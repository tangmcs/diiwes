#!/usr/bin/env python3
"""Parallel trainer for the paper ES experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DIIWES, StandardES
from core.policies import DiscretePolicy, MLPPolicy, make_layer_slices
from utilities import ObsNormalizer


CONDITIONS = {
    "standard_es",
    "no_curvature",
    "diag_curvature",
}

_WORKER_ENV = None
_WORKER_POLICY = None
_WORKER_MAX_STEPS = None


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else dict(config)


def _make_env(
    env_name: str,
    env_kwargs: dict[str, Any] | None = None,
    *,
    frame_stack: int = 1,
    fire_reset: bool = False,
    fire_reset_steps: list[int] | tuple[int, ...] | None = None,
    fire_on_life_loss: bool = False,
    action_indices: list[int] | tuple[int, ...] | None = None,
):
    import gymnasium as gym

    try:
        import ale_py

        gym.register_envs(ale_py)
    except Exception:
        pass

    class FireResetWrapper(gym.Wrapper):
        def __init__(
            self,
            env: Any,
            action_steps: list[int] | tuple[int, ...],
            *,
            fire_after_life_loss: bool = False,
        ) -> None:
            super().__init__(env)
            if not hasattr(env.action_space, "n"):
                raise ValueError("fire_reset requires a discrete action space")
            if not action_steps:
                raise ValueError("fire_reset_steps must contain at least one action")
            self.action_steps = [int(action) for action in action_steps]
            n_actions = int(env.action_space.n)
            if min(self.action_steps) < 0 or max(self.action_steps) >= n_actions:
                raise ValueError(f"fire_reset_steps must be valid actions in [0, {n_actions})")
            meanings = self._action_meanings()
            if meanings is not None and not any("FIRE" in meanings[action] for action in self.action_steps):
                raise ValueError("fire_reset=True requires at least one FIRE action in fire_reset_steps")
            self.fire_after_life_loss = bool(fire_after_life_loss)
            self.lives = self._lives()

        def _action_meanings(self) -> list[str] | None:
            base = getattr(self.env, "unwrapped", self.env)
            if hasattr(base, "get_action_meanings"):
                return list(base.get_action_meanings())
            return None

        def _lives(self) -> int | None:
            base = getattr(self.env, "unwrapped", self.env)
            ale = getattr(base, "ale", None)
            if ale is not None and hasattr(ale, "lives"):
                return int(ale.lives())
            return None

        def _fire_sequence(self, obs: Any, reward: float = 0.0):
            info: dict[str, Any] = {}
            terminated = False
            truncated = False
            for action in self.action_steps:
                obs, step_reward, terminated, truncated, info = self.env.step(action)
                reward += float(step_reward)
                if terminated or truncated:
                    break
            self.lives = self._lives()
            return obs, reward, terminated, truncated, info

        def reset(self, **kwargs: Any):
            obs, info = self.env.reset(**kwargs)
            obs, _, terminated, truncated, _ = self._fire_sequence(obs)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
                obs, _, terminated, truncated, _ = self._fire_sequence(obs)
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
            return obs, info

        def step(self, action: Any):
            old_lives = self.lives
            obs, reward, terminated, truncated, info = self.env.step(action)
            new_lives = self._lives()
            lost_life = old_lives is not None and new_lives is not None and new_lives < old_lives
            if self.fire_after_life_loss and lost_life and not terminated and not truncated:
                obs, reward, terminated, truncated, info = self._fire_sequence(obs, float(reward))
            else:
                self.lives = new_lives
            return obs, reward, terminated, truncated, info

    class FrameStackWrapper(gym.Wrapper):
        def __init__(self, env: Any, n_frames: int) -> None:
            super().__init__(env)
            if not isinstance(env.observation_space, gym.spaces.Box):
                raise ValueError("frame_stack requires a Box observation space")
            self.n_frames = int(n_frames)
            self.frames: deque[np.ndarray] = deque(maxlen=self.n_frames)
            low = np.repeat(env.observation_space.low[None, ...], self.n_frames, axis=0)
            high = np.repeat(env.observation_space.high[None, ...], self.n_frames, axis=0)
            self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

        def _stacked_obs(self) -> np.ndarray:
            return np.stack(list(self.frames), axis=0)

        def reset(self, **kwargs: Any):
            obs, info = self.env.reset(**kwargs)
            self.frames.clear()
            for _ in range(self.n_frames):
                self.frames.append(np.asarray(obs).copy())
            return self._stacked_obs(), info

        def step(self, action: Any):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(np.asarray(obs).copy())
            return self._stacked_obs(), reward, terminated, truncated, info

        def get_action_meanings(self) -> list[str]:
            if hasattr(self.env, "get_action_meanings"):
                return list(self.env.get_action_meanings())
            base = getattr(self.env, "unwrapped", self.env)
            if hasattr(base, "get_action_meanings"):
                return list(base.get_action_meanings())
            return []

    class ActionSubsetWrapper(gym.ActionWrapper):
        def __init__(self, env: Any, indices: list[int] | tuple[int, ...]) -> None:
            super().__init__(env)
            if not hasattr(env.action_space, "n"):
                raise ValueError("action_indices requires a discrete action space")
            if not indices:
                raise ValueError("action_indices must contain at least one action")
            self.action_indices = [int(action) for action in indices]
            n_actions = int(env.action_space.n)
            if min(self.action_indices) < 0 or max(self.action_indices) >= n_actions:
                raise ValueError(f"action_indices must be valid actions in [0, {n_actions})")
            self.action_space = gym.spaces.Discrete(len(self.action_indices))

        def action(self, action: int) -> int:
            return self.action_indices[int(action)]

        def get_action_meanings(self) -> list[str]:
            base = getattr(self.env, "unwrapped", self.env)
            if hasattr(base, "get_action_meanings"):
                meanings = list(base.get_action_meanings())
                return [meanings[action] for action in self.action_indices]
            return [str(action) for action in self.action_indices]

    env = gym.make(env_name, **(env_kwargs or {}))
    if fire_reset:
        env = FireResetWrapper(env, fire_reset_steps or [1], fire_after_life_loss=fire_on_life_loss)
    if action_indices is not None:
        env = ActionSubsetWrapper(env, action_indices)
    frame_stack = int(frame_stack)
    if frame_stack <= 0:
        raise ValueError("frame_stack must be positive")
    if frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack)
    return env


def make_policy(config: dict[str, Any], env: Any) -> MLPPolicy | DiscretePolicy:
    hidden_dims = config.get("hidden_dims", [64, 64])
    activation = config.get("activation", "tanh")
    output_activation = config.get("output_activation", "tanh")
    ob_dim = int(np.prod(env.observation_space.shape))
    if hasattr(env.action_space, "n"):
        n_actions = int(env.action_space.n)
        mlp = MLPPolicy(
            ob_dim=ob_dim,
            ac_dim=n_actions,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=None,
        )
        return DiscretePolicy(mlp, n_actions)

    ac_dim = int(np.prod(env.action_space.shape))
    return MLPPolicy(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        hidden_dims=hidden_dims,
        activation=activation,
        output_activation=output_activation,
    )


def _init_worker(config: dict[str, Any]) -> None:
    global _WORKER_ENV, _WORKER_POLICY, _WORKER_MAX_STEPS
    _WORKER_ENV = _make_env(
        config["env_name"],
        config.get("env_kwargs"),
        frame_stack=config.get("frame_stack", 1),
        fire_reset=config.get("fire_reset", False),
        fire_reset_steps=config.get("fire_reset_steps"),
        fire_on_life_loss=config.get("fire_on_life_loss", False),
        action_indices=config.get("action_indices"),
    )
    _WORKER_POLICY = make_policy(config, _WORKER_ENV)
    _WORKER_MAX_STEPS = int(config.get("max_episode_steps", 1000))


def _scale_obs(obs: Any, obs_scale: float) -> np.ndarray:
    obs_array = np.asarray(obs, dtype=np.float64)
    if obs_scale != 1.0:
        obs_array = obs_array / obs_scale
    return obs_array


def _evaluate_params(task: tuple[np.ndarray, int, Any, Any, bool, float]) -> tuple[float, list[np.ndarray]]:
    params, rollout_seed, obs_mean, obs_var, collect_obs, obs_scale = task
    obs, _ = _WORKER_ENV.reset(seed=int(rollout_seed))
    if hasattr(_WORKER_ENV.action_space, "seed"):
        _WORKER_ENV.action_space.seed(int(rollout_seed))

    observations: list[np.ndarray] = []
    total_reward = 0.0
    for _ in range(_WORKER_MAX_STEPS):
        obs_scaled = _scale_obs(obs, obs_scale)
        if obs_mean is not None and obs_var is not None:
            obs_in = (obs_scaled - obs_mean) / np.sqrt(obs_var + 1e-8)
        else:
            obs_in = obs_scaled
        action = _WORKER_POLICY.act(obs_in, params)
        if hasattr(_WORKER_ENV.action_space, "low"):
            action = np.clip(action, _WORKER_ENV.action_space.low, _WORKER_ENV.action_space.high)
        obs, reward, terminated, truncated, _ = _WORKER_ENV.step(action)
        total_reward += float(reward)
        if collect_obs:
            observations.append(_scale_obs(obs, obs_scale).copy())
        if terminated or truncated:
            break
    return total_reward, observations


def _condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    config = dict(config)
    config["condition"] = condition

    if condition == "standard_es":
        config["algorithm"] = "standard_es"
    elif condition == "no_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = False
        config["curvature_mode"] = "diag"
    elif condition == "diag_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
    else:
        raise ValueError(f"unknown condition: {condition}")
    return config


def make_optimizer(
    config: dict[str, Any],
    num_params: int,
    policy: MLPPolicy | DiscretePolicy,
    seed: int,
) -> StandardES | DIIWES:
    if config["algorithm"] == "standard_es":
        return StandardES(
            num_params=num_params,
            population_size=config.get("population_size", 200),
            learning_rate=config.get("learning_rate", 0.02),
            noise_std=config.get("noise_std", 0.02),
            l2_coeff=config.get("l2_coeff", 0.0),
            antithetic=config.get("antithetic", True),
            rank_fitness=config.get("rank_fitness", True),
            max_grad_norm=config.get("max_grad_norm", 0.0),
            max_param_norm=config.get("max_param_norm", None),
            seed=seed,
        )

    block_slices = None
    if config.get("curvature_mode", "diag") == "block" and config.get("block_structure") == "layer":
        block_slices = make_layer_slices(policy)

    return DIIWES(
        num_params=num_params,
        population_size=config.get("population_size", 200),
        learning_rate=config.get("learning_rate", 0.02),
        noise_std=config.get("noise_std", 0.02),
        l2_coeff=config.get("l2_coeff", 0.0),
        buffer_size=config.get("buffer_size", 1024),
        reuse_fraction=config.get("reuse_fraction", 0.2),
        min_importance_weight=config.get("min_importance_weight", 1e-3),
        max_importance_weight=config.get("max_importance_weight", 10.0),
        implicit_damping=config.get("implicit_damping", 0.1),
        antithetic=config.get("antithetic", True),
        rank_fitness=config.get("rank_fitness", True),
        max_grad_norm=config.get("max_grad_norm", 0.0),
        max_sample_age=config.get("max_sample_age", 3),
        buffer_sampling=config.get("buffer_sampling", "random"),
        elite_quantile=config.get("elite_quantile", 0.25),
        max_param_norm=config.get("max_param_norm", None),
        seed=seed,
        use_curvature=config.get("use_curvature", True),
        curvature_mode=config.get("curvature_mode", "diag"),
        curvature_beta=config.get("curvature_beta", 0.99),
        curvature_clip=config.get("curvature_clip", 1e3),
        min_step_multiplier=config.get("min_step_multiplier", 0.05),
        trust_radius=config.get("trust_radius", None),
        ess_min_ratio=config.get("ess_min_ratio", 0.2),
        block_slices=block_slices,
        use_leave_one_out_curvature_baseline=config.get("use_leave_one_out_curvature_baseline", True),
        bias_correct_curvature_ema=config.get("bias_correct_curvature_ema", True),
    )


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return None


def _history_record(
    iteration: int,
    eval_reward: float,
    best_reward: float,
    best_fitness_iter: float,
    best_fitness_so_far: float,
    fresh_fitness: np.ndarray,
    info: dict[str, Any],
    learning_rate: float,
    elapsed: float,
) -> dict[str, Any]:
    record = {
        "iteration": int(iteration),
        "eval_reward": float(eval_reward),
        "best_reward": float(best_reward),
        "best_fitness_iter": float(best_fitness_iter),
        "best_fitness_so_far": float(best_fitness_so_far),
        "mean_fitness": float(info.get("mean_fitness", np.mean(fresh_fitness) if len(fresh_fitness) else 0.0)),
        "max_fitness": float(info.get("max_fitness", np.max(fresh_fitness) if len(fresh_fitness) else 0.0)),
        "n_fresh": int(info.get("n_fresh", len(fresh_fitness))),
        "n_reused": int(info.get("n_reused", 0)),
        "ess_ratio": float(info.get("ess_ratio", 1.0)),
        "used_replay": bool(info.get("used_replay", False)),
        "grad_norm": float(info.get("grad_norm", 0.0)),
        "step_norm": float(info.get("step_norm", 0.0)),
        "lr": float(learning_rate),
        "time": float(elapsed),
    }
    for key, value in info.items():
        if key in record:
            continue
        scalar = _json_scalar(value)
        if scalar is not None:
            record[key] = scalar
    return record


def _format_progress(record: dict[str, Any], verbose: bool) -> str:
    if not verbose:
        return (
            f"Iter {record['iteration']:4d} | "
            f"Eval {record['eval_reward']:8.2f} | "
            f"Best {record['best_reward']:8.2f}"
        )
    return (
        f"Iter {record['iteration']:4d} | "
        f"Eval {record['eval_reward']:8.2f} | "
        f"Best {record['best_reward']:8.2f} | "
        f"Step {record['step_norm']:.3f} | "
        f"PreTrust {record.get('pre_trust_step_norm', record['step_norm']):.3f} | "
        f"Curv {record.get('curv_mean', 0.0):.3e} | "
        f"Fresh {record['n_fresh']} | "
        f"Reused {record['n_reused']} | "
        f"Time {record['time']:.1f}s"
    )


def train(
    config: dict[str, Any],
    seed: int,
    output_dir: str,
    n_workers: int,
    verbose: bool = False,
) -> tuple[float, np.ndarray]:
    config = dict(config)
    verbose = bool(verbose or config.get("verbose", False))
    np.random.seed(seed)

    env = _make_env(
        config["env_name"],
        config.get("env_kwargs"),
        frame_stack=config.get("frame_stack", 1),
        fire_reset=config.get("fire_reset", False),
        fire_reset_steps=config.get("fire_reset_steps"),
        fire_on_life_loss=config.get("fire_on_life_loss", False),
        action_indices=config.get("action_indices"),
    )
    policy = make_policy(config, env)

    optimizer = make_optimizer(config, policy.num_params, policy, seed)
    use_curvature = bool(getattr(optimizer, "use_curvature", False))
    curvature_mode = str(getattr(optimizer, "curvature_mode", "none"))
    trust_radius = getattr(optimizer, "trust_radius", None)

    obs_normalizer = ObsNormalizer(env.observation_space.shape) if config.get("use_obs_norm", False) else None
    params = np.random.randn(policy.num_params) * float(config.get("init_param_std", 0.1))
    optimizer.current_params = params.copy()

    n_iterations = int(config.get("n_iterations", 500))
    eval_episodes = int(config.get("eval_episodes", 3))
    eval_interval = int(config.get("eval_interval", 1))
    log_interval = int(config.get("log_interval", 10))
    base_lr = float(config.get("learning_rate", 0.02))
    lr_decay = float(config.get("lr_decay", 1.0))
    evaluate_center_fitness = bool(config.get("evaluate_center_fitness", False))
    common_rollout_seed = bool(config.get("common_rollout_seed", False))
    obs_scale = float(config.get("obs_scale", 1.0))
    if obs_scale <= 0.0:
        raise ValueError("obs_scale must be positive")

    print(
        f"Run: {config['condition']} | env={config['env_name']} | seed={seed} | "
        f"iterations={n_iterations} | workers={n_workers}",
        flush=True,
    )
    if verbose:
        print(f"Policy: {policy.__class__.__name__}, {policy.num_params} parameters", flush=True)
        print(
            f"Optimizer: algorithm={config['algorithm']} | curvature={use_curvature} | "
            f"mode={curvature_mode} | trust_radius={trust_radius} | lr={base_lr}",
            flush=True,
        )

    pool = Pool(processes=n_workers, initializer=_init_worker, initargs=(config,))
    history: list[dict[str, Any]] = []
    best_reward = -np.inf
    best_fitness_so_far = -np.inf
    best_params = params.copy()
    last_eval_reward = -np.inf

    try:
        for iteration in range(n_iterations):
            start = time.time()
            optimizer.learning_rate = base_lr * (lr_decay**iteration)
            if obs_normalizer is not None:
                obs_mean, obs_var = obs_normalizer.get_mean_var()
            else:
                obs_mean, obs_var = None, None

            noise, ask_info = optimizer.ask()
            is_reused = np.asarray(ask_info["is_reused"], dtype=bool)
            fresh_indices = np.where(~is_reused)[0]

            fresh_tasks = []
            for local_idx, noise_idx in enumerate(fresh_indices):
                theta_eval = params + optimizer.noise_std * noise[noise_idx]
                if common_rollout_seed:
                    rollout_seed = seed
                else:
                    rollout_seed = seed + 100_000 * iteration + local_idx
                fresh_tasks.append((theta_eval, rollout_seed, obs_mean, obs_var, obs_normalizer is not None, obs_scale))
            fresh_results = pool.map(_evaluate_params, fresh_tasks) if fresh_tasks else []
            fresh_fitness = np.asarray([result[0] for result in fresh_results], dtype=np.float64)

            if obs_normalizer is not None and fresh_results:
                observations = [obs for _, rollout_obs in fresh_results for obs in rollout_obs]
                if observations:
                    obs_normalizer.update_batch(np.asarray(observations, dtype=np.float64))

            center_fitness = None
            if evaluate_center_fitness:
                center = pool.map(
                    _evaluate_params,
                    [(params.copy(), seed + 900_000 + iteration, obs_mean, obs_var, False, obs_scale)],
                )
                center_fitness = float(center[0][0])

            params, info = optimizer.tell(
                params,
                noise,
                fresh_fitness,
                ask_info,
                center_fitness=center_fitness,
            )
            optimizer.current_params = params.copy()

            best_fitness_iter = float(np.max(fresh_fitness)) if len(fresh_fitness) else float(info.get("max_fitness", -np.inf))
            best_fitness_so_far = max(best_fitness_so_far, best_fitness_iter)

            should_eval = iteration % eval_interval == 0 or iteration == n_iterations - 1
            if should_eval:
                eval_tasks = []
                for eval_idx in range(eval_episodes):
                    if common_rollout_seed:
                        rollout_seed = seed
                    else:
                        rollout_seed = seed + 1_000_000 + 10_000 * iteration + eval_idx
                    eval_tasks.append((params.copy(), rollout_seed, obs_mean, obs_var, False, obs_scale))
                eval_results = pool.map(_evaluate_params, eval_tasks)
                last_eval_reward = float(np.mean([result[0] for result in eval_results]))
                if last_eval_reward > best_reward:
                    best_reward = last_eval_reward
                    best_params = params.copy()

            elapsed = time.time() - start
            record = _history_record(
                iteration,
                last_eval_reward,
                best_reward,
                best_fitness_iter,
                best_fitness_so_far,
                fresh_fitness,
                info,
                optimizer.learning_rate,
                elapsed,
            )
            history.append(record)

            if iteration % log_interval == 0 or iteration == n_iterations - 1:
                print(_format_progress(record, verbose), flush=True)
    finally:
        pool.close()
        pool.join()
        env.close()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "best_params.npy"), best_params)
    np.save(os.path.join(output_dir, "final_params.npy"), params)
    if hasattr(optimizer, "hessian_ema"):
        np.save(os.path.join(output_dir, "hessian_ema.npy"), optimizer.hessian_ema)
    if obs_normalizer is not None:
        np.savez(os.path.join(output_dir, "obs_norm.npz"), **obs_normalizer.get_state())
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "seed": int(seed)}, f, indent=2)

    print(f"Training complete. Best reward: {best_reward:.2f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    return best_reward, best_params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--trust-radius", type=float, default=None)
    parser.add_argument("--reuse-fraction", type=float, default=None)
    parser.add_argument(
        "--rank-fitness",
        choices=("true", "false"),
        default=None,
        help="Override whether ES uses rank-shaped fitness for the policy gradient.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print detailed optimizer diagnostics while training.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.trust_radius is not None:
        config["trust_radius"] = float(args.trust_radius)
    if args.reuse_fraction is not None:
        config["reuse_fraction"] = float(args.reuse_fraction)
    if args.rank_fitness is not None:
        config["rank_fitness"] = args.rank_fitness == "true"
    if args.iterations is not None:
        config["n_iterations"] = int(args.iterations)
    config = _condition_config(config, args.condition)

    if args.workers is None:
        n_workers = max(1, len(os.sched_getaffinity(0)) - 2)
    else:
        n_workers = int(args.workers)
    train(config, args.seed, args.output, n_workers, verbose=args.verbose)


if __name__ == "__main__":
    main()

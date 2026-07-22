#!/usr/bin/env python3
"""Parallel trainer for the paper ES experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DIIWES, StandardES
from core.policies import DiscretePolicy, MLPPolicy, make_layer_slices
from utilities import ObsNormalizer


CONDITIONS = {
    "standard_es",
    "standard_es_trust",
    "no_curvature",
    "diag_curvature",
    "global_curvature",
    "block_curvature",
    "directional_curvature",
    "normalized_diag_curvature",
    "normalized_block_curvature",
}

LR_SCHEDULES = {"constant", "exponential", "inverse_linear", "inverse_sqrt"}

_WORKER_ENV = None
_WORKER_POLICY = None
_WORKER_MAX_STEPS = None


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return {} if config is None else dict(config)


def learning_rate_at_iteration(
    initial_learning_rate: float,
    iteration: int,
    schedule: str = "exponential",
    decay: float = 1.0,
) -> float:
    """Resolve the scalar step size without changing optimizer semantics."""
    alpha0 = float(initial_learning_rate)
    step = int(iteration)
    name = str(schedule).lower()
    if not np.isfinite(alpha0) or alpha0 <= 0.0:
        raise ValueError("initial_learning_rate must be finite and positive")
    if step < 0:
        raise ValueError("iteration must be nonnegative")
    if name not in LR_SCHEDULES:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if name == "constant":
        return alpha0
    if name == "exponential":
        gamma = float(decay)
        if not np.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("lr_decay must be finite and positive")
        return float(alpha0 * (gamma**step))
    if name == "inverse_linear":
        return float(alpha0 / (step + 1.0))
    return float(alpha0 / np.sqrt(step + 1.0))


def _source_digest(config_path: str) -> str:
    """Hash the exact optimizer/trainer/config inputs used by a run."""
    root = Path(__file__).resolve().parents[1]
    config = Path(config_path).resolve()
    source_paths = [
        root / "core" / "__init__.py",
        root / "core" / "diiwes.py",
        root / "core" / "policies.py",
        root / "core" / "standard_es.py",
        root / "experiments" / "train.py",
        root / "utilities" / "__init__.py",
        root / "utilities" / "obs_norm.py",
        config,
    ]
    digest = hashlib.sha256()
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            label = path.relative_to(root).as_posix()
        except ValueError:
            label = str(path)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def _evaluate_params(task: tuple[np.ndarray, int, Any, Any, bool, float]) -> tuple[float, list[np.ndarray], int]:
    params, rollout_seed, obs_mean, obs_var, collect_obs, obs_scale = task
    obs, _ = _WORKER_ENV.reset(seed=int(rollout_seed))
    if hasattr(_WORKER_ENV.action_space, "seed"):
        _WORKER_ENV.action_space.seed(int(rollout_seed))

    observations: list[np.ndarray] = []
    total_reward = 0.0
    steps = 0
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
        steps += 1
        total_reward += float(reward)
        if collect_obs:
            observations.append(_scale_obs(obs, obs_scale).copy())
        if terminated or truncated:
            break
    return total_reward, observations, steps


def _condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    config = dict(config)
    config["condition"] = condition

    if condition == "standard_es":
        config["algorithm"] = "standard_es"
        config["use_trust_radius_for_standard_es"] = False
    elif condition == "standard_es_trust":
        config["algorithm"] = "standard_es"
        config["use_trust_radius_for_standard_es"] = True
    elif condition == "no_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = False
        config["curvature_mode"] = "diag"
    elif condition == "diag_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
        config["curvature_step_mode"] = "dampen"
    elif condition == "global_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "global"
        config["curvature_step_mode"] = "dampen"
    elif condition == "block_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "block"
        config["block_structure"] = "layer"
        config["curvature_step_mode"] = "dampen"
    elif condition == "directional_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "directional"
        config["curvature_step_mode"] = "dampen"
    elif condition == "normalized_diag_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
        config["curvature_step_mode"] = "normalized"
    elif condition == "normalized_block_curvature":
        config["algorithm"] = "semi_implicit_curvature_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "block"
        config["block_structure"] = "layer"
        config["curvature_step_mode"] = "normalized"
    else:
        raise ValueError(f"unknown condition: {condition}")
    if config["algorithm"] == "semi_implicit_curvature_es":
        config.setdefault("curvature_fitness", "raw")
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
            trust_radius=config.get("trust_radius", None)
            if config.get("use_trust_radius_for_standard_es", False)
            else None,
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
        curvature_fitness=config.get("curvature_fitness", "raw"),
        curvature_mode=config.get("curvature_mode", "diag"),
        curvature_step_mode=config.get("curvature_step_mode", "dampen"),
        curvature_beta=config.get("curvature_beta", 0.99),
        curvature_clip=config.get("curvature_clip", 1e3),
        min_step_multiplier=config.get("min_step_multiplier", 0.05),
        trust_radius=config.get("trust_radius", None),
        ess_min_ratio=config.get("ess_min_ratio", 0.2),
        block_slices=block_slices,
        use_leave_one_out_curvature_baseline=config.get("use_leave_one_out_curvature_baseline", True),
        bias_correct_curvature_ema=config.get("bias_correct_curvature_ema", True),
    )


def _file_sha256(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 digest of one initialization artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_initial_state(
    config: dict[str, Any],
    policy: MLPPolicy | DiscretePolicy,
    obs_normalizer: ObsNormalizer | None,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a matched policy-gradient checkpoint or use legacy random init."""
    params_path = config.get("initial_params_path")
    obs_path = config.get("initial_obs_norm_path")
    metadata: dict[str, Any] = {}
    if params_path is None:
        if obs_path is not None:
            raise ValueError("initial_obs_norm_path requires initial_params_path")
        params = np.random.randn(policy.num_params) * float(
            config.get("init_param_std", 0.1)
        )
        metadata["parameter_initialization"] = "seeded_gaussian"
        metadata["initialization_seed"] = int(seed)
        return params.astype(np.float64, copy=False), metadata

    params_file = Path(params_path).resolve()
    if not params_file.is_file():
        raise FileNotFoundError(params_file)
    params = np.load(params_file, allow_pickle=False)
    params = np.asarray(params, dtype=np.float64)
    expected_shape = (int(policy.num_params),)
    if params.shape != expected_shape:
        raise ValueError(
            f"initial parameter checkpoint has shape {params.shape}, expected {expected_shape}"
        )
    if not np.all(np.isfinite(params)):
        raise FloatingPointError("initial parameter checkpoint contains non-finite values")
    params_digest = _file_sha256(params_file)
    expected_params_digest = config.get("expected_initial_params_sha256")
    if expected_params_digest and params_digest != expected_params_digest:
        raise RuntimeError(
            "initial parameter digest mismatch: "
            f"expected {expected_params_digest}, found {params_digest}"
        )
    metadata.update(
        {
            "parameter_initialization": "policy_gradient_checkpoint",
            "initial_params_path": str(params_file),
            "initial_params_sha256": params_digest,
        }
    )

    if obs_path is None:
        if obs_normalizer is not None:
            raise ValueError(
                "a policy-gradient checkpoint with use_obs_norm=true requires "
                "initial_obs_norm_path"
            )
        return params.copy(), metadata
    if obs_normalizer is None:
        raise ValueError("initial_obs_norm_path requires use_obs_norm=true")

    obs_file = Path(obs_path).resolve()
    if not obs_file.is_file():
        raise FileNotFoundError(obs_file)
    with np.load(obs_file, allow_pickle=False) as state:
        required = {"mean", "var", "count"}
        missing = required.difference(state.files)
        if missing:
            raise ValueError(
                f"initial observation normalizer is missing {sorted(missing)}"
            )
        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(np.asarray(state["count"]).item())
    if mean.shape != obs_normalizer.shape or var.shape != obs_normalizer.shape:
        raise ValueError(
            "initial observation normalizer shape mismatch: "
            f"mean={mean.shape}, var={var.shape}, expected={obs_normalizer.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)):
        raise FloatingPointError("initial observation normalizer is non-finite")
    if np.any(var < 0.0) or not np.isfinite(count) or count <= 0.0:
        raise ValueError("initial observation variance/count must be valid")
    obs_normalizer.mean = mean.copy()
    obs_normalizer.var = var.copy()
    obs_normalizer.count = count
    obs_digest = _file_sha256(obs_file)
    expected_obs_digest = config.get("expected_initial_obs_norm_sha256")
    if expected_obs_digest and obs_digest != expected_obs_digest:
        raise RuntimeError(
            "initial observation-normalizer digest mismatch: "
            f"expected {expected_obs_digest}, found {obs_digest}"
        )
    metadata.update(
        {
            "initial_obs_norm_path": str(obs_file),
            "initial_obs_norm_sha256": obs_digest,
            "initial_obs_norm_count": count,
        }
    )
    return params.copy(), metadata


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
    train_env_steps: int,
    train_env_steps_iter: int,
    eval_env_steps: int,
    eval_env_steps_iter: int,
) -> dict[str, Any]:
    total_env_steps = int(train_env_steps) + int(eval_env_steps)
    total_env_steps_iter = int(train_env_steps_iter) + int(eval_env_steps_iter)
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
        "env_steps": int(train_env_steps),
        "env_steps_iter": int(train_env_steps_iter),
        "train_env_steps": int(train_env_steps),
        "train_env_steps_iter": int(train_env_steps_iter),
        "eval_env_steps": int(eval_env_steps),
        "eval_env_steps_iter": int(eval_env_steps_iter),
        "total_env_steps": total_env_steps,
        "total_env_steps_iter": total_env_steps_iter,
    }
    for key, value in info.items():
        if key in record:
            continue
        scalar = _json_scalar(value)
        if scalar is not None:
            record[key] = scalar
    return record


def _format_progress(record: dict[str, Any], verbose: bool) -> str:
    clipping = ""
    if "curvature_clip_frac" in record and "multiplier_floor_clip_frac" in record:
        clipping = (
            f" | CurvCap {100.0 * float(record['curvature_clip_frac']):5.1f}%"
            f" | MultFloor {100.0 * float(record['multiplier_floor_clip_frac']):5.1f}%"
        )
    if not verbose:
        return (
            f"Iter {record['iteration']:4d} | "
            f"Eval {record['eval_reward']:8.2f} | "
            f"Best {record['best_reward']:8.2f}"
            f"{clipping}"
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
        f"{clipping}"
    )


def _validated_coordinate_vector(
    optimizer: Any, attribute: str, num_params: int
) -> np.ndarray:
    """Return one exact finite float64 optimizer vector or fail the run."""
    value = getattr(optimizer, attribute, None)
    if value is None:
        raise RuntimeError(f"optimizer did not expose {attribute} after tell()")
    array = np.asarray(value)
    expected_shape = (int(num_params),)
    if array.shape != expected_shape:
        raise ValueError(
            f"optimizer {attribute} has shape {array.shape}, expected {expected_shape}"
        )
    if array.dtype != np.dtype(np.float64):
        raise TypeError(
            f"optimizer {attribute} has dtype {array.dtype}, expected float64"
        )
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(f"optimizer {attribute} contains non-finite values")
    return array


def _write_coordinate_history_row(
    optimizer: Any,
    hessian_history: np.memmap,
    multiplier_history: np.memmap,
    iteration: int,
    num_params: int,
) -> None:
    """Persist and flush the exact vectors for one completed optimizer step."""
    expected_history_shape = (hessian_history.shape[0], int(num_params))
    if hessian_history.shape != expected_history_shape:
        raise ValueError(
            "hessian history has shape "
            f"{hessian_history.shape}, expected {expected_history_shape}"
        )
    if multiplier_history.shape != expected_history_shape:
        raise ValueError(
            "multiplier history has shape "
            f"{multiplier_history.shape}, expected {expected_history_shape}"
        )
    if hessian_history.dtype != np.dtype(np.float64):
        raise TypeError("hessian history must have dtype float64")
    if multiplier_history.dtype != np.dtype(np.float64):
        raise TypeError("multiplier history must have dtype float64")
    row = int(iteration)
    if row < 0 or row >= hessian_history.shape[0]:
        raise IndexError(
            f"coordinate history row {row} is outside 0..{hessian_history.shape[0] - 1}"
        )

    hessian = _validated_coordinate_vector(
        optimizer, "hessian_for_step_vector", num_params
    )
    multiplier = _validated_coordinate_vector(
        optimizer, "step_multiplier_vector", num_params
    )
    hessian_history[row, :] = hessian
    multiplier_history[row, :] = multiplier
    hessian_history.flush()
    multiplier_history.flush()


def _flush_close_memmap(value: np.memmap | None) -> None:
    """Flush and deterministically close a NumPy memmap when one was opened."""
    if value is None:
        return
    try:
        value.flush()
    finally:
        mapping = getattr(value, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def _should_record_coordinate_history(
    config: dict[str, Any], optimizer: StandardES | DIIWES
) -> bool:
    """Limit full-coordinate artifacts to the locked diagonal dampen arm."""
    return bool(
        config.get("condition") == "diag_curvature"
        and isinstance(optimizer, DIIWES)
        and optimizer.use_curvature
        and optimizer.curvature_mode == "diag"
        and optimizer.curvature_step_mode == "dampen"
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
    os.makedirs(output_dir, exist_ok=True)
    canonical_history_path = os.path.join(output_dir, "history.json")
    if os.path.exists(canonical_history_path):
        raise FileExistsError(
            f"refusing to overwrite completed run artifact: {canonical_history_path}"
        )

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
    curvature_fitness = str(getattr(optimizer, "curvature_fitness", "none"))
    curvature_mode = str(getattr(optimizer, "curvature_mode", "none"))
    trust_radius = getattr(optimizer, "trust_radius", None)

    obs_normalizer = ObsNormalizer(env.observation_space.shape) if config.get("use_obs_norm", False) else None
    params, initialization_metadata = _load_initial_state(
        config, policy, obs_normalizer, seed
    )
    config.update(initialization_metadata)
    optimizer.current_params = params.copy()

    n_iterations = int(config.get("n_iterations", 500))
    eval_episodes = int(config.get("eval_episodes", 3))
    eval_interval = int(config.get("eval_interval", 1))
    log_interval = int(config.get("log_interval", 10))
    base_lr = float(config.get("learning_rate", 0.02))
    lr_decay = float(config.get("lr_decay", 1.0))
    lr_schedule = str(config.get("lr_schedule", "exponential")).lower()
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown learning-rate schedule: {lr_schedule}")
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
            f"mode={curvature_mode} | curvature_fitness={curvature_fitness} | "
            f"trust_radius={trust_radius} | lr={base_lr} | schedule={lr_schedule}",
            flush=True,
        )

    # Create the run directory before training so diagnostic records survive a
    # timeout or interrupted run.  ``history.json`` remains the canonical
    # completed-run artifact; the append-only JSONL file is the live audit log.
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "seed": int(seed)}, f, indent=2)
    record_coordinate_history = _should_record_coordinate_history(config, optimizer)

    history_jsonl = None
    hessian_history: np.memmap | None = None
    multiplier_history: np.memmap | None = None
    pool = None
    history: list[dict[str, Any]] = []
    best_reward = -np.inf
    best_fitness_so_far = -np.inf
    best_params = params.copy()
    last_eval_reward = -np.inf
    train_env_steps = 0
    eval_env_steps = 0

    try:
        history_jsonl = open(
            os.path.join(output_dir, "history.jsonl"), "w", encoding="utf-8"
        )
        if record_coordinate_history:
            coordinate_shape = (n_iterations, policy.num_params)
            hessian_history = np.lib.format.open_memmap(
                os.path.join(output_dir, "hessian_for_step_history.npy"),
                mode="w+",
                dtype=np.float64,
                shape=coordinate_shape,
            )
            multiplier_history = np.lib.format.open_memmap(
                os.path.join(output_dir, "step_multiplier_history.npy"),
                mode="w+",
                dtype=np.float64,
                shape=coordinate_shape,
            )
        pool = Pool(processes=n_workers, initializer=_init_worker, initargs=(config,))
        for iteration in range(n_iterations):
            start = time.time()
            optimizer.learning_rate = learning_rate_at_iteration(
                base_lr,
                iteration,
                schedule=lr_schedule,
                decay=lr_decay,
            )
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
            train_env_steps_iter = int(sum(result[2] for result in fresh_results))
            train_env_steps += train_env_steps_iter

            if obs_normalizer is not None and fresh_results:
                observations = [obs for _, rollout_obs, _ in fresh_results for obs in rollout_obs]
                if observations:
                    obs_normalizer.update_batch(np.asarray(observations, dtype=np.float64))

            center_fitness = None
            center_env_steps_iter = 0
            if evaluate_center_fitness:
                center = pool.map(
                    _evaluate_params,
                    [(params.copy(), seed + 900_000 + iteration, obs_mean, obs_var, False, obs_scale)],
                )
                center_fitness = float(center[0][0])
                center_env_steps_iter = int(center[0][2])
                train_env_steps_iter += center_env_steps_iter
                train_env_steps += center_env_steps_iter

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
            eval_env_steps_iter = 0
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
                eval_env_steps_iter = int(sum(result[2] for result in eval_results))
                eval_env_steps += eval_env_steps_iter
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
                train_env_steps,
                train_env_steps_iter,
                eval_env_steps,
                eval_env_steps_iter,
            )
            if record_coordinate_history:
                if hessian_history is None or multiplier_history is None:
                    raise RuntimeError("coordinate history memmaps were not initialized")
                _write_coordinate_history_row(
                    optimizer,
                    hessian_history,
                    multiplier_history,
                    iteration,
                    policy.num_params,
                )
            history.append(record)
            history_jsonl.write(json.dumps(record, separators=(",", ":")) + "\n")
            history_jsonl.flush()

            if iteration % log_interval == 0 or iteration == n_iterations - 1:
                print(_format_progress(record, verbose), flush=True)
    finally:
        try:
            _flush_close_memmap(hessian_history)
        finally:
            try:
                _flush_close_memmap(multiplier_history)
            finally:
                try:
                    if history_jsonl is not None:
                        history_jsonl.close()
                finally:
                    try:
                        if pool is not None:
                            pool.close()
                            pool.join()
                    finally:
                        env.close()

    np.save(os.path.join(output_dir, "best_params.npy"), best_params)
    np.save(os.path.join(output_dir, "final_params.npy"), params)
    if hasattr(optimizer, "hessian_ema"):
        np.save(os.path.join(output_dir, "hessian_ema.npy"), optimizer.hessian_ema)
    if obs_normalizer is not None:
        np.savez(os.path.join(output_dir, "obs_norm.npz"), **obs_normalizer.get_state())
    with open(canonical_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best reward: {best_reward:.2f}", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    return best_reward, best_params


def _parse_optional_float(value: str) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "off", "false"}:
        return None
    return float(value)


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Override the configured number of candidate policies per update.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Override replay-buffer capacity; use 0 with reuse_fraction=0 for fresh-only runs.",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=sorted(LR_SCHEDULES),
        default=None,
        help=(
            "Learning-rate sequence. Existing configs default to exponential "
            "with their configured lr_decay (1.0 means constant)."
        ),
    )
    parser.add_argument(
        "--trust-radius",
        type=str,
        default=None,
        help="Override trust radius. Use 'none' to disable trust clipping.",
    )
    parser.add_argument("--reuse-fraction", type=float, default=None)
    parser.add_argument("--implicit-damping", type=float, default=None)
    parser.add_argument(
        "--curvature-mode",
        choices=("diag", "global", "block", "directional"),
        default=None,
        help="Override the curvature estimator for DIIWES variants.",
    )
    parser.add_argument(
        "--curvature-step-mode",
        choices=("dampen", "normalized"),
        default=None,
        help="Use either shrink-only damping or RMS-normalized curvature preconditioning.",
    )
    parser.add_argument("--curvature-beta", type=float, default=None)
    parser.add_argument(
        "--curvature-fitness",
        choices=("raw", "standardized"),
        default=None,
        help="Fitness scale used by the Stein curvature estimator.",
    )
    parser.add_argument("--evaluate-center-fitness", type=_parse_bool, default=None)
    parser.add_argument("--bias-correct-curvature-ema", type=_parse_bool, default=None)
    parser.add_argument("--leave-one-out-curvature-baseline", type=_parse_bool, default=None)
    parser.add_argument(
        "--rank-fitness",
        choices=("true", "false"),
        default=None,
        help="Override whether ES uses rank-shaped fitness for the policy gradient.",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--initial-params",
        default=None,
        help="Policy-gradient parameter checkpoint used as the shared ES center.",
    )
    parser.add_argument(
        "--initial-obs-norm",
        default=None,
        help="Observation-normalization state paired with --initial-params.",
    )
    parser.add_argument("--initial-params-sha256", default=None)
    parser.add_argument("--initial-obs-norm-sha256", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print detailed optimizer diagnostics while training.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config = _condition_config(config, args.condition)
    if args.population_size is not None:
        if args.population_size <= 0:
            raise ValueError("population_size must be positive")
        config["population_size"] = int(args.population_size)
    if args.buffer_size is not None:
        if args.buffer_size < 0:
            raise ValueError("buffer_size must be nonnegative")
        config["buffer_size"] = int(args.buffer_size)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.lr_schedule is not None:
        config["lr_schedule"] = args.lr_schedule
    if args.trust_radius is not None:
        config["trust_radius"] = _parse_optional_float(args.trust_radius)
    if args.reuse_fraction is not None:
        config["reuse_fraction"] = float(args.reuse_fraction)
    if args.implicit_damping is not None:
        config["implicit_damping"] = float(args.implicit_damping)
    if args.curvature_mode is not None:
        config["curvature_mode"] = args.curvature_mode
        if args.curvature_mode == "block":
            config.setdefault("block_structure", "layer")
    if args.curvature_step_mode is not None:
        config["curvature_step_mode"] = args.curvature_step_mode
    if args.curvature_beta is not None:
        config["curvature_beta"] = float(args.curvature_beta)
    if args.curvature_fitness is not None:
        config["curvature_fitness"] = args.curvature_fitness
    if args.evaluate_center_fitness is not None:
        config["evaluate_center_fitness"] = bool(args.evaluate_center_fitness)
    if args.bias_correct_curvature_ema is not None:
        config["bias_correct_curvature_ema"] = bool(args.bias_correct_curvature_ema)
    if args.leave_one_out_curvature_baseline is not None:
        config["use_leave_one_out_curvature_baseline"] = bool(args.leave_one_out_curvature_baseline)
    if args.rank_fitness is not None:
        config["rank_fitness"] = args.rank_fitness == "true"
    if args.iterations is not None:
        config["n_iterations"] = int(args.iterations)
    if args.initial_params is not None:
        config["initial_params_path"] = str(Path(args.initial_params).resolve())
    if args.initial_obs_norm is not None:
        config["initial_obs_norm_path"] = str(Path(args.initial_obs_norm).resolve())
    if args.initial_params_sha256 is not None:
        config["expected_initial_params_sha256"] = args.initial_params_sha256
    if args.initial_obs_norm_sha256 is not None:
        config["expected_initial_obs_norm_sha256"] = args.initial_obs_norm_sha256

    config.setdefault("lr_schedule", "exponential")
    config["initial_learning_rate"] = float(config.get("learning_rate", 0.02))
    actual_source_sha = _source_digest(args.config)
    expected_source_sha = os.environ.get("PAPER_EXPECTED_SOURCE_SHA")
    if expected_source_sha and actual_source_sha != expected_source_sha:
        raise RuntimeError(
            "source digest mismatch: "
            f"expected {expected_source_sha}, found {actual_source_sha}"
        )
    config["source_sha256"] = actual_source_sha

    if args.workers is None:
        n_workers = max(1, len(os.sched_getaffinity(0)) - 2)
    else:
        n_workers = int(args.workers)
    train(config, args.seed, args.output, n_workers, verbose=args.verbose)


if __name__ == "__main__":
    main()

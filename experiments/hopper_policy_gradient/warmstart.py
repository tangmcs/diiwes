#!/usr/bin/env python3
"""Train a NumPy PPO warm start compatible with the repository Hopper policy.

The actor has exactly the same 11-64-64-3 tanh parameter layout used by
``core.MLPPolicy``.  A squashed diagonal Gaussian is used during training;
the saved checkpoint is its deterministic tanh mean.  The implementation is
dependency-light because the cluster environment intentionally has no PyTorch
or Stable-Baselines3 installation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.policies import MLPPolicy  # noqa: E402
from utilities import ObsNormalizer  # noqa: E402


EXPERIMENT_VERSION = "1.0.0"
_WORKER_ENV: Any = None
_WORKER_MAX_STEPS = 1000


@dataclass(frozen=True)
class PPOConfig:
    """Locked policy-gradient initialization protocol."""

    env_name: str = "Hopper-v5"
    hidden_dims: tuple[int, ...] = (64, 64)
    max_episode_steps: int = 1000
    updates: int = 300
    batch_episodes: int = 32
    ppo_epochs: int = 6
    minibatch_size: int = 2048
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    max_grad_norm: float = 0.5
    action_std_start: float = 0.6
    action_std_end: float = 0.15
    eval_interval: int = 5
    eval_episodes: int = 10
    early_stop_return: float = 2000.0
    early_stop_patience: int = 3
    master_seed: int = 0

    def validate(self) -> None:
        if self.env_name != "Hopper-v5":
            raise ValueError("this locked warm start requires Hopper-v5")
        if self.hidden_dims != (64, 64):
            raise ValueError("hidden_dims must be exactly (64, 64)")
        for name in (
            "max_episode_steps",
            "updates",
            "batch_episodes",
            "ppo_epochs",
            "minibatch_size",
            "eval_interval",
            "eval_episodes",
            "early_stop_patience",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        for name in (
            "actor_learning_rate",
            "critic_learning_rate",
            "max_grad_norm",
            "action_std_start",
            "action_std_end",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.early_stop_return <= 0.0:
            raise ValueError("early_stop_return must be positive")
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")


@dataclass
class AdamState:
    mean: np.ndarray
    second: np.ndarray
    step: int = 0


def _dimensions(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> list[int]:
    return [int(input_dim), *(int(value) for value in hidden_dims), int(output_dim)]


def parameter_count(dimensions: Sequence[int]) -> int:
    return sum(
        int(dimensions[index]) * int(dimensions[index + 1])
        + int(dimensions[index + 1])
        for index in range(len(dimensions) - 1)
    )


def initialize_network(dimensions: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Xavier-initialize one tanh MLP in repository flattening order."""
    parts: list[np.ndarray] = []
    for index in range(len(dimensions) - 1):
        fan_in, fan_out = int(dimensions[index]), int(dimensions[index + 1])
        scale = math.sqrt(2.0 / (fan_in + fan_out))
        if index == len(dimensions) - 2:
            scale *= 0.1
        parts.extend(
            (
                rng.normal(scale=scale, size=(fan_in, fan_out)).ravel(),
                np.zeros(fan_out, dtype=np.float64),
            )
        )
    return np.concatenate(parts).astype(np.float64, copy=False)


def unpack_network(
    params: np.ndarray, dimensions: Sequence[int]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return read-only views of flattened weights and biases."""
    values = np.asarray(params, dtype=np.float64)
    if values.shape != (parameter_count(dimensions),):
        raise ValueError("network parameter shape does not match dimensions")
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for index in range(len(dimensions) - 1):
        fan_in, fan_out = int(dimensions[index]), int(dimensions[index + 1])
        weight_size = fan_in * fan_out
        weight = values[cursor : cursor + weight_size].reshape(fan_in, fan_out)
        cursor += weight_size
        bias = values[cursor : cursor + fan_out]
        cursor += fan_out
        layers.append((weight, bias))
    return layers


def network_forward(
    params: np.ndarray,
    inputs: np.ndarray,
    dimensions: Sequence[int],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Evaluate a hidden-tanh, linear-output network and retain activations."""
    current = np.asarray(inputs, dtype=np.float64)
    one_row = current.ndim == 1
    if one_row:
        current = current[None, :]
    if current.ndim != 2 or current.shape[1] != int(dimensions[0]):
        raise ValueError("network inputs have the wrong shape")
    activations = [current]
    for index, (weight, bias) in enumerate(unpack_network(params, dimensions)):
        current = current @ weight + bias
        if index < len(dimensions) - 2:
            current = np.tanh(current)
        activations.append(current)
    return (current[0] if one_row else current), activations


def network_gradient(
    params: np.ndarray,
    inputs: np.ndarray,
    output_gradient: np.ndarray,
    dimensions: Sequence[int],
) -> np.ndarray:
    """Backpropagate a supplied scalar-objective gradient through the MLP."""
    output, activations = network_forward(params, inputs, dimensions)
    delta = np.asarray(output_gradient, dtype=np.float64)
    if output.ndim == 1:
        delta = delta.reshape(1, -1)
    if delta.shape != activations[-1].shape:
        raise ValueError("output_gradient has the wrong shape")
    layers = unpack_network(params, dimensions)
    gradients: list[tuple[np.ndarray, np.ndarray]] = [
        (np.empty(0), np.empty(0)) for _ in layers
    ]
    for index in range(len(layers) - 1, -1, -1):
        weight, _ = layers[index]
        gradients[index] = (
            activations[index].T @ delta,
            np.sum(delta, axis=0),
        )
        if index > 0:
            delta = (delta @ weight.T) * (1.0 - activations[index] ** 2)
    return np.concatenate(
        [part.ravel() for pair in gradients for part in pair]
    ).astype(np.float64, copy=False)


def gaussian_log_probability(
    latent_actions: np.ndarray,
    means: np.ndarray,
    action_std: float,
) -> np.ndarray:
    """Log density before tanh; its Jacobian cancels in PPO ratios."""
    latent = np.asarray(latent_actions, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variance = float(action_std) ** 2
    return -0.5 * np.sum(
        (latent - means) ** 2 / variance
        + math.log(2.0 * math.pi * variance),
        axis=-1,
    )


def ppo_actor_gradient(
    params: np.ndarray,
    observations: np.ndarray,
    latent_actions: np.ndarray,
    old_log_probabilities: np.ndarray,
    advantages: np.ndarray,
    action_std: float,
    clip_ratio: float,
    dimensions: Sequence[int],
) -> tuple[np.ndarray, dict[str, float]]:
    """Return the gradient of the clipped PPO objective for one minibatch."""
    means, _ = network_forward(params, observations, dimensions)
    new_log_probabilities = gaussian_log_probability(
        latent_actions, means, action_std
    )
    log_ratio = np.clip(new_log_probabilities - old_log_probabilities, -20.0, 20.0)
    ratios = np.exp(log_ratio)
    advantages = np.asarray(advantages, dtype=np.float64)
    active = np.where(
        advantages >= 0.0,
        ratios <= 1.0 + clip_ratio,
        ratios >= 1.0 - clip_ratio,
    )
    coefficient = np.where(active, advantages * ratios, 0.0)
    output_gradient = (
        coefficient[:, None]
        * (np.asarray(latent_actions) - means)
        / (float(action_std) ** 2)
        / len(observations)
    )
    gradient = network_gradient(
        params, observations, output_gradient, dimensions
    )
    clipped_ratios = np.clip(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio)
    objective = float(np.mean(np.minimum(ratios * advantages, clipped_ratios * advantages)))
    diagnostics = {
        "actor_objective": objective,
        "approx_kl": float(np.mean(old_log_probabilities - new_log_probabilities)),
        "clip_fraction": float(np.mean(~active)),
    }
    return gradient, diagnostics


def _clip_gradient(gradient: np.ndarray, maximum: float) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(gradient))
    if norm > maximum:
        gradient = gradient * (float(maximum) / (norm + 1e-12))
    return gradient, norm


def _adam_update(
    params: np.ndarray,
    gradient: np.ndarray,
    state: AdamState,
    learning_rate: float,
    *,
    ascent: bool,
) -> np.ndarray:
    state.step += 1
    state.mean *= 0.9
    state.mean += 0.1 * gradient
    state.second *= 0.999
    state.second += 0.001 * gradient * gradient
    corrected_mean = state.mean / (1.0 - 0.9**state.step)
    corrected_second = state.second / (1.0 - 0.999**state.step)
    step = float(learning_rate) * corrected_mean / (
        np.sqrt(corrected_second) + 1e-8
    )
    return params + step if ascent else params - step


def _update_normalizer_from_moments(
    normalizer: ObsNormalizer,
    observation_sum: np.ndarray,
    observation_square_sum: np.ndarray,
    observation_count: int,
) -> None:
    if observation_count <= 0:
        return
    batch_mean = observation_sum / observation_count
    batch_var = np.maximum(
        observation_square_sum / observation_count - batch_mean**2,
        0.0,
    )
    delta = batch_mean - normalizer.mean
    total_count = normalizer.count + observation_count
    combined = (
        normalizer.var * normalizer.count
        + batch_var * observation_count
        + delta**2 * normalizer.count * observation_count / total_count
    )
    normalizer.mean = normalizer.mean + delta * observation_count / total_count
    normalizer.var = combined / total_count
    normalizer.count = float(total_count)


def _worker_init(env_name: str, max_steps: int) -> None:
    global _WORKER_ENV, _WORKER_MAX_STEPS
    import gymnasium as gym

    _WORKER_ENV = gym.make(env_name)
    _WORKER_MAX_STEPS = int(max_steps)


def _collect_rollout(
    task: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        float,
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
    ]
) -> dict[str, Any]:
    (
        actor_params,
        critic_params,
        obs_mean,
        obs_var,
        action_std,
        env_seed,
        action_seed,
        actor_dimensions,
        critic_dimensions,
    ) = task
    observation, _ = _WORKER_ENV.reset(seed=int(env_seed))
    if hasattr(_WORKER_ENV.action_space, "seed"):
        _WORKER_ENV.action_space.seed(int(env_seed))
    action_rng = np.random.default_rng(int(action_seed))
    observations: list[np.ndarray] = []
    latent_actions: list[np.ndarray] = []
    log_probabilities: list[float] = []
    rewards: list[float] = []
    values: list[float] = []
    observation_sum = np.zeros_like(obs_mean, dtype=np.float64)
    observation_square_sum = np.zeros_like(obs_mean, dtype=np.float64)
    for _ in range(_WORKER_MAX_STEPS):
        raw = np.asarray(observation, dtype=np.float64).ravel()
        normalized = (raw - obs_mean) / np.sqrt(obs_var + 1e-8)
        mean, _ = network_forward(actor_params, normalized, actor_dimensions)
        value, _ = network_forward(critic_params, normalized, critic_dimensions)
        latent = mean + float(action_std) * action_rng.standard_normal(mean.shape)
        action = np.tanh(latent)
        action = np.clip(action, _WORKER_ENV.action_space.low, _WORKER_ENV.action_space.high)
        next_observation, reward, terminated, truncated, _ = _WORKER_ENV.step(action)
        observations.append(normalized)
        latent_actions.append(latent)
        log_probabilities.append(
            float(gaussian_log_probability(latent[None, :], mean[None, :], action_std)[0])
        )
        rewards.append(float(reward))
        values.append(float(np.asarray(value).item()))
        observation_sum += raw
        observation_square_sum += raw * raw
        observation = next_observation
        if terminated or truncated:
            break
    return {
        "observations": np.asarray(observations, dtype=np.float64),
        "latent_actions": np.asarray(latent_actions, dtype=np.float64),
        "old_log_probabilities": np.asarray(log_probabilities, dtype=np.float64),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "values": np.asarray(values, dtype=np.float64),
        "episode_return": float(np.sum(rewards)),
        "observation_sum": observation_sum,
        "observation_square_sum": observation_square_sum,
        "observation_count": len(observations),
    }


def _evaluate_rollout(
    task: tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, ...]]
) -> float:
    actor_params, obs_mean, obs_var, env_seed, actor_dimensions = task
    observation, _ = _WORKER_ENV.reset(seed=int(env_seed))
    total = 0.0
    for _ in range(_WORKER_MAX_STEPS):
        raw = np.asarray(observation, dtype=np.float64).ravel()
        normalized = (raw - obs_mean) / np.sqrt(obs_var + 1e-8)
        mean, _ = network_forward(actor_params, normalized, actor_dimensions)
        action = np.tanh(mean)
        observation, reward, terminated, truncated, _ = _WORKER_ENV.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def _advantages_and_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.empty_like(rewards)
    running = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * gae_lambda * running
        advantages[index] = running
        next_value = values[index]
    return advantages, advantages + values


def _action_std(config: PPOConfig, update: int) -> float:
    fraction = min(max((update - 1) / max(config.updates - 1, 1), 0.0), 1.0)
    return float(
        config.action_std_start
        * (config.action_std_end / config.action_std_start) ** fraction
    )


def _seed(master: int, *parts: int) -> int:
    sequence = np.random.SeedSequence([int(master), *(int(part) for part in parts)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _evaluate(
    pool: Pool,
    actor_params: np.ndarray,
    normalizer: ObsNormalizer,
    actor_dimensions: tuple[int, ...],
    config: PPOConfig,
) -> float:
    tasks = [
        (
            actor_params,
            normalizer.mean,
            normalizer.var,
            _seed(config.master_seed, 900, episode),
            actor_dimensions,
        )
        for episode in range(config.eval_episodes)
    ]
    returns = pool.map(_evaluate_rollout, tasks)
    return float(np.mean(returns))


def _write_history(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest() -> str:
    """Hash every repository source file that defines the saved checkpoint."""
    repository = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repository / "core" / "policies.py",
        repository / "utilities" / "obs_norm.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(repository)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def train_warmstart(
    config: PPOConfig,
    output_dir: str | os.PathLike[str],
    workers: int,
) -> dict[str, Any]:
    """Run PPO and save the best deterministic actor and matching normalizer."""
    config.validate()
    if workers < 1:
        raise ValueError("workers must be positive")
    actual_source_digest = source_digest()
    expected_source_digest = os.environ.get("PAPER_EXPECTED_PG_SOURCE_SHA")
    if expected_source_digest and actual_source_digest != expected_source_digest:
        raise RuntimeError(
            "policy-gradient source digest mismatch: "
            f"expected {expected_source_digest}, found {actual_source_digest}"
        )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

    import gymnasium as gym

    probe = gym.make(config.env_name)
    observation_dim = int(np.prod(probe.observation_space.shape))
    action_dim = int(np.prod(probe.action_space.shape))
    probe.close()
    actor_dimensions = tuple(
        _dimensions(observation_dim, config.hidden_dims, action_dim)
    )
    critic_dimensions = tuple(
        _dimensions(observation_dim, config.hidden_dims, 1)
    )
    policy = MLPPolicy(
        observation_dim,
        action_dim,
        hidden_dims=config.hidden_dims,
        activation="tanh",
        output_activation="tanh",
    )
    if policy.num_params != parameter_count(actor_dimensions):
        raise RuntimeError("PPO actor layout does not match core.MLPPolicy")

    rng = np.random.default_rng(_seed(config.master_seed, 1))
    actor_params = initialize_network(actor_dimensions, rng)
    critic_params = initialize_network(critic_dimensions, rng)
    actor_adam = AdamState(np.zeros_like(actor_params), np.zeros_like(actor_params))
    critic_adam = AdamState(np.zeros_like(critic_params), np.zeros_like(critic_params))
    normalizer = ObsNormalizer((observation_dim,))
    best_params = actor_params.copy()
    best_normalizer = normalizer.get_state()
    best_eval = -math.inf
    target_hits = 0
    history: list[dict[str, Any]] = []

    pool = Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(config.env_name, config.max_episode_steps),
    )
    try:
        initial_eval = _evaluate(
            pool, actor_params, normalizer, actor_dimensions, config
        )
        best_eval = initial_eval
        history.append(
            {
                "update": 0,
                "eval_return": initial_eval,
                "batch_mean_return": "",
                "batch_max_return": "",
                "batch_steps": 0,
                "action_std": config.action_std_start,
                "actor_objective": "",
                "critic_loss": "",
                "approx_kl": "",
                "clip_fraction": "",
                "actor_gradient_norm": 0.0,
                "critic_gradient_norm": 0.0,
                "parameter_norm": float(np.linalg.norm(actor_params)),
            }
        )
        print(f"PPO update 0 | eval {initial_eval:.2f}", flush=True)

        for update in range(1, config.updates + 1):
            action_std = _action_std(config, update)
            tasks = [
                (
                    actor_params,
                    critic_params,
                    normalizer.mean,
                    normalizer.var,
                    action_std,
                    _seed(config.master_seed, 100, update, episode),
                    _seed(config.master_seed, 101, update, episode),
                    actor_dimensions,
                    critic_dimensions,
                )
                for episode in range(config.batch_episodes)
            ]
            rollouts = pool.map(_collect_rollout, tasks)
            observation_sum = np.sum(
                [row["observation_sum"] for row in rollouts], axis=0
            )
            square_sum = np.sum(
                [row["observation_square_sum"] for row in rollouts], axis=0
            )
            observation_count = int(
                sum(row["observation_count"] for row in rollouts)
            )

            advantages_parts: list[np.ndarray] = []
            returns_parts: list[np.ndarray] = []
            for row in rollouts:
                advantages, returns = _advantages_and_returns(
                    row["rewards"],
                    row["values"],
                    config.gamma,
                    config.gae_lambda,
                )
                advantages_parts.append(advantages)
                returns_parts.append(returns)
            observations = np.concatenate(
                [row["observations"] for row in rollouts], axis=0
            )
            latent_actions = np.concatenate(
                [row["latent_actions"] for row in rollouts], axis=0
            )
            old_log_probabilities = np.concatenate(
                [row["old_log_probabilities"] for row in rollouts], axis=0
            )
            advantages = np.concatenate(advantages_parts)
            returns = np.concatenate(returns_parts)
            advantages = (advantages - np.mean(advantages)) / (
                np.std(advantages) + 1e-8
            )

            last_actor_diagnostics = {
                "actor_objective": 0.0,
                "approx_kl": 0.0,
                "clip_fraction": 0.0,
            }
            actor_gradient_norm = 0.0
            critic_gradient_norm = 0.0
            critic_loss = 0.0
            sample_count = len(observations)
            for epoch in range(config.ppo_epochs):
                order_rng = np.random.default_rng(
                    _seed(config.master_seed, 200, update, epoch)
                )
                order = order_rng.permutation(sample_count)
                for start in range(0, sample_count, config.minibatch_size):
                    indices = order[start : start + config.minibatch_size]
                    actor_gradient, last_actor_diagnostics = ppo_actor_gradient(
                        actor_params,
                        observations[indices],
                        latent_actions[indices],
                        old_log_probabilities[indices],
                        advantages[indices],
                        action_std,
                        config.clip_ratio,
                        actor_dimensions,
                    )
                    actor_gradient, actor_gradient_norm = _clip_gradient(
                        actor_gradient, config.max_grad_norm
                    )
                    actor_params = _adam_update(
                        actor_params,
                        actor_gradient,
                        actor_adam,
                        config.actor_learning_rate,
                        ascent=True,
                    )

                    predicted, _ = network_forward(
                        critic_params, observations[indices], critic_dimensions
                    )
                    predicted = np.asarray(predicted).reshape(-1)
                    residual = predicted - returns[indices]
                    critic_loss = float(0.5 * np.mean(residual**2))
                    critic_gradient = network_gradient(
                        critic_params,
                        observations[indices],
                        (residual / len(indices))[:, None],
                        critic_dimensions,
                    )
                    critic_gradient, critic_gradient_norm = _clip_gradient(
                        critic_gradient, config.max_grad_norm
                    )
                    critic_params = _adam_update(
                        critic_params,
                        critic_gradient,
                        critic_adam,
                        config.critic_learning_rate,
                        ascent=False,
                    )

            if not np.all(np.isfinite(actor_params)):
                raise FloatingPointError("PPO actor parameters became non-finite")
            if not np.all(np.isfinite(critic_params)):
                raise FloatingPointError("PPO critic parameters became non-finite")

            _update_normalizer_from_moments(
                normalizer, observation_sum, square_sum, observation_count
            )
            should_evaluate = update % config.eval_interval == 0 or update == config.updates
            eval_return: float | str = ""
            if should_evaluate:
                eval_return = _evaluate(
                    pool, actor_params, normalizer, actor_dimensions, config
                )
                if float(eval_return) > best_eval:
                    best_eval = float(eval_return)
                    best_params = actor_params.copy()
                    best_normalizer = normalizer.get_state()
                if float(eval_return) >= config.early_stop_return:
                    target_hits += 1
                else:
                    target_hits = 0
            batch_returns = [row["episode_return"] for row in rollouts]
            history.append(
                {
                    "update": update,
                    "eval_return": eval_return,
                    "batch_mean_return": float(np.mean(batch_returns)),
                    "batch_max_return": float(np.max(batch_returns)),
                    "batch_steps": sample_count,
                    "action_std": action_std,
                    "actor_objective": last_actor_diagnostics["actor_objective"],
                    "critic_loss": critic_loss,
                    "approx_kl": last_actor_diagnostics["approx_kl"],
                    "clip_fraction": last_actor_diagnostics["clip_fraction"],
                    "actor_gradient_norm": actor_gradient_norm,
                    "critic_gradient_norm": critic_gradient_norm,
                    "parameter_norm": float(np.linalg.norm(actor_params)),
                }
            )
            print(
                f"PPO update {update:3d} | batch {np.mean(batch_returns):8.2f} "
                f"| eval {eval_return if eval_return != '' else '-'} "
                f"| best {best_eval:.2f} | steps {sample_count}",
                flush=True,
            )
            if target_hits >= config.early_stop_patience:
                break
    finally:
        pool.close()
        pool.join()

    params_path = output / "policy_params.npy"
    obs_path = output / "obs_norm.npz"
    history_csv = output / "history.csv"
    history_json = output / "history.json"
    np.save(params_path, best_params)
    np.savez(obs_path, **best_normalizer)
    _write_history(history_csv, history)
    history_json.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_sha256": actual_source_digest,
        "status": "complete",
        "method": "numpy_ppo_policy_gradient",
        "environment": config.env_name,
        "config": asdict(config),
        "observation_dimension": observation_dim,
        "action_dimension": action_dim,
        "actor_dimensions": list(actor_dimensions),
        "actor_parameter_count": len(best_params),
        "best_eval_return": best_eval,
        "updates_completed": int(history[-1]["update"]),
        "policy_params": params_path.name,
        "policy_params_sha256": _sha256(params_path),
        "obs_norm": obs_path.name,
        "obs_norm_sha256": _sha256(obs_path),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "workers": workers,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--updates", type=int, default=300)
    parser.add_argument("--batch-episodes", type=int, default=32)
    parser.add_argument("--ppo-epochs", type=int, default=6)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--master-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = PPOConfig(
        updates=args.updates,
        batch_episodes=args.batch_episodes,
        ppo_epochs=args.ppo_epochs,
        eval_episodes=args.eval_episodes,
        master_seed=args.master_seed,
    )
    train_warmstart(config, args.output_dir, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

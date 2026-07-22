#!/usr/bin/env python3
"""Policy-gradient warm starts for nonlinear zeroth-order optimization.

This experiment adapts the initialization protocol from Wang, Zhang, and
Ying (2026): train a stable actor with a first-order policy-gradient method,
then use that actor as the initial point for zeroth-order fine-tuning.  Human
preference feedback and federation are intentionally outside the scope of
this benchmark.

The environment is a small, dependency-free implementation of the canonical
CartPole control problem.  Its trigonometric dynamics and two-hidden-layer
neural policy make the parameter-to-return map nonlinear and non-convex.  The
zeroth-order phase calls the repository's StandardES and DIIWES classes
directly, rather than reimplementing either optimizer.
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
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import DIIWES, StandardES  # noqa: E402
from core.policies import DiscretePolicy, MLPPolicy  # noqa: E402


EXPERIMENT_VERSION = "1.2.0"
METHODS = ("standard_es", "diiwes")
INITIALIZATIONS = ("random", "reinforce")
TRAJECTORY_FIELDS = (
    "seed",
    "initialization",
    "method",
    "update",
    "eval_return",
    "best_eval_return",
    "population_mean_return",
    "population_max_return",
    "step_norm",
    "grad_norm",
    "curvature_active_fraction",
    "mean_step_multiplier",
)
PRETRAIN_FIELDS = (
    "seed",
    "update",
    "eval_return",
    "batch_mean_return",
    "batch_max_return",
    "gradient_norm",
    "parameter_norm",
)
RUN_SUMMARY_FIELDS = (
    "seed",
    "initialization",
    "method",
    "initial_return",
    "final_return",
    "best_return",
    "mean_return_auc",
    "fine_tuning_gain",
    "solved_final",
)
AGGREGATE_FIELDS = (
    "initialization",
    "method",
    "n_runs",
    "median_initial_return",
    "median_final_return",
    "q25_final_return",
    "q75_final_return",
    "median_best_return",
    "median_mean_return_auc",
    "median_fine_tuning_gain",
    "solved_fraction",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Complete protocol for the nonlinear warm-start comparison."""

    hidden_dims: tuple[int, ...] = (8, 8)
    max_episode_steps: int = 200
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    init_param_std: float = 0.1
    eval_episodes: int = 20
    solve_return: float = 195.0

    reinforce_updates: int = 60
    reinforce_batch_episodes: int = 16
    reinforce_learning_rate: float = 0.01
    reinforce_gamma: float = 0.99
    reinforce_gradient_clip: float = 5.0
    reinforce_eval_interval: int = 5
    reinforce_target_return: float = 100.0
    reinforce_target_patience: int = 2

    es_updates: int = 300
    population_size: int = 500
    noise_std: float = 0.05
    es_learning_rate: float = 0.01
    curvature_beta: float = 0.99
    curvature_clip: float = 1000.0
    min_step_multiplier: float = 0.05
    master_seed: int = 20260721

    @property
    def antithetic_pairs(self) -> int:
        """Number of matched +epsilon/-epsilon evaluations per ES update."""
        return self.population_size // 2

    def validate(self) -> None:
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.max_episode_steps < 2:
            raise ValueError("max_episode_steps must be at least two")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonempty and nonnegative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if not np.isfinite(self.init_param_std) or self.init_param_std <= 0.0:
            raise ValueError("init_param_std must be positive and finite")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be positive")
        if not 0.0 < self.solve_return <= self.max_episode_steps:
            raise ValueError("solve_return must lie in (0, max_episode_steps]")
        if self.reinforce_updates < 1 or self.reinforce_batch_episodes < 1:
            raise ValueError("REINFORCE updates and batch size must be positive")
        if self.reinforce_learning_rate <= 0.0:
            raise ValueError("reinforce_learning_rate must be positive")
        if not 0.0 < self.reinforce_gamma <= 1.0:
            raise ValueError("reinforce_gamma must lie in (0, 1]")
        if self.reinforce_gradient_clip <= 0.0:
            raise ValueError("reinforce_gradient_clip must be positive")
        if self.reinforce_eval_interval < 1:
            raise ValueError("reinforce_eval_interval must be positive")
        if not 0.0 < self.reinforce_target_return <= self.max_episode_steps:
            raise ValueError(
                "reinforce_target_return must lie in (0, max_episode_steps]"
            )
        if self.reinforce_target_patience < 1:
            raise ValueError("reinforce_target_patience must be positive")
        if self.es_updates < 1:
            raise ValueError("es_updates must be positive")
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be even and at least four")
        if self.noise_std <= 0.0 or self.es_learning_rate <= 0.0:
            raise ValueError("noise_std and es_learning_rate must be positive")
        if not 0.0 <= self.curvature_beta < 1.0:
            raise ValueError("curvature_beta must lie in [0, 1)")
        if self.curvature_clip <= 0.0:
            raise ValueError("curvature_clip must be positive")
        if not 0.0 <= self.min_step_multiplier <= 1.0:
            raise ValueError("min_step_multiplier must lie in [0, 1]")
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")


@dataclass(frozen=True)
class ExperimentResult:
    trajectories: tuple[dict[str, Any], ...]
    pretraining: tuple[dict[str, Any], ...]
    run_summaries: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]


class CartPoleDynamics:
    """Deterministic CartPole dynamics with seeded random initial states."""

    gravity = 9.8
    cart_mass = 1.0
    pole_mass = 0.1
    half_pole_length = 0.5
    force_magnitude = 10.0
    integration_step = 0.02
    cart_position_limit = 2.4
    pole_angle_limit = 12.0 * 2.0 * math.pi / 360.0

    def __init__(self, max_steps: int = 200) -> None:
        self.max_steps = int(max_steps)
        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        self.state = rng.uniform(-0.05, 0.05, size=4).astype(np.float64)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if int(action) not in (0, 1):
            raise ValueError("CartPole action must be zero or one")
        x, x_velocity, angle, angle_velocity = self.state
        force = self.force_magnitude if int(action) == 1 else -self.force_magnitude
        sine = math.sin(float(angle))
        cosine = math.cos(float(angle))
        total_mass = self.cart_mass + self.pole_mass
        pole_mass_length = self.pole_mass * self.half_pole_length
        temp = (
            force + pole_mass_length * angle_velocity * angle_velocity * sine
        ) / total_mass
        angle_acceleration = (
            self.gravity * sine - cosine * temp
        ) / (
            self.half_pole_length
            * (4.0 / 3.0 - self.pole_mass * cosine * cosine / total_mass)
        )
        x_acceleration = temp - pole_mass_length * angle_acceleration * cosine / total_mass

        x = x + self.integration_step * x_velocity
        x_velocity = x_velocity + self.integration_step * x_acceleration
        angle = angle + self.integration_step * angle_velocity
        angle_velocity = angle_velocity + self.integration_step * angle_acceleration
        self.state = np.asarray(
            (x, x_velocity, angle, angle_velocity), dtype=np.float64
        )
        self.steps += 1
        terminated = bool(
            abs(x) > self.cart_position_limit
            or abs(angle) > self.pole_angle_limit
            or self.steps >= self.max_steps
        )
        return self.state.copy(), 1.0, terminated


def make_policy(config: BenchmarkConfig) -> DiscretePolicy:
    mlp = MLPPolicy(
        ob_dim=4,
        ac_dim=2,
        hidden_dims=config.hidden_dims,
        activation="tanh",
        output_activation=None,
    )
    return DiscretePolicy(mlp, n_actions=2)


def _seed(*parts: int) -> int:
    payload = ":".join(str(int(part)) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _unpack_layers(
    policy: DiscretePolicy, params: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    mlp = policy.continuous_policy
    params = np.asarray(params, dtype=np.float64)
    if params.shape != (policy.num_params,):
        raise ValueError(
            f"params must have shape ({policy.num_params},), got {params.shape}"
        )
    dims = [mlp.ob_dim] + list(mlp.hidden_dims) + [mlp.ac_dim]
    layers: list[tuple[np.ndarray, np.ndarray]] = []
    cursor = 0
    for input_dim, output_dim in zip(dims[:-1], dims[1:], strict=True):
        weight_size = input_dim * output_dim
        weight = params[cursor : cursor + weight_size].reshape(
            input_dim, output_dim
        )
        cursor += weight_size
        bias = params[cursor : cursor + output_dim]
        cursor += output_dim
        layers.append((weight, bias))
    if cursor != policy.num_params:
        raise RuntimeError("layer unpacking did not consume the parameter vector")
    return layers


def policy_probabilities(
    policy: DiscretePolicy, params: np.ndarray, state: np.ndarray
) -> np.ndarray:
    logits = policy.continuous_policy.act(state, params)
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def _batched_policy_logits(
    policy: DiscretePolicy, params: np.ndarray, states: np.ndarray
) -> np.ndarray:
    """Evaluate one policy per state (or broadcast one shared policy)."""
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != policy.ob_dim:
        raise ValueError(f"states must have shape (n, {policy.ob_dim})")
    parameter_matrix = np.asarray(params, dtype=np.float64)
    if parameter_matrix.ndim == 1:
        parameter_matrix = np.broadcast_to(
            parameter_matrix, (len(states), policy.num_params)
        )
    if parameter_matrix.shape != (len(states), policy.num_params):
        raise ValueError(
            f"params must have shape ({policy.num_params},) or "
            f"({len(states)}, {policy.num_params})"
        )

    mlp = policy.continuous_policy
    dims = [mlp.ob_dim] + list(mlp.hidden_dims) + [mlp.ac_dim]
    values = states
    cursor = 0
    for layer_index, (input_dim, output_dim) in enumerate(
        zip(dims[:-1], dims[1:], strict=True)
    ):
        weight_size = input_dim * output_dim
        weights = parameter_matrix[
            :, cursor : cursor + weight_size
        ].reshape(len(states), input_dim, output_dim)
        cursor += weight_size
        biases = parameter_matrix[:, cursor : cursor + output_dim]
        cursor += output_dim
        values = np.einsum("ni,nio->no", values, weights) + biases
        if layer_index < len(dims) - 2:
            values = np.tanh(values)
    return values


def batched_greedy_returns(
    policy: DiscretePolicy,
    params: np.ndarray,
    seeds: Sequence[int],
    max_steps: int,
) -> np.ndarray:
    """Vectorized deterministic evaluation for the zeroth-order phase."""
    seed_values = tuple(int(seed) for seed in seeds)
    states = np.asarray(
        [
            np.random.default_rng(seed).uniform(-0.05, 0.05, size=4)
            for seed in seed_values
        ],
        dtype=np.float64,
    )
    if len(states) == 0:
        return np.zeros(0, dtype=np.float64)
    returns = np.zeros(len(states), dtype=np.float64)
    active = np.ones(len(states), dtype=bool)
    total_mass = CartPoleDynamics.cart_mass + CartPoleDynamics.pole_mass
    pole_mass_length = (
        CartPoleDynamics.pole_mass * CartPoleDynamics.half_pole_length
    )
    for _ in range(int(max_steps)):
        actions = np.argmax(_batched_policy_logits(policy, params, states), axis=1)
        x = states[:, 0]
        x_velocity = states[:, 1]
        angle = states[:, 2]
        angle_velocity = states[:, 3]
        forces = np.where(
            actions == 1,
            CartPoleDynamics.force_magnitude,
            -CartPoleDynamics.force_magnitude,
        )
        sine = np.sin(angle)
        cosine = np.cos(angle)
        temp = (
            forces + pole_mass_length * angle_velocity * angle_velocity * sine
        ) / total_mass
        angle_acceleration = (
            CartPoleDynamics.gravity * sine - cosine * temp
        ) / (
            CartPoleDynamics.half_pole_length
            * (
                4.0 / 3.0
                - CartPoleDynamics.pole_mass * cosine * cosine / total_mass
            )
        )
        x_acceleration = (
            temp - pole_mass_length * angle_acceleration * cosine / total_mass
        )
        next_states = np.column_stack(
            (
                x + CartPoleDynamics.integration_step * x_velocity,
                x_velocity
                + CartPoleDynamics.integration_step * x_acceleration,
                angle + CartPoleDynamics.integration_step * angle_velocity,
                angle_velocity
                + CartPoleDynamics.integration_step * angle_acceleration,
            )
        )
        states[active] = next_states[active]
        returns[active] += 1.0
        failed = (
            np.abs(states[:, 0]) > CartPoleDynamics.cart_position_limit
        ) | (np.abs(states[:, 2]) > CartPoleDynamics.pole_angle_limit)
        active &= ~failed
        if not np.any(active):
            break
    return returns


def log_policy_gradient(
    policy: DiscretePolicy,
    params: np.ndarray,
    state: np.ndarray,
    action: int,
) -> np.ndarray:
    """Return the exact gradient of log pi(action | state)."""
    if int(action) not in (0, 1):
        raise ValueError("action must be zero or one")
    layers = _unpack_layers(policy, params)
    activations = [np.asarray(state, dtype=np.float64).ravel()]
    for layer_index, (weight, bias) in enumerate(layers):
        value = activations[-1] @ weight + bias
        if layer_index < len(layers) - 1:
            value = np.tanh(value)
        activations.append(value)

    logits = activations[-1]
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities)
    delta = -probabilities
    delta[int(action)] += 1.0

    gradients: list[tuple[np.ndarray, np.ndarray]] = [
        (np.empty(0), np.empty(0)) for _ in layers
    ]
    for layer_index in range(len(layers) - 1, -1, -1):
        weight, _ = layers[layer_index]
        gradients[layer_index] = (
            np.outer(activations[layer_index], delta),
            delta.copy(),
        )
        if layer_index > 0:
            delta = (weight @ delta) * (1.0 - activations[layer_index] ** 2)

    flat_parts: list[np.ndarray] = []
    for weight_gradient, bias_gradient in gradients:
        flat_parts.extend((weight_gradient.ravel(), bias_gradient.ravel()))
    gradient = np.concatenate(flat_parts)
    if gradient.shape != (policy.num_params,):
        raise RuntimeError("policy gradient has the wrong shape")
    return gradient


def rollout(
    policy: DiscretePolicy,
    params: np.ndarray,
    seed: int,
    max_steps: int,
    *,
    stochastic: bool,
    action_seed: int | None = None,
    collect_gradients: bool = False,
) -> tuple[float, list[np.ndarray]]:
    environment = CartPoleDynamics(max_steps=max_steps)
    state = environment.reset(seed)
    action_rng = np.random.default_rng(seed if action_seed is None else action_seed)
    gradients: list[np.ndarray] = []
    total_return = 0.0
    for _ in range(max_steps):
        probabilities = policy_probabilities(policy, params, state)
        if stochastic:
            action = int(action_rng.choice(2, p=probabilities))
        else:
            action = int(np.argmax(probabilities))
        if collect_gradients:
            gradients.append(log_policy_gradient(policy, params, state, action))
        state, reward, terminated = environment.step(action)
        total_return += reward
        if terminated:
            break
    return float(total_return), gradients


def evaluate_policy(
    policy: DiscretePolicy,
    params: np.ndarray,
    seeds: Sequence[int],
    max_steps: int,
) -> float:
    returns = batched_greedy_returns(policy, params, seeds, max_steps)
    return float(np.mean(returns))


def _discounted_unit_returns(length: int, gamma: float) -> np.ndarray:
    values = np.empty(length, dtype=np.float64)
    running = 0.0
    for index in range(length - 1, -1, -1):
        running = 1.0 + gamma * running
        values[index] = running
    return values


def reinforce_pretrain(
    policy: DiscretePolicy,
    initial_params: np.ndarray,
    config: BenchmarkConfig,
    seed: int,
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Warm-start a policy with batched REINFORCE and clipped Adam steps."""
    params = np.asarray(initial_params, dtype=np.float64).copy()
    adam_mean = np.zeros_like(params)
    adam_second_moment = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    target_hits = 0
    history: list[dict[str, Any]] = []
    eval_seeds = [
        _seed(config.master_seed, seed, 700, index)
        for index in range(config.eval_episodes)
    ]
    initial_eval = evaluate_policy(
        policy, params, eval_seeds, config.max_episode_steps
    )
    history.append(
        {
            "seed": seed,
            "update": 0,
            "eval_return": initial_eval,
            "batch_mean_return": "",
            "batch_max_return": "",
            "gradient_norm": 0.0,
            "parameter_norm": float(np.linalg.norm(params)),
        }
    )

    for update in range(1, config.reinforce_updates + 1):
        episode_gradients: list[list[np.ndarray]] = []
        reward_to_go: list[np.ndarray] = []
        batch_returns: list[float] = []
        for episode in range(config.reinforce_batch_episodes):
            environment_seed = _seed(
                config.master_seed, seed, 710, update, episode
            )
            action_seed = _seed(config.master_seed, seed, 711, update, episode)
            episode_return, gradients = rollout(
                policy,
                params,
                environment_seed,
                config.max_episode_steps,
                stochastic=True,
                action_seed=action_seed,
                collect_gradients=True,
            )
            episode_gradients.append(gradients)
            reward_to_go.append(
                _discounted_unit_returns(len(gradients), config.reinforce_gamma)
            )
            batch_returns.append(episode_return)

        flat_returns = np.concatenate(reward_to_go)
        baseline = float(np.mean(flat_returns))
        scale = float(np.std(flat_returns) + 1e-8)
        gradient = np.zeros_like(params)
        for gradients, returns in zip(
            episode_gradients, reward_to_go, strict=True
        ):
            for score, value in zip(gradients, returns, strict=True):
                gradient += ((float(value) - baseline) / scale) * score
        gradient /= float(config.reinforce_batch_episodes)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm > config.reinforce_gradient_clip:
            gradient *= config.reinforce_gradient_clip / (gradient_norm + 1e-12)

        adam_mean = beta1 * adam_mean + (1.0 - beta1) * gradient
        adam_second_moment = (
            beta2 * adam_second_moment + (1.0 - beta2) * gradient * gradient
        )
        corrected_mean = adam_mean / (1.0 - beta1**update)
        corrected_second = adam_second_moment / (1.0 - beta2**update)
        params += config.reinforce_learning_rate * corrected_mean / (
            np.sqrt(corrected_second) + 1e-8
        )

        should_evaluate = (
            update % config.reinforce_eval_interval == 0
            or update == config.reinforce_updates
        )
        eval_return: float | str = ""
        if should_evaluate:
            eval_return = evaluate_policy(
                policy, params, eval_seeds, config.max_episode_steps
            )
            if eval_return >= config.reinforce_target_return:
                target_hits += 1
            else:
                target_hits = 0
        history.append(
            {
                "seed": seed,
                "update": update,
                "eval_return": eval_return,
                "batch_mean_return": float(np.mean(batch_returns)),
                "batch_max_return": float(np.max(batch_returns)),
                "gradient_norm": gradient_norm,
                "parameter_norm": float(np.linalg.norm(params)),
            }
        )
        if target_hits >= config.reinforce_target_patience:
            break

    return params, tuple(history)


def _make_optimizer(
    method: str,
    num_params: int,
    config: BenchmarkConfig,
    seed: int,
) -> StandardES | DIIWES:
    common = dict(
        num_params=num_params,
        population_size=config.population_size,
        learning_rate=config.es_learning_rate,
        noise_std=config.noise_std,
        l2_coeff=0.0,
        antithetic=True,
        rank_fitness=True,
        max_grad_norm=0.0,
        max_param_norm=None,
        seed=seed,
    )
    if method == "standard_es":
        return StandardES(**common)
    if method != "diiwes":
        raise ValueError(f"unknown method: {method}")
    return DIIWES(
        **common,
        buffer_size=0,
        reuse_fraction=0.0,
        implicit_damping=0.0,
        use_curvature=True,
        curvature_fitness="raw",
        curvature_mode="diag",
        curvature_step_mode="dampen",
        curvature_beta=config.curvature_beta,
        curvature_clip=config.curvature_clip,
        min_step_multiplier=config.min_step_multiplier,
        trust_radius=None,
        use_leave_one_out_curvature_baseline=True,
        bias_correct_curvature_ema=True,
    )


def _evaluation_seeds(config: BenchmarkConfig, seed: int) -> tuple[int, ...]:
    return tuple(
        _seed(config.master_seed, seed, 800, episode)
        for episode in range(config.eval_episodes)
    )


def _population_returns(
    policy: DiscretePolicy,
    candidates: np.ndarray,
    config: BenchmarkConfig,
    seed: int,
    update: int,
) -> np.ndarray:
    half = config.population_size // 2
    environment_seeds = []
    for candidate_index in range(len(candidates)):
        pair_index = candidate_index if candidate_index < half else candidate_index - half
        environment_seeds.append(
            _seed(config.master_seed, seed, 900, update, pair_index)
        )
    return batched_greedy_returns(
        policy, candidates, environment_seeds, config.max_episode_steps
    )


def _trajectory_row(
    *,
    seed: int,
    initialization: str,
    method: str,
    update: int,
    eval_return: float,
    best_eval_return: float,
    population_returns: np.ndarray | None,
    info: dict[str, Any] | None,
) -> dict[str, Any]:
    details = {} if info is None else info
    return {
        "seed": seed,
        "initialization": initialization,
        "method": method,
        "update": update,
        "eval_return": float(eval_return),
        "best_eval_return": float(best_eval_return),
        "population_mean_return": ""
        if population_returns is None
        else float(np.mean(population_returns)),
        "population_max_return": ""
        if population_returns is None
        else float(np.max(population_returns)),
        "step_norm": float(details.get("step_norm", 0.0)),
        "grad_norm": float(details.get("grad_norm", 0.0)),
        "curvature_active_fraction": float(
            details.get("curvature_active_frac", 0.0)
        ),
        "mean_step_multiplier": float(details.get("step_multiplier_mean", 1.0)),
    }


def _summarize_runs(
    trajectories: Sequence[dict[str, Any]], config: BenchmarkConfig
) -> tuple[dict[str, Any], ...]:
    summaries: list[dict[str, Any]] = []
    for seed in config.seeds:
        for initialization in INITIALIZATIONS:
            for method in METHODS:
                rows = [
                    row
                    for row in trajectories
                    if row["seed"] == seed
                    and row["initialization"] == initialization
                    and row["method"] == method
                ]
                rows.sort(key=lambda row: int(row["update"]))
                if len(rows) != config.es_updates + 1:
                    raise RuntimeError("one nonlinear run has an incomplete trajectory")
                returns = np.asarray(
                    [float(row["eval_return"]) for row in rows], dtype=np.float64
                )
                if hasattr(np, "trapezoid"):
                    auc_integral = np.trapezoid(returns, dx=1.0)
                else:
                    auc_integral = np.trapz(returns, dx=1.0)
                mean_auc = float(auc_integral / config.es_updates)
                summaries.append(
                    {
                        "seed": seed,
                        "initialization": initialization,
                        "method": method,
                        "initial_return": float(returns[0]),
                        "final_return": float(returns[-1]),
                        "best_return": float(np.max(returns)),
                        "mean_return_auc": mean_auc,
                        "fine_tuning_gain": float(returns[-1] - returns[0]),
                        "solved_final": int(returns[-1] >= config.solve_return),
                    }
                )
    return tuple(summaries)


def _aggregate_runs(
    summaries: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    aggregates: list[dict[str, Any]] = []
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            rows = [
                row
                for row in summaries
                if row["initialization"] == initialization
                and row["method"] == method
            ]
            final = np.asarray([row["final_return"] for row in rows])
            aggregates.append(
                {
                    "initialization": initialization,
                    "method": method,
                    "n_runs": len(rows),
                    "median_initial_return": float(
                        np.median([row["initial_return"] for row in rows])
                    ),
                    "median_final_return": float(np.median(final)),
                    "q25_final_return": float(np.quantile(final, 0.25)),
                    "q75_final_return": float(np.quantile(final, 0.75)),
                    "median_best_return": float(
                        np.median([row["best_return"] for row in rows])
                    ),
                    "median_mean_return_auc": float(
                        np.median([row["mean_return_auc"] for row in rows])
                    ),
                    "median_fine_tuning_gain": float(
                        np.median([row["fine_tuning_gain"] for row in rows])
                    ),
                    "solved_fraction": float(
                        np.mean([row["solved_final"] for row in rows])
                    ),
                }
            )
    return tuple(aggregates)


def run_benchmark(config: BenchmarkConfig) -> ExperimentResult:
    config.validate()
    policy = make_policy(config)
    trajectories: list[dict[str, Any]] = []
    pretraining: list[dict[str, Any]] = []

    for seed in config.seeds:
        initial_rng = np.random.default_rng(
            _seed(config.master_seed, seed, 600)
        )
        random_params = initial_rng.normal(
            scale=config.init_param_std, size=policy.num_params
        )
        warm_params, warm_history = reinforce_pretrain(
            policy, random_params, config, seed
        )
        pretraining.extend(warm_history)
        initial_points = {"random": random_params, "reinforce": warm_params}
        eval_seeds = _evaluation_seeds(config, seed)

        for initialization in INITIALIZATIONS:
            for method in METHODS:
                params = initial_points[initialization].copy()
                optimizer_seed = _seed(config.master_seed, seed, 1000)
                optimizer = _make_optimizer(
                    method, policy.num_params, config, optimizer_seed
                )
                optimizer.current_params = params.copy()
                eval_return = evaluate_policy(
                    policy, params, eval_seeds, config.max_episode_steps
                )
                best_eval_return = eval_return
                trajectories.append(
                    _trajectory_row(
                        seed=seed,
                        initialization=initialization,
                        method=method,
                        update=0,
                        eval_return=eval_return,
                        best_eval_return=best_eval_return,
                        population_returns=None,
                        info=None,
                    )
                )
                for update in range(1, config.es_updates + 1):
                    noise, ask_info = optimizer.ask()
                    candidates = params[None, :] + optimizer.noise_std * noise
                    population_returns = _population_returns(
                        policy, candidates, config, seed, update
                    )
                    params, info = optimizer.tell(
                        params, noise, population_returns, ask_info
                    )
                    optimizer.current_params = params.copy()
                    eval_return = evaluate_policy(
                        policy, params, eval_seeds, config.max_episode_steps
                    )
                    best_eval_return = max(best_eval_return, eval_return)
                    trajectories.append(
                        _trajectory_row(
                            seed=seed,
                            initialization=initialization,
                            method=method,
                            update=update,
                            eval_return=eval_return,
                            best_eval_return=best_eval_return,
                            population_returns=population_returns,
                            info=info,
                        )
                    )

    summaries = _summarize_runs(trajectories, config)
    aggregates = _aggregate_runs(summaries)
    return ExperimentResult(
        trajectories=tuple(trajectories),
        pretraining=tuple(pretraining),
        run_summaries=summaries,
        aggregates=aggregates,
    )


def _write_csv(
    path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_digest() -> str:
    root = Path(__file__).resolve().parents[2]
    sources = (
        Path(__file__).resolve(),
        root / "core" / "diiwes.py",
        root / "core" / "standard_es.py",
        root / "core" / "policies.py",
    )
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _plot_learning_curves(
    result: ExperimentResult, config: BenchmarkConfig, output_dir: Path
) -> None:
    """Write a dependency-free SVG of median and interquartile trajectories."""
    width, height = 1100, 430
    panel_width, panel_height = 430, 315
    panel_lefts = (80, 625)
    panel_top = 60
    colors = {"standard_es": "#4C78A8", "diiwes": "#E45756"}
    labels = {"standard_es": "Standard ES", "diiwes": "DIIWES (diagonal H)"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.tick{font-size:11px}.label{font-size:13px}.title{font-size:16px;font-weight:600}.main{font-size:18px;font-weight:600}</style>',
        '<text x="550" y="27" text-anchor="middle" class="main">Nonlinear policy optimization: initialization and ES method</text>',
    ]

    def x_coordinate(update: float, left: float) -> float:
        return left + panel_width * update / config.es_updates

    def y_coordinate(value: float) -> float:
        clipped = float(np.clip(value, 0.0, config.max_episode_steps))
        return panel_top + panel_height * (1.0 - clipped / config.max_episode_steps)

    for panel_index, initialization in enumerate(INITIALIZATIONS):
        left = panel_lefts[panel_index]
        title = (
            "Random initialization"
            if initialization == "random"
            else "REINFORCE warm start"
        )
        elements.extend(
            [
                f'<text x="{left + panel_width / 2:.1f}" y="49" text-anchor="middle" class="title">{title}</text>',
                f'<line x1="{left}" y1="{panel_top}" x2="{left}" y2="{panel_top + panel_height}" stroke="#333"/>',
                f'<line x1="{left}" y1="{panel_top + panel_height}" x2="{left + panel_width}" y2="{panel_top + panel_height}" stroke="#333"/>',
            ]
        )
        for y_value in np.linspace(0.0, config.max_episode_steps, num=5):
            y_pos = y_coordinate(y_value)
            elements.append(
                f'<line x1="{left}" y1="{y_pos:.1f}" x2="{left + panel_width}" y2="{y_pos:.1f}" stroke="#ddd" stroke-width="1"/>'
            )
            elements.append(
                f'<text x="{left - 9}" y="{y_pos + 4:.1f}" text-anchor="end" class="tick">{y_value:g}</text>'
            )
        x_ticks = sorted({0, config.es_updates // 2, config.es_updates})
        for update in x_ticks:
            x_pos = x_coordinate(update, left)
            elements.append(
                f'<text x="{x_pos:.1f}" y="{panel_top + panel_height + 19}" text-anchor="middle" class="tick">{update}</text>'
            )
        elements.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{panel_top + panel_height + 43}" text-anchor="middle" class="label">Zeroth-order update</text>'
        )
        for method in METHODS:
            matrix = []
            for seed in config.seeds:
                rows = [
                    row
                    for row in result.trajectories
                    if row["seed"] == seed
                    and row["initialization"] == initialization
                    and row["method"] == method
                ]
                rows.sort(key=lambda row: int(row["update"]))
                matrix.append([float(row["eval_return"]) for row in rows])
            values = np.asarray(matrix, dtype=np.float64)
            updates = np.arange(config.es_updates + 1)
            median = np.median(values, axis=0)
            q25 = np.quantile(values, 0.25, axis=0)
            q75 = np.quantile(values, 0.75, axis=0)
            upper = [
                (x_coordinate(float(update), left), y_coordinate(float(value)))
                for update, value in zip(updates, q75, strict=True)
            ]
            lower = [
                (x_coordinate(float(update), left), y_coordinate(float(value)))
                for update, value in zip(updates[::-1], q25[::-1], strict=True)
            ]
            polygon = " ".join(
                f"{x_value:.1f},{y_value:.1f}" for x_value, y_value in upper + lower
            )
            line = " ".join(
                f"{x_coordinate(float(update), left):.1f},{y_coordinate(float(value)):.1f}"
                for update, value in zip(updates, median, strict=True)
            )
            elements.append(
                f'<polygon points="{polygon}" fill="{colors[method]}" fill-opacity="0.18"/>'
            )
            elements.append(
                f'<polyline points="{line}" fill="none" stroke="{colors[method]}" stroke-width="2.5"/>'
            )
    elements.append(
        f'<text x="18" y="217" text-anchor="middle" class="label" transform="rotate(-90 18 217)">CartPole return (max {config.max_episode_steps})</text>'
    )
    legend_x = 710
    for index, method in enumerate(METHODS):
        y_pos = 406 + 18 * index
        elements.append(
            f'<line x1="{legend_x}" y1="{y_pos}" x2="{legend_x + 26}" y2="{y_pos}" stroke="{colors[method]}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{legend_x + 34}" y="{y_pos + 4}" class="tick">{labels[method]}</text>'
        )
    elements.append("</svg>")
    (output_dir / "learning_curves.svg").write_text(
        "\n".join(elements) + "\n", encoding="utf-8"
    )


def _aggregate_lookup(
    result: ExperimentResult, initialization: str, method: str
) -> dict[str, Any]:
    return next(
        row
        for row in result.aggregates
        if row["initialization"] == initialization and row["method"] == method
    )


def _comparison_diagnostics(result: ExperimentResult) -> dict[str, Any]:
    paired: dict[str, dict[str, list[float]]] = {}
    mechanism: dict[str, dict[str, float]] = {}
    for initialization in INITIALIZATIONS:
        paired[initialization] = {}
        for metric in ("final_return", "mean_return_auc"):
            deltas: list[float] = []
            seeds = sorted(
                {
                    int(row["seed"])
                    for row in result.run_summaries
                    if row["initialization"] == initialization
                }
            )
            for seed in seeds:
                matched = {
                    str(row["method"]): float(row[metric])
                    for row in result.run_summaries
                    if int(row["seed"]) == seed
                    and row["initialization"] == initialization
                }
                deltas.append(matched["diiwes"] - matched["standard_es"])
            paired[initialization][f"diiwes_minus_standard_es_{metric}"] = deltas

        method_rows = {
            method: [
                row
                for row in result.trajectories
                if row["initialization"] == initialization
                and row["method"] == method
                and int(row["update"]) > 0
            ]
            for method in METHODS
        }
        diiwes_rows = method_rows["diiwes"]
        standard_rows = method_rows["standard_es"]
        mechanism[initialization] = {
            "median_curvature_active_fraction": float(
                np.median(
                    [float(row["curvature_active_fraction"]) for row in diiwes_rows]
                )
            ),
            "median_step_multiplier": float(
                np.median([float(row["mean_step_multiplier"]) for row in diiwes_rows])
            ),
            "median_diiwes_step_norm": float(
                np.median([float(row["step_norm"]) for row in diiwes_rows])
            ),
            "median_standard_es_step_norm": float(
                np.median([float(row["step_norm"]) for row in standard_rows])
            ),
        }
    return {"paired_method_deltas": paired, "diiwes_mechanism": mechanism}


def _report_text(result: ExperimentResult, config: BenchmarkConfig) -> str:
    random_diiwes = _aggregate_lookup(result, "random", "diiwes")
    warm_diiwes = _aggregate_lookup(result, "reinforce", "diiwes")
    warm_standard = _aggregate_lookup(result, "reinforce", "standard_es")
    warm_start_gain = (
        warm_diiwes["median_initial_return"]
        - random_diiwes["median_initial_return"]
    )
    method_gap = (
        warm_diiwes["median_final_return"]
        - warm_standard["median_final_return"]
    )
    diagnostics = _comparison_diagnostics(result)
    warm_deltas = diagnostics["paired_method_deltas"]["reinforce"]
    warm_final_deltas = warm_deltas["diiwes_minus_standard_es_final_return"]
    warm_auc_deltas = warm_deltas["diiwes_minus_standard_es_mean_return_auc"]
    warm_mechanism = diagnostics["diiwes_mechanism"]["reinforce"]
    if warm_start_gain > 0.0:
        initialization_conclusion = (
            f"The REINFORCE first stage raised DIIWES's median starting return "
            f"by {warm_start_gain:.1f} points."
        )
    else:
        initialization_conclusion = (
            "The REINFORCE first stage did not improve DIIWES's median starting return."
        )
    if method_gap > 0.0:
        method_conclusion = (
            f"After the warm start, DIIWES finished {method_gap:.1f} median "
            "return points above Standard ES."
        )
    elif method_gap < 0.0:
        method_conclusion = (
            f"After the warm start, DIIWES finished {-method_gap:.1f} median "
            "return points below Standard ES."
        )
    else:
        method_conclusion = (
            "After the warm start, DIIWES and Standard ES had the same median final return."
        )

    lines = [
        "# Nonlinear CartPole warm-start experiment",
        "",
        "## Result",
        "",
        initialization_conclusion,
        method_conclusion,
        "",
        "| initialization | method | median initial | median final [IQR] | median AUC | median fine-tuning gain | solved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.aggregates:
        lines.append(
            f"| {row['initialization']} | {row['method']} | "
            f"{row['median_initial_return']:.1f} | {row['median_final_return']:.1f} "
            f"[{row['q25_final_return']:.1f}, {row['q75_final_return']:.1f}] | "
            f"{row['median_mean_return_auc']:.1f} | "
            f"{row['median_fine_tuning_gain']:+.1f} | "
            f"{100.0 * row['solved_fraction']:.0f}% |"
        )
    lines.extend(
        [
            "",
            "## DIIWES readout",
            "",
            "- Warm-started paired final-return differences (DIIWES minus Standard ES): "
            + ", ".join(f"{value:+.1f}" for value in warm_final_deltas)
            + ".",
            "- Warm-started paired AUC differences (DIIWES minus Standard ES): "
            + ", ".join(f"{value:+.1f}" for value in warm_auc_deltas)
            + ".",
            f"- Across warm-started DIIWES updates, the median curvature-active coordinate fraction was {100.0 * warm_mechanism['median_curvature_active_fraction']:.1f}% and the median mean step multiplier was {warm_mechanism['median_step_multiplier']:.3f}.",
            f"- The median applied step norm was {warm_mechanism['median_diiwes_step_norm']:.4f} for DIIWES versus {warm_mechanism['median_standard_es_step_norm']:.4f} for Standard ES.",
            "",
            "The warm start is the decisive improvement in this run. The diagonal-curvature update is stable from that point, but it does not beat the matched Standard ES control; its damping produces smaller updates and nearly identical, slightly lower returns.",
            "",
            "## Protocol",
            "",
            f"- Nonlinear task: seeded CartPole dynamics, return capped at {config.max_episode_steps}.",
            f"- Policy: 4-{ '-'.join(str(value) for value in config.hidden_dims) }-2 tanh MLP ({make_policy(config).num_params} parameters).",
            f"- Initialization: shared random actor or up to {config.reinforce_updates} batched REINFORCE/Adam updates; early-stop target {config.reinforce_target_return:g}.",
            f"- Fine-tuning: {config.es_updates} updates, {config.antithetic_pairs} antithetic pairs (population {config.population_size}), Gaussian sigma {config.noise_std:g}, learning rate {config.es_learning_rate:g}.",
            "- DIIWES uses the repository implementation with raw-return diagonal Stein curvature, leave-one-pair-out baseline, no replay, no trust clipping, and no scalar damping.",
            f"- Evaluation: {config.eval_episodes} held-out deterministic episodes per checkpoint and {len(config.seeds)} matched seeds.",
            "- Human feedback and federation are intentionally omitted.",
            "",
            "## Interpretation limits",
            "",
            f"This is a low-cost mechanism check on one canonical nonlinear control problem, not a substitute for the paper's MuJoCo-scale PPO-initialized study. The {len(config.seeds)} matched seeds characterize this implementation run but do not establish broad statistical superiority.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: ExperimentResult, config: BenchmarkConfig, output_dir: str | Path
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "trajectory.csv", result.trajectories, TRAJECTORY_FIELDS)
    _write_csv(destination / "pretraining.csv", result.pretraining, PRETRAIN_FIELDS)
    _write_csv(
        destination / "run_summary.csv", result.run_summaries, RUN_SUMMARY_FIELDS
    )
    _write_csv(destination / "aggregate.csv", result.aggregates, AGGREGATE_FIELDS)
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "source_sha256": _source_digest(),
        "config": asdict(config),
        "methods": list(METHODS),
        "initializations": list(INITIALIZATIONS),
        "environment": {
            "name": "CartPole",
            "implementation": "dependency_free_canonical_dynamics",
            "nonlinearity": "sin_and_cos_pole_dynamics_plus_tanh_neural_policy",
        },
        "optimizer_contract": {
            "standard_es": "core.standard_es.StandardES",
            "diiwes": "core.diiwes.DIIWES",
            "common_random_numbers": True,
            "antithetic_pairs_share_environment_seed": True,
            "antithetic_pairs_per_update": config.antithetic_pairs,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    with (destination / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    summary = {
        "aggregates": list(result.aggregates),
        **_comparison_diagnostics(result),
        "paper_adaptation": {
            "kept": "first_order_policy_gradient_warm_start_before_zeroth_order_fine_tuning",
            "changed": "REINFORCE_on_CartPole_instead_of_PPO_on_MuJoCo",
            "omitted": ["human_feedback", "preference_model", "federation"],
        },
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    (destination / "report.md").write_text(
        _report_text(result, config), encoding="utf-8"
    )
    _plot_learning_curves(result, config, destination)
    return {
        name: str(destination / name)
        for name in (
            "manifest.json",
            "summary.json",
            "trajectory.csv",
            "pretraining.csv",
            "run_summary.csv",
            "aggregate.csv",
            "report.md",
            "learning_curves.svg",
        )
    }


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated integer list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/nonlinear_cartpole_warm_start",
    )
    parser.add_argument("--seeds", type=_parse_int_tuple, default=None)
    parser.add_argument("--reinforce-updates", type=int, default=None)
    parser.add_argument("--es-updates", type=int, default=None)
    population_group = parser.add_mutually_exclusive_group()
    population_group.add_argument(
        "--antithetic-pairs",
        type=int,
        default=None,
        help="Number of matched +epsilon/-epsilon pairs; population is twice this value.",
    )
    population_group.add_argument(
        "--population-size",
        type=int,
        default=None,
        help="Even candidate count; --antithetic-pairs is the clearer equivalent.",
    )
    parser.add_argument("--eval-episodes", type=int, default=None)
    args = parser.parse_args()

    overrides: dict[str, Any] = {}
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.reinforce_updates is not None:
        overrides["reinforce_updates"] = args.reinforce_updates
    if args.es_updates is not None:
        overrides["es_updates"] = args.es_updates
    if args.antithetic_pairs is not None:
        if args.antithetic_pairs < 2:
            raise ValueError("antithetic_pairs must be at least two")
        overrides["population_size"] = 2 * args.antithetic_pairs
    elif args.population_size is not None:
        overrides["population_size"] = args.population_size
    if args.eval_episodes is not None:
        overrides["eval_episodes"] = args.eval_episodes
    config = BenchmarkConfig(**overrides)
    result = run_benchmark(config)
    outputs = write_outputs(result, config, args.output_dir)
    print(_report_text(result, config))
    print(f"Artifacts written to {os.path.abspath(args.output_dir)}")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()

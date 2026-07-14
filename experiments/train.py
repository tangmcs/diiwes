#!/usr/bin/env python3
"""Parallel trainer for the paper ES experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import socket
import subprocess
import sys
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    AdamES,
    ClipUpES,
    ConcaveCurvatureES,
    DIIWES,
    EndpointImplicitES,
    LinearizedImplicitES,
    LOPOGradientES,
    MomentumES,
    SNES,
    StandardES,
)
from core.policies import DiscretePolicy, MLPPolicy, make_layer_slices
from experiments.lagged_subspace_study_lock import (
    require_checkpoint_generation_provenance_locks,
    study_sha256_for_checkpoint_config,
)
from utilities import ObsNormalizer


CONDITIONS = {
    "standard_es",
    "momentum_es",
    "adam_es",
    "clipup_es",
    "snes",
    "no_curvature",
    "scalar_damped_es",
    "diag_curvature",
    "diag_curvature_raw",
    "diag_curvature_matched_rank",
    "global_curvature",
    "block_curvature",
    "directional_curvature",
    "endpoint_implicit_es",
    "linearized_implicit_es",
    "concave_diagonal_curvature_es",
    "concave_block_curvature_es",
    "concave_block_ema_curvature_es",
    "concave_block_ema_isotropic_control_es",
    "concave_block_ols_ema_curvature_es",
    "concave_block_lopo_u_stat",
    "concave_block_lopo_u_stat_isotropic_control",
    "lopo_gradient_only_es",
}

LR_SCHEDULES = {"constant", "exponential", "inverse_linear", "inverse_sqrt"}
SNES_FINAL_SEARCH_STD_ARTIFACT = "snes_search_std.npy"
HESSIAN_EMA_ARTIFACT = "hessian_ema.npy"
CHECKPOINT_CAPTURE_MANIFEST = "checkpoint_capture.json"
CHECKPOINT_TRAINING_CONFIG_ARTIFACT = "checkpoint_training_config.json"
CHECKPOINT_CAPTURE_DIRECTORY = "checkpoints"
CHECKPOINT_GRADIENT_ARCHIVE_LENGTH = 10
LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL = (
    "lagged_subspace_frozen_checkpoint_v1"
)

_LAGGED_SUBSPACE_CHECKPOINT_TASKS = frozenset(
    {"Hopper-v5", "Walker2d-v5", "HalfCheetah-v5"}
)
_LAGGED_SUBSPACE_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "bias_correct_curvature_ema",
        "block_structure",
        "clipup_max_speed",
        "clipup_momentum",
        "curvature_attenuation_mode",
        "curvature_confidence_z",
        "curvature_estimator",
        "curvature_ema",
        "curvature_fitness",
        "curvature_mode",
        "curvature_rank_utility_mode",
        "curvature_step_mode",
        "endpoint_ratio_diagnostic_cap",
        "endpoint_ratio_diagnostic_floor",
        "ess_min_ratio",
        "gradient_clipping",
        "heldout_eval_episodes",
        "heldout_training_step_budget",
        "implicit_iterations",
        "implicit_tolerance",
        "importance_sampling",
        "importance_sampling_enabled",
        "linear_min_abs_diagonal",
        "lr_decay",
        "max_importance_weight",
        "min_abs_diagonal",
        "min_importance_weight",
        "min_step_multiplier",
        "momentum_beta",
        "optimizer_momentum",
        "outcome_based_checkpoint_selection",
        "parameter_projection",
        "picard_iteration",
        "picard_iteration_enabled",
        "post_outcome_record_exclusion",
        "snes_sigma_learning_rate",
        "trust_radius",
        "trust_region",
        "trust_region_enabled",
        "use_leave_one_out_curvature_baseline",
        "use_trust_radius_for_standard_es",
    }
)

_NAMED_LOPO_CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "lopo_gradient_only_es": {
        "antithetic": True,
        "l2_coeff": 0.0,
        "implicit_damping": 0.0,
        "scalar_damping": 0.0,
        "min_replay_weight_mass": 0.0,
        "curvature_beta": 0.0,
        "curvature_clip": 0.0,
        "curvature_fitness": "none",
        "curvature_mode": "none",
        "curvature_estimator": "none",
        "curvature_confidence_z": None,
        "curvature_rank_utility_mode": "lopo_rank_u_statistic",
        "curvature_attenuation_mode": "none",
        "rank_fitness": True,
        "evaluate_center_fitness": False,
        "use_leave_one_out_curvature_baseline": False,
        "bias_correct_curvature_ema": False,
    },
    "concave_block_lopo_u_stat": {
        "antithetic": True,
        "l2_coeff": 0.0,
        "implicit_damping": 0.0,
        "scalar_damping": 0.0,
        "min_replay_weight_mass": 0.0,
        "curvature_beta": 0.0,
        "curvature_clip": 0.0,
        "curvature_fitness": "matched",
        "curvature_mode": "block",
        "curvature_estimator": "stein_moment",
        "curvature_confidence_z": None,
        "curvature_rank_utility_mode": "lopo_rank_u_statistic",
        "curvature_attenuation_mode": "structured",
        "rank_fitness": True,
        "evaluate_center_fitness": False,
        "use_leave_one_out_curvature_baseline": False,
        "bias_correct_curvature_ema": False,
    },
    "concave_block_lopo_u_stat_isotropic_control": {
        "antithetic": True,
        "l2_coeff": 0.0,
        "implicit_damping": 0.0,
        "scalar_damping": 0.0,
        "min_replay_weight_mass": 0.0,
        "curvature_beta": 0.0,
        "curvature_clip": 0.0,
        "curvature_fitness": "matched",
        "curvature_mode": "block",
        "curvature_estimator": "stein_moment",
        "curvature_confidence_z": None,
        "curvature_rank_utility_mode": "lopo_rank_u_statistic",
        "curvature_attenuation_mode": "isotropic_norm_matched",
        "rank_fitness": True,
        "evaluate_center_fitness": False,
        "use_leave_one_out_curvature_baseline": False,
        "bias_correct_curvature_ema": False,
    },
}

_WORKER_ENV = None
_WORKER_POLICY = None
_WORKER_MAX_STEPS = None


def _named_lopo_value_matches(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) is expected
    if isinstance(expected, float):
        try:
            numeric = float(actual)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(numeric) and numeric == expected)
    return actual == expected


def _validate_named_lopo_condition_semantics(
    config: dict[str, Any], condition: str | None = None
) -> None:
    resolved_condition = str(
        config.get("condition", "") if condition is None else condition
    )
    expected = _NAMED_LOPO_CONDITION_SPECS.get(resolved_condition)
    if expected is None:
        return
    for key, expected_value in expected.items():
        if key in config and not _named_lopo_value_matches(
            config[key], expected_value
        ):
            raise ValueError(
                f"named LOPO condition {resolved_condition} locks "
                f"{key}={expected_value!r}; got {config[key]!r}"
            )


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


def _validate_no_replay_protocol(config: dict[str, Any]) -> None:
    if config.get("replay_enabled", False) is not False:
        raise ValueError("replay_enabled must be false for the no-replay protocol")
    for key in ("reuse_fraction", "buffer_size"):
        try:
            value = float(config.get(key, 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be zero for the no-replay protocol") from error
        if not np.isfinite(value) or value != 0.0:
            raise ValueError(f"{key} must be zero for the no-replay protocol")


def _fresh_only_diagnostic_matches(actual: Any, expected: float) -> bool:
    """Allow only floating-point roundoff in fresh-only unit-mass diagnostics."""
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(value)
        and np.isclose(value, expected, rtol=1e-9, atol=1e-12)
    )


def _condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    config = dict(config)
    _validate_named_lopo_condition_semantics(config, condition)
    config["diagnostic_schema_version"] = 2
    obsolete_keys = (
        "trust_radius",
        "use_trust_radius_for_standard_es",
        "min_step_multiplier",
        "curvature_step_mode",
    )
    present_obsolete = [key for key in obsolete_keys if key in config]
    if present_obsolete:
        raise ValueError(
            "retired trust/floor settings are not allowed: " + ", ".join(present_obsolete)
        )
    try:
        configured_max_grad_norm = float(config.get("max_grad_norm", 0.0))
    except (TypeError, ValueError):
        configured_max_grad_norm = float("nan")
    if not np.isfinite(configured_max_grad_norm) or configured_max_grad_norm != 0.0:
        raise ValueError("max_grad_norm must be zero for the no-norm-control protocol")
    if config.get("max_param_norm") is not None:
        raise ValueError("max_param_norm is not allowed for the no-norm-control protocol")
    config["max_grad_norm"] = 0.0
    config["max_param_norm"] = None
    _validate_no_replay_protocol(config)
    config["replay_enabled"] = False
    config["reuse_fraction"] = 0.0
    config["buffer_size"] = 0
    config["condition"] = condition

    if condition == "standard_es":
        config["algorithm"] = "standard_es"
        config["use_curvature"] = False
    elif condition == "momentum_es":
        config["algorithm"] = "momentum_es"
        config["use_curvature"] = False
        config.setdefault("momentum_beta", 0.9)
    elif condition == "adam_es":
        config["algorithm"] = "adam_es"
        config["use_curvature"] = False
        config.setdefault("adam_beta1", 0.9)
        config.setdefault("adam_beta2", 0.999)
        config.setdefault("adam_epsilon", 1e-8)
    elif condition == "clipup_es":
        config["algorithm"] = "clipup_es"
        config["use_curvature"] = False
        config.setdefault("clipup_momentum", 0.9)
        config.setdefault("clipup_max_speed", 0.15)
    elif condition == "snes":
        config["algorithm"] = "snes"
        config["use_curvature"] = False
        config["rank_fitness"] = True
    elif condition in {"no_curvature", "scalar_damped_es"}:
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = False
        config["curvature_mode"] = "diag"
    elif condition == "diag_curvature":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
        config["curvature_fitness"] = "raw"
    elif condition == "diag_curvature_raw":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
        config["curvature_fitness"] = "raw"
    elif condition == "diag_curvature_matched_rank":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "diag"
        config["curvature_fitness"] = "matched"
    elif condition == "global_curvature":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "global"
    elif condition == "block_curvature":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "block"
        config["block_structure"] = "layer"
    elif condition == "directional_curvature":
        config["algorithm"] = "curvature_preconditioned_es"
        config["use_curvature"] = True
        config["curvature_mode"] = "directional"
    elif condition == "endpoint_implicit_es":
        config["algorithm"] = "endpoint_implicit_es"
        config["use_curvature"] = False
        config["implicit_damping"] = 0.0
    elif condition == "linearized_implicit_es":
        config["algorithm"] = "linearized_implicit_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "diag"
        config["curvature_beta"] = 0.0
        config["implicit_damping"] = 0.0
    elif condition == "concave_diagonal_curvature_es":
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "diag"
        config["curvature_beta"] = 0.0
        config["curvature_rank_utility_mode"] = "pooled_centered_ranks"
        config["implicit_damping"] = 0.0
    elif condition == "concave_block_curvature_es":
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "block"
        config["curvature_beta"] = 0.0
        config["curvature_rank_utility_mode"] = "pooled_centered_ranks"
        config["implicit_damping"] = 0.0
    elif condition == "concave_block_ema_curvature_es":
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "block"
        config["curvature_beta"] = 0.9
        config["curvature_estimator"] = "stein_moment"
        config["curvature_confidence_z"] = None
        config["curvature_rank_utility_mode"] = "pooled_centered_ranks"
        config["curvature_attenuation_mode"] = "structured"
        config["implicit_damping"] = 0.0
    elif condition == "concave_block_ema_isotropic_control_es":
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "block"
        config["curvature_beta"] = 0.9
        config["curvature_estimator"] = "stein_moment"
        config["curvature_confidence_z"] = None
        config["curvature_rank_utility_mode"] = "pooled_centered_ranks"
        config["curvature_attenuation_mode"] = "isotropic_norm_matched"
        config["implicit_damping"] = 0.0
    elif condition == "concave_block_ols_ema_curvature_es":
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config["curvature_fitness"] = "matched"
        config["curvature_mode"] = "block"
        config["curvature_beta"] = 0.9
        config["curvature_estimator"] = "block_joint_ols"
        config["curvature_confidence_z"] = 1.645
        config["curvature_rank_utility_mode"] = "pooled_centered_ranks"
        config["curvature_attenuation_mode"] = "structured"
        config["implicit_damping"] = 0.0
    elif condition == "lopo_gradient_only_es":
        config["algorithm"] = "lopo_gradient_es"
        config["use_curvature"] = False
        config.update(_NAMED_LOPO_CONDITION_SPECS[condition])
        config["implicit_damping"] = 0.0
    elif condition in {
        "concave_block_lopo_u_stat",
        "concave_block_lopo_u_stat_isotropic_control",
    }:
        config["algorithm"] = "concave_curvature_es"
        config["use_curvature"] = True
        config.update(_NAMED_LOPO_CONDITION_SPECS[condition])
        config["implicit_damping"] = 0.0
    else:
        raise ValueError(f"unknown condition: {condition}")
    if config["algorithm"] == "curvature_preconditioned_es":
        config.setdefault("curvature_fitness", "raw")
    _validate_named_lopo_condition_semantics(config, condition)
    return config


def make_optimizer(
    config: dict[str, Any],
    num_params: int,
    policy: MLPPolicy | DiscretePolicy,
    seed: int,
) -> (
    StandardES
    | MomentumES
    | AdamES
    | ClipUpES
    | SNES
    | DIIWES
    | EndpointImplicitES
    | LinearizedImplicitES
    | ConcaveCurvatureES
    | LOPOGradientES
):
    common_kwargs = {
        "num_params": num_params,
        "population_size": config.get("population_size", 200),
        "learning_rate": config.get("learning_rate", 0.02),
        "noise_std": config.get("noise_std", 0.02),
        "l2_coeff": config.get("l2_coeff", 0.0),
        "antithetic": config.get("antithetic", True),
        "rank_fitness": config.get("rank_fitness", True),
        "max_grad_norm": config.get("max_grad_norm", 0.0),
        "max_param_norm": config.get("max_param_norm", None),
        "seed": seed,
    }
    if config["algorithm"] == "standard_es":
        return StandardES(**common_kwargs)
    if config["algorithm"] == "momentum_es":
        return MomentumES(
            **common_kwargs,
            momentum_beta=config.get("momentum_beta", 0.9),
        )
    if config["algorithm"] == "adam_es":
        return AdamES(
            **common_kwargs,
            adam_beta1=config.get("adam_beta1", 0.9),
            adam_beta2=config.get("adam_beta2", 0.999),
            adam_epsilon=config.get("adam_epsilon", 1e-8),
        )
    if config["algorithm"] == "clipup_es":
        return ClipUpES(
            **common_kwargs,
            clipup_momentum=config.get("clipup_momentum", 0.9),
            clipup_max_speed=config.get("clipup_max_speed", 0.15),
        )
    if config["algorithm"] == "snes":
        return SNES(
            **common_kwargs,
            snes_sigma_learning_rate=config.get(
                "snes_sigma_learning_rate"
            ),
        )
    if config["algorithm"] == "endpoint_implicit_es":
        return EndpointImplicitES(
            **common_kwargs,
            implicit_damping=config.get("implicit_damping", 0.0),
            implicit_iterations=config.get("implicit_iterations", 10),
            implicit_tolerance=config.get("implicit_tolerance", 1e-5),
            diagnostic_ratio_floor=config.get("endpoint_ratio_diagnostic_floor", 1e-3),
            diagnostic_ratio_cap=config.get("endpoint_ratio_diagnostic_cap", 10.0),
        )
    if config["algorithm"] == "linearized_implicit_es":
        return LinearizedImplicitES(
            **common_kwargs,
            implicit_damping=config.get("implicit_damping", 0.0),
            min_abs_diagonal=config.get("linear_min_abs_diagonal", 1e-12),
        )
    if config["algorithm"] == "lopo_gradient_es":
        return LOPOGradientES(
            **common_kwargs,
            implicit_damping=config.get("implicit_damping", 0.0),
        )
    if config["algorithm"] == "concave_curvature_es":
        curvature_structure = str(config.get("curvature_mode", "diag"))
        block_slices = (
            make_layer_slices(policy) if curvature_structure == "block" else None
        )
        return ConcaveCurvatureES(
            **common_kwargs,
            implicit_damping=config.get("implicit_damping", 0.0),
            curvature_structure=curvature_structure,
            block_slices=block_slices,
            curvature_beta=config.get("curvature_beta", 0.0),
            curvature_estimator=config.get("curvature_estimator", "stein_moment"),
            curvature_confidence_z=config.get("curvature_confidence_z"),
            rank_utility_mode=config.get(
                "curvature_rank_utility_mode", "pooled_centered_ranks"
            ),
            attenuation_mode=config.get(
                "curvature_attenuation_mode", "structured"
            ),
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
        buffer_size=config.get("buffer_size", 0),
        reuse_fraction=config.get("reuse_fraction", 0.0),
        min_importance_weight=config.get("min_importance_weight", 1e-3),
        max_importance_weight=config.get("max_importance_weight", 10.0),
        scalar_damping=config.get("scalar_damping", 0.1),
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
        curvature_beta=config.get("curvature_beta", 0.99),
        curvature_clip=config.get("curvature_clip", 1e3),
        ess_min_ratio=config.get("ess_min_ratio", 0.2),
        min_replay_weight_mass=config.get("min_replay_weight_mass", 0.01),
        block_slices=block_slices,
        use_leave_one_out_curvature_baseline=config.get("use_leave_one_out_curvature_baseline", True),
        bias_correct_curvature_ema=config.get("bias_correct_curvature_ema", True),
    )


def _learning_rate_at(
    initial_learning_rate: float,
    iteration: int,
    schedule: str,
    *,
    exponential_decay: float = 1.0,
) -> float:
    initial_learning_rate = float(initial_learning_rate)
    iteration = int(iteration)
    schedule = str(schedule).lower()
    if not np.isfinite(initial_learning_rate) or initial_learning_rate <= 0.0:
        raise ValueError("initial learning rate must be finite and positive")
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    if schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown learning-rate schedule: {schedule}")
    if schedule == "constant":
        learning_rate = initial_learning_rate
    elif schedule == "exponential":
        if not np.isfinite(exponential_decay) or not (0.0 < exponential_decay <= 1.0):
            raise ValueError("exponential decay must be finite and in (0, 1]")
        learning_rate = initial_learning_rate * exponential_decay**iteration
    elif schedule == "inverse_sqrt":
        learning_rate = initial_learning_rate / np.sqrt(iteration + 1.0)
    else:
        learning_rate = initial_learning_rate / (iteration + 1.0)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise FloatingPointError("learning-rate schedule produced an invalid value")
    return float(learning_rate)


def _keyed_rollout_seed(seed: int, stream: int, iteration: int, index: int) -> int:
    components = [int(seed), int(stream), int(iteration), int(index)]
    if any(component < 0 for component in components):
        raise ValueError("seed components must be nonnegative")

    # Cantor pairing is injective over nonnegative integers.  Keeping the full
    # Python integer avoids the birthday collisions caused by hashing every
    # rollout key into a single uint32 seed.
    def pair(left: int, right: int) -> int:
        total = left + right
        return total * (total + 1) // 2 + right

    return pair(pair(components[0], components[1]), pair(components[2], components[3]))


def _training_rollout_seed(
    seed: int,
    iteration: int,
    local_index: int,
    common: bool,
    fresh_count: int,
    antithetic: bool,
) -> int:
    local_index = int(local_index)
    seed_index = local_index
    if common and antithetic:
        pair_count = int(fresh_count) // 2
        if pair_count <= local_index < 2 * pair_count:
            seed_index = local_index - pair_count
    return _keyed_rollout_seed(seed, 0, iteration, seed_index)


def _evaluation_rollout_seed(seed: int, eval_index: int) -> int:
    return _keyed_rollout_seed(seed, 1, 0, eval_index)


def _center_rollout_seed(seed: int, iteration: int) -> int:
    return _keyed_rollout_seed(seed, 2, iteration, 0)


def _calibration_rollout_seed(seed: int, calibration_index: int) -> int:
    return _keyed_rollout_seed(seed, 3, 0, calibration_index)


def _heldout_evaluation_rollout_seed(seed: int, eval_index: int) -> int:
    """Return a fixed held-out seed from a stream unused during training."""
    return _keyed_rollout_seed(seed, 4, 0, eval_index)


def _resolve_heldout_evaluation_config(
    config: dict[str, Any],
) -> dict[str, int] | None:
    enabled = config.get("heldout_evaluation_enabled", False)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("heldout_evaluation_enabled must be boolean")
    if not bool(enabled):
        return None

    resolved: dict[str, int] = {}
    for key in ("heldout_training_step_budget", "heldout_eval_episodes"):
        if key not in config:
            raise ValueError(f"{key} is required when held-out evaluation is enabled")
        value = config[key]
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a positive integer")
        try:
            integer_value = int(value)
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{key} must be a positive integer") from error
        if (
            not np.isfinite(numeric_value)
            or numeric_value != float(integer_value)
            or integer_value <= 0
        ):
            raise ValueError(f"{key} must be a positive integer")
        resolved[key] = integer_value
    return resolved


def _resolve_online_evaluation_enabled(config: dict[str, Any]) -> bool:
    enabled = config.get("online_evaluation_enabled", True)
    if not isinstance(enabled, (bool, np.bool_)):
        raise ValueError("online_evaluation_enabled must be boolean")
    return bool(enabled)


def _resolve_checkpoint_capture_config(
    config: dict[str, Any], n_iterations: int
) -> dict[str, Any] | None:
    raw_generations = config.get("checkpoint_capture_generations")
    configured_archive_length = config.get(
        "checkpoint_gradient_archive_length",
        CHECKPOINT_GRADIENT_ARCHIVE_LENGTH,
    )
    if isinstance(configured_archive_length, (bool, np.bool_)):
        raise ValueError("checkpoint_gradient_archive_length must equal 10")
    try:
        archive_length = int(configured_archive_length)
        numeric_archive_length = float(configured_archive_length)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "checkpoint_gradient_archive_length must equal 10"
        ) from error
    if (
        not np.isfinite(numeric_archive_length)
        or numeric_archive_length != float(archive_length)
        or archive_length != CHECKPOINT_GRADIENT_ARCHIVE_LENGTH
    ):
        raise ValueError("checkpoint_gradient_archive_length must equal 10")
    if raw_generations is None:
        if "checkpoint_gradient_archive_length" in config:
            raise ValueError(
                "checkpoint_capture_generations is required when configuring "
                "the gradient archive"
            )
        return None
    if (
        isinstance(raw_generations, (str, bytes, bool, np.bool_))
        or not isinstance(raw_generations, (list, tuple))
        or not raw_generations
    ):
        raise ValueError(
            "checkpoint_capture_generations must be a nonempty integer list"
        )
    generations: list[int] = []
    for value in raw_generations:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                "checkpoint_capture_generations must contain only integers"
            )
        try:
            generation = int(value)
            numeric_generation = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "checkpoint_capture_generations must contain only integers"
            ) from error
        if (
            not np.isfinite(numeric_generation)
            or numeric_generation != float(generation)
        ):
            raise ValueError(
                "checkpoint_capture_generations must contain only integers"
            )
        generations.append(generation)
    if generations != sorted(set(generations)):
        raise ValueError(
            "checkpoint_capture_generations must be unique and strictly increasing"
        )
    if generations[0] < archive_length:
        raise ValueError(
            "each checkpoint generation must have ten strictly prior gradients"
        )
    if generations[-1] > int(n_iterations):
        raise ValueError(
            "checkpoint generations cannot exceed configured iterations"
        )
    return {
        "schema_version": 1,
        "generations": generations,
        "gradient_archive_length": archive_length,
        "selection_policy": "fixed_config_generations_only",
        "reward_selection_used": False,
        "current_generation_gradient_excluded": True,
        "artifact_manifest": CHECKPOINT_CAPTURE_MANIFEST,
        "artifact_directory": CHECKPOINT_CAPTURE_DIRECTORY,
        "training_config_artifact": CHECKPOINT_TRAINING_CONFIG_ARTIFACT,
    }


def _production_checkpoint_value_matches(actual: Any, expected: Any) -> bool:
    """Compare a resolved config value without accepting truthy substitutes."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return isinstance(actual, (bool, np.bool_)) and bool(actual) is expected
    if isinstance(expected, int):
        return (
            not isinstance(actual, (bool, np.bool_))
            and isinstance(actual, (int, np.integer))
            and int(actual) == expected
        )
    if isinstance(expected, float):
        if isinstance(actual, (bool, np.bool_)):
            return False
        try:
            numeric = float(actual)
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(np.isfinite(numeric) and numeric == expected)
    if isinstance(expected, list):
        return isinstance(actual, list) and actual == expected
    return actual == expected


def _validate_lagged_subspace_checkpoint_protocol(
    config: dict[str, Any],
    *,
    seed: Any,
    checkpoint_settings: dict[str, Any] | None,
) -> None:
    """Fail closed on any drift from the preregistered checkpoint generator."""
    protocol = config.get("checkpoint_capture_protocol")
    if protocol is None:
        return
    if protocol != LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL:
        raise ValueError(
            "checkpoint_capture_protocol must equal "
            f"{LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL!r}; got {protocol!r}"
        )
    if checkpoint_settings is None:
        raise ValueError(
            f"{LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL} requires checkpoint capture"
        )

    expected_config: dict[str, Any] = {
        "algorithm": "standard_es",
        "antithetic": True,
        "buffer_size": 0,
        "checkpoint_capture_generations": [50, 150, 250],
        "checkpoint_gradient_archive_length": 10,
        "common_rollout_seed": True,
        "condition": "standard_es",
        "curvature_beta": 0.0,
        "curvature_clip": 0.0,
        "eval_episodes": 0,
        "evaluate_center_fitness": False,
        "heldout_evaluation_enabled": False,
        "hidden_dims": [64, 64],
        "implicit_damping": 0.0,
        "init_param_std": 0.1,
        "l2_coeff": 0.0,
        "learning_rate": 1e-4,
        "lr_schedule": "constant",
        "max_episode_steps": 1000,
        "max_grad_norm": 0.0,
        "max_param_norm": None,
        "min_replay_weight_mass": 0.0,
        "n_iterations": 250,
        "noise_std": 0.02,
        "obs_norm_calibration_episodes": 3,
        "obs_norm_mode": "frozen_after_calibration",
        "online_evaluation_enabled": False,
        "output_activation": "tanh",
        "population_size": 200,
        "rank_fitness": True,
        "replay_enabled": False,
        "reuse_fraction": 0.0,
        "scalar_damping": 0.0,
        "training_env_step_budget": None,
        "use_curvature": False,
        "use_obs_norm": True,
        "activation": "tanh",
    }
    mismatches = [
        f"{key}={config.get(key)!r} (expected {expected!r})"
        for key, expected in expected_config.items()
        if not _production_checkpoint_value_matches(
            config.get(key), expected
        )
    ]
    if config.get("env_name") not in _LAGGED_SUBSPACE_CHECKPOINT_TASKS:
        mismatches.append(
            f"env_name={config.get('env_name')!r} "
            f"(expected one of {sorted(_LAGGED_SUBSPACE_CHECKPOINT_TASKS)!r})"
        )
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) not in range(300, 320)
    ):
        mismatches.append(f"seed={seed!r} (expected an integer from 300 through 319)")
    if checkpoint_settings.get("generations") != [50, 150, 250]:
        mismatches.append(
            "resolved checkpoint generations are not [50, 150, 250]"
        )
    if checkpoint_settings.get("gradient_archive_length") != 10:
        mismatches.append("resolved checkpoint gradient archive length is not 10")

    forbidden = sorted(
        key for key in _LAGGED_SUBSPACE_FORBIDDEN_CONFIG_KEYS if key in config
    )
    if forbidden:
        mismatches.append(
            "forbidden mechanism-specific settings are present: "
            + ", ".join(forbidden)
        )
    if mismatches:
        raise ValueError(
            f"{LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL} configuration drift: "
            + "; ".join(mismatches)
        )


def _resolve_training_env_step_budget(config: dict[str, Any]) -> int | None:
    """Resolve the optional complete-generation training-step budget."""
    value = config.get("training_env_step_budget")
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("training_env_step_budget must be a positive integer")
    try:
        integer_value = int(value)
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "training_env_step_budget must be a positive integer"
        ) from error
    if (
        not np.isfinite(numeric_value)
        or numeric_value != float(integer_value)
        or integer_value <= 0
    ):
        raise ValueError("training_env_step_budget must be a positive integer")
    return integer_value


def _training_budget_metadata(
    target: int,
    reached: int,
    max_iterations: int,
    stopping_reason: str,
) -> dict[str, Any]:
    target = int(target)
    reached = int(reached)
    max_iterations = int(max_iterations)
    if target <= 0 or reached < 0 or max_iterations <= 0:
        raise ValueError("invalid training-budget metadata")
    return {
        "target": target,
        "reached": reached,
        "overshoot": max(0, reached - target),
        "unit": "training_environment_steps",
        "stopping_reason": str(stopping_reason),
        "generation_boundary": "first_complete_generation_at_or_above_target",
        "max_iterations_safety_cap": max_iterations,
    }


def _with_training_budget_status(
    status: dict[str, Any],
    target: int | None,
    reached: int,
    max_iterations: int,
    stopping_reason: str,
) -> dict[str, Any]:
    if target is None:
        return status
    result = dict(status)
    result["training_budget"] = _training_budget_metadata(
        target,
        reached,
        max_iterations,
        stopping_reason,
    )
    return result


def _select_heldout_checkpoint_steps(
    successive_training_steps: list[int] | tuple[int, ...],
    budget: int,
) -> list[int]:
    """Select the initial center and every center through the first budget crossing."""
    budget = int(budget)
    if budget <= 0:
        raise ValueError("held-out training-step budget must be positive")
    steps = [int(value) for value in successive_training_steps]
    if not steps or steps[0] != 0:
        raise ValueError("held-out checkpoint steps must start at zero")
    if any(current <= previous for previous, current in zip(steps, steps[1:])):
        raise ValueError("held-out checkpoint steps must be strictly increasing")
    for index, step in enumerate(steps):
        if step >= budget:
            return steps[: index + 1]
    return steps


def _heldout_metrics_at_budget(
    training_steps: list[int] | tuple[int, ...] | np.ndarray,
    returns: list[float] | tuple[float, ...] | np.ndarray,
    budget: int,
) -> tuple[float, float]:
    """Compute normalized trapezoidal AUC and the interpolated return at budget."""
    budget = int(budget)
    if budget <= 0:
        raise ValueError("held-out training-step budget must be positive")
    x = np.asarray(training_steps, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("held-out steps and returns must be equal-length vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("held-out steps and returns must be finite")
    if x[0] != 0.0 or np.any(np.diff(x) <= 0.0):
        raise ValueError("held-out checkpoint steps must start at zero and increase")
    if x[-1] < float(budget):
        raise ValueError("held-out checkpoints do not cover the training-step budget")

    return_at_budget = float(np.interp(float(budget), x, y))
    below_budget = x < float(budget)
    x_cut = np.concatenate((x[below_budget], [float(budget)]))
    y_cut = np.concatenate((y[below_budget], [return_at_budget]))
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(y_cut, x_cut) / float(budget))
    if not np.isfinite(auc) or not np.isfinite(return_at_budget):
        raise FloatingPointError("held-out evaluation produced a non-finite metric")
    return auc, return_at_budget


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return None


_HISTORY_CURVATURE_VECTOR_FIELDS = frozenset(
    {
        "h_split_first_components",
        "h_split_second_components",
        "curvature_same_generation_components",
        "curvature_same_generation_se_components",
        "curvature_step_state_components",
        "curvature_step_state_se_components",
        "curvature_confidence_upper_components",
        "curvature_block_sizes",
        "curvature_raw_components",
        "curvature_ema_components",
        "curvature_ema_variance_components",
        "curvature_bias_corrected_ema_components",
        "curvature_step_components",
        "concave_curvature_components",
        "denominator_components",
    }
)
_HISTORY_CURVATURE_VECTOR_MAX_LENGTH = 64


def _history_curvature_vector(
    key: str, value: Any
) -> tuple[list[Any] | None, int]:
    """Validate and serialize one explicitly approved component vector."""
    if key not in _HISTORY_CURVATURE_VECTOR_FIELDS:
        raise KeyError(f"history vector field is not approved: {key}")
    vector = np.asarray(value)
    if vector.ndim != 1:
        raise ValueError(f"history vector {key} must be one-dimensional")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"history vector {key} must be numeric") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"history vector {key} must contain only finite values")
    length = int(len(vector))
    if length > _HISTORY_CURVATURE_VECTOR_MAX_LENGTH:
        return None, length
    return vector.tolist(), length


def _write_json_atomic(path: str, value: Any) -> None:
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
    os.replace(temporary_path, path)


def _write_bytes_atomic(path: str, payload: bytes) -> None:
    temporary_path = f"{path}.tmp.{os.getpid()}"
    with open(temporary_path, "wb") as stream:
        stream.write(payload)
    os.replace(temporary_path, path)


def _write_npz_atomic(path: str, **arrays: np.ndarray) -> None:
    temporary_path = f"{path}.tmp.{os.getpid()}.npz"
    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            if "/" in name or "\\" in name:
                raise ValueError("NPZ array names cannot contain path separators")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            entry = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            entry.compress_type = zipfile.ZIP_STORED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())
    os.replace(temporary_path, path)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _labeled_arrays_sha256(
    values: list[tuple[str, np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for label, value in values:
        encoded_label = str(label).encode("utf-8")
        digest.update(len(encoded_label).to_bytes(8, "big"))
        digest.update(encoded_label)
        digest.update(bytes.fromhex(_array_sha256(value)))
    return digest.hexdigest()


def _checkpoint_training_config_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
    excluded = {
        "_config_path",
        "provenance",
        "resolved_checkpoint_capture",
        "training_budget",
    }
    return {
        key: config[key]
        for key in sorted(config)
        if key not in excluded
    }


def _checkpoint_capture_status_metadata(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": manifest["status"],
        "artifact": CHECKPOINT_CAPTURE_MANIFEST,
        "requested_generations": list(manifest["requested_generations"]),
        "captured_generations": list(manifest["captured_generations"]),
        "gradient_archive_length": int(manifest["gradient_archive_length"]),
        "selection_policy": manifest["selection_policy"],
        "reward_selection_used": False,
        "current_generation_gradient_excluded": True,
    }


def _with_checkpoint_capture_status(
    status: dict[str, Any],
    manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    if manifest is None:
        return status
    result = dict(status)
    result["checkpoint_capture"] = _checkpoint_capture_status_metadata(
        manifest
    )
    return result


def _write_checkpoint_capture_manifest(
    output_dir: str, manifest: dict[str, Any]
) -> None:
    _write_json_atomic(
        os.path.join(output_dir, CHECKPOINT_CAPTURE_MANIFEST),
        manifest,
    )


def _capture_checkpoint_artifact(
    *,
    output_dir: str,
    generation: int,
    params: np.ndarray,
    obs_normalizer: ObsNormalizer | None,
    gradient_archive: deque[tuple[int, np.ndarray]],
    settings: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    generation = int(generation)
    archive_length = int(settings["gradient_archive_length"])
    expected_generations = list(
        range(generation - archive_length, generation)
    )
    archived_generations = [int(index) for index, _ in gradient_archive]
    if archived_generations != expected_generations:
        raise RuntimeError(
            "checkpoint gradient archive is not exactly the ten strictly "
            f"prior generations for theta_{generation}: expected "
            f"{expected_generations}, got {archived_generations}"
        )
    center_params = np.asarray(params, dtype=np.float64)
    proposal_gradients = np.stack(
        [np.asarray(gradient, dtype=np.float64) for _, gradient in gradient_archive]
    )
    if (
        center_params.ndim != 1
        or proposal_gradients.shape != (archive_length, len(center_params))
        or not np.all(np.isfinite(center_params))
        or not np.all(np.isfinite(proposal_gradients))
    ):
        raise FloatingPointError("checkpoint parameters or gradients are invalid")
    gradient_generations = np.asarray(expected_generations, dtype=np.int64)
    if obs_normalizer is None:
        obs_enabled = np.asarray(False, dtype=np.bool_)
        obs_mean = np.zeros(0, dtype=np.float64)
        obs_var = np.zeros(0, dtype=np.float64)
        obs_count = np.asarray(0.0, dtype=np.float64)
    else:
        obs_state = obs_normalizer.get_state()
        obs_enabled = np.asarray(True, dtype=np.bool_)
        obs_mean = np.asarray(obs_state["mean"], dtype=np.float64)
        obs_var = np.asarray(obs_state["var"], dtype=np.float64)
        obs_count = np.asarray(obs_state["count"], dtype=np.float64)
        if (
            not np.all(np.isfinite(obs_mean))
            or not np.all(np.isfinite(obs_var))
            or not np.isfinite(float(obs_count))
            or np.any(obs_var < 0.0)
            or float(obs_count) < 0.0
        ):
            raise FloatingPointError(
                "checkpoint observation-normalizer state is invalid"
            )

    relative_path = os.path.join(
        CHECKPOINT_CAPTURE_DIRECTORY,
        f"checkpoint_generation_{generation:06d}.npz",
    )
    artifact_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    _write_npz_atomic(
        artifact_path,
        schema_version=np.asarray(2, dtype=np.int64),
        checkpoint_generation=np.asarray(generation, dtype=np.int64),
        study_source_sha256=np.asarray(
            manifest["source_sha256"].encode("ascii"), dtype="S64"
        ),
        training_config_sha256=np.asarray(
            manifest["training_config_sha256"].encode("ascii"), dtype="S64"
        ),
        center_params=center_params,
        obs_normalizer_enabled=obs_enabled,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=obs_count,
        gradient_generations=gradient_generations,
        proposal_gradients=proposal_gradients,
    )

    params_sha256 = _array_sha256(center_params)
    obs_state_sha256 = _labeled_arrays_sha256(
        [
            ("enabled", obs_enabled),
            ("mean", obs_mean),
            ("var", obs_var),
            ("count", obs_count),
        ]
    )
    gradient_rows = [
        {
            "generation": int(index),
            "sha256": _array_sha256(proposal_gradients[row]),
        }
        for row, index in enumerate(expected_generations)
    ]
    indexed_gradient_archive_sha256 = _labeled_arrays_sha256(
        [
            ("gradient_generations", gradient_generations),
            ("proposal_gradients", proposal_gradients),
        ]
    )
    payload_identity = {
        "schema_version": 2,
        "checkpoint_generation": generation,
        "center_params_sha256": params_sha256,
        "observation_normalizer_state_sha256": obs_state_sha256,
        "indexed_gradient_archive_sha256": indexed_gradient_archive_sha256,
        "training_config_sha256": manifest["training_config_sha256"],
        "source_sha256": manifest["source_sha256"],
    }
    artifact_metadata = {
        "checkpoint_generation": generation,
        "checkpoint_index": settings["generations"].index(generation),
        "artifact": relative_path,
        "artifact_sha256": _sha256_file(artifact_path),
        "payload_identity_sha256": _sha256_bytes(
            _canonical_json_bytes(payload_identity)
        ),
        "center_semantics": f"theta_{generation}_after_{generation}_updates",
        "center_params_sha256": params_sha256,
        "observation_normalizer_enabled": bool(obs_enabled),
        "observation_normalizer_state_sha256": obs_state_sha256,
        "gradient_archive_length": archive_length,
        "gradient_generations": expected_generations,
        "gradient_generation_start": expected_generations[0],
        "gradient_generation_end": expected_generations[-1],
        "last_applied_gradient_generation": generation - 1,
        "current_checkpoint_gradient_generation": generation,
        "current_checkpoint_gradient_included": False,
        "strictly_prior_gradient_archive": True,
        "capture_timing": (
            "after_theta_update_before_sampling_checkpoint_generation"
        ),
        "proposal_gradient_hashes": gradient_rows,
        "indexed_gradient_archive_sha256": indexed_gradient_archive_sha256,
        "training_config_sha256": manifest["training_config_sha256"],
        "source_sha256": manifest["source_sha256"],
    }
    manifest["artifacts"].append(artifact_metadata)
    manifest["captured_generations"].append(generation)
    _write_checkpoint_capture_manifest(output_dir, manifest)


def _append_json_line(path: str, value: Any) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        json.dump(value, stream, separators=(",", ":"))
        stream.write("\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_heldout_checkpoint(
    params: np.ndarray,
    obs_normalizer: ObsNormalizer | None,
    training_env_steps: int,
    source_iteration: int | None,
) -> dict[str, Any]:
    if obs_normalizer is None:
        obs_mean, obs_var = None, None
    else:
        obs_mean, obs_var = obs_normalizer.get_mean_var()
    return {
        "params": np.asarray(params, dtype=np.float64).copy(),
        "obs_mean": None if obs_mean is None else obs_mean.copy(),
        "obs_var": None if obs_var is None else obs_var.copy(),
        "training_env_steps": int(training_env_steps),
        "source_iteration": source_iteration,
    }


def _heldout_status_metadata(
    settings: dict[str, int],
    state: str,
    checkpoint_count: int,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "status": state,
        "artifact": "heldout_evaluation.json",
        "training_step_budget": settings["heldout_training_step_budget"],
        "episodes_per_checkpoint": settings["heldout_eval_episodes"],
        "checkpoint_count": int(checkpoint_count),
    }
    if evaluation is not None:
        metadata["normalized_auc_at_budget"] = float(
            evaluation["normalized_auc_at_budget"]
        )
        metadata["return_at_budget"] = float(evaluation["return_at_budget"])
    return metadata


def _with_heldout_status(
    status: dict[str, Any],
    settings: dict[str, int] | None,
    state: str,
    checkpoint_count: int,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if settings is None:
        return status
    result = dict(status)
    result["heldout_evaluation"] = _heldout_status_metadata(
        settings,
        state,
        checkpoint_count,
        evaluation,
    )
    return result


def _evaluate_heldout_checkpoints(
    pool: Pool,
    checkpoints: list[dict[str, Any]],
    *,
    seed: int,
    settings: dict[str, int],
    obs_scale: float,
) -> dict[str, Any]:
    budget = settings["heldout_training_step_budget"]
    episodes = settings["heldout_eval_episodes"]
    checkpoint_steps = [int(row["training_env_steps"]) for row in checkpoints]
    selected_steps = _select_heldout_checkpoint_steps(checkpoint_steps, budget)
    if selected_steps != checkpoint_steps or checkpoint_steps[-1] < budget:
        raise RuntimeError(
            "held-out checkpoints must end at the first center crossing the budget"
        )

    rollout_seeds = [
        _heldout_evaluation_rollout_seed(seed, eval_index)
        for eval_index in range(episodes)
    ]
    records: list[dict[str, Any]] = []
    total_env_steps = 0
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        tasks = [
            (
                checkpoint["params"],
                rollout_seed,
                checkpoint["obs_mean"],
                checkpoint["obs_var"],
                False,
                obs_scale,
            )
            for rollout_seed in rollout_seeds
        ]
        results = pool.map(_evaluate_params, tasks)
        episode_returns = [float(result[0]) for result in results]
        episode_env_steps = [int(result[2]) for result in results]
        if (
            not np.all(np.isfinite(episode_returns))
            or any(value <= 0 for value in episode_env_steps)
        ):
            raise FloatingPointError("held-out evaluation produced an invalid rollout")
        total_env_steps += sum(episode_env_steps)
        records.append(
            {
                "checkpoint_index": checkpoint_index,
                "source_iteration": checkpoint["source_iteration"],
                "training_env_steps": checkpoint["training_env_steps"],
                "mean_return": float(np.mean(episode_returns)),
                "episode_returns": episode_returns,
                "episode_env_steps": episode_env_steps,
            }
        )

    auc, return_at_budget = _heldout_metrics_at_budget(
        checkpoint_steps,
        [record["mean_return"] for record in records],
        budget,
    )
    return {
        "schema_version": 1,
        "training_step_budget": budget,
        "episodes_per_checkpoint": episodes,
        "checkpoint_selection": "initial_and_every_center_through_first_budget_crossing",
        "rollout_seed_stream": 4,
        "rollout_seeds": rollout_seeds,
        "common_seed_bank_across_checkpoints": True,
        "optimizer_or_checkpoint_selection_uses_heldout_results": False,
        "observation_normalizer_state": "frozen_per_checkpoint",
        "checkpoint_count": len(records),
        "heldout_evaluation_env_steps": int(total_env_steps),
        "normalized_auc_at_budget": auc,
        "return_at_budget": return_at_budget,
        "checkpoints": records,
    }


def _source_digest(config_path: str | None) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    relative_paths = [
        "core/__init__.py",
        "core/diiwes.py",
        "core/implicit_es.py",
        "core/standard_es.py",
        "core/policies.py",
        "docs/hopper_fresh_optimizer_development_protocol.md",
        "docs/hopper_hessian_confirmation_preregistration.md",
        "docs/lagged_subspace_frozen_checkpoint_protocol.md",
        "environment.yml",
        "experiments/manifests/hopper_fresh_optimizer_development.json",
        "experiments/train.py",
        "requirement.txt",
        "scripts/diagnose_structured_curvature.py",
        "scripts/submit_hopper_hessian_confirmation.sh",
        "scripts/submit_hopper_hessian_fix_sweep.sh",
        "scripts/submit_hopper_no_replay_sweep.sh",
        "scripts/submit_hopper_fresh_optimizer_development.sh",
        "scripts/summarize_hopper_fresh_optimizer_development.py",
        "scripts/summarize_hopper_hessian_confirmation.py",
        "scripts/summarize_hopper_implicit_sweep.py",
        "utilities/__init__.py",
        "utilities/obs_norm.py",
    ]
    labeled_paths = [
        (relative_path, os.path.join(root, relative_path))
        for relative_path in relative_paths
    ]
    if config_path:
        labeled_paths.append(("experiment_config", os.path.abspath(config_path)))
    digest = hashlib.sha256()
    for label, path in sorted(labeled_paths):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with open(path, "rb") as stream:
            digest.update(stream.read())
    return digest.hexdigest()


def _git_revision() -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("gymnasium", "mujoco", "PyYAML"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _resolved_optimizer_config(
    optimizer: (
        StandardES
        | MomentumES
        | AdamES
        | ClipUpES
        | SNES
        | DIIWES
        | EndpointImplicitES
        | LinearizedImplicitES
        | ConcaveCurvatureES
        | LOPOGradientES
    ),
) -> dict[str, Any]:
    resolved = {
        "type": optimizer.__class__.__name__,
        "population_size": int(optimizer.population_size),
        "initial_learning_rate": float(optimizer.learning_rate),
        "noise_std": float(optimizer.noise_std),
        "rank_fitness": bool(optimizer.rank_fitness),
        "l2_coeff": float(optimizer.l2_coeff),
        "antithetic": bool(optimizer.antithetic),
        "max_grad_norm": float(optimizer.max_grad_norm),
        "max_param_norm": optimizer.max_param_norm,
        "trust_region": False,
        "replay_enabled": False,
        "persists_hessian_ema_artifact": bool(
            getattr(optimizer, "persist_hessian_ema_artifact", False)
        ),
        "hessian_ema_artifact": (
            HESSIAN_EMA_ARTIFACT
            if getattr(optimizer, "persist_hessian_ema_artifact", False)
            else None
        ),
    }
    if isinstance(optimizer, SNES):
        resolved.update(
            {
                "method": "snes",
                "update_rule": "separable_gaussian_natural_gradient",
                "initial_coordinate_sigma": float(optimizer.noise_std),
                "mean_learning_rate_source": "learning_rate_schedule",
                "sigma_learning_rate": float(
                    optimizer.snes_sigma_learning_rate
                ),
                "default_sigma_learning_rate": float(
                    optimizer.snes_default_sigma_learning_rate
                ),
                "uses_default_sigma_learning_rate": bool(
                    optimizer.snes_uses_default_sigma_learning_rate
                ),
                "utility_shaping": "canonical_log_rank_tie_averaged",
                "search_covariance": "learned_diagonal",
                "sigma_parameterization": "coordinate_std_exponential_update",
                "sigma_clipping": False,
                "final_search_std_artifact": SNES_FINAL_SEARCH_STD_ARTIFACT,
                "final_search_std_artifact_semantics": (
                    "final_optimizer_state_for_audit_not_a_resume_checkpoint"
                ),
                "antithetic_sampling": bool(optimizer.antithetic),
                "reference_deviations": [
                    "configured_population_size",
                    "optional_antithetic_sampling",
                    "tie_averaged_rank_utilities",
                    "mean_rate_may_use_shared_scheduler",
                    "no_adaptation_sampling_or_restarts",
                ],
            }
        )
    elif isinstance(optimizer, MomentumES):
        resolved.update(
            {
                "method": "momentum_es",
                "update_rule": "heavy_ball_momentum",
                "momentum_beta": float(optimizer.momentum_beta),
            }
        )
    elif isinstance(optimizer, AdamES):
        resolved.update(
            {
                "method": "adam_es",
                "update_rule": "bias_corrected_adam",
                "adam_beta1": float(optimizer.adam_beta1),
                "adam_beta2": float(optimizer.adam_beta2),
                "adam_epsilon": float(optimizer.adam_epsilon),
                "adam_bias_correction": True,
            }
        )
    elif isinstance(optimizer, ClipUpES):
        resolved.update(
            {
                "method": "clipup_es",
                "update_rule": "normalized_gradient_momentum_velocity_clip",
                "clipup_momentum": float(optimizer.clipup_momentum),
                "clipup_max_speed": float(optimizer.clipup_max_speed),
                "clipup_step_size_source": "learning_rate_schedule",
                "clipup_gradient_normalization": True,
                "clipup_velocity_clipping": True,
            }
        )
    elif isinstance(optimizer, EndpointImplicitES):
        resolved.update(
            {
                "method": "endpoint_implicit",
                "implicit_damping": float(optimizer.implicit_damping),
                "implicit_iterations": int(optimizer.implicit_iterations),
                "implicit_tolerance": float(optimizer.implicit_tolerance),
                "importance_ratio_clipping": False,
                "diagnostic_ratio_floor": float(optimizer.diagnostic_ratio_floor),
                "diagnostic_ratio_cap": float(optimizer.diagnostic_ratio_cap),
                "solver_type": "picard_endpoint_implicit",
                "endpoint_weights_recomputed": True,
                "endpoint_scores_recomputed": True,
            }
        )
    elif isinstance(optimizer, LinearizedImplicitES):
        resolved.update(
            {
                "method": "linearized_implicit",
                "implicit_damping": float(optimizer.implicit_damping),
                "curvature_fitness": optimizer.curvature_fitness,
                "curvature_mode": optimizer.curvature_mode,
                "curvature_beta": 0.0,
                "curvature_clipping": False,
                "min_abs_diagonal": float(optimizer.min_abs_diagonal),
                "solver_type": "signed_diagonal_linearized_implicit",
            }
        )
    elif isinstance(optimizer, LOPOGradientES):
        pair_count = optimizer.population_size // 2
        resolved.update(
            {
                "method": "lopo_gradient_only",
                "update_rule": "explicit_exact_lopo_rank_gradient",
                "sample_reuse": False,
                "importance_weighting": False,
                "implicit_damping": float(optimizer.implicit_damping),
                "curvature_used": False,
                "curvature_fitness": "none",
                "curvature_mode": "none",
                "curvature_structure": "none",
                "curvature_beta": 0.0,
                "curvature_estimator": "none",
                "curvature_rank_utility_mode": "lopo_rank_u_statistic",
                "curvature_attenuation_mode": "none",
                "curvature_components": 0,
                "lopo_rank_utility_semantics": (
                    "exact_leave_own_antithetic_pair_out_midranks_no_recentering"
                ),
                "lopo_centering_operation_applied": False,
                "lopo_structural_zero_sum": True,
                "lopo_c_m": float(
                    2.0 * (pair_count - 1.0) / (2.0 * pair_count - 1.0)
                ),
                "lopo_matching_scope": (
                    "population_current_mid_cdf_stop_gradient"
                ),
                "raw_lopo_block_moment_endpoint_jacobian_claim_applicability": (
                    "not_applicable_no_curvature_operator"
                ),
                "raw_lopo_block_moment_endpoint_jacobian_nonuse_reason": (
                    "gradient_only_condition_has_no_curvature_operator"
                ),
                "full_endpoint_jacobian_operator_claim": False,
                "projected_curvature_operator_endpoint_jacobian_claim": False,
                "off_proposal_endpoint_jacobian_claim": False,
                "global_adaptive_rank_objective_hessian_claim": False,
                "raw_return_hessian_claim": False,
                "attribution_role": "lopo_gradient_without_curvature",
                "solver_type": "explicit_lopo_rank_gradient",
            }
        )
    elif isinstance(optimizer, ConcaveCurvatureES):
        resolved.update(
            {
                "method": "concave_curvature",
                "implicit_damping": float(optimizer.implicit_damping),
                "curvature_fitness": optimizer.curvature_fitness,
                "curvature_mode": optimizer.curvature_mode,
                "curvature_structure": optimizer.curvature_structure,
                "curvature_beta": float(optimizer.curvature_beta),
                "curvature_same_generation": bool(
                    optimizer.curvature_same_generation
                ),
                "curvature_estimator": optimizer.curvature_estimator,
                "curvature_rank_utility_mode": optimizer.rank_utility_mode,
                "curvature_confidence_z": optimizer.curvature_confidence_z,
                "curvature_attenuation_mode": optimizer.attenuation_mode,
                "curvature_clipping": False,
                "curvature_projection": "concave",
                "curvature_components": int(optimizer.num_curvature_components),
                "solver_type": (
                    (
                        "concave_projected_block_lopo_rank_u_statistic"
                        if optimizer.attenuation_mode == "structured"
                        else (
                            "concave_projected_block_lopo_rank_u_statistic_"
                            "isotropic_norm_control"
                        )
                    )
                    if optimizer.rank_utility_mode == "lopo_rank_u_statistic"
                    else (
                        f"concave_projected_{optimizer.curvature_structure}"
                        if optimizer.attenuation_mode == "structured"
                        else "concave_projected_block_isotropic_attenuation_control"
                    )
                ),
            }
        )
        if optimizer.rank_utility_mode == "lopo_rank_u_statistic":
            resolved.update(
                {
                    "sample_reuse": False,
                    "importance_weighting": False,
                    "curvature_rank_utility_semantics": (
                        "exact_leave_own_antithetic_pair_out_midranks_no_recentering"
                    ),
                    "lopo_centering_operation_applied": False,
                    "lopo_structural_zero_sum": True,
                    "curvature_standard_error_method": (
                        "delete_one_antithetic_pair_order_two_u_statistic"
                    ),
                    "curvature_standard_error_scope": (
                        "componentwise_asymptotic_non_simultaneous"
                    ),
                    "curvature_standard_error_target": (
                        "raw_same_generation_block_u_statistic"
                    ),
                    "curvature_standard_error_optimization_coverage_calibrated": False,
                    "curvature_within_pair_dependence_allowed": True,
                    "curvature_across_pair_inference_assumption": (
                        "iid_nondegenerate_pair_clusters"
                    ),
                    "curvature_inference_assumptions_runtime_verified": False,
                    "curvature_minimum_antithetic_pairs": 3,
                    "lopo_matching_scope": (
                        "population_current_mid_cdf_stop_gradient_and_block_curvature"
                    ),
                    "raw_lopo_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average": True,
                    "raw_lopo_block_moment_endpoint_jacobian_scope": (
                        "at_proposal_frozen_lopo_utility_self_normalized_map_"
                        "raw_preprojection_block_average_of_diagonal"
                    ),
                    "full_endpoint_jacobian_operator_claim": False,
                    "projected_curvature_operator_endpoint_jacobian_claim": False,
                    "off_proposal_endpoint_jacobian_claim": False,
                    "global_adaptive_rank_objective_hessian_claim": False,
                    "raw_return_hessian_claim": False,
                    "attribution_role": (
                        "lopo_structured_block_curvature"
                        if optimizer.attenuation_mode == "structured"
                        else "lopo_isotropic_norm_matched_attenuation_control"
                    ),
                }
            )
    elif isinstance(optimizer, DIIWES):
        resolved.update(
            {
                "reuse_fraction": float(optimizer.reuse_fraction),
                "buffer_size": int(optimizer.buffer_size),
                "buffer_sampling": optimizer.buffer_sampling,
                "min_importance_weight": float(optimizer.min_importance_weight),
                "max_importance_weight": float(optimizer.max_importance_weight),
                "max_sample_age": int(optimizer.max_sample_age),
                "ess_min_ratio": float(optimizer.ess_min_ratio),
                "min_replay_weight_mass": float(optimizer.min_replay_weight_mass),
                "scalar_damping": float(optimizer.scalar_damping),
                "use_curvature": bool(optimizer.use_curvature),
                "curvature_fitness": optimizer.curvature_fitness,
                "curvature_mode": optimizer.curvature_mode,
                "curvature_beta": float(optimizer.curvature_beta),
                "curvature_clip": float(optimizer.curvature_clip),
                "use_leave_one_out_curvature_baseline": bool(
                    optimizer.use_leave_one_out_curvature_baseline
                ),
                "bias_correct_curvature_ema": bool(optimizer.bias_correct_curvature_ema),
                "solver_type": "projected_diagonal_closed_form",
            }
        )
    return resolved


def _history_record(
    iteration: int,
    eval_reward: float | None,
    best_reward: float | None,
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
    initial_eval_reward: float | None,
    initial_eval_env_steps: int,
    normalization_calibration_env_steps: int,
) -> dict[str, Any]:
    total_env_steps = (
        int(normalization_calibration_env_steps)
        + int(train_env_steps)
        + int(eval_env_steps)
    )
    total_env_steps_iter = int(train_env_steps_iter) + int(eval_env_steps_iter)
    record = {
        "iteration": int(iteration),
        "eval_reward": (
            None if eval_reward is None else float(eval_reward)
        ),
        "best_reward": (
            None if best_reward is None else float(best_reward)
        ),
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
        "iteration_compute_seconds": float(elapsed),
        "env_steps": int(train_env_steps),
        "env_steps_iter": int(train_env_steps_iter),
        "train_env_steps": int(train_env_steps),
        "train_env_steps_iter": int(train_env_steps_iter),
        "training_env_steps": int(train_env_steps),
        "training_env_steps_iter": int(train_env_steps_iter),
        "eval_env_steps": int(eval_env_steps),
        "eval_env_steps_iter": int(eval_env_steps_iter),
        "initial_eval_reward": (
            None
            if initial_eval_reward is None
            else float(initial_eval_reward)
        ),
        "initial_eval_env_steps": int(initial_eval_env_steps),
        "normalization_calibration_env_steps": int(normalization_calibration_env_steps),
        "total_env_steps": total_env_steps,
        "total_env_steps_iter": total_env_steps_iter,
    }
    for key, value in info.items():
        if key in record:
            continue
        if key in _HISTORY_CURVATURE_VECTOR_FIELDS:
            serialized, length = _history_curvature_vector(key, value)
            record[f"{key}_length"] = length
            if serialized is None:
                record[f"{key}_omitted"] = True
                record[f"{key}_serialization"] = (
                    "omitted_length_exceeds_64"
                )
            else:
                record[key] = serialized
                record[f"{key}_omitted"] = False
                record[f"{key}_serialization"] = "persisted"
            continue
        scalar = _json_scalar(value)
        if scalar is not None:
            record[key] = scalar
    return record


def _format_progress(record: dict[str, Any], verbose: bool) -> str:
    if record["eval_reward"] is None:
        evaluation = "disabled"
        best = "disabled"
    else:
        evaluation = f"{record['eval_reward']:8.2f}"
        best = f"{record['best_reward']:8.2f}"
    if not verbose:
        return (
            f"Iter {record['iteration']:4d} | "
            f"Eval {evaluation} | "
            f"Best {best}"
        )
    return (
        f"Iter {record['iteration']:4d} | "
        f"Eval {evaluation} | "
        f"Best {best} | "
        f"LR {record['lr']:.4g} | "
        f"Step {record['step_norm']:.3f} | "
        f"Proposed {record.get('proposed_step_norm', record['step_norm']):.3f} | "
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
) -> tuple[float | None, np.ndarray]:
    config = dict(config)
    _validate_named_lopo_condition_semantics(config)
    _validate_no_replay_protocol(config)
    online_evaluation_enabled = _resolve_online_evaluation_enabled(config)
    heldout_settings = _resolve_heldout_evaluation_config(config)
    training_step_budget = _resolve_training_env_step_budget(config)
    if (
        training_step_budget is not None
        and heldout_settings is not None
        and heldout_settings["heldout_training_step_budget"]
        > training_step_budget
    ):
        raise ValueError(
            "heldout_training_step_budget cannot exceed "
            "training_env_step_budget"
        )
    verbose = bool(verbose or config.get("verbose", False))
    requested_seed = seed
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must be in [0, 2**32 - 1]")
    n_iterations = int(config.get("n_iterations", 500))
    checkpoint_capture_settings = _resolve_checkpoint_capture_config(
        config, n_iterations
    )
    _validate_lagged_subspace_checkpoint_protocol(
        config,
        seed=requested_seed,
        checkpoint_settings=checkpoint_capture_settings,
    )
    lagged_subspace_source_sha256: str | None = None
    lagged_subspace_provenance_locks: dict[str, str] | None = None
    if (
        config.get("checkpoint_capture_protocol")
        == LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL
    ):
        # This must precede environment construction and worker-pool creation.
        # It is one study-wide lock over all task configs and both stages, not
        # the legacy per-config digest retained for older studies below.
        lagged_subspace_provenance_locks = (
            require_checkpoint_generation_provenance_locks()
        )
        lagged_subspace_source_sha256 = (
            lagged_subspace_provenance_locks["source_sha256"]
        )
        config_path = config.get("_config_path")
        if not isinstance(config_path, str):
            raise ValueError(
                "production checkpoint generation requires the selected "
                "locked config path"
            )
        if (
            study_sha256_for_checkpoint_config(
                config_path, expected_env_name=str(config.get("env_name"))
            )
            != lagged_subspace_source_sha256
        ):
            raise ValueError(
                "selected checkpoint config does not share the enforced "
                "study source lock"
            )
    if n_workers <= 0:
        raise ValueError("n_workers must be positive")
    config["n_workers"] = int(n_workers)

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
    if optimizer.max_grad_norm != 0.0 or optimizer.max_param_norm is not None:
        raise ValueError("the experiment protocol forbids gradient or parameter norm projection")
    use_curvature = bool(getattr(optimizer, "use_curvature", False))
    curvature_fitness = str(getattr(optimizer, "curvature_fitness", "none"))
    curvature_mode = str(getattr(optimizer, "curvature_mode", "none"))

    obs_normalizer = ObsNormalizer(env.observation_space.shape) if config.get("use_obs_norm", False) else None
    if obs_normalizer is None:
        obs_norm_mode = "none"
        obs_norm_calibration_episodes = 0
    else:
        obs_norm_mode = str(config.get("obs_norm_mode", "online")).lower()
        if obs_norm_mode not in {"online", "frozen_after_calibration"}:
            raise ValueError("obs_norm_mode must be online or frozen_after_calibration")
        obs_norm_calibration_episodes = int(config.get("obs_norm_calibration_episodes", 0))
        if obs_norm_mode == "frozen_after_calibration" and obs_norm_calibration_episodes <= 0:
            raise ValueError("frozen observation normalization requires calibration episodes")
        if (
            obs_norm_mode == "online"
            and isinstance(optimizer, DIIWES)
            and optimizer.reuse_fraction > 0.0
        ):
            raise ValueError("online observation normalization is incompatible with replayed fitness")
    config["obs_norm_mode"] = obs_norm_mode
    config["obs_norm_calibration_episodes"] = obs_norm_calibration_episodes
    initialization_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    params = initialization_rng.standard_normal(policy.num_params)
    params *= float(config.get("init_param_std", 0.1))
    optimizer.current_params = params.copy()

    eval_episodes = int(config.get("eval_episodes", 3))
    eval_interval = int(config.get("eval_interval", 1))
    log_interval = int(config.get("log_interval", 10))
    base_lr = float(config.get("learning_rate", 0.02))
    lr_decay = float(config.get("lr_decay", 1.0))
    lr_schedule = str(
        config.get("lr_schedule", "exponential" if "lr_decay" in config else "constant")
    ).lower()
    _learning_rate_at(base_lr, 0, lr_schedule, exponential_decay=lr_decay)
    config["lr_schedule"] = lr_schedule
    config["online_evaluation_enabled"] = online_evaluation_enabled
    evaluate_center_fitness = bool(config.get("evaluate_center_fitness", False))
    common_rollout_seed = bool(config.get("common_rollout_seed", False))
    obs_scale = float(config.get("obs_scale", 1.0))
    if obs_scale <= 0.0:
        raise ValueError("obs_scale must be positive")
    if (
        n_iterations <= 0
        or (online_evaluation_enabled and eval_episodes <= 0)
        or eval_interval <= 0
        or log_interval <= 0
    ):
        raise ValueError("iteration and evaluation counts must be valid positive values")
    if evaluate_center_fitness and not use_curvature:
        raise ValueError("evaluate_center_fitness requires a curvature-enabled optimizer")
    if evaluate_center_fitness and curvature_fitness == "matched" and optimizer.rank_fitness:
        raise ValueError("matched rank curvature cannot transform a standalone center fitness")
    if checkpoint_capture_settings is not None:
        if online_evaluation_enabled:
            raise ValueError(
                "checkpoint generator runs require "
                "online_evaluation_enabled=false"
            )
        if type(optimizer) is not StandardES:
            raise ValueError(
                "checkpoint capture requires the plain StandardES optimizer"
            )
        if (
            not optimizer.rank_fitness
            or not optimizer.antithetic
            or optimizer.l2_coeff != 0.0
            or use_curvature
            or evaluate_center_fitness
        ):
            raise ValueError(
                "checkpoint capture requires antithetic centered-rank Standard "
                "ES without curvature, center evaluation, or L2"
            )
        if training_step_budget is not None:
            raise ValueError(
                "checkpoint capture requires fixed generations without a "
                "training-step early-stop budget"
            )
        if heldout_settings is not None:
            raise ValueError(
                "checkpoint capture cannot be combined with held-out evaluation"
            )

    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, "history.json")
    history_jsonl_path = os.path.join(output_dir, "history.jsonl")
    status_path = os.path.join(output_dir, "status.json")
    config_path = os.path.join(output_dir, "config.json")
    heldout_evaluation_path = os.path.join(output_dir, "heldout_evaluation.json")
    summary_path = os.path.join(output_dir, "summary.json")
    if os.path.exists(status_path) and not config.get("allow_overwrite", False):
        raise FileExistsError(f"output directory already contains a run status: {output_dir}")
    started_at = _utc_now()
    config["seed"] = int(seed)
    config["resolved_optimizer"] = _resolved_optimizer_config(optimizer)
    config["resolved_online_evaluation"] = {
        "enabled": online_evaluation_enabled,
        "episodes_per_evaluation": (
            eval_episodes if online_evaluation_enabled else 0
        ),
        "initial_evaluation": online_evaluation_enabled,
        "periodic_evaluation": online_evaluation_enabled,
        "final_evaluation": online_evaluation_enabled,
        "best_policy_selection_by_return": online_evaluation_enabled,
        "best_policy_artifacts": online_evaluation_enabled,
    }
    if training_step_budget is not None:
        config["training_env_step_budget"] = training_step_budget
        config["training_budget"] = _training_budget_metadata(
            training_step_budget,
            0,
            n_iterations,
            "running",
        )
    if heldout_settings is not None:
        config["resolved_heldout_evaluation"] = {
            "enabled": True,
            "artifact": os.path.basename(heldout_evaluation_path),
            "training_step_budget": heldout_settings[
                "heldout_training_step_budget"
            ],
            "episodes_per_checkpoint": heldout_settings["heldout_eval_episodes"],
            "checkpoint_selection": (
                "initial_and_every_center_through_first_budget_crossing"
            ),
            "execution_phase": "post_training",
            "rollout_seed_stream": 4,
            "common_seed_bank_across_checkpoints": True,
            "optimizer_or_checkpoint_selection_uses_heldout_results": False,
            "observation_normalizer_state": "frozen_per_checkpoint",
        }
    source_sha256 = (
        lagged_subspace_source_sha256
        if lagged_subspace_source_sha256 is not None
        else _source_digest(config.get("_config_path"))
    )
    checkpoint_capture_manifest: dict[str, Any] | None = None
    checkpoint_training_config_bytes: bytes | None = None
    if checkpoint_capture_settings is not None:
        checkpoint_training_config = _checkpoint_training_config_payload(
            config
        )
        checkpoint_training_config_bytes = _canonical_json_bytes(
            checkpoint_training_config
        )
        checkpoint_training_config_sha256 = _sha256_bytes(
            checkpoint_training_config_bytes
        )
        config["resolved_checkpoint_capture"] = {
            **checkpoint_capture_settings,
            "enabled": True,
            "source_sha256": source_sha256,
            "training_config_sha256": checkpoint_training_config_sha256,
            "training_gradient_semantics": (
                "repository_centered_rank_standard_es_proposal_gradient"
            ),
            "checkpoint_selection_by_reward": False,
            "online_evaluation_enabled": online_evaluation_enabled,
        }
        checkpoint_capture_manifest = {
            "schema_version": 1,
            "status": "running",
            "requested_generations": list(
                checkpoint_capture_settings["generations"]
            ),
            "captured_generations": [],
            "expected_checkpoint_count": len(
                checkpoint_capture_settings["generations"]
            ),
            "gradient_archive_length": CHECKPOINT_GRADIENT_ARCHIVE_LENGTH,
            "selection_policy": "fixed_config_generations_only",
            "reward_selection_used": False,
            "current_generation_gradient_excluded": True,
            "online_evaluation_enabled": online_evaluation_enabled,
            "training_config_artifact": (
                CHECKPOINT_TRAINING_CONFIG_ARTIFACT
            ),
            "training_config_sha256": checkpoint_training_config_sha256,
            "training_config_hash_scope": (
                "resolved_training_config_excluding_runtime_provenance_paths_"
                "and_checkpoint_hash_metadata"
            ),
            "source_sha256": source_sha256,
            "training_gradient_semantics": (
                "repository_centered_rank_standard_es_proposal_gradient"
            ),
            "artifacts": [],
            "validated_generator_controls": {
                "plain_standard_es": True,
                "rank_fitness": True,
                "antithetic": True,
                "replay": False,
                "importance_sampling": False,
                "trust_region": False,
                "picard_iteration": False,
                "gradient_clipping": False,
                "parameter_projection": False,
                "curvature": False,
                "curvature_clipping": False,
                "l2": False,
                "checkpoint_selection_by_reward": False,
            },
        }
    config["provenance"] = {
        "git_revision": _git_revision(),
        "source_sha256": source_sha256,
        "expected_source_sha256": os.environ.get("PAPER_EXPECTED_SOURCE_SHA"),
        "argv": list(sys.argv),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "numpy": np.__version__,
        "dependencies": _dependency_versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "started_at": started_at,
        "rng_scheme": {
            "optimizer": "numpy.RandomState(run_seed)",
            "parameter_initialization": "numpy.default_rng(SeedSequence([run_seed, 1]))",
            "rollout": "injective Cantor encoding of (run_seed, stream, iteration, index)",
        },
    }
    optional_provenance_environment = {
        "PAPER_EXPECTED_MANIFEST_SHA256": "expected_manifest_sha256",
        "PAPER_EXPECTED_PROTOCOL_SHA256": "expected_protocol_sha256",
        "PAPER_EXPECTED_ANALYZER_SHA256": "expected_analyzer_sha256",
        "PAPER_EXPECTED_LAUNCHER_SHA256": "expected_launcher_sha256",
        "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256": "expected_launcher_sha256",
        "PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256": (
            "expected_dependency_lock_sha256"
        ),
        "PAPER_SOURCE_GIT_REVISION": "source_git_revision",
        "PAPER_REPO_DIR": "source_repo_dir",
    }
    for environment_key, provenance_key in optional_provenance_environment.items():
        value = os.environ.get(environment_key)
        if value:
            config["provenance"][provenance_key] = value
    if lagged_subspace_provenance_locks is not None:
        config["provenance"].update(
            {
                "expected_source_sha256": (
                    lagged_subspace_provenance_locks["source_sha256"]
                ),
                "expected_manifest_sha256": (
                    lagged_subspace_provenance_locks["manifest_sha256"]
                ),
                "expected_protocol_sha256": (
                    lagged_subspace_provenance_locks["protocol_sha256"]
                ),
                "expected_analyzer_sha256": (
                    lagged_subspace_provenance_locks["analyzer_sha256"]
                ),
                "expected_launcher_sha256": (
                    lagged_subspace_provenance_locks["launcher_sha256"]
                ),
                "expected_dependency_lock_sha256": (
                    lagged_subspace_provenance_locks[
                        "dependency_lock_sha256"
                    ]
                ),
            }
        )
    if heldout_settings is not None:
        config["provenance"]["rng_scheme"]["heldout_evaluation"] = (
            "stream=4 with fixed (run_seed, episode_index) bank"
        )
    _write_json_atomic(config_path, config)
    if checkpoint_capture_manifest is not None:
        if checkpoint_training_config_bytes is None:
            raise RuntimeError("checkpoint training config was not serialized")
        _write_bytes_atomic(
            os.path.join(
                output_dir, CHECKPOINT_TRAINING_CONFIG_ARTIFACT
            ),
            checkpoint_training_config_bytes,
        )
        _write_checkpoint_capture_manifest(
            output_dir, checkpoint_capture_manifest
        )
    _write_json_atomic(history_path, [])
    with open(history_jsonl_path, "w", encoding="utf-8"):
        pass
    _write_json_atomic(
        status_path,
        _with_checkpoint_capture_status(
            _with_heldout_status(
                _with_training_budget_status(
                    {
                        "status": "running",
                        "started_at": started_at,
                        "expected_iterations": n_iterations,
                        "completed_iterations": 0,
                        "history_records": os.path.basename(
                            history_jsonl_path
                        ),
                    },
                    training_step_budget,
                    0,
                    n_iterations,
                    "running",
                ),
                heldout_settings,
                "pending_training",
                0,
            ),
            checkpoint_capture_manifest,
        ),
    )

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
            f"initial_lr={base_lr} | lr_schedule={lr_schedule}",
            flush=True,
        )

    pool: Pool | None = None
    history: list[dict[str, Any]] = []
    best_reward: float | None = (
        -np.inf if online_evaluation_enabled else None
    )
    best_fitness_so_far = -np.inf
    best_params = params.copy()
    last_eval_reward: float | None = (
        -np.inf if online_evaluation_enabled else None
    )
    initial_eval_reward: float | None = None
    initial_eval_env_steps = 0
    train_env_steps = 0
    eval_env_steps = 0
    normalization_calibration_env_steps = 0
    best_obs_norm_state: dict[str, Any] | None = None
    heldout_checkpoints: list[dict[str, Any]] = []
    heldout_evaluation: dict[str, Any] | None = None
    training_stopping_reason = "running"
    proposal_gradient_archive: deque[tuple[int, np.ndarray]] = deque(
        maxlen=CHECKPOINT_GRADIENT_ARCHIVE_LENGTH
    )

    try:
        pool = Pool(processes=n_workers, initializer=_init_worker, initargs=(config,))
        if obs_norm_mode == "frozen_after_calibration":
            calibration_mean, calibration_var = obs_normalizer.get_mean_var()
            calibration_tasks = [
                (
                    params.copy(),
                    _calibration_rollout_seed(seed, calibration_idx),
                    calibration_mean,
                    calibration_var,
                    True,
                    obs_scale,
                )
                for calibration_idx in range(obs_norm_calibration_episodes)
            ]
            calibration_results = pool.map(_evaluate_params, calibration_tasks)
            calibration_observations = [
                observation
                for _, rollout_observations, _ in calibration_results
                for observation in rollout_observations
            ]
            if not calibration_observations:
                raise RuntimeError("observation-normalization calibration collected no observations")
            obs_normalizer.update_batch(
                np.asarray(calibration_observations, dtype=np.float64)
            )
            normalization_calibration_env_steps = int(
                sum(result[2] for result in calibration_results)
            )
        if heldout_settings is not None:
            heldout_checkpoints.append(
                _make_heldout_checkpoint(params, obs_normalizer, 0, None)
            )
        if online_evaluation_enabled:
            if obs_normalizer is not None:
                initial_obs_mean, initial_obs_var = (
                    obs_normalizer.get_mean_var()
                )
            else:
                initial_obs_mean, initial_obs_var = None, None
            initial_eval_tasks = [
                (
                    params.copy(),
                    _evaluation_rollout_seed(seed, eval_idx),
                    initial_obs_mean,
                    initial_obs_var,
                    False,
                    obs_scale,
                )
                for eval_idx in range(eval_episodes)
            ]
            initial_eval_results = pool.map(
                _evaluate_params, initial_eval_tasks
            )
            initial_eval_reward = float(
                np.mean([result[0] for result in initial_eval_results])
            )
            initial_eval_env_steps = int(
                sum(result[2] for result in initial_eval_results)
            )
            eval_env_steps = initial_eval_env_steps
            last_eval_reward = initial_eval_reward
            best_reward = initial_eval_reward
            best_params = params.copy()
            if obs_normalizer is not None:
                best_obs_norm_state = obs_normalizer.get_state()

        for iteration in range(n_iterations):
            start = time.time()
            optimizer.learning_rate = _learning_rate_at(
                base_lr,
                iteration,
                lr_schedule,
                exponential_decay=lr_decay,
            )
            if obs_normalizer is not None:
                obs_mean, obs_var = obs_normalizer.get_mean_var()
            else:
                obs_mean, obs_var = None, None

            noise, ask_info = optimizer.ask()
            is_reused = np.asarray(ask_info["is_reused"], dtype=bool)
            fresh_indices = np.where(~is_reused)[0]
            if np.any(is_reused) or len(fresh_indices) != optimizer.population_size:
                raise RuntimeError("no-replay protocol received a reused or incomplete batch")

            fresh_tasks = []
            for local_idx, noise_idx in enumerate(fresh_indices):
                theta_eval = optimizer.candidate_params(
                    params, noise[noise_idx]
                )
                rollout_seed = _training_rollout_seed(
                    seed,
                    iteration,
                    local_idx,
                    common_rollout_seed,
                    len(fresh_indices),
                    optimizer.antithetic,
                )
                collect_training_observations = obs_norm_mode == "online"
                fresh_tasks.append(
                    (
                        theta_eval,
                        rollout_seed,
                        obs_mean,
                        obs_var,
                        collect_training_observations,
                        obs_scale,
                    )
                )
            fresh_results = pool.map(_evaluate_params, fresh_tasks) if fresh_tasks else []
            fresh_fitness = np.asarray([result[0] for result in fresh_results], dtype=np.float64)
            train_env_steps_iter = int(sum(result[2] for result in fresh_results))
            train_env_steps += train_env_steps_iter

            if obs_norm_mode == "online" and fresh_results:
                observations = [obs for _, rollout_obs, _ in fresh_results for obs in rollout_obs]
                if observations:
                    obs_normalizer.update_batch(np.asarray(observations, dtype=np.float64))

            center_fitness = None
            center_env_steps_iter = 0
            if evaluate_center_fitness:
                center_seed = (
                    _training_rollout_seed(
                        seed,
                        iteration,
                        0,
                        True,
                        len(fresh_indices),
                        optimizer.antithetic,
                    )
                    if common_rollout_seed
                    else _center_rollout_seed(seed, iteration)
                )
                center = pool.map(
                    _evaluate_params,
                    [(params.copy(), center_seed, obs_mean, obs_var, False, obs_scale)],
                )
                center_fitness = float(center[0][0])
                center_env_steps_iter = int(center[0][2])
                train_env_steps_iter += center_env_steps_iter
                train_env_steps += center_env_steps_iter

            params_before_update = params.copy()
            checkpoint_proposal_gradient: np.ndarray | None = None
            if checkpoint_capture_settings is not None:
                checkpoint_proposal_gradient = np.asarray(
                    optimizer._gradient(noise, fresh_fitness),
                    dtype=np.float64,
                ).copy()
                if (
                    checkpoint_proposal_gradient.shape
                    != (optimizer.num_params,)
                    or not np.all(np.isfinite(checkpoint_proposal_gradient))
                ):
                    raise FloatingPointError(
                        "checkpoint proposal gradient is invalid"
                    )

            params, info = optimizer.tell(
                params,
                noise,
                fresh_fitness,
                ask_info,
                center_fitness=center_fitness,
            )
            if (
                any(
                    key not in info
                    for key in (
                        "n_fresh",
                        "n_reused",
                        "used_replay",
                        "replay_weight_mass",
                        "fresh_weight_mass",
                        "buffer_size",
                        "ess",
                        "ess_ratio",
                        "ess_normalized",
                        "importance_weight_min",
                        "importance_weight_mean",
                        "importance_weight_max",
                    )
                )
                or int(info.get("n_fresh", optimizer.population_size))
                != optimizer.population_size
                or int(info.get("n_reused", 0)) != 0
                or bool(info.get("used_replay", False))
                or float(info.get("replay_weight_mass", 0.0)) != 0.0
                or not _fresh_only_diagnostic_matches(
                    info.get("fresh_weight_mass", 1.0), 1.0
                )
                or int(info.get("buffer_size", 0)) != 0
                or not _fresh_only_diagnostic_matches(
                    info.get("ess", optimizer.population_size),
                    float(optimizer.population_size),
                )
                or not _fresh_only_diagnostic_matches(
                    info.get("ess_ratio", 1.0), 1.0
                )
                or not _fresh_only_diagnostic_matches(
                    info.get("ess_normalized", 1.0), 1.0
                )
                or not _fresh_only_diagnostic_matches(
                    info.get("importance_weight_min", 1.0), 1.0
                )
                or not _fresh_only_diagnostic_matches(
                    info.get("importance_weight_mean", 1.0), 1.0
                )
                or not _fresh_only_diagnostic_matches(
                    info.get("importance_weight_max", 1.0), 1.0
                )
                or (
                    "mean_importance_weight" in info
                    and not _fresh_only_diagnostic_matches(
                        info["mean_importance_weight"], 1.0
                    )
                )
                or (
                    "max_importance_weight" in info
                    and not _fresh_only_diagnostic_matches(
                        info["max_importance_weight"], 1.0
                    )
                )
                or (
                    isinstance(optimizer, DIIWES)
                    and len(optimizer.sample_buffer) != 0
                )
            ):
                raise RuntimeError("no-replay protocol detected replay in optimizer diagnostics")
            optimizer.current_params = params.copy()

            if checkpoint_capture_settings is not None:
                if checkpoint_proposal_gradient is None:
                    raise RuntimeError(
                        "checkpoint proposal gradient was not computed"
                    )
                expected_params = params_before_update + (
                    optimizer.learning_rate * checkpoint_proposal_gradient
                )
                if not np.array_equal(params, expected_params):
                    raise RuntimeError(
                        "checkpoint generator update is not the unclipped "
                        "explicit Standard-ES proposal-gradient step"
                    )
                proposal_gradient_archive.append(
                    (iteration, checkpoint_proposal_gradient)
                )
                completed_generation = iteration + 1
                if completed_generation in checkpoint_capture_settings[
                    "generations"
                ]:
                    if checkpoint_capture_manifest is None:
                        raise RuntimeError(
                            "checkpoint capture manifest is unavailable"
                        )
                    _capture_checkpoint_artifact(
                        output_dir=output_dir,
                        generation=completed_generation,
                        params=params,
                        obs_normalizer=obs_normalizer,
                        gradient_archive=proposal_gradient_archive,
                        settings=checkpoint_capture_settings,
                        manifest=checkpoint_capture_manifest,
                    )

            if heldout_settings is not None:
                candidate_steps = [
                    int(checkpoint["training_env_steps"])
                    for checkpoint in heldout_checkpoints
                ] + [train_env_steps]
                selected_steps = _select_heldout_checkpoint_steps(
                    candidate_steps,
                    heldout_settings["heldout_training_step_budget"],
                )
                if len(selected_steps) == len(candidate_steps):
                    heldout_checkpoints.append(
                        _make_heldout_checkpoint(
                            params,
                            obs_normalizer,
                            train_env_steps,
                            iteration,
                        )
                    )

            best_fitness_iter = float(np.max(fresh_fitness)) if len(fresh_fitness) else float(info.get("max_fitness", -np.inf))
            best_fitness_so_far = max(best_fitness_so_far, best_fitness_iter)

            training_budget_reached = (
                training_step_budget is not None
                and train_env_steps >= training_step_budget
            )
            if training_budget_reached:
                training_stopping_reason = "training_env_step_budget_reached"
            should_eval = (
                online_evaluation_enabled
                and (
                    iteration % eval_interval == 0
                    or iteration == n_iterations - 1
                    or training_budget_reached
                )
            )
            eval_env_steps_iter = 0
            if should_eval:
                if obs_normalizer is not None:
                    eval_obs_mean, eval_obs_var = obs_normalizer.get_mean_var()
                else:
                    eval_obs_mean, eval_obs_var = None, None
                eval_tasks = []
                for eval_idx in range(eval_episodes):
                    rollout_seed = _evaluation_rollout_seed(seed, eval_idx)
                    eval_tasks.append(
                        (
                            params.copy(),
                            rollout_seed,
                            eval_obs_mean,
                            eval_obs_var,
                            False,
                            obs_scale,
                        )
                    )
                eval_results = pool.map(_evaluate_params, eval_tasks)
                last_eval_reward = float(np.mean([result[0] for result in eval_results]))
                eval_env_steps_iter = int(sum(result[2] for result in eval_results))
                eval_env_steps += eval_env_steps_iter
                if last_eval_reward > best_reward:
                    best_reward = last_eval_reward
                    best_params = params.copy()
                    if obs_normalizer is not None:
                        best_obs_norm_state = obs_normalizer.get_state()

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
                initial_eval_reward,
                initial_eval_env_steps,
                normalization_calibration_env_steps,
            )
            history.append(record)
            _append_json_line(history_jsonl_path, record)
            _write_json_atomic(
                status_path,
                _with_checkpoint_capture_status(
                    _with_heldout_status(
                        _with_training_budget_status(
                            {
                                "status": "running",
                                "started_at": started_at,
                                "expected_iterations": n_iterations,
                                "completed_iterations": len(history),
                                "history_records": os.path.basename(
                                    history_jsonl_path
                                ),
                            },
                            training_step_budget,
                            train_env_steps,
                            n_iterations,
                            training_stopping_reason,
                        ),
                        heldout_settings,
                        "pending_training",
                        len(heldout_checkpoints),
                    ),
                    checkpoint_capture_manifest,
                ),
            )

            if (
                iteration % log_interval == 0
                or iteration == n_iterations - 1
                or training_budget_reached
            ):
                print(_format_progress(record, verbose), flush=True)
            if training_budget_reached:
                break
        if (
            training_step_budget is not None
            and train_env_steps < training_step_budget
        ):
            training_stopping_reason = (
                "max_iterations_exhausted_before_training_env_step_budget"
            )
            raise RuntimeError(
                "max-iteration safety cap exhausted before "
                f"training_env_step_budget: reached {train_env_steps} of "
                f"{training_step_budget} training environment steps after "
                f"{n_iterations} complete generations"
            )
        if checkpoint_capture_settings is not None:
            if checkpoint_capture_manifest is None:
                raise RuntimeError("checkpoint capture manifest is unavailable")
            requested = list(checkpoint_capture_settings["generations"])
            captured = list(
                checkpoint_capture_manifest["captured_generations"]
            )
            if captured != requested:
                raise RuntimeError(
                    "checkpoint capture did not produce every fixed generation: "
                    f"expected {requested}, got {captured}"
                )
        if heldout_settings is not None:
            heldout_evaluation = _evaluate_heldout_checkpoints(
                pool,
                heldout_checkpoints,
                seed=seed,
                settings=heldout_settings,
                obs_scale=obs_scale,
            )
    except BaseException as error:
        if training_step_budget is not None:
            if training_stopping_reason == "running":
                training_stopping_reason = (
                    "failed_before_training_env_step_budget"
                )
            budget_metadata = _training_budget_metadata(
                training_step_budget,
                train_env_steps,
                n_iterations,
                training_stopping_reason,
            )
            config["training_budget"] = budget_metadata
            _write_json_atomic(config_path, config)
            _write_json_atomic(
                summary_path,
                {
                    "status": "failed",
                    "completed_iterations": len(history),
                    "training_budget": budget_metadata,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        if checkpoint_capture_manifest is not None:
            checkpoint_capture_manifest["status"] = "failed"
            checkpoint_capture_manifest["failure_phase"] = "training"
            checkpoint_capture_manifest["completed_iterations"] = len(history)
            checkpoint_capture_manifest["error_type"] = type(error).__name__
            checkpoint_capture_manifest["error"] = str(error)
            _write_checkpoint_capture_manifest(
                output_dir, checkpoint_capture_manifest
            )
        _write_json_atomic(history_path, history)
        _write_json_atomic(
            status_path,
            _with_checkpoint_capture_status(
                _with_heldout_status(
                    _with_training_budget_status(
                        {
                            "status": "failed",
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "expected_iterations": n_iterations,
                            "completed_iterations": len(history),
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "history_records": os.path.basename(
                                history_jsonl_path
                            ),
                        },
                        training_step_budget,
                        train_env_steps,
                        n_iterations,
                        training_stopping_reason,
                    ),
                    heldout_settings,
                    "not_completed",
                    len(heldout_checkpoints),
                ),
                checkpoint_capture_manifest,
            ),
        )
        raise
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        env.close()

    try:
        if training_step_budget is not None:
            budget_metadata = _training_budget_metadata(
                training_step_budget,
                train_env_steps,
                n_iterations,
                training_stopping_reason,
            )
            config["training_budget"] = budget_metadata
            _write_json_atomic(config_path, config)
        if online_evaluation_enabled:
            np.save(os.path.join(output_dir, "best_params.npy"), best_params)
        np.save(os.path.join(output_dir, "final_params.npy"), params)
        if isinstance(optimizer, SNES):
            final_search_std = np.asarray(optimizer.search_std, dtype=np.float64)
            if (
                final_search_std.shape != (optimizer.num_params,)
                or not np.all(np.isfinite(final_search_std))
                or np.any(final_search_std <= 0.0)
            ):
                raise FloatingPointError(
                    "SNES final search standard deviation is invalid"
                )
            np.save(
                os.path.join(output_dir, SNES_FINAL_SEARCH_STD_ARTIFACT),
                final_search_std,
            )
        if getattr(optimizer, "persist_hessian_ema_artifact", False):
            if not hasattr(optimizer, "hessian_ema"):
                raise RuntimeError(
                    "optimizer advertises a Hessian EMA artifact without state"
                )
            hessian_ema = np.asarray(optimizer.hessian_ema, dtype=np.float64)
            if hessian_ema.ndim != 1 or not np.all(np.isfinite(hessian_ema)):
                raise FloatingPointError(
                    "optimizer Hessian EMA state is invalid"
                )
            np.save(
                os.path.join(output_dir, HESSIAN_EMA_ARTIFACT),
                hessian_ema,
            )
        if obs_normalizer is not None:
            np.savez(os.path.join(output_dir, "obs_norm.npz"), **obs_normalizer.get_state())
            if online_evaluation_enabled:
                if best_obs_norm_state is None:
                    raise RuntimeError(
                        "best observation-normalizer state was not recorded"
                    )
                np.savez(
                    os.path.join(output_dir, "best_obs_norm.npz"),
                    **best_obs_norm_state,
                )
        if heldout_settings is not None:
            if heldout_evaluation is None:
                raise RuntimeError("held-out evaluation was not completed")
            _write_json_atomic(heldout_evaluation_path, heldout_evaluation)
        finished_at = _utc_now()
        _write_json_atomic(history_path, history)
        if checkpoint_capture_manifest is not None:
            checkpoint_capture_manifest["status"] = "complete"
            checkpoint_capture_manifest["completed_iterations"] = len(history)
            checkpoint_capture_manifest["checkpoint_count"] = len(
                checkpoint_capture_manifest["artifacts"]
            )
            _write_checkpoint_capture_manifest(
                output_dir, checkpoint_capture_manifest
            )
        if training_step_budget is not None:
            _write_json_atomic(
                summary_path,
                {
                    "status": "complete",
                    "completed_iterations": len(history),
                    "best_reward": (
                        None if best_reward is None else float(best_reward)
                    ),
                    "initial_eval_reward": (
                        None
                        if initial_eval_reward is None
                        else float(initial_eval_reward)
                    ),
                    "training_budget": budget_metadata,
                },
            )
        _write_json_atomic(
            status_path,
            _with_checkpoint_capture_status(
                _with_heldout_status(
                    _with_training_budget_status(
                        {
                            "status": "complete",
                            "started_at": started_at,
                            "finished_at": finished_at,
                            "expected_iterations": n_iterations,
                            "completed_iterations": len(history),
                            "best_reward": (
                                None
                                if best_reward is None
                                else float(best_reward)
                            ),
                            "initial_eval_reward": (
                                None
                                if initial_eval_reward is None
                                else float(initial_eval_reward)
                            ),
                            "normalization_calibration_env_steps": (
                                normalization_calibration_env_steps
                            ),
                            "history_records": os.path.basename(
                                history_jsonl_path
                            ),
                        },
                        training_step_budget,
                        train_env_steps,
                        n_iterations,
                        training_stopping_reason,
                    ),
                    heldout_settings,
                    "complete",
                    len(heldout_checkpoints),
                    heldout_evaluation,
                ),
                checkpoint_capture_manifest,
            ),
        )
    except BaseException as error:
        if training_step_budget is not None:
            budget_metadata = _training_budget_metadata(
                training_step_budget,
                train_env_steps,
                n_iterations,
                training_stopping_reason,
            )
            config["training_budget"] = budget_metadata
            _write_json_atomic(config_path, config)
            _write_json_atomic(
                summary_path,
                {
                    "status": "failed",
                    "completed_iterations": len(history),
                    "training_budget": budget_metadata,
                    "error_type": type(error).__name__,
                    "error": f"artifact finalization failed: {error}",
                },
            )
        if checkpoint_capture_manifest is not None:
            checkpoint_capture_manifest["status"] = "failed"
            checkpoint_capture_manifest["failure_phase"] = (
                "artifact_finalization"
            )
            checkpoint_capture_manifest["completed_iterations"] = len(history)
            checkpoint_capture_manifest["error_type"] = type(error).__name__
            checkpoint_capture_manifest["error"] = (
                f"artifact finalization failed: {error}"
            )
            _write_checkpoint_capture_manifest(
                output_dir, checkpoint_capture_manifest
            )
        _write_json_atomic(
            status_path,
            _with_checkpoint_capture_status(
                _with_heldout_status(
                    _with_training_budget_status(
                        {
                            "status": "failed",
                            "started_at": started_at,
                            "finished_at": _utc_now(),
                            "expected_iterations": n_iterations,
                            "completed_iterations": len(history),
                            "error_type": type(error).__name__,
                            "error": (
                                f"artifact finalization failed: {error}"
                            ),
                            "history_records": os.path.basename(
                                history_jsonl_path
                            ),
                        },
                        training_step_budget,
                        train_env_steps,
                        n_iterations,
                        training_stopping_reason,
                    ),
                    heldout_settings,
                    "not_completed",
                    len(heldout_checkpoints),
                    heldout_evaluation,
                ),
                checkpoint_capture_manifest,
            ),
        )
        raise

    if online_evaluation_enabled:
        print(f"Training complete. Best reward: {best_reward:.2f}", flush=True)
    else:
        print("Training complete. Online evaluation disabled.", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)
    return (
        best_reward,
        best_params if online_evaluation_enabled else params.copy(),
    )


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


_OPTIMIZER_OVERRIDE_CONDITIONS = {
    "momentum_beta": "momentum_es",
    "adam_beta1": "adam_es",
    "adam_beta2": "adam_es",
    "adam_epsilon": "adam_es",
    "clipup_momentum": "clipup_es",
    "clipup_max_speed": "clipup_es",
    "snes_sigma_learning_rate": "snes",
}


def _apply_optimizer_cli_overrides(
    config: dict[str, Any],
    condition: str,
    overrides: dict[str, float | None],
) -> dict[str, Any]:
    """Apply validated optimizer-specific CLI values without cross-condition drift."""
    result = dict(config)
    for key, required_condition in _OPTIMIZER_OVERRIDE_CONDITIONS.items():
        value = overrides.get(key)
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if condition != required_condition:
            raise ValueError(
                f"{flag} is only valid with --condition {required_condition}"
            )
        numeric_value = float(value)
        if key in {
            "momentum_beta",
            "adam_beta1",
            "adam_beta2",
            "clipup_momentum",
        }:
            if not np.isfinite(numeric_value) or not 0.0 <= numeric_value < 1.0:
                raise ValueError(f"{flag} must be finite and in [0, 1)")
        elif not np.isfinite(numeric_value) or numeric_value <= 0.0:
            raise ValueError(f"{flag} must be finite and positive")
        result[key] = numeric_value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", required=True, choices=sorted(CONDITIONS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-schedule", choices=sorted(LR_SCHEDULES), default=None)
    parser.add_argument("--lr-decay", type=float, default=None)
    parser.add_argument("--momentum-beta", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=None)
    parser.add_argument("--adam-beta2", type=float, default=None)
    parser.add_argument("--adam-epsilon", type=float, default=None)
    parser.add_argument("--clipup-momentum", type=float, default=None)
    parser.add_argument("--clipup-max-speed", type=float, default=None)
    parser.add_argument("--snes-sigma-learning-rate", type=float, default=None)
    parser.add_argument("--reuse-fraction", type=float, default=None)
    parser.add_argument("--min-replay-weight-mass", type=float, default=None)
    parser.add_argument("--scalar-damping", type=float, default=None)
    parser.add_argument(
        "--curvature-mode",
        choices=("diag", "global", "block", "directional"),
        default=None,
        help="Override the curvature estimator for DIIWES variants.",
    )
    parser.add_argument("--curvature-beta", type=float, default=None)
    parser.add_argument(
        "--curvature-fitness",
        choices=("matched", "raw", "standardized"),
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
    parser.add_argument("--training-env-step-budget", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", help="Print detailed optimizer diagnostics while training.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config["_config_path"] = os.path.abspath(args.config)
    config = _condition_config(config, args.condition)
    if args.learning_rate is not None:
        config["learning_rate"] = float(args.learning_rate)
    if args.lr_schedule is not None:
        config["lr_schedule"] = args.lr_schedule
    if args.lr_decay is not None:
        config["lr_decay"] = float(args.lr_decay)
    try:
        config = _apply_optimizer_cli_overrides(
            config,
            args.condition,
            {
                "momentum_beta": args.momentum_beta,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "adam_epsilon": args.adam_epsilon,
                "clipup_momentum": args.clipup_momentum,
                "clipup_max_speed": args.clipup_max_speed,
                "snes_sigma_learning_rate": args.snes_sigma_learning_rate,
            },
        )
    except ValueError as error:
        parser.error(str(error))
    if args.reuse_fraction is not None:
        config["reuse_fraction"] = float(args.reuse_fraction)
    if args.min_replay_weight_mass is not None:
        config["min_replay_weight_mass"] = float(args.min_replay_weight_mass)
    if args.scalar_damping is not None:
        config["scalar_damping"] = float(args.scalar_damping)
    if args.curvature_mode is not None:
        config["curvature_mode"] = args.curvature_mode
        if args.curvature_mode == "block":
            config.setdefault("block_structure", "layer")
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
    if args.training_env_step_budget is not None:
        config["training_env_step_budget"] = int(
            args.training_env_step_budget
        )

    try:
        _validate_named_lopo_condition_semantics(config, args.condition)
    except ValueError as error:
        parser.error(str(error))
    _validate_no_replay_protocol(config)
    try:
        resolved_training_step_budget = _resolve_training_env_step_budget(config)
    except ValueError as error:
        parser.error(str(error))
    if resolved_training_step_budget is not None:
        config["training_env_step_budget"] = resolved_training_step_budget

    if args.workers is None:
        n_workers = max(1, len(os.sched_getaffinity(0)) - 2)
    else:
        n_workers = int(args.workers)
    train(config, args.seed, args.output, n_workers, verbose=args.verbose)


if __name__ == "__main__":
    main()

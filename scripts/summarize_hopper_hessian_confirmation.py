#!/usr/bin/env python3
"""Strict validator and paired analyzer for the Hopper Hessian confirmation."""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Sequence

import numpy as np


CONDITIONS = (
    "standard_es",
    "concave_block_ema_curvature_es",
    "concave_block_ema_isotropic_control_es",
)
SEEDS = tuple(range(100, 110))
EXPECTED_ITERATIONS = 500
INITIAL_LEARNING_RATE = 10.0
LR_SCHEDULE = "inverse_linear"
TRAINING_STEP_BUDGET = 75_000
HELDOUT_EPISODES = 20
POPULATION_SIZE = 200
DIAGNOSTIC_SCHEMA_VERSION = 2
CONFIG_RELATIVE_PATH = "configs/mujuco/hopper_hessian_confirmation_no_replay.yaml"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.realpath(os.path.join(REPO_ROOT, CONFIG_RELATIVE_PATH))
T_CRITICAL_975_DF9 = 2.2621571627409915

STRUCTURED = "concave_block_ema_curvature_es"
ISOTROPIC = "concave_block_ema_isotropic_control_es"
STANDARD = "standard_es"

EXPECTED_COMMON_CONFIG: dict[str, Any] = {
    "env_name": "Hopper-v5",
    "population_size": POPULATION_SIZE,
    "learning_rate": INITIAL_LEARNING_RATE,
    "noise_std": 0.02,
    "l2_coeff": 0.0,
    "rank_fitness": True,
    "antithetic": True,
    "max_grad_norm": 0.0,
    "max_param_norm": None,
    "hidden_dims": [64, 64],
    "activation": "tanh",
    "output_activation": "tanh",
    "init_param_std": 0.1,
    "n_iterations": EXPECTED_ITERATIONS,
    "eval_episodes": 3,
    "eval_interval": 500,
    "log_interval": 10,
    "max_episode_steps": 1000,
    "use_obs_norm": True,
    "obs_norm_mode": "frozen_after_calibration",
    "obs_norm_calibration_episodes": 3,
    "heldout_evaluation_enabled": True,
    "heldout_training_step_budget": TRAINING_STEP_BUDGET,
    "heldout_eval_episodes": HELDOUT_EPISODES,
    "replay_enabled": False,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "common_rollout_seed": True,
    "implicit_damping": 0.0,
    "linear_min_abs_diagonal": 1e-12,
    "evaluate_center_fitness": False,
    "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
    "lr_schedule": LR_SCHEDULE,
}

CONFIG_RUNTIME_KEYS = {
    "_config_path",
    "algorithm",
    "condition",
    "curvature_beta",
    "provenance",
    "resolved_heldout_evaluation",
    "resolved_optimizer",
    "seed",
    "use_curvature",
}
CURVATURE_CONFIG_KEYS = {
    "curvature_attenuation_mode",
    "curvature_confidence_z",
    "curvature_estimator",
    "curvature_fitness",
    "curvature_mode",
}

HELDOUT_TOP_LEVEL_KEYS = {
    "schema_version",
    "training_step_budget",
    "episodes_per_checkpoint",
    "checkpoint_selection",
    "rollout_seed_stream",
    "rollout_seeds",
    "common_seed_bank_across_checkpoints",
    "optimizer_or_checkpoint_selection_uses_heldout_results",
    "observation_normalizer_state",
    "checkpoint_count",
    "heldout_evaluation_env_steps",
    "normalized_auc_at_budget",
    "return_at_budget",
    "checkpoints",
}
HELDOUT_CHECKPOINT_KEYS = {
    "checkpoint_index",
    "source_iteration",
    "training_env_steps",
    "mean_return",
    "episode_returns",
    "episode_env_steps",
}

RUN_FIELDS = (
    "condition",
    "seed",
    "task_id",
    "heldout_auc_at_75000",
    "heldout_return_at_75000",
    "heldout_checkpoint_count",
    "heldout_evaluation_env_steps",
    "run_dir",
)


class ConfirmationValidationError(ValueError):
    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__(
            f"Hopper Hessian confirmation failed with {len(self.issues)} issue(s)"
        )


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: str) -> list[Any]:
    records: list[Any] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL record at line {line_number}: {error}"
                ) from error
    return records


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool) or expected is None:
        return actual is expected
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return bool(
            np.isfinite(value)
            and np.isclose(value, expected, rtol=1e-12, atol=1e-12)
        )
    return actual == expected


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    return bool(
        np.isfinite(left_value)
        and np.isfinite(right_value)
        and np.isclose(
            left_value,
            right_value,
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _task_id(condition: str, seed: int) -> int:
    condition_index = CONDITIONS.index(condition)
    seed_index = SEEDS.index(seed)
    slot = (condition_index - seed_index) % len(CONDITIONS)
    return len(CONDITIONS) * seed_index + slot


def _keyed_seed(seed: int, stream: int, iteration: int, index: int) -> int:
    def pair(left: int, right: int) -> int:
        total = left + right
        return total * (total + 1) // 2 + right

    return pair(pair(seed, stream), pair(iteration, index))


def _heldout_seed_bank(seed: int) -> list[int]:
    return [_keyed_seed(seed, 4, 0, index) for index in range(HELDOUT_EPISODES)]


def _metrics_at_budget(
    training_steps: Sequence[int], returns: Sequence[float]
) -> tuple[float, float]:
    x = np.asarray(training_steps, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    if (
        x.ndim != 1
        or y.ndim != 1
        or len(x) != len(y)
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or x[0] != 0.0
        or np.any(np.diff(x) <= 0.0)
        or x[-1] < TRAINING_STEP_BUDGET
    ):
        raise ValueError("held-out checkpoints do not validly cover the budget")
    at_budget = float(np.interp(float(TRAINING_STEP_BUDGET), x, y))
    below = x < float(TRAINING_STEP_BUDGET)
    x_cut = np.concatenate((x[below], [float(TRAINING_STEP_BUDGET)]))
    y_cut = np.concatenate((y[below], [at_budget]))
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(y_cut, x_cut) / float(TRAINING_STEP_BUDGET))
    return auc, at_budget


def _expected_checkpoint_sequence(
    history: Sequence[dict[str, Any]],
) -> tuple[list[int], list[int | None]]:
    steps = [0]
    iterations: list[int | None] = [None]
    for index, record in enumerate(history):
        step = int(record["training_env_steps"])
        steps.append(step)
        iterations.append(index)
        if step >= TRAINING_STEP_BUDGET:
            return steps, iterations
    raise ValueError("training history never crosses the held-out budget")


def _validate_provenance(
    config: dict[str, Any],
    run_dir: str,
    condition: str,
    seed: int,
    expected_source_sha: str,
    issues: list[str],
) -> tuple[str | None, str | None, int | None]:
    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        issues.append(f"{run_dir}: provenance is missing")
        return None, None, None
    source_sha = provenance.get("source_sha256")
    locked_sha = provenance.get("expected_source_sha256")
    if source_sha != expected_source_sha or locked_sha != expected_source_sha:
        issues.append(f"{run_dir}: source digest lock is invalid")
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or any(
        not isinstance(dependencies.get(name), str)
        for name in ("gymnasium", "mujoco", "PyYAML")
    ):
        issues.append(f"{run_dir}: dependency provenance is incomplete")
    rng = provenance.get("rng_scheme")
    if not isinstance(rng, dict) or rng.get("heldout_evaluation") != (
        "stream=4 with fixed (run_seed, episode_index) bank"
    ):
        issues.append(f"{run_dir}: held-out RNG provenance is invalid")

    array_job_id = provenance.get("slurm_array_job_id")
    task_id_raw = provenance.get("slurm_array_task_id")
    if not isinstance(array_job_id, str) or re.fullmatch(r"[0-9]+", array_job_id) is None:
        issues.append(f"{run_dir}: Slurm array job id is invalid")
        array_job_id = None
    try:
        if isinstance(task_id_raw, bool):
            raise ValueError
        task_id = int(task_id_raw)
    except (TypeError, ValueError):
        issues.append(f"{run_dir}: Slurm task id is invalid")
        task_id = None
    expected_task = _task_id(condition, seed)
    if task_id is not None and task_id != expected_task:
        issues.append(
            f"{run_dir}: task id {task_id} does not match rotated mapping {expected_task}"
        )
    if array_job_id is not None and task_id is not None:
        expected_name = (
            f"{condition}_inverse_linear_a10_seed{seed}_"
            f"job{array_job_id}_task{task_id}"
        )
        if os.path.basename(run_dir) != expected_name:
            issues.append(f"{run_dir}: run directory does not match its exact cell mapping")

    argv = provenance.get("argv")
    if not isinstance(argv, list) or len(argv) != 17:
        issues.append(f"{run_dir}: trainer argv is invalid")
    else:
        expected_fixed = [
            "experiments/train.py",
            "--config",
            CONFIG_RELATIVE_PATH,
            "--condition",
            condition,
            "--learning-rate",
            "10",
            "--lr-schedule",
            LR_SCHEDULE,
            "--reuse-fraction",
            "0",
            "--seed",
            str(seed),
        ]
        if argv[:13] != expected_fixed or argv[13] != "--workers" or argv[15] != "--output":
            issues.append(f"{run_dir}: trainer argv deviates from the locked protocol")
        try:
            workers = int(argv[14])
        except (TypeError, ValueError):
            workers = 0
        if workers <= 0:
            issues.append(f"{run_dir}: trainer worker count is invalid")
        if os.path.abspath(str(argv[16])) != os.path.abspath(run_dir):
            issues.append(f"{run_dir}: trainer output argument does not match run directory")
    return source_sha if isinstance(source_sha, str) else None, array_job_id, task_id


def _is_locked_config_path(value: Any) -> bool:
    """Accept the locked config from either this checkout or a relocated one."""

    if not isinstance(value, str) or not value:
        return False
    normalized = os.path.normpath(value)
    if os.path.realpath(normalized) == CONFIG_PATH:
        return True
    relative = os.path.normpath(CONFIG_RELATIVE_PATH)
    return os.path.isabs(normalized) and normalized.endswith(os.sep + relative)


def _validate_config(
    config: dict[str, Any],
    run_dir: str,
    condition: str,
    seed: int,
    issues: list[str],
) -> None:
    for key, expected in EXPECTED_COMMON_CONFIG.items():
        if not _matches(config.get(key), expected):
            issues.append(f"{run_dir}: config.{key} is not locked to {expected!r}")
    if not _is_locked_config_path(config.get("_config_path")):
        issues.append(f"{run_dir}: config path is not the locked confirmation config")
    if config.get("condition") != condition or config.get("seed") != seed:
        issues.append(f"{run_dir}: condition/seed metadata is inconsistent")

    is_curvature = condition != STANDARD
    expected_keys = set(EXPECTED_COMMON_CONFIG) | CONFIG_RUNTIME_KEYS
    if is_curvature:
        expected_keys |= CURVATURE_CONFIG_KEYS
    missing = sorted(expected_keys - set(config))
    unexpected = sorted(set(config) - expected_keys)
    if missing:
        issues.append(f"{run_dir}: config is missing keys {missing}")
    if unexpected:
        issues.append(f"{run_dir}: config has unexpected keys {unexpected}")

    if is_curvature:
        attenuation = "structured" if condition == STRUCTURED else "isotropic_norm_matched"
        expected_values = {
            "algorithm": "concave_curvature_es",
            "use_curvature": True,
            "curvature_beta": 0.9,
            "curvature_fitness": "matched",
            "curvature_mode": "block",
            "curvature_estimator": "stein_moment",
            "curvature_confidence_z": None,
            "curvature_attenuation_mode": attenuation,
        }
    else:
        expected_values = {
            "algorithm": "standard_es",
            "use_curvature": False,
            "curvature_beta": 0.0,
        }
    for key, expected in expected_values.items():
        if not _matches(config.get(key), expected):
            issues.append(f"{run_dir}: config.{key} is invalid for {condition}")

    resolved_heldout = config.get("resolved_heldout_evaluation")
    expected_heldout = {
        "enabled": True,
        "artifact": "heldout_evaluation.json",
        "training_step_budget": TRAINING_STEP_BUDGET,
        "episodes_per_checkpoint": HELDOUT_EPISODES,
        "checkpoint_selection": "initial_and_every_center_through_first_budget_crossing",
        "execution_phase": "post_training",
        "rollout_seed_stream": 4,
        "common_seed_bank_across_checkpoints": True,
        "optimizer_or_checkpoint_selection_uses_heldout_results": False,
        "observation_normalizer_state": "frozen_per_checkpoint",
    }
    if resolved_heldout != expected_heldout:
        issues.append(f"{run_dir}: resolved held-out evaluation is invalid")

    resolved = config.get("resolved_optimizer")
    if not isinstance(resolved, dict):
        issues.append(f"{run_dir}: resolved optimizer is missing")
        return
    common_resolved = {
        "type": "StandardES" if not is_curvature else "ConcaveCurvatureES",
        "population_size": POPULATION_SIZE,
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "noise_std": 0.02,
        "rank_fitness": True,
        "l2_coeff": 0.0,
        "antithetic": True,
        "max_grad_norm": 0.0,
        "max_param_norm": None,
        "trust_region": False,
        "replay_enabled": False,
    }
    for key, expected in common_resolved.items():
        if not _matches(resolved.get(key), expected):
            issues.append(f"{run_dir}: resolved_optimizer.{key} is invalid")
    if not is_curvature:
        expected_resolved_keys = set(common_resolved)
    else:
        attenuation = "structured" if condition == STRUCTURED else "isotropic_norm_matched"
        solver = (
            "concave_projected_block"
            if condition == STRUCTURED
            else "concave_projected_block_isotropic_attenuation_control"
        )
        curvature_resolved = {
            "method": "concave_curvature",
            "implicit_damping": 0.0,
            "curvature_fitness": "matched",
            "curvature_mode": "block",
            "curvature_structure": "block",
            "curvature_beta": 0.9,
            "curvature_same_generation": False,
            "curvature_estimator": "stein_moment",
            "curvature_confidence_z": None,
            "curvature_attenuation_mode": attenuation,
            "curvature_clipping": False,
            "curvature_projection": "concave",
            "curvature_components": 3,
            "solver_type": solver,
        }
        for key, expected in curvature_resolved.items():
            if not _matches(resolved.get(key), expected):
                issues.append(f"{run_dir}: resolved_optimizer.{key} is invalid")
        expected_resolved_keys = set(common_resolved) | set(curvature_resolved)
    if set(resolved) != expected_resolved_keys:
        issues.append(f"{run_dir}: resolved optimizer schema is not exact")


def _validate_history(
    history: Any,
    run_dir: str,
    condition: str,
    issues: list[str],
) -> None:
    if not isinstance(history, list):
        issues.append(f"{run_dir}: history is not a list")
        return
    if len(history) != EXPECTED_ITERATIONS:
        issues.append(f"{run_dir}: history does not contain 500 updates")
    previous_steps = 0
    forbidden_fields = {
        "trust_active",
        "trust_scale",
        "trust_radius",
        "pre_trust_step_norm",
        "multiplier_floor_frac",
        "implicit_converged",
        "implicit_iterations",
        "implicit_relative_residual",
    }
    exact_fresh_values = {
        "n_fresh": POPULATION_SIZE,
        "n_reused": 0,
        "used_replay": False,
        "replay_weight_mass": 0.0,
        "fresh_weight_mass": 1.0,
        "buffer_size": 0,
        "ess": float(POPULATION_SIZE),
        "ess_ratio": 1.0,
        "ess_normalized": 1.0,
        "importance_weight_mean": 1.0,
        "importance_weight_min": 1.0,
        "importance_weight_max": 1.0,
        "parameter_projection_active": False,
        "curvature_clip_frac": 0.0,
    }
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            issues.append(f"{run_dir}: history[{index}] is not an object")
            continue
        if record.get("iteration") != index:
            issues.append(f"{run_dir}: history[{index}] has the wrong iteration")
        expected_lr = INITIAL_LEARNING_RATE / float(index + 1)
        if not _close(record.get("lr"), expected_lr):
            issues.append(f"{run_dir}: history[{index}] violates the 10/(t+1) schedule")
        for field, expected in exact_fresh_values.items():
            if field not in record or not _matches(record.get(field), expected):
                issues.append(
                    f"{run_dir}: history[{index}].{field} is not fresh-only {expected!r}"
                )
        for alias in ("mean_importance_weight", "max_importance_weight"):
            if alias in record and not _matches(record[alias], 1.0):
                issues.append(f"{run_dir}: history[{index}].{alias} is not one")
        if forbidden_fields.intersection(record):
            issues.append(f"{run_dir}: history[{index}] contains Picard/trust metadata")
        if any(str(field).startswith("endpoint_") for field in record):
            issues.append(f"{run_dir}: history[{index}] contains endpoint weighting")
        for field, value in record.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not np.isfinite(float(value)):
                issues.append(f"{run_dir}: history[{index}].{field} is non-finite")

        try:
            cumulative_steps = int(record["training_env_steps"])
            iteration_steps = int(record["training_env_steps_iter"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] has invalid training steps")
        else:
            if (
                cumulative_steps <= previous_steps
                or iteration_steps <= 0
                or cumulative_steps - previous_steps != iteration_steps
            ):
                issues.append(f"{run_dir}: history[{index}] training-step sequence is invalid")
            previous_steps = cumulative_steps
        if not _close(record.get("grad_norm"), record.get("grad_norm_before_clip")):
            issues.append(f"{run_dir}: history[{index}] reports gradient norm clipping")
        if not _close(record.get("step_norm"), record.get("proposed_step_norm")):
            issues.append(f"{run_dir}: history[{index}] reports parameter projection")

        if condition == STANDARD:
            if record.get("solver_type") != "none" or "solve_success" in record:
                issues.append(f"{run_dir}: Standard ES has nonstandard solver metadata")
            ratio = record.get("step_norm_ratio")
            if not _close(ratio, 1.0, tolerance=1e-8):
                issues.append(f"{run_dir}: Standard ES step ratio is not one")
            if any(
                field in record
                for field in (
                    "curvature_attenuation_mode",
                    "curvature_estimator",
                    "curvature_same_generation",
                    "hessian_ema_count",
                )
            ):
                issues.append(f"{run_dir}: Standard ES contains curvature metadata")
            continue

        attenuation = "structured" if condition == STRUCTURED else "isotropic_norm_matched"
        solver = (
            "concave_projected_block"
            if condition == STRUCTURED
            else "concave_projected_block_isotropic_attenuation_control"
        )
        exact_curvature = {
            "solver_type": solver,
            "solve_success": True,
            "implicit_damping": 0.0,
            "curvature_beta": 0.9,
            "curvature_same_generation": False,
            "curvature_step_state": "bias_corrected_ema",
            "curvature_estimator": "stein_moment",
            "curvature_attenuation_mode": attenuation,
            "curvature_confidence_gate_enabled": False,
            "curvature_confidence_gate_frac": 0.0,
            "curvature_fitness": "matched",
            "curvature_matches_gradient": True,
            "curvature_mode": "block",
            "curvature_structure": "block",
            "curvature_components": 3,
            "curvature_block_size_min": 195,
            "curvature_block_size_max": 4160,
            "hessian_pairs": 100,
            "hessian_ema_count": index + 1,
            "curvature_same_generation_se_available": True,
            "linear_nonpositive_diagonal_frac": 0.0,
        }
        for field, expected in exact_curvature.items():
            if field not in record or not _matches(record.get(field), expected):
                issues.append(f"{run_dir}: history[{index}].{field} is invalid")
        numeric_bounds = {
            "linear_relative_residual": (0.0, 1e-10),
            "structured_reference_relative_residual": (0.0, 1e-10),
            "attenuation_norm_match_relative_error": (0.0, 1e-10),
            "denominator_min": (1.0, float("inf")),
            "linear_condition_estimate": (1.0, float("inf")),
            "step_norm_ratio": (0.0, 1.0 + 1e-10),
            "curvature_projection_frac": (0.0, 1.0),
            "curvature_active_frac": (0.0, 1.0),
        }
        for field, (lower, upper) in numeric_bounds.items():
            try:
                value = float(record[field])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}].{field} is missing/invalid")
            else:
                if not np.isfinite(value) or not lower <= value <= upper:
                    issues.append(f"{run_dir}: history[{index}].{field} is out of bounds")
        if not _close(
            record.get("step_norm"),
            record.get("structured_reference_step_norm"),
            tolerance=1e-10,
        ):
            issues.append(f"{run_dir}: history[{index}] does not norm-match its structured step")
        try:
            attenuation_scale = float(record["isotropic_attenuation_scale"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] attenuation scale is invalid")
        else:
            if condition == STRUCTURED:
                if attenuation_scale != 1.0:
                    issues.append(f"{run_dir}: structured arm applies isotropic attenuation")
            elif not np.isfinite(attenuation_scale) or not 0.0 < attenuation_scale <= 1.0:
                issues.append(f"{run_dir}: isotropic attenuation scale is out of bounds")


def _validate_heldout(
    artifact: Any,
    history: list[dict[str, Any]],
    status: dict[str, Any],
    run_dir: str,
    seed: int,
    issues: list[str],
) -> tuple[float | None, float | None]:
    if not isinstance(artifact, dict):
        issues.append(f"{run_dir}: heldout_evaluation.json is not an object")
        return None, None
    if set(artifact) != HELDOUT_TOP_LEVEL_KEYS:
        issues.append(f"{run_dir}: held-out top-level schema is not exact")
    exact_values = {
        "schema_version": 1,
        "training_step_budget": TRAINING_STEP_BUDGET,
        "episodes_per_checkpoint": HELDOUT_EPISODES,
        "checkpoint_selection": "initial_and_every_center_through_first_budget_crossing",
        "rollout_seed_stream": 4,
        "rollout_seeds": _heldout_seed_bank(seed),
        "common_seed_bank_across_checkpoints": True,
        "optimizer_or_checkpoint_selection_uses_heldout_results": False,
        "observation_normalizer_state": "frozen_per_checkpoint",
    }
    for field, expected in exact_values.items():
        if artifact.get(field) != expected:
            issues.append(f"{run_dir}: held-out {field} is invalid")
    try:
        expected_steps, expected_iterations = _expected_checkpoint_sequence(history)
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{run_dir}: cannot reconstruct held-out checkpoints: {error}")
        return None, None
    checkpoints = artifact.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != len(expected_steps):
        issues.append(f"{run_dir}: held-out checkpoint count is invalid")
        return None, None
    if artifact.get("checkpoint_count") != len(checkpoints):
        issues.append(f"{run_dir}: held-out checkpoint_count disagrees with records")

    means: list[float] = []
    total_env_steps = 0
    for index, (checkpoint, expected_step, expected_iteration) in enumerate(
        zip(checkpoints, expected_steps, expected_iterations, strict=True)
    ):
        if not isinstance(checkpoint, dict) or set(checkpoint) != HELDOUT_CHECKPOINT_KEYS:
            issues.append(f"{run_dir}: held-out checkpoint[{index}] schema is invalid")
            continue
        if (
            checkpoint.get("checkpoint_index") != index
            or checkpoint.get("source_iteration") != expected_iteration
            or checkpoint.get("training_env_steps") != expected_step
        ):
            issues.append(f"{run_dir}: held-out checkpoint[{index}] is not an actual center")
        returns = checkpoint.get("episode_returns")
        episode_steps = checkpoint.get("episode_env_steps")
        if not isinstance(returns, list) or len(returns) != HELDOUT_EPISODES:
            issues.append(f"{run_dir}: held-out checkpoint[{index}] return bank is invalid")
            continue
        if not isinstance(episode_steps, list) or len(episode_steps) != HELDOUT_EPISODES:
            issues.append(f"{run_dir}: held-out checkpoint[{index}] step bank is invalid")
            continue
        try:
            return_values = np.asarray(returns, dtype=np.float64)
            step_values = [int(value) for value in episode_steps]
        except (TypeError, ValueError):
            issues.append(f"{run_dir}: held-out checkpoint[{index}] episodes are invalid")
            continue
        if (
            not np.all(np.isfinite(return_values))
            or any(
                isinstance(raw, bool)
                or raw != integer
                or not 1 <= integer <= 1000
                for raw, integer in zip(episode_steps, step_values, strict=True)
            )
        ):
            issues.append(f"{run_dir}: held-out checkpoint[{index}] rollout values are invalid")
        recomputed_mean = float(np.mean(return_values))
        if not _close(checkpoint.get("mean_return"), recomputed_mean):
            issues.append(f"{run_dir}: held-out checkpoint[{index}] mean is not recomputed")
        means.append(recomputed_mean)
        total_env_steps += sum(step_values)
    if len(means) != len(expected_steps):
        return None, None
    if artifact.get("heldout_evaluation_env_steps") != total_env_steps:
        issues.append(f"{run_dir}: held-out environment-step total is invalid")
    try:
        auc, return_at_budget = _metrics_at_budget(expected_steps, means)
    except ValueError as error:
        issues.append(f"{run_dir}: held-out metric inputs are invalid: {error}")
        return None, None
    if not _close(artifact.get("normalized_auc_at_budget"), auc):
        issues.append(f"{run_dir}: held-out AUC was not recomputed")
    if not _close(artifact.get("return_at_budget"), return_at_budget):
        issues.append(f"{run_dir}: held-out return at budget was not recomputed")

    heldout_status = status.get("heldout_evaluation")
    if not isinstance(heldout_status, dict):
        issues.append(f"{run_dir}: held-out status metadata is missing")
    else:
        expected_status = {
            "status": "complete",
            "artifact": "heldout_evaluation.json",
            "training_step_budget": TRAINING_STEP_BUDGET,
            "episodes_per_checkpoint": HELDOUT_EPISODES,
            "checkpoint_count": len(checkpoints),
        }
        for field, expected in expected_status.items():
            if heldout_status.get(field) != expected:
                issues.append(f"{run_dir}: held-out status.{field} is invalid")
        if not _close(heldout_status.get("normalized_auc_at_budget"), auc) or not _close(
            heldout_status.get("return_at_budget"), return_at_budget
        ):
            issues.append(f"{run_dir}: held-out status metrics disagree with artifact")
    return auc, return_at_budget


def validate_and_collect(
    root: str,
    *,
    expected_source_sha: str,
    job_output_dir: str = "job_outputs",
) -> list[dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{64}", expected_source_sha) is None:
        raise ValueError("expected_source_sha must be a lowercase SHA-256 digest")
    expected_cells = {(condition, seed) for seed in SEEDS for condition in CONDITIONS}
    candidates: dict[tuple[str, int], list[str]] = defaultdict(list)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    issues: list[str] = []
    source_hashes: set[str] = set()
    array_job_ids: set[str] = set()

    run_dirs = sorted(
        {
            os.path.dirname(path)
            for filename in (
                "config.json",
                "history.json",
                "history.jsonl",
                "status.json",
                "heldout_evaluation.json",
            )
            for path in glob.glob(os.path.join(root, "**", filename), recursive=True)
        }
    )
    for run_dir in run_dirs:
        paths = {
            name: os.path.join(run_dir, name)
            for name in (
                "config.json",
                "history.json",
                "history.jsonl",
                "status.json",
                "heldout_evaluation.json",
            )
        }
        if not all(os.path.isfile(path) for path in paths.values()):
            issues.append(f"{run_dir}: confirmation artifacts are incomplete")
            continue
        try:
            config = _read_json(paths["config.json"])
            history = _read_json(paths["history.json"])
            history_jsonl = _read_jsonl(paths["history.jsonl"])
            status = _read_json(paths["status.json"])
            heldout = _read_json(paths["heldout_evaluation.json"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_dir}: artifact is unreadable: {error}")
            continue
        if not isinstance(config, dict) or not isinstance(status, dict):
            issues.append(f"{run_dir}: config/status is not an object")
            continue
        if history_jsonl != history:
            issues.append(f"{run_dir}: history JSON and JSONL do not match exactly")
        condition = config.get("condition")
        seed = config.get("seed")
        if condition not in CONDITIONS or seed not in SEEDS:
            issues.append(f"{run_dir}: unexpected confirmation cell {(condition, seed)!r}")
            continue
        cell = (str(condition), int(seed))
        candidates[cell].append(run_dir)
        _validate_config(config, run_dir, cell[0], cell[1], issues)
        source_sha, array_job_id, task_id = _validate_provenance(
            config,
            run_dir,
            cell[0],
            cell[1],
            expected_source_sha,
            issues,
        )
        if source_sha is not None:
            source_hashes.add(source_sha)
        if array_job_id is not None:
            array_job_ids.add(array_job_id)
        if array_job_id is not None and task_id is not None:
            stderr_path = os.path.join(
                job_output_dir,
                f"hopper_hconf_{array_job_id}_{task_id}.err",
            )
            if not os.path.isfile(stderr_path):
                issues.append(f"{run_dir}: Slurm stderr artifact is missing")
            elif os.path.getsize(stderr_path) != 0:
                issues.append(f"{run_dir}: Slurm stderr artifact is nonempty")
        _validate_history(history, run_dir, cell[0], issues)
        if (
            status.get("status") != "complete"
            or status.get("expected_iterations") != EXPECTED_ITERATIONS
            or status.get("completed_iterations") != EXPECTED_ITERATIONS
            or status.get("history_records") != "history.jsonl"
        ):
            issues.append(f"{run_dir}: run status is not complete and exact")
        if isinstance(history, list) and history:
            if not _close(status.get("initial_eval_reward"), history[0].get("initial_eval_reward")):
                issues.append(f"{run_dir}: status initial reward disagrees with history")
        auc, return_at_budget = _validate_heldout(
            heldout,
            history if isinstance(history, list) else [],
            status,
            run_dir,
            cell[1],
            issues,
        )
        if auc is not None and return_at_budget is not None and task_id is not None:
            rows[cell] = {
                "condition": cell[0],
                "seed": cell[1],
                "task_id": task_id,
                "heldout_auc_at_75000": auc,
                "heldout_return_at_75000": return_at_budget,
                "heldout_checkpoint_count": heldout.get("checkpoint_count"),
                "heldout_evaluation_env_steps": heldout.get(
                    "heldout_evaluation_env_steps"
                ),
                "run_dir": run_dir,
                "_history": history,
                "_heldout": heldout,
            }

    for cell in sorted(expected_cells):
        found = candidates.get(cell, [])
        if len(found) != 1:
            issues.append(f"cell {cell!r}: expected one run, found {len(found)}")
    if len(run_dirs) != len(expected_cells):
        issues.append(
            f"confirmation root has {len(run_dirs)} run directories, expected 30"
        )
    if source_hashes != {expected_source_sha}:
        issues.append("confirmation runs do not share the expected source digest")
    if len(array_job_ids) != 1:
        issues.append("confirmation runs do not share one Slurm array job id")

    for seed in SEEDS:
        seed_rows = [rows.get((condition, seed)) for condition in CONDITIONS]
        if any(row is None for row in seed_rows):
            continue
        typed_rows = [row for row in seed_rows if row is not None]
        initial_rewards = [row["_history"][0]["initial_eval_reward"] for row in typed_rows]
        if not all(_close(value, initial_rewards[0]) for value in initial_rewards[1:]):
            issues.append(f"seed {seed}: initial evaluation rewards are not matched")
        initial_heldout = [row["_heldout"]["checkpoints"][0] for row in typed_rows]
        initial_return_banks = [checkpoint["episode_returns"] for checkpoint in initial_heldout]
        if any(bank != initial_return_banks[0] for bank in initial_return_banks[1:]):
            issues.append(f"seed {seed}: initial held-out return banks are not matched")
        structured = rows[(STRUCTURED, seed)]["_history"][0]
        isotropic = rows[(ISOTROPIC, seed)]["_history"][0]
        for field in (
            "explicit_step_norm",
            "structured_reference_step_norm",
            "step_norm",
            "h_raw_mean",
            "h_raw_std",
            "denominator_min",
            "denominator_max",
        ):
            if not _close(structured.get(field), isotropic.get(field), tolerance=1e-10):
                issues.append(
                    f"seed {seed}: structured/control first-update {field} is unmatched"
                )

    if issues:
        raise ConfirmationValidationError(issues)
    return [rows[(condition, seed)] for seed in SEEDS for condition in CONDITIONS]


def exact_two_sided_sign_flip_p(differences: Sequence[float]) -> tuple[float, int, int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.shape != (len(SEEDS),) or not np.all(np.isfinite(values)):
        raise ValueError("sign-flip test requires exactly ten finite paired differences")
    observed = abs(float(np.mean(values)))
    threshold = float(np.nextafter(observed, -np.inf))
    extreme = 0
    assignments = 2 ** len(values)
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= threshold:
            extreme += 1
    return float(extreme / assignments), extreme, assignments


def _holm_adjust(raw_p_values: Sequence[float]) -> list[float]:
    if len(raw_p_values) != 2:
        raise ValueError("the preregistered Holm family contains exactly two contrasts")
    order = sorted(range(2), key=lambda index: (raw_p_values[index], index))
    adjusted = [0.0, 0.0]
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (2 - rank) * float(raw_p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def analyze_primary_contrasts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_cell = {(row["condition"], int(row["seed"])): row for row in rows}
    specifications = (
        ("block_ema_minus_standard", STRUCTURED, STANDARD),
        ("block_ema_minus_isotropic", STRUCTURED, ISOTROPIC),
    )
    contrasts: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for name, treatment, comparator in specifications:
        differences = [
            float(by_cell[(treatment, seed)]["heldout_auc_at_75000"])
            - float(by_cell[(comparator, seed)]["heldout_auc_at_75000"])
            for seed in SEEDS
        ]
        values = np.asarray(differences, dtype=np.float64)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        median = float(np.median(values))
        half_width = T_CRITICAL_975_DF9 * sd / np.sqrt(len(values))
        raw_p, extreme, assignments = exact_two_sided_sign_flip_p(values)
        raw_p_values.append(raw_p)
        contrasts.append(
            {
                "name": name,
                "treatment": treatment,
                "comparator": comparator,
                "outcome": "heldout_normalized_auc_at_75000_training_steps",
                "n_pairs": len(values),
                "seeds": list(SEEDS),
                "paired_differences": differences,
                "mean_difference": mean,
                "sample_sd": sd,
                "median_difference": median,
                "t_ci_95_lower": float(mean - half_width),
                "t_ci_95_upper": float(mean + half_width),
                "wins": int(np.sum(values > 0.0)),
                "ties": int(np.sum(values == 0.0)),
                "losses": int(np.sum(values < 0.0)),
                "sign_flip_p_raw": raw_p,
                "sign_flip_extreme_assignments": extreme,
                "sign_flip_total_assignments": assignments,
            }
        )
    adjusted = _holm_adjust(raw_p_values)
    for record, value in zip(contrasts, adjusted, strict=True):
        record["holm_adjusted_p"] = value
        record["holm_reject_0_05"] = bool(value < 0.05)
    claim = all(
        record["mean_difference"] > 0.0
        and record["holm_adjusted_p"] < 0.05
        for record in contrasts
    )
    return {
        "schema_version": 1,
        "primary_outcome": "heldout_normalized_auc_at_75000_training_steps",
        "multiple_testing_family": [record["name"] for record in contrasts],
        "multiple_testing_method": "Holm across exactly two preregistered contrasts",
        "alpha": 0.05,
        "contrasts": contrasts,
        "confirmation_claim_supported": bool(claim),
        "claim_rule": "both paired means > 0 and both Holm-adjusted p-values < 0.05",
    }


def _stage_csv(path: str, rows: Sequence[dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return temporary


def _stage_json(path: str, value: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
    return temporary


def write_outputs(
    rows: Sequence[dict[str, Any]],
    analysis: dict[str, Any],
    *,
    run_output: str,
    contrast_output: str,
) -> None:
    staged: list[tuple[str, str]] = []
    try:
        staged.append((_stage_csv(run_output, rows), run_output))
        staged.append((_stage_json(contrast_output, analysis), contrast_output))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


def summarize(
    root: str,
    *,
    expected_source_sha: str,
    run_output: str,
    contrast_output: str,
    job_output_dir: str = "job_outputs",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = validate_and_collect(
        root,
        expected_source_sha=expected_source_sha,
        job_output_dir=job_output_dir,
    )
    analysis = analyze_primary_contrasts(rows)
    write_outputs(
        rows,
        analysis,
        run_output=run_output,
        contrast_output=contrast_output,
    )
    return rows, analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--contrast-output", default=None)
    parser.add_argument("--job-output-dir", default="job_outputs")
    args = parser.parse_args()
    run_output = args.run_output or os.path.join(
        args.root, "validated_confirmation_runs.csv"
    )
    contrast_output = args.contrast_output or os.path.join(
        args.root, "confirmation_primary_contrasts.json"
    )
    try:
        rows, analysis = summarize(
            args.root,
            expected_source_sha=args.expected_source_sha,
            run_output=run_output,
            contrast_output=contrast_output,
            job_output_dir=args.job_output_dir,
        )
    except ConfirmationValidationError as error:
        for issue in error.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        raise SystemExit(2) from error
    print(
        f"Validated {len(rows)} confirmation runs; "
        f"claim_supported={analysis['confirmation_claim_supported']}; "
        f"wrote {run_output} and {contrast_output}"
    )


if __name__ == "__main__":
    main()

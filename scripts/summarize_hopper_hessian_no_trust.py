#!/usr/bin/env python3
"""Validate and summarize the trust-free Hopper Hessian experiment matrix.

The production matrix contains 80 runs:

* ``standard_es`` and ``diag_curvature``;
* 500 candidates per update;
* no replay or replay buffer;
* inverse-square-root and inverse-linear learning-rate schedules;
* initial learning rates 10 and 30; and
* seeds 0 through 9.

This module intentionally depends only on NumPy and the Python standard
library.  Validation is strict: a summary is never written from a partial,
mixed-source, trust-rescaled, or protocol-deviating matrix.
"""

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
from typing import Any, Iterable, Sequence

import numpy as np


CONDITIONS = ("standard_es", "diag_curvature")
LR_SCHEDULES = ("inverse_sqrt", "inverse_linear")
INITIAL_LEARNING_RATES = (10.0, 30.0)
ALPHA0S = INITIAL_LEARNING_RATES
SEEDS = tuple(range(10))
EXPECTED_ITERATIONS = 500
EXPECTED_ENV = "Hopper-v5"
EXPECTED_POPULATION_SIZE = 500
EXPECTED_PARAMETER_COUNT = 5123
HESSIAN_FOR_STEP_HISTORY_FILENAME = "hessian_for_step_history.npy"
STEP_MULTIPLIER_HISTORY_FILENAME = "step_multiplier_history.npy"

# Values inherited from configs/mujuco/hopper.yaml on main, with the requested
# population increase and fresh-only replay overrides. Keeping these locked
# makes the Hessian arm an interpretable change to Standard ES instead of
# another optimizer redesign.
EXPECTED_COMMON_CONFIG: dict[str, Any] = {
    "env_name": EXPECTED_ENV,
    "population_size": EXPECTED_POPULATION_SIZE,
    "noise_std": 0.02,
    "l2_coeff": 0.0,
    "rank_fitness": True,
    "antithetic": True,
    "hidden_dims": [64, 64],
    "activation": "tanh",
    "output_activation": "tanh",
    "init_param_std": 0.1,
    "eval_episodes": 3,
    "eval_interval": 1,
    "log_interval": 10,
    "max_episode_steps": 1000,
    "use_obs_norm": True,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "buffer_sampling": "random",
    "min_importance_weight": 0.001,
    "max_importance_weight": 10.0,
    "max_sample_age": 3,
    "ess_min_ratio": 0.2,
    "common_rollout_seed": True,
    "implicit_damping": 0.0,
    "use_curvature": True,
    "curvature_mode": "diag",
    "curvature_beta": 0.99,
    "curvature_clip": 1000.0,
    "min_step_multiplier": 0.05,
    "evaluate_center_fitness": False,
    "use_leave_one_out_curvature_baseline": True,
    "bias_correct_curvature_ema": True,
}

EXPECTED_DIAG_CONFIG: dict[str, Any] = {
    "algorithm": "semi_implicit_curvature_es",
    "use_curvature": True,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "buffer_sampling": "random",
    "min_importance_weight": 0.001,
    "max_importance_weight": 10.0,
    "max_sample_age": 3,
    "ess_min_ratio": 0.2,
    "implicit_damping": 0.0,
    "curvature_fitness": "raw",
    "curvature_mode": "diag",
    "curvature_step_mode": "dampen",
    "curvature_beta": 0.99,
    "curvature_clip": 1000.0,
    "min_step_multiplier": 0.05,
    "evaluate_center_fitness": False,
    "use_leave_one_out_curvature_baseline": True,
    "bias_correct_curvature_ema": True,
}

RUNTIME_CONFIG_KEYS = {
    "algorithm",
    "condition",
    "initial_learning_rate",
    "learning_rate",
    "lr_schedule",
    "n_iterations",
    "seed",
    "source_sha256",
    "trust_radius",
}
STANDARD_CONFIG_KEYS = (
    set(EXPECTED_COMMON_CONFIG)
    | RUNTIME_CONFIG_KEYS
    | {"use_trust_radius_for_standard_es"}
)
DIAG_CONFIG_KEYS = (
    set(EXPECTED_COMMON_CONFIG)
    | RUNTIME_CONFIG_KEYS
    | {"curvature_fitness", "curvature_step_mode"}
)

PERFORMANCE_FIELDS = (
    "mean_eval_return",
    "iteration_auc",
    "final_eval_return",
    "last_10_eval_return",
    "best_eval_return",
)

MECHANISM_FIELDS = (
    "mean_hessian_pairs",
    "mean_h_split_correlation",
    "h_split_correlation_available_fraction",
    "mean_h_split_sign_agreement",
    "mean_h_split_relative_disagreement",
    "mean_h_temporal_correlation",
    "h_temporal_correlation_available_fraction",
    "mean_h_temporal_sign_agreement",
    "mean_division_relative_residual",
    "max_division_relative_residual",
    "mean_applied_relative_residual",
    "max_applied_relative_residual",
    "mean_linear_condition_estimate",
    "median_linear_condition_estimate",
    "max_linear_condition_estimate",
    "minimum_linear_abs_diagonal",
    "maximum_linear_abs_diagonal",
    "curvature_coordinate_count",
    "mean_curvature_active_count",
    "mean_curvature_active_frac",
    "mean_curvature_preclip_mean",
    "max_curvature_preclip_max",
    "total_curvature_clip_count",
    "mean_curvature_clip_count",
    "max_curvature_clip_count",
    "mean_curvature_clip_frac",
    "max_curvature_clip_frac",
    "curvature_clip_active_iteration_fraction",
    "mean_curvature_clip_excess_mean",
    "mean_curvature_clip_excess_per_clipped_coordinate",
    "max_curvature_clip_excess_max",
    "multiplier_coordinate_count",
    "minimum_raw_step_multiplier",
    "maximum_raw_step_multiplier",
    "total_multiplier_floor_clip_count",
    "mean_multiplier_floor_clip_count",
    "max_multiplier_floor_clip_count",
    "mean_multiplier_floor_clip_frac",
    "max_multiplier_floor_clip_frac",
    "multiplier_floor_clip_active_iteration_fraction",
    "mean_multiplier_floor_clip_deficit_mean",
    "mean_multiplier_floor_clip_deficit_per_clipped_coordinate",
    "max_multiplier_floor_clip_deficit_max",
    "total_multiplier_ceiling_clip_count",
    "multiplier_ceiling_clip_active_iteration_fraction",
    "mean_multiplier_floor_frac",
    "max_multiplier_floor_frac",
    "floor_active_iteration_fraction",
)

RUN_FIELDS = (
    "env",
    "condition",
    "lr_schedule",
    "initial_learning_rate",
    "seed",
    "iterations",
    "final_train_env_steps",
    *PERFORMANCE_FIELDS,
    *MECHANISM_FIELDS,
    "source_sha256",
    "slurm_job_id",
    "slurm_task_id",
    "run_dir",
)

GROUP_FIELDS = (
    "env",
    "condition",
    "lr_schedule",
    "initial_learning_rate",
    "runs",
    "source_sha256",
    *(name for field in PERFORMANCE_FIELDS for name in (f"{field}_mean", f"{field}_std")),
    *(f"{field}_mean" for field in MECHANISM_FIELDS),
)


class HessianSweepValidationError(ValueError):
    """Raised when any part of the locked matrix is unsafe to summarize."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__(
            f"Hopper Hessian no-trust validation failed with {len(self.issues)} issue(s)"
        )


# Short alias used by the other sweep summarizers in this repository.
ValidationError = HessianSweepValidationError


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


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


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _parse_seed(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        seed = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return seed if np.isfinite(numeric) and numeric == seed else None


def _match_alpha(value: Any, expected: Sequence[float]) -> float | None:
    numeric = _finite_float(value)
    if numeric is None:
        return None
    matches = [
        float(candidate)
        for candidate in expected
        if np.isclose(numeric, candidate, rtol=1e-12, atol=1e-12)
    ]
    return matches[0] if len(matches) == 1 else None


def learning_rate_at_iteration(alpha0: float, iteration: int, schedule: str) -> float:
    """Return the preregistered scalar learning rate for an iteration."""

    alpha = float(alpha0)
    step = int(iteration)
    if not np.isfinite(alpha) or alpha <= 0.0 or step < 0:
        raise ValueError("alpha0 must be positive and iteration must be nonnegative")
    if schedule == "inverse_sqrt":
        return float(alpha / np.sqrt(step + 1.0))
    if schedule == "inverse_linear":
        return float(alpha / (step + 1.0))
    raise ValueError(f"unsupported learning-rate schedule {schedule!r}")


def _production_task_id(
    condition: str, schedule: str, alpha0: float, seed: int
) -> int:
    """Return the fixed 0..79 launcher index for a production cell."""

    return (
        CONDITIONS.index(condition)
        * len(LR_SCHEDULES)
        * len(INITIAL_LEARNING_RATES)
        * len(SEEDS)
        + LR_SCHEDULES.index(schedule)
        * len(INITIAL_LEARNING_RATES)
        * len(SEEDS)
        + INITIAL_LEARNING_RATES.index(float(alpha0)) * len(SEEDS)
        + SEEDS.index(int(seed))
    )


def _expected_run_name_prefix(
    condition: str, schedule: str, alpha0: float, seed: int
) -> str:
    return f"{condition}_{schedule}_a{alpha0:g}_seed{seed}_job"


def _run_name_metadata(
    run_dir: str,
    *,
    condition: str,
    schedule: str,
    alpha0: float,
    seed: int,
) -> tuple[str | None, int | None, str | None]:
    """Validate and decode the launcher's job/task suffix."""

    basename = os.path.basename(os.path.normpath(run_dir))
    prefix = re.escape(_expected_run_name_prefix(condition, schedule, alpha0, seed))
    match = re.fullmatch(prefix + r"([0-9]+)_task([0-9]+)", basename)
    if match is None:
        return None, None, f"{run_dir}: run directory does not encode its exact matrix cell"
    job_id, task_text = match.groups()
    task_id = int(task_text)
    expected_task_id = _production_task_id(condition, schedule, alpha0, seed)
    if task_id != expected_task_id:
        return (
            job_id,
            task_id,
            f"{run_dir}: task id {task_id} does not match matrix cell {expected_task_id}",
        )
    return job_id, task_id, None


def _candidate_run_dirs(root: str) -> list[str]:
    directories: set[str] = set()
    for filename in ("config.json", "history.json", "status.json"):
        directories.update(
            os.path.dirname(path)
            for path in glob.glob(
                os.path.join(root, "**", filename), recursive=True
            )
        )
    return sorted(directories)


def _nonfinite_paths(value: Any, path: str = "history") -> list[str]:
    paths: list[str] = []
    if isinstance(value, bool) or value is None:
        return paths
    if isinstance(value, (int, float)):
        if not np.isfinite(float(value)):
            paths.append(path)
        return paths
    if isinstance(value, dict):
        for key, child in value.items():
            paths.extend(_nonfinite_paths(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_nonfinite_paths(child, f"{path}[{index}]"))
    return paths


def _config_issues(
    config: dict[str, Any],
    *,
    run_dir: str,
    condition: str,
    schedule: str,
    alpha0: float,
    seed: int,
    expected_iterations: int,
    expected_source_sha: str | None,
) -> list[str]:
    issues: list[str] = []
    for key, expected in EXPECTED_COMMON_CONFIG.items():
        if not _matches(config.get(key), expected):
            issues.append(
                f"{run_dir}: config.{key}={config.get(key)!r}, expected {expected!r}"
            )
    exact = {
        "condition": condition,
        "seed": seed,
        "learning_rate": alpha0,
        "lr_schedule": schedule,
        "n_iterations": expected_iterations,
        "trust_radius": None,
    }
    for key, expected in exact.items():
        if not _matches(config.get(key), expected):
            issues.append(
                f"{run_dir}: config.{key}={config.get(key)!r}, expected {expected!r}"
            )
    if "trust_radius" not in config:
        issues.append(f"{run_dir}: config.trust_radius is missing; explicit null is required")
    if "initial_learning_rate" in config and not _matches(
        config.get("initial_learning_rate"), alpha0
    ):
        issues.append(f"{run_dir}: initial_learning_rate disagrees with learning_rate")
    if "lr_decay" in config and not _matches(config.get("lr_decay"), 1.0):
        issues.append(f"{run_dir}: lr_decay must remain 1.0 for this schedule sweep")

    source_sha = config.get("source_sha256")
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha) is None:
        issues.append(f"{run_dir}: config.source_sha256 is not a lowercase SHA-256 digest")
    elif expected_source_sha is not None and source_sha != expected_source_sha:
        issues.append(
            f"{run_dir}: source digest {source_sha} does not match {expected_source_sha}"
        )

    if condition == "standard_es":
        expected_keys = STANDARD_CONFIG_KEYS
        if config.get("algorithm") != "standard_es":
            issues.append(f"{run_dir}: standard_es resolved to the wrong algorithm")
        if config.get("use_trust_radius_for_standard_es") is not False:
            issues.append(f"{run_dir}: Standard ES trust-radius mode is not disabled")
    elif condition == "diag_curvature":
        expected_keys = DIAG_CONFIG_KEYS
        for key, expected in EXPECTED_DIAG_CONFIG.items():
            if not _matches(config.get(key), expected):
                issues.append(
                    f"{run_dir}: config.{key}={config.get(key)!r}, "
                    f"expected main default {expected!r}"
                )
    else:  # Guard direct calls independently of grid validation.
        expected_keys = set()
        issues.append(f"{run_dir}: unsupported condition {condition!r}")

    missing_keys = sorted(expected_keys - set(config))
    unexpected_keys = sorted(set(config) - expected_keys)
    if missing_keys:
        issues.append(f"{run_dir}: config is missing main keys {missing_keys}")
    if unexpected_keys:
        issues.append(f"{run_dir}: config has non-main keys {unexpected_keys}")

    nonfinite = _nonfinite_paths(config, "config")
    if nonfinite:
        issues.append(
            f"{run_dir}: non-finite config value(s): {', '.join(nonfinite[:10])}"
        )
    return issues


def _required_numeric(
    record: dict[str, Any], field: str, context: str, issues: list[str]
) -> float | None:
    if field not in record:
        issues.append(f"{context} is missing {field}")
        return None
    value = _finite_float(record[field])
    if value is None:
        issues.append(f"{context}.{field} is not finite numeric data")
    return value


def _optional_numeric(
    record: dict[str, Any], field: str, context: str, issues: list[str]
) -> float | None:
    """Read a numeric diagnostic which is mathematically allowed to be undefined."""

    if field not in record or record[field] is None:
        return None
    value = _finite_float(record[field])
    if value is None:
        issues.append(f"{context}.{field} is present but not finite numeric data")
    return value


def _required_bool(
    record: dict[str, Any], field: str, context: str, issues: list[str]
) -> bool | None:
    if field not in record:
        issues.append(f"{context} is missing {field}")
        return None
    value = record[field]
    if not isinstance(value, bool):
        issues.append(f"{context}.{field} is not boolean data")
        return None
    return value


def _is_nonnegative_integer(value: float, *, positive: bool = False) -> bool:
    return value == int(value) and value >= (1.0 if positive else 0.0)


def _validate_diag_record(
    record: dict[str, Any], index: int, context: str, issues: list[str]
) -> None:
    exact = {
        "curvature_mode": "diag",
        "curvature_step_mode": "dampen",
        "curvature_fitness": "raw",
        "lambda": 0.0,
        "reuse_fraction": 0.0,
        "buffer_size": 0,
        "used_replay": False,
    }
    for field, expected in exact.items():
        if not _matches(record.get(field), expected):
            issues.append(f"{context}.{field} violates the main diagonal protocol")

    fresh_only_weights = {
        "replay_weight_mass": 0.0,
        "fresh_weight_mass": 1.0,
        "importance_weight_mean": 1.0,
        "importance_weight_min": 1.0,
        "importance_weight_max": 1.0,
        "w_min": 1.0 / EXPECTED_POPULATION_SIZE,
        "w_max": 1.0 / EXPECTED_POPULATION_SIZE,
        "clip_frac": 0.0,
    }
    for field, expected in fresh_only_weights.items():
        value = _required_numeric(record, field, context, issues)
        if value is not None and not np.isclose(
            value, expected, rtol=1e-10, atol=1e-12
        ):
            issues.append(
                f"{context}.{field}={value!r}, expected fresh-only value {expected!r}"
            )

    hessian_pairs = _required_numeric(record, "hessian_pairs", context, issues)
    expected_pairs = 250
    if hessian_pairs is not None and hessian_pairs != expected_pairs:
        issues.append(
            f"{context}.hessian_pairs={hessian_pairs!r}, expected {expected_pairs}"
        )

    # Pearson correlation is undefined for a constant Hessian vector.  The
    # producer represents that state as None, which _history_record omits.
    # Sign agreement and relative disagreement remain defined and required.
    split_corr = _optional_numeric(record, "h_split_correlation", context, issues)
    split_sign = _required_numeric(record, "h_split_sign_agreement", context, issues)
    split_disagreement = _required_numeric(
        record, "h_split_relative_disagreement", context, issues
    )
    if split_corr is not None and not (-1.0 - 1e-12 <= split_corr <= 1.0 + 1e-12):
        issues.append(f"{context}.h_split_correlation is outside [-1, 1]")
    if split_sign is not None and not (0.0 <= split_sign <= 1.0):
        issues.append(f"{context}.h_split_sign_agreement is outside [0, 1]")
    if split_disagreement is not None and split_disagreement < 0.0:
        issues.append(f"{context}.h_split_relative_disagreement is negative")

    # Iteration zero has no prior Hessian vector.  Every later iteration must
    # carry the temporal comparison rather than silently dropping instability.
    if index > 0:
        temporal_corr = _optional_numeric(
            record, "h_temporal_correlation", context, issues
        )
        temporal_sign = _required_numeric(
            record, "h_temporal_sign_agreement", context, issues
        )
        if temporal_corr is not None and not (
            -1.0 - 1e-12 <= temporal_corr <= 1.0 + 1e-12
        ):
            issues.append(f"{context}.h_temporal_correlation is outside [-1, 1]")
        if temporal_sign is not None and not (0.0 <= temporal_sign <= 1.0):
            issues.append(f"{context}.h_temporal_sign_agreement is outside [0, 1]")

    curvature_coordinates = _required_numeric(
        record, "curvature_coordinate_count", context, issues
    )
    curvature_active_count = _required_numeric(
        record, "curvature_active_count", context, issues
    )
    curvature_active_frac = _required_numeric(
        record, "curvature_active_frac", context, issues
    )
    curvature_preclip_mean = _required_numeric(
        record, "curvature_preclip_mean", context, issues
    )
    curvature_preclip_max = _required_numeric(
        record, "curvature_preclip_max", context, issues
    )
    curvature_clip_count = _required_numeric(
        record, "curvature_clip_count", context, issues
    )
    curvature_clip_frac = _required_numeric(
        record, "curvature_clip_frac", context, issues
    )
    curvature_clip_active = _required_bool(
        record, "curvature_clip_active", context, issues
    )
    curvature_excess_mean = _required_numeric(
        record, "curvature_clip_excess_mean", context, issues
    )
    curvature_excess_max = _required_numeric(
        record, "curvature_clip_excess_max", context, issues
    )
    curv_mean = _required_numeric(record, "curv_mean", context, issues)
    curv_max = _required_numeric(record, "curv_max", context, issues)
    curv_min = _required_numeric(record, "curv_min", context, issues)

    curvature_count_valid = bool(
        curvature_coordinates is not None
        and _is_nonnegative_integer(curvature_coordinates, positive=True)
    )
    if curvature_coordinates is not None and not curvature_count_valid:
        issues.append(f"{context}.curvature_coordinate_count is not a positive integer")
    elif curvature_coordinates is not None and curvature_coordinates != EXPECTED_PARAMETER_COUNT:
        issues.append(
            f"{context}.curvature_coordinate_count={curvature_coordinates:g}, "
            f"expected {EXPECTED_PARAMETER_COUNT} for the locked Hopper policy"
        )
    if curvature_active_count is not None and (
        not _is_nonnegative_integer(curvature_active_count)
        or (curvature_count_valid and curvature_active_count > curvature_coordinates)
    ):
        issues.append(f"{context}.curvature_active_count is invalid")
    if curvature_clip_count is not None and (
        not _is_nonnegative_integer(curvature_clip_count)
        or (curvature_count_valid and curvature_clip_count > curvature_coordinates)
        or (
            curvature_active_count is not None
            and curvature_clip_count > curvature_active_count
        )
    ):
        issues.append(f"{context}.curvature_clip_count is invalid")
    if curvature_active_frac is not None and not (0.0 <= curvature_active_frac <= 1.0):
        issues.append(f"{context}.curvature_active_frac is outside [0, 1]")
    if curvature_clip_frac is not None and not (0.0 <= curvature_clip_frac <= 1.0):
        issues.append(f"{context}.curvature_clip_frac is outside [0, 1]")
    if curvature_count_valid and curvature_active_count is not None and curvature_active_frac is not None:
        if not np.isclose(
            curvature_active_frac,
            curvature_active_count / curvature_coordinates,
            rtol=1e-12,
            atol=1e-12,
        ):
            issues.append(
                f"{context}.curvature_active_frac disagrees with count/coordinates"
            )
    if curvature_count_valid and curvature_clip_count is not None and curvature_clip_frac is not None:
        if not np.isclose(
            curvature_clip_frac,
            curvature_clip_count / curvature_coordinates,
            rtol=1e-12,
            atol=1e-12,
        ):
            issues.append(
                f"{context}.curvature_clip_frac disagrees with count/coordinates"
            )
    if curvature_clip_count is not None and curvature_clip_active is not None:
        if curvature_clip_active is not (curvature_clip_count > 0.0):
            issues.append(f"{context}.curvature_clip_active disagrees with clip count")
    if curvature_preclip_mean is not None and curvature_preclip_mean < 0.0:
        issues.append(f"{context}.curvature_preclip_mean is negative")
    if curvature_preclip_max is not None and curvature_preclip_max < 0.0:
        issues.append(f"{context}.curvature_preclip_max is negative")
    if (
        curvature_preclip_mean is not None
        and curvature_preclip_max is not None
        and curvature_preclip_max < curvature_preclip_mean
    ):
        issues.append(f"{context}: pre-clip maximum is below the pre-clip mean")
    if curvature_active_count is not None and curvature_preclip_max is not None:
        if (curvature_active_count > 0.0) is not (curvature_preclip_max > 0.0):
            issues.append(
                f"{context}: curvature active count disagrees with pre-clip maximum"
            )

    curvature_cap = float(EXPECTED_DIAG_CONFIG["curvature_clip"])
    if curvature_clip_active is True:
        if curvature_preclip_max is not None and not curvature_preclip_max > curvature_cap:
            issues.append(f"{context}: active curvature cap lacks a strict exceedance")
        if curvature_excess_mean is not None and curvature_excess_mean <= 0.0:
            issues.append(f"{context}.curvature_clip_excess_mean is not positive")
        if curvature_excess_max is not None and curvature_excess_max <= 0.0:
            issues.append(f"{context}.curvature_clip_excess_max is not positive")
        if curv_max is not None and not np.isclose(
            curv_max, curvature_cap, rtol=1e-12, atol=1e-12
        ):
            issues.append(f"{context}.curv_max does not reach the configured cap")
    elif curvature_clip_active is False:
        if curvature_preclip_max is not None and curvature_preclip_max > curvature_cap:
            issues.append(f"{context}: inactive curvature cap has a strict exceedance")
        if (
            curvature_preclip_max is not None
            and curv_max is not None
            and not np.isclose(
                curv_max, curvature_preclip_max, rtol=1e-12, atol=1e-12
            )
        ):
            issues.append(f"{context}: inactive cap changed the curvature maximum")
        for field, value in (
            ("curvature_clip_excess_mean", curvature_excess_mean),
            ("curvature_clip_excess_max", curvature_excess_max),
        ):
            if value is not None and value != 0.0:
                issues.append(f"{context}.{field} must be zero when clipping is inactive")
    if (
        curvature_excess_mean is not None
        and curvature_excess_max is not None
        and (
            curvature_excess_mean < 0.0
            or (
                curvature_excess_max < curvature_excess_mean
                and not np.isclose(
                    curvature_excess_max,
                    curvature_excess_mean,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        )
    ):
        issues.append(f"{context}: curvature clip excess statistics are invalid")
    if (
        curvature_clip_active is True
        and curvature_preclip_max is not None
        and curvature_excess_max is not None
        and not np.isclose(
            curvature_excess_max,
            curvature_preclip_max - curvature_cap,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        issues.append(f"{context}: curvature maximum/excess arithmetic is inconsistent")
    if curv_min is not None and curv_min < 0.0:
        issues.append(f"{context}.curv_min is negative")
    if curv_max is not None and curv_max > curvature_cap + 1e-12:
        issues.append(f"{context}.curv_max exceeds the configured cap")
    if curv_min is not None and curv_mean is not None and curv_max is not None and not (
        curv_min <= curv_mean <= curv_max
    ):
        issues.append(f"{context}: clipped curvature min/mean/max are inconsistent")
    if (
        curvature_count_valid
        and curvature_preclip_mean is not None
        and curv_mean is not None
        and curvature_clip_count is not None
        and curvature_excess_mean is not None
    ):
        expected_curv_mean = curvature_preclip_mean - (
            curvature_clip_count * curvature_excess_mean / curvature_coordinates
        )
        if not np.isclose(curv_mean, expected_curv_mean, rtol=1e-10, atol=1e-10):
            issues.append(f"{context}: pre/post curvature means are inconsistent")

    multiplier_coordinates = _required_numeric(
        record, "multiplier_coordinate_count", context, issues
    )
    raw_multiplier_min = _required_numeric(
        record, "raw_step_multiplier_min", context, issues
    )
    raw_multiplier_max = _required_numeric(
        record, "raw_step_multiplier_max", context, issues
    )
    multiplier_diagnostics_exact = _required_bool(
        record, "multiplier_clipping_diagnostics_exact", context, issues
    )
    if multiplier_diagnostics_exact is not True:
        issues.append(
            f"{context}.multiplier_clipping_diagnostics_exact must be true"
        )
    floor_clip_count = _required_numeric(
        record, "multiplier_floor_clip_count", context, issues
    )
    floor_clip_frac = _required_numeric(
        record, "multiplier_floor_clip_frac", context, issues
    )
    floor_clip_active = _required_bool(
        record, "multiplier_floor_clip_active", context, issues
    )
    floor_deficit_mean = _required_numeric(
        record, "multiplier_floor_clip_deficit_mean", context, issues
    )
    floor_deficit_max = _required_numeric(
        record, "multiplier_floor_clip_deficit_max", context, issues
    )
    ceiling_clip_count = _required_numeric(
        record, "multiplier_ceiling_clip_count", context, issues
    )
    ceiling_clip_frac = _required_numeric(
        record, "multiplier_ceiling_clip_frac", context, issues
    )
    ceiling_clip_active = _required_bool(
        record, "multiplier_ceiling_clip_active", context, issues
    )
    ceiling_excess_mean = _required_numeric(
        record, "multiplier_ceiling_clip_excess_mean", context, issues
    )
    ceiling_excess_max = _required_numeric(
        record, "multiplier_ceiling_clip_excess_max", context, issues
    )
    multiplier_min = _required_numeric(record, "step_multiplier_min", context, issues)
    multiplier_max = _required_numeric(record, "step_multiplier_max", context, issues)
    legacy_floor_frac = _required_numeric(
        record, "multiplier_floor_frac", context, issues
    )

    multiplier_count_valid = bool(
        multiplier_coordinates is not None
        and _is_nonnegative_integer(multiplier_coordinates, positive=True)
    )
    if multiplier_coordinates is not None and not multiplier_count_valid:
        issues.append(f"{context}.multiplier_coordinate_count is not a positive integer")
    elif multiplier_coordinates is not None and multiplier_coordinates != EXPECTED_PARAMETER_COUNT:
        issues.append(
            f"{context}.multiplier_coordinate_count={multiplier_coordinates:g}, "
            f"expected {EXPECTED_PARAMETER_COUNT} for the locked Hopper policy"
        )
    if (
        curvature_count_valid
        and multiplier_count_valid
        and multiplier_coordinates != curvature_coordinates
    ):
        issues.append(f"{context}: curvature/multiplier coordinate counts disagree")
    if floor_clip_count is not None and (
        not _is_nonnegative_integer(floor_clip_count)
        or (multiplier_count_valid and floor_clip_count > multiplier_coordinates)
    ):
        issues.append(f"{context}.multiplier_floor_clip_count is invalid")
    if floor_clip_frac is not None and not (0.0 <= floor_clip_frac <= 1.0):
        issues.append(f"{context}.multiplier_floor_clip_frac is outside [0, 1]")
    if multiplier_count_valid and floor_clip_count is not None and floor_clip_frac is not None:
        if not np.isclose(
            floor_clip_frac,
            floor_clip_count / multiplier_coordinates,
            rtol=1e-12,
            atol=1e-12,
        ):
            issues.append(
                f"{context}.multiplier_floor_clip_frac disagrees with count/coordinates"
            )
    if floor_clip_count is not None and floor_clip_active is not None:
        if floor_clip_active is not (floor_clip_count > 0.0):
            issues.append(
                f"{context}.multiplier_floor_clip_active disagrees with clip count"
            )
    if raw_multiplier_min is not None and raw_multiplier_min <= 0.0:
        issues.append(f"{context}.raw_step_multiplier_min is not positive")
    if (
        raw_multiplier_min is not None
        and raw_multiplier_max is not None
        and raw_multiplier_max < raw_multiplier_min
    ):
        issues.append(f"{context}: raw multiplier maximum is below its minimum")
    if raw_multiplier_max is not None and raw_multiplier_max > 1.0 + 1e-12:
        issues.append(f"{context}: raw multiplier exceeds the configured upper bound")
    for field, value, expected in (
        ("multiplier_ceiling_clip_count", ceiling_clip_count, 0.0),
        ("multiplier_ceiling_clip_frac", ceiling_clip_frac, 0.0),
        ("multiplier_ceiling_clip_excess_mean", ceiling_excess_mean, 0.0),
        ("multiplier_ceiling_clip_excess_max", ceiling_excess_max, 0.0),
    ):
        if value is not None and value != expected:
            issues.append(f"{context}.{field} must be zero in the locked protocol")
    if ceiling_clip_active is not None and ceiling_clip_active is not False:
        issues.append(
            f"{context}.multiplier_ceiling_clip_active must be false in the locked protocol"
        )

    multiplier_floor = float(EXPECTED_DIAG_CONFIG["min_step_multiplier"])
    if floor_clip_active is True:
        if raw_multiplier_min is not None and not raw_multiplier_min < multiplier_floor:
            issues.append(f"{context}: active multiplier floor lacks a strict deficit")
        if floor_deficit_mean is not None and floor_deficit_mean <= 0.0:
            issues.append(
                f"{context}.multiplier_floor_clip_deficit_mean is not positive"
            )
        if floor_deficit_max is not None and floor_deficit_max <= 0.0:
            issues.append(
                f"{context}.multiplier_floor_clip_deficit_max is not positive"
            )
    elif floor_clip_active is False:
        if raw_multiplier_min is not None and raw_multiplier_min < multiplier_floor:
            issues.append(f"{context}: inactive multiplier floor has a strict deficit")
        for field, value in (
            ("multiplier_floor_clip_deficit_mean", floor_deficit_mean),
            ("multiplier_floor_clip_deficit_max", floor_deficit_max),
        ):
            if value is not None and value != 0.0:
                issues.append(f"{context}.{field} must be zero when clipping is inactive")
    if (
        floor_deficit_mean is not None
        and floor_deficit_max is not None
        and (
            floor_deficit_mean < 0.0
            or (
                floor_deficit_max < floor_deficit_mean
                and not np.isclose(
                    floor_deficit_max,
                    floor_deficit_mean,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        )
    ):
        issues.append(f"{context}: multiplier floor deficit statistics are invalid")
    if (
        floor_clip_active is True
        and raw_multiplier_min is not None
        and floor_deficit_max is not None
        and not np.isclose(
            floor_deficit_max,
            multiplier_floor - raw_multiplier_min,
            rtol=1e-12,
            atol=1e-12,
        )
    ):
        issues.append(f"{context}: multiplier minimum/deficit arithmetic is inconsistent")
    if multiplier_min is not None and raw_multiplier_min is not None:
        expected_multiplier_min = max(raw_multiplier_min, multiplier_floor)
        if not np.isclose(
            multiplier_min, expected_multiplier_min, rtol=1e-12, atol=1e-12
        ):
            issues.append(f"{context}: raw/applied multiplier minima are inconsistent")
    if multiplier_max is not None and raw_multiplier_max is not None:
        expected_multiplier_max = min(raw_multiplier_max, 1.0)
        if not np.isclose(
            multiplier_max, expected_multiplier_max, rtol=1e-12, atol=1e-12
        ):
            issues.append(f"{context}: raw/applied multiplier maxima are inconsistent")
    if legacy_floor_frac is not None:
        if not (0.0 <= legacy_floor_frac <= 1.0):
            issues.append(f"{context}.multiplier_floor_frac is outside [0, 1]")
        if floor_clip_frac is not None and legacy_floor_frac + 1e-12 < floor_clip_frac:
            issues.append(
                f"{context}: legacy at-floor fraction is below strict clip fraction"
            )

    division = _required_numeric(record, "division_relative_residual", context, issues)
    applied = _required_numeric(record, "applied_relative_residual", context, issues)
    linear_condition = _required_numeric(record, "linear_condition_estimate", context, issues)
    minimum = _required_numeric(record, "linear_min_abs_diagonal", context, issues)
    maximum = _required_numeric(record, "linear_max_abs_diagonal", context, issues)
    if division is not None and division < 0.0:
        issues.append(f"{context}.division_relative_residual is negative")
    if applied is not None and applied < 0.0:
        issues.append(f"{context}.applied_relative_residual is negative")
    if linear_condition is not None and linear_condition < 1.0 - 1e-12:
        issues.append(f"{context}.linear_condition_estimate is below one")
    if minimum is not None and minimum <= 0.0:
        issues.append(f"{context}.linear_min_abs_diagonal is not positive")
    if minimum is not None and maximum is not None and maximum < minimum:
        issues.append(f"{context}: maximum absolute diagonal is below the minimum")
    if minimum is not None and maximum is not None and linear_condition is not None:
        expected_condition = maximum / minimum
        if not np.isclose(
            linear_condition, expected_condition, rtol=1e-10, atol=1e-12
        ):
            issues.append(
                f"{context}.linear_condition_estimate disagrees with diagonal extrema"
            )
    learning_rate = _finite_float(record.get("lr"))
    if (
        learning_rate is not None
        and curv_min is not None
        and curv_max is not None
        and minimum is not None
        and maximum is not None
    ):
        expected_minimum = 1.0 + learning_rate * curv_min
        expected_maximum = 1.0 + learning_rate * curv_max
        if not np.isclose(minimum, expected_minimum, rtol=1e-10, atol=1e-10):
            issues.append(
                f"{context}.linear_min_abs_diagonal disagrees with curvature and lr"
            )
        if not np.isclose(maximum, expected_maximum, rtol=1e-10, atol=1e-10):
            issues.append(
                f"{context}.linear_max_abs_diagonal disagrees with curvature and lr"
            )
    if maximum is not None and raw_multiplier_min is not None:
        if not np.isclose(
            raw_multiplier_min, 1.0 / maximum, rtol=1e-10, atol=1e-12
        ):
            issues.append(
                f"{context}.raw_step_multiplier_min disagrees with linear diagonal"
            )
    if minimum is not None and raw_multiplier_max is not None:
        if not np.isclose(
            raw_multiplier_max, 1.0 / minimum, rtol=1e-10, atol=1e-12
        ):
            issues.append(
                f"{context}.raw_step_multiplier_max disagrees with linear diagonal"
            )


def _load_coordinate_history(
    path: str,
    *,
    label: str,
    expected_iterations: int,
    issues: list[str],
) -> np.ndarray | None:
    """Load one required full-coordinate diagnostic matrix without copying it."""

    if not os.path.isfile(path):
        issues.append(f"{path}: required diagonal coordinate artifact is missing")
        return None
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as error:
        issues.append(f"{path}: unreadable {label} coordinate artifact: {error}")
        return None
    if not isinstance(values, np.ndarray):
        issues.append(f"{path}: {label} coordinate artifact is not a NumPy array")
        return None
    if values.dtype != np.dtype(np.float64):
        issues.append(
            f"{path}: {label} coordinate artifact has dtype {values.dtype}, "
            "expected float64"
        )
        return None
    expected_shape = (expected_iterations, EXPECTED_PARAMETER_COUNT)
    if values.shape != expected_shape:
        issues.append(
            f"{path}: {label} coordinate artifact has shape {values.shape}, "
            f"expected {expected_shape}"
        )
        return None
    if not bool(np.all(np.isfinite(values))):
        issues.append(f"{path}: {label} coordinate artifact contains non-finite values")
        return None
    return values


def _check_coordinate_statistic(
    record: dict[str, Any],
    field: str,
    expected: bool | int | float,
    context: str,
    issues: list[str],
) -> None:
    """Cross-check one scalar history field against the coordinate artifacts."""

    if field not in record:
        issues.append(
            f"{context} is missing coordinate-crosschecked field {field}"
        )
        return
    actual = record[field]
    if isinstance(expected, bool):
        matches = isinstance(actual, bool) and actual is expected
    elif isinstance(expected, int):
        numeric = _finite_float(actual)
        matches = numeric is not None and numeric == expected
    else:
        numeric = _finite_float(actual)
        matches = numeric is not None and bool(
            np.isclose(numeric, expected, rtol=1e-12, atol=1e-12)
        )
    if not matches:
        issues.append(
            f"{context}.{field}={actual!r} disagrees with reconstructed "
            f"coordinate artifacts ({expected!r})"
        )


def _diag_coordinate_artifact_issues(
    history: Any,
    *,
    run_dir: str,
    expected_iterations: int,
) -> list[str]:
    """Validate full Hessian/multiplier histories and their scalar projections."""

    issues: list[str] = []
    hessian_path = os.path.join(run_dir, HESSIAN_FOR_STEP_HISTORY_FILENAME)
    multiplier_path = os.path.join(run_dir, STEP_MULTIPLIER_HISTORY_FILENAME)
    hessian_history = _load_coordinate_history(
        hessian_path,
        label="Hessian-for-step",
        expected_iterations=expected_iterations,
        issues=issues,
    )
    multiplier_history = _load_coordinate_history(
        multiplier_path,
        label="step-multiplier",
        expected_iterations=expected_iterations,
        issues=issues,
    )
    if hessian_history is None or multiplier_history is None:
        return issues
    if (
        not isinstance(history, list)
        or len(history) != expected_iterations
        or any(not isinstance(record, dict) for record in history)
    ):
        return issues

    coordinate_count = EXPECTED_PARAMETER_COUNT
    curvature_cap = float(EXPECTED_DIAG_CONFIG["curvature_clip"])
    multiplier_floor = float(EXPECTED_DIAG_CONFIG["min_step_multiplier"])
    for index, record in enumerate(history):
        context = f"{run_dir}: history[{index}]"
        learning_rate = _finite_float(record.get("lr"))
        if learning_rate is None:
            continue

        hessian_for_step = np.asarray(hessian_history[index])
        saved_multiplier = np.asarray(multiplier_history[index])
        curvature_preclip = np.maximum(-hessian_for_step, 0.0)
        curvature_active_mask = curvature_preclip > 0.0
        curvature_clip_mask = curvature_preclip > curvature_cap
        curvature = np.clip(curvature_preclip, 0.0, curvature_cap)
        linear_diagonal = 1.0 + learning_rate * curvature
        raw_multiplier = 1.0 / linear_diagonal
        applied_multiplier = np.clip(raw_multiplier, multiplier_floor, 1.0)

        if not np.allclose(
            saved_multiplier,
            applied_multiplier,
            rtol=0.0,
            atol=0.0,
        ):
            mismatch_count = int(np.count_nonzero(saved_multiplier != applied_multiplier))
            issues.append(
                f"{multiplier_path}: iteration {index} differs from the exactly "
                f"reconstructed applied multiplier at {mismatch_count} coordinate(s)"
            )

        curvature_active_count = int(np.count_nonzero(curvature_active_mask))
        curvature_clip_count = int(np.count_nonzero(curvature_clip_mask))
        if curvature_clip_count:
            curvature_excess = (
                curvature_preclip[curvature_clip_mask] - curvature_cap
            )
            curvature_excess_mean = float(np.mean(curvature_excess))
            curvature_excess_max = float(np.max(curvature_excess))
        else:
            curvature_excess_mean = 0.0
            curvature_excess_max = 0.0

        floor_clip_mask = raw_multiplier < multiplier_floor
        floor_clip_count = int(np.count_nonzero(floor_clip_mask))
        if floor_clip_count:
            floor_deficit = multiplier_floor - raw_multiplier[floor_clip_mask]
            floor_deficit_mean = float(np.mean(floor_deficit))
            floor_deficit_max = float(np.max(floor_deficit))
        else:
            floor_deficit_mean = 0.0
            floor_deficit_max = 0.0

        ceiling_clip_mask = raw_multiplier > 1.0
        ceiling_clip_count = int(np.count_nonzero(ceiling_clip_mask))
        if ceiling_clip_count:
            ceiling_excess = raw_multiplier[ceiling_clip_mask] - 1.0
            ceiling_excess_mean = float(np.mean(ceiling_excess))
            ceiling_excess_max = float(np.max(ceiling_excess))
        else:
            ceiling_excess_mean = 0.0
            ceiling_excess_max = 0.0

        reconstructed: dict[str, bool | int | float] = {
            "h_step_mean": float(np.mean(hessian_for_step)),
            "h_step_min": float(np.min(hessian_for_step)),
            "h_step_max": float(np.max(hessian_for_step)),
            "curvature_coordinate_count": coordinate_count,
            "curvature_active_count": curvature_active_count,
            "curvature_active_frac": curvature_active_count / coordinate_count,
            "curvature_preclip_mean": float(np.mean(curvature_preclip)),
            "curvature_preclip_max": float(np.max(curvature_preclip)),
            "curvature_clip_count": curvature_clip_count,
            "curvature_clip_frac": curvature_clip_count / coordinate_count,
            "curvature_clip_active": curvature_clip_count > 0,
            "curvature_clip_excess_mean": curvature_excess_mean,
            "curvature_clip_excess_max": curvature_excess_max,
            "curv_mean": float(np.mean(curvature)),
            "curv_min": float(np.min(curvature)),
            "curv_max": float(np.max(curvature)),
            "multiplier_coordinate_count": coordinate_count,
            "raw_step_multiplier_min": float(np.min(raw_multiplier)),
            "raw_step_multiplier_max": float(np.max(raw_multiplier)),
            "multiplier_floor_clip_count": floor_clip_count,
            "multiplier_floor_clip_frac": floor_clip_count / coordinate_count,
            "multiplier_floor_clip_active": floor_clip_count > 0,
            "multiplier_floor_clip_deficit_mean": floor_deficit_mean,
            "multiplier_floor_clip_deficit_max": floor_deficit_max,
            "multiplier_ceiling_clip_count": ceiling_clip_count,
            "multiplier_ceiling_clip_frac": ceiling_clip_count / coordinate_count,
            "multiplier_ceiling_clip_active": ceiling_clip_count > 0,
            "multiplier_ceiling_clip_excess_mean": ceiling_excess_mean,
            "multiplier_ceiling_clip_excess_max": ceiling_excess_max,
            "step_multiplier_min": float(np.min(applied_multiplier)),
            "step_multiplier_max": float(np.max(applied_multiplier)),
            "step_multiplier_mean": float(np.mean(applied_multiplier)),
            "step_multiplier_std": float(np.std(applied_multiplier)),
            "step_multiplier_cv": float(
                np.std(applied_multiplier) / (np.mean(applied_multiplier) + 1e-12)
            ),
            "hessian_shrinkage_median": float(np.median(applied_multiplier)),
            "hessian_shrinkage_p90": float(
                np.percentile(applied_multiplier, 90.0)
            ),
            "hessian_shrinkage_max": float(np.max(applied_multiplier)),
            "multiplier_floor_frac": float(
                np.mean(applied_multiplier <= multiplier_floor + 1e-12)
            ),
            "linear_min_abs_diagonal": float(np.min(linear_diagonal)),
            "linear_max_abs_diagonal": float(np.max(linear_diagonal)),
            "linear_condition_estimate": float(
                np.max(linear_diagonal) / np.min(linear_diagonal)
            ),
        }
        for field, expected in reconstructed.items():
            _check_coordinate_statistic(record, field, expected, context, issues)
    return issues


def _history_issues(
    history: Any,
    *,
    run_dir: str,
    condition: str,
    schedule: str,
    alpha0: float,
    expected_iterations: int,
) -> list[str]:
    if not isinstance(history, list):
        return [f"{run_dir}: history.json is not a list"]
    issues: list[str] = []
    if len(history) != expected_iterations:
        issues.append(
            f"{run_dir}: incomplete history: expected {expected_iterations}, found {len(history)}"
        )

    nonfinite = _nonfinite_paths(history)
    if nonfinite:
        suffix = " ..." if len(nonfinite) > 10 else ""
        issues.append(
            f"{run_dir}: non-finite history value(s): "
            f"{', '.join(nonfinite[:10])}{suffix}"
        )

    train_steps: list[float] = []
    eval_returns: list[float] = []
    best_returns: list[float] = []
    schedule_mismatches: list[int] = []
    trust_active: list[int] = []
    trust_scaled: list[int] = []
    curvature_coordinate_counts: list[int] = []
    multiplier_coordinate_counts: list[int] = []
    for index, record in enumerate(history):
        context = f"{run_dir}: history[{index}]"
        if not isinstance(record, dict):
            issues.append(f"{context} is not an object")
            continue
        if record.get("iteration") != index:
            issues.append(f"{context}.iteration={record.get('iteration')!r}, expected {index}")

        lr = _required_numeric(record, "lr", context, issues)
        expected_lr = learning_rate_at_iteration(alpha0, index, schedule)
        if lr is not None and not np.isclose(
            lr, expected_lr, rtol=1e-12, atol=1e-12
        ):
            schedule_mismatches.append(index)
        if "learning_rate" in record:
            recorded_alpha = _finite_float(record.get("learning_rate"))
            if recorded_alpha is None or not np.isclose(
                recorded_alpha, expected_lr, rtol=1e-12, atol=1e-12
            ):
                issues.append(f"{context}.learning_rate disagrees with lr")

        if record.get("trust_active") is not False:
            trust_active.append(index)
        scale = _required_numeric(record, "trust_scale", context, issues)
        if scale is not None and not np.isclose(scale, 1.0, rtol=0.0, atol=1e-12):
            trust_scaled.append(index)

        for field in ("mean_fitness", "max_fitness", "grad_norm", "step_norm"):
            _required_numeric(record, field, context, issues)
        n_fresh = _required_numeric(record, "n_fresh", context, issues)
        n_reused = _required_numeric(record, "n_reused", context, issues)
        sigma = _required_numeric(record, "sigma", context, issues)
        if n_fresh is not None and n_reused is not None:
            valid_counts = (
                n_fresh >= 0.0
                and n_reused >= 0.0
                and n_fresh == int(n_fresh)
                and n_reused == int(n_reused)
                and n_fresh + n_reused == float(EXPECTED_POPULATION_SIZE)
            )
            if not valid_counts:
                issues.append(
                    f"{context}: fresh/reused counts do not form population "
                    f"{EXPECTED_POPULATION_SIZE}"
                )
            if n_fresh != EXPECTED_POPULATION_SIZE or n_reused != 0.0:
                issues.append(
                    f"{context}: fresh-only population counts are "
                    f"{n_fresh}/{n_reused}, expected "
                    f"{EXPECTED_POPULATION_SIZE}/0"
                )
        if sigma is not None and not np.isclose(
            sigma, 0.02, rtol=1e-12, atol=1e-12
        ):
            issues.append(f"{context}.sigma is not the main Hopper value 0.02")
        evaluation = _required_numeric(record, "eval_reward", context, issues)
        best = _required_numeric(record, "best_reward", context, issues)
        step_count = _required_numeric(record, "train_env_steps", context, issues)
        if evaluation is not None:
            eval_returns.append(evaluation)
        if best is not None:
            best_returns.append(best)
        if step_count is not None:
            train_steps.append(step_count)

        if condition == "diag_curvature":
            _validate_diag_record(record, index, context, issues)
            curvature_coordinates = _finite_float(
                record.get("curvature_coordinate_count")
            )
            multiplier_coordinates = _finite_float(
                record.get("multiplier_coordinate_count")
            )
            if curvature_coordinates is not None and _is_nonnegative_integer(
                curvature_coordinates, positive=True
            ):
                curvature_coordinate_counts.append(int(curvature_coordinates))
            if multiplier_coordinates is not None and _is_nonnegative_integer(
                multiplier_coordinates, positive=True
            ):
                multiplier_coordinate_counts.append(int(multiplier_coordinates))

    if schedule_mismatches:
        issues.append(
            f"{run_dir}: lr deviates from {schedule} at iteration(s) "
            f"{schedule_mismatches[:10]}"
        )
    if trust_active:
        issues.append(
            f"{run_dir}: trust region activated at iteration(s) {trust_active[:10]}"
        )
    if trust_scaled:
        issues.append(
            f"{run_dir}: trust_scale is not one at iteration(s) {trust_scaled[:10]}"
        )
    if curvature_coordinate_counts and len(set(curvature_coordinate_counts)) != 1:
        issues.append(f"{run_dir}: curvature coordinate count changes across iterations")
    if multiplier_coordinate_counts and len(set(multiplier_coordinate_counts)) != 1:
        issues.append(f"{run_dir}: multiplier coordinate count changes across iterations")
    if len(train_steps) == len(history) and train_steps:
        steps = np.asarray(train_steps, dtype=np.float64)
        if steps[0] <= 0.0 or np.any(np.diff(steps) <= 0.0):
            issues.append(f"{run_dir}: train_env_steps is not positive and strictly increasing")
    if len(eval_returns) == len(history) and len(best_returns) == len(history):
        expected_best = np.maximum.accumulate(np.asarray(eval_returns, dtype=np.float64))
        actual_best = np.asarray(best_returns, dtype=np.float64)
        if not np.allclose(actual_best, expected_best, rtol=1e-10, atol=1e-10):
            issues.append(f"{run_dir}: best_reward is not the cumulative best eval_reward")
    return issues


def _mean(history: Sequence[dict[str, Any]], field: str, *, start: int = 0) -> float:
    return float(np.mean([float(record[field]) for record in history[start:]]))


def _maximum(history: Sequence[dict[str, Any]], field: str) -> float:
    return float(np.max([float(record[field]) for record in history]))


def _optional_values(
    history: Sequence[dict[str, Any]], field: str, *, start: int = 0
) -> list[float]:
    return [
        float(record[field])
        for record in history[start:]
        if field in record and record[field] is not None
    ]


def _row_from_run(
    config: dict[str, Any],
    history: list[dict[str, Any]],
    run_dir: str,
    *,
    alpha0: float,
    job_id: str | None,
    task_id: int | None,
) -> dict[str, Any]:
    returns = np.asarray([float(record["eval_reward"]) for record in history])
    x = np.arange(len(returns), dtype=np.float64)
    integrate = getattr(np, "trapezoid", np.trapz)
    iteration_auc = (
        float(integrate(returns, x) / x[-1]) if len(returns) > 1 else float(returns[0])
    )
    row: dict[str, Any] = {
        "env": config["env_name"],
        "condition": config["condition"],
        "lr_schedule": config["lr_schedule"],
        "initial_learning_rate": alpha0,
        "seed": int(config["seed"]),
        "iterations": len(history),
        "final_train_env_steps": int(history[-1]["train_env_steps"]),
        "mean_eval_return": float(np.mean(returns)),
        "iteration_auc": iteration_auc,
        "final_eval_return": float(returns[-1]),
        "last_10_eval_return": float(np.mean(returns[-min(10, len(returns)) :])),
        "best_eval_return": float(np.max(returns)),
        "source_sha256": config["source_sha256"],
        "slurm_job_id": job_id,
        "slurm_task_id": task_id,
        "run_dir": run_dir,
    }
    if config["condition"] == "diag_curvature":
        split_correlations = _optional_values(history, "h_split_correlation")
        temporal_correlations = _optional_values(
            history, "h_temporal_correlation", start=1
        )
        temporal_comparisons = max(0, len(history) - 1)
        condition_values = np.asarray(
            [float(record["linear_condition_estimate"]) for record in history]
        )
        floor_values = np.asarray(
            [float(record["multiplier_floor_frac"]) for record in history]
        )
        curvature_active_counts = np.asarray(
            [float(record["curvature_active_count"]) for record in history]
        )
        curvature_active_fracs = np.asarray(
            [float(record["curvature_active_frac"]) for record in history]
        )
        curvature_clip_counts = np.asarray(
            [float(record["curvature_clip_count"]) for record in history]
        )
        curvature_clip_fracs = np.asarray(
            [float(record["curvature_clip_frac"]) for record in history]
        )
        curvature_clip_active = np.asarray(
            [bool(record["curvature_clip_active"]) for record in history]
        )
        multiplier_floor_clip_counts = np.asarray(
            [float(record["multiplier_floor_clip_count"]) for record in history]
        )
        multiplier_floor_clip_fracs = np.asarray(
            [float(record["multiplier_floor_clip_frac"]) for record in history]
        )
        multiplier_floor_clip_active = np.asarray(
            [bool(record["multiplier_floor_clip_active"]) for record in history]
        )
        multiplier_ceiling_clip_counts = np.asarray(
            [float(record["multiplier_ceiling_clip_count"]) for record in history]
        )
        multiplier_ceiling_clip_active = np.asarray(
            [bool(record["multiplier_ceiling_clip_active"]) for record in history]
        )
        total_curvature_clip_count = int(np.sum(curvature_clip_counts))
        total_multiplier_floor_clip_count = int(
            np.sum(multiplier_floor_clip_counts)
        )
        curvature_excess_means = np.asarray(
            [float(record["curvature_clip_excess_mean"]) for record in history]
        )
        multiplier_deficit_means = np.asarray(
            [
                float(record["multiplier_floor_clip_deficit_mean"])
                for record in history
            ]
        )
        row.update(
            {
                "mean_hessian_pairs": _mean(history, "hessian_pairs"),
                "mean_h_split_correlation": (
                    float(np.mean(split_correlations)) if split_correlations else None
                ),
                "h_split_correlation_available_fraction": float(
                    len(split_correlations) / len(history)
                ),
                "mean_h_split_sign_agreement": _mean(
                    history, "h_split_sign_agreement"
                ),
                "mean_h_split_relative_disagreement": _mean(
                    history, "h_split_relative_disagreement"
                ),
                "mean_h_temporal_correlation": (
                    float(np.mean(temporal_correlations))
                    if temporal_correlations
                    else None
                ),
                "h_temporal_correlation_available_fraction": float(
                    len(temporal_correlations) / temporal_comparisons
                    if temporal_comparisons
                    else 0.0
                ),
                "mean_h_temporal_sign_agreement": _mean(
                    history, "h_temporal_sign_agreement", start=1
                ),
                "mean_division_relative_residual": _mean(
                    history, "division_relative_residual"
                ),
                "max_division_relative_residual": _maximum(
                    history, "division_relative_residual"
                ),
                "mean_applied_relative_residual": _mean(
                    history, "applied_relative_residual"
                ),
                "max_applied_relative_residual": _maximum(
                    history, "applied_relative_residual"
                ),
                "mean_linear_condition_estimate": float(np.mean(condition_values)),
                "median_linear_condition_estimate": float(np.median(condition_values)),
                "max_linear_condition_estimate": float(np.max(condition_values)),
                "minimum_linear_abs_diagonal": float(
                    np.min([record["linear_min_abs_diagonal"] for record in history])
                ),
                "maximum_linear_abs_diagonal": float(
                    np.max([record["linear_max_abs_diagonal"] for record in history])
                ),
                "curvature_coordinate_count": int(
                    history[0]["curvature_coordinate_count"]
                ),
                "mean_curvature_active_count": float(
                    np.mean(curvature_active_counts)
                ),
                "mean_curvature_active_frac": float(
                    np.mean(curvature_active_fracs)
                ),
                "mean_curvature_preclip_mean": _mean(
                    history, "curvature_preclip_mean"
                ),
                "max_curvature_preclip_max": _maximum(
                    history, "curvature_preclip_max"
                ),
                "total_curvature_clip_count": total_curvature_clip_count,
                "mean_curvature_clip_count": float(
                    np.mean(curvature_clip_counts)
                ),
                "max_curvature_clip_count": int(
                    np.max(curvature_clip_counts)
                ),
                "mean_curvature_clip_frac": float(
                    np.mean(curvature_clip_fracs)
                ),
                "max_curvature_clip_frac": float(
                    np.max(curvature_clip_fracs)
                ),
                "curvature_clip_active_iteration_fraction": float(
                    np.mean(curvature_clip_active)
                ),
                "mean_curvature_clip_excess_mean": _mean(
                    history, "curvature_clip_excess_mean"
                ),
                "mean_curvature_clip_excess_per_clipped_coordinate": (
                    float(
                        np.sum(curvature_clip_counts * curvature_excess_means)
                        / total_curvature_clip_count
                    )
                    if total_curvature_clip_count
                    else 0.0
                ),
                "max_curvature_clip_excess_max": _maximum(
                    history, "curvature_clip_excess_max"
                ),
                "multiplier_coordinate_count": int(
                    history[0]["multiplier_coordinate_count"]
                ),
                "minimum_raw_step_multiplier": float(
                    np.min([record["raw_step_multiplier_min"] for record in history])
                ),
                "maximum_raw_step_multiplier": float(
                    np.max([record["raw_step_multiplier_max"] for record in history])
                ),
                "total_multiplier_floor_clip_count": (
                    total_multiplier_floor_clip_count
                ),
                "mean_multiplier_floor_clip_count": float(
                    np.mean(multiplier_floor_clip_counts)
                ),
                "max_multiplier_floor_clip_count": int(
                    np.max(multiplier_floor_clip_counts)
                ),
                "mean_multiplier_floor_clip_frac": float(
                    np.mean(multiplier_floor_clip_fracs)
                ),
                "max_multiplier_floor_clip_frac": float(
                    np.max(multiplier_floor_clip_fracs)
                ),
                "multiplier_floor_clip_active_iteration_fraction": float(
                    np.mean(multiplier_floor_clip_active)
                ),
                "mean_multiplier_floor_clip_deficit_mean": _mean(
                    history, "multiplier_floor_clip_deficit_mean"
                ),
                "mean_multiplier_floor_clip_deficit_per_clipped_coordinate": (
                    float(
                        np.sum(
                            multiplier_floor_clip_counts
                            * multiplier_deficit_means
                        )
                        / total_multiplier_floor_clip_count
                    )
                    if total_multiplier_floor_clip_count
                    else 0.0
                ),
                "max_multiplier_floor_clip_deficit_max": _maximum(
                    history, "multiplier_floor_clip_deficit_max"
                ),
                "total_multiplier_ceiling_clip_count": int(
                    np.sum(multiplier_ceiling_clip_counts)
                ),
                "multiplier_ceiling_clip_active_iteration_fraction": float(
                    np.mean(multiplier_ceiling_clip_active)
                ),
                "mean_multiplier_floor_frac": float(np.mean(floor_values)),
                "max_multiplier_floor_frac": float(np.max(floor_values)),
                "floor_active_iteration_fraction": float(np.mean(floor_values > 0.0)),
            }
        )
    else:
        row.update({field: None for field in MECHANISM_FIELDS})
    return row


def _validate_grid_arguments(
    conditions: Sequence[str],
    schedules: Sequence[str],
    alpha0s: Sequence[float],
    seeds: Sequence[int],
    expected_iterations: int,
    expected_source_sha: str | None,
) -> None:
    if not conditions or not schedules or not alpha0s or not seeds:
        raise ValueError("condition, schedule, alpha0, and seed grids must be nonempty")
    for label, values in (
        ("conditions", conditions),
        ("schedules", schedules),
        ("alpha0 values", alpha0s),
        ("seeds", seeds),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} contain duplicates")
    if set(conditions) - set(CONDITIONS):
        raise ValueError("conditions must be a subset of the production matrix")
    if set(schedules) - set(LR_SCHEDULES):
        raise ValueError("schedules must be a subset of the production matrix")
    if set(float(value) for value in alpha0s) - set(INITIAL_LEARNING_RATES):
        raise ValueError("alpha0 values must be a subset of the production matrix")
    if set(int(value) for value in seeds) - set(SEEDS):
        raise ValueError("seeds must be a subset of the production matrix")
    if expected_iterations <= 0:
        raise ValueError("expected_iterations must be positive")
    if expected_source_sha is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_source_sha
    ) is None:
        raise ValueError("expected_source_sha must be a lowercase SHA-256 digest")


def validate_and_collect(
    root: str,
    *,
    conditions: Sequence[str] = CONDITIONS,
    lr_schedules: Sequence[str] = LR_SCHEDULES,
    initial_learning_rates: Sequence[float] = INITIAL_LEARNING_RATES,
    seeds: Sequence[int] = SEEDS,
    expected_iterations: int = EXPECTED_ITERATIONS,
    expected_source_sha: str | None = None,
    validate_run_names: bool = True,
) -> list[dict[str, Any]]:
    """Return run rows only if the complete requested matrix is valid."""

    conditions = tuple(str(value) for value in conditions)
    lr_schedules = tuple(str(value) for value in lr_schedules)
    initial_learning_rates = tuple(float(value) for value in initial_learning_rates)
    seeds = tuple(int(value) for value in seeds)
    _validate_grid_arguments(
        conditions,
        lr_schedules,
        initial_learning_rates,
        seeds,
        expected_iterations,
        expected_source_sha,
    )
    expected_order = [
        (condition, schedule, alpha0, seed)
        for condition in conditions
        for schedule in lr_schedules
        for alpha0 in initial_learning_rates
        for seed in seeds
    ]
    expected = set(expected_order)
    candidates: dict[tuple[str, str, float, int], list[str]] = defaultdict(list)
    rows: dict[str, dict[str, Any]] = {}
    source_hashes: set[str] = set()
    issues: list[str] = []

    for run_dir in _candidate_run_dirs(root):
        config_path = os.path.join(run_dir, "config.json")
        history_path = os.path.join(run_dir, "history.json")
        if not os.path.exists(config_path) or not os.path.exists(history_path):
            issues.append(f"{run_dir}: missing config.json or history.json")
            continue
        try:
            config = _read_json(config_path)
            history = _read_json(history_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_dir}: unreadable result artifact: {error}")
            continue
        if not isinstance(config, dict):
            issues.append(f"{run_dir}: config.json is not an object")
            continue

        condition = str(config.get("condition", ""))
        schedule = str(config.get("lr_schedule", ""))
        alpha0 = _match_alpha(config.get("learning_rate"), initial_learning_rates)
        seed = _parse_seed(config.get("seed"))
        if alpha0 is None or seed is None:
            issues.append(f"{run_dir}: invalid learning-rate or seed cell metadata")
            continue
        cell = (condition, schedule, alpha0, seed)
        if cell not in expected:
            issues.append(f"{run_dir}: unexpected matrix cell {cell}")
            continue
        candidates[cell].append(run_dir)

        job_id: str | None = None
        task_id: int | None = None
        if validate_run_names:
            job_id, task_id, name_issue = _run_name_metadata(
                run_dir,
                condition=condition,
                schedule=schedule,
                alpha0=alpha0,
                seed=seed,
            )
            if name_issue is not None:
                issues.append(name_issue)

        config_problems = _config_issues(
            config,
            run_dir=run_dir,
            condition=condition,
            schedule=schedule,
            alpha0=alpha0,
            seed=seed,
            expected_iterations=expected_iterations,
            expected_source_sha=expected_source_sha,
        )
        issues.extend(config_problems)
        source_sha = config.get("source_sha256")
        if isinstance(source_sha, str) and re.fullmatch(r"[0-9a-f]{64}", source_sha):
            source_hashes.add(source_sha)

        status_path = os.path.join(run_dir, "status.json")
        if os.path.exists(status_path):
            try:
                status = _read_json(status_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                issues.append(f"{run_dir}: unreadable status.json: {error}")
            else:
                if not isinstance(status, dict):
                    issues.append(f"{run_dir}: status.json is not an object")
                elif status.get("status") != "complete" or status.get(
                    "completed_iterations"
                ) != expected_iterations:
                    issues.append(f"{run_dir}: status.json does not certify completion")

        history_problems = _history_issues(
            history,
            run_dir=run_dir,
            condition=condition,
            schedule=schedule,
            alpha0=alpha0,
            expected_iterations=expected_iterations,
        )
        issues.extend(history_problems)
        coordinate_artifact_problems: list[str] = []
        if condition == "diag_curvature":
            coordinate_artifact_problems = _diag_coordinate_artifact_issues(
                history,
                run_dir=run_dir,
                expected_iterations=expected_iterations,
            )
            issues.extend(coordinate_artifact_problems)
        if (
            not config_problems
            and not history_problems
            and not coordinate_artifact_problems
        ):
            rows[run_dir] = _row_from_run(
                config,
                history,
                run_dir,
                alpha0=alpha0,
                job_id=job_id,
                task_id=task_id,
            )

    for cell in expected_order:
        found = candidates.get(cell, [])
        if not found:
            issues.append(f"missing matrix cell {cell}")
        elif len(found) > 1:
            issues.append(f"duplicate matrix cell {cell}: {found}")
    if len(source_hashes) > 1:
        issues.append(f"source digest mismatch across runs: {sorted(source_hashes)}")
    if issues:
        raise HessianSweepValidationError(issues)
    return [rows[candidates[cell][0]] for cell in expected_order]


# Compatibility with the naming used by scripts/summarize_no_trust_sweep.py.
validate_and_collect_runs = validate_and_collect


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate validated runs by condition, schedule, and initial alpha."""

    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["condition"]),
                str(row["lr_schedule"]),
                float(row["initial_learning_rate"]),
            )
        ].append(row)
    output: list[dict[str, Any]] = []
    for (condition, schedule, alpha0), group in sorted(groups.items()):
        source_hashes = {str(row["source_sha256"]) for row in group}
        if len(source_hashes) != 1:
            raise ValueError("cannot aggregate mixed source digests")
        record: dict[str, Any] = {
            "env": EXPECTED_ENV,
            "condition": condition,
            "lr_schedule": schedule,
            "initial_learning_rate": alpha0,
            "runs": len(group),
            "source_sha256": next(iter(source_hashes)),
        }
        for field in PERFORMANCE_FIELDS:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            record[f"{field}_mean"] = float(np.mean(values))
            record[f"{field}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        for field in MECHANISM_FIELDS:
            values = [float(row[field]) for row in group if row.get(field) is not None]
            record[f"{field}_mean"] = float(np.mean(values)) if values else None
        output.append(record)
    return output


aggregate_runs = aggregate


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be a finite one-dimensional array")
    nonzero = values[values != 0.0]
    if len(nonzero) == 0:
        return 1.0
    if len(nonzero) > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 nonzero pairs")
    observed = abs(float(np.sum(nonzero)))
    tolerance = 1e-12 * max(1.0, observed, float(np.sum(np.abs(nonzero))))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(nonzero)):
        permuted = float(np.dot(nonzero, np.asarray(signs, dtype=np.float64)))
        if abs(permuted) >= observed - tolerance:
            extreme += 1
    return float(extreme / (2 ** len(nonzero)))


def paired_diag_minus_standard(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return seed-paired diagonal-Hessian minus Standard ES contrasts."""

    index: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    source_hashes = {str(row["source_sha256"]) for row in rows}
    environments = {str(row["env"]) for row in rows}
    iteration_counts = {int(row["iterations"]) for row in rows}
    if len(source_hashes) != 1 or len(environments) != 1 or len(iteration_counts) != 1:
        raise ValueError("paired rows do not share source, environment, and iteration count")
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["lr_schedule"]),
            float(row["initial_learning_rate"]),
            int(row["seed"]),
        )
        if key in index:
            raise ValueError(f"duplicate paired row {key}")
        index[key] = row

    schedules = sorted({str(row["lr_schedule"]) for row in rows})
    alpha0s = sorted({float(row["initial_learning_rate"]) for row in rows})
    cells: list[dict[str, Any]] = []
    for schedule in schedules:
        for alpha0 in alpha0s:
            baseline_seeds = {
                seed
                for condition, candidate_schedule, candidate_alpha, seed in index
                if condition == "standard_es"
                and candidate_schedule == schedule
                and candidate_alpha == alpha0
            }
            diag_seeds = {
                seed
                for condition, candidate_schedule, candidate_alpha, seed in index
                if condition == "diag_curvature"
                and candidate_schedule == schedule
                and candidate_alpha == alpha0
            }
            if not baseline_seeds or baseline_seeds != diag_seeds:
                raise ValueError(
                    "paired contrast requires identical nonempty seed sets for "
                    f"{schedule}, alpha0={alpha0:g}"
                )
            seeds = sorted(baseline_seeds)
            metrics: dict[str, Any] = {}
            for field in PERFORMANCE_FIELDS:
                standard = np.asarray(
                    [
                        float(index[("standard_es", schedule, alpha0, seed)][field])
                        for seed in seeds
                    ]
                )
                diagonal = np.asarray(
                    [
                        float(index[("diag_curvature", schedule, alpha0, seed)][field])
                        for seed in seeds
                    ]
                )
                differences = diagonal - standard
                metrics[field] = {
                    "standard_es_mean": float(np.mean(standard)),
                    "diag_curvature_mean": float(np.mean(diagonal)),
                    "paired_mean_difference": float(np.mean(differences)),
                    "paired_median_difference": float(np.median(differences)),
                    "paired_sample_sd": (
                        float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
                    ),
                    "wins": int(np.sum(differences > 0.0)),
                    "losses": int(np.sum(differences < 0.0)),
                    "ties": int(np.sum(differences == 0.0)),
                    "exact_two_sided_sign_flip_p": _exact_sign_flip_p(differences),
                    "differences_by_seed": [
                        {"seed": int(seed), "difference": float(difference)}
                        for seed, difference in zip(seeds, differences)
                    ],
                }
            diag_rows = [
                index[("diag_curvature", schedule, alpha0, seed)] for seed in seeds
            ]
            mechanism: dict[str, Any] = {}
            for field in MECHANISM_FIELDS:
                values = [
                    float(row[field])
                    for row in diag_rows
                    if row.get(field) is not None
                ]
                mechanism[field] = {
                    "valid_runs": len(values),
                    "mean_across_runs": float(np.mean(values)) if values else None,
                    "minimum_across_runs": float(np.min(values)) if values else None,
                    "maximum_across_runs": float(np.max(values)) if values else None,
                }
            cells.append(
                {
                    "lr_schedule": schedule,
                    "initial_learning_rate": alpha0,
                    "paired_runs": len(seeds),
                    "metrics": metrics,
                    "diag_curvature_mechanism": mechanism,
                }
            )
    return {
        "schema_version": 1,
        "baseline_condition": "standard_es",
        "condition": "diag_curvature",
        "difference_direction": "diag_curvature_minus_standard_es",
        "primary_metric": "iteration_auc",
        "environment": next(iter(environments)),
        "iterations": next(iter(iteration_counts)),
        "source_sha256": next(iter(source_hashes)),
        "cells": cells,
    }


paired_contrasts = paired_diag_minus_standard


def _write_csv(
    path: str, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    root: str,
    *,
    expected_source_sha: str,
    run_output: str,
    group_output: str,
    paired_output: str,
    conditions: Sequence[str] = CONDITIONS,
    lr_schedules: Sequence[str] = LR_SCHEDULES,
    initial_learning_rates: Sequence[float] = INITIAL_LEARNING_RATES,
    seeds: Sequence[int] = SEEDS,
    expected_iterations: int = EXPECTED_ITERATIONS,
    validate_run_names: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(conditions) != 2 or set(conditions) != set(CONDITIONS):
        raise ValueError("summarization requires both standard_es and diag_curvature")
    rows = validate_and_collect(
        root,
        conditions=conditions,
        lr_schedules=lr_schedules,
        initial_learning_rates=initial_learning_rates,
        seeds=seeds,
        expected_iterations=expected_iterations,
        expected_source_sha=expected_source_sha,
        validate_run_names=validate_run_names,
    )
    groups = aggregate(rows)
    paired = paired_diag_minus_standard(rows)
    _write_csv(run_output, rows, RUN_FIELDS)
    _write_csv(group_output, groups, GROUP_FIELDS)
    os.makedirs(os.path.dirname(os.path.abspath(paired_output)), exist_ok=True)
    with open(paired_output, "w", encoding="utf-8") as stream:
        json.dump(paired, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return rows, groups, paired


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Result root containing the 80-run matrix.")
    parser.add_argument(
        "--expected-source-sha",
        required=True,
        help="Exact lowercase source_sha256 recorded by the locked launcher.",
    )
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--group-output", "--summary-output", dest="group_output", default=None)
    parser.add_argument("--paired-output", "--contrast-output", dest="paired_output", default=None)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--lr-schedules", nargs="+", default=list(LR_SCHEDULES))
    parser.add_argument(
        "--initial-learning-rates",
        nargs="+",
        type=float,
        default=list(INITIAL_LEARNING_RATES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument(
        "--expected-iterations", type=int, default=EXPECTED_ITERATIONS
    )
    parser.add_argument(
        "--skip-run-name-check",
        action="store_true",
        help="Allow copied/renamed run directories while retaining content validation.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_output = args.run_output or os.path.join(args.root, "validated_runs.csv")
    group_output = args.group_output or os.path.join(args.root, "validated_groups.csv")
    paired_output = args.paired_output or os.path.join(
        args.root, "paired_diag_minus_standard.json"
    )
    try:
        rows, groups, _ = summarize(
            args.root,
            expected_source_sha=args.expected_source_sha,
            run_output=run_output,
            group_output=group_output,
            paired_output=paired_output,
            conditions=args.conditions,
            lr_schedules=args.lr_schedules,
            initial_learning_rates=args.initial_learning_rates,
            seeds=args.seeds,
            expected_iterations=args.expected_iterations,
            validate_run_names=not args.skip_run_name_check,
        )
    except HessianSweepValidationError as error:
        for issue in error.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 2
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        f"Validated {len(rows)} runs and {len(groups)} groups; wrote "
        f"{run_output}, {group_output}, and {paired_output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

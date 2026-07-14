#!/usr/bin/env python3
"""Validate and summarize the no-replay, no-trust Hopper diagnostic sweep."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_CONDITIONS = (
    "standard_es",
    "scalar_damped_es",
    "diag_curvature_raw",
    "diag_curvature_matched_rank",
)
DEFAULT_INITIAL_LEARNING_RATES = (0.25, 1.0, 3.0, 10.0, 30.0)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_EXPECTED_ITERATIONS = 500
DEFAULT_AUC_TRAIN_STEP_BUDGET = 75_000
DEFAULT_EXPECTED_ENV = "Hopper-v5"
DEFAULT_EXPECTED_SCHEDULE = "inverse_sqrt"
DEFAULT_OBS_NORM_CALIBRATION_EPISODES = 3

EXPECTED_COMMON_CONFIG = {
    "population_size": 200,
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
    "common_rollout_seed": True,
    "replay_enabled": False,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "buffer_sampling": "random",
    "min_importance_weight": 0.001,
    "max_importance_weight": 10.0,
    "max_sample_age": 3,
    "ess_min_ratio": 0.2,
    "min_replay_weight_mass": 0.01,
    "scalar_damping": 0.1,
    "curvature_mode": "diag",
    "curvature_beta": 0.99,
    "curvature_clip": 1000.0,
    "evaluate_center_fitness": False,
    "use_leave_one_out_curvature_baseline": True,
    "bias_correct_curvature_ema": True,
    "max_grad_norm": 0.0,
    "max_param_norm": None,
}

EXPECTED_RESOLVED_COMMON = {
    "population_size": 200,
    "noise_std": 0.02,
    "rank_fitness": True,
    "l2_coeff": 0.0,
    "antithetic": True,
    "max_grad_norm": 0.0,
    "max_param_norm": None,
    "trust_region": False,
    "replay_enabled": False,
}

EXPECTED_DIIWES_RESOLVED = {
    "reuse_fraction": 0.0,
    "buffer_size": 0,
    "buffer_sampling": "random",
    "min_importance_weight": 0.001,
    "max_importance_weight": 10.0,
    "max_sample_age": 3,
    "ess_min_ratio": 0.2,
    "min_replay_weight_mass": 0.01,
    "scalar_damping": 0.1,
    "curvature_mode": "diag",
    "curvature_beta": 0.99,
    "curvature_clip": 1000.0,
    "use_leave_one_out_curvature_baseline": True,
    "bias_correct_curvature_ema": True,
    "solver_type": "projected_diagonal_closed_form",
}

EXPECTED_OPTIONAL_CONFIG_DEFAULTS = {
    "env_kwargs": None,
    "frame_stack": 1,
    "fire_reset": False,
    "fire_reset_steps": None,
    "fire_on_life_loss": False,
    "action_indices": None,
    "obs_scale": 1.0,
}

ALLOWED_CONFIG_KEYS = set(EXPECTED_COMMON_CONFIG) | set(EXPECTED_OPTIONAL_CONFIG_DEFAULTS) | {
    "_config_path",
    "algorithm",
    "condition",
    "curvature_fitness",
    "env_name",
    "learning_rate",
    "lr_schedule",
    "n_iterations",
    "obs_norm_calibration_episodes",
    "obs_norm_mode",
    "provenance",
    "resolved_optimizer",
    "seed",
    "use_curvature",
    "use_obs_norm",
}

RUN_FIELDS = [
    "env",
    "condition",
    "seed",
    "initial_learning_rate",
    "lr_schedule",
    "iterations",
    "final_training_env_steps",
    "auc_training_env_step_budget",
    "training_env_step_auc",
    "return_at_training_env_step_budget",
    "initial_return",
    "final_return_after_updates",
    "last_10_return",
    "best_return",
    "mean_step_norm",
    "mean_curvature_clip_frac",
    "mean_replay_weight_mass",
    "replay_use_fraction",
    "replay_mass_rejection_fraction",
    "replay_ess_rejection_fraction",
    "mean_signed_nonpositive_diagonal_frac",
    "median_signed_condition_estimate",
    "signed_system_positive_fraction",
    "solve_failure_count",
    "run_dir",
]

SUMMARY_FIELDS = [
    "env",
    "condition",
    "initial_learning_rate",
    "lr_schedule",
    "runs",
    "auc_training_env_step_budget",
    "training_env_step_auc_mean",
    "training_env_step_auc_std",
    "return_at_training_env_step_budget_mean",
    "return_at_training_env_step_budget_std",
    "initial_return_mean",
    "initial_return_std",
    "final_return_after_updates_mean",
    "final_return_after_updates_std",
    "last_10_return_mean",
    "last_10_return_std",
    "best_return_mean",
    "best_return_std",
    "replay_use_fraction_mean",
    "replay_mass_rejection_fraction_mean",
    "replay_ess_rejection_fraction_mean",
    "mean_signed_nonpositive_diagonal_frac_mean",
    "median_signed_condition_estimate_mean",
    "signed_system_positive_fraction_mean",
    "solve_failure_count",
]


class SweepValidationError(ValueError):
    """Raised when a result root does not contain the complete expected sweep."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__(f"sweep validation failed with {len(self.issues)} issue(s)")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.mean(values)) if values else None


def _median(records: list[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.median(values)) if values else None


def _training_env_step_auc(records: Sequence[dict[str, Any]], budget: int) -> float:
    """Average return on a fixed training-step interval, truncated at ``budget``."""
    if budget <= 0:
        raise ValueError("AUC training-step budget must be positive")
    if not records:
        raise ValueError("cannot compute AUC from empty history")

    x = np.asarray([record["training_env_steps"] for record in records], dtype=np.float64)
    y = np.asarray([record["eval_reward"] for record in records], dtype=np.float64)
    initial_values = np.asarray(
        [record["initial_eval_reward"] for record in records], dtype=np.float64
    )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("AUC inputs must be finite")
    if not np.all(np.isfinite(initial_values)) or not np.allclose(
        initial_values, initial_values[0], rtol=0.0, atol=0.0
    ):
        raise ValueError("initial_eval_reward must be finite and constant within a run")
    if np.any(x < 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("training_env_steps must be nonnegative and strictly increasing")
    if x[-1] < budget:
        raise ValueError(
            f"history ends at {int(x[-1])} training steps, below AUC budget {budget}"
        )

    x = np.concatenate(([0.0], x))
    y = np.concatenate(([initial_values[0]], y))

    below = x < float(budget)
    x_cut = x[below]
    y_cut = y[below]
    y_at_budget = float(np.interp(float(budget), x, y))
    x_cut = np.concatenate((x_cut, [float(budget)]))
    y_cut = np.concatenate((y_cut, [y_at_budget]))
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y_cut, x_cut) / float(budget))


def _return_at_training_env_step_budget(
    records: Sequence[dict[str, Any]], budget: int
) -> float:
    if not records:
        raise ValueError("cannot interpolate return from empty history")
    x = np.asarray([0.0] + [float(record["training_env_steps"]) for record in records])
    y = np.asarray(
        [float(records[0]["initial_eval_reward"])]
        + [float(record["eval_reward"]) for record in records]
    )
    if budget <= 0 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("return-at-budget inputs must be finite and budget must be positive")
    if np.any(np.diff(x) <= 0.0) or x[-1] < budget:
        raise ValueError("history does not provide increasing coverage through the budget")
    return float(np.interp(float(budget), x, y))


def _candidate_run_dirs(root: str) -> list[str]:
    directories: set[str] = set()
    for filename in ("config.json", "history.json", "status.json"):
        pattern = os.path.join(root, "**", filename)
        directories.update(os.path.dirname(path) for path in glob.glob(pattern, recursive=True))
    return sorted(directories)


def _configured_schedule(config: dict[str, Any]) -> str:
    default = "exponential" if "lr_decay" in config else "constant"
    return str(config.get("lr_schedule", default)).lower()


def _parse_seed(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("seed cannot be boolean")
    seed = int(value)
    if float(value) != float(seed):
        raise ValueError("seed must be an integer")
    return seed


def _match_learning_rate(value: Any, expected: Sequence[float]) -> float | None:
    try:
        learning_rate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(learning_rate):
        return None
    for candidate in expected:
        if np.isclose(learning_rate, float(candidate), rtol=1e-12, atol=1e-12):
            return float(candidate)
    return None


def _protocol_value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, float):
        try:
            actual_float = float(actual)
        except (TypeError, ValueError):
            return False
        return bool(
            np.isfinite(actual_float)
            and np.isclose(actual_float, expected, rtol=1e-12, atol=1e-12)
        )
    return actual == expected


def _mapping_protocol_issues(
    mapping: dict[str, Any], expected: dict[str, Any], *, run_dir: str, label: str
) -> list[str]:
    return [
        f"{run_dir}: {label}.{key}={mapping.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if not _protocol_value_matches(mapping.get(key), value)
    ]


def _cell_text(cell: tuple[str, float, int]) -> str:
    condition, learning_rate, seed = cell
    return f"condition={condition}, alpha0={learning_rate:g}, seed={seed}"


def _nonfinite_history_fields(history: Sequence[Any]) -> list[str]:
    failures: list[str] = []
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            continue
        for field, value in record.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not np.isfinite(float(value)):
                failures.append(f"history[{index}].{field}")
    return failures


def _validate_history(
    history: Any,
    *,
    run_dir: str,
    expected_iterations: int,
    auc_train_step_budget: int,
    initial_learning_rate: float | None,
    require_projected_solve: bool,
    require_signed_diagnostics: bool,
) -> list[str]:
    issues: list[str] = []
    if not isinstance(history, list):
        return [f"{run_dir}: history.json is not a list"]
    if len(history) != expected_iterations:
        issues.append(
            f"{run_dir}: incomplete history: expected {expected_iterations} iterations, "
            f"found {len(history)}"
        )

    nonfinite = _nonfinite_history_fields(history)
    if nonfinite:
        preview = ", ".join(nonfinite[:5])
        suffix = " ..." if len(nonfinite) > 5 else ""
        issues.append(f"{run_dir}: nonfinite history value(s): {preview}{suffix}")

    training_steps: list[float] = []
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            issues.append(f"{run_dir}: history[{index}] is not an object")
            continue
        if record.get("iteration") != index:
            issues.append(
                f"{run_dir}: history[{index}].iteration is {record.get('iteration')!r}, expected {index}"
            )
        for field in ("eval_reward", "training_env_steps", "initial_eval_reward"):
            if field not in record:
                issues.append(f"{run_dir}: history[{index}] is missing {field}")
        if record.get("n_reused", 0) != 0:
            issues.append(f"{run_dir}: history[{index}] reused samples in a no-replay run")
        if record.get("n_fresh") != EXPECTED_COMMON_CONFIG["population_size"]:
            issues.append(
                f"{run_dir}: history[{index}].n_fresh={record.get('n_fresh')!r}, "
                f"expected {EXPECTED_COMMON_CONFIG['population_size']}"
            )
        if record.get("used_replay", False) is not False:
            issues.append(f"{run_dir}: history[{index}] reports active replay")
        for field in (
            "replay_weight_mass",
            "n_replay_candidates",
            "n_replay_overlapping",
            "buffer_size",
        ):
            try:
                replay_value = float(record.get(field, 0.0))
            except (TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}].{field} is not numeric")
            else:
                if replay_value != 0.0:
                    issues.append(
                        f"{run_dir}: history[{index}].{field}={replay_value!r} in a no-replay run"
                    )
        if initial_learning_rate is not None:
            expected_lr = float(initial_learning_rate / np.sqrt(index + 1.0))
            try:
                realized_lr = float(record["lr"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}] has no numeric lr")
            else:
                if not np.isclose(realized_lr, expected_lr, rtol=1e-12, atol=1e-12):
                    issues.append(
                        f"{run_dir}: history[{index}].lr={realized_lr!r}, "
                        f"expected {expected_lr!r}"
                    )
        if require_projected_solve:
            if record.get("solve_success") is not True:
                issues.append(f"{run_dir}: history[{index}] lacks a successful projected solve")
            if record.get("solver_type") != "projected_diagonal_closed_form":
                issues.append(f"{run_dir}: history[{index}] has the wrong solver_type")
            try:
                residual = float(record["linear_relative_residual"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}] lacks a numeric solve residual")
            else:
                if not np.isfinite(residual) or residual > 1e-10:
                    issues.append(
                        f"{run_dir}: history[{index}] projected solve residual is {residual!r}"
                    )
        if require_signed_diagnostics:
            for field in (
                "signed_linear_diagonal_min",
                "signed_linear_min_abs_diagonal",
                "signed_linear_condition_estimate",
                "signed_linear_nonpositive_diagonal_frac",
                "signed_system_finite",
                "signed_system_invertible",
                "signed_system_positive",
            ):
                if field not in record:
                    issues.append(
                        f"{run_dir}: history[{index}] is missing signed diagnostic {field}"
                    )
        if record.get("parameter_projection_active") is True:
            issues.append(f"{run_dir}: history[{index}] reports an active parameter projection")
        forbidden_fields = {
            "trust_active",
            "trust_scale",
            "pre_trust_step_norm",
            "no_curv_pre_trust_step_norm",
        }
        present_forbidden = sorted(forbidden_fields.intersection(record))
        if present_forbidden:
            issues.append(
                f"{run_dir}: history[{index}] contains retired trust field(s): "
                f"{', '.join(present_forbidden)}"
            )
        if "training_env_steps" in record:
            try:
                training_steps.append(float(record["training_env_steps"]))
            except (TypeError, ValueError):
                issues.append(
                    f"{run_dir}: history[{index}].training_env_steps is not numeric"
                )

    if len(training_steps) == len(history) and training_steps:
        steps = np.asarray(training_steps, dtype=np.float64)
        if np.all(np.isfinite(steps)):
            if np.any(steps < 0.0) or np.any(np.diff(steps) <= 0.0):
                issues.append(f"{run_dir}: training_env_steps is not strictly increasing")
            if steps[-1] < auc_train_step_budget:
                issues.append(
                    f"{run_dir}: insufficient training-step coverage: ends at "
                    f"{int(steps[-1])}, below AUC budget {auc_train_step_budget}"
                )
    return issues


def _row_from_run(
    config: dict[str, Any],
    history: list[dict[str, Any]],
    run_dir: str,
    learning_rate: float,
    auc_train_step_budget: int,
) -> dict[str, Any]:
    final_window = history[-min(10, len(history)) :]
    return {
        "env": config["env_name"],
        "condition": config["condition"],
        "seed": _parse_seed(config["seed"]),
        "initial_learning_rate": learning_rate,
        "lr_schedule": _configured_schedule(config),
        "iterations": len(history),
        "final_training_env_steps": int(history[-1]["training_env_steps"]),
        "auc_training_env_step_budget": int(auc_train_step_budget),
        "training_env_step_auc": _training_env_step_auc(history, auc_train_step_budget),
        "return_at_training_env_step_budget": _return_at_training_env_step_budget(
            history, auc_train_step_budget
        ),
        "initial_return": float(history[0]["initial_eval_reward"]),
        "final_return_after_updates": float(history[-1]["eval_reward"]),
        "last_10_return": float(np.mean([record["eval_reward"] for record in final_window])),
        "best_return": float(
            max(
                float(history[0]["initial_eval_reward"]),
                max(record["eval_reward"] for record in history),
            )
        ),
        "mean_step_norm": _mean(history, "step_norm"),
        "mean_curvature_clip_frac": _mean(history, "curvature_clip_frac"),
        "mean_replay_weight_mass": _mean(history, "replay_weight_mass"),
        "replay_use_fraction": _mean(history, "used_replay"),
        "replay_mass_rejection_fraction": _mean(history, "replay_mass_rejected"),
        "replay_ess_rejection_fraction": _mean(history, "replay_ess_rejected"),
        "mean_signed_nonpositive_diagonal_frac": _mean(
            history, "signed_linear_nonpositive_diagonal_frac"
        ),
        "median_signed_condition_estimate": _median(
            history, "signed_linear_condition_estimate"
        ),
        "signed_system_positive_fraction": _mean(history, "signed_system_positive"),
        "solve_failure_count": sum(record.get("solve_success") is False for record in history),
        "run_dir": run_dir,
    }


def _resolved_optimizer_issues(
    config: dict[str, Any],
    *,
    condition: str,
    learning_rate: float | None,
    run_dir: str,
) -> list[str]:
    resolved = config.get("resolved_optimizer")
    if not isinstance(resolved, dict):
        return [f"{run_dir}: missing resolved_optimizer certification"]

    issues: list[str] = []
    issues.extend(
        _mapping_protocol_issues(
            resolved,
            EXPECTED_RESOLVED_COMMON,
            run_dir=run_dir,
            label="resolved_optimizer",
        )
    )
    if learning_rate is not None and _match_learning_rate(
        resolved.get("initial_learning_rate"), [learning_rate]
    ) is None:
        issues.append(f"{run_dir}: resolved initial learning rate does not match config")

    if condition == "standard_es":
        if resolved.get("type") != "StandardES":
            issues.append(f"{run_dir}: standard_es did not resolve to StandardES")
    elif condition in {"no_curvature", "scalar_damped_es"}:
        if resolved.get("type") != "DIIWES" or resolved.get("use_curvature") is not False:
            issues.append(f"{run_dir}: scalar-damped control did not resolve to curvature-disabled DIIWES")
    elif condition in {"diag_curvature", "diag_curvature_raw"}:
        if (
            resolved.get("type") != "DIIWES"
            or resolved.get("use_curvature") is not True
            or resolved.get("curvature_mode") != "diag"
            or resolved.get("curvature_fitness") != "raw"
            or resolved.get("solver_type") != "projected_diagonal_closed_form"
        ):
            issues.append(f"{run_dir}: diag_curvature resolved optimizer violates the protocol")
    elif condition == "diag_curvature_matched_rank":
        if (
            resolved.get("type") != "DIIWES"
            or resolved.get("use_curvature") is not True
            or resolved.get("curvature_mode") != "diag"
            or resolved.get("curvature_fitness") != "matched"
            or resolved.get("solver_type") != "projected_diagonal_closed_form"
        ):
            issues.append(
                f"{run_dir}: matched-rank curvature resolved optimizer violates the protocol"
            )
    if resolved.get("type") == "DIIWES":
        issues.extend(
            _mapping_protocol_issues(
                resolved,
                EXPECTED_DIIWES_RESOLVED,
                run_dir=run_dir,
                label="resolved_optimizer",
            )
        )
    return issues


def validate_and_collect_runs(
    root: str,
    *,
    conditions: Sequence[str] = DEFAULT_CONDITIONS,
    initial_learning_rates: Sequence[float] = DEFAULT_INITIAL_LEARNING_RATES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_iterations: int = DEFAULT_EXPECTED_ITERATIONS,
    auc_train_step_budget: int = DEFAULT_AUC_TRAIN_STEP_BUDGET,
    expected_env: str = DEFAULT_EXPECTED_ENV,
    expected_schedule: str = DEFAULT_EXPECTED_SCHEDULE,
    obs_norm_calibration_episodes: int = DEFAULT_OBS_NORM_CALIBRATION_EPISODES,
) -> list[dict[str, Any]]:
    """Return validated rows or raise with every detected protocol violation."""
    conditions = tuple(str(value) for value in conditions)
    initial_learning_rates = tuple(float(value) for value in initial_learning_rates)
    seeds = tuple(int(value) for value in seeds)
    if not conditions or not initial_learning_rates or not seeds:
        raise ValueError("expected condition, learning-rate, and seed sets must be nonempty")
    if expected_iterations <= 0 or auc_train_step_budget <= 0:
        raise ValueError("expected iterations and AUC training-step budget must be positive")
    if len(set(conditions)) != len(conditions):
        raise ValueError("expected conditions contain duplicates")
    if len(set(initial_learning_rates)) != len(initial_learning_rates):
        raise ValueError("expected initial learning rates contain duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("expected seeds contain duplicates")

    expected_cells = {
        (condition, learning_rate, seed)
        for condition in conditions
        for learning_rate in initial_learning_rates
        for seed in seeds
    }
    candidates_by_cell: dict[tuple[str, float, int], list[str]] = defaultdict(list)
    valid_rows_by_dir: dict[str, dict[str, Any]] = {}
    source_hashes_by_dir: dict[str, str] = {}
    issues: list[str] = []

    for run_dir in _candidate_run_dirs(root):
        config_path = os.path.join(run_dir, "config.json")
        history_path = os.path.join(run_dir, "history.json")
        status_path = os.path.join(run_dir, "status.json")
        if not os.path.exists(config_path):
            issues.append(f"{run_dir}: result artifacts exist without config.json")
            continue
        try:
            config = _read_json(config_path)
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"{run_dir}: cannot read config.json: {error}")
            continue
        if not isinstance(config, dict):
            issues.append(f"{run_dir}: config.json is not an object")
            continue

        condition = str(config.get("condition", config.get("algorithm", "")))
        learning_rate = _match_learning_rate(config.get("learning_rate"), initial_learning_rates)
        try:
            seed = _parse_seed(config.get("seed"))
        except (TypeError, ValueError) as error:
            seed = None
            issues.append(f"{run_dir}: invalid seed: {error}")

        metadata_valid = True
        if str(config.get("env_name", "")) != expected_env:
            issues.append(
                f"{run_dir}: env_name={config.get('env_name')!r}, expected {expected_env!r}"
            )
            metadata_valid = False
        if condition not in conditions:
            issues.append(f"{run_dir}: unexpected condition {condition!r}")
            metadata_valid = False
        if learning_rate is None:
            issues.append(
                f"{run_dir}: unexpected initial learning rate {config.get('learning_rate')!r}"
            )
            metadata_valid = False
        schedule = _configured_schedule(config)
        if schedule != expected_schedule:
            issues.append(
                f"{run_dir}: lr_schedule={schedule!r}, expected {expected_schedule!r}"
            )
            metadata_valid = False
        if seed not in seeds:
            if seed is not None:
                issues.append(f"{run_dir}: unexpected seed {seed}")
            metadata_valid = False
        try:
            configured_iterations = int(config.get("n_iterations"))
        except (TypeError, ValueError):
            configured_iterations = None
        if configured_iterations != expected_iterations:
            issues.append(
                f"{run_dir}: n_iterations={config.get('n_iterations')!r}, "
                f"expected {expected_iterations}"
            )
            metadata_valid = False
        common_config_issues = _mapping_protocol_issues(
            config,
            EXPECTED_COMMON_CONFIG,
            run_dir=run_dir,
            label="config",
        )
        if common_config_issues:
            issues.extend(common_config_issues)
            metadata_valid = False
        optional_config_issues = [
            f"{run_dir}: config.{key}={config.get(key)!r}, expected default {expected!r}"
            for key, expected in EXPECTED_OPTIONAL_CONFIG_DEFAULTS.items()
            if not _protocol_value_matches(config.get(key, expected), expected)
        ]
        unexpected_config_keys = sorted(set(config) - ALLOWED_CONFIG_KEYS)
        if unexpected_config_keys:
            optional_config_issues.append(
                f"{run_dir}: unexpected config key(s): {', '.join(unexpected_config_keys)}"
            )
        if optional_config_issues:
            issues.extend(optional_config_issues)
            metadata_valid = False
        if config.get("use_obs_norm") is not True:
            issues.append(f"{run_dir}: use_obs_norm must be true for the Hopper protocol")
            metadata_valid = False
        if config.get("obs_norm_mode") != "frozen_after_calibration":
            issues.append(f"{run_dir}: observation normalization must be frozen after calibration")
            metadata_valid = False
        if config.get("obs_norm_calibration_episodes") != obs_norm_calibration_episodes:
            issues.append(
                f"{run_dir}: obs_norm_calibration_episodes="
                f"{config.get('obs_norm_calibration_episodes')!r}, "
                f"expected {obs_norm_calibration_episodes}"
            )
            metadata_valid = False
        if any(key in config for key in ("trust_radius", "use_trust_radius_for_standard_es")):
            issues.append(f"{run_dir}: trust-region configuration is not allowed")
            metadata_valid = False
        resolved_issues = _resolved_optimizer_issues(
            config,
            condition=condition,
            learning_rate=learning_rate,
            run_dir=run_dir,
        )
        if resolved_issues:
            issues.extend(resolved_issues)
            metadata_valid = False
        provenance = config.get("provenance")
        source_hash = provenance.get("source_sha256") if isinstance(provenance, dict) else None
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            issues.append(f"{run_dir}: missing or invalid provenance source_sha256")
            metadata_valid = False
        else:
            source_hashes_by_dir[run_dir] = source_hash

        cell: tuple[str, float, int] | None = None
        if condition in conditions and learning_rate is not None and seed in seeds:
            cell = (condition, learning_rate, int(seed))
            candidates_by_cell[cell].append(run_dir)

        history: Any = None
        if not os.path.exists(history_path):
            issues.append(f"{run_dir}: missing history.json")
        else:
            try:
                history = _read_json(history_path)
            except (OSError, json.JSONDecodeError) as error:
                issues.append(f"{run_dir}: cannot read history.json: {error}")

        status: Any = None
        if not os.path.exists(status_path):
            issues.append(f"{run_dir}: missing status.json")
        else:
            try:
                status = _read_json(status_path)
            except (OSError, json.JSONDecodeError) as error:
                issues.append(f"{run_dir}: cannot read status.json: {error}")
        if status is not None:
            if not isinstance(status, dict):
                issues.append(f"{run_dir}: status.json is not an object")
            else:
                if status.get("status") != "complete":
                    issues.append(
                        f"{run_dir}: run status is {status.get('status')!r}, expected 'complete'"
                    )
                if status.get("expected_iterations") != expected_iterations:
                    issues.append(
                        f"{run_dir}: status expected_iterations={status.get('expected_iterations')!r}, "
                        f"expected {expected_iterations}"
                    )
                if status.get("completed_iterations") != expected_iterations:
                    issues.append(
                        f"{run_dir}: status completed_iterations={status.get('completed_iterations')!r}, "
                        f"expected {expected_iterations}"
                    )

        history_issues: list[str] = []
        if history is not None:
            history_issues = _validate_history(
                history,
                run_dir=run_dir,
                expected_iterations=expected_iterations,
                auc_train_step_budget=auc_train_step_budget,
                initial_learning_rate=learning_rate,
                require_projected_solve=condition != "standard_es",
                require_signed_diagnostics=condition
                in {"diag_curvature", "diag_curvature_raw", "diag_curvature_matched_rank"},
            )
            issues.extend(history_issues)

        status_valid = (
            isinstance(status, dict)
            and status.get("status") == "complete"
            and status.get("expected_iterations") == expected_iterations
            and status.get("completed_iterations") == expected_iterations
        )
        if (
            metadata_valid
            and cell is not None
            and status_valid
            and isinstance(history, list)
            and not history_issues
        ):
            try:
                valid_rows_by_dir[run_dir] = _row_from_run(
                    config,
                    history,
                    run_dir,
                    learning_rate,
                    auc_train_step_budget,
                )
            except (KeyError, TypeError, ValueError) as error:
                issues.append(f"{run_dir}: cannot summarize validated history: {error}")

    for cell in sorted(expected_cells):
        candidates = candidates_by_cell.get(cell, [])
        if not candidates:
            issues.append(f"missing run: {_cell_text(cell)}")
        elif len(candidates) > 1:
            issues.append(
                f"duplicate runs for {_cell_text(cell)}: {', '.join(sorted(candidates))}"
            )

    unique_source_hashes = set(source_hashes_by_dir.values())
    if len(unique_source_hashes) > 1:
        issues.append(
            f"source mismatch: found {len(unique_source_hashes)} distinct provenance hashes"
        )

    for seed in seeds:
        seed_rows = [row for row in valid_rows_by_dir.values() if row["seed"] == seed]
        if seed_rows:
            initial_returns = np.asarray(
                [row["initial_return"] for row in seed_rows], dtype=np.float64
            )
            if not np.allclose(initial_returns, initial_returns[0], rtol=0.0, atol=1e-10):
                issues.append(
                    f"seed={seed}: initial policy return differs across experiment cells"
                )

    if issues:
        raise SweepValidationError(sorted(set(issues)))

    rows = [valid_rows_by_dir[candidates_by_cell[cell][0]] for cell in sorted(expected_cells)]
    return rows


def aggregate_runs(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["env"]),
            str(row["condition"]),
            float(row["initial_learning_rate"]),
            str(row["lr_schedule"]),
        )
        grouped[key].append(row)

    summaries = []
    for (env, condition, initial_lr, schedule), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "env": env,
            "condition": condition,
            "initial_learning_rate": initial_lr,
            "lr_schedule": schedule,
            "runs": len(group),
            "auc_training_env_step_budget": int(group[0]["auc_training_env_step_budget"]),
            "solve_failure_count": sum(int(row["solve_failure_count"]) for row in group),
        }
        for metric in (
            "training_env_step_auc",
            "return_at_training_env_step_budget",
            "initial_return",
            "final_return_after_updates",
            "last_10_return",
            "best_return",
        ):
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))
        for metric in (
            "replay_use_fraction",
            "replay_mass_rejection_fraction",
            "replay_ess_rejection_fraction",
            "mean_signed_nonpositive_diagonal_frac",
            "median_signed_condition_estimate",
            "signed_system_positive_fraction",
        ):
            values = np.asarray(
                [row[metric] for row in group if row[metric] is not None],
                dtype=np.float64,
            )
            summary[f"{metric}_mean"] = float(np.mean(values)) if len(values) else None
        summaries.append(summary)
    return summaries


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Result root containing exactly the expected run matrix.")
    parser.add_argument("--run-output", default="plots/no_replay_no_trust_run_summary.csv")
    parser.add_argument("--summary-output", default="plots/no_replay_no_trust_group_summary.csv")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument(
        "--initial-learning-rates",
        nargs="+",
        type=float,
        default=list(DEFAULT_INITIAL_LEARNING_RATES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--expected-iterations", type=int, default=DEFAULT_EXPECTED_ITERATIONS)
    parser.add_argument(
        "--auc-train-step-budget",
        type=int,
        default=DEFAULT_AUC_TRAIN_STEP_BUDGET,
    )
    parser.add_argument("--expected-env", default=DEFAULT_EXPECTED_ENV)
    parser.add_argument("--expected-schedule", default=DEFAULT_EXPECTED_SCHEDULE)
    parser.add_argument(
        "--obs-norm-calibration-episodes",
        type=int,
        default=DEFAULT_OBS_NORM_CALIBRATION_EPISODES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        runs = validate_and_collect_runs(
            args.root,
            conditions=args.conditions,
            initial_learning_rates=args.initial_learning_rates,
            seeds=args.seeds,
            expected_iterations=args.expected_iterations,
            auc_train_step_budget=args.auc_train_step_budget,
            expected_env=args.expected_env,
            expected_schedule=args.expected_schedule,
            obs_norm_calibration_episodes=args.obs_norm_calibration_episodes,
        )
    except SweepValidationError as error:
        print(str(error), file=sys.stderr)
        for issue in error.issues:
            print(f"- {issue}", file=sys.stderr)
        return 2

    summaries = aggregate_runs(runs)
    _write_csv(args.run_output, RUN_FIELDS, runs)
    _write_csv(args.summary_output, SUMMARY_FIELDS, summaries)
    print(f"Validated and wrote {len(runs)} runs to {args.run_output}")
    print(f"Wrote {len(summaries)} groups to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

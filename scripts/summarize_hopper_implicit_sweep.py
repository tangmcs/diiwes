#!/usr/bin/env python3
"""Validate the no-Picard Hopper Standard ES versus Hessian sweep."""

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


CONDITIONS = ("standard_es", "linearized_implicit_es")
HESSIAN_FIX_CONDITIONS = (
    "standard_es",
    "linearized_implicit_es",
    "concave_diagonal_curvature_es",
    "concave_block_curvature_es",
    "concave_block_ema_curvature_es",
)
LR_SCHEDULES = ("inverse_sqrt", "inverse_linear")
INITIAL_LEARNING_RATES = (10.0, 30.0)
SEEDS = tuple(range(10))
HESSIAN_FIX_SEEDS = tuple(range(5))
EXPECTED_ITERATIONS = 500
AUC_TRAINING_STEP_BUDGET = 75_000
LEGACY_DIAGNOSTIC_SCHEMA_VERSION = 1
CURRENT_DIAGNOSTIC_SCHEMA_VERSION = 2

COMMON_CONFIG = {
    "env_name": "Hopper-v5",
    "population_size": 200,
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
    "eval_episodes": 3,
    "eval_interval": 1,
    "log_interval": 10,
    "max_episode_steps": 1000,
    "use_obs_norm": True,
    "obs_norm_mode": "frozen_after_calibration",
    "obs_norm_calibration_episodes": 3,
    "replay_enabled": False,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "common_rollout_seed": True,
    "implicit_damping": 0.0,
    "linear_min_abs_diagonal": 1e-12,
    "evaluate_center_fitness": False,
}

ALGORITHMS = {
    "standard_es": "standard_es",
    "linearized_implicit_es": "linearized_implicit_es",
    "concave_diagonal_curvature_es": "concave_curvature_es",
    "concave_block_curvature_es": "concave_curvature_es",
    "concave_block_ema_curvature_es": "concave_curvature_es",
}

ALLOWED_CONFIG_KEYS = set(COMMON_CONFIG) | {
    "_config_path",
    "algorithm",
    "condition",
    "curvature_beta",
    "curvature_fitness",
    "curvature_mode",
    "diagnostic_schema_version",
    "learning_rate",
    "lr_schedule",
    "n_iterations",
    "provenance",
    "resolved_optimizer",
    "seed",
    "use_curvature",
}

RESOLVED_TYPES = {
    "standard_es": ("StandardES", "none"),
    "linearized_implicit_es": (
        "LinearizedImplicitES",
        "signed_diagonal_linearized_implicit",
    ),
    "concave_diagonal_curvature_es": (
        "ConcaveCurvatureES",
        "concave_projected_diag",
    ),
    "concave_block_curvature_es": (
        "ConcaveCurvatureES",
        "concave_projected_block",
    ),
    "concave_block_ema_curvature_es": (
        "ConcaveCurvatureES",
        "concave_projected_block",
    ),
}

RUN_FIELDS = (
    "condition",
    "lr_schedule",
    "initial_learning_rate",
    "seed",
    "iterations",
    "initial_return",
    "return_at_training_step_budget",
    "training_step_auc",
    "final_return",
    "last_10_return",
    "best_return",
    "implicit_convergence_fraction",
    "mean_implicit_relative_residual",
    "mean_hessian_split_correlation",
    "mean_hessian_split_sign_agreement",
    "mean_hessian_split_relative_disagreement",
    "mean_hessian_temporal_correlation",
    "mean_hessian_temporal_sign_agreement",
    "mean_hessian_temporal_relative_disagreement",
    "mean_curvature_projection_frac",
    "mean_curvature_active_frac",
    "max_linear_relative_residual",
    "min_linear_absolute_diagonal",
    "mean_linear_condition_estimate",
    "median_linear_condition_estimate",
    "max_linear_condition_estimate",
    "mean_linear_nonpositive_diagonal_frac",
    "median_step_norm_ratio",
    "max_step_norm_ratio",
    "run_dir",
)

SUMMARY_FIELDS = (
    "condition",
    "lr_schedule",
    "initial_learning_rate",
    "runs",
    "training_step_auc_mean",
    "training_step_auc_std",
    "return_at_training_step_budget_mean",
    "return_at_training_step_budget_std",
    "final_return_mean",
    "final_return_std",
    "last_10_return_mean",
    "last_10_return_std",
    "best_return_mean",
    "best_return_std",
    "implicit_convergence_fraction_mean",
    "mean_implicit_relative_residual_mean",
    "mean_hessian_split_correlation_mean",
    "mean_hessian_split_sign_agreement_mean",
    "mean_hessian_split_relative_disagreement_mean",
    "mean_hessian_temporal_correlation_mean",
    "mean_hessian_temporal_sign_agreement_mean",
    "mean_hessian_temporal_relative_disagreement_mean",
    "mean_curvature_projection_frac_mean",
    "mean_curvature_active_frac_mean",
    "max_linear_relative_residual_mean",
    "min_linear_absolute_diagonal_mean",
    "mean_linear_condition_estimate_mean",
    "median_linear_condition_estimate_mean",
    "max_linear_condition_estimate_mean",
    "mean_linear_nonpositive_diagonal_frac_mean",
    "median_step_norm_ratio_mean",
    "max_step_norm_ratio_mean",
)

PAIRED_PERFORMANCE_FIELDS = (
    "training_step_auc",
    "return_at_training_step_budget",
    "final_return",
    "best_return",
)


class ValidationError(ValueError):
    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__(f"Hessian sweep validation failed with {len(self.issues)} issue(s)")


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_json_lines(path: str) -> list[Any]:
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
        try:
            value = float(actual)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(value) and np.isclose(value, expected, rtol=1e-12, atol=1e-12))
    return actual == expected


def _mean(records: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.mean(values)) if values else None


def _minimum(records: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.min(values)) if values else None


def _maximum(records: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.max(values)) if values else None


def _median(records: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [float(record[field]) for record in records if field in record]
    return float(np.median(values)) if values else None


def _training_step_metrics(
    history: Sequence[dict[str, Any]], budget: int
) -> tuple[float, float]:
    x = np.asarray([0.0] + [float(row["training_env_steps"]) for row in history])
    y = np.asarray(
        [float(history[0]["initial_eval_reward"])]
        + [float(row["eval_reward"]) for row in history]
    )
    if np.any(np.diff(x) <= 0.0) or x[-1] < budget:
        raise ValueError(f"history does not cover the {budget} training-step budget")
    at_budget = float(np.interp(float(budget), x, y))
    below = x < budget
    x_cut = np.concatenate((x[below], [float(budget)]))
    y_cut = np.concatenate((y[below], [at_budget]))
    integrate = getattr(np, "trapezoid", np.trapz)
    auc = float(integrate(y_cut, x_cut) / float(budget))
    return auc, at_budget


def _history_issues(
    history: Any,
    *,
    run_dir: str,
    condition: str,
    learning_rate: float,
    lr_schedule: str,
    population_size: int,
    diagnostic_schema_version: int,
    expected_iterations: int,
    budget: int,
) -> list[str]:
    if not isinstance(history, list):
        return [f"{run_dir}: history.json is not a list"]
    issues: list[str] = []
    if len(history) != expected_iterations:
        issues.append(
            f"{run_dir}: expected {expected_iterations} history records, found {len(history)}"
        )
    forbidden = {
        "trust_active",
        "trust_scale",
        "trust_radius",
        "pre_trust_step_norm",
        "multiplier_floor_frac",
    }
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            issues.append(f"{run_dir}: history[{index}] is not an object")
            continue
        if record.get("iteration") != index:
            issues.append(f"{run_dir}: history[{index}] has the wrong iteration index")
        if lr_schedule == "inverse_sqrt":
            expected_lr = learning_rate / np.sqrt(index + 1.0)
        elif lr_schedule == "inverse_linear":
            expected_lr = learning_rate / (index + 1.0)
        else:
            issues.append(f"{run_dir}: unsupported learning-rate schedule {lr_schedule!r}")
            expected_lr = float("nan")
        try:
            actual_lr = float(record["lr"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] has no numeric learning rate")
        else:
            if not np.isclose(actual_lr, expected_lr, rtol=1e-12, atol=1e-12):
                issues.append(
                    f"{run_dir}: history[{index}] deviates from {lr_schedule} schedule"
                )
        common_required = (
            "n_fresh",
            "n_reused",
            "used_replay",
            "replay_weight_mass",
            "fresh_weight_mass",
            "buffer_size",
            "ess",
            "ess_ratio",
            "ess_normalized",
            "importance_weight_mean",
            "importance_weight_min",
            "importance_weight_max",
            "parameter_projection_active",
        )
        for field in common_required:
            if field not in record:
                issues.append(f"{run_dir}: history[{index}] is missing {field}")
        if (
            record.get("n_fresh") != population_size
            or record.get("n_reused") != 0
        ):
            issues.append(f"{run_dir}: history[{index}] is not a complete fresh population")
        if record.get("used_replay") is not False:
            issues.append(f"{run_dir}: history[{index}] reports replay")
        exact_runtime_values = {
            "replay_weight_mass": 0.0,
            "fresh_weight_mass": 1.0,
            "buffer_size": 0.0,
            "ess": float(population_size),
            "ess_ratio": 1.0,
            "ess_normalized": 1.0,
            "importance_weight_mean": 1.0,
            "importance_weight_min": 1.0,
            "importance_weight_max": 1.0,
        }
        for field, expected_value in exact_runtime_values.items():
            try:
                raw_value = record[field]
                if isinstance(raw_value, bool):
                    raise TypeError
                value = float(raw_value)
            except (TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}].{field} is not numeric")
            except KeyError:
                continue
            else:
                if not np.isfinite(value) or value != expected_value:
                    issues.append(
                        f"{run_dir}: history[{index}].{field}={value!r}, "
                        f"expected {expected_value!r}"
                    )
        for field in ("mean_importance_weight", "max_importance_weight"):
            if field not in record:
                continue
            try:
                raw_value = record[field]
                if isinstance(raw_value, bool):
                    raise TypeError
                value = float(raw_value)
            except (TypeError, ValueError):
                issues.append(f"{run_dir}: history[{index}].{field} is not numeric")
            else:
                if not np.isfinite(value) or value != 1.0:
                    issues.append(
                        f"{run_dir}: history[{index}].{field}={value!r}, expected 1.0"
                    )
        if record.get("parameter_projection_active") is not False:
            issues.append(f"{run_dir}: history[{index}] reports parameter projection")
        present = forbidden.intersection(record)
        if present:
            issues.append(
                f"{run_dir}: history[{index}] contains trust/floor fields: {sorted(present)}"
            )
        for field, value in record.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not np.isfinite(float(value)):
                issues.append(f"{run_dir}: history[{index}].{field} is non-finite")

        if condition == "standard_es":
            if record.get("solver_type") != "none":
                issues.append(f"{run_dir}: Standard ES has a nonstandard solver")
        elif condition == "linearized_implicit_es":
            required = (
                "linear_relative_residual",
                "linear_min_abs_diagonal",
                "linear_condition_estimate",
                "linear_nonpositive_diagonal_frac",
                "hessian_pairs",
                "h_raw_std",
                "h_split_correlation",
                "h_split_sign_agreement",
                "h_temporal_correlation",
                "h_temporal_sign_agreement",
                "step_norm_ratio",
            )
            if record.get("solver_type") != "signed_diagonal_linearized_implicit":
                issues.append(f"{run_dir}: linearized arm has the wrong solver")
            if record.get("solve_success") is not True:
                issues.append(f"{run_dir}: signed diagonal arithmetic solve failed")
            if record.get("hessian_pairs") != 100:
                issues.append(f"{run_dir}: linearized arm did not use 100 Hessian pairs")
            try:
                residual = float(record["linear_relative_residual"])
            except (KeyError, TypeError, ValueError):
                residual = float("inf")
            if not np.isfinite(residual) or residual > 1e-10:
                issues.append(f"{run_dir}: signed diagonal arithmetic residual is invalid")
            try:
                minimum_absolute = float(record["linear_min_abs_diagonal"])
                condition_estimate = float(record["linear_condition_estimate"])
                nonpositive_fraction = float(record["linear_nonpositive_diagonal_frac"])
                split_correlation = float(record["h_split_correlation"])
                split_sign = float(record["h_split_sign_agreement"])
                temporal_correlation = float(record["h_temporal_correlation"])
                temporal_sign = float(record["h_temporal_sign_agreement"])
                step_norm_ratio = float(record["step_norm_ratio"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{run_dir}: signed/Hessian diagnostics are invalid")
            else:
                if minimum_absolute <= 0.0 or condition_estimate < 1.0:
                    issues.append(f"{run_dir}: signed diagonal scale/condition is invalid")
                if not (0.0 <= nonpositive_fraction <= 1.0):
                    issues.append(f"{run_dir}: signed nonpositive fraction is invalid")
                if not (-1.0 <= split_correlation <= 1.0):
                    issues.append(f"{run_dir}: Hessian split correlation is invalid")
                if not (0.0 <= split_sign <= 1.0):
                    issues.append(f"{run_dir}: Hessian split sign agreement is invalid")
                if not (-1.0 <= temporal_correlation <= 1.0):
                    issues.append(f"{run_dir}: Hessian temporal correlation is invalid")
                if not (0.0 <= temporal_sign <= 1.0):
                    issues.append(f"{run_dir}: Hessian temporal sign agreement is invalid")
                if step_norm_ratio < 0.0:
                    issues.append(f"{run_dir}: linearized step ratio is negative")
            for field in required:
                if field not in record:
                    issues.append(f"{run_dir}: linearized history is missing {field}")
        elif condition in {
            "concave_diagonal_curvature_es",
            "concave_block_curvature_es",
            "concave_block_ema_curvature_es",
        }:
            expected_structure = (
                "diag"
                if condition == "concave_diagonal_curvature_es"
                else "block"
            )
            expected_beta = (
                0.9 if condition == "concave_block_ema_curvature_es" else 0.0
            )
            expected_same_generation = expected_beta == 0.0
            expected_components = 5123 if expected_structure == "diag" else 3
            required = (
                "linear_relative_residual",
                "linear_min_abs_diagonal",
                "linear_condition_estimate",
                "linear_nonpositive_diagonal_frac",
                "hessian_pairs",
                "h_raw_std",
                "h_split_correlation",
                "h_split_sign_agreement",
                "h_split_relative_disagreement",
                "h_temporal_correlation",
                "h_temporal_sign_agreement",
                "h_temporal_relative_disagreement",
                "curvature_projection_frac",
                "curvature_clip_frac",
                "curvature_components",
                "curvature_block_size_min",
                "curvature_block_size_max",
                "curvature_beta",
                "hessian_ema_count",
                "step_norm_ratio",
            )
            if diagnostic_schema_version >= CURRENT_DIAGNOSTIC_SCHEMA_VERSION:
                required += ("curvature_same_generation",)
            expected_solver = f"concave_projected_{expected_structure}"
            if record.get("solver_type") != expected_solver:
                issues.append(f"{run_dir}: concave curvature arm has the wrong solver")
            if record.get("solve_success") is not True:
                issues.append(f"{run_dir}: concave curvature arithmetic solve failed")
            if "implicit_converged" in record:
                issues.append(f"{run_dir}: arithmetic solve is mislabeled implicit convergence")
            if record.get("hessian_pairs") != 100:
                issues.append(f"{run_dir}: concave curvature arm did not use 100 pairs")
            if record.get("curvature_mode") != expected_structure:
                issues.append(f"{run_dir}: concave curvature structure is wrong")
            if record.get("curvature_matches_gradient") is not True:
                issues.append(f"{run_dir}: concave curvature does not match the gradient transform")
            if "curvature_same_generation" in record and (
                record.get("curvature_same_generation") is not expected_same_generation
            ):
                issues.append(
                    f"{run_dir}: concave curvature same-generation metadata is wrong"
                )
            try:
                residual = float(record["linear_relative_residual"])
                denominator_min = float(record["linear_min_abs_diagonal"])
                condition_estimate = float(record["linear_condition_estimate"])
                nonpositive_fraction = float(record["linear_nonpositive_diagonal_frac"])
                split_correlation = float(record["h_split_correlation"])
                split_sign = float(record["h_split_sign_agreement"])
                split_disagreement = float(record["h_split_relative_disagreement"])
                temporal_correlation = float(record["h_temporal_correlation"])
                temporal_sign = float(record["h_temporal_sign_agreement"])
                temporal_disagreement = float(record["h_temporal_relative_disagreement"])
                projection_fraction = float(record["curvature_projection_frac"])
                clip_fraction = float(record["curvature_clip_frac"])
                component_count = int(record["curvature_components"])
                beta = float(record["curvature_beta"])
                ema_count = int(record["hessian_ema_count"])
                step_ratio = float(record["step_norm_ratio"])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{run_dir}: concave curvature diagnostics are invalid")
            else:
                if not np.isfinite(residual) or residual > 1e-10:
                    issues.append(f"{run_dir}: concave curvature residual is invalid")
                if denominator_min < 1.0 or condition_estimate < 1.0:
                    issues.append(f"{run_dir}: concave denominator lost positivity")
                if nonpositive_fraction != 0.0:
                    issues.append(f"{run_dir}: concave denominator is nonpositive")
                if not (-1.0 <= split_correlation <= 1.0):
                    issues.append(f"{run_dir}: concave split correlation is invalid")
                if not (0.0 <= split_sign <= 1.0) or split_disagreement < 0.0:
                    issues.append(f"{run_dir}: concave split diagnostics are invalid")
                if not (-1.0 <= temporal_correlation <= 1.0):
                    issues.append(f"{run_dir}: concave temporal correlation is invalid")
                if not (0.0 <= temporal_sign <= 1.0) or temporal_disagreement < 0.0:
                    issues.append(f"{run_dir}: concave temporal diagnostics are invalid")
                if not (0.0 <= projection_fraction <= 1.0) or clip_fraction != 0.0:
                    issues.append(f"{run_dir}: concave projection/clipping diagnostics are invalid")
                if component_count != expected_components:
                    issues.append(f"{run_dir}: concave curvature component count is wrong")
                if not np.isclose(beta, expected_beta, rtol=0.0, atol=0.0):
                    issues.append(f"{run_dir}: concave curvature EMA beta is wrong")
                if ema_count != index + 1:
                    issues.append(f"{run_dir}: concave curvature EMA count is wrong")
                if not (0.0 <= step_ratio <= 1.0 + 1e-10):
                    issues.append(f"{run_dir}: concave step amplifies the explicit step")
            if expected_structure == "diag":
                if record.get("curvature_block_size_min") != 1 or record.get(
                    "curvature_block_size_max"
                ) != 1:
                    issues.append(f"{run_dir}: diagonal curvature blocks are invalid")
            elif record.get("curvature_block_size_min") != 195 or record.get(
                "curvature_block_size_max"
            ) != 4160:
                issues.append(f"{run_dir}: layer-block partition is invalid")
            for field in required:
                if field not in record:
                    issues.append(f"{run_dir}: concave history is missing {field}")
        else:
            issues.append(f"{run_dir}: history uses unsupported condition {condition!r}")

    if history:
        try:
            _training_step_metrics(history, budget)
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{run_dir}: {error}")
    return issues


def _run_row(
    config: dict[str, Any], history: list[dict[str, Any]], run_dir: str, budget: int
) -> dict[str, Any]:
    auc, at_budget = _training_step_metrics(history, budget)
    condition = str(config["condition"])
    initial = float(history[0]["initial_eval_reward"])
    return {
        "condition": condition,
        "lr_schedule": str(config["lr_schedule"]),
        "initial_learning_rate": float(config["learning_rate"]),
        "seed": int(config["seed"]),
        "iterations": len(history),
        "initial_return": initial,
        "return_at_training_step_budget": at_budget,
        "training_step_auc": auc,
        "final_return": float(history[-1]["eval_reward"]),
        "last_10_return": float(np.mean([row["eval_reward"] for row in history[-10:]])),
        "best_return": float(max([initial] + [float(row["eval_reward"]) for row in history])),
        "implicit_convergence_fraction": _mean(history, "implicit_converged"),
        "mean_implicit_relative_residual": _mean(history, "implicit_relative_residual"),
        "mean_hessian_split_correlation": _mean(history, "h_split_correlation"),
        "mean_hessian_split_sign_agreement": _mean(history, "h_split_sign_agreement"),
        "mean_hessian_split_relative_disagreement": _mean(
            history, "h_split_relative_disagreement"
        ),
        "mean_hessian_temporal_correlation": _mean(history, "h_temporal_correlation"),
        "mean_hessian_temporal_sign_agreement": _mean(
            history, "h_temporal_sign_agreement"
        ),
        "mean_hessian_temporal_relative_disagreement": _mean(
            history, "h_temporal_relative_disagreement"
        ),
        "mean_curvature_projection_frac": _mean(
            history, "curvature_projection_frac"
        ),
        "mean_curvature_active_frac": _mean(history, "curvature_active_frac"),
        "max_linear_relative_residual": _maximum(history, "linear_relative_residual"),
        "min_linear_absolute_diagonal": _minimum(history, "linear_min_abs_diagonal"),
        "mean_linear_condition_estimate": _mean(history, "linear_condition_estimate"),
        "median_linear_condition_estimate": _median(
            history, "linear_condition_estimate"
        ),
        "max_linear_condition_estimate": _maximum(
            history, "linear_condition_estimate"
        ),
        "mean_linear_nonpositive_diagonal_frac": _mean(
            history, "linear_nonpositive_diagonal_frac"
        ),
        "median_step_norm_ratio": _median(history, "step_norm_ratio"),
        "max_step_norm_ratio": _maximum(history, "step_norm_ratio"),
        "run_dir": run_dir,
    }


def validate_and_collect(
    root: str,
    *,
    conditions: Sequence[str] = CONDITIONS,
    lr_schedules: Sequence[str] = LR_SCHEDULES,
    learning_rates: Sequence[float] = INITIAL_LEARNING_RATES,
    seeds: Sequence[int] = SEEDS,
    expected_iterations: int = EXPECTED_ITERATIONS,
    budget: int = AUC_TRAINING_STEP_BUDGET,
    expected_source_sha: str | None = None,
) -> list[dict[str, Any]]:
    conditions = tuple(str(value) for value in conditions)
    lr_schedules = tuple(str(value) for value in lr_schedules)
    learning_rates = tuple(float(value) for value in learning_rates)
    seeds = tuple(int(value) for value in seeds)
    if not conditions or not lr_schedules or not learning_rates or not seeds:
        raise ValueError("condition, schedule, learning-rate, and seed grids must be nonempty")
    for label, values in (
        ("conditions", conditions),
        ("learning-rate schedules", lr_schedules),
        ("learning rates", learning_rates),
        ("seeds", seeds),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{label} contain duplicates")
    if expected_source_sha is not None and re.fullmatch(
        r"[0-9a-f]{64}", expected_source_sha
    ) is None:
        raise ValueError("expected_source_sha must be a lowercase SHA-256 digest")
    expected = {
        (str(condition), str(lr_schedule), float(learning_rate), int(seed))
        for condition in conditions
        for lr_schedule in lr_schedules
        for learning_rate in learning_rates
        for seed in seeds
    }
    candidates: dict[tuple[str, str, float, int], list[str]] = defaultdict(list)
    rows: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    array_job_ids: set[str] = set()
    issues: list[str] = []

    run_dirs = sorted(
        {
            os.path.dirname(path)
            for filename in ("config.json", "history.json", "status.json")
            for path in glob.glob(os.path.join(root, "**", filename), recursive=True)
        }
    )
    for run_dir in run_dirs:
        config_path = os.path.join(run_dir, "config.json")
        history_path = os.path.join(run_dir, "history.json")
        history_jsonl_path = os.path.join(run_dir, "history.jsonl")
        status_path = os.path.join(run_dir, "status.json")
        if not all(
            os.path.exists(path)
            for path in (config_path, history_path, history_jsonl_path, status_path)
        ):
            issues.append(f"{run_dir}: missing config, JSON/JSONL history, or status artifact")
            continue
        try:
            config = _read_json(config_path)
            history = _read_json(history_path)
            history_jsonl = _read_json_lines(history_jsonl_path)
            status = _read_json(status_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_dir}: unreadable artifact: {error}")
            continue
        if not isinstance(config, dict) or not isinstance(status, dict):
            issues.append(f"{run_dir}: config/status artifact has the wrong type")
            continue
        if history_jsonl != history:
            issues.append(f"{run_dir}: history.jsonl does not match history.json")
        condition = str(config.get("condition", ""))
        try:
            lr_schedule = str(config["lr_schedule"])
            learning_rate = float(config["learning_rate"])
            seed = int(config["seed"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: invalid condition cell metadata")
            continue
        cell = (condition, lr_schedule, learning_rate, seed)
        if cell in expected:
            candidates[cell].append(run_dir)
        else:
            issues.append(f"{run_dir}: unexpected cell {cell}")

        for key, expected_value in COMMON_CONFIG.items():
            if not _matches(config.get(key), expected_value):
                issues.append(
                    f"{run_dir}: config.{key}={config.get(key)!r}, expected {expected_value!r}"
                )
        unexpected_keys = sorted(set(config) - ALLOWED_CONFIG_KEYS)
        if unexpected_keys:
            issues.append(f"{run_dir}: unexpected config keys: {unexpected_keys}")
        raw_diagnostic_schema_version = config.get(
            "diagnostic_schema_version", LEGACY_DIAGNOSTIC_SCHEMA_VERSION
        )
        if (
            isinstance(raw_diagnostic_schema_version, bool)
            or not isinstance(raw_diagnostic_schema_version, int)
            or raw_diagnostic_schema_version
            not in {
                LEGACY_DIAGNOSTIC_SCHEMA_VERSION,
                CURRENT_DIAGNOSTIC_SCHEMA_VERSION,
            }
        ):
            issues.append(f"{run_dir}: diagnostic schema version is invalid")
            diagnostic_schema_version = LEGACY_DIAGNOSTIC_SCHEMA_VERSION
        else:
            diagnostic_schema_version = raw_diagnostic_schema_version
        if config.get("n_iterations") != expected_iterations:
            issues.append(f"{run_dir}: configured iteration count is not {expected_iterations}")
        if lr_schedule not in lr_schedules:
            issues.append(f"{run_dir}: unexpected learning-rate schedule {lr_schedule!r}")
        if config.get("algorithm") != ALGORITHMS.get(condition):
            issues.append(f"{run_dir}: condition resolved to the wrong algorithm")
        if condition in {
            "concave_diagonal_curvature_es",
            "concave_block_curvature_es",
            "concave_block_ema_curvature_es",
        }:
            expected_structure = (
                "diag"
                if condition == "concave_diagonal_curvature_es"
                else "block"
            )
            expected_beta = (
                0.9 if condition == "concave_block_ema_curvature_es" else 0.0
            )
            if (
                config.get("curvature_fitness") != "matched"
                or config.get("curvature_mode") != expected_structure
                or not _matches(config.get("curvature_beta"), expected_beta)
                or config.get("implicit_damping") != 0.0
            ):
                issues.append(f"{run_dir}: concave condition config is invalid")
        if any(key in config for key in ("trust_radius", "min_step_multiplier")):
            issues.append(f"{run_dir}: retired trust/floor configuration is present")

        resolved = config.get("resolved_optimizer")
        expected_type, expected_solver = RESOLVED_TYPES.get(condition, (None, None))
        if not isinstance(resolved, dict):
            issues.append(f"{run_dir}: missing resolved optimizer record")
        else:
            common_resolved = {
                "type": expected_type,
                "population_size": 200,
                "initial_learning_rate": learning_rate,
                "noise_std": 0.02,
                "rank_fitness": True,
                "l2_coeff": 0.0,
                "antithetic": True,
                "max_grad_norm": 0.0,
                "max_param_norm": None,
                "trust_region": False,
                "replay_enabled": False,
            }
            for key, expected_value in common_resolved.items():
                if not _matches(resolved.get(key), expected_value):
                    issues.append(f"{run_dir}: resolved_optimizer.{key} is invalid")
            if condition != "standard_es":
                if resolved.get("solver_type") != expected_solver:
                    issues.append(f"{run_dir}: resolved optimizer has the wrong solver")
                if resolved.get("implicit_damping") != 0.0:
                    issues.append(f"{run_dir}: implicit damping is nonzero")
            if condition == "linearized_implicit_es" and (
                resolved.get("curvature_fitness") != "matched"
                or resolved.get("curvature_beta") != 0.0
                or resolved.get("curvature_clipping") is not False
            ):
                issues.append(f"{run_dir}: linearized Hessian protocol is invalid")
            if condition in {
                "concave_diagonal_curvature_es",
                "concave_block_curvature_es",
                "concave_block_ema_curvature_es",
            }:
                expected_structure = (
                    "diag"
                    if condition == "concave_diagonal_curvature_es"
                    else "block"
                )
                expected_beta = (
                    0.9
                    if condition == "concave_block_ema_curvature_es"
                    else 0.0
                )
                expected_same_generation = expected_beta == 0.0
                expected_components = 5123 if expected_structure == "diag" else 3
                concave_resolved_invalid = (
                    resolved.get("curvature_fitness") != "matched"
                    or resolved.get("curvature_structure") != expected_structure
                    or resolved.get("curvature_projection") != "concave"
                    or resolved.get("curvature_clipping") is not False
                    or not _matches(resolved.get("curvature_beta"), expected_beta)
                    or resolved.get("curvature_components") != expected_components
                )
                if diagnostic_schema_version >= CURRENT_DIAGNOSTIC_SCHEMA_VERSION:
                    concave_resolved_invalid = (
                        concave_resolved_invalid
                        or resolved.get("curvature_same_generation")
                        is not expected_same_generation
                    )
                elif "curvature_same_generation" in resolved:
                    concave_resolved_invalid = (
                        concave_resolved_invalid
                        or resolved.get("curvature_same_generation")
                        is not expected_same_generation
                    )
                if concave_resolved_invalid:
                    issues.append(f"{run_dir}: concave curvature protocol is invalid")

        provenance = config.get("provenance")
        source_hash = provenance.get("source_sha256") if isinstance(provenance, dict) else None
        locked_hash = (
            provenance.get("expected_source_sha256")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(source_hash, str) or re.fullmatch(r"[0-9a-f]{64}", source_hash) is None:
            issues.append(f"{run_dir}: missing source digest")
        else:
            hashes[run_dir] = source_hash
        if (
            not isinstance(locked_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", locked_hash) is None
            or locked_hash != source_hash
        ):
            issues.append(f"{run_dir}: expected/actual source digest lock is invalid")
        if expected_source_sha is not None and source_hash != expected_source_sha:
            issues.append(f"{run_dir}: source digest differs from requested submission digest")
        dependencies = provenance.get("dependencies") if isinstance(provenance, dict) else None
        if not isinstance(dependencies, dict) or any(
            not isinstance(dependencies.get(name), str)
            for name in ("gymnasium", "mujoco", "PyYAML")
        ):
            issues.append(f"{run_dir}: dependency-version provenance is incomplete")
        if isinstance(provenance, dict):
            array_job_id = provenance.get("slurm_array_job_id")
            task_id_raw = provenance.get("slurm_array_task_id")
        else:
            array_job_id = None
            task_id_raw = None
        if not isinstance(array_job_id, str) or re.fullmatch(r"[0-9]+", array_job_id) is None:
            issues.append(f"{run_dir}: Slurm array job provenance is invalid")
        else:
            array_job_ids.add(array_job_id)
        try:
            task_id = int(task_id_raw)
        except (TypeError, ValueError):
            issues.append(f"{run_dir}: Slurm array task provenance is invalid")
        else:
            if cell in expected:
                expected_task_id = (
                    conditions.index(condition)
                    * len(lr_schedules)
                    * len(learning_rates)
                    * len(seeds)
                    + lr_schedules.index(lr_schedule)
                    * len(learning_rates)
                    * len(seeds)
                    + learning_rates.index(learning_rate) * len(seeds)
                    + seeds.index(seed)
                )
                if task_id != expected_task_id:
                    issues.append(
                        f"{run_dir}: task id {task_id} does not match cell mapping "
                        f"{expected_task_id}"
                    )
                suffix = f"_job{array_job_id}_task{task_id}"
                if not os.path.basename(run_dir).endswith(suffix):
                    issues.append(f"{run_dir}: output directory does not encode job/task mapping")
        if status.get("history_records") != "history.jsonl":
            issues.append(f"{run_dir}: status does not identify history.jsonl")
        if status.get("status") != "complete" or status.get("completed_iterations") != expected_iterations:
            issues.append(f"{run_dir}: run status is not complete")

        history_problems = _history_issues(
            history,
            run_dir=run_dir,
            condition=condition,
            learning_rate=learning_rate,
            lr_schedule=lr_schedule,
            population_size=int(COMMON_CONFIG["population_size"]),
            diagnostic_schema_version=diagnostic_schema_version,
            expected_iterations=expected_iterations,
            budget=budget,
        )
        issues.extend(history_problems)
        if not history_problems and status.get("status") == "complete":
            rows[run_dir] = _run_row(config, history, run_dir, budget)

    for cell in sorted(expected):
        found = candidates.get(cell, [])
        if not found:
            issues.append(f"missing cell {cell}")
        elif len(found) > 1:
            issues.append(f"duplicate cell {cell}: {found}")
    if hashes and len(set(hashes.values())) != 1:
        issues.append(f"source digest mismatch across runs: {sorted(set(hashes.values()))}")
    if array_job_ids and len(array_job_ids) != 1:
        issues.append(f"multiple Slurm array job ids found: {sorted(array_job_ids)}")

    for seed in seeds:
        initial = [
            row["initial_return"]
            for row in rows.values()
            if int(row["seed"]) == int(seed)
        ]
        if initial and not np.allclose(initial, initial[0], rtol=1e-12, atol=1e-12):
            issues.append(f"seed {seed}: matched runs have unequal initial returns")
    if issues:
        raise ValidationError(issues)
    return [rows[candidates[cell][0]] for cell in sorted(expected)]


def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
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
    metric_fields = (
        "training_step_auc",
        "return_at_training_step_budget",
        "final_return",
        "last_10_return",
        "best_return",
    )
    diagnostic_fields = (
        "implicit_convergence_fraction",
        "mean_implicit_relative_residual",
        "mean_hessian_split_correlation",
        "mean_hessian_split_sign_agreement",
        "mean_hessian_split_relative_disagreement",
        "mean_hessian_temporal_correlation",
        "mean_hessian_temporal_sign_agreement",
        "mean_hessian_temporal_relative_disagreement",
        "mean_curvature_projection_frac",
        "mean_curvature_active_frac",
        "max_linear_relative_residual",
        "min_linear_absolute_diagonal",
        "mean_linear_condition_estimate",
        "median_linear_condition_estimate",
        "max_linear_condition_estimate",
        "mean_linear_nonpositive_diagonal_frac",
        "median_step_norm_ratio",
        "max_step_norm_ratio",
    )
    for (condition, lr_schedule, learning_rate), group in sorted(groups.items()):
        record: dict[str, Any] = {
            "condition": condition,
            "lr_schedule": lr_schedule,
            "initial_learning_rate": learning_rate,
            "runs": len(group),
        }
        for field in metric_fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            record[f"{field}_mean"] = float(np.mean(values))
            record[f"{field}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for field in diagnostic_fields:
            values = [float(row[field]) for row in group if row[field] is not None]
            record[f"{field}_mean"] = float(np.mean(values)) if values else None
        output.append(record)
    return output


def _exact_sign_flip_p(differences: np.ndarray) -> float:
    """Return the exact two-sided paired sign-flip p-value for a mean contrast."""
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


def paired_contrasts(
    rows: Sequence[dict[str, Any]],
    *,
    baseline_condition: str = "standard_es",
) -> dict[str, Any]:
    """Build seed-paired performance contrasts against Standard ES.

    The output is deliberately separate from the grouped means: it preserves
    the paired differences that reveal whether an apparent average gain is
    consistent across seeds or driven by an outlier.
    """
    index: dict[tuple[str, str, float, int], dict[str, Any]] = {}
    conditions: set[str] = set()
    schedules: set[str] = set()
    learning_rates: set[float] = set()
    for row in rows:
        condition = str(row["condition"])
        schedule = str(row["lr_schedule"])
        learning_rate = float(row["initial_learning_rate"])
        seed = int(row["seed"])
        key = (condition, schedule, learning_rate, seed)
        if key in index:
            raise ValueError(f"duplicate paired-contrast row {key}")
        index[key] = row
        conditions.add(condition)
        schedules.add(schedule)
        learning_rates.add(learning_rate)
    if baseline_condition not in conditions:
        raise ValueError(f"missing baseline condition {baseline_condition!r}")

    cells: list[dict[str, Any]] = []
    for condition in sorted(conditions - {baseline_condition}):
        for schedule in sorted(schedules):
            for learning_rate in sorted(learning_rates):
                baseline_seeds = {
                    seed
                    for candidate, candidate_schedule, candidate_rate, seed in index
                    if candidate == baseline_condition
                    and candidate_schedule == schedule
                    and candidate_rate == learning_rate
                }
                condition_seeds = {
                    seed
                    for candidate, candidate_schedule, candidate_rate, seed in index
                    if candidate == condition
                    and candidate_schedule == schedule
                    and candidate_rate == learning_rate
                }
                if baseline_seeds != condition_seeds or not baseline_seeds:
                    raise ValueError(
                        "paired contrast requires identical nonempty seed sets for "
                        f"{condition}, {schedule}, alpha_0={learning_rate:g}"
                    )
                seeds = sorted(baseline_seeds)
                metrics: dict[str, Any] = {}
                for field in PAIRED_PERFORMANCE_FIELDS:
                    baseline = np.asarray(
                        [
                            float(index[(baseline_condition, schedule, learning_rate, seed)][field])
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    treatment = np.asarray(
                        [
                            float(index[(condition, schedule, learning_rate, seed)][field])
                            for seed in seeds
                        ],
                        dtype=np.float64,
                    )
                    differences = treatment - baseline
                    metrics[field] = {
                        "baseline_mean": float(np.mean(baseline)),
                        "condition_mean": float(np.mean(treatment)),
                        "paired_mean_difference": float(np.mean(differences)),
                        "paired_median_difference": float(np.median(differences)),
                        "paired_sample_sd": (
                            float(np.std(differences, ddof=1))
                            if len(differences) > 1
                            else 0.0
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
                cells.append(
                    {
                        "condition": condition,
                        "lr_schedule": schedule,
                        "initial_learning_rate": learning_rate,
                        "paired_runs": len(seeds),
                        "metrics": metrics,
                    }
                )
    return {
        "schema_version": 1,
        "baseline_condition": baseline_condition,
        "difference_direction": f"condition_minus_{baseline_condition}",
        "primary_metric": "training_step_auc",
        "training_step_auc_budget": AUC_TRAINING_STEP_BUDGET,
        "cells": cells,
    }


def _write_csv(path: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument(
        "--protocol",
        choices=("signed_diagnostic", "hessian_fix"),
        default="signed_diagnostic",
    )
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--contrast-output", default=None)
    parser.add_argument("--expected-source-sha", required=True)
    args = parser.parse_args()
    run_output = args.run_output or os.path.join(args.root, "validated_runs.csv")
    summary_output = args.summary_output or os.path.join(args.root, "validated_summary.csv")
    contrast_output = args.contrast_output or os.path.join(
        args.root, "paired_contrasts.json"
    )
    conditions = CONDITIONS
    seeds = SEEDS
    if args.protocol == "hessian_fix":
        conditions = HESSIAN_FIX_CONDITIONS
        seeds = HESSIAN_FIX_SEEDS
    try:
        rows = validate_and_collect(
            args.root,
            conditions=conditions,
            seeds=seeds,
            expected_source_sha=args.expected_source_sha,
        )
    except ValidationError as error:
        for issue in error.issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        raise SystemExit(2) from error
    summaries = aggregate(rows)
    contrasts = paired_contrasts(rows)
    _write_csv(run_output, rows, RUN_FIELDS)
    _write_csv(summary_output, summaries, SUMMARY_FIELDS)
    os.makedirs(os.path.dirname(os.path.abspath(contrast_output)), exist_ok=True)
    with open(contrast_output, "w", encoding="utf-8") as stream:
        json.dump(contrasts, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"Validated {len(rows)} runs; wrote {run_output}, {summary_output}, "
        f"and {contrast_output}"
    )


if __name__ == "__main__":
    main()

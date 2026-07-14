#!/usr/bin/env python3
"""Strict validator and descriptive summary for the Hopper development screen."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from experiments.train import _source_digest


CONFIG_RELATIVE_PATH = "configs/mujuco/hopper_fresh_optimizer_development.yaml"
MANIFEST_RELATIVE_PATH = (
    "experiments/manifests/hopper_fresh_optimizer_development.json"
)
LAUNCHER_RELATIVE_PATH = "scripts/submit_hopper_fresh_optimizer_development.sh"
CONFIG_PATH = os.path.realpath(os.path.join(REPO_ROOT, CONFIG_RELATIVE_PATH))
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, MANIFEST_RELATIVE_PATH)
DEFAULT_LAUNCHER_PATH = os.path.join(REPO_ROOT, LAUNCHER_RELATIVE_PATH)

STUDY = "hopper_fresh_optimizer_development"
SEEDS = (200, 201, 202)
EXPECTED_ITERATIONS = 250
EXPECTED_CELLS = 33
EXPECTED_RUNS = 99
POPULATION_SIZE = 200
EVAL_INTERVAL = 10
EVAL_EPISODES = 5
DIAGNOSTIC_SCHEMA_VERSION = 2

STANDARD = "standard_es"
MOMENTUM = "momentum_es"
ADAM = "adam_es"
CLIPUP = "clipup_es"
STRUCTURED = "concave_block_ema_curvature_es"
ISOTROPIC = "concave_block_ema_isotropic_control_es"
OLS_GATE = "concave_block_ols_ema_curvature_es"
CONDITIONS = (
    STANDARD,
    MOMENTUM,
    ADAM,
    CLIPUP,
    STRUCTURED,
    ISOTROPIC,
    OLS_GATE,
)
CONDITION_COUNTS = {
    STANDARD: 7,
    MOMENTUM: 3,
    ADAM: 4,
    CLIPUP: 4,
    STRUCTURED: 6,
    ISOTROPIC: 6,
    OLS_GATE: 3,
}
CURVATURE_CONDITIONS = {STRUCTURED, ISOTROPIC, OLS_GATE}

EXPECTED_COMMON_CONFIG: dict[str, Any] = {
    "env_name": "Hopper-v5",
    "population_size": POPULATION_SIZE,
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
    "eval_episodes": EVAL_EPISODES,
    "eval_interval": EVAL_INTERVAL,
    "log_interval": 10,
    "max_episode_steps": 1000,
    "use_obs_norm": True,
    "obs_norm_mode": "frozen_after_calibration",
    "obs_norm_calibration_episodes": 3,
    "heldout_evaluation_enabled": False,
    "replay_enabled": False,
    "buffer_size": 0,
    "reuse_fraction": 0.0,
    "common_rollout_seed": True,
    "implicit_damping": 0.0,
    "linear_min_abs_diagonal": 1e-12,
    "evaluate_center_fitness": False,
    "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
}

CONFIG_RUNTIME_KEYS = {
    "_config_path",
    "algorithm",
    "condition",
    "learning_rate",
    "lr_schedule",
    "curvature_beta",
    "provenance",
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
OPTIMIZER_CONFIG_KEYS = {
    MOMENTUM: {"momentum_beta"},
    ADAM: {"adam_beta1", "adam_beta2", "adam_epsilon"},
    CLIPUP: {"clipup_momentum", "clipup_max_speed"},
}

RUN_FIELDS = (
    "study",
    "analysis_designation",
    "exploratory",
    "cell_id",
    "label",
    "condition",
    "seed",
    "seed_index",
    "task_id",
    "array_job_id",
    "learning_rate",
    "lr_schedule",
    "momentum_beta",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "clipup_momentum",
    "clipup_max_speed",
    "initial_return",
    "evaluation_generation_auc",
    "final_return",
    "best_return",
    "evaluation_point_count",
    "training_env_steps",
    "eval_env_steps",
    "normalization_calibration_env_steps",
    "total_env_steps",
    "mean_grad_norm",
    "mean_step_norm",
    "median_step_norm",
    "max_step_norm",
    "final_step_norm",
    "first_step_over_sigma",
    "mean_step_over_sigma",
    "max_step_over_sigma",
    "local_step_fraction",
    "mean_explicit_step_norm",
    "mean_step_norm_ratio",
    "solve_success_fraction",
    "clipup_clipped_updates",
    "clipup_clip_fraction",
    "clipup_mean_clip_scale",
    "clipup_min_clip_scale",
    "clipup_max_velocity_norm",
    "mean_h_split_correlation",
    "mean_h_split_sign_agreement",
    "mean_h_split_relative_disagreement",
    "mean_h_temporal_correlation",
    "mean_h_temporal_sign_agreement",
    "mean_h_temporal_relative_disagreement",
    "mean_curvature_projection_frac",
    "mean_curvature_projection_parameter_frac",
    "mean_curvature_active_frac",
    "mean_h_raw_std",
    "mean_denominator_condition",
    "mean_isotropic_attenuation_scale",
    "mean_attenuation_norm_match_relative_error",
    "mean_curvature_confidence_pass_frac",
    "mean_curvature_confidence_gate_frac",
    "mean_regression_r_squared",
    "mean_regression_design_condition",
    "source_sha256",
    "manifest_sha256",
    "launcher_sha256",
    "run_dir",
)

GROUP_METRICS = (
    "evaluation_generation_auc",
    "final_return",
    "best_return",
    "training_env_steps",
    "eval_env_steps",
    "total_env_steps",
    "mean_grad_norm",
    "mean_step_norm",
    "max_step_norm",
    "first_step_over_sigma",
    "mean_step_over_sigma",
    "max_step_over_sigma",
    "local_step_fraction",
    "mean_explicit_step_norm",
    "mean_step_norm_ratio",
    "solve_success_fraction",
    "clipup_clip_fraction",
    "clipup_mean_clip_scale",
    "mean_h_split_correlation",
    "mean_h_split_sign_agreement",
    "mean_h_split_relative_disagreement",
    "mean_h_temporal_correlation",
    "mean_h_temporal_sign_agreement",
    "mean_h_temporal_relative_disagreement",
    "mean_curvature_projection_frac",
    "mean_curvature_projection_parameter_frac",
    "mean_curvature_active_frac",
    "mean_h_raw_std",
    "mean_denominator_condition",
    "mean_isotropic_attenuation_scale",
    "mean_attenuation_norm_match_relative_error",
    "mean_curvature_confidence_pass_frac",
    "mean_curvature_confidence_gate_frac",
    "mean_regression_r_squared",
    "mean_regression_design_condition",
)

CONTRAST_METRICS = (
    "evaluation_generation_auc",
    "final_return",
    "best_return",
    "mean_step_norm",
    "mean_step_norm_ratio",
    "mean_step_over_sigma",
    "mean_h_split_correlation",
    "mean_h_split_sign_agreement",
    "mean_h_split_relative_disagreement",
    "mean_h_temporal_correlation",
    "mean_h_temporal_sign_agreement",
    "mean_isotropic_attenuation_scale",
)


class DevelopmentValidationError(RuntimeError):
    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        if isinstance(actual, bool):
            return False
        try:
            return bool(
                np.isfinite(float(actual))
                and np.isclose(float(actual), expected, rtol=1e-12, atol=1e-12)
            )
        except (TypeError, ValueError):
            return False
    return actual == expected


def _close(left: Any, right: Any, *, tolerance: float = 1e-10) -> bool:
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


def _validate_digest(value: str, name: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: str) -> list[Any]:
    records: list[Any] = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            records.append(json.loads(line))
    return records


def _cell_expected_keys(condition: str) -> set[str]:
    keys = {"cell_id", "label", "condition", "learning_rate", "lr_schedule"}
    keys |= OPTIMIZER_CONFIG_KEYS.get(condition, set())
    return keys


def load_and_validate_manifest(
    path: str,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    expected_sha256 = _validate_digest(expected_sha256, "expected_manifest_sha256")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise DevelopmentValidationError(
            [
                f"manifest digest mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            ]
        )
    manifest = _read_json(path)
    issues: list[str] = []
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "study",
        "exploratory",
        "seeds",
        "iterations",
        "cells",
    }:
        raise DevelopmentValidationError(["manifest top-level schema is not exact"])
    exact_top = {
        "schema_version": 1,
        "study": STUDY,
        "exploratory": True,
        "seeds": list(SEEDS),
        "iterations": EXPECTED_ITERATIONS,
    }
    for key, expected in exact_top.items():
        if manifest.get(key) != expected:
            issues.append(f"manifest.{key} is not {expected!r}")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or len(cells) != EXPECTED_CELLS:
        issues.append(f"manifest must contain exactly {EXPECTED_CELLS} cells")
        cells = []
    labels: set[str] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            issues.append(f"manifest.cells[{index}] is not an object")
            continue
        condition = cell.get("condition")
        if condition not in CONDITIONS:
            issues.append(f"manifest.cells[{index}] has unknown condition {condition!r}")
            continue
        if set(cell) != _cell_expected_keys(str(condition)):
            issues.append(f"manifest.cells[{index}] schema is not exact")
        if cell.get("cell_id") != index:
            issues.append(f"manifest.cells[{index}].cell_id is not {index}")
        label = cell.get("label")
        if not isinstance(label, str) or re.fullmatch(r"[a-z0-9_]+", label) is None:
            issues.append(f"manifest.cells[{index}].label is invalid")
        elif label in labels:
            issues.append(f"manifest label {label!r} is duplicated")
        else:
            labels.add(label)
        try:
            learning_rate = float(cell.get("learning_rate"))
        except (TypeError, ValueError):
            learning_rate = float("nan")
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            issues.append(f"manifest.cells[{index}].learning_rate is invalid")
        if cell.get("lr_schedule") not in {"constant", "inverse_sqrt"}:
            issues.append(f"manifest.cells[{index}].lr_schedule is invalid")
        for key in OPTIMIZER_CONFIG_KEYS.get(str(condition), set()):
            try:
                value = float(cell[key])
            except (KeyError, TypeError, ValueError):
                value = float("nan")
            if key.endswith(("beta", "beta1", "beta2", "momentum")):
                valid = np.isfinite(value) and 0.0 <= value < 1.0
            else:
                valid = np.isfinite(value) and value > 0.0
            if not valid:
                issues.append(f"manifest.cells[{index}].{key} is invalid")
    if Counter(cell.get("condition") for cell in cells if isinstance(cell, dict)) != Counter(
        CONDITION_COUNTS
    ):
        issues.append("manifest condition counts do not match the 33-cell protocol")
    if issues:
        raise DevelopmentValidationError(issues)
    return manifest, actual_sha256


def task_id_for(cell_id: int, seed_index: int, n_cells: int = EXPECTED_CELLS) -> int:
    cell_id = int(cell_id)
    seed_index = int(seed_index)
    if not 0 <= cell_id < n_cells or not 0 <= seed_index < len(SEEDS):
        raise ValueError("cell_id or seed_index is outside the development matrix")
    slot_index = (cell_id - 11 * seed_index) % n_cells
    return slot_index * len(SEEDS) + seed_index


def mapping_for_task(task_id: int, n_cells: int = EXPECTED_CELLS) -> tuple[int, int]:
    task_id = int(task_id)
    total = n_cells * len(SEEDS)
    if not 0 <= task_id < total:
        raise ValueError("task_id is outside the development matrix")
    slot_index, seed_index = divmod(task_id, len(SEEDS))
    cell_id = (slot_index + 11 * seed_index) % n_cells
    return cell_id, seed_index


def _condition_config_values(cell: dict[str, Any]) -> dict[str, Any]:
    condition = str(cell["condition"])
    if condition == STANDARD:
        values = {
            "algorithm": "standard_es",
            "use_curvature": False,
            "curvature_beta": 0.0,
        }
    elif condition == MOMENTUM:
        values = {
            "algorithm": "momentum_es",
            "use_curvature": False,
            "curvature_beta": 0.0,
            "momentum_beta": cell["momentum_beta"],
        }
    elif condition == ADAM:
        values = {
            "algorithm": "adam_es",
            "use_curvature": False,
            "curvature_beta": 0.0,
            "adam_beta1": cell["adam_beta1"],
            "adam_beta2": cell["adam_beta2"],
            "adam_epsilon": cell["adam_epsilon"],
        }
    elif condition == CLIPUP:
        values = {
            "algorithm": "clipup_es",
            "use_curvature": False,
            "curvature_beta": 0.0,
            "clipup_momentum": cell["clipup_momentum"],
            "clipup_max_speed": cell["clipup_max_speed"],
        }
    else:
        estimator = "block_joint_ols" if condition == OLS_GATE else "stein_moment"
        confidence = 1.645 if condition == OLS_GATE else None
        attenuation = "isotropic_norm_matched" if condition == ISOTROPIC else "structured"
        values = {
            "algorithm": "concave_curvature_es",
            "use_curvature": True,
            "curvature_beta": 0.9,
            "curvature_fitness": "matched",
            "curvature_mode": "block",
            "curvature_estimator": estimator,
            "curvature_confidence_z": confidence,
            "curvature_attenuation_mode": attenuation,
        }
    return values


def _resolved_optimizer_values(cell: dict[str, Any]) -> dict[str, Any]:
    condition = str(cell["condition"])
    common: dict[str, Any] = {
        "type": {
            STANDARD: "StandardES",
            MOMENTUM: "MomentumES",
            ADAM: "AdamES",
            CLIPUP: "ClipUpES",
        }.get(condition, "ConcaveCurvatureES"),
        "population_size": POPULATION_SIZE,
        "initial_learning_rate": cell["learning_rate"],
        "noise_std": 0.02,
        "rank_fitness": True,
        "l2_coeff": 0.0,
        "antithetic": True,
        "max_grad_norm": 0.0,
        "max_param_norm": None,
        "trust_region": False,
        "replay_enabled": False,
    }
    if condition == MOMENTUM:
        common.update(
            {
                "method": "momentum_es",
                "update_rule": "heavy_ball_momentum",
                "momentum_beta": cell["momentum_beta"],
            }
        )
    elif condition == ADAM:
        common.update(
            {
                "method": "adam_es",
                "update_rule": "bias_corrected_adam",
                "adam_beta1": cell["adam_beta1"],
                "adam_beta2": cell["adam_beta2"],
                "adam_epsilon": cell["adam_epsilon"],
                "adam_bias_correction": True,
            }
        )
    elif condition == CLIPUP:
        common.update(
            {
                "method": "clipup_es",
                "update_rule": "normalized_gradient_momentum_velocity_clip",
                "clipup_momentum": cell["clipup_momentum"],
                "clipup_max_speed": cell["clipup_max_speed"],
                "clipup_step_size_source": "learning_rate_schedule",
                "clipup_gradient_normalization": True,
                "clipup_velocity_clipping": True,
            }
        )
    elif condition in CURVATURE_CONDITIONS:
        attenuation = "isotropic_norm_matched" if condition == ISOTROPIC else "structured"
        estimator = "block_joint_ols" if condition == OLS_GATE else "stein_moment"
        confidence = 1.645 if condition == OLS_GATE else None
        solver = (
            "concave_projected_block_isotropic_attenuation_control"
            if condition == ISOTROPIC
            else "concave_projected_block"
        )
        common.update(
            {
                "method": "concave_curvature",
                "implicit_damping": 0.0,
                "curvature_fitness": "matched",
                "curvature_mode": "block",
                "curvature_structure": "block",
                "curvature_beta": 0.9,
                "curvature_same_generation": False,
                "curvature_estimator": estimator,
                "curvature_confidence_z": confidence,
                "curvature_attenuation_mode": attenuation,
                "curvature_clipping": False,
                "curvature_projection": "concave",
                "curvature_components": 3,
                "solver_type": solver,
            }
        )
    return common


def _validate_config(
    config: dict[str, Any],
    run_dir: str,
    cell: dict[str, Any],
    seed: int,
    issues: list[str],
) -> None:
    condition = str(cell["condition"])
    expected = dict(EXPECTED_COMMON_CONFIG)
    expected.update(
        {
            "condition": condition,
            "seed": seed,
            "learning_rate": cell["learning_rate"],
            "lr_schedule": cell["lr_schedule"],
        }
    )
    expected.update(_condition_config_values(cell))
    for key, value in expected.items():
        if not _matches(config.get(key), value):
            issues.append(f"{run_dir}: config.{key} is not locked to {value!r}")
    configured_path = os.path.normpath(str(config.get("_config_path", "")))
    expected_suffix = os.sep + os.path.normpath(CONFIG_RELATIVE_PATH)
    if not os.path.isabs(configured_path) or not configured_path.endswith(
        expected_suffix
    ):
        issues.append(f"{run_dir}: config path is not the development protocol path")

    expected_keys = set(EXPECTED_COMMON_CONFIG) | CONFIG_RUNTIME_KEYS
    expected_keys |= OPTIMIZER_CONFIG_KEYS.get(condition, set())
    if condition in CURVATURE_CONDITIONS:
        expected_keys |= CURVATURE_CONFIG_KEYS
    missing = sorted(expected_keys - set(config))
    unexpected = sorted(set(config) - expected_keys)
    if missing:
        issues.append(f"{run_dir}: config is missing keys {missing}")
    if unexpected:
        issues.append(f"{run_dir}: config has unexpected keys {unexpected}")
    forbidden = [
        key
        for key in config
        if "trust" in str(key).lower()
        or str(key).startswith("endpoint_")
        or key in {"implicit_iterations", "implicit_tolerance"}
    ]
    if forbidden:
        issues.append(f"{run_dir}: config contains trust/Picard keys {sorted(forbidden)}")

    resolved = config.get("resolved_optimizer")
    expected_resolved = _resolved_optimizer_values(cell)
    if not isinstance(resolved, dict):
        issues.append(f"{run_dir}: resolved_optimizer is missing")
        return
    for key, value in expected_resolved.items():
        if not _matches(resolved.get(key), value):
            issues.append(
                f"{run_dir}: resolved_optimizer.{key} is not {value!r}"
            )
    if set(resolved) != set(expected_resolved):
        issues.append(f"{run_dir}: resolved optimizer schema is not exact")
    if any("trust" in str(key).lower() and resolved[key] is not False for key in resolved):
        issues.append(f"{run_dir}: resolved optimizer enables trust control")
    if any("picard" in str(value).lower() for value in resolved.values()):
        issues.append(f"{run_dir}: resolved optimizer contains a Picard solver")


def _expected_argv(
    cell: dict[str, Any],
    seed: int,
    run_dir: str,
    workers: str,
) -> list[str]:
    argv = [
        "experiments/train.py",
        "--config",
        CONFIG_RELATIVE_PATH,
        "--condition",
        str(cell["condition"]),
        "--learning-rate",
        str(cell["learning_rate"]),
        "--lr-schedule",
        str(cell["lr_schedule"]),
        "--reuse-fraction",
        "0",
        "--seed",
        str(seed),
        "--workers",
        workers,
        "--output",
        run_dir,
    ]
    condition = str(cell["condition"])
    if condition == MOMENTUM:
        argv.extend(["--momentum-beta", str(cell["momentum_beta"])])
    elif condition == ADAM:
        argv.extend(
            [
                "--adam-beta1",
                str(cell["adam_beta1"]),
                "--adam-beta2",
                str(cell["adam_beta2"]),
                "--adam-epsilon",
                str(cell["adam_epsilon"]),
            ]
        )
    elif condition == CLIPUP:
        argv.extend(
            [
                "--clipup-momentum",
                str(cell["clipup_momentum"]),
                "--clipup-max-speed",
                str(cell["clipup_max_speed"]),
            ]
        )
    return argv


def _validate_provenance(
    config: dict[str, Any],
    run_dir: str,
    cell: dict[str, Any],
    seed: int,
    expected_task_id: int,
    expected_source_sha: str,
    expected_manifest_sha: str,
    expected_launcher_sha: str,
    issues: list[str],
) -> tuple[str | None, str | None, int | None]:
    provenance = config.get("provenance")
    if not isinstance(provenance, dict):
        issues.append(f"{run_dir}: provenance is missing")
        return None, None, None
    exact_hashes = {
        "source_sha256": expected_source_sha,
        "expected_source_sha256": expected_source_sha,
        "expected_manifest_sha256": expected_manifest_sha,
        "expected_launcher_sha256": expected_launcher_sha,
    }
    for key, expected in exact_hashes.items():
        if provenance.get(key) != expected:
            issues.append(f"{run_dir}: provenance.{key} does not match the lock")
    dependencies = provenance.get("dependencies")
    if not isinstance(dependencies, dict) or any(
        not isinstance(dependencies.get(name), str)
        for name in ("gymnasium", "mujoco", "PyYAML")
    ):
        issues.append(f"{run_dir}: dependency provenance is incomplete")
    rng = provenance.get("rng_scheme")
    expected_rng = {
        "optimizer": "numpy.RandomState(run_seed)",
        "parameter_initialization": (
            "numpy.default_rng(SeedSequence([run_seed, 1]))"
        ),
        "rollout": (
            "injective Cantor encoding of (run_seed, stream, iteration, index)"
        ),
    }
    if not isinstance(rng, dict) or rng != expected_rng:
        issues.append(f"{run_dir}: RNG provenance is not exact")
    if "source_git_revision" in provenance and re.fullmatch(
        r"[0-9a-f]{40}|unavailable", str(provenance["source_git_revision"])
    ) is None:
        issues.append(f"{run_dir}: source_git_revision is invalid")
    if "source_repo_dir" in provenance and not os.path.isabs(
        str(provenance["source_repo_dir"])
    ):
        issues.append(f"{run_dir}: source_repo_dir is not absolute")

    array_job_id = provenance.get("slurm_array_job_id")
    task_raw = provenance.get("slurm_array_task_id")
    if not isinstance(array_job_id, str) or re.fullmatch(r"[0-9]+", array_job_id) is None:
        issues.append(f"{run_dir}: Slurm array job id is invalid")
        array_job_id = None
    try:
        if isinstance(task_raw, bool):
            raise ValueError
        task_id = int(task_raw)
    except (TypeError, ValueError):
        issues.append(f"{run_dir}: Slurm task id is invalid")
        task_id = None
    if task_id is not None and task_id != expected_task_id:
        issues.append(
            f"{run_dir}: task id {task_id} does not match rotated mapping "
            f"{expected_task_id}"
        )

    argv = provenance.get("argv")
    if not isinstance(argv, list):
        issues.append(f"{run_dir}: trainer argv is missing")
    else:
        try:
            worker_index = argv.index("--workers") + 1
            workers = str(int(argv[worker_index]))
            if int(workers) <= 0:
                raise ValueError
        except (ValueError, IndexError, TypeError):
            workers = "invalid"
        if workers == "invalid":
            issues.append(f"{run_dir}: trainer worker count is invalid")
        else:
            expected_argv = _expected_argv(cell, seed, run_dir, workers)
            normalized = list(argv)
            try:
                output_index = normalized.index("--output") + 1
                normalized[output_index] = os.path.abspath(str(normalized[output_index]))
                expected_argv[output_index] = os.path.abspath(expected_argv[output_index])
            except (ValueError, IndexError):
                pass
            if normalized != expected_argv:
                issues.append(f"{run_dir}: trainer argv deviates from the manifest cell")
    return (
        provenance.get("source_sha256")
        if isinstance(provenance.get("source_sha256"), str)
        else None,
        array_job_id,
        task_id,
    )


def _finite_number(
    record: dict[str, Any],
    key: str,
    run_dir: str,
    index: int,
    issues: list[str],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float | None:
    try:
        if isinstance(record.get(key), bool):
            raise ValueError
        value = float(record[key])
    except (KeyError, TypeError, ValueError):
        issues.append(f"{run_dir}: history[{index}].{key} is missing or invalid")
        return None
    if not np.isfinite(value) or (lower is not None and value < lower) or (
        upper is not None and value > upper
    ):
        issues.append(f"{run_dir}: history[{index}].{key} is out of bounds")
        return None
    return value


def _mean_optional(values: Sequence[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    array = np.asarray(numeric, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        return None
    return float(np.mean(array))


def _serialized_vector(
    record: dict[str, Any],
    key: str,
    run_dir: str,
    index: int,
    issues: list[str],
    *,
    expected_length: int = 3,
) -> np.ndarray | None:
    raw = record.get(key)
    try:
        vector = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        vector = np.empty(0)
    if (
        vector.shape != (expected_length,)
        or not np.all(np.isfinite(vector))
        or record.get(f"{key}_length") != expected_length
        or record.get(f"{key}_omitted") is not False
        or record.get(f"{key}_serialization") != "persisted"
    ):
        issues.append(f"{run_dir}: history[{index}].{key} serialization is invalid")
        return None
    return vector


def _vector_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    )
    if denominator <= 1e-15:
        return 1.0 if np.allclose(left, right, rtol=1e-12, atol=1e-12) else 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def evaluation_generation_metrics(
    history: Sequence[dict[str, Any]],
    *,
    expected_iterations: int = EXPECTED_ITERATIONS,
) -> dict[str, Any]:
    """Compute AUC from the initial center and records with real eval rollouts."""
    if not history:
        raise ValueError("history is empty")
    initial = float(history[0]["initial_eval_reward"])
    generations = [0]
    returns = [initial]
    for index, record in enumerate(history):
        eval_steps = int(record["eval_env_steps_iter"])
        if eval_steps > 0:
            generations.append(index + 1)
            returns.append(float(record["eval_reward"]))
    x = np.asarray(generations, dtype=np.float64)
    y = np.asarray(returns, dtype=np.float64)
    if (
        len(x) < 2
        or x[0] != 0.0
        or x[-1] != float(expected_iterations)
        or np.any(np.diff(x) <= 0.0)
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("evaluation points do not cover the generation horizon")
    integrate = getattr(np, "trapezoid", np.trapz)
    return {
        "initial_return": initial,
        "evaluation_generation_auc": float(
            integrate(y, x) / float(expected_iterations)
        ),
        "final_return": float(y[-1]),
        "best_return": float(np.max(y)),
        "evaluation_point_count": int(len(y)),
        "evaluation_generations": [int(value) for value in generations],
        "evaluation_returns": [float(value) for value in returns],
    }


def _validate_curvature_record(
    record: dict[str, Any],
    run_dir: str,
    index: int,
    condition: str,
    issues: list[str],
) -> None:
    attenuation = "isotropic_norm_matched" if condition == ISOTROPIC else "structured"
    estimator = "block_joint_ols" if condition == OLS_GATE else "stein_moment"
    solver = (
        "concave_projected_block_isotropic_attenuation_control"
        if condition == ISOTROPIC
        else "concave_projected_block"
    )
    exact = {
        "solver_type": solver,
        "solve_success": True,
        "implicit_damping": 0.0,
        "fitness_transform": "centered_rank",
        "curvature_clip_frac": 0.0,
        "curvature_estimator": estimator,
        "curvature_attenuation_mode": attenuation,
        "curvature_fitness": "matched",
        "curvature_matches_gradient": True,
        "curvature_mode": "block",
        "curvature_structure": "block",
        "curvature_beta": 0.9,
        "curvature_same_generation": False,
        "curvature_components": 3,
        "curvature_component_count": 3,
        "hessian_pairs": 100,
        "hessian_ema_count": index + 1,
        "curvature_step_state": "bias_corrected_ema",
        "curvature_same_generation_se_available": True,
        "h_split_available": True,
        "h_split_rank_semantics": (
            "independent_centered_ranks_per_disjoint_pair_half"
        ),
        "h_split_pair_partition": "first_vs_second_antithetic_pair_halves",
        "h_split_first_pair_count": 50,
        "h_split_second_pair_count": 50,
        "h_temporal_available": index > 0,
        "curvature_confidence_gate_enabled": condition == OLS_GATE,
        "curvature_confidence_z": 1.645 if condition == OLS_GATE else None,
    }
    for key, expected in exact.items():
        if not _matches(record.get(key), expected):
            issues.append(
                f"{run_dir}: history[{index}].{key} is not {expected!r}"
            )
    block_sizes = _serialized_vector(
        record,
        "curvature_block_sizes",
        run_dir,
        index,
        issues,
    )
    if block_sizes is not None and not np.array_equal(
        block_sizes, np.asarray([768.0, 4160.0, 195.0])
    ):
        issues.append(f"{run_dir}: history[{index}] curvature block sizes are invalid")
    for key in (
        "curvature_same_generation_components",
        "curvature_same_generation_se_components",
        "curvature_step_state_components",
        "curvature_step_state_se_components",
        "curvature_confidence_upper_components",
        "curvature_raw_components",
        "curvature_ema_components",
        "curvature_ema_variance_components",
        "curvature_bias_corrected_ema_components",
        "curvature_step_components",
        "concave_curvature_components",
        "denominator_components",
    ):
        _serialized_vector(record, key, run_dir, index, issues)
    first = _serialized_vector(
        record, "h_split_first_components", run_dir, index, issues
    )
    second = _serialized_vector(
        record, "h_split_second_components", run_dir, index, issues
    )
    if first is not None and second is not None:
        correlation = _vector_correlation(first, second)
        sign_agreement = float(np.mean(np.sign(first) == np.sign(second)))
        disagreement = float(
            np.linalg.norm(first - second)
            / max(float(np.linalg.norm(first)), float(np.linalg.norm(second)), 1e-12)
        )
        for key, expected in (
            ("h_split_correlation", correlation),
            ("h_split_sign_agreement", sign_agreement),
            ("h_split_relative_disagreement", disagreement),
        ):
            if not _close(record.get(key), expected):
                issues.append(
                    f"{run_dir}: history[{index}].{key} does not match split vectors"
                )

    bounded = {
        "h_split_correlation": (-1.0, 1.0),
        "h_split_sign_agreement": (0.0, 1.0),
        "h_split_relative_disagreement": (0.0, None),
        "h_temporal_correlation": (-1.0, 1.0),
        "h_temporal_sign_agreement": (0.0, 1.0),
        "h_temporal_relative_disagreement": (0.0, None),
        "curvature_projection_frac": (0.0, 1.0),
        "curvature_projection_parameter_frac": (0.0, 1.0),
        "curvature_active_frac": (0.0, 1.0),
        "curvature_confidence_pass_frac": (0.0, 1.0),
        "curvature_confidence_gate_frac": (0.0, 1.0),
        "attenuation_norm_match_relative_error": (0.0, 1e-10),
        "linear_relative_residual": (0.0, 1e-10),
        "structured_reference_relative_residual": (0.0, 1e-10),
        "denominator_min": (1.0, None),
        "denominator_max": (1.0, None),
        "denominator_condition": (1.0, None),
        "h_raw_std": (0.0, None),
        "step_norm_ratio": (0.0, 1.0 + 1e-8),
    }
    for key, (lower, upper) in bounded.items():
        _finite_number(
            record,
            key,
            run_dir,
            index,
            issues,
            lower=lower,
            upper=upper,
        )
    attenuation_scale = _finite_number(
        record,
        "isotropic_attenuation_scale",
        run_dir,
        index,
        issues,
        lower=np.nextafter(0.0, 1.0),
        upper=1.0,
    )
    if condition != ISOTROPIC and attenuation_scale is not None and attenuation_scale != 1.0:
        issues.append(f"{run_dir}: history[{index}] structured arm is attenuated isotropically")
    if not _close(record.get("step_norm"), record.get("structured_reference_step_norm")):
        issues.append(f"{run_dir}: history[{index}] is not norm-matched to structured step")
    if condition != OLS_GATE and not _matches(
        record.get("curvature_confidence_gate_frac"), 0.0
    ):
        issues.append(f"{run_dir}: history[{index}] ungated curvature has gate attenuation")
    if condition == OLS_GATE:
        regression_exact = {
            "regression_rank": 4,
            "regression_parameters": 4,
            "regression_residual_dof": 96,
        }
        for key, expected in regression_exact.items():
            if record.get(key) != expected:
                issues.append(f"{run_dir}: history[{index}].{key} is invalid")
        for key in (
            "regression_residual_std",
            "regression_r_squared",
            "regression_design_condition",
        ):
            _finite_number(record, key, run_dir, index, issues)


def _validate_adaptive_record(
    record: dict[str, Any],
    run_dir: str,
    index: int,
    cell: dict[str, Any],
    issues: list[str],
) -> None:
    condition = str(cell["condition"])
    if condition == MOMENTUM:
        exact = {
            "optimizer_type": "momentum",
            "momentum_beta": cell["momentum_beta"],
            "momentum_iteration": index + 1,
        }
        for key, expected in exact.items():
            if not _matches(record.get(key), expected):
                issues.append(f"{run_dir}: history[{index}].{key} is invalid")
        _finite_number(
            record, "momentum_buffer_norm", run_dir, index, issues, lower=0.0
        )
    elif condition == ADAM:
        exact = {
            "optimizer_type": "adam",
            "adam_beta1": cell["adam_beta1"],
            "adam_beta2": cell["adam_beta2"],
            "adam_epsilon": cell["adam_epsilon"],
            "adam_iteration": index + 1,
            "adam_first_moment_bias_correction": (
                1.0 - float(cell["adam_beta1"]) ** (index + 1)
            ),
            "adam_second_moment_bias_correction": (
                1.0 - float(cell["adam_beta2"]) ** (index + 1)
            ),
        }
        for key, expected in exact.items():
            if not _matches(record.get(key), expected):
                issues.append(f"{run_dir}: history[{index}].{key} is invalid")
        for key in ("adam_first_moment_norm", "adam_second_moment_norm"):
            _finite_number(record, key, run_dir, index, issues, lower=0.0)
    elif condition == CLIPUP:
        exact = {
            "optimizer_type": "clipup",
            "clipup_momentum": cell["clipup_momentum"],
            "clipup_max_speed": cell["clipup_max_speed"],
            "clipup_iteration": index + 1,
        }
        for key, expected in exact.items():
            if not _matches(record.get(key), expected):
                issues.append(f"{run_dir}: history[{index}].{key} is invalid")
        if not _close(record.get("clipup_step_size"), record.get("lr")):
            issues.append(f"{run_dir}: history[{index}] ClipUp step size is not scheduled LR")
        input_gradient_norm = _finite_number(
            record,
            "clipup_input_gradient_norm",
            run_dir,
            index,
            issues,
            lower=0.0,
        )
        normalized_step_norm = _finite_number(
            record,
            "clipup_normalized_gradient_step_norm",
            run_dir,
            index,
            issues,
            lower=0.0,
        )
        zero_gradient = record.get("clipup_zero_gradient")
        if not isinstance(zero_gradient, bool):
            issues.append(f"{run_dir}: history[{index}] ClipUp zero-gradient flag is invalid")
        elif input_gradient_norm is not None and zero_gradient != (
            input_gradient_norm == 0.0
        ):
            issues.append(f"{run_dir}: history[{index}] ClipUp zero-gradient flag disagrees")
        expected_normalized_norm = 0.0 if zero_gradient is True else float(record["lr"])
        if normalized_step_norm is not None and not _close(
            normalized_step_norm, expected_normalized_norm
        ):
            issues.append(f"{run_dir}: history[{index}] ClipUp normalized step is invalid")
        before = _finite_number(
            record,
            "clipup_velocity_norm_before_clip",
            run_dir,
            index,
            issues,
            lower=0.0,
        )
        velocity = _finite_number(
            record,
            "clipup_velocity_norm",
            run_dir,
            index,
            issues,
            lower=0.0,
            upper=float(cell["clipup_max_speed"]) + 1e-10,
        )
        scale = _finite_number(
            record,
            "clipup_velocity_clip_scale",
            run_dir,
            index,
            issues,
            lower=np.nextafter(0.0, 1.0),
            upper=1.0,
        )
        clipped = record.get("clipup_velocity_clipped")
        if not isinstance(clipped, bool):
            issues.append(f"{run_dir}: history[{index}] ClipUp clipped flag is invalid")
        elif before is not None and velocity is not None and scale is not None:
            expected_clipped = before > float(cell["clipup_max_speed"])
            if clipped != expected_clipped:
                issues.append(f"{run_dir}: history[{index}] ClipUp clipped flag disagrees")
            if not _close(velocity, before * scale):
                issues.append(f"{run_dir}: history[{index}] ClipUp scale is inconsistent")
        if velocity is not None and not _close(record.get("step_norm"), velocity):
            issues.append(f"{run_dir}: history[{index}] ClipUp step/velocity mismatch")


def _validate_history(
    history: Any,
    run_dir: str,
    cell: dict[str, Any],
    issues: list[str],
) -> dict[str, Any] | None:
    if not isinstance(history, list):
        issues.append(f"{run_dir}: history is not a list")
        return None
    if len(history) != EXPECTED_ITERATIONS:
        issues.append(f"{run_dir}: history does not contain 250 complete updates")
        return None
    condition = str(cell["condition"])
    previous_training_steps = 0
    previous_eval_steps: int | None = None
    previous_eval_reward: float | None = None
    running_best: float | None = None
    initial_reward: float | None = None
    initial_eval_steps: int | None = None
    calibration_steps: int | None = None
    for index, record in enumerate(history):
        if not isinstance(record, dict):
            issues.append(f"{run_dir}: history[{index}] is not an object")
            continue
        if record.get("iteration") != index:
            issues.append(f"{run_dir}: history[{index}] has the wrong iteration")
        expected_lr = float(cell["learning_rate"])
        if cell["lr_schedule"] == "inverse_sqrt":
            expected_lr /= np.sqrt(index + 1.0)
        for key in ("lr", "learning_rate"):
            if not _close(record.get(key), expected_lr):
                issues.append(f"{run_dir}: history[{index}].{key} violates the schedule")
        exact_fresh = {
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
            "clip_frac": 0.0,
            "clip_fraction": 0.0,
            "parameter_projection_active": False,
            "curvature_clip_frac": 0.0,
            "sigma": 0.02,
            "eval_count": POPULATION_SIZE * (index + 1),
        }
        for key, expected in exact_fresh.items():
            if not _matches(record.get(key), expected):
                issues.append(
                    f"{run_dir}: history[{index}].{key} violates fresh-only protocol"
                )
        for alias in ("mean_importance_weight", "max_importance_weight"):
            if alias in record and not _matches(record[alias], 1.0):
                issues.append(
                    f"{run_dir}: history[{index}].{alias} violates fresh-only protocol"
                )
        forbidden = [
            key
            for key in record
            if "trust" in str(key).lower()
            or str(key).startswith("endpoint_")
            or key in {
                "implicit_iterations",
                "implicit_converged",
                "implicit_relative_residual",
            }
        ]
        if forbidden:
            issues.append(
                f"{run_dir}: history[{index}] contains trust/Picard fields {sorted(forbidden)}"
            )
        for key in (
            "grad_norm",
            "grad_norm_before_clip",
            "param_norm",
            "param_change",
            "step_norm",
            "proposed_step_norm",
            "explicit_step_norm",
            "explicit_gradient_step_norm",
            "step_norm_ratio",
            "time",
            "iteration_compute_seconds",
        ):
            _finite_number(record, key, run_dir, index, issues, lower=0.0)
        if not _close(record.get("grad_norm"), record.get("grad_norm_before_clip")):
            issues.append(f"{run_dir}: history[{index}] applies gradient clipping")
        for key in ("param_change", "proposed_step_norm"):
            if not _close(record.get("step_norm"), record.get(key)):
                issues.append(f"{run_dir}: history[{index}].{key} disagrees with step norm")

        try:
            train_iter = int(record["training_env_steps_iter"])
            train_total = int(record["training_env_steps"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] training steps are invalid")
            train_iter = 0
            train_total = previous_training_steps
        if not 200 <= train_iter <= 200_000 or train_total != previous_training_steps + train_iter:
            issues.append(f"{run_dir}: history[{index}] training-step accounting is invalid")
        for alias in ("train_env_steps", "env_steps"):
            if record.get(alias) != train_total:
                issues.append(f"{run_dir}: history[{index}].{alias} disagrees")
        for alias in ("train_env_steps_iter", "env_steps_iter"):
            if record.get(alias) != train_iter:
                issues.append(f"{run_dir}: history[{index}].{alias} disagrees")
        previous_training_steps = train_total

        try:
            eval_iter = int(record["eval_env_steps_iter"])
            eval_total = int(record["eval_env_steps"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] evaluation steps are invalid")
            eval_iter = 0
            eval_total = previous_eval_steps or 0
        expected_eval = index % EVAL_INTERVAL == 0 or index == EXPECTED_ITERATIONS - 1
        if expected_eval != (eval_iter > 0) or (eval_iter > 0 and not 5 <= eval_iter <= 5_000):
            issues.append(f"{run_dir}: history[{index}] evaluation cadence is invalid")
        if previous_eval_steps is None:
            try:
                first_eval_steps = int(record["initial_eval_env_steps"])
            except (KeyError, TypeError, ValueError):
                first_eval_steps = 0
            if not 5 <= first_eval_steps <= 5_000:
                issues.append(f"{run_dir}: initial evaluation steps are invalid")
            initial_eval_steps = first_eval_steps
            previous_eval_steps = first_eval_steps
        if eval_total != previous_eval_steps + eval_iter:
            issues.append(f"{run_dir}: history[{index}] evaluation-step accounting is invalid")
        previous_eval_steps = eval_total

        try:
            record_initial = float(record["initial_eval_reward"])
            eval_reward = float(record["eval_reward"])
            best_reward = float(record["best_reward"])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{run_dir}: history[{index}] evaluation returns are invalid")
            continue
        if not all(np.isfinite(value) for value in (record_initial, eval_reward, best_reward)):
            issues.append(f"{run_dir}: history[{index}] evaluation returns are non-finite")
        if initial_reward is None:
            initial_reward = record_initial
            previous_eval_reward = record_initial
            running_best = record_initial
            try:
                calibration_steps = int(record["normalization_calibration_env_steps"])
            except (KeyError, TypeError, ValueError):
                calibration_steps = 0
            if not 3 <= calibration_steps <= 3_000:
                issues.append(f"{run_dir}: calibration-step count is invalid")
        elif not _close(record_initial, initial_reward):
            issues.append(f"{run_dir}: history[{index}] initial return changed")
        if record.get("initial_eval_env_steps") != initial_eval_steps:
            issues.append(f"{run_dir}: history[{index}] initial eval steps changed")
        if record.get("normalization_calibration_env_steps") != calibration_steps:
            issues.append(f"{run_dir}: history[{index}] calibration steps changed")
        if expected_eval:
            previous_eval_reward = eval_reward
            running_best = max(float(running_best), eval_reward)
        elif not _close(eval_reward, previous_eval_reward):
            issues.append(f"{run_dir}: history[{index}] carried evaluation return changed")
        if not _close(best_reward, running_best):
            issues.append(f"{run_dir}: history[{index}] best return is inconsistent")

        expected_total = int(calibration_steps or 0) + train_total + eval_total
        if record.get("total_env_steps") != expected_total:
            issues.append(f"{run_dir}: history[{index}] total environment steps disagree")
        if record.get("total_env_steps_iter") != train_iter + eval_iter:
            issues.append(f"{run_dir}: history[{index}] per-update environment steps disagree")

        if condition in CURVATURE_CONDITIONS:
            _validate_curvature_record(record, run_dir, index, condition, issues)
        else:
            if record.get("solver_type") != "none":
                issues.append(f"{run_dir}: history[{index}] non-curvature solver is invalid")
            _validate_adaptive_record(record, run_dir, index, cell, issues)

    try:
        evaluation = evaluation_generation_metrics(history)
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{run_dir}: evaluation metric inputs are invalid: {error}")
        return None
    if evaluation["evaluation_point_count"] != 27:
        issues.append(f"{run_dir}: evaluation curve does not contain 27 actual points")

    step_norms = np.asarray([record["step_norm"] for record in history], dtype=float)
    sigma = float(history[0]["sigma"])
    step_over_sigma = step_norms / sigma
    grad_norms = np.asarray([record["grad_norm"] for record in history], dtype=float)
    explicit_norms = np.asarray(
        [record["explicit_step_norm"] for record in history], dtype=float
    )
    step_ratios = np.asarray([record["step_norm_ratio"] for record in history], dtype=float)
    metrics: dict[str, Any] = {
        **evaluation,
        "training_env_steps": int(history[-1]["training_env_steps"]),
        "eval_env_steps": int(history[-1]["eval_env_steps"]),
        "normalization_calibration_env_steps": int(
            history[-1]["normalization_calibration_env_steps"]
        ),
        "total_env_steps": int(history[-1]["total_env_steps"]),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "mean_step_norm": float(np.mean(step_norms)),
        "median_step_norm": float(np.median(step_norms)),
        "max_step_norm": float(np.max(step_norms)),
        "final_step_norm": float(step_norms[-1]),
        "first_step_over_sigma": float(step_over_sigma[0]),
        "mean_step_over_sigma": float(np.mean(step_over_sigma)),
        "max_step_over_sigma": float(np.max(step_over_sigma)),
        "local_step_fraction": float(np.mean(step_over_sigma <= 1.0)),
        "mean_explicit_step_norm": float(np.mean(explicit_norms)),
        "mean_step_norm_ratio": float(np.mean(step_ratios)),
        "solve_success_fraction": (
            float(np.mean([bool(record["solve_success"]) for record in history]))
            if condition in CURVATURE_CONDITIONS
            else None
        ),
    }
    if condition == CLIPUP:
        clipped = np.asarray(
            [bool(record["clipup_velocity_clipped"]) for record in history]
        )
        scales = np.asarray(
            [record["clipup_velocity_clip_scale"] for record in history], dtype=float
        )
        velocities = np.asarray(
            [record["clipup_velocity_norm"] for record in history], dtype=float
        )
        metrics.update(
            {
                "clipup_clipped_updates": int(np.sum(clipped)),
                "clipup_clip_fraction": float(np.mean(clipped)),
                "clipup_mean_clip_scale": float(np.mean(scales)),
                "clipup_min_clip_scale": float(np.min(scales)),
                "clipup_max_velocity_norm": float(np.max(velocities)),
            }
        )
    else:
        metrics.update(
            {
                "clipup_clipped_updates": None,
                "clipup_clip_fraction": None,
                "clipup_mean_clip_scale": None,
                "clipup_min_clip_scale": None,
                "clipup_max_velocity_norm": None,
            }
        )
    curvature_summary_fields = {
        "mean_h_split_correlation": "h_split_correlation",
        "mean_h_split_sign_agreement": "h_split_sign_agreement",
        "mean_h_split_relative_disagreement": "h_split_relative_disagreement",
        "mean_curvature_projection_frac": "curvature_projection_frac",
        "mean_curvature_projection_parameter_frac": (
            "curvature_projection_parameter_frac"
        ),
        "mean_curvature_active_frac": "curvature_active_frac",
        "mean_h_raw_std": "h_raw_std",
        "mean_denominator_condition": "denominator_condition",
        "mean_isotropic_attenuation_scale": "isotropic_attenuation_scale",
        "mean_attenuation_norm_match_relative_error": (
            "attenuation_norm_match_relative_error"
        ),
        "mean_curvature_confidence_pass_frac": "curvature_confidence_pass_frac",
        "mean_curvature_confidence_gate_frac": "curvature_confidence_gate_frac",
        "mean_regression_r_squared": "regression_r_squared",
        "mean_regression_design_condition": "regression_design_condition",
    }
    for output_key, record_key in curvature_summary_fields.items():
        metrics[output_key] = (
            _mean_optional([record.get(record_key) for record in history])
            if condition in CURVATURE_CONDITIONS
            else None
        )
    temporal_fields = {
        "mean_h_temporal_correlation": "h_temporal_correlation",
        "mean_h_temporal_sign_agreement": "h_temporal_sign_agreement",
        "mean_h_temporal_relative_disagreement": (
            "h_temporal_relative_disagreement"
        ),
    }
    for output_key, record_key in temporal_fields.items():
        metrics[output_key] = (
            _mean_optional(
                [
                    record.get(record_key)
                    for record in history
                    if record.get("h_temporal_available") is True
                ]
            )
            if condition in CURVATURE_CONDITIONS
            else None
        )
    return metrics


RUN_NAME_PATTERN = re.compile(
    r"^cell(?P<cell_id>[0-9]+)_(?P<label>[a-z0-9_]+)_"
    r"seed(?P<seed>[0-9]+)_job(?P<job_id>[0-9]+)_task(?P<task_id>[0-9]+)$"
)


def _validate_status(
    status: dict[str, Any],
    history: list[dict[str, Any]],
    metrics: dict[str, Any] | None,
    run_dir: str,
    issues: list[str],
) -> None:
    exact = {
        "status": "complete",
        "expected_iterations": EXPECTED_ITERATIONS,
        "completed_iterations": EXPECTED_ITERATIONS,
        "history_records": "history.jsonl",
    }
    for key, expected in exact.items():
        if status.get(key) != expected:
            issues.append(f"{run_dir}: status.{key} is not {expected!r}")
    if "training_budget" in status:
        issues.append(f"{run_dir}: status unexpectedly contains a training budget")
    if not history or metrics is None:
        return
    comparisons = {
        "initial_eval_reward": metrics["initial_return"],
        "best_reward": metrics["best_return"],
        "normalization_calibration_env_steps": metrics[
            "normalization_calibration_env_steps"
        ],
    }
    for key, expected in comparisons.items():
        if not _close(status.get(key), expected):
            issues.append(f"{run_dir}: status.{key} disagrees with history")


def validate_and_collect(
    root: str,
    *,
    manifest_path: str = DEFAULT_MANIFEST_PATH,
    launcher_path: str = DEFAULT_LAUNCHER_PATH,
    expected_source_sha: str,
    expected_manifest_sha: str,
    expected_launcher_sha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_source_sha = _validate_digest(expected_source_sha, "expected_source_sha")
    expected_manifest_sha = _validate_digest(
        expected_manifest_sha, "expected_manifest_sha"
    )
    expected_launcher_sha = _validate_digest(
        expected_launcher_sha, "expected_launcher_sha"
    )
    local_source_sha = _source_digest(CONFIG_PATH)
    if local_source_sha != expected_source_sha:
        raise DevelopmentValidationError(
            [
                f"analyzer checkout source digest mismatch: expected "
                f"{expected_source_sha}, found {local_source_sha}"
            ]
        )
    manifest, manifest_sha = load_and_validate_manifest(
        manifest_path, expected_sha256=expected_manifest_sha
    )
    launcher_sha = _sha256_file(launcher_path)
    if launcher_sha != expected_launcher_sha:
        raise DevelopmentValidationError(
            [
                f"launcher digest mismatch: expected {expected_launcher_sha}, "
                f"found {launcher_sha}"
            ]
        )
    cells = manifest["cells"]
    expected_mappings = {
        (int(cell["cell_id"]), seed)
        for cell in cells
        for seed in SEEDS
    }
    candidates: dict[tuple[int, int], list[str]] = defaultdict(list)
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    issues: list[str] = []
    source_hashes: set[str] = set()
    array_job_ids: set[str] = set()
    task_ids: set[int] = set()
    run_dirs = sorted(
        {
            os.path.dirname(path)
            for filename in ("config.json", "history.json", "history.jsonl", "status.json")
            for path in glob.glob(os.path.join(root, "**", filename), recursive=True)
        }
    )
    for run_dir in run_dirs:
        paths = {
            name: os.path.join(run_dir, name)
            for name in ("config.json", "history.json", "history.jsonl", "status.json")
        }
        if not all(os.path.isfile(path) for path in paths.values()):
            issues.append(f"{run_dir}: development artifacts are incomplete")
            continue
        match = RUN_NAME_PATTERN.fullmatch(os.path.basename(run_dir))
        if match is None:
            issues.append(f"{run_dir}: run directory name is not a manifest mapping")
            continue
        cell_id = int(match.group("cell_id"))
        seed = int(match.group("seed"))
        task_from_name = int(match.group("task_id"))
        if not 0 <= cell_id < len(cells) or seed not in SEEDS:
            issues.append(f"{run_dir}: directory identifies an unexpected cell/seed")
            continue
        cell = cells[cell_id]
        seed_index = SEEDS.index(seed)
        expected_task = task_id_for(cell_id, seed_index, len(cells))
        mapping = (cell_id, seed)
        candidates[mapping].append(run_dir)
        expected_name = (
            f"cell{cell_id}_{cell['label']}_seed{seed}_job{match.group('job_id')}_"
            f"task{expected_task}"
        )
        if os.path.basename(run_dir) != expected_name or task_from_name != expected_task:
            issues.append(f"{run_dir}: directory does not match the rotated task mapping")
        try:
            config = _read_json(paths["config.json"])
            history = _read_json(paths["history.json"])
            history_jsonl = _read_jsonl(paths["history.jsonl"])
            status = _read_json(paths["status.json"])
        except (OSError, ValueError, json.JSONDecodeError) as error:
            issues.append(f"{run_dir}: artifact is unreadable: {error}")
            continue
        if not isinstance(config, dict) or not isinstance(status, dict):
            issues.append(f"{run_dir}: config/status is not an object")
            continue
        if history_jsonl != history:
            issues.append(f"{run_dir}: history JSON and JSONL do not match exactly")
        _validate_config(config, run_dir, cell, seed, issues)
        source_sha, array_job_id, provenance_task = _validate_provenance(
            config,
            run_dir,
            cell,
            seed,
            expected_task,
            expected_source_sha,
            expected_manifest_sha,
            expected_launcher_sha,
            issues,
        )
        if source_sha is not None:
            source_hashes.add(source_sha)
        if array_job_id is not None:
            array_job_ids.add(array_job_id)
            if array_job_id != match.group("job_id"):
                issues.append(f"{run_dir}: directory/provenance array job ids disagree")
        if provenance_task is not None:
            task_ids.add(provenance_task)
            if provenance_task != task_from_name:
                issues.append(f"{run_dir}: directory/provenance task ids disagree")
        metrics = _validate_history(history, run_dir, cell, issues)
        _validate_status(
            status,
            history if isinstance(history, list) else [],
            metrics,
            run_dir,
            issues,
        )
        if metrics is None or array_job_id is None or provenance_task is None:
            continue
        row = {
            "study": STUDY,
            "analysis_designation": "exploratory_development_screen",
            "exploratory": True,
            "cell_id": cell_id,
            "label": cell["label"],
            "condition": cell["condition"],
            "seed": seed,
            "seed_index": seed_index,
            "task_id": expected_task,
            "array_job_id": array_job_id,
            "learning_rate": cell["learning_rate"],
            "lr_schedule": cell["lr_schedule"],
            "momentum_beta": cell.get("momentum_beta"),
            "adam_beta1": cell.get("adam_beta1"),
            "adam_beta2": cell.get("adam_beta2"),
            "adam_epsilon": cell.get("adam_epsilon"),
            "clipup_momentum": cell.get("clipup_momentum"),
            "clipup_max_speed": cell.get("clipup_max_speed"),
            **{key: metrics.get(key) for key in RUN_FIELDS if key in metrics},
            "source_sha256": source_sha,
            "manifest_sha256": manifest_sha,
            "launcher_sha256": launcher_sha,
            "run_dir": os.path.abspath(run_dir),
            "_history": history,
        }
        rows[mapping] = row

    for mapping in sorted(expected_mappings):
        found = candidates.get(mapping, [])
        if len(found) != 1:
            issues.append(f"cell/seed {mapping!r}: expected one run, found {len(found)}")
    if len(run_dirs) != EXPECTED_RUNS:
        issues.append(
            f"development root has {len(run_dirs)} run directories, expected {EXPECTED_RUNS}"
        )
    if source_hashes != {expected_source_sha}:
        issues.append("development runs do not share the expected source digest")
    if len(array_job_ids) != 1:
        issues.append("development runs do not share one Slurm array job id")
    if task_ids != set(range(EXPECTED_RUNS)):
        issues.append("development provenance does not cover task ids 0 through 98 exactly")

    for seed in SEEDS:
        seed_rows = [rows.get((cell_id, seed)) for cell_id in range(len(cells))]
        if any(row is None for row in seed_rows):
            continue
        typed = [row for row in seed_rows if row is not None]
        first = typed[0]
        for row in typed[1:]:
            for field in (
                "initial_return",
                "normalization_calibration_env_steps",
            ):
                if not _close(row[field], first[field]):
                    issues.append(f"seed {seed}: initial {field} is not matched")
            for field in (
                "mean_fitness",
                "max_fitness",
                "min_fitness",
                "training_env_steps_iter",
            ):
                if not _close(row["_history"][0].get(field), first["_history"][0].get(field)):
                    issues.append(f"seed {seed}: first-generation {field} is not matched")

    structured_cells = {
        (float(cell["learning_rate"]), str(cell["lr_schedule"])): cell
        for cell in cells
        if cell["condition"] == STRUCTURED
    }
    isotropic_cells = {
        (float(cell["learning_rate"]), str(cell["lr_schedule"])): cell
        for cell in cells
        if cell["condition"] == ISOTROPIC
    }
    if set(structured_cells) != set(isotropic_cells):
        issues.append("structured/isotropic manifest cells are not matched")
    for key in sorted(set(structured_cells) & set(isotropic_cells)):
        structured_cell = structured_cells[key]
        isotropic_cell = isotropic_cells[key]
        for seed in SEEDS:
            structured_row = rows.get((int(structured_cell["cell_id"]), seed))
            isotropic_row = rows.get((int(isotropic_cell["cell_id"]), seed))
            if structured_row is None or isotropic_row is None:
                continue
            for field in (
                "explicit_step_norm",
                "structured_reference_step_norm",
                "step_norm",
                "h_raw_mean",
                "h_raw_std",
                "h_split_correlation",
                "h_split_sign_agreement",
                "h_split_relative_disagreement",
            ):
                if not _close(
                    structured_row["_history"][0].get(field),
                    isotropic_row["_history"][0].get(field),
                ):
                    issues.append(
                        f"seed {seed}, lr/schedule {key}: first-update {field} is unmatched"
                    )
    if issues:
        raise DevelopmentValidationError(issues)
    ordered = [
        rows[(int(cell["cell_id"]), seed)]
        for cell in cells
        for seed in SEEDS
    ]
    metadata = {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "launcher_sha256": launcher_sha,
        "source_sha256": expected_source_sha,
        "analyzer_checkout_source_sha256": local_source_sha,
        "array_job_id": next(iter(array_job_ids)),
    }
    return ordered, metadata


GROUP_ID_FIELDS = (
    "study",
    "analysis_designation",
    "exploratory",
    "cell_id",
    "label",
    "condition",
    "learning_rate",
    "lr_schedule",
    "momentum_beta",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "clipup_momentum",
    "clipup_max_speed",
    "runs",
    "seeds",
)
GROUP_FIELDS = GROUP_ID_FIELDS + tuple(
    name
    for metric in GROUP_METRICS
    for name in (f"{metric}_mean", f"{metric}_sample_sd")
) + (
    "evaluation_generation_auc_min",
    "evaluation_generation_auc_max",
    "final_return_min",
    "final_return_max",
    "best_return_min",
    "best_return_max",
)


def aggregate_runs(
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    by_cell: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[int(row["cell_id"])].append(row)
    groups: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        cell_id = int(cell["cell_id"])
        cell_rows = sorted(by_cell[cell_id], key=lambda row: int(row["seed"]))
        if len(cell_rows) != len(SEEDS) or [row["seed"] for row in cell_rows] != list(SEEDS):
            raise ValueError(f"cell {cell_id} does not contain all development seeds")
        group: dict[str, Any] = {
            "study": STUDY,
            "analysis_designation": "exploratory_development_screen",
            "exploratory": True,
            "cell_id": cell_id,
            "label": cell["label"],
            "condition": cell["condition"],
            "learning_rate": cell["learning_rate"],
            "lr_schedule": cell["lr_schedule"],
            "momentum_beta": cell.get("momentum_beta"),
            "adam_beta1": cell.get("adam_beta1"),
            "adam_beta2": cell.get("adam_beta2"),
            "adam_epsilon": cell.get("adam_epsilon"),
            "clipup_momentum": cell.get("clipup_momentum"),
            "clipup_max_speed": cell.get("clipup_max_speed"),
            "runs": len(cell_rows),
            "seeds": ";".join(str(seed) for seed in SEEDS),
        }
        for metric in GROUP_METRICS:
            values = [row.get(metric) for row in cell_rows]
            numeric = np.asarray(
                [float(value) for value in values if value is not None], dtype=float
            )
            if len(numeric) == 0:
                mean = None
                sample_sd = None
            elif len(numeric) != len(cell_rows) or not np.all(np.isfinite(numeric)):
                raise ValueError(f"cell {cell_id} has incomplete metric {metric}")
            else:
                mean = float(np.mean(numeric))
                sample_sd = float(np.std(numeric, ddof=1))
            group[f"{metric}_mean"] = mean
            group[f"{metric}_sample_sd"] = sample_sd
        for metric in (
            "evaluation_generation_auc",
            "final_return",
            "best_return",
        ):
            values = np.asarray([row[metric] for row in cell_rows], dtype=float)
            group[f"{metric}_min"] = float(np.min(values))
            group[f"{metric}_max"] = float(np.max(values))
        groups.append(group)
    return groups


def paired_structured_minus_isotropic(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["condition"]),
            float(row["learning_rate"]),
            str(row["lr_schedule"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    structured_cells = sorted(
        {
            (float(row["learning_rate"]), str(row["lr_schedule"]))
            for row in rows
            if row["condition"] == STRUCTURED
        }
    )
    contrasts: list[dict[str, Any]] = []
    for learning_rate, schedule in structured_cells:
        record: dict[str, Any] = {
            "analysis_designation": "exploratory_descriptive_contrast",
            "exploratory": True,
            "treatment": STRUCTURED,
            "comparator": ISOTROPIC,
            "learning_rate": learning_rate,
            "lr_schedule": schedule,
            "seeds": list(SEEDS),
            "n_pairs": len(SEEDS),
            "inference": "descriptive paired differences only",
            "metrics": {},
        }
        for metric in CONTRAST_METRICS:
            differences: list[float] = []
            for seed in SEEDS:
                structured = by_key[(STRUCTURED, learning_rate, schedule, seed)]
                isotropic = by_key[(ISOTROPIC, learning_rate, schedule, seed)]
                if structured.get(metric) is None or isotropic.get(metric) is None:
                    differences = []
                    break
                differences.append(float(structured[metric]) - float(isotropic[metric]))
            if not differences:
                continue
            values = np.asarray(differences, dtype=float)
            record["metrics"][metric] = {
                "paired_differences": differences,
                "mean_difference": float(np.mean(values)),
                "sample_sd": float(np.std(values, ddof=1)),
                "median_difference": float(np.median(values)),
                "wins": int(np.sum(values > 0.0)),
                "ties": int(np.sum(values == 0.0)),
                "losses": int(np.sum(values < 0.0)),
            }
        contrasts.append(record)
    if len(contrasts) != 6:
        raise ValueError("expected six matched structured/isotropic contrast cells")
    return contrasts


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in RUN_FIELDS}


def build_summary_document(
    rows: Sequence[dict[str, Any]],
    groups: Sequence[dict[str, Any]],
    contrasts: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    *,
    root: str,
    manifest_path: str,
    launcher_path: str,
    run_output: str,
    group_output: str,
    json_output: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": STUDY,
        "analysis_designation": "exploratory_development_screen",
        "exploratory": True,
        "confirmatory_analysis_performed": False,
        "inferential_tests_performed": False,
        "p_values_computed": False,
        "claim_selection_performed": False,
        "interpretation": (
            "Development-only optimizer calibration; descriptive results cannot "
            "support a confirmatory superiority claim."
        ),
        "inputs": {
            "result_root": os.path.abspath(root),
            "config": CONFIG_RELATIVE_PATH,
            "manifest": os.path.abspath(manifest_path),
            "launcher": os.path.abspath(launcher_path),
            "source_sha256": metadata["source_sha256"],
            "manifest_sha256": metadata["manifest_sha256"],
            "launcher_sha256": metadata["launcher_sha256"],
            "slurm_array_job_id": metadata["array_job_id"],
        },
        "design": {
            "seeds": list(SEEDS),
            "manifest_cells": EXPECTED_CELLS,
            "validated_runs": EXPECTED_RUNS,
            "updates_per_run": EXPECTED_ITERATIONS,
            "population_size": POPULATION_SIZE,
            "fresh_candidates_per_update": POPULATION_SIZE,
            "candidate_policy_rollouts_per_run": (
                EXPECTED_ITERATIONS * POPULATION_SIZE
            ),
            "fixed_budget_unit": "candidate_policy_rollouts",
            "environment_transition_budget_equalized": False,
            "transition_sample_efficiency_claim_permitted": False,
            "evaluation_episodes": EVAL_EPISODES,
            "evaluation_interval": EVAL_INTERVAL,
            "evaluation_point_count_including_initial": 27,
            "task_mapping": (
                "slot=task_id//3; seed_index=task_id%3; "
                "cell_id=(slot+11*seed_index)%33"
            ),
        },
        "validated_invariants": {
            "all_manifest_seed_task_mappings_present_once": True,
            "complete_250_record_json_and_jsonl_histories": True,
            "source_manifest_launcher_hash_locks_exact": True,
            "resolved_optimizer_hyperparameters_exact": True,
            "fresh_only": True,
            "replay_disabled": True,
            "importance_sampling_disabled": True,
            "trust_region_disabled": True,
            "parameter_and_gradient_projection_disabled": True,
            "curvature_clipping_disabled": True,
            "picard_iteration_excluded": True,
            "clipup_velocity_clipping_is_baseline_internal": True,
        },
        "metric_definitions": {
            "evaluation_generation_auc": (
                "Trapezoidal return-versus-generation area divided by 250, "
                "using generation 0 initial_eval_reward and only history records "
                "with eval_env_steps_iter > 0."
            ),
            "final_return": "Return at the actual evaluation after generation 250.",
            "best_return": (
                "Maximum over the initial evaluation and actual evaluation rollouts; "
                "carried-forward non-evaluation values are excluded."
            ),
            "training_env_steps": "Fresh candidate rollout steps only.",
            "eval_env_steps": "Initial and online evaluation rollout steps.",
            "total_env_steps": (
                "Observation-normalization calibration plus training plus online "
                "evaluation environment steps."
            ),
            "update_norm_summaries": (
                "Descriptive mean, median, maximum, final, explicit-step, and "
                "implicit-to-explicit norm ratio summaries across 250 updates."
            ),
            "step_over_sigma": (
                "Applied parameter-update norm divided by the fixed perturbation "
                "scale sigma=0.02. Values at or below one are counted by "
                "local_step_fraction; this is a locality diagnostic, not a proof "
                "that a first-order surrogate is accurate."
            ),
            "clipup_clipping_summaries": (
                "ClipUp velocity-clipping frequency, scale, and maximum velocity; "
                "not trust-region clipping and not part of the curvature method."
            ),
            "curvature_reliability_summaries": (
                "Means of independent split-half and lag-one temporal correlation, "
                "sign agreement, and relative disagreement diagnostics."
            ),
            "attenuation_summaries": (
                "Curvature projection activity, step ratio, isotropic scale, norm-match "
                "error, heuristic OLS confidence-adjusted shrinkage activity, and OLS "
                "fit diagnostics. The OLS standard error is not inferentially calibrated."
            ),
            "paired_structured_minus_isotropic": (
                "Seed-paired structured minus norm-matched isotropic differences at "
                "identical learning-rate and schedule cells; descriptive only."
            ),
        },
        "outputs": {
            "validated_run_csv": os.path.abspath(run_output),
            "grouped_csv": os.path.abspath(group_output),
            "summary_json": os.path.abspath(json_output),
            "run_csv_fields": list(RUN_FIELDS),
            "group_csv_fields": list(GROUP_FIELDS),
        },
        "manifest": metadata["manifest"],
        "validated_runs": [_public_row(row) for row in rows],
        "grouped_results": list(groups),
        "paired_structured_minus_isotropic": list(contrasts),
    }


def _stage_csv(path: str, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
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
    groups: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    *,
    run_output: str,
    group_output: str,
    json_output: str,
) -> None:
    staged: list[tuple[str, str]] = []
    try:
        staged.append(
            (
                _stage_csv(
                    run_output,
                    [_public_row(row) for row in rows],
                    RUN_FIELDS,
                ),
                run_output,
            )
        )
        staged.append((_stage_csv(group_output, groups, GROUP_FIELDS), group_output))
        staged.append((_stage_json(json_output, summary), json_output))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            if os.path.exists(temporary):
                os.unlink(temporary)


def summarize(
    root: str,
    *,
    manifest_path: str,
    launcher_path: str,
    expected_source_sha: str,
    expected_manifest_sha: str,
    expected_launcher_sha: str,
    run_output: str,
    group_output: str,
    json_output: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, metadata = validate_and_collect(
        root,
        manifest_path=manifest_path,
        launcher_path=launcher_path,
        expected_source_sha=expected_source_sha,
        expected_manifest_sha=expected_manifest_sha,
        expected_launcher_sha=expected_launcher_sha,
    )
    groups = aggregate_runs(rows, metadata["manifest"])
    contrasts = paired_structured_minus_isotropic(rows)
    summary = build_summary_document(
        rows,
        groups,
        contrasts,
        metadata,
        root=root,
        manifest_path=manifest_path,
        launcher_path=launcher_path,
        run_output=run_output,
        group_output=group_output,
        json_output=json_output,
    )
    write_outputs(
        rows,
        groups,
        summary,
        run_output=run_output,
        group_output=group_output,
        json_output=json_output,
    )
    return rows, groups, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--launcher", default=DEFAULT_LAUNCHER_PATH)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--expected-manifest-sha", required=True)
    parser.add_argument("--expected-launcher-sha", required=True)
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--group-output", default=None)
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    run_output = args.run_output or os.path.join(
        args.root, "validated_development_runs.csv"
    )
    group_output = args.group_output or os.path.join(
        args.root, "development_grouped_summary.csv"
    )
    json_output = args.json_output or os.path.join(
        args.root, "development_summary.json"
    )
    try:
        rows, groups, summary = summarize(
            args.root,
            manifest_path=args.manifest,
            launcher_path=args.launcher,
            expected_source_sha=args.expected_source_sha,
            expected_manifest_sha=args.expected_manifest_sha,
            expected_launcher_sha=args.expected_launcher_sha,
            run_output=run_output,
            group_output=group_output,
            json_output=json_output,
        )
    except (DevelopmentValidationError, ValueError, OSError) as error:
        if isinstance(error, DevelopmentValidationError):
            for issue in error.issues:
                print(f"ERROR: {issue}", file=sys.stderr)
        else:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(
        f"Validated {len(rows)} exploratory development runs in {len(groups)} "
        f"manifest cells; confirmatory_analysis={summary['confirmatory_analysis_performed']}; "
        f"wrote {run_output}, {group_output}, and {json_output}"
    )


if __name__ == "__main__":
    main()

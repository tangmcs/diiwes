#!/usr/bin/env python3
"""Render deterministic paper outputs from the locked analysis JSON only.

The renderer is deliberately downstream of scientific analysis.  It accepts
only the complete schema emitted by the locked lagged-subspace analyzer,
checks all identities and derived decisions, and never reads the manifest,
audit index, rollout fragments, or live scheduler state.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import hmac
import io
import json
import math
import os
import platform
import re
import shutil
import stat
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, ft2font
from PIL import __version__ as PILLOW_VERSION
from PIL import features as pillow_features


SCHEMA_VERSION = 1
STUDY = "lagged_subspace_frozen_checkpoint"
ANALYSIS_DESIGNATION = "preregistered_frozen_checkpoint_mechanism_diagnostic"
CLAIM_BOUNDARY = (
    "frozen_checkpoint_mechanism_only_not_optimizer_or_sample_efficiency"
)
PRIMARY_Q = 0.5
MECHANISM_BOUND_METHOD = (
    "distribution_free_one_sided_median_order_statistics"
)
MECHANISM_FAMILYWISE_ERROR_UPPER_BOUND = 0.015460968017578125
ENDPOINT_FAMILY_ALPHA = 0.034539031982421875
COMBINED_FALSE_ADVANCE_UPPER_BOUND = 0.05
REQUIRED_PASSING_TASK_COUNT = 2
EXPECTED_NUMPY_VERSION = "1.26.4"
EXPECTED_MATPLOTLIB_VERSION = "3.10.5"
GENERATOR_PATH = "scripts/render_lagged_subspace_paper_outputs.py"
INPUT_PATH = "analysis.json"
OUTPUT_DIRECTORY = "paper_outputs"
COMMAND_WORKING_DIRECTORY = "artifact_root"
LAYOUT_CONTRACT = (
    "Run from the artifact directory containing analysis.json; paper_outputs "
    "must be its absent sibling; PYTHON names the locked interpreter and "
    "REPOSITORY_ROOT names the repository root."
)

TASKS = (
    (0, "Hopper-v5"),
    (1, "Walker2d-v5"),
    (2, "HalfCheetah-v5"),
)
METRICS = ("L", "D", "H", "E")
Q_VALUES = (0.25, 0.5, 1.0)
ARMS = ("structured", "isotropic", "explicit", "random")
CONTROLS = ("isotropic", "explicit", "random")
GATE_THRESHOLDS = {
    "L": 1.0,
    "D": 0.01,
    "H": 0.25,
    "E": 0.5,
}
GATE_KEYS = (
    "locality",
    "material_action",
    "high_sample_replication",
    "operational_reliability",
    "directional_endpoint",
    "alpha_calibration_resolved",
    "required_diagnostics_resolved",
    "random_control_valid",
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "study",
    "analysis_designation",
    "primary_q",
    "mechanism_bound_method",
    "mechanism_familywise_error_upper_bound",
    "endpoint_family_alpha",
    "combined_false_advance_upper_bound",
    "top_level_unit",
    "task_results",
    "descriptive_locality",
    "descriptive_return_contrasts",
    "passing_task_count",
    "required_passing_task_count",
    "mechanism_advances_to_optimizer_pilot",
    "claim_boundary",
}
TASK_RESULT_KEYS = {
    "task_index",
    "env_name",
    "seed_mean_contrast",
    "strict_positive_seed_count",
    "strict_tie_seed_count",
    "seed_count",
    "seed_level_probability_of_improvement",
    "raw_one_sided_sign_p",
    "holm_adjusted_one_sided_sign_p",
    "seed_statistics",
    "simultaneous_bounds",
    "gate_conditions",
    "task_pass",
}
BOUND_KEYS = {
    "estimate",
    "one_sided_bound",
    "resolved",
    "order_index_zero_based",
}
LOCALITY_KEYS = {
    "task_index",
    "env_name",
    "q",
    "arm",
    "repeated_measure_count",
    "first_step_over_sigma",
    "mean_step_over_sigma",
    "median_step_over_sigma",
    "percentile_95_step_over_sigma",
    "maximum_step_over_sigma",
    "fraction_at_or_below_0_25",
    "fraction_at_or_below_0_5",
    "fraction_at_or_below_1_0",
    "inference",
}
RETURN_CONTRAST_KEYS = {
    "task_index",
    "env_name",
    "q",
    "contrast",
    "paired_difference_count",
    "training_seed_cluster_count",
    "paired_mean",
    "paired_median",
    "paired_interquartile_mean",
    "paired_checkpoint_partition_episode_probability_of_improvement",
    "seed_cluster_bootstrap_mean_interval_95",
    "bootstrap_seed",
    "multiplicity_role",
}

OUTPUT_SPECS = {
    "table_mechanism_gates.csv": {
        "selectors": [
            "task_results[*].task_index",
            "task_results[*].env_name",
            "task_results[*].simultaneous_bounds",
            "task_results[*].gate_conditions",
            "task_results[*].task_pass",
        ],
        "transformation_id": "mechanism_gates_csv_v1_float10g_na",
        "caption_id": "tab:mechanism-gates",
    },
    "table_mechanism_gates.tex": {
        "selectors": [
            "task_results[*].env_name",
            "task_results[*].simultaneous_bounds",
            "task_results[*].gate_conditions",
            "task_results[*].task_pass",
        ],
        "transformation_id": "mechanism_gates_tex_v1_float6g_dash_yes_no",
        "caption_id": "tab:mechanism-gates",
    },
    "table_endpoint_sign_tests.csv": {
        "selectors": [
            "task_results[*].{task_index,env_name,seed_mean_contrast,"
            "strict_positive_seed_count,strict_tie_seed_count,seed_count,"
            "raw_one_sided_sign_p,holm_adjusted_one_sided_sign_p}",
            "endpoint_family_alpha",
        ],
        "transformation_id": "endpoint_sign_tests_csv_v1_float10g",
        "caption_id": "tab:endpoint-sign-tests",
    },
    "table_endpoint_sign_tests.tex": {
        "selectors": [
            "task_results[*].{env_name,seed_mean_contrast,"
            "strict_positive_seed_count,strict_tie_seed_count,seed_count,"
            "raw_one_sided_sign_p,holm_adjusted_one_sided_sign_p}",
            "endpoint_family_alpha",
        ],
        "transformation_id": "endpoint_sign_tests_tex_v1_float6g",
        "caption_id": "tab:endpoint-sign-tests",
    },
    "figure_mechanism_bounds.pdf": {
        "selectors": [
            "task_results[*].env_name",
            "task_results[*].simultaneous_bounds",
        ],
        "transformation_id": "mechanism_bounds_figure_v1_locked_thresholds",
        "caption_id": "fig:mechanism-bounds",
    },
    "figure_mechanism_bounds.png": {
        "selectors": [
            "task_results[*].env_name",
            "task_results[*].simultaneous_bounds",
        ],
        "transformation_id": "mechanism_bounds_figure_v1_locked_thresholds",
        "caption_id": "fig:mechanism-bounds",
    },
    "figure_endpoint_contrasts.pdf": {
        "selectors": [
            "task_results[*].{env_name,seed_mean_contrast}",
            "descriptive_return_contrasts[?(@.q==0.5)]",
            "descriptive_return_contrasts[*].bootstrap_seed",
        ],
        "transformation_id": "endpoint_contrasts_figure_v1_seed_cluster_bootstrap",
        "caption_id": "fig:endpoint-contrasts",
    },
    "figure_endpoint_contrasts.png": {
        "selectors": [
            "task_results[*].{env_name,seed_mean_contrast}",
            "descriptive_return_contrasts[?(@.q==0.5)]",
            "descriptive_return_contrasts[*].bootstrap_seed",
        ],
        "transformation_id": "endpoint_contrasts_figure_v1_seed_cluster_bootstrap",
        "caption_id": "fig:endpoint-contrasts",
    },
    "figure_locality_sensitivity.pdf": {
        "selectors": ["descriptive_locality[*]"],
        "transformation_id": "locality_sensitivity_figure_v1_mean_and_p95_descriptive",
        "caption_id": "fig:locality-sensitivity",
    },
    "figure_locality_sensitivity.png": {
        "selectors": ["descriptive_locality[*]"],
        "transformation_id": "locality_sensitivity_figure_v1_mean_and_p95_descriptive",
        "caption_id": "fig:locality-sensitivity",
    },
}


class PaperOutputError(RuntimeError):
    """Raised when input or output cannot satisfy the release contract."""


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PaperOutputError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _reproduction_command(input_sha256: str) -> str:
    return (
        f'"$PYTHON" "$REPOSITORY_ROOT/{GENERATOR_PATH}" {INPUT_PATH} '
        f"--output-dir {OUTPUT_DIRECTORY} "
        f"--expected-analysis-sha256 {input_sha256}"
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: Any, label: str) -> float:
    if not _is_number(value) or not math.isfinite(float(value)):
        raise PaperOutputError(f"{label} must be a finite number")
    return float(value)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PaperOutputError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise PaperOutputError(f"{label} must be Boolean")
    return value


def _exact_keys(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PaperOutputError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PaperOutputError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )
    return value


def _exact_float(value: Any, expected: float, label: str) -> None:
    actual = _finite_number(value, label)
    if actual != expected:
        raise PaperOutputError(f"{label} must equal the locked value {expected!r}")


def _close(left: float, right: float, *, tolerance: float = 1e-12) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _binomial_upper_tail(successes: int, trials: int) -> float:
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials)
    )


def _holm_adjust(raw: Sequence[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    adjusted = [1.0] * len(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def _read_analysis(
    path: str | os.PathLike[str], expected_sha256: str
) -> tuple[dict[str, Any], str]:
    expected_sha256 = _validate_sha256(
        expected_sha256, "expected analysis SHA-256"
    )
    if os.path.basename(os.fspath(path)) != INPUT_PATH:
        raise PaperOutputError(f"analysis input must be named {INPUT_PATH}")
    if not hasattr(os, "O_NOFOLLOW"):
        raise PaperOutputError("this release requires O_NOFOLLOW input support")
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise PaperOutputError(
                    "analysis input must be a non-symlink regular file"
                )
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                descriptor = -1
                payload = stream.read()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        actual_sha256 = _sha256_bytes(payload)
        if not hmac.compare_digest(actual_sha256, expected_sha256):
            raise PaperOutputError(
                "analysis SHA-256 does not match the explicit expected digest"
            )

        def reject_duplicate_keys(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key {key!r}")
                result[key] = value
            return result

        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except PaperOutputError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise PaperOutputError(
                "analysis input must be a non-symlink regular file"
            ) from error
        raise PaperOutputError(f"cannot read analysis JSON: {error}") from error
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PaperOutputError(f"cannot parse analysis JSON: {error}") from error
    return dict(_exact_keys(value, TOP_LEVEL_KEYS, "analysis")), actual_sha256


def _validate_bound(value: Any, label: str) -> dict[str, Any]:
    bound = dict(_exact_keys(value, BOUND_KEYS, label))
    resolved = _boolean(bound["resolved"], f"{label}.resolved")
    if resolved:
        estimate = _finite_number(bound["estimate"], f"{label}.estimate")
        one_sided = _finite_number(
            bound["one_sided_bound"], f"{label}.one_sided_bound"
        )
        order_index = _integer(
            bound["order_index_zero_based"],
            f"{label}.order_index_zero_based",
        )
        if order_index not in (3, 16):
            raise PaperOutputError(f"{label} has an unlocked order index")
        bound.update(
            estimate=estimate,
            one_sided_bound=one_sided,
            order_index_zero_based=order_index,
        )
    elif any(
        bound[key] is not None
        for key in ("estimate", "one_sided_bound", "order_index_zero_based")
    ):
        raise PaperOutputError(f"{label} unresolved fields must all be null")
    return bound


def _validate_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    if (
        _integer(analysis["schema_version"], "analysis.schema_version")
        != SCHEMA_VERSION
    ):
        raise PaperOutputError("unsupported analysis schema version")
    identities = {
        "study": STUDY,
        "analysis_designation": ANALYSIS_DESIGNATION,
        "mechanism_bound_method": MECHANISM_BOUND_METHOD,
        "top_level_unit": "training_seed",
        "claim_boundary": CLAIM_BOUNDARY,
    }
    for key, expected in identities.items():
        if analysis[key] != expected:
            raise PaperOutputError(
                f"analysis {key} differs from the locked study identity"
            )
    _exact_float(analysis["primary_q"], PRIMARY_Q, "analysis.primary_q")
    _exact_float(
        analysis["mechanism_familywise_error_upper_bound"],
        MECHANISM_FAMILYWISE_ERROR_UPPER_BOUND,
        "analysis.mechanism_familywise_error_upper_bound",
    )
    _exact_float(
        analysis["endpoint_family_alpha"],
        ENDPOINT_FAMILY_ALPHA,
        "analysis.endpoint_family_alpha",
    )
    _exact_float(
        analysis["combined_false_advance_upper_bound"],
        COMBINED_FALSE_ADVANCE_UPPER_BOUND,
        "analysis.combined_false_advance_upper_bound",
    )
    if analysis["required_passing_task_count"] != REQUIRED_PASSING_TASK_COUNT:
        raise PaperOutputError("required passing-task count differs from the lock")

    raw_tasks = analysis["task_results"]
    if not isinstance(raw_tasks, list) or len(raw_tasks) != len(TASKS):
        raise PaperOutputError("task_results is partial or duplicated")
    tasks: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    supplied_adjusted_p_values: list[float] = []
    for position, (expected_index, expected_env) in enumerate(TASKS):
        label = f"task_results[{position}]"
        task = dict(_exact_keys(raw_tasks[position], TASK_RESULT_KEYS, label))
        task_index = _integer(task["task_index"], f"{label}.task_index")
        if task_index != expected_index or task["env_name"] != expected_env:
            raise PaperOutputError(f"{label} is not in locked task order")
        seed_count = _integer(task["seed_count"], f"{label}.seed_count", minimum=1)
        if seed_count != 20:
            raise PaperOutputError(f"{label}.seed_count differs from the lock")
        wins = _integer(
            task["strict_positive_seed_count"],
            f"{label}.strict_positive_seed_count",
        )
        ties = _integer(
            task["strict_tie_seed_count"], f"{label}.strict_tie_seed_count"
        )
        if wins + ties > seed_count:
            raise PaperOutputError(f"{label} has impossible sign counts")
        contrast = _finite_number(
            task["seed_mean_contrast"], f"{label}.seed_mean_contrast"
        )
        probability = _finite_number(
            task["seed_level_probability_of_improvement"],
            f"{label}.seed_level_probability_of_improvement",
        )
        expected_probability = (wins + 0.5 * ties) / seed_count
        if not _close(probability, expected_probability):
            raise PaperOutputError(f"{label} probability disagrees with sign counts")
        supplied_raw_p = _finite_number(
            task["raw_one_sided_sign_p"], f"{label}.raw_one_sided_sign_p"
        )
        raw_p = _binomial_upper_tail(wins, seed_count)
        if not _close(supplied_raw_p, raw_p):
            raise PaperOutputError(f"{label} raw sign p-value is inconsistent")
        supplied_adjusted_p = _finite_number(
            task["holm_adjusted_one_sided_sign_p"],
            f"{label}.holm_adjusted_one_sided_sign_p",
        )

        seed_statistics = _exact_keys(
            task["seed_statistics"], set(METRICS), f"{label}.seed_statistics"
        )
        normalized_seed_statistics: dict[str, list[float]] = {}
        for metric in METRICS:
            values = seed_statistics[metric]
            if not isinstance(values, list) or len(values) != seed_count:
                raise PaperOutputError(
                    f"{label}.seed_statistics.{metric} is partial"
                )
            normalized_seed_statistics[metric] = [
                _finite_number(value, f"{label}.seed_statistics.{metric}[{index}]")
                for index, value in enumerate(values)
            ]
        bounds = _exact_keys(
            task["simultaneous_bounds"],
            set(METRICS),
            f"{label}.simultaneous_bounds",
        )
        normalized_bounds = {
            metric: _validate_bound(
                bounds[metric], f"{label}.simultaneous_bounds.{metric}"
            )
            for metric in METRICS
        }
        for metric in METRICS:
            bound = normalized_bounds[metric]
            if not bound["resolved"]:
                raise PaperOutputError(
                    f"{label}.simultaneous_bounds.{metric} is unresolved despite "
                    "20 finite seed statistics"
                )
            values = normalized_seed_statistics[metric]
            ordered = sorted(values)
            order_index = 3 if metric == "D" else 16
            expected_estimate = float(np.median(np.asarray(values, dtype=np.float64)))
            if (
                bound["order_index_zero_based"] != order_index
                or not _close(bound["estimate"], expected_estimate)
                or not _close(bound["one_sided_bound"], ordered[order_index])
            ):
                raise PaperOutputError(
                    f"{label}.simultaneous_bounds.{metric} is inconsistent"
                )
            normalized_bounds[metric] = {
                "estimate": expected_estimate,
                "one_sided_bound": float(ordered[order_index]),
                "resolved": True,
                "order_index_zero_based": order_index,
            }

        gates = dict(
            _exact_keys(
                task["gate_conditions"], set(GATE_KEYS), f"{label}.gate_conditions"
            )
        )
        for key in GATE_KEYS:
            gates[key] = _boolean(gates[key], f"{label}.gate_conditions.{key}")
        calibration = gates["alpha_calibration_resolved"]
        expected_bound_gates = {
            "locality": bool(
                calibration
                and normalized_bounds["L"]["resolved"]
                and normalized_bounds["L"]["one_sided_bound"]
                <= GATE_THRESHOLDS["L"]
            ),
            "material_action": bool(
                calibration
                and normalized_bounds["D"]["resolved"]
                and normalized_bounds["D"]["one_sided_bound"]
                > GATE_THRESHOLDS["D"]
            ),
            "high_sample_replication": bool(
                calibration
                and normalized_bounds["H"]["resolved"]
                and normalized_bounds["H"]["one_sided_bound"]
                < GATE_THRESHOLDS["H"]
            ),
            "operational_reliability": bool(
                calibration
                and normalized_bounds["E"]["resolved"]
                and normalized_bounds["E"]["one_sided_bound"]
                < GATE_THRESHOLDS["E"]
            ),
        }
        for key, expected in expected_bound_gates.items():
            if gates[key] != expected:
                raise PaperOutputError(f"{label}.gate_conditions.{key} is inconsistent")
            gates[key] = expected
        supplied_task_pass = _boolean(task["task_pass"], f"{label}.task_pass")
        task.update(
            seed_mean_contrast=contrast,
            seed_level_probability_of_improvement=expected_probability,
            raw_one_sided_sign_p=raw_p,
            holm_adjusted_one_sided_sign_p=supplied_adjusted_p,
            seed_statistics=normalized_seed_statistics,
            simultaneous_bounds=normalized_bounds,
            gate_conditions=gates,
            task_pass=supplied_task_pass,
        )
        tasks.append(task)
        raw_p_values.append(raw_p)
        supplied_adjusted_p_values.append(supplied_adjusted_p)

    expected_adjusted = _holm_adjust(raw_p_values)
    for position, (task, supplied, expected) in enumerate(
        zip(tasks, supplied_adjusted_p_values, expected_adjusted)
    ):
        if not _close(supplied, expected):
            raise PaperOutputError("Holm-adjusted endpoint p-values are inconsistent")
        task["holm_adjusted_one_sided_sign_p"] = expected
        calibration = task["gate_conditions"]["alpha_calibration_resolved"]
        directional_endpoint = bool(
            calibration
            and task["seed_mean_contrast"] > 0.0
            and expected < ENDPOINT_FAMILY_ALPHA
        )
        if task["gate_conditions"]["directional_endpoint"] != directional_endpoint:
            raise PaperOutputError(
                f"task_results[{position}].gate_conditions.directional_endpoint "
                "is inconsistent"
            )
        task["gate_conditions"]["directional_endpoint"] = directional_endpoint
        task_pass = all(task["gate_conditions"].values())
        if task["task_pass"] != task_pass:
            raise PaperOutputError(f"task_results[{position}].task_pass is inconsistent")
        task["task_pass"] = task_pass

    raw_locality = analysis["descriptive_locality"]
    if not isinstance(raw_locality, list) or len(raw_locality) != 36:
        raise PaperOutputError("descriptive_locality is partial or duplicated")
    locality_by_key: dict[tuple[int, float, str], dict[str, Any]] = {}
    numeric_locality_fields = (
        "first_step_over_sigma",
        "mean_step_over_sigma",
        "median_step_over_sigma",
        "percentile_95_step_over_sigma",
        "maximum_step_over_sigma",
        "fraction_at_or_below_0_25",
        "fraction_at_or_below_0_5",
        "fraction_at_or_below_1_0",
    )
    for position, raw in enumerate(raw_locality):
        label = f"descriptive_locality[{position}]"
        row = dict(_exact_keys(raw, LOCALITY_KEYS, label))
        task_index = _integer(row["task_index"], f"{label}.task_index")
        task_map = dict(TASKS)
        if task_index not in task_map or row["env_name"] != task_map[task_index]:
            raise PaperOutputError(f"{label} has an unsupported task identity")
        q = _finite_number(row["q"], f"{label}.q")
        if q not in Q_VALUES or row["arm"] not in ARMS:
            raise PaperOutputError(f"{label} has an unsupported q or arm")
        key = (task_index, q, row["arm"])
        if key in locality_by_key:
            raise PaperOutputError(f"{label} duplicates a locality cell")
        if row["repeated_measure_count"] != 1200:
            raise PaperOutputError(f"{label} repeated-measure count differs from lock")
        for field in numeric_locality_fields:
            row[field] = _finite_number(row[field], f"{label}.{field}")
        if any(row[field] < 0.0 for field in numeric_locality_fields[:5]):
            raise PaperOutputError(f"{label} contains a negative norm summary")
        if not (
            row["median_step_over_sigma"]
            <= row["percentile_95_step_over_sigma"]
            <= row["maximum_step_over_sigma"]
            and row["first_step_over_sigma"] <= row["maximum_step_over_sigma"]
            and row["mean_step_over_sigma"] <= row["maximum_step_over_sigma"]
        ):
            raise PaperOutputError(f"{label} norm summaries are not ordered")
        if any(
            not 0.0 <= row[field] <= 1.0
            for field in (
                "fraction_at_or_below_0_25",
                "fraction_at_or_below_0_5",
                "fraction_at_or_below_1_0",
            )
        ):
            raise PaperOutputError(f"{label} contains an invalid fraction")
        if not (
            row["fraction_at_or_below_0_25"]
            <= row["fraction_at_or_below_0_5"]
            <= row["fraction_at_or_below_1_0"]
        ):
            raise PaperOutputError(f"{label} empirical fractions are not ordered")
        if row["inference"] != "descriptive_repeated_measures_only":
            raise PaperOutputError(f"{label} contains unsupported inference")
        locality_by_key[key] = row
    expected_locality_keys = {
        (task_index, q, arm)
        for task_index, _ in TASKS
        for q in Q_VALUES
        for arm in ARMS
    }
    if set(locality_by_key) != expected_locality_keys:
        raise PaperOutputError("descriptive_locality cell set is incomplete")

    raw_returns = analysis["descriptive_return_contrasts"]
    if not isinstance(raw_returns, list) or len(raw_returns) != 27:
        raise PaperOutputError("descriptive_return_contrasts is partial or duplicated")
    returns_by_key: dict[tuple[int, float, str], dict[str, Any]] = {}
    bootstrap_seeds: set[int] = set()
    return_numeric_fields = (
        "paired_mean",
        "paired_median",
        "paired_interquartile_mean",
        "paired_checkpoint_partition_episode_probability_of_improvement",
    )
    for position, raw in enumerate(raw_returns):
        label = f"descriptive_return_contrasts[{position}]"
        row = dict(_exact_keys(raw, RETURN_CONTRAST_KEYS, label))
        task_index = _integer(row["task_index"], f"{label}.task_index")
        task_map = dict(TASKS)
        if task_index not in task_map or row["env_name"] != task_map[task_index]:
            raise PaperOutputError(f"{label} has an unsupported task identity")
        q = _finite_number(row["q"], f"{label}.q")
        expected_prefix = "structured_minus_"
        contrast_name = row["contrast"]
        if (
            q not in Q_VALUES
            or not isinstance(contrast_name, str)
            or not contrast_name.startswith(expected_prefix)
            or contrast_name[len(expected_prefix) :] not in CONTROLS
        ):
            raise PaperOutputError(f"{label} has an unsupported q or contrast")
        control = contrast_name[len(expected_prefix) :]
        key = (task_index, q, control)
        if key in returns_by_key:
            raise PaperOutputError(f"{label} duplicates a return-contrast cell")
        if row["paired_difference_count"] != 12000:
            raise PaperOutputError(f"{label} paired count differs from the lock")
        if row["training_seed_cluster_count"] != 20:
            raise PaperOutputError(f"{label} cluster count differs from the lock")
        for field in return_numeric_fields:
            row[field] = _finite_number(row[field], f"{label}.{field}")
        probability = row[
            "paired_checkpoint_partition_episode_probability_of_improvement"
        ]
        if not 0.0 <= probability <= 1.0:
            raise PaperOutputError(f"{label} probability is outside [0, 1]")
        interval = row["seed_cluster_bootstrap_mean_interval_95"]
        if not isinstance(interval, list) or len(interval) != 2:
            raise PaperOutputError(f"{label} bootstrap interval is malformed")
        interval = [
            _finite_number(value, f"{label}.bootstrap_interval[{index}]")
            for index, value in enumerate(interval)
        ]
        if interval[0] > interval[1]:
            raise PaperOutputError(f"{label} bootstrap interval is reversed")
        row["seed_cluster_bootstrap_mean_interval_95"] = interval
        row["bootstrap_seed"] = _integer(
            row["bootstrap_seed"], f"{label}.bootstrap_seed"
        )
        bootstrap_seeds.add(row["bootstrap_seed"])
        expected_role = (
            "primary_holm_family"
            if q == PRIMARY_Q and control == "isotropic"
            else (
                "secondary_no_p_value_reported"
                if q == PRIMARY_Q
                else "descriptive_sensitivity_no_p_value"
            )
        )
        if row["multiplicity_role"] != expected_role:
            raise PaperOutputError(f"{label} multiplicity role is unsupported")
        returns_by_key[key] = row
    expected_return_keys = {
        (task_index, q, control)
        for task_index, _ in TASKS
        for q in Q_VALUES
        for control in CONTROLS
    }
    if set(returns_by_key) != expected_return_keys:
        raise PaperOutputError("descriptive return-contrast cell set is incomplete")
    if len(bootstrap_seeds) != 1:
        raise PaperOutputError(
            "descriptive return contrasts must carry one shared bootstrap seed "
            "for the protocol-required common resample-index matrix"
        )
    for task in tasks:
        primary = returns_by_key[(task["task_index"], PRIMARY_Q, "isotropic")]
        if not _close(task["seed_mean_contrast"], primary["paired_mean"]):
            raise PaperOutputError(
                "primary endpoint mean disagrees across analysis sections"
            )

    passing = sum(task["task_pass"] for task in tasks)
    if (
        _integer(
            analysis["passing_task_count"], "analysis.passing_task_count"
        )
        != passing
    ):
        raise PaperOutputError("passing_task_count is inconsistent")
    advances = _boolean(
        analysis["mechanism_advances_to_optimizer_pilot"],
        "analysis.mechanism_advances_to_optimizer_pilot",
    )
    if advances != (passing >= REQUIRED_PASSING_TASK_COUNT):
        raise PaperOutputError("mechanism advancement decision is inconsistent")
    normalized = dict(analysis)
    normalized.update(
        task_results=tasks,
        descriptive_locality=[
            locality_by_key[(task_index, q, arm)]
            for task_index, _ in TASKS
            for q in Q_VALUES
            for arm in ARMS
        ],
        descriptive_return_contrasts=[
            returns_by_key[(task_index, q, control)]
            for task_index, _ in TASKS
            for q in Q_VALUES
            for control in CONTROLS
        ],
        shared_bootstrap_seed=next(iter(bootstrap_seeds)),
    )
    return normalized


def _format_number(value: float | None, significant_digits: int) -> str:
    if value is None:
        return "NA"
    return format(float(value), f".{significant_digits}g")


def _csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _mechanism_table_csv(analysis: Mapping[str, Any]) -> bytes:
    fields = ["task_index", "env_name"]
    for metric in METRICS:
        fields.extend(
            [
                f"{metric}_estimate",
                f"{metric}_one_sided_bound",
                f"{metric}_threshold",
                f"{metric}_resolved",
            ]
        )
    fields.extend([f"gate_{key}" for key in GATE_KEYS])
    fields.append("task_pass")
    rows = []
    for task in analysis["task_results"]:
        row: dict[str, Any] = {
            "task_index": task["task_index"],
            "env_name": task["env_name"],
        }
        for metric in METRICS:
            bound = task["simultaneous_bounds"][metric]
            row.update(
                {
                    f"{metric}_estimate": _format_number(bound["estimate"], 10),
                    f"{metric}_one_sided_bound": _format_number(
                        bound["one_sided_bound"], 10
                    ),
                    f"{metric}_threshold": _format_number(
                        GATE_THRESHOLDS[metric], 10
                    ),
                    f"{metric}_resolved": str(bound["resolved"]).lower(),
                }
            )
        row.update(
            {
                f"gate_{key}": str(task["gate_conditions"][key]).lower()
                for key in GATE_KEYS
            }
        )
        row["task_pass"] = str(task["task_pass"]).lower()
        rows.append(row)
    return _csv_bytes(fields, rows)


def _mechanism_table_tex(analysis: Mapping[str, Any]) -> bytes:
    gate_headers = ("Loc.", "Mat.", "High", "Oper.", "End.", "Cal.", "Diag.", "Rand.")
    lines = [
        r"\begin{tabular}{lrrrrccccccccc}",
        r"\toprule",
        "Task & L & D & H & E & " + " & ".join(gate_headers) + r" & Pass \\",
        r"\midrule",
    ]
    for task in analysis["task_results"]:
        values = [_tex_escape(task["env_name"])]
        for metric in METRICS:
            value = task["simultaneous_bounds"][metric]["one_sided_bound"]
            values.append("--" if value is None else _format_number(value, 6))
        values.extend(
            "Yes" if task["gate_conditions"][key] else "No" for key in GATE_KEYS
        )
        values.append("Yes" if task["task_pass"] else "No")
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "% L, D, H, and E are preregistered simultaneous one-sided bounds.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _endpoint_table_csv(analysis: Mapping[str, Any]) -> bytes:
    fields = [
        "task_index",
        "env_name",
        "seed_mean_contrast",
        "strict_positive_seed_count",
        "strict_tie_seed_count",
        "seed_count",
        "raw_one_sided_sign_p",
        "holm_adjusted_one_sided_sign_p",
        "locked_alpha_threshold",
    ]
    rows = []
    for task in analysis["task_results"]:
        rows.append(
            {
                "task_index": task["task_index"],
                "env_name": task["env_name"],
                "seed_mean_contrast": _format_number(task["seed_mean_contrast"], 10),
                "strict_positive_seed_count": task["strict_positive_seed_count"],
                "strict_tie_seed_count": task["strict_tie_seed_count"],
                "seed_count": task["seed_count"],
                "raw_one_sided_sign_p": _format_number(
                    task["raw_one_sided_sign_p"], 10
                ),
                "holm_adjusted_one_sided_sign_p": _format_number(
                    task["holm_adjusted_one_sided_sign_p"], 10
                ),
                "locked_alpha_threshold": _format_number(
                    analysis["endpoint_family_alpha"], 10
                ),
            }
        )
    return _csv_bytes(fields, rows)


def _endpoint_table_tex(analysis: Mapping[str, Any]) -> bytes:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Task & Mean contrast & Wins & Ties & Seeds & Raw $p$ & Holm $p$ \\",
        r"\midrule",
    ]
    for task in analysis["task_results"]:
        values = [
            _tex_escape(task["env_name"]),
            _format_number(task["seed_mean_contrast"], 6),
            str(task["strict_positive_seed_count"]),
            str(task["strict_tie_seed_count"]),
            str(task["seed_count"]),
            _format_number(task["raw_one_sided_sign_p"], 6),
            _format_number(task["holm_adjusted_one_sided_sign_p"], 6),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.extend(
        [
            r"\midrule",
            r"\multicolumn{7}{l}{Locked one-sided family threshold: "
            + _format_number(analysis["endpoint_family_alpha"], 6)
            + r".} \\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _configure_plotting() -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "pdf.compression": 9,
            "savefig.dpi": 180,
        }
    )


def _runtime_provenance() -> dict[str, str | None]:
    font_path = font_manager.findfont("DejaVu Sans", fallback_to_default=False)
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": os.path.basename(sys.executable),
        "system": platform.system(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "matplotlib_backend": str(matplotlib.get_backend()),
        "freetype": ft2font.__freetype_version__,
        "dejavu_sans_font_file": os.path.basename(font_path),
        "dejavu_sans_font_sha256": _sha256_file(font_path),
        "pillow": PILLOW_VERSION,
        "libpng": pillow_features.version("libpng"),
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
    }


def _save_figure_pair(
    figure: matplotlib.figure.Figure,
    directory: str,
    stem: str,
) -> None:
    pdf_path = os.path.join(directory, f"{stem}.pdf")
    png_path = os.path.join(directory, f"{stem}.png")
    pdf_metadata = {
        "Title": None,
        "Author": None,
        "Subject": None,
        "Keywords": None,
        "Creator": None,
        "Producer": None,
        "CreationDate": None,
        "ModDate": None,
    }
    png_metadata = {
        "Software": None,
        "Author": None,
        "Creation Time": None,
    }
    figure.savefig(pdf_path, format="pdf", metadata=pdf_metadata)
    figure.savefig(png_path, format="png", metadata=png_metadata)
    plt.close(figure)


def _mechanism_bounds_figure(analysis: Mapping[str, Any]) -> matplotlib.figure.Figure:
    labels = [task["env_name"].replace("-v5", "") for task in analysis["task_results"]]
    x = np.arange(len(labels), dtype=np.float64)
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.0), squeeze=False)
    metric_titles = {
        "L": "L: locality",
        "D": "D: material action",
        "H": "H: high-sample error",
        "E": "E: operational error",
    }
    for axis, metric in zip(axes[0], METRICS):
        values = [
            task["simultaneous_bounds"][metric]["one_sided_bound"]
            for task in analysis["task_results"]
        ]
        estimates = [
            task["simultaneous_bounds"][metric]["estimate"]
            for task in analysis["task_results"]
        ]
        threshold = GATE_THRESHOLDS[metric]
        axis.axhline(
            threshold,
            color="#B22222",
            linestyle="--",
            linewidth=1.2,
            label="Locked threshold",
        )
        for index, (bound, estimate) in enumerate(zip(values, estimates)):
            if bound is None:
                axis.scatter(
                    [index],
                    [threshold],
                    marker="x",
                    color="#B22222",
                    zorder=3,
                    label="Unresolved" if index == 0 else None,
                )
            else:
                axis.scatter(
                    [index],
                    [estimate],
                    marker="o",
                    facecolors="none",
                    edgecolors="#666666",
                    s=28,
                    zorder=3,
                    label="Seed median" if index == 0 else None,
                )
                axis.scatter(
                    [index],
                    [bound],
                    marker="D",
                    color="#1F5A94",
                    s=30,
                    zorder=4,
                    label="One-sided bound" if index == 0 else None,
                )
        axis.set_title(metric_titles[metric])
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        direction = ">" if metric == "D" else ("<=" if metric == "L" else "<")
        axis.set_xlabel(f"Pass direction: bound {direction} threshold")
    axes[0, 0].set_ylabel("Metric value")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=max(1, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.26, top=0.78, wspace=0.38)
    return figure


def _endpoint_contrasts_figure(
    analysis: Mapping[str, Any],
) -> matplotlib.figure.Figure:
    rows = {
        (row["task_index"], row["q"], row["contrast"].removeprefix("structured_minus_")): row
        for row in analysis["descriptive_return_contrasts"]
    }
    labels = [env.replace("-v5", "") for _, env in TASKS]
    x = np.arange(len(TASKS), dtype=np.float64)
    offsets = {"isotropic": -0.18, "explicit": 0.0, "random": 0.18}
    styles = {
        "isotropic": ("#B22222", "o", "Structured - isotropic (primary)"),
        "explicit": ("#1F5A94", "s", "Structured - explicit (descriptive)"),
        "random": ("#555555", "^", "Structured - random (descriptive)"),
    }
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.axhline(0.0, color="#222222", linewidth=0.9)
    for control in CONTROLS:
        color, marker, label = styles[control]
        means = []
        interval_lower = []
        interval_upper = []
        for task_index, _ in TASKS:
            row = rows[(task_index, PRIMARY_Q, control)]
            mean = (
                analysis["task_results"][task_index]["seed_mean_contrast"]
                if control == "isotropic"
                else row["paired_mean"]
            )
            interval = row["seed_cluster_bootstrap_mean_interval_95"]
            means.append(mean)
            interval_lower.append(interval[0])
            interval_upper.append(interval[1])
        face = color if control == "isotropic" else "white"
        positions = x + offsets[control]
        axis.vlines(
            positions,
            interval_lower,
            interval_upper,
            color=color,
            linewidth=1.3,
        )
        cap_half_width = 0.025
        axis.hlines(
            interval_lower,
            positions - cap_half_width,
            positions + cap_half_width,
            color=color,
            linewidth=1.3,
        )
        axis.hlines(
            interval_upper,
            positions - cap_half_width,
            positions + cap_half_width,
            color=color,
            linewidth=1.3,
        )
        axis.plot(
            positions,
            means,
            linestyle="none",
            marker=marker,
            color=color,
            markerfacecolor=face,
            markeredgecolor=color,
            markersize=5.5,
            label=label,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Paired return contrast")
    axis.set_xlabel("Task (primary q = 0.5)")
    axis.set_title("Seed-cluster means and descriptive 95% bootstrap intervals")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.legend(frameon=False, ncol=1, loc="best")
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.88)
    return figure


def _locality_figure(analysis: Mapping[str, Any]) -> matplotlib.figure.Figure:
    rows = {
        (row["task_index"], row["q"], row["arm"]): row
        for row in analysis["descriptive_locality"]
    }
    colors = {
        "structured": "#B22222",
        "isotropic": "#1F5A94",
        "explicit": "#2E7D32",
        "random": "#555555",
    }
    markers = {"structured": "o", "isotropic": "s", "explicit": "^", "random": "D"}
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, squeeze=False)
    q_array = np.asarray(Q_VALUES, dtype=np.float64)
    for axis, (task_index, env_name) in zip(axes[0], TASKS):
        for arm in ARMS:
            arm_rows = [rows[(task_index, q, arm)] for q in Q_VALUES]
            means = [row["mean_step_over_sigma"] for row in arm_rows]
            p95 = [row["percentile_95_step_over_sigma"] for row in arm_rows]
            axis.plot(
                q_array,
                means,
                color=colors[arm],
                marker=markers[arm],
                label=arm.capitalize(),
            )
            axis.plot(
                q_array,
                p95,
                color=colors[arm],
                linestyle=":",
                linewidth=1.0,
                alpha=0.9,
            )
        axis.set_title(env_name.replace("-v5", ""))
        axis.set_xticks(q_array, ["0.25", "0.5", "1.0"])
        axis.set_xlabel("Locality calibration q")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[0, 0].set_ylabel("Step norm / sigma")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
    )
    figure.text(
        0.5,
        0.015,
        "Descriptive only. Solid: mean; dotted: 95th percentile.",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.80, wspace=0.28)
    return figure


def _write_bytes(path: str, value: bytes) -> None:
    with open(path, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def render_paper_outputs(
    analysis_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    expected_analysis_sha256: str,
) -> dict[str, Any]:
    """Validate one analysis and atomically create a write-once output tree."""

    analysis_path = os.path.abspath(os.fspath(analysis_path))
    output_dir = os.path.abspath(os.fspath(output_dir))
    if os.path.basename(output_dir) != OUTPUT_DIRECTORY:
        raise PaperOutputError(f"output directory must be named {OUTPUT_DIRECTORY}")
    if os.path.dirname(analysis_path) != os.path.dirname(output_dir):
        raise PaperOutputError(
            "analysis.json and paper_outputs must be siblings in one artifact root"
        )
    if os.path.lexists(output_dir):
        raise PaperOutputError("paper output directory already exists; refusing overwrite")
    if np.__version__ != EXPECTED_NUMPY_VERSION:
        raise PaperOutputError(
            f"unsupported NumPy version {np.__version__}; expected {EXPECTED_NUMPY_VERSION}"
        )
    if matplotlib.__version__ != EXPECTED_MATPLOTLIB_VERSION:
        raise PaperOutputError(
            "unsupported Matplotlib version "
            f"{matplotlib.__version__}; expected {EXPECTED_MATPLOTLIB_VERSION}"
        )
    raw_analysis, input_sha256 = _read_analysis(
        analysis_path, expected_analysis_sha256
    )
    analysis = _validate_analysis(raw_analysis)
    generator_file = Path(__file__).resolve()
    generator_sha256 = _sha256_file(generator_file)
    parent = os.path.dirname(output_dir)
    os.makedirs(parent, exist_ok=True)
    staged = tempfile.mkdtemp(prefix=f".{OUTPUT_DIRECTORY}.staging.", dir=parent)
    try:
        _configure_plotting()
        _write_bytes(
            os.path.join(staged, "table_mechanism_gates.csv"),
            _mechanism_table_csv(analysis),
        )
        _write_bytes(
            os.path.join(staged, "table_mechanism_gates.tex"),
            _mechanism_table_tex(analysis),
        )
        _write_bytes(
            os.path.join(staged, "table_endpoint_sign_tests.csv"),
            _endpoint_table_csv(analysis),
        )
        _write_bytes(
            os.path.join(staged, "table_endpoint_sign_tests.tex"),
            _endpoint_table_tex(analysis),
        )
        _save_figure_pair(
            _mechanism_bounds_figure(analysis), staged, "figure_mechanism_bounds"
        )
        _save_figure_pair(
            _endpoint_contrasts_figure(analysis),
            staged,
            "figure_endpoint_contrasts",
        )
        _save_figure_pair(
            _locality_figure(analysis), staged, "figure_locality_sensitivity"
        )

        actual_names = sorted(os.listdir(staged))
        if actual_names != sorted(OUTPUT_SPECS):
            raise PaperOutputError(
                f"renderer produced an incomplete output set: {actual_names}"
            )
        runtime_provenance = _runtime_provenance()
        reproduction_command = _reproduction_command(input_sha256)
        entries = []
        for name in sorted(OUTPUT_SPECS):
            spec = OUTPUT_SPECS[name]
            entries.append(
                {
                    "output_path": f"{OUTPUT_DIRECTORY}/{name}",
                    "output_sha256": _sha256_file(os.path.join(staged, name)),
                    "generator_path": GENERATOR_PATH,
                    "generator_sha256": generator_sha256,
                    "input_path": INPUT_PATH,
                    "input_sha256": input_sha256,
                    "input_sha256_expectation": "explicit_required_argument",
                    "json_selectors": spec["selectors"],
                    "transformation_id": spec["transformation_id"],
                    "command": reproduction_command,
                    "command_working_directory": COMMAND_WORKING_DIRECTORY,
                    "layout_contract": LAYOUT_CONTRACT,
                    "software_versions": {
                        "matplotlib": matplotlib.__version__,
                        "numpy": np.__version__,
                    },
                    "runtime_provenance": runtime_provenance,
                    "caption_id": spec["caption_id"],
                    "claim_boundary": CLAIM_BOUNDARY,
                    "locked_constants": (
                        {"gate_thresholds": GATE_THRESHOLDS}
                        if "mechanism" in name
                        else (
                            {"endpoint_family_alpha": ENDPOINT_FAMILY_ALPHA}
                            if "endpoint_sign_tests" in name
                            else (
                                {
                                    "primary_q": PRIMARY_Q,
                                    "primary_control": "isotropic",
                                    "secondary_controls": ["explicit", "random"],
                                    "shared_bootstrap_seed": analysis[
                                        "shared_bootstrap_seed"
                                    ],
                                    "bootstrap_index_reuse": (
                                        "one_common_resample_index_matrix_across_all_cells"
                                    ),
                                }
                                if "endpoint_contrasts" in name
                                else {
                                    "q_values": list(Q_VALUES),
                                    "arms": list(ARMS),
                                    "summaries": ["mean", "percentile_95"],
                                    "inference": "descriptive_only",
                                }
                            )
                        )
                    ),
                }
            )
        manifest = {
            "schema_version": 1,
            "study": STUDY,
            "analysis_designation": ANALYSIS_DESIGNATION,
            "claim_boundary": CLAIM_BOUNDARY,
            "validated_protocol_invariants": {
                "descriptive_bootstrap": (
                    "one_shared_seed_required_for_one_common_resample_index_matrix"
                )
            },
            "manifest_scope": "ten_rendered_outputs_excluding_this_nonrecursive_manifest",
            "outputs": entries,
        }
        _write_bytes(
            os.path.join(staged, "paper_output_manifest.json"),
            _canonical_json_bytes(manifest),
        )
        os.rename(staged, output_dir)
        staged = ""
        return manifest
    except BaseException:
        if staged:
            shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render locked lagged-subspace paper tables and figures"
    )
    parser.add_argument("analysis", help="validated analysis.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-analysis-sha256", required=True)
    args = parser.parse_args()
    manifest = render_paper_outputs(
        args.analysis,
        args.output_dir,
        expected_analysis_sha256=args.expected_analysis_sha256,
    )
    print(
        f"Rendered {len(manifest['outputs'])} deterministic outputs to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()

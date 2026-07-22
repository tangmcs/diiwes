#!/usr/bin/env python3
"""Measure curvature-estimation error versus sample size and dimension.

This companion to :mod:`experiments.curvature_sample_size.benchmark` keeps the
same synthetic objectives and calls the checked-out
``DIIWES._estimate_fresh_curvature`` implementation.  Unlike the original
single-dimension benchmark, it aggregates coordinate errors online so that
thousands of dimensions do not create millions of Python dictionaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.curvature_sample_size.benchmark import (  # noqa: E402
    CASES,
    CASE_LABELS,
    estimate_with_repository_method,
    linear_coefficients,
    nonlinear_nonconvex,
    nonlinear_parameters,
    nonlinear_smoothed_diagonal_hessian,
)


EXPERIMENT_VERSION = "1.0.0"
RUN_FIELDS = (
    "dimension",
    "case",
    "repetition",
    "pair_count",
    "function_evaluations",
    "pairs_per_dimension",
    "rmse",
    "mae",
    "max_abs_error",
    "mean_error",
    "estimate_norm",
    "relative_l2_error",
    "sign_accuracy",
)
AGGREGATE_FIELDS = (
    "dimension",
    "case",
    "case_label",
    "pair_count",
    "function_evaluations",
    "pairs_per_dimension",
    "repetitions",
    "median_rmse",
    "q25_rmse",
    "q75_rmse",
    "mean_rmse",
    "std_rmse",
    "bias_rmse",
    "median_mae",
    "median_max_abs_error",
    "median_relative_l2_error",
    "mean_sign_accuracy",
    "target_rms",
)
RATE_FIELDS = (
    "dimension",
    "case",
    "case_label",
    "minimum_pair_count",
    "maximum_pair_count",
    "points_in_fit",
    "log_log_slope",
    "slope_standard_error",
    "r_squared",
    "rmse_reduction_factor",
)


@dataclass(frozen=True)
class DimensionSweepConfig:
    """Complete, reproducible high-dimensional sample-size protocol."""

    dimensions: tuple[int, ...] = (100, 1000, 2000)
    sigma: float = 0.1
    pair_counts: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 250, 500, 1000)
    repetitions: int = 20
    linear_noise_std: float = 0.05
    master_seed: int = 20260721
    rate_fit_min_pairs: int = 32
    sign_threshold: float = 0.05

    def validate(self) -> None:
        if not self.dimensions or any(value < 2 for value in self.dimensions):
            raise ValueError("dimensions must be nonempty and at least two")
        if tuple(sorted(set(self.dimensions))) != self.dimensions:
            raise ValueError("dimensions must be strictly increasing and unique")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        if not self.pair_counts or any(value < 2 for value in self.pair_counts):
            raise ValueError("pair_counts must be nonempty and at least two")
        if tuple(sorted(set(self.pair_counts))) != self.pair_counts:
            raise ValueError("pair_counts must be strictly increasing and unique")
        if self.repetitions < 2:
            raise ValueError("repetitions must be at least two")
        if self.linear_noise_std < 0.0 or not np.isfinite(self.linear_noise_std):
            raise ValueError("linear_noise_std must be nonnegative and finite")
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        if self.rate_fit_min_pairs < 2:
            raise ValueError("rate_fit_min_pairs must be at least two")
        if self.sign_threshold < 0.0:
            raise ValueError("sign_threshold must be nonnegative")


@dataclass(frozen=True)
class DimensionSweepResult:
    """In-memory output without per-coordinate Python records."""

    run_metrics: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]
    convergence_rates: tuple[dict[str, Any], ...]
    scaling_model: dict[str, Any]
    summary: dict[str, Any]


def _metric_row(
    dimension: int,
    case: str,
    repetition: int,
    pair_count: int,
    estimate: np.ndarray,
    target: np.ndarray,
    sign_threshold: float,
) -> dict[str, Any]:
    error = np.asarray(estimate) - np.asarray(target)
    target_norm = float(np.linalg.norm(target))
    sign_mask = np.abs(target) >= sign_threshold
    relative = (
        float(np.linalg.norm(error) / target_norm) if target_norm > 1e-15 else None
    )
    sign_accuracy = (
        float(np.mean(np.sign(estimate[sign_mask]) == np.sign(target[sign_mask])))
        if np.any(sign_mask)
        else None
    )
    return {
        "dimension": dimension,
        "case": case,
        "repetition": repetition,
        "pair_count": pair_count,
        "function_evaluations": 2 * pair_count,
        "pairs_per_dimension": pair_count / dimension,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "estimate_norm": float(np.linalg.norm(estimate)),
        "relative_l2_error": relative,
        "sign_accuracy": sign_accuracy,
    }


def _aggregate(
    run_metrics: Sequence[dict[str, Any]],
    error_sums: dict[tuple[int, str, int], np.ndarray],
    targets: dict[tuple[int, str], np.ndarray],
    config: DimensionSweepConfig,
) -> tuple[dict[str, Any], ...]:
    aggregates: list[dict[str, Any]] = []
    for dimension in config.dimensions:
        for case in CASES:
            target = targets[(dimension, case)]
            target_rms = float(np.sqrt(np.mean(target**2)))
            for pair_count in config.pair_counts:
                runs = [
                    row
                    for row in run_metrics
                    if row["dimension"] == dimension
                    and row["case"] == case
                    and row["pair_count"] == pair_count
                ]
                rmse = np.asarray([row["rmse"] for row in runs], dtype=np.float64)
                mae = np.asarray([row["mae"] for row in runs], dtype=np.float64)
                max_abs = np.asarray(
                    [row["max_abs_error"] for row in runs], dtype=np.float64
                )
                relative = np.asarray(
                    [
                        row["relative_l2_error"]
                        for row in runs
                        if row["relative_l2_error"] is not None
                    ],
                    dtype=np.float64,
                )
                signs = np.asarray(
                    [
                        row["sign_accuracy"]
                        for row in runs
                        if row["sign_accuracy"] is not None
                    ],
                    dtype=np.float64,
                )
                mean_error = error_sums[(dimension, case, pair_count)] / len(runs)
                aggregates.append(
                    {
                        "dimension": dimension,
                        "case": case,
                        "case_label": CASE_LABELS[case],
                        "pair_count": pair_count,
                        "function_evaluations": 2 * pair_count,
                        "pairs_per_dimension": pair_count / dimension,
                        "repetitions": len(runs),
                        "median_rmse": float(np.median(rmse)),
                        "q25_rmse": float(np.quantile(rmse, 0.25)),
                        "q75_rmse": float(np.quantile(rmse, 0.75)),
                        "mean_rmse": float(np.mean(rmse)),
                        "std_rmse": float(np.std(rmse, ddof=1)),
                        "bias_rmse": float(np.sqrt(np.mean(mean_error**2))),
                        "median_mae": float(np.median(mae)),
                        "median_max_abs_error": float(np.median(max_abs)),
                        "median_relative_l2_error": (
                            float(np.median(relative)) if relative.size else None
                        ),
                        "mean_sign_accuracy": (
                            float(np.mean(signs)) if signs.size else None
                        ),
                        "target_rms": target_rms,
                    }
                )
    return tuple(aggregates)


def _fit_log_log_rate(
    dimension: int,
    case: str,
    aggregates: Sequence[dict[str, Any]],
    config: DimensionSweepConfig,
) -> dict[str, Any]:
    all_rows = [
        row
        for row in aggregates
        if row["dimension"] == dimension and row["case"] == case
    ]
    selected = [
        row
        for row in all_rows
        if row["pair_count"] >= config.rate_fit_min_pairs
        and row["median_rmse"] > 1e-14
    ]
    reduction = (
        float(all_rows[0]["median_rmse"] / all_rows[-1]["median_rmse"])
        if all_rows[-1]["median_rmse"] > 0.0
        else None
    )
    result = {
        "dimension": dimension,
        "case": case,
        "case_label": CASE_LABELS[case],
        "minimum_pair_count": selected[0]["pair_count"] if selected else None,
        "maximum_pair_count": selected[-1]["pair_count"] if selected else None,
        "points_in_fit": len(selected),
        "log_log_slope": None,
        "slope_standard_error": None,
        "r_squared": None,
        "rmse_reduction_factor": reduction,
    }
    if len(selected) < 3:
        return result
    x = np.log([row["pair_count"] for row in selected])
    y = np.log([row["median_rmse"] for row in selected])
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    degrees = len(x) - 2
    covariance = (
        (ss_res / degrees) * np.linalg.inv(design.T @ design)
        if degrees > 0
        else np.full((2, 2), np.nan)
    )
    result.update(
        {
            "log_log_slope": float(coefficients[1]),
            "slope_standard_error": float(np.sqrt(covariance[1, 1])),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        }
    )
    return result


def _fit_dimension_sample_model(
    aggregates: Sequence[dict[str, Any]],
    config: DimensionSweepConfig,
) -> dict[str, Any]:
    """Fit log(RMSE) = intercept + alpha log(d) + beta log(N)."""
    selected = [
        row
        for row in aggregates
        if row["case"] == "nonlinear_nonconvex"
        and row["pair_count"] >= config.rate_fit_min_pairs
        and row["median_rmse"] > 0.0
    ]
    if len(selected) < 4:
        return {
            "formula": "log(RMSE) = intercept + alpha*log(dimension) + beta*log(pair_count)",
            "rows_in_fit": len(selected),
            "intercept": None,
            "dimension_exponent_alpha": None,
            "sample_size_exponent_beta": None,
            "r_squared": None,
        }
    y = np.log([row["median_rmse"] for row in selected])
    design = np.column_stack(
        [
            np.ones(len(selected)),
            np.log([row["dimension"] for row in selected]),
            np.log([row["pair_count"] for row in selected]),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "formula": "log(RMSE) = intercept + alpha*log(dimension) + beta*log(pair_count)",
        "rows_in_fit": len(selected),
        "minimum_pair_count": config.rate_fit_min_pairs,
        "intercept": float(coefficients[0]),
        "dimension_exponent_alpha": float(coefficients[1]),
        "sample_size_exponent_beta": float(coefficients[2]),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
    }


def _build_summary(
    aggregates: Sequence[dict[str, Any]],
    rates: Sequence[dict[str, Any]],
    scaling_model: dict[str, Any],
    config: DimensionSweepConfig,
) -> dict[str, Any]:
    first_pairs = config.pair_counts[0]
    last_pairs = config.pair_counts[-1]
    by_key = {
        (row["dimension"], row["case"], row["pair_count"]): row
        for row in aggregates
    }
    rates_by_key = {
        (row["dimension"], row["case"]): row for row in rates
    }
    dimension_results: dict[str, Any] = {}
    for dimension in config.dimensions:
        first = by_key[(dimension, "nonlinear_nonconvex", first_pairs)]
        last = by_key[(dimension, "nonlinear_nonconvex", last_pairs)]
        rate = rates_by_key[(dimension, "nonlinear_nonconvex")]
        deterministic = [
            row
            for row in aggregates
            if row["dimension"] == dimension
            and row["case"] == "linear_deterministic"
        ]
        dimension_results[str(dimension)] = {
            "nonlinear_median_rmse_first": first["median_rmse"],
            "nonlinear_median_rmse_last": last["median_rmse"],
            "nonlinear_relative_l2_error_last": last["median_relative_l2_error"],
            "nonlinear_sign_accuracy_last": last["mean_sign_accuracy"],
            "nonlinear_log_log_slope": rate["log_log_slope"],
            "nonlinear_rmse_reduction_factor": rate["rmse_reduction_factor"],
            "deterministic_linear_max_median_error": max(
                row["median_max_abs_error"] for row in deterministic
            ),
        }
    return {
        "status": "completed",
        "sample_size_definition": "antithetic perturbation pairs",
        "function_evaluations_per_pair": 2,
        "dimensions": list(config.dimensions),
        "first_pair_count": first_pairs,
        "last_pair_count": last_pairs,
        "dimension_sample_scaling_model": scaling_model,
        "by_dimension": dimension_results,
    }


def run_dimension_sweep(config: DimensionSweepConfig) -> DimensionSweepResult:
    """Run all requested dimensions while aggregating coordinate bias online."""
    config.validate()
    max_pairs = max(config.pair_counts)
    run_metrics: list[dict[str, Any]] = []
    error_sums: dict[tuple[int, str, int], np.ndarray] = {}
    targets: dict[tuple[int, str], np.ndarray] = {}

    for dimension in config.dimensions:
        theta, frequencies = nonlinear_parameters(dimension)
        nonlinear_target = nonlinear_smoothed_diagonal_hessian(
            theta, frequencies, config.sigma
        )
        linear_target = np.zeros(dimension, dtype=np.float64)
        targets[(dimension, "nonlinear_nonconvex")] = nonlinear_target
        targets[(dimension, "linear_deterministic")] = linear_target
        targets[(dimension, "linear_noisy")] = linear_target
        for case in CASES:
            for pair_count in config.pair_counts:
                error_sums[(dimension, case, pair_count)] = np.zeros(
                    dimension, dtype=np.float64
                )

        coefficients = linear_coefficients(dimension)
        for repetition in range(config.repetitions):
            seed_sequence = np.random.SeedSequence(
                [config.master_seed, dimension, repetition]
            )
            perturbation_seed, observation_seed = seed_sequence.spawn(2)
            perturbation_rng = np.random.default_rng(perturbation_seed)
            observation_rng = np.random.default_rng(observation_seed)
            eps_all = perturbation_rng.standard_normal((max_pairs, dimension))

            nonlinear_plus = nonlinear_nonconvex(
                theta + config.sigma * eps_all, frequencies
            )
            nonlinear_minus = nonlinear_nonconvex(
                theta - config.sigma * eps_all, frequencies
            )
            deterministic_plus = (config.sigma * eps_all) @ coefficients
            deterministic_minus = -deterministic_plus
            noisy_plus = deterministic_plus + observation_rng.normal(
                scale=config.linear_noise_std, size=max_pairs
            )
            noisy_minus = deterministic_minus + observation_rng.normal(
                scale=config.linear_noise_std, size=max_pairs
            )
            case_arrays = {
                "nonlinear_nonconvex": (
                    nonlinear_plus,
                    nonlinear_minus,
                    nonlinear_target,
                ),
                "linear_deterministic": (
                    deterministic_plus,
                    deterministic_minus,
                    linear_target,
                ),
                "linear_noisy": (noisy_plus, noisy_minus, linear_target),
            }
            for pair_count in config.pair_counts:
                eps = eps_all[:pair_count]
                for case in CASES:
                    plus, minus, target = case_arrays[case]
                    estimate = estimate_with_repository_method(
                        eps,
                        plus[:pair_count],
                        minus[:pair_count],
                        config.sigma,
                    )
                    run_metrics.append(
                        _metric_row(
                            dimension,
                            case,
                            repetition,
                            pair_count,
                            estimate,
                            target,
                            config.sign_threshold,
                        )
                    )
                    error_sums[(dimension, case, pair_count)] += estimate - target

    aggregates = _aggregate(run_metrics, error_sums, targets, config)
    rates = tuple(
        _fit_log_log_rate(dimension, case, aggregates, config)
        for dimension in config.dimensions
        for case in CASES
    )
    scaling_model = _fit_dimension_sample_model(aggregates, config)
    summary = _build_summary(aggregates, rates, scaling_model, config)
    return DimensionSweepResult(
        run_metrics=tuple(run_metrics),
        aggregates=aggregates,
        convergence_rates=rates,
        scaling_model=scaling_model,
        summary=summary,
    )


def _number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not applicable"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def _chart_rows(
    result: DimensionSweepResult,
    config: DimensionSweepConfig,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    str,
    str,
]:
    """Build chart rows by executing the SQL preserved in report provenance."""
    first_pairs = config.pair_counts[0]
    last_pairs = config.pair_counts[-1]
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE aggregate_results (
            dimension INTEGER,
            case_id TEXT,
            case_label TEXT,
            pair_count INTEGER,
            function_evaluations INTEGER,
            pairs_per_dimension REAL,
            repetitions INTEGER,
            median_rmse REAL,
            q25_rmse REAL,
            q75_rmse REAL,
            median_relative_l2_error REAL,
            mean_sign_accuracy REAL
        )
        """
    )
    connection.executemany(
        "INSERT INTO aggregate_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["dimension"],
                row["case"],
                row["case_label"],
                row["pair_count"],
                row["function_evaluations"],
                row["pairs_per_dimension"],
                row["repetitions"],
                row["median_rmse"],
                row["q25_rmse"],
                row["q75_rmse"],
                row["median_relative_l2_error"],
                row["mean_sign_accuracy"],
            )
            for row in result.aggregates
        ],
    )
    connection.execute(
        """
        CREATE TABLE convergence_rates (
            dimension INTEGER,
            case_id TEXT,
            log_log_slope REAL,
            rmse_reduction_factor REAL
        )
        """
    )
    connection.executemany(
        "INSERT INTO convergence_rates VALUES (?, ?, ?, ?)",
        [
            (
                row["dimension"],
                row["case"],
                row["log_log_slope"],
                row["rmse_reduction_factor"],
            )
            for row in result.convergence_rates
        ],
    )
    absolute_sql = """
SELECT
    dimension,
    case_id AS "case",
    case_label,
    pair_count,
    CAST(pair_count AS TEXT) AS pair_count_label,
    function_evaluations,
    pairs_per_dimension,
    repetitions,
    median_rmse,
    q25_rmse,
    q75_rmse,
    median_relative_l2_error,
    mean_sign_accuracy,
    'd = ' || dimension AS series,
    'solid' AS line_style
FROM aggregate_results
WHERE case_id = 'nonlinear_nonconvex'
ORDER BY pair_count, dimension
""".strip()
    normalized_sql = f"""
WITH baselines AS (
    SELECT dimension, median_rmse AS baseline_rmse
    FROM aggregate_results
    WHERE case_id = 'nonlinear_nonconvex' AND pair_count = {first_pairs}
), observed AS (
    SELECT
        a.dimension,
        a.case_id AS "case",
        a.case_label,
        a.pair_count,
        CAST(a.pair_count AS TEXT) AS pair_count_label,
        a.function_evaluations,
        a.pairs_per_dimension,
        a.repetitions,
        a.median_rmse,
        a.q25_rmse,
        a.q75_rmse,
        a.median_relative_l2_error,
        a.mean_sign_accuracy,
        'd = ' || a.dimension AS series,
        a.median_rmse / b.baseline_rmse AS normalized_rmse,
        'solid' AS line_style
    FROM aggregate_results AS a
    JOIN baselines AS b USING (dimension)
    WHERE a.case_id = 'nonlinear_nonconvex'
), reference AS (
    SELECT DISTINCT
        NULL AS dimension,
        'reference' AS "case",
        'Inverse-square-root reference' AS case_label,
        pair_count,
        CAST(pair_count AS TEXT) AS pair_count_label,
        function_evaluations,
        NULL AS pairs_per_dimension,
        {config.repetitions} AS repetitions,
        NULL AS median_rmse,
        NULL AS q25_rmse,
        NULL AS q75_rmse,
        NULL AS median_relative_l2_error,
        NULL AS mean_sign_accuracy,
        'Inverse-square-root reference' AS series,
        sqrt({first_pairs}.0 / pair_count) AS normalized_rmse,
        'dotted' AS line_style
    FROM aggregate_results
    WHERE case_id = 'nonlinear_nonconvex'
)
SELECT * FROM observed
UNION ALL
SELECT * FROM reference
ORDER BY pair_count, series
""".strip()
    endpoint_sql = f"""
SELECT
    a.dimension,
    a.pair_count AS largest_pair_count,
    a.median_rmse,
    a.median_relative_l2_error,
    a.mean_sign_accuracy,
    r.log_log_slope,
    r.rmse_reduction_factor
FROM aggregate_results AS a
JOIN convergence_rates AS r
  ON a.dimension = r.dimension AND a.case_id = r.case_id
WHERE a.case_id = 'nonlinear_nonconvex' AND a.pair_count = {last_pairs}
ORDER BY a.dimension
""".strip()
    absolute_rows = [dict(row) for row in connection.execute(absolute_sql)]
    normalized_rows = [dict(row) for row in connection.execute(normalized_sql)]
    endpoint_rows = [dict(row) for row in connection.execute(endpoint_sql)]
    connection.close()
    return (
        absolute_rows,
        normalized_rows,
        endpoint_rows,
        absolute_sql,
        normalized_sql,
        endpoint_sql,
    )


def build_report_artifact(
    result: DimensionSweepResult,
    config: DimensionSweepConfig,
    generated_at: str,
) -> dict[str, Any]:
    """Build a canonical technical report with two dimension-comparison plots."""
    (
        absolute_rows,
        normalized_rows,
        endpoint_rows,
        absolute_sql,
        normalized_sql,
        endpoint_sql,
    ) = _chart_rows(result, config)
    model = result.scaling_model
    dimensions_text = ", ".join(f"{value:,}" for value in config.dimensions)
    first_pairs = config.pair_counts[0]
    last_pairs = config.pair_counts[-1]
    by_dimension = result.summary["by_dimension"]
    slopes = [by_dimension[str(value)]["nonlinear_log_log_slope"] for value in config.dimensions]
    first_dimension = config.dimensions[0]
    last_dimension = config.dimensions[-1]
    first_end = by_dimension[str(first_dimension)]["nonlinear_median_rmse_last"]
    last_end = by_dimension[str(last_dimension)]["nonlinear_median_rmse_last"]

    aggregate_source = {
        "id": "dimension_sweep_aggregate",
        "label": "High-dimensional curvature sweep aggregates",
        "path": "aggregate.csv",
    }
    absolute_source = {
        "id": "dimension_sweep_absolute_query",
        "label": "Absolute high-dimensional curvature error query",
        "path": "aggregate.csv",
        "query": {
            "engine": "sqlite",
            "sql": absolute_sql,
            "description": (
                "Selects nonlinear median coordinate RMSE by dimension and pair count."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": ["aggregate_results"],
            "filters": ["case_id = nonlinear_nonconvex"],
            "metric_definitions": [
                "median_rmse = median across repetitions of within-run coordinate RMSE"
            ],
        },
    }
    normalized_source = {
        "id": "dimension_sweep_normalized_query",
        "label": "Normalized high-dimensional curvature error query",
        "path": "aggregate.csv",
        "query": {
            "engine": "sqlite",
            "sql": normalized_sql,
            "description": (
                "Normalizes nonlinear median RMSE to the smallest pair count within "
                "each dimension and adds an inverse-square-root reference."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": ["aggregate_results"],
            "filters": [
                "case_id = nonlinear_nonconvex",
                f"baseline pair_count = {first_pairs}",
            ],
            "metric_definitions": [
                "normalized_rmse = median_rmse / within-dimension baseline_rmse",
                "inverse-square-root reference = sqrt(smallest pair count / pair count)",
            ],
        },
    }
    endpoint_source = {
        "id": "dimension_sweep_endpoint_query",
        "label": "High-dimensional curvature endpoint and rate query",
        "path": "convergence_rates.csv",
        "query": {
            "engine": "sqlite",
            "sql": endpoint_sql,
            "description": (
                "Joins nonlinear error at the largest pair count to its fitted "
                "convergence rate for each dimension."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": ["aggregate_results", "convergence_rates"],
            "filters": [
                "case_id = nonlinear_nonconvex",
                f"pair_count = {last_pairs}",
            ],
            "metric_definitions": [
                "log_log_slope = OLS slope of log(median_rmse) on log(pair_count)",
                "rmse_reduction_factor = first-pair-count RMSE / last-pair-count RMSE",
            ],
        },
    }
    technical_summary = (
        "## Error falls with sample size, but the same sample budget is less accurate in higher dimensions\n\n"
        f"Across dimensions {dimensions_text}, nonlinear curvature RMSE decreased as the "
        f"number of antithetic pairs increased. The fitted per-dimension log-log slopes "
        f"range from **{_number(min(slopes), 2)}** to **{_number(max(slopes), 2)}**, "
        "so the convergence rate remains close to the inverse-square-root Monte Carlo "
        "benchmark even as dimension grows.\n\n"
        f"The pooled model estimates RMSE proportional to dimension^"
        f"**{_number(model['dimension_exponent_alpha'], 2)}** and pair count^"
        f"**{_number(model['sample_size_exponent_beta'], 2)}** (R²="
        f"**{_number(model['r_squared'], 3)}**). At {last_pairs:,} pairs, median RMSE "
        f"is **{_number(first_end)}** for d={first_dimension:,} and "
        f"**{_number(last_end)}** for d={last_dimension:,}."
    )
    absolute_findings = (
        "## Absolute error rises with dimension at a fixed number of pairs\n\n"
        "Each curve reports median coordinate RMSE against the exact diagonal Hessian "
        f"of the smoothed nonlinear objective. At every tested sample size, the "
        "higher-dimensional problems retain more error. This means a fixed population "
        "size should not be expected to provide dimension-invariant curvature quality."
    )
    normalized_findings = (
        "## The convergence shape remains approximately Monte Carlo-like\n\n"
        f"Normalizing each dimension to its error at {first_pairs} pairs removes the "
        "level difference and isolates the sample-size rate. The three curves track the "
        "inverse-square-root reference closely over the fitted range, so added samples "
        "continue to help at high dimension; they do not erase the higher starting "
        "variance at the same absolute sample count."
    )
    definitions = (
        "## Scope and metric definitions\n\n"
        f"The sweep uses dimensions {dimensions_text}, pair counts "
        f"{', '.join(map(str, config.pair_counts))}, {config.repetitions} independent "
        f"repetitions, and Gaussian perturbation scale sigma={config.sigma}. Sample "
        "size is the number of antithetic pairs, so one pair costs two objective "
        "evaluations. RMSE is computed across coordinates for each repetition and then "
        "summarized by its median across repetitions."
    )
    methodology = (
        "## The estimator and analytic target are unchanged\n\n"
        "The nonlinear objective is the same sine-plus-quartic function with adjacent "
        "bilinear coupling used in the original 12-dimensional benchmark. Its "
        "Gaussian-smoothed diagonal Hessian is analytic. Every estimate calls "
        "`DIIWES._estimate_fresh_curvature` with raw fitness, diagonal mode, antithetic "
        "pairs, and the leave-one-out pair baseline. Common random-number prefixes are "
        "used across pair counts within a repetition. Coordinate bias is accumulated "
        "online rather than stored as per-coordinate records."
    )
    limitations = (
        "## The result is descriptive and local\n\n"
        "The pooled exponents summarize three dimensions, one point, one smoothing "
        "scale, and one synthetic objective; they are not a universal complexity law. "
        f"With {config.repetitions} repetitions, medians and slopes are adequate for a "
        "comparative sweep, but repetition-mean bias estimates are noisier than in the "
        "original 200-repetition study. The experiment evaluates the local diagonal "
        "estimator, not full-matrix recovery or optimizer return."
    )
    next_steps = (
        "## Scale pair count with dimension when curvature quality matters\n\n"
        "Use the absolute-error plot to choose a population budget for the required "
        "curvature accuracy, and report pairs per dimension alongside raw pair count. "
        "For an optimizer study, test budgets proportional to dimension and compare "
        "curvature repeatability, return, and wall-clock cost rather than assuming a "
        "fixed pair count transfers across model sizes."
    )
    further = (
        "## Further questions\n\n"
        "The next useful checks are pair counts above the largest tested dimension, "
        "multiple smoothing scales, additional evaluation points, and noisy nonlinear "
        "objectives. Those tests would show whether the observed dimension exponent is "
        "stable or specific to this objective and sample-size range."
    )

    charts = [
        {
            "id": "absolute_rmse_by_dimension",
            "title": "Nonlinear curvature RMSE by dimension and pair count",
            "subtitle": (
                f"Median coordinate RMSE over {config.repetitions} repetitions; "
                "each pair costs two function evaluations"
            ),
            "showDescription": True,
            "type": "line",
            "intent": "trend",
            "question": "How does absolute curvature error vary with sample size and dimension?",
            "rationale": (
                "A multi-series line chart compares the ordered sample-size response "
                "across three dimensions."
            ),
            "comparisonContext": {
                "baseline": "exact diagonal Hessian of the Gaussian-smoothed objective",
                "denominator": "coordinates within each repetition",
                "grain": "dimension by antithetic pair count",
                "unit": "curvature RMSE",
            },
            "dataset": "absolute_rmse_by_dimension",
            "sourceId": "dimension_sweep_absolute_query",
            "xAxisTitle": "Antithetic pairs (2 evaluations each)",
            "yAxisTitle": "Median coordinate RMSE",
            "encodings": {
                "x": {
                    "field": "pair_count_label",
                    "type": "ordinal",
                    "label": "Antithetic pairs",
                },
                "y": {
                    "field": "median_rmse",
                    "type": "quantitative",
                    "label": "Median RMSE",
                    "format": "number",
                },
                "color": {"field": "series", "type": "nominal", "label": "Dimension"},
                "lineStyle": {
                    "field": "line_style",
                    "type": "nominal",
                    "label": "Line style",
                },
                "tooltip": [
                    {
                        "field": "dimension",
                        "type": "quantitative",
                        "label": "Dimension",
                        "format": "number",
                    },
                    {
                        "field": "median_relative_l2_error",
                        "type": "quantitative",
                        "label": "Median relative L2 error",
                        "format": "number",
                    },
                    {
                        "field": "pairs_per_dimension",
                        "type": "quantitative",
                        "label": "Pairs per dimension",
                        "format": "number",
                    },
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "labels": {"values": "endpoints"},
            "legend": {"position": "bottom", "sort": "spec"},
            "palette": {"kind": "categorical", "name": "blue-orange-neutral"},
            "settings": {"showPoints": "always"},
            "surface": {"surface": "card", "viewMode": "both"},
        },
        {
            "id": "normalized_rmse_by_dimension",
            "title": "Normalized nonlinear curvature RMSE by pair count",
            "subtitle": (
                f"Each dimension equals 1 at {first_pairs} pairs; neutral dotted line "
                "is the inverse-square-root reference"
            ),
            "showDescription": True,
            "type": "line",
            "intent": "trend",
            "question": "Does the sample-size convergence rate change with dimension?",
            "rationale": (
                "Normalization removes level differences so the convergence shapes "
                "can be compared with a Monte Carlo reference."
            ),
            "comparisonContext": {
                "baseline": f"median RMSE at {first_pairs} pairs within each dimension",
                "denominator": "within-dimension smallest-batch median RMSE",
                "grain": "dimension by antithetic pair count",
                "normalization": "divide by within-dimension baseline",
                "unit": "ratio",
            },
            "dataset": "normalized_rmse_by_dimension",
            "sourceId": "dimension_sweep_normalized_query",
            "xAxisTitle": "Antithetic pairs (2 evaluations each)",
            "yAxisTitle": "RMSE / within-dimension baseline RMSE",
            "encodings": {
                "x": {
                    "field": "pair_count_label",
                    "type": "ordinal",
                    "label": "Antithetic pairs",
                },
                "y": {
                    "field": "normalized_rmse",
                    "type": "quantitative",
                    "label": "Normalized RMSE",
                    "format": "number",
                },
                "color": {"field": "series", "type": "nominal", "label": "Series"},
                "lineStyle": {
                    "field": "line_style",
                    "type": "nominal",
                    "label": "Line style",
                },
                "tooltip": [
                    {
                        "field": "pair_count",
                        "type": "quantitative",
                        "label": "Antithetic pairs",
                        "format": "number",
                    },
                    {
                        "field": "function_evaluations",
                        "type": "quantitative",
                        "label": "Function evaluations",
                        "format": "number",
                    },
                ],
            },
            "valueFormat": "number",
            "layout": "full",
            "labels": {"values": "endpoints"},
            "legend": {"position": "bottom", "sort": "spec"},
            "palette": {"kind": "categorical", "name": "blue-orange-neutral"},
            "settings": {"showPoints": "always"},
            "surface": {"surface": "card", "viewMode": "both"},
        },
    ]
    tables = [
        {
            "id": "dimension_endpoint_table",
            "title": f"Nonlinear results at {last_pairs:,} antithetic pairs",
            "subtitle": "Slope is fitted over the declared high-sample range",
            "showDescription": True,
            "dataset": "dimension_endpoints",
            "sourceId": "dimension_sweep_endpoint_query",
            "defaultSort": {"field": "dimension", "direction": "asc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "dimension", "label": "Dimension", "format": "number"},
                {"field": "largest_pair_count", "label": "Pairs", "format": "number"},
                {"field": "median_rmse", "label": "Median RMSE", "format": "number"},
                {
                    "field": "median_relative_l2_error",
                    "label": "Relative L2 error",
                    "format": "number",
                },
                {
                    "field": "mean_sign_accuracy",
                    "label": "Sign accuracy",
                    "format": "percent",
                },
                {"field": "log_log_slope", "label": "Fitted slope", "format": "number"},
                {
                    "field": "rmse_reduction_factor",
                    "label": "Sweep reduction factor",
                    "format": "number",
                },
            ],
        }
    ]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Curvature sample size across dimensions",
        "description": (
            "Controlled comparison of DIIWES diagonal curvature-estimation error at "
            "100, 1,000, and 2,000 dimensions."
        ),
        "generatedAt": generated_at,
        "charts": charts,
        "tables": tables,
        "sources": [
            aggregate_source,
            absolute_source,
            normalized_source,
            endpoint_source,
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Curvature sample size across dimensions",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": technical_summary,
                "sourceId": "dimension_sweep_aggregate",
            },
            {
                "id": "absolute_findings",
                "type": "markdown",
                "body": absolute_findings,
                "sourceId": "dimension_sweep_aggregate",
            },
            {"id": "absolute_chart", "type": "chart", "chartId": "absolute_rmse_by_dimension"},
            {
                "id": "normalized_findings",
                "type": "markdown",
                "body": normalized_findings,
                "sourceId": "dimension_sweep_aggregate",
            },
            {
                "id": "normalized_chart",
                "type": "chart",
                "chartId": "normalized_rmse_by_dimension",
            },
            {"id": "endpoints", "type": "table", "tableId": "dimension_endpoint_table"},
            {
                "id": "definitions",
                "type": "markdown",
                "body": definitions,
                "sourceId": "dimension_sweep_aggregate",
            },
            {"id": "methodology", "type": "markdown", "body": methodology},
            {"id": "limitations", "type": "markdown", "body": limitations},
            {"id": "next_steps", "type": "markdown", "body": next_steps},
            {"id": "further_questions", "type": "markdown", "body": further},
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "absolute_rmse_by_dimension": absolute_rows,
                "normalized_rmse_by_dimension": normalized_rows,
                "dimension_endpoints": endpoint_rows,
            },
        },
        "sources": [
            aggregate_source,
            absolute_source,
            normalized_source,
            endpoint_source,
        ],
    }


def _source_notes(config: DimensionSweepConfig) -> str:
    return f"""# High-dimensional curvature sweep report notes

## Reporting job

- Question: How does diagonal curvature-estimation error change with antithetic sample size at dimensions {', '.join(map(str, config.dimensions))}?
- Audience: technical.
- Baseline: exact diagonal Hessian of the Gaussian-smoothed nonlinear objective.
- Comparison: absolute and within-dimension-normalized median coordinate RMSE.

## Required-structure map

- Title: title block.
- Technical summary: technical-summary block.
- Key findings with visual evidence: absolute and normalized findings plus two line charts.
- Scope, data, and metric definitions: definitions block.
- Methodology/model specification: methodology block.
- Limitations, uncertainty, and robustness: limitations block and linear controls retained in aggregate.csv.
- Recommended next steps: next-steps block.
- Further questions: further-questions block.

## Chart map

- Absolute-error section: trend / multi-series line; pair count, median nonlinear RMSE, and dimension; shows fixed-budget dimension dependence.
- Normalized-error section: trend / multi-series line; pair count, within-dimension normalized RMSE, dimension, and inverse-square-root reference; shows convergence-shape similarity.
- Palette: blue/orange/neutral roots with line style carrying reference identity.
- Delivery: native charts inside `report.html`.

## Data and calculation notes

- Synthetic functions; no external data source or time window.
- {config.repetitions} independent repetitions with master seed {config.master_seed}.
- Common random-number prefixes across pair counts within each dimension and repetition.
- Sample size is antithetic pairs; total function evaluations are twice the pair count.
- Per-coordinate bias is accumulated online; no multi-million-row coordinate CSV is produced.

## Omission notes

- No uncertainty-band chart: quartiles are retained in `aggregate.csv`, while the two plots prioritize the cross-dimension comparison.
- No parallel static chart: portable HTML is the selected report delivery mode.
"""


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    result: DimensionSweepResult,
    config: DimensionSweepConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write compact evidence, provenance, and the canonical report payload."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    paths = {
        "run_metrics.csv": output / "run_metrics.csv",
        "aggregate.csv": output / "aggregate.csv",
        "convergence_rates.csv": output / "convergence_rates.csv",
        "scaling_model.json": output / "scaling_model.json",
        "summary.json": output / "summary.json",
        "experiment_manifest.json": output / "experiment_manifest.json",
        "artifact.json": output / "artifact.json",
        "source_notes.md": output / "source_notes.md",
    }
    _write_csv(paths["run_metrics.csv"], result.run_metrics, RUN_FIELDS)
    _write_csv(paths["aggregate.csv"], result.aggregates, AGGREGATE_FIELDS)
    _write_csv(paths["convergence_rates.csv"], result.convergence_rates, RATE_FIELDS)
    paths["scaling_model.json"].write_text(
        json.dumps(result.scaling_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["summary.json"].write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_path = Path(__file__).resolve()
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": generated_at,
        "config": asdict(config),
        "cases": list(CASES),
        "case_labels": CASE_LABELS,
        "estimator_contract": {
            "implementation": "core.DIIWES._estimate_fresh_curvature",
            "curvature_fitness": "raw",
            "curvature_mode": "diag",
            "pairing": "antithetic",
            "baseline": "leave_one_out_pair_sum",
            "target": "diagonal Hessian of Gaussian-smoothed objective",
        },
        "memory_contract": {
            "coordinate_storage": "online error sums only",
            "reason": "avoid dimension-by-repetition Python coordinate records",
        },
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "outputs": list(paths) + ["report.html"],
    }
    paths["experiment_manifest.json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    artifact = build_report_artifact(result, config, generated_at)
    paths["artifact.json"].write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["source_notes.md"].write_text(_source_notes(config), encoding="utf-8")
    return paths


def _parse_positive_ints(value: str, label: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"at least one {label[:-1]} is required")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/curvature_sample_size/dimension_sweep",
        help="Directory for compact CSV, JSON, notes, and report payloads.",
    )
    parser.add_argument(
        "--dimensions",
        type=lambda value: _parse_positive_ints(value, "dimensions"),
        default=(100, 1000, 2000),
    )
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument(
        "--pair-counts",
        type=lambda value: _parse_positive_ints(value, "pair counts"),
        default=(4, 8, 16, 32, 64, 128, 250, 500, 1000),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--linear-noise-std", type=float, default=0.05)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--rate-fit-min-pairs", type=int, default=32)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke protocol: dimensions 10,20; pair counts 4,8,16; 3 repetitions.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dimensions = args.dimensions
    pair_counts = args.pair_counts
    repetitions = args.repetitions
    rate_fit_min_pairs = args.rate_fit_min_pairs
    if args.quick:
        dimensions = (10, 20)
        pair_counts = (4, 8, 16)
        repetitions = 3
        rate_fit_min_pairs = 4
    config = DimensionSweepConfig(
        dimensions=tuple(dimensions),
        sigma=args.sigma,
        pair_counts=tuple(pair_counts),
        repetitions=repetitions,
        linear_noise_std=args.linear_noise_std,
        master_seed=args.master_seed,
        rate_fit_min_pairs=rate_fit_min_pairs,
    )
    result = run_dimension_sweep(config)
    outputs = write_outputs(result, config, args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"Wrote {len(outputs)} source artifacts to {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

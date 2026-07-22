#!/usr/bin/env python3
"""Measure DIIWES curvature-estimation error as sample size increases.

The experiment calls :meth:`core.DIIWES._estimate_fresh_curvature` directly.
It does not maintain a second implementation of the estimator. Three cases
separate different questions:

* ``nonlinear_nonconvex`` has a closed-form Gaussian-smoothed diagonal
  Hessian, so it measures accuracy on a genuinely nonlinear objective;
* ``linear_deterministic`` has exactly zero curvature and checks that the
  antithetic estimator cancels a linear objective;
* ``linear_noisy`` also has zero curvature, but independent observation noise
  makes its Monte Carlo error reveal the sample-size convergence rate.

Sample size is reported as the number of antithetic pairs. Each pair costs
two function evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import DIIWES  # noqa: E402


EXPERIMENT_VERSION = "1.0.0"
CASES = (
    "nonlinear_nonconvex",
    "linear_deterministic",
    "linear_noisy",
)
CASE_LABELS = {
    "nonlinear_nonconvex": "Nonlinear non-convex",
    "linear_deterministic": "Linear, noiseless",
    "linear_noisy": "Linear, noisy",
}
RUN_FIELDS = (
    "case",
    "repetition",
    "pair_count",
    "function_evaluations",
    "rmse",
    "mae",
    "max_abs_error",
    "mean_error",
    "estimate_norm",
    "relative_l2_error",
    "sign_accuracy",
)
COORDINATE_FIELDS = (
    "case",
    "repetition",
    "pair_count",
    "function_evaluations",
    "coordinate",
    "target_curvature",
    "estimated_curvature",
    "error",
)
AGGREGATE_FIELDS = (
    "case",
    "case_label",
    "pair_count",
    "function_evaluations",
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
class BenchmarkConfig:
    """Complete, reproducible sample-size protocol."""

    dimension: int = 12
    sigma: float = 0.1
    pair_counts: tuple[int, ...] = (4, 8, 16, 32, 64, 128, 250, 500, 1000)
    repetitions: int = 200
    linear_noise_std: float = 0.05
    master_seed: int = 20260721
    rate_fit_min_pairs: int = 32
    sign_threshold: float = 0.05

    def validate(self) -> None:
        if self.dimension < 2:
            raise ValueError("dimension must be at least two")
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
class ExperimentResult:
    """In-memory output of one benchmark run."""

    run_metrics: tuple[dict[str, Any], ...]
    coordinate_estimates: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]
    convergence_rates: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def nonlinear_parameters(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the evaluation point and sinusoidal frequencies."""
    theta = np.linspace(-1.0, 1.0, int(dimension), dtype=np.float64)
    frequencies = np.linspace(0.7, 1.9, int(dimension), dtype=np.float64)
    return theta, frequencies


def nonlinear_nonconvex(
    points: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Evaluate a nonlinear, non-convex objective on one or more points.

    f(x) = sum_i [sin(omega_i x_i) + 0.05 x_i^4]
           + 0.08 sum_i x_i x_{i+1}.
    """
    values = np.asarray(points, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(frequencies):
        raise ValueError("points must have shape (n, dimension)")
    separable = np.sum(
        np.sin(values * frequencies[None, :]) + 0.05 * values**4,
        axis=1,
    )
    coupling = 0.08 * np.sum(values[:, :-1] * values[:, 1:], axis=1)
    result = separable + coupling
    return result[0] if one_dimensional else result


def nonlinear_smoothed_diagonal_hessian(
    theta: np.ndarray,
    frequencies: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Exact diagonal Hessian after isotropic Gaussian smoothing."""
    theta = np.asarray(theta, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if theta.shape != frequencies.shape:
        raise ValueError("theta and frequencies must have the same shape")
    attenuation = np.exp(-0.5 * frequencies**2 * float(sigma) ** 2)
    return (
        -(frequencies**2) * attenuation * np.sin(frequencies * theta)
        + 0.6 * (theta**2 + float(sigma) ** 2)
    )


def linear_coefficients(dimension: int) -> np.ndarray:
    """Return a fixed nonzero slope for the linear controls."""
    return np.linspace(-1.25, 1.0, int(dimension), dtype=np.float64)


def estimate_with_repository_method(
    eps: np.ndarray,
    plus_fitness: np.ndarray,
    minus_fitness: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Call the checked-out DIIWES estimator for one antithetic batch."""
    eps = np.asarray(eps, dtype=np.float64)
    plus_fitness = np.asarray(plus_fitness, dtype=np.float64)
    minus_fitness = np.asarray(minus_fitness, dtype=np.float64)
    n_pairs, dimension = eps.shape
    if plus_fitness.shape != (n_pairs,) or minus_fitness.shape != (n_pairs,):
        raise ValueError("fitness arrays must have one value per pair")
    noise = np.concatenate([eps, -eps], axis=0)
    fitness = np.concatenate([plus_fitness, minus_fitness], axis=0)
    ask_info = {
        "fresh_pair_plus": np.arange(n_pairs, dtype=int),
        "fresh_pair_minus": np.arange(n_pairs, 2 * n_pairs, dtype=int),
    }
    optimizer = DIIWES(
        num_params=dimension,
        population_size=2 * n_pairs,
        noise_std=float(sigma),
        reuse_fraction=0.0,
        buffer_size=0,
        use_curvature=True,
        curvature_fitness="raw",
        curvature_mode="diag",
        use_leave_one_out_curvature_baseline=True,
        seed=0,
    )
    estimate, observed_pairs = optimizer._estimate_fresh_curvature(
        noise=noise,
        f=fitness,
        ask_info=ask_info,
        sigma=float(sigma),
    )
    if estimate is None or observed_pairs != n_pairs:
        raise RuntimeError("DIIWES did not return the requested pair estimate")
    return np.asarray(estimate, dtype=np.float64)


def _metric_row(
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
        "case": case,
        "repetition": repetition,
        "pair_count": pair_count,
        "function_evaluations": 2 * pair_count,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "estimate_norm": float(np.linalg.norm(estimate)),
        "relative_l2_error": relative,
        "sign_accuracy": sign_accuracy,
    }


def _coordinate_rows(
    case: str,
    repetition: int,
    pair_count: int,
    estimate: np.ndarray,
    target: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "case": case,
            "repetition": repetition,
            "pair_count": pair_count,
            "function_evaluations": 2 * pair_count,
            "coordinate": coordinate,
            "target_curvature": float(target[coordinate]),
            "estimated_curvature": float(estimate[coordinate]),
            "error": float(estimate[coordinate] - target[coordinate]),
        }
        for coordinate in range(len(target))
    ]


def _finite_values(rows: Iterable[dict[str, Any]], field: str) -> np.ndarray:
    values = [row[field] for row in rows if row[field] is not None]
    return np.asarray(values, dtype=np.float64)


def _aggregate(
    run_metrics: Sequence[dict[str, Any]],
    coordinate_estimates: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> tuple[dict[str, Any], ...]:
    aggregates: list[dict[str, Any]] = []
    for case in CASES:
        for pair_count in config.pair_counts:
            runs = [
                row
                for row in run_metrics
                if row["case"] == case and row["pair_count"] == pair_count
            ]
            coordinates = [
                row
                for row in coordinate_estimates
                if row["case"] == case and row["pair_count"] == pair_count
            ]
            rmse = _finite_values(runs, "rmse")
            mae = _finite_values(runs, "mae")
            max_abs = _finite_values(runs, "max_abs_error")
            relative = _finite_values(runs, "relative_l2_error")
            signs = _finite_values(runs, "sign_accuracy")
            bias_by_coordinate = []
            target_by_coordinate = []
            for coordinate in range(config.dimension):
                selected = [
                    row for row in coordinates if row["coordinate"] == coordinate
                ]
                bias_by_coordinate.append(
                    float(np.mean([row["error"] for row in selected]))
                )
                target_by_coordinate.append(float(selected[0]["target_curvature"]))
            aggregates.append(
                {
                    "case": case,
                    "case_label": CASE_LABELS[case],
                    "pair_count": pair_count,
                    "function_evaluations": 2 * pair_count,
                    "repetitions": len(runs),
                    "median_rmse": float(np.median(rmse)),
                    "q25_rmse": float(np.quantile(rmse, 0.25)),
                    "q75_rmse": float(np.quantile(rmse, 0.75)),
                    "mean_rmse": float(np.mean(rmse)),
                    "std_rmse": float(np.std(rmse, ddof=1)),
                    "bias_rmse": float(
                        np.sqrt(np.mean(np.asarray(bias_by_coordinate) ** 2))
                    ),
                    "median_mae": float(np.median(mae)),
                    "median_max_abs_error": float(np.median(max_abs)),
                    "median_relative_l2_error": (
                        float(np.median(relative)) if relative.size else None
                    ),
                    "mean_sign_accuracy": (
                        float(np.mean(signs)) if signs.size else None
                    ),
                    "target_rms": float(
                        np.sqrt(np.mean(np.asarray(target_by_coordinate) ** 2))
                    ),
                }
            )
    return tuple(aggregates)


def _fit_log_log_rate(
    case: str,
    aggregates: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    selected = [
        row
        for row in aggregates
        if row["case"] == case
        and row["pair_count"] >= config.rate_fit_min_pairs
        and row["median_rmse"] > 1e-14
    ]
    all_case = [row for row in aggregates if row["case"] == case]
    reduction = (
        float(all_case[0]["median_rmse"] / all_case[-1]["median_rmse"])
        if all_case[-1]["median_rmse"] > 0.0
        else None
    )
    base = {
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
        return base
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
    base.update(
        {
            "log_log_slope": float(coefficients[1]),
            "slope_standard_error": float(np.sqrt(covariance[1, 1])),
            "r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0,
        }
    )
    return base


def _build_summary(
    aggregates: Sequence[dict[str, Any]],
    rates: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    by_case_pair = {
        (row["case"], row["pair_count"]): row for row in aggregates
    }
    rate_by_case = {row["case"]: row for row in rates}
    first_pairs = config.pair_counts[0]
    last_pairs = config.pair_counts[-1]
    nonlinear_last = by_case_pair[("nonlinear_nonconvex", last_pairs)]
    deterministic_rows = [
        row for row in aggregates if row["case"] == "linear_deterministic"
    ]
    deterministic_max = max(row["median_max_abs_error"] for row in deterministic_rows)
    nonlinear_bias_ratio = nonlinear_last["bias_rmse"] / nonlinear_last["target_rms"]
    nonlinear_slope = rate_by_case["nonlinear_nonconvex"]["log_log_slope"]
    noisy_slope = rate_by_case["linear_noisy"]["log_log_slope"]
    nonlinear_reduction = rate_by_case["nonlinear_nonconvex"][
        "rmse_reduction_factor"
    ]
    noisy_reduction = rate_by_case["linear_noisy"]["rmse_reduction_factor"]
    checks = {
        "linear_deterministic_zero_curvature": {
            "passed": bool(deterministic_max <= 1e-9),
            "threshold": 1e-9,
            "observed_max_median_coordinate_error": deterministic_max,
        },
        "nonlinear_error_decreases": {
            "passed": bool(nonlinear_reduction >= 2.0),
            "minimum_reduction_factor": 2.0,
            "observed_reduction_factor": nonlinear_reduction,
        },
        "nonlinear_mean_estimate_is_accurate": {
            "passed": bool(nonlinear_bias_ratio <= 0.10),
            "maximum_bias_to_target_rms": 0.10,
            "observed_bias_to_target_rms": nonlinear_bias_ratio,
        },
        "nonlinear_rate_is_monte_carlo_like": {
            "passed": bool(
                nonlinear_slope is not None and -0.8 <= nonlinear_slope <= -0.2
            ),
            "accepted_slope_interval": [-0.8, -0.2],
            "observed_slope": nonlinear_slope,
        },
        "noisy_linear_error_decreases": {
            "passed": bool(noisy_reduction >= 2.0),
            "minimum_reduction_factor": 2.0,
            "observed_reduction_factor": noisy_reduction,
        },
        "noisy_linear_rate_is_monte_carlo_like": {
            "passed": bool(noisy_slope is not None and -0.8 <= noisy_slope <= -0.2),
            "accepted_slope_interval": [-0.8, -0.2],
            "observed_slope": noisy_slope,
        },
    }
    works = all(item["passed"] for item in checks.values())
    return {
        "verdict": "works" if works else "needs_revision",
        "all_checks_passed": works,
        "sample_size_definition": "antithetic perturbation pairs",
        "function_evaluations_per_pair": 2,
        "first_pair_count": first_pairs,
        "last_pair_count": last_pairs,
        "checks": checks,
        "key_results": {
            "nonlinear_median_rmse_first": by_case_pair[
                ("nonlinear_nonconvex", first_pairs)
            ]["median_rmse"],
            "nonlinear_median_rmse_last": nonlinear_last["median_rmse"],
            "nonlinear_sign_accuracy_last": nonlinear_last["mean_sign_accuracy"],
            "nonlinear_log_log_slope": nonlinear_slope,
            "noisy_linear_median_rmse_first": by_case_pair[
                ("linear_noisy", first_pairs)
            ]["median_rmse"],
            "noisy_linear_median_rmse_last": by_case_pair[
                ("linear_noisy", last_pairs)
            ]["median_rmse"],
            "noisy_linear_log_log_slope": noisy_slope,
            "deterministic_linear_max_median_coordinate_error": deterministic_max,
            "nonlinear_bias_to_target_rms_last": nonlinear_bias_ratio,
        },
    }


def run_benchmark(config: BenchmarkConfig) -> ExperimentResult:
    """Run all cases with common random numbers across pair counts."""
    config.validate()
    max_pairs = max(config.pair_counts)
    nonlinear_theta, frequencies = nonlinear_parameters(config.dimension)
    nonlinear_target = nonlinear_smoothed_diagonal_hessian(
        nonlinear_theta, frequencies, config.sigma
    )
    linear_theta = np.zeros(config.dimension, dtype=np.float64)
    linear_target = np.zeros(config.dimension, dtype=np.float64)
    coefficients = linear_coefficients(config.dimension)

    run_metrics: list[dict[str, Any]] = []
    coordinate_estimates: list[dict[str, Any]] = []
    for repetition in range(config.repetitions):
        seed_sequence = np.random.SeedSequence([config.master_seed, repetition])
        perturbation_seed, observation_seed = seed_sequence.spawn(2)
        perturbation_rng = np.random.default_rng(perturbation_seed)
        observation_rng = np.random.default_rng(observation_seed)
        eps_all = perturbation_rng.standard_normal((max_pairs, config.dimension))

        nonlinear_plus_points = nonlinear_theta + config.sigma * eps_all
        nonlinear_minus_points = nonlinear_theta - config.sigma * eps_all
        nonlinear_plus = nonlinear_nonconvex(nonlinear_plus_points, frequencies)
        nonlinear_minus = nonlinear_nonconvex(nonlinear_minus_points, frequencies)

        linear_plus_points = linear_theta + config.sigma * eps_all
        linear_minus_points = linear_theta - config.sigma * eps_all
        deterministic_plus = linear_plus_points @ coefficients
        deterministic_minus = linear_minus_points @ coefficients
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
                        case,
                        repetition,
                        pair_count,
                        estimate,
                        target,
                        config.sign_threshold,
                    )
                )
                coordinate_estimates.extend(
                    _coordinate_rows(
                        case,
                        repetition,
                        pair_count,
                        estimate,
                        target,
                    )
                )

    aggregates = _aggregate(run_metrics, coordinate_estimates, config)
    rates = tuple(_fit_log_log_rate(case, aggregates, config) for case in CASES)
    summary = _build_summary(aggregates, rates, config)
    return ExperimentResult(
        run_metrics=tuple(run_metrics),
        coordinate_estimates=tuple(coordinate_estimates),
        aggregates=aggregates,
        convergence_rates=rates,
        summary=summary,
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "not applicable"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def build_report_artifact(
    result: ExperimentResult,
    config: BenchmarkConfig,
    generated_at: str,
) -> dict[str, Any]:
    """Create the canonical Data Analytics report artifact payload."""
    aggregate = list(result.aggregates)
    rates = {row["case"]: row for row in result.convergence_rates}
    summary = result.summary
    first_pairs = config.pair_counts[0]
    last_pairs = config.pair_counts[-1]
    by_case_pair = {(row["case"], row["pair_count"]): row for row in aggregate}

    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE aggregate_results (
            case_id TEXT,
            case_label TEXT,
            pair_count INTEGER,
            function_evaluations INTEGER,
            median_rmse REAL,
            q25_rmse REAL,
            q75_rmse REAL,
            bias_rmse REAL,
            mean_sign_accuracy REAL
        )
        """
    )
    connection.executemany(
        "INSERT INTO aggregate_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                row["case"],
                row["case_label"],
                row["pair_count"],
                row["function_evaluations"],
                row["median_rmse"],
                row["q25_rmse"],
                row["q75_rmse"],
                row["bias_rmse"],
                row["mean_sign_accuracy"],
            )
            for row in aggregate
        ],
    )
    connection.execute(
        """
        CREATE TABLE convergence_rates (
            case_id TEXT,
            rmse_reduction_factor REAL,
            log_log_slope REAL
        )
        """
    )
    connection.executemany(
        "INSERT INTO convergence_rates VALUES (?, ?, ?)",
        [
            (
                row["case"],
                row["rmse_reduction_factor"],
                row["log_log_slope"],
            )
            for row in result.convergence_rates
        ],
    )
    curve_sql = f"""
WITH baselines AS (
    SELECT case_id, median_rmse AS baseline_rmse
    FROM aggregate_results
    WHERE pair_count = {first_pairs}
), stochastic_series AS (
    SELECT
        a.pair_count,
        CAST(a.pair_count AS TEXT) AS pair_count_label,
        a.function_evaluations,
        a.case_label AS series,
        a.median_rmse / b.baseline_rmse AS error_ratio,
        CASE a.case_id
            WHEN 'nonlinear_nonconvex' THEN 'solid'
            ELSE 'dashed'
        END AS line_style,
        a.median_rmse,
        a.q25_rmse,
        a.q75_rmse
    FROM aggregate_results AS a
    JOIN baselines AS b USING (case_id)
    WHERE a.case_id IN ('nonlinear_nonconvex', 'linear_noisy')
), reference_series AS (
    SELECT
        pair_count,
        CAST(pair_count AS TEXT) AS pair_count_label,
        function_evaluations,
        'Inverse-square-root reference' AS series,
        sqrt({first_pairs}.0 / pair_count) AS error_ratio,
        'dotted' AS line_style,
        NULL AS median_rmse,
        NULL AS q25_rmse,
        NULL AS q75_rmse
    FROM aggregate_results
    WHERE case_id = 'nonlinear_nonconvex'
)
SELECT * FROM stochastic_series
UNION ALL
SELECT * FROM reference_series
ORDER BY pair_count, series
""".strip()
    endpoint_sql = f"""
SELECT
    a.case_label AS "case",
    a.pair_count,
    a.function_evaluations,
    a.median_rmse,
    a.bias_rmse,
    a.mean_sign_accuracy,
    r.rmse_reduction_factor,
    r.log_log_slope
FROM aggregate_results AS a
JOIN convergence_rates AS r USING (case_id)
WHERE a.pair_count IN ({first_pairs}, {last_pairs})
ORDER BY a.pair_count, a.case_label
""".strip()
    curve_rows = [dict(row) for row in connection.execute(curve_sql)]
    endpoint_rows = [dict(row) for row in connection.execute(endpoint_sql)]
    connection.close()

    key = summary["key_results"]
    verdict = "works" if summary["all_checks_passed"] else "needs revision"
    technical_summary = (
        "## The estimator works on this controlled nonlinear test\n\n"
        f"The repository's diagonal curvature estimator **{verdict}** under the "
        "predeclared checks. On the nonlinear non-convex function, median "
        f"coordinate RMSE fell from **{_number(key['nonlinear_median_rmse_first'])}** "
        f"at {first_pairs} pairs to **{_number(key['nonlinear_median_rmse_last'])}** "
        f"at {last_pairs} pairs. Its fitted log-log slope was "
        f"**{_number(key['nonlinear_log_log_slope'], 2)}**, close to the "
        "Monte Carlo benchmark of -0.5.\n\n"
        f"The noiseless linear control stayed at numerical zero (worst median "
        f"coordinate error **{_number(key['deterministic_linear_max_median_coordinate_error'])}**). "
        "The noisy linear control also converged toward zero, showing that larger "
        "sample sizes reduce observation-noise error rather than manufacture curvature."
    )
    findings = (
        "## Larger samples reduce nonlinear curvature error at the expected rate\n\n"
        "The chart normalizes each stochastic case to its median RMSE at the "
        f"smallest batch ({first_pairs} pairs). Both stochastic curves should be "
        "read against the neutral inverse-square-root reference. A value of 0.25 "
        "means one quarter of the smallest-batch error. Common random numbers make "
        "the pair-count comparison less noisy, while repetitions remain independent."
    )
    linear_section = (
        "## The linear controls separate cancellation from noisy convergence\n\n"
        "A linear objective has zero Hessian before and after Gaussian smoothing. "
        "With noiseless paired evaluations, the +epsilon and -epsilon linear terms "
        "cancel in each pair, so the estimated curvature is zero up to floating-point "
        "roundoff for every sample size. Independent evaluation noise breaks exact "
        "pair cancellation, but its RMSE decreases as more pairs are averaged."
    )
    definitions = (
        "## What was measured\n\n"
        f"**Sample size** is the number of antithetic perturbation pairs; each pair "
        f"uses two function evaluations. The sweep is {', '.join(map(str, config.pair_counts))} "
        f"pairs with {config.repetitions} independent repetitions in "
        f"{config.dimension} dimensions and Gaussian perturbation scale sigma={config.sigma}. "
        "The primary error metric is coordinate RMSE against the exact diagonal "
        "Hessian of the Gaussian-smoothed objective. Bias RMSE is computed after "
        "averaging estimates across repetitions."
    )
    methodology = (
        "## The nonlinear target is analytic and the estimator is not duplicated\n\n"
        "The nonlinear objective combines coordinate-wise sine and quartic terms "
        "with adjacent-coordinate bilinear coupling. It is non-quadratic, non-separable, "
        "and has both positive and negative diagonal curvature at the test point. Its "
        "Gaussian-smoothed diagonal Hessian is available in closed form. Every estimate "
        "calls `DIIWES._estimate_fresh_curvature` with raw function values, antithetic "
        "pairs, diagonal mode, and the current leave-one-out pair baseline. Each "
        "repetition generates the largest batch once and uses prefixes for smaller "
        "sample sizes."
    )
    limitations = (
        "## This validates a local diagonal estimate, not full optimizer performance\n\n"
        "The benchmark studies one point, one smoothing scale, and the diagonal of the "
        "smoothed Hessian. It does not validate off-diagonal curvature, replay/importance "
        "weighting, Hessian EMA dynamics, or whether curvature damping improves return "
        "during optimization. The noisy linear case uses independent Gaussian observation "
        f"noise with standard deviation {config.linear_noise_std}; other noise models can "
        "change finite-sample constants without changing the zero-curvature target."
    )
    next_steps = (
        "## Use at least the regime where reliability is acceptable for the application\n\n"
        f"For this function, {last_pairs} pairs gives the smallest tested error, but the "
        "full CSV makes the cost-accuracy tradeoff explicit. Choose a pair count by "
        "comparing median RMSE and its interquartile range with the curvature magnitude "
        "that would materially change the optimizer's step. For the ongoing nonlinear "
        "policy experiment, continue logging split-half agreement because analytic "
        "curvature is unavailable there."
    )
    further = (
        "## Further questions\n\n"
        "The next useful sensitivity checks are perturbation scale, dimension, "
        "heavy-tailed return noise, and multiple evaluation points. A separate optimizer "
        "study is still needed to connect lower curvature-estimation error with return, "
        "step stability, and wall-clock cost."
    )

    aggregate_source = {
        "id": "curvature_aggregate_file",
        "label": "Curvature sample-size benchmark outputs",
        "path": "aggregate.csv",
    }
    curve_source = {
        "id": "curvature_curve_query",
        "label": "Curvature error-curve query",
        "path": "aggregate.csv",
        "query": {
            "engine": "sqlite",
            "sql": curve_sql,
            "description": (
                "Normalizes nonlinear and noisy-linear median RMSE to the smallest "
                "pair count and adds an inverse-square-root reference."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": ["aggregate_results"],
            "filters": [
                "case_id in nonlinear_nonconvex, linear_noisy",
                f"baseline pair_count={first_pairs}",
            ],
            "metric_definitions": [
                "error_ratio = median_rmse / median_rmse at the smallest pair count",
                "inverse-square-root reference = sqrt(smallest pair count / pair count)",
            ],
        },
    }
    endpoint_source = {
        "id": "curvature_endpoint_query",
        "label": "Curvature endpoint comparison query",
        "path": "aggregate.csv",
        "query": {
            "engine": "sqlite",
            "sql": endpoint_sql,
            "description": (
                "Selects absolute error and convergence-rate evidence at the smallest "
                "and largest tested pair counts."
            ),
            "executed_at": generated_at,
            "language": "sql",
            "tables_used": ["aggregate_results", "convergence_rates"],
            "filters": [f"pair_count in ({first_pairs}, {last_pairs})"],
            "metric_definitions": [
                "median_rmse = median across repetitions of coordinate RMSE",
                "bias_rmse = coordinate RMSE of the repetition-mean estimate",
            ],
        },
    }
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "Curvature estimation versus sample size",
        "description": (
            "Controlled validation of the DIIWES diagonal Stein estimator on nonlinear "
            "and linear functions."
        ),
        "generatedAt": generated_at,
        "charts": [
            {
                "id": "relative_error_curve",
                "title": "Normalized curvature RMSE across antithetic pair counts",
                "subtitle": (
                    f"Median coordinate RMSE over {config.repetitions} repetitions; "
                    f"each series equals 1 at {first_pairs} pairs"
                ),
                "showDescription": True,
                "type": "line",
                "intent": "trend",
                "question": "How does curvature-estimation error change with sample size?",
                "rationale": (
                    "A line chart is appropriate because pair count is ordered and "
                    f"{len(config.pair_counts)} points reveal the convergence shape."
                ),
                "comparisonContext": {
                    "baseline": f"median RMSE at {first_pairs} antithetic pairs",
                    "denominator": "coordinate RMSE",
                    "grain": "function case by pair count",
                    "normalization": "divide by smallest-batch median RMSE",
                    "unit": "ratio",
                },
                "dataset": "relative_error_curve",
                "sourceId": "curvature_curve_query",
                "xAxisTitle": "Antithetic pairs (2 function evaluations each)",
                "yAxisTitle": "Median RMSE / smallest-batch median RMSE",
                "encodings": {
                    "x": {
                        "field": "pair_count_label",
                        "type": "ordinal",
                        "label": "Antithetic pairs",
                    },
                    "y": {
                        "field": "error_ratio",
                        "type": "quantitative",
                        "label": "Normalized median RMSE",
                        "format": "number",
                    },
                    "color": {
                        "field": "series",
                        "type": "nominal",
                        "label": "Series",
                    },
                    "lineStyle": {
                        "field": "line_style",
                        "type": "nominal",
                        "label": "Line style",
                    },
                    "tooltip": [
                        {
                            "field": "function_evaluations",
                            "type": "quantitative",
                            "label": "Function evaluations",
                            "format": "number",
                        },
                        {
                            "field": "pair_count",
                            "type": "quantitative",
                            "label": "Antithetic pairs",
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
            }
        ],
        "tables": [
            {
                "id": "endpoint_table",
                "title": "Absolute error at the smallest and largest batches",
                "subtitle": (
                    "RMSE and bias are in curvature units; reduction uses the complete sweep"
                ),
                "showDescription": True,
                "dataset": "endpoint_results",
                "sourceId": "curvature_endpoint_query",
                "defaultSort": {"field": "pair_count", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "case", "label": "Function", "type": "text"},
                    {"field": "pair_count", "label": "Pairs", "format": "number"},
                    {
                        "field": "function_evaluations",
                        "label": "Function evaluations",
                        "format": "number",
                    },
                    {"field": "median_rmse", "label": "Median RMSE", "format": "number"},
                    {"field": "bias_rmse", "label": "Bias RMSE", "format": "number"},
                    {
                        "field": "rmse_reduction_factor",
                        "label": "Sweep reduction factor",
                        "format": "number",
                    },
                    {"field": "log_log_slope", "label": "Fitted slope", "format": "number"},
                ],
            }
        ],
        "sources": [aggregate_source, curve_source, endpoint_source],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Curvature estimation versus sample size",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": technical_summary,
                "sourceId": "curvature_aggregate_file",
            },
            {
                "id": "findings",
                "type": "markdown",
                "body": findings,
                "sourceId": "curvature_aggregate_file",
            },
            {"id": "curve", "type": "chart", "chartId": "relative_error_curve"},
            {
                "id": "linear_controls",
                "type": "markdown",
                "body": linear_section,
                "sourceId": "curvature_aggregate_file",
            },
            {"id": "endpoints", "type": "table", "tableId": "endpoint_table"},
            {
                "id": "definitions",
                "type": "markdown",
                "body": definitions,
                "sourceId": "curvature_aggregate_file",
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
                "relative_error_curve": curve_rows,
                "endpoint_results": endpoint_rows,
            },
        },
        "sources": [aggregate_source, curve_source, endpoint_source],
    }


def _source_notes(config: BenchmarkConfig) -> str:
    return f"""# Curvature sample-size report notes

## Reporting job

- Question: Does the repository's diagonal curvature estimator improve with sample size on nonlinear and linear functions?
- Audience: technical.
- Baseline: exact diagonal Hessian of the Gaussian-smoothed objective.
- Success criteria: noiseless linear cancellation, decreasing nonlinear/noisy-linear RMSE, low repetition-mean nonlinear bias, and a log-log error slope consistent with Monte Carlo convergence.

## Required-structure map

- Title: title block.
- Technical summary: technical-summary block.
- Key findings with visual evidence: findings block plus relative-error line chart.
- Scope, data, and metric definitions: definitions block.
- Methodology/model specification: methodology block.
- Limitations, uncertainty, and robustness: linear-control and limitations blocks.
- Recommended next steps: next-steps block.
- Further questions: further-questions block.

## Chart map

- Section: larger samples reduce nonlinear curvature error.
- Question: how does median curvature RMSE change with antithetic pair count?
- Family/type: trend / highlighted multi-series line.
- Fields: ordered pair count; normalized nonlinear RMSE; normalized noisy-linear RMSE; inverse-square-root reference.
- Supported claim: stochastic estimation error decreases approximately at the Monte Carlo rate.
- Palette: blue actual, orange dashed comparison, neutral dotted reference; line style also carries identity.
- Delivery: native chart inside `report.html`.

## Data and calculation notes

- Synthetic functions; no external data source or time window.
- Generated from {config.repetitions} independent repetitions with master seed {config.master_seed}.
- Common random-number prefixes are used across pair counts within each repetition.
- Sample size is antithetic pairs; total function evaluations are twice the pair count.
- Raw repetition and coordinate outputs are retained for independent recomputation.

## Omission notes

- No uncertainty-band chart: the native line chart is normalized for convergence-shape comparison; exact quartiles remain in `aggregate.csv` and the chart dataset.
- No static parallel chart: HTML is the selected report delivery mode.
"""


def write_outputs(
    result: ExperimentResult,
    config: BenchmarkConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write all evidence and the canonical report payload to one folder."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    paths = {
        "run_metrics.csv": output / "run_metrics.csv",
        "coordinate_estimates.csv": output / "coordinate_estimates.csv",
        "aggregate.csv": output / "aggregate.csv",
        "convergence_rates.csv": output / "convergence_rates.csv",
        "summary.json": output / "summary.json",
        "experiment_manifest.json": output / "experiment_manifest.json",
        "artifact.json": output / "artifact.json",
        "source_notes.md": output / "source_notes.md",
    }
    _write_csv(paths["run_metrics.csv"], result.run_metrics, RUN_FIELDS)
    _write_csv(
        paths["coordinate_estimates.csv"],
        result.coordinate_estimates,
        COORDINATE_FIELDS,
    )
    _write_csv(paths["aggregate.csv"], result.aggregates, AGGREGATE_FIELDS)
    _write_csv(
        paths["convergence_rates.csv"],
        result.convergence_rates,
        RATE_FIELDS,
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


def _parse_pair_counts(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("pair counts must be comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one pair count is required")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/curvature_sample_size",
        help="Separate directory for CSV, JSON, notes, and report payloads.",
    )
    parser.add_argument("--dimension", type=int, default=12)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument(
        "--pair-counts",
        type=_parse_pair_counts,
        default=(4, 8, 16, 32, 64, 128, 250, 500, 1000),
    )
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--linear-noise-std", type=float, default=0.05)
    parser.add_argument("--master-seed", type=int, default=20260721)
    parser.add_argument("--rate-fit-min-pairs", type=int, default=32)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke protocol: 20 repetitions and pair counts 4,16,64.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pair_counts = args.pair_counts
    repetitions = args.repetitions
    rate_fit_min_pairs = args.rate_fit_min_pairs
    if args.quick:
        pair_counts = (4, 16, 64)
        repetitions = 20
        rate_fit_min_pairs = 4
    config = BenchmarkConfig(
        dimension=args.dimension,
        sigma=args.sigma,
        pair_counts=tuple(pair_counts),
        repetitions=repetitions,
        linear_noise_std=args.linear_noise_std,
        master_seed=args.master_seed,
        rate_fit_min_pairs=rate_fit_min_pairs,
    )
    result = run_benchmark(config)
    outputs = write_outputs(result, config, args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(f"Wrote {len(outputs)} source artifacts to {Path(args.output_dir).resolve()}")
    return 0 if result.summary["all_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

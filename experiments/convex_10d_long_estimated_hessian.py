#!/usr/bin/env python3
"""Ten-dimensional convex comparison of explicit and implicit ES.

This experiment deliberately compares exactly two updates on shifted,
deterministic, diagonal quadratics:

* explicit ES: ``x <- x - alpha_t * g_hat``;
* linearly implicit ES:
  ``x <- x - alpha_t * g_hat / (1 + alpha_t * h_hat)``.

The gradient and signed diagonal Hessian estimate use the same antithetic
candidate evaluations.  The Hessian is a raw leave-one-pair-out Stein
estimate.  There is no oracle curvature, replay, trust region, damping,
projection, clipping, rank shaping, or other safeguard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


EXPERIMENT_VERSION = "1.0.0"
DIMENSION = 10
TARGET_FRACTION = 1.0e-4
METHODS = ("explicit_es", "linearized_implicit_es")
METHOD_LABELS = {
    "explicit_es": "Explicit ES",
    "linearized_implicit_es": "Implicit ES",
}


@dataclass(frozen=True)
class QuadraticProblem:
    key: str
    condition_number: int
    hessian_diagonal: tuple[float, ...]
    optimum: tuple[float, ...]

    @property
    def initial_point(self) -> tuple[float, ...]:
        return tuple(value + 1.0 for value in self.optimum)


def _problem(condition_number: int) -> QuadraticProblem:
    return QuadraticProblem(
        key=f"kappa{condition_number}",
        condition_number=condition_number,
        hessian_diagonal=tuple(
            float(value)
            for value in np.geomspace(1.0, float(condition_number), DIMENSION)
        ),
        optimum=tuple(float(value) for value in np.linspace(-1.5, 1.5, DIMENSION)),
    )


PROBLEMS = tuple(_problem(value) for value in (2, 4, 8))


@dataclass(frozen=True)
class ExperimentConfig:
    population_size: int = 2000
    updates: int = 300
    seeds: tuple[int, ...] = tuple(range(10))
    sigma: float = 0.1
    initial_learning_rate: float = 0.5
    accuracy_populations: tuple[int, ...] = (100, 200, 500, 1000, 2000, 4000)
    accuracy_replicates: int = 50
    master_seed: int = 20260715

    def validate(self) -> None:
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be even and at least four")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if self.initial_learning_rate <= 0.0:
            raise ValueError("initial_learning_rate must be positive")
        if any(size < 4 or size % 2 for size in self.accuracy_populations):
            raise ValueError("accuracy populations must be even and at least four")
        if tuple(sorted(set(self.accuracy_populations))) != self.accuracy_populations:
            raise ValueError("accuracy populations must be sorted and unique")
        if self.accuracy_replicates < 1:
            raise ValueError("accuracy_replicates must be positive")


@dataclass(frozen=True)
class Estimate:
    gradient: np.ndarray
    hessian_diagonal: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    trajectories: tuple[dict[str, Any], ...]
    optimization_summary: tuple[dict[str, Any], ...]
    decision_metrics: tuple[dict[str, Any], ...]
    decision_summary: tuple[dict[str, Any], ...]
    curvature_updates: tuple[dict[str, Any], ...]
    curvature_accuracy: tuple[dict[str, Any], ...]
    curvature_accuracy_summary: tuple[dict[str, Any], ...]
    learning_rates: tuple[dict[str, Any], ...]


COORDINATE_FIELDS = tuple(f"x{index}" for index in range(1, DIMENSION + 1))
TRUE_H_FIELDS = tuple(f"h{index}_true" for index in range(1, DIMENSION + 1))
ESTIMATE_H_FIELDS = tuple(
    f"h{index}_estimate" for index in range(1, DIMENSION + 1)
)
ERROR_H_FIELDS = tuple(f"h{index}_error" for index in range(1, DIMENSION + 1))
DENOMINATOR_FIELDS = tuple(
    f"denominator_{index}" for index in range(1, DIMENSION + 1)
)
MULTIPLIER_FIELDS = tuple(
    f"multiplier_{index}" for index in range(1, DIMENSION + 1)
)

TRAJECTORY_FIELDS = (
    "problem",
    "condition_number",
    "dimension",
    "seed",
    "method",
    "update",
    "alpha_used",
    *COORDINATE_FIELDS,
    "objective_gap",
    "fraction_initial_loss",
)
OPTIMIZATION_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "method",
    "update",
    "alpha_used",
    "median_fraction_initial_loss",
    "q25_fraction_initial_loss",
    "q75_fraction_initial_loss",
)
DECISION_METRIC_FIELDS = (
    "problem",
    "condition_number",
    "seed",
    "method",
    "peak_fraction_initial_loss",
    "mean_fraction_initial_loss",
    "first_update_fraction_le_1e_minus_4",
    "reached_target",
    "final_fraction_initial_loss",
)
DECISION_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "method",
    "seed_count",
    "reached_target_count",
    "median_peak_fraction",
    "q25_peak_fraction",
    "q75_peak_fraction",
    "median_mean_fraction",
    "q25_mean_fraction",
    "q75_mean_fraction",
    "median_first_update",
    "q25_first_update",
    "q75_first_update",
    "median_final_fraction",
    "q25_final_fraction",
    "q75_final_fraction",
)
CURVATURE_UPDATE_FIELDS = (
    "problem",
    "condition_number",
    "seed",
    "update",
    "alpha",
    "evaluated_candidates",
    "antithetic_pairs",
    *TRUE_H_FIELDS,
    *ESTIMATE_H_FIELDS,
    *ERROR_H_FIELDS,
    *DENOMINATOR_FIELDS,
    *MULTIPLIER_FIELDS,
    "negative_hessian_coordinates",
    "nonpositive_denominator_coordinates",
    "minimum_denominator",
    "maximum_absolute_multiplier",
)
CURVATURE_ACCURACY_FIELDS = (
    "problem",
    "condition_number",
    "population_size",
    "antithetic_pairs",
    "seed",
    "replicate",
    *TRUE_H_FIELDS,
    *ESTIMATE_H_FIELDS,
    *ERROR_H_FIELDS,
    "relative_squared_error_mean",
)
CURVATURE_ACCURACY_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "population_size",
    "antithetic_pairs",
    "estimates_per_seed",
    "seed_count",
    "relative_rmse_mean",
    "relative_rmse_ci95_low",
    "relative_rmse_ci95_high",
    *tuple(f"h{index}_mean_estimate" for index in range(1, DIMENSION + 1)),
    *tuple(f"h{index}_rmse" for index in range(1, DIMENSION + 1)),
)
LEARNING_RATE_FIELDS = ("update_index_t", "alpha_t")


def learning_rate(config: ExperimentConfig, update_index: int) -> float:
    """Return alpha_t = 0.5 / sqrt(t + 1), with zero-based t."""
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    return float(config.initial_learning_rate / np.sqrt(update_index + 1.0))


def quadratic_loss(point: np.ndarray, problem: QuadraticProblem) -> float:
    point_array = np.asarray(point, dtype=np.float64)
    optimum = np.asarray(problem.optimum, dtype=np.float64)
    if point_array.shape != (DIMENSION,):
        raise ValueError(f"point must have shape ({DIMENSION},)")
    displacement = point_array - optimum
    hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
    return float(0.5 * np.dot(hessian, displacement**2))


def leave_one_pair_out_baseline(pair_sums: np.ndarray) -> np.ndarray:
    """For pair k, return the mean pair sum over all other pairs."""
    values = np.asarray(pair_sums, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two one-dimensional pair sums are required")
    return (float(np.sum(values)) - values) / float(values.size - 1)


def estimate_gradient_and_hessian(
    point: np.ndarray,
    perturbations: np.ndarray,
    sigma: float,
    problem: QuadraticProblem,
) -> Estimate:
    """Return raw antithetic gradient and signed LOO Stein diagonal Hessian."""
    point_array = np.asarray(point, dtype=np.float64)
    epsilon = np.asarray(perturbations, dtype=np.float64)
    if point_array.shape != (DIMENSION,):
        raise ValueError(f"point must have shape ({DIMENSION},)")
    if epsilon.ndim != 2 or epsilon.shape[1] != DIMENSION:
        raise ValueError(f"perturbations must have shape (pairs, {DIMENSION})")
    if epsilon.shape[0] < 2 or sigma <= 0.0:
        raise ValueError("at least two pairs and positive sigma are required")

    optimum = np.asarray(problem.optimum, dtype=np.float64)
    hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
    plus = point_array[None, :] + sigma * epsilon - optimum[None, :]
    minus = point_array[None, :] - sigma * epsilon - optimum[None, :]
    f_plus = 0.5 * np.sum(hessian[None, :] * plus**2, axis=1)
    f_minus = 0.5 * np.sum(hessian[None, :] * minus**2, axis=1)

    gradient = np.mean((f_plus - f_minus)[:, None] * epsilon, axis=0) / (
        2.0 * sigma
    )
    pair_sums = f_plus + f_minus
    baseline = leave_one_pair_out_baseline(pair_sums)
    pair_signal = pair_sums - baseline
    hessian_diagonal = np.mean(
        pair_signal[:, None] * (epsilon**2 - 1.0), axis=0
    ) / (2.0 * sigma**2)
    return Estimate(gradient=gradient, hessian_diagonal=hessian_diagonal)


def linearly_implicit_step(
    gradient: np.ndarray, hessian_diagonal: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the raw signed diagonal solve, without clipping or damping."""
    gradient_array = np.asarray(gradient, dtype=np.float64)
    hessian_array = np.asarray(hessian_diagonal, dtype=np.float64)
    if gradient_array.shape != (DIMENSION,) or hessian_array.shape != (DIMENSION,):
        raise ValueError(f"gradient and Hessian must have shape ({DIMENSION},)")
    denominator = 1.0 + alpha * hessian_array
    if np.any(denominator == 0.0):
        raise FloatingPointError("estimated diagonal implicit system is singular")
    multiplier = 1.0 / denominator
    step = -alpha * multiplier * gradient_array
    if not np.all(np.isfinite(step)):
        raise FloatingPointError("linearly implicit step is nonfinite")
    return step, denominator, multiplier


def _optimization_perturbations(
    config: ExperimentConfig, seed: int, update_index: int
) -> np.ndarray:
    # Problem and method are intentionally absent: identical Gaussian draws
    # are used across both methods and all three problems.
    sequence = np.random.SeedSequence(
        [config.master_seed, 0, int(seed), int(update_index)]
    )
    return np.random.default_rng(sequence).normal(
        size=(config.population_size // 2, DIMENSION)
    )


def _accuracy_perturbations(
    config: ExperimentConfig, population: int, seed: int, replicate: int
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [config.master_seed, 1, int(population), int(seed), int(replicate)]
    )
    return np.random.default_rng(sequence).normal(
        size=(population // 2, DIMENSION)
    )


def _coordinate_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}{index}": float(value)
        for index, value in enumerate(np.asarray(values), start=1)
    }


def _trajectory_row(
    problem: QuadraticProblem,
    seed: int,
    method: str,
    update: int,
    alpha_used: float | str,
    point: np.ndarray,
) -> dict[str, Any]:
    gap = quadratic_loss(point, problem)
    initial_gap = quadratic_loss(np.asarray(problem.initial_point), problem)
    return {
        "problem": problem.key,
        "condition_number": problem.condition_number,
        "dimension": DIMENSION,
        "seed": seed,
        "method": method,
        "update": update,
        "alpha_used": alpha_used,
        **_coordinate_values("x", point),
        "objective_gap": gap,
        "fraction_initial_loss": gap / initial_gap,
    }


def _summarize_optimization(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    summary: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for method in METHODS:
            selected_method = [
                row
                for row in rows
                if row["problem"] == problem.key and row["method"] == method
            ]
            for update in range(config.updates + 1):
                fractions = np.asarray(
                    [
                        row["fraction_initial_loss"]
                        for row in selected_method
                        if row["update"] == update
                    ],
                    dtype=np.float64,
                )
                if fractions.size != len(config.seeds):
                    raise RuntimeError("optimization trajectory is incomplete")
                summary.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "method": method,
                        "update": update,
                        "alpha_used": ""
                        if update == 0
                        else learning_rate(config, update - 1),
                        "median_fraction_initial_loss": float(np.median(fractions)),
                        "q25_fraction_initial_loss": float(
                            np.quantile(fractions, 0.25)
                        ),
                        "q75_fraction_initial_loss": float(
                            np.quantile(fractions, 0.75)
                        ),
                    }
                )
    return tuple(summary)


def _decision_metrics(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    metrics: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for method in METHODS:
            for seed in config.seeds:
                selected = [
                    row
                    for row in rows
                    if row["problem"] == problem.key
                    and row["method"] == method
                    and row["seed"] == seed
                ]
                selected.sort(key=lambda row: int(row["update"]))
                if len(selected) != config.updates + 1:
                    raise RuntimeError("decision trajectory is incomplete")
                fractions = np.asarray(
                    [row["fraction_initial_loss"] for row in selected],
                    dtype=np.float64,
                )
                reached_indices = np.flatnonzero(fractions <= TARGET_FRACTION)
                first_update: int | None = (
                    int(reached_indices[0]) if reached_indices.size else None
                )
                metrics.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "seed": seed,
                        "method": method,
                        "peak_fraction_initial_loss": float(np.max(fractions)),
                        "mean_fraction_initial_loss": float(np.mean(fractions)),
                        "first_update_fraction_le_1e_minus_4": first_update,
                        "reached_target": int(first_update is not None),
                        "final_fraction_initial_loss": float(fractions[-1]),
                    }
                )

    summary: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for method in METHODS:
            selected = [
                row
                for row in metrics
                if row["problem"] == problem.key and row["method"] == method
            ]
            peak = np.asarray(
                [row["peak_fraction_initial_loss"] for row in selected]
            )
            mean = np.asarray(
                [row["mean_fraction_initial_loss"] for row in selected]
            )
            final = np.asarray(
                [row["final_fraction_initial_loss"] for row in selected]
            )
            first = np.asarray(
                [
                    row["first_update_fraction_le_1e_minus_4"]
                    for row in selected
                    if row["reached_target"]
                ],
                dtype=np.float64,
            )
            if first.size == 0:
                first_quantiles = (None, None, None)
            else:
                first_quantiles = tuple(
                    float(value) for value in np.quantile(first, (0.5, 0.25, 0.75))
                )
            summary.append(
                {
                    "problem": problem.key,
                    "condition_number": problem.condition_number,
                    "method": method,
                    "seed_count": len(selected),
                    "reached_target_count": int(sum(row["reached_target"] for row in selected)),
                    "median_peak_fraction": float(np.median(peak)),
                    "q25_peak_fraction": float(np.quantile(peak, 0.25)),
                    "q75_peak_fraction": float(np.quantile(peak, 0.75)),
                    "median_mean_fraction": float(np.median(mean)),
                    "q25_mean_fraction": float(np.quantile(mean, 0.25)),
                    "q75_mean_fraction": float(np.quantile(mean, 0.75)),
                    "median_first_update": first_quantiles[0],
                    "q25_first_update": first_quantiles[1],
                    "q75_first_update": first_quantiles[2],
                    "median_final_fraction": float(np.median(final)),
                    "q25_final_fraction": float(np.quantile(final, 0.25)),
                    "q75_final_fraction": float(np.quantile(final, 0.75)),
                }
            )
    return tuple(metrics), tuple(summary)


def run_optimization(
    config: ExperimentConfig,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    trajectories: list[dict[str, Any]] = []
    curvature: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        initial = np.asarray(problem.initial_point, dtype=np.float64)
        true_hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
        for seed in config.seeds:
            states = {method: initial.copy() for method in METHODS}
            for method in METHODS:
                trajectories.append(
                    _trajectory_row(problem, seed, method, 0, "", states[method])
                )

            for update_index in range(config.updates):
                alpha = learning_rate(config, update_index)
                epsilon = _optimization_perturbations(config, seed, update_index)

                explicit = estimate_gradient_and_hessian(
                    states["explicit_es"], epsilon, config.sigma, problem
                )
                states["explicit_es"] -= alpha * explicit.gradient

                implicit = estimate_gradient_and_hessian(
                    states["linearized_implicit_es"], epsilon, config.sigma, problem
                )
                step, denominator, multiplier = linearly_implicit_step(
                    implicit.gradient, implicit.hessian_diagonal, alpha
                )
                states["linearized_implicit_es"] += step

                error = implicit.hessian_diagonal - true_hessian
                curvature.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "seed": seed,
                        "update": update_index + 1,
                        "alpha": alpha,
                        "evaluated_candidates": config.population_size,
                        "antithetic_pairs": config.population_size // 2,
                        **{
                            f"h{index}_true": float(value)
                            for index, value in enumerate(true_hessian, start=1)
                        },
                        **{
                            f"h{index}_estimate": float(value)
                            for index, value in enumerate(
                                implicit.hessian_diagonal, start=1
                            )
                        },
                        **{
                            f"h{index}_error": float(value)
                            for index, value in enumerate(error, start=1)
                        },
                        **{
                            f"denominator_{index}": float(value)
                            for index, value in enumerate(denominator, start=1)
                        },
                        **{
                            f"multiplier_{index}": float(value)
                            for index, value in enumerate(multiplier, start=1)
                        },
                        "negative_hessian_coordinates": int(
                            np.count_nonzero(implicit.hessian_diagonal < 0.0)
                        ),
                        "nonpositive_denominator_coordinates": int(
                            np.count_nonzero(denominator <= 0.0)
                        ),
                        "minimum_denominator": float(np.min(denominator)),
                        "maximum_absolute_multiplier": float(
                            np.max(np.abs(multiplier))
                        ),
                    }
                )

                for method in METHODS:
                    if not np.all(np.isfinite(states[method])):
                        raise FloatingPointError(
                            f"{method} became nonfinite on {problem.key}, "
                            f"seed={seed}, update={update_index + 1}"
                        )
                    trajectories.append(
                        _trajectory_row(
                            problem,
                            seed,
                            method,
                            update_index + 1,
                            alpha,
                            states[method],
                        )
                    )

    optimization_summary = _summarize_optimization(config, trajectories)
    decision_metrics, decision_summary = _decision_metrics(config, trajectories)
    return (
        tuple(trajectories),
        optimization_summary,
        decision_metrics,
        decision_summary,
        tuple(curvature),
    )


def _t95(sample_count: int) -> float:
    values = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
    }
    if sample_count < 2:
        return float("nan")
    return values.get(sample_count, 1.96)


def run_curvature_accuracy(
    config: ExperimentConfig,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    raw: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        point = np.asarray(problem.initial_point, dtype=np.float64)
        true_hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
        for population in config.accuracy_populations:
            for seed in config.seeds:
                for replicate in range(config.accuracy_replicates):
                    epsilon = _accuracy_perturbations(
                        config, population, seed, replicate
                    )
                    estimate = estimate_gradient_and_hessian(
                        point, epsilon, config.sigma, problem
                    ).hessian_diagonal
                    error = estimate - true_hessian
                    raw.append(
                        {
                            "problem": problem.key,
                            "condition_number": problem.condition_number,
                            "population_size": population,
                            "antithetic_pairs": population // 2,
                            "seed": seed,
                            "replicate": replicate,
                            **{
                                f"h{index}_true": float(value)
                                for index, value in enumerate(true_hessian, start=1)
                            },
                            **{
                                f"h{index}_estimate": float(value)
                                for index, value in enumerate(estimate, start=1)
                            },
                            **{
                                f"h{index}_error": float(value)
                                for index, value in enumerate(error, start=1)
                            },
                            "relative_squared_error_mean": float(
                                np.mean((error / true_hessian) ** 2)
                            ),
                        }
                    )

    summary: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for population in config.accuracy_populations:
            selected = [
                row
                for row in raw
                if row["problem"] == problem.key
                and row["population_size"] == population
            ]
            seed_rmse: list[float] = []
            for seed in config.seeds:
                squared_errors = [
                    row["relative_squared_error_mean"]
                    for row in selected
                    if row["seed"] == seed
                ]
                if len(squared_errors) != config.accuracy_replicates:
                    raise RuntimeError("curvature accuracy rows are incomplete")
                seed_rmse.append(float(np.sqrt(np.mean(squared_errors))))
            seed_rmse_array = np.asarray(seed_rmse, dtype=np.float64)
            center = float(np.mean(seed_rmse_array))
            half_width = float(
                _t95(len(config.seeds))
                * np.std(seed_rmse_array, ddof=1)
                / np.sqrt(len(config.seeds))
            )
            estimates = np.asarray(
                [
                    [row[f"h{index}_estimate"] for index in range(1, DIMENSION + 1)]
                    for row in selected
                ],
                dtype=np.float64,
            )
            errors = np.asarray(
                [
                    [row[f"h{index}_error"] for index in range(1, DIMENSION + 1)]
                    for row in selected
                ],
                dtype=np.float64,
            )
            summary.append(
                {
                    "problem": problem.key,
                    "condition_number": problem.condition_number,
                    "population_size": population,
                    "antithetic_pairs": population // 2,
                    "estimates_per_seed": config.accuracy_replicates,
                    "seed_count": len(config.seeds),
                    "relative_rmse_mean": center,
                    "relative_rmse_ci95_low": max(0.0, center - half_width),
                    "relative_rmse_ci95_high": center + half_width,
                    **{
                        f"h{index}_mean_estimate": float(
                            np.mean(estimates[:, index - 1])
                        )
                        for index in range(1, DIMENSION + 1)
                    },
                    **{
                        f"h{index}_rmse": float(
                            np.sqrt(np.mean(errors[:, index - 1] ** 2))
                        )
                        for index in range(1, DIMENSION + 1)
                    },
                }
            )
    return tuple(raw), tuple(summary)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()
    (
        trajectories,
        optimization_summary,
        decision_metrics,
        decision_summary,
        curvature_updates,
    ) = run_optimization(config)
    curvature_accuracy, curvature_accuracy_summary = run_curvature_accuracy(config)
    learning_rates = tuple(
        {
            "update_index_t": update_index,
            "alpha_t": learning_rate(config, update_index),
        }
        for update_index in range(config.updates)
    )
    return ExperimentResult(
        trajectories=trajectories,
        optimization_summary=optimization_summary,
        decision_metrics=decision_metrics,
        decision_summary=decision_summary,
        curvature_updates=curvature_updates,
        curvature_accuracy=curvature_accuracy,
        curvature_accuracy_summary=curvature_accuracy_summary,
        learning_rates=learning_rates,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(
    figure: plt.Figure,
    pdf_path: Path,
    png_path: Path,
    title: str,
    subject: str,
) -> None:
    temporary_pdf = pdf_path.with_name(f".{pdf_path.name}.tmp")
    temporary_png = png_path.with_name(f".{png_path.name}.tmp")
    figure.savefig(
        temporary_pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Author": "DIIWES experiment", "Subject": subject},
    )
    figure.savefig(
        temporary_png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title},
    )
    plt.close(figure)
    os.replace(temporary_pdf, pdf_path)
    os.replace(temporary_png, png_path)


def plot_optimization(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot the two methods across three condition numbers."""
    _style()
    colors = {"explicit_es": "#6B7280", "linearized_implicit_es": "#1F5AA6"}
    styles = {"explicit_es": "--", "linearized_implicit_es": "-"}
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.55), sharey=True)
    handles: list[Any] = []
    floor = 1.0e-18
    for panel, (axis, problem) in enumerate(zip(axes, PROBLEMS, strict=True)):
        for method in METHODS:
            selected = [
                row
                for row in summary
                if row["problem"] == problem.key and row["method"] == method
            ]
            selected.sort(key=lambda row: int(row["update"]))
            x = np.asarray([row["update"] for row in selected])
            median = np.asarray([row["median_fraction_initial_loss"] for row in selected])
            lower = np.asarray([row["q25_fraction_initial_loss"] for row in selected])
            upper = np.asarray([row["q75_fraction_initial_loss"] for row in selected])
            line = axis.plot(
                x,
                np.maximum(median, floor),
                color=colors[method],
                linestyle=styles[method],
                linewidth=2.1,
                label=METHOD_LABELS[method],
                zorder=3,
            )[0]
            axis.fill_between(
                x,
                np.maximum(lower, floor),
                np.maximum(upper, floor),
                color=colors[method],
                alpha=0.12,
                linewidth=0.0,
            )
            if panel == 0:
                handles.append(line)
        axis.axhline(TARGET_FRACTION, color="#B8BEC7", linewidth=0.8, zorder=1)
        axis.set_title(rf"$\kappa={problem.condition_number}$", pad=6)
        axis.set_yscale("log")
        axis.set_xlim(0, config.updates)
        axis.set_xlabel("Update")
        axis.grid(axis="y", color="#DEE2E8", linewidth=0.65, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.8)
    axes[0].set_ylabel("Loss / initial loss")
    figure.suptitle("Optimization on 10-D shifted quadratics", y=0.99, fontsize=14)
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        frameon=False,
        handlelength=2.7,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.75, wspace=0.18)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Optimization on 10-D shifted quadratics",
        "Median and interquartile range across matched seeds",
    )


def plot_curvature_accuracy(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot curvature RMSE versus evaluated candidates, with direct labels."""
    _style()
    colors = {2: "#1F5AA6", 4: "#C46A1A", 8: "#2F7D5A"}
    markers = {2: "o", 4: "s", 8: "^"}
    styles = {2: "-", 4: "--", 8: "-."}
    label_offsets = {2: -3, 4: 4, 8: 10}
    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    for problem in PROBLEMS:
        selected = [row for row in summary if row["problem"] == problem.key]
        selected.sort(key=lambda row: int(row["population_size"]))
        population = np.asarray([row["population_size"] for row in selected])
        center = 100.0 * np.asarray([row["relative_rmse_mean"] for row in selected])
        lower = 100.0 * np.asarray([row["relative_rmse_ci95_low"] for row in selected])
        upper = 100.0 * np.asarray([row["relative_rmse_ci95_high"] for row in selected])
        color = colors[problem.condition_number]
        axis.fill_between(population, lower, upper, color=color, alpha=0.10, linewidth=0)
        axis.plot(
            population,
            center,
            color=color,
            linestyle=styles[problem.condition_number],
            marker=markers[problem.condition_number],
            markersize=4.5,
            linewidth=2.0,
        )
        axis.annotate(
            rf"$\kappa={problem.condition_number}$",
            xy=(population[-1], center[-1]),
            xytext=(7, label_offsets[problem.condition_number]),
            textcoords="offset points",
            va="center",
            color=color,
            fontsize=9.5,
        )
    axis.axvline(config.population_size, color="#9CA3AF", linestyle=":", linewidth=1.1)
    axis.text(
        config.population_size,
        axis.get_ylim()[1] if axis.get_ylim()[1] > 0 else 100,
        "  production B",
        ha="left",
        va="top",
        color="#6B7280",
        fontsize=8.5,
    )
    axis.set_xscale("log")
    axis.set_xticks(config.accuracy_populations)
    axis.set_xticklabels([str(value) for value in config.accuracy_populations])
    axis.set_xlim(min(config.accuracy_populations) * 0.9, max(config.accuracy_populations) * 1.38)
    axis.set_ylim(bottom=5.0)
    axis.set_xlabel("Evaluated candidates per update, B")
    axis.set_ylabel("Relative Hessian RMSE (%)")
    axis.set_title("Diagonal curvature accuracy", pad=10, fontsize=14)
    axis.grid(axis="y", color="#DEE2E8", linewidth=0.65, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)
    figure.subplots_adjust(left=0.11, right=0.88, bottom=0.15, top=0.88)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Diagonal curvature accuracy",
        "Relative diagonal Hessian RMSE versus candidate population",
    )


def plot_stability_summary(
    pdf_path: Path,
    png_path: Path,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Compact decision view of overshoot and updates to target."""
    _style()
    colors = {"explicit_es": "#6B7280", "linearized_implicit_es": "#1F5AA6"}
    markers = {"explicit_es": "o", "linearized_implicit_es": "s"}
    offsets = {"explicit_es": -0.08, "linearized_implicit_es": 0.08}
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.65))
    x = np.arange(len(PROBLEMS), dtype=np.float64)
    handles: list[Any] = []
    for method in METHODS:
        selected = [row for row in summary if row["method"] == method]
        selected.sort(key=lambda row: int(row["condition_number"]))
        xpos = x + offsets[method]
        peak = np.asarray([row["median_peak_fraction"] for row in selected])
        peak_low = np.asarray([row["q25_peak_fraction"] for row in selected])
        peak_high = np.asarray([row["q75_peak_fraction"] for row in selected])
        handle = axes[0].errorbar(
            xpos,
            peak,
            yerr=np.vstack((peak - peak_low, peak_high - peak)),
            color=colors[method],
            marker=markers[method],
            linestyle="none",
            markersize=6,
            capsize=3,
            elinewidth=1.4,
            label=METHOD_LABELS[method],
        )
        handles.append(handle)
        target = np.asarray([row["median_first_update"] for row in selected], dtype=float)
        target_low = np.asarray([row["q25_first_update"] for row in selected], dtype=float)
        target_high = np.asarray([row["q75_first_update"] for row in selected], dtype=float)
        axes[1].errorbar(
            xpos,
            target,
            yerr=np.vstack((target - target_low, target_high - target)),
            color=colors[method],
            marker=markers[method],
            linestyle="none",
            markersize=6,
            capsize=3,
            elinewidth=1.4,
        )
    axes[0].axhline(1.0, color="#B8BEC7", linewidth=0.9)
    axes[0].set_title("Peak loss", pad=7)
    axes[0].set_ylabel("Peak / initial loss")
    axes[1].set_title(r"Time to $10^{-4}$", pad=7)
    axes[1].set_ylabel("Updates")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([rf"$\kappa={problem.condition_number}$" for problem in PROBLEMS])
        axis.grid(axis="y", color="#DEE2E8", linewidth=0.65, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.8)
    figure.suptitle("Stability and progress", y=0.99, fontsize=14)
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.88),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.16, top=0.70, wspace=0.30)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Stability and progress",
        "Peak normalized loss and updates to target across matched seeds",
    )


def _atomic_write_csv(
    path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_report(config: ExperimentConfig, result: ExperimentResult) -> str:
    lines = [
        "# Explicit versus linearly implicit ES in 10 dimensions",
        "",
        "## Question",
        "",
        "Does an estimated-Hessian implicit update reduce learning-rate overshoot "
        "as the convex problem becomes stiffer?",
        "",
        "## Controlled protocol",
        "",
        r"Each problem is \(f(x)=\tfrac12(x-x^\star)^T H(x-x^\star)\), where",
        r"\(x^\star\) is the same nonzero 10-vector, \(x_0=x^\star+\mathbf 1\), and",
        r"the ten diagonal eigenvalues are geometrically spaced from 1 to \(\kappa\).",
        "",
        f"- Problems: dimension {DIMENSION}, condition numbers 2, 4, and 8.",
        r"- Schedule: \(\alpha_t=0.5/\sqrt{t+1}\), zero-based \(t\).",
        f"- Optimization: {config.updates} updates, {config.population_size} candidates "
        f"({config.population_size // 2} antithetic directions), "
        f"{len(config.seeds)} matched seeds, sigma={config.sigma:g}.",
        "- Exactly two methods: raw explicit ES and raw linearly implicit ES.",
        "- The same candidates provide gradient and curvature; curvature adds zero evaluations.",
        "- No replay, trust region, damping, projection, clipping, rank shaping, or oracle Hessian.",
        "",
        "## Update and estimator",
        "",
        r"Explicit: \(x_{t+1}=x_t-\alpha_t\widehat g_t\).",
        "",
        r"Implicit: \(x_{t+1}=x_t-\alpha_t(I+\alpha_t\operatorname{diag}(\widehat h_t))^{-1}\widehat g_t\).",
        "",
        r"For \(m=B/2\) directions, \(s_k=f(x+\sigma\epsilon_k)+f(x-\sigma\epsilon_k)\) and",
        r"\(b_k=(m-1)^{-1}\sum_{\ell\ne k}s_\ell\). The signed diagonal estimate is",
        r"\[\widehat h_j=\frac1m\sum_{k=1}^m\frac{(s_k-b_k)(\epsilon_{k,j}^2-1)}{2\sigma^2}.\]",
        "",
        "## Decision metrics (median across seeds)",
        "",
        "| Problem | Method | Peak / initial | Mean / initial | First update <= 1e-4 | Final / initial |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for problem in PROBLEMS:
        for method in METHODS:
            row = next(
                item
                for item in result.decision_summary
                if item["problem"] == problem.key and item["method"] == method
            )
            first_update = (
                "not reached"
                if row["median_first_update"] is None
                else f"{row['median_first_update']:.4g}"
            )
            lines.append(
                f"| kappa={problem.condition_number} | {METHOD_LABELS[method]} | "
                f"{row['median_peak_fraction']:.4g} | "
                f"{row['median_mean_fraction']:.4g} | "
                f"{first_update} | "
                f"{row['median_final_fraction']:.4g} |"
            )
    lines.extend(
        [
            "",
            f"## Curvature accuracy at production B={config.population_size}",
            "",
            "Relative RMSE is sqrt(mean_j((h_hat_j-h_j)/h_j)^2), first pooled "
            "within each seed and then averaged across seeds.",
            "",
            "| Problem | Relative RMSE | 95% CI |",
            "| :--- | ---: | :--- |",
        ]
    )
    for problem in PROBLEMS:
        row = next(
            item
            for item in result.curvature_accuracy_summary
            if item["problem"] == problem.key
            and item["population_size"] == config.population_size
        )
        lines.append(
            f"| kappa={problem.condition_number} | "
            f"{100 * row['relative_rmse_mean']:.2f}% | "
            f"[{100 * row['relative_rmse_ci95_low']:.2f}%, "
            f"{100 * row['relative_rmse_ci95_high']:.2f}%] |"
        )
    negative = sum(
        int(row["negative_hessian_coordinates"]) for row in result.curvature_updates
    )
    nonpositive = sum(
        int(row["nonpositive_denominator_coordinates"])
        for row in result.curvature_updates
    )
    total_coordinates = len(result.curvature_updates) * DIMENSION
    lines.extend(
        [
            "",
            "## Raw implicit-system diagnostics",
            "",
            f"Across {total_coordinates:,} estimated coordinate updates, "
            f"{negative:,} Hessian entries were negative and {nonpositive:,} "
            "implicit denominators were nonpositive. No value was clipped.",
            "",
            "## Interpretation",
            "",
            "The implicit procedure is not uniformly faster: on mild problems its "
            "curvature denominator makes conservative steps. Its benefit appears on "
            "the stiff problem, where explicit ES overshoots while implicit ES keeps "
            "the initial aggressive learning rate stable. The curvature-accuracy "
            "curve shows the sampling cost behind that stabilization.",
            "",
            "Every trajectory, per-update Hessian/denominator/multiplier, accuracy "
            "replicate, schedule value, and plotted aggregate is retained in CSV. "
            "`manifest.json` records hashes and row counts.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: str | os.PathLike[str],
    config: ExperimentConfig,
    result: ExperimentResult,
) -> dict[str, str]:
    root = Path(output_dir)
    figures = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": root / "optimization_trajectories.csv",
        "optimization_summary": root / "optimization_summary.csv",
        "decision_metrics": root / "decision_metrics.csv",
        "decision_summary": root / "decision_summary.csv",
        "curvature_updates": root / "curvature_updates.csv",
        "curvature_accuracy": root / "curvature_accuracy.csv",
        "curvature_accuracy_summary": root / "curvature_accuracy_summary.csv",
        "learning_rates": root / "learning_rate_schedule.csv",
        "optimization_pdf": figures / "optimization_trajectories.pdf",
        "optimization_png": figures / "optimization_trajectories.png",
        "accuracy_pdf": figures / "curvature_accuracy.pdf",
        "accuracy_png": figures / "curvature_accuracy.png",
        "stability_pdf": figures / "stability_summary.pdf",
        "stability_png": figures / "stability_summary.png",
        "report": root / "report.md",
        "manifest": root / "manifest.json",
    }
    _atomic_write_csv(paths["trajectories"], TRAJECTORY_FIELDS, result.trajectories)
    _atomic_write_csv(
        paths["optimization_summary"],
        OPTIMIZATION_SUMMARY_FIELDS,
        result.optimization_summary,
    )
    _atomic_write_csv(
        paths["decision_metrics"], DECISION_METRIC_FIELDS, result.decision_metrics
    )
    _atomic_write_csv(
        paths["decision_summary"], DECISION_SUMMARY_FIELDS, result.decision_summary
    )
    _atomic_write_csv(
        paths["curvature_updates"], CURVATURE_UPDATE_FIELDS, result.curvature_updates
    )
    _atomic_write_csv(
        paths["curvature_accuracy"],
        CURVATURE_ACCURACY_FIELDS,
        result.curvature_accuracy,
    )
    _atomic_write_csv(
        paths["curvature_accuracy_summary"],
        CURVATURE_ACCURACY_SUMMARY_FIELDS,
        result.curvature_accuracy_summary,
    )
    _atomic_write_csv(
        paths["learning_rates"], LEARNING_RATE_FIELDS, result.learning_rates
    )
    plot_optimization(
        paths["optimization_pdf"],
        paths["optimization_png"],
        config,
        result.optimization_summary,
    )
    plot_curvature_accuracy(
        paths["accuracy_pdf"],
        paths["accuracy_png"],
        config,
        result.curvature_accuracy_summary,
    )
    plot_stability_summary(
        paths["stability_pdf"], paths["stability_png"], result.decision_summary
    )
    _atomic_write_text(paths["report"], _render_report(config, result))

    tracked: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        if name == "manifest":
            continue
        tracked[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    row_counts = {
        "trajectories": len(result.trajectories),
        "optimization_summary": len(result.optimization_summary),
        "decision_metrics": len(result.decision_metrics),
        "decision_summary": len(result.decision_summary),
        "curvature_updates": len(result.curvature_updates),
        "curvature_accuracy": len(result.curvature_accuracy),
        "curvature_accuracy_summary": len(result.curvature_accuracy_summary),
        "learning_rates": len(result.learning_rates),
    }
    for name, count in row_counts.items():
        tracked[name]["rows"] = count

    manifest = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "does_estimated_curvature_stabilize_aggressive_ES_on_stiffer_10D_quadratics",
        "methods": {
            "explicit_es": "x_next = x - alpha_t * g_hat",
            "linearized_implicit_es": "x_next = x - alpha_t * g_hat / (1 + alpha_t * h_hat_diag)",
        },
        "learning_rate_schedule": {
            "formula": "alpha_t = 0.5 / sqrt(t + 1)",
            "zero_based_t": True,
            "first": learning_rate(config, 0),
            "last": learning_rate(config, config.updates - 1),
        },
        "evaluation_accounting": {
            "optimization_candidates_per_update_per_method": config.population_size,
            "optimization_antithetic_pairs_per_update_per_method": config.population_size // 2,
            "candidate_evaluations_per_method_run": config.population_size * config.updates,
            "hessian_additional_candidate_evaluations": 0,
            "definition": "B candidates are B/2 Gaussian directions, each evaluated at +epsilon and -epsilon",
        },
        "problems": [
            {
                **asdict(problem),
                "initial_point": list(problem.initial_point),
                "dimension": DIMENSION,
                "objective": "0.5 * (x - optimum)^T diag(hessian_diagonal) (x - optimum)",
            }
            for problem in PROBLEMS
        ],
        "estimators": {
            "gradient": "raw antithetic loss difference",
            "hessian_diagonal": "raw signed diagonal Stein estimate",
            "baseline": "leave one antithetic pair out",
        },
        "decision_metrics": {
            "target_fraction_initial_loss": TARGET_FRACTION,
            "peak_includes_initial_state": True,
        },
        "common_random_numbers": {
            "across_methods": True,
            "across_problems": True,
            "optimization_seed_sequence": "[master_seed, 0, seed, update_index]",
        },
        "excluded_components": [
            "replay",
            "trust_region",
            "additive_damping",
            "curvature_projection",
            "curvature_clipping",
            "multiplier_clipping",
            "rank_shaping",
            "oracle_curvature_optimizer",
        ],
        "validation": {
            "singular_or_nonfinite_runs": 0,
            "all_expected_rows_present": True,
        },
        "config": asdict(config),
        "files": tracked,
        "provenance": {
            "source_file": "experiments/convex_10d_long_estimated_hessian.py",
            "source_sha256": _sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
    }
    _atomic_write_text(
        paths["manifest"],
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return {name: str(path) for name, path in paths.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="reports/convex_10d_long_presentation",
        help="output folder for CSV, report, figures, and manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig()
    result = run_experiment(config)
    outputs = write_outputs(args.output_dir, config, result)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

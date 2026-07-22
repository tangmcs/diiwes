#!/usr/bin/env python3
"""Initial-learning-rate sweep for explicit and linearly implicit ES.

The experiment uses deterministic shifted 10-D diagonal quadratics and varies
the initial learning rate in ``alpha_t = alpha_0 / sqrt(t + 1)``.  It compares
exactly two raw methods:

* explicit ES: ``x <- x - alpha_t * g_hat``;
* linearly implicit ES:
  ``x <- x - alpha_t * g_hat / (1 + alpha_t * h_hat)``.

The gradient and signed leave-one-pair-out diagonal Hessian estimate use the
same antithetic candidate evaluations.  There is no replay, trust region,
damping, projection, clipping, rank shaping, oracle curvature, or fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:  # Package import used by tests and ``python -m``.
    from experiments.convex_10d_long_estimated_hessian import (  # noqa: E402
        DIMENSION,
        METHODS,
        PROBLEMS,
        TARGET_FRACTION,
        estimate_gradient_and_hessian,
        linearly_implicit_step,
        quadratic_loss,
    )
except ModuleNotFoundError as error:  # Direct ``python experiments/file.py``.
    if error.name != "experiments":
        raise
    from convex_10d_long_estimated_hessian import (  # type: ignore[no-redef]  # noqa: E402
        DIMENSION,
        METHODS,
        PROBLEMS,
        TARGET_FRACTION,
        estimate_gradient_and_hessian,
        linearly_implicit_step,
        quadratic_loss,
    )


EXPERIMENT_VERSION = "1.0.0"
DEFAULT_OUTPUT_DIR = "reports/convex_10d_initial_lr_sweep"
INITIAL_LEARNING_RATES = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
METHOD_LABELS = {
    "explicit_es": "Explicit ES",
    "linearized_implicit_es": "Linearized implicit ES",
}
TRAJECTORY_VISUALIZATION_FLOOR = 1.0e-20


@dataclass(frozen=True)
class ExperimentConfig:
    population_size: int = 2000
    updates: int = 300
    seeds: tuple[int, ...] = tuple(range(10))
    sigma: float = 0.1
    initial_learning_rates: tuple[float, ...] = INITIAL_LEARNING_RATES
    master_seed: int = 20260716

    def validate(self) -> None:
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be even and at least four")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be nonempty and unique")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonnegative")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if not self.initial_learning_rates:
            raise ValueError("initial_learning_rates must be nonempty")
        if any(value <= 0.0 for value in self.initial_learning_rates):
            raise ValueError("initial learning rates must be positive")
        if tuple(sorted(set(self.initial_learning_rates))) != self.initial_learning_rates:
            raise ValueError("initial learning rates must be sorted and unique")


@dataclass(frozen=True)
class ExperimentResult:
    trajectories: tuple[dict[str, Any], ...]
    optimization_summary: tuple[dict[str, Any], ...]
    decision_metrics: tuple[dict[str, Any], ...]
    decision_summary: tuple[dict[str, Any], ...]
    curvature_updates: tuple[dict[str, Any], ...]
    curvature_diagnostics_by_seed: tuple[dict[str, Any], ...]
    curvature_diagnostics: tuple[dict[str, Any], ...]
    learning_rates: tuple[dict[str, Any], ...]


COORDINATE_FIELDS = tuple(f"x{index}" for index in range(1, DIMENSION + 1))
TRUE_H_FIELDS = tuple(f"h{index}_true" for index in range(1, DIMENSION + 1))
ESTIMATE_H_FIELDS = tuple(
    f"h{index}_estimate" for index in range(1, DIMENSION + 1)
)
DENOMINATOR_FIELDS = tuple(
    f"denominator_{index}" for index in range(1, DIMENSION + 1)
)
MULTIPLIER_FIELDS = tuple(
    f"multiplier_{index}" for index in range(1, DIMENSION + 1)
)

TRAJECTORY_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "dimension",
    "seed",
    "method",
    "update",
    "alpha_used",
    *COORDINATE_FIELDS,
    "objective_gap",
    "fraction_initial_loss",
    "finite",
)
OPTIMIZATION_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "method",
    "update",
    "alpha_used",
    "finite_seed_count",
    "median_fraction_initial_loss",
    "q25_fraction_initial_loss",
    "q75_fraction_initial_loss",
)
DECISION_METRIC_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "seed",
    "method",
    "peak_fraction_initial_loss",
    "mean_fraction_initial_loss",
    "first_update_fraction_le_1e_minus_4",
    "reached_target",
    "final_fraction_initial_loss",
    "finite_run",
    "failure_update",
)
DECISION_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "method",
    "seed_count",
    "finite_run_count",
    "reached_target_count",
    "all_seeds_reached_target",
    "median_peak_fraction",
    "q25_peak_fraction",
    "q75_peak_fraction",
    "median_mean_fraction",
    "q25_mean_fraction",
    "q75_mean_fraction",
    "median_first_update_reached",
    "q25_first_update_reached",
    "q75_first_update_reached",
    "median_first_update_all_seeds",
    "q25_first_update_all_seeds",
    "q75_first_update_all_seeds",
    "median_final_fraction",
    "q25_final_fraction",
    "q75_final_fraction",
)
CURVATURE_UPDATE_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "seed",
    "update",
    "alpha",
    "evaluated_candidates",
    "antithetic_pairs",
    *TRUE_H_FIELDS,
    *ESTIMATE_H_FIELDS,
    *DENOMINATOR_FIELDS,
    *MULTIPLIER_FIELDS,
    "relative_squared_error_mean",
    "negative_hessian_coordinates",
    "nonpositive_denominator_coordinates",
    "minimum_denominator",
    "maximum_absolute_multiplier",
)
CURVATURE_DIAGNOSTIC_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "seed",
    "updates",
    "coordinate_updates",
    "relative_hessian_rmse",
    "negative_hessian_coordinates",
    "negative_hessian_fraction",
    "nonpositive_denominator_coordinates",
    "nonpositive_denominator_fraction",
    "minimum_denominator",
    "maximum_absolute_multiplier",
)
CURVATURE_DIAGNOSTIC_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "initial_learning_rate",
    "seed_count",
    "updates_per_seed",
    "coordinate_updates",
    "relative_hessian_rmse",
    "negative_hessian_coordinates",
    "negative_hessian_fraction",
    "nonpositive_denominator_coordinates",
    "nonpositive_denominator_fraction",
    "minimum_denominator",
    "maximum_absolute_multiplier",
)
LEARNING_RATE_FIELDS = (
    "initial_learning_rate",
    "update_index_t",
    "update_number",
    "alpha_t",
)


def learning_rate(initial_learning_rate: float, update_index: int) -> float:
    """Return alpha_t = alpha_0 / sqrt(t + 1), with zero-based t."""
    if initial_learning_rate <= 0.0:
        raise ValueError("initial_learning_rate must be positive")
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    return float(initial_learning_rate / np.sqrt(update_index + 1.0))


def exact_gradient_initial_step_reference(condition_number: float) -> float:
    """Return the exact-gradient initial-step reference 2 / lambda_max."""
    if condition_number <= 0.0:
        raise ValueError("condition_number must be positive")
    # All test Hessians have lambda_min=1 and lambda_max=kappa.
    return float(2.0 / condition_number)


def optimization_perturbations(
    config: ExperimentConfig, seed: int, update_index: int
) -> np.ndarray:
    """Draw CRN perturbations independent of problem, alpha_0, and method."""
    sequence = np.random.SeedSequence(
        [config.master_seed, int(seed), int(update_index)]
    )
    return np.random.default_rng(sequence).normal(
        size=(config.population_size // 2, DIMENSION)
    )


def _coordinate_values(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}{index}": float(value)
        for index, value in enumerate(np.asarray(values), start=1)
    }


def _trajectory_row(
    problem: Any,
    alpha0: float,
    seed: int,
    method: str,
    update: int,
    alpha_used: float | str,
    point: np.ndarray,
) -> dict[str, Any]:
    finite = bool(np.all(np.isfinite(point)))
    if finite:
        gap = quadratic_loss(point, problem)
        initial_gap = quadratic_loss(np.asarray(problem.initial_point), problem)
        coordinates: dict[str, Any] = _coordinate_values("x", point)
        fraction: float | str = gap / initial_gap
    else:
        gap = ""
        fraction = ""
        coordinates = {field: "" for field in COORDINATE_FIELDS}
    return {
        "problem": problem.key,
        "condition_number": problem.condition_number,
        "initial_learning_rate": alpha0,
        "dimension": DIMENSION,
        "seed": seed,
        "method": method,
        "update": update,
        "alpha_used": alpha_used,
        **coordinates,
        "objective_gap": gap,
        "fraction_initial_loss": fraction,
        "finite": int(finite),
    }


def run_optimization(
    config: ExperimentConfig,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Run every problem/alpha/seed using common Gaussian directions."""
    trajectories: list[dict[str, Any]] = []
    curvature_updates: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        initial = np.asarray(problem.initial_point, dtype=np.float64)
        true_hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
        for alpha0 in config.initial_learning_rates:
            for seed in config.seeds:
                states = {method: initial.copy() for method in METHODS}
                active = {method: True for method in METHODS}
                for method in METHODS:
                    trajectories.append(
                        _trajectory_row(problem, alpha0, seed, method, 0, "", states[method])
                    )

                for update_index in range(config.updates):
                    alpha = learning_rate(alpha0, update_index)
                    epsilon = optimization_perturbations(config, seed, update_index)

                    if active["explicit_es"]:
                        estimate = estimate_gradient_and_hessian(
                            states["explicit_es"], epsilon, config.sigma, problem
                        )
                        states["explicit_es"] -= alpha * estimate.gradient
                        active["explicit_es"] = bool(
                            np.all(np.isfinite(states["explicit_es"]))
                        )

                    if active["linearized_implicit_es"]:
                        estimate = estimate_gradient_and_hessian(
                            states["linearized_implicit_es"],
                            epsilon,
                            config.sigma,
                            problem,
                        )
                        step, denominator, multiplier = linearly_implicit_step(
                            estimate.gradient, estimate.hessian_diagonal, alpha
                        )
                        states["linearized_implicit_es"] += step
                        active["linearized_implicit_es"] = bool(
                            np.all(np.isfinite(states["linearized_implicit_es"]))
                        )
                        error = estimate.hessian_diagonal - true_hessian
                        curvature_updates.append(
                            {
                                "problem": problem.key,
                                "condition_number": problem.condition_number,
                                "initial_learning_rate": alpha0,
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
                                        estimate.hessian_diagonal, start=1
                                    )
                                },
                                **{
                                    f"denominator_{index}": float(value)
                                    for index, value in enumerate(denominator, start=1)
                                },
                                **{
                                    f"multiplier_{index}": float(value)
                                    for index, value in enumerate(multiplier, start=1)
                                },
                                "relative_squared_error_mean": float(
                                    np.mean((error / true_hessian) ** 2)
                                ),
                                "negative_hessian_coordinates": int(
                                    np.count_nonzero(estimate.hessian_diagonal < 0.0)
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
                        trajectories.append(
                            _trajectory_row(
                                problem,
                                alpha0,
                                seed,
                                method,
                                update_index + 1,
                                alpha,
                                states[method],
                            )
                        )
    return tuple(trajectories), tuple(curvature_updates)


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"),) * 3
    return tuple(float(value) for value in np.quantile(array, (0.5, 0.25, 0.75)))


def summarize_optimization(
    config: ExperimentConfig, trajectories: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[str, float, str, int], list[float]] = defaultdict(list)
    for row in trajectories:
        if row["finite"]:
            grouped[
                (
                    str(row["problem"]),
                    float(row["initial_learning_rate"]),
                    str(row["method"]),
                    int(row["update"]),
                )
            ].append(float(row["fraction_initial_loss"]))
    output: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            for method in METHODS:
                for update in range(config.updates + 1):
                    values = grouped[(problem.key, alpha0, method, update)]
                    center, low, high = _quantiles(values)
                    output.append(
                        {
                            "problem": problem.key,
                            "condition_number": problem.condition_number,
                            "initial_learning_rate": alpha0,
                            "method": method,
                            "update": update,
                            "alpha_used": ""
                            if update == 0
                            else learning_rate(alpha0, update - 1),
                            "finite_seed_count": len(values),
                            "median_fraction_initial_loss": center,
                            "q25_fraction_initial_loss": low,
                            "q75_fraction_initial_loss": high,
                        }
                    )
    return tuple(output)


def decision_metrics(
    config: ExperimentConfig, trajectories: Sequence[dict[str, Any]]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[str, float, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[
            (
                str(row["problem"]),
                float(row["initial_learning_rate"]),
                int(row["seed"]),
                str(row["method"]),
            )
        ].append(row)

    per_seed: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            for seed in config.seeds:
                for method in METHODS:
                    selected = grouped[(problem.key, alpha0, seed, method)]
                    selected.sort(key=lambda row: int(row["update"]))
                    finite_rows = [row for row in selected if row["finite"]]
                    finite_run = len(finite_rows) == config.updates + 1
                    failure_update: int | str = ""
                    if not finite_run:
                        failure_update = int(
                            next(row["update"] for row in selected if not row["finite"])
                        )
                    fractions = np.asarray(
                        [float(row["fraction_initial_loss"]) for row in finite_rows],
                        dtype=np.float64,
                    )
                    reached = np.flatnonzero(fractions <= TARGET_FRACTION)
                    first_update: int | str = ""
                    if reached.size:
                        first_update = int(finite_rows[int(reached[0])]["update"])
                    final_fraction: float | str = ""
                    if finite_run:
                        final_fraction = float(fractions[-1])
                    per_seed.append(
                        {
                            "problem": problem.key,
                            "condition_number": problem.condition_number,
                            "initial_learning_rate": alpha0,
                            "seed": seed,
                            "method": method,
                            "peak_fraction_initial_loss": float(np.max(fractions)),
                            "mean_fraction_initial_loss": float(np.mean(fractions)),
                            "first_update_fraction_le_1e_minus_4": first_update,
                            "reached_target": int(first_update != ""),
                            "final_fraction_initial_loss": final_fraction,
                            "finite_run": int(finite_run),
                            "failure_update": failure_update,
                        }
                    )

    aggregate: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            for method in METHODS:
                selected = [
                    row
                    for row in per_seed
                    if row["problem"] == problem.key
                    and row["initial_learning_rate"] == alpha0
                    and row["method"] == method
                ]
                peak = [float(row["peak_fraction_initial_loss"]) for row in selected]
                mean = [float(row["mean_fraction_initial_loss"]) for row in selected]
                final = [
                    float(row["final_fraction_initial_loss"])
                    for row in selected
                    if row["final_fraction_initial_loss"] != ""
                ]
                first = [
                    float(row["first_update_fraction_le_1e_minus_4"])
                    for row in selected
                    if row["reached_target"]
                ]
                peak_q = _quantiles(peak)
                mean_q = _quantiles(mean)
                final_q = _quantiles(final)
                reached_q = _quantiles(first)
                all_reached = len(first) == len(selected)
                all_q: tuple[float | str, float | str, float | str]
                all_q = reached_q if all_reached else ("", "", "")
                aggregate.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "initial_learning_rate": alpha0,
                        "method": method,
                        "seed_count": len(selected),
                        "finite_run_count": int(sum(row["finite_run"] for row in selected)),
                        "reached_target_count": len(first),
                        "all_seeds_reached_target": int(all_reached),
                        "median_peak_fraction": peak_q[0],
                        "q25_peak_fraction": peak_q[1],
                        "q75_peak_fraction": peak_q[2],
                        "median_mean_fraction": mean_q[0],
                        "q25_mean_fraction": mean_q[1],
                        "q75_mean_fraction": mean_q[2],
                        "median_first_update_reached": reached_q[0]
                        if first
                        else "",
                        "q25_first_update_reached": reached_q[1] if first else "",
                        "q75_first_update_reached": reached_q[2] if first else "",
                        "median_first_update_all_seeds": all_q[0],
                        "q25_first_update_all_seeds": all_q[1],
                        "q75_first_update_all_seeds": all_q[2],
                        "median_final_fraction": final_q[0],
                        "q25_final_fraction": final_q[1],
                        "q75_final_fraction": final_q[2],
                    }
                )
    return tuple(per_seed), tuple(aggregate)


def curvature_diagnostics(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    grouped: dict[tuple[str, float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["problem"]),
                float(row["initial_learning_rate"]),
                int(row["seed"]),
            )
        ].append(row)
    by_seed: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            for seed in config.seeds:
                selected = grouped[(problem.key, alpha0, seed)]
                coordinate_updates = len(selected) * DIMENSION
                by_seed.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "initial_learning_rate": alpha0,
                        "seed": seed,
                        "updates": len(selected),
                        "coordinate_updates": coordinate_updates,
                        "relative_hessian_rmse": float(
                            np.sqrt(
                                np.mean(
                                    [row["relative_squared_error_mean"] for row in selected]
                                )
                            )
                        ),
                        "negative_hessian_coordinates": int(
                            sum(row["negative_hessian_coordinates"] for row in selected)
                        ),
                        "negative_hessian_fraction": float(
                            sum(row["negative_hessian_coordinates"] for row in selected)
                            / coordinate_updates
                        ),
                        "nonpositive_denominator_coordinates": int(
                            sum(
                                row["nonpositive_denominator_coordinates"]
                                for row in selected
                            )
                        ),
                        "nonpositive_denominator_fraction": float(
                            sum(
                                row["nonpositive_denominator_coordinates"]
                                for row in selected
                            )
                            / coordinate_updates
                        ),
                        "minimum_denominator": float(
                            min(row["minimum_denominator"] for row in selected)
                        ),
                        "maximum_absolute_multiplier": float(
                            max(row["maximum_absolute_multiplier"] for row in selected)
                        ),
                    }
                )

    aggregate: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            selected = [
                row
                for row in by_seed
                if row["problem"] == problem.key
                and row["initial_learning_rate"] == alpha0
            ]
            coordinate_updates = int(sum(row["coordinate_updates"] for row in selected))
            negative = int(sum(row["negative_hessian_coordinates"] for row in selected))
            nonpositive = int(
                sum(row["nonpositive_denominator_coordinates"] for row in selected)
            )
            # Weight seed-level MSE equally; every seed has the same number of updates.
            relative_rmse = float(
                np.sqrt(np.mean([row["relative_hessian_rmse"] ** 2 for row in selected]))
            )
            aggregate.append(
                {
                    "problem": problem.key,
                    "condition_number": problem.condition_number,
                    "initial_learning_rate": alpha0,
                    "seed_count": len(selected),
                    "updates_per_seed": config.updates,
                    "coordinate_updates": coordinate_updates,
                    "relative_hessian_rmse": relative_rmse,
                    "negative_hessian_coordinates": negative,
                    "negative_hessian_fraction": negative / coordinate_updates,
                    "nonpositive_denominator_coordinates": nonpositive,
                    "nonpositive_denominator_fraction": nonpositive / coordinate_updates,
                    "minimum_denominator": float(
                        min(row["minimum_denominator"] for row in selected)
                    ),
                    "maximum_absolute_multiplier": float(
                        max(row["maximum_absolute_multiplier"] for row in selected)
                    ),
                }
            )
    return tuple(by_seed), tuple(aggregate)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()
    trajectories, curvature_updates = run_optimization(config)
    optimization_summary = summarize_optimization(config, trajectories)
    per_seed, aggregate = decision_metrics(config, trajectories)
    curvature_by_seed, curvature_summary = curvature_diagnostics(
        config, curvature_updates
    )
    learning_rates = tuple(
        {
            "initial_learning_rate": alpha0,
            "update_index_t": update_index,
            "update_number": update_index + 1,
            "alpha_t": learning_rate(alpha0, update_index),
        }
        for alpha0 in config.initial_learning_rates
        for update_index in range(config.updates)
    )
    return ExperimentResult(
        trajectories=trajectories,
        optimization_summary=optimization_summary,
        decision_metrics=per_seed,
        decision_summary=aggregate,
        curvature_updates=curvature_updates,
        curvature_diagnostics_by_seed=curvature_by_seed,
        curvature_diagnostics=curvature_summary,
        learning_rates=learning_rates,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


COLORS = {"explicit_es": "#6B7280", "linearized_implicit_es": "#1F5AA6"}
STYLES = {"explicit_es": "--", "linearized_implicit_es": "-"}
MARKERS = {"explicit_es": "o", "linearized_implicit_es": "s"}


def _finish_axis(axis: plt.Axes) -> None:
    axis.grid(axis="y", color="#DEE2E8", linewidth=0.65, alpha=0.85)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.0, width=0.8)


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


def _summary_rows(
    summary: Sequence[dict[str, Any]], problem_key: str, method: str
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in summary
        if row["problem"] == problem_key and row["method"] == method
    ]
    selected.sort(key=lambda row: float(row["initial_learning_rate"]))
    return selected


def plot_learning_rate_robustness(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot time-averaged normalized loss against initial learning rate."""
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True)
    handles: list[Any] = []
    for panel, (axis, problem) in enumerate(zip(axes, PROBLEMS, strict=True)):
        for method in METHODS:
            selected = _summary_rows(summary, problem.key, method)
            x = np.asarray([row["initial_learning_rate"] for row in selected])
            center = np.asarray([row["median_mean_fraction"] for row in selected])
            low = np.asarray([row["q25_mean_fraction"] for row in selected])
            high = np.asarray([row["q75_mean_fraction"] for row in selected])
            line = axis.plot(
                x,
                center,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKERS[method],
                markersize=4.2,
                linewidth=2.0,
                label=METHOD_LABELS[method],
                zorder=3,
            )[0]
            axis.fill_between(x, low, high, color=COLORS[method], alpha=0.12, linewidth=0)
            if panel == 0:
                handles.append(line)
        reference = exact_gradient_initial_step_reference(problem.condition_number)
        axis.axvline(reference, color="#9CA3AF", linestyle=":", linewidth=1.1)
        axis.set_yscale("log")
        axis.set_xscale("log", base=2)
        axis.set_xlim(
            min(config.initial_learning_rates) / 1.15,
            max(config.initial_learning_rates) * 1.06,
        )
        axis.set_xticks(config.initial_learning_rates)
        axis.set_xticklabels([f"{value:g}" for value in config.initial_learning_rates])
        axis.set_title(rf"$\kappa={problem.condition_number}$")
        axis.set_xlabel(r"Initial rate $\alpha_0$")
        _finish_axis(axis)
    axes[0].set_ylabel("Mean loss / initial loss")
    figure.suptitle("Learning-rate robustness", y=0.99, fontsize=14)
    handles.append(
        plt.Line2D([], [], color="#9CA3AF", linestyle=":", linewidth=1.1)
    )
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS]
        + ["Exact-gradient reference"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=3,
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.16, top=0.72, wspace=0.28)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Learning-rate robustness",
        "Median time-averaged normalized loss and interquartile range over 300 updates",
    )


def plot_mentor_peak_summary(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot a concise mentor-facing summary of peak normalized loss."""
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), sharex=True)
    handles: list[Any] = []
    for panel, (axis, problem) in enumerate(zip(axes, PROBLEMS, strict=True)):
        for method in METHODS:
            selected = _summary_rows(summary, problem.key, method)
            x = np.asarray([row["initial_learning_rate"] for row in selected])
            center = np.asarray([row["median_peak_fraction"] for row in selected])
            low = np.asarray([row["q25_peak_fraction"] for row in selected])
            high = np.asarray([row["q75_peak_fraction"] for row in selected])
            line = axis.plot(
                x,
                center,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKERS[method],
                markersize=4.2,
                linewidth=2.0,
                label=METHOD_LABELS[method],
                zorder=3,
            )[0]
            axis.fill_between(
                x,
                low,
                high,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
            if panel == 0:
                handles.append(line)

        reference = exact_gradient_initial_step_reference(problem.condition_number)
        axis.axvline(reference, color="#9CA3AF", linestyle=":", linewidth=1.1)
        axis.axhline(1.0, color="#C4C9D1", linewidth=0.8, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlim(
            min(config.initial_learning_rates) / 1.15,
            max(config.initial_learning_rates) * 1.06,
        )
        axis.set_xticks(config.initial_learning_rates)
        axis.set_xticklabels([f"{value:g}" for value in config.initial_learning_rates])
        axis.set_title(rf"$\kappa={problem.condition_number}$")
        axis.set_xlabel(r"Initial rate $\alpha_0$")
        _finish_axis(axis)

    axes[0].set_ylabel("Peak loss / initial loss")
    figure.suptitle("Peak loss across initial learning rates", y=0.985, fontsize=14)
    figure.text(
        0.5,
        0.905,
        (
            r"$\alpha_t=\alpha_0/\sqrt{t+1}$"
            f"  ·  {config.population_size:,} candidates/update"
            f"  ·  {config.updates} updates"
            f"  ·  median of {len(config.seeds)} matched seeds"
        ),
        ha="center",
        va="center",
        fontsize=8.8,
        color="#596273",
    )
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=2,
        frameon=False,
        handlelength=2.6,
    )
    figure.text(
        0.99,
        0.018,
        (
            "Dotted: exact-gradient first-step reference "
            "(not a finite-sample guarantee)  ·  Log y-scales vary by panel"
        ),
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#6B7280",
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.17, top=0.68, wspace=0.28
    )
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Peak loss across initial learning rates",
        "Median peak normalized loss and interquartile range over matched seeds",
    )


def plot_stability_vs_learning_rate(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot overshoot and uncensored target time versus alpha_0."""
    _style()
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), sharex=True)
    handles: list[Any] = []
    for column, problem in enumerate(PROBLEMS):
        for method in METHODS:
            selected = _summary_rows(summary, problem.key, method)
            x = np.asarray([row["initial_learning_rate"] for row in selected])
            peak = np.asarray([row["median_peak_fraction"] for row in selected])
            peak_low = np.asarray([row["q25_peak_fraction"] for row in selected])
            peak_high = np.asarray([row["q75_peak_fraction"] for row in selected])
            line = axes[0, column].plot(
                x,
                peak,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKERS[method],
                markersize=4.0,
                linewidth=1.9,
                label=METHOD_LABELS[method],
            )[0]
            axes[0, column].fill_between(
                x, peak_low, peak_high, color=COLORS[method], alpha=0.12, linewidth=0
            )
            if column == 0:
                handles.append(line)

            target = np.asarray(
                [
                    np.nan
                    if row["median_first_update_all_seeds"] == ""
                    else float(row["median_first_update_all_seeds"])
                    for row in selected
                ]
            )
            target_low = np.asarray(
                [
                    np.nan
                    if row["q25_first_update_all_seeds"] == ""
                    else float(row["q25_first_update_all_seeds"])
                    for row in selected
                ]
            )
            target_high = np.asarray(
                [
                    np.nan
                    if row["q75_first_update_all_seeds"] == ""
                    else float(row["q75_first_update_all_seeds"])
                    for row in selected
                ]
            )
            axes[1, column].plot(
                x,
                target,
                color=COLORS[method],
                linestyle=STYLES[method],
                marker=MARKERS[method],
                markersize=4.0,
                linewidth=1.9,
            )
            axes[1, column].fill_between(
                x,
                target_low,
                target_high,
                color=COLORS[method],
                alpha=0.12,
                linewidth=0,
            )
            censored = np.isnan(target)
            axes[1, column].scatter(
                x[censored],
                np.full(np.count_nonzero(censored), config.updates),
                color=COLORS[method],
                marker="x",
                s=28,
                linewidths=1.4,
                zorder=4,
            )

        reference = exact_gradient_initial_step_reference(problem.condition_number)
        for row in range(2):
            axes[row, column].axvline(
                reference, color="#9CA3AF", linestyle=":", linewidth=1.0
            )
            axes[row, column].set_xscale("log", base=2)
            axes[row, column].set_xlim(
                min(config.initial_learning_rates) / 1.15,
                max(config.initial_learning_rates) * 1.06,
            )
            axes[row, column].set_xticks(config.initial_learning_rates)
            axes[row, column].set_xticklabels(
                [f"{value:g}" for value in config.initial_learning_rates]
            )
            _finish_axis(axes[row, column])
        axes[0, column].set_title(rf"$\kappa={problem.condition_number}$")
        axes[0, column].set_yscale("log")
        axes[0, column].axhline(1.0, color="#C4C9D1", linewidth=0.8)
        axes[1, column].set_ylim(0, 1.04 * config.updates)
        axes[1, column].set_xlabel(r"Initial rate $\alpha_0$")
    axes[0, 0].set_ylabel("Peak loss / initial loss")
    axes[1, 0].set_ylabel(r"Updates to $10^{-4}$")
    figure.suptitle("Stability and convergence", y=0.995, fontsize=14)
    handles.extend(
        [
            plt.Line2D([], [], color="#9CA3AF", linestyle=":", linewidth=1.0),
            plt.Line2D(
                [],
                [],
                color="#374151",
                linestyle="none",
                marker="x",
                markersize=5,
            ),
        ]
    )
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS]
        + ["Exact-gradient reference", "Not all seeds reached"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.10, top=0.81, hspace=0.30, wspace=0.25)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Stability and convergence",
        "Peak normalized loss and uncensored updates to target versus initial learning rate",
    )


def _trajectory_values(
    summary: Sequence[dict[str, Any]], problem_key: str, alpha0: float, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected = [
        row
        for row in summary
        if row["problem"] == problem_key
        and row["initial_learning_rate"] == alpha0
        and row["method"] == method
    ]
    selected.sort(key=lambda row: int(row["update"]))
    return (
        np.asarray([row["update"] for row in selected]),
        np.asarray([row["median_fraction_initial_loss"] for row in selected]),
        np.asarray([row["q25_fraction_initial_loss"] for row in selected]),
        np.asarray([row["q75_fraction_initial_loss"] for row in selected]),
    )


def _draw_trajectory_panel(
    axis: plt.Axes,
    summary: Sequence[dict[str, Any]],
    problem_key: str,
    alpha0: float,
    add_labels: bool,
) -> list[Any]:
    handles: list[Any] = []
    floor = TRAJECTORY_VISUALIZATION_FLOOR
    for method in METHODS:
        x, center, low, high = _trajectory_values(
            summary, problem_key, alpha0, method
        )
        line = axis.plot(
            x,
            np.maximum(center, floor),
            color=COLORS[method],
            linestyle=STYLES[method],
            linewidth=1.65,
            label=METHOD_LABELS[method] if add_labels else None,
            zorder=3,
        )[0]
        axis.fill_between(
            x,
            np.maximum(low, floor),
            np.maximum(high, floor),
            color=COLORS[method],
            alpha=0.10,
            linewidth=0,
        )
        handles.append(line)
    axis.axhline(TARGET_FRACTION, color="#B8BEC7", linewidth=0.7)
    axis.set_yscale("log")
    _finish_axis(axis)
    return handles


def plot_trajectory_grid(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Plot all problem/rate trajectories as row-scaled small multiples."""
    _style()
    figure, axes = plt.subplots(
        3,
        len(config.initial_learning_rates),
        figsize=(15.6, 7.4),
        sharex=True,
        sharey="row",
    )
    handles: list[Any] = []
    for row_index, problem in enumerate(PROBLEMS):
        for column, alpha0 in enumerate(config.initial_learning_rates):
            panel_handles = _draw_trajectory_panel(
                axes[row_index, column],
                summary,
                problem.key,
                alpha0,
                add_labels=row_index == 0 and column == 0,
            )
            if row_index == 0 and column == 0:
                handles = panel_handles
            if row_index == 0:
                axes[row_index, column].set_title(rf"$\alpha_0={alpha0:g}$")
            if column == 0:
                axes[row_index, column].set_ylabel(
                    rf"$\kappa={problem.condition_number}$" + "\nLoss / initial"
                )
            if row_index == len(PROBLEMS) - 1:
                axes[row_index, column].set_xlabel("Update")
            axes[row_index, column].set_xlim(0, config.updates)
    figure.suptitle("Optimization trajectories", y=0.995, fontsize=14)
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=2,
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.075, top=0.87, hspace=0.27, wspace=0.16)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Optimization trajectories",
        "Median and interquartile normalized loss over ten matched seeds",
    )


def plot_matched_aggressiveness(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Compare kappa * alpha_0 = 4 across the three problems."""
    _style()
    settings = tuple(
        (problem, 4.0 / problem.condition_number) for problem in PROBLEMS
    )
    missing = [
        alpha0
        for _, alpha0 in settings
        if alpha0 not in config.initial_learning_rates
    ]
    if missing:
        raise ValueError(
            f"matched-aggressiveness rates are absent from the sweep: {missing}"
        )
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True, sharey=True)
    handles: list[Any] = []
    for panel, (axis, (problem, alpha0)) in enumerate(
        zip(axes, settings, strict=True)
    ):
        panel_handles = _draw_trajectory_panel(
            axis, summary, problem.key, alpha0, add_labels=panel == 0
        )
        if panel == 0:
            handles = panel_handles
        axis.set_xlim(0, config.updates)
        axis.set_xlabel("Update")
        axis.set_title(
            rf"$\kappa={problem.condition_number},\ \alpha_0={alpha0:g}$"
        )
    axes[0].set_ylabel("Loss / initial loss")
    figure.suptitle(r"Matched initial aggressiveness: $\alpha_0\kappa=4$", y=0.99, fontsize=14)
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.87),
        ncol=2,
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.71, wspace=0.18)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Matched initial aggressiveness",
        "Median and interquartile normalized loss at alpha_0 times kappa equal to four",
    )


def plot_problem_trajectories(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
    problem: Any,
) -> None:
    """Plot one readable trajectory row for a condition number."""
    _style()
    figure, axes = plt.subplots(
        1,
        len(config.initial_learning_rates),
        figsize=(15.6, 2.95),
        sharex=True,
        sharey=True,
    )
    handles: list[Any] = []
    for column, alpha0 in enumerate(config.initial_learning_rates):
        panel_handles = _draw_trajectory_panel(
            axes[column], summary, problem.key, alpha0, add_labels=column == 0
        )
        if column == 0:
            handles = panel_handles
        axes[column].set_title(rf"$\alpha_0={alpha0:g}$")
        axes[column].set_xlabel("Update")
        axes[column].set_xlim(0, config.updates)
    axes[0].set_ylabel("Loss / initial loss")
    figure.suptitle(rf"Optimization trajectories ($\kappa={problem.condition_number}$)", y=0.995, fontsize=13)
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.68, wspace=0.15)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        rf"Optimization trajectories (kappa={problem.condition_number})",
        "Median and interquartile normalized loss over ten matched seeds",
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    return "not reached" if value == "" else f"{float(value):.4g}"


def _render_report(config: ExperimentConfig, result: ExperimentResult) -> str:
    lines = [
        "# Initial-learning-rate robustness on shifted 10-D quadratics",
        "",
        "## Protocol",
        "",
        r"The schedule is \(\alpha_t=\alpha_0/\sqrt{t+1}\), with zero-based \(t\),",
        f"and alpha0 in {list(config.initial_learning_rates)}. Each run uses "
        f"{config.population_size} candidates ({config.population_size // 2} "
        f"antithetic directions), {config.updates} updates, and {len(config.seeds)} matched seeds.",
        "The three diagonal Hessians have geometrically spaced eigenvalues from 1 "
        "to kappa=2, 4, or 8. The same Gaussian directions are used across methods, "
        "problems, and alpha0 settings.",
        "",
        "Exactly two methods are compared: raw explicit ES and raw linearly implicit "
        "ES using the signed diagonal leave-one-pair-out Hessian estimate. Gradient and "
        "curvature use the same candidate returns, so curvature adds zero evaluations. "
        "There is no replay, trust region, damping, projection, clipping, rank shaping, "
        "oracle curvature, or fallback.",
        "",
        "## Median decision metrics across seeds",
        "",
        "| kappa | alpha0 | method | peak / initial | mean / initial | target seeds | first update (all seeds) | final / initial |",
        "| ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |",
    ]
    for problem in PROBLEMS:
        for alpha0 in config.initial_learning_rates:
            for method in METHODS:
                row = next(
                    item
                    for item in result.decision_summary
                    if item["problem"] == problem.key
                    and item["initial_learning_rate"] == alpha0
                    and item["method"] == method
                )
                lines.append(
                    f"| {problem.condition_number} | {alpha0:g} | {METHOD_LABELS[method]} | "
                    f"{row['median_peak_fraction']:.4g} | {row['median_mean_fraction']:.4g} | "
                    f"{row['reached_target_count']}/{row['seed_count']} | "
                    f"{_fmt(row['median_first_update_all_seeds'])} | "
                    f"{row['median_final_fraction']:.4g} |"
                )

    lines.extend(
        [
            "",
            "## Curvature and denominator diagnostics",
            "",
            "| kappa | alpha0 | Hessian RMSE | negative h | nonpositive denominator | minimum denominator | max multiplier magnitude |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result.curvature_diagnostics:
        lines.append(
            f"| {row['condition_number']} | {row['initial_learning_rate']:g} | "
            f"{100 * row['relative_hessian_rmse']:.2f}% | "
            f"{row['negative_hessian_coordinates']} | "
            f"{row['nonpositive_denominator_coordinates']} | "
            f"{row['minimum_denominator']:.4g} | "
            f"{row['maximum_absolute_multiplier']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Reading the figures",
            "",
            "The dotted vertical line is the exact-gradient initial-step reference "
            r"\(2/\kappa\). It is not a finite-sample stability guarantee or a fitted threshold. "
            "The target-time panels plot a value only when all "
            f"{len(config.seeds)} seeds reach "
            r"\(10^{-4}\). A cross at the upper edge marks a setting where at least "
            "one seed did not reach the target; the CSV remains blank rather than "
            "inventing an update number.",
            "",
            "Every raw trajectory and exact per-update Hessian, denominator, and "
            "multiplier is retained in CSV. `manifest.json` records row counts and hashes.",
            f"Trajectory figures use a log-display floor of {TRAJECTORY_VISUALIZATION_FLOOR:g}; "
            "the CSV values are not floored.",
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
    paths: dict[str, Path] = {
        "trajectories": root / "optimization_trajectories.csv",
        "optimization_summary": root / "optimization_summary.csv",
        "decision_metrics": root / "decision_metrics.csv",
        "decision_summary": root / "decision_summary.csv",
        "curvature_updates": root / "curvature_updates.csv",
        "curvature_diagnostics_by_seed": root / "curvature_diagnostics_by_seed.csv",
        "curvature_diagnostics": root / "curvature_diagnostics.csv",
        "learning_rates": root / "learning_rate_schedule.csv",
        "robustness_pdf": figures / "learning_rate_robustness.pdf",
        "robustness_png": figures / "learning_rate_robustness.png",
        "mentor_peak_pdf": figures / "mentor_peak_loss_vs_initial_rate.pdf",
        "mentor_peak_png": figures / "mentor_peak_loss_vs_initial_rate.png",
        "stability_pdf": figures / "stability_vs_learning_rate.pdf",
        "stability_png": figures / "stability_vs_learning_rate.png",
        "grid_pdf": figures / "trajectory_grid.pdf",
        "grid_png": figures / "trajectory_grid.png",
        "matched_pdf": figures / "matched_aggressiveness_trajectories.pdf",
        "matched_png": figures / "matched_aggressiveness_trajectories.png",
        "report": root / "report.md",
        "manifest": root / "manifest.json",
    }
    for problem in PROBLEMS:
        paths[f"trajectory_kappa{problem.condition_number}_pdf"] = (
            figures / f"trajectories_kappa{problem.condition_number}.pdf"
        )
        paths[f"trajectory_kappa{problem.condition_number}_png"] = (
            figures / f"trajectories_kappa{problem.condition_number}.png"
        )

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
        paths["curvature_diagnostics_by_seed"],
        CURVATURE_DIAGNOSTIC_FIELDS,
        result.curvature_diagnostics_by_seed,
    )
    _atomic_write_csv(
        paths["curvature_diagnostics"],
        CURVATURE_DIAGNOSTIC_SUMMARY_FIELDS,
        result.curvature_diagnostics,
    )
    _atomic_write_csv(
        paths["learning_rates"], LEARNING_RATE_FIELDS, result.learning_rates
    )

    plot_learning_rate_robustness(
        paths["robustness_pdf"], paths["robustness_png"], config, result.decision_summary
    )
    plot_mentor_peak_summary(
        paths["mentor_peak_pdf"],
        paths["mentor_peak_png"],
        config,
        result.decision_summary,
    )
    plot_stability_vs_learning_rate(
        paths["stability_pdf"], paths["stability_png"], config, result.decision_summary
    )
    plot_trajectory_grid(
        paths["grid_pdf"], paths["grid_png"], config, result.optimization_summary
    )
    plot_matched_aggressiveness(
        paths["matched_pdf"],
        paths["matched_png"],
        config,
        result.optimization_summary,
    )
    for problem in PROBLEMS:
        plot_problem_trajectories(
            paths[f"trajectory_kappa{problem.condition_number}_pdf"],
            paths[f"trajectory_kappa{problem.condition_number}_png"],
            config,
            result.optimization_summary,
            problem,
        )
    _atomic_write_text(paths["report"], _render_report(config, result))

    tracked: dict[str, dict[str, Any]] = {}
    row_counts = {
        "trajectories": len(result.trajectories),
        "optimization_summary": len(result.optimization_summary),
        "decision_metrics": len(result.decision_metrics),
        "decision_summary": len(result.decision_summary),
        "curvature_updates": len(result.curvature_updates),
        "curvature_diagnostics_by_seed": len(result.curvature_diagnostics_by_seed),
        "curvature_diagnostics": len(result.curvature_diagnostics),
        "learning_rates": len(result.learning_rates),
    }
    for name, path in paths.items():
        if name == "manifest":
            continue
        tracked[name] = {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        if name in row_counts:
            tracked[name]["rows"] = row_counts[name]

    nonfinite_runs = sum(
        int(not row["finite_run"]) for row in result.decision_metrics
    )
    nonpositive_denominators = sum(
        int(row["nonpositive_denominator_coordinates"])
        for row in result.curvature_diagnostics
    )
    problem_count = len(PROBLEMS)
    rate_count = len(config.initial_learning_rates)
    seed_count = len(config.seeds)
    method_count = len(METHODS)
    expected_row_counts = {
        "trajectories": problem_count
        * rate_count
        * seed_count
        * method_count
        * (config.updates + 1),
        "optimization_summary": problem_count
        * rate_count
        * method_count
        * (config.updates + 1),
        "decision_metrics": problem_count * rate_count * seed_count * method_count,
        "decision_summary": problem_count * rate_count * method_count,
        "curvature_updates": problem_count
        * rate_count
        * seed_count
        * config.updates,
        "curvature_diagnostics_by_seed": problem_count * rate_count * seed_count,
        "curvature_diagnostics": problem_count * rate_count,
        "learning_rates": rate_count * config.updates,
    }
    all_expected_rows_present = all(
        row_counts[name] == expected
        for name, expected in expected_row_counts.items()
    )
    dependency = Path(__file__).resolve().with_name(
        "convex_10d_long_estimated_hessian.py"
    )
    manifest = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "how_initial_learning_rate_changes_explicit_vs_implicit_ES_stability",
        "methods": {
            "explicit_es": "x_next = x - alpha_t * g_hat",
            "linearized_implicit_es": "x_next = x - alpha_t * g_hat / (1 + alpha_t * h_hat_diag)",
        },
        "learning_rate_schedule": {
            "formula": "alpha_t = alpha_0 / sqrt(t + 1)",
            "zero_based_t": True,
            "initial_learning_rates": list(config.initial_learning_rates),
            "last_rates": {
                str(value): learning_rate(value, config.updates - 1)
                for value in config.initial_learning_rates
            },
        },
        "explicit_exact_gradient_stability_references": {
            str(problem.condition_number): exact_gradient_initial_step_reference(
                problem.condition_number
            )
            for problem in PROBLEMS
        },
        "evaluation_accounting": {
            "candidates_per_update_per_method": config.population_size,
            "antithetic_directions_per_update_per_method": config.population_size // 2,
            "candidate_evaluations_per_method_run": config.population_size
            * config.updates,
            "hessian_additional_candidate_evaluations": 0,
            "definition": "B candidates are B/2 directions, each evaluated at +epsilon and -epsilon",
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
            "shared_candidate_returns": True,
        },
        "decision_metrics": {
            "target_fraction_initial_loss": TARGET_FRACTION,
            "peak_includes_initial_state": True,
            "mean_includes_all_301_states": True,
            "unreached_target_encoding": "blank/NaN; never encoded as update 301",
        },
        "visualization": {
            "trajectory_log_floor": TRAJECTORY_VISUALIZATION_FLOOR,
            "floor_applies_to_figures_only": True,
            "raw_csv_values_are_unfloored": True,
        },
        "common_random_numbers": {
            "across_methods": True,
            "across_problems": True,
            "across_initial_learning_rates": True,
            "optimization_seed_sequence": "[master_seed, seed, update_index]",
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
            "fallback_update",
        ],
        "validation": {
            "nonfinite_runs": nonfinite_runs,
            "nonpositive_implicit_denominator_coordinates": nonpositive_denominators,
            "all_expected_rows_present": all_expected_rows_present,
            "expected_row_counts": expected_row_counts,
            "actual_row_counts": row_counts,
        },
        "config": asdict(config),
        "files": tracked,
        "provenance": {
            "source_file": "experiments/convex_10d_initial_lr_sweep.py",
            "source_sha256": _sha256(Path(__file__).resolve()),
            "estimator_dependency": str(dependency.relative_to(Path.cwd())),
            "estimator_dependency_sha256": _sha256(dependency),
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
        default=DEFAULT_OUTPUT_DIR,
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

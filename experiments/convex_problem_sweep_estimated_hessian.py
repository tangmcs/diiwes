#!/usr/bin/env python3
"""Advisor-facing convex sweep for explicit and linearly implicit ES.

The optimizer comparison deliberately contains exactly two methods:

* explicit ES: ``x <- x - alpha_t * g_hat``;
* linearly implicit ES:
  ``x <- x - alpha_t * g_hat / (1 + alpha_t * h_hat)``.

Both methods use raw antithetic gradient estimates.  The implicit method uses
a raw leave-one-pair-out diagonal Stein Hessian estimate from the same
evaluated population.  There is no oracle curvature, replay, trust region,
damping, projection, clipping, or rank shaping.
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
METHODS = ("explicit_es", "linearized_implicit_es")
METHOD_LABELS = {
    "explicit_es": "Explicit ES",
    "linearized_implicit_es": "Implicit ES (estimated H)",
}


@dataclass(frozen=True)
class QuadraticProblem:
    key: str
    label: str
    hessian_diagonal: tuple[float, float]
    optimum: tuple[float, float]

    @property
    def condition_number(self) -> float:
        return max(self.hessian_diagonal) / min(self.hessian_diagonal)

    @property
    def initial_point(self) -> tuple[float, float]:
        return self.optimum[0] + 1.0, self.optimum[1] + 1.0


PROBLEMS = (
    QuadraticProblem(
        key="kappa2",
        label=r"$H=\mathrm{diag}(1,2)$",
        hessian_diagonal=(1.0, 2.0),
        optimum=(2.0, -1.0),
    ),
    QuadraticProblem(
        key="kappa4",
        label=r"$H=\mathrm{diag}(1,4)$",
        hessian_diagonal=(1.0, 4.0),
        optimum=(-1.0, 2.0),
    ),
    QuadraticProblem(
        key="kappa8",
        label=r"$H=\mathrm{diag}(1,8)$",
        hessian_diagonal=(1.0, 8.0),
        optimum=(1.5, 1.0),
    ),
)


@dataclass(frozen=True)
class ExperimentConfig:
    population_size: int = 500
    updates: int = 100
    seeds: tuple[int, ...] = tuple(range(10))
    sigma: float = 0.1
    initial_learning_rate: float = 0.5
    accuracy_populations: tuple[int, ...] = (20, 50, 100, 200, 500, 1000)
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
        if not self.accuracy_populations:
            raise ValueError("accuracy_populations must be nonempty")
        if any(size < 4 or size % 2 for size in self.accuracy_populations):
            raise ValueError("all accuracy populations must be even and >= 4")
        if tuple(sorted(set(self.accuracy_populations))) != self.accuracy_populations:
            raise ValueError("accuracy_populations must be sorted and unique")
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
    curvature_updates: tuple[dict[str, Any], ...]
    curvature_accuracy: tuple[dict[str, Any], ...]
    curvature_accuracy_summary: tuple[dict[str, Any], ...]
    learning_rates: tuple[dict[str, Any], ...]


TRAJECTORY_FIELDS = (
    "problem",
    "condition_number",
    "seed",
    "method",
    "update",
    "alpha_used",
    "x1",
    "x2",
    "objective_gap",
    "fraction_initial_loss",
)
OPTIMIZATION_SUMMARY_FIELDS = (
    "problem",
    "condition_number",
    "method",
    "update",
    "alpha_used",
    "median_objective_gap",
    "q25_objective_gap",
    "q75_objective_gap",
    "median_fraction_initial_loss",
    "q25_fraction_initial_loss",
    "q75_fraction_initial_loss",
)
CURVATURE_UPDATE_FIELDS = (
    "problem",
    "condition_number",
    "seed",
    "update",
    "alpha",
    "evaluated_candidates",
    "antithetic_pairs",
    "h1_true",
    "h2_true",
    "h1_estimate",
    "h2_estimate",
    "h1_error",
    "h2_error",
    "denominator_1",
    "denominator_2",
    "multiplier_1",
    "multiplier_2",
)
CURVATURE_ACCURACY_FIELDS = (
    "problem",
    "condition_number",
    "population_size",
    "antithetic_pairs",
    "seed",
    "replicate",
    "h1_true",
    "h2_true",
    "h1_estimate",
    "h2_estimate",
    "h1_error",
    "h2_error",
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
    "h1_mean_estimate",
    "h2_mean_estimate",
    "h1_rmse",
    "h2_rmse",
)
LEARNING_RATE_FIELDS = ("update_index_t", "alpha_t")


def learning_rate(config: ExperimentConfig, update_index: int) -> float:
    """Return alpha_t = alpha_0 / sqrt(t + 1), using zero-based t."""
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    return float(config.initial_learning_rate / np.sqrt(update_index + 1.0))


def quadratic_loss(point: np.ndarray, problem: QuadraticProblem) -> float:
    displacement = np.asarray(point, dtype=np.float64) - np.asarray(
        problem.optimum, dtype=np.float64
    )
    return float(
        0.5
        * np.dot(np.asarray(problem.hessian_diagonal, dtype=np.float64), displacement**2)
    )


def leave_one_pair_out_baseline(pair_sums: np.ndarray) -> np.ndarray:
    """For pair k, return the mean pair sum over all pairs except k."""
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
    """Raw antithetic gradient and LOO diagonal Stein Hessian estimates."""
    point = np.asarray(point, dtype=np.float64)
    eps = np.asarray(perturbations, dtype=np.float64)
    if point.shape != (2,) or eps.ndim != 2 or eps.shape[1] != 2:
        raise ValueError("point must be (2,) and perturbations must be (pairs, 2)")
    if eps.shape[0] < 2 or sigma <= 0.0:
        raise ValueError("at least two pairs and positive sigma are required")

    optimum = np.asarray(problem.optimum, dtype=np.float64)
    hessian = np.asarray(problem.hessian_diagonal, dtype=np.float64)
    plus = point[None, :] + sigma * eps - optimum[None, :]
    minus = point[None, :] - sigma * eps - optimum[None, :]
    f_plus = 0.5 * np.sum(hessian[None, :] * plus**2, axis=1)
    f_minus = 0.5 * np.sum(hessian[None, :] * minus**2, axis=1)

    gradient = np.mean((f_plus - f_minus)[:, None] * eps, axis=0) / (
        2.0 * sigma
    )
    pair_sums = f_plus + f_minus
    pair_signal = pair_sums - leave_one_pair_out_baseline(pair_sums)
    hessian_diagonal = np.mean(
        pair_signal[:, None] * (eps**2 - 1.0), axis=0
    ) / (2.0 * sigma**2)
    return Estimate(gradient=gradient, hessian_diagonal=hessian_diagonal)


def linearly_implicit_step(
    gradient: np.ndarray, hessian_diagonal: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the raw diagonal linearized implicit update without safeguards."""
    denominator = 1.0 + alpha * np.asarray(hessian_diagonal, dtype=np.float64)
    if np.any(denominator == 0.0):
        raise FloatingPointError("estimated diagonal implicit system is singular")
    multiplier = 1.0 / denominator
    step = -alpha * multiplier * np.asarray(gradient, dtype=np.float64)
    if not np.all(np.isfinite(step)):
        raise FloatingPointError("linearly implicit step is nonfinite")
    return step, denominator, multiplier


def _optimization_perturbations(
    config: ExperimentConfig, seed: int, update_index: int
) -> np.ndarray:
    # Problem and method are deliberately absent: this is common random numbers
    # across both methods and all three quadratics.
    sequence = np.random.SeedSequence(
        [config.master_seed, 0, int(seed), int(update_index)]
    )
    return np.random.default_rng(sequence).normal(
        size=(config.population_size // 2, 2)
    )


def _accuracy_perturbations(
    config: ExperimentConfig, population: int, seed: int, replicate: int
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [config.master_seed, 1, int(population), int(seed), int(replicate)]
    )
    return np.random.default_rng(sequence).normal(size=(population // 2, 2))


def _trajectory_row(
    problem: QuadraticProblem,
    seed: int,
    method: str,
    update: int,
    alpha_used: float | str,
    point: np.ndarray,
) -> dict[str, Any]:
    objective_gap = quadratic_loss(point, problem)
    initial_gap = quadratic_loss(np.asarray(problem.initial_point), problem)
    return {
        "problem": problem.key,
        "condition_number": problem.condition_number,
        "seed": seed,
        "method": method,
        "update": update,
        "alpha_used": alpha_used,
        "x1": float(point[0]),
        "x2": float(point[1]),
        "objective_gap": objective_gap,
        "fraction_initial_loss": objective_gap / initial_gap,
    }


def _summarize_optimization(
    config: ExperimentConfig, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    summary: list[dict[str, Any]] = []
    for problem in PROBLEMS:
        for method in METHODS:
            for update in range(config.updates + 1):
                values = np.asarray(
                    [
                        row["objective_gap"]
                        for row in rows
                        if row["problem"] == problem.key
                        and row["method"] == method
                        and row["update"] == update
                    ],
                    dtype=np.float64,
                )
                fractions = np.asarray(
                    [
                        row["fraction_initial_loss"]
                        for row in rows
                        if row["problem"] == problem.key
                        and row["method"] == method
                        and row["update"] == update
                    ],
                    dtype=np.float64,
                )
                if values.size != len(config.seeds):
                    raise RuntimeError("optimization trajectory is incomplete")
                if fractions.size != len(config.seeds):
                    raise RuntimeError("normalized optimization trajectory is incomplete")
                summary.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "method": method,
                        "update": update,
                        "alpha_used": ""
                        if update == 0
                        else learning_rate(config, update - 1),
                        "median_objective_gap": float(np.median(values)),
                        "q25_objective_gap": float(np.quantile(values, 0.25)),
                        "q75_objective_gap": float(np.quantile(values, 0.75)),
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


def run_optimization(
    config: ExperimentConfig,
) -> tuple[
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
                eps = _optimization_perturbations(config, seed, update_index)

                explicit_estimate = estimate_gradient_and_hessian(
                    states["explicit_es"], eps, config.sigma, problem
                )
                states["explicit_es"] -= alpha * explicit_estimate.gradient

                implicit_estimate = estimate_gradient_and_hessian(
                    states["linearized_implicit_es"], eps, config.sigma, problem
                )
                step, denominator, multiplier = linearly_implicit_step(
                    implicit_estimate.gradient,
                    implicit_estimate.hessian_diagonal,
                    alpha,
                )
                states["linearized_implicit_es"] += step

                estimate_h = implicit_estimate.hessian_diagonal
                curvature.append(
                    {
                        "problem": problem.key,
                        "condition_number": problem.condition_number,
                        "seed": seed,
                        "update": update_index + 1,
                        "alpha": alpha,
                        "evaluated_candidates": config.population_size,
                        "antithetic_pairs": config.population_size // 2,
                        "h1_true": true_hessian[0],
                        "h2_true": true_hessian[1],
                        "h1_estimate": float(estimate_h[0]),
                        "h2_estimate": float(estimate_h[1]),
                        "h1_error": float(estimate_h[0] - true_hessian[0]),
                        "h2_error": float(estimate_h[1] - true_hessian[1]),
                        "denominator_1": float(denominator[0]),
                        "denominator_2": float(denominator[1]),
                        "multiplier_1": float(multiplier[0]),
                        "multiplier_2": float(multiplier[1]),
                    }
                )

                for method in METHODS:
                    if not np.all(np.isfinite(states[method])):
                        raise FloatingPointError(
                            f"{method} became nonfinite for {problem.key}, "
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

    summary = _summarize_optimization(config, trajectories)
    return tuple(trajectories), summary, tuple(curvature)


def _t95(seed_count: int) -> float:
    # Two-sided 95% Student-t critical values for the small seed counts used
    # here; normal approximation is sufficient beyond 30.
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
        21: 2.086,
        22: 2.080,
        23: 2.074,
        24: 2.069,
        25: 2.064,
        26: 2.060,
        27: 2.056,
        28: 2.052,
        29: 2.048,
        30: 2.045,
    }
    if seed_count < 2:
        return float("nan")
    return table.get(seed_count, 1.96)


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
                    eps = _accuracy_perturbations(
                        config, population, seed, replicate
                    )
                    estimate = estimate_gradient_and_hessian(
                        point, eps, config.sigma, problem
                    ).hessian_diagonal
                    error = estimate - true_hessian
                    relative_squared_error = float(
                        np.mean((error / true_hessian) ** 2)
                    )
                    raw.append(
                        {
                            "problem": problem.key,
                            "condition_number": problem.condition_number,
                            "population_size": population,
                            "antithetic_pairs": population // 2,
                            "seed": seed,
                            "replicate": replicate,
                            "h1_true": true_hessian[0],
                            "h2_true": true_hessian[1],
                            "h1_estimate": float(estimate[0]),
                            "h2_estimate": float(estimate[1]),
                            "h1_error": float(error[0]),
                            "h2_error": float(error[1]),
                            "relative_squared_error_mean": relative_squared_error,
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
                seed_values = [
                    row["relative_squared_error_mean"]
                    for row in selected
                    if row["seed"] == seed
                ]
                if len(seed_values) != config.accuracy_replicates:
                    raise RuntimeError("curvature accuracy rows are incomplete")
                seed_rmse.append(float(np.sqrt(np.mean(seed_values))))
            seed_rmse_array = np.asarray(seed_rmse, dtype=np.float64)
            center = float(np.mean(seed_rmse_array))
            half_width = float(
                _t95(len(config.seeds))
                * np.std(seed_rmse_array, ddof=1)
                / np.sqrt(len(config.seeds))
            )
            estimates = np.asarray(
                [[row["h1_estimate"], row["h2_estimate"]] for row in selected],
                dtype=np.float64,
            )
            errors = np.asarray(
                [[row["h1_error"], row["h2_error"]] for row in selected],
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
                    "h1_mean_estimate": float(np.mean(estimates[:, 0])),
                    "h2_mean_estimate": float(np.mean(estimates[:, 1])),
                    "h1_rmse": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
                    "h2_rmse": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
                }
            )
    return tuple(raw), tuple(summary)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    config.validate()
    trajectories, optimization_summary, curvature_updates = run_optimization(
        config
    )
    curvature_accuracy, curvature_accuracy_summary = run_curvature_accuracy(
        config
    )
    schedule = tuple(
        {
            "update_index_t": update_index,
            "alpha_t": learning_rate(config, update_index),
        }
        for update_index in range(config.updates)
    )
    return ExperimentResult(
        trajectories=trajectories,
        optimization_summary=optimization_summary,
        curvature_updates=curvature_updates,
        curvature_accuracy=curvature_accuracy,
        curvature_accuracy_summary=curvature_accuracy_summary,
        learning_rates=schedule,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 10.0,
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
    """Three concise panels with a single two-method legend."""
    _style()
    colors = {"explicit_es": "#6B7280", "linearized_implicit_es": "#1F5AA6"}
    styles = {"explicit_es": "--", "linearized_implicit_es": "-"}
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.75), sharey=True)
    handles: list[Any] = []
    positive_floor = 1e-30

    for panel, (axis, problem) in enumerate(zip(axes, PROBLEMS, strict=True)):
        for method in METHODS:
            selected = [
                row
                for row in summary
                if row["problem"] == problem.key and row["method"] == method
            ]
            selected.sort(key=lambda row: int(row["update"]))
            x = np.asarray([row["update"] for row in selected], dtype=np.int64)
            median = np.asarray(
                [row["median_fraction_initial_loss"] for row in selected],
                dtype=np.float64,
            )
            lower = np.asarray(
                [row["q25_fraction_initial_loss"] for row in selected],
                dtype=np.float64,
            )
            upper = np.asarray(
                [row["q75_fraction_initial_loss"] for row in selected],
                dtype=np.float64,
            )
            line = axis.plot(
                x,
                np.maximum(median, positive_floor),
                color=colors[method],
                linestyle=styles[method],
                linewidth=2.2,
                label=METHOD_LABELS[method],
                zorder=3,
            )[0]
            axis.fill_between(
                x,
                np.maximum(lower, positive_floor),
                np.maximum(upper, positive_floor),
                color=colors[method],
                alpha=0.13,
                linewidth=0.0,
                zorder=2,
            )
            if panel == 0:
                handles.append(line)
        axis.set_title(problem.label, pad=7)
        axis.set_yscale("log")
        axis.set_xlabel("Update")
        axis.set_xlim(0, config.updates)
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.8)
    axes[0].set_ylabel("Fraction of initial loss")
    figure.suptitle(
        "Explicit and linearly implicit ES",
        x=0.5,
        y=0.99,
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.915,
        (
            r"$\alpha_t=0.5/\sqrt{t+1}$"
            f"   |   B={config.population_size}   |   {len(config.seeds)} matched seeds"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.14, top=0.72, wspace=0.18)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Explicit and linearly implicit ES across convex quadratics",
        "Fraction of initial loss under a common decreasing learning-rate schedule",
    )


def plot_curvature_accuracy(
    pdf_path: Path,
    png_path: Path,
    config: ExperimentConfig,
    summary: Sequence[dict[str, Any]],
) -> None:
    """Small multiples avoid a three-series legend."""
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.55), sharey=True)
    color = "#1F5AA6"
    for axis, problem in zip(axes, PROBLEMS, strict=True):
        selected = [row for row in summary if row["problem"] == problem.key]
        selected.sort(key=lambda row: int(row["population_size"]))
        population = np.asarray(
            [row["population_size"] for row in selected], dtype=np.float64
        )
        center = 100.0 * np.asarray(
            [row["relative_rmse_mean"] for row in selected], dtype=np.float64
        )
        lower = 100.0 * np.asarray(
            [row["relative_rmse_ci95_low"] for row in selected], dtype=np.float64
        )
        upper = 100.0 * np.asarray(
            [row["relative_rmse_ci95_high"] for row in selected], dtype=np.float64
        )
        axis.fill_between(
            population, lower, upper, color=color, alpha=0.14, linewidth=0.0
        )
        axis.plot(
            population,
            center,
            color=color,
            marker="o",
            markersize=4.5,
            linewidth=2.1,
            zorder=3,
        )
        axis.set_title(problem.label, pad=7)
        axis.set_xscale("log")
        axis.set_xticks(config.accuracy_populations)
        axis.set_xticklabels([str(value) for value in config.accuracy_populations])
        axis.set_xlabel("Evaluated candidates")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.0, width=0.8)
    axes[0].set_ylabel("Relative Hessian RMSE (%)")
    figure.suptitle(
        "Diagonal Hessian estimation accuracy",
        x=0.5,
        y=0.985,
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.895,
        f"B candidates (B/2 antithetic pairs)   |   {config.accuracy_replicates} estimates per seed",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.76, wspace=0.18)
    _save_figure(
        figure,
        pdf_path,
        png_path,
        "Diagonal Hessian estimation accuracy",
        "Relative RMSE versus evaluated antithetic population",
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
    final_rows = [
        row for row in result.optimization_summary if row["update"] == config.updates
    ]
    accuracy_500 = [
        row
        for row in result.curvature_accuracy_summary
        if row["population_size"] == config.population_size
    ]
    lines = [
        "# Explicit versus linearly implicit ES on shifted convex quadratics",
        "",
        "## Protocol",
        "",
        r"Each problem is \(f(x)=\tfrac12(x-x^\star)^T H(x-x^\star)\),",
        r"initialized at \(x_0=x^\star+(1,1)\).",
        "",
        "| Problem | Hessian | Optimum | Condition number |",
        "| :--- | :--- | :--- | ---: |",
    ]
    for problem in PROBLEMS:
        lines.append(
            f"| {problem.key} | `diag{problem.hessian_diagonal}` | "
            f"`{problem.optimum}` | {problem.condition_number:g} |"
        )
    lines.extend(
        [
            "",
            r"- Explicit: \(x_{t+1}=x_t-\alpha_t\widehat g_t\).",
            r"- Implicit: \(x_{t+1}=x_t-\alpha_t(I+\alpha_t\widehat H_t)^{-1}\widehat g_t\).",
            r"- Schedule: \(\alpha_t=0.5/\sqrt{t+1}\), from 0.5 to 0.05.",
            f"- Optimization: {config.updates} updates, {config.population_size} "
            f"evaluated candidates ({config.population_size // 2} antithetic pairs), "
            f"{len(config.seeds)} matched seeds, sigma={config.sigma:g}.",
            "- No replay, trust region, damping, projection, clipping, rank shaping, "
            "or exact-curvature optimizer is used.",
            f"- All {len(PROBLEMS) * len(METHODS) * len(config.seeds)} method/problem/seed runs completed with finite states.",
            r"- The optimization figure plots \((f(x_t)-f^\star)/(f(x_0)-f^\star)\): every panel starts at 1 and lower is better. Raw gaps remain in the CSV.",
            "",
            "## Curvature estimate",
            "",
            r"For pair \(k\), let \(s_k=f(x+\sigma\epsilon_k)+f(x-\sigma\epsilon_k)\) and",
            r"\(b_k=(m-1)^{-1}\sum_{\ell\ne k}s_\ell\). The diagonal estimate is",
            r"\[\widehat H_{jj}=\frac1m\sum_{k=1}^m"
            r"\frac{(s_k-b_k)(\epsilon_{k,j}^2-1)}{2\sigma^2}.\]",
            "The same candidate evaluations supply both the gradient and Hessian "
            "estimate; the implicit method requires no additional evaluations.",
            "",
            "## Final median objective gap",
            "",
            "| Problem | Explicit ES | Implicit ES |",
            "| :--- | ---: | ---: |",
        ]
    )
    for problem in PROBLEMS:
        values = {
            row["method"]: row["median_objective_gap"]
            for row in final_rows
            if row["problem"] == problem.key
        }
        lines.append(
            f"| {problem.key} | {values['explicit_es']:.6g} | "
            f"{values['linearized_implicit_es']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Hessian accuracy at N=500",
            "",
            "Relative RMSE is computed coordinate-wise against the known diagonal "
            "and then averaged over independent estimates; lower is better.",
            "",
            "| Problem | Relative RMSE | Mean h1 estimate | Mean h2 estimate |",
            "| :--- | ---: | ---: | ---: |",
        ]
    )
    for problem in PROBLEMS:
        row = next(item for item in accuracy_500 if item["problem"] == problem.key)
        lines.append(
            f"| {problem.key} | {100.0 * row['relative_rmse_mean']:.2f}% | "
            f"{row['h1_mean_estimate']:.4g} | {row['h2_mean_estimate']:.4g} |"
        )
    lines.extend(
        [
            "",
            "The CSV files retain every trajectory, update-level Hessian estimate, "
            "accuracy replicate, and plotted aggregate. `manifest.json` records "
            "configuration, file hashes, row counts, and source provenance.",
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
        "curvature_updates": root / "curvature_updates.csv",
        "curvature_accuracy": root / "curvature_accuracy.csv",
        "curvature_accuracy_summary": root / "curvature_accuracy_summary.csv",
        "learning_rates": root / "learning_rate_schedule.csv",
        "optimization_pdf": figures / "optimization_trajectories.pdf",
        "optimization_png": figures / "optimization_trajectories.png",
        "accuracy_pdf": figures / "curvature_accuracy.pdf",
        "accuracy_png": figures / "curvature_accuracy.png",
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
        "curvature_updates": len(result.curvature_updates),
        "curvature_accuracy": len(result.curvature_accuracy),
        "curvature_accuracy_summary": len(result.curvature_accuracy_summary),
        "learning_rates": len(result.learning_rates),
    }
    for name, rows in row_counts.items():
        tracked[name]["rows"] = rows

    manifest = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "explicit_ES_versus_linearly_implicit_ES_across_shifted_convex_quadratics",
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
            "optimization_candidates_per_update": config.population_size,
            "optimization_antithetic_pairs_per_update": config.population_size // 2,
            "hessian_additional_candidate_evaluations": 0,
            "definition": "B candidates are B/2 Gaussian directions, each evaluated at +epsilon and -epsilon",
        },
        "problems": [
            {
                **asdict(problem),
                "initial_point": list(problem.initial_point),
                "condition_number": problem.condition_number,
                "objective": "0.5 * (x - optimum)^T diag(hessian_diagonal) (x - optimum)",
            }
            for problem in PROBLEMS
        ],
        "estimators": {
            "gradient": "raw antithetic loss difference",
            "hessian_diagonal": "raw diagonal Stein estimate",
            "baseline": "leave one antithetic pair out",
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
            "source_file": "experiments/convex_problem_sweep_estimated_hessian.py",
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
        default="reports/convex_problem_sweep_presentation",
        help="folder for auditable CSV, PDF, PNG, report, and manifest outputs",
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

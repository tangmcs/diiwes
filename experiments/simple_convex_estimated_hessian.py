#!/usr/bin/env python3
"""Minimal explicit-versus-implicit ES experiment on a convex quadratic.

The experiment intentionally has only two optimization methods:

* explicit ES: ``x <- x - alpha * g_hat``;
* linearly implicit ES: ``x <- x - alpha * g_hat / (1 + alpha * h_hat)``.

Both ``g_hat`` and the diagonal ``h_hat`` are raw-return Monte Carlo
estimates.  The Hessian estimator uses antithetic pair sums and a
leave-one-pair-out baseline.  There is no replay, trust region, additive
damping, projection, clipping, rank transform, or exact-curvature oracle.
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


EXPERIMENT_VERSION = "1.2.0"
METHODS = ("explicit_es", "linearized_implicit_es")
METHOD_LABELS = {
    "explicit_es": "Explicit ES",
    "linearized_implicit_es": "Linearly implicit ES (estimated Hessian)",
}
RUN_FIELDS = ("seed", "alpha", "method", "update", "x1", "x2", "loss")
CURVATURE_FIELDS = (
    "seed",
    "alpha",
    "update",
    "h11_estimate",
    "h22_estimate",
    "h11_true",
    "h22_true",
    "h11_error",
    "h22_error",
    "denominator_1",
    "denominator_2",
    "implicit_multiplier_1",
    "implicit_multiplier_2",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for the advisor-facing experiment."""

    population_size: int = 500
    iterations: int = 30
    alphas: tuple[float, float] = (0.25, 1.0)
    seeds: tuple[int, ...] = tuple(range(10))
    sigma: float = 0.1
    hessian_diagonal: tuple[float, float] = (1.0, 2.0)
    initial_point: tuple[float, float] = (1.0, 1.0)
    optimum_point: tuple[float, float] = (0.0, 0.0)
    problem_name: str = "origin"
    master_seed: int = 20260715

    def validate(self) -> None:
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be even and at least four")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if len(self.alphas) != 2 or any(alpha <= 0.0 for alpha in self.alphas):
            raise ValueError("alphas must contain two positive step sizes")
        if not self.alphas[0] < self.alphas[1]:
            raise ValueError("alphas must be ordered as safe then large")
        if not self.seeds or any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be nonempty and nonnegative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if len(self.hessian_diagonal) != 2 or any(
            value <= 0.0 for value in self.hessian_diagonal
        ):
            raise ValueError("the two Hessian eigenvalues must be positive")
        if len(self.initial_point) != 2:
            raise ValueError("initial_point must be two dimensional")
        if len(self.optimum_point) != 2:
            raise ValueError("optimum_point must be two dimensional")
        if self.problem_name not in {"origin", "shifted"}:
            raise ValueError("problem_name must be 'origin' or 'shifted'")


@dataclass(frozen=True)
class MonteCarloEstimate:
    gradient: np.ndarray
    hessian_diagonal: np.ndarray


@dataclass(frozen=True)
class ExperimentResult:
    runs: tuple[dict[str, Any], ...]
    curvature: tuple[dict[str, Any], ...]


def config_for_problem(problem: str) -> ExperimentConfig:
    """Return the fixed advisor-facing protocol for a named quadratic."""
    if problem == "origin":
        return ExperimentConfig()
    if problem == "shifted":
        return ExperimentConfig(
            initial_point=(3.0, 0.0),
            optimum_point=(2.0, -1.0),
            problem_name="shifted",
        )
    raise ValueError("problem must be 'origin' or 'shifted'")


def quadratic_loss(
    point: np.ndarray,
    hessian_diagonal: np.ndarray,
    optimum_point: np.ndarray | None = None,
) -> float:
    """Return f(x) = 0.5 (x-x*)^T H (x-x*) for diagonal positive H."""
    point = np.asarray(point, dtype=np.float64)
    hessian_diagonal = np.asarray(hessian_diagonal, dtype=np.float64)
    optimum = (
        np.zeros_like(point)
        if optimum_point is None
        else np.asarray(optimum_point, dtype=np.float64)
    )
    displacement = point - optimum
    return float(0.5 * np.dot(hessian_diagonal, displacement**2))


def evaluate_antithetic(
    point: np.ndarray,
    perturbations: np.ndarray,
    sigma: float,
    hessian_diagonal: np.ndarray,
    optimum_point: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate deterministic loss at x +/- sigma epsilon."""
    plus_points = point[None, :] + sigma * perturbations
    minus_points = point[None, :] - sigma * perturbations
    optimum = (
        np.zeros_like(point)
        if optimum_point is None
        else np.asarray(optimum_point, dtype=np.float64)
    )
    plus = 0.5 * np.sum(
        hessian_diagonal * (plus_points - optimum) ** 2, axis=1
    )
    minus = 0.5 * np.sum(
        hessian_diagonal * (minus_points - optimum) ** 2, axis=1
    )
    return plus, minus


def leave_one_pair_out_baseline(pair_sums: np.ndarray) -> np.ndarray:
    """For each pair, return the mean pair sum over all other pairs."""
    pair_sums = np.asarray(pair_sums, dtype=np.float64)
    if pair_sums.ndim != 1 or len(pair_sums) < 2:
        raise ValueError("at least two one-dimensional pair sums are required")
    return (float(np.sum(pair_sums)) - pair_sums) / float(len(pair_sums) - 1)


def estimate_gradient_and_diagonal_hessian(
    point: np.ndarray,
    perturbations: np.ndarray,
    sigma: float,
    hessian_diagonal: np.ndarray,
    optimum_point: np.ndarray | None = None,
) -> MonteCarloEstimate:
    """Estimate the raw antithetic gradient and diagonal Stein Hessian."""
    perturbations = np.asarray(perturbations, dtype=np.float64)
    if perturbations.ndim != 2 or perturbations.shape[1] != 2:
        raise ValueError("perturbations must have shape (pair_count, 2)")
    plus, minus = evaluate_antithetic(
        np.asarray(point, dtype=np.float64),
        perturbations,
        sigma,
        np.asarray(hessian_diagonal, dtype=np.float64),
        optimum_point,
    )
    gradient = np.mean(
        (plus - minus)[:, None] * perturbations, axis=0
    ) / (2.0 * sigma)

    pair_sums = plus + minus
    baseline = leave_one_pair_out_baseline(pair_sums)
    pair_signal = pair_sums - baseline
    diagonal = np.mean(
        pair_signal[:, None] * (perturbations**2 - 1.0), axis=0
    ) / (2.0 * sigma**2)
    return MonteCarloEstimate(gradient=gradient, hessian_diagonal=diagonal)


def explicit_step(gradient: np.ndarray, alpha: float) -> np.ndarray:
    return -alpha * np.asarray(gradient, dtype=np.float64)


def linearly_implicit_step(
    gradient: np.ndarray,
    hessian_diagonal: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the diagonal linearized implicit equation without safeguards."""
    denominator = 1.0 + alpha * np.asarray(
        hessian_diagonal, dtype=np.float64
    )
    if np.any(denominator == 0.0):
        raise FloatingPointError("the unsafeguarded implicit system is singular")
    multiplier = 1.0 / denominator
    step = -alpha * multiplier * np.asarray(gradient, dtype=np.float64)
    if not np.all(np.isfinite(step)):
        raise FloatingPointError("the unsafeguarded implicit step is nonfinite")
    return step, denominator, multiplier


def _perturbations(config: ExperimentConfig, seed: int, update: int) -> np.ndarray:
    """Generate a batch shared across methods and step-size panels."""
    sequence = np.random.SeedSequence([config.master_seed, seed, update])
    rng = np.random.default_rng(sequence)
    return rng.normal(size=(config.population_size // 2, 2))


def _run_row(
    seed: int,
    alpha: float,
    method: str,
    update: int,
    point: np.ndarray,
    hessian_diagonal: np.ndarray,
    optimum_point: np.ndarray,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "alpha": alpha,
        "method": method,
        "update": update,
        "x1": float(point[0]),
        "x2": float(point[1]),
        "loss": quadratic_loss(point, hessian_diagonal, optimum_point),
    }


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run both methods with matched Monte Carlo batches."""
    config.validate()
    hessian = np.asarray(config.hessian_diagonal, dtype=np.float64)
    initial = np.asarray(config.initial_point, dtype=np.float64)
    optimum = np.asarray(config.optimum_point, dtype=np.float64)
    runs: list[dict[str, Any]] = []
    curvature: list[dict[str, Any]] = []

    for alpha in config.alphas:
        for seed in config.seeds:
            states = {method: initial.copy() for method in METHODS}
            for method in METHODS:
                runs.append(
                    _run_row(
                        seed,
                        alpha,
                        method,
                        0,
                        states[method],
                        hessian,
                        optimum,
                    )
                )

            for update in range(1, config.iterations + 1):
                # This exact perturbation matrix is used by both methods.  The
                # alpha is deliberately absent from the seed sequence so both
                # panels also use matched random numbers.
                eps = _perturbations(config, seed, update - 1)

                explicit_estimate = estimate_gradient_and_diagonal_hessian(
                    states["explicit_es"],
                    eps,
                    config.sigma,
                    hessian,
                    optimum,
                )
                states["explicit_es"] += explicit_step(
                    explicit_estimate.gradient, alpha
                )

                implicit_estimate = estimate_gradient_and_diagonal_hessian(
                    states["linearized_implicit_es"],
                    eps,
                    config.sigma,
                    hessian,
                    optimum,
                )
                implicit_update, denominator, multiplier = linearly_implicit_step(
                    implicit_estimate.gradient,
                    implicit_estimate.hessian_diagonal,
                    alpha,
                )
                states["linearized_implicit_es"] += implicit_update

                estimate_h = implicit_estimate.hessian_diagonal
                curvature.append(
                    {
                        "seed": seed,
                        "alpha": alpha,
                        "update": update,
                        "h11_estimate": float(estimate_h[0]),
                        "h22_estimate": float(estimate_h[1]),
                        "h11_true": float(hessian[0]),
                        "h22_true": float(hessian[1]),
                        "h11_error": float(estimate_h[0] - hessian[0]),
                        "h22_error": float(estimate_h[1] - hessian[1]),
                        "denominator_1": float(denominator[0]),
                        "denominator_2": float(denominator[1]),
                        "implicit_multiplier_1": float(multiplier[0]),
                        "implicit_multiplier_2": float(multiplier[1]),
                    }
                )
                for method in METHODS:
                    if not np.all(np.isfinite(states[method])):
                        raise FloatingPointError(
                            f"{method} became nonfinite at alpha={alpha}, "
                            f"seed={seed}, update={update}"
                        )
                    runs.append(
                        _run_row(
                            seed,
                            alpha,
                            method,
                            update,
                            states[method],
                            hessian,
                            optimum,
                        )
                    )

    return ExperimentResult(runs=tuple(runs), curvature=tuple(curvature))


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


def _objective_plain(config: ExperimentConfig) -> str:
    if config.problem_name == "origin":
        return "0.5 * (x1^2 + 2*x2^2)"
    return "0.5 * ((x1 - 2)^2 + 2*(x2 + 1)^2)"


def _objective_latex(config: ExperimentConfig) -> str:
    if config.problem_name == "origin":
        return r"f(x)=\frac{1}{2}(x_1^2+2x_2^2)"
    return r"f(x)=\frac{1}{2}[(x_1-2)^2+2(x_2+1)^2]"


def _plot_trajectories(
    path_pdf: Path,
    path_png: Path,
    config: ExperimentConfig,
    runs: Sequence[dict[str, Any]],
) -> None:
    colors = {
        "explicit_es": "#66717E",
        "linearized_implicit_es": "#1F5AA6",
    }
    linestyles = {"explicit_es": "--", "linearized_implicit_es": "-"}
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), constrained_layout=False)
    updates = np.arange(config.iterations + 1)

    handles = []
    for panel, (axis, alpha) in enumerate(zip(axes, config.alphas, strict=True)):
        for method in METHODS:
            matrix = np.asarray(
                [
                    [
                        float(row["loss"])
                        for row in runs
                        if float(row["alpha"]) == alpha
                        and int(row["seed"]) == seed
                        and row["method"] == method
                    ]
                    for seed in config.seeds
                ],
                dtype=np.float64,
            )
            if matrix.shape != (len(config.seeds), config.iterations + 1):
                raise RuntimeError("trajectory rows are incomplete or out of order")
            median = np.median(matrix, axis=0)
            lower = np.quantile(matrix, 0.25, axis=0)
            upper = np.quantile(matrix, 0.75, axis=0)
            positive_floor = np.finfo(np.float64).tiny
            line = axis.plot(
                updates,
                np.maximum(median, positive_floor),
                color=colors[method],
                linestyle=linestyles[method],
                linewidth=2.25,
                label=METHOD_LABELS[method],
                zorder=3,
            )[0]
            axis.fill_between(
                updates,
                np.maximum(lower, positive_floor),
                np.maximum(upper, positive_floor),
                color=colors[method],
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )
            if panel == 0:
                handles.append(line)

        label = "Safe step" if panel == 0 else "Stability boundary"
        axis.set_title(f"({chr(97 + panel)}) {label}:  $\\alpha={alpha:g}$", pad=8)
        axis.set_yscale("log")
        axis.set_xlabel("Update")
        axis.grid(axis="y", color="#D9DEE5", linewidth=0.7, alpha=0.85)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out", length=3.5, width=0.8)
    axes[0].set_ylabel(r"Objective  $f(x_t)$")

    figure.suptitle(
        "Explicit and linearly implicit ES",
        x=0.5,
        y=0.975,
        fontsize=14,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.905,
        (
            f"${_objective_latex(config)}$; "
            f"population {config.population_size}; median and interquartile range "
            f"over {len(config.seeds)} matched seeds"
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
        bbox_to_anchor=(0.5, 0.825),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.2,
    )
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.13, top=0.69, wspace=0.25)

    metadata = {
        "Title": (
            "Explicit and linearly implicit ES on a "
            f"{config.problem_name}-centered convex quadratic"
        ),
        "Author": "DIIWES experiment",
        "Subject": "Raw-loss trajectories with an estimated diagonal Hessian",
    }
    temporary_pdf = path_pdf.with_name(f".{path_pdf.name}.tmp")
    temporary_png = path_png.with_name(f".{path_png.name}.tmp")
    figure.savefig(
        temporary_pdf,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    figure.savefig(
        temporary_png,
        format="png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": metadata["Title"]},
    )
    plt.close(figure)
    os.replace(temporary_pdf, path_pdf)
    os.replace(temporary_png, path_png)


def _median_final_loss(
    result: ExperimentResult, alpha: float, method: str, iterations: int
) -> float:
    values = [
        float(row["loss"])
        for row in result.runs
        if float(row["alpha"]) == alpha
        and row["method"] == method
        and int(row["update"]) == iterations
    ]
    return float(np.median(values))


def _median_final_point(
    result: ExperimentResult, alpha: float, method: str, iterations: int
) -> tuple[float, float]:
    values = np.asarray(
        [
            [float(row["x1"]), float(row["x2"])]
            for row in result.runs
            if float(row["alpha"]) == alpha
            and row["method"] == method
            and int(row["update"]) == iterations
        ],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != 2:
        raise RuntimeError("final-coordinate rows are incomplete")
    median = np.median(values, axis=0)
    return float(median[0]), float(median[1])


def _render_report(config: ExperimentConfig, result: ExperimentResult) -> str:
    h_values = np.asarray(
        [
            [row["h11_estimate"], row["h22_estimate"]]
            for row in result.curvature
        ],
        dtype=np.float64,
    )
    h_mean = np.mean(h_values, axis=0)
    h_std = np.std(h_values, axis=0, ddof=1)
    safe_alpha, boundary_alpha = config.alphas
    initial_loss = quadratic_loss(
        np.asarray(config.initial_point),
        np.asarray(config.hessian_diagonal),
        np.asarray(config.optimum_point),
    )
    safe_explicit = tuple(
        1.0 - safe_alpha * value for value in config.hessian_diagonal
    )
    safe_implicit = tuple(
        1.0 / (1.0 + safe_alpha * value)
        for value in config.hessian_diagonal
    )
    boundary_explicit = tuple(
        1.0 - boundary_alpha * value for value in config.hessian_diagonal
    )
    boundary_implicit = tuple(
        1.0 / (1.0 + boundary_alpha * value)
        for value in config.hessian_diagonal
    )

    def result_row(alpha: float, method: str, label: str) -> str:
        x1, x2 = _median_final_point(result, alpha, method, config.iterations)
        loss = _median_final_loss(result, alpha, method, config.iterations)
        return f"| {alpha:g} | {label} | {loss:.6g} | {x1:.6g} | {x2:.6g} |"

    lines = [
        "# Minimal estimated-Hessian convex experiment",
        "",
        "## Question",
        "",
        "Does the linearly implicit ES update remain convergent at the explicit",
        "stability boundary when both methods use matched sampled gradients?",
        "",
        "## Problem and two methods",
        "",
        f"The deterministic objective is \\({_objective_latex(config)}\\),",
        rf"with optimum \(x^\star={config.optimum_point}\), so the known Hessian is",
        r"\(H=\operatorname{diag}(1,2)\). The code does",
        "not give this known Hessian to the implicit optimizer; it estimates the",
        "diagonal from the same raw antithetic population.",
        "",
        r"- Explicit ES: \(x_{t+1}=x_t-\alpha\widehat g_t\).",
        r"- Implicit ES: \(x_{t+1}=x_t-\alpha(I+\alpha\widehat H_t)^{-1}\widehat g_t\),",
        r"  using the estimated diagonal \(\widehat H_t\).",
        "",
        "The shifted run is a translation-invariance check: its initial",
        r"displacement is \(x_0-x^\star=(1,1)\), exactly matching the",
        "origin-centered run. Therefore the two problems should have identical",
        "loss, gradient-displacement, and Hessian-estimate trajectories; only",
        "the absolute coordinates should be translated by the new optimum.",
        "",
        "## Fixed settings",
        "",
        f"- Initial point: `{config.initial_point}`; initial loss: "
        f"`{initial_loss:g}`; optimum: `{config.optimum_point}`.",
        f"- Population: `{config.population_size}` = "
        f"`{config.population_size // 2}` antithetic pairs.",
        f"- Updates: `{config.iterations}`; perturbation scale: `{config.sigma:g}`.",
        f"- Matched seeds: `{len(config.seeds)}`.",
        f"- Safe step size: `{safe_alpha:g}`; explicit stability boundary: `{boundary_alpha:g}`.",
        "- The environment is deterministic. The only randomness is ES sampling.",
        "- Each method and each step-size panel receives the same perturbations",
        "  for a given seed and update.",
        "",
        "There is no replay, trust region, additive damping, curvature projection,",
        "curvature clipping, multiplier clipping, rank shaping, or oracle update.",
        "The coordinate-wise factor `1 / (1 + alpha * h_hat)` is the direct",
        "solution of the linearized implicit equation, not an added safeguard.",
        "",
        "For this quadratic, the largest eigenvalue is 2, so explicit gradient",
        r"descent is stable only for \(0<\alpha<2/\lambda_{\max}=1\). The",
        "true-curvature contraction factors below explain the two selected panels;",
        "the optimizer itself still uses sampled gradients and Hessian estimates.",
        "",
        "| Step size | Explicit factors | Implicit factors |",
        "| ---: | ---: | ---: |",
        (
            f"| {safe_alpha:g} | \\({safe_explicit[0]:.2f},\\ "
            f"{safe_explicit[1]:.2f}\\) | \\({safe_implicit[0]:.2f},\\ "
            f"{safe_implicit[1]:.2f}\\) |"
        ),
        (
            f"| {boundary_alpha:g} | \\({boundary_explicit[0]:.2f},\\ "
            f"{boundary_explicit[1]:.2f}\\) | \\({boundary_implicit[0]:.2f},\\ "
            f"{boundary_implicit[1]:.2f}\\) |"
        ),
        "",
        "## Result",
        "",
        "Median final convex loss and component-wise median final coordinates",
        "across matched seeds:",
        "",
        "| Step size | Method | Final loss | Final median x1 | Final median x2 |",
        "| ---: | :--- | ---: | ---: | ---: |",
        result_row(safe_alpha, "explicit_es", "Explicit ES"),
        result_row(
            safe_alpha,
            "linearized_implicit_es",
            "Implicit ES (estimated Hessian)",
        ),
        result_row(boundary_alpha, "explicit_es", "Explicit ES"),
        result_row(
            boundary_alpha,
            "linearized_implicit_es",
            "Implicit ES (estimated Hessian)",
        ),
        "",
        "At the safe step size, explicit ES moves faster because it does not",
        "attenuate the update. At the stability boundary, its high-curvature",
        "displacement coordinate oscillates instead of contracting, whereas the implicit",
        "denominator uses estimated curvature and continues to converge.",
        "",
        "## Curvature check",
        "",
        "Across all recorded implicit updates, the estimated diagonal mean and",
        "standard deviation were:",
        "",
        "| Coordinate | True Hessian | Estimate mean | Estimate SD |",
        "| ---: | ---: | ---: | ---: |",
        f"| 1 | {config.hessian_diagonal[0]:g} | {h_mean[0]:.6g} | {h_std[0]:.6g} |",
        f"| 2 | {config.hessian_diagonal[1]:g} | {h_mean[1]:.6g} | {h_std[1]:.6g} |",
        "",
        "`curvature.csv` records every exact estimate, denominator, and implicit",
        "multiplier. `runs.csv` records the raw loss and both coordinates at every",
        "update. The figure uses raw convex loss and contains no aggregate score.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output_dir: str | os.PathLike[str],
    config: ExperimentConfig,
    result: ExperimentResult,
) -> dict[str, str]:
    """Write CSV data, the clean figure, report, and provenance manifest."""
    root = Path(output_dir)
    figures = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": root / "runs.csv",
        "curvature": root / "curvature.csv",
        "figure_pdf": figures / "explicit_vs_implicit.pdf",
        "figure_png": figures / "explicit_vs_implicit.png",
        "report": root / "report.md",
        "manifest": root / "manifest.json",
    }

    _atomic_write_csv(paths["runs"], RUN_FIELDS, result.runs)
    _atomic_write_csv(paths["curvature"], CURVATURE_FIELDS, result.curvature)
    _plot_trajectories(
        paths["figure_pdf"], paths["figure_png"], config, result.runs
    )
    _atomic_write_text(paths["report"], _render_report(config, result))

    tracked = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in paths.items()
        if name != "manifest"
    }
    tracked["runs"]["rows"] = len(result.runs)
    tracked["curvature"]["rows"] = len(result.curvature)
    manifest = {
        "schema_version": 1,
        "experiment_version": EXPERIMENT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "explicit_ES_versus_linearized_implicit_ES_with_estimated_diagonal_Hessian",
        "problem": {
            "name": config.problem_name,
            "dimension": 2,
            "objective": _objective_plain(config),
            "initial_point": list(config.initial_point),
            "optimum_point": list(config.optimum_point),
            "minimum_value": 0.0,
            "convex": True,
            "deterministic_evaluations": True,
            "known_hessian_used_for_scoring_only": list(config.hessian_diagonal),
        },
        "methods": {
            "explicit_es": "x_next = x - alpha * g_hat",
            "linearized_implicit_es": "x_next = x - alpha * g_hat / (1 + alpha * h_hat_diag)",
        },
        "estimators": {
            "gradient": "raw antithetic return difference",
            "hessian_diagonal": "raw diagonal Stein estimate",
            "curvature_baseline": "leave one antithetic pair out",
        },
        "common_random_numbers": {
            "across_methods": True,
            "across_step_sizes": True,
            "seed_sequence": "[master_seed, seed, update]",
        },
        "excluded_components": [
            "replay",
            "trust_region",
            "additive_damping",
            "curvature_projection",
            "curvature_clipping",
            "multiplier_clipping",
            "rank_shaping",
            "oracle_optimizer",
        ],
        "config": asdict(config),
        "files": tracked,
        "provenance": {
            "source_file": "experiments/simple_convex_estimated_hessian.py",
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
        "--problem",
        choices=("origin", "shifted"),
        default="origin",
        help="quadratic optimum: origin=(0,0), shifted=(2,-1)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="folder for CSV, report, manifest, PDF, and PNG outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = config_for_problem(args.problem)
    result = run_experiment(config)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            "results/simple_convex_estimated_hessian"
            if args.problem == "origin"
            else "results/simple_shifted_convex_estimated_hessian"
        )
    outputs = write_outputs(output_dir, config, result)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

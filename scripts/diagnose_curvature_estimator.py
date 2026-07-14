#!/usr/bin/env python3
"""Measure production Stein-curvature stability on known quadratics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import DIIWES
from core.standard_es import centered_ranks


FIELDS = [
    "dimension",
    "pairs",
    "replicates",
    "alpha",
    "raw_relative_rmse_mean",
    "raw_relative_rmse_std",
    "raw_sign_accuracy_mean",
    "raw_clip_fraction_mean",
    "rank_split_correlation_mean",
    "rank_split_sign_agreement_mean",
    "raw_step_relative_error_mean",
    "exact_solve_relative_residual_max",
    "estimated_projected_solve_relative_residual_max",
    "estimated_signed_nonpositive_fraction_mean",
    "estimated_signed_condition_median",
    "estimated_signed_solve_relative_residual_max",
]


def _quadratic_returns(eps: np.ndarray, sigma: float, hessian_diag: np.ndarray) -> np.ndarray:
    points = sigma * eps
    return 0.5 * np.sum(hessian_diag[None, :] * points * points, axis=1)


def _estimate(
    eps: np.ndarray,
    fitness: np.ndarray,
    sigma: float,
    *,
    center_fitness: float | None,
) -> np.ndarray:
    n_pairs, dimension = eps.shape
    noise = np.concatenate([eps, -eps], axis=0)
    ask_info = {
        "fresh_pair_plus": np.arange(n_pairs),
        "fresh_pair_minus": np.arange(n_pairs, 2 * n_pairs),
    }
    optimizer = DIIWES(
        num_params=dimension,
        population_size=2 * n_pairs,
        noise_std=sigma,
        reuse_fraction=0.0,
        curvature_mode="diag",
    )
    estimate, count = optimizer._estimate_fresh_curvature(
        noise,
        fitness,
        ask_info,
        sigma,
        center_f_for_curv=center_fitness,
    )
    if estimate is None or count != n_pairs:
        raise RuntimeError("production curvature estimator returned an incomplete estimate")
    return estimate


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def diagnose(
    dimension: int,
    pairs: int,
    replicates: int,
    sigma: float,
    curvature_clip: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed + dimension * 10_000 + pairs)
    hessian_diag = -np.geomspace(0.25, 4.0, dimension)
    hessian_norm = float(np.linalg.norm(hessian_diag))
    raw_relative_rmse = []
    raw_sign_accuracy = []
    raw_clip_fraction = []
    rank_split_correlation = []
    rank_split_sign_agreement = []
    raw_step_relative_error = []
    exact_solve_relative_residual = []
    estimated_projected_solve_relative_residual = []
    estimated_signed_nonpositive_fraction = []
    estimated_signed_condition = []
    estimated_signed_solve_relative_residual = []
    gradient = np.ones(dimension, dtype=np.float64) / np.sqrt(float(dimension))
    true_curvature = np.maximum(-hessian_diag, 0.0)
    exact_denom = 1.0 + alpha * (0.1 + true_curvature)
    rhs = alpha * gradient
    exact_step = rhs / exact_denom

    for _ in range(replicates):
        raw_estimates = []
        rank_estimates = []
        for _split in range(2):
            eps = rng.randn(pairs, dimension)
            plus = _quadratic_returns(eps, sigma, hessian_diag)
            minus = plus.copy()
            raw_fitness = np.concatenate([plus, minus])
            raw_estimates.append(_estimate(eps, raw_fitness, sigma, center_fitness=0.0))
            rank_estimates.append(
                _estimate(eps, centered_ranks(raw_fitness), sigma, center_fitness=None)
            )

        raw_estimate = raw_estimates[0]
        raw_relative_rmse.append(float(np.linalg.norm(raw_estimate - hessian_diag) / hessian_norm))
        raw_sign_accuracy.append(float(np.mean(np.sign(raw_estimate) == np.sign(hessian_diag))))
        raw_clip_fraction.append(float(np.mean(np.maximum(-raw_estimate, 0.0) > curvature_clip)))
        estimated_curvature = np.clip(
            np.maximum(-raw_estimate, 0.0), 0.0, curvature_clip
        )
        estimated_denom = 1.0 + alpha * (0.1 + estimated_curvature)
        estimated_step = rhs / estimated_denom
        raw_step_relative_error.append(
            float(
                np.linalg.norm(estimated_step - exact_step)
                / max(float(np.linalg.norm(exact_step)), 1e-12)
            )
        )
        exact_solve_relative_residual.append(
            float(
                np.linalg.norm(exact_denom * exact_step - rhs)
                / max(float(np.linalg.norm(rhs)), 1e-12)
            )
        )
        estimated_projected_solve_relative_residual.append(
            float(
                np.linalg.norm(estimated_denom * estimated_step - rhs)
                / max(float(np.linalg.norm(rhs)), 1e-12)
            )
        )
        estimated_signed_denom = 1.0 + alpha * 0.1 - alpha * raw_estimate
        estimated_signed_abs = np.abs(estimated_signed_denom)
        estimated_signed_nonpositive_fraction.append(
            float(np.mean(estimated_signed_denom <= 0.0))
        )
        estimated_signed_condition.append(
            float(
                np.max(estimated_signed_abs)
                / max(float(np.min(estimated_signed_abs)), 1e-12)
            )
        )
        estimated_signed_step = rhs / estimated_signed_denom
        estimated_signed_solve_relative_residual.append(
            float(
                np.linalg.norm(
                    estimated_signed_denom * estimated_signed_step - rhs
                )
                / max(float(np.linalg.norm(rhs)), 1e-12)
            )
        )
        rank_split_correlation.append(_safe_correlation(rank_estimates[0], rank_estimates[1]))
        rank_split_sign_agreement.append(
            float(np.mean(np.sign(rank_estimates[0]) == np.sign(rank_estimates[1])))
        )

    return {
        "dimension": dimension,
        "pairs": pairs,
        "replicates": replicates,
        "alpha": alpha,
        "raw_relative_rmse_mean": float(np.nanmean(raw_relative_rmse)),
        "raw_relative_rmse_std": float(np.nanstd(raw_relative_rmse)),
        "raw_sign_accuracy_mean": float(np.nanmean(raw_sign_accuracy)),
        "raw_clip_fraction_mean": float(np.nanmean(raw_clip_fraction)),
        "rank_split_correlation_mean": float(np.nanmean(rank_split_correlation)),
        "rank_split_sign_agreement_mean": float(np.nanmean(rank_split_sign_agreement)),
        "raw_step_relative_error_mean": float(np.nanmean(raw_step_relative_error)),
        "exact_solve_relative_residual_max": float(
            np.max(exact_solve_relative_residual)
        ),
        "estimated_projected_solve_relative_residual_max": float(
            np.max(estimated_projected_solve_relative_residual)
        ),
        "estimated_signed_nonpositive_fraction_mean": float(
            np.mean(estimated_signed_nonpositive_fraction)
        ),
        "estimated_signed_condition_median": float(
            np.median(estimated_signed_condition)
        ),
        "estimated_signed_solve_relative_residual_max": float(
            np.max(estimated_signed_solve_relative_residual)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimensions", type=int, nargs="+", default=[5, 100, 1000, 5123])
    parser.add_argument("--pairs", type=int, default=80)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--curvature-clip", type=float, default=1000.0)
    parser.add_argument("--alpha", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="", help="Optional destination CSV path.")
    args = parser.parse_args()

    rows = [
        diagnose(
            dimension,
            args.pairs,
            args.replicates,
            args.sigma,
            args.curvature_clip,
            args.alpha,
            args.seed,
        )
        for dimension in args.dimensions
    ]
    stream = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()
    if args.output:
        print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

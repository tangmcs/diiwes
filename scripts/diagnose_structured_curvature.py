#!/usr/bin/env python3
"""Compare diagonal and layer-block matched-rank curvature stability."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.standard_es import centered_ranks


FIELDS = (
    "dimension",
    "pairs",
    "replicates",
    "alpha",
    "diag_split_correlation_mean",
    "diag_split_sign_agreement_mean",
    "block_split_correlation_mean",
    "block_split_sign_agreement_mean",
    "block_split_relative_disagreement_median",
    "signed_diag_min_abs_denominator_median",
    "signed_diag_condition_median",
    "signed_diag_max_amplification_p95",
    "signed_block_min_abs_denominator_median",
    "signed_block_condition_median",
    "signed_block_max_amplification_p95",
    "concave_diag_min_denominator",
    "concave_diag_max_amplification",
    "concave_block_min_denominator",
    "concave_block_max_amplification",
)


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - float(np.mean(left))
    right = right - float(np.mean(right))
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def _relative_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-12)
    )


def _estimate(
    rng: np.random.RandomState,
    hessian: np.ndarray,
    block_slices: Sequence[slice],
    pairs: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = rng.standard_normal((pairs, len(hessian)))
    returns = 0.5 * sigma**2 * np.sum(
        hessian[None, :] * epsilon * epsilon, axis=1
    )
    utilities = centered_ranks(np.concatenate([returns, returns]))
    pair_utility = utilities[:pairs] + utilities[pairs:]
    diagonal = np.mean(
        pair_utility[:, None] * (epsilon * epsilon - 1.0), axis=0
    )
    diagonal /= 2.0 * sigma**2
    blocks = np.asarray(
        [float(np.mean(diagonal[block])) for block in block_slices],
        dtype=np.float64,
    )
    return diagonal, blocks


def diagnose(
    *,
    block_sizes: Sequence[int],
    pairs: int,
    replicates: int,
    sigma: float,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    dimension = int(sum(block_sizes))
    hessian = -np.geomspace(0.25, 4.0, dimension)
    boundaries = np.cumsum([0, *[int(size) for size in block_sizes]])
    block_slices = tuple(
        slice(int(boundaries[index]), int(boundaries[index + 1]))
        for index in range(len(block_sizes))
    )
    rng = np.random.RandomState(seed)

    diag_correlations: list[float] = []
    diag_signs: list[float] = []
    block_correlations: list[float] = []
    block_signs: list[float] = []
    block_disagreements: list[float] = []
    signed_diag_minima: list[float] = []
    signed_diag_conditions: list[float] = []
    signed_diag_amplifications: list[float] = []
    signed_block_minima: list[float] = []
    signed_block_conditions: list[float] = []
    signed_block_amplifications: list[float] = []
    concave_diag_minima: list[float] = []
    concave_diag_amplifications: list[float] = []
    concave_block_minima: list[float] = []
    concave_block_amplifications: list[float] = []

    for _ in range(replicates):
        first_diag, first_block = _estimate(
            rng, hessian, block_slices, pairs, sigma
        )
        second_diag, second_block = _estimate(
            rng, hessian, block_slices, pairs, sigma
        )
        diag_correlations.append(_correlation(first_diag, second_diag))
        diag_signs.append(float(np.mean(np.sign(first_diag) == np.sign(second_diag))))
        block_correlations.append(_correlation(first_block, second_block))
        block_signs.append(
            float(np.mean(np.sign(first_block) == np.sign(second_block)))
        )
        block_disagreements.append(
            _relative_disagreement(first_block, second_block)
        )

        for diagonal, blocks in (
            (first_diag, first_block),
            (second_diag, second_block),
        ):
            signed_diag = 1.0 - alpha * diagonal
            signed_block = 1.0 - alpha * blocks
            signed_diag_abs = np.abs(signed_diag)
            signed_block_abs = np.abs(signed_block)
            signed_diag_minima.append(float(np.min(signed_diag_abs)))
            signed_diag_conditions.append(
                float(np.max(signed_diag_abs) / np.min(signed_diag_abs))
            )
            signed_diag_amplifications.append(float(np.max(1.0 / signed_diag_abs)))
            signed_block_minima.append(float(np.min(signed_block_abs)))
            signed_block_conditions.append(
                float(np.max(signed_block_abs) / np.min(signed_block_abs))
            )
            signed_block_amplifications.append(
                float(np.max(1.0 / signed_block_abs))
            )

            concave_diag = 1.0 + alpha * np.maximum(-diagonal, 0.0)
            concave_block = 1.0 + alpha * np.maximum(-blocks, 0.0)
            concave_diag_minima.append(float(np.min(concave_diag)))
            concave_diag_amplifications.append(float(np.max(1.0 / concave_diag)))
            concave_block_minima.append(float(np.min(concave_block)))
            concave_block_amplifications.append(
                float(np.max(1.0 / concave_block))
            )

    return {
        "dimension": dimension,
        "pairs": int(pairs),
        "replicates": int(replicates),
        "alpha": float(alpha),
        "diag_split_correlation_mean": float(np.mean(diag_correlations)),
        "diag_split_sign_agreement_mean": float(np.mean(diag_signs)),
        "block_split_correlation_mean": float(np.mean(block_correlations)),
        "block_split_sign_agreement_mean": float(np.mean(block_signs)),
        "block_split_relative_disagreement_median": float(
            np.median(block_disagreements)
        ),
        "signed_diag_min_abs_denominator_median": float(
            np.median(signed_diag_minima)
        ),
        "signed_diag_condition_median": float(np.median(signed_diag_conditions)),
        "signed_diag_max_amplification_p95": float(
            np.percentile(signed_diag_amplifications, 95.0)
        ),
        "signed_block_min_abs_denominator_median": float(
            np.median(signed_block_minima)
        ),
        "signed_block_condition_median": float(
            np.median(signed_block_conditions)
        ),
        "signed_block_max_amplification_p95": float(
            np.percentile(signed_block_amplifications, 95.0)
        ),
        "concave_diag_min_denominator": float(np.min(concave_diag_minima)),
        "concave_diag_max_amplification": float(
            np.max(concave_diag_amplifications)
        ),
        "concave_block_min_denominator": float(np.min(concave_block_minima)),
        "concave_block_max_amplification": float(
            np.max(concave_block_amplifications)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[768, 4160, 195])
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--alphas", type=float, nargs="+", default=[10.0, 30.0])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if any(size <= 0 for size in args.block_sizes):
        raise ValueError("block sizes must be positive")
    if args.pairs <= 1 or args.replicates <= 0 or args.sigma <= 0.0:
        raise ValueError("pairs, replicates, and sigma must be positive")

    rows = [
        diagnose(
            block_sizes=args.block_sizes,
            pairs=args.pairs,
            replicates=args.replicates,
            sigma=args.sigma,
            alpha=alpha,
            seed=args.seed,
        )
        for alpha in args.alphas
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

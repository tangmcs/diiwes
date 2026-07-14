#!/usr/bin/env python3
"""Reliability benchmark for the rank surrogate used by implicit ES.

This benchmark is separate from the raw-return curvature benchmark.  It uses
the exact production ``centered_ranks`` transform and compares it with two
alternatives:

* a fixed empirical reference-CDF transform built from an independent sample;
* two-fold cross-fitted ranks, with antithetic pairs kept in the same fold.

All transforms feed the production covariance-score formula.  For the two
alternative transforms, the independent high-sample target is

    E[(U_ref(Y) - E U_ref(Y)) (epsilon epsilon^T - I)] / sigma^2,

where ``U_ref`` is held fixed.  It is called a reference-CDF covariance-score
target here, not the Hessian of the original return objective.
The expectation may subtract the fixed population mean because the Gaussian
score has mean zero; the Monte Carlo target does not recenter its realized
target batch.

For one frozen antithetic batch, the production matrix is also the Jacobian of
the endpoint gradient with respect to its endpoint displacement when batch
utilities are held fixed.  A central finite-difference check verifies that
identity.  Across iid antithetic pairs, dividing that matrix by
``(population - 2) / (population - 1)`` gives an order-two U-statistic that is
unbiased for the Hessian of the current-CDF transformed objective when the CDF
is held fixed.  The benchmark estimates that matching target with independent
pair-of-pairs.  This is not the raw-return Hessian or the total Hessian of an
adaptively reranked objective.

``paired_crn`` gives each plus/minus pair one shared additive observation-noise
draw.  It models coupling, but does not claim to reproduce simulator noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.implicit_es import EndpointImplicitES, LinearizedImplicitES
from core.standard_es import centered_ranks, centered_ranks_from_reference
from experiments.curvature_reliability_benchmark import make_blocks, make_surface


SURFACES = ("diagonal", "block_isotropic", "rotated", "saddle")
TRANSFORMS = (
    "same_batch_centered_rank",
    "independent_reference_cdf",
    "cross_fitted_rank",
)
STRUCTURES = ("diag", "block")
NOISE_COUPLINGS = ("none", "independent", "paired_crn")


@dataclass(frozen=True)
class RankBenchmarkConfig:
    dimensions: tuple[int, ...] = (8, 32)
    populations: tuple[int, ...] = (40, 200)
    sigmas: tuple[float, ...] = (0.02, 0.1)
    surfaces: tuple[str, ...] = SURFACES
    linear_scales: tuple[float, ...] = (0.0, 1.0)
    noise_couplings: tuple[str, ...] = NOISE_COUPLINGS
    observation_noise_std: float = 0.1
    repetitions: int = 100
    num_blocks: int = 4
    reference_size: int = 50_000
    target_size: int = 100_000
    seed: int = 20260712

    def validate(self) -> None:
        if not self.dimensions or any(value < 2 for value in self.dimensions):
            raise ValueError("dimensions must contain integers of at least two")
        if not self.populations or any(
            value < 8 or value % 4 for value in self.populations
        ):
            raise ValueError(
                "populations must contain multiples of four of at least eight"
            )
        if not self.sigmas or any(
            not np.isfinite(value) or value <= 0.0 for value in self.sigmas
        ):
            raise ValueError("sigmas must contain positive finite values")
        if not self.surfaces or any(value not in SURFACES for value in self.surfaces):
            raise ValueError(f"surfaces must be selected from {SURFACES}")
        if not self.linear_scales or any(
            not np.isfinite(value) or value < 0.0 for value in self.linear_scales
        ):
            raise ValueError("linear_scales must contain nonnegative finite values")
        if not self.noise_couplings or any(
            value not in NOISE_COUPLINGS for value in self.noise_couplings
        ):
            raise ValueError(
                f"noise_couplings must be selected from {NOISE_COUPLINGS}"
            )
        if (
            not np.isfinite(self.observation_noise_std)
            or self.observation_noise_std < 0.0
        ):
            raise ValueError("observation_noise_std must be nonnegative and finite")
        if self.repetitions < 2:
            raise ValueError("repetitions must be at least two")
        if self.num_blocks < 2:
            raise ValueError("num_blocks must be at least two")
        if self.reference_size < 100:
            raise ValueError("reference_size must be at least 100")
        if self.target_size < 200 or self.target_size % 2:
            raise ValueError("target_size must be an even integer of at least 200")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")


@dataclass(frozen=True)
class ReferenceTarget:
    sorted_reference: np.ndarray
    diagonal: np.ndarray
    diagonal_standard_error: np.ndarray
    diagonal_split_first: np.ndarray
    diagonal_split_second: np.ndarray
    block: np.ndarray
    block_standard_error: np.ndarray
    block_split_first: np.ndarray
    block_split_second: np.ndarray


@dataclass(frozen=True)
class SameBatchMatchingTarget:
    """Independent estimate of the population target of corrected ranks."""

    diagonal: np.ndarray
    diagonal_standard_error: np.ndarray
    diagonal_split_first: np.ndarray
    diagonal_split_second: np.ndarray
    block: np.ndarray
    block_standard_error: np.ndarray
    block_split_first: np.ndarray
    block_split_second: np.ndarray
    kernel_samples: int


def same_batch_finite_population_factor(pair_count: int) -> float:
    """Return the exact finite-m scale of the same-batch rank Jacobian."""
    pair_count = int(pair_count)
    if pair_count < 2:
        raise ValueError("same-batch curvature requires at least two pairs")
    return float(2 * (pair_count - 1) / (2 * pair_count - 1))


def _tie_comparison(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return 1, 0, or -1 with exact midrank semantics on ties."""
    left = np.asarray(left)
    right = np.asarray(right)
    return (left > right).astype(np.float64) - (left < right).astype(np.float64)


def leave_one_pair_out_rank_utilities(
    fitness: np.ndarray, pair_count: int
) -> np.ndarray:
    """Score each observation against only the other antithetic pairs.

    These utilities use an empirical mid-CDF with ``2 * (m - 1)`` reference
    observations.  They are not recentered on the target batch.  Their
    gradient and covariance-score averages are matched order-two U-statistics
    for the current-CDF stop-gradient population quantities.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    pair_count = int(pair_count)
    if pair_count < 2 or fitness.shape != (2 * pair_count,):
        raise ValueError("LOPO ranks require at least two complete pairs")
    utilities = np.empty(2 * pair_count, dtype=np.float64)
    all_pairs = np.arange(pair_count)
    for pair_index in range(pair_count):
        reference_pairs = all_pairs[all_pairs != pair_index]
        reference_indices = np.concatenate(
            (reference_pairs, reference_pairs + pair_count)
        )
        reference = fitness[reference_indices]
        for sample_index in (pair_index, pair_index + pair_count):
            utilities[sample_index] = float(
                np.sum(_tie_comparison(fitness[sample_index], reference))
                / (4.0 * (pair_count - 1))
            )
    return utilities


def same_batch_pair_u_statistic_matrix(
    eps_half: np.ndarray, fitness: np.ndarray, sigma: float
) -> np.ndarray:
    """Return the unbiased pair U-statistic matching same-batch curvature.

    The identity

        production_matrix = finite_population_factor * returned_matrix

    holds for every complete antithetic batch, including tied returns.
    This direct O(m^2 d^2) implementation is intended for validation, not the
    optimizer hot path.
    """
    eps_half = np.asarray(eps_half, dtype=np.float64)
    fitness = np.asarray(fitness, dtype=np.float64)
    if eps_half.ndim != 2:
        raise ValueError("eps_half must be a matrix")
    pair_count, dimension = eps_half.shape
    if pair_count < 2 or fitness.shape != (2 * pair_count,):
        raise ValueError("fitness must contain two returns for every pair")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    score = np.einsum("mi,mj->mij", eps_half, eps_half, optimize=True)
    score -= np.eye(dimension)[None, :, :]
    total = np.zeros((dimension, dimension), dtype=np.float64)
    for first in range(pair_count):
        first_returns = fitness[[first, first + pair_count]]
        for second in range(first + 1, pair_count):
            second_returns = fitness[[second, second + pair_count]]
            comparison_sum = float(
                np.sum(
                    _tie_comparison(
                        first_returns[:, None], second_returns[None, :]
                    )
                )
            )
            total += comparison_sum * (score[first] - score[second])
    total /= 16.0 * sigma**2
    total /= pair_count * (pair_count - 1) / 2.0
    return 0.5 * (total + total.T)


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def _surface_gradient(
    dimension: int, surface_index: int, seed: int
) -> np.ndarray:
    direction = _rng(seed, 1103, surface_index, dimension).normal(size=dimension)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-15:
        raise FloatingPointError("sampled zero surface-gradient direction")
    return direction / norm


def evaluate_quadratic(
    eps: np.ndarray,
    hessian: np.ndarray,
    gradient_direction: np.ndarray,
    sigma: float,
    linear_scale: float,
) -> np.ndarray:
    eps = np.asarray(eps, dtype=np.float64)
    linear = sigma * linear_scale * (eps @ gradient_direction)
    quadratic = 0.5 * sigma**2 * np.einsum(
        "bi,ij,bj->b", eps, hessian, eps, optimize=True
    )
    return linear + quadratic


def evaluate_antithetic_batch(
    eps_half: np.ndarray,
    hessian: np.ndarray,
    gradient_direction: np.ndarray,
    sigma: float,
    linear_scale: float,
    *,
    noise_coupling: str,
    observation_noise_std: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if noise_coupling not in NOISE_COUPLINGS:
        raise ValueError(f"unknown noise coupling {noise_coupling!r}")
    eps_half = np.asarray(eps_half, dtype=np.float64)
    plus = evaluate_quadratic(
        eps_half, hessian, gradient_direction, sigma, linear_scale
    )
    minus = evaluate_quadratic(
        -eps_half, hessian, gradient_direction, sigma, linear_scale
    )
    if noise_coupling == "none" or observation_noise_std == 0.0:
        return np.concatenate((plus, minus)), np.zeros((2, len(eps_half)))
    if noise_coupling == "independent":
        noise = rng.normal(
            scale=observation_noise_std, size=(2, len(eps_half))
        )
    else:
        shared = rng.normal(scale=observation_noise_std, size=len(eps_half))
        noise = np.vstack((shared, shared))
    return np.concatenate((plus + noise[0], minus + noise[1])), noise


def reference_cdf_utilities(
    values: np.ndarray, sorted_reference: np.ndarray
) -> np.ndarray:
    """Efficiently reproduce centered_ranks_from_reference for sorted input."""
    values = np.asarray(values)
    reference = np.asarray(sorted_reference).ravel()
    if len(reference) < 2 or np.any(reference[1:] < reference[:-1]):
        raise ValueError("sorted_reference must contain at least two sorted values")
    flat = values.ravel()
    left = np.searchsorted(reference, flat, side="left")
    right = np.searchsorted(reference, flat, side="right")
    positions = np.where(right > left, 0.5 * (left + right - 1), left)
    utilities = positions.astype(np.float64) / float(len(reference) - 1) - 0.5
    return np.clip(utilities, -0.5, 0.5).reshape(values.shape)


def cross_fitted_rank_utilities(
    fitness: np.ndarray, pair_count: int
) -> np.ndarray:
    """Two-fold cross-fitting that never uses a pair in its own reference CDF."""
    fitness = np.asarray(fitness, dtype=np.float64)
    if fitness.shape != (2 * pair_count,) or pair_count < 2:
        raise ValueError("cross-fitting requires a complete antithetic population")
    utilities = np.empty_like(fitness, dtype=np.float64)
    pair_indices = np.arange(pair_count)
    folds = pair_indices % 2
    for heldout_fold in (0, 1):
        heldout_pairs = pair_indices[folds == heldout_fold]
        reference_pairs = pair_indices[folds != heldout_fold]
        if len(heldout_pairs) == 0 or len(reference_pairs) == 0:
            raise ValueError("each cross-fit fold must contain at least one pair")
        heldout = np.concatenate((heldout_pairs, heldout_pairs + pair_count))
        reference = np.concatenate(
            (reference_pairs, reference_pairs + pair_count)
        )
        utilities[heldout] = centered_ranks_from_reference(
            fitness[heldout], fitness[reference]
        )
    return utilities - float(np.mean(utilities))


def transform_utilities(
    fitness: np.ndarray,
    transform: str,
    *,
    sorted_reference: np.ndarray,
    pair_count: int,
) -> np.ndarray:
    if transform == "same_batch_centered_rank":
        utilities = centered_ranks(fitness).astype(np.float64, copy=False)
    elif transform == "independent_reference_cdf":
        utilities = reference_cdf_utilities(fitness, sorted_reference)
    elif transform == "cross_fitted_rank":
        return cross_fitted_rank_utilities(fitness, pair_count)
    else:
        raise ValueError(f"unknown transform {transform!r}")
    return utilities - float(np.mean(utilities))


def conditional_covariance_score_matrix(
    noise: np.ndarray, utilities: np.ndarray, sigma: float
) -> np.ndarray:
    """Frozen-utility endpoint-gradient Jacobian for a centered antithetic batch."""
    noise = np.asarray(noise, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    if noise.ndim != 2 or utilities.shape != (len(noise),):
        raise ValueError("noise and utilities have incompatible shapes")
    if not np.allclose(np.mean(noise, axis=0), 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError("conditional formula requires centered antithetic noise")
    centered = utilities - float(np.mean(utilities))
    matrix = np.einsum(
        "b,bi,bj->ij", centered, noise, noise, optimize=True
    ) / (len(noise) * sigma**2)
    # The -I score term is exactly zero because centered utilities sum to zero.
    return 0.5 * (matrix + matrix.T)


def endpoint_gradient_with_frozen_utilities(
    noise: np.ndarray,
    utilities: np.ndarray,
    delta: np.ndarray,
    sigma: float,
) -> np.ndarray:
    noise = np.asarray(noise, dtype=np.float64)
    utilities = np.asarray(utilities, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    logits = noise @ (delta / sigma)
    weights = np.exp(logits - float(np.max(logits)))
    weights /= float(np.sum(weights))
    utility_mean = float(np.dot(weights, utilities))
    centered = utilities - utility_mean
    transformed_noise = noise - delta[None, :] / sigma
    return (weights * centered) @ transformed_noise / sigma


def finite_difference_frozen_jacobian(
    noise: np.ndarray,
    utilities: np.ndarray,
    sigma: float,
    *,
    relative_step: float = 1e-5,
) -> np.ndarray:
    dimension = noise.shape[1]
    step = sigma * relative_step
    jacobian = np.empty((dimension, dimension), dtype=np.float64)
    for coordinate in range(dimension):
        delta = np.zeros(dimension, dtype=np.float64)
        delta[coordinate] = step
        plus = endpoint_gradient_with_frozen_utilities(
            noise, utilities, delta, sigma
        )
        minus = endpoint_gradient_with_frozen_utilities(
            noise, utilities, -delta, sigma
        )
        jacobian[:, coordinate] = (plus - minus) / (2.0 * step)
    return jacobian


def verify_production_conditional_jacobian(seed: int = 20260712) -> dict[str, Any]:
    dimension = 6
    population = 20
    pair_count = population // 2
    sigma = 0.17
    eps_half = _rng(seed, 4001).normal(size=(pair_count, dimension))
    noise = np.concatenate((eps_half, -eps_half), axis=0)
    fitness = _rng(seed, 4003).normal(size=population)

    optimizer = LinearizedImplicitES(
        num_params=dimension,
        population_size=population,
        noise_std=sigma,
        rank_fitness=True,
    )
    utilities, transform = optimizer._utilities(fitness)
    ask_info = {
        "fresh_pair_plus": np.arange(pair_count),
        "fresh_pair_minus": np.arange(pair_count, population),
    }
    production_diagonal, _ = optimizer._matched_diagonal_hessian(
        noise, utilities, ask_info
    )
    analytic = conditional_covariance_score_matrix(noise, utilities, sigma)
    numerical = finite_difference_frozen_jacobian(noise, utilities, sigma)

    endpoint = EndpointImplicitES(
        num_params=dimension,
        population_size=population,
        noise_std=sigma,
        rank_fitness=True,
    )
    core_gradient, _ = endpoint._endpoint_gradient(
        noise, utilities, np.zeros(dimension)
    )
    independent_gradient = endpoint_gradient_with_frozen_utilities(
        noise, utilities, np.zeros(dimension), sigma
    )
    errors = {
        "fitness_transform": transform,
        "conditional_quantity": "frozen_utility_endpoint_gradient_jacobian",
        "objective_hessian": False,
        "production_diagonal_max_abs_error": float(
            np.max(np.abs(production_diagonal - np.diag(analytic)))
        ),
        "finite_difference_max_abs_error": float(
            np.max(np.abs(numerical - analytic))
        ),
        "finite_difference_relative_frobenius_error": float(
            np.linalg.norm(numerical - analytic)
            / max(float(np.linalg.norm(analytic)), 1e-15)
        ),
        "endpoint_gradient_max_abs_error": float(
            np.max(np.abs(core_gradient - independent_gradient))
        ),
        "finite_difference_relative_step": relative_step_value(),
    }
    if (
        errors["production_diagonal_max_abs_error"] > 1e-10
        or errors["finite_difference_relative_frobenius_error"] > 1e-8
        or errors["endpoint_gradient_max_abs_error"] > 1e-10
    ):
        raise AssertionError(f"conditional Jacobian verification failed: {errors}")
    return errors


def relative_step_value() -> float:
    return 1e-5


def pair_covariance_score_contributions(
    eps_half: np.ndarray, utilities: np.ndarray, sigma: float
) -> np.ndarray:
    pair_count, dimension = eps_half.shape
    if utilities.shape != (2 * pair_count,):
        raise ValueError("utilities do not match the antithetic pairs")
    pair_utility = utilities[:pair_count] + utilities[pair_count:]
    return pair_utility[:, None] * (eps_half**2 - 1.0) / (2.0 * sigma**2)


def pool_components(
    diagonal_values: np.ndarray, blocks: Sequence[slice]
) -> np.ndarray:
    diagonal_values = np.asarray(diagonal_values, dtype=np.float64)
    return np.column_stack(
        [np.mean(diagonal_values[:, block], axis=1) for block in blocks]
    )


def split_covariance_score_estimates(
    fitness: np.ndarray,
    eps_half: np.ndarray,
    sigma: float,
    transform: str,
    *,
    sorted_reference: np.ndarray,
    blocks: Sequence[slice],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], str, str]:
    """Recompute each split transform instead of slicing full-batch utilities."""
    fitness = np.asarray(fitness, dtype=np.float64)
    eps_half = np.asarray(eps_half, dtype=np.float64)
    pair_count = len(eps_half)
    if fitness.shape != (2 * pair_count,) or pair_count % 2:
        raise ValueError("split estimates require an even number of complete pairs")

    if transform == "cross_fitted_rank":
        pair_groups = (
            np.arange(pair_count)[np.arange(pair_count) % 2 == 0],
            np.arange(pair_count)[np.arange(pair_count) % 2 == 1],
        )
        semantics = "dependent_reciprocal_cross_fit_fold_agreement"
        independence = "dependent_through_reciprocal_fold_references"
    else:
        split = pair_count // 2
        pair_groups = (np.arange(split), np.arange(split, pair_count))
        if transform == "same_batch_centered_rank":
            semantics = "independent_centered_ranks_per_disjoint_pair_half"
            independence = "independent_disjoint_pair_halves"
        elif transform == "independent_reference_cdf":
            semantics = "fixed_reference_cdf_centered_per_disjoint_pair_half"
            independence = "conditional_on_shared_fixed_reference_cdf"
        else:
            raise ValueError(f"unknown transform {transform!r}")

    diagonal_estimates: list[np.ndarray] = []
    block_estimates: list[np.ndarray] = []
    all_pairs = np.arange(pair_count)
    for pair_indices in pair_groups:
        sample_indices = np.concatenate(
            (pair_indices, pair_indices + pair_count)
        )
        half_fitness = fitness[sample_indices]
        if transform == "same_batch_centered_rank":
            half_utilities = centered_ranks(half_fitness)
        elif transform == "independent_reference_cdf":
            half_utilities = reference_cdf_utilities(
                half_fitness, sorted_reference
            )
        else:
            reference_pairs = np.setdiff1d(
                all_pairs, pair_indices, assume_unique=True
            )
            reference_indices = np.concatenate(
                (reference_pairs, reference_pairs + pair_count)
            )
            half_utilities = centered_ranks_from_reference(
                half_fitness, fitness[reference_indices]
            )
        half_utilities = half_utilities - float(np.mean(half_utilities))
        contributions = pair_covariance_score_contributions(
            eps_half[pair_indices], half_utilities, sigma
        )
        diagonal_estimates.append(np.mean(contributions, axis=0))
        block_estimates.append(
            np.mean(pool_components(contributions, blocks), axis=0)
        )
    return (
        {
            "diag": (diagonal_estimates[0], diagonal_estimates[1]),
            "block": (block_estimates[0], block_estimates[1]),
        },
        semantics,
        independence,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if (
        len(left) < 2
        or float(np.std(left)) <= 1e-15
        or float(np.std(right)) <= 1e-15
    ):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _relative_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-15)
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _reference_target(
    *,
    reference_eps: np.ndarray,
    target_eps: np.ndarray,
    reference_base: np.ndarray,
    target_base: np.ndarray,
    sigma: float,
    observation_noise_std: float,
    reference_noise: np.ndarray,
    target_noise: np.ndarray,
    blocks: Sequence[slice],
) -> ReferenceTarget:
    reference_returns = reference_base + observation_noise_std * reference_noise
    sorted_reference = np.sort(reference_returns, kind="mergesort")
    target_returns = target_base + observation_noise_std * target_noise
    # Keep the fixed transform unchanged.  Re-centering this target batch would
    # instead define a finite-n conditional covariance/Jacobian target.
    utilities = reference_cdf_utilities(target_returns, sorted_reference)
    contributions = utilities[:, None] * (target_eps**2 - 1.0) / sigma**2
    diagonal = np.mean(contributions, axis=0)
    standard_error = np.std(contributions, axis=0, ddof=1) / np.sqrt(
        len(contributions)
    )
    block_contributions = pool_components(contributions, blocks)
    block = np.mean(block_contributions, axis=0)
    block_standard_error = np.std(
        block_contributions, axis=0, ddof=1
    ) / np.sqrt(len(block_contributions))
    split = len(contributions) // 2
    return ReferenceTarget(
        sorted_reference=sorted_reference,
        diagonal=diagonal,
        diagonal_standard_error=standard_error,
        diagonal_split_first=np.mean(contributions[:split], axis=0),
        diagonal_split_second=np.mean(contributions[split:], axis=0),
        block=block,
        block_standard_error=block_standard_error,
        block_split_first=np.mean(block_contributions[:split], axis=0),
        block_split_second=np.mean(block_contributions[split:], axis=0),
    )


def _same_batch_matching_target(
    *,
    eps_half: np.ndarray,
    plus_returns: np.ndarray,
    minus_returns: np.ndarray,
    sigma: float,
    blocks: Sequence[slice],
) -> SameBatchMatchingTarget:
    """Estimate the corrected-rank target with independent pair-of-pairs.

    Adjacent antithetic pairs form one independent order-two kernel draw.  The
    kernel uses only pairwise return comparisons, so no empirical reference
    CDF or current-target centering enters this population-target estimate.
    """
    eps_half = np.asarray(eps_half, dtype=np.float64)
    plus_returns = np.asarray(plus_returns, dtype=np.float64)
    minus_returns = np.asarray(minus_returns, dtype=np.float64)
    pair_count, dimension = eps_half.shape
    if pair_count < 4 or pair_count % 2:
        raise ValueError("matching target requires an even number of at least four pairs")
    if plus_returns.shape != (pair_count,) or minus_returns.shape != (pair_count,):
        raise ValueError("matching target returns do not match eps_half")

    first = np.arange(0, pair_count, 2)
    second = first + 1
    comparison_sum = (
        _tie_comparison(plus_returns[first], plus_returns[second])
        + _tie_comparison(plus_returns[first], minus_returns[second])
        + _tie_comparison(minus_returns[first], plus_returns[second])
        + _tie_comparison(minus_returns[first], minus_returns[second])
    )
    # The identity matrices cancel in S(first) - S(second).
    score_difference = eps_half[first] ** 2 - eps_half[second] ** 2
    contributions = (
        comparison_sum[:, None] * score_difference / (16.0 * sigma**2)
    )
    if not np.all(np.isfinite(contributions)):
        raise FloatingPointError("same-batch matching target is non-finite")
    diagonal = np.mean(contributions, axis=0)
    diagonal_se = np.std(contributions, axis=0, ddof=1) / np.sqrt(
        len(contributions)
    )
    block_contributions = pool_components(contributions, blocks)
    block = np.mean(block_contributions, axis=0)
    block_se = np.std(block_contributions, axis=0, ddof=1) / np.sqrt(
        len(block_contributions)
    )
    split = len(contributions) // 2
    if split == 0 or split == len(contributions):
        raise ValueError("matching target requires at least two kernel samples")
    return SameBatchMatchingTarget(
        diagonal=diagonal,
        diagonal_standard_error=diagonal_se,
        diagonal_split_first=np.mean(contributions[:split], axis=0),
        diagonal_split_second=np.mean(contributions[split:], axis=0),
        block=block,
        block_standard_error=block_se,
        block_split_first=np.mean(block_contributions[:split], axis=0),
        block_split_second=np.mean(block_contributions[split:], axis=0),
        kernel_samples=len(contributions),
    )


def _summarize_estimates(
    estimates: Sequence[np.ndarray],
    split_first: Sequence[np.ndarray],
    split_second: Sequence[np.ndarray],
    target: np.ndarray,
    target_se: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(estimates, dtype=np.float64)
    first = np.asarray(split_first, dtype=np.float64)
    second = np.asarray(split_second, dtype=np.float64)
    repeated_target = np.broadcast_to(target, values.shape)
    error = values - repeated_target
    mean_component_error = np.mean(values, axis=0) - target

    predicted_negative = values < 0.0
    true_negative = repeated_target < 0.0
    true_positive = int(np.sum(predicted_negative & true_negative))
    false_positive = int(np.sum(predicted_negative & ~true_negative))
    false_negative = int(np.sum(~predicted_negative & true_negative))
    true_negative_count = int(np.sum(~predicted_negative & ~true_negative))

    reliable_component = np.abs(target) > 1.96 * target_se
    reliable_mask = np.broadcast_to(reliable_component, values.shape)
    reliable_true = true_negative & reliable_mask
    reliable_predicted = predicted_negative & reliable_mask
    reliable_tp = int(np.sum(reliable_predicted & reliable_true))
    reliable_fp = int(np.sum(reliable_predicted & ~true_negative & reliable_mask))
    reliable_fn = int(np.sum(~predicted_negative & reliable_true))

    split_correlations = [
        _safe_correlation(left, right) for left, right in zip(first, second, strict=True)
    ]
    finite_correlations = [value for value in split_correlations if value is not None]
    split_sign = np.mean(np.sign(first) == np.sign(second), axis=1)
    split_disagreement = np.asarray(
        [
            _relative_disagreement(left, right)
            for left, right in zip(first, second, strict=True)
        ]
    )
    target_rms = float(np.linalg.norm(target) / np.sqrt(len(target)))
    rmse = float(np.sqrt(np.mean(error**2)))
    component_bias_rmse = float(np.sqrt(np.mean(mean_component_error**2)))
    return {
        "bias": float(np.mean(error)),
        "component_bias_rmse": component_bias_rmse,
        "relative_component_bias_rmse": float(
            component_bias_rmse / max(target_rms, 1e-15)
        ),
        "rmse": rmse,
        "relative_rmse": float(rmse / max(target_rms, 1e-15)),
        "correlation": _safe_correlation(values, repeated_target),
        "mean_estimate_norm": float(np.mean(np.linalg.norm(values, axis=1))),
        "target_norm": float(np.linalg.norm(target)),
        "target_rms": target_rms,
        "target_mean_standard_error": float(np.mean(target_se)),
        "target_max_standard_error": float(np.max(target_se)),
        "target_components_reliable_1_96se": int(np.sum(reliable_component)),
        "sign_precision": _ratio(true_positive, true_positive + false_positive),
        "sign_recall": _ratio(true_positive, true_positive + false_negative),
        "sign_accuracy": float(
            (true_positive + true_negative_count) / predicted_negative.size
        ),
        "reliable_sign_precision": _ratio(
            reliable_tp, reliable_tp + reliable_fp
        ),
        "reliable_sign_recall": _ratio(
            reliable_tp, reliable_tp + reliable_fn
        ),
        "split_correlation_mean": (
            float(np.mean(finite_correlations)) if finite_correlations else None
        ),
        "split_correlation_median": (
            float(np.median(finite_correlations)) if finite_correlations else None
        ),
        "split_sign_agreement_mean": float(np.mean(split_sign)),
        "split_sign_agreement_median": float(np.median(split_sign)),
        "split_relative_disagreement_mean": float(np.mean(split_disagreement)),
        "split_relative_disagreement_median": float(
            np.median(split_disagreement)
        ),
    }


def run_benchmark(config: RankBenchmarkConfig) -> list[dict[str, Any]]:
    config.validate()
    rows: list[dict[str, Any]] = []
    surface_indices = {name: index for index, name in enumerate(SURFACES)}

    for dimension in config.dimensions:
        blocks = make_blocks(dimension, config.num_blocks)
        for surface_name in config.surfaces:
            surface_index = surface_indices[surface_name]
            surface = make_surface(
                surface_name,
                dimension,
                blocks,
                seed=config.seed + 1013 * surface_index,
            )
            gradient_direction = _surface_gradient(
                dimension, surface_index, config.seed
            )
            reference_eps = _rng(
                config.seed, 5003, surface_index, dimension
            ).normal(size=(config.reference_size, dimension))
            target_eps = _rng(
                config.seed, 5009, surface_index, dimension
            ).normal(size=(config.target_size, dimension))
            reference_quadratic = 0.5 * np.einsum(
                "bi,ij,bj->b",
                reference_eps,
                surface.hessian,
                reference_eps,
                optimize=True,
            )
            target_quadratic = 0.5 * np.einsum(
                "bi,ij,bj->b",
                target_eps,
                surface.hessian,
                target_eps,
                optimize=True,
            )
            reference_linear = reference_eps @ gradient_direction
            target_linear = target_eps @ gradient_direction
            reference_noise = _rng(
                config.seed, 5011, surface_index, dimension
            ).normal(size=config.reference_size)
            target_noise = _rng(
                config.seed, 5021, surface_index, dimension
            ).normal(size=config.target_size)
            target_noise_minus = _rng(
                config.seed, 5023, surface_index, dimension
            ).normal(size=config.target_size)

            for linear_scale in config.linear_scales:
                for sigma in config.sigmas:
                    reference_base = (
                        sigma * linear_scale * reference_linear
                        + sigma**2 * reference_quadratic
                    )
                    target_base = (
                        sigma * linear_scale * target_linear
                        + sigma**2 * target_quadratic
                    )
                    target_minus_base = (
                        -sigma * linear_scale * target_linear
                        + sigma**2 * target_quadratic
                    )
                    targets = {
                        False: _reference_target(
                            reference_eps=reference_eps,
                            target_eps=target_eps,
                            reference_base=reference_base,
                            target_base=target_base,
                            sigma=sigma,
                            observation_noise_std=0.0,
                            reference_noise=reference_noise,
                            target_noise=target_noise,
                            blocks=blocks,
                        ),
                        True: _reference_target(
                            reference_eps=reference_eps,
                            target_eps=target_eps,
                            reference_base=reference_base,
                            target_base=target_base,
                            sigma=sigma,
                            observation_noise_std=config.observation_noise_std,
                            reference_noise=reference_noise,
                            target_noise=target_noise,
                            blocks=blocks,
                        ),
                    }
                    matching_targets = {
                        False: _same_batch_matching_target(
                            eps_half=target_eps,
                            plus_returns=target_base,
                            minus_returns=target_minus_base,
                            sigma=sigma,
                            blocks=blocks,
                        ),
                        True: _same_batch_matching_target(
                            eps_half=target_eps,
                            plus_returns=(
                                target_base
                                + config.observation_noise_std * target_noise
                            ),
                            minus_returns=(
                                target_minus_base
                                + config.observation_noise_std * target_noise_minus
                            ),
                            sigma=sigma,
                            blocks=blocks,
                        ),
                    }

                    for population in config.populations:
                        pair_count = population // 2
                        for coupling in config.noise_couplings:
                            noisy = coupling != "none"
                            target = targets[noisy]
                            matching_target = matching_targets[noisy]
                            rank_factor = same_batch_finite_population_factor(
                                pair_count
                            )
                            split_rank_factor = same_batch_finite_population_factor(
                                pair_count // 2
                            )
                            estimates: dict[
                                tuple[str, str], list[np.ndarray]
                            ] = {
                                (transform, structure): []
                                for transform in TRANSFORMS
                                for structure in STRUCTURES
                            }
                            split_first: dict[
                                tuple[str, str], list[np.ndarray]
                            ] = {key: [] for key in estimates}
                            split_second: dict[
                                tuple[str, str], list[np.ndarray]
                            ] = {key: [] for key in estimates}
                            split_semantics: dict[str, str] = {}
                            split_independence: dict[str, str] = {}

                            for repetition in range(config.repetitions):
                                eps_half = _rng(
                                    config.seed,
                                    6007,
                                    surface_index,
                                    dimension,
                                    population,
                                    repetition,
                                ).normal(size=(pair_count, dimension))
                                fitness, _ = evaluate_antithetic_batch(
                                    eps_half,
                                    surface.hessian,
                                    gradient_direction,
                                    sigma,
                                    linear_scale,
                                    noise_coupling=coupling,
                                    observation_noise_std=(
                                        config.observation_noise_std if noisy else 0.0
                                    ),
                                    rng=_rng(
                                        config.seed,
                                        6011,
                                        surface_index,
                                        dimension,
                                        population,
                                        repetition,
                                    ),
                                )
                                for transform in TRANSFORMS:
                                    utilities = transform_utilities(
                                        fitness,
                                        transform,
                                        sorted_reference=target.sorted_reference,
                                        pair_count=pair_count,
                                    )
                                    diagonal_contributions = (
                                        pair_covariance_score_contributions(
                                            eps_half, utilities, sigma
                                        )
                                    )
                                    component_contributions = {
                                        "diag": diagonal_contributions,
                                        "block": pool_components(
                                            diagonal_contributions, blocks
                                        ),
                                    }
                                    split_estimates, semantics, independence = (
                                        split_covariance_score_estimates(
                                            fitness,
                                            eps_half,
                                            sigma,
                                            transform,
                                            sorted_reference=target.sorted_reference,
                                            blocks=blocks,
                                        )
                                    )
                                    split_semantics[transform] = semantics
                                    split_independence[transform] = independence
                                    for structure in STRUCTURES:
                                        key = (transform, structure)
                                        contributions = component_contributions[
                                            structure
                                        ]
                                        estimate = np.mean(contributions, axis=0)
                                        first_split = split_estimates[structure][0]
                                        second_split = split_estimates[structure][1]
                                        if transform == "same_batch_centered_rank":
                                            # J_current / c_m is exactly the
                                            # leave-one-pair-out rank U-statistic.
                                            estimate = estimate / rank_factor
                                            first_split = (
                                                first_split / split_rank_factor
                                            )
                                            second_split = (
                                                second_split / split_rank_factor
                                            )
                                        estimates[key].append(estimate)
                                        split_first[key].append(first_split)
                                        split_second[key].append(second_split)

                            for transform in TRANSFORMS:
                                for structure in STRUCTURES:
                                    same_batch = (
                                        transform == "same_batch_centered_rank"
                                    )
                                    reported_target = (
                                        matching_target if same_batch else target
                                    )
                                    if structure == "diag":
                                        component_target = reported_target.diagonal
                                        component_target_se = (
                                            reported_target.diagonal_standard_error
                                        )
                                        target_split_first = (
                                            reported_target.diagonal_split_first
                                        )
                                        target_split_second = (
                                            reported_target.diagonal_split_second
                                        )
                                    else:
                                        component_target = reported_target.block
                                        component_target_se = (
                                            reported_target.block_standard_error
                                        )
                                        target_split_first = (
                                            reported_target.block_split_first
                                        )
                                        target_split_second = (
                                            reported_target.block_split_second
                                        )
                                    target_split_correlation = _safe_correlation(
                                        target_split_first, target_split_second
                                    )
                                    target_split_sign = float(
                                        np.mean(
                                            np.sign(target_split_first)
                                            == np.sign(target_split_second)
                                        )
                                    )
                                    key = (transform, structure)
                                    row: dict[str, Any] = {
                                        "surface": surface_name,
                                        "dimension": dimension,
                                        "population": population,
                                        "antithetic_pairs": pair_count,
                                        "sigma": sigma,
                                        "linear_scale": linear_scale,
                                        "noise_coupling": coupling,
                                        "observation_noise_std": (
                                            config.observation_noise_std
                                            if noisy
                                            else 0.0
                                        ),
                                        "transform": transform,
                                        "structure": structure,
                                        "method": f"{transform}_{structure}",
                                        "components": (
                                            dimension
                                            if structure == "diag"
                                            else len(blocks)
                                        ),
                                        "repetitions": config.repetitions,
                                        "num_blocks": len(blocks),
                                        "reference_size": config.reference_size,
                                        "target_size": config.target_size,
                                        "target_kernel_samples": (
                                            matching_target.kernel_samples
                                            if same_batch
                                            else 0
                                        ),
                                        "estimand": (
                                            "current_return_mid_cdf_stop_gradient_"
                                            "population_hessian"
                                            if same_batch
                                            else
                                            "fixed_independent_reference_cdf_"
                                            "population_covariance_score"
                                        ),
                                        "target_current_batch_centered": False,
                                        "estimate_current_batch_centered": True,
                                        "finite_population_rank_factor": (
                                            rank_factor if same_batch else 1.0
                                        ),
                                        "reported_estimate_divided_by_rank_factor": (
                                            same_batch
                                        ),
                                        "finite_n_conditional_jacobian_equals_"
                                        "fixed_transform_population_target": False,
                                        "conditionally_unbiased_for_fixed_"
                                        "transform_population_target": False,
                                        "unconditionally_unbiased_for_reported_"
                                        "population_target": same_batch,
                                        "finite_n_estimator_semantics": (
                                            "bias_corrected_same_batch_rank_"
                                            "leave_one_pair_out_u_statistic"
                                            if same_batch
                                            else
                                            "batch_centered_reference_cdf_"
                                            "conditional_jacobian"
                                            if transform
                                            == "independent_reference_cdf"
                                            else
                                            "cross_fitted_batch_defined_"
                                            "conditional_jacobian"
                                        ),
                                        "estimate_interpretation": (
                                            "unbiased_leave_one_pair_out_rank_"
                                            "population_curvature_u_statistic"
                                            if same_batch
                                            else
                                            "conditional_frozen_utility_endpoint_"
                                            "gradient_jacobian"
                                        ),
                                        "raw_production_interpretation": (
                                            "conditional_frozen_utility_endpoint_"
                                            "gradient_jacobian"
                                        ),
                                        "original_return_objective_hessian": False,
                                        "global_adaptive_rank_objective_hessian": False,
                                        "fixed_transform_smoothed_objective_"
                                        "hessian_target": True,
                                        "same_batch_production_semantics": same_batch,
                                        "matching_pair_of_pairs_target": same_batch,
                                        "reported_curvature_equals_lopo_"
                                        "u_statistic": same_batch,
                                        "production_gradient_equals_lopo_"
                                        "u_statistic": False,
                                        "rescaling_curvature_alone_makes_"
                                        "implicit_system_population_matched": False,
                                        "principled_population_matched_update_"
                                        "requires_lopo_gradient_and_curvature": (
                                            same_batch
                                        ),
                                        "cross_fit_pair_preserving": (
                                            transform == "cross_fitted_rank"
                                        ),
                                        "split_semantics": split_semantics[
                                            transform
                                        ],
                                        "split_independence": split_independence[
                                            transform
                                        ],
                                        "target_split_correlation": (
                                            target_split_correlation
                                        ),
                                        "target_split_sign_agreement": (
                                            target_split_sign
                                        ),
                                    }
                                    row.update(
                                        _summarize_estimates(
                                            estimates[key],
                                            split_first[key],
                                            split_second[key],
                                            component_target,
                                            component_target_se,
                                        )
                                    )
                                    rows.append(row)
    return rows


def _atomic_write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_csv(path: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty rank benchmark")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_outputs(
    output: str,
    rows: Sequence[dict[str, Any]],
    config: RankBenchmarkConfig,
    verification: dict[str, Any],
    *,
    metadata_output: str | None = None,
) -> tuple[str, str]:
    metadata = metadata_output or f"{os.path.splitext(output)[0]}.json"
    _atomic_write_csv(output, rows)
    _atomic_write_json(
        metadata,
        {
            "schema_version": 2,
            "benchmark": "rank_surrogate_covariance_score_reliability",
            "config": asdict(config),
            "scope": {
                "rl_environment_evaluated": False,
                "original_return_objective_hessian_evaluated": False,
                "production_same_batch_rank_semantics_evaluated": True,
                "conditional_quantity": (
                    "frozen-utility endpoint-gradient Jacobian"
                ),
                "same_batch_population_target": (
                    "current-return mid-CDF stop-gradient Hessian estimated "
                    "by independent pair-of-pairs U-statistic kernels"
                ),
                "other_transform_population_target": (
                    "uncentered fixed-transform covariance score from an "
                    "independent high-sample reference CDF"
                ),
                "population_target_current_batch_centered": False,
                "reported_estimates_current_batch_centered": True,
                "finite_n_target_distinction": (
                    "raw same-batch curvature is exactly its conditional "
                    "Jacobian and is c_m times the LOPO U-statistic; reported "
                    "same-batch curvature divides by c_m. Other transforms "
                    "retain their separate fixed-reference discrepancy semantics"
                ),
                "gradient_boundary": (
                    "the pooled-rank production gradient retains a within-pair "
                    "comparison term; rescaling curvature alone does not make "
                    "the finite-m implicit system population matched"
                ),
                "population_matched_future_estimator": (
                    "leave-one-pair-out rank utilities for both gradient and "
                    "curvature; not used by the production optimizer"
                ),
                "cross_fit_definition": (
                    "two folds preserving complete antithetic pairs"
                ),
                "noise_model": (
                    "controlled additive observation noise; paired_crn shares "
                    "one draw within each antithetic pair"
                ),
            },
            "production_conditional_jacobian_verification": verification,
            "rows": list(rows),
        },
    )
    return output, metadata


def _parse_csv(value: str, cast: Any, name: str) -> tuple[Any, ...]:
    try:
        result = tuple(cast(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {error}") from error
    if not result:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/rank_surrogate_reliability_benchmark.csv",
    )
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--dimensions", default="8,32")
    parser.add_argument("--populations", default="40,200")
    parser.add_argument("--sigmas", default="0.02,0.1")
    parser.add_argument("--surfaces", default=",".join(SURFACES))
    parser.add_argument("--linear-scales", default="0,1")
    parser.add_argument(
        "--noise-couplings", default=",".join(NOISE_COUPLINGS)
    )
    parser.add_argument("--observation-noise-std", type=float, default=0.1)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--reference-size", type=int, default=50_000)
    parser.add_argument("--target-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = RankBenchmarkConfig(
        dimensions=_parse_csv(args.dimensions, int, "dimensions"),
        populations=_parse_csv(args.populations, int, "populations"),
        sigmas=_parse_csv(args.sigmas, float, "sigmas"),
        surfaces=_parse_csv(args.surfaces, str, "surfaces"),
        linear_scales=_parse_csv(args.linear_scales, float, "linear scales"),
        noise_couplings=_parse_csv(
            args.noise_couplings, str, "noise couplings"
        ),
        observation_noise_std=args.observation_noise_std,
        repetitions=args.repetitions,
        num_blocks=args.num_blocks,
        reference_size=args.reference_size,
        target_size=args.target_size,
        seed=args.seed,
    )
    config.validate()
    verification = verify_production_conditional_jacobian(config.seed)
    rows = run_benchmark(config)
    output, metadata = write_outputs(
        args.output,
        rows,
        config,
        verification,
        metadata_output=args.metadata_output,
    )
    print(
        f"Wrote {len(rows)} rank-surrogate cells to {output}; metadata={metadata}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Controlled trajectory benchmark for implicit ES curvature surrogates.

This benchmark is intentionally separate from the RL trainer.  It uses
quadratic objectives with known full Hessians and compares exact-gradient
sanity dynamics with Monte Carlo ES dynamics under common random numbers.

The raw-fitness Monte Carlo regime has a literal Gaussian-smoothed Hessian
target.  The centered-rank regime does not: its sampled statistic is labeled a
frozen-rank covariance-score surrogate.  Synthetic results from this script
are mechanism diagnostics, not evidence of general RL performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BENCHMARK_VERSION = "1.0.0"
SCHEMA_VERSION = 1
MAX_FINITE_DIAGNOSTIC = 1.0e300

CASE_NAMES = (
    "block_aligned_concave",
    "rotated_concave",
    "rotated_indefinite",
    "block_aligned_additive_noise",
)
EXACT_CASE_NAMES = (
    "block_aligned_concave",
    "rotated_concave",
    "rotated_indefinite",
)
FITNESS_TRANSFORMS = ("raw", "same_batch_centered_rank")

EXACT_METHODS = (
    "explicit_exact_gradient",
    "oracle_full_implicit",
    "oracle_diagonal_approximation_signed",
    "oracle_block_approximation_signed",
    "concave_projected_block",
    "norm_matched_isotropic",
)
RAW_MC_METHODS = (
    "explicit_es",
    "oracle_full_implicit",
    "sampled_signed_diagonal",
    "sampled_signed_block",
    "concave_projected_block",
    "norm_matched_isotropic",
)
RANK_MC_METHODS = (
    "explicit_es",
    "frozen_batch_full_signed",
    "sampled_signed_diagonal",
    "sampled_signed_block",
    "concave_projected_block",
    "norm_matched_isotropic",
)

TRAJECTORY_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "method",
    "alpha",
    "seed",
    "iteration",
    "objective",
    "objective_gap",
    "parameter_norm",
    "gradient_norm",
    "active_update",
    "carried_forward",
    "diverged",
    "divergence_reason",
    "step_norm",
    "candidate_norm_before_cap",
    "explicit_step_norm",
    "step_amplification",
    "inverse_operator_norm",
    "denominator_min_abs",
    "denominator_nonpositive_fraction",
    "solve_success",
    "norm_match_relative_error",
    "curvature_diag_rmse",
    "curvature_block_rmse",
    "true_step_improvement",
    "structured_counterfactual_improvement",
    "isotropic_counterfactual_improvement",
    "structured_directional_benefit",
    "structured_isotropic_cosine",
    "observation_noise_std",
    "curvature_target_kind",
)

SUMMARY_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "method",
    "alpha",
    "seed",
    "has_finite_maximum",
    "diverged",
    "divergence_iteration",
    "divergence_reason",
    "active_updates",
    "initial_objective",
    "final_objective",
    "best_objective",
    "mean_trapezoidal_objective_auc",
    "initial_gap",
    "final_gap",
    "final_gap_ratio",
    "normalized_gap_auc",
    "converged_gap_1e_3",
    "max_parameter_norm",
    "max_step_amplification",
    "min_denominator_margin",
    "mean_nonpositive_denominator_fraction",
    "mean_curvature_diag_rmse",
    "mean_curvature_block_rmse",
    "mean_structured_directional_benefit",
    "structured_directional_win_fraction",
    "max_norm_match_relative_error",
)

DIRECTIONAL_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "alpha",
    "seed",
    "iteration",
    "reference_state_method",
    "structured_improvement",
    "isotropic_improvement",
    "structured_directional_benefit",
    "structured_step_norm",
    "isotropic_step_norm",
    "norm_match_relative_error",
    "structured_isotropic_cosine",
)

AGGREGATE_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "method",
    "alpha",
    "n_runs",
    "n_diverged",
    "n_nondiverged",
    "divergence_rate",
    "boundary_inclusive_mean_final_objective",
    "boundary_inclusive_mean_objective_auc",
    "boundary_inclusive_mean_final_gap_ratio",
    "boundary_inclusive_mean_normalized_gap_auc",
    "nondiverged_mean_final_objective",
    "nondiverged_mean_objective_auc",
    "nondiverged_mean_final_gap_ratio",
    "nondiverged_mean_normalized_gap_auc",
    "mean_max_step_amplification",
    "minimum_denominator_margin",
    "mean_structured_directional_benefit",
    "mean_structured_directional_win_fraction",
)

DIRECTIONAL_AGGREGATE_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "alpha",
    "reference_state_method",
    "reference_runs",
    "reference_run_divergence_rate",
    "expected_reference_steps",
    "observed_reference_steps",
    "reference_horizon_completion_fraction",
    "mean_structured_improvement",
    "mean_isotropic_improvement",
    "mean_structured_directional_benefit",
    "median_structured_directional_benefit",
    "structured_directional_win_fraction",
    "max_norm_match_relative_error",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    dimension: int = 12
    num_blocks: int = 3
    population_size: int = 96
    iterations: int = 30
    sigma: float = 0.1
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    mc_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    cases: tuple[str, ...] = CASE_NAMES
    fitness_transforms: tuple[str, ...] = FITNESS_TRANSFORMS
    additive_noise_std: float = 0.05
    master_seed: int = 20260712
    divergence_threshold: float = 1.0e6
    objective_gap_divergence_ratio: float = 1.0e6
    singular_tolerance: float = 1.0e-10

    def validate(self) -> None:
        if self.dimension < 4:
            raise ValueError("dimension must be at least four")
        if self.num_blocks < 2 or self.dimension % self.num_blocks:
            raise ValueError("num_blocks must divide dimension and be at least two")
        if self.population_size < 8 or self.population_size % 2:
            raise ValueError("population_size must be an even integer of at least eight")
        if self.iterations < 2:
            raise ValueError("iterations must be at least two")
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be positive and finite")
        if not self.alphas or any(
            not np.isfinite(value) or value <= 0.0 for value in self.alphas
        ):
            raise ValueError("alphas must contain positive finite values")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must be unique")
        if not self.mc_seeds or any(value < 0 for value in self.mc_seeds):
            raise ValueError("mc_seeds must contain nonnegative integers")
        if len(set(self.mc_seeds)) != len(self.mc_seeds):
            raise ValueError("mc_seeds must be unique")
        if not self.cases or any(value not in CASE_NAMES for value in self.cases):
            raise ValueError(f"cases must be selected from {CASE_NAMES}")
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("cases must be unique")
        if not self.fitness_transforms or any(
            value not in FITNESS_TRANSFORMS for value in self.fitness_transforms
        ):
            raise ValueError(
                f"fitness_transforms must be selected from {FITNESS_TRANSFORMS}"
            )
        if len(set(self.fitness_transforms)) != len(self.fitness_transforms):
            raise ValueError("fitness_transforms must be unique")
        if not np.isfinite(self.additive_noise_std) or self.additive_noise_std < 0.0:
            raise ValueError("additive_noise_std must be nonnegative and finite")
        if self.master_seed < 0:
            raise ValueError("master_seed must be nonnegative")
        if (
            not np.isfinite(self.divergence_threshold)
            or self.divergence_threshold <= 1.0
        ):
            raise ValueError("divergence_threshold must be finite and greater than one")
        if (
            not np.isfinite(self.objective_gap_divergence_ratio)
            or self.objective_gap_divergence_ratio <= 1.0
        ):
            raise ValueError(
                "objective_gap_divergence_ratio must be finite and greater than one"
            )
        if (
            not np.isfinite(self.singular_tolerance)
            or self.singular_tolerance <= 0.0
        ):
            raise ValueError("singular_tolerance must be positive and finite")


@dataclass(frozen=True)
class QuadraticCase:
    name: str
    hessian: np.ndarray
    blocks: tuple[slice, ...]
    initial_params: np.ndarray
    observation_noise_std: float
    has_finite_maximum: bool
    interpretation: str

    @property
    def dimension(self) -> int:
        return int(self.hessian.shape[0])

    @property
    def diagonal(self) -> np.ndarray:
        return np.diag(self.hessian).copy()

    @property
    def block_targets(self) -> np.ndarray:
        diagonal = self.diagonal
        return np.asarray(
            [float(np.mean(diagonal[block])) for block in self.blocks],
            dtype=np.float64,
        )

    def objective(self, params: np.ndarray) -> float:
        params = np.asarray(params, dtype=np.float64)
        return float(0.5 * params @ self.hessian @ params)

    def gradient(self, params: np.ndarray) -> np.ndarray:
        return self.hessian @ np.asarray(params, dtype=np.float64)

    def gap(self, params: np.ndarray) -> float | None:
        if not self.has_finite_maximum:
            return None
        value = -self.objective(params)
        return float(max(value, 0.0))


@dataclass(frozen=True)
class Estimates:
    gradient: np.ndarray
    diagonal: np.ndarray
    block: np.ndarray
    full_linearization: np.ndarray | None
    diagonal_rmse: float | None
    block_rmse: float | None
    target_kind: str


@dataclass(frozen=True)
class Proposal:
    step: np.ndarray
    denominator_min_abs: float
    denominator_nonpositive_fraction: float
    inverse_operator_norm: float
    step_amplification: float
    solve_success: bool
    failure_reason: str
    norm_match_relative_error: float


@dataclass(frozen=True)
class DirectionalComparison:
    structured_step: np.ndarray
    isotropic_step: np.ndarray
    structured_improvement: float
    isotropic_improvement: float
    benefit: float
    norm_match_relative_error: float
    cosine: float


@dataclass
class MethodState:
    params: np.ndarray
    diverged: bool = False
    divergence_iteration: int | None = None
    divergence_reason: str = ""


@dataclass(frozen=True)
class BenchmarkResult:
    trajectories: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]
    directional: tuple[dict[str, Any], ...]
    aggregates: tuple[dict[str, Any], ...]
    directional_aggregates: tuple[dict[str, Any], ...]
    case_metadata: tuple[dict[str, Any], ...]
    validation: dict[str, Any]


def make_blocks(dimension: int, num_blocks: int) -> tuple[slice, ...]:
    if dimension < 1 or num_blocks < 1 or dimension % num_blocks:
        raise ValueError("num_blocks must evenly divide dimension")
    width = dimension // num_blocks
    return tuple(
        slice(index * width, (index + 1) * width)
        for index in range(num_blocks)
    )


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def _orthogonal_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def _stable_norm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    scale = float(np.max(np.abs(vector))) if vector.size else 0.0
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("inf")
    scaled = float(np.linalg.norm(vector / scale))
    value = scale * scaled
    return float(value) if np.isfinite(value) else float("inf")


def _bounded_nonnegative(value: float) -> float:
    if not np.isfinite(value):
        return MAX_FINITE_DIAGNOSTIC
    return float(min(max(value, 0.0), MAX_FINITE_DIAGNOSTIC))


def _reciprocal_margin(margin: float) -> float:
    if margin <= 1.0 / MAX_FINITE_DIAGNOSTIC:
        return MAX_FINITE_DIAGNOSTIC
    return _bounded_nonnegative(1.0 / margin)


def _block_expand(values: np.ndarray, blocks: Sequence[slice], dimension: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(blocks),):
        raise ValueError("block values have the wrong shape")
    expanded = np.empty(dimension, dtype=np.float64)
    for value, block in zip(values, blocks, strict=True):
        expanded[block] = value
    return expanded


def make_cases(config: BenchmarkConfig) -> tuple[QuadraticCase, ...]:
    config.validate()
    blocks = make_blocks(config.dimension, config.num_blocks)
    block_values = -np.geomspace(0.1, 2.0, config.num_blocks)
    aligned_diagonal = _block_expand(block_values, blocks, config.dimension)
    aligned_hessian = np.diag(aligned_diagonal)

    aligned_rng = _rng(config.master_seed, 1001)
    aligned_initial = aligned_rng.normal(size=config.dimension)
    aligned_initial *= 2.0 / _stable_norm(aligned_initial)

    rotated_rng = _rng(config.master_seed, 1002, config.dimension)
    rotated_q = _orthogonal_matrix(config.dimension, rotated_rng)
    rotated_hessian = rotated_q @ np.diag(aligned_diagonal) @ rotated_q.T
    rotated_initial = rotated_rng.normal(size=config.dimension)
    rotated_initial *= 2.0 / _stable_norm(rotated_initial)

    negative_count = (2 * config.dimension) // 3
    positive_count = config.dimension - negative_count
    indefinite_eigenvalues = np.concatenate(
        (
            -np.geomspace(0.1, 2.0, negative_count),
            np.geomspace(0.07, 0.6, positive_count),
        )
    )
    indefinite_rng = _rng(config.master_seed, 1003, config.dimension)
    indefinite_q = _orthogonal_matrix(config.dimension, indefinite_rng)
    indefinite_hessian = (
        indefinite_q @ np.diag(indefinite_eigenvalues) @ indefinite_q.T
    )
    indefinite_initial = indefinite_rng.normal(size=config.dimension)
    indefinite_initial *= 2.0 / _stable_norm(indefinite_initial)

    all_cases = {
        "block_aligned_concave": QuadraticCase(
            name="block_aligned_concave",
            hessian=aligned_hessian,
            blocks=blocks,
            initial_params=aligned_initial,
            observation_noise_std=0.0,
            has_finite_maximum=True,
            interpretation="correctly_specified_block_isotropic_concave",
        ),
        "rotated_concave": QuadraticCase(
            name="rotated_concave",
            hessian=0.5 * (rotated_hessian + rotated_hessian.T),
            blocks=blocks,
            initial_params=rotated_initial,
            observation_noise_std=0.0,
            has_finite_maximum=True,
            interpretation="dense_rotated_concave_block_misspecified",
        ),
        "rotated_indefinite": QuadraticCase(
            name="rotated_indefinite",
            hessian=0.5 * (indefinite_hessian + indefinite_hessian.T),
            blocks=blocks,
            initial_params=indefinite_initial,
            observation_noise_std=0.0,
            has_finite_maximum=False,
            interpretation="dense_rotated_saddle_stress_only_no_finite_maximum",
        ),
        "block_aligned_additive_noise": QuadraticCase(
            name="block_aligned_additive_noise",
            hessian=aligned_hessian.copy(),
            blocks=blocks,
            initial_params=aligned_initial.copy(),
            observation_noise_std=float(config.additive_noise_std),
            has_finite_maximum=True,
            interpretation="correctly_specified_block_isotropic_with_independent_additive_noise",
        ),
    }
    return tuple(all_cases[name] for name in config.cases)


def _block_approximation(case: QuadraticCase) -> np.ndarray:
    approximation = np.zeros_like(case.hessian)
    for value, block in zip(case.block_targets, case.blocks, strict=True):
        approximation[block, block] = value * np.eye(block.stop - block.start)
    return approximation


def case_metadata(case: QuadraticCase) -> dict[str, Any]:
    eigenvalues = np.linalg.eigvalsh(case.hessian)
    hessian_norm = float(np.linalg.norm(case.hessian))
    diagonal_residual = float(
        np.linalg.norm(case.hessian - np.diag(case.diagonal))
        / max(hessian_norm, 1e-15)
    )
    block_residual = float(
        np.linalg.norm(case.hessian - _block_approximation(case))
        / max(hessian_norm, 1e-15)
    )
    return {
        "name": case.name,
        "dimension": case.dimension,
        "block_bounds": [[int(block.start), int(block.stop)] for block in case.blocks],
        "eigenvalues": [float(value) for value in eigenvalues],
        "diagonal": [float(value) for value in case.diagonal],
        "block_targets": [float(value) for value in case.block_targets],
        "has_finite_maximum": bool(case.has_finite_maximum),
        "observation_noise_std": float(case.observation_noise_std),
        "interpretation": case.interpretation,
        "off_diagonal_relative_frobenius_norm": diagonal_residual,
        "block_misspecification_relative_frobenius_norm": block_residual,
        "initial_parameter_norm": _stable_norm(case.initial_params),
        "initial_objective": case.objective(case.initial_params),
        "hessian_sha256": hashlib.sha256(
            np.ascontiguousarray(case.hessian, dtype="<f8").tobytes()
        ).hexdigest(),
    }


def centered_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-aware centered ranks in [-0.5, 0.5]."""
    values = np.asarray(values)
    flat = values.ravel()
    if len(flat) == 0:
        return flat.astype(np.float64).reshape(values.shape)
    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(len(flat), dtype=np.float64)
    start = 0
    while start < len(flat):
        end = start + 1
        while end < len(flat) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    if len(flat) > 1:
        ranks = ranks / float(len(flat) - 1) - 0.5
    else:
        ranks.fill(0.0)
    return ranks.reshape(values.shape)


def evaluate_antithetic(
    case: QuadraticCase,
    params: np.ndarray,
    eps: np.ndarray,
    sigma: float,
    noise_plus: np.ndarray,
    noise_minus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.float64)
    gradient = case.gradient(params)
    center = case.objective(params)
    linear = sigma * (eps @ gradient)
    quadratic = 0.5 * sigma**2 * np.einsum(
        "bi,ij,bj->b", eps, case.hessian, eps, optimize=True
    )
    plus = center + linear + quadratic
    minus = center - linear + quadratic
    if case.observation_noise_std > 0.0:
        plus = plus + case.observation_noise_std * noise_plus
        minus = minus + case.observation_noise_std * noise_minus
    return plus.astype(np.float64), minus.astype(np.float64)


def estimate_mc(
    case: QuadraticCase,
    params: np.ndarray,
    eps: np.ndarray,
    plus: np.ndarray,
    minus: np.ndarray,
    sigma: float,
    fitness_transform: str,
) -> Estimates:
    pair_count, dimension = eps.shape
    if pair_count < 2 or dimension != case.dimension:
        raise ValueError("Monte Carlo estimate has incompatible dimensions")
    if fitness_transform == "raw":
        gradient = np.mean(
            (plus - minus)[:, None] * eps, axis=0
        ) / (2.0 * sigma)
        pair_sum = plus + minus
        baseline = (float(np.sum(pair_sum)) - pair_sum) / float(pair_count - 1)
        pair_signal = pair_sum - baseline
        diagonal_contributions = (
            pair_signal[:, None] * (eps**2 - 1.0) / (2.0 * sigma**2)
        )
        block_features = np.column_stack(
            [
                np.mean(eps[:, block] ** 2, axis=1) - 1.0
                for block in case.blocks
            ]
        )
        block_contributions = (
            pair_signal[:, None] * block_features / (2.0 * sigma**2)
        )
        diagonal = np.mean(diagonal_contributions, axis=0)
        block = np.mean(block_contributions, axis=0)
        return Estimates(
            gradient=gradient,
            diagonal=diagonal,
            block=block,
            full_linearization=None,
            diagonal_rmse=float(np.sqrt(np.mean((diagonal - case.diagonal) ** 2))),
            block_rmse=float(np.sqrt(np.mean((block - case.block_targets) ** 2))),
            target_kind="raw_gaussian_smoothed_hessian",
        )
    if fitness_transform != "same_batch_centered_rank":
        raise ValueError(f"unknown fitness transform {fitness_transform!r}")

    noise = np.concatenate((eps, -eps), axis=0)
    utilities = centered_ranks(np.concatenate((plus, minus)))
    utilities = utilities - float(np.mean(utilities))
    gradient = np.mean(utilities[:, None] * noise, axis=0) / sigma
    pair_utility = utilities[:pair_count] + utilities[pair_count:]
    diagonal = np.mean(
        pair_utility[:, None] * (eps**2 - 1.0) / (2.0 * sigma**2),
        axis=0,
    )
    block_features = np.column_stack(
        [
            np.mean(eps[:, block] ** 2, axis=1) - 1.0
            for block in case.blocks
        ]
    )
    block = np.mean(
        pair_utility[:, None] * block_features / (2.0 * sigma**2),
        axis=0,
    )
    full = (
        noise.T @ (utilities[:, None] * noise)
    ) / (len(noise) * sigma**2)
    full = 0.5 * (full + full.T)
    if not np.allclose(np.diag(full), diagonal, rtol=1e-11, atol=1e-11):
        raise FloatingPointError("rank full Jacobian and diagonal statistic disagree")
    return Estimates(
        gradient=gradient,
        diagonal=diagonal,
        block=block,
        full_linearization=full,
        diagonal_rmse=None,
        block_rmse=None,
        target_kind="frozen_same_batch_rank_covariance_score_no_literal_hessian_target",
    )


def exact_estimates(case: QuadraticCase, params: np.ndarray) -> Estimates:
    return Estimates(
        gradient=case.gradient(params),
        diagonal=case.diagonal,
        block=case.block_targets,
        full_linearization=case.hessian,
        diagonal_rmse=0.0,
        block_rmse=0.0,
        target_kind="known_quadratic_hessian",
    )


def _amplification(step: np.ndarray, explicit: np.ndarray) -> float:
    explicit_norm = _stable_norm(explicit)
    step_norm = _stable_norm(step)
    if explicit_norm <= 1e-15:
        return 0.0 if step_norm <= 1e-15 else MAX_FINITE_DIAGNOSTIC
    return _bounded_nonnegative(step_norm / explicit_norm)


def _full_proposal(
    gradient: np.ndarray,
    matrix: np.ndarray,
    alpha: float,
    singular_tolerance: float,
) -> Proposal:
    system = np.eye(len(gradient), dtype=np.float64) - alpha * matrix
    eigenvalues = np.linalg.eigvalsh(0.5 * (system + system.T))
    margin = float(np.min(np.abs(eigenvalues)))
    nonpositive = float(np.mean(eigenvalues <= 0.0))
    if margin < singular_tolerance:
        return Proposal(
            step=np.zeros_like(gradient),
            denominator_min_abs=margin,
            denominator_nonpositive_fraction=nonpositive,
            inverse_operator_norm=_reciprocal_margin(margin),
            step_amplification=MAX_FINITE_DIAGNOSTIC,
            solve_success=False,
            failure_reason="near_singular_full_system",
            norm_match_relative_error=0.0,
        )
    rhs = alpha * gradient
    try:
        step = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return Proposal(
            step=np.zeros_like(gradient),
            denominator_min_abs=margin,
            denominator_nonpositive_fraction=nonpositive,
            inverse_operator_norm=_reciprocal_margin(margin),
            step_amplification=MAX_FINITE_DIAGNOSTIC,
            solve_success=False,
            failure_reason="full_system_solve_failure",
            norm_match_relative_error=0.0,
        )
    success = bool(np.all(np.isfinite(step)))
    return Proposal(
        step=step if success else np.zeros_like(gradient),
        denominator_min_abs=margin,
        denominator_nonpositive_fraction=nonpositive,
        inverse_operator_norm=_reciprocal_margin(margin),
        step_amplification=(
            _amplification(step, rhs) if success else MAX_FINITE_DIAGNOSTIC
        ),
        solve_success=success,
        failure_reason="" if success else "nonfinite_full_system_step",
        norm_match_relative_error=0.0,
    )


def _diagonal_proposal(
    gradient: np.ndarray,
    diagonal_curvature: np.ndarray,
    alpha: float,
    singular_tolerance: float,
) -> Proposal:
    denominator = 1.0 - alpha * np.asarray(diagonal_curvature, dtype=np.float64)
    margin = float(np.min(np.abs(denominator)))
    nonpositive = float(np.mean(denominator <= 0.0))
    if margin < singular_tolerance:
        return Proposal(
            step=np.zeros_like(gradient),
            denominator_min_abs=margin,
            denominator_nonpositive_fraction=nonpositive,
            inverse_operator_norm=_reciprocal_margin(margin),
            step_amplification=MAX_FINITE_DIAGNOSTIC,
            solve_success=False,
            failure_reason="near_singular_diagonal_system",
            norm_match_relative_error=0.0,
        )
    rhs = alpha * gradient
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        step = rhs / denominator
    success = bool(np.all(np.isfinite(step)))
    return Proposal(
        step=step if success else np.zeros_like(gradient),
        denominator_min_abs=margin,
        denominator_nonpositive_fraction=nonpositive,
        inverse_operator_norm=_reciprocal_margin(margin),
        step_amplification=(
            _amplification(step, rhs) if success else MAX_FINITE_DIAGNOSTIC
        ),
        solve_success=success,
        failure_reason="" if success else "nonfinite_diagonal_step",
        norm_match_relative_error=0.0,
    )


def _structured_steps(
    gradient: np.ndarray,
    block_curvature: np.ndarray,
    blocks: Sequence[slice],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    concave = np.maximum(-np.asarray(block_curvature, dtype=np.float64), 0.0)
    block_denominator = 1.0 + alpha * concave
    denominator = _block_expand(block_denominator, blocks, len(gradient))
    explicit = alpha * gradient
    structured = explicit / denominator
    explicit_norm = _stable_norm(explicit)
    structured_norm = _stable_norm(structured)
    scale = 1.0 if explicit_norm <= 1e-15 else structured_norm / explicit_norm
    scale = float(min(max(scale, 0.0), 1.0))
    isotropic = scale * explicit
    norm_error = abs(_stable_norm(isotropic) - structured_norm) / max(
        structured_norm, 1e-15
    )
    return structured, isotropic, block_denominator, float(norm_error)


def propose_step(
    method: str,
    estimates: Estimates,
    case: QuadraticCase,
    alpha: float,
    singular_tolerance: float,
) -> Proposal:
    gradient = estimates.gradient
    explicit = alpha * gradient
    if method in {"explicit_exact_gradient", "explicit_es"}:
        return Proposal(
            step=explicit,
            denominator_min_abs=1.0,
            denominator_nonpositive_fraction=0.0,
            inverse_operator_norm=1.0,
            step_amplification=1.0 if _stable_norm(explicit) > 1e-15 else 0.0,
            solve_success=True,
            failure_reason="",
            norm_match_relative_error=0.0,
        )
    if method == "oracle_full_implicit":
        return _full_proposal(
            gradient, case.hessian, alpha, singular_tolerance
        )
    if method == "frozen_batch_full_signed":
        if estimates.full_linearization is None:
            raise ValueError("frozen-batch full method requires a full linearization")
        return _full_proposal(
            gradient, estimates.full_linearization, alpha, singular_tolerance
        )
    if method in {
        "oracle_diagonal_approximation_signed",
        "sampled_signed_diagonal",
    }:
        return _diagonal_proposal(
            gradient, estimates.diagonal, alpha, singular_tolerance
        )
    if method in {
        "oracle_block_approximation_signed",
        "sampled_signed_block",
    }:
        expanded = _block_expand(estimates.block, case.blocks, case.dimension)
        return _diagonal_proposal(
            gradient, expanded, alpha, singular_tolerance
        )
    if method in {"concave_projected_block", "norm_matched_isotropic"}:
        structured, isotropic, block_denominator, norm_error = _structured_steps(
            gradient, estimates.block, case.blocks, alpha
        )
        step = structured if method == "concave_projected_block" else isotropic
        margin = float(np.min(block_denominator))
        scale = _amplification(step, explicit)
        inverse_norm = scale if method == "norm_matched_isotropic" else float(
            1.0 / margin
        )
        return Proposal(
            step=step,
            denominator_min_abs=margin if method == "concave_projected_block" else (
                _reciprocal_margin(scale)
            ),
            denominator_nonpositive_fraction=0.0,
            inverse_operator_norm=_bounded_nonnegative(inverse_norm),
            step_amplification=scale,
            solve_success=bool(np.all(np.isfinite(step))),
            failure_reason="" if np.all(np.isfinite(step)) else "nonfinite_safe_step",
            norm_match_relative_error=norm_error,
        )
    raise ValueError(f"unknown method {method!r}")


def directional_comparison(
    case: QuadraticCase,
    params: np.ndarray,
    estimates: Estimates,
    alpha: float,
) -> DirectionalComparison:
    structured, isotropic, _, norm_error = _structured_steps(
        estimates.gradient, estimates.block, case.blocks, alpha
    )
    current = case.objective(params)
    structured_improvement = case.objective(params + structured) - current
    isotropic_improvement = case.objective(params + isotropic) - current
    structured_norm = _stable_norm(structured)
    isotropic_norm = _stable_norm(isotropic)
    if structured_norm <= 1e-15 and isotropic_norm <= 1e-15:
        cosine = 1.0
    elif structured_norm <= 1e-15 or isotropic_norm <= 1e-15:
        cosine = 0.0
    else:
        cosine = float(
            np.dot(structured, isotropic) / (structured_norm * isotropic_norm)
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))
    return DirectionalComparison(
        structured_step=structured,
        isotropic_step=isotropic,
        structured_improvement=float(structured_improvement),
        isotropic_improvement=float(isotropic_improvement),
        benefit=float(structured_improvement - isotropic_improvement),
        norm_match_relative_error=float(norm_error),
        cosine=cosine,
    )


def _cap_or_diverge(
    state: MethodState,
    proposal: Proposal,
    estimates: Estimates,
    case: QuadraticCase,
    iteration: int,
    divergence_threshold: float,
    objective_gap_divergence_ratio: float,
) -> tuple[np.ndarray, float, str]:
    candidate = state.params + proposal.step
    candidate_norm = _stable_norm(candidate)
    reason = proposal.failure_reason
    if not proposal.solve_success or not np.all(np.isfinite(candidate)):
        reason = reason or "nonfinite_candidate"
    elif candidate_norm > divergence_threshold:
        reason = "parameter_norm_threshold"
    else:
        value = case.objective(candidate)
        if np.isfinite(value):
            candidate_gap = case.gap(candidate)
            initial_gap = case.gap(case.initial_params)
            if (
                candidate_gap is not None
                and initial_gap is not None
                and candidate_gap
                > objective_gap_divergence_ratio * max(initial_gap, 1e-15)
            ):
                reason = "objective_gap_ratio_threshold"
            else:
                return candidate, candidate_norm, ""
        else:
            reason = "nonfinite_objective"

    if reason == "objective_gap_ratio_threshold":
        candidate_gap = case.gap(candidate)
        initial_gap = case.gap(case.initial_params)
        if candidate_gap is None or initial_gap is None or candidate_gap <= 0.0:
            raise RuntimeError("objective-gap divergence requires a positive finite gap")
        target_gap = objective_gap_divergence_ratio * max(initial_gap, 1e-15)
        capped = candidate * math.sqrt(target_gap / candidate_gap)
        state.diverged = True
        state.divergence_iteration = iteration
        state.divergence_reason = reason
        return capped, _bounded_nonnegative(candidate_norm), reason

    direction = candidate if np.all(np.isfinite(candidate)) else estimates.gradient
    direction_norm = _stable_norm(direction)
    if not np.isfinite(direction_norm) or direction_norm <= 1e-15:
        direction = state.params
        direction_norm = _stable_norm(direction)
    if not np.isfinite(direction_norm) or direction_norm <= 1e-15:
        direction = np.zeros(case.dimension, dtype=np.float64)
        direction[0] = 1.0
        direction_norm = 1.0
    capped = direction * (divergence_threshold / direction_norm)
    state.diverged = True
    state.divergence_iteration = iteration
    state.divergence_reason = reason
    return capped, _bounded_nonnegative(candidate_norm), reason


def _initial_row(
    regime: str,
    transform: str,
    case: QuadraticCase,
    method: str,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "regime": regime,
        "fitness_transform": transform,
        "case": case.name,
        "method": method,
        "alpha": float(alpha),
        "seed": int(seed),
        "iteration": 0,
        "objective": case.objective(case.initial_params),
        "objective_gap": case.gap(case.initial_params),
        "parameter_norm": _stable_norm(case.initial_params),
        "gradient_norm": _stable_norm(case.gradient(case.initial_params)),
        "active_update": False,
        "carried_forward": False,
        "diverged": False,
        "divergence_reason": "",
        "step_norm": None,
        "candidate_norm_before_cap": None,
        "explicit_step_norm": None,
        "step_amplification": None,
        "inverse_operator_norm": None,
        "denominator_min_abs": None,
        "denominator_nonpositive_fraction": None,
        "solve_success": None,
        "norm_match_relative_error": None,
        "curvature_diag_rmse": None,
        "curvature_block_rmse": None,
        "true_step_improvement": None,
        "structured_counterfactual_improvement": None,
        "isotropic_counterfactual_improvement": None,
        "structured_directional_benefit": None,
        "structured_isotropic_cosine": None,
        "observation_noise_std": float(case.observation_noise_std),
        "curvature_target_kind": "not_evaluated_at_initial_state",
    }


def _carried_row(
    previous: dict[str, Any],
    iteration: int,
    state: MethodState,
) -> dict[str, Any]:
    row = dict(previous)
    row.update(
        {
            "iteration": int(iteration),
            "active_update": False,
            "carried_forward": True,
            "diverged": True,
            "divergence_reason": state.divergence_reason,
            "step_norm": None,
            "candidate_norm_before_cap": None,
            "explicit_step_norm": None,
            "step_amplification": None,
            "inverse_operator_norm": None,
            "denominator_min_abs": None,
            "denominator_nonpositive_fraction": None,
            "solve_success": None,
            "norm_match_relative_error": None,
            "curvature_diag_rmse": None,
            "curvature_block_rmse": None,
            "true_step_improvement": None,
            "structured_counterfactual_improvement": None,
            "isotropic_counterfactual_improvement": None,
            "structured_directional_benefit": None,
            "structured_isotropic_cosine": None,
        }
    )
    return row


def _sample_iteration(
    config: BenchmarkConfig,
    case_index: int,
    alpha_index: int,
    seed: int,
    iteration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = _rng(
        config.master_seed,
        2001,
        case_index,
        alpha_index,
        seed,
        iteration,
    )
    pair_count = config.population_size // 2
    eps = rng.normal(size=(pair_count, config.dimension))
    noise_plus = rng.normal(size=pair_count)
    noise_minus = rng.normal(size=pair_count)
    return eps, noise_plus, noise_minus


def _run_group(
    config: BenchmarkConfig,
    case: QuadraticCase,
    case_index: int,
    alpha: float,
    alpha_index: int,
    regime: str,
    transform: str,
    seed: int,
    methods: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    states = {
        method: MethodState(params=case.initial_params.copy()) for method in methods
    }
    method_rows = {
        method: [_initial_row(regime, transform, case, method, alpha, seed)]
        for method in methods
    }
    directional_rows: list[dict[str, Any]] = []
    reference_method = methods[0]

    for iteration in range(1, config.iterations + 1):
        if regime == "monte_carlo_es":
            eps, noise_plus, noise_minus = _sample_iteration(
                config, case_index, alpha_index, seed, iteration
            )
        else:
            eps = noise_plus = noise_minus = None

        for method in methods:
            state = states[method]
            if state.diverged:
                method_rows[method].append(
                    _carried_row(method_rows[method][-1], iteration, state)
                )
                continue

            if regime == "exact_gradient_sanity":
                estimates = exact_estimates(case, state.params)
            else:
                if eps is None or noise_plus is None or noise_minus is None:
                    raise RuntimeError("Monte Carlo samples were not generated")
                plus, minus = evaluate_antithetic(
                    case,
                    state.params,
                    eps,
                    config.sigma,
                    noise_plus,
                    noise_minus,
                )
                estimates = estimate_mc(
                    case,
                    state.params,
                    eps,
                    plus,
                    minus,
                    config.sigma,
                    transform,
                )

            comparison = directional_comparison(
                case, state.params, estimates, alpha
            )
            if method == reference_method:
                directional_rows.append(
                    {
                        "regime": regime,
                        "fitness_transform": transform,
                        "case": case.name,
                        "alpha": float(alpha),
                        "seed": int(seed),
                        "iteration": int(iteration),
                        "reference_state_method": reference_method,
                        "structured_improvement": comparison.structured_improvement,
                        "isotropic_improvement": comparison.isotropic_improvement,
                        "structured_directional_benefit": comparison.benefit,
                        "structured_step_norm": _stable_norm(comparison.structured_step),
                        "isotropic_step_norm": _stable_norm(comparison.isotropic_step),
                        "norm_match_relative_error": comparison.norm_match_relative_error,
                        "structured_isotropic_cosine": comparison.cosine,
                    }
                )

            proposal = propose_step(
                method,
                estimates,
                case,
                alpha,
                config.singular_tolerance,
            )
            before_objective = case.objective(state.params)
            new_params, candidate_norm, divergence_reason = _cap_or_diverge(
                state,
                proposal,
                estimates,
                case,
                iteration,
                config.divergence_threshold,
                config.objective_gap_divergence_ratio,
            )
            state.params = new_params
            after_objective = case.objective(new_params)
            explicit_step = alpha * estimates.gradient
            row = {
                "regime": regime,
                "fitness_transform": transform,
                "case": case.name,
                "method": method,
                "alpha": float(alpha),
                "seed": int(seed),
                "iteration": int(iteration),
                "objective": after_objective,
                "objective_gap": case.gap(new_params),
                "parameter_norm": _stable_norm(new_params),
                "gradient_norm": _stable_norm(estimates.gradient),
                "active_update": True,
                "carried_forward": False,
                "diverged": bool(state.diverged),
                "divergence_reason": divergence_reason,
                "step_norm": _bounded_nonnegative(_stable_norm(proposal.step)),
                "candidate_norm_before_cap": _bounded_nonnegative(candidate_norm),
                "explicit_step_norm": _bounded_nonnegative(_stable_norm(explicit_step)),
                "step_amplification": proposal.step_amplification,
                "inverse_operator_norm": proposal.inverse_operator_norm,
                "denominator_min_abs": proposal.denominator_min_abs,
                "denominator_nonpositive_fraction": (
                    proposal.denominator_nonpositive_fraction
                ),
                "solve_success": proposal.solve_success,
                "norm_match_relative_error": proposal.norm_match_relative_error,
                "curvature_diag_rmse": estimates.diagonal_rmse,
                "curvature_block_rmse": estimates.block_rmse,
                "true_step_improvement": after_objective - before_objective,
                "structured_counterfactual_improvement": (
                    comparison.structured_improvement
                ),
                "isotropic_counterfactual_improvement": (
                    comparison.isotropic_improvement
                ),
                "structured_directional_benefit": comparison.benefit,
                "structured_isotropic_cosine": comparison.cosine,
                "observation_noise_std": float(case.observation_noise_std),
                "curvature_target_kind": estimates.target_kind,
            }
            method_rows[method].append(row)

    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for method in methods:
        rows = method_rows[method]
        trajectories.extend(rows)
        summaries.append(_summarize_rows(rows, case, config.iterations))
    return trajectories, summaries, directional_rows


def _mean_trapezoid(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2:
        raise ValueError("trapezoidal AUC needs at least two points")
    return float(
        (0.5 * array[0] + np.sum(array[1:-1]) + 0.5 * array[-1])
        / (len(array) - 1)
    )


def _nullable_mean(values: Iterable[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return None if not available else float(np.mean(available))


def _summarize_rows(
    rows: Sequence[dict[str, Any]],
    case: QuadraticCase,
    iterations: int,
) -> dict[str, Any]:
    if len(rows) != iterations + 1:
        raise ValueError("trajectory does not have the configured horizon")
    objectives = [float(row["objective"]) for row in rows]
    gaps = [row["objective_gap"] for row in rows]
    active = [row for row in rows[1:] if row["active_update"]]
    step_amplifications = [
        float(row["step_amplification"])
        for row in active
        if row["step_amplification"] is not None
        and np.isfinite(row["step_amplification"])
    ]
    margins = [
        float(row["denominator_min_abs"])
        for row in active
        if row["denominator_min_abs"] is not None
        and np.isfinite(row["denominator_min_abs"])
    ]
    nonpositive = [
        float(row["denominator_nonpositive_fraction"])
        for row in active
        if row["denominator_nonpositive_fraction"] is not None
    ]
    benefits = [
        float(row["structured_directional_benefit"])
        for row in active
        if row["structured_directional_benefit"] is not None
    ]
    norm_errors = [
        float(row["norm_match_relative_error"])
        for row in active
        if row["norm_match_relative_error"] is not None
    ]
    initial_gap = gaps[0]
    final_gap = gaps[-1]
    if case.has_finite_maximum:
        if initial_gap is None or final_gap is None:
            raise RuntimeError("finite-maximum case is missing a gap")
        initial_gap_float = float(initial_gap)
        final_gap_float = float(final_gap)
        final_gap_ratio = final_gap_float / max(initial_gap_float, 1e-15)
        gap_auc = _mean_trapezoid([float(value) for value in gaps])
        normalized_gap_auc = gap_auc / max(initial_gap_float, 1e-15)
        converged = bool(final_gap_ratio <= 1e-3)
    else:
        initial_gap_float = final_gap_float = final_gap_ratio = None
        normalized_gap_auc = None
        converged = False
    divergence_rows = [row for row in rows if row["diverged"]]
    first_divergence = divergence_rows[0] if divergence_rows else None
    first = rows[0]
    return {
        "regime": first["regime"],
        "fitness_transform": first["fitness_transform"],
        "case": first["case"],
        "method": first["method"],
        "alpha": first["alpha"],
        "seed": first["seed"],
        "has_finite_maximum": bool(case.has_finite_maximum),
        "diverged": bool(first_divergence is not None),
        "divergence_iteration": (
            None if first_divergence is None else int(first_divergence["iteration"])
        ),
        "divergence_reason": (
            "" if first_divergence is None else first_divergence["divergence_reason"]
        ),
        "active_updates": int(len(active)),
        "initial_objective": objectives[0],
        "final_objective": objectives[-1],
        "best_objective": float(max(objectives)),
        "mean_trapezoidal_objective_auc": _mean_trapezoid(objectives),
        "initial_gap": initial_gap_float,
        "final_gap": final_gap_float,
        "final_gap_ratio": final_gap_ratio,
        "normalized_gap_auc": normalized_gap_auc,
        "converged_gap_1e_3": converged,
        "max_parameter_norm": float(max(float(row["parameter_norm"]) for row in rows)),
        "max_step_amplification": (
            None if not step_amplifications else float(max(step_amplifications))
        ),
        "min_denominator_margin": (
            None if not margins else float(min(margins))
        ),
        "mean_nonpositive_denominator_fraction": (
            None if not nonpositive else float(np.mean(nonpositive))
        ),
        "mean_curvature_diag_rmse": _nullable_mean(
            row["curvature_diag_rmse"] for row in active
        ),
        "mean_curvature_block_rmse": _nullable_mean(
            row["curvature_block_rmse"] for row in active
        ),
        "mean_structured_directional_benefit": (
            None if not benefits else float(np.mean(benefits))
        ),
        "structured_directional_win_fraction": (
            None if not benefits else float(np.mean(np.asarray(benefits) > 0.0))
        ),
        "max_norm_match_relative_error": (
            None if not norm_errors else float(max(norm_errors))
        ),
    }


def aggregate_summaries(
    summaries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in summaries:
        key = (
            row["regime"],
            row["fitness_transform"],
            row["case"],
            row["method"],
            float(row["alpha"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        rows = groups[key]
        nondiverged = [row for row in rows if not row["diverged"]]
        margins = [
            float(row["min_denominator_margin"])
            for row in rows
            if row["min_denominator_margin"] is not None
        ]
        output.append(
            {
                "regime": key[0],
                "fitness_transform": key[1],
                "case": key[2],
                "method": key[3],
                "alpha": key[4],
                "n_runs": len(rows),
                "n_diverged": int(sum(bool(row["diverged"]) for row in rows)),
                "n_nondiverged": len(nondiverged),
                "divergence_rate": float(np.mean([row["diverged"] for row in rows])),
                "boundary_inclusive_mean_final_objective": float(
                    np.mean([row["final_objective"] for row in rows])
                ),
                "boundary_inclusive_mean_objective_auc": float(
                    np.mean([row["mean_trapezoidal_objective_auc"] for row in rows])
                ),
                "boundary_inclusive_mean_final_gap_ratio": _nullable_mean(
                    row["final_gap_ratio"] for row in rows
                ),
                "boundary_inclusive_mean_normalized_gap_auc": _nullable_mean(
                    row["normalized_gap_auc"] for row in rows
                ),
                "nondiverged_mean_final_objective": _nullable_mean(
                    row["final_objective"] for row in nondiverged
                ),
                "nondiverged_mean_objective_auc": _nullable_mean(
                    row["mean_trapezoidal_objective_auc"] for row in nondiverged
                ),
                "nondiverged_mean_final_gap_ratio": _nullable_mean(
                    row["final_gap_ratio"] for row in nondiverged
                ),
                "nondiverged_mean_normalized_gap_auc": _nullable_mean(
                    row["normalized_gap_auc"] for row in nondiverged
                ),
                "mean_max_step_amplification": _nullable_mean(
                    row["max_step_amplification"] for row in rows
                ),
                "minimum_denominator_margin": (
                    None if not margins else float(min(margins))
                ),
                "mean_structured_directional_benefit": _nullable_mean(
                    row["mean_structured_directional_benefit"] for row in rows
                ),
                "mean_structured_directional_win_fraction": _nullable_mean(
                    row["structured_directional_win_fraction"] for row in rows
                ),
            }
        )
    return output


def aggregate_directional_comparisons(
    directional: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    iterations: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in directional:
        key = (
            row["regime"],
            row["fitness_transform"],
            row["case"],
            float(row["alpha"]),
            row["reference_state_method"],
        )
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        rows = groups[key]
        reference_summaries = [
            row
            for row in summaries
            if row["regime"] == key[0]
            and row["fitness_transform"] == key[1]
            and row["case"] == key[2]
            and float(row["alpha"]) == key[3]
            and row["method"] == key[4]
        ]
        if not reference_summaries:
            raise ValueError(f"missing reference summaries for directional group {key}")
        expected_steps = len(reference_summaries) * iterations
        benefits = np.asarray(
            [row["structured_directional_benefit"] for row in rows],
            dtype=np.float64,
        )
        output.append(
            {
                "regime": key[0],
                "fitness_transform": key[1],
                "case": key[2],
                "alpha": key[3],
                "reference_state_method": key[4],
                "reference_runs": len(reference_summaries),
                "reference_run_divergence_rate": float(
                    np.mean([row["diverged"] for row in reference_summaries])
                ),
                "expected_reference_steps": expected_steps,
                "observed_reference_steps": len(rows),
                "reference_horizon_completion_fraction": float(
                    len(rows) / expected_steps
                ),
                "mean_structured_improvement": float(
                    np.mean([row["structured_improvement"] for row in rows])
                ),
                "mean_isotropic_improvement": float(
                    np.mean([row["isotropic_improvement"] for row in rows])
                ),
                "mean_structured_directional_benefit": float(np.mean(benefits)),
                "median_structured_directional_benefit": float(np.median(benefits)),
                "structured_directional_win_fraction": float(
                    np.mean(benefits > 0.0)
                ),
                "max_norm_match_relative_error": float(
                    max(row["norm_match_relative_error"] for row in rows)
                ),
            }
        )
    return output


def _finite_numbers(value: Any, path: str = "root") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite number at {path}: {value}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_numbers(child, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_numbers(child, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported metadata value at {path}: {type(value).__name__}")


def validate_result(
    config: BenchmarkConfig,
    cases: Sequence[QuadraticCase],
    trajectories: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    directional: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    exact_cases = [case for case in cases if case.name in EXACT_CASE_NAMES]
    expected_exact = len(exact_cases) * len(config.alphas) * len(EXACT_METHODS)
    expected_mc = 0
    for transform in config.fitness_transforms:
        methods = RAW_MC_METHODS if transform == "raw" else RANK_MC_METHODS
        expected_mc += (
            len(cases) * len(config.alphas) * len(config.mc_seeds) * len(methods)
        )
    expected_summaries = expected_exact + expected_mc
    if len(summaries) != expected_summaries:
        raise ValueError(
            f"expected {expected_summaries} summaries, found {len(summaries)}"
        )
    expected_trajectories = expected_summaries * (config.iterations + 1)
    if len(trajectories) != expected_trajectories:
        raise ValueError(
            f"expected {expected_trajectories} trajectory rows, found {len(trajectories)}"
        )
    summary_keys = [
        (
            row["regime"],
            row["fitness_transform"],
            row["case"],
            row["method"],
            row["alpha"],
            row["seed"],
        )
        for row in summaries
    ]
    if len(set(summary_keys)) != len(summary_keys):
        raise ValueError("duplicate run summaries")
    trajectory_keys = [
        (
            row["regime"],
            row["fitness_transform"],
            row["case"],
            row["method"],
            row["alpha"],
            row["seed"],
            row["iteration"],
        )
        for row in trajectories
    ]
    if len(set(trajectory_keys)) != len(trajectory_keys):
        raise ValueError("duplicate trajectory rows")
    for row in trajectories:
        if row["method"] in {
            "concave_projected_block",
            "norm_matched_isotropic",
        } and row["active_update"]:
            amplification = row["step_amplification"]
            if amplification is None or amplification > 1.0 + 1e-10:
                raise ValueError("safe attenuation amplified an explicit step")
            if row["denominator_nonpositive_fraction"] != 0.0:
                raise ValueError("safe attenuation has a nonpositive denominator")
        if row["method"] == "norm_matched_isotropic" and row["active_update"]:
            error = row["norm_match_relative_error"]
            if error is None or error > 1e-10:
                raise ValueError("isotropic control failed its norm match")
    for case in cases:
        case_rows = [row for row in trajectories if row["case"] == case.name]
        if case.has_finite_maximum:
            if any(row["objective_gap"] is None for row in case_rows):
                raise ValueError("finite-maximum case is missing objective gaps")
            if any(float(row["objective_gap"]) < -1e-10 for row in case_rows):
                raise ValueError("concave objective has a negative gap")
        elif any(row["objective_gap"] is not None for row in case_rows):
            raise ValueError("indefinite case must not report a finite optimum gap")
    for row in directional:
        if row["norm_match_relative_error"] > 1e-10:
            raise ValueError("directional comparison is not norm matched")
    _finite_numbers(list(trajectories), "trajectories")
    _finite_numbers(list(summaries), "summaries")
    _finite_numbers(list(directional), "directional")
    return {
        "complete_matrix": True,
        "strict_finite_numbers": True,
        "unique_run_keys": True,
        "unique_trajectory_keys": True,
        "safe_methods_never_amplify": True,
        "isotropic_control_norm_matched": True,
        "indefinite_gap_suppressed": True,
        "expected_summary_rows": expected_summaries,
        "expected_trajectory_rows": expected_trajectories,
        "directional_rows": len(directional),
    }


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    config.validate()
    cases = make_cases(config)
    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    directional: list[dict[str, Any]] = []

    for case_index, case in enumerate(cases):
        if case.name in EXACT_CASE_NAMES:
            for alpha_index, alpha in enumerate(config.alphas):
                group = _run_group(
                    config,
                    case,
                    case_index,
                    alpha,
                    alpha_index,
                    "exact_gradient_sanity",
                    "exact",
                    0,
                    EXACT_METHODS,
                )
                trajectories.extend(group[0])
                summaries.extend(group[1])
                directional.extend(group[2])

        for transform in config.fitness_transforms:
            methods = RAW_MC_METHODS if transform == "raw" else RANK_MC_METHODS
            for alpha_index, alpha in enumerate(config.alphas):
                for seed in config.mc_seeds:
                    group = _run_group(
                        config,
                        case,
                        case_index,
                        alpha,
                        alpha_index,
                        "monte_carlo_es",
                        transform,
                        seed,
                        methods,
                    )
                    trajectories.extend(group[0])
                    summaries.extend(group[1])
                    directional.extend(group[2])

    aggregates = aggregate_summaries(summaries)
    directional_aggregates = aggregate_directional_comparisons(
        directional, summaries, config.iterations
    )
    metadata = tuple(case_metadata(case) for case in cases)
    validation = validate_result(
        config, cases, trajectories, summaries, directional
    )
    expected_aggregate_rows = (
        len([case for case in cases if case.name in EXACT_CASE_NAMES])
        * len(config.alphas)
        * len(EXACT_METHODS)
        + len(cases)
        * len(config.alphas)
        * sum(
            len(RAW_MC_METHODS) if transform == "raw" else len(RANK_MC_METHODS)
            for transform in config.fitness_transforms
        )
    )
    expected_directional_aggregate_rows = (
        len([case for case in cases if case.name in EXACT_CASE_NAMES])
        * len(config.alphas)
        + len(cases) * len(config.alphas) * len(config.fitness_transforms)
    )
    if len(aggregates) != expected_aggregate_rows:
        raise ValueError("descriptive aggregate matrix is incomplete")
    if len(directional_aggregates) != expected_directional_aggregate_rows:
        raise ValueError("directional aggregate matrix is incomplete")
    for row in aggregates:
        if row["n_diverged"] + row["n_nondiverged"] != row["n_runs"]:
            raise ValueError("aggregate divergence counts do not sum to n_runs")
        if not np.isclose(
            row["divergence_rate"], row["n_diverged"] / row["n_runs"]
        ):
            raise ValueError("aggregate divergence rate disagrees with counts")
        if row["n_nondiverged"] == 0 and any(
            row[key] is not None
            for key in (
                "nondiverged_mean_final_objective",
                "nondiverged_mean_objective_auc",
                "nondiverged_mean_final_gap_ratio",
                "nondiverged_mean_normalized_gap_auc",
            )
        ):
            raise ValueError("all-diverged aggregate has nondiverged performance")
    for row in directional_aggregates:
        completion = row["reference_horizon_completion_fraction"]
        if not 0.0 < completion <= 1.0:
            raise ValueError("directional reference completion must be in (0, 1]")
        if row["max_norm_match_relative_error"] > 1e-10:
            raise ValueError("directional aggregate contains a failed norm match")
    validation["divergence_first_aggregate_schema"] = True
    validation["directional_aggregate_rows"] = len(directional_aggregates)
    validation["aggregate_rows"] = len(aggregates)
    _finite_numbers(list(aggregates), "aggregates")
    _finite_numbers(list(directional_aggregates), "directional_aggregates")
    return BenchmarkResult(
        trajectories=tuple(trajectories),
        summaries=tuple(summaries),
        directional=tuple(directional),
        aggregates=tuple(aggregates),
        directional_aggregates=tuple(directional_aggregates),
        case_metadata=metadata,
        validation=validation,
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row[key]) for key in fieldnames})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


def _method_metadata() -> dict[str, dict[str, Any]]:
    return {
        "explicit_es": {
            "curvature": "none",
            "update": "alpha_times_sampled_gradient",
        },
        "explicit_exact_gradient": {
            "curvature": "none",
            "update": "alpha_times_exact_gradient",
        },
        "oracle_full_implicit": {
            "curvature": "known_full_quadratic_hessian",
            "update": "solve_(I-alpha_H)_delta=alpha_g",
        },
        "frozen_batch_full_signed": {
            "curvature": "full_frozen_rank_batch_jacobian_not_population_hessian",
            "update": "solve_(I-alpha_J_batch)_delta=alpha_g",
        },
        "oracle_diagonal_approximation_signed": {
            "curvature": "known_hessian_diagonal_only_not_full_hessian",
            "update": "signed_diagonal_solve",
        },
        "sampled_signed_diagonal": {
            "curvature": "sampled_diagonal_statistic",
            "update": "signed_diagonal_solve",
        },
        "oracle_block_approximation_signed": {
            "curvature": "known_block_mean_diagonal_approximation_not_full_hessian",
            "update": "signed_block_scalar_solve",
        },
        "sampled_signed_block": {
            "curvature": "sampled_block_mean_statistic",
            "update": "signed_block_scalar_solve",
        },
        "concave_projected_block": {
            "curvature": "max(-sampled_or_exact_block_statistic,0)",
            "update": "structured_denominator_at_least_one",
        },
        "norm_matched_isotropic": {
            "curvature": "used_only_to_compute_reference_step_norm",
            "update": "explicit_direction_scaled_to_structured_step_norm",
        },
    }


def _render_report(
    config: BenchmarkConfig,
    result: BenchmarkResult,
    manifest_name: str,
) -> str:
    directional_alpha = 1.0 if 1.0 in config.alphas else float(config.alphas[0])
    lines = [
        "# Implicit Quadratic Optimization Benchmark",
        "",
        "This is a deterministic synthetic mechanism benchmark. It does not",
        "establish performance on reinforcement-learning tasks.",
        "",
        "## Protocol",
        "",
        f"- Dimension: `{config.dimension}`",
        f"- Blocks: `{config.num_blocks}`",
        f"- Population: `{config.population_size}`",
        f"- Iterations: `{config.iterations}`",
        f"- Perturbation scale: `{config.sigma}`",
        f"- Learning rates: `{list(config.alphas)}`",
        f"- Monte Carlo seeds: `{list(config.mc_seeds)}`",
        f"- Parameter-norm divergence threshold: `{config.divergence_threshold}`",
        "- Finite-optimum gap-ratio divergence threshold: "
        f"`{config.objective_gap_divergence_ratio}`",
        "- All methods in a cell use identical perturbations and additive-noise draws.",
        "- Indefinite cases have no finite objective gap and are stress tests only.",
        "",
        "## Exact Block-Aligned Sanity",
        "",
        "Divergence is reported before performance. Gap AUC is shown only for",
        "nondiverged runs; boundary-capped values remain audit fields, not",
        "ordinary performance observations.",
        "",
        "| Alpha | Method | Diverged | Nondiverged gap AUC | Final gap ratio | Max amplification |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    selected = [
        row
        for row in result.summaries
        if row["regime"] == "exact_gradient_sanity"
        and row["case"] == "block_aligned_concave"
    ]
    selected.sort(key=lambda row: (float(row["alpha"]), str(row["method"])))
    for row in selected:
        if row["diverged"]:
            auc = "--"
            ratio = "boundary-capped"
        else:
            auc = f"{float(row['normalized_gap_auc']):.6g}"
            ratio = f"{float(row['final_gap_ratio']):.6g}"
        lines.append(
            "| {alpha:g} | `{method}` | {diverged} | {auc} | {ratio} | {amp:.6g} |".format(
                alpha=float(row["alpha"]),
                method=row["method"],
                diverged="yes" if row["diverged"] else "no",
                auc=auc,
                ratio=ratio,
                amp=float(row["max_step_amplification"]),
            )
        )
    lines.extend(
        [
            "",
            f"## Equal-Norm Directional Control At Alpha {directional_alpha:g}",
            "",
            "These values use the explicit method's states and compare true one-step",
            "improvement for structured and isotropic steps with identical norm.",
            "Completion below one means the explicit reference path diverged, in which",
            "case its directional mean must not be treated as a full-horizon result.",
            "",
            "| Transform | Case | Reference divergence | Reference completion | Mean structured-minus-isotropic benefit | Win fraction |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    directional_selected = [
        row
        for row in result.directional_aggregates
        if row["regime"] == "monte_carlo_es"
        and float(row["alpha"]) == directional_alpha
        and row["case"] != "rotated_indefinite"
    ]
    directional_selected.sort(
        key=lambda row: (str(row["fitness_transform"]), str(row["case"]))
    )
    for row in directional_selected:
        lines.append(
            "| `{transform}` | `{case}` | {divergence:.3f} | {completion:.3f} | {benefit:.6g} | {wins:.3f} |".format(
                transform=row["fitness_transform"],
                case=row["case"],
                divergence=float(row["reference_run_divergence_rate"]),
                completion=float(row["reference_horizon_completion_fraction"]),
                benefit=float(row["mean_structured_directional_benefit"]),
                wins=float(row["structured_directional_win_fraction"]),
            )
        )
    lines.extend(
        [
            "",
            "## Artifact Contract",
            "",
            f"- Manifest: `{manifest_name}`",
            "- Per-run outcomes: `run_summary.csv`",
            "- Per-update trajectories: `trajectories.csv`",
            "- Common explicit-state directional controls: `directional_comparison.csv`",
            "- Explicit-state directional aggregates: `directional_aggregate.csv`",
            "- Across-seed descriptive aggregates: `aggregate_summary.csv`",
            "",
            "Aggregate performance fields are split into boundary-inclusive audit",
            "values and nondiverged-only descriptives. When divergence rate is nonzero,",
            "it is the primary outcome; nondiverged means are survivor descriptives,",
            "not estimates for the complete method population.",
            "",
            "The raw-fitness regime targets the known quadratic Hessian. The",
            "same-batch-rank regime reports a frozen-rank covariance-score surrogate",
            "and intentionally has no literal-Hessian RMSE. Structured-versus-isotropic",
            "benefit is the difference in true one-step objective improvement between",
            "equal-norm counterfactual steps from the same state.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    output_dir: str | os.PathLike[str],
    config: BenchmarkConfig,
    result: BenchmarkResult,
) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "trajectories": root / "trajectories.csv",
        "run_summary": root / "run_summary.csv",
        "directional_comparison": root / "directional_comparison.csv",
        "aggregate_summary": root / "aggregate_summary.csv",
        "directional_aggregate": root / "directional_aggregate.csv",
        "manifest": root / "benchmark_manifest.json",
        "report": root / "report.md",
    }
    _atomic_write_csv(
        paths["trajectories"], TRAJECTORY_FIELDS, result.trajectories
    )
    _atomic_write_csv(paths["run_summary"], SUMMARY_FIELDS, result.summaries)
    _atomic_write_csv(
        paths["directional_comparison"], DIRECTIONAL_FIELDS, result.directional
    )
    _atomic_write_csv(
        paths["aggregate_summary"], AGGREGATE_FIELDS, result.aggregates
    )
    _atomic_write_csv(
        paths["directional_aggregate"],
        DIRECTIONAL_AGGREGATE_FIELDS,
        result.directional_aggregates,
    )
    data_files = {
        key: {
            "path": path.name,
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
            "rows": {
                "trajectories": len(result.trajectories),
                "run_summary": len(result.summaries),
                "directional_comparison": len(result.directional),
                "aggregate_summary": len(result.aggregates),
                "directional_aggregate": len(result.directional_aggregates),
            }[key],
        }
        for key, path in paths.items()
        if key in {
            "trajectories",
            "run_summary",
            "directional_comparison",
            "aggregate_summary",
            "directional_aggregate",
        }
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "scope": {
            "synthetic_quadratic_only": True,
            "rl_environment_evaluated": False,
            "general_performance_claim_supported": False,
            "raw_fitness_literal_hessian_target": True,
            "same_batch_rank_literal_hessian_target": False,
            "indefinite_case_has_finite_maximum": False,
        },
        "config": asdict(config),
        "cases": list(result.case_metadata),
        "methods": _method_metadata(),
        "metric_definitions": {
            "objective_gap": "zero_minus_objective_for_strictly_concave_cases_only",
            "mean_trapezoidal_objective_auc": "trapezoidal_mean_over_update_index_higher_is_better",
            "normalized_gap_auc": "trapezoidal_mean_gap_divided_by_initial_gap_lower_is_better",
            "divergence": "parameter_norm_or_objective_gap_boundary_nonfinite_value_or_singular_system",
            "objective_gap_divergence": "finite_optimum_gap_exceeds_configured_multiple_of_initial_gap",
            "post_divergence_policy": "applicable_boundary_state_carried_forward_to_fixed_horizon",
            "aggregate_performance_policy": "report_divergence_first_boundary_inclusive_values_for_audit_and_nondiverged_only_values_as_survivor_descriptives",
            "step_amplification": "step_norm_divided_by_alpha_gradient_norm",
            "denominator_margin": "minimum_absolute_eigenvalue_or_diagonal_denominator",
            "structured_directional_benefit": "true_improvement_structured_minus_true_improvement_equal_norm_isotropic",
        },
        "randomness": {
            "generator": "numpy_PCG64_via_default_rng",
            "seed_sequence": "[master_seed,2001,case_index,alpha_index,mc_seed,iteration]",
            "common_across_methods": True,
            "common_across_fitness_transforms": True,
            "additive_noise": "independent_plus_minus_draws_common_across_methods",
        },
        "raw_estimator": {
            "gradient": "antithetic_raw_return_difference",
            "curvature": "second_order_gaussian_score",
            "baseline": "leave_one_antithetic_pair_out_pair_sum",
        },
        "rank_estimator": {
            "transform": "tie_aware_same_batch_centered_rank",
            "interpretation": "frozen_batch_covariance_score_not_literal_population_hessian",
        },
        "validation": result.validation,
        "files": data_files,
        "provenance": {
            "source_file": "experiments/implicit_quadratic_optimization_benchmark.py",
            "source_sha256": _source_sha256(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    _finite_numbers(payload, "manifest")
    _atomic_write_text(
        paths["manifest"],
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write_text(
        paths["report"],
        _render_report(config, result, paths["manifest"].name),
    )
    return {key: str(path) for key, path in paths.items()}


def _parse_csv_numbers(value: str, cast: type) -> tuple[Any, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("list must not be empty")
    try:
        return tuple(cast(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/implicit_quadratic_optimization_benchmark",
    )
    parser.add_argument("--dimension", type=int, default=12)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--alphas", default="0.1,1,10")
    parser.add_argument("--mc-seeds", default="0,1,2,3,4")
    parser.add_argument("--cases", default=",".join(CASE_NAMES))
    parser.add_argument(
        "--fitness-transforms", default=",".join(FITNESS_TRANSFORMS)
    )
    parser.add_argument("--additive-noise-std", type=float, default=0.05)
    parser.add_argument("--master-seed", type=int, default=20260712)
    parser.add_argument("--divergence-threshold", type=float, default=1.0e6)
    parser.add_argument(
        "--objective-gap-divergence-ratio", type=float, default=1.0e6
    )
    parser.add_argument("--singular-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a small smoke-test matrix while preserving all cases and methods.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    if args.quick:
        dimension = 8
        num_blocks = 2
        population = 32
        iterations = 6
        alphas = (0.1, 10.0)
        seeds = (0,)
    else:
        dimension = args.dimension
        num_blocks = args.num_blocks
        population = args.population_size
        iterations = args.iterations
        alphas = _parse_csv_numbers(args.alphas, float)
        seeds = _parse_csv_numbers(args.mc_seeds, int)
    config = BenchmarkConfig(
        dimension=dimension,
        num_blocks=num_blocks,
        population_size=population,
        iterations=iterations,
        sigma=args.sigma,
        alphas=alphas,
        mc_seeds=seeds,
        cases=tuple(part.strip() for part in args.cases.split(",") if part.strip()),
        fitness_transforms=tuple(
            part.strip()
            for part in args.fitness_transforms.split(",")
            if part.strip()
        ),
        additive_noise_std=args.additive_noise_std,
        master_seed=args.master_seed,
        divergence_threshold=args.divergence_threshold,
        objective_gap_divergence_ratio=args.objective_gap_divergence_ratio,
        singular_tolerance=args.singular_tolerance,
    )
    config.validate()
    return config


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    result = run_benchmark(config)
    outputs = write_outputs(args.output_dir, config, result)
    print(
        "Validated implicit quadratic benchmark: "
        f"summaries={len(result.summaries)}, "
        f"trajectories={len(result.trajectories)}, "
        f"manifest={outputs['manifest']}"
    )


if __name__ == "__main__":
    main()

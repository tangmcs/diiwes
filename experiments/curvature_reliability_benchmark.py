#!/usr/bin/env python3
"""Controlled reliability benchmark for ES curvature estimators.

The benchmark uses centered quadratic objectives with known Hessians.  It is
deliberately independent of an RL environment and of the optimizer update
loop.  Four estimators are compared on identical antithetic samples:

* signed Stein diagonal moment;
* pooled block Stein moment;
* joint block OLS;
* one-sided confidence-gated joint block OLS.

For additive noise, ``independent`` uses independent noise for the plus,
minus, and center evaluations.  ``crn`` uses the same additive draw for that
plus/center/minus triplet, so the center second-difference control variate
cancels it exactly.  This is an idealized, explicit CRN assumption rather than
an assertion about a stochastic simulator.

Accuracy metrics for the first three methods target signed diagonal or block
mean curvature.  The confidence-gated method reports its effective concave
curvature, ``min(estimate + z * se, 0)``, against the true concave component.
Its ``raw_*`` metrics and interval coverage still evaluate the underlying
signed OLS estimate.  Signed methods use ``1 - alpha * h_hat`` for resonance
diagnostics; the gate uses its positive, no-amplification denominator.

The estimator input is a raw centered return second difference.  No rank
transformation is applied.  Therefore this benchmark tests raw-return
Stein/OLS formulas and numerical helper arithmetic, not the batch-dependent
same-generation rank surrogate used by the Hopper experiments.  Moment
standard errors treat pair contributions as independent even when a
leave-one-out baseline couples them, and OLS uses the classical homoskedastic
formula.  Reported coverage is finite-sample empirical calibration, not a
coverage guarantee.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np


SURFACE_NAMES = (
    "diagonal",
    "block_isotropic",
    "rotated",
    "saddle",
    "rotated_saddle",
)
NOISE_MODES = ("none", "independent", "crn")
METHOD_NAMES = (
    "signed_stein_diagonal",
    "pooled_block_moment",
    "joint_block_ols",
    "confidence_gated_block_ols",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    dimensions: tuple[int, ...] = (16, 64)
    populations: tuple[int, ...] = (64, 200)
    sigmas: tuple[float, ...] = (0.02, 0.1)
    surfaces: tuple[str, ...] = SURFACE_NAMES
    noise_modes: tuple[str, ...] = NOISE_MODES
    noise_stds: tuple[float, ...] = (1.0,)
    repetitions: int = 100
    num_blocks: int = 4
    seed: int = 20260712
    learning_rate: float = 10.0
    gate_z: float = 1.645
    coverage_z: float = 1.959963984540054
    resonance_tolerance: float = 0.05
    moment_baseline: str = "loo"

    def validate(self) -> None:
        if not self.dimensions or any(value < 2 for value in self.dimensions):
            raise ValueError("dimensions must contain integers of at least two")
        if not self.populations or any(
            value < 4 or value % 2 for value in self.populations
        ):
            raise ValueError("populations must contain even integers of at least four")
        if not self.sigmas or any(
            not np.isfinite(value) or value <= 0.0 for value in self.sigmas
        ):
            raise ValueError("sigmas must contain positive finite values")
        if not self.surfaces or any(
            value not in SURFACE_NAMES for value in self.surfaces
        ):
            raise ValueError(f"surfaces must be selected from {SURFACE_NAMES}")
        if not self.noise_modes or any(
            value not in NOISE_MODES for value in self.noise_modes
        ):
            raise ValueError(f"noise_modes must be selected from {NOISE_MODES}")
        if not self.noise_stds or any(
            not np.isfinite(value) or value < 0.0 for value in self.noise_stds
        ):
            raise ValueError("noise_stds must contain nonnegative finite values")
        if self.repetitions < 2:
            raise ValueError("repetitions must be at least two")
        if self.num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.gate_z) or self.gate_z < 0.0:
            raise ValueError("gate_z must be nonnegative and finite")
        if not np.isfinite(self.coverage_z) or self.coverage_z <= 0.0:
            raise ValueError("coverage_z must be positive and finite")
        if (
            not np.isfinite(self.resonance_tolerance)
            or self.resonance_tolerance <= 0.0
        ):
            raise ValueError("resonance_tolerance must be positive and finite")
        if self.moment_baseline not in {"none", "loo"}:
            raise ValueError("moment_baseline must be one of none, loo")

        for dimension in self.dimensions:
            component_count = min(self.num_blocks, dimension)
            for population in self.populations:
                pair_count = population // 2
                if pair_count <= component_count + 1:
                    raise ValueError(
                        "joint OLS needs more antithetic pairs than block "
                        f"parameters: d={dimension}, population={population}, "
                        f"blocks={component_count}"
                    )


@dataclass(frozen=True)
class QuadraticSurface:
    name: str
    hessian: np.ndarray
    blocks: tuple[slice, ...]

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

    def centered_pair_response(self, eps: np.ndarray, sigma: float) -> np.ndarray:
        """Return f(+sigma eps) + f(-sigma eps) - 2 f(0)."""
        eps = np.asarray(eps, dtype=np.float64)
        if eps.ndim != 2 or eps.shape[1] != self.dimension:
            raise ValueError("eps has the wrong shape for this surface")
        return sigma**2 * np.einsum(
            "bi,ij,bj->b", eps, self.hessian, eps, optimize=True
        )


@dataclass(frozen=True)
class Estimate:
    value: np.ndarray
    standard_error: np.ndarray
    contributions: np.ndarray | None = None
    diagnostics: dict[str, float | int] | None = None


def make_blocks(dimension: int, num_blocks: int) -> tuple[slice, ...]:
    if dimension < 1 or num_blocks < 1:
        raise ValueError("dimension and num_blocks must be positive")
    count = min(dimension, num_blocks)
    base, extra = divmod(dimension, count)
    blocks: list[slice] = []
    start = 0
    for index in range(count):
        width = base + (1 if index < extra else 0)
        blocks.append(slice(start, start + width))
        start += width
    return tuple(blocks)


def _orthogonal_matrix(dimension: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def _block_matrix(blocks: Sequence[slice], values: np.ndarray) -> np.ndarray:
    dimension = int(blocks[-1].stop)
    diagonal = np.empty(dimension, dtype=np.float64)
    for block, value in zip(blocks, values, strict=True):
        diagonal[block] = value
    return np.diag(diagonal)


def make_surface(
    name: str,
    dimension: int,
    blocks: Sequence[slice],
    *,
    seed: int,
) -> QuadraticSurface:
    if name not in SURFACE_NAMES:
        raise ValueError(f"unknown surface {name!r}")
    blocks_tuple = tuple(blocks)
    if not blocks_tuple or int(blocks_tuple[-1].stop) != dimension:
        raise ValueError("blocks must cover the surface dimension")

    rng = np.random.default_rng(np.random.SeedSequence([seed, dimension, 1701]))
    if name == "diagonal":
        eigenvalues = -np.geomspace(0.25, 3.0, dimension)
        hessian = np.diag(eigenvalues)
    elif name == "block_isotropic":
        values = -np.geomspace(0.25, 3.0, len(blocks_tuple))
        hessian = _block_matrix(blocks_tuple, values)
    elif name == "rotated":
        eigenvalues = -np.geomspace(0.1, 4.0, dimension)
        q = _orthogonal_matrix(dimension, rng)
        hessian = q @ np.diag(eigenvalues) @ q.T
    elif name == "saddle":
        magnitudes = np.geomspace(0.35, 2.5, len(blocks_tuple))
        signs = np.where(np.arange(len(blocks_tuple)) % 2 == 0, -1.0, 1.0)
        hessian = _block_matrix(blocks_tuple, signs * magnitudes)
    else:
        half = (dimension + 1) // 2
        negative = -np.geomspace(0.25, 3.0, half)
        positive = np.geomspace(0.25, 3.0, dimension - half)
        eigenvalues = np.concatenate((negative, positive))
        rng.shuffle(eigenvalues)
        q = _orthogonal_matrix(dimension, rng)
        hessian = q @ np.diag(eigenvalues) @ q.T

    hessian = 0.5 * (hessian + hessian.T)
    return QuadraticSurface(
        name=name,
        hessian=hessian.astype(np.float64, copy=False),
        blocks=blocks_tuple,
    )


def make_pair_response(
    surface: QuadraticSurface,
    eps: np.ndarray,
    sigma: float,
    *,
    noise_mode: str,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if noise_mode not in NOISE_MODES:
        raise ValueError(f"unknown noise mode {noise_mode!r}")
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be nonnegative and finite")
    response = surface.centered_pair_response(eps, sigma)
    if noise_mode == "none" or noise_std == 0.0:
        return response
    pair_count = len(eps)
    if noise_mode == "independent":
        noise = rng.normal(scale=noise_std, size=(pair_count, 3))
        return response + noise[:, 0] + noise[:, 1] - 2.0 * noise[:, 2]

    # Consume the stochastic stream but return the algebraically exact CRN
    # cancellation.  Computing (response + eta) + eta - 2 eta in floating
    # point can erase low-sigma response bits before the subtraction.
    rng.normal(scale=noise_std, size=pair_count)
    return response


def _moment_contributions(
    pair_response: np.ndarray,
    features: np.ndarray,
    sigma: float,
    *,
    baseline: str,
) -> np.ndarray:
    response = np.asarray(pair_response, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if response.ndim != 1 or features.ndim != 2 or len(response) != len(features):
        raise ValueError("moment response/features have incompatible shapes")
    if len(response) < 2:
        raise ValueError("moment estimation needs at least two pairs")
    if baseline == "none":
        centered = response
    elif baseline == "loo":
        centered = response - (np.sum(response) - response) / (len(response) - 1)
    else:
        raise ValueError("baseline must be one of none, loo")
    return centered[:, None] * features / (2.0 * sigma**2)


def _estimate_from_contributions(contributions: np.ndarray) -> Estimate:
    contributions = np.asarray(contributions, dtype=np.float64)
    value = np.mean(contributions, axis=0)
    standard_error = np.std(contributions, axis=0, ddof=1) / np.sqrt(
        len(contributions)
    )
    return Estimate(
        value=value.astype(np.float64, copy=False),
        standard_error=standard_error.astype(np.float64, copy=False),
        contributions=contributions,
    )


def estimate_stein_diagonal(
    eps: np.ndarray,
    pair_response: np.ndarray,
    sigma: float,
    *,
    baseline: str = "loo",
) -> Estimate:
    eps = np.asarray(eps, dtype=np.float64)
    features = eps**2 - 1.0
    return _estimate_from_contributions(
        _moment_contributions(pair_response, features, sigma, baseline=baseline)
    )


def estimate_pooled_block_moment(
    eps: np.ndarray,
    pair_response: np.ndarray,
    sigma: float,
    blocks: Sequence[slice],
    *,
    baseline: str = "loo",
) -> Estimate:
    eps = np.asarray(eps, dtype=np.float64)
    features = np.column_stack(
        [np.mean(eps[:, block] ** 2, axis=1) - 1.0 for block in blocks]
    )
    return _estimate_from_contributions(
        _moment_contributions(pair_response, features, sigma, baseline=baseline)
    )


def estimate_joint_block_ols(
    eps: np.ndarray,
    pair_response: np.ndarray,
    sigma: float,
    blocks: Sequence[slice],
) -> Estimate:
    eps = np.asarray(eps, dtype=np.float64)
    response = np.asarray(pair_response, dtype=np.float64)
    if eps.ndim != 2 or response.shape != (len(eps),):
        raise ValueError("OLS response/perturbations have incompatible shapes")
    features = np.column_stack(
        [np.mean(eps[:, block] ** 2, axis=1) - 1.0 for block in blocks]
    )
    design = np.column_stack((np.ones(len(eps), dtype=np.float64), features))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, response, rcond=None
    )
    parameter_count = design.shape[1]
    if rank != parameter_count:
        raise FloatingPointError(
            f"joint OLS design rank {rank}, expected {parameter_count}"
        )
    residual_dof = len(response) - parameter_count
    if residual_dof <= 0:
        raise FloatingPointError("joint OLS has no residual degrees of freedom")
    residual = response - design @ coefficients
    residual_variance = float(np.dot(residual, residual) / residual_dof)
    covariance = residual_variance * np.linalg.pinv(
        design.T @ design, rcond=1e-12
    )
    coefficient_variance = np.maximum(np.diag(covariance)[1:], 0.0)
    block_sizes = np.asarray(
        [int(block.stop) - int(block.start) for block in blocks],
        dtype=np.float64,
    )
    scale = sigma**2 * block_sizes
    value = coefficients[1:] / scale
    standard_error = np.sqrt(coefficient_variance) / scale
    condition = float(
        np.max(singular_values) / max(float(np.min(singular_values)), 1e-300)
    )
    diagnostics: dict[str, float | int] = {
        "rank": int(rank),
        "residual_dof": int(residual_dof),
        "residual_std": float(np.sqrt(residual_variance)),
        "design_condition": condition,
    }
    return Estimate(
        value=value.astype(np.float64, copy=False),
        standard_error=standard_error.astype(np.float64, copy=False),
        diagnostics=diagnostics,
    )


def verify_against_core_helpers(seed: int = 9917) -> dict[str, float]:
    """Cross-check independent formulas against current optimizer helpers."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from core.implicit_es import ConcaveCurvatureES

    rng = np.random.default_rng(seed)
    dimension = 8
    pair_count = 20
    population = 2 * pair_count
    sigma = 0.17
    blocks = make_blocks(dimension, 4)
    eps = rng.normal(size=(pair_count, dimension))
    response = rng.normal(size=pair_count)

    independent_diag = estimate_stein_diagonal(
        eps, response, sigma, baseline="none"
    )
    diag_optimizer = ConcaveCurvatureES(
        num_params=dimension,
        population_size=population,
        noise_std=sigma,
        curvature_structure="diag",
    )
    core_diag = diag_optimizer._stein_moment_estimate(eps, response)

    independent_block = estimate_pooled_block_moment(
        eps, response, sigma, blocks, baseline="none"
    )
    block_optimizer = ConcaveCurvatureES(
        num_params=dimension,
        population_size=population,
        noise_std=sigma,
        curvature_structure="block",
        block_slices=blocks,
    )
    core_block = block_optimizer._stein_moment_estimate(eps, response)

    independent_ols = estimate_joint_block_ols(eps, response, sigma, blocks)
    ols_optimizer = ConcaveCurvatureES(
        num_params=dimension,
        population_size=population,
        noise_std=sigma,
        curvature_structure="block",
        block_slices=blocks,
        curvature_estimator="block_joint_ols",
    )
    core_ols = ols_optimizer._fit_block_joint_ols(eps, response)

    errors = {
        "stein_value_max_abs_error": float(
            np.max(np.abs(independent_diag.value - core_diag[0]))
        ),
        "stein_se_max_abs_error": float(
            np.max(np.abs(independent_diag.standard_error - core_diag[2]))
        ),
        "block_value_max_abs_error": float(
            np.max(np.abs(independent_block.value - core_block[0]))
        ),
        "block_se_max_abs_error": float(
            np.max(np.abs(independent_block.standard_error - core_block[2]))
        ),
        "ols_value_max_abs_error": float(
            np.max(np.abs(independent_ols.value - core_ols[0]))
        ),
        "ols_se_max_abs_error": float(
            np.max(np.abs(independent_ols.standard_error - core_ols[1]))
        ),
    }
    if any(value > 1e-10 for value in errors.values()):
        raise AssertionError(f"independent estimator formulas disagree: {errors}")
    return errors


def _expand_components(
    values: np.ndarray, blocks: Sequence[slice], dimension: int
) -> np.ndarray:
    expanded = np.empty(dimension, dtype=np.float64)
    for value, block in zip(values, blocks, strict=True):
        expanded[block] = value
    return expanded


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


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _noise_settings(config: BenchmarkConfig) -> list[tuple[str, float]]:
    settings: list[tuple[str, float]] = []
    for mode in config.noise_modes:
        if mode == "none":
            settings.append((mode, 0.0))
        else:
            settings.extend((mode, value) for value in config.noise_stds)
    return settings


def _rng(master_seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([master_seed, *keys]))


def _method_summary(
    *,
    method: str,
    estimates: Sequence[Estimate],
    signed_target: np.ndarray,
    blocks: Sequence[slice],
    diagonal_target: np.ndarray,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    raw = np.asarray([estimate.value for estimate in estimates], dtype=np.float64)
    se = np.asarray(
        [estimate.standard_error for estimate in estimates], dtype=np.float64
    )
    repeated_signed_target = np.broadcast_to(signed_target, raw.shape)
    confidence_upper = raw + config.gate_z * se
    gated = method == "confidence_gated_block_ols"
    if gated:
        reported = np.minimum(confidence_upper, 0.0)
        accuracy_target = np.broadcast_to(np.minimum(signed_target, 0.0), raw.shape)
        predicted_concave = confidence_upper < 0.0
        denominator = 1.0 - config.learning_rate * reported
    else:
        reported = raw
        accuracy_target = repeated_signed_target
        predicted_concave = raw < 0.0
        denominator = 1.0 - config.learning_rate * raw

    raw_error = raw - repeated_signed_target
    error = reported - accuracy_target
    true_concave = repeated_signed_target < 0.0
    true_positive = int(np.sum(predicted_concave & true_concave))
    false_positive = int(np.sum(predicted_concave & ~true_concave))
    false_negative = int(np.sum(~predicted_concave & true_concave))
    true_negative = int(np.sum(~predicted_concave & ~true_concave))
    coverage = np.abs(raw_error) <= config.coverage_z * se + 1e-12

    absolute_denominator = np.abs(denominator)
    amplification = 1.0 / np.maximum(absolute_denominator, 1e-300)
    resonance = absolute_denominator < config.resonance_tolerance
    nonpositive = denominator <= 0.0

    dimension = len(diagonal_target)
    if method == "signed_stein_diagonal":
        expanded = reported
        expanded_target = np.broadcast_to(diagonal_target, expanded.shape)
    else:
        expanded = np.asarray(
            [
                _expand_components(values, blocks, dimension)
                for values in reported
            ]
        )
        target = np.minimum(diagonal_target, 0.0) if gated else diagonal_target
        expanded_target = np.broadcast_to(target, expanded.shape)

    return {
        "method": method,
        "input_fitness": "raw_centered_second_difference",
        "rank_surrogate_evaluated": False,
        "standard_error_method": (
            "naive_pair_contribution_se_with_loo_dependence"
            if method in {"signed_stein_diagonal", "pooled_block_moment"}
            and config.moment_baseline == "loo"
            else "classical_iid_pair_contribution_se"
            if method in {"signed_stein_diagonal", "pooled_block_moment"}
            else "classical_homoskedastic_ols"
        ),
        "coverage_interpretation": "empirical_finite_sample_not_guaranteed",
        "estimand": (
            "concave_block_mean_diagonal"
            if gated
            else "diagonal_hessian"
            if method == "signed_stein_diagonal"
            else "block_mean_diagonal_hessian"
        ),
        "components": int(raw.shape[1]),
        "replications": int(raw.shape[0]),
        "bias": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "correlation": _safe_correlation(reported, accuracy_target),
        "raw_bias": float(np.mean(raw_error)),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "raw_correlation": _safe_correlation(raw, repeated_signed_target),
        "expanded_rmse": float(
            np.sqrt(np.mean((expanded - expanded_target) ** 2))
        ),
        "mean_reported_standard_error": float(np.mean(se)),
        "coverage": float(np.mean(coverage)),
        "coverage_z": config.coverage_z,
        "sign_precision": _ratio(true_positive, true_positive + false_positive),
        "sign_recall": _ratio(true_positive, true_positive + false_negative),
        "sign_true_positive": true_positive,
        "sign_false_positive": false_positive,
        "sign_false_negative": false_negative,
        "sign_true_negative": true_negative,
        "gate_activation_rate": (
            float(np.mean(predicted_concave)) if gated else None
        ),
        "resonance_probability": float(np.mean(resonance)),
        "nonpositive_denominator_probability": float(np.mean(nonpositive)),
        "max_amplification": float(np.max(amplification)),
        "p99_amplification": float(np.quantile(amplification, 0.99)),
        "mean_amplification": float(np.mean(amplification)),
    }


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    config.validate()
    rows: list[dict[str, Any]] = []
    surface_indices = {name: index for index, name in enumerate(SURFACE_NAMES)}
    noise_indices = {name: index for index, name in enumerate(NOISE_MODES)}

    for dimension in config.dimensions:
        blocks = make_blocks(dimension, config.num_blocks)
        for surface_name in config.surfaces:
            surface_index = surface_indices[surface_name]
            surface = make_surface(
                surface_name,
                dimension,
                blocks,
                seed=config.seed + 1009 * surface_index,
            )
            for population in config.populations:
                pair_count = population // 2
                for sigma_index, sigma in enumerate(config.sigmas):
                    for noise_setting_index, (noise_mode, noise_std) in enumerate(
                        _noise_settings(config)
                    ):
                        noise_index = noise_indices[noise_mode]
                        estimates: dict[str, list[Estimate]] = {
                            name: [] for name in METHOD_NAMES
                        }
                        design_conditions: list[float] = []
                        for repetition in range(config.repetitions):
                            eps_rng = _rng(
                                config.seed,
                                2003,
                                surface_index,
                                dimension,
                                population,
                                sigma_index,
                                repetition,
                            )
                            eps = eps_rng.normal(size=(pair_count, dimension))
                            noise_rng = _rng(
                                config.seed,
                                3011,
                                surface_index,
                                dimension,
                                population,
                                sigma_index,
                                noise_index,
                                noise_setting_index,
                                repetition,
                            )
                            response = make_pair_response(
                                surface,
                                eps,
                                sigma,
                                noise_mode=noise_mode,
                                noise_std=noise_std,
                                rng=noise_rng,
                            )
                            estimates["signed_stein_diagonal"].append(
                                estimate_stein_diagonal(
                                    eps,
                                    response,
                                    sigma,
                                    baseline=config.moment_baseline,
                                )
                            )
                            estimates["pooled_block_moment"].append(
                                estimate_pooled_block_moment(
                                    eps,
                                    response,
                                    sigma,
                                    blocks,
                                    baseline=config.moment_baseline,
                                )
                            )
                            ols = estimate_joint_block_ols(
                                eps, response, sigma, blocks
                            )
                            estimates["joint_block_ols"].append(ols)
                            estimates["confidence_gated_block_ols"].append(ols)
                            if ols.diagnostics is not None:
                                design_conditions.append(
                                    float(ols.diagnostics["design_condition"])
                                )

                        common = {
                            "surface": surface_name,
                            "dimension": dimension,
                            "population": population,
                            "antithetic_pairs": pair_count,
                            "sigma": sigma,
                            "noise_mode": noise_mode,
                            "noise_std": noise_std,
                            "num_blocks": len(blocks),
                            "min_block_size": min(
                                int(block.stop) - int(block.start) for block in blocks
                            ),
                            "max_block_size": max(
                                int(block.stop) - int(block.start) for block in blocks
                            ),
                            "learning_rate": config.learning_rate,
                            "gate_z": config.gate_z,
                            "moment_baseline": config.moment_baseline,
                            "median_ols_design_condition": float(
                                np.median(design_conditions)
                            ),
                            "true_diagonal_min": float(np.min(surface.diagonal)),
                            "true_diagonal_max": float(np.max(surface.diagonal)),
                            "true_block_min": float(np.min(surface.block_targets)),
                            "true_block_max": float(np.max(surface.block_targets)),
                        }
                        for method in METHOD_NAMES:
                            row = dict(common)
                            row.update(
                                _method_summary(
                                    method=method,
                                    estimates=estimates[method],
                                    signed_target=(
                                        surface.diagonal
                                        if method == "signed_stein_diagonal"
                                        else surface.block_targets
                                    ),
                                    blocks=blocks,
                                    diagonal_target=surface.diagonal,
                                    config=config,
                                )
                            )
                            rows.append(row)
    return rows


def _atomic_write_json(path: str, value: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
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
        raise ValueError("cannot write an empty benchmark")
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
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


def write_benchmark_outputs(
    output: str,
    rows: Sequence[dict[str, Any]],
    config: BenchmarkConfig,
    helper_verification: dict[str, float],
    *,
    metadata_output: str | None = None,
) -> tuple[str, str]:
    metadata_path = metadata_output or f"{os.path.splitext(output)[0]}.json"
    _atomic_write_csv(output, rows)
    payload = {
        "schema_version": 1,
        "benchmark": "controlled_es_curvature_reliability",
        "scope": {
            "input_fitness": "raw_centered_second_difference",
            "rank_surrogate_evaluated": False,
            "rl_environment_evaluated": False,
            "core_helper_verification": (
                "arithmetic equivalence for identical numeric pair responses only"
            ),
            "coverage": (
                "finite-sample empirical calibration; not a guarantee"
            ),
            "moment_standard_error": (
                "naive pair-contribution SE; LOO contributions are dependent"
            ),
            "ols_standard_error": "classical homoskedastic OLS SE",
        },
        "config": asdict(config),
        "core_helper_verification": helper_verification,
        "noise_semantics": {
            "none": "deterministic centered second difference",
            "independent": (
                "independent additive noise in plus, minus, and center evaluations"
            ),
            "crn": (
                "one shared additive draw in each plus/center/minus triplet; "
                "the centered second difference cancels it"
            ),
        },
        "rows": list(rows),
    }
    _atomic_write_json(metadata_path, payload)
    return output, metadata_path


def _parse_csv(value: str, cast: Any, name: str) -> tuple[Any, ...]:
    try:
        parsed = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {error}") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="docs/curvature_reliability_benchmark.csv",
        help="aggregated CSV output",
    )
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--dimensions", default="16,64")
    parser.add_argument("--populations", default="64,200")
    parser.add_argument("--sigmas", default="0.02,0.1")
    parser.add_argument("--surfaces", default=",".join(SURFACE_NAMES))
    parser.add_argument("--noise-modes", default=",".join(NOISE_MODES))
    parser.add_argument("--noise-stds", default="1.0")
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--learning-rate", type=float, default=10.0)
    parser.add_argument("--gate-z", type=float, default=1.645)
    parser.add_argument(
        "--coverage-z", type=float, default=1.959963984540054
    )
    parser.add_argument("--resonance-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--moment-baseline", choices=("none", "loo"), default="loo"
    )
    parser.add_argument(
        "--skip-core-verification",
        action="store_true",
        help="skip exact formula checks against the optimizer helper methods",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig(
        dimensions=_parse_csv(args.dimensions, int, "dimensions"),
        populations=_parse_csv(args.populations, int, "populations"),
        sigmas=_parse_csv(args.sigmas, float, "sigmas"),
        surfaces=_parse_csv(args.surfaces, str, "surfaces"),
        noise_modes=_parse_csv(args.noise_modes, str, "noise modes"),
        noise_stds=_parse_csv(args.noise_stds, float, "noise stds"),
        repetitions=args.repetitions,
        num_blocks=args.num_blocks,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gate_z=args.gate_z,
        coverage_z=args.coverage_z,
        resonance_tolerance=args.resonance_tolerance,
        moment_baseline=args.moment_baseline,
    )
    config.validate()
    helper_verification = (
        {} if args.skip_core_verification else verify_against_core_helpers(config.seed)
    )
    rows = run_benchmark(config)
    output, metadata = write_benchmark_outputs(
        args.output,
        rows,
        config,
        helper_verification,
        metadata_output=args.metadata_output,
    )
    print(
        f"Wrote {len(rows)} aggregated method cells to {output}; "
        f"metadata={metadata}"
    )


if __name__ == "__main__":
    main()

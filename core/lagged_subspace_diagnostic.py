"""Pure numerical primitives for the lagged-subspace LOPO diagnostic.

This module evaluates a frozen antithetic population.  It never constructs a
basis from the current population, evaluates an environment, updates a policy,
or selects a checkpoint.  ``build_lagged_bases`` accepts only a lagged gradient
archive; population evaluators require bases locked before the current batch.

The curvature quantity is a rank-based current-CDF stop-gradient subspace
moment.  It is not a raw-return Hessian or the Hessian of a globally adaptive
rank objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


SUBSPACE_DIMENSION = 3
_FLOAT_EPS = np.finfo(np.float64).eps


class UnresolvedDiagnosticError(ValueError):
    """Raised when a required normalized quantity has a zero denominator."""


@dataclass(frozen=True)
class BasisProvenance:
    """Caller assertion that both bases are previsible to the current batch."""

    primary_reference: str
    random_reference: str
    locked_before_current_batch: bool
    uses_current_noise: bool = False
    uses_current_returns: bool = False

    @classmethod
    def strictly_lagged(
        cls,
        primary_reference: str,
        random_reference: str,
    ) -> "BasisProvenance":
        return cls(
            primary_reference=primary_reference,
            random_reference=random_reference,
            locked_before_current_batch=True,
            uses_current_noise=False,
            uses_current_returns=False,
        )

    def validate(self) -> None:
        if not self.primary_reference.strip():
            raise ValueError("primary basis provenance must be nonempty")
        if not self.random_reference.strip():
            raise ValueError("random-control basis provenance must be nonempty")
        if not self.locked_before_current_batch:
            raise ValueError("bases must be locked before the current batch")
        if self.uses_current_noise or self.uses_current_returns:
            raise ValueError("current-batch basis construction is forbidden")


@dataclass(frozen=True)
class ClaimMetadata:
    gradient_estimand: str = "current_return_mid_cdf_stop_gradient"
    curvature_estimand: str = (
        "previsible_subspace_current_return_mid_cdf_stop_gradient_hessian"
    )
    endpoint_identity_scope: str = (
        "raw_preprojection_subspace_jacobian_at_proposal_with_frozen_lopo_utilities"
    )
    raw_return_hessian: bool = False
    globally_adaptive_rank_objective_hessian: bool = False
    full_jacobian_estimated: bool = False
    projected_operator_is_endpoint_jacobian: bool = False
    optimizer_confirmation: bool = False
    trajectory_improvement_claim: bool = False
    transition_sample_efficiency_claim: bool = False
    basis_conditioning: str = "fixed_before_current_population"


CLAIM_METADATA = ClaimMetadata()


@dataclass(frozen=True)
class JackknifeEstimate:
    estimate: np.ndarray
    delete_estimates: np.ndarray
    covariance: np.ndarray | None
    standard_error: np.ndarray
    component_labels: tuple[str, ...]


@dataclass(frozen=True)
class LaggedBases:
    primary: np.ndarray
    random_control: np.ndarray
    lagged_block_gradients: np.ndarray
    primary_fallback_blocks: tuple[bool, ...]
    random_uses_primary_fallback_blocks: tuple[bool, ...]
    provenance: BasisProvenance


@dataclass(frozen=True)
class LOPOPopulationEstimate:
    utilities: np.ndarray
    utility_sum: float
    utility_zero_sum_tolerance: float
    gradient: np.ndarray
    curvature: np.ndarray
    random_curvature: np.ndarray
    noise_layout: str


@dataclass(frozen=True)
class LOPOEstimate:
    utilities: np.ndarray
    utility_sum: float
    utility_zero_sum_tolerance: float
    gradient: np.ndarray
    curvature: np.ndarray
    random_curvature: np.ndarray
    gradient_jackknife: JackknifeEstimate
    curvature_vech_jackknife: JackknifeEstimate
    delete_curvatures: np.ndarray
    random_delete_curvatures: np.ndarray
    noise_layout: str


@dataclass(frozen=True)
class StepSet:
    structured: np.ndarray
    isotropic: np.ndarray
    explicit: np.ndarray
    random_raw: np.ndarray
    random: np.ndarray
    endpoints: Mapping[str, np.ndarray]
    curvature_eigenvalues: np.ndarray
    random_curvature_eigenvalues: np.ndarray
    concave_eigenvalues: np.ndarray
    random_concave_eigenvalues: np.ndarray
    multipliers: np.ndarray
    random_multipliers: np.ndarray
    structured_solve_relative_residual: float
    random_solve_relative_residual: float
    isotropic_norm_match_relative_error: float
    random_norm_match_relative_error: float | None
    gradient_direction_defined: bool
    random_direction_defined: bool
    random_control_valid: bool


@dataclass(frozen=True)
class LocalitySummary:
    first: float
    mean: float
    median: float
    percentile_95: float
    maximum: float
    fraction_at_or_below_0_25: float
    fraction_at_or_below_0_5: float
    fraction_at_or_below_1_0: float


@dataclass(frozen=True)
class ActionMetrics:
    anisotropic_action: np.ndarray
    material_fraction: float | None
    structured_explicit_cosine: float | None
    structured_explicit_angle_degrees: float | None
    structured_random_cosine: float | None
    alpha_max_concave_eigenvalue: float
    multiplier_standard_deviation: float
    multiplier_range: float


@dataclass(frozen=True)
class NonlinearJackknife:
    eigenvalues: np.ndarray
    delete_eigenvalues: np.ndarray
    eigenvalue_covariance: np.ndarray
    structured_step: np.ndarray
    delete_structured_steps: np.ndarray
    structured_action_covariance_trace: float
    anisotropic_action: np.ndarray
    delete_anisotropic_actions: np.ndarray
    anisotropic_action_covariance_trace: float
    anisotropic_action_aligned_variance: float | None
    repeated_eigenvalue_unresolved: bool
    projection_boundary_unresolved: bool
    zero_anisotropic_action_unresolved: bool


@dataclass(frozen=True)
class FrozenEndpointDiagnostics:
    delta: np.ndarray
    unnormalized_objective: float
    unnormalized_gradient: np.ndarray
    self_normalized_gradient: np.ndarray
    full_jacobian_action: np.ndarray
    restricted_jacobian_action: np.ndarray
    self_normalized_jacobian_action: np.ndarray
    full_linearization_residual: float | None
    restricted_linearization_residual: float | None
    self_normalized_linearization_residual: float | None
    normalized_ess_ratio: float
    ratio_coefficient_of_variation: float
    mean_unnormalized_ratio_minus_one: float
    log_ratio_span: float


@dataclass(frozen=True)
class ActionReliabilityMetrics:
    reference_action: np.ndarray
    replication_action: np.ndarray
    material_fraction: float | None
    high_sample_relative_disagreement: float | None
    operational_rms_relative_error: float | None
    operational_structured_step_cosines: np.ndarray
    operational_structured_step_relative_errors: np.ndarray
    operational_anisotropic_action_cosines: np.ndarray
    operational_anisotropic_action_relative_errors: np.ndarray


@dataclass(frozen=True)
class CurvatureReliabilityMetrics:
    replication_frobenius_relative_error: float | None
    operational_frobenius_relative_errors: np.ndarray
    reference_negative_eigenvalue_count: int
    replication_negative_eigenvalue_count: int
    operational_negative_eigenvalue_counts: np.ndarray
    replication_eigenvalue_sign_agreement: float
    operational_eigenvalue_sign_agreements: np.ndarray


@dataclass(frozen=True)
class LaggedSubspaceDiagnostic:
    estimate: LOPOEstimate
    steps: StepSet
    locality: Mapping[str, LocalitySummary]
    action_metrics: ActionMetrics
    nonlinear_jackknife: NonlinearJackknife
    endpoint_diagnostics: Mapping[str, FrozenEndpointDiagnostics]
    gradient_endpoint_relative_error: float
    subspace_jacobian_relative_error: float
    self_normalized_gradient_relative_error: float
    self_normalized_jacobian_relative_error: float
    basis_provenance: BasisProvenance
    claim_metadata: ClaimMetadata


def _as_finite_array(name: str, value: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _comparison_sign(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return sign(left - right) without subtracting finite extremes."""

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    return np.greater(left_array, right_array).astype(np.int8) - np.less(
        left_array, right_array
    ).astype(np.int8)


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        _FLOAT_EPS,
    )
    return float(np.linalg.norm(actual - expected) / denominator)


def _stable_norm(vector: np.ndarray) -> float:
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.linalg.norm(vector / scale))


def _optional_cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= _FLOAT_EPS:
        return None
    return float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))


def _validate_basis(name: str, basis: np.ndarray, dimension: int) -> np.ndarray:
    basis = _as_finite_array(name, basis)
    expected_shape = (dimension, SUBSPACE_DIMENSION)
    if basis.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")
    gram = basis.T @ basis
    if not np.allclose(
        gram,
        np.eye(SUBSPACE_DIMENSION),
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError(f"{name} columns must be orthonormal")
    return basis.copy()


def _validated_blocks(
    blocks: Sequence[slice | np.ndarray | Sequence[int]],
    dimension: int,
) -> tuple[np.ndarray, ...]:
    if len(blocks) != SUBSPACE_DIMENSION:
        raise ValueError("exactly three parameter blocks are required")
    validated: list[np.ndarray] = []
    used = np.zeros(dimension, dtype=bool)
    base = np.arange(dimension)
    for index, block in enumerate(blocks):
        if isinstance(block, slice):
            indices = base[block]
        else:
            raw = np.asarray(block)
            if raw.dtype == bool:
                if raw.shape != (dimension,):
                    raise ValueError(f"block {index} boolean mask has wrong shape")
                indices = np.flatnonzero(raw)
            else:
                if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                    raise ValueError(f"block {index} must contain integer indices")
                indices = raw.astype(np.int64, copy=False)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"block {index} is empty")
        if np.any(indices < 0) or np.any(indices >= dimension):
            raise ValueError(f"block {index} contains an out-of-range index")
        if np.unique(indices).size != indices.size:
            raise ValueError(f"block {index} contains duplicate indices")
        if np.any(used[indices]):
            raise ValueError("parameter blocks must be disjoint")
        used[indices] = True
        validated.append(indices.copy())
    if not np.all(used):
        raise ValueError("parameter blocks must partition the full parameter vector")
    return tuple(validated)


def build_lagged_bases(
    lagged_gradients: np.ndarray | Sequence[float],
    blocks: Sequence[slice | np.ndarray | Sequence[int]],
    *,
    primary_fallback_seed: int,
    random_permutation_seed: int,
    primary_reference: str,
    random_reference: str,
) -> LaggedBases:
    """Build the protocol's primary and signed-permutation lagged bases.

    ``lagged_gradients`` is chronological: row zero is ``g[t-10]`` and row
    nine is ``g[t-1]``.  This matches the persisted checkpoint archive.  No
    current population is accepted by this function.

    The protocol does not specify a random-control unit vector when a lagged
    block gradient is exactly zero.  In that case this implementation applies
    the locked signed permutation to the primary locked Gaussian fallback and
    records the event in ``random_uses_primary_fallback_blocks``.
    """

    lagged = _as_finite_array("lagged_gradients", lagged_gradients)
    if lagged.ndim != 2 or lagged.shape[0] != 10:
        raise ValueError("lagged_gradients must have shape (10, d)")
    dimension = lagged.shape[1]
    if dimension < SUBSPACE_DIMENSION:
        raise ValueError("parameter dimension must be at least three")
    validated_blocks = _validated_blocks(blocks, dimension)
    if not isinstance(primary_fallback_seed, (int, np.integer)):
        raise ValueError("primary_fallback_seed must be an integer")
    if not isinstance(random_permutation_seed, (int, np.integer)):
        raise ValueError("random_permutation_seed must be an integer")

    # The newest archived gradient g[t-1] receives weight one; the oldest
    # g[t-10] receives weight 0.9**9.
    weights = 0.9 ** np.arange(9, -1, -1, dtype=np.float64)
    lagged_mean = np.sum(weights[:, None] * lagged, axis=0) / np.sum(weights)
    primary = np.zeros((dimension, SUBSPACE_DIMENSION), dtype=np.float64)
    random_control = np.zeros_like(primary)
    block_gradients = np.zeros_like(primary)
    primary_fallbacks: list[bool] = []
    random_fallbacks: list[bool] = []
    primary_rng = np.random.default_rng(int(primary_fallback_seed))
    random_rng = np.random.default_rng(int(random_permutation_seed))

    for block_index, indices in enumerate(validated_blocks):
        block_gradient = lagged_mean[indices]
        block_gradients[indices, block_index] = block_gradient
        used_fallback = bool(np.all(block_gradient == 0.0))
        if used_fallback:
            primary_values = primary_rng.standard_normal(indices.size)
            fallback_norm = _stable_norm(primary_values)
            if fallback_norm == 0.0:
                raise RuntimeError("Gaussian fallback unexpectedly had zero norm")
            primary_values /= fallback_norm
        else:
            norm = _stable_norm(block_gradient)
            primary_values = block_gradient / norm
        primary[indices, block_index] = primary_values

        permutation = random_rng.permutation(indices.size)
        signs = (
            2.0
            * random_rng.integers(
                0, 2, size=indices.size, dtype=np.int64
            ).astype(np.float64)
            - 1.0
        )
        random_values = signs * primary_values[permutation]
        random_values /= np.linalg.norm(random_values)
        random_control[indices, block_index] = random_values
        primary_fallbacks.append(used_fallback)
        random_fallbacks.append(used_fallback)

    provenance = BasisProvenance.strictly_lagged(
        primary_reference=primary_reference,
        random_reference=random_reference,
    )
    _validate_basis("primary lagged basis", primary, dimension)
    _validate_basis("random-control lagged basis", random_control, dimension)
    return LaggedBases(
        primary=primary,
        random_control=random_control,
        lagged_block_gradients=block_gradients,
        primary_fallback_blocks=tuple(primary_fallbacks),
        random_uses_primary_fallback_blocks=tuple(random_fallbacks),
        provenance=provenance,
    )


def _coerce_pair_noise(
    noise: np.ndarray | Sequence[float],
    pair_count: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    array = _as_finite_array("noise", noise)
    if array.shape == (pair_count, dimension):
        pair_noise = array.copy()
        signed = np.stack((pair_noise, -pair_noise), axis=1)
        return pair_noise, signed, "positive_half_with_constructed_antithetic_mates"
    if array.shape == (pair_count, 2, dimension):
        if not np.array_equal(array[:, 1, :], -array[:, 0, :]):
            raise ValueError("noise pairs must be exactly antithetic")
        return array[:, 0, :].copy(), array.copy(), "paired_interleaved"
    if array.shape == (2 * pair_count, dimension):
        if not np.array_equal(array[pair_count:], -array[:pair_count]):
            raise ValueError("noise halves must be exactly antithetic")
        pair_noise = array[:pair_count].copy()
        signed = np.stack((pair_noise, -pair_noise), axis=1)
        return pair_noise, signed, "positive_then_negative_halves"
    raise ValueError(
        "noise must have shape (m, d), (m, 2, d), or (2m, d)"
    )


def _lopo_utility_numerators(
    paired_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    returns = _as_finite_array("paired_returns", paired_returns)
    if returns.ndim != 2 or returns.shape[1] != 2:
        raise ValueError("paired_returns must have shape (m, 2)")
    pair_count = returns.shape[0]
    if pair_count < 2:
        raise ValueError("LOPO utilities require at least two pairs")

    flat = returns.reshape(-1)
    ordered = np.sort(flat, kind="mergesort")
    less = np.searchsorted(ordered, flat, side="left")
    greater = flat.size - np.searchsorted(ordered, flat, side="right")
    all_comparisons = less - greater
    mate_indices = np.arange(flat.size) ^ 1
    mate_comparisons = _comparison_sign(flat, flat[mate_indices]).astype(np.int64)
    numerators = (all_comparisons - mate_comparisons).reshape(pair_count, 2)
    if int(np.sum(numerators, dtype=np.int64)) != 0:
        raise RuntimeError("LOPO comparison numerators violated structural zero-sum")
    return returns, numerators


def exact_lopo_utilities(
    paired_returns: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return exact tie-aware leave-own-pair-out rank utilities.

    The implementation uses sorted tie groups, not a dense candidate-by-
    candidate comparison matrix.  No target-population recentering is applied.
    """

    returns, numerators = _lopo_utility_numerators(
        np.asarray(paired_returns, dtype=np.float64)
    )
    pair_count = returns.shape[0]
    utilities = numerators.astype(np.float64) / (4.0 * (pair_count - 1))
    tolerance = 1e-12 * max(1.0, float(np.sum(np.abs(utilities))))
    if abs(float(np.sum(utilities))) > tolerance:
        raise RuntimeError("LOPO utilities violated floating-point zero-sum tolerance")
    return utilities


def _vech(matrix: np.ndarray) -> np.ndarray:
    indices = np.tril_indices(SUBSPACE_DIMENSION)
    return np.asarray(matrix[indices], dtype=np.float64)


def _unvech(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    expected = SUBSPACE_DIMENSION * (SUBSPACE_DIMENSION + 1) // 2
    if vector.shape != (expected,):
        raise ValueError(f"vech vector must have shape ({expected},)")
    result = np.zeros((SUBSPACE_DIMENSION, SUBSPACE_DIMENSION), dtype=np.float64)
    indices = np.tril_indices(SUBSPACE_DIMENSION)
    result[indices] = vector
    result[(indices[1], indices[0])] = vector
    return result


def _unvech_rows(vectors: np.ndarray) -> np.ndarray:
    return np.stack([_unvech(row) for row in np.asarray(vectors)], axis=0)


def _jackknife_from_kernel_row_sums(
    row_sums: np.ndarray,
    labels: tuple[str, ...],
    expected_estimate: np.ndarray | None = None,
    *,
    full_covariance: bool = True,
) -> JackknifeEstimate:
    row_sums = _as_finite_array("kernel row sums", row_sums)
    if row_sums.ndim != 2:
        raise ValueError("kernel row sums must be a matrix")
    pair_count, components = row_sums.shape
    if pair_count < 3:
        raise ValueError("delete-pair jackknife requires at least three pairs")
    if components != len(labels):
        raise ValueError("jackknife component labels do not match row sums")

    total = 0.5 * np.sum(row_sums, axis=0)
    pair_denominator = pair_count * (pair_count - 1) / 2.0
    delete_denominator = (pair_count - 1) * (pair_count - 2) / 2.0
    estimate = total / pair_denominator
    delete_estimates = (total[None, :] - row_sums) / delete_denominator
    delete_mean = np.mean(delete_estimates, axis=0)
    centered = delete_estimates - delete_mean
    variance = ((pair_count - 1.0) / pair_count) * np.sum(centered**2, axis=0)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    if full_covariance:
        covariance = ((pair_count - 1.0) / pair_count) * (
            centered.T @ centered
        )
        covariance = 0.5 * (covariance + covariance.T)
    else:
        covariance = None

    if expected_estimate is not None:
        expected = np.asarray(expected_estimate, dtype=np.float64)
        if not np.allclose(estimate, expected, rtol=5e-11, atol=5e-13):
            raise RuntimeError("direct LOPO estimate disagrees with U-statistic kernel")
        estimate = expected.copy()
    return JackknifeEstimate(
        estimate=estimate,
        delete_estimates=delete_estimates,
        covariance=covariance,
        standard_error=standard_error,
        component_labels=labels,
    )


def _paired_weighted_sign_query_sums(
    paired_returns: np.ndarray,
    pair_noise: np.ndarray,
) -> np.ndarray:
    """Sum the two weighted sign queries for every antithetic pair.

    Candidate events have weights ``+epsilon_k`` and ``-epsilon_k``.  Their
    total is structurally zero, so tie-group prefix sums can accumulate the
    two query results directly into an ``m x d`` matrix.  This avoids all
    ``2m x d`` query/weight matrices and every ``m x m x d`` object.
    """

    pair_count, dimension = pair_noise.shape
    values = paired_returns.reshape(-1)
    pair_indices = np.repeat(np.arange(pair_count), 2)
    signs = np.tile(np.asarray([1.0, -1.0]), pair_count)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_pairs = pair_indices[order]
    sorted_signs = signs[order]
    boundaries = np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([values.size])))

    prefix = np.zeros(dimension, dtype=np.float64)
    query_sums = np.zeros((pair_count, dimension), dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        group_pairs = sorted_pairs[start:end]
        group_signs = sorted_signs[start:end]
        group_weight = np.sum(
            group_signs[:, None] * pair_noise[group_pairs], axis=0
        )
        # greater - less with a structurally zero total event weight.
        query = -2.0 * prefix - group_weight
        unique_pairs, counts = np.unique(group_pairs, return_counts=True)
        query_sums[unique_pairs] += counts[:, None] * query
        prefix += group_weight
    return query_sums


def gradient_u_statistic_row_sums(
    pair_noise: np.ndarray,
    paired_returns: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Return full-gradient order-two kernel row sums without m^2*d state."""

    pair_noise = _as_finite_array("pair_noise", pair_noise)
    returns, numerators = _lopo_utility_numerators(paired_returns)
    pair_count = returns.shape[0]
    if pair_noise.ndim != 2 or pair_noise.shape[0] != pair_count:
        raise ValueError("pair_noise must have shape (m, d)")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")

    own_pair_kernel = 2.0 * _comparison_sign(returns[:, 0], returns[:, 1])
    incoming = _paired_weighted_sign_query_sums(returns, pair_noise)
    incoming -= own_pair_kernel[:, None] * pair_noise
    outgoing = numerators[:, 0] - numerators[:, 1]
    row_sums = (
        outgoing[:, None] * pair_noise + incoming
    ) / (16.0 * sigma)
    return row_sums


def _pair_comparison_sums(paired_returns: np.ndarray) -> np.ndarray:
    returns = _as_finite_array("paired_returns", paired_returns)
    pair_count = returns.shape[0]
    comparison = np.zeros((pair_count, pair_count), dtype=np.int8)
    for first_sign in range(2):
        left = returns[:, first_sign, None]
        for second_sign in range(2):
            right = returns[None, :, second_sign]
            term = np.greater(left, right).astype(np.int8)
            term -= np.less(left, right).astype(np.int8)
            comparison += term
    np.fill_diagonal(comparison, 0)
    if not np.array_equal(comparison, -comparison.T):
        raise RuntimeError("pair comparison matrix is not antisymmetric")
    return comparison


def _curvature_kernel_row_sums(
    pair_noise: np.ndarray,
    paired_returns: np.ndarray,
    basis: np.ndarray,
    sigma: float,
    comparison: np.ndarray | None = None,
) -> np.ndarray:
    projected = pair_noise @ basis
    score = np.einsum("mi,mj->mij", projected, projected, optimize=True)
    score -= np.eye(SUBSPACE_DIMENSION)[None, :, :]
    score_flat = score.reshape(score.shape[0], -1)
    if comparison is None:
        comparison = _pair_comparison_sums(paired_returns)
    row_total = np.sum(comparison, axis=1, dtype=np.int64).astype(np.float64)
    weighted_other = comparison.astype(np.float64) @ score_flat
    row_sums = (
        score_flat * row_total[:, None] - weighted_other
    ) / (16.0 * sigma**2)
    return row_sums.reshape(-1, SUBSPACE_DIMENSION, SUBSPACE_DIMENSION)


def _direct_lopo_estimate(
    pair_noise: np.ndarray,
    paired_returns: np.ndarray,
    basis: np.ndarray,
    random_basis: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    utilities = exact_lopo_utilities(paired_returns)
    pair_count = pair_noise.shape[0]
    pair_difference = utilities[:, 0] - utilities[:, 1]
    gradient = np.sum(pair_difference[:, None] * pair_noise, axis=0)
    gradient /= 2.0 * pair_count * sigma

    def curvature_for(selected_basis: np.ndarray) -> np.ndarray:
        projected = pair_noise @ selected_basis
        score = np.einsum("mi,mj->mij", projected, projected, optimize=True)
        score -= np.eye(SUBSPACE_DIMENSION)[None, :, :]
        pair_sum = utilities[:, 0] + utilities[:, 1]
        matrix = np.sum(pair_sum[:, None, None] * score, axis=0)
        matrix /= 2.0 * pair_count * sigma**2
        return 0.5 * (matrix + matrix.T)

    return utilities, gradient, curvature_for(basis), curvature_for(random_basis)


def estimate_lopo_population(
    theta: np.ndarray | Sequence[float],
    noise: np.ndarray | Sequence[float],
    paired_returns: np.ndarray | Sequence[float],
    sigma: float,
    basis: np.ndarray | Sequence[float],
    random_basis: np.ndarray | Sequence[float],
    *,
    basis_provenance: BasisProvenance,
) -> LOPOPopulationEstimate:
    """Compute the O(m*d) direct population gradient and two 3x3 moments."""

    theta = _as_finite_array("theta", theta)
    if theta.ndim != 1 or theta.size < SUBSPACE_DIMENSION:
        raise ValueError("theta must be a vector with dimension at least three")
    returns = _as_finite_array("paired_returns", paired_returns)
    if returns.ndim != 2 or returns.shape[1] != 2 or returns.shape[0] < 2:
        raise ValueError("paired_returns must have shape (m, 2) with m >= 2")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    basis_provenance.validate()
    basis = _validate_basis("basis", np.asarray(basis), theta.size)
    random_basis = _validate_basis(
        "random_basis", np.asarray(random_basis), theta.size
    )
    pair_noise, _, layout = _coerce_pair_noise(
        np.asarray(noise), returns.shape[0], theta.size
    )
    utilities, gradient, curvature, random_curvature = _direct_lopo_estimate(
        pair_noise, returns, basis, random_basis, sigma
    )
    utility_sum = float(np.sum(utilities))
    zero_tolerance = 1e-12 * max(1.0, float(np.sum(np.abs(utilities))))
    return LOPOPopulationEstimate(
        utilities=utilities,
        utility_sum=utility_sum,
        utility_zero_sum_tolerance=zero_tolerance,
        gradient=gradient,
        curvature=curvature,
        random_curvature=random_curvature,
        noise_layout=layout,
    )


def estimate_lopo_with_jackknife(
    theta: np.ndarray | Sequence[float],
    noise: np.ndarray | Sequence[float],
    paired_returns: np.ndarray | Sequence[float],
    sigma: float,
    basis: np.ndarray | Sequence[float],
    random_basis: np.ndarray | Sequence[float],
    *,
    basis_provenance: BasisProvenance,
) -> LOPOEstimate:
    """Estimate matched LOPO gradient/subspace curvature and delete jackknives."""

    population = estimate_lopo_population(
        theta,
        noise,
        paired_returns,
        sigma,
        basis,
        random_basis,
        basis_provenance=basis_provenance,
    )
    theta = _as_finite_array("theta", theta)
    returns = _as_finite_array("paired_returns", paired_returns)
    if returns.shape[0] < 3:
        raise ValueError("paired_returns must have shape (m, 2) with m >= 3")
    basis = _validate_basis("basis", np.asarray(basis), theta.size)
    random_basis = _validate_basis(
        "random_basis", np.asarray(random_basis), theta.size
    )
    pair_noise, _, _ = _coerce_pair_noise(
        np.asarray(noise), returns.shape[0], theta.size
    )

    utilities = population.utilities
    gradient = population.gradient
    curvature = population.curvature
    random_curvature = population.random_curvature

    gradient_rows = gradient_u_statistic_row_sums(pair_noise, returns, sigma)
    gradient_labels = tuple(f"g[{index}]" for index in range(theta.size))
    gradient_jackknife = _jackknife_from_kernel_row_sums(
        gradient_rows,
        gradient_labels,
        expected_estimate=gradient,
        full_covariance=False,
    )

    comparison = _pair_comparison_sums(returns)
    curvature_rows = _curvature_kernel_row_sums(
        pair_noise, returns, basis, sigma, comparison
    )
    curvature_vech_rows = np.stack([_vech(row) for row in curvature_rows])
    curvature_labels = (
        "B[0,0]",
        "B[1,0]",
        "B[1,1]",
        "B[2,0]",
        "B[2,1]",
        "B[2,2]",
    )
    curvature_jackknife = _jackknife_from_kernel_row_sums(
        curvature_vech_rows,
        curvature_labels,
        expected_estimate=_vech(curvature),
    )
    delete_curvatures = _unvech_rows(curvature_jackknife.delete_estimates)

    random_rows = _curvature_kernel_row_sums(
        pair_noise, returns, random_basis, sigma, comparison
    )
    random_vech_rows = np.stack([_vech(row) for row in random_rows])
    random_jackknife = _jackknife_from_kernel_row_sums(
        random_vech_rows,
        curvature_labels,
        expected_estimate=_vech(random_curvature),
    )
    random_delete_curvatures = _unvech_rows(random_jackknife.delete_estimates)

    return LOPOEstimate(
        utilities=utilities,
        utility_sum=population.utility_sum,
        utility_zero_sum_tolerance=population.utility_zero_sum_tolerance,
        gradient=gradient,
        curvature=curvature,
        random_curvature=random_curvature,
        gradient_jackknife=gradient_jackknife,
        curvature_vech_jackknife=curvature_jackknife,
        delete_curvatures=delete_curvatures,
        random_delete_curvatures=random_delete_curvatures,
        noise_layout=population.noise_layout,
    )


def calibrate_locality_rate(
    gradient: np.ndarray | Sequence[float],
    sigma: float,
    q: float,
) -> float:
    gradient = _as_finite_array("gradient", gradient)
    if gradient.ndim != 1:
        raise ValueError("gradient must be a vector")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not np.isfinite(q) or q <= 0.0:
        raise ValueError("q must be finite and positive")
    norm = _stable_norm(gradient)
    if norm == 0.0:
        raise UnresolvedDiagnosticError("cannot calibrate locality from zero gradient")
    rate = float(q * sigma / norm)
    if not np.isfinite(rate):
        raise FloatingPointError("locality calibration produced a nonfinite rate")
    return rate


def _structured_step(
    gradient: np.ndarray,
    curvature: np.ndarray,
    basis: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(curvature)
    concave = np.maximum(-eigenvalues, 0.0)
    multipliers = 1.0 / (1.0 + alpha * concave)
    projected_gradient = basis.T @ gradient
    solved_coordinates = eigenvectors @ (
        multipliers * (eigenvectors.T @ projected_gradient)
    )
    perpendicular = gradient - basis @ projected_gradient
    step = alpha * (perpendicular + basis @ solved_coordinates)

    system = np.eye(SUBSPACE_DIMENSION)
    system += alpha * (eigenvectors * concave[None, :]) @ eigenvectors.T
    residual = system @ solved_coordinates - projected_gradient
    residual_denominator = max(
        float(np.linalg.norm(projected_gradient)), _FLOAT_EPS
    )
    relative_residual = float(np.linalg.norm(residual) / residual_denominator)
    return step, eigenvalues, concave, multipliers, relative_residual


def compute_four_steps(
    theta: np.ndarray | Sequence[float],
    gradient: np.ndarray | Sequence[float],
    curvature: np.ndarray | Sequence[float],
    basis: np.ndarray | Sequence[float],
    random_curvature: np.ndarray | Sequence[float],
    random_basis: np.ndarray | Sequence[float],
    alpha: float,
) -> StepSet:
    theta = _as_finite_array("theta", theta)
    gradient = _as_finite_array("gradient", gradient)
    if theta.ndim != 1 or gradient.shape != theta.shape:
        raise ValueError("theta and gradient must be same-shaped vectors")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    basis = _validate_basis("basis", np.asarray(basis), theta.size)
    random_basis = _validate_basis(
        "random_basis", np.asarray(random_basis), theta.size
    )
    curvature = _as_finite_array("curvature", curvature)
    random_curvature = _as_finite_array("random_curvature", random_curvature)
    expected_shape = (SUBSPACE_DIMENSION, SUBSPACE_DIMENSION)
    if (
        curvature.shape != expected_shape
        or random_curvature.shape != expected_shape
    ):
        raise ValueError("curvature matrices must have shape (3, 3)")
    if not np.allclose(curvature, curvature.T, rtol=1e-12, atol=1e-14):
        raise ValueError("curvature must be symmetric")
    if not np.allclose(
        random_curvature, random_curvature.T, rtol=1e-12, atol=1e-14
    ):
        raise ValueError("random_curvature must be symmetric")

    structured, eigenvalues, concave, multipliers, residual = _structured_step(
        gradient, curvature, basis, alpha
    )
    (
        random_raw,
        random_eigenvalues,
        random_concave,
        random_multipliers,
        random_residual,
    ) = _structured_step(gradient, random_curvature, random_basis, alpha)
    explicit = alpha * gradient
    structured_norm = float(np.linalg.norm(structured))
    gradient_norm = float(np.linalg.norm(gradient))
    random_raw_norm = float(np.linalg.norm(random_raw))

    if gradient_norm <= _FLOAT_EPS:
        isotropic = np.zeros_like(gradient)
        gradient_direction_defined = False
    else:
        isotropic = structured_norm * gradient / gradient_norm
        gradient_direction_defined = True
    isotropic_error = abs(float(np.linalg.norm(isotropic)) - structured_norm)
    isotropic_error /= max(structured_norm, _FLOAT_EPS)

    if random_raw_norm <= _FLOAT_EPS:
        random = np.zeros_like(random_raw)
        random_direction_defined = False
        random_control_valid = bool(structured_norm <= _FLOAT_EPS)
        random_error = 0.0 if random_control_valid else None
    else:
        random = structured_norm * random_raw / random_raw_norm
        random_direction_defined = True
        random_control_valid = True
        random_error = abs(float(np.linalg.norm(random)) - structured_norm)
        random_error /= max(structured_norm, _FLOAT_EPS)

    vectors = {
        "structured": structured,
        "isotropic": isotropic,
        "explicit": explicit,
        "random": random,
    }
    endpoints = {name: theta + step for name, step in vectors.items()}
    return StepSet(
        structured=structured,
        isotropic=isotropic,
        explicit=explicit,
        random_raw=random_raw,
        random=random,
        endpoints=endpoints,
        curvature_eigenvalues=eigenvalues,
        random_curvature_eigenvalues=random_eigenvalues,
        concave_eigenvalues=concave,
        random_concave_eigenvalues=random_concave,
        multipliers=multipliers,
        random_multipliers=random_multipliers,
        structured_solve_relative_residual=residual,
        random_solve_relative_residual=random_residual,
        isotropic_norm_match_relative_error=float(isotropic_error),
        random_norm_match_relative_error=(
            None if random_error is None else float(random_error)
        ),
        gradient_direction_defined=gradient_direction_defined,
        random_direction_defined=random_direction_defined,
        random_control_valid=random_control_valid,
    )


def summarize_locality(
    steps: np.ndarray | Sequence[float],
    sigma: float,
) -> LocalitySummary:
    steps = _as_finite_array("steps", steps)
    if steps.ndim == 1:
        steps = steps[None, :]
    if steps.ndim != 2 or steps.shape[0] == 0:
        raise ValueError("steps must be a nonempty vector or matrix")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    ratios = np.linalg.norm(steps, axis=1) / sigma
    return LocalitySummary(
        first=float(ratios[0]),
        mean=float(np.mean(ratios)),
        median=float(np.median(ratios)),
        percentile_95=float(np.percentile(ratios, 95.0, method="linear")),
        maximum=float(np.max(ratios)),
        fraction_at_or_below_0_25=float(np.mean(ratios <= 0.25)),
        fraction_at_or_below_0_5=float(np.mean(ratios <= 0.5)),
        fraction_at_or_below_1_0=float(np.mean(ratios <= 1.0)),
    )


def compute_action_metrics(steps: StepSet, alpha: float) -> ActionMetrics:
    action = steps.structured - steps.isotropic
    structured_norm = float(np.linalg.norm(steps.structured))
    material = (
        None
        if structured_norm <= _FLOAT_EPS
        else float(np.linalg.norm(action) / structured_norm)
    )
    explicit_cosine = _optional_cosine(steps.structured, steps.explicit)
    angle = (
        None
        if explicit_cosine is None
        else float(np.degrees(np.arccos(explicit_cosine)))
    )
    random_cosine = _optional_cosine(steps.structured, steps.random)
    return ActionMetrics(
        anisotropic_action=action,
        material_fraction=material,
        structured_explicit_cosine=explicit_cosine,
        structured_explicit_angle_degrees=angle,
        structured_random_cosine=random_cosine,
        alpha_max_concave_eigenvalue=float(
            alpha * np.max(steps.concave_eigenvalues)
        ),
        multiplier_standard_deviation=float(np.std(steps.multipliers)),
        multiplier_range=float(np.ptp(steps.multipliers)),
    )


def _jackknife_covariance(values: np.ndarray) -> np.ndarray:
    values = _as_finite_array("jackknife values", values)
    count = values.shape[0]
    centered = values - np.mean(values, axis=0)
    flat = centered.reshape(count, -1)
    return ((count - 1.0) / count) * (flat.T @ flat)


def recompute_eigen_action_jackknife(
    estimate: LOPOEstimate,
    theta: np.ndarray | Sequence[float],
    basis: np.ndarray | Sequence[float],
    random_basis: np.ndarray | Sequence[float],
    alpha: float,
    *,
    degeneracy_tolerance: float = 1e-10,
) -> NonlinearJackknife:
    theta = _as_finite_array("theta", theta)
    basis = _validate_basis("basis", np.asarray(basis), theta.size)
    random_basis = _validate_basis(
        "random_basis", np.asarray(random_basis), theta.size
    )
    full_steps = compute_four_steps(
        theta,
        estimate.gradient,
        estimate.curvature,
        basis,
        estimate.random_curvature,
        random_basis,
        alpha,
    )
    delete_gradients = estimate.gradient_jackknife.delete_estimates
    delete_steps: list[np.ndarray] = []
    delete_actions: list[np.ndarray] = []
    delete_eigenvalues: list[np.ndarray] = []
    for index in range(delete_gradients.shape[0]):
        current = compute_four_steps(
            theta,
            delete_gradients[index],
            estimate.delete_curvatures[index],
            basis,
            estimate.random_delete_curvatures[index],
            random_basis,
            alpha,
        )
        delete_steps.append(current.structured)
        delete_actions.append(current.structured - current.isotropic)
        delete_eigenvalues.append(current.curvature_eigenvalues)
    step_array = np.stack(delete_steps)
    action_array = np.stack(delete_actions)
    eigen_array = np.stack(delete_eigenvalues)
    eigen_covariance = _jackknife_covariance(eigen_array)
    count = step_array.shape[0]
    factor = (count - 1.0) / count
    centered_steps = step_array - np.mean(step_array, axis=0)
    centered_actions = action_array - np.mean(action_array, axis=0)
    step_covariance_trace = float(factor * np.sum(centered_steps**2))
    action_covariance_trace = float(factor * np.sum(centered_actions**2))

    anisotropic_action = full_steps.structured - full_steps.isotropic
    action_norm = float(np.linalg.norm(anisotropic_action))
    if action_norm <= degeneracy_tolerance:
        aligned_variance = None
        zero_action = True
    else:
        direction = anisotropic_action / action_norm
        projections = centered_actions @ direction
        aligned_variance = float(
            ((count - 1.0) / count) * np.dot(projections, projections)
        )
        zero_action = False

    eigenvalues = full_steps.curvature_eigenvalues
    gaps = np.diff(np.sort(eigenvalues))
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    repeated = bool(np.any(np.abs(gaps) <= degeneracy_tolerance * scale))
    boundary = bool(
        np.any(np.abs(eigenvalues) <= degeneracy_tolerance * scale)
    )
    return NonlinearJackknife(
        eigenvalues=eigenvalues,
        delete_eigenvalues=eigen_array,
        eigenvalue_covariance=eigen_covariance,
        structured_step=full_steps.structured,
        delete_structured_steps=step_array,
        structured_action_covariance_trace=step_covariance_trace,
        anisotropic_action=anisotropic_action,
        delete_anisotropic_actions=action_array,
        anisotropic_action_covariance_trace=action_covariance_trace,
        anisotropic_action_aligned_variance=aligned_variance,
        repeated_eigenvalue_unresolved=repeated,
        projection_boundary_unresolved=boundary,
        zero_anisotropic_action_unresolved=zero_action,
    )


def _validate_signed_noise_utilities(
    signed_noise: np.ndarray,
    utilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    signed_noise = _as_finite_array("signed_noise", signed_noise)
    utilities = _as_finite_array("utilities", utilities)
    if signed_noise.ndim != 2 or utilities.shape != (signed_noise.shape[0],):
        raise ValueError("signed_noise and utilities have inconsistent shapes")
    if signed_noise.shape[0] % 2 != 0:
        raise ValueError("signed_noise must contain complete antithetic pairs")
    if not np.array_equal(signed_noise[1::2], -signed_noise[0::2]):
        raise ValueError("signed_noise must contain exact interleaved antithetic pairs")
    return signed_noise, utilities


def frozen_endpoint_gradient(
    signed_noise: np.ndarray | Sequence[float],
    utilities: np.ndarray | Sequence[float],
    delta: np.ndarray | Sequence[float],
    sigma: float,
    *,
    self_normalized: bool = False,
) -> np.ndarray:
    signed_noise, utilities = _validate_signed_noise_utilities(
        np.asarray(signed_noise), np.asarray(utilities)
    )
    delta = _as_finite_array("delta", delta)
    if delta.shape != (signed_noise.shape[1],):
        raise ValueError("delta has the wrong dimension")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    log_ratio = signed_noise @ delta / sigma
    log_ratio -= float(np.dot(delta, delta)) / (2.0 * sigma**2)

    if self_normalized:
        shifted = log_ratio - float(np.max(log_ratio))
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        utility_mean = float(np.dot(weights, utilities))
        centered = utilities - utility_mean
        return np.sum(
            weights[:, None]
            * centered[:, None]
            * (signed_noise - delta[None, :] / sigma),
            axis=0,
        ) / sigma

    with np.errstate(over="raise", invalid="raise"):
        try:
            ratio = np.exp(log_ratio)
        except FloatingPointError as error:
            raise FloatingPointError(
                "unnormalized frozen endpoint ratio is nonfinite"
            ) from error
    return np.mean(
        (utilities * ratio)[:, None]
        * (signed_noise - delta[None, :] / sigma),
        axis=0,
    ) / sigma


def frozen_endpoint_objective(
    signed_noise: np.ndarray | Sequence[float],
    utilities: np.ndarray | Sequence[float],
    delta: np.ndarray | Sequence[float],
    sigma: float,
) -> float:
    signed_noise, utilities = _validate_signed_noise_utilities(
        np.asarray(signed_noise), np.asarray(utilities)
    )
    delta = _as_finite_array("delta", delta)
    if delta.shape != (signed_noise.shape[1],):
        raise ValueError("delta has the wrong dimension")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    log_ratio = signed_noise @ delta / sigma
    log_ratio -= float(np.dot(delta, delta)) / (2.0 * sigma**2)
    with np.errstate(over="raise", invalid="raise"):
        try:
            ratio = np.exp(log_ratio)
        except FloatingPointError as error:
            raise FloatingPointError(
                "unnormalized frozen endpoint ratio is nonfinite"
            ) from error
    return float(np.mean(utilities * ratio))


def frozen_endpoint_jacobians(
    signed_noise: np.ndarray | Sequence[float],
    utilities: np.ndarray | Sequence[float],
    sigma: float,
    *,
    max_dimension: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full proposal Jacobians for small-dimensional validation.

    Production diagnostics should use :func:`frozen_endpoint_jacobian_actions`,
    which remains O(m*d).  The explicit guard prevents an accidental dense
    ``d x d`` allocation on policy-sized vectors.
    """

    signed_noise, utilities = _validate_signed_noise_utilities(
        np.asarray(signed_noise), np.asarray(utilities)
    )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if not isinstance(max_dimension, (int, np.integer)) or max_dimension < 1:
        raise ValueError("max_dimension must be a positive integer")
    if signed_noise.shape[1] > int(max_dimension):
        raise ValueError(
            "full Jacobian exceeds max_dimension; use Jacobian actions instead"
        )
    mean_noise = np.mean(signed_noise, axis=0)
    if float(np.linalg.norm(mean_noise)) > 1e-13:
        raise ValueError("self-normalized Jacobian formula requires antithetic mean zero")
    second = np.einsum(
        "n,ni,nj->ij", utilities, signed_noise, signed_noise, optimize=True
    ) / signed_noise.shape[0]
    utility_mean = float(np.mean(utilities))
    noise_second = signed_noise.T @ signed_noise / signed_noise.shape[0]
    identity = np.eye(signed_noise.shape[1])
    unnormalized = (second - utility_mean * identity) / sigma**2
    self_normalized = (second - utility_mean * noise_second) / sigma**2
    return (
        0.5 * (unnormalized + unnormalized.T),
        0.5 * (self_normalized + self_normalized.T),
    )


def frozen_endpoint_jacobian_actions(
    signed_noise: np.ndarray | Sequence[float],
    utilities: np.ndarray | Sequence[float],
    delta: np.ndarray | Sequence[float],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    signed_noise, utilities = _validate_signed_noise_utilities(
        np.asarray(signed_noise), np.asarray(utilities)
    )
    delta = _as_finite_array("delta", delta)
    if delta.shape != (signed_noise.shape[1],):
        raise ValueError("delta has the wrong dimension")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    projections = signed_noise @ delta
    utility_mean = float(np.mean(utilities))
    first = np.mean(
        utilities[:, None] * signed_noise * projections[:, None], axis=0
    )
    unnormalized = (first - utility_mean * delta) / sigma**2
    noise_action = np.mean(
        signed_noise * projections[:, None], axis=0
    )
    self_normalized = (first - utility_mean * noise_action) / sigma**2
    return unnormalized, self_normalized


def _linearization_residual(
    endpoint_difference: np.ndarray,
    linear_action: np.ndarray,
) -> float | None:
    denominator = max(
        float(np.linalg.norm(endpoint_difference)),
        float(np.linalg.norm(linear_action)),
    )
    if denominator <= _FLOAT_EPS:
        return None
    return float(np.linalg.norm(endpoint_difference - linear_action) / denominator)


def frozen_endpoint_diagnostics(
    signed_noise: np.ndarray | Sequence[float],
    utilities: np.ndarray | Sequence[float],
    delta: np.ndarray | Sequence[float],
    sigma: float,
    basis: np.ndarray | Sequence[float],
    subspace_curvature: np.ndarray | Sequence[float],
) -> FrozenEndpointDiagnostics:
    signed_noise, utilities = _validate_signed_noise_utilities(
        np.asarray(signed_noise), np.asarray(utilities)
    )
    delta = _as_finite_array("delta", delta)
    basis = _validate_basis("basis", np.asarray(basis), signed_noise.shape[1])
    subspace_curvature = _as_finite_array(
        "subspace_curvature", subspace_curvature
    )
    if subspace_curvature.shape != (SUBSPACE_DIMENSION, SUBSPACE_DIMENSION):
        raise ValueError("subspace_curvature must have shape (3, 3)")
    if not np.allclose(
        subspace_curvature,
        subspace_curvature.T,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("subspace_curvature must be symmetric")

    zero = np.zeros_like(delta)
    gradient_zero = frozen_endpoint_gradient(
        signed_noise, utilities, zero, sigma, self_normalized=False
    )
    gradient_delta = frozen_endpoint_gradient(
        signed_noise, utilities, delta, sigma, self_normalized=False
    )
    sn_zero = frozen_endpoint_gradient(
        signed_noise, utilities, zero, sigma, self_normalized=True
    )
    sn_delta = frozen_endpoint_gradient(
        signed_noise, utilities, delta, sigma, self_normalized=True
    )
    full_action, sn_action = frozen_endpoint_jacobian_actions(
        signed_noise, utilities, delta, sigma
    )
    restricted_action = basis @ (subspace_curvature @ (basis.T @ delta))
    endpoint_difference = gradient_delta - gradient_zero
    sn_difference = sn_delta - sn_zero

    log_ratio = signed_noise @ delta / sigma
    log_ratio -= float(np.dot(delta, delta)) / (2.0 * sigma**2)
    with np.errstate(over="raise", invalid="raise"):
        try:
            ratio = np.exp(log_ratio)
        except FloatingPointError as error:
            raise FloatingPointError(
                "unnormalized frozen endpoint ratio is nonfinite"
            ) from error
    shifted = log_ratio - float(np.max(log_ratio))
    normalized = np.exp(shifted)
    normalized /= np.sum(normalized)
    count = normalized.size
    ess_ratio = float(1.0 / (count * np.dot(normalized, normalized)))
    ratio_cv = float(np.sqrt(max(count * np.dot(normalized, normalized) - 1.0, 0.0)))

    return FrozenEndpointDiagnostics(
        delta=delta,
        unnormalized_objective=frozen_endpoint_objective(
            signed_noise, utilities, delta, sigma
        ),
        unnormalized_gradient=gradient_delta,
        self_normalized_gradient=sn_delta,
        full_jacobian_action=full_action,
        restricted_jacobian_action=restricted_action,
        self_normalized_jacobian_action=sn_action,
        full_linearization_residual=_linearization_residual(
            endpoint_difference, full_action
        ),
        restricted_linearization_residual=_linearization_residual(
            endpoint_difference, restricted_action
        ),
        self_normalized_linearization_residual=_linearization_residual(
            sn_difference, sn_action
        ),
        normalized_ess_ratio=ess_ratio,
        ratio_coefficient_of_variation=ratio_cv,
        mean_unnormalized_ratio_minus_one=float(np.mean(ratio) - 1.0),
        log_ratio_span=float(np.ptp(log_ratio)),
    )


def compare_anisotropic_actions(
    reference_structured: np.ndarray | Sequence[float],
    reference_isotropic: np.ndarray | Sequence[float],
    replication_structured: np.ndarray | Sequence[float],
    replication_isotropic: np.ndarray | Sequence[float],
    operational_structured: np.ndarray | Sequence[float],
    operational_isotropic: np.ndarray | Sequence[float],
) -> ActionReliabilityMetrics:
    reference_structured = _as_finite_array(
        "reference_structured", reference_structured
    )
    reference_isotropic = _as_finite_array(
        "reference_isotropic", reference_isotropic
    )
    replication_structured = _as_finite_array(
        "replication_structured", replication_structured
    )
    replication_isotropic = _as_finite_array(
        "replication_isotropic", replication_isotropic
    )
    operational_structured = _as_finite_array(
        "operational_structured", operational_structured
    )
    operational_isotropic = _as_finite_array(
        "operational_isotropic", operational_isotropic
    )
    dimension = reference_structured.shape
    for name, array in (
        ("reference_isotropic", reference_isotropic),
        ("replication_structured", replication_structured),
        ("replication_isotropic", replication_isotropic),
    ):
        if array.shape != dimension:
            raise ValueError(f"{name} has the wrong shape")
    if operational_structured.ndim == 1:
        operational_structured = operational_structured[None, :]
    if operational_isotropic.ndim == 1:
        operational_isotropic = operational_isotropic[None, :]
    if (
        operational_structured.shape != operational_isotropic.shape
        or operational_structured.shape[1:] != dimension
        or operational_structured.shape[0] == 0
    ):
        raise ValueError("operational action arrays have inconsistent shapes")

    reference_action = reference_structured - reference_isotropic
    replication_action = replication_structured - replication_isotropic
    operational_actions = operational_structured - operational_isotropic
    reference_structured_norm = float(np.linalg.norm(reference_structured))
    reference_action_norm = float(np.linalg.norm(reference_action))
    replication_action_norm = float(np.linalg.norm(replication_action))
    material = (
        None
        if reference_structured_norm <= _FLOAT_EPS
        else reference_action_norm / reference_structured_norm
    )
    high_denominator = 0.5 * (reference_action_norm + replication_action_norm)
    high_error = (
        None
        if high_denominator <= _FLOAT_EPS
        else float(
            np.linalg.norm(reference_action - replication_action)
            / high_denominator
        )
    )
    operational_error = (
        None
        if reference_action_norm <= _FLOAT_EPS
        else float(
            np.sqrt(
                np.mean(
                    np.sum((operational_actions - reference_action) ** 2, axis=1)
                )
            )
            / reference_action_norm
        )
    )
    step_cosines = np.asarray(
        [
            np.nan
            if _optional_cosine(step, reference_structured) is None
            else _optional_cosine(step, reference_structured)
            for step in operational_structured
        ],
        dtype=np.float64,
    )
    reference_norm = float(np.linalg.norm(reference_structured))
    step_relative_errors = np.full(operational_structured.shape[0], np.nan)
    if reference_norm > _FLOAT_EPS:
        step_relative_errors = (
            np.linalg.norm(
                operational_structured - reference_structured[None, :], axis=1
            )
            / reference_norm
        )
    action_cosines = np.asarray(
        [
            np.nan
            if _optional_cosine(action, reference_action) is None
            else _optional_cosine(action, reference_action)
            for action in operational_actions
        ],
        dtype=np.float64,
    )
    action_relative_errors = np.full(operational_actions.shape[0], np.nan)
    if reference_action_norm > _FLOAT_EPS:
        action_relative_errors = (
            np.linalg.norm(
                operational_actions - reference_action[None, :], axis=1
            )
            / reference_action_norm
        )
    return ActionReliabilityMetrics(
        reference_action=reference_action,
        replication_action=replication_action,
        material_fraction=(None if material is None else float(material)),
        high_sample_relative_disagreement=high_error,
        operational_rms_relative_error=operational_error,
        operational_structured_step_cosines=step_cosines,
        operational_structured_step_relative_errors=step_relative_errors,
        operational_anisotropic_action_cosines=action_cosines,
        operational_anisotropic_action_relative_errors=action_relative_errors,
    )


def compare_subspace_curvatures(
    reference_curvature: np.ndarray | Sequence[float],
    replication_curvature: np.ndarray | Sequence[float],
    operational_curvatures: np.ndarray | Sequence[float],
) -> CurvatureReliabilityMetrics:
    """Compare independent/operational 3x3 moments with a fixed reference."""

    reference = _as_finite_array("reference_curvature", reference_curvature)
    replication = _as_finite_array(
        "replication_curvature", replication_curvature
    )
    operational = _as_finite_array(
        "operational_curvatures", operational_curvatures
    )
    expected = (SUBSPACE_DIMENSION, SUBSPACE_DIMENSION)
    if reference.shape != expected or replication.shape != expected:
        raise ValueError("reference and replication curvatures must be 3x3")
    if operational.ndim == 2:
        operational = operational[None, :, :]
    if operational.ndim != 3 or operational.shape[1:] != expected:
        raise ValueError("operational_curvatures must have shape (r, 3, 3)")
    if operational.shape[0] == 0:
        raise ValueError("at least one operational curvature is required")
    for name, matrix in (
        ("reference_curvature", reference),
        ("replication_curvature", replication),
    ):
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} must be symmetric")
    if not np.allclose(
        operational,
        np.swapaxes(operational, 1, 2),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("operational_curvatures must be symmetric")

    reference_norm = float(np.linalg.norm(reference, ord="fro"))
    if reference_norm <= _FLOAT_EPS:
        replication_error = None
        operational_errors = np.full(operational.shape[0], np.nan)
    else:
        replication_error = float(
            np.linalg.norm(replication - reference, ord="fro")
            / reference_norm
        )
        operational_errors = (
            np.linalg.norm(operational - reference[None, :, :], axis=(1, 2))
            / reference_norm
        )

    reference_eigenvalues = np.linalg.eigvalsh(reference)
    replication_eigenvalues = np.linalg.eigvalsh(replication)
    operational_eigenvalues = np.linalg.eigvalsh(operational)
    reference_signs = np.sign(reference_eigenvalues)
    replication_sign_agreement = float(
        np.mean(np.sign(replication_eigenvalues) == reference_signs)
    )
    operational_sign_agreements = np.mean(
        np.sign(operational_eigenvalues) == reference_signs[None, :], axis=1
    )
    return CurvatureReliabilityMetrics(
        replication_frobenius_relative_error=replication_error,
        operational_frobenius_relative_errors=operational_errors,
        reference_negative_eigenvalue_count=int(
            np.count_nonzero(reference_eigenvalues < 0.0)
        ),
        replication_negative_eigenvalue_count=int(
            np.count_nonzero(replication_eigenvalues < 0.0)
        ),
        operational_negative_eigenvalue_counts=np.count_nonzero(
            operational_eigenvalues < 0.0, axis=1
        ),
        replication_eigenvalue_sign_agreement=replication_sign_agreement,
        operational_eigenvalue_sign_agreements=operational_sign_agreements,
    )


def analyze_lagged_subspace_population(
    theta: np.ndarray | Sequence[float],
    noise: np.ndarray | Sequence[float],
    paired_returns: np.ndarray | Sequence[float],
    sigma: float,
    basis: np.ndarray | Sequence[float],
    random_basis: np.ndarray | Sequence[float],
    alpha: float,
    *,
    basis_provenance: BasisProvenance,
    endpoint_reference_noise: np.ndarray | Sequence[float] | None = None,
    endpoint_reference_utilities: np.ndarray | Sequence[float] | None = None,
    endpoint_reference_curvature: np.ndarray | Sequence[float] | None = None,
    endpoint_reference_gradient: np.ndarray | Sequence[float] | None = None,
) -> LaggedSubspaceDiagnostic:
    """Run population-level diagnostics for one frozen antithetic batch.

    Cross-bank curvature/action comparisons and paired environment returns are
    intentionally separate runner-level operations.
    """

    theta_array = _as_finite_array("theta", theta)
    returns = _as_finite_array("paired_returns", paired_returns)
    estimate = estimate_lopo_with_jackknife(
        theta_array,
        noise,
        returns,
        sigma,
        basis,
        random_basis,
        basis_provenance=basis_provenance,
    )
    basis_array = _validate_basis("basis", np.asarray(basis), theta_array.size)
    random_basis_array = _validate_basis(
        "random_basis", np.asarray(random_basis), theta_array.size
    )
    pair_noise, signed_pairs, _ = _coerce_pair_noise(
        np.asarray(noise), returns.shape[0], theta_array.size
    )
    del pair_noise
    signed_noise = signed_pairs.reshape(-1, theta_array.size)
    utilities = estimate.utilities.reshape(-1)

    endpoint_values = (
        endpoint_reference_noise,
        endpoint_reference_utilities,
        endpoint_reference_curvature,
        endpoint_reference_gradient,
    )
    if any(value is not None for value in endpoint_values):
        if any(value is None for value in endpoint_values):
            raise ValueError("frozen endpoint reference must be supplied completely")
        reference_noise = _as_finite_array(
            "endpoint_reference_noise", endpoint_reference_noise
        )
        if reference_noise.ndim == 3 and reference_noise.shape[1:] == (
            2,
            theta_array.size,
        ):
            reference_noise = reference_noise.reshape(-1, theta_array.size)
        reference_utilities = _as_finite_array(
            "endpoint_reference_utilities", endpoint_reference_utilities
        ).reshape(-1)
        signed_noise, utilities = _validate_signed_noise_utilities(
            reference_noise, reference_utilities
        )
        endpoint_curvature = _as_finite_array(
            "endpoint_reference_curvature", endpoint_reference_curvature
        )
        endpoint_gradient = _as_finite_array(
            "endpoint_reference_gradient", endpoint_reference_gradient
        )
        if endpoint_curvature.shape != (SUBSPACE_DIMENSION, SUBSPACE_DIMENSION):
            raise ValueError("endpoint_reference_curvature must have shape (3, 3)")
        if not np.allclose(
            endpoint_curvature, endpoint_curvature.T, rtol=1e-12, atol=1e-14
        ):
            raise ValueError("endpoint_reference_curvature must be symmetric")
        if endpoint_gradient.shape != theta_array.shape:
            raise ValueError("endpoint_reference_gradient has the wrong dimension")
    else:
        endpoint_curvature = estimate.curvature
        endpoint_gradient = estimate.gradient

    steps = compute_four_steps(
        theta_array,
        estimate.gradient,
        estimate.curvature,
        basis_array,
        estimate.random_curvature,
        random_basis_array,
        alpha,
    )
    step_vectors = {
        "structured": steps.structured,
        "isotropic": steps.isotropic,
        "explicit": steps.explicit,
        "random": steps.random,
    }
    locality = {
        name: summarize_locality(step, sigma)
        for name, step in step_vectors.items()
    }
    action_metrics = compute_action_metrics(steps, alpha)
    nonlinear = recompute_eigen_action_jackknife(
        estimate,
        theta_array,
        basis_array,
        random_basis_array,
        alpha,
    )
    endpoints = {
        name: frozen_endpoint_diagnostics(
            signed_noise,
            utilities,
            step,
            sigma,
            basis_array,
            endpoint_curvature,
        )
        for name, step in step_vectors.items()
    }

    zero = np.zeros(theta_array.size, dtype=np.float64)
    unnormalized_gradient = frozen_endpoint_gradient(
        signed_noise, utilities, zero, sigma, self_normalized=False
    )
    self_normalized_gradient = frozen_endpoint_gradient(
        signed_noise, utilities, zero, sigma, self_normalized=True
    )
    projected_unnormalized = np.column_stack(
        [
            basis_array.T
            @ frozen_endpoint_jacobian_actions(
                signed_noise, utilities, basis_array[:, index], sigma
            )[0]
            for index in range(SUBSPACE_DIMENSION)
        ]
    )
    projected_self_normalized = np.column_stack(
        [
            basis_array.T
            @ frozen_endpoint_jacobian_actions(
                signed_noise, utilities, basis_array[:, index], sigma
            )[1]
            for index in range(SUBSPACE_DIMENSION)
        ]
    )
    return LaggedSubspaceDiagnostic(
        estimate=estimate,
        steps=steps,
        locality=locality,
        action_metrics=action_metrics,
        nonlinear_jackknife=nonlinear,
        endpoint_diagnostics=endpoints,
        gradient_endpoint_relative_error=_relative_error(
            endpoint_gradient, unnormalized_gradient
        ),
        subspace_jacobian_relative_error=_relative_error(
            endpoint_curvature, projected_unnormalized
        ),
        self_normalized_gradient_relative_error=_relative_error(
            unnormalized_gradient, self_normalized_gradient
        ),
        self_normalized_jacobian_relative_error=_relative_error(
            projected_unnormalized, projected_self_normalized
        ),
        basis_provenance=basis_provenance,
        claim_metadata=CLAIM_METADATA,
    )


__all__ = [
    "ActionMetrics",
    "ActionReliabilityMetrics",
    "BasisProvenance",
    "CLAIM_METADATA",
    "ClaimMetadata",
    "CurvatureReliabilityMetrics",
    "FrozenEndpointDiagnostics",
    "JackknifeEstimate",
    "LOPOPopulationEstimate",
    "LOPOEstimate",
    "LaggedBases",
    "LaggedSubspaceDiagnostic",
    "LocalitySummary",
    "NonlinearJackknife",
    "StepSet",
    "UnresolvedDiagnosticError",
    "analyze_lagged_subspace_population",
    "build_lagged_bases",
    "calibrate_locality_rate",
    "compare_anisotropic_actions",
    "compare_subspace_curvatures",
    "compute_action_metrics",
    "compute_four_steps",
    "estimate_lopo_with_jackknife",
    "estimate_lopo_population",
    "exact_lopo_utilities",
    "frozen_endpoint_diagnostics",
    "frozen_endpoint_gradient",
    "frozen_endpoint_jacobian_actions",
    "frozen_endpoint_jacobians",
    "frozen_endpoint_objective",
    "gradient_u_statistic_row_sums",
    "recompute_eigen_action_jackknife",
    "summarize_locality",
]

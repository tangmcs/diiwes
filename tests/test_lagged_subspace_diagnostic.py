"""Tests for the pure lagged-subspace LOPO diagnostic."""

from __future__ import annotations

import unittest

import numpy as np

from core.lagged_subspace_diagnostic import (
    BasisProvenance,
    UnresolvedDiagnosticError,
    analyze_lagged_subspace_population,
    build_lagged_bases,
    calibrate_locality_rate,
    compare_anisotropic_actions,
    compare_subspace_curvatures,
    compute_action_metrics,
    compute_four_steps,
    estimate_lopo_population,
    estimate_lopo_with_jackknife,
    exact_lopo_utilities,
    frozen_endpoint_diagnostics,
    frozen_endpoint_gradient,
    frozen_endpoint_jacobians,
    gradient_u_statistic_row_sums,
    recompute_eigen_action_jackknife,
    summarize_locality,
)


def _orthonormal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    return np.linalg.qr(rng.normal(size=(dimension, 3)))[0]


def _provenance() -> BasisProvenance:
    return BasisProvenance.strictly_lagged(
        primary_reference="sha256:lagged-gradient-archive",
        random_reference="sha256:locked-signed-permutations",
    )


def _signed_noise(pair_noise: np.ndarray) -> np.ndarray:
    return np.stack((pair_noise, -pair_noise), axis=1).reshape(
        2 * pair_noise.shape[0], pair_noise.shape[1]
    )


def _brute_lopo_utilities(returns: np.ndarray) -> np.ndarray:
    pair_count = returns.shape[0]
    result = np.zeros_like(returns, dtype=np.float64)
    for pair in range(pair_count):
        for sign in range(2):
            total = 0.0
            for other in range(pair_count):
                if other == pair:
                    continue
                for other_sign in range(2):
                    left = returns[pair, sign]
                    right = returns[other, other_sign]
                    total += float(left > right) - float(left < right)
            result[pair, sign] = total / (4.0 * (pair_count - 1))
    return result


class LaggedSubspaceDiagnosticTests(unittest.TestCase):
    def test_lagged_and_signed_permutation_bases_are_previsible(self) -> None:
        rng = np.random.default_rng(11)
        lagged = rng.normal(size=(10, 9))
        lagged[:, :2] = 0.0
        blocks = (slice(0, 2), slice(2, 5), slice(5, 9))
        result = build_lagged_bases(
            lagged,
            blocks,
            primary_fallback_seed=17,
            random_permutation_seed=19,
            primary_reference="archive-before-bank-a",
            random_reference="permutations-before-bank-a",
        )
        repeated = build_lagged_bases(
            lagged,
            blocks,
            primary_fallback_seed=17,
            random_permutation_seed=19,
            primary_reference="archive-before-bank-a",
            random_reference="permutations-before-bank-a",
        )

        self.assertTrue(np.array_equal(result.primary, repeated.primary))
        self.assertTrue(
            np.array_equal(result.random_control, repeated.random_control)
        )
        self.assertTrue(np.allclose(result.primary.T @ result.primary, np.eye(3)))
        self.assertTrue(
            np.allclose(result.random_control.T @ result.random_control, np.eye(3))
        )
        self.assertEqual(result.primary_fallback_blocks, (True, False, False))
        self.assertEqual(
            result.random_uses_primary_fallback_blocks, (True, False, False)
        )
        self.assertTrue(result.provenance.locked_before_current_batch)
        self.assertFalse(result.provenance.uses_current_noise)
        self.assertFalse(result.provenance.uses_current_returns)

        weights = 0.9 ** np.arange(9, -1, -1)
        expected = np.sum(weights[:, None] * lagged, axis=0) / np.sum(weights)
        for column, block in enumerate(blocks):
            indices = np.arange(9)[block]
            outside = np.setdiff1d(np.arange(9), indices)
            self.assertTrue(np.all(result.primary[outside, column] == 0.0))
            self.assertTrue(
                np.all(result.random_control[outside, column] == 0.0)
            )
            self.assertTrue(
                np.array_equal(
                    result.lagged_block_gradients[indices, column],
                    expected[indices],
                )
            )
            self.assertTrue(
                np.allclose(
                    np.sort(np.abs(result.primary[indices, column])),
                    np.sort(np.abs(result.random_control[indices, column])),
                )
            )

    def test_chronological_archive_weights_newest_gradient_most(self) -> None:
        lagged = np.zeros((10, 6), dtype=np.float64)
        # Persisted rows are g[t-10], ..., g[t-1].  Orthogonal sentinels make
        # a reversal visible in the first block's weighted direction.
        lagged[0, 0] = 1.0
        lagged[9, 1] = 1.0
        lagged[:, 2] = 1.0
        lagged[:, 4] = 1.0
        result = build_lagged_bases(
            lagged,
            (slice(0, 2), slice(2, 4), slice(4, 6)),
            primary_fallback_seed=23,
            random_permutation_seed=29,
            primary_reference="chronological-g[t-10]-through-g[t-1]",
            random_reference="locked-before-current-batch",
        )

        first_block = result.lagged_block_gradients[:2, 0]
        self.assertAlmostEqual(first_block[1] / first_block[0], 1.0 / 0.9**9)
        expected_direction = np.asarray([0.9**9, 1.0], dtype=np.float64)
        expected_direction /= np.linalg.norm(expected_direction)
        self.assertTrue(
            np.allclose(result.primary[:2, 0], expected_direction, atol=1e-15)
        )

    def test_tie_safe_lopo_utilities_match_literal_comparisons(self) -> None:
        returns = np.asarray(
            [
                [1.0, 1.0],
                [3.0, -2.0],
                [1.0, 4.0],
                [-2.0, 0.5],
                [3.0, 3.0],
            ]
        )
        expected = _brute_lopo_utilities(returns)
        actual = exact_lopo_utilities(returns)
        shifted = exact_lopo_utilities(7.0 * returns + 5.0)
        self.assertTrue(np.array_equal(actual, expected))
        self.assertTrue(np.array_equal(actual, shifted))
        self.assertLessEqual(
            abs(float(np.sum(actual))),
            1e-12 * max(1.0, float(np.sum(np.abs(actual)))),
        )
        self.assertTrue(
            np.array_equal(exact_lopo_utilities(np.ones((4, 2))), np.zeros((4, 2)))
        )

    def test_lopo_comparisons_are_safe_for_extreme_finite_returns(self) -> None:
        limit = np.finfo(np.float64).max
        extreme = np.asarray(
            [[limit, -limit], [0.0, limit], [-limit, 0.0]], dtype=np.float64
        )
        rank_equivalent = np.asarray(
            [[2.0, -2.0], [0.0, 2.0], [-2.0, 0.0]], dtype=np.float64
        )
        pair_noise = np.asarray(
            [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0]], dtype=np.float64
        )

        with np.errstate(over="raise", invalid="raise"):
            extreme_utilities = exact_lopo_utilities(extreme)
            extreme_rows = gradient_u_statistic_row_sums(
                pair_noise, extreme, sigma=0.2
            )

        self.assertTrue(
            np.array_equal(extreme_utilities, exact_lopo_utilities(rank_equivalent))
        )
        self.assertTrue(
            np.array_equal(
                extreme_rows,
                gradient_u_statistic_row_sums(
                    pair_noise, rank_equivalent, sigma=0.2
                ),
            )
        )

    def test_exhaustive_small_m_lopo_scaling_and_kernel_identities(self) -> None:
        sigma = 0.3
        basis = np.eye(3)
        pair_noise_all = np.asarray(
            [[0.5, -1.0, 0.25], [1.5, 0.75, -0.5], [-0.25, 1.25, 2.0]],
            dtype=np.float64,
        )
        values = np.asarray([0.0, 1.0, 2.0])

        for pair_count in (2, 3):
            pair_noise = pair_noise_all[:pair_count]
            c_m = 2.0 * (pair_count - 1.0) / (2.0 * pair_count - 1.0)
            projected = pair_noise @ basis
            score = np.einsum("mi,mj->mij", projected, projected)
            score -= np.eye(3)[None, :, :]

            for value_indices in np.ndindex(*(3,) * (2 * pair_count)):
                returns = values[np.asarray(value_indices)].reshape(pair_count, 2)
                lopo = exact_lopo_utilities(returns)
                flat = returns.reshape(-1)
                comparisons = np.greater(flat[:, None], flat[None, :]).astype(
                    np.int8
                )
                comparisons -= np.less(flat[:, None], flat[None, :]).astype(
                    np.int8
                )
                pooled = comparisons.sum(axis=1).reshape(pair_count, 2)
                pooled = pooled.astype(np.float64) / (
                    2.0 * (2.0 * pair_count - 1.0)
                )
                mate_sign = np.greater(returns[:, 0], returns[:, 1]).astype(
                    np.int8
                )
                mate_sign -= np.less(returns[:, 0], returns[:, 1]).astype(
                    np.int8
                )
                mate = np.column_stack((mate_sign, -mate_sign))

                np.testing.assert_allclose(
                    pooled,
                    c_m * lopo + mate / (2.0 * (2.0 * pair_count - 1.0)),
                    rtol=0.0,
                    atol=1e-15,
                )

                pooled_gradient = np.sum(
                    (pooled[:, 0] - pooled[:, 1])[:, None] * pair_noise,
                    axis=0,
                ) / (2.0 * pair_count * sigma)
                lopo_gradient = np.sum(
                    (lopo[:, 0] - lopo[:, 1])[:, None] * pair_noise,
                    axis=0,
                ) / (2.0 * pair_count * sigma)
                expected_remainder = np.sum(
                    mate_sign[:, None] * pair_noise, axis=0
                ) / (
                    2.0
                    * pair_count
                    * sigma
                    * (2.0 * pair_count - 1.0)
                )
                np.testing.assert_allclose(
                    pooled_gradient,
                    c_m * lopo_gradient + expected_remainder,
                    rtol=0.0,
                    atol=2e-15,
                )

                pooled_curvature = np.sum(
                    (pooled[:, 0] + pooled[:, 1])[:, None, None] * score,
                    axis=0,
                ) / (2.0 * pair_count * sigma**2)
                lopo_curvature = np.sum(
                    (lopo[:, 0] + lopo[:, 1])[:, None, None] * score,
                    axis=0,
                ) / (2.0 * pair_count * sigma**2)
                np.testing.assert_allclose(
                    pooled_curvature,
                    c_m * lopo_curvature,
                    rtol=0.0,
                    atol=2e-15,
                )

                estimate = estimate_lopo_population(
                    np.zeros(3),
                    pair_noise,
                    returns,
                    sigma,
                    basis,
                    basis,
                    basis_provenance=_provenance(),
                )
                np.testing.assert_allclose(
                    estimate.gradient, lopo_gradient, rtol=0.0, atol=2e-15
                )
                np.testing.assert_allclose(
                    estimate.curvature, lopo_curvature, rtol=0.0, atol=2e-15
                )
                if pair_count == 3:
                    jackknife = estimate_lopo_with_jackknife(
                        np.zeros(3),
                        pair_noise,
                        returns,
                        sigma,
                        basis,
                        basis,
                        basis_provenance=_provenance(),
                    )
                    np.testing.assert_allclose(
                        jackknife.gradient,
                        lopo_gradient,
                        rtol=0.0,
                        atol=2e-15,
                    )
                    np.testing.assert_allclose(
                        jackknife.curvature,
                        lopo_curvature,
                        rtol=0.0,
                        atol=2e-15,
                    )

    def test_fast_gradient_kernel_rows_match_brute_force(self) -> None:
        rng = np.random.default_rng(23)
        pair_count, dimension, sigma = 7, 6, 0.17
        noise = rng.normal(size=(pair_count, dimension))
        returns = rng.integers(-2, 4, size=(pair_count, 2)).astype(np.float64)
        actual = gradient_u_statistic_row_sums(noise, returns, sigma)
        expected = np.zeros_like(actual)
        signed_values = (1.0, -1.0)
        for first in range(pair_count):
            for second in range(pair_count):
                if first == second:
                    continue
                kernel = np.zeros(dimension)
                for first_sign, first_value in enumerate(signed_values):
                    for second_sign, second_value in enumerate(signed_values):
                        comparison = np.sign(
                            returns[first, first_sign]
                            - returns[second, second_sign]
                        )
                        kernel += comparison * (
                            first_value * noise[first]
                            - second_value * noise[second]
                        )
                expected[first] += kernel / (16.0 * sigma)
        self.assertTrue(np.allclose(actual, expected, atol=2e-14))

    def test_delete_pair_jackknife_matches_literal_recomputation(self) -> None:
        rng = np.random.default_rng(29)
        pair_count, dimension, sigma = 8, 7, 0.2
        theta = rng.normal(size=dimension)
        noise = rng.normal(size=(pair_count, dimension))
        returns = rng.integers(-3, 5, size=(pair_count, 2)).astype(np.float64)
        basis = _orthonormal(rng, dimension)
        random_basis = _orthonormal(rng, dimension)
        result = estimate_lopo_with_jackknife(
            theta,
            noise,
            returns,
            sigma,
            basis,
            random_basis,
            basis_provenance=_provenance(),
        )
        self.assertIsNone(result.gradient_jackknife.covariance)
        self.assertEqual(result.curvature_vech_jackknife.covariance.shape, (6, 6))
        for deleted in range(pair_count):
            keep = np.arange(pair_count) != deleted
            literal = estimate_lopo_population(
                theta,
                noise[keep],
                returns[keep],
                sigma,
                basis,
                random_basis,
                basis_provenance=_provenance(),
            )
            self.assertTrue(
                np.allclose(
                    result.gradient_jackknife.delete_estimates[deleted],
                    literal.gradient,
                    atol=2e-13,
                )
            )
            self.assertTrue(
                np.allclose(
                    result.delete_curvatures[deleted],
                    literal.curvature,
                    atol=2e-12,
                )
            )
            self.assertTrue(
                np.allclose(
                    result.random_delete_curvatures[deleted],
                    literal.random_curvature,
                    atol=2e-12,
                )
            )

    def test_noiseless_quadratic_has_correct_gradient_and_curvature_sign(self) -> None:
        rng = np.random.default_rng(31)
        pair_count, sigma = 20_000, 0.25
        noise = rng.normal(size=(pair_count, 3))
        basis = np.eye(3)
        random_basis = _orthonormal(rng, 3)

        center = np.zeros(3)
        plus = -0.5 * np.sum((center + sigma * noise) ** 2, axis=1)
        minus = -0.5 * np.sum((center - sigma * noise) ** 2, axis=1)
        centered = estimate_lopo_population(
            center,
            noise,
            np.column_stack((plus, minus)),
            sigma,
            basis,
            random_basis,
            basis_provenance=_provenance(),
        )
        self.assertTrue(np.array_equal(centered.gradient, np.zeros(3)))
        self.assertTrue(np.all(np.linalg.eigvalsh(centered.curvature) < -2.5))

        center = np.asarray([0.15, -0.1, 0.08])
        plus = -0.5 * np.sum((center + sigma * noise) ** 2, axis=1)
        minus = -0.5 * np.sum((center - sigma * noise) ** 2, axis=1)
        displaced = estimate_lopo_population(
            center,
            noise,
            np.column_stack((plus, minus)),
            sigma,
            basis,
            random_basis,
            basis_provenance=_provenance(),
        )
        cosine = np.dot(displaced.gradient, -center)
        cosine /= np.linalg.norm(displaced.gradient) * np.linalg.norm(center)
        self.assertGreater(cosine, 0.995)
        self.assertTrue(np.all(np.linalg.eigvalsh(displaced.curvature) < -1.5))

    def test_endpoint_jacobians_match_finite_differences_and_subspace_moment(self) -> None:
        rng = np.random.default_rng(37)
        pair_count, dimension, sigma = 10, 5, 0.3
        theta = rng.normal(size=dimension)
        noise = rng.normal(size=(pair_count, dimension))
        returns = rng.integers(-2, 4, size=(pair_count, 2)).astype(np.float64)
        basis = _orthonormal(rng, dimension)
        random_basis = _orthonormal(rng, dimension)
        estimate = estimate_lopo_population(
            theta,
            noise,
            returns,
            sigma,
            basis,
            random_basis,
            basis_provenance=_provenance(),
        )
        signed_noise = _signed_noise(noise)
        utilities = estimate.utilities.reshape(-1)
        jacobian, self_normalized = frozen_endpoint_jacobians(
            signed_noise, utilities, sigma
        )
        self.assertTrue(
            np.allclose(basis.T @ jacobian @ basis, estimate.curvature, atol=2e-13)
        )
        self.assertTrue(np.allclose(jacobian, self_normalized, atol=2e-13))

        step = 1e-6
        numerical = np.empty_like(jacobian)
        numerical_sn = np.empty_like(jacobian)
        for column in range(dimension):
            offset = np.zeros(dimension)
            offset[column] = step
            numerical[:, column] = (
                frozen_endpoint_gradient(
                    signed_noise, utilities, offset, sigma
                )
                - frozen_endpoint_gradient(
                    signed_noise, utilities, -offset, sigma
                )
            ) / (2.0 * step)
            numerical_sn[:, column] = (
                frozen_endpoint_gradient(
                    signed_noise,
                    utilities,
                    offset,
                    sigma,
                    self_normalized=True,
                )
                - frozen_endpoint_gradient(
                    signed_noise,
                    utilities,
                    -offset,
                    sigma,
                    self_normalized=True,
                )
            ) / (2.0 * step)
        self.assertTrue(np.allclose(numerical, jacobian, rtol=2e-8, atol=2e-8))
        self.assertTrue(
            np.allclose(numerical_sn, self_normalized, rtol=2e-8, atol=2e-8)
        )

    def test_constant_utility_shift_is_self_normalization_negative_control(self) -> None:
        rng = np.random.default_rng(41)
        noise = rng.normal(size=(9, 4))
        returns = rng.integers(-2, 3, size=(9, 2)).astype(np.float64)
        utilities = exact_lopo_utilities(returns).reshape(-1)
        signed_noise = _signed_noise(noise)
        sigma = 0.23
        jacobian, self_normalized = frozen_endpoint_jacobians(
            signed_noise, utilities, sigma
        )
        self.assertTrue(np.allclose(jacobian, self_normalized, atol=2e-13))

        shift = 0.7
        shifted = utilities + shift
        shifted_jacobian, shifted_self_normalized = frozen_endpoint_jacobians(
            signed_noise, shifted, sigma
        )
        second = signed_noise.T @ signed_noise / signed_noise.shape[0]
        expected_gap = shift * (np.eye(4) - second) / sigma**2
        self.assertTrue(
            np.allclose(
                shifted_self_normalized - shifted_jacobian,
                expected_gap,
                atol=3e-13,
            )
        )
        self.assertGreater(np.linalg.norm(expected_gap), 0.1)

    def test_rotations_and_equal_norm_controls(self) -> None:
        rng = np.random.default_rng(43)
        dimension = 8
        theta = rng.normal(size=dimension)
        gradient = rng.normal(size=dimension)
        basis = _orthonormal(rng, dimension)
        random_basis = _orthonormal(rng, dimension)
        raw = rng.normal(size=(3, 3))
        curvature = 0.5 * (raw + raw.T)
        random_raw = rng.normal(size=(3, 3))
        random_curvature = 0.5 * (random_raw + random_raw.T)
        alpha = 0.4
        result = compute_four_steps(
            theta,
            gradient,
            curvature,
            basis,
            random_curvature,
            random_basis,
            alpha,
        )
        structured_norm = np.linalg.norm(result.structured)
        self.assertAlmostEqual(np.linalg.norm(result.isotropic), structured_norm)
        self.assertAlmostEqual(np.linalg.norm(result.random), structured_norm)
        self.assertLessEqual(structured_norm, np.linalg.norm(result.explicit) + 1e-12)
        self.assertLess(result.structured_solve_relative_residual, 1e-13)
        self.assertLess(result.random_solve_relative_residual, 1e-13)

        rotation = np.linalg.qr(rng.normal(size=(dimension, dimension)))[0]
        rotated = compute_four_steps(
            rotation @ theta,
            rotation @ gradient,
            curvature,
            rotation @ basis,
            random_curvature,
            rotation @ random_basis,
            alpha,
        )
        for name in ("structured", "isotropic", "explicit", "random"):
            self.assertTrue(
                np.allclose(
                    getattr(rotated, name),
                    rotation @ getattr(result, name),
                    atol=2e-13,
                )
            )
        metrics = compute_action_metrics(result, alpha)
        self.assertGreaterEqual(metrics.alpha_max_concave_eigenvalue, 0.0)

    def test_complete_diagnostic_and_nonlinear_delete_recomputation(self) -> None:
        rng = np.random.default_rng(47)
        pair_count, dimension, sigma = 9, 6, 0.2
        theta = rng.normal(size=dimension)
        noise = rng.normal(size=(pair_count, dimension))
        returns = rng.integers(-3, 5, size=(pair_count, 2)).astype(np.float64)
        basis = _orthonormal(rng, dimension)
        random_basis = _orthonormal(rng, dimension)
        alpha = 0.03
        result = analyze_lagged_subspace_population(
            theta,
            np.stack((noise, -noise), axis=1),
            returns,
            sigma,
            basis,
            random_basis,
            alpha,
            basis_provenance=_provenance(),
        )
        self.assertLess(result.gradient_endpoint_relative_error, 1e-12)
        self.assertLess(result.subspace_jacobian_relative_error, 1e-12)
        self.assertLess(result.self_normalized_gradient_relative_error, 1e-12)
        self.assertLess(result.self_normalized_jacobian_relative_error, 1e-12)
        self.assertEqual(
            set(result.endpoint_diagnostics),
            {"structured", "isotropic", "explicit", "random"},
        )
        self.assertEqual(result.nonlinear_jackknife.delete_eigenvalues.shape, (9, 3))
        self.assertEqual(
            result.nonlinear_jackknife.delete_structured_steps.shape,
            (9, dimension),
        )
        direct = recompute_eigen_action_jackknife(
            result.estimate,
            theta,
            basis,
            random_basis,
            alpha,
        )
        self.assertTrue(
            np.array_equal(
                direct.delete_anisotropic_actions,
                result.nonlinear_jackknife.delete_anisotropic_actions,
            )
        )
        self.assertFalse(result.claim_metadata.raw_return_hessian)
        self.assertFalse(result.claim_metadata.optimizer_confirmation)

    def test_locality_and_action_reliability_metrics(self) -> None:
        steps = np.asarray([[0.1, 0.0], [0.3, 0.0], [0.6, 0.0], [1.2, 0.0]])
        locality = summarize_locality(steps, sigma=1.0)
        self.assertEqual(locality.first, 0.1)
        self.assertEqual(locality.maximum, 1.2)
        self.assertEqual(locality.fraction_at_or_below_0_25, 0.25)
        self.assertEqual(locality.fraction_at_or_below_0_5, 0.5)
        self.assertEqual(locality.fraction_at_or_below_1_0, 0.75)

        reference_structured = np.asarray([2.0, 1.0])
        reference_isotropic = np.asarray([2.0, 0.0])
        replication_structured = np.asarray([2.0, 0.8])
        replication_isotropic = np.asarray([2.0, 0.0])
        operational_structured = np.asarray([[2.0, 0.9], [2.0, 1.1]])
        operational_isotropic = np.asarray([[2.0, 0.0], [2.0, 0.0]])
        reliability = compare_anisotropic_actions(
            reference_structured,
            reference_isotropic,
            replication_structured,
            replication_isotropic,
            operational_structured,
            operational_isotropic,
        )
        self.assertAlmostEqual(
            reliability.material_fraction, 1.0 / np.sqrt(5.0)
        )
        self.assertAlmostEqual(
            reliability.high_sample_relative_disagreement, 0.2 / 0.9
        )
        self.assertAlmostEqual(reliability.operational_rms_relative_error, 0.1)
        self.assertTrue(
            np.allclose(
                reliability.operational_anisotropic_action_relative_errors,
                np.asarray([0.1, 0.1]),
            )
        )
        self.assertTrue(
            np.all(np.isfinite(reliability.operational_structured_step_cosines))
        )

        reference_curvature = np.diag([-2.0, -1.0, 0.5])
        replication_curvature = np.diag([-1.8, -1.1, 0.4])
        operational_curvatures = np.stack(
            (
                np.diag([-2.1, -0.9, 0.6]),
                np.diag([-1.9, 0.2, 0.4]),
            )
        )
        curvature = compare_subspace_curvatures(
            reference_curvature,
            replication_curvature,
            operational_curvatures,
        )
        self.assertEqual(curvature.reference_negative_eigenvalue_count, 2)
        self.assertEqual(curvature.replication_negative_eigenvalue_count, 2)
        self.assertTrue(
            np.array_equal(
                curvature.operational_negative_eigenvalue_counts,
                np.asarray([2, 1]),
            )
        )
        self.assertEqual(curvature.replication_eigenvalue_sign_agreement, 1.0)
        self.assertTrue(
            np.allclose(
                curvature.operational_eigenvalue_sign_agreements,
                np.asarray([1.0, 2.0 / 3.0]),
            )
        )

    def test_degenerate_and_invalid_inputs_fail_explicitly(self) -> None:
        rng = np.random.default_rng(53)
        pair_count, dimension = 5, 5
        theta = np.zeros(dimension)
        noise = rng.normal(size=(pair_count, dimension))
        returns = np.ones((pair_count, 2))
        basis = _orthonormal(rng, dimension)
        random_basis = _orthonormal(rng, dimension)
        result = analyze_lagged_subspace_population(
            theta,
            noise,
            returns,
            0.2,
            basis,
            random_basis,
            0.1,
            basis_provenance=_provenance(),
        )
        self.assertTrue(np.array_equal(result.estimate.gradient, np.zeros(dimension)))
        self.assertTrue(np.array_equal(result.estimate.curvature, np.zeros((3, 3))))
        self.assertIsNone(result.action_metrics.material_fraction)
        self.assertTrue(result.nonlinear_jackknife.zero_anisotropic_action_unresolved)
        self.assertTrue(result.steps.random_control_valid)
        self.assertFalse(result.steps.gradient_direction_defined)
        self.assertIsNone(
            result.endpoint_diagnostics["structured"].full_linearization_residual
        )

        with self.assertRaises(UnresolvedDiagnosticError):
            calibrate_locality_rate(np.zeros(dimension), 0.2, 0.5)
        zero_steps = compute_four_steps(
            theta,
            rng.normal(size=dimension),
            np.diag([-2.0, 0.0, 3.0]),
            basis,
            np.diag([-1.0, 2.0, 4.0]),
            random_basis,
            0.0,
        )
        for step in (
            zero_steps.structured,
            zero_steps.isotropic,
            zero_steps.explicit,
            zero_steps.random,
        ):
            self.assertTrue(np.array_equal(step, np.zeros(dimension)))
        with self.assertRaises(ValueError):
            estimate_lopo_population(
                theta,
                np.stack((noise, noise), axis=1),
                returns,
                0.2,
                basis,
                random_basis,
                basis_provenance=_provenance(),
            )
        bad_provenance = BasisProvenance(
            primary_reference="current",
            random_reference="current",
            locked_before_current_batch=False,
            uses_current_returns=True,
        )
        with self.assertRaises(ValueError):
            estimate_lopo_population(
                theta,
                noise,
                returns,
                0.2,
                basis,
                random_basis,
                basis_provenance=bad_provenance,
            )
        nonorthonormal = basis.copy()
        nonorthonormal[:, 1] = nonorthonormal[:, 0]
        with self.assertRaises(ValueError):
            estimate_lopo_population(
                theta,
                noise,
                returns,
                0.2,
                nonorthonormal,
                random_basis,
                basis_provenance=_provenance(),
            )
        signed_noise = _signed_noise(noise)
        with self.assertRaises(ValueError):
            frozen_endpoint_jacobians(
                signed_noise,
                np.zeros(2 * pair_count),
                0.2,
                max_dimension=3,
            )
        with self.assertRaises(ValueError):
            build_lagged_bases(
                rng.normal(size=(10, dimension)),
                (slice(0, 2), slice(1, 3), slice(3, 5)),
                primary_fallback_seed=1,
                random_permutation_seed=2,
                primary_reference="archive",
                random_reference="permutations",
            )


if __name__ == "__main__":
    unittest.main()

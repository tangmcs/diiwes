#!/usr/bin/env python3
"""Tests for the controlled implicit quadratic trajectory benchmark."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest

import numpy as np

from experiments.implicit_quadratic_optimization_benchmark import (
    EXACT_METHODS,
    BenchmarkConfig,
    Estimates,
    centered_ranks,
    directional_comparison,
    estimate_mc,
    evaluate_antithetic,
    exact_estimates,
    make_cases,
    propose_step,
    run_benchmark,
    write_outputs,
)


class ImplicitQuadraticOptimizationBenchmarkTests(unittest.TestCase):
    def test_case_geometry_and_noise_contract(self) -> None:
        config = BenchmarkConfig(
            dimension=12,
            num_blocks=3,
            population_size=24,
            iterations=2,
            alphas=(0.1,),
            mc_seeds=(0,),
        )
        cases = {case.name: case for case in make_cases(config)}

        aligned = cases["block_aligned_concave"]
        self.assertTrue(np.all(np.linalg.eigvalsh(aligned.hessian) < 0.0))
        for value, block in zip(
            aligned.block_targets, aligned.blocks, strict=True
        ):
            expected = value * np.eye(block.stop - block.start)
            self.assertTrue(np.allclose(aligned.hessian[block, block], expected))
        self.assertEqual(aligned.observation_noise_std, 0.0)
        self.assertIsNotNone(aligned.gap(aligned.initial_params))

        rotated = cases["rotated_concave"]
        self.assertTrue(np.all(np.linalg.eigvalsh(rotated.hessian) < 0.0))
        off_diagonal = rotated.hessian - np.diag(np.diag(rotated.hessian))
        self.assertGreater(np.linalg.norm(off_diagonal), 0.1)

        indefinite = cases["rotated_indefinite"]
        eigenvalues = np.linalg.eigvalsh(indefinite.hessian)
        self.assertTrue(np.any(eigenvalues < 0.0))
        self.assertTrue(np.any(eigenvalues > 0.0))
        self.assertFalse(indefinite.has_finite_maximum)
        self.assertIsNone(indefinite.gap(indefinite.initial_params))

        noisy = cases["block_aligned_additive_noise"]
        self.assertTrue(np.array_equal(noisy.hessian, aligned.hessian))
        self.assertTrue(np.array_equal(noisy.initial_params, aligned.initial_params))
        self.assertEqual(noisy.observation_noise_std, config.additive_noise_std)

    def test_exact_full_implicit_matches_closed_form(self) -> None:
        config = BenchmarkConfig(
            dimension=8,
            num_blocks=2,
            population_size=16,
            iterations=2,
            alphas=(10.0,),
            mc_seeds=(0,),
            cases=("block_aligned_concave",),
        )
        case = make_cases(config)[0]
        estimates = exact_estimates(case, case.initial_params)
        alpha = 10.0
        full = propose_step(
            "oracle_full_implicit",
            estimates,
            case,
            alpha,
            config.singular_tolerance,
        )
        expected_endpoint = np.linalg.solve(
            np.eye(case.dimension) - alpha * case.hessian,
            case.initial_params,
        )
        self.assertTrue(
            np.allclose(case.initial_params + full.step, expected_endpoint)
        )
        self.assertTrue(full.solve_success)
        self.assertLessEqual(full.step_amplification, 1.0 + 1e-12)

        for method in (
            "oracle_diagonal_approximation_signed",
            "oracle_block_approximation_signed",
            "concave_projected_block",
        ):
            proposal = propose_step(
                method,
                estimates,
                case,
                alpha,
                config.singular_tolerance,
            )
            self.assertTrue(np.allclose(proposal.step, full.step, atol=1e-12))
            self.assertGreaterEqual(proposal.denominator_min_abs, 1.0)

        isotropic = propose_step(
            "norm_matched_isotropic",
            estimates,
            case,
            alpha,
            config.singular_tolerance,
        )
        self.assertAlmostEqual(
            np.linalg.norm(isotropic.step), np.linalg.norm(full.step), places=12
        )
        self.assertLessEqual(isotropic.norm_match_relative_error, 1e-12)

        diagonal_projected = propose_step(
            "concave_projected_diagonal",
            estimates,
            case,
            alpha,
            config.singular_tolerance,
        )
        diagonal_isotropic = propose_step(
            "diagonal_norm_matched_isotropic",
            estimates,
            case,
            alpha,
            config.singular_tolerance,
        )
        self.assertTrue(
            np.allclose(diagonal_projected.step, full.step, atol=1e-12)
        )
        np.testing.assert_allclose(
            np.linalg.norm(diagonal_isotropic.step),
            np.linalg.norm(diagonal_projected.step),
            rtol=1e-12,
            atol=0.0,
        )
        self.assertLessEqual(
            diagonal_isotropic.norm_match_relative_error, 1e-12
        )

    def test_mc_oracle_diagonal_uses_true_curvature(self) -> None:
        config = BenchmarkConfig(
            dimension=8,
            num_blocks=2,
            population_size=16,
            iterations=2,
            alphas=(0.75,),
            mc_seeds=(0,),
            cases=("rotated_concave",),
        )
        case = make_cases(config)[0]
        estimates = Estimates(
            gradient=np.linspace(-0.5, 0.5, case.dimension),
            diagonal=np.full(case.dimension, 100.0),
            block=np.full(len(case.blocks), 100.0),
            full_linearization=None,
            diagonal_rmse=None,
            block_rmse=None,
            target_kind="test_sampled_estimate",
        )
        proposal = propose_step(
            "oracle_true_diagonal",
            estimates,
            case,
            config.alphas[0],
            config.singular_tolerance,
        )
        expected = (
            config.alphas[0]
            * estimates.gradient
            / (1.0 - config.alphas[0] * case.diagonal)
        )
        self.assertTrue(np.allclose(proposal.step, expected))

    def test_raw_estimators_converge_to_known_quadratic_targets(self) -> None:
        config = BenchmarkConfig(
            dimension=6,
            num_blocks=3,
            population_size=20,
            iterations=2,
            alphas=(0.1,),
            mc_seeds=(0,),
            cases=("block_aligned_concave",),
        )
        case = make_cases(config)[0]
        rng = np.random.default_rng(29)
        pair_count = 120_000
        eps = rng.normal(size=(pair_count, case.dimension))
        zeros = np.zeros(pair_count, dtype=np.float64)
        plus, minus = evaluate_antithetic(
            case, case.initial_params, eps, config.sigma, zeros, zeros
        )
        estimate = estimate_mc(
            case,
            case.initial_params,
            eps,
            plus,
            minus,
            config.sigma,
            "raw",
        )

        self.assertTrue(
            np.allclose(
                estimate.gradient,
                case.gradient(case.initial_params),
                rtol=0.025,
                atol=0.025,
            )
        )
        self.assertTrue(
            np.allclose(estimate.diagonal, case.diagonal, rtol=0.06, atol=0.04)
        )
        self.assertTrue(
            np.allclose(
                estimate.block, case.block_targets, rtol=0.06, atol=0.04
            )
        )
        self.assertEqual(
            estimate.target_kind, "raw_gaussian_smoothed_hessian"
        )

    def test_rank_full_diagonal_and_reward_scale_invariance(self) -> None:
        config = BenchmarkConfig(
            dimension=8,
            num_blocks=2,
            population_size=32,
            iterations=2,
            alphas=(0.1,),
            mc_seeds=(0,),
            cases=("rotated_concave",),
        )
        case = make_cases(config)[0]
        rng = np.random.default_rng(31)
        eps = rng.normal(size=(16, case.dimension))
        zeros = np.zeros(16, dtype=np.float64)
        plus, minus = evaluate_antithetic(
            case, case.initial_params, eps, config.sigma, zeros, zeros
        )
        estimate = estimate_mc(
            case,
            case.initial_params,
            eps,
            plus,
            minus,
            config.sigma,
            "same_batch_centered_rank",
        )
        scaled = estimate_mc(
            case,
            case.initial_params,
            eps,
            7.0 * plus + 3.0,
            7.0 * minus + 3.0,
            config.sigma,
            "same_batch_centered_rank",
        )

        self.assertTrue(np.array_equal(estimate.gradient, scaled.gradient))
        self.assertTrue(np.array_equal(estimate.diagonal, scaled.diagonal))
        self.assertTrue(np.array_equal(estimate.block, scaled.block))
        self.assertTrue(
            np.allclose(
                np.diag(estimate.full_linearization),
                estimate.diagonal,
                atol=1e-12,
            )
        )
        self.assertIsNone(estimate.diagonal_rmse)
        self.assertIn("no_literal_hessian_target", estimate.target_kind)
        self.assertTrue(
            np.array_equal(
                centered_ranks(np.concatenate((plus, minus))),
                centered_ranks(np.concatenate((7.0 * plus + 3.0, 7.0 * minus + 3.0))),
            )
        )

    def test_directional_control_is_exactly_norm_matched(self) -> None:
        config = BenchmarkConfig(
            dimension=8,
            num_blocks=2,
            population_size=16,
            iterations=2,
            alphas=(1.0,),
            mc_seeds=(0,),
            cases=("rotated_concave",),
        )
        case = make_cases(config)[0]
        comparison = directional_comparison(
            case,
            case.initial_params,
            exact_estimates(case, case.initial_params),
            1.0,
        )
        self.assertAlmostEqual(
            np.linalg.norm(comparison.structured_step),
            np.linalg.norm(comparison.isotropic_step),
            places=12,
        )
        self.assertAlmostEqual(
            comparison.benefit,
            comparison.structured_improvement - comparison.isotropic_improvement,
            places=14,
        )
        self.assertLessEqual(comparison.norm_match_relative_error, 1e-12)
        self.assertGreaterEqual(comparison.cosine, -1.0)
        self.assertLessEqual(comparison.cosine, 1.0)

    def test_norm_matching_is_preserved_for_machine_scale_steps(self) -> None:
        config = BenchmarkConfig(
            dimension=6,
            num_blocks=3,
            population_size=16,
            iterations=2,
            alphas=(0.75,),
            mc_seeds=(0,),
            cases=("block_aligned_concave",),
        )
        case = make_cases(config)[0]
        estimates = Estimates(
            gradient=np.full(case.dimension, 4.0e-16),
            diagonal=np.zeros(case.dimension),
            block=np.asarray((-0.1, -0.45, -2.0)),
            full_linearization=None,
            diagonal_rmse=None,
            block_rmse=None,
            target_kind="test_machine_scale_gradient",
        )
        explicit_norm = np.linalg.norm(config.alphas[0] * estimates.gradient)
        self.assertGreater(explicit_norm, 0.0)
        self.assertLessEqual(explicit_norm, 1.0e-15)

        structured = propose_step(
            "concave_projected_block",
            estimates,
            case,
            config.alphas[0],
            config.singular_tolerance,
        )
        isotropic = propose_step(
            "norm_matched_isotropic",
            estimates,
            case,
            config.alphas[0],
            config.singular_tolerance,
        )

        structured_norm = np.linalg.norm(structured.step)
        isotropic_norm = np.linalg.norm(isotropic.step)
        self.assertGreater(structured_norm, 0.0)
        self.assertLess(structured_norm, explicit_norm)
        np.testing.assert_allclose(
            isotropic_norm,
            structured_norm,
            rtol=1.0e-14,
            atol=0.0,
        )
        self.assertLessEqual(isotropic.norm_match_relative_error, 1.0e-12)
        np.testing.assert_allclose(
            isotropic.step_amplification,
            structured_norm / explicit_norm,
            rtol=1.0e-14,
            atol=0.0,
        )

    def test_small_matrix_is_deterministic_complete_and_safe(self) -> None:
        config = BenchmarkConfig(
            dimension=6,
            num_blocks=3,
            population_size=24,
            iterations=4,
            alphas=(0.1, 10.0),
            mc_seeds=(0,),
        )
        first = run_benchmark(config)
        second = run_benchmark(config)
        self.assertEqual(first, second)
        self.assertEqual(len(first.summaries), 184)
        self.assertEqual(len(first.trajectories), 184 * 5)
        self.assertTrue(first.validation["complete_matrix"])
        self.assertTrue(first.validation["safe_methods_never_amplify"])
        self.assertTrue(first.validation["divergence_first_aggregate_schema"])

        safe_rows = [
            row
            for row in first.trajectories
            if row["active_update"]
            and row["method"]
            in {
                "concave_projected_diagonal",
                "diagonal_norm_matched_isotropic",
                "concave_projected_block",
                "norm_matched_isotropic",
            }
        ]
        self.assertTrue(safe_rows)
        self.assertTrue(
            all(row["step_amplification"] <= 1.0 + 1e-10 for row in safe_rows)
        )
        isotropic = [
            row for row in safe_rows if row["method"] == "norm_matched_isotropic"
        ]
        self.assertTrue(
            all(row["norm_match_relative_error"] <= 1e-10 for row in isotropic)
        )

        aligned_full = [
            row["objective"]
            for row in first.trajectories
            if row["regime"] == "exact_gradient_sanity"
            and row["case"] == "block_aligned_concave"
            and row["alpha"] == 10.0
            and row["method"] == "oracle_full_implicit"
        ]
        aligned_block = [
            row["objective"]
            for row in first.trajectories
            if row["regime"] == "exact_gradient_sanity"
            and row["case"] == "block_aligned_concave"
            and row["alpha"] == 10.0
            and row["method"] == "oracle_block_approximation_signed"
        ]
        self.assertEqual(aligned_full, aligned_block)

        indefinite = [
            row for row in first.trajectories if row["case"] == "rotated_indefinite"
        ]
        self.assertTrue(indefinite)
        self.assertTrue(all(row["objective_gap"] is None for row in indefinite))

        diverged_aggregate = next(
            row
            for row in first.aggregates
            if row["regime"] == "exact_gradient_sanity"
            and row["case"] == "block_aligned_concave"
            and row["method"] == "explicit_exact_gradient"
            and row["alpha"] == 10.0
        )
        self.assertEqual(diverged_aggregate["divergence_rate"], 1.0)
        self.assertEqual(diverged_aggregate["n_nondiverged"], 0)
        self.assertIsNone(
            diverged_aggregate["nondiverged_mean_normalized_gap_auc"]
        )
        self.assertIsNotNone(
            diverged_aggregate[
                "boundary_inclusive_mean_normalized_gap_auc"
            ]
        )
        self.assertTrue(first.directional_aggregates)
        self.assertTrue(
            all(
                row["reference_horizon_completion_fraction"] <= 1.0
                for row in first.directional_aggregates
            )
        )

    def test_singular_diagnostic_path_remains_strictly_finite(self) -> None:
        config = BenchmarkConfig(
            dimension=6,
            num_blocks=3,
            population_size=12,
            iterations=2,
            alphas=(1.0,),
            mc_seeds=(0,),
            cases=("block_aligned_concave",),
        )
        case = make_cases(config)[0]
        estimates = Estimates(
            gradient=np.ones(case.dimension),
            diagonal=np.ones(case.dimension),
            block=np.ones(len(case.blocks)),
            full_linearization=np.eye(case.dimension),
            diagonal_rmse=None,
            block_rmse=None,
            target_kind="test",
        )
        proposal = propose_step(
            "sampled_signed_diagonal",
            estimates,
            case,
            1.0,
            config.singular_tolerance,
        )
        self.assertFalse(proposal.solve_success)
        self.assertEqual(proposal.failure_reason, "near_singular_diagonal_system")
        self.assertTrue(np.isfinite(proposal.step_amplification))
        self.assertTrue(np.isfinite(proposal.inverse_operator_norm))

    def test_outputs_are_byte_deterministic_and_hash_locked(self) -> None:
        config = BenchmarkConfig(
            dimension=6,
            num_blocks=3,
            population_size=12,
            iterations=2,
            alphas=(0.1,),
            mc_seeds=(0,),
            cases=("block_aligned_concave",),
            fitness_transforms=("raw",),
        )
        result = run_benchmark(config)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = write_outputs(first_dir, config, result)
            second = write_outputs(second_dir, config, result)
            for key in first:
                with open(first[key], "rb") as stream:
                    first_bytes = stream.read()
                with open(second[key], "rb") as stream:
                    second_bytes = stream.read()
                self.assertEqual(first_bytes, second_bytes, key)
            with open(first["manifest"], encoding="utf-8") as stream:
                manifest = json.load(stream)
            for key, metadata in manifest["files"].items():
                with open(first[key], "rb") as stream:
                    digest = hashlib.sha256(stream.read()).hexdigest()
                self.assertEqual(digest, metadata["sha256"])

        self.assertEqual(manifest["schema_version"], 1)
        self.assertTrue(manifest["validation"]["complete_matrix"])
        self.assertFalse(manifest["scope"]["same_batch_rank_literal_hessian_target"])
        self.assertEqual(len(manifest["provenance"]["source_sha256"]), 64)

    def test_invalid_configs_are_rejected_before_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "divide dimension"):
            BenchmarkConfig(dimension=7, num_blocks=3).validate()
        with self.assertRaisesRegex(ValueError, "even integer"):
            BenchmarkConfig(population_size=15).validate()
        with self.assertRaisesRegex(ValueError, "unique"):
            BenchmarkConfig(alphas=(1.0, 1.0)).validate()
        with self.assertRaisesRegex(ValueError, "selected"):
            BenchmarkConfig(cases=("unknown",)).validate()
        self.assertEqual(len(EXACT_METHODS), 8)


if __name__ == "__main__":
    unittest.main()

"""Tests for the controlled ES curvature-reliability benchmark."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from experiments.curvature_reliability_benchmark import (
    BenchmarkConfig,
    estimate_joint_block_ols,
    estimate_pooled_block_moment,
    estimate_stein_diagonal,
    make_blocks,
    make_pair_response,
    make_surface,
    run_benchmark,
    verify_against_core_helpers,
    write_benchmark_outputs,
)


class CurvatureReliabilityBenchmarkTests(unittest.TestCase):
    def test_surface_families_have_known_structure(self) -> None:
        blocks = make_blocks(12, 4)
        diagonal = make_surface("diagonal", 12, blocks, seed=11)
        self.assertTrue(
            np.allclose(diagonal.hessian, np.diag(np.diag(diagonal.hessian)))
        )
        self.assertTrue(np.all(diagonal.diagonal < 0.0))

        block = make_surface("block_isotropic", 12, blocks, seed=11)
        for target, section in zip(block.block_targets, blocks, strict=True):
            expected = target * np.eye(section.stop - section.start)
            self.assertTrue(
                np.allclose(block.hessian[section, section], expected)
            )
        self.assertTrue(
            np.allclose(block.hessian, np.diag(np.diag(block.hessian)))
        )

        saddle = make_surface("saddle", 12, blocks, seed=11)
        self.assertTrue(np.any(saddle.block_targets < 0.0))
        self.assertTrue(np.any(saddle.block_targets > 0.0))

        rotated = make_surface("rotated", 12, blocks, seed=11)
        rotated_again = make_surface("rotated", 12, blocks, seed=11)
        self.assertTrue(np.array_equal(rotated.hessian, rotated_again.hessian))
        self.assertTrue(np.allclose(rotated.hessian, rotated.hessian.T))
        self.assertGreater(
            np.linalg.norm(rotated.hessian - np.diag(rotated.diagonal)), 0.1
        )
        self.assertTrue(np.all(np.linalg.eigvalsh(rotated.hessian) < 0.0))

        rotated_saddle = make_surface("rotated_saddle", 12, blocks, seed=11)
        eigenvalues = np.linalg.eigvalsh(rotated_saddle.hessian)
        self.assertTrue(np.any(eigenvalues < 0.0))
        self.assertTrue(np.any(eigenvalues > 0.0))

    def test_joint_ols_exactly_recovers_block_isotropic_quadratic(self) -> None:
        blocks = make_blocks(8, 4)
        surface = make_surface("block_isotropic", 8, blocks, seed=29)
        rng = np.random.default_rng(31)
        eps = rng.normal(size=(200, 8))
        sigma = 0.2
        response = surface.centered_pair_response(eps, sigma)

        estimate = estimate_joint_block_ols(eps, response, sigma, blocks)

        self.assertTrue(
            np.allclose(estimate.value, surface.block_targets, atol=1e-12)
        )
        self.assertTrue(np.all(estimate.standard_error < 1e-12))
        self.assertEqual(estimate.diagnostics["rank"], 5)
        self.assertEqual(estimate.diagnostics["residual_dof"], 195)

    def test_moment_estimators_converge_to_their_known_targets(self) -> None:
        blocks = make_blocks(6, 3)
        surface = make_surface("diagonal", 6, blocks, seed=41)
        rng = np.random.default_rng(43)
        eps = rng.normal(size=(150_000, 6))
        sigma = 0.3
        response = surface.centered_pair_response(eps, sigma)

        diagonal = estimate_stein_diagonal(
            eps, response, sigma, baseline="loo"
        )
        pooled = estimate_pooled_block_moment(
            eps, response, sigma, blocks, baseline="loo"
        )

        self.assertTrue(
            np.allclose(diagonal.value, surface.diagonal, rtol=0.04, atol=0.04)
        )
        self.assertTrue(
            np.allclose(
                pooled.value, surface.block_targets, rtol=0.04, atol=0.04
            )
        )

    def test_crn_cancels_shared_additive_noise(self) -> None:
        blocks = make_blocks(6, 3)
        surface = make_surface("saddle", 6, blocks, seed=47)
        eps = np.random.default_rng(53).normal(size=(20, 6))
        sigma = 0.1
        deterministic = make_pair_response(
            surface,
            eps,
            sigma,
            noise_mode="none",
            noise_std=0.0,
            rng=np.random.default_rng(59),
        )
        crn = make_pair_response(
            surface,
            eps,
            sigma,
            noise_mode="crn",
            noise_std=100.0,
            rng=np.random.default_rng(61),
        )
        independent = make_pair_response(
            surface,
            eps,
            sigma,
            noise_mode="independent",
            noise_std=100.0,
            rng=np.random.default_rng(61),
        )

        self.assertTrue(np.array_equal(crn, deterministic))
        self.assertFalse(np.allclose(independent, deterministic))

    def test_independent_formulas_match_optimizer_helpers(self) -> None:
        errors = verify_against_core_helpers(seed=67)
        self.assertEqual(set(errors), {
            "stein_value_max_abs_error",
            "stein_se_max_abs_error",
            "block_value_max_abs_error",
            "block_se_max_abs_error",
            "ols_value_max_abs_error",
            "ols_se_max_abs_error",
        })
        self.assertTrue(all(value <= 1e-12 for value in errors.values()))

    def test_benchmark_is_deterministic_and_gate_cannot_amplify(self) -> None:
        config = BenchmarkConfig(
            dimensions=(6,),
            populations=(20,),
            sigmas=(0.1,),
            surfaces=("saddle", "rotated_saddle"),
            noise_modes=("none", "independent", "crn"),
            noise_stds=(0.5,),
            repetitions=5,
            num_blocks=2,
            seed=71,
            learning_rate=10.0,
        )

        first = run_benchmark(config)
        second = run_benchmark(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2 * 3 * 4)
        gates = [
            row
            for row in first
            if row["method"] == "confidence_gated_block_ols"
        ]
        self.assertEqual(len(gates), 6)
        for row in gates:
            self.assertLessEqual(row["max_amplification"], 1.0 + 1e-12)
            self.assertEqual(row["resonance_probability"], 0.0)
            self.assertEqual(row["nonpositive_denominator_probability"], 0.0)
            self.assertGreaterEqual(row["gate_activation_rate"], 0.0)
            self.assertLessEqual(row["gate_activation_rate"], 1.0)

    def test_outputs_are_strict_and_self_describing(self) -> None:
        config = BenchmarkConfig(
            dimensions=(4,),
            populations=(16,),
            sigmas=(0.2,),
            surfaces=("block_isotropic",),
            noise_modes=("none",),
            repetitions=3,
            num_blocks=2,
            seed=73,
        )
        rows = run_benchmark(config)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "benchmark.csv")
            json_path = os.path.join(directory, "benchmark.json")
            outputs = write_benchmark_outputs(
                csv_path,
                rows,
                config,
                {"verified": 0.0},
                metadata_output=json_path,
            )
            self.assertEqual(outputs, (csv_path, json_path))
            with open(csv_path, newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            with open(json_path, encoding="utf-8") as stream:
                payload = json.load(stream)

        self.assertEqual(len(csv_rows), 4)
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["config"]["seed"], 73)
        self.assertEqual(len(payload["rows"]), 4)
        self.assertIn("noise_semantics", payload)
        self.assertFalse(payload["scope"]["rank_surrogate_evaluated"])
        self.assertFalse(payload["scope"]["rl_environment_evaluated"])
        self.assertTrue(
            all(not row["rank_surrogate_evaluated"] for row in payload["rows"])
        )
        self.assertTrue(
            all(
                row["coverage_interpretation"]
                == "empirical_finite_sample_not_guaranteed"
                for row in payload["rows"]
            )
        )

    def test_invalid_grid_is_rejected_before_sampling(self) -> None:
        with self.assertRaisesRegex(ValueError, "even integers"):
            BenchmarkConfig(populations=(15,)).validate()
        with self.assertRaisesRegex(ValueError, "more antithetic pairs"):
            BenchmarkConfig(
                dimensions=(8,), populations=(8,), num_blocks=4
            ).validate()


if __name__ == "__main__":
    unittest.main()

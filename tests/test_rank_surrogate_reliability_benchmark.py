"""Tests for the controlled rank-surrogate reliability benchmark."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from core.implicit_es import LinearizedImplicitES
from core.standard_es import centered_ranks, centered_ranks_from_reference
from experiments.curvature_reliability_benchmark import make_blocks, make_surface
from experiments.rank_surrogate_reliability_benchmark import (
    RankBenchmarkConfig,
    conditional_covariance_score_matrix,
    cross_fitted_rank_utilities,
    endpoint_gradient_with_frozen_utilities,
    evaluate_antithetic_batch,
    finite_difference_frozen_jacobian,
    leave_one_pair_out_rank_utilities,
    pair_covariance_score_contributions,
    reference_cdf_utilities,
    run_benchmark,
    same_batch_finite_population_factor,
    same_batch_pair_u_statistic_matrix,
    split_covariance_score_estimates,
    transform_utilities,
    verify_production_conditional_jacobian,
    write_outputs,
)


class RankSurrogateReliabilityBenchmarkTests(unittest.TestCase):
    def test_same_batch_transform_and_diagonal_match_production(self) -> None:
        eps_half = np.asarray(
            [
                [0.2, -1.1, 0.5, 1.2],
                [1.3, 0.4, -0.8, 0.1],
                [-0.6, 0.9, 1.1, -0.2],
                [0.7, -0.3, -1.4, 0.8],
            ],
            dtype=np.float64,
        )
        noise = np.concatenate((eps_half, -eps_half), axis=0)
        fitness = np.asarray([3.0, 1.0, 1.0, -2.0, 2.0, 0.5, -1.0, 4.0])
        sigma = 0.2
        utilities = transform_utilities(
            fitness,
            "same_batch_centered_rank",
            sorted_reference=np.sort(fitness),
            pair_count=4,
        )
        self.assertTrue(np.allclose(utilities, centered_ranks(fitness), atol=1e-16))

        independent = np.mean(
            pair_covariance_score_contributions(eps_half, utilities, sigma),
            axis=0,
        )
        optimizer = LinearizedImplicitES(
            num_params=4,
            population_size=8,
            noise_std=sigma,
            rank_fitness=True,
        )
        production_utilities, _ = optimizer._utilities(fitness)
        production, _ = optimizer._matched_diagonal_hessian(
            noise,
            production_utilities,
            {
                "fresh_pair_plus": np.arange(4),
                "fresh_pair_minus": np.arange(4, 8),
            },
        )
        self.assertTrue(np.array_equal(utilities, production_utilities))
        self.assertTrue(np.allclose(independent, production, atol=1e-14))

    def test_tied_ranks_equal_scaled_lopo_pair_u_statistic(self) -> None:
        eps_half = np.asarray(
            [
                [0.2, -1.1, 0.5],
                [1.3, 0.4, -0.8],
                [-0.6, 0.9, 1.1],
                [0.7, -0.3, -1.4],
            ],
            dtype=np.float64,
        )
        # Several exact ties exercise the midrank comparison convention.
        fitness = np.asarray([1.0, 1.0, 3.0, -2.0, 1.0, 0.5, -2.0, 4.0])
        pair_count, dimension = eps_half.shape
        sigma = 0.2
        noise = np.concatenate((eps_half, -eps_half), axis=0)
        utilities = centered_ranks(fitness)
        production = conditional_covariance_score_matrix(
            noise, utilities, sigma
        )
        factor = same_batch_finite_population_factor(pair_count)
        pair_u_statistic = same_batch_pair_u_statistic_matrix(
            eps_half, fitness, sigma
        )

        lopo_utilities = leave_one_pair_out_rank_utilities(
            fitness, pair_count
        )
        score = np.einsum("bi,bj->bij", noise, noise, optimize=True)
        score -= np.eye(dimension)[None, :, :]
        lopo_curvature = np.mean(
            lopo_utilities[:, None, None] * score, axis=0
        ) / sigma**2

        mate_comparison = np.sign(
            fitness[:pair_count] - fitness[pair_count:]
        )
        mate_term = np.concatenate((mate_comparison, -mate_comparison))
        mate_term /= 2.0 * (2 * pair_count - 1)
        self.assertTrue(
            np.allclose(
                utilities, factor * lopo_utilities + mate_term, atol=1e-15
            )
        )
        self.assertTrue(
            np.allclose(production, factor * pair_u_statistic, atol=1e-13)
        )
        self.assertTrue(
            np.allclose(pair_u_statistic, lopo_curvature, atol=1e-13)
        )

        production_gradient = np.mean(
            utilities[:, None] * noise, axis=0
        ) / sigma
        lopo_gradient = np.mean(
            lopo_utilities[:, None] * noise, axis=0
        ) / sigma
        within_pair_remainder = np.mean(
            eps_half * mate_comparison[:, None], axis=0
        ) / (2.0 * sigma * (2 * pair_count - 1))
        self.assertTrue(
            np.allclose(
                production_gradient,
                factor * lopo_gradient + within_pair_remainder,
                atol=1e-14,
            )
        )

    def test_same_batch_population_factor_on_squared_gaussian(self) -> None:
        # For Y = epsilon^2 and sigma = 1, the frozen-current-CDF score
        # identity has H_stop = 1 / pi. Antithetic members tie exactly, so this
        # also exercises the midrank convention in a population calculation.
        rng = np.random.default_rng(20260712)
        repetitions = 200_000
        pair_count = 5
        eps = rng.normal(size=(repetitions, pair_count))
        fitness = eps**2
        pair_rank = np.argsort(
            np.argsort(fitness, axis=1), axis=1
        )
        pair_utility = (
            (2.0 * pair_rank + 0.5) / (2 * pair_count - 1) - 0.5
        )
        production = np.mean(
            pair_utility * (eps**2 - 1.0), axis=1
        )
        expected = (
            same_batch_finite_population_factor(pair_count) / np.pi
        )
        self.assertAlmostEqual(float(np.mean(production)), expected, delta=0.002)

    def test_frozen_utility_matrix_is_endpoint_jacobian(self) -> None:
        rng = np.random.default_rng(17)
        eps_half = rng.normal(size=(20, 5))
        noise = np.concatenate((eps_half, -eps_half), axis=0)
        utilities = centered_ranks(rng.normal(size=40))
        sigma = 0.13

        analytic = conditional_covariance_score_matrix(noise, utilities, sigma)
        numerical = finite_difference_frozen_jacobian(
            noise, utilities, sigma
        )
        gradient = endpoint_gradient_with_frozen_utilities(
            noise, utilities, np.zeros(5), sigma
        )

        self.assertTrue(np.allclose(analytic, analytic.T, atol=1e-14))
        self.assertTrue(np.allclose(analytic, numerical, rtol=1e-8, atol=1e-8))
        self.assertEqual(gradient.shape, (5,))

        verification = verify_production_conditional_jacobian(seed=19)
        self.assertFalse(verification["objective_hessian"])
        self.assertLess(
            verification["finite_difference_relative_frobenius_error"], 1e-8
        )
        self.assertLess(
            verification["production_diagonal_max_abs_error"], 1e-12
        )

    def test_conditional_rank_jacobian_is_not_analytic_objective_hessian(self) -> None:
        blocks = make_blocks(6, 3)
        surface = make_surface("saddle", 6, blocks, seed=23)
        rng = np.random.default_rng(29)
        eps_half = rng.normal(size=(30, 6))
        noise = np.concatenate((eps_half, -eps_half), axis=0)
        fitness = 0.5 * 0.1**2 * np.einsum(
            "bi,ij,bj->b", noise, surface.hessian, noise, optimize=True
        )
        utilities = centered_ranks(fitness)
        conditional = conditional_covariance_score_matrix(
            noise, utilities, sigma=0.1
        )

        self.assertGreater(np.linalg.norm(conditional - surface.hessian), 1.0)

    def test_reference_cdf_mapping_matches_core_helper(self) -> None:
        reference = np.asarray([-2.0, -1.0, -1.0, 0.5, 3.0, 4.0])
        values = np.asarray([-5.0, -1.0, -0.5, 4.0, 9.0])
        expected = centered_ranks_from_reference(values, reference)
        actual = reference_cdf_utilities(values, np.sort(reference))
        self.assertTrue(np.array_equal(actual, expected))

    def test_cross_fit_preserves_pairs_and_excludes_own_fold(self) -> None:
        pair_count = 4
        fitness = np.asarray([1.0, 3.0, 2.0, 4.0, 0.5, 2.5, 1.5, 3.5])
        actual = cross_fitted_rank_utilities(fitness, pair_count)
        expected = np.empty(2 * pair_count)
        for fold in (0, 1):
            heldout_pairs = np.arange(pair_count)[np.arange(pair_count) % 2 == fold]
            reference_pairs = np.arange(pair_count)[np.arange(pair_count) % 2 != fold]
            heldout = np.concatenate(
                (heldout_pairs, heldout_pairs + pair_count)
            )
            reference = np.concatenate(
                (reference_pairs, reference_pairs + pair_count)
            )
            expected[heldout] = centered_ranks_from_reference(
                fitness[heldout], fitness[reference]
            )
        expected -= np.mean(expected)

        self.assertTrue(np.array_equal(actual, expected))

    def test_split_same_batch_reranks_each_disjoint_pair_half(self) -> None:
        eps_half = np.asarray(
            [
                [0.2, -1.0],
                [1.2, 0.3],
                [-0.5, 0.8],
                [0.9, -0.4],
            ]
        )
        fitness = np.asarray([0.0, 100.0, 2.0, 3.0, 1.0, 101.0, 4.0, 5.0])
        blocks = make_blocks(2, 2)
        estimates, semantics, independence = split_covariance_score_estimates(
            fitness,
            eps_half,
            0.2,
            "same_batch_centered_rank",
            sorted_reference=np.sort(fitness),
            blocks=blocks,
        )
        first_indices = np.asarray([0, 1, 4, 5])
        first_utilities = centered_ranks(fitness[first_indices])
        expected_first = np.mean(
            pair_covariance_score_contributions(
                eps_half[:2], first_utilities, 0.2
            ),
            axis=0,
        )

        self.assertTrue(np.allclose(estimates["diag"][0], expected_first))
        self.assertEqual(
            semantics, "independent_centered_ranks_per_disjoint_pair_half"
        )
        self.assertEqual(independence, "independent_disjoint_pair_halves")

    def test_paired_crn_shares_only_with_antithetic_partner(self) -> None:
        blocks = make_blocks(6, 3)
        surface = make_surface("diagonal", 6, blocks, seed=31)
        eps_half = np.random.default_rng(37).normal(size=(8, 6))
        direction = np.ones(6) / np.sqrt(6.0)
        independent_fitness, independent_noise = evaluate_antithetic_batch(
            eps_half,
            surface.hessian,
            direction,
            0.1,
            1.0,
            noise_coupling="independent",
            observation_noise_std=0.2,
            rng=np.random.default_rng(41),
        )
        paired_fitness, paired_noise = evaluate_antithetic_batch(
            eps_half,
            surface.hessian,
            direction,
            0.1,
            1.0,
            noise_coupling="paired_crn",
            observation_noise_std=0.2,
            rng=np.random.default_rng(41),
        )

        self.assertTrue(np.array_equal(paired_noise[0], paired_noise[1]))
        self.assertTrue(np.array_equal(paired_noise[0], independent_noise[0]))
        self.assertFalse(np.array_equal(independent_noise[0], independent_noise[1]))
        self.assertEqual(independent_fitness.shape, paired_fitness.shape)

    def test_small_benchmark_is_deterministic_and_self_consistent(self) -> None:
        config = RankBenchmarkConfig(
            dimensions=(6,),
            populations=(20,),
            sigmas=(0.1,),
            surfaces=("saddle",),
            linear_scales=(0.0,),
            noise_couplings=("none", "independent", "paired_crn"),
            observation_noise_std=0.2,
            repetitions=4,
            num_blocks=2,
            reference_size=1_000,
            target_size=2_000,
            seed=43,
        )
        first = run_benchmark(config)
        second = run_benchmark(config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3 * 3 * 2)
        production = [
            row
            for row in first
            if row["transform"] == "same_batch_centered_rank"
        ]
        self.assertTrue(all(row["same_batch_production_semantics"] for row in production))
        self.assertTrue(
            all(not row["original_return_objective_hessian"] for row in first)
        )
        self.assertTrue(all(not row["target_current_batch_centered"] for row in first))
        self.assertTrue(all(row["estimate_current_batch_centered"] for row in first))
        for row in first:
            self.assertGreaterEqual(row["split_sign_agreement_mean"], 0.0)
            self.assertLessEqual(row["split_sign_agreement_mean"], 1.0)

        keyed = {
            (row["noise_coupling"], row["method"]): row for row in first
        }
        for method in {
            row["method"] for row in first if row["noise_coupling"] == "independent"
        }:
            independent = keyed[("independent", method)]
            paired = keyed[("paired_crn", method)]
            self.assertEqual(independent["target_norm"], paired["target_norm"])
            self.assertEqual(
                independent["target_mean_standard_error"],
                paired["target_mean_standard_error"],
            )

    def test_outputs_state_conditional_scope(self) -> None:
        config = RankBenchmarkConfig(
            dimensions=(4,),
            populations=(16,),
            sigmas=(0.2,),
            surfaces=("block_isotropic",),
            linear_scales=(0.0,),
            noise_couplings=("none",),
            repetitions=3,
            num_blocks=2,
            reference_size=500,
            target_size=1_000,
            seed=47,
        )
        rows = run_benchmark(config)
        verification = verify_production_conditional_jacobian(seed=47)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = os.path.join(directory, "rank.csv")
            json_path = os.path.join(directory, "rank.json")
            write_outputs(
                csv_path,
                rows,
                config,
                verification,
                metadata_output=json_path,
            )
            with open(csv_path, newline="", encoding="utf-8") as stream:
                csv_rows = list(csv.DictReader(stream))
            with open(json_path, encoding="utf-8") as stream:
                payload = json.load(stream)

        self.assertEqual(len(csv_rows), 6)
        self.assertEqual(payload["schema_version"], 2)
        self.assertTrue(
            payload["scope"]["production_same_batch_rank_semantics_evaluated"]
        )
        self.assertFalse(payload["scope"]["original_return_objective_hessian_evaluated"])
        self.assertFalse(
            payload["scope"]["population_target_current_batch_centered"]
        )
        self.assertTrue(payload["scope"]["reported_estimates_current_batch_centered"])
        self.assertIn("within-pair", payload["scope"]["gradient_boundary"])
        self.assertFalse(
            payload["production_conditional_jacobian_verification"][
                "objective_hessian"
            ]
        )

    def test_invalid_grid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiples of four"):
            RankBenchmarkConfig(populations=(18,)).validate()
        with self.assertRaisesRegex(ValueError, "reference_size"):
            RankBenchmarkConfig(reference_size=20).validate()
        with self.assertRaisesRegex(ValueError, "target_size"):
            RankBenchmarkConfig(target_size=999).validate()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Tests for the 10-D initial-learning-rate convex sweep."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.convex_10d_initial_lr_sweep import (
    CURVATURE_UPDATE_FIELDS,
    DIMENSION,
    INITIAL_LEARNING_RATES,
    METHODS,
    PROBLEMS,
    TARGET_FRACTION,
    ExperimentConfig,
    exact_gradient_initial_step_reference,
    learning_rate,
    optimization_perturbations,
    run_experiment,
    write_outputs,
)


class Convex10DInitialLearningRateSweepTests(unittest.TestCase):
    def test_fixed_protocol(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(DIMENSION, 10)
        self.assertEqual(METHODS, ("explicit_es", "linearized_implicit_es"))
        self.assertEqual(config.population_size, 2000)
        self.assertEqual(config.updates, 300)
        self.assertEqual(config.seeds, tuple(range(10)))
        self.assertEqual(config.sigma, 0.1)
        self.assertEqual(
            INITIAL_LEARNING_RATES,
            (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00),
        )
        self.assertEqual(config.initial_learning_rates, INITIAL_LEARNING_RATES)
        self.assertEqual([problem.condition_number for problem in PROBLEMS], [2, 4, 8])
        for problem in PROBLEMS:
            np.testing.assert_allclose(
                problem.hessian_diagonal,
                np.geomspace(1.0, problem.condition_number, DIMENSION),
            )

    def test_schedule_and_exact_gradient_references(self) -> None:
        self.assertEqual(learning_rate(2.0, 0), 2.0)
        self.assertAlmostEqual(learning_rate(0.5, 99), 0.05)
        self.assertAlmostEqual(learning_rate(2.0, 299), 2.0 / np.sqrt(300.0))
        self.assertEqual(exact_gradient_initial_step_reference(2), 1.0)
        self.assertEqual(exact_gradient_initial_step_reference(4), 0.5)
        self.assertEqual(exact_gradient_initial_step_reference(8), 0.25)
        with self.assertRaises(ValueError):
            learning_rate(0.0, 0)
        with self.assertRaises(ValueError):
            learning_rate(0.5, -1)
        with self.assertRaises(ValueError):
            exact_gradient_initial_step_reference(0)

    def test_common_random_numbers_exclude_problem_rate_and_method(self) -> None:
        config = replace(ExperimentConfig(), population_size=100)
        first = optimization_perturbations(config, seed=3, update_index=7)
        second = optimization_perturbations(config, seed=3, update_index=7)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (50, DIMENSION))
        self.assertFalse(
            np.array_equal(
                first, optimization_perturbations(config, seed=3, update_index=8)
            )
        )

    def test_small_run_contract_and_uncensored_target_encoding(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=3,
            seeds=(0, 1),
            initial_learning_rates=(0.1, 1.0, 2.0),
        )
        result = run_experiment(config)
        rate_count = len(config.initial_learning_rates)
        self.assertEqual(
            len(result.trajectories),
            len(PROBLEMS)
            * rate_count
            * len(METHODS)
            * len(config.seeds)
            * (config.updates + 1),
        )
        self.assertEqual(
            len(result.optimization_summary),
            len(PROBLEMS) * rate_count * len(METHODS) * (config.updates + 1),
        )
        self.assertEqual(
            len(result.decision_metrics),
            len(PROBLEMS) * rate_count * len(METHODS) * len(config.seeds),
        )
        self.assertEqual(
            len(result.decision_summary), len(PROBLEMS) * rate_count * len(METHODS)
        )
        self.assertEqual(
            len(result.curvature_updates),
            len(PROBLEMS) * rate_count * len(config.seeds) * config.updates,
        )
        self.assertEqual(
            len(result.curvature_diagnostics), len(PROBLEMS) * rate_count
        )
        self.assertEqual(
            len(result.learning_rates), rate_count * config.updates
        )
        self.assertEqual(TARGET_FRACTION, 1.0e-4)
        for row in result.decision_summary:
            self.assertEqual(row["reached_target_count"], 0)
            self.assertEqual(row["median_first_update_all_seeds"], "")
            self.assertNotEqual(row["median_first_update_all_seeds"], config.updates + 1)
        for row in result.decision_metrics:
            self.assertIn("finite_run", row)
            self.assertIn("failure_update", row)
            self.assertEqual(row["first_update_fraction_le_1e_minus_4"], "")

    def test_raw_curvature_and_denominator_fields_are_retained(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=2,
            seeds=(0,),
            initial_learning_rates=(0.5, 2.0),
        )
        result = run_experiment(config)
        row = result.curvature_updates[0]
        self.assertEqual(set(row), set(CURVATURE_UPDATE_FIELDS))
        for index in range(1, DIMENSION + 1):
            self.assertIn(f"h{index}_true", row)
            self.assertIn(f"h{index}_estimate", row)
            self.assertIn(f"denominator_{index}", row)
            self.assertIn(f"multiplier_{index}", row)
            self.assertAlmostEqual(
                row[f"multiplier_{index}"], 1.0 / row[f"denominator_{index}"]
            )

    def test_outputs_hashes_figures_and_manifest(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=8,
            seeds=(0, 1),
            initial_learning_rates=(0.1, 0.5, 1.0, 2.0),
        )
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = {
                name: Path(path)
                for name, path in write_outputs(temporary, config, result).items()
            }
            self.assertIn("mentor_peak_pdf", outputs)
            self.assertIn("mentor_peak_png", outputs)
            for path in outputs.values():
                self.assertTrue(path.is_file(), path)
            for name, path in outputs.items():
                if name.endswith("_pdf"):
                    self.assertTrue(path.read_bytes().startswith(b"%PDF"), name)
                if name.endswith("_png"):
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG"), name)
                    self.assertGreater(path.stat().st_size, 20_000, name)

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertTrue(manifest["common_random_numbers"]["across_methods"])
            self.assertTrue(manifest["common_random_numbers"]["across_problems"])
            self.assertTrue(
                manifest["common_random_numbers"]["across_initial_learning_rates"]
            )
            self.assertEqual(
                manifest["evaluation_accounting"][
                    "hessian_additional_candidate_evaluations"
                ],
                0,
            )
            self.assertEqual(
                manifest["decision_metrics"]["unreached_target_encoding"],
                "blank/NaN; never encoded as update 301",
            )
            self.assertTrue(manifest["validation"]["all_expected_rows_present"])
            self.assertEqual(
                manifest["validation"]["actual_row_counts"],
                manifest["validation"]["expected_row_counts"],
            )
            for excluded in (
                "trust_region",
                "additive_damping",
                "curvature_clipping",
                "multiplier_clipping",
                "fallback_update",
            ):
                self.assertIn(excluded, manifest["excluded_components"])
            for metadata in manifest["files"].values():
                artifact = Path(temporary) / metadata["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
                self.assertEqual(artifact.stat().st_size, metadata["bytes"])


if __name__ == "__main__":
    unittest.main()

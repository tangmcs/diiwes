#!/usr/bin/env python3
"""Focused tests for the 10-D long convex implicit-ES experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.convex_10d_long_estimated_hessian import (
    DIMENSION,
    METHODS,
    PROBLEMS,
    TARGET_FRACTION,
    ExperimentConfig,
    _optimization_perturbations,
    estimate_gradient_and_hessian,
    learning_rate,
    linearly_implicit_step,
    quadratic_loss,
    run_experiment,
    write_outputs,
)


class Convex10DLongTests(unittest.TestCase):
    def test_fixed_protocol(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(DIMENSION, 10)
        self.assertEqual(METHODS, ("explicit_es", "linearized_implicit_es"))
        self.assertEqual(config.population_size, 2000)
        self.assertEqual(config.updates, 300)
        self.assertEqual(config.seeds, tuple(range(10)))
        self.assertEqual(config.sigma, 0.1)
        self.assertEqual(config.initial_learning_rate, 0.5)
        self.assertEqual(
            config.accuracy_populations, (100, 200, 500, 1000, 2000, 4000)
        )
        self.assertEqual(config.accuracy_replicates, 50)
        self.assertEqual([problem.condition_number for problem in PROBLEMS], [2, 4, 8])
        for problem in PROBLEMS:
            np.testing.assert_allclose(
                problem.hessian_diagonal,
                np.geomspace(1.0, problem.condition_number, DIMENSION),
            )

    def test_shifted_quadratics(self) -> None:
        expected_optimum = np.linspace(-1.5, 1.5, DIMENSION)
        for problem in PROBLEMS:
            optimum = np.asarray(problem.optimum)
            initial = np.asarray(problem.initial_point)
            np.testing.assert_allclose(optimum, expected_optimum)
            np.testing.assert_allclose(initial, expected_optimum + 1.0)
            self.assertGreater(np.linalg.norm(optimum), 0.0)
            self.assertEqual(quadratic_loss(optimum, problem), 0.0)
            self.assertAlmostEqual(
                quadratic_loss(initial, problem),
                0.5 * sum(problem.hessian_diagonal),
            )

    def test_learning_rate_schedule(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(learning_rate(config, 0), 0.5)
        self.assertAlmostEqual(learning_rate(config, 99), 0.05)
        self.assertAlmostEqual(learning_rate(config, 299), 0.5 / np.sqrt(300.0))
        with self.assertRaises(ValueError):
            learning_rate(config, -1)

    def test_raw_estimators_target_gradient_and_hessian(self) -> None:
        problem = PROBLEMS[1]
        point = np.asarray(problem.optimum) + np.linspace(-0.8, 0.7, DIMENSION)
        epsilon = np.random.default_rng(2718).normal(size=(250_000, DIMENSION))
        estimate = estimate_gradient_and_hessian(point, epsilon, 0.1, problem)
        true_hessian = np.asarray(problem.hessian_diagonal)
        true_gradient = true_hessian * (point - np.asarray(problem.optimum))
        np.testing.assert_allclose(
            estimate.gradient, true_gradient, rtol=0.025, atol=0.025
        )
        np.testing.assert_allclose(
            estimate.hessian_diagonal, true_hessian, rtol=0.08, atol=0.08
        )

    def test_signed_implicit_solve_has_no_safeguard(self) -> None:
        gradient = np.linspace(-2.0, 2.0, DIMENSION)
        hessian = np.linspace(-1.5, 2.0, DIMENSION)
        step, denominator, multiplier = linearly_implicit_step(
            gradient, hessian, alpha=0.5
        )
        np.testing.assert_allclose(denominator, 1.0 + 0.5 * hessian)
        np.testing.assert_allclose(multiplier, 1.0 / denominator)
        np.testing.assert_allclose(step, -0.5 * gradient / denominator)
        self.assertGreater(multiplier[0], 1.0)

    def test_quadratic_curvature_estimate_is_state_independent(self) -> None:
        config = replace(ExperimentConfig(), population_size=200)
        problem = PROBLEMS[2]
        epsilon = _optimization_perturbations(config, seed=3, update_index=4)
        first = estimate_gradient_and_hessian(
            np.asarray(problem.initial_point), epsilon, config.sigma, problem
        )
        second = estimate_gradient_and_hessian(
            np.asarray(problem.initial_point) + np.linspace(-0.3, 0.4, DIMENSION),
            epsilon,
            config.sigma,
            problem,
        )
        np.testing.assert_allclose(
            first.hessian_diagonal, second.hessian_diagonal, atol=2e-10, rtol=0.0
        )

    def test_accuracy_improves_with_population(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=2,
            seeds=tuple(range(5)),
            accuracy_populations=(100, 1000),
            accuracy_replicates=15,
        )
        result = run_experiment(config)
        for problem in PROBLEMS:
            values = {
                row["population_size"]: row["relative_rmse_mean"]
                for row in result.curvature_accuracy_summary
                if row["problem"] == problem.key
            }
            self.assertLess(values[1000], values[100])

    def test_output_contract_hashes_and_figures(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=20,
            seeds=(0, 1),
            accuracy_populations=(100, 200),
            accuracy_replicates=3,
        )
        result = run_experiment(config)
        expected_trajectories = (
            len(PROBLEMS) * len(METHODS) * len(config.seeds) * (config.updates + 1)
        )
        expected_curvature = len(PROBLEMS) * len(config.seeds) * config.updates
        expected_accuracy = (
            len(PROBLEMS)
            * len(config.accuracy_populations)
            * len(config.seeds)
            * config.accuracy_replicates
        )
        self.assertEqual(len(result.trajectories), expected_trajectories)
        self.assertEqual(len(result.curvature_updates), expected_curvature)
        self.assertEqual(len(result.curvature_accuracy), expected_accuracy)
        self.assertEqual(len(result.decision_metrics), len(PROBLEMS) * len(METHODS) * 2)
        self.assertEqual(TARGET_FRACTION, 1e-4)
        for row in result.decision_metrics:
            self.assertIn("first_update_fraction_le_1e_minus_4", row)
            self.assertGreaterEqual(row["mean_fraction_initial_loss"], 0.0)
            self.assertLessEqual(
                row["mean_fraction_initial_loss"],
                row["peak_fraction_initial_loss"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            outputs = {
                key: Path(value)
                for key, value in write_outputs(temporary, config, result).items()
            }
            for path in outputs.values():
                self.assertTrue(path.is_file(), path)
            for name in ("optimization_pdf", "accuracy_pdf", "stability_pdf"):
                self.assertTrue(outputs[name].read_bytes().startswith(b"%PDF"))
            for name in ("optimization_png", "accuracy_png", "stability_png"):
                self.assertTrue(outputs[name].read_bytes().startswith(b"\x89PNG"))
                self.assertGreater(outputs[name].stat().st_size, 30_000)

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["methods"]), set(METHODS))
            self.assertTrue(manifest["common_random_numbers"]["across_methods"])
            self.assertTrue(manifest["common_random_numbers"]["across_problems"])
            self.assertEqual(
                manifest["evaluation_accounting"][
                    "hessian_additional_candidate_evaluations"
                ],
                0,
            )
            self.assertIn("curvature_clipping", manifest["excluded_components"])
            self.assertIn("multiplier_clipping", manifest["excluded_components"])
            self.assertEqual(
                manifest["files"]["trajectories"]["rows"], expected_trajectories
            )
            for metadata in manifest["files"].values():
                artifact = Path(temporary) / metadata["path"]
                self.assertEqual(
                    hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    metadata["sha256"],
                )
                self.assertEqual(artifact.stat().st_size, metadata["bytes"])

            with outputs["learning_rates"].open(
                newline="", encoding="utf-8"
            ) as handle:
                schedule = list(csv.DictReader(handle))
            self.assertEqual(len(schedule), config.updates)
            self.assertEqual(float(schedule[0]["alpha_t"]), 0.5)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Focused tests for the advisor-facing convex problem sweep."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.convex_problem_sweep_estimated_hessian import (
    METHODS,
    PROBLEMS,
    ExperimentConfig,
    _optimization_perturbations,
    estimate_gradient_and_hessian,
    learning_rate,
    linearly_implicit_step,
    quadratic_loss,
    run_experiment,
    write_outputs,
)


class ConvexProblemSweepTests(unittest.TestCase):
    def test_fixed_protocol(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(METHODS, ("explicit_es", "linearized_implicit_es"))
        self.assertEqual(config.population_size, 500)
        self.assertEqual(config.updates, 100)
        self.assertEqual(config.seeds, tuple(range(10)))
        self.assertEqual(config.sigma, 0.1)
        self.assertEqual(config.initial_learning_rate, 0.5)
        self.assertEqual(
            [problem.hessian_diagonal for problem in PROBLEMS],
            [(1.0, 2.0), (1.0, 4.0), (1.0, 8.0)],
        )
        self.assertEqual(
            [problem.initial_point for problem in PROBLEMS],
            [(3.0, 0.0), (0.0, 3.0), (2.5, 2.0)],
        )

    def test_learning_rate_schedule(self) -> None:
        config = ExperimentConfig()
        self.assertEqual(learning_rate(config, 0), 0.5)
        self.assertAlmostEqual(learning_rate(config, 1), 0.5 / np.sqrt(2.0))
        self.assertAlmostEqual(learning_rate(config, 9), 0.5 / np.sqrt(10.0))
        self.assertEqual(learning_rate(config, 99), 0.05)
        with self.assertRaises(ValueError):
            learning_rate(config, -1)

    def test_shifted_objectives_have_nonzero_optima(self) -> None:
        for problem in PROBLEMS:
            optimum = np.asarray(problem.optimum)
            initial = np.asarray(problem.initial_point)
            self.assertNotEqual(problem.optimum, (0.0, 0.0))
            self.assertEqual(quadratic_loss(optimum, problem), 0.0)
            expected = 0.5 * sum(problem.hessian_diagonal)
            self.assertEqual(quadratic_loss(initial, problem), expected)

    def test_raw_estimators_target_gradient_and_diagonal_hessian(self) -> None:
        problem = PROBLEMS[1]
        point = np.asarray([0.7, -1.2])
        rng = np.random.default_rng(914)
        estimate = estimate_gradient_and_hessian(
            point,
            rng.normal(size=(200_000, 2)),
            sigma=0.1,
            problem=problem,
        )
        true_gradient = np.asarray(problem.hessian_diagonal) * (
            point - np.asarray(problem.optimum)
        )
        np.testing.assert_allclose(
            estimate.gradient, true_gradient, rtol=0.015, atol=0.025
        )
        np.testing.assert_allclose(
            estimate.hessian_diagonal,
            problem.hessian_diagonal,
            rtol=0.035,
            atol=0.035,
        )

    def test_implicit_update_has_no_projection_or_clipping(self) -> None:
        gradient = np.asarray([2.0, -3.0])
        hessian_estimate = np.asarray([1.5, -0.5])
        step, denominator, multiplier = linearly_implicit_step(
            gradient, hessian_estimate, alpha=0.5
        )
        np.testing.assert_allclose(denominator, [1.75, 0.75])
        np.testing.assert_allclose(multiplier, 1.0 / denominator)
        np.testing.assert_allclose(step, -0.5 * gradient / denominator)
        self.assertGreater(multiplier[1], 1.0)

    def test_common_random_numbers_are_state_independent_for_curvature(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=2,
            seeds=(0, 1),
            accuracy_populations=(20, 100),
            accuracy_replicates=2,
        )
        eps = _optimization_perturbations(config, seed=1, update_index=0)
        problem = PROBLEMS[0]
        first = estimate_gradient_and_hessian(
            np.asarray(problem.initial_point), eps, config.sigma, problem
        )
        second = estimate_gradient_and_hessian(
            np.asarray(problem.initial_point) + np.asarray([0.3, -0.4]),
            eps,
            config.sigma,
            problem,
        )
        # On an exact quadratic the LOO curvature signal removes the
        # state-dependent constant, leaving the same batch Hessian estimate.
        np.testing.assert_allclose(
            first.hessian_diagonal, second.hessian_diagonal, atol=1e-10, rtol=0.0
        )

    def test_accuracy_improves_with_more_candidate_evaluations(self) -> None:
        config = replace(
            ExperimentConfig(),
            seeds=tuple(range(5)),
            updates=2,
            accuracy_populations=(20, 500),
            accuracy_replicates=20,
        )
        result = run_experiment(config)
        for problem in PROBLEMS:
            values = {
                row["population_size"]: row["relative_rmse_mean"]
                for row in result.curvature_accuracy_summary
                if row["problem"] == problem.key
            }
            self.assertLess(values[500], values[20])

    def test_output_contract_hashes_and_figures(self) -> None:
        config = replace(
            ExperimentConfig(),
            population_size=100,
            updates=3,
            seeds=(0, 1),
            accuracy_populations=(20, 100),
            accuracy_replicates=3,
        )
        result = run_experiment(config)
        expected_trajectories = (
            len(PROBLEMS) * len(config.seeds) * len(METHODS) * (config.updates + 1)
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

        with tempfile.TemporaryDirectory() as temporary:
            outputs = {
                key: Path(value)
                for key, value in write_outputs(temporary, config, result).items()
            }
            for path in outputs.values():
                self.assertTrue(path.is_file(), path)
            self.assertTrue(outputs["optimization_pdf"].read_bytes().startswith(b"%PDF"))
            self.assertTrue(outputs["accuracy_pdf"].read_bytes().startswith(b"%PDF"))
            self.assertTrue(outputs["optimization_png"].read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(outputs["accuracy_png"].read_bytes().startswith(b"\x89PNG"))
            self.assertGreater(outputs["optimization_png"].stat().st_size, 50_000)
            self.assertGreater(outputs["accuracy_png"].stat().st_size, 50_000)

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["methods"]), set(METHODS))
            self.assertTrue(manifest["common_random_numbers"]["across_methods"])
            self.assertEqual(
                manifest["evaluation_accounting"]["hessian_additional_candidate_evaluations"],
                0,
            )
            self.assertIn("curvature_clipping", manifest["excluded_components"])
            self.assertEqual(manifest["files"]["trajectories"]["rows"], expected_trajectories)
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

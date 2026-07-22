#!/usr/bin/env python3
"""Tests for the minimal estimated-Hessian convex experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from experiments.simple_convex_estimated_hessian import (
    CURVATURE_FIELDS,
    METHODS,
    ExperimentConfig,
    config_for_problem,
    estimate_gradient_and_diagonal_hessian,
    leave_one_pair_out_baseline,
    linearly_implicit_step,
    quadratic_loss,
    run_experiment,
    write_outputs,
)


class SimpleConvexEstimatedHessianTests(unittest.TestCase):
    def test_leave_one_pair_out_baseline(self) -> None:
        pair_sums = np.asarray([2000.0, 2004.0, 1996.0])
        baseline = leave_one_pair_out_baseline(pair_sums)
        np.testing.assert_allclose(baseline, [2000.0, 1998.0, 2002.0])
        np.testing.assert_allclose(pair_sums - baseline, [0.0, 6.0, -6.0])

    def test_raw_estimators_target_known_gradient_and_hessian(self) -> None:
        point = np.asarray([1.5, -0.75])
        hessian = np.asarray([1.0, 2.0])
        rng = np.random.default_rng(39)
        estimate = estimate_gradient_and_diagonal_hessian(
            point,
            rng.normal(size=(200_000, 2)),
            sigma=0.1,
            hessian_diagonal=hessian,
        )
        np.testing.assert_allclose(
            estimate.gradient, hessian * point, rtol=0.015, atol=0.015
        )
        np.testing.assert_allclose(
            estimate.hessian_diagonal, hessian, rtol=0.035, atol=0.025
        )

    def test_shifted_problem_targets_displacement_gradient(self) -> None:
        config = config_for_problem("shifted")
        self.assertEqual(config.initial_point, (3.0, 0.0))
        self.assertEqual(config.optimum_point, (2.0, -1.0))
        self.assertEqual(
            config_for_problem("origin"),
            ExperimentConfig(),
            "the original experiment must remain the default",
        )
        hessian = np.asarray(config.hessian_diagonal)
        optimum = np.asarray(config.optimum_point)
        point = np.asarray([0.5, 0.25])
        self.assertEqual(quadratic_loss(optimum, hessian, optimum), 0.0)
        self.assertEqual(
            quadratic_loss(np.asarray(config.initial_point), hessian, optimum),
            1.5,
        )

        rng = np.random.default_rng(72)
        estimate = estimate_gradient_and_diagonal_hessian(
            point,
            rng.normal(size=(200_000, 2)),
            sigma=config.sigma,
            hessian_diagonal=hessian,
            optimum_point=optimum,
        )
        np.testing.assert_allclose(
            estimate.gradient,
            hessian * (point - optimum),
            rtol=0.015,
            atol=0.02,
        )
        np.testing.assert_allclose(
            estimate.hessian_diagonal, hessian, rtol=0.035, atol=0.025
        )

    def test_implicit_step_is_unsafeguarded_linear_solution(self) -> None:
        gradient = np.asarray([2.0, -3.0])
        estimate = np.asarray([1.5, -0.25])
        alpha = 1.0
        step, denominator, multiplier = linearly_implicit_step(
            gradient, estimate, alpha
        )
        np.testing.assert_allclose(denominator, 1.0 + alpha * estimate)
        np.testing.assert_allclose(multiplier, 1.0 / denominator)
        np.testing.assert_allclose(step, -alpha * gradient / denominator)
        # Negative estimated curvature is intentionally neither projected nor
        # clipped: its multiplier is greater than one.
        self.assertGreater(multiplier[1], 1.0)

    def test_common_random_numbers_and_exactly_two_methods(self) -> None:
        config = ExperimentConfig(
            population_size=100,
            iterations=3,
            seeds=(0, 1),
        )
        result = run_experiment(config)
        self.assertEqual({row["method"] for row in result.runs}, set(METHODS))

        # On this quadratic the LOO diagonal estimate is state independent.
        # Equality across alpha therefore verifies the shared perturbation batch.
        for seed in config.seeds:
            for update in range(1, config.iterations + 1):
                matched = [
                    row
                    for row in result.curvature
                    if row["seed"] == seed and row["update"] == update
                ]
                self.assertEqual(len(matched), 2)
                np.testing.assert_allclose(
                    [matched[0]["h11_estimate"], matched[0]["h22_estimate"]],
                    [matched[1]["h11_estimate"], matched[1]["h22_estimate"]],
                    rtol=0.0,
                    atol=1e-10,
                )

    def test_default_protocol_shows_the_stability_boundary(self) -> None:
        config = ExperimentConfig()
        result = run_experiment(config)

        def median_final(alpha: float, method: str) -> float:
            return float(
                np.median(
                    [
                        row["loss"]
                        for row in result.runs
                        if row["alpha"] == alpha
                        and row["method"] == method
                        and row["update"] == config.iterations
                    ]
                )
            )

        safe, boundary = config.alphas
        self.assertLess(
            median_final(safe, "explicit_es"),
            median_final(safe, "linearized_implicit_es"),
        )
        self.assertGreater(median_final(boundary, "explicit_es"), 1e-2)
        self.assertLess(
            median_final(boundary, "linearized_implicit_es"), 1e-12
        )

    def test_shifted_protocol_converges_to_nonzero_optimum(self) -> None:
        config = config_for_problem("shifted")
        result = run_experiment(config)
        initial_rows = [row for row in result.runs if row["update"] == 0]
        self.assertTrue(initial_rows)
        self.assertTrue(all(row["x1"] == 3.0 for row in initial_rows))
        self.assertTrue(all(row["x2"] == 0.0 for row in initial_rows))
        self.assertTrue(all(row["loss"] == 1.5 for row in initial_rows))

        final_implicit = np.asarray(
            [
                [row["x1"], row["x2"]]
                for row in result.runs
                if row["alpha"] == config.alphas[1]
                and row["method"] == "linearized_implicit_es"
                and row["update"] == config.iterations
            ]
        )
        np.testing.assert_allclose(
            np.median(final_implicit, axis=0),
            config.optimum_point,
            rtol=0.0,
            atol=1e-8,
        )

    def test_shift_is_exactly_translation_invariant(self) -> None:
        origin_config = replace(
            config_for_problem("origin"),
            population_size=100,
            iterations=3,
            seeds=(0, 1),
        )
        shifted_config = replace(
            config_for_problem("shifted"),
            population_size=100,
            iterations=3,
            seeds=(0, 1),
        )
        origin = run_experiment(origin_config)
        shifted = run_experiment(shifted_config)
        self.assertEqual(len(origin.runs), len(shifted.runs))
        for origin_row, shifted_row in zip(origin.runs, shifted.runs, strict=True):
            self.assertEqual(
                (origin_row["seed"], origin_row["alpha"], origin_row["method"], origin_row["update"]),
                (shifted_row["seed"], shifted_row["alpha"], shifted_row["method"], shifted_row["update"]),
            )
            self.assertAlmostEqual(origin_row["loss"], shifted_row["loss"], places=12)
            self.assertAlmostEqual(
                origin_row["x1"] + shifted_config.optimum_point[0],
                shifted_row["x1"],
                places=12,
            )
            self.assertAlmostEqual(
                origin_row["x2"] + shifted_config.optimum_point[1],
                shifted_row["x2"],
                places=12,
            )
        for origin_row, shifted_row in zip(
            origin.curvature, shifted.curvature, strict=True
        ):
            for field in CURVATURE_FIELDS:
                self.assertAlmostEqual(origin_row[field], shifted_row[field], places=10)

    def test_output_contract_and_advisor_figure(self) -> None:
        config = ExperimentConfig(
            population_size=100,
            iterations=3,
            seeds=(0, 1),
        )
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = {
                key: Path(value)
                for key, value in write_outputs(temporary, config, result).items()
            }
            for path in outputs.values():
                self.assertTrue(path.is_file(), path)

            with outputs["runs"].open(newline="", encoding="utf-8") as handle:
                run_rows = list(csv.DictReader(handle))
            with outputs["curvature"].open(
                newline="", encoding="utf-8"
            ) as handle:
                curvature_rows = list(csv.DictReader(handle))
            self.assertEqual(
                len(run_rows),
                len(config.alphas)
                * len(config.seeds)
                * len(METHODS)
                * (config.iterations + 1),
            )
            self.assertEqual(
                len(curvature_rows),
                len(config.alphas) * len(config.seeds) * config.iterations,
            )

            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["methods"]), set(METHODS))
            self.assertTrue(manifest["common_random_numbers"]["across_methods"])
            self.assertIn("curvature_clipping", manifest["excluded_components"])
            self.assertNotIn("normalized_gap_auc", json.dumps(manifest))
            for metadata in manifest["files"].values():
                artifact = Path(temporary) / metadata["path"]
                digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
                self.assertEqual(digest, metadata["sha256"])
                self.assertEqual(artifact.stat().st_size, metadata["bytes"])

            source = Path(manifest["provenance"]["source_file"])
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                manifest["provenance"]["source_sha256"],
            )

            self.assertTrue(outputs["figure_pdf"].read_bytes().startswith(b"%PDF"))
            self.assertTrue(outputs["figure_png"].read_bytes().startswith(b"\x89PNG"))
            self.assertGreater(outputs["figure_pdf"].stat().st_size, 10_000)
            self.assertGreater(outputs["figure_png"].stat().st_size, 50_000)

    def test_shifted_outputs_identify_objective_and_optimum(self) -> None:
        config = ExperimentConfig(
            population_size=100,
            iterations=3,
            seeds=(0, 1),
            initial_point=(3.0, 0.0),
            optimum_point=(2.0, -1.0),
            problem_name="shifted",
        )
        result = run_experiment(config)
        with tempfile.TemporaryDirectory() as temporary:
            outputs = {
                key: Path(value)
                for key, value in write_outputs(temporary, config, result).items()
            }
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["problem"]["name"], "shifted")
            self.assertEqual(manifest["problem"]["initial_point"], [3.0, 0.0])
            self.assertEqual(manifest["problem"]["optimum_point"], [2.0, -1.0])
            self.assertIn("(x1 - 2)^2", manifest["problem"]["objective"])

            report = outputs["report"].read_text(encoding="utf-8")
            self.assertIn("optimum: `(2.0, -1.0)`", report)
            self.assertIn("translation-invariance check", report)
            self.assertIn("Final median x1", report)
            self.assertIn("Final median x2", report)


if __name__ == "__main__":
    unittest.main()

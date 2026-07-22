#!/usr/bin/env python3
"""Tests for the isolated curvature sample-size benchmark."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest

import numpy as np

from experiments.curvature_sample_size.benchmark import (
    CASES,
    BenchmarkConfig,
    build_report_artifact,
    estimate_with_repository_method,
    nonlinear_nonconvex,
    nonlinear_parameters,
    nonlinear_smoothed_diagonal_hessian,
    run_benchmark,
    write_outputs,
)
from experiments.curvature_sample_size.dimension_sweep import (
    DimensionSweepConfig,
    build_report_artifact as build_dimension_report_artifact,
    run_dimension_sweep,
    write_outputs as write_dimension_outputs,
)


class CurvatureSampleSizeBenchmarkTests(unittest.TestCase):
    def test_smoothed_nonlinear_hessian_matches_finite_differences(self) -> None:
        dimension = 5
        sigma = 0.13
        theta, frequencies = nonlinear_parameters(dimension)
        analytic = nonlinear_smoothed_diagonal_hessian(theta, frequencies, sigma)

        def smoothed(point: np.ndarray) -> float:
            attenuation = np.exp(-0.5 * frequencies**2 * sigma**2)
            sine = np.sum(attenuation * np.sin(frequencies * point))
            quartic = 0.05 * np.sum(
                point**4 + 6.0 * point**2 * sigma**2 + 3.0 * sigma**4
            )
            coupling = 0.08 * np.sum(point[:-1] * point[1:])
            return float(sine + quartic + coupling)

        step = 2e-4
        numerical = np.empty(dimension)
        center = smoothed(theta)
        for coordinate in range(dimension):
            plus = theta.copy()
            minus = theta.copy()
            plus[coordinate] += step
            minus[coordinate] -= step
            numerical[coordinate] = (
                smoothed(plus) - 2.0 * center + smoothed(minus)
            ) / step**2
        np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-6)

    def test_function_is_nonlinear_and_locally_nonconvex(self) -> None:
        theta, frequencies = nonlinear_parameters(12)
        sigma = 0.1
        hessian = nonlinear_smoothed_diagonal_hessian(theta, frequencies, sigma)
        self.assertTrue(np.any(hessian > 0.0))
        self.assertTrue(np.any(hessian < 0.0))
        midpoint = 0.5 * (
            nonlinear_nonconvex(theta - 0.2, frequencies)
            + nonlinear_nonconvex(theta + 0.2, frequencies)
        )
        self.assertNotAlmostEqual(
            float(nonlinear_nonconvex(theta, frequencies)), float(midpoint), places=6
        )

    def test_repository_estimator_cancels_linear_function(self) -> None:
        rng = np.random.default_rng(4)
        eps = rng.standard_normal((64, 6))
        coefficients = np.linspace(-1.0, 1.0, 6)
        sigma = 0.1
        plus = (sigma * eps) @ coefficients
        minus = (-sigma * eps) @ coefficients
        estimate = estimate_with_repository_method(eps, plus, minus, sigma)
        np.testing.assert_allclose(estimate, 0.0, rtol=0.0, atol=1e-12)

    def test_small_run_and_output_contract(self) -> None:
        config = BenchmarkConfig(
            dimension=4,
            pair_counts=(4, 8, 16),
            repetitions=4,
            rate_fit_min_pairs=4,
            master_seed=9,
        )
        result = run_benchmark(config)
        self.assertEqual(
            len(result.run_metrics), len(CASES) * config.repetitions * 3
        )
        self.assertEqual(
            len(result.coordinate_estimates),
            len(result.run_metrics) * config.dimension,
        )
        self.assertEqual(len(result.aggregates), len(CASES) * 3)
        self.assertEqual(len(result.convergence_rates), len(CASES))
        deterministic = [
            row for row in result.aggregates if row["case"] == "linear_deterministic"
        ]
        self.assertLess(max(row["median_max_abs_error"] for row in deterministic), 1e-10)

        with tempfile.TemporaryDirectory() as directory:
            paths = write_outputs(result, config, directory)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
            with paths["aggregate.csv"].open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(CASES) * 3)
            manifest = json.loads(paths["experiment_manifest.json"].read_text())
            self.assertEqual(manifest["experiment_version"], "1.0.0")
            self.assertEqual(
                manifest["estimator_contract"]["implementation"],
                "core.DIIWES._estimate_fresh_curvature",
            )
            artifact = json.loads(paths["artifact.json"].read_text())
            self.assertEqual(artifact["surface"], "report")
            self.assertEqual(artifact["snapshot"]["status"], "ready")
            self.assertIn("relative_error_curve", artifact["snapshot"]["datasets"])

    def test_report_artifact_has_technical_reading_path(self) -> None:
        config = BenchmarkConfig(
            dimension=3,
            pair_counts=(4, 8, 16),
            repetitions=3,
            rate_fit_min_pairs=4,
            master_seed=10,
        )
        result = run_benchmark(config)
        artifact = build_report_artifact(result, config, "2026-07-21T00:00:00+00:00")
        blocks = artifact["manifest"]["blocks"]
        self.assertEqual(blocks[0]["body"], "# Curvature estimation versus sample size")
        headings = [
            block.get("body", "").split("\n", 1)[0]
            for block in blocks
            if block["type"] == "markdown"
        ]
        for expected in (
            "## The estimator works on this controlled nonlinear test",
            "## What was measured",
            "## The nonlinear target is analytic and the estimator is not duplicated",
            "## This validates a local diagonal estimate, not full optimizer performance",
            "## Further questions",
        ):
            self.assertIn(expected, headings)

    def test_dimension_sweep_is_compact_and_report_has_two_plots(self) -> None:
        config = DimensionSweepConfig(
            dimensions=(3, 5),
            pair_counts=(4, 8, 16),
            repetitions=3,
            rate_fit_min_pairs=4,
            master_seed=11,
        )
        result = run_dimension_sweep(config)
        expected_runs = len(config.dimensions) * len(CASES) * 3 * 3
        self.assertEqual(len(result.run_metrics), expected_runs)
        self.assertEqual(len(result.aggregates), len(config.dimensions) * len(CASES) * 3)
        self.assertEqual(
            len(result.convergence_rates), len(config.dimensions) * len(CASES)
        )
        self.assertNotIn("coordinate_estimates", result.__dataclass_fields__)
        self.assertEqual(result.scaling_model["rows_in_fit"], 6)

        artifact = build_dimension_report_artifact(
            result, config, "2026-07-21T00:00:00+00:00"
        )
        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(len(artifact["manifest"]["charts"]), 2)
        self.assertIn(
            "absolute_rmse_by_dimension", artifact["snapshot"]["datasets"]
        )
        self.assertIn(
            "normalized_rmse_by_dimension", artifact["snapshot"]["datasets"]
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_dimension_outputs(result, config, directory)
            self.assertNotIn("coordinate_estimates.csv", paths)
            for path in paths.values():
                self.assertTrue(path.is_file(), path)
            manifest = json.loads(paths["experiment_manifest.json"].read_text())
            self.assertEqual(
                manifest["memory_contract"]["coordinate_storage"],
                "online error sums only",
            )


if __name__ == "__main__":
    unittest.main()

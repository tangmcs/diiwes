from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import implicit_quadratic_optimization_benchmark as benchmark
from scripts.plotting import analyze_plot_convex_estimated_hessian as analysis


def _write_quick_artifact(
    root: Path,
    *,
    iterations: int = 4,
    cases: tuple[str, ...] = (
        "block_aligned_concave",
        "rotated_concave",
        "block_aligned_additive_noise",
    ),
) -> tuple[benchmark.BenchmarkConfig, benchmark.BenchmarkResult, Path]:
    config = benchmark.BenchmarkConfig(
        dimension=6,
        num_blocks=3,
        population_size=12,
        iterations=iterations,
        sigma=0.1,
        alphas=(0.5, 1.0),
        mc_seeds=(0, 1),
        cases=cases,
        fitness_transforms=("raw",),
        additive_noise_std=0.05,
        master_seed=20260715,
    )
    result = benchmark.run_benchmark(config)
    artifact = root / "benchmark"
    benchmark.write_outputs(artifact, config, result)
    return config, result, artifact


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_end_to_end_analysis_outputs_are_auditable(tmp_path: Path) -> None:
    config, result, artifact = _write_quick_artifact(tmp_path)
    output = tmp_path / "analysis"

    paths = analysis.analyze(artifact, output, checkpoints=(2, 4))

    checkpoint_path = Path(paths["checkpoint_runs"])
    aggregate_path = Path(paths["checkpoint_aggregate"])
    with checkpoint_path.open(newline="") as stream:
        checkpoint_rows = list(csv.DictReader(stream))
    assert len(checkpoint_rows) == len(result.summaries) * 2
    assert {int(row["checkpoint"]) for row in checkpoint_rows} == {2, 4}

    explicit_source = [
        row
        for row in result.trajectories
        if row["regime"] == "monte_carlo_es"
        and row["fitness_transform"] == "raw"
        and row["case"] == "block_aligned_concave"
        and row["method"] == "explicit_es"
        and float(row["alpha"]) == 0.5
        and int(row["seed"]) == 0
        and int(row["iteration"]) <= 2
    ]
    explicit_source.sort(key=lambda row: int(row["iteration"]))
    ratios = np.asarray(
        [float(row["objective_gap"]) for row in explicit_source]
    ) / float(explicit_source[0]["objective_gap"])
    expected_auc = (0.5 * ratios[0] + ratios[1] + 0.5 * ratios[2]) / 2.0
    derived = next(
        row
        for row in checkpoint_rows
        if row["regime"] == "monte_carlo_es"
        and row["case"] == "block_aligned_concave"
        and row["method"] == "explicit_es"
        and float(row["alpha"]) == 0.5
        and int(row["seed"]) == 0
        and int(row["checkpoint"]) == 2
    )
    assert float(derived["normalized_gap_auc"]) == pytest.approx(expected_auc)

    with aggregate_path.open(newline="") as stream:
        aggregate_rows = list(csv.DictReader(stream))
    assert aggregate_rows
    assert {row["method"] for row in aggregate_rows}.issuperset(
        analysis.PRIMARY_METHODS
    )

    with np.load(paths["curvature_estimates"], allow_pickle=False) as archive:
        assert archive["reward_hessian_diagonal_estimate"].shape == (
            len(config.cases),
            len(config.alphas),
            len(config.mc_seeds),
            config.iterations,
            config.dimension,
        )
        assert archive["active_mask"].dtype == np.bool_
        assert np.any(archive["active_mask"])
        assert np.allclose(
            archive["true_reward_hessian_diagonal"],
            np.stack([case.diagonal for case in benchmark.make_cases(config)]),
        )
        assert np.allclose(
            archive["convex_loss_curvature_estimate"],
            -archive["reward_hessian_diagonal_estimate"],
            equal_nan=True,
        )
        assert np.allclose(
            archive["projected_convex_loss_curvature"],
            np.maximum(archive["convex_loss_curvature_estimate"], 0.0),
            equal_nan=True,
        )
        alpha = archive["alphas"][None, :, None, None, None]
        assert np.allclose(
            archive["projected_diagonal_multiplier"],
            1.0
            / (
                1.0
                + alpha * archive["projected_convex_loss_curvature"]
            ),
            equal_nan=True,
        )
        assert np.allclose(
            archive["true_convex_loss_curvature_diagonal"],
            -archive["true_reward_hessian_diagonal"],
        )

    curvature_index = json.loads(Path(paths["curvature_index"]).read_text())
    assert curvature_index["npz_sha256"] == _sha256(
        Path(paths["curvature_estimates"])
    )
    assert (
        curvature_index["presentation_sign_convention"]
        == "convex_loss_curvature_equals_negative_reward_hessian"
    )
    assert "projected_diagonal_multiplier" in curvature_index["presentation_arrays"]

    with Path(paths["curvature_diagnostics"]).open(newline="") as stream:
        curvature_diagnostics = list(csv.DictReader(stream))
    assert len(curvature_diagnostics) == len(config.cases) * len(config.alphas)
    alpha_one = [
        row for row in curvature_diagnostics if float(row["alpha"]) == 1.0
    ]
    assert len(alpha_one) == len(config.cases)
    assert all(0.0 <= float(row["median_projected_multiplier"]) <= 1.0 for row in alpha_one)
    assert all(0.0 <= float(row["no_attenuation_fraction"]) <= 1.0 for row in alpha_one)

    for stem in (
        "step_size_robustness",
        "alpha1_trajectories",
        "curvature_calibration",
    ):
        pdf = output / "figures" / f"{stem}.pdf"
        png = output / "figures" / f"{stem}.png"
        assert pdf.read_bytes().startswith(b"%PDF")
        assert png.read_bytes().startswith(b"\x89PNG")

    manifest = json.loads(Path(paths["analysis_manifest"]).read_text())
    assert manifest["validation"]["benchmark_manifest_validated"] is True
    assert manifest["validation"]["curvature_replay"]["max_relative_gap_error"] <= 1e-10
    for metadata in manifest["outputs"].values():
        path = output / metadata["path"]
        assert metadata["sha256"] == _sha256(path)
    summary = Path(paths["academic_summary"]).read_text()
    assert "Estimated-Hessian Implicit Optimization" in summary
    assert "convex-loss curvature" in summary
    assert "Full-Hessian oracle" in summary


def test_manifest_hash_tampering_is_rejected(tmp_path: Path) -> None:
    _, _, artifact = _write_quick_artifact(
        tmp_path,
        iterations=2,
        cases=("block_aligned_concave",),
    )
    with (artifact / "trajectories.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")

    with pytest.raises(analysis.AnalysisError, match="SHA-256 mismatch"):
        analysis.validate_benchmark_artifact(artifact)

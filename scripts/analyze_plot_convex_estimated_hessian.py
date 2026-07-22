#!/usr/bin/env python3
"""Analyze and plot the population convex estimated-Hessian benchmark.

The analyzer is deliberately separate from the benchmark.  It validates the
immutable benchmark artifact, streams trajectories one run at a time, replays
the raw diagonal Stein estimator at the explicit-ES reference states, and
writes presentation-ready static figures plus auditable derived tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Matplotlib must not use the unwritable home cache on the cluster.
if "MPLCONFIGDIR" not in os.environ:
    mpl_root = Path(tempfile.gettempdir()) / f"diiwes-mpl-{os.getuid()}"
    mpl_root.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_root)
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from experiments import implicit_quadratic_optimization_benchmark as benchmark  # noqa: E402


ANALYZER_VERSION = "1.1.0"
DEFAULT_CHECKPOINTS = (10, 30, 100, 300, 500)
REPLAY_GAP_RELATIVE_TOLERANCE = 1e-10
REPLAY_RMSE_ABSOLUTE_TOLERANCE = 1e-8
PRIMARY_METHODS = (
    "explicit_es",
    "sampled_signed_diagonal",
    "concave_projected_diagonal",
    "diagonal_norm_matched_isotropic",
    "oracle_true_diagonal",
    "oracle_full_implicit",
)
METHOD_LABELS = {
    "explicit_es": "Explicit ES",
    "sampled_signed_diagonal": "Estimated diagonal",
    "concave_projected_diagonal": "Projected estimated diagonal",
    "diagonal_norm_matched_isotropic": "Equal-norm scalar",
    "oracle_true_diagonal": "True-diagonal implicit",
    "oracle_full_implicit": "Full-Hessian oracle",
}
METHOD_STYLES = {
    "explicit_es": dict(color="#4b5563", linestyle="--", marker="o"),
    "sampled_signed_diagonal": dict(color="#2563eb", linestyle="-", marker="o"),
    "concave_projected_diagonal": dict(
        color="#174a7e", linestyle="-", marker="^"
    ),
    "diagonal_norm_matched_isotropic": dict(
        color="#d97706", linestyle="-.", marker="s"
    ),
    "oracle_true_diagonal": dict(color="#9ca3af", linestyle=":", marker="D"),
    "oracle_full_implicit": dict(color="#111827", linestyle="--", marker="x"),
}
CASE_LABELS = {
    "block_aligned_concave": "Aligned",
    "rotated_concave": "Rotated",
    "block_aligned_additive_noise": "Additive noise",
    "rotated_indefinite": "Indefinite",
}

CHECKPOINT_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "method",
    "alpha",
    "seed",
    "checkpoint",
    "diverged_by_checkpoint",
    "divergence_iteration",
    "initial_gap",
    "gap_at_checkpoint",
    "gap_ratio_at_checkpoint",
    "normalized_gap_auc",
)
AGGREGATE_FIELDS = (
    "regime",
    "fitness_transform",
    "case",
    "method",
    "alpha",
    "checkpoint",
    "n_runs",
    "n_diverged",
    "n_nondiverged",
    "divergence_rate",
    "divergence_wilson_low",
    "divergence_wilson_high",
    "mean_gap_ratio",
    "sd_gap_ratio",
    "mean_normalized_gap_auc",
    "sd_normalized_gap_auc",
    "ci95_low_normalized_gap_auc",
    "ci95_high_normalized_gap_auc",
    "nondiverged_mean_gap_ratio",
    "nondiverged_mean_normalized_gap_auc",
)
CURVATURE_DIAGNOSTIC_FIELDS = (
    "case",
    "alpha",
    "n_active_batches",
    "n_coordinate_estimates",
    "mean_curvature_bias",
    "coordinate_curvature_rmse",
    "median_batch_curvature_rmse",
    "p95_batch_curvature_rmse",
    "nonpositive_curvature_estimate_fraction",
    "signed_denominator_nonpositive_fraction",
    "signed_denominator_abs_below_0_1_fraction",
    "mean_projected_multiplier",
    "median_projected_multiplier",
    "p05_projected_multiplier",
    "p95_projected_multiplier",
    "no_attenuation_fraction",
)


class AnalysisError(ValueError):
    """Raised when an input artifact or derived invariant is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            next(reader)
        except StopIteration as error:
            raise AnalysisError(f"empty CSV: {path}") from error
        return sum(1 for _ in reader)


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise AnalysisError(f"artifact path escapes result directory: {relative}") from error
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AnalysisError(f"non-finite JSON constant {value!r} in {path}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise AnalysisError(f"expected a JSON object in {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _atomic_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_figure(fig: Any, path: Path, fmt: str, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
    os.replace(temporary, path)


def _config_from_manifest(manifest: dict[str, Any]) -> benchmark.BenchmarkConfig:
    raw = dict(manifest.get("config", {}))
    for name in ("alphas", "mc_seeds", "cases", "fitness_transforms"):
        if name in raw:
            raw[name] = tuple(raw[name])
    try:
        config = benchmark.BenchmarkConfig(**raw)
        config.validate()
    except (TypeError, ValueError) as error:
        raise AnalysisError(f"invalid benchmark config: {error}") from error
    return config


def validate_benchmark_artifact(input_dir: Path) -> tuple[dict[str, Any], benchmark.BenchmarkConfig, dict[str, Path], dict[str, Any]]:
    input_dir = input_dir.resolve()
    manifest_path = input_dir / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise AnalysisError(f"missing benchmark manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, dict) or "trajectories" not in files:
        raise AnalysisError("benchmark manifest has no trajectories artifact")

    resolved: dict[str, Path] = {}
    checks: dict[str, Any] = {}
    for key, metadata in files.items():
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            raise AnalysisError(f"invalid file metadata for {key}")
        path = _safe_child(input_dir, metadata["path"])
        if not path.is_file():
            raise AnalysisError(f"missing declared artifact: {path}")
        actual_hash = _sha256(path)
        actual_bytes = path.stat().st_size
        actual_rows = _csv_rows(path)
        if actual_hash != metadata.get("sha256"):
            raise AnalysisError(f"SHA-256 mismatch for {path.name}")
        if actual_bytes != int(metadata.get("bytes", -1)):
            raise AnalysisError(f"byte-size mismatch for {path.name}")
        if actual_rows != int(metadata.get("rows", -1)):
            raise AnalysisError(f"row-count mismatch for {path.name}")
        resolved[key] = path
        checks[key] = {"sha256": actual_hash, "bytes": actual_bytes, "rows": actual_rows}

    provenance = manifest.get("provenance", {})
    source_rel = provenance.get("source_file")
    source_hash = provenance.get("source_sha256")
    if not isinstance(source_rel, str) or not isinstance(source_hash, str):
        raise AnalysisError("benchmark provenance is missing source identity")
    source_path = _safe_child(REPO_ROOT, source_rel)
    if not source_path.is_file() or _sha256(source_path) != source_hash:
        raise AnalysisError("checked-out benchmark source does not match the run manifest")

    config = _config_from_manifest(manifest)
    if tuple(config.fitness_transforms) != ("raw",):
        raise AnalysisError("this analyzer requires the raw-fitness Hessian regime only")
    missing = [name for name in PRIMARY_METHODS if name not in manifest.get("methods", {})]
    if missing:
        raise AnalysisError(f"benchmark predates required diagonal controls: {missing}")
    validation = manifest.get("validation", {})
    if not validation.get("complete_matrix") or not validation.get("strict_finite_numbers"):
        raise AnalysisError("benchmark manifest does not certify a complete finite matrix")
    return manifest, config, resolved, {"manifest_sha256": _sha256(manifest_path), "files": checks}


def _bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise AnalysisError(f"invalid Boolean {value!r} in {field}")


def _mean_trapezoid(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise AnalysisError("prefix AUC needs at least two observations")
    array = np.asarray(values, dtype=np.float64)
    return float((0.5 * array[0] + np.sum(array[1:-1]) + 0.5 * array[-1]) / (len(array) - 1))


def _run_key(row: dict[str, str]) -> tuple[str, str, str, str, float, int]:
    return (
        row["regime"],
        row["fitness_transform"],
        row["case"],
        row["method"],
        float(row["alpha"]),
        int(row["seed"]),
    )


def _summarize_run(key: tuple[str, str, str, str, float, int], rows: Sequence[dict[str, str]], checkpoints: Sequence[int], total_iterations: int) -> list[dict[str, Any]]:
    iterations = [int(row["iteration"]) for row in rows]
    if iterations != list(range(total_iterations + 1)):
        raise AnalysisError(f"trajectory is incomplete or unordered for run {key}")
    if any(row["objective_gap"] == "" for row in rows):
        return []
    gaps = np.asarray([float(row["objective_gap"]) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(gaps)) or gaps[0] <= 0.0:
        raise AnalysisError(f"invalid objective gaps for run {key}")
    divergence_iterations = [
        int(row["iteration"]) for row in rows if _bool(row["diverged"], "diverged")
    ]
    first_divergence = min(divergence_iterations) if divergence_iterations else None
    output: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        ratios = gaps[: checkpoint + 1] / gaps[0]
        output.append(
            {
                "regime": key[0],
                "fitness_transform": key[1],
                "case": key[2],
                "method": key[3],
                "alpha": key[4],
                "seed": key[5],
                "checkpoint": checkpoint,
                "diverged_by_checkpoint": first_divergence is not None and first_divergence <= checkpoint,
                "divergence_iteration": first_divergence,
                "initial_gap": float(gaps[0]),
                "gap_at_checkpoint": float(gaps[checkpoint]),
                "gap_ratio_at_checkpoint": float(ratios[-1]),
                "normalized_gap_auc": _mean_trapezoid(ratios),
            }
        )
    return output


def stream_trajectories(path: Path, config: benchmark.BenchmarkConfig, checkpoints: Sequence[int]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], list[dict[str, Any]]], dict[tuple[str, str, int], list[float]]]:
    required = {
        "regime", "fitness_transform", "case", "method", "alpha", "seed",
        "iteration", "objective_gap", "active_update", "diverged", "curvature_diag_rmse",
    }
    checkpoint_rows: list[dict[str, Any]] = []
    explicit_reference: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    curves: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    seen: set[tuple[str, str, str, str, float, int]] = set()
    current_key: tuple[str, str, str, str, float, int] | None = None
    current_rows: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal current_key, current_rows
        if current_key is None:
            return
        checkpoint_rows.extend(_summarize_run(current_key, current_rows, checkpoints, config.iterations))
        regime, transform, case, method, alpha, seed = current_key
        if (
            regime == "monte_carlo_es" and transform == "raw"
            and method == "explicit_es"
        ):
            compact = []
            for row in current_rows:
                compact.append(
                    {
                        "iteration": int(row["iteration"]),
                        "objective_gap": None if row["objective_gap"] == "" else float(row["objective_gap"]),
                        "active_update": _bool(row["active_update"], "active_update"),
                        "diverged": _bool(row["diverged"], "diverged"),
                        "curvature_diag_rmse": None if row["curvature_diag_rmse"] == "" else float(row["curvature_diag_rmse"]),
                    }
                )
            explicit_reference[(case, format(alpha, ".17g"), seed)] = compact
        if (
            regime == "monte_carlo_es" and transform == "raw"
            and method in PRIMARY_METHODS and math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-12)
            and current_rows[0]["objective_gap"] != ""
        ):
            initial = float(current_rows[0]["objective_gap"])
            for row in current_rows:
                ratio = float(row["objective_gap"]) / initial
                curves[(case, method, int(row["iteration"]))].append(ratio)
        current_rows = []

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise AnalysisError(f"trajectory schema is missing {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = _run_key(row)
            if current_key is None:
                current_key = key
                seen.add(key)
            elif key != current_key:
                flush()
                if key in seen:
                    raise AnalysisError(f"trajectory run is not contiguous: {key}")
                seen.add(key)
                current_key = key
            current_rows.append(row)
    flush()
    if not checkpoint_rows:
        raise AnalysisError("no finite-optimum checkpoint rows were derived")
    return checkpoint_rows, explicit_reference, curves


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else None
    return mean, sd


def _t_critical_95(df: int) -> float:
    table = (math.nan, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)
    return table[df] if 1 <= df <= 30 else 1.96


def _mean_ci(values: Sequence[float]) -> tuple[float | None, float | None, float | None, float | None]:
    mean, sd = _mean_sd(values)
    if mean is None or sd is None:
        return mean, sd, None, None
    half = _t_critical_95(len(values) - 1) * sd / math.sqrt(len(values))
    return mean, sd, mean - half, mean + half


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise AnalysisError("Wilson interval needs a positive denominator")
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def aggregate_checkpoints(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["regime"], row["fitness_transform"], row["case"], row["method"], float(row["alpha"]), int(row["checkpoint"]))
        groups[key].append(row)
    output = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        members = groups[key]
        diverged = [row for row in members if row["diverged_by_checkpoint"]]
        nondiverged = [row for row in members if not row["diverged_by_checkpoint"]]
        gap = [float(row["gap_ratio_at_checkpoint"]) for row in members]
        auc = [float(row["normalized_gap_auc"]) for row in members]
        mean_gap, sd_gap = _mean_sd(gap)
        mean_auc, sd_auc, ci_low, ci_high = _mean_ci(auc)
        wilson_low, wilson_high = _wilson(len(diverged), len(members))
        output.append({
            "regime": key[0], "fitness_transform": key[1], "case": key[2],
            "method": key[3], "alpha": key[4], "checkpoint": key[5],
            "n_runs": len(members), "n_diverged": len(diverged), "n_nondiverged": len(nondiverged),
            "divergence_rate": len(diverged) / len(members),
            "divergence_wilson_low": wilson_low, "divergence_wilson_high": wilson_high,
            "mean_gap_ratio": mean_gap, "sd_gap_ratio": sd_gap,
            "mean_normalized_gap_auc": mean_auc, "sd_normalized_gap_auc": sd_auc,
            "ci95_low_normalized_gap_auc": ci_low, "ci95_high_normalized_gap_auc": ci_high,
            "nondiverged_mean_gap_ratio": _mean_sd([float(row["gap_ratio_at_checkpoint"]) for row in nondiverged])[0],
            "nondiverged_mean_normalized_gap_auc": _mean_sd([float(row["normalized_gap_auc"]) for row in nondiverged])[0],
        })
    return output


def replay_curvature(config: benchmark.BenchmarkConfig, explicit_reference: dict[tuple[str, str, int], list[dict[str, Any]]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cases = benchmark.make_cases(config)
    shape = (len(cases), len(config.alphas), len(config.mc_seeds), config.iterations, config.dimension)
    estimates = np.full(shape, np.nan, dtype=np.float64)
    active = np.zeros(shape[:-1], dtype=np.bool_)
    max_gap_error = 0.0
    max_rmse_error = 0.0
    checked_updates = 0
    for case_index, case in enumerate(cases):
        for alpha_index, alpha in enumerate(config.alphas):
            for seed_index, seed in enumerate(config.mc_seeds):
                key = (case.name, format(float(alpha), ".17g"), int(seed))
                expected = explicit_reference.get(key)
                if expected is None:
                    raise AnalysisError(f"missing explicit-ES reference trajectory {key}")
                state = benchmark.MethodState(params=case.initial_params.copy())
                for iteration in range(1, config.iterations + 1):
                    expected_row = expected[iteration]
                    if state.diverged:
                        if expected_row["active_update"]:
                            raise AnalysisError(f"replay/reference active-state mismatch at {key}, iteration {iteration}")
                        continue
                    eps, noise_plus, noise_minus = benchmark._sample_iteration(config, case_index, alpha_index, int(seed), iteration)
                    plus, minus = benchmark.evaluate_antithetic(case, state.params, eps, config.sigma, noise_plus, noise_minus)
                    estimate = benchmark.estimate_mc(case, state.params, eps, plus, minus, config.sigma, "raw")
                    estimates[case_index, alpha_index, seed_index, iteration - 1] = estimate.diagonal
                    active[case_index, alpha_index, seed_index, iteration - 1] = True
                    if estimate.diagonal_rmse is not None and expected_row["curvature_diag_rmse"] is not None:
                        max_rmse_error = max(max_rmse_error, abs(estimate.diagonal_rmse - expected_row["curvature_diag_rmse"]))
                    proposal = benchmark.propose_step("explicit_es", estimate, case, float(alpha), config.singular_tolerance)
                    new_params, _, _ = benchmark._cap_or_diverge(state, proposal, estimate, case, iteration, config.divergence_threshold, config.objective_gap_divergence_ratio)
                    state.params = new_params
                    actual_gap = case.gap(new_params)
                    expected_gap = expected_row["objective_gap"]
                    if actual_gap is not None and expected_gap is not None:
                        error = abs(actual_gap - expected_gap) / max(1.0, abs(expected_gap))
                        max_gap_error = max(max_gap_error, error)
                    if bool(state.diverged) != bool(expected_row["diverged"]):
                        raise AnalysisError(f"replay/reference divergence mismatch at {key}, iteration {iteration}")
                    checked_updates += 1
    if (
        max_gap_error > REPLAY_GAP_RELATIVE_TOLERANCE
        or max_rmse_error > REPLAY_RMSE_ABSOLUTE_TOLERANCE
    ):
        raise AnalysisError(f"deterministic replay mismatch: gap={max_gap_error:g}, RMSE={max_rmse_error:g}")
    convex_loss_curvature = -estimates
    projected_curvature = np.maximum(convex_loss_curvature, 0.0)
    alpha_broadcast = np.asarray(config.alphas, dtype=np.float64)[
        None, :, None, None, None
    ]
    projected_multiplier = 1.0 / (1.0 + alpha_broadcast * projected_curvature)
    true_reward_diagonal = np.stack([case.diagonal for case in cases])
    arrays = {
        "reward_hessian_diagonal_estimate": estimates,
        "convex_loss_curvature_estimate": convex_loss_curvature,
        "projected_convex_loss_curvature": projected_curvature,
        "projected_diagonal_multiplier": projected_multiplier,
        "true_reward_hessian_diagonal": true_reward_diagonal,
        "true_convex_loss_curvature_diagonal": -true_reward_diagonal,
        "active_mask": active,
        "case_names": np.asarray([case.name for case in cases], dtype="U64"),
        "alphas": np.asarray(config.alphas, dtype=np.float64),
        "seeds": np.asarray(config.mc_seeds, dtype=np.int64),
        "iterations": np.arange(1, config.iterations + 1, dtype=np.int64),
        "coordinates": np.arange(1, config.dimension + 1, dtype=np.int64),
    }
    validation = {
        "checked_active_updates": checked_updates,
        "max_relative_gap_error": max_gap_error,
        "max_relative_gap_tolerance": REPLAY_GAP_RELATIVE_TOLERANCE,
        "max_absolute_rmse_error": max_rmse_error,
        "max_absolute_rmse_tolerance": REPLAY_RMSE_ABSOLUTE_TOLERANCE,
    }
    return arrays, validation


def summarize_curvature(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Summarize per-update estimator noise and the resulting attenuation."""
    estimates = arrays["convex_loss_curvature_estimate"]
    truth = arrays["true_convex_loss_curvature_diagonal"]
    multipliers = arrays["projected_diagonal_multiplier"]
    active = arrays["active_mask"]
    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(arrays["case_names"]):
        for alpha_index, alpha in enumerate(arrays["alphas"]):
            mask = active[case_index, alpha_index]
            selected = estimates[case_index, alpha_index][mask]
            selected_multipliers = multipliers[case_index, alpha_index][mask]
            if selected.size == 0:
                continue
            errors = selected - truth[case_index]
            batch_rmse = np.sqrt(np.mean(errors**2, axis=1))
            signed_denominator = 1.0 + float(alpha) * selected
            rows.append(
                {
                    "case": str(case),
                    "alpha": float(alpha),
                    "n_active_batches": int(selected.shape[0]),
                    "n_coordinate_estimates": int(selected.size),
                    "mean_curvature_bias": float(np.mean(errors)),
                    "coordinate_curvature_rmse": float(np.sqrt(np.mean(errors**2))),
                    "median_batch_curvature_rmse": float(np.median(batch_rmse)),
                    "p95_batch_curvature_rmse": float(np.quantile(batch_rmse, 0.95)),
                    "nonpositive_curvature_estimate_fraction": float(np.mean(selected <= 0.0)),
                    "signed_denominator_nonpositive_fraction": float(np.mean(signed_denominator <= 0.0)),
                    "signed_denominator_abs_below_0_1_fraction": float(np.mean(np.abs(signed_denominator) < 0.1)),
                    "mean_projected_multiplier": float(np.mean(selected_multipliers)),
                    "median_projected_multiplier": float(np.median(selected_multipliers)),
                    "p05_projected_multiplier": float(np.quantile(selected_multipliers, 0.05)),
                    "p95_projected_multiplier": float(np.quantile(selected_multipliers, 0.95)),
                    "no_attenuation_fraction": float(np.mean(selected_multipliers == 1.0)),
                }
            )
    return rows


def _style_axes(axis: Any) -> None:
    axis.grid(True, color="#d1d5db", linewidth=0.6, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8)


def _apply_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Serif", "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "lines.linewidth": 1.6, "savefig.transparent": False,
    })


def plot_step_size(aggregates: Sequence[dict[str, Any]], cases: Sequence[str], checkpoint: int, figures: Path) -> list[Path]:
    _apply_plot_style()
    fig, axes = plt.subplots(2, len(cases), figsize=(3.25 * len(cases), 5.5), squeeze=False, sharex="col")
    for column, case in enumerate(cases):
        top, bottom = axes[0, column], axes[1, column]
        for method in PRIMARY_METHODS:
            rows = [row for row in aggregates if row["regime"] == "monte_carlo_es" and row["fitness_transform"] == "raw" and row["case"] == case and row["method"] == method and int(row["checkpoint"]) == checkpoint]
            rows.sort(key=lambda row: float(row["alpha"]))
            if not rows:
                continue
            x = np.asarray([float(row["alpha"]) for row in rows])
            style = METHOD_STYLES[method]
            top.plot(x, [100.0 * float(row["divergence_rate"]) for row in rows], label=METHOD_LABELS[method], markersize=3.5, **style)
            auc = [float(row["mean_normalized_gap_auc"]) if float(row["divergence_rate"]) == 0.0 else np.nan for row in rows]
            bottom.plot(x, auc, markersize=3.5, **style)
        top.set_title(CASE_LABELS.get(case, case))
        top.set_ylim(-3.0, 103.0)
        bottom.set_yscale("log")
        bottom.set_xscale("log")
        top.set_xscale("log")
        bottom.set_xlabel(r"Step size $\alpha$")
        if column == 0:
            top.set_ylabel("Divergence (%)")
            bottom.set_ylabel("Normalized gap AUC")
        _style_axes(top); _style_axes(bottom)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Step-size robustness", y=1.065, fontsize=12)
    fig.tight_layout()
    outputs = [figures / "step_size_robustness.pdf", figures / "step_size_robustness.png"]
    _save_figure(fig, outputs[0], "pdf"); _save_figure(fig, outputs[1], "png", dpi=300)
    plt.close(fig)
    return outputs


def plot_alpha_one(curves: dict[tuple[str, str, int], list[float]], cases: Sequence[str], iterations: int, figures: Path) -> list[Path]:
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(cases), figsize=(3.25 * len(cases), 3.3), squeeze=False, sharey=True)
    x = np.arange(iterations + 1)
    for column, case in enumerate(cases):
        axis = axes[0, column]
        for method in PRIMARY_METHODS:
            means, lows, highs = [], [], []
            for iteration in x:
                values = curves.get((case, method, int(iteration)), [])
                mean, _, low, high = _mean_ci(values)
                means.append(np.nan if mean is None or mean <= 0.0 else mean)
                # A symmetric interval for an arithmetic mean can cross zero.
                # On a logarithmic axis, omit that unsupported lower segment
                # instead of drawing an arbitrary floor that looks precise.
                lows.append(np.nan if low is None or low <= 0.0 else low)
                highs.append(np.nan if high is None or high <= 0.0 else high)
            style = METHOD_STYLES[method]
            axis.plot(x, means, label=METHOD_LABELS[method], color=style["color"], linestyle=style["linestyle"])
            if np.any(np.isfinite(lows)):
                axis.fill_between(x, lows, highs, color=style["color"], alpha=0.08, linewidth=0)
        axis.set_title(CASE_LABELS.get(case, case))
        axis.set_yscale("log")
        axis.set_xlabel("Update")
        if column == 0:
            axis.set_ylabel("Normalized loss gap")
        _style_axes(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(r"Convergence at $\alpha=1$", y=1.09, fontsize=12)
    fig.tight_layout()
    outputs = [figures / "alpha1_trajectories.pdf", figures / "alpha1_trajectories.png"]
    _save_figure(fig, outputs[0], "pdf"); _save_figure(fig, outputs[1], "png", dpi=300)
    plt.close(fig)
    return outputs


def plot_curvature(arrays: dict[str, np.ndarray], figures: Path) -> list[Path]:
    _apply_plot_style()
    estimates = arrays["convex_loss_curvature_estimate"]
    truth = arrays["true_convex_loss_curvature_diagonal"]
    multipliers = arrays["projected_diagonal_multiplier"]
    active = arrays["active_mask"]
    cases = [str(value) for value in arrays["case_names"]]
    alpha_matches = np.flatnonzero(np.isclose(arrays["alphas"], 1.0, rtol=0.0, atol=1e-12))
    if len(alpha_matches) != 1:
        raise AnalysisError("curvature figure requires exactly one alpha=1 entry")
    alpha_index = int(alpha_matches[0])
    alpha = float(arrays["alphas"][alpha_index])
    fig, axes = plt.subplots(
        2,
        len(cases),
        figsize=(3.25 * len(cases), 5.6),
        squeeze=False,
        sharex="col",
    )
    for case_index, case in enumerate(cases):
        top, bottom = axes[0, case_index], axes[1, case_index]
        mask = active[case_index, alpha_index]
        selected = estimates[case_index, alpha_index][mask]
        selected_multipliers = multipliers[case_index, alpha_index][mask]
        if selected.size == 0:
            raise AnalysisError(f"no active alpha=1 curvature batches for {case}")
        curvature_median = np.median(selected, axis=0)
        curvature_low, curvature_high = np.quantile(selected, [0.05, 0.95], axis=0)
        multiplier_median = np.median(selected_multipliers, axis=0)
        multiplier_low, multiplier_high = np.quantile(
            selected_multipliers, [0.05, 0.95], axis=0
        )
        true_multiplier = 1.0 / (1.0 + alpha * np.maximum(truth[case_index], 0.0))
        x = np.arange(1, estimates.shape[-1] + 1)
        top.plot(
            x,
            truth[case_index],
            color="#111827",
            linestyle="--",
            marker="s",
            markersize=3,
            label="True curvature",
        )
        top.errorbar(
            x,
            curvature_median,
            yerr=np.vstack(
                (
                    curvature_median - curvature_low,
                    curvature_high - curvature_median,
                )
            ),
            color="#2563eb",
            linestyle="none",
            marker="o",
            markersize=3,
            capsize=2,
            label="Curvature: median [5%, 95%]",
        )
        top.axhline(0.0, color="#6b7280", linewidth=0.7)
        top.set_title(CASE_LABELS.get(case, case))
        if case_index == 0:
            top.set_ylabel("Convex-loss diagonal curvature")
        bottom.plot(
            x,
            true_multiplier,
            color="#111827",
            linestyle="--",
            marker="s",
            markersize=3,
            label="True multiplier",
        )
        bottom.errorbar(
            x,
            multiplier_median,
            yerr=np.vstack(
                (
                    multiplier_median - multiplier_low,
                    multiplier_high - multiplier_median,
                )
            ),
            color="#d97706",
            linestyle="none",
            marker="o",
            markersize=3,
            capsize=2,
            label="Multiplier: median [5%, 95%]",
        )
        bottom.set_ylim(-0.03, 1.03)
        bottom.set_xlabel("Coordinate")
        if case_index == 0:
            bottom.set_ylabel(r"Projected multiplier $m_j$")
        _style_axes(top)
        _style_axes(bottom)
    top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
    bottom_handles, bottom_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        top_handles + bottom_handles,
        top_labels + bottom_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    fig.suptitle(r"Curvature estimates and attenuation at $\alpha=1$", y=1.065, fontsize=12)
    fig.tight_layout()
    outputs = [figures / "curvature_calibration.pdf", figures / "curvature_calibration.png"]
    _save_figure(fig, outputs[0], "pdf"); _save_figure(fig, outputs[1], "png", dpi=300)
    plt.close(fig)
    return outputs


def _artifact_entry(path: Path, rows: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"path": path.name if path.parent.name != "figures" else f"figures/{path.name}", "bytes": path.stat().st_size, "sha256": _sha256(path)}
    if rows is not None:
        entry["rows"] = rows
    return entry


def _academic_summary(
    config: benchmark.BenchmarkConfig,
    aggregates: Sequence[dict[str, Any]],
    curvature_diagnostics: Sequence[dict[str, Any]],
    checkpoint: int,
    curvature_validation: dict[str, Any],
) -> str:
    lines = [
        "# Estimated-Hessian Implicit Optimization on Convex Quadratics", "",
        "## Technical summary", "",
        "This controlled benchmark estimates diagonal curvature from antithetic objective evaluations and applies it in implicit updates. It is a synthetic mechanism study, not reinforcement-learning evidence.", "",
        "The primary comparisons separate sampled curvature, projected sampled curvature, an equal-step-norm scalar control, an exact-diagonal reference, and a full-Hessian oracle. Divergence is interpreted before AUC; survivor-only performance is not used as the primary result.", "",
        "## Results at alpha = 1", "",
        f"The table reports the {checkpoint}-update divergence rate and boundary-inclusive normalized loss-gap AUC. Lower AUC is better.", "",
        "| Case | Method | Divergence | Gap AUC |", "| --- | --- | ---: | ---: |",
    ]
    selected = [row for row in aggregates if row["regime"] == "monte_carlo_es" and row["fitness_transform"] == "raw" and row["method"] in PRIMARY_METHODS and int(row["checkpoint"]) == checkpoint and math.isclose(float(row["alpha"]), 1.0, abs_tol=1e-12)]
    selected.sort(key=lambda row: (str(row["case"]), PRIMARY_METHODS.index(str(row["method"]))))
    for row in selected:
        lines.append(f"| {CASE_LABELS.get(str(row['case']), row['case'])} | {METHOD_LABELS[str(row['method'])]} | {100.0 * float(row['divergence_rate']):.1f}% | {float(row['mean_normalized_gap_auc']):.6g} |")
    diagnostic_alpha_one = sorted(
        (
            row
            for row in curvature_diagnostics
            if math.isclose(float(row["alpha"]), 1.0, abs_tol=1e-12)
        ),
        key=lambda row: str(row["case"]),
    )
    lines.extend([
        "", "## Hessian-estimate diagnostics at alpha = 1", "",
        "These are per-update diagnostics, before averaging estimates across training. Near-singular means that the absolute unprojected signed denominator is below 0.1; the multiplier column describes the projected update.", "",
        "| Case | Median batch RMSE | Nonpositive estimate | Signed denominator near zero | Median multiplier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in diagnostic_alpha_one:
        lines.append(
            f"| {CASE_LABELS.get(str(row['case']), row['case'])} | "
            f"{float(row['median_batch_curvature_rmse']):.4g} | "
            f"{100.0 * float(row['nonpositive_curvature_estimate_fraction']):.1f}% | "
            f"{100.0 * float(row['signed_denominator_abs_below_0_1_fraction']):.2f}% | "
            f"{float(row['median_projected_multiplier']):.3f} |"
        )
    lines.extend([
        "", "## Figures", "",
        "![Step-size robustness](figures/step_size_robustness.png)", "",
        "Divergence is shown separately above AUC. AUC points are omitted when any seed diverged.", "",
        "![Alpha-one trajectories](figures/alpha1_trajectories.png)", "",
        "Curves are arithmetic means of the normalized convex-loss gap; shaded bands are pointwise 95% t intervals across matched seeds. Boundary-capped failures remain in the trajectories.", "",
        "![Curvature calibration](figures/curvature_calibration.png)", "",
        "The code maximizes a concave reward, so convex-loss curvature is the negative reward Hessian. Markers summarize individual alpha-one update batches; bars span their 5th to 95th percentiles. The lower row shows the corresponding projected implicit multiplier.", "",
        "## Protocol and definitions", "",
        f"- Dimension: {config.dimension}; population: {config.population_size} ({config.population_size // 2} antithetic pairs).",
        f"- Updates: {config.iterations}; perturbation scale: {config.sigma}.",
        f"- Step sizes: {', '.join(format(float(value), 'g') for value in config.alphas)}.",
        f"- Seeds: {', '.join(map(str, config.mc_seeds))}.",
        "- Fitness: raw objective values; curvature: leave-one-pair-out diagonal Stein estimate.",
        "- No replay, trust region, or additive scalar damping is used inside the Hessian arms; equal-norm scalar attenuation is a separately labeled control.", "",
        "For checkpoint h, normalized gap AUC is the trapezoidal mean of f(x_t)/f(x_0) from update 0 through h. The exact-diagonal and full-Hessian oracle arms use the same sampled gradient as the estimated-curvature methods.", "",
        "## Validation and limitations", "",
        f"Deterministic replay checked {curvature_validation['checked_active_updates']} active explicit-reference updates; maximum relative loss-gap disagreement was {curvature_validation['max_relative_gap_error']:.3g}.", "",
        "The rotated problem distinguishes diagonal approximation error from full-Hessian geometry. Ten synthetic seeds provide descriptive uncertainty only. Results do not establish performance on Hopper or other reinforcement-learning tasks.", "",
    ])
    return "\n".join(lines)


def analyze(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str] | None = None, checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS) -> dict[str, str]:
    source_root = Path(input_dir).resolve()
    destination = source_root if output_dir is None else Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    manifest, config, files, input_validation = validate_benchmark_artifact(source_root)
    checkpoints = tuple(sorted(set(int(value) for value in checkpoints)))
    if not checkpoints or checkpoints[0] <= 0 or checkpoints[-1] > config.iterations:
        raise AnalysisError(f"checkpoints must lie in [1, {config.iterations}]")
    if not any(math.isclose(float(alpha), 1.0, abs_tol=1e-12) for alpha in config.alphas):
        raise AnalysisError("alpha=1 is required for the trajectory figure")

    checkpoint_rows, explicit_reference, curves = stream_trajectories(files["trajectories"], config, checkpoints)
    aggregate_rows = aggregate_checkpoints(checkpoint_rows)
    checkpoint_path = destination / "checkpoint_runs.csv"
    aggregate_path = destination / "checkpoint_aggregate.csv"
    _atomic_csv(checkpoint_path, CHECKPOINT_FIELDS, checkpoint_rows)
    _atomic_csv(aggregate_path, AGGREGATE_FIELDS, aggregate_rows)

    arrays, curvature_validation = replay_curvature(config, explicit_reference)
    curvature_diagnostic_rows = summarize_curvature(arrays)
    curvature_path = destination / "curvature_estimates.npz"
    _atomic_npz(curvature_path, **arrays)
    curvature_diagnostics_path = destination / "curvature_diagnostics.csv"
    _atomic_csv(
        curvature_diagnostics_path,
        CURVATURE_DIAGNOSTIC_FIELDS,
        curvature_diagnostic_rows,
    )
    curvature_index_path = destination / "curvature_estimates_index.json"
    curvature_index = {
        "schema_version": 1,
        "npz_file": curvature_path.name,
        "npz_sha256": _sha256(curvature_path),
        "axis_order": ["case", "alpha", "seed", "iteration", "coordinate"],
        "estimate_array": "reward_hessian_diagonal_estimate",
        "truth_array": "true_reward_hessian_diagonal",
        "presentation_arrays": {
            "convex_loss_curvature_estimate": "negative reward_hessian_diagonal_estimate",
            "projected_convex_loss_curvature": "maximum of convex-loss curvature and zero",
            "projected_diagonal_multiplier": "1 / (1 + alpha * projected_convex_loss_curvature)",
            "true_convex_loss_curvature_diagonal": "negative true reward-Hessian diagonal",
        },
        "active_mask": "active_mask",
        "presentation_sign_convention": "convex_loss_curvature_equals_negative_reward_hessian",
        "estimator": "raw_antithetic_leave_one_pair_out_diagonal_Stein",
        "shape": list(arrays["reward_hessian_diagonal_estimate"].shape),
        "validation": curvature_validation,
    }
    _atomic_json(curvature_index_path, curvature_index)

    finite_cases = [metadata["name"] for metadata in manifest.get("cases", []) if metadata.get("has_finite_maximum")]
    max_checkpoint = max(checkpoints)
    figure_paths = []
    figure_paths.extend(plot_step_size(aggregate_rows, finite_cases, max_checkpoint, figures))
    figure_paths.extend(plot_alpha_one(curves, finite_cases, config.iterations, figures))
    figure_paths.extend(plot_curvature(arrays, figures))

    summary_path = destination / "academic_summary.md"
    _atomic_text(
        summary_path,
        _academic_summary(
            config,
            aggregate_rows,
            curvature_diagnostic_rows,
            max_checkpoint,
            curvature_validation,
        ),
    )
    outputs: dict[str, Path] = {
        "checkpoint_runs": checkpoint_path, "checkpoint_aggregate": aggregate_path,
        "curvature_estimates": curvature_path,
        "curvature_diagnostics": curvature_diagnostics_path,
        "curvature_index": curvature_index_path,
        "academic_summary": summary_path,
    }
    outputs.update({path.stem + "_" + path.suffix.lstrip("."): path for path in figure_paths})
    analysis_manifest_path = destination / "analysis_manifest.json"
    output_entries = {}
    for key, path in outputs.items():
        rows = _csv_rows(path) if path.suffix == ".csv" else None
        output_entries[key] = _artifact_entry(path, rows)
    analysis_manifest = {
        "schema_version": 1, "analyzer_version": ANALYZER_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {"directory": str(source_root), **input_validation},
        "config": {"checkpoints": list(checkpoints), "primary_methods": list(PRIMARY_METHODS), "alpha_one_trajectory": True},
        "validation": {"benchmark_manifest_validated": True, "trajectory_stream_complete": True, "curvature_replay": curvature_validation},
        "outputs": output_entries,
        "provenance": {"script": str(Path(__file__).resolve().relative_to(REPO_ROOT)), "script_sha256": _sha256(Path(__file__).resolve()), "python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
    }
    _atomic_json(analysis_manifest_path, analysis_manifest)
    outputs["analysis_manifest"] = analysis_manifest_path
    return {key: str(path) for key, path in outputs.items()}


def _parse_checkpoints(value: str) -> tuple[int, ...]:
    try:
        output = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not output:
        raise argparse.ArgumentTypeError("checkpoint list must not be empty")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoints", type=_parse_checkpoints, default=DEFAULT_CHECKPOINTS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outputs = analyze(args.input_dir, args.output_dir, args.checkpoints)
    print(f"Validated and analyzed convex benchmark: {outputs['analysis_manifest']}")


if __name__ == "__main__":
    main()

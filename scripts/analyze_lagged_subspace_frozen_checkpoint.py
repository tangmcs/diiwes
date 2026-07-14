#!/usr/bin/env python3
"""Fail-closed validator and preregistered analyzer for the subspace diagnostic.

The input is an audit index. Large numerical arrays remain in hashed artifacts;
the index records their deterministic identities, scalar invariants, and endpoint
observations. No aggregate or inferential value is accepted from the producer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from core.lagged_subspace_diagnostic import (
    BasisProvenance,
    analyze_lagged_subspace_population,
    frozen_endpoint_diagnostics,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST_PATH = os.path.join(
    REPO_ROOT, "experiments/manifests/lagged_subspace_frozen_checkpoint.json"
)
STUDY = "lagged_subspace_frozen_checkpoint"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
NORM_TOLERANCE = 1e-10
UTILITY_TOLERANCE = 1e-12
ORIGIN_MAP_TOLERANCE = 1e-10

TOP_LEVEL_KEYS = {
    "schema_version",
    "study",
    "designation",
    "manifest_sha256",
    "provenance",
    "analysis_declaration",
    "training_runs",
    "checkpoints",
    "banks",
    "partitions",
    "checkpoint_metrics",
    "center_endpoints",
    "endpoints",
    "budget",
}
PROVENANCE_KEYS = {
    "source_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "analyzer_sha256",
    "launcher_sha256",
    "dependency_lock_sha256",
    "source_snapshot_path",
    "stderr_empty",
    "documented_infrastructure_failures",
    "record_sha256",
}
TRAINING_KEYS = {
    "training_id",
    "task_index",
    "env_name",
    "training_seed",
    "updates",
    "population_size",
    "candidate_rollouts",
    "calibration_rollouts",
    "online_evaluation_rollouts",
    "checkpoint_generations",
    "training_transitions",
    "calibration_transitions",
    "training_log_sha256",
    "training_log_path",
    "stderr_sha256",
    "stderr_empty",
    "record_sha256",
}
CHECKPOINT_KEYS = {
    "checkpoint_id",
    "training_id",
    "task_index",
    "env_name",
    "training_seed",
    "generation",
    "parameter_sha256",
    "observation_normalizer_sha256",
    "training_config_sha256",
    "source_sha256",
    "prior_gradient_indices",
    "prior_gradient_sha256",
    "lagged_block_norms",
    "lagged_block_exact_zero",
    "primary_gaussian_fallback_used",
    "random_control_permuted_fallback_used",
    "fallback_column_sha256",
    "basis_seed",
    "random_control_seed",
    "basis_sha256",
    "random_basis_sha256",
    "basis_locked_before_bank_sampling",
    "checkpoint_artifact_sha256",
    "checkpoint_artifact_path",
    "training_config_path",
    "basis_artifact_path",
    "basis_artifact_sha256",
    "record_sha256",
}
BANK_KEYS = {
    "bank_id",
    "checkpoint_id",
    "bank",
    "pair_count",
    "candidate_rollouts",
    "pair_indices",
    "perturbation_seeds",
    "rollout_seeds_plus",
    "rollout_seeds_minus",
    "perturbations_sha256",
    "returns_sha256",
    "transitions_sha256",
    "jackknife_sha256",
    "raw_bank_path",
    "raw_bank_sha256",
    "diagnostics_path",
    "diagnostics_sha256",
    "candidate_transitions",
    "antithetic_max_abs_error",
    "exact_antithetic",
    "shared_rollout_seed_within_pair",
    "lopo_utility_sum",
    "lopo_utility_abs_sum",
    "lopo_gradient_curvature_shared",
    "dsn_da_relative_error",
    "jsn_ja_relative_error",
    "finite_u_statistic",
    "finite_jackknife",
    "finite_eigensystem",
    "q_summaries",
    "diagnostics_path",
    "diagnostics_sha256",
    "stderr_sha256",
    "stderr_empty",
    "record_sha256",
}
PARTITION_KEYS = {
    "partition_id",
    "checkpoint_id",
    "partition_index",
    "partition_seed",
    "pair_indices",
    "pair_count",
    "lopo_utility_sum",
    "lopo_utility_abs_sum",
    "lopo_gradient_curvature_shared",
    "finite_u_statistic",
    "finite_jackknife",
    "finite_eigensystem",
    "q_summaries",
    "diagnostics_path",
    "diagnostics_sha256",
    "record_sha256",
}
Q_SUMMARY_KEYS = {
    "q",
    "alpha",
    "alpha_resolved",
    "alpha_unresolved_reason",
    "gradient_norm",
    "structured_norm",
    "isotropic_norm",
    "explicit_norm",
    "random_norm",
    "random_raw_norm",
    "anisotropic_action_norm",
    "anisotropic_minus_bank_a_norm",
    "structured_step_over_sigma",
    "structured_solve_residual",
    "random_solve_residual",
    "structured_isotropic_relative_norm_error",
    "structured_random_relative_norm_error",
    "random_control_valid",
    "material_denominator_resolved",
    "finite",
    "action_sha256",
}
METRIC_KEYS = {
    "metric_id",
    "checkpoint_id",
    "q",
    "d_material",
    "e_high",
    "e_100",
    "material_resolved",
    "high_sample_resolved",
    "operational_resolved",
    "high_sample_action_difference_norm",
    "partition_action_sq_error_mean",
    "record_sha256",
}
CENTER_KEYS = {
    "center_endpoint_id",
    "checkpoint_id",
    "episode_index",
    "rollout_seed",
    "return",
    "transitions",
    "rollout_artifact_path",
    "rollout_artifact_sha256",
    "record_sha256",
}
ENDPOINT_KEYS = {
    "endpoint_id",
    "checkpoint_id",
    "partition_index",
    "q",
    "arm",
    "episode_index",
    "rollout_seed",
    "return",
    "transitions",
    "action_sha256",
    "rollout_artifact_path",
    "rollout_artifact_sha256",
    "record_sha256",
}
BUDGET_KEYS = {
    "checkpoint_training_candidate_rollouts",
    "normalization_calibration_rollouts",
    "bank_candidate_rollouts",
    "endpoint_arm_rollouts",
    "checkpoint_center_rollouts",
    "total_policy_rollouts",
    "checkpoint_training_transitions",
    "normalization_calibration_transitions",
    "bank_transitions",
    "endpoint_arm_transitions",
    "checkpoint_center_transitions",
    "total_environment_transitions",
}
INFERENCE_LEAKAGE_KEYS = {
    "p_value",
    "adjusted_p_value",
    "gate_pass",
    "study_pass",
    "confidence_bound",
    "selected_checkpoint",
    "selected_partition",
    "selected_task",
    "claim_selected",
}


class SubspaceValidationError(RuntimeError):
    """Raised only after collecting every detectable validation issue."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__(
            f"lagged-subspace validation failed with {len(self.issues)} issue(s)"
        )


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _labeled_arrays_sha256(values: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for label, value in values:
        encoded = str(label).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_array_sha256(np.asarray(value))))
    return digest.hexdigest()


def _verified_artifact_path(
    artifact_root: str,
    relative_path: Any,
    expected_sha256: Any,
    label: str,
    issues: list[str],
    cache: dict[str, str],
) -> str | None:
    """Resolve and independently hash one root-contained regular file."""

    if (
        not isinstance(relative_path, str)
        or not relative_path
        or os.path.isabs(relative_path)
        or "\\" in relative_path
        or os.path.normpath(relative_path) != relative_path
        or relative_path == ".."
        or relative_path.startswith("../")
    ):
        issues.append(f"{label}: artifact path is not a normalized relative path")
        return None
    root = os.path.realpath(artifact_root)
    candidate = os.path.realpath(os.path.join(root, relative_path))
    try:
        contained = os.path.commonpath([root, candidate]) == root
    except ValueError:
        contained = False
    if not contained:
        issues.append(f"{label}: artifact path escapes the locked artifact root")
        return None
    if not os.path.isfile(candidate):
        issues.append(f"{label}: referenced artifact is missing or not a regular file")
        return None
    if not _is_sha256(expected_sha256):
        issues.append(f"{label}: expected artifact digest is invalid")
        return None
    actual = cache.get(candidate)
    if actual is None:
        actual = _sha256_file(candidate)
        cache[candidate] = actual
    if actual != expected_sha256:
        issues.append(
            f"{label}: artifact digest mismatch (expected {expected_sha256}, found {actual})"
        )
        return None
    return candidate


def _load_npz_exact(
    path: str | None,
    expected_keys: set[str],
    label: str,
    issues: list[str],
) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                issues.append(
                    f"{label}: NPZ schema is not exact "
                    f"(missing={sorted(expected_keys - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - expected_keys)})"
                )
                return None
            return {name: np.asarray(archive[name]) for name in archive.files}
    except (OSError, ValueError, TypeError) as error:
        issues.append(f"{label}: cannot read NPZ artifact: {error}")
        return None


CHECKPOINT_NPZ_KEYS = {
    "schema_version",
    "checkpoint_generation",
    "study_source_sha256",
    "training_config_sha256",
    "center_params",
    "obs_normalizer_enabled",
    "obs_mean",
    "obs_var",
    "obs_count",
    "gradient_generations",
    "proposal_gradients",
}
BASIS_NPZ_KEYS = {
    "primary_basis",
    "random_basis",
    "lagged_block_norms",
    "lagged_block_exact_zero",
    "primary_gaussian_fallback_used",
    "random_control_permuted_fallback_used",
    "fallback_columns",
}
RAW_BANK_NPZ_KEYS = {
    "paired_returns",
    "paired_transitions",
    "perturbation_seeds",
    "rollout_seeds_plus",
    "rollout_seeds_minus",
}
ENDPOINT_NPZ_KEYS = {
    "center_returns",
    "center_transitions",
    "endpoint_returns",
    "endpoint_transitions",
    "rollout_seeds",
}


def _diagnostic_npz_keys(manifest: Mapping[str, Any]) -> set[str]:
    keys = {
        "utilities",
        "gradient_sha256",
        "curvature",
        "random_curvature",
        "gradient_component_variance_sha256",
        "step_sha256",
        "curvature_vech_covariance",
        "curvature_eigenvalues",
        "negative_eigenvalue_count",
        "jackknife_eigenvalue_se",
        "structured_action_covariance_trace",
        "anisotropic_action_covariance_trace",
        "anisotropic_action_aligned_variance",
        "repeated_eigenvalue_unresolved",
        "projection_boundary_unresolved",
        "zero_anisotropic_action_unresolved",
        "b_frobenius_absolute_error_to_bank_a",
        "negative_eigenvalue_sign_agreement_to_bank_a",
        "anisotropic_action_cosine_to_bank_a",
        "anisotropic_action_relative_error_to_bank_a",
        "r_full",
        "r_sub",
        "r_sn",
        "r_full_unresolved",
        "r_sub_unresolved",
        "r_sn_unresolved",
        "normalized_ess_ratio",
        "ratio_coefficient_of_variation",
        "mean_unnormalized_ratio_minus_one",
        "log_ratio_span",
        "alpha_max_concave_eigenvalue",
        "structured_explicit_angle_degrees",
        "multiplier_standard_deviation",
        "multiplier_range",
        "gradient_endpoint_relative_error",
        "subspace_jacobian_relative_error",
        "self_normalized_gradient_relative_error",
        "self_normalized_jacobian_relative_error",
    }
    return keys


def _diagnostic_jackknife_sha256(data: Mapping[str, np.ndarray]) -> str:
    labels = (
        "gradient_component_variance",
        "curvature_vech_covariance",
        "jackknife_eigenvalue_se",
        "structured_action_covariance_trace",
        "anisotropic_action_covariance_trace",
        "anisotropic_action_aligned_variance",
    )
    return _labeled_arrays_sha256([(label, data[label]) for label in labels])


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _fixed_ascii_sha256_matches(value: np.ndarray, expected: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("S64"):
        return False
    try:
        decoded = array.item().decode("ascii")
    except (AttributeError, UnicodeDecodeError):
        return False
    return decoded == expected


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if not _finite(left) or not _finite(right):
        return False
    return bool(
        np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance)
    )


def _relative_error(left: float, right: float, epsilon: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), epsilon)


def _stable_norm(vector: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError("stable norm requires finite values")
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.linalg.norm(vector / scale))


def _append_schema_issue(
    record: Any, expected: set[str], label: str, issues: list[str]
) -> bool:
    if not isinstance(record, dict):
        issues.append(f"{label}: record is not an object")
        return False
    if set(record) != expected:
        missing = sorted(expected - set(record))
        extra = sorted(set(record) - expected)
        issues.append(f"{label}: schema is not exact (missing={missing}, extra={extra})")
        return False
    return True


def _validate_record_hash(record: Mapping[str, Any], label: str, issues: list[str]) -> None:
    expected = _record_sha256(record)
    actual = record.get("record_sha256")
    if actual != expected:
        issues.append(f"{label}: record SHA-256 mismatch")


def _find_inference_leakage(value: Any, path: str = "artifact") -> list[str]:
    issues: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in INFERENCE_LEAKAGE_KEYS:
                issues.append(f"{child_path}: producer-supplied inferential result is forbidden")
            issues.extend(_find_inference_leakage(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            issues.extend(_find_inference_leakage(child, f"{path}[{index}]"))
    return issues


def derive_seed(manifest: Mapping[str, Any], namespace: str, *coordinates: Any) -> int:
    """Return the protocol's versioned uint64 seed for one named stream."""

    rng = manifest["rng"]
    namespace_id = rng["namespaces"][namespace]
    payload = [manifest["study"], rng["master_seed"], namespace_id, *coordinates]
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def training_id_for(manifest: Mapping[str, Any], task_index: int, seed: int) -> int:
    seeds = manifest["training_seeds"]
    return task_index * len(seeds) + seeds.index(seed)


def checkpoint_id_for(
    manifest: Mapping[str, Any], task_index: int, seed: int, generation: int
) -> int:
    training_id = training_id_for(manifest, task_index, seed)
    return training_id * len(manifest["checkpoint_generations"]) + manifest[
        "checkpoint_generations"
    ].index(generation)


def checkpoint_coordinates(
    manifest: Mapping[str, Any], checkpoint_id: int
) -> tuple[int, int, int]:
    generations = manifest["checkpoint_generations"]
    seeds = manifest["training_seeds"]
    checkpoint_count = len(manifest["tasks"]) * len(seeds) * len(generations)
    if isinstance(checkpoint_id, bool) or not 0 <= checkpoint_id < checkpoint_count:
        raise ValueError(f"checkpoint id {checkpoint_id!r} is out of range")
    training_id, generation_index = divmod(checkpoint_id, len(generations))
    task_index, seed_index = divmod(training_id, len(seeds))
    return task_index, seeds[seed_index], generations[generation_index]


def bank_id_for(manifest: Mapping[str, Any], checkpoint_id: int, bank: str) -> int:
    return checkpoint_id * len(manifest["dimensions"]["banks"]) + manifest[
        "dimensions"
    ]["banks"].index(bank)


def partition_id_for(manifest: Mapping[str, Any], checkpoint_id: int, index: int) -> int:
    return checkpoint_id * manifest["dimensions"]["bank_b_partition_count"] + index


def metric_id_for(manifest: Mapping[str, Any], checkpoint_id: int, q: float) -> int:
    return checkpoint_id * len(manifest["dimensions"]["locality_q"]) + manifest[
        "dimensions"
    ]["locality_q"].index(q)


def center_endpoint_id_for(
    manifest: Mapping[str, Any], checkpoint_id: int, episode_index: int
) -> int:
    return checkpoint_id * manifest["dimensions"]["endpoint_episodes"] + episode_index


def endpoint_id_for(
    manifest: Mapping[str, Any],
    checkpoint_id: int,
    q: float,
    partition_index: int,
    arm: str,
    episode_index: int,
) -> int:
    dims = manifest["dimensions"]
    q_index = dims["locality_q"].index(q)
    arm_index = dims["endpoint_arms"].index(arm)
    value = checkpoint_id * len(dims["locality_q"]) + q_index
    value = value * dims["bank_b_partition_count"] + partition_index
    value = value * len(dims["endpoint_arms"]) + arm_index
    return value * dims["endpoint_episodes"] + episode_index


def checkpoint_seed(
    manifest: Mapping[str, Any], namespace: str, checkpoint_id: int
) -> int:
    task_index, seed, generation = checkpoint_coordinates(manifest, checkpoint_id)
    return derive_seed(manifest, namespace, task_index, seed, generation)


def reconstruct_lagged_bases(
    proposal_gradients: np.ndarray,
    block_sizes: Sequence[int],
    *,
    lagged_decay: float,
    basis_seed_value: int,
    random_seed_value: int,
) -> dict[str, np.ndarray]:
    """Reconstruct the previsible bases from chronological prior gradients."""

    gradients = np.asarray(proposal_gradients, dtype=np.float64)
    sizes = [int(value) for value in block_sizes]
    if (
        gradients.ndim != 2
        or not np.all(np.isfinite(gradients))
        or len(sizes) != 3
        or any(value <= 0 for value in sizes)
        or sum(sizes) != gradients.shape[1]
        or not np.isfinite(lagged_decay)
        or not 0.0 < lagged_decay <= 1.0
    ):
        raise ValueError("lagged basis inputs are invalid")
    archive_length = gradients.shape[0]
    weights = lagged_decay ** np.arange(
        archive_length - 1, -1, -1, dtype=np.float64
    )
    lagged = np.sum(weights[:, None] * gradients, axis=0) / np.sum(weights)
    primary = np.zeros((gradients.shape[1], 3), dtype=np.float64)
    random_basis = np.zeros_like(primary)
    fallback_columns = np.zeros_like(primary)
    norms = np.zeros(3, dtype=np.float64)
    zero_mask = np.zeros(3, dtype=np.bool_)
    basis_rng = np.random.Generator(np.random.PCG64(int(basis_seed_value)))
    random_rng = np.random.Generator(np.random.PCG64(int(random_seed_value)))
    start = 0
    for block_index, block_size in enumerate(sizes):
        stop = start + block_size
        block = np.asarray(lagged[start:stop], dtype=np.float64)
        block_scale = float(np.max(np.abs(block), initial=0.0))
        norm = (
            0.0
            if block_scale == 0.0
            else float(block_scale * np.linalg.norm(block / block_scale))
        )
        norms[block_index] = norm
        zero = bool(np.all(block == 0.0))
        zero_mask[block_index] = zero
        if zero:
            fallback = basis_rng.standard_normal(block_size)
            fallback_scale = float(np.max(np.abs(fallback), initial=0.0))
            fallback_norm = (
                0.0
                if fallback_scale == 0.0
                else float(
                    fallback_scale * np.linalg.norm(fallback / fallback_scale)
                )
            )
            if fallback_norm == 0.0:
                raise FloatingPointError("deterministic Gaussian fallback is zero")
            fallback /= fallback_norm
            source = fallback
        else:
            source = block / norm
        primary[start:stop, block_index] = source
        if zero:
            fallback_columns[start:stop, block_index] = fallback
        permutation = random_rng.permutation(block_size)
        signs = (
            2
            * random_rng.integers(
                0, 2, size=block_size, dtype=np.int64
            )
            - 1
        )
        random_values = signs * source[permutation]
        random_values /= np.linalg.norm(random_values)
        random_basis[start:stop, block_index] = random_values
        start = stop
    return {
        "primary_basis": primary,
        "random_basis": random_basis,
        "lagged_block_norms": norms,
        "lagged_block_exact_zero": zero_mask,
        "primary_gaussian_fallback_used": zero_mask.copy(),
        "random_control_permuted_fallback_used": zero_mask.copy(),
        "fallback_columns": fallback_columns,
    }


def pair_seed(
    manifest: Mapping[str, Any],
    namespace: str,
    checkpoint_id: int,
    bank: str,
    pair_index: int,
) -> int:
    task_index, seed, generation = checkpoint_coordinates(manifest, checkpoint_id)
    bank_index = manifest["dimensions"]["banks"].index(bank)
    return derive_seed(
        manifest,
        namespace,
        task_index,
        seed,
        generation,
        bank_index,
        pair_index,
    )


def _regenerate_signed_noise(seeds: Sequence[int], dimension: int) -> np.ndarray:
    plus = np.stack(
        [
            np.random.Generator(np.random.PCG64(int(seed))).standard_normal(dimension)
            for seed in seeds
        ]
    )
    return np.stack((plus, -plus), axis=1)


def endpoint_seed(
    manifest: Mapping[str, Any], checkpoint_id: int, episode_index: int
) -> int:
    task_index, seed, generation = checkpoint_coordinates(manifest, checkpoint_id)
    return derive_seed(
        manifest, "endpoint", task_index, seed, generation, episode_index
    )


def bank_b_partition(manifest: Mapping[str, Any], checkpoint_id: int) -> list[list[int]]:
    dims = manifest["dimensions"]
    seed = checkpoint_seed(manifest, "bank_b_partition", checkpoint_id)

    def ordering_key(pair_index: int) -> tuple[bytes, int]:
        digest = hashlib.sha256(_canonical_bytes([seed, pair_index])).digest()
        return digest, pair_index

    permutation = sorted(range(dims["pairs_per_bank"]), key=ordering_key)
    width = dims["pairs_per_partition"]
    return [
        permutation[index * width : (index + 1) * width]
        for index in range(dims["bank_b_partition_count"])
    ]


def _validate_manifest_structure(
    manifest: Any, *, require_preregistered: bool
) -> list[str]:
    issues: list[str] = []
    expected_top = {
        "schema_version",
        "study",
        "designation",
        "protocol_status",
        "protocol",
        "tasks",
        "training_seeds",
        "checkpoint_generations",
        "dimensions",
        "training",
        "rng",
        "endpoint_evaluation",
        "analysis",
        "budget",
        "prohibitions",
        "required_hash_locks",
        "claim_boundary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_top:
        return ["manifest top-level schema is not exact"]
    if manifest.get("schema_version") != 1 or manifest.get("study") != STUDY:
        issues.append("manifest study/schema version is invalid")
    if manifest.get("protocol_status") != "final_locked_before_environment_outcomes":
        issues.append("manifest protocol status is not final and pre-outcome locked")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        issues.append("manifest tasks are empty or invalid")
        tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or set(task) != {
            "task_index",
            "env_name",
            "observation_dim",
            "action_dim",
            "parameter_count",
            "policy_block_sizes",
            "policy_block_ranges",
        }:
            issues.append(f"manifest task {index} schema is invalid")
        elif (
            task["task_index"] != index
            or not isinstance(task["env_name"], str)
            or isinstance(task["observation_dim"], bool)
            or not isinstance(task["observation_dim"], int)
            or task["observation_dim"] <= 0
            or isinstance(task["action_dim"], bool)
            or not isinstance(task["action_dim"], int)
            or task["action_dim"] <= 0
            or isinstance(task["parameter_count"], bool)
            or not isinstance(task["parameter_count"], int)
            or task["parameter_count"] <= 0
            or not isinstance(task["policy_block_sizes"], list)
            or len(task["policy_block_sizes"]) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in task["policy_block_sizes"]
            )
        ):
            issues.append(f"manifest task {index} mapping is invalid")
        elif (
            sum(task["policy_block_sizes"]) != task["parameter_count"]
            or task["policy_block_ranges"]
            != [
                [0, task["policy_block_sizes"][0]],
                [
                    task["policy_block_sizes"][0],
                    task["policy_block_sizes"][0]
                    + task["policy_block_sizes"][1],
                ],
                [
                    task["policy_block_sizes"][0]
                    + task["policy_block_sizes"][1],
                    task["parameter_count"],
                ],
            ]
        ):
            issues.append(f"manifest task {index} policy ranges are invalid")
    seeds = manifest.get("training_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        issues.append("manifest training seeds are invalid or duplicated")
        seeds = []
    generations = manifest.get("checkpoint_generations")
    if (
        not isinstance(generations, list)
        or not generations
        or generations != sorted(set(generations))
    ):
        issues.append("manifest checkpoint generations are invalid")
        generations = []
    dims = manifest.get("dimensions")
    required_dims = {
        "policy_hidden_widths",
        "layer_blocks",
        "population_size",
        "antithetic_pairs_per_training_update",
        "training_updates",
        "lagged_gradient_count",
        "lagged_decay",
        "noise_std",
        "banks",
        "pairs_per_bank",
        "bank_b_partition_count",
        "pairs_per_partition",
        "locality_q",
        "endpoint_arms",
        "endpoint_episodes",
        "calibration_episodes",
    }
    if not isinstance(dims, dict) or set(dims) != required_dims:
        issues.append("manifest dimensions schema is not exact")
        dims = {}
    if dims:
        if dims["banks"] != ["A", "B"]:
            issues.append("manifest must contain exactly independent banks A and B")
        if dims["locality_q"] != [0.25, 0.5, 1.0]:
            issues.append("manifest locality grid is not exact")
        if dims["endpoint_arms"] != ["structured", "isotropic", "explicit", "random"]:
            issues.append("manifest endpoint arms are not exact")
        if dims["population_size"] != 2 * dims["antithetic_pairs_per_training_update"]:
            issues.append("training population is not exactly antithetic")
        if dims["pairs_per_bank"] != (
            dims["bank_b_partition_count"] * dims["pairs_per_partition"]
        ):
            issues.append("Bank-B partition dimensions do not cover the bank exactly")
        if generations and max(generations) > dims["training_updates"]:
            issues.append("checkpoint generation exceeds training updates")
        if generations and min(generations) < dims["lagged_gradient_count"]:
            issues.append("checkpoint lacks its required strictly prior gradients")
    rng = manifest.get("rng")
    if not isinstance(rng, dict) or set(rng) != {
        "scheme",
        "master_seed",
        "serialization",
        "uint64_extraction",
        "partition_permutation",
        "basis_gaussian",
        "random_signed_permutation",
        "perturbation_gaussian",
        "namespaces",
    }:
        issues.append("manifest RNG schema is not exact")
    elif (
        rng["scheme"] != "sha256_uint64_v1"
        or rng["basis_gaussian"]
        != "numpy_generator_pcg64_conditional_standard_normal_by_zero_block_v1"
        or rng["random_signed_permutation"]
        != "numpy_generator_pcg64_permutation_then_int64_rademacher_and_renormalize_by_block_v1"
        or rng["perturbation_gaussian"]
        != "numpy_generator_pcg64_standard_normal_per_pair_v1"
        or set(rng["namespaces"])
        != {
            "basis",
            "random_control",
            "bank_perturbation",
            "bank_rollout",
            "bank_b_partition",
            "endpoint",
            "cluster_bootstrap",
        }
        or len(set(rng["namespaces"].values())) != len(rng["namespaces"])
    ):
        issues.append("manifest RNG namespaces are invalid")
    if set(manifest.get("prohibitions", {}).values()) != {False}:
        issues.append("every prohibited mechanism must be disabled")
    required_locks = manifest.get("required_hash_locks")
    if required_locks != [
        "source_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "analyzer_sha256",
        "launcher_sha256",
        "dependency_lock_sha256",
    ]:
        issues.append("manifest required hash locks are not exact")
    protocol = manifest.get("protocol")
    if (
        not isinstance(protocol, dict)
        or set(protocol) != {"path", "sha256"}
        or not _is_sha256(protocol.get("sha256"))
    ):
        issues.append("manifest protocol lock is invalid")
    if dims and tasks and seeds and generations:
        expected = {
            "checkpoint_training_candidate_rollouts": len(tasks)
            * len(seeds)
            * dims["training_updates"]
            * dims["population_size"],
            "normalization_calibration_rollouts": len(tasks)
            * len(seeds)
            * dims["calibration_episodes"],
            "bank_candidate_rollouts": len(tasks)
            * len(seeds)
            * len(generations)
            * len(dims["banks"])
            * dims["pairs_per_bank"]
            * 2,
            "endpoint_arm_rollouts": len(tasks)
            * len(seeds)
            * len(generations)
            * len(dims["locality_q"])
            * dims["bank_b_partition_count"]
            * len(dims["endpoint_arms"])
            * dims["endpoint_episodes"],
            "checkpoint_center_rollouts": len(tasks)
            * len(seeds)
            * len(generations)
            * dims["endpoint_episodes"],
        }
        expected["total_policy_rollouts"] = sum(expected.values())
        expected["environment_transitions_are_separate"] = True
        if manifest.get("budget") != expected:
            issues.append("manifest rollout budget does not equal its design dimensions")
    analysis = manifest.get("analysis", {})
    if analysis.get("zero_gradient_calibration") != {
        "detection": "scaled_l2_exact_zero_only",
        "alpha_sentinel": 0.0,
        "unresolved_reason": "bank_a_gradient_exact_zero",
        "all_four_steps_zero": True,
        "gate_policy": "fail_all_affected_task_conditions",
    }:
        issues.append("manifest zero-gradient calibration policy is invalid")
    if analysis.get("primary_q") != 0.5:
        issues.append("manifest primary locality stratum is not q=0.5")
    if analysis.get("simultaneous_metric_count") != 4 * len(tasks):
        issues.append("manifest simultaneous metric family size is invalid")
    if analysis.get("primary_multiplicity") != "holm_across_exactly_3_tasks":
        issues.append("manifest primary multiplicity rule is invalid")
    if analysis.get("secondary_multiplicity") != "holm_across_exactly_6_task_by_control_contrasts":
        issues.append("manifest secondary multiplicity rule is invalid")
    if analysis.get("sensitivity_inference") != "descriptive_only":
        issues.append("nonprimary locality strata must remain descriptive")
    try:
        bound_seed_count = int(analysis["mechanism_seed_count"])
        lower_index = int(analysis["lower_order_index_zero_based"])
        upper_index = int(analysis["upper_order_index_zero_based"])
        numerator = int(analysis["per_bound_error_numerator"])
        denominator = int(analysis["per_bound_error_denominator"])
        certified_denominator = 2**bound_seed_count
        certified_numerator = sum(
            math.comb(bound_seed_count, index)
            for index in range(lower_index + 1)
        )
        delta = numerator / denominator
        simultaneous = 1.0 - analysis["simultaneous_metric_count"] * delta
        endpoint_alpha = 0.05 - analysis["simultaneous_metric_count"] * delta
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        issues.append("manifest distribution-free bound constants are invalid")
    else:
        if (
            analysis.get("mechanism_bound_method")
            != "distribution_free_one_sided_median_order_statistics"
            or bound_seed_count != len(seeds)
            or not 0 <= lower_index < upper_index < bound_seed_count
            or upper_index != bound_seed_count - 1 - lower_index
            or denominator != certified_denominator
            or numerator != certified_numerator
            or not _close(analysis.get("per_bound_error"), delta)
            or not _close(
                analysis.get("simultaneous_coverage_lower_bound"), simultaneous
            )
            or not _close(analysis.get("endpoint_family_alpha"), endpoint_alpha)
            or not _close(
                analysis.get("combined_false_advance_upper_bound"), 0.05
            )
            or not _close(
                analysis.get("gate_thresholds", {}).get(
                    "endpoint_adjusted_one_sided_p_upper"
                ),
                endpoint_alpha,
            )
        ):
            issues.append("manifest distribution-free bound design is inconsistent")
    if require_preregistered:
        expected_tasks = ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5"]
        if [task.get("env_name") for task in tasks] != expected_tasks:
            issues.append("production manifest task set is not preregistered")
        if [task.get("policy_block_sizes") for task in tasks] != [
            [768, 4160, 195],
            [1152, 4160, 390],
            [1152, 4160, 390],
        ]:
            issues.append("production policy layer block sizes are not exact")
        if [
            (
                task.get("observation_dim"),
                task.get("action_dim"),
                task.get("parameter_count"),
                task.get("policy_block_ranges"),
            )
            for task in tasks
        ] != [
            (11, 3, 5123, [[0, 768], [768, 4928], [4928, 5123]]),
            (17, 6, 5702, [[0, 1152], [1152, 5312], [5312, 5702]]),
            (17, 6, 5702, [[0, 1152], [1152, 5312], [5312, 5702]]),
        ]:
            issues.append("production policy dimension/range locks are not exact")
        if seeds != list(range(300, 320)):
            issues.append("production manifest seeds are not 300 through 319")
        if (
            analysis.get("mechanism_seed_count") != 20
            or analysis.get("lower_order_index_zero_based") != 3
            or analysis.get("upper_order_index_zero_based") != 16
            or analysis.get("per_bound_error_numerator") != 1351
            or analysis.get("per_bound_error_denominator") != 1048576
        ):
            issues.append("production median order-statistic constants are not exact")
        if generations != [50, 150, 250]:
            issues.append("production checkpoint generations are not exact")
        if dims and (
            dims["pairs_per_bank"] != 2000
            or dims["bank_b_partition_count"] != 20
            or dims["pairs_per_partition"] != 100
            or dims["training_updates"] != 250
        ):
            issues.append("production high-sample dimensions are not preregistered")
    return issues


def load_and_validate_manifest(
    path: str, *, expected_sha256: str, require_preregistered: bool = True
) -> tuple[dict[str, Any], str]:
    if not _is_sha256(expected_sha256):
        raise SubspaceValidationError(["expected manifest digest is invalid"])
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise SubspaceValidationError(
            [f"manifest digest mismatch: expected {expected_sha256}, found {actual}"]
        )
    manifest = _read_json(path)
    issues = _validate_manifest_structure(
        manifest, require_preregistered=require_preregistered
    )
    if issues:
        raise SubspaceValidationError(issues)
    protocol_path = os.path.join(REPO_ROOT, manifest["protocol"]["path"])
    if not os.path.isfile(protocol_path):
        raise SubspaceValidationError(["manifest protocol file is missing"])
    protocol_sha256 = _sha256_file(protocol_path)
    if protocol_sha256 != manifest["protocol"]["sha256"]:
        raise SubspaceValidationError(
            [
                "manifest embedded protocol digest is stale: "
                f"expected {manifest['protocol']['sha256']}, found {protocol_sha256}"
            ]
        )
    return manifest, actual


def _expected_analysis_declaration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    analysis = manifest["analysis"]
    return {
        "primary_q": analysis["primary_q"],
        "top_level_cluster": analysis["top_level_cluster"],
        "checkpoints_are_repeated_measures": True,
        "partitions_are_repeated_measures": True,
        "endpoint_seeds_are_repeated_measures": True,
        "paired_across_tasks_by_seed_number": analysis[
            "paired_across_tasks_by_seed_number"
        ],
        "seed_reductions": analysis["seed_reductions"],
        "simultaneous_metric_count": analysis["simultaneous_metric_count"],
        "mechanism_bound_method": analysis["mechanism_bound_method"],
        "mechanism_seed_count": analysis["mechanism_seed_count"],
        "lower_order_index_zero_based": analysis["lower_order_index_zero_based"],
        "upper_order_index_zero_based": analysis["upper_order_index_zero_based"],
        "per_bound_error": analysis["per_bound_error"],
        "simultaneous_coverage_lower_bound": analysis[
            "simultaneous_coverage_lower_bound"
        ],
        "endpoint_family_alpha": analysis["endpoint_family_alpha"],
        "primary_endpoint_test": analysis["endpoint_test"],
        "primary_multiplicity": analysis["primary_multiplicity"],
        "secondary_multiplicity": analysis["secondary_multiplicity"],
        "sensitivity_inference": analysis["sensitivity_inference"],
        "zero_gradient_calibration": analysis["zero_gradient_calibration"],
        "no_outcome_selection": True,
        "no_record_exclusions": True,
        "complete_case_policy": "refuse_analysis_if_any_planned_record_is_invalid",
    }


def _validate_q_summary(
    value: Any,
    label: str,
    manifest: Mapping[str, Any],
    expected_q: float,
    locked_alpha: float | None,
    issues: list[str],
    locked_alpha_resolved: bool | None = None,
    locked_alpha_unresolved_reason: str | None = None,
) -> None:
    if not _append_schema_issue(value, Q_SUMMARY_KEYS, label, issues):
        return
    if value["q"] != expected_q:
        issues.append(f"{label}: q identity is invalid")
    numeric = Q_SUMMARY_KEYS - {
        "q",
        "alpha_resolved",
        "alpha_unresolved_reason",
        "random_control_valid",
        "material_denominator_resolved",
        "finite",
        "action_sha256",
    }
    for key in numeric:
        if not _finite(value[key]):
            issues.append(f"{label}.{key}: value is nonfinite")
    if value["finite"] is not True:
        issues.append(f"{label}: producer marked numerical result nonfinite")
    zero_policy = manifest["analysis"]["zero_gradient_calibration"]
    resolved = value["alpha_resolved"]
    reason = value["alpha_unresolved_reason"]
    if not isinstance(resolved, bool):
        issues.append(f"{label}: alpha_resolved is not boolean")
    elif resolved:
        if reason is not None or not _finite(value["alpha"]) or float(value["alpha"]) <= 0.0:
            issues.append(f"{label}: resolved alpha state is inconsistent")
    elif (
        reason != zero_policy["unresolved_reason"]
        or not _close(value["alpha"], zero_policy["alpha_sentinel"])
    ):
        issues.append(f"{label}: unresolved alpha sentinel/reason is inconsistent")
    if locked_alpha_resolved is not None and (
        resolved is not locked_alpha_resolved
        or reason != locked_alpha_unresolved_reason
    ):
        issues.append(f"{label}: alpha resolution state is not frozen from complete Bank A")
    if not isinstance(value["action_sha256"], dict) or set(value["action_sha256"]) != set(
        manifest["dimensions"]["endpoint_arms"]
    ) or any(not _is_sha256(item) for item in value["action_sha256"].values()):
        issues.append(f"{label}: action digest map is invalid")
    if locked_alpha is not None and not _close(value["alpha"], locked_alpha):
        issues.append(f"{label}: alpha is not frozen from complete Bank A")
    epsilon = manifest["analysis"]["machine_epsilon"]
    sigma = manifest["dimensions"]["noise_std"]
    if _finite(value["structured_norm"]) and _finite(value["isotropic_norm"]):
        error = _relative_error(
            float(value["structured_norm"]), float(value["isotropic_norm"]), epsilon
        )
        if error > NORM_TOLERANCE or not _close(
            error,
            value["structured_isotropic_relative_norm_error"],
            tolerance=1e-10,
        ):
            issues.append(f"{label}: structured/isotropic norm match is invalid")
    if value["random_control_valid"] is True:
        error = _relative_error(
            float(value["structured_norm"]), float(value["random_norm"]), epsilon
        )
        if error > NORM_TOLERANCE or not _close(
            error,
            value["structured_random_relative_norm_error"],
            tolerance=1e-10,
        ):
            issues.append(f"{label}: structured/random norm match is invalid")
    elif not (
        _close(value["random_raw_norm"], 0.0)
        and float(value["structured_norm"]) > 0.0
    ):
        issues.append(f"{label}: invalid random control is not the prespecified zero case")
    expected_locality = float(value["structured_norm"]) / sigma
    if not _close(expected_locality, value["structured_step_over_sigma"], tolerance=1e-10):
        issues.append(f"{label}: structured locality ratio is inconsistent")
    if locked_alpha is not None:
        expected_explicit = locked_alpha * float(value["gradient_norm"])
        if not _close(expected_explicit, value["explicit_norm"], tolerance=1e-10):
            issues.append(f"{label}: explicit step does not equal locked alpha times gradient")
    if resolved is False:
        zero_fields = (
            "structured_norm",
            "isotropic_norm",
            "explicit_norm",
            "random_norm",
            "random_raw_norm",
            "anisotropic_action_norm",
            "anisotropic_minus_bank_a_norm",
            "structured_step_over_sigma",
        )
        if any(not _close(value[field], 0.0) for field in zero_fields):
            issues.append(f"{label}: unresolved alpha did not retain four zero steps")
    for residual in ("structured_solve_residual", "random_solve_residual"):
        if float(value[residual]) > NORM_TOLERANCE:
            issues.append(f"{label}: {residual} exceeds numerical tolerance")


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _validate_diagnostic_data(
    data: Mapping[str, np.ndarray] | None,
    *,
    label: str,
    manifest: Mapping[str, Any],
    theta: np.ndarray,
    signed_noise: np.ndarray,
    paired_returns: np.ndarray,
    basis: np.ndarray,
    random_basis: np.ndarray,
    q_summaries: Sequence[Mapping[str, Any]],
    reference_curvature: np.ndarray | None,
    reference_actions: Mapping[float, np.ndarray] | None,
    issues: list[str],
    endpoint_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Recompute LOPO, jackknife, four steps, and Section-10 diagnostics."""

    if data is None:
        return None
    q_values = manifest["dimensions"]["locality_q"]
    arms = manifest["dimensions"]["endpoint_arms"]
    q_count = len(q_values)
    arm_count = len(arms)
    pair_count = paired_returns.shape[0]
    dimension = theta.size
    numeric_shapes: dict[str, tuple[int, ...]] = {
        "utilities": (pair_count, 2),
        "curvature": (3, 3),
        "random_curvature": (3, 3),
        "curvature_vech_covariance": (6, 6),
        "curvature_eigenvalues": (q_count, 3),
        "negative_eigenvalue_count": (),
        "jackknife_eigenvalue_se": (q_count, 3),
        "structured_action_covariance_trace": (q_count,),
        "anisotropic_action_covariance_trace": (q_count,),
        "anisotropic_action_aligned_variance": (q_count,),
        "b_frobenius_absolute_error_to_bank_a": (),
        "negative_eigenvalue_sign_agreement_to_bank_a": (),
        "anisotropic_action_cosine_to_bank_a": (q_count,),
        "anisotropic_action_relative_error_to_bank_a": (q_count,),
        "r_full": (q_count, arm_count),
        "r_sub": (q_count, arm_count),
        "r_sn": (q_count, arm_count),
        "normalized_ess_ratio": (q_count, arm_count),
        "ratio_coefficient_of_variation": (q_count, arm_count),
        "mean_unnormalized_ratio_minus_one": (q_count, arm_count),
        "log_ratio_span": (q_count, arm_count),
        "alpha_max_concave_eigenvalue": (q_count,),
        "structured_explicit_angle_degrees": (q_count,),
        "multiplier_standard_deviation": (q_count,),
        "multiplier_range": (q_count,),
        "gradient_endpoint_relative_error": (),
        "subspace_jacobian_relative_error": (),
        "self_normalized_gradient_relative_error": (),
        "self_normalized_jacobian_relative_error": (),
    }
    boolean_shapes = {
        "repeated_eigenvalue_unresolved": (q_count,),
        "projection_boundary_unresolved": (q_count,),
        "zero_anisotropic_action_unresolved": (q_count,),
        "r_full_unresolved": (q_count, arm_count),
        "r_sub_unresolved": (q_count, arm_count),
        "r_sn_unresolved": (q_count, arm_count),
    }
    for name, shape in numeric_shapes.items():
        value = np.asarray(data[name])
        if value.shape != shape or not np.issubdtype(value.dtype, np.number) or not np.all(
            np.isfinite(value)
        ):
            issues.append(f"{label}.{name}: shape, dtype, or finiteness is invalid")
    for name, shape in boolean_shapes.items():
        value = np.asarray(data[name])
        if value.shape != shape or value.dtype != np.bool_:
            issues.append(f"{label}.{name}: unresolved-mask schema is invalid")
    for name in ("gradient_sha256", "gradient_component_variance_sha256"):
        value = np.asarray(data[name])
        if value.shape != () or value.dtype != np.dtype("S64"):
            issues.append(f"{label}.{name}: digest scalar schema is invalid")
    step_hashes = np.asarray(data["step_sha256"])
    if step_hashes.shape != (q_count, arm_count) or step_hashes.dtype != np.dtype("S64"):
        issues.append(f"{label}.step_sha256: digest matrix schema is invalid")
    if any(issue.startswith(f"{label}.") for issue in issues):
        return None
    nonnegative = (
        "jackknife_eigenvalue_se",
        "structured_action_covariance_trace",
        "anisotropic_action_covariance_trace",
        "anisotropic_action_aligned_variance",
        "b_frobenius_absolute_error_to_bank_a",
        "anisotropic_action_relative_error_to_bank_a",
        "r_full",
        "r_sub",
        "r_sn",
        "ratio_coefficient_of_variation",
        "log_ratio_span",
        "alpha_max_concave_eigenvalue",
        "multiplier_standard_deviation",
        "multiplier_range",
        "gradient_endpoint_relative_error",
        "subspace_jacobian_relative_error",
        "self_normalized_gradient_relative_error",
        "self_normalized_jacobian_relative_error",
    )
    for name in nonnegative:
        if np.any(np.asarray(data[name], dtype=np.float64) < 0.0):
            issues.append(f"{label}.{name}: diagnostic must be nonnegative")
    if np.any(np.asarray(data["normalized_ess_ratio"]) < 0.0) or np.any(
        np.asarray(data["normalized_ess_ratio"]) > 1.0 + 1e-12
    ):
        issues.append(f"{label}.normalized_ess_ratio: value is outside [0,1]")
    if np.any(np.abs(np.asarray(data["anisotropic_action_cosine_to_bank_a"])) > 1.0 + 1e-12):
        issues.append(f"{label}: anisotropic action cosine is outside [-1,1]")
    if not 0.0 <= float(data["negative_eigenvalue_sign_agreement_to_bank_a"]) <= 1.0:
        issues.append(f"{label}: eigenvalue sign agreement is outside [0,1]")

    provenance = BasisProvenance.strictly_lagged(
        primary_reference=f"{label}:primary",
        random_reference=f"{label}:random",
    )
    endpoint_kwargs: dict[str, Any] = {}
    if endpoint_reference is not None:
        endpoint_kwargs = {
            "endpoint_reference_noise": endpoint_reference.get("signed_noise"),
            "endpoint_reference_utilities": endpoint_reference.get("utilities"),
            "endpoint_reference_curvature": endpoint_reference.get("curvature"),
            "endpoint_reference_gradient": endpoint_reference.get("gradient"),
        }
    recomputed_by_q: list[Any] = []
    try:
        for q_index, q in enumerate(q_values):
            recomputed_by_q.append(
                analyze_lagged_subspace_population(
                    theta,
                    signed_noise,
                    paired_returns,
                    manifest["dimensions"]["noise_std"],
                    basis,
                    random_basis,
                    float(q_summaries[q_index]["alpha"]),
                    basis_provenance=provenance,
                    **endpoint_kwargs,
                )
            )
    except (ValueError, FloatingPointError, RuntimeError, np.linalg.LinAlgError) as error:
        issues.append(f"{label}: independent diagnostic recomputation failed: {error}")
        return None
    first = recomputed_by_q[0]
    if endpoint_reference is None:
        endpoint_reference = {
            "signed_noise": signed_noise,
            "utilities": first.estimate.utilities,
            "gradient": first.estimate.gradient,
            "curvature": first.estimate.curvature,
            "origin_metrics": {
                "gradient_endpoint_relative_error": first.gradient_endpoint_relative_error,
                "subspace_jacobian_relative_error": first.subspace_jacobian_relative_error,
                "self_normalized_gradient_relative_error": first.self_normalized_gradient_relative_error,
                "self_normalized_jacobian_relative_error": first.self_normalized_jacobian_relative_error,
            },
        }
    try:
        endpoint_signed_noise = np.asarray(
            endpoint_reference["signed_noise"], dtype=np.float64
        ).reshape(-1, dimension)
        endpoint_utilities = np.asarray(
            endpoint_reference["utilities"], dtype=np.float64
        ).reshape(-1)
        endpoint_curvature = np.asarray(
            endpoint_reference["curvature"], dtype=np.float64
        )
        endpoint_gradient = np.asarray(
            endpoint_reference["gradient"], dtype=np.float64
        )
        endpoint_origin_metrics = endpoint_reference["origin_metrics"]
        if (
            endpoint_signed_noise.shape[0] != endpoint_utilities.size
            or endpoint_gradient.shape != (dimension,)
            or endpoint_curvature.shape != (3, 3)
            or set(endpoint_origin_metrics) != {
                "gradient_endpoint_relative_error",
                "subspace_jacobian_relative_error",
                "self_normalized_gradient_relative_error",
                "self_normalized_jacobian_relative_error",
            }
            or any(not _finite(value) for value in endpoint_origin_metrics.values())
        ):
            raise ValueError("frozen endpoint reference schema is invalid")
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"{label}: frozen Bank-A endpoint reference is invalid: {error}")
        return None
    gradient_variance = np.square(first.estimate.gradient_jackknife.standard_error)
    exact_arrays = {
        "utilities": first.estimate.utilities,
        "curvature": first.estimate.curvature,
        "random_curvature": first.estimate.random_curvature,
        "curvature_vech_covariance": first.estimate.curvature_vech_jackknife.covariance,
    }
    for name, expected in exact_arrays.items():
        if expected is None or not np.allclose(
            np.asarray(data[name]), np.asarray(expected), rtol=1e-10, atol=1e-12
        ):
            issues.append(f"{label}.{name}: disagrees with independent LOPO recomputation")
    if not _fixed_ascii_sha256_matches(
        data["gradient_sha256"], _array_sha256(first.estimate.gradient)
    ):
        issues.append(f"{label}.gradient_sha256: regenerated gradient digest is invalid")
    if not _fixed_ascii_sha256_matches(
        data["gradient_component_variance_sha256"],
        _array_sha256(gradient_variance),
    ):
        issues.append(
            f"{label}.gradient_component_variance_sha256: regenerated variance digest is invalid"
        )
    actions: dict[float, np.ndarray] = {}
    expected_jk_eigen = []
    for q_index, (q, recomputed) in enumerate(zip(q_values, recomputed_by_q)):
        steps = recomputed.steps
        step_map = {
            "structured": steps.structured,
            "isotropic": steps.isotropic,
            "explicit": steps.explicit,
            "random": steps.random,
        }
        summary = q_summaries[q_index]
        action = steps.structured - steps.isotropic
        actions[q] = action
        expected_values = {
            "gradient_norm": _stable_norm(recomputed.estimate.gradient),
            "structured_norm": np.linalg.norm(steps.structured),
            "isotropic_norm": np.linalg.norm(steps.isotropic),
            "explicit_norm": np.linalg.norm(steps.explicit),
            "random_norm": np.linalg.norm(steps.random),
            "random_raw_norm": np.linalg.norm(steps.random_raw),
            "anisotropic_action_norm": np.linalg.norm(action),
            "structured_solve_residual": steps.structured_solve_relative_residual,
            "random_solve_residual": steps.random_solve_relative_residual,
            "structured_isotropic_relative_norm_error": steps.isotropic_norm_match_relative_error,
            "structured_random_relative_norm_error": (
                0.0
                if steps.random_norm_match_relative_error is None
                else steps.random_norm_match_relative_error
            ),
        }
        for name, expected in expected_values.items():
            if not _close(summary[name], expected, tolerance=1e-10):
                issues.append(f"{label}, q={q}: summary {name} disagrees with raw arrays")
        if summary["random_control_valid"] is not steps.random_control_valid:
            issues.append(f"{label}, q={q}: random-control validity is inconsistent")
        for arm, expected in step_map.items():
            expected_hash = _array_sha256(expected)
            stored_hash = np.asarray(data["step_sha256"])[
                q_index, arms.index(arm)
            ].item().decode("ascii")
            if stored_hash != expected_hash or summary["action_sha256"][arm] != expected_hash:
                issues.append(f"{label}, q={q}, arm={arm}: action content digest is invalid")
        endpoint_rows = [
            frozen_endpoint_diagnostics(
                endpoint_signed_noise,
                endpoint_utilities,
                step_map[arm],
                manifest["dimensions"]["noise_std"],
                basis,
                endpoint_curvature,
            )
            for arm in arms
        ]
        endpoint_expected = {
            "r_full": [row.full_linearization_residual for row in endpoint_rows],
            "r_sub": [row.restricted_linearization_residual for row in endpoint_rows],
            "r_sn": [row.self_normalized_linearization_residual for row in endpoint_rows],
            "normalized_ess_ratio": [row.normalized_ess_ratio for row in endpoint_rows],
            "ratio_coefficient_of_variation": [
                row.ratio_coefficient_of_variation for row in endpoint_rows
            ],
            "mean_unnormalized_ratio_minus_one": [
                row.mean_unnormalized_ratio_minus_one for row in endpoint_rows
            ],
            "log_ratio_span": [row.log_ratio_span for row in endpoint_rows],
        }
        residual_masks = {
            "r_full": np.asarray(data["r_full_unresolved"])[q_index],
            "r_sub": np.asarray(data["r_sub_unresolved"])[q_index],
            "r_sn": np.asarray(data["r_sn_unresolved"])[q_index],
        }
        for name, values in endpoint_expected.items():
            expected_mask = np.asarray([value is None for value in values])
            if name in residual_masks:
                if not np.array_equal(residual_masks[name], expected_mask):
                    issues.append(f"{label}, q={q}: {name} unresolved mask is inconsistent")
                retained = np.asarray(
                    [0.0 if value is None else value for value in values]
                )
            else:
                retained = np.asarray(values)
            if not np.allclose(
                np.asarray(data[name])[q_index], retained, rtol=1e-10, atol=1e-12
            ):
                issues.append(f"{label}, q={q}: {name} disagrees with recomputation")
        action_metrics = recomputed.action_metrics
        scalar_expected = {
            "alpha_max_concave_eigenvalue": action_metrics.alpha_max_concave_eigenvalue,
            "structured_explicit_angle_degrees": action_metrics.structured_explicit_angle_degrees,
            "multiplier_standard_deviation": action_metrics.multiplier_standard_deviation,
            "multiplier_range": action_metrics.multiplier_range,
            "structured_action_covariance_trace": recomputed.nonlinear_jackknife.structured_action_covariance_trace,
            "anisotropic_action_covariance_trace": recomputed.nonlinear_jackknife.anisotropic_action_covariance_trace,
            "anisotropic_action_aligned_variance": (
                0.0
                if recomputed.nonlinear_jackknife.anisotropic_action_aligned_variance is None
                else recomputed.nonlinear_jackknife.anisotropic_action_aligned_variance
            ),
        }
        for name, expected in scalar_expected.items():
            if expected is not None and not _close(
                np.asarray(data[name])[q_index], expected, tolerance=1e-10
            ):
                issues.append(f"{label}, q={q}: {name} disagrees with recomputation")
        expected_jk_eigen.append(
            np.sqrt(
                np.maximum(
                    np.diag(recomputed.nonlinear_jackknife.eigenvalue_covariance),
                    0.0,
                )
            )
        )
        nonlinear_flags = {
            "repeated_eigenvalue_unresolved": recomputed.nonlinear_jackknife.repeated_eigenvalue_unresolved,
            "projection_boundary_unresolved": recomputed.nonlinear_jackknife.projection_boundary_unresolved,
            "zero_anisotropic_action_unresolved": recomputed.nonlinear_jackknife.zero_anisotropic_action_unresolved,
        }
        for name, expected in nonlinear_flags.items():
            if bool(np.asarray(data[name])[q_index]) is not bool(expected):
                issues.append(f"{label}, q={q}: {name} is inconsistent")
        expected_eigen = recomputed.steps.curvature_eigenvalues
        if not np.allclose(
            np.asarray(data["curvature_eigenvalues"])[q_index],
            expected_eigen,
            rtol=1e-10,
            atol=1e-12,
        ):
            issues.append(f"{label}, q={q}: eigenvalues disagree with recomputation")
    if not np.allclose(
        np.asarray(data["jackknife_eigenvalue_se"]),
        np.asarray(expected_jk_eigen),
        rtol=1e-10,
        atol=1e-12,
    ):
        issues.append(f"{label}: nonlinear eigenvalue jackknife SE is inconsistent")
    expected_negative = int(
        np.sum(recomputed_by_q[0].steps.curvature_eigenvalues < 0.0)
    )
    if int(np.asarray(data["negative_eigenvalue_count"]).item()) != expected_negative:
        issues.append(f"{label}: negative-eigenvalue count is inconsistent")
    q_independent = {
        name: endpoint_origin_metrics[name]
        for name in (
            "gradient_endpoint_relative_error",
            "subspace_jacobian_relative_error",
            "self_normalized_gradient_relative_error",
            "self_normalized_jacobian_relative_error",
        )
    }
    for name, expected in q_independent.items():
        if not _close(data[name], expected, tolerance=1e-10):
            issues.append(f"{label}.{name}: Jacobian diagnostic is inconsistent")
    if reference_curvature is None:
        reference_curvature = first.estimate.curvature
    expected_frobenius = float(np.linalg.norm(first.estimate.curvature - reference_curvature))
    if not _close(data["b_frobenius_absolute_error_to_bank_a"], expected_frobenius, tolerance=1e-10):
        issues.append(f"{label}: Bank-A curvature Frobenius error is inconsistent")
    signs = np.sign(np.linalg.eigvalsh(first.estimate.curvature))
    reference_signs = np.sign(np.linalg.eigvalsh(reference_curvature))
    expected_sign = float(np.mean(signs == reference_signs))
    if not _close(
        data["negative_eigenvalue_sign_agreement_to_bank_a"],
        expected_sign,
        tolerance=1e-10,
    ):
        issues.append(f"{label}: eigenvalue sign agreement is inconsistent")
    if reference_actions is None:
        reference_actions = actions
    for q_index, q in enumerate(q_values):
        expected_reference = reference_actions[q]
        expected_cosine = _cosine(actions[q], expected_reference)
        expected_relative = float(
            np.linalg.norm(actions[q] - expected_reference)
            / max(np.linalg.norm(expected_reference), np.finfo(np.float64).eps)
        )
        expected_distance = float(np.linalg.norm(actions[q] - expected_reference))
        if not _close(
            q_summaries[q_index]["anisotropic_minus_bank_a_norm"],
            expected_distance,
            tolerance=1e-10,
        ):
            issues.append(f"{label}, q={q}: Bank-A action distance is inconsistent")
        if not _close(
            np.asarray(data["anisotropic_action_cosine_to_bank_a"])[q_index],
            expected_cosine,
            tolerance=1e-10,
        ) or not _close(
            np.asarray(data["anisotropic_action_relative_error_to_bank_a"])[q_index],
            expected_relative,
            tolerance=1e-10,
        ):
            issues.append(f"{label}, q={q}: Bank-A action comparison is inconsistent")
    primary_index = q_values.index(manifest["analysis"]["primary_q"])
    return {
        "curvature": first.estimate.curvature,
        "actions": actions,
        "gradient": first.estimate.gradient,
        "endpoint_reference": endpoint_reference,
        "jackknife_sha256": _diagnostic_jackknife_sha256(
            {
                "gradient_component_variance": gradient_variance,
                **{
                    name: np.asarray(data[name])
                    for name in (
                        "curvature_vech_covariance",
                        "jackknife_eigenvalue_se",
                        "structured_action_covariance_trace",
                        "anisotropic_action_covariance_trace",
                        "anisotropic_action_aligned_variance",
                    )
                },
            }
        ),
        "primary_diagnostics_resolved": bool(
            not np.any(np.asarray(data["r_full_unresolved"])[primary_index])
            and not np.any(np.asarray(data["r_sub_unresolved"])[primary_index])
            and not np.any(np.asarray(data["r_sn_unresolved"])[primary_index])
            and not bool(
                np.asarray(data["zero_anisotropic_action_unresolved"])[primary_index]
            )
            and not bool(
                np.asarray(data["projection_boundary_unresolved"])[primary_index]
            )
            and not bool(
                np.asarray(data["repeated_eigenvalue_unresolved"])[primary_index]
            )
        ),
    }


def validate_artifact(
    artifact: Any,
    manifest: Mapping[str, Any],
    *,
    expected_hashes: Mapping[str, str],
    artifact_root: str,
    require_preregistered_manifest: bool = True,
) -> dict[str, Any]:
    """Validate the complete audit index and return indexed records.

    The function never drops a bad record. Any issue rejects the complete study.
    """

    issues = _validate_manifest_structure(
        manifest, require_preregistered=require_preregistered_manifest
    )
    if not os.path.isdir(artifact_root):
        issues.append("artifact root is missing or not a directory")
    file_cache: dict[str, str] = {}
    issues.extend(_find_inference_leakage(artifact))
    if not isinstance(artifact, dict) or set(artifact) != TOP_LEVEL_KEYS:
        issues.append("artifact top-level schema is not exact")
        raise SubspaceValidationError(issues)
    if artifact["schema_version"] != 1 or artifact["study"] != STUDY:
        issues.append("artifact study/schema version is invalid")
    if artifact["designation"] != manifest["designation"]:
        issues.append("artifact designation is inconsistent with manifest")
    required_hashes = manifest["required_hash_locks"]
    if set(expected_hashes) != set(required_hashes) or any(
        not _is_sha256(value) for value in expected_hashes.values()
    ):
        issues.append("expected hash-lock set is incomplete or invalid")
    provenance = artifact["provenance"]
    if _append_schema_issue(provenance, PROVENANCE_KEYS, "provenance", issues):
        _validate_record_hash(provenance, "provenance", issues)
        for key in required_hashes:
            if provenance.get(key) != expected_hashes.get(key):
                issues.append(f"provenance.{key}: does not match the enforced lock")
        if artifact["manifest_sha256"] != provenance["manifest_sha256"]:
            issues.append("artifact manifest digest disagrees with provenance")
        if provenance["protocol_sha256"] != manifest["protocol"]["sha256"]:
            issues.append("protocol digest does not match manifest")
        if provenance["stderr_empty"] is not True:
            issues.append("study-level stderr is not empty")
        if provenance["documented_infrastructure_failures"] != []:
            issues.append("final artifact contains unresolved infrastructure failures")
        if not isinstance(provenance["source_snapshot_path"], str) or not provenance[
            "source_snapshot_path"
        ]:
            issues.append("source snapshot path is missing")
    if artifact["analysis_declaration"] != _expected_analysis_declaration(manifest):
        issues.append("analysis declaration changes clustering, gates, or multiplicity")

    dims = manifest["dimensions"]
    tasks = manifest["tasks"]
    seeds = manifest["training_seeds"]
    generations = manifest["checkpoint_generations"]
    q_values = dims["locality_q"]
    arms = dims["endpoint_arms"]
    expected_training = len(tasks) * len(seeds)
    expected_checkpoints = expected_training * len(generations)
    expected_banks = expected_checkpoints * 2
    expected_partitions = expected_checkpoints * dims["bank_b_partition_count"]
    expected_metrics = expected_checkpoints * len(q_values)
    expected_centers = expected_checkpoints * dims["endpoint_episodes"]
    expected_endpoints = (
        expected_checkpoints
        * len(q_values)
        * dims["bank_b_partition_count"]
        * len(arms)
        * dims["endpoint_episodes"]
    )
    expected_counts = {
        "training_runs": expected_training,
        "checkpoints": expected_checkpoints,
        "banks": expected_banks,
        "partitions": expected_partitions,
        "checkpoint_metrics": expected_metrics,
        "center_endpoints": expected_centers,
        "endpoints": expected_endpoints,
    }
    for field, count in expected_counts.items():
        value = artifact[field]
        if not isinstance(value, list):
            issues.append(f"{field}: is not a list")
        elif len(value) != count:
            issues.append(f"{field}: contains {len(value)} records, expected {count}")

    training_by_id: dict[int, dict[str, Any]] = {}
    training_transition_sum = 0
    calibration_transition_sum = 0
    for index, record in enumerate(artifact["training_runs"] if isinstance(artifact["training_runs"], list) else []):
        label = f"training_runs[{index}]"
        if not _append_schema_issue(record, TRAINING_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        identity = record["training_id"]
        if identity in training_by_id:
            issues.append(f"{label}: duplicate training id {identity}")
        training_by_id[identity] = record
        if record["task_index"] not in range(len(tasks)) or record["training_seed"] not in seeds:
            issues.append(f"{label}: task/seed is outside the manifest")
            continue
        expected_id = training_id_for(manifest, record["task_index"], record["training_seed"])
        if identity != expected_id:
            issues.append(f"{label}: training id mapping is invalid")
        if record["env_name"] != tasks[record["task_index"]]["env_name"]:
            issues.append(f"{label}: environment name is invalid")
        exact = {
            "updates": dims["training_updates"],
            "population_size": dims["population_size"],
            "candidate_rollouts": dims["training_updates"] * dims["population_size"],
            "calibration_rollouts": dims["calibration_episodes"],
            "online_evaluation_rollouts": 0,
            "checkpoint_generations": generations,
            "stderr_sha256": EMPTY_SHA256,
            "stderr_empty": True,
        }
        for key, expected in exact.items():
            if record[key] != expected:
                issues.append(f"{label}.{key}: is not locked to {expected!r}")
        for key in ("training_log_sha256",):
            if not _is_sha256(record[key]):
                issues.append(f"{label}.{key}: digest is invalid")
        _verified_artifact_path(
            artifact_root,
            record["training_log_path"],
            record["training_log_sha256"],
            f"{label}.training_log",
            issues,
            file_cache,
        )
        if not isinstance(record["training_transitions"], int) or record["training_transitions"] <= 0:
            issues.append(f"{label}: training transitions are invalid")
        else:
            training_transition_sum += record["training_transitions"]
        if not isinstance(record["calibration_transitions"], int) or record["calibration_transitions"] <= 0:
            issues.append(f"{label}: calibration transitions are invalid")
        else:
            calibration_transition_sum += record["calibration_transitions"]
    if set(training_by_id) != set(range(expected_training)):
        issues.append("training-run identities are partial, duplicated, or noncontiguous")

    checkpoint_by_id: dict[int, dict[str, Any]] = {}
    checkpoint_numeric: dict[int, dict[str, np.ndarray]] = {}
    for index, record in enumerate(artifact["checkpoints"] if isinstance(artifact["checkpoints"], list) else []):
        label = f"checkpoints[{index}]"
        if not _append_schema_issue(record, CHECKPOINT_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        checkpoint_id = record["checkpoint_id"]
        if checkpoint_id in checkpoint_by_id:
            issues.append(f"{label}: duplicate checkpoint id {checkpoint_id}")
        checkpoint_by_id[checkpoint_id] = record
        try:
            task_index, seed, generation = checkpoint_coordinates(manifest, checkpoint_id)
        except (TypeError, ValueError) as error:
            issues.append(f"{label}: {error}")
            continue
        training_id = training_id_for(manifest, task_index, seed)
        exact = {
            "training_id": training_id,
            "task_index": task_index,
            "env_name": tasks[task_index]["env_name"],
            "training_seed": seed,
            "generation": generation,
            "source_sha256": expected_hashes.get("source_sha256"),
            "prior_gradient_indices": list(
                range(generation - dims["lagged_gradient_count"], generation)
            ),
            "basis_seed": checkpoint_seed(manifest, "basis", checkpoint_id),
            "random_control_seed": checkpoint_seed(
                manifest, "random_control", checkpoint_id
            ),
        }
        for key, expected in exact.items():
            if record[key] != expected:
                issues.append(f"{label}.{key}: checkpoint lineage is invalid")
        prior_hashes = record["prior_gradient_sha256"]
        if not isinstance(prior_hashes, list) or len(prior_hashes) != dims["lagged_gradient_count"] or any(
            not _is_sha256(value) for value in prior_hashes
        ):
            issues.append(f"{label}: strictly prior gradient digest list is invalid")
        block_count = dims["layer_blocks"]
        block_norms = record["lagged_block_norms"]
        zero_mask = record["lagged_block_exact_zero"]
        primary_fallback = record["primary_gaussian_fallback_used"]
        random_fallback = record["random_control_permuted_fallback_used"]
        fallback_hashes = record["fallback_column_sha256"]
        valid_zero_mask = (
            isinstance(zero_mask, list)
            and len(zero_mask) == block_count
            and all(isinstance(value, bool) for value in zero_mask)
        )
        if (
            not isinstance(block_norms, list)
            or len(block_norms) != block_count
            or any(not _finite(value) or float(value) < 0.0 for value in block_norms)
        ):
            issues.append(f"{label}: lagged block norms are invalid")
        elif not valid_zero_mask:
            issues.append(f"{label}: exact-zero block mask is invalid")
        else:
            expected_zero = [float(value) == 0.0 for value in block_norms]
            if zero_mask != expected_zero:
                issues.append(f"{label}: fallback mask does not use exact zero")
        if not valid_zero_mask or primary_fallback != zero_mask or random_fallback != zero_mask:
            issues.append(
                f"{label}: primary/random fallback records do not match the exact-zero mask"
            )
        if (
            not isinstance(fallback_hashes, list)
            or len(fallback_hashes) != block_count
            or not valid_zero_mask
            or any(
                (used and not _is_sha256(value)) or (not used and value is not None)
                for used, value in zip(zero_mask, fallback_hashes)
            )
        ):
            issues.append(f"{label}: fallback column digest record is invalid")
        if record["basis_locked_before_bank_sampling"] is not True:
            issues.append(f"{label}: basis was not locked before current banks")
        for key in (
            "parameter_sha256",
            "observation_normalizer_sha256",
            "training_config_sha256",
            "basis_sha256",
            "random_basis_sha256",
            "checkpoint_artifact_sha256",
        ):
            if not _is_sha256(record[key]):
                issues.append(f"{label}.{key}: digest is invalid")
        checkpoint_path = _verified_artifact_path(
            artifact_root,
            record["checkpoint_artifact_path"],
            record["checkpoint_artifact_sha256"],
            f"{label}.checkpoint_artifact",
            issues,
            file_cache,
        )
        _verified_artifact_path(
            artifact_root,
            record["training_config_path"],
            record["training_config_sha256"],
            f"{label}.training_config",
            issues,
            file_cache,
        )
        basis_path = _verified_artifact_path(
            artifact_root,
            record["basis_artifact_path"],
            record["basis_artifact_sha256"],
            f"{label}.basis_artifact",
            issues,
            file_cache,
        )
        checkpoint_data = _load_npz_exact(
            checkpoint_path, CHECKPOINT_NPZ_KEYS, f"{label}.checkpoint_artifact", issues
        )
        basis_data = _load_npz_exact(
            basis_path, BASIS_NPZ_KEYS, f"{label}.basis_artifact", issues
        )
        if checkpoint_data is not None:
            center = np.asarray(checkpoint_data["center_params"])
            proposal = np.asarray(checkpoint_data["proposal_gradients"])
            gradient_generations = np.asarray(
                checkpoint_data["gradient_generations"]
            )
            if (
                np.asarray(checkpoint_data["schema_version"]).shape != ()
                or int(checkpoint_data["schema_version"]) != 2
                or np.asarray(checkpoint_data["checkpoint_generation"]).shape != ()
                or int(checkpoint_data["checkpoint_generation"]) != generation
                or not _fixed_ascii_sha256_matches(
                    checkpoint_data["study_source_sha256"],
                    expected_hashes["source_sha256"],
                )
                or not _fixed_ascii_sha256_matches(
                    checkpoint_data["training_config_sha256"],
                    record["training_config_sha256"],
                )
                or center.ndim != 1
                or center.size != tasks[task_index]["parameter_count"]
                or not np.all(np.isfinite(center))
                or proposal.shape != (dims["lagged_gradient_count"], center.size)
                or not np.all(np.isfinite(proposal))
                or gradient_generations.tolist() != record["prior_gradient_indices"]
            ):
                issues.append(f"{label}: raw checkpoint lineage/array schema is invalid")
            else:
                if _array_sha256(center) != record["parameter_sha256"]:
                    issues.append(f"{label}: parameter content digest is invalid")
                expected_gradient_hashes = [
                    _array_sha256(proposal[row]) for row in range(proposal.shape[0])
                ]
                if expected_gradient_hashes != record["prior_gradient_sha256"]:
                    issues.append(f"{label}: prior-gradient content digests are invalid")
                enabled = np.asarray(checkpoint_data["obs_normalizer_enabled"])
                obs_mean = np.asarray(checkpoint_data["obs_mean"])
                obs_var = np.asarray(checkpoint_data["obs_var"])
                obs_count = np.asarray(checkpoint_data["obs_count"])
                obs_digest = _labeled_arrays_sha256(
                    [
                        ("enabled", enabled),
                        ("mean", obs_mean),
                        ("var", obs_var),
                        ("count", obs_count),
                    ]
                )
                if (
                    obs_digest != record["observation_normalizer_sha256"]
                    or not np.all(np.isfinite(obs_mean))
                    or not np.all(np.isfinite(obs_var))
                    or np.any(obs_var < 0.0)
                    or not np.all(np.isfinite(obs_count))
                ):
                    issues.append(f"{label}: observation-normalizer artifact is invalid")
        if checkpoint_data is not None and basis_data is not None:
            center = np.asarray(checkpoint_data["center_params"], dtype=np.float64)
            primary = np.asarray(basis_data["primary_basis"], dtype=np.float64)
            random_basis = np.asarray(basis_data["random_basis"], dtype=np.float64)
            basis_norms = np.asarray(basis_data["lagged_block_norms"])
            basis_zero = np.asarray(basis_data["lagged_block_exact_zero"])
            basis_primary_fallback = np.asarray(
                basis_data["primary_gaussian_fallback_used"]
            )
            basis_random_fallback = np.asarray(
                basis_data["random_control_permuted_fallback_used"]
            )
            fallback_columns = np.asarray(basis_data["fallback_columns"])
            if (
                primary.shape != (center.size, dims["layer_blocks"])
                or random_basis.shape != primary.shape
                or fallback_columns.shape != primary.shape
                or basis_norms.shape != (dims["layer_blocks"],)
                or basis_zero.shape != (dims["layer_blocks"],)
                or basis_primary_fallback.shape != (dims["layer_blocks"],)
                or basis_random_fallback.shape != (dims["layer_blocks"],)
                or basis_zero.dtype != np.bool_
                or basis_primary_fallback.dtype != np.bool_
                or basis_random_fallback.dtype != np.bool_
                or not np.all(np.isfinite(primary))
                or not np.all(np.isfinite(random_basis))
                or not np.all(np.isfinite(fallback_columns))
                or not np.allclose(primary.T @ primary, np.eye(dims["layer_blocks"]), rtol=1e-10, atol=1e-12)
                or not np.allclose(random_basis.T @ random_basis, np.eye(dims["layer_blocks"]), rtol=1e-10, atol=1e-12)
            ):
                issues.append(f"{label}: basis artifact schema/orthonormality is invalid")
            else:
                try:
                    reconstructed = reconstruct_lagged_bases(
                        np.asarray(
                            checkpoint_data["proposal_gradients"],
                            dtype=np.float64,
                        ),
                        tasks[task_index]["policy_block_sizes"],
                        lagged_decay=dims["lagged_decay"],
                        basis_seed_value=record["basis_seed"],
                        random_seed_value=record["random_control_seed"],
                    )
                except (ValueError, FloatingPointError) as error:
                    issues.append(f"{label}: basis reconstruction failed: {error}")
                    reconstructed = None
                if reconstructed is not None:
                    for field in (
                        "primary_basis",
                        "random_basis",
                        "lagged_block_norms",
                        "fallback_columns",
                    ):
                        if not np.array_equal(
                            np.asarray(basis_data[field]), reconstructed[field]
                        ):
                            issues.append(
                                f"{label}: {field} disagrees with independent lagged reconstruction"
                            )
                    for field in (
                        "lagged_block_exact_zero",
                        "primary_gaussian_fallback_used",
                        "random_control_permuted_fallback_used",
                    ):
                        if not np.array_equal(
                            np.asarray(basis_data[field]), reconstructed[field]
                        ):
                            issues.append(
                                f"{label}: {field} disagrees with independent lagged reconstruction"
                            )
                if _array_sha256(primary) != record["basis_sha256"] or _array_sha256(
                    random_basis
                ) != record["random_basis_sha256"]:
                    issues.append(f"{label}: basis content digest is invalid")
                if not np.array_equal(basis_norms, np.asarray(record["lagged_block_norms"])):
                    issues.append(f"{label}: basis lagged-block norms disagree with index")
                if (
                    basis_zero.tolist() != record["lagged_block_exact_zero"]
                    or basis_primary_fallback.tolist()
                    != record["primary_gaussian_fallback_used"]
                    or basis_random_fallback.tolist()
                    != record["random_control_permuted_fallback_used"]
                ):
                    issues.append(f"{label}: basis fallback metadata disagrees with index")
                expected_fallback_hashes = [
                    _array_sha256(fallback_columns[:, block])
                    if bool(basis_zero[block])
                    else None
                    for block in range(dims["layer_blocks"])
                ]
                if expected_fallback_hashes != record["fallback_column_sha256"]:
                    issues.append(f"{label}: fallback column content digest is invalid")
                checkpoint_numeric[checkpoint_id] = {
                    "theta": center,
                    "basis": primary,
                    "random_basis": random_basis,
                }
    if set(checkpoint_by_id) != set(range(expected_checkpoints)):
        issues.append("checkpoint identities are partial, duplicated, or noncontiguous")

    bank_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    bank_numeric: dict[tuple[int, str], dict[str, Any]] = {}
    bank_transition_sum = 0
    all_perturbation_seeds: set[int] = set()
    all_bank_rollout_seeds: set[int] = set()
    for index, record in enumerate(artifact["banks"] if isinstance(artifact["banks"], list) else []):
        label = f"banks[{index}]"
        if not _append_schema_issue(record, BANK_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        key = (record["checkpoint_id"], record["bank"])
        if key in bank_by_key:
            issues.append(f"{label}: duplicate checkpoint/bank identity")
        bank_by_key[key] = record
        if record["checkpoint_id"] not in checkpoint_by_id or record["bank"] not in dims["banks"]:
            issues.append(f"{label}: checkpoint/bank identity is outside manifest")
            continue
        expected_id = bank_id_for(manifest, record["checkpoint_id"], record["bank"])
        if record["bank_id"] != expected_id:
            issues.append(f"{label}: bank id mapping is invalid")
        pair_indices = list(range(dims["pairs_per_bank"]))
        expected_perturbation = [
            pair_seed(
                manifest,
                "bank_perturbation",
                record["checkpoint_id"],
                record["bank"],
                pair_index,
            )
            for pair_index in pair_indices
        ]
        expected_rollout = [
            pair_seed(
                manifest,
                "bank_rollout",
                record["checkpoint_id"],
                record["bank"],
                pair_index,
            )
            for pair_index in pair_indices
        ]
        exact = {
            "pair_count": dims["pairs_per_bank"],
            "candidate_rollouts": 2 * dims["pairs_per_bank"],
            "pair_indices": pair_indices,
            "perturbation_seeds": expected_perturbation,
            "rollout_seeds_plus": expected_rollout,
            "rollout_seeds_minus": expected_rollout,
            "exact_antithetic": True,
            "shared_rollout_seed_within_pair": True,
            "lopo_gradient_curvature_shared": True,
            "finite_u_statistic": True,
            "finite_jackknife": True,
            "finite_eigensystem": True,
            "stderr_sha256": EMPTY_SHA256,
            "stderr_empty": True,
        }
        for field, expected in exact.items():
            if record[field] != expected:
                issues.append(f"{label}.{field}: bank mapping/invariant is invalid")
        duplicate_perturbation = all_perturbation_seeds.intersection(expected_perturbation)
        duplicate_rollout = all_bank_rollout_seeds.intersection(expected_rollout)
        if duplicate_perturbation or duplicate_rollout:
            issues.append(f"{label}: bank streams are not globally disjoint")
        all_perturbation_seeds.update(expected_perturbation)
        all_bank_rollout_seeds.update(expected_rollout)
        for field in (
            "perturbations_sha256",
            "returns_sha256",
            "transitions_sha256",
            "jackknife_sha256",
        ):
            if not _is_sha256(record[field]):
                issues.append(f"{label}.{field}: digest is invalid")
        raw_path = _verified_artifact_path(
            artifact_root,
            record["raw_bank_path"],
            record["raw_bank_sha256"],
            f"{label}.raw_bank",
            issues,
            file_cache,
        )
        diagnostic_path = _verified_artifact_path(
            artifact_root,
            record["diagnostics_path"],
            record["diagnostics_sha256"],
            f"{label}.diagnostics",
            issues,
            file_cache,
        )
        raw_data = _load_npz_exact(
            raw_path, RAW_BANK_NPZ_KEYS, f"{label}.raw_bank", issues
        )
        diagnostic_data = _load_npz_exact(
            diagnostic_path,
            _diagnostic_npz_keys(manifest),
            f"{label}.diagnostics",
            issues,
        )
        checkpoint_state = checkpoint_numeric.get(record["checkpoint_id"])
        if raw_data is not None and checkpoint_state is not None:
            dimension = checkpoint_state["theta"].size
            paired_returns = np.asarray(raw_data["paired_returns"])
            paired_transitions = np.asarray(raw_data["paired_transitions"])
            raw_perturbation_seeds = np.asarray(raw_data["perturbation_seeds"])
            raw_rollout_plus = np.asarray(raw_data["rollout_seeds_plus"])
            raw_rollout_minus = np.asarray(raw_data["rollout_seeds_minus"])
            signed_noise = _regenerate_signed_noise(expected_perturbation, dimension)
            raw_valid = (
                paired_returns.shape == (dims["pairs_per_bank"], 2)
                and paired_transitions.shape == (dims["pairs_per_bank"], 2)
                and raw_perturbation_seeds.shape == (dims["pairs_per_bank"],)
                and raw_rollout_plus.shape == (dims["pairs_per_bank"],)
                and raw_rollout_minus.shape == (dims["pairs_per_bank"],)
                and np.all(np.isfinite(paired_returns))
                and np.issubdtype(paired_transitions.dtype, np.integer)
                and np.all(paired_transitions > 0)
            )
            if not raw_valid:
                issues.append(f"{label}: raw bank array schema is invalid")
            else:
                antithetic_error = float(
                    np.max(np.abs(signed_noise[:, 0, :] + signed_noise[:, 1, :]))
                )
                if antithetic_error != 0.0 or not _close(
                    antithetic_error, record["antithetic_max_abs_error"]
                ):
                    issues.append(f"{label}: raw perturbations are not exactly antithetic")
                content_hashes = {
                    "perturbations_sha256": _array_sha256(signed_noise),
                    "returns_sha256": _array_sha256(paired_returns),
                    "transitions_sha256": _array_sha256(paired_transitions),
                }
                for field, expected in content_hashes.items():
                    if record[field] != expected:
                        issues.append(f"{label}.{field}: raw-array digest is invalid")
                if (
                    raw_perturbation_seeds.tolist() != record["perturbation_seeds"]
                    or raw_rollout_plus.tolist() != record["rollout_seeds_plus"]
                    or raw_rollout_minus.tolist() != record["rollout_seeds_minus"]
                ):
                    issues.append(f"{label}: raw bank seed arrays disagree with mapping")
                if int(np.sum(paired_transitions)) != record["candidate_transitions"]:
                    issues.append(f"{label}: bank transition total disagrees with raw array")
                if diagnostic_data is not None:
                    bank_numeric[key] = {
                        **checkpoint_state,
                        "perturbation_seeds": np.asarray(
                            expected_perturbation, dtype=np.uint64
                        ),
                        "paired_returns": paired_returns,
                        "paired_transitions": paired_transitions,
                        "diagnostics": diagnostic_data,
                    }
        if not _close(record["antithetic_max_abs_error"], 0.0):
            issues.append(f"{label}: perturbations are not exactly antithetic")
        utility_bound = UTILITY_TOLERANCE * max(
            1.0, float(record["lopo_utility_abs_sum"])
        )
        if not _finite(record["lopo_utility_sum"]) or abs(float(record["lopo_utility_sum"])) > utility_bound:
            issues.append(f"{label}: LOPO utility sum identity fails")
        for field in ("dsn_da_relative_error", "jsn_ja_relative_error"):
            if not _finite(record[field]) or float(record[field]) > ORIGIN_MAP_TOLERANCE:
                issues.append(f"{label}.{field}: at-origin map agreement fails")
        if not isinstance(record["candidate_transitions"], int) or record["candidate_transitions"] <= 0:
            issues.append(f"{label}: bank transition count is invalid")
        else:
            bank_transition_sum += record["candidate_transitions"]
        q_summaries = record["q_summaries"]
        if not isinstance(q_summaries, list) or len(q_summaries) != len(q_values):
            issues.append(f"{label}: q summaries are partial")
        else:
            for q_index, q in enumerate(q_values):
                _validate_q_summary(
                    q_summaries[q_index],
                    f"{label}.q_summaries[{q_index}]",
                    manifest,
                    q,
                    None,
                    issues,
                )
    expected_bank_keys = {
        (checkpoint_id, bank)
        for checkpoint_id in range(expected_checkpoints)
        for bank in dims["banks"]
    }
    if set(bank_by_key) != expected_bank_keys:
        issues.append("bank identities are partial, duplicated, or incomplete")

    # Complete Bank A locks alpha for every other estimate at a checkpoint.
    alpha_by_checkpoint_q: dict[tuple[int, float], float] = {}
    alpha_resolution_by_checkpoint_q: dict[
        tuple[int, float], tuple[bool, str | None]
    ] = {}
    zero_policy = manifest["analysis"]["zero_gradient_calibration"]
    for checkpoint_id in range(expected_checkpoints):
        bank_a = bank_by_key.get((checkpoint_id, "A"))
        if not bank_a or not isinstance(bank_a.get("q_summaries"), list):
            continue
        for q_index, q in enumerate(q_values):
            if q_index >= len(bank_a["q_summaries"]):
                continue
            summary = bank_a["q_summaries"][q_index]
            if not isinstance(summary, dict) or not _finite(summary.get("gradient_norm")):
                continue
            gradient_norm = float(summary["gradient_norm"])
            if gradient_norm == 0.0:
                expected_alpha = float(zero_policy["alpha_sentinel"])
                expected_resolved = False
                expected_reason = str(zero_policy["unresolved_reason"])
            elif gradient_norm > 0.0:
                expected_alpha = q * dims["noise_std"] / gradient_norm
                expected_resolved = True
                expected_reason = None
            else:
                issues.append(f"checkpoint {checkpoint_id}, q={q}: Bank-A gradient norm is invalid")
                continue
            if not _close(summary.get("alpha"), expected_alpha, tolerance=1e-10):
                issues.append(f"checkpoint {checkpoint_id}, q={q}: alpha is not q*sigma/||g_A||")
            alpha_by_checkpoint_q[(checkpoint_id, q)] = expected_alpha
            alpha_resolution_by_checkpoint_q[(checkpoint_id, q)] = (
                expected_resolved,
                expected_reason,
            )
            for bank in ("A", "B"):
                bank_record = bank_by_key.get((checkpoint_id, bank))
                if bank_record and q_index < len(bank_record.get("q_summaries", [])):
                    bank_summary = bank_record["q_summaries"][q_index]
                    if not _close(
                        bank_summary.get("alpha"),
                        expected_alpha,
                        tolerance=1e-10,
                    ) or (
                        bank_summary.get("alpha_resolved") is not expected_resolved
                        or bank_summary.get("alpha_unresolved_reason") != expected_reason
                    ):
                        issues.append(
                            f"checkpoint {checkpoint_id}, bank {bank}, q={q}: alpha state is not Bank-A locked"
                        )

    bank_recomputed: dict[tuple[int, str], dict[str, Any]] = {}
    for checkpoint_id in range(expected_checkpoints):
        reference: dict[str, Any] | None = None
        for bank in ("A", "B"):
            key = (checkpoint_id, bank)
            numeric = bank_numeric.get(key)
            record = bank_by_key.get(key)
            if numeric is None or record is None:
                continue
            signed_noise = _regenerate_signed_noise(
                numeric["perturbation_seeds"], numeric["theta"].size
            )
            result = _validate_diagnostic_data(
                numeric["diagnostics"],
                label=f"checkpoint {checkpoint_id}, bank {bank} diagnostics",
                manifest=manifest,
                theta=numeric["theta"],
                signed_noise=signed_noise,
                paired_returns=numeric["paired_returns"],
                basis=numeric["basis"],
                random_basis=numeric["random_basis"],
                q_summaries=record["q_summaries"],
                reference_curvature=(None if reference is None else reference["curvature"]),
                reference_actions=(None if reference is None else reference["actions"]),
                issues=issues,
                endpoint_reference=(
                    None if reference is None else reference["endpoint_reference"]
                ),
            )
            if result is None:
                continue
            bank_recomputed[key] = {
                name: value
                for name, value in result.items()
                if name != "endpoint_reference"
            }
            if result["jackknife_sha256"] != record["jackknife_sha256"]:
                issues.append(
                    f"checkpoint {checkpoint_id}, bank {bank}: jackknife content digest is invalid"
                )
            if bank == "A":
                reference = result
            utility_sum = float(np.sum(numeric["diagnostics"]["utilities"]))
            utility_abs = float(np.sum(np.abs(numeric["diagnostics"]["utilities"])))
            if not _close(record["lopo_utility_sum"], utility_sum, tolerance=1e-10) or not _close(
                record["lopo_utility_abs_sum"], utility_abs, tolerance=1e-10
            ):
                issues.append(
                    f"checkpoint {checkpoint_id}, bank {bank}: utility diagnostics disagree with arrays"
                )
            if not _close(
                record["dsn_da_relative_error"],
                numeric["diagnostics"]["self_normalized_gradient_relative_error"],
                tolerance=1e-10,
            ) or not _close(
                record["jsn_ja_relative_error"],
                numeric["diagnostics"]["self_normalized_jacobian_relative_error"],
                tolerance=1e-10,
            ):
                issues.append(
                    f"checkpoint {checkpoint_id}, bank {bank}: at-origin map diagnostics disagree with arrays"
                )

    partition_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    partition_numeric: dict[tuple[int, int], dict[str, Any]] = {}
    for index, record in enumerate(artifact["partitions"] if isinstance(artifact["partitions"], list) else []):
        label = f"partitions[{index}]"
        if not _append_schema_issue(record, PARTITION_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        key = (record["checkpoint_id"], record["partition_index"])
        if key in partition_by_key:
            issues.append(f"{label}: duplicate checkpoint/partition identity")
        partition_by_key[key] = record
        if record["checkpoint_id"] not in checkpoint_by_id or record["partition_index"] not in range(
            dims["bank_b_partition_count"]
        ):
            issues.append(f"{label}: partition identity is outside manifest")
            continue
        expected_partition = bank_b_partition(manifest, record["checkpoint_id"])[
            record["partition_index"]
        ]
        exact = {
            "partition_id": partition_id_for(
                manifest, record["checkpoint_id"], record["partition_index"]
            ),
            "partition_seed": checkpoint_seed(
                manifest, "bank_b_partition", record["checkpoint_id"]
            ),
            "pair_indices": expected_partition,
            "pair_count": dims["pairs_per_partition"],
            "lopo_gradient_curvature_shared": True,
            "finite_u_statistic": True,
            "finite_jackknife": True,
            "finite_eigensystem": True,
        }
        for field, expected in exact.items():
            if record[field] != expected:
                issues.append(f"{label}.{field}: partition mapping/invariant is invalid")
        utility_bound = UTILITY_TOLERANCE * max(
            1.0, float(record["lopo_utility_abs_sum"])
        )
        if not _finite(record["lopo_utility_sum"]) or abs(float(record["lopo_utility_sum"])) > utility_bound:
            issues.append(f"{label}: LOPO utility sum identity fails")
        q_summaries = record["q_summaries"]
        if not isinstance(q_summaries, list) or len(q_summaries) != len(q_values):
            issues.append(f"{label}: q summaries are partial")
        else:
            for q_index, q in enumerate(q_values):
                _validate_q_summary(
                    q_summaries[q_index],
                    f"{label}.q_summaries[{q_index}]",
                    manifest,
                    q,
                    alpha_by_checkpoint_q.get((record["checkpoint_id"], q)),
                    issues,
                    *alpha_resolution_by_checkpoint_q.get(
                        (record["checkpoint_id"], q), (None, None)
                    ),
                )
        diagnostic_path = _verified_artifact_path(
            artifact_root,
            record["diagnostics_path"],
            record["diagnostics_sha256"],
            f"{label}.diagnostics",
            issues,
            file_cache,
        )
        diagnostic_data = _load_npz_exact(
            diagnostic_path,
            _diagnostic_npz_keys(manifest),
            f"{label}.diagnostics",
            issues,
        )
        bank_b_numeric = bank_numeric.get((record["checkpoint_id"], "B"))
        if diagnostic_data is not None and bank_b_numeric is not None:
            selected = np.asarray(record["pair_indices"], dtype=np.int64)
            partition_numeric[key] = {
                "theta": bank_b_numeric["theta"],
                "basis": bank_b_numeric["basis"],
                "random_basis": bank_b_numeric["random_basis"],
                "perturbation_seeds": bank_b_numeric["perturbation_seeds"][selected],
                "paired_returns": bank_b_numeric["paired_returns"][selected],
                "diagnostics": diagnostic_data,
            }
    expected_partition_keys = {
        (checkpoint_id, partition_index)
        for checkpoint_id in range(expected_checkpoints)
        for partition_index in range(dims["bank_b_partition_count"])
    }
    if set(partition_by_key) != expected_partition_keys:
        issues.append("partition identities are partial, duplicated, or incomplete")
    partition_recomputed: dict[tuple[int, int], dict[str, Any]] = {}
    for checkpoint_id in range(expected_checkpoints):
        records = [
            partition_by_key.get((checkpoint_id, index))
            for index in range(dims["bank_b_partition_count"])
        ]
        if any(record is None for record in records):
            continue
        flattened = [pair for record in records for pair in record["pair_indices"]]
        if sorted(flattened) != list(range(dims["pairs_per_bank"])) or len(set(flattened)) != len(flattened):
            issues.append(f"checkpoint {checkpoint_id}: Bank-B partitions overlap or are incomplete")

    for checkpoint_id in range(expected_checkpoints):
        reference = bank_recomputed.get((checkpoint_id, "A"))
        reference_numeric = bank_numeric.get((checkpoint_id, "A"))
        if reference is None or reference_numeric is None:
            continue
        reference_signed_noise = _regenerate_signed_noise(
            reference_numeric["perturbation_seeds"],
            reference_numeric["theta"].size,
        )
        endpoint_reference = {
            "signed_noise": reference_signed_noise,
            "utilities": np.asarray(reference_numeric["diagnostics"]["utilities"]),
            "gradient": reference["gradient"],
            "curvature": reference["curvature"],
            "origin_metrics": {
                name: float(reference_numeric["diagnostics"][name])
                for name in (
                    "gradient_endpoint_relative_error",
                    "subspace_jacobian_relative_error",
                    "self_normalized_gradient_relative_error",
                    "self_normalized_jacobian_relative_error",
                )
            },
        }
        for partition_index in range(dims["bank_b_partition_count"]):
            key = (checkpoint_id, partition_index)
            numeric = partition_numeric.get(key)
            record = partition_by_key.get(key)
            if numeric is None or record is None:
                continue
            signed_noise = _regenerate_signed_noise(
                numeric["perturbation_seeds"], numeric["theta"].size
            )
            result = _validate_diagnostic_data(
                numeric["diagnostics"],
                label=(
                    f"checkpoint {checkpoint_id}, partition {partition_index} diagnostics"
                ),
                manifest=manifest,
                theta=numeric["theta"],
                signed_noise=signed_noise,
                paired_returns=numeric["paired_returns"],
                basis=numeric["basis"],
                random_basis=numeric["random_basis"],
                q_summaries=record["q_summaries"],
                reference_curvature=reference["curvature"],
                reference_actions=reference["actions"],
                issues=issues,
                endpoint_reference=endpoint_reference,
            )
            if result is None:
                continue
            partition_recomputed[key] = {
                name: value
                for name, value in result.items()
                if name != "endpoint_reference"
            }
            utility_sum = float(np.sum(numeric["diagnostics"]["utilities"]))
            utility_abs = float(np.sum(np.abs(numeric["diagnostics"]["utilities"])))
            if not _close(record["lopo_utility_sum"], utility_sum, tolerance=1e-10) or not _close(
                record["lopo_utility_abs_sum"], utility_abs, tolerance=1e-10
            ):
                issues.append(
                    f"checkpoint {checkpoint_id}, partition {partition_index}: utility diagnostics disagree with arrays"
                )

    metric_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    for index, record in enumerate(artifact["checkpoint_metrics"] if isinstance(artifact["checkpoint_metrics"], list) else []):
        label = f"checkpoint_metrics[{index}]"
        if not _append_schema_issue(record, METRIC_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        key = (record["checkpoint_id"], record["q"])
        if key in metric_by_key:
            issues.append(f"{label}: duplicate checkpoint/q metric")
        metric_by_key[key] = record
        if record["checkpoint_id"] not in checkpoint_by_id or record["q"] not in q_values:
            issues.append(f"{label}: checkpoint/q identity is outside manifest")
            continue
        expected_id = metric_id_for(manifest, record["checkpoint_id"], record["q"])
        if record["metric_id"] != expected_id:
            issues.append(f"{label}: metric id mapping is invalid")
        for field in (
            "d_material",
            "e_high",
            "e_100",
            "high_sample_action_difference_norm",
            "partition_action_sq_error_mean",
        ):
            if not _finite(record[field]) or float(record[field]) < 0.0:
                issues.append(f"{label}.{field}: metric is invalid")
        q_index = q_values.index(record["q"])
        bank_a = bank_by_key.get((record["checkpoint_id"], "A"))
        bank_b = bank_by_key.get((record["checkpoint_id"], "B"))
        partitions = [
            partition_by_key.get((record["checkpoint_id"], partition_index))
            for partition_index in range(dims["bank_b_partition_count"])
        ]
        if not bank_a or not bank_b or any(partition is None for partition in partitions):
            continue
        a_summary = bank_a["q_summaries"][q_index]
        b_summary = bank_b["q_summaries"][q_index]
        epsilon = manifest["analysis"]["machine_epsilon"]
        a_norm = float(a_summary["anisotropic_action_norm"])
        b_norm = float(b_summary["anisotropic_action_norm"])
        struct_norm = float(a_summary["structured_norm"])
        expected_d = a_norm / max(struct_norm, epsilon)
        expected_high = float(record["high_sample_action_difference_norm"]) / max(
            0.5 * (a_norm + b_norm), epsilon
        )
        distances = [
            float(partition["q_summaries"][q_index]["anisotropic_minus_bank_a_norm"])
            for partition in partitions
        ]
        expected_sq = float(np.mean(np.square(distances)))
        expected_e100 = math.sqrt(expected_sq) / max(a_norm, epsilon)
        if not _close(record["d_material"], expected_d, tolerance=1e-10):
            issues.append(f"{label}: D_material is inconsistent")
        if not _close(record["e_high"], expected_high, tolerance=1e-10):
            issues.append(f"{label}: E_high is inconsistent")
        if not _close(record["partition_action_sq_error_mean"], expected_sq, tolerance=1e-10):
            issues.append(f"{label}: partition action error mean is inconsistent")
        if not _close(record["e_100"], expected_e100, tolerance=1e-10):
            issues.append(f"{label}: E_100 is inconsistent")
        expected_resolution = struct_norm > epsilon and a_norm > epsilon
        if record["material_resolved"] is not (struct_norm > epsilon):
            issues.append(f"{label}: material denominator resolution is inconsistent")
        if record["high_sample_resolved"] is not (0.5 * (a_norm + b_norm) > epsilon):
            issues.append(f"{label}: high-sample denominator resolution is inconsistent")
        if record["operational_resolved"] is not (a_norm > epsilon):
            issues.append(f"{label}: operational denominator resolution is inconsistent")
        if expected_resolution is False and record["d_material"] != expected_d:
            issues.append(f"{label}: unresolved material metric is not retained deterministically")
    expected_metric_keys = {
        (checkpoint_id, q)
        for checkpoint_id in range(expected_checkpoints)
        for q in q_values
    }
    if set(metric_by_key) != expected_metric_keys:
        issues.append("checkpoint metric identities are partial, duplicated, or incomplete")

    center_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    endpoint_batch_by_checkpoint: dict[int, dict[str, Any]] = {}
    center_transition_sum = 0
    endpoint_seed_set: set[int] = set()
    for index, record in enumerate(artifact["center_endpoints"] if isinstance(artifact["center_endpoints"], list) else []):
        label = f"center_endpoints[{index}]"
        if not _append_schema_issue(record, CENTER_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        key = (record["checkpoint_id"], record["episode_index"])
        if key in center_by_key:
            issues.append(f"{label}: duplicate center endpoint identity")
        center_by_key[key] = record
        if record["checkpoint_id"] not in checkpoint_by_id or record["episode_index"] not in range(
            dims["endpoint_episodes"]
        ):
            issues.append(f"{label}: center endpoint identity is outside manifest")
            continue
        expected_seed = endpoint_seed(
            manifest, record["checkpoint_id"], record["episode_index"]
        )
        if record["center_endpoint_id"] != center_endpoint_id_for(
            manifest, record["checkpoint_id"], record["episode_index"]
        ) or record["rollout_seed"] != expected_seed:
            issues.append(f"{label}: center endpoint mapping is invalid")
        endpoint_seed_set.add(expected_seed)
        rollout_path = _verified_artifact_path(
            artifact_root,
            record["rollout_artifact_path"],
            record["rollout_artifact_sha256"],
            f"{label}.rollout_artifact",
            issues,
            file_cache,
        )
        existing_batch = endpoint_batch_by_checkpoint.get(record["checkpoint_id"])
        if existing_batch is not None and (
            existing_batch["path"] != rollout_path
            or existing_batch["sha256"] != record["rollout_artifact_sha256"]
        ):
            issues.append(f"{label}: checkpoint endpoint artifact is not unique")
        elif existing_batch is None:
            rollout_data = _load_npz_exact(
                rollout_path,
                ENDPOINT_NPZ_KEYS,
                f"{label}.rollout_artifact",
                issues,
            )
            if rollout_data is not None:
                expected_shape = (
                    len(q_values),
                    dims["bank_b_partition_count"],
                    len(arms),
                    dims["endpoint_episodes"],
                )
                if (
                    rollout_data["center_returns"].shape
                    != (dims["endpoint_episodes"],)
                    or rollout_data["center_transitions"].shape
                    != (dims["endpoint_episodes"],)
                    or rollout_data["endpoint_returns"].shape != expected_shape
                    or rollout_data["endpoint_transitions"].shape != expected_shape
                    or rollout_data["rollout_seeds"].shape
                    != (dims["endpoint_episodes"],)
                    or not np.all(np.isfinite(rollout_data["center_returns"]))
                    or not np.all(np.isfinite(rollout_data["endpoint_returns"]))
                    or not np.issubdtype(
                        rollout_data["center_transitions"].dtype, np.integer
                    )
                    or not np.issubdtype(
                        rollout_data["endpoint_transitions"].dtype, np.integer
                    )
                    or np.any(rollout_data["center_transitions"] <= 0)
                    or np.any(rollout_data["endpoint_transitions"] <= 0)
                ):
                    issues.append(f"{label}: endpoint rollout artifact schema is invalid")
                else:
                    endpoint_batch_by_checkpoint[record["checkpoint_id"]] = {
                        "path": rollout_path,
                        "sha256": record["rollout_artifact_sha256"],
                        "data": rollout_data,
                    }
                    expected_seeds = [
                        endpoint_seed(manifest, record["checkpoint_id"], episode)
                        for episode in range(dims["endpoint_episodes"])
                    ]
                    if rollout_data["rollout_seeds"].tolist() != expected_seeds:
                        issues.append(f"{label}: endpoint rollout seed array is invalid")
        batch = endpoint_batch_by_checkpoint.get(record["checkpoint_id"])
        if batch is not None:
            data = batch["data"]
            if not _close(data["center_returns"][record["episode_index"]], record["return"]):
                issues.append(f"{label}: center return disagrees with rollout artifact")
            if int(data["center_transitions"][record["episode_index"]]) != record[
                "transitions"
            ]:
                issues.append(f"{label}: center transitions disagree with rollout artifact")
        if not _finite(record["return"]):
            issues.append(f"{label}: return is nonfinite")
        if not isinstance(record["transitions"], int) or record["transitions"] <= 0:
            issues.append(f"{label}: transition count is invalid")
        else:
            center_transition_sum += record["transitions"]
    expected_center_keys = {
        (checkpoint_id, episode_index)
        for checkpoint_id in range(expected_checkpoints)
        for episode_index in range(dims["endpoint_episodes"])
    }
    if set(center_by_key) != expected_center_keys:
        issues.append("center endpoint identities are partial, duplicated, or incomplete")
    if endpoint_seed_set.intersection(all_perturbation_seeds) or endpoint_seed_set.intersection(
        all_bank_rollout_seeds
    ):
        issues.append("endpoint seed stream overlaps a curvature-bank stream")

    endpoint_by_key: dict[tuple[int, int, float, str, int], dict[str, Any]] = {}
    endpoint_transition_sum = 0
    for index, record in enumerate(artifact["endpoints"] if isinstance(artifact["endpoints"], list) else []):
        label = f"endpoints[{index}]"
        if not _append_schema_issue(record, ENDPOINT_KEYS, label, issues):
            continue
        _validate_record_hash(record, label, issues)
        key = (
            record["checkpoint_id"],
            record["partition_index"],
            record["q"],
            record["arm"],
            record["episode_index"],
        )
        if key in endpoint_by_key:
            issues.append(f"{label}: duplicate endpoint identity")
        endpoint_by_key[key] = record
        if (
            record["checkpoint_id"] not in checkpoint_by_id
            or record["partition_index"] not in range(dims["bank_b_partition_count"])
            or record["q"] not in q_values
            or record["arm"] not in arms
            or record["episode_index"] not in range(dims["endpoint_episodes"])
        ):
            issues.append(f"{label}: endpoint identity is outside manifest")
            continue
        expected_id = endpoint_id_for(
            manifest,
            record["checkpoint_id"],
            record["q"],
            record["partition_index"],
            record["arm"],
            record["episode_index"],
        )
        expected_seed = endpoint_seed(
            manifest, record["checkpoint_id"], record["episode_index"]
        )
        partition = partition_by_key.get(
            (record["checkpoint_id"], record["partition_index"])
        )
        expected_action = None
        if partition:
            expected_action = partition["q_summaries"][q_values.index(record["q"])][
                "action_sha256"
            ][record["arm"]]
        if record["endpoint_id"] != expected_id or record["rollout_seed"] != expected_seed:
            issues.append(f"{label}: endpoint id/seed mapping is invalid")
        if record["action_sha256"] != expected_action:
            issues.append(f"{label}: endpoint action digest does not match its partition estimate")
        rollout_path = _verified_artifact_path(
            artifact_root,
            record["rollout_artifact_path"],
            record["rollout_artifact_sha256"],
            f"{label}.rollout_artifact",
            issues,
            file_cache,
        )
        batch = endpoint_batch_by_checkpoint.get(record["checkpoint_id"])
        if batch is None or batch["path"] != rollout_path or batch["sha256"] != record[
            "rollout_artifact_sha256"
        ]:
            issues.append(f"{label}: endpoint does not reference the locked checkpoint rollout artifact")
        else:
            data = batch["data"]
            coordinates = (
                q_values.index(record["q"]),
                record["partition_index"],
                arms.index(record["arm"]),
                record["episode_index"],
            )
            if not _close(data["endpoint_returns"][coordinates], record["return"]):
                issues.append(f"{label}: endpoint return disagrees with rollout artifact")
            if int(data["endpoint_transitions"][coordinates]) != record["transitions"]:
                issues.append(f"{label}: endpoint transitions disagree with rollout artifact")
        if not _finite(record["return"]):
            issues.append(f"{label}: return is nonfinite")
        if not isinstance(record["transitions"], int) or record["transitions"] <= 0:
            issues.append(f"{label}: transition count is invalid")
        else:
            endpoint_transition_sum += record["transitions"]
    expected_endpoint_keys = {
        (checkpoint_id, partition_index, q, arm, episode_index)
        for checkpoint_id in range(expected_checkpoints)
        for partition_index in range(dims["bank_b_partition_count"])
        for q in q_values
        for arm in arms
        for episode_index in range(dims["endpoint_episodes"])
    }
    if set(endpoint_by_key) != expected_endpoint_keys:
        issues.append("endpoint identities are partial, duplicated, or incomplete")

    budget = artifact["budget"]
    if not isinstance(budget, dict) or set(budget) != BUDGET_KEYS:
        issues.append("artifact budget schema is not exact")
    else:
        rollout_expected = dict(manifest["budget"])
        rollout_expected.pop("environment_transitions_are_separate")
        for key, expected in rollout_expected.items():
            if budget[key] != expected:
                issues.append(f"budget.{key}: is {budget[key]!r}, expected {expected!r}")
        transition_expected = {
            "checkpoint_training_transitions": training_transition_sum,
            "normalization_calibration_transitions": calibration_transition_sum,
            "bank_transitions": bank_transition_sum,
            "endpoint_arm_transitions": endpoint_transition_sum,
            "checkpoint_center_transitions": center_transition_sum,
        }
        transition_expected["total_environment_transitions"] = sum(
            transition_expected.values()
        )
        for key, expected in transition_expected.items():
            if budget[key] != expected:
                issues.append(f"budget.{key}: does not equal validated record totals")

    if issues:
        raise SubspaceValidationError(issues)
    return {
        "training_by_id": training_by_id,
        "checkpoint_by_id": checkpoint_by_id,
        "bank_by_key": bank_by_key,
        "partition_by_key": partition_by_key,
        "metric_by_key": metric_by_key,
        "center_by_key": center_by_key,
        "endpoint_by_key": endpoint_by_key,
        "partition_recomputed": partition_recomputed,
        "bank_recomputed": bank_recomputed,
    }


def _binomial_upper_tail(successes: int, trials: int) -> float:
    return float(
        sum(math.comb(trials, value) for value in range(successes, trials + 1))
        / (2**trials)
    )


def _holm_adjust(raw: Sequence[float]) -> list[float]:
    order = sorted(range(len(raw)), key=lambda index: raw[index])
    adjusted = [1.0] * len(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(raw[index])))
        adjusted[index] = running
    return adjusted


def _interquartile_mean(values: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if ordered.size == 0:
        raise ValueError("interquartile mean requires observations")
    lower = int(math.floor(0.25 * ordered.size))
    upper = int(math.ceil(0.75 * ordered.size))
    return float(np.mean(ordered[lower:upper]))


def _simultaneous_bounds(
    seed_statistics: Mapping[tuple[int, str], np.ndarray],
    manifest: Mapping[str, Any],
) -> dict[tuple[int, str], dict[str, float | bool | None]]:
    analysis = manifest["analysis"]
    keys = [
        (task["task_index"], metric)
        for task in manifest["tasks"]
        for metric in ("L", "D", "H", "E")
    ]
    result: dict[tuple[int, str], dict[str, float | bool | None]] = {}
    expected_count = analysis["mechanism_seed_count"]
    for key in keys:
        values = np.asarray(seed_statistics[key], dtype=np.float64)
        if values.shape != (expected_count,) or not np.all(np.isfinite(values)):
            result[key] = {
                "estimate": None,
                "one_sided_bound": None,
                "resolved": False,
                "order_index_zero_based": None,
            }
            continue
        ordered = np.sort(values, kind="stable")
        index = (
            analysis["lower_order_index_zero_based"]
            if key[1] == "D"
            else analysis["upper_order_index_zero_based"]
        )
        result[key] = {
            "estimate": float(np.median(values)),
            "one_sided_bound": float(ordered[index]),
            "resolved": True,
            "order_index_zero_based": int(index),
        }
    return result


def analyze_validated(
    validated: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply only the preregistered seed reductions, bounds, and sign tests."""

    dims = manifest["dimensions"]
    primary_q = manifest["analysis"]["primary_q"]
    q_index = dims["locality_q"].index(primary_q)
    checkpoint_by_id = validated["checkpoint_by_id"]
    partition_by_key = validated["partition_by_key"]
    metric_by_key = validated["metric_by_key"]
    endpoint_by_key = validated["endpoint_by_key"]
    seed_statistics: dict[tuple[int, str], list[float]] = defaultdict(list)
    endpoint_seed_differences: dict[tuple[int, int], float] = {}
    random_control_valid_by_task: dict[int, bool] = {
        task["task_index"]: True for task in manifest["tasks"]
    }
    resolved_by_task: dict[int, bool] = {
        task["task_index"]: True for task in manifest["tasks"]
    }
    alpha_resolved_by_task: dict[int, bool] = {
        task["task_index"]: True for task in manifest["tasks"]
    }
    for task in manifest["tasks"]:
        task_index = task["task_index"]
        for seed in manifest["training_seeds"]:
            checkpoint_ids = [
                checkpoint_id_for(manifest, task_index, seed, generation)
                for generation in manifest["checkpoint_generations"]
            ]
            locality = []
            d_values = []
            h_values = []
            e_values = []
            endpoint_differences = []
            for checkpoint_id in checkpoint_ids:
                metric = metric_by_key[(checkpoint_id, primary_q)]
                d_values.append(float(metric["d_material"]))
                h_values.append(float(metric["e_high"]))
                e_values.append(float(metric["e_100"]))
                if not (
                    metric["material_resolved"]
                    and metric["high_sample_resolved"]
                    and metric["operational_resolved"]
                ):
                    resolved_by_task[task_index] = False
                for bank in ("A", "B"):
                    diagnostic = validated["bank_recomputed"].get(
                        (checkpoint_id, bank)
                    )
                    if diagnostic is None or not diagnostic[
                        "primary_diagnostics_resolved"
                    ]:
                        resolved_by_task[task_index] = False
                for partition_index in range(dims["bank_b_partition_count"]):
                    summary = partition_by_key[(checkpoint_id, partition_index)][
                        "q_summaries"
                    ][q_index]
                    locality.append(float(summary["structured_step_over_sigma"]))
                    if not summary["alpha_resolved"]:
                        alpha_resolved_by_task[task_index] = False
                    if not summary["random_control_valid"]:
                        random_control_valid_by_task[task_index] = False
                    diagnostic = validated["partition_recomputed"].get(
                        (checkpoint_id, partition_index)
                    )
                    if diagnostic is None or not diagnostic[
                        "primary_diagnostics_resolved"
                    ]:
                        resolved_by_task[task_index] = False
                    for episode_index in range(dims["endpoint_episodes"]):
                        structured = endpoint_by_key[
                            (
                                checkpoint_id,
                                partition_index,
                                primary_q,
                                "structured",
                                episode_index,
                            )
                        ]["return"]
                        isotropic = endpoint_by_key[
                            (
                                checkpoint_id,
                                partition_index,
                                primary_q,
                                "isotropic",
                                episode_index,
                            )
                        ]["return"]
                        endpoint_differences.append(float(structured) - float(isotropic))
            seed_statistics[(task_index, "L")].append(
                float(
                    np.quantile(
                        locality,
                        0.95,
                        method=manifest["analysis"]["quantile_method"],
                    )
                )
            )
            seed_statistics[(task_index, "D")].append(float(np.median(d_values)))
            seed_statistics[(task_index, "H")].append(float(np.median(h_values)))
            seed_statistics[(task_index, "E")].append(float(np.median(e_values)))
            endpoint_seed_differences[(task_index, seed)] = float(
                np.mean(endpoint_differences)
            )
    bounds = _simultaneous_bounds(
        {key: np.asarray(value) for key, value in seed_statistics.items()},
        manifest,
    )
    raw_p = []
    endpoint_descriptive = []
    for task in manifest["tasks"]:
        task_index = task["task_index"]
        values = [
            endpoint_seed_differences[(task_index, seed)]
            for seed in manifest["training_seeds"]
        ]
        successes = sum(value > 0.0 for value in values)
        ties = sum(value == 0.0 for value in values)
        raw = _binomial_upper_tail(successes, len(values))
        raw_p.append(raw)
        endpoint_descriptive.append(
            {
                "task_index": task_index,
                "env_name": task["env_name"],
                "seed_mean_contrast": float(np.mean(values)),
                "strict_positive_seed_count": successes,
                "strict_tie_seed_count": ties,
                "seed_count": len(values),
                "seed_level_probability_of_improvement": float(
                    (successes + 0.5 * ties) / len(values)
                ),
                "raw_one_sided_sign_p": raw,
            }
        )
    adjusted = _holm_adjust(raw_p)
    thresholds = manifest["analysis"]["gate_thresholds"]
    task_results = []
    for task, endpoint, adjusted_p in zip(
        manifest["tasks"], endpoint_descriptive, adjusted
    ):
        task_index = task["task_index"]
        locality = bounds[(task_index, "L")]
        material = bounds[(task_index, "D")]
        high = bounds[(task_index, "H")]
        operational = bounds[(task_index, "E")]
        calibration_resolved = alpha_resolved_by_task[task_index]
        conditions = {
            "locality": bool(
                calibration_resolved
                and locality["resolved"]
                and locality["one_sided_bound"] <= thresholds["locality_upper"]
            ),
            "material_action": bool(
                calibration_resolved
                and material["resolved"]
                and material["one_sided_bound"] > thresholds["material_lower"]
            ),
            "high_sample_replication": bool(
                calibration_resolved
                and high["resolved"]
                and high["one_sided_bound"] < thresholds["high_sample_upper"]
            ),
            "operational_reliability": bool(
                calibration_resolved
                and operational["resolved"]
                and operational["one_sided_bound"] < thresholds["operational_upper"]
            ),
            "directional_endpoint": bool(
                calibration_resolved
                and endpoint["seed_mean_contrast"] > 0.0
                and adjusted_p < thresholds["endpoint_adjusted_one_sided_p_upper"]
            ),
            "alpha_calibration_resolved": calibration_resolved,
            "required_diagnostics_resolved": bool(
                calibration_resolved and resolved_by_task[task_index]
            ),
            "random_control_valid": bool(
                calibration_resolved and random_control_valid_by_task[task_index]
            ),
        }
        task_results.append(
            {
                **endpoint,
                "holm_adjusted_one_sided_sign_p": adjusted_p,
                "seed_statistics": {
                    metric: seed_statistics[(task_index, metric)]
                    for metric in ("L", "D", "H", "E")
                },
                "simultaneous_bounds": {
                    metric: bounds[(task_index, metric)]
                    for metric in ("L", "D", "H", "E")
                },
                "gate_conditions": conditions,
                "task_pass": all(conditions.values()),
            }
        )
    passing = sum(result["task_pass"] for result in task_results)
    descriptive_locality = []
    norm_fields = {
        "structured": "structured_norm",
        "isotropic": "isotropic_norm",
        "explicit": "explicit_norm",
        "random": "random_norm",
    }
    for task in manifest["tasks"]:
        task_index = task["task_index"]
        for q_index, q in enumerate(dims["locality_q"]):
            for arm, field in norm_fields.items():
                ratios = []
                for seed in manifest["training_seeds"]:
                    for generation in manifest["checkpoint_generations"]:
                        checkpoint_id = checkpoint_id_for(
                            manifest, task_index, seed, generation
                        )
                        for partition_index in range(
                            dims["bank_b_partition_count"]
                        ):
                            summary = partition_by_key[
                                (checkpoint_id, partition_index)
                            ]["q_summaries"][q_index]
                            ratios.append(float(summary[field]) / dims["noise_std"])
                values = np.asarray(ratios, dtype=np.float64)
                descriptive_locality.append(
                    {
                        "task_index": task_index,
                        "env_name": task["env_name"],
                        "q": q,
                        "arm": arm,
                        "repeated_measure_count": int(values.size),
                        "first_step_over_sigma": float(values[0]),
                        "mean_step_over_sigma": float(np.mean(values)),
                        "median_step_over_sigma": float(np.median(values)),
                        "percentile_95_step_over_sigma": float(
                            np.quantile(
                                values,
                                0.95,
                                method=manifest["analysis"]["quantile_method"],
                            )
                        ),
                        "maximum_step_over_sigma": float(np.max(values)),
                        "fraction_at_or_below_0_25": float(
                            np.mean(values <= 0.25)
                        ),
                        "fraction_at_or_below_0_5": float(np.mean(values <= 0.5)),
                        "fraction_at_or_below_1_0": float(np.mean(values <= 1.0)),
                        "inference": "descriptive_repeated_measures_only",
                    }
                )

    descriptive_return_contrasts = []
    for task in manifest["tasks"]:
        task_index = task["task_index"]
        for q in dims["locality_q"]:
            for control in ("isotropic", "explicit", "random"):
                seed_blocks: dict[int, list[float]] = {
                    seed: [] for seed in manifest["training_seeds"]
                }
                for seed in manifest["training_seeds"]:
                    for generation in manifest["checkpoint_generations"]:
                        checkpoint_id = checkpoint_id_for(
                            manifest, task_index, seed, generation
                        )
                        for partition_index in range(
                            dims["bank_b_partition_count"]
                        ):
                            for episode_index in range(dims["endpoint_episodes"]):
                                structured = endpoint_by_key[
                                    (
                                        checkpoint_id,
                                        partition_index,
                                        q,
                                        "structured",
                                        episode_index,
                                    )
                                ]["return"]
                                comparison = endpoint_by_key[
                                    (
                                        checkpoint_id,
                                        partition_index,
                                        q,
                                        control,
                                        episode_index,
                                    )
                                ]["return"]
                                seed_blocks[seed].append(
                                    float(structured) - float(comparison)
                                )
                flattened = np.asarray(
                    [
                        value
                        for seed in manifest["training_seeds"]
                        for value in seed_blocks[seed]
                    ],
                    dtype=np.float64,
                )
                seed_means = np.asarray(
                    [
                        np.mean(seed_blocks[seed])
                        for seed in manifest["training_seeds"]
                    ],
                    dtype=np.float64,
                )
                bootstrap_seed = derive_seed(
                    manifest,
                    "cluster_bootstrap",
                    "descriptive_return",
                    task_index,
                    q,
                    control,
                )
                rng = np.random.default_rng(bootstrap_seed)
                indices = rng.integers(
                    0,
                    len(seed_means),
                    size=(
                        manifest["analysis"]["bootstrap_resamples"],
                        len(seed_means),
                    ),
                )
                bootstrap_means = np.mean(seed_means[indices], axis=1)
                interval = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                    method=manifest["analysis"]["quantile_method"],
                )
                descriptive_return_contrasts.append(
                    {
                        "task_index": task_index,
                        "env_name": task["env_name"],
                        "q": q,
                        "contrast": f"structured_minus_{control}",
                        "paired_difference_count": int(flattened.size),
                        "training_seed_cluster_count": len(seed_means),
                        "paired_mean": float(np.mean(flattened)),
                        "paired_median": float(np.median(flattened)),
                        "paired_interquartile_mean": _interquartile_mean(
                            flattened
                        ),
                        "paired_checkpoint_partition_episode_probability_of_improvement": float(
                            np.mean(flattened > 0.0)
                            + 0.5 * np.mean(flattened == 0.0)
                        ),
                        "seed_cluster_bootstrap_mean_interval_95": [
                            float(interval[0]),
                            float(interval[1]),
                        ],
                        "bootstrap_seed": bootstrap_seed,
                        "multiplicity_role": (
                            "primary_holm_family"
                            if q == primary_q and control == "isotropic"
                            else (
                                "secondary_no_p_value_reported"
                                if q == primary_q
                                else "descriptive_sensitivity_no_p_value"
                            )
                        ),
                    }
                )
    return {
        "schema_version": 1,
        "study": STUDY,
        "analysis_designation": "preregistered_frozen_checkpoint_mechanism_diagnostic",
        "primary_q": primary_q,
        "mechanism_bound_method": manifest["analysis"]["mechanism_bound_method"],
        "mechanism_familywise_error_upper_bound": 1.0
        - manifest["analysis"]["simultaneous_coverage_lower_bound"],
        "endpoint_family_alpha": manifest["analysis"]["endpoint_family_alpha"],
        "combined_false_advance_upper_bound": manifest["analysis"]
        ["combined_false_advance_upper_bound"],
        "top_level_unit": "training_seed",
        "task_results": task_results,
        "descriptive_locality": descriptive_locality,
        "descriptive_return_contrasts": descriptive_return_contrasts,
        "passing_task_count": passing,
        "required_passing_task_count": manifest["analysis"]["minimum_passing_tasks"],
        "mechanism_advances_to_optimizer_pilot": passing
        >= manifest["analysis"]["minimum_passing_tasks"],
        "claim_boundary": manifest["claim_boundary"],
    }


def _atomic_json(path: str, value: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(prefix=".subspace_analysis_", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    except BaseException:
        try:
            os.unlink(staged)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument(
        "--artifact-root",
        help="locked root for every relative artifact path (defaults to index directory)",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest, manifest_sha = load_and_validate_manifest(
        args.manifest, expected_sha256=args.expected_manifest_sha256
    )
    expected_hashes = {
        "source_sha256": args.expected_source_sha256,
        "manifest_sha256": manifest_sha,
        "protocol_sha256": args.expected_protocol_sha256,
        "analyzer_sha256": args.expected_analyzer_sha256,
        "launcher_sha256": args.expected_launcher_sha256,
        "dependency_lock_sha256": args.expected_dependency_lock_sha256,
    }
    artifact = _read_json(args.artifact)
    validated = validate_artifact(
        artifact,
        manifest,
        expected_hashes=expected_hashes,
        artifact_root=(
            args.artifact_root
            if args.artifact_root is not None
            else os.path.dirname(os.path.abspath(args.artifact))
        ),
        require_preregistered_manifest=True,
    )
    result = analyze_validated(validated, manifest)
    _atomic_json(args.output, result)
    print(
        f"Validated complete diagnostic and wrote {args.output}; "
        f"passing_tasks={result['passing_task_count']}/3"
    )


if __name__ == "__main__":
    main()

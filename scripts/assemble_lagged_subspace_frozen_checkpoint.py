#!/usr/bin/env python3
"""Assemble the complete frozen-checkpoint diagnostic audit index.

This is deliberately a structural and provenance pass.  It accepts every
predeclared training run and checkpoint fragment or rejects the study; it never
computes scientific outcomes, chooses checkpoints, or drops records.  The
separate analyzer independently reconstructs and validates all numerical
quantities after this index has been committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import analyze_lagged_subspace_frozen_checkpoint as analyzer
from experiments.lagged_subspace_study_lock import (
    DEPENDENCY_BUNDLE_PATH,
    LAUNCHER_BUNDLE_PATH,
    StudySourceLockError,
    compute_lagged_subspace_study_sha256,
    validate_hash_bundle,
)


STUDY = "lagged_subspace_frozen_checkpoint"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
FRAGMENT_KEYS = {
    "schema_version",
    "study",
    "status",
    "manifest_sha256",
    "checkpoint_id",
    "task_index",
    "training_seed",
    "generation",
    "checkpoint",
    "banks",
    "partitions",
    "checkpoint_metrics",
    "center_endpoints",
    "endpoints",
    "transition_totals",
    "lineage_artifacts",
    "artifact_inventory",
    "no_outcome_selection",
    "no_record_exclusion",
    "fragment_sha256",
}
TRANSITION_TOTAL_KEYS = {"bank", "center", "endpoint"}
LINEAGE_KEYS = {"capture_manifest_path", "capture_manifest_sha256"}
INVENTORY_KEYS = {"path", "sha256"}
FORBIDDEN_TRAINING_ARTIFACTS = {
    "best_params.npy",
    "best_obs_norm.npz",
    "heldout_evaluation.json",
    "summary.json",
    "hessian_ema.npy",
    "snes_search_std.npy",
}
TRAINING_FILES = {
    "checkpoint_capture.json",
    "checkpoint_training_config.json",
    "config.json",
    "final_params.npy",
    "history.json",
    "history.jsonl",
    "obs_norm.npz",
    "status.json",
}


class AssemblyError(RuntimeError):
    """A fail-closed assembly contract violation."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _stamp(record: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if "record_sha256" in result:
        raise AssemblyError("cannot stamp a record that already has record_sha256")
    result["record_sha256"] = _record_sha256(result)
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _strict_json_loads(payload: str) -> Any:
    def object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token!r}")

    return json.loads(
        payload,
        object_pairs_hook=object_without_duplicates,
        parse_constant=reject_nonfinite,
    )


def _read_json(path: str, label: str) -> Any:

    try:
        with open(path, encoding="utf-8") as stream:
            return _strict_json_loads(stream.read())
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise AssemblyError(f"{label}: cannot read strict JSON: {error}") from error


def _exact_keys(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else type(value).__name__
        raise AssemblyError(f"{label}: schema is not exact; found {actual!r}")
    return value


def _validate_record(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    record = dict(_exact_keys(value, keys, label))
    if not _is_sha256(record["record_sha256"]):
        raise AssemblyError(f"{label}: record digest is invalid")
    actual = _record_sha256(record)
    if actual != record["record_sha256"]:
        raise AssemblyError(f"{label}: record digest mismatch")
    return record


def _normalized_relative(path: Any, label: str) -> str:
    if (
        not isinstance(path, str)
        or not path
        or os.path.isabs(path)
        or os.path.normpath(path) != path
        or path == ".."
        or path.startswith("../")
        or "\\" in path
    ):
        raise AssemblyError(f"{label}: path is not normalized and root-relative")
    return path


def _inside_root(root: str, relative: Any, label: str) -> str:
    relative = _normalized_relative(relative, label)
    root_real = os.path.realpath(root)
    candidate = os.path.join(root_real, relative)
    candidate_real = os.path.realpath(candidate)
    if os.path.commonpath([root_real, candidate_real]) != root_real:
        raise AssemblyError(f"{label}: path escapes the artifact root")
    return candidate


def _regular_file(root: str, relative: Any, label: str) -> str:
    path = _inside_root(root, relative, label)
    if os.path.islink(path) or not os.path.isfile(path):
        raise AssemblyError(f"{label}: expected a non-symlink regular file")
    return path


def _directory(root: str, relative: Any, label: str) -> str:
    path = _inside_root(root, relative, label)
    if os.path.islink(path) or not os.path.isdir(path):
        raise AssemblyError(f"{label}: expected a non-symlink directory")
    return path


def _verify_file(
    root: str, relative: Any, expected_sha256: Any, label: str
) -> str:
    if not _is_sha256(expected_sha256):
        raise AssemblyError(f"{label}: expected digest is invalid")
    path = _regular_file(root, relative, label)
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise AssemblyError(
            f"{label}: digest mismatch; expected {expected_sha256}, found {actual}"
        )
    return path


def _reject_symlinks(root: str, label: str) -> None:
    for directory, names, files in os.walk(root, followlinks=False):
        for name in [*names, *files]:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                raise AssemblyError(f"{label}: symlink is forbidden: {path}")


def _reject_inference(value: Any, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in analyzer.INFERENCE_LEAKAGE_KEYS:
                raise AssemblyError(f"{path}.{key}: inferential field is forbidden")
            _reject_inference(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_inference(item, f"{path}[{index}]")


def _atomic_json(path: str, value: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, staged = tempfile.mkstemp(prefix=".subspace_assembly_", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(staged)
        except FileNotFoundError:
            pass
        raise


def _training_id(manifest: Mapping[str, Any], task_index: int, seed: int) -> int:
    return analyzer.training_id_for(manifest, task_index, seed)


def _checkpoint_id(
    manifest: Mapping[str, Any], task_index: int, seed: int, generation: int
) -> int:
    return analyzer.checkpoint_id_for(manifest, task_index, seed, generation)


def _validate_stderr_inventory(
    artifact_root: str,
    relative_root: str,
    *,
    prefix: str,
    count: int,
) -> None:
    root = _directory(artifact_root, relative_root, f"{prefix} stderr root")
    expected = {f"{prefix}_{index:06d}.stderr" for index in range(count)}
    actual = set(os.listdir(root))
    if actual != expected:
        raise AssemblyError(
            f"{prefix} stderr inventory is partial or has extras: "
            f"expected {len(expected)}, found {len(actual)}"
        )
    for filename in sorted(expected):
        relative = os.path.join(relative_root, filename)
        path = _regular_file(artifact_root, relative, f"stderr {filename}")
        if os.path.getsize(path) != 0 or _sha256_file(path) != EMPTY_SHA256:
            raise AssemblyError(f"stderr {filename}: file is not empty")


def _validate_training_config(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    task_index: int,
    seed: int,
    expected_hashes: Mapping[str, str],
    label: str,
) -> None:
    dims = manifest["dimensions"]
    training = manifest["training"]
    exact = {
        "checkpoint_capture_protocol": "lagged_subspace_frozen_checkpoint_v1",
        "env_name": manifest["tasks"][task_index]["env_name"],
        "seed": seed,
        "population_size": dims["population_size"],
        "learning_rate": training["learning_rate"],
        "lr_schedule": training["learning_rate_schedule"],
        "noise_std": dims["noise_std"],
        "l2_coeff": 0.0,
        "rank_fitness": True,
        "antithetic": True,
        "max_grad_norm": 0.0,
        "max_param_norm": None,
        "n_iterations": dims["training_updates"],
        "online_evaluation_enabled": False,
        "eval_episodes": 0,
        "use_obs_norm": True,
        "obs_norm_mode": "frozen_after_calibration",
        "obs_norm_calibration_episodes": dims["calibration_episodes"],
        "heldout_evaluation_enabled": False,
        "checkpoint_capture_generations": manifest["checkpoint_generations"],
        "checkpoint_gradient_archive_length": dims["lagged_gradient_count"],
        "replay_enabled": False,
        "buffer_size": 0,
        "reuse_fraction": 0.0,
        "common_rollout_seed": True,
        "evaluate_center_fitness": False,
        "condition": "standard_es",
        "algorithm": "standard_es",
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise AssemblyError(f"{label}.{key}: expected {expected!r}")
    for forbidden in (
        "trust_radius",
        "importance_sampling",
        "picard_iteration",
        "curvature_ema",
    ):
        if forbidden in config:
            raise AssemblyError(f"{label}.{forbidden}: forbidden control is present")
    provenance = config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AssemblyError(f"{label}.provenance: missing")
    provenance_expectations = {
        "source_sha256": expected_hashes["source_sha256"],
        "expected_source_sha256": expected_hashes["source_sha256"],
        "expected_manifest_sha256": expected_hashes["manifest_sha256"],
        "expected_protocol_sha256": expected_hashes["protocol_sha256"],
        "expected_analyzer_sha256": expected_hashes["analyzer_sha256"],
        "expected_launcher_sha256": expected_hashes["launcher_sha256"],
        "expected_dependency_lock_sha256": expected_hashes[
            "dependency_lock_sha256"
        ],
    }
    for key, expected in provenance_expectations.items():
        if provenance.get(key) != expected:
            raise AssemblyError(f"{label}.provenance.{key}: lock mismatch")


def _read_history(run_dir: str, updates: int, population: int) -> tuple[list[Any], int]:
    history_path = os.path.join(run_dir, "history.json")
    history = _read_json(history_path, "training history")
    if not isinstance(history, list) or len(history) != updates:
        raise AssemblyError("training history is incomplete")
    jsonl_path = os.path.join(run_dir, "history.jsonl")
    rows: list[Any] = []
    try:
        with open(jsonl_path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise AssemblyError(
                        f"history.jsonl:{line_number}: blank records are forbidden"
                    )
                rows.append(_strict_json_loads(line))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssemblyError(f"training history JSONL is invalid: {error}") from error
    if rows != history:
        raise AssemblyError("history.json and history.jsonl are not exactly identical")
    cumulative = 0
    for index, row in enumerate(history):
        if not isinstance(row, Mapping) or row.get("iteration") != index:
            raise AssemblyError(f"history[{index}]: iteration mapping is invalid")
        exact = {
            "n_fresh": population,
            "n_reused": 0,
            "used_replay": False,
            "eval_reward": None,
            "best_reward": None,
            "eval_env_steps": 0,
            "eval_env_steps_iter": 0,
            "initial_eval_reward": None,
            "initial_eval_env_steps": 0,
        }
        for key, expected in exact.items():
            if row.get(key) != expected:
                raise AssemblyError(f"history[{index}].{key}: training protocol drift")
        step_count = row.get("training_env_steps_iter")
        if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count <= 0:
            raise AssemblyError(f"history[{index}]: transition increment is invalid")
        cumulative += step_count
        if row.get("training_env_steps") != cumulative:
            raise AssemblyError(f"history[{index}]: transition cumulative sum is invalid")
    return history, cumulative


def _training_record(
    *,
    artifact_root: str,
    relative_run_dir: str,
    manifest: Mapping[str, Any],
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = _directory(artifact_root, relative_run_dir, "training run")
    _reject_symlinks(run_dir, "training run")
    entries = set(os.listdir(run_dir))
    if entries & FORBIDDEN_TRAINING_ARTIFACTS:
        raise AssemblyError(
            f"{relative_run_dir}: best/evaluation artifact is forbidden"
        )
    if entries != TRAINING_FILES | {"checkpoints"}:
        raise AssemblyError(
            f"{relative_run_dir}: training artifact inventory is not exact"
        )
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    if os.path.islink(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
        raise AssemblyError(f"{relative_run_dir}: checkpoint directory is invalid")

    config = _read_json(os.path.join(run_dir, "config.json"), "training config")
    if not isinstance(config, Mapping):
        raise AssemblyError("training config is not an object")
    env_names = [task["env_name"] for task in manifest["tasks"]]
    if config.get("env_name") not in env_names or config.get("seed") not in manifest[
        "training_seeds"
    ]:
        raise AssemblyError(f"{relative_run_dir}: task/seed is outside the manifest")
    task_index = env_names.index(config["env_name"])
    seed = config["seed"]
    training_id = _training_id(manifest, task_index, seed)
    expected_name = f"training_{training_id:06d}"
    if os.path.basename(relative_run_dir) != expected_name:
        raise AssemblyError(
            f"{relative_run_dir}: directory identity must be {expected_name}"
        )
    _validate_training_config(
        config, manifest, task_index, seed, expected_hashes, "training config"
    )

    dims = manifest["dimensions"]
    status = _read_json(os.path.join(run_dir, "status.json"), "training status")
    if (
        not isinstance(status, Mapping)
        or status.get("status") != "complete"
        or status.get("expected_iterations") != dims["training_updates"]
        or status.get("completed_iterations") != dims["training_updates"]
        or status.get("best_reward") is not None
        or status.get("initial_eval_reward") is not None
    ):
        raise AssemblyError(f"{relative_run_dir}: training status is incomplete")
    calibration = status.get("normalization_calibration_env_steps")
    if isinstance(calibration, bool) or not isinstance(calibration, int) or calibration <= 0:
        raise AssemblyError(f"{relative_run_dir}: calibration transitions are invalid")
    history, training_transitions = _read_history(
        run_dir, dims["training_updates"], dims["population_size"]
    )
    if any(row.get("normalization_calibration_env_steps") != calibration for row in history):
        raise AssemblyError(f"{relative_run_dir}: calibration accounting drifted")

    capture = _read_json(
        os.path.join(run_dir, "checkpoint_capture.json"), "checkpoint capture"
    )
    if not isinstance(capture, Mapping):
        raise AssemblyError("checkpoint capture is not an object")
    exact_capture = {
        "status": "complete",
        "requested_generations": manifest["checkpoint_generations"],
        "captured_generations": manifest["checkpoint_generations"],
        "expected_checkpoint_count": len(manifest["checkpoint_generations"]),
        "checkpoint_count": len(manifest["checkpoint_generations"]),
        "gradient_archive_length": dims["lagged_gradient_count"],
        "selection_policy": "fixed_config_generations_only",
        "reward_selection_used": False,
        "current_generation_gradient_excluded": True,
        "online_evaluation_enabled": False,
        "source_sha256": expected_hashes["source_sha256"],
    }
    for key, expected in exact_capture.items():
        if capture.get(key) != expected:
            raise AssemblyError(f"{relative_run_dir}.capture.{key}: lineage mismatch")
    controls = capture.get("validated_generator_controls")
    if not isinstance(controls, Mapping) or controls.get("plain_standard_es") is not True:
        raise AssemblyError(f"{relative_run_dir}: generator controls are missing")
    if any(
        controls.get(key) is not False
        for key in (
            "replay",
            "importance_sampling",
            "trust_region",
            "picard_iteration",
            "gradient_clipping",
            "parameter_projection",
            "curvature",
            "curvature_clipping",
            "l2",
            "checkpoint_selection_by_reward",
        )
    ):
        raise AssemblyError(f"{relative_run_dir}: a prohibited generator control is active")
    if controls.get("rank_fitness") is not True or controls.get("antithetic") is not True:
        raise AssemblyError(f"{relative_run_dir}: rank/antithetic controls are invalid")

    training_config_relative = os.path.join(
        relative_run_dir, "checkpoint_training_config.json"
    )
    training_config_path = os.path.join(run_dir, "checkpoint_training_config.json")
    training_config_sha = _sha256_file(training_config_path)
    if capture.get("training_config_sha256") != training_config_sha:
        raise AssemblyError(f"{relative_run_dir}: checkpoint config digest mismatch")
    checkpoint_config = _read_json(
        training_config_path, "checkpoint training config"
    )
    if not isinstance(checkpoint_config, Mapping):
        raise AssemblyError(f"{relative_run_dir}: checkpoint config is not an object")
    excluded_config_keys = {
        "_config_path",
        "provenance",
        "resolved_checkpoint_capture",
        "training_budget",
    }
    expected_checkpoint_config = {
        key: config[key]
        for key in sorted(config)
        if key not in excluded_config_keys
    }
    if checkpoint_config != expected_checkpoint_config:
        raise AssemblyError(
            f"{relative_run_dir}: checkpoint config is not the locked runtime payload"
        )
    artifacts = capture.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(
        manifest["checkpoint_generations"]
    ):
        raise AssemblyError(f"{relative_run_dir}: checkpoint capture is partial")
    capture_by_generation: dict[int, Mapping[str, Any]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise AssemblyError(f"{relative_run_dir}: invalid checkpoint metadata")
        generation = item.get("checkpoint_generation")
        if generation in capture_by_generation:
            raise AssemblyError(f"{relative_run_dir}: duplicate checkpoint generation")
        capture_by_generation[generation] = item
        expected_path = os.path.join(
            "checkpoints", f"checkpoint_generation_{generation:06d}.npz"
        )
        if item.get("artifact") != expected_path:
            raise AssemblyError(f"{relative_run_dir}: checkpoint path is not canonical")
        full_relative = os.path.join(relative_run_dir, expected_path)
        checkpoint_path = _verify_file(
            artifact_root,
            full_relative,
            item.get("artifact_sha256"),
            f"{relative_run_dir} checkpoint {generation}",
        )
        try:
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                if set(archive.files) != analyzer.CHECKPOINT_NPZ_KEYS:
                    raise AssemblyError(
                        f"{relative_run_dir}: checkpoint NPZ schema is not exact"
                    )
                schema_version = np.asarray(archive["schema_version"])
                embedded_generation = np.asarray(archive["checkpoint_generation"])
                embedded_source = np.asarray(archive["study_source_sha256"])
                embedded_config = np.asarray(archive["training_config_sha256"])
        except (OSError, ValueError, TypeError) as error:
            raise AssemblyError(
                f"{relative_run_dir}: cannot inspect checkpoint lineage: {error}"
            ) from error
        if (
            schema_version.shape != ()
            or int(schema_version) != 2
            or embedded_generation.shape != ()
            or int(embedded_generation) != generation
            or embedded_source.shape != ()
            or embedded_source.dtype != np.dtype("S64")
            or bytes(embedded_source).decode("ascii")
            != expected_hashes["source_sha256"]
            or embedded_config.shape != ()
            or embedded_config.dtype != np.dtype("S64")
            or bytes(embedded_config).decode("ascii") != training_config_sha
        ):
            raise AssemblyError(
                f"{relative_run_dir}: embedded checkpoint lineage is invalid"
            )
        if (
            item.get("source_sha256") != expected_hashes["source_sha256"]
            or item.get("training_config_sha256") != training_config_sha
            or item.get("strictly_prior_gradient_archive") is not True
            or item.get("current_checkpoint_gradient_included") is not False
            or item.get("last_applied_gradient_generation") != generation - 1
        ):
            raise AssemblyError(f"{relative_run_dir}: checkpoint lineage is invalid")
    if set(capture_by_generation) != set(manifest["checkpoint_generations"]):
        raise AssemblyError(f"{relative_run_dir}: checkpoint generations are incomplete")
    expected_checkpoint_files = {
        f"checkpoint_generation_{generation:06d}.npz"
        for generation in manifest["checkpoint_generations"]
    }
    if set(os.listdir(checkpoint_dir)) != expected_checkpoint_files:
        raise AssemblyError(f"{relative_run_dir}: checkpoint file inventory is not exact")

    history_relative = os.path.join(relative_run_dir, "history.jsonl")
    history_sha = _sha256_file(os.path.join(run_dir, "history.jsonl"))
    record = _stamp(
        {
            "training_id": training_id,
            "task_index": task_index,
            "env_name": config["env_name"],
            "training_seed": seed,
            "updates": dims["training_updates"],
            "population_size": dims["population_size"],
            "candidate_rollouts": dims["training_updates"] * dims["population_size"],
            "calibration_rollouts": dims["calibration_episodes"],
            "online_evaluation_rollouts": 0,
            "checkpoint_generations": list(manifest["checkpoint_generations"]),
            "training_transitions": training_transitions,
            "calibration_transitions": calibration,
            "training_log_sha256": history_sha,
            "training_log_path": history_relative,
            "stderr_sha256": EMPTY_SHA256,
            "stderr_empty": True,
        }
    )
    context = {
        "run_dir": relative_run_dir,
        "training_config_sha256": training_config_sha,
        "training_config_path": training_config_relative,
        "capture_sha256": _sha256_file(os.path.join(run_dir, "checkpoint_capture.json")),
        "capture": capture,
        "capture_by_generation": capture_by_generation,
    }
    return record, context


def _validate_q_summaries(records: Sequence[Mapping[str, Any]], label: str) -> None:
    for record_index, record in enumerate(records):
        summaries = record.get("q_summaries")
        if not isinstance(summaries, list):
            raise AssemblyError(f"{label}[{record_index}].q_summaries: not a list")
        for q_index, summary in enumerate(summaries):
            _exact_keys(
                summary,
                analyzer.Q_SUMMARY_KEYS,
                f"{label}[{record_index}].q_summaries[{q_index}]",
            )


def _fragment_records(
    *,
    artifact_root: str,
    diagnostic_root: str,
    checkpoint_id: int,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    expected_hashes: Mapping[str, str],
    training_context: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    relative_dir = os.path.join(diagnostic_root, f"checkpoint_{checkpoint_id:06d}")
    fragment_dir = _directory(artifact_root, relative_dir, "checkpoint fragment")
    _reject_symlinks(fragment_dir, "checkpoint fragment")
    index_relative = os.path.join(relative_dir, "checkpoint_index.json")
    fragment = _read_json(
        _regular_file(artifact_root, index_relative, "checkpoint fragment index"),
        "checkpoint fragment index",
    )
    _exact_keys(fragment, FRAGMENT_KEYS, "checkpoint fragment")
    stored_fragment_hash = fragment["fragment_sha256"]
    if not _is_sha256(stored_fragment_hash):
        raise AssemblyError("checkpoint fragment digest is invalid")
    fragment_payload = dict(fragment)
    fragment_payload.pop("fragment_sha256")
    if hashlib.sha256(_canonical_bytes(fragment_payload)).hexdigest() != stored_fragment_hash:
        raise AssemblyError("checkpoint fragment digest mismatch")
    _reject_inference(fragment, "checkpoint_fragment")

    task_index, seed, generation = analyzer.checkpoint_coordinates(
        manifest, checkpoint_id
    )
    exact = {
        "schema_version": 1,
        "study": STUDY,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "checkpoint_id": checkpoint_id,
        "task_index": task_index,
        "training_seed": seed,
        "generation": generation,
        "no_outcome_selection": True,
        "no_record_exclusion": True,
    }
    for key, expected in exact.items():
        if fragment[key] != expected:
            raise AssemblyError(f"checkpoint {checkpoint_id}.{key}: fragment mismatch")

    inventory = fragment["artifact_inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise AssemblyError(f"checkpoint {checkpoint_id}: artifact inventory is empty")
    expected_files: set[str] = set()
    inventory_paths: list[str] = []
    for index, item in enumerate(inventory):
        item = _exact_keys(item, INVENTORY_KEYS, f"inventory[{index}]")
        path = _normalized_relative(item["path"], f"inventory[{index}].path")
        if os.path.dirname(path) != relative_dir or path in expected_files:
            raise AssemblyError(f"inventory[{index}]: path is outside/duplicate")
        _verify_file(artifact_root, path, item["sha256"], f"inventory[{index}]")
        expected_files.add(path)
        inventory_paths.append(path)
    if inventory_paths != sorted(inventory_paths):
        raise AssemblyError(
            f"checkpoint {checkpoint_id}: artifact inventory is not deterministic"
        )
    actual_files = {
        os.path.join(relative_dir, name)
        for name in os.listdir(fragment_dir)
        if os.path.isfile(os.path.join(fragment_dir, name))
    }
    if actual_files != expected_files | {index_relative}:
        raise AssemblyError(
            f"checkpoint {checkpoint_id}: fragment inventory has extras/missing files"
        )
    if any(os.path.isdir(os.path.join(fragment_dir, name)) for name in os.listdir(fragment_dir)):
        raise AssemblyError(
            f"checkpoint {checkpoint_id}: nested fragment directories are forbidden"
        )
    expected_basenames = {
        "checkpoint.npz",
        "checkpoint_training_config.json",
        "checkpoint_capture.json",
        "basis.npz",
        "bank_A_raw.npz",
        "bank_A_diagnostics.npz",
        "bank_B_raw.npz",
        "bank_B_diagnostics.npz",
        "endpoints.npz",
        *{
            f"partition_{index:02d}_diagnostics.npz"
            for index in range(manifest["dimensions"]["bank_b_partition_count"])
        },
    }
    if {os.path.basename(path) for path in expected_files} != expected_basenames:
        raise AssemblyError(
            f"checkpoint {checkpoint_id}: fragment artifact set is not exact"
        )

    dimensions = manifest["dimensions"]
    specs = {
        "checkpoint": (analyzer.CHECKPOINT_KEYS, 1),
        "banks": (analyzer.BANK_KEYS, len(dimensions["banks"])),
        "partitions": (
            analyzer.PARTITION_KEYS,
            dimensions["bank_b_partition_count"],
        ),
        "checkpoint_metrics": (
            analyzer.METRIC_KEYS,
            len(dimensions["locality_q"]),
        ),
        "center_endpoints": (
            analyzer.CENTER_KEYS,
            dimensions["endpoint_episodes"],
        ),
        "endpoints": (
            analyzer.ENDPOINT_KEYS,
            len(dimensions["locality_q"])
            * dimensions["bank_b_partition_count"]
            * len(dimensions["endpoint_arms"])
            * dimensions["endpoint_episodes"],
        ),
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for field, (keys, expected_count) in specs.items():
        raw = [fragment[field]] if field == "checkpoint" else fragment[field]
        if not isinstance(raw, list) or len(raw) != expected_count:
            raise AssemblyError(
                f"checkpoint {checkpoint_id}.{field}: expected {expected_count} records"
            )
        result[field] = [
            _validate_record(value, keys, f"checkpoint {checkpoint_id}.{field}[{index}]")
            for index, value in enumerate(raw)
        ]
    _validate_q_summaries(result["banks"], "banks")
    _validate_q_summaries(result["partitions"], "partitions")

    checkpoint = result["checkpoint"][0]
    if (
        checkpoint["checkpoint_id"] != checkpoint_id
        or checkpoint["training_id"] != checkpoint_id
        // len(manifest["checkpoint_generations"])
        or checkpoint["task_index"] != task_index
        or checkpoint["env_name"] != manifest["tasks"][task_index]["env_name"]
        or checkpoint["training_seed"] != seed
        or checkpoint["generation"] != generation
        or checkpoint["source_sha256"] != expected_hashes["source_sha256"]
        or checkpoint["training_config_sha256"]
        != training_context["training_config_sha256"]
    ):
        raise AssemblyError(f"checkpoint {checkpoint_id}: checkpoint lineage mismatch")
    capture_item = training_context["capture_by_generation"][generation]
    if checkpoint["checkpoint_artifact_sha256"] != capture_item["artifact_sha256"]:
        raise AssemblyError(f"checkpoint {checkpoint_id}: source checkpoint copy mismatch")
    _verify_file(
        artifact_root,
        checkpoint["checkpoint_artifact_path"],
        checkpoint["checkpoint_artifact_sha256"],
        f"checkpoint {checkpoint_id}.checkpoint_artifact",
    )
    _verify_file(
        artifact_root,
        checkpoint["training_config_path"],
        checkpoint["training_config_sha256"],
        f"checkpoint {checkpoint_id}.training_config",
    )
    _verify_file(
        artifact_root,
        checkpoint["basis_artifact_path"],
        checkpoint["basis_artifact_sha256"],
        f"checkpoint {checkpoint_id}.basis",
    )
    lineage = _exact_keys(fragment["lineage_artifacts"], LINEAGE_KEYS, "lineage")
    if lineage["capture_manifest_sha256"] != training_context["capture_sha256"]:
        raise AssemblyError(f"checkpoint {checkpoint_id}: capture copy mismatch")
    _verify_file(
        artifact_root,
        lineage["capture_manifest_path"],
        lineage["capture_manifest_sha256"],
        f"checkpoint {checkpoint_id}.capture",
    )

    referenced_paths = {
        checkpoint["checkpoint_artifact_path"],
        checkpoint["training_config_path"],
        checkpoint["basis_artifact_path"],
        lineage["capture_manifest_path"],
    }
    expected_bank_names = list(dimensions["banks"])
    if [record["bank"] for record in result["banks"]] != expected_bank_names:
        raise AssemblyError(f"checkpoint {checkpoint_id}: bank order/identity is invalid")
    expected_pair_indices = list(range(dimensions["pairs_per_bank"]))
    all_bank_seeds: set[int] = set()
    for bank in result["banks"]:
        bank_name = bank["bank"]
        if (
            bank["checkpoint_id"] != checkpoint_id
            or bank["bank_id"]
            != analyzer.bank_id_for(manifest, checkpoint_id, bank_name)
            or bank["pair_count"] != dimensions["pairs_per_bank"]
            or bank["candidate_rollouts"] != 2 * dimensions["pairs_per_bank"]
            or bank["pair_indices"] != expected_pair_indices
            or [summary["q"] for summary in bank["q_summaries"]]
            != list(dimensions["locality_q"])
        ):
            raise AssemblyError(f"checkpoint {checkpoint_id}: bank identity/count is invalid")
        seed_lists = (
            bank["perturbation_seeds"],
            bank["rollout_seeds_plus"],
            bank["rollout_seeds_minus"],
        )
        if any(
            not isinstance(values, list)
            or len(values) != dimensions["pairs_per_bank"]
            or len(set(values)) != len(values)
            for values in seed_lists
        ):
            raise AssemblyError(f"checkpoint {checkpoint_id}: bank seed counts are invalid")
        if bank["rollout_seeds_plus"] != bank["rollout_seeds_minus"]:
            raise AssemblyError(f"checkpoint {checkpoint_id}: antithetic CRN pairing changed")
        current = set(bank["perturbation_seeds"]) | set(bank["rollout_seeds_plus"])
        if set(bank["perturbation_seeds"]) & set(bank["rollout_seeds_plus"]):
            raise AssemblyError(f"checkpoint {checkpoint_id}: bank RNG streams overlap")
        if current & all_bank_seeds:
            raise AssemblyError(f"checkpoint {checkpoint_id}: Bank A/B RNG streams overlap")
        all_bank_seeds.update(current)
        for path_key, hash_key in (
            ("raw_bank_path", "raw_bank_sha256"),
            ("diagnostics_path", "diagnostics_sha256"),
        ):
            referenced_paths.add(bank[path_key])
            _verify_file(
                artifact_root,
                bank[path_key],
                bank[hash_key],
                f"checkpoint {checkpoint_id}.bank.{path_key}",
            )
        if bank["stderr_empty"] is not True or bank["stderr_sha256"] != EMPTY_SHA256:
            raise AssemblyError(f"checkpoint {checkpoint_id}: bank stderr claim is invalid")
    expected_partitions = analyzer.bank_b_partition(manifest, checkpoint_id)
    for partition_index, partition in enumerate(result["partitions"]):
        if (
            partition["checkpoint_id"] != checkpoint_id
            or partition["partition_index"] != partition_index
            or partition["partition_id"]
            != analyzer.partition_id_for(manifest, checkpoint_id, partition_index)
            or partition["pair_count"] != dimensions["pairs_per_partition"]
            or partition["pair_indices"] != expected_partitions[partition_index]
            or [summary["q"] for summary in partition["q_summaries"]]
            != list(dimensions["locality_q"])
        ):
            raise AssemblyError(
                f"checkpoint {checkpoint_id}: partition identity/coverage is invalid"
            )
        referenced_paths.add(partition["diagnostics_path"])
        _verify_file(
            artifact_root,
            partition["diagnostics_path"],
            partition["diagnostics_sha256"],
            f"checkpoint {checkpoint_id}.partition",
        )
    for q, metric in zip(
        dimensions["locality_q"], result["checkpoint_metrics"], strict=True
    ):
        if (
            metric["checkpoint_id"] != checkpoint_id
            or metric["q"] != q
            or metric["metric_id"]
            != analyzer.metric_id_for(manifest, checkpoint_id, q)
        ):
            raise AssemblyError(f"checkpoint {checkpoint_id}: metric identity is invalid")
    for episode, center in enumerate(result["center_endpoints"]):
        if (
            center["checkpoint_id"] != checkpoint_id
            or center["episode_index"] != episode
            or center["center_endpoint_id"]
            != analyzer.center_endpoint_id_for(manifest, checkpoint_id, episode)
        ):
            raise AssemblyError(f"checkpoint {checkpoint_id}: center identity is invalid")
    endpoint_keys = set()
    for endpoint in result["endpoints"]:
        key = (
            endpoint["q"],
            endpoint["partition_index"],
            endpoint["arm"],
            endpoint["episode_index"],
        )
        if key in endpoint_keys:
            raise AssemblyError(f"checkpoint {checkpoint_id}: duplicate endpoint coordinate")
        endpoint_keys.add(key)
        try:
            expected_endpoint_id = analyzer.endpoint_id_for(
                manifest,
                checkpoint_id,
                endpoint["q"],
                endpoint["partition_index"],
                endpoint["arm"],
                endpoint["episode_index"],
            )
        except (ValueError, TypeError) as error:
            raise AssemblyError(
                f"checkpoint {checkpoint_id}: endpoint coordinate is invalid"
            ) from error
        if (
            endpoint["checkpoint_id"] != checkpoint_id
            or endpoint["endpoint_id"] != expected_endpoint_id
        ):
            raise AssemblyError(f"checkpoint {checkpoint_id}: endpoint identity is invalid")
    expected_endpoint_keys = {
        (q, partition, arm, episode)
        for q in dimensions["locality_q"]
        for partition in range(dimensions["bank_b_partition_count"])
        for arm in dimensions["endpoint_arms"]
        for episode in range(dimensions["endpoint_episodes"])
    }
    if endpoint_keys != expected_endpoint_keys:
        raise AssemblyError(f"checkpoint {checkpoint_id}: endpoint grid is incomplete")
    for field in ("center_endpoints", "endpoints"):
        for record in result[field]:
            referenced_paths.add(record["rollout_artifact_path"])
            _verify_file(
                artifact_root,
                record["rollout_artifact_path"],
                record["rollout_artifact_sha256"],
                f"checkpoint {checkpoint_id}.{field}.rollout",
            )
    if referenced_paths != expected_files:
        raise AssemblyError(
            f"checkpoint {checkpoint_id}: inventory has unreferenced or cross-fragment files"
        )

    transitions = _exact_keys(
        fragment["transition_totals"], TRANSITION_TOTAL_KEYS, "transition totals"
    )
    recomputed = {
        "bank": sum(record["candidate_transitions"] for record in result["banks"]),
        "center": sum(record["transitions"] for record in result["center_endpoints"]),
        "endpoint": sum(record["transitions"] for record in result["endpoints"]),
    }
    if dict(transitions) != recomputed:
        raise AssemblyError(f"checkpoint {checkpoint_id}: transition totals mismatch")
    result["checkpoints"] = result.pop("checkpoint")
    return result


def assemble(
    *,
    artifact_root: str,
    manifest_path: str,
    expected_hashes: Mapping[str, str],
    source_snapshot_path: str,
    launcher_lock_path: str,
    dependency_lock_path: str,
    training_root: str = "training_runs",
    diagnostic_root: str = "checkpoint_artifacts",
    training_stderr_root: str = "stderr/training",
    diagnostic_stderr_root: str = "stderr/diagnostic",
    require_preregistered_manifest: bool = True,
) -> dict[str, Any]:
    """Validate every raw input and return the deterministic audit index."""

    artifact_root = os.path.abspath(artifact_root)
    if os.path.islink(artifact_root) or not os.path.isdir(artifact_root):
        raise AssemblyError("artifact root must be an existing non-symlink directory")
    required_locks = {
        "source_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "analyzer_sha256",
        "launcher_sha256",
        "dependency_lock_sha256",
    }
    if set(expected_hashes) != required_locks or any(
        not _is_sha256(value) for value in expected_hashes.values()
    ):
        raise AssemblyError("expected provenance lock set is incomplete or invalid")
    if os.path.islink(manifest_path) or not os.path.isfile(manifest_path):
        raise AssemblyError("manifest must be an existing non-symlink regular file")
    manifest, manifest_sha = analyzer.load_and_validate_manifest(
        manifest_path,
        expected_sha256=expected_hashes["manifest_sha256"],
        require_preregistered=require_preregistered_manifest,
    )
    if manifest_sha != expected_hashes["manifest_sha256"]:
        raise AssemblyError("manifest lock changed after validation")
    protocol_relative = _normalized_relative(
        manifest["protocol"]["path"], "manifest protocol"
    )
    protocol_path = os.path.join(analyzer.REPO_ROOT, protocol_relative)
    if os.path.commonpath(
        [os.path.realpath(analyzer.REPO_ROOT), os.path.realpath(protocol_path)]
    ) != os.path.realpath(analyzer.REPO_ROOT):
        raise AssemblyError("manifest protocol path escapes the repository root")
    lock_files = {
        "protocol_sha256": protocol_path,
        "analyzer_sha256": analyzer.__file__,
        "launcher_sha256": launcher_lock_path,
        "dependency_lock_sha256": dependency_lock_path,
    }
    for key, path in lock_files.items():
        if os.path.islink(path) or not os.path.isfile(path):
            raise AssemblyError(f"{key}: lock file is missing or a symlink")
        actual = _sha256_file(path)
        if actual != expected_hashes[key]:
            raise AssemblyError(f"{key}: expected {expected_hashes[key]}, found {actual}")
    if manifest["protocol"]["sha256"] != expected_hashes["protocol_sha256"]:
        raise AssemblyError("manifest protocol digest disagrees with enforced lock")
    snapshot = _directory(artifact_root, source_snapshot_path, "source snapshot")
    _reject_symlinks(snapshot, "source snapshot")
    if require_preregistered_manifest:
        try:
            snapshot_sha256 = compute_lagged_subspace_study_sha256(snapshot)
        except StudySourceLockError as error:
            raise AssemblyError(
                f"source snapshot failed its exact composite inventory: {error}"
            ) from error
        if snapshot_sha256 != expected_hashes["source_sha256"]:
            raise AssemblyError(
                "source snapshot composite digest disagrees with the global study lock"
            )
        try:
            validate_hash_bundle(
                snapshot,
                LAUNCHER_BUNDLE_PATH,
                expected_bundle_sha256=expected_hashes["launcher_sha256"],
                expected_kind="launchers",
            )
            validate_hash_bundle(
                snapshot,
                DEPENDENCY_BUNDLE_PATH,
                expected_bundle_sha256=expected_hashes[
                    "dependency_lock_sha256"
                ],
                expected_kind="dependency_locks",
            )
        except StudySourceLockError as error:
            raise AssemblyError(
                f"source snapshot provenance bundle is invalid: {error}"
            ) from error

    tasks = manifest["tasks"]
    seeds = manifest["training_seeds"]
    generations = manifest["checkpoint_generations"]
    dims = manifest["dimensions"]
    training_count = len(tasks) * len(seeds)
    checkpoint_count = training_count * len(generations)
    training_directory = _directory(artifact_root, training_root, "training root")
    expected_training_names = {
        f"training_{index:06d}" for index in range(training_count)
    }
    if set(os.listdir(training_directory)) != expected_training_names:
        raise AssemblyError("training directory set is partial, duplicated, or has extras")
    diagnostic_directory = _directory(artifact_root, diagnostic_root, "diagnostic root")
    expected_diagnostic_names = {
        f"checkpoint_{index:06d}" for index in range(checkpoint_count)
    }
    if set(os.listdir(diagnostic_directory)) != expected_diagnostic_names:
        raise AssemblyError("diagnostic fragment set is partial, duplicated, or has extras")
    _validate_stderr_inventory(
        artifact_root, training_stderr_root, prefix="training", count=training_count
    )
    _validate_stderr_inventory(
        artifact_root,
        diagnostic_stderr_root,
        prefix="checkpoint",
        count=checkpoint_count,
    )

    training_records: list[dict[str, Any]] = []
    training_context: dict[int, dict[str, Any]] = {}
    for training_id in range(training_count):
        relative = os.path.join(training_root, f"training_{training_id:06d}")
        record, context = _training_record(
            artifact_root=artifact_root,
            relative_run_dir=relative,
            manifest=manifest,
            expected_hashes=expected_hashes,
        )
        if record["training_id"] != training_id or training_id in training_context:
            raise AssemblyError("training identities are duplicated or noncontiguous")
        training_records.append(record)
        training_context[training_id] = context

    merged = {
        "checkpoints": [],
        "banks": [],
        "partitions": [],
        "checkpoint_metrics": [],
        "center_endpoints": [],
        "endpoints": [],
    }
    for checkpoint_id in range(checkpoint_count):
        training_id = checkpoint_id // len(generations)
        records = _fragment_records(
            artifact_root=artifact_root,
            diagnostic_root=diagnostic_root,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            expected_hashes=expected_hashes,
            training_context=training_context[training_id],
        )
        for field in merged:
            merged[field].extend(records[field])

    identity_fields = {
        "checkpoints": "checkpoint_id",
        "banks": "bank_id",
        "partitions": "partition_id",
        "checkpoint_metrics": "metric_id",
        "center_endpoints": "center_endpoint_id",
        "endpoints": "endpoint_id",
    }
    for field, identity in identity_fields.items():
        merged[field].sort(key=lambda record: record[identity])
        found = [record[identity] for record in merged[field]]
        if found != list(range(len(found))):
            raise AssemblyError(f"{field}: identities are partial, duplicate, or noncontiguous")

    rollout_budget = {
        "checkpoint_training_candidate_rollouts": sum(
            record["candidate_rollouts"] for record in training_records
        ),
        "normalization_calibration_rollouts": sum(
            record["calibration_rollouts"] for record in training_records
        ),
        "bank_candidate_rollouts": sum(
            record["candidate_rollouts"] for record in merged["banks"]
        ),
        "endpoint_arm_rollouts": len(merged["endpoints"]),
        "checkpoint_center_rollouts": len(merged["center_endpoints"]),
    }
    rollout_budget["total_policy_rollouts"] = sum(rollout_budget.values())
    expected_rollout_budget = dict(manifest["budget"])
    expected_rollout_budget.pop("environment_transitions_are_separate")
    if rollout_budget != expected_rollout_budget:
        raise AssemblyError("recomputed policy-rollout budget disagrees with manifest")
    transition_budget = {
        "checkpoint_training_transitions": sum(
            record["training_transitions"] for record in training_records
        ),
        "normalization_calibration_transitions": sum(
            record["calibration_transitions"] for record in training_records
        ),
        "bank_transitions": sum(
            record["candidate_transitions"] for record in merged["banks"]
        ),
        "endpoint_arm_transitions": sum(
            record["transitions"] for record in merged["endpoints"]
        ),
        "checkpoint_center_transitions": sum(
            record["transitions"] for record in merged["center_endpoints"]
        ),
    }
    transition_budget["total_environment_transitions"] = sum(
        transition_budget.values()
    )
    budget = {**rollout_budget, **transition_budget}
    if set(budget) != analyzer.BUDGET_KEYS:
        raise AssemblyError("assembled budget schema drifted from analyzer")

    provenance = _stamp(
        {
            **dict(expected_hashes),
            "source_snapshot_path": _normalized_relative(
                source_snapshot_path, "source snapshot"
            ),
            "stderr_empty": True,
            "documented_infrastructure_failures": [],
        }
    )
    artifact = {
        "schema_version": 1,
        "study": STUDY,
        "designation": manifest["designation"],
        "manifest_sha256": manifest_sha,
        "provenance": provenance,
        "analysis_declaration": analyzer._expected_analysis_declaration(manifest),
        "training_runs": training_records,
        **merged,
        "budget": budget,
    }
    if set(artifact) != analyzer.TOP_LEVEL_KEYS:
        raise AssemblyError("assembled top-level schema drifted from analyzer")
    _reject_inference(artifact)
    return artifact


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-snapshot-path", required=True)
    parser.add_argument("--launcher-lock", required=True)
    parser.add_argument("--dependency-lock", required=True)
    parser.add_argument("--training-root", default="training_runs")
    parser.add_argument("--diagnostic-root", default="checkpoint_artifacts")
    parser.add_argument("--training-stderr-root", default="stderr/training")
    parser.add_argument("--diagnostic-stderr-root", default="stderr/diagnostic")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    parser.add_argument("--output", required=True, help="root-relative audit-index path")
    parser.add_argument(
        "--fixture-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_root = os.path.abspath(args.artifact_root)
    output_relative = _normalized_relative(args.output, "output")
    output_path = _inside_root(artifact_root, output_relative, "output")
    if os.path.lexists(output_path):
        raise AssemblyError("output already exists; immutable assembly refuses overwrite")
    expected_hashes = {
        "source_sha256": args.expected_source_sha256,
        "manifest_sha256": args.expected_manifest_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "analyzer_sha256": args.expected_analyzer_sha256,
        "launcher_sha256": args.expected_launcher_sha256,
        "dependency_lock_sha256": args.expected_dependency_lock_sha256,
    }
    artifact = assemble(
        artifact_root=artifact_root,
        manifest_path=args.manifest,
        expected_hashes=expected_hashes,
        source_snapshot_path=args.source_snapshot_path,
        launcher_lock_path=args.launcher_lock,
        dependency_lock_path=args.dependency_lock,
        training_root=args.training_root,
        diagnostic_root=args.diagnostic_root,
        training_stderr_root=args.training_stderr_root,
        diagnostic_stderr_root=args.diagnostic_stderr_root,
        require_preregistered_manifest=not args.fixture_mode,
    )
    _atomic_json(output_path, artifact)
    readback = _read_json(output_path, "assembled audit index")
    if readback != artifact:
        raise AssemblyError("atomic audit-index readback changed its content")
    print(
        f"Assembled {len(artifact['training_runs'])} training runs and "
        f"{len(artifact['checkpoints'])} checkpoints into {output_path}"
    )


if __name__ == "__main__":
    main()

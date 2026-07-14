#!/usr/bin/env python3
"""Validate the complete checkpoint-generation stage and commit a report.

This validator is intentionally narrower than the final artifact assembler. It
validates all checkpoint-generating runs before the diagnostic array is
submitted, without inspecting or requiring any diagnostic result.  The strict
per-run and stderr contracts are delegated to the immutable assembler helpers
so this gate cannot accept a weaker checkpoint stage than final assembly.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import tempfile
from typing import Any, Mapping

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import lagged_subspace_study_lock as study_lock
from scripts import analyze_lagged_subspace_frozen_checkpoint as analyzer
from scripts import assemble_lagged_subspace_frozen_checkpoint as assembler


AssemblyError = assembler.AssemblyError
STUDY = assembler.STUDY
EXPECTED_PRODUCTION_TRAINING_RUNS = 60
EXPECTED_PRODUCTION_CHECKPOINTS = 180
ASSEMBLER_RELATIVE_PATH = "scripts/assemble_lagged_subspace_frozen_checkpoint.py"
ANALYZER_RELATIVE_PATH = "scripts/analyze_lagged_subspace_frozen_checkpoint.py"
STUDY_LOCK_RELATIVE_PATH = "experiments/lagged_subspace_study_lock.py"


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return hashlib.sha256(assembler._canonical_bytes(payload)).hexdigest()


def _stamp_report(report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    if "report_sha256" in result:
        raise AssemblyError("cannot stamp a report that already has report_sha256")
    result["report_sha256"] = _report_sha256(result)
    return result


def _resolve_snapshot(artifact_root: str, value: str) -> tuple[str, str]:
    if not isinstance(value, str) or not value:
        raise AssemblyError("source snapshot path is empty")
    if os.path.isabs(value):
        candidate = os.path.abspath(value)
        if os.path.islink(candidate):
            raise AssemblyError("source snapshot must not be a symlink")
        root_real = os.path.realpath(artifact_root)
        candidate_real = os.path.realpath(candidate)
        if os.path.commonpath([root_real, candidate_real]) != root_real:
            raise AssemblyError("source snapshot must be inside the artifact root")
        relative = os.path.relpath(candidate_real, root_real)
        relative = assembler._normalized_relative(relative, "source snapshot")
        snapshot = assembler._directory(
            artifact_root, relative, "source snapshot"
        )
    else:
        relative = assembler._normalized_relative(value, "source snapshot")
        snapshot = assembler._directory(
            artifact_root, relative, "source snapshot"
        )
    assembler._reject_symlinks(snapshot, "source snapshot")
    return snapshot, relative


def _load_and_validate_manifest(
    path: str,
    *,
    expected_sha256: str,
    require_preregistered: bool,
) -> tuple[dict[str, Any], str]:
    """Run the locked structural checks without consulting a mutable repo root."""

    actual = assembler._sha256_file(path)
    if actual != expected_sha256:
        raise AssemblyError(
            f"manifest digest mismatch: expected {expected_sha256}, found {actual}"
        )
    manifest = analyzer._read_json(path)
    issues = analyzer._validate_manifest_structure(
        manifest, require_preregistered=require_preregistered
    )
    if issues:
        raise analyzer.SubspaceValidationError(issues)
    return manifest, actual


def _validate_provenance(
    *,
    artifact_root: str,
    manifest_path: str,
    source_snapshot_path: str,
    launcher_lock_path: str,
    dependency_lock_path: str,
    expected_hashes: Mapping[str, str],
    require_preregistered_manifest: bool,
) -> tuple[dict[str, Any], str, str, str, str]:
    required_locks = {
        "source_sha256",
        "manifest_sha256",
        "protocol_sha256",
        "analyzer_sha256",
        "launcher_sha256",
        "dependency_lock_sha256",
    }
    if set(expected_hashes) != required_locks or any(
        not assembler._is_sha256(value) for value in expected_hashes.values()
    ):
        raise AssemblyError("expected provenance lock set is incomplete or invalid")
    if os.path.islink(manifest_path) or not os.path.isfile(manifest_path):
        raise AssemblyError("manifest must be an existing non-symlink regular file")

    snapshot, snapshot_relative = _resolve_snapshot(
        artifact_root, source_snapshot_path
    )
    manifest, manifest_sha = _load_and_validate_manifest(
        manifest_path,
        expected_sha256=expected_hashes["manifest_sha256"],
        require_preregistered=require_preregistered_manifest,
    )
    if manifest_sha != expected_hashes["manifest_sha256"]:
        raise AssemblyError("manifest lock changed after validation")

    protocol_relative = assembler._normalized_relative(
        manifest["protocol"]["path"], "manifest protocol"
    )
    if require_preregistered_manifest:
        protocol_path = os.path.join(snapshot, protocol_relative)
        analyzer_path = os.path.join(snapshot, ANALYZER_RELATIVE_PATH)
    else:
        protocol_path = os.path.join(analyzer.REPO_ROOT, protocol_relative)
        analyzer_path = analyzer.__file__
    lock_files = {
        "protocol_sha256": protocol_path,
        "analyzer_sha256": analyzer_path,
        "launcher_sha256": launcher_lock_path,
        "dependency_lock_sha256": dependency_lock_path,
    }
    for key, path in lock_files.items():
        if os.path.islink(path) or not os.path.isfile(path):
            raise AssemblyError(f"{key}: lock file is missing or a symlink")
        actual = assembler._sha256_file(path)
        if actual != expected_hashes[key]:
            raise AssemblyError(
                f"{key}: expected {expected_hashes[key]}, found {actual}"
            )
    if manifest["protocol"]["sha256"] != expected_hashes["protocol_sha256"]:
        raise AssemblyError("manifest protocol digest disagrees with enforced lock")

    if require_preregistered_manifest:
        try:
            snapshot_sha256 = assembler.compute_lagged_subspace_study_sha256(
                snapshot
            )
        except assembler.StudySourceLockError as error:
            raise AssemblyError(
                f"source snapshot failed its exact composite inventory: {error}"
            ) from error
        if snapshot_sha256 != expected_hashes["source_sha256"]:
            raise AssemblyError(
                "source snapshot composite digest disagrees with the global study lock"
            )
        try:
            assembler.validate_hash_bundle(
                snapshot,
                assembler.LAUNCHER_BUNDLE_PATH,
                expected_bundle_sha256=expected_hashes["launcher_sha256"],
                expected_kind="launchers",
            )
            assembler.validate_hash_bundle(
                snapshot,
                assembler.DEPENDENCY_BUNDLE_PATH,
                expected_bundle_sha256=expected_hashes[
                    "dependency_lock_sha256"
                ],
                expected_kind="dependency_locks",
            )
        except assembler.StudySourceLockError as error:
            raise AssemblyError(
                f"source snapshot provenance bundle is invalid: {error}"
            ) from error

        imported_analyzer_sha = assembler._sha256_file(analyzer.__file__)
        if imported_analyzer_sha != expected_hashes["analyzer_sha256"]:
            raise AssemblyError(
                "imported analyzer differs from the analyzer in the source snapshot"
            )
        snapshot_assembler = os.path.join(snapshot, ASSEMBLER_RELATIVE_PATH)
        if os.path.islink(snapshot_assembler) or not os.path.isfile(
            snapshot_assembler
        ):
            raise AssemblyError("source snapshot assembler is missing or a symlink")
        assembler_sha = assembler._sha256_file(assembler.__file__)
        if assembler._sha256_file(snapshot_assembler) != assembler_sha:
            raise AssemblyError(
                "imported assembler differs from the assembler in the source snapshot"
            )
        snapshot_study_lock = os.path.join(snapshot, STUDY_LOCK_RELATIVE_PATH)
        if os.path.islink(snapshot_study_lock) or not os.path.isfile(
            snapshot_study_lock
        ):
            raise AssemblyError("source snapshot study lock is missing or a symlink")
        study_lock_sha = assembler._sha256_file(study_lock.__file__)
        if assembler._sha256_file(snapshot_study_lock) != study_lock_sha:
            raise AssemblyError(
                "imported study lock differs from the study lock in the source snapshot"
            )
    else:
        assembler_sha = assembler._sha256_file(assembler.__file__)
        study_lock_sha = assembler._sha256_file(study_lock.__file__)

    return (
        manifest,
        manifest_sha,
        snapshot_relative,
        assembler_sha,
        study_lock_sha,
    )


def validate_checkpoint_stage(
    *,
    artifact_root: str,
    manifest_path: str,
    source_snapshot_path: str,
    launcher_lock_path: str,
    dependency_lock_path: str,
    expected_hashes: Mapping[str, str],
    training_root: str = "training_runs",
    training_stderr_root: str = "stderr/training",
    require_preregistered_manifest: bool = True,
) -> dict[str, Any]:
    """Validate every training run and return a deterministic stage report."""

    artifact_root = os.path.abspath(artifact_root)
    if os.path.islink(artifact_root) or not os.path.isdir(artifact_root):
        raise AssemblyError("artifact root must be an existing non-symlink directory")
    training_root = assembler._normalized_relative(training_root, "training root")
    training_stderr_root = assembler._normalized_relative(
        training_stderr_root, "training stderr root"
    )
    (
        manifest,
        manifest_sha,
        snapshot_relative,
        assembler_sha,
        study_lock_sha,
    ) = _validate_provenance(
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        source_snapshot_path=source_snapshot_path,
        launcher_lock_path=launcher_lock_path,
        dependency_lock_path=dependency_lock_path,
        expected_hashes=expected_hashes,
        require_preregistered_manifest=require_preregistered_manifest,
    )

    training_count = len(manifest["tasks"]) * len(manifest["training_seeds"])
    checkpoint_count = training_count * len(manifest["checkpoint_generations"])
    if require_preregistered_manifest and (
        training_count != EXPECTED_PRODUCTION_TRAINING_RUNS
        or checkpoint_count != EXPECTED_PRODUCTION_CHECKPOINTS
    ):
        raise AssemblyError(
            "preregistered checkpoint stage must contain exactly 60 training runs "
            "and 180 checkpoints"
        )

    training_directory = assembler._directory(
        artifact_root, training_root, "training root"
    )
    expected_names = {
        f"training_{training_id:06d}" for training_id in range(training_count)
    }
    if set(os.listdir(training_directory)) != expected_names:
        raise AssemblyError(
            "training directory set is partial, duplicated, symlinked, or has extras"
        )
    assembler._validate_stderr_inventory(
        artifact_root,
        training_stderr_root,
        prefix="training",
        count=training_count,
    )

    training_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    generations = list(manifest["checkpoint_generations"])
    for training_id in range(training_count):
        relative_run_dir = os.path.join(
            training_root, f"training_{training_id:06d}"
        )
        record, context = assembler._training_record(
            artifact_root=artifact_root,
            relative_run_dir=relative_run_dir,
            manifest=manifest,
            expected_hashes=expected_hashes,
        )
        if record["training_id"] != training_id:
            raise AssemblyError("training identities are duplicated or noncontiguous")
        training_records.append(record)
        for generation in generations:
            checkpoint_id = analyzer.checkpoint_id_for(
                manifest,
                record["task_index"],
                record["training_seed"],
                generation,
            )
            capture_item = context["capture_by_generation"][generation]
            checkpoint_records.append(
                assembler._stamp(
                    {
                        "checkpoint_id": checkpoint_id,
                        "training_id": training_id,
                        "task_index": record["task_index"],
                        "env_name": record["env_name"],
                        "training_seed": record["training_seed"],
                        "generation": generation,
                        "checkpoint_artifact_path": os.path.join(
                            relative_run_dir, capture_item["artifact"]
                        ),
                        "checkpoint_artifact_sha256": capture_item[
                            "artifact_sha256"
                        ],
                        "training_config_path": context[
                            "training_config_path"
                        ],
                        "training_config_sha256": context[
                            "training_config_sha256"
                        ],
                        "capture_manifest_path": os.path.join(
                            relative_run_dir, "checkpoint_capture.json"
                        ),
                        "capture_manifest_sha256": context["capture_sha256"],
                        "strictly_prior_gradient_archive": True,
                        "reward_selection_used": False,
                    }
                )
            )

    checkpoint_records.sort(key=lambda record: record["checkpoint_id"])
    if [record["checkpoint_id"] for record in checkpoint_records] != list(
        range(checkpoint_count)
    ):
        raise AssemblyError(
            "checkpoint identities are partial, duplicate, or noncontiguous"
        )

    candidate_rollouts = sum(
        record["candidate_rollouts"] for record in training_records
    )
    calibration_rollouts = sum(
        record["calibration_rollouts"] for record in training_records
    )
    training_transitions = sum(
        record["training_transitions"] for record in training_records
    )
    calibration_transitions = sum(
        record["calibration_transitions"] for record in training_records
    )
    provenance = assembler._stamp(
        {
            **dict(expected_hashes),
            "source_snapshot_path": snapshot_relative,
            "validator_sha256": assembler._sha256_file(__file__),
            "assembler_sha256": assembler_sha,
            "study_lock_sha256": study_lock_sha,
            "stderr_empty": True,
            "validation_mode": (
                "preregistered" if require_preregistered_manifest else "fixture"
            ),
        }
    )
    report = _stamp_report(
        {
            "schema_version": 1,
            "study": STUDY,
            "stage": "checkpoint_generation",
            "status": "validated",
            "designation": manifest["designation"],
            "manifest_sha256": manifest_sha,
            "provenance": provenance,
            "counts": {
                "training_runs": training_count,
                "checkpoints": checkpoint_count,
                "training_stderr_files": training_count,
            },
            "budget": {
                "checkpoint_training_candidate_rollouts": candidate_rollouts,
                "normalization_calibration_rollouts": calibration_rollouts,
                "total_policy_rollouts": candidate_rollouts
                + calibration_rollouts,
                "checkpoint_training_transitions": training_transitions,
                "normalization_calibration_transitions": calibration_transitions,
                "total_environment_transitions": training_transitions
                + calibration_transitions,
            },
            "training_runs": training_records,
            "checkpoints": checkpoint_records,
            "no_outcome_selection": True,
            "no_forbidden_controls": True,
        }
    )
    assembler._reject_inference(report, "checkpoint stage report")
    return report


def _atomic_json_write_once(path: str, value: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    if os.path.lexists(path):
        raise AssemblyError(
            "output already exists; immutable checkpoint validation refuses overwrite"
        )
    descriptor, staged = tempfile.mkstemp(
        prefix=".checkpoint_stage_validation_", dir=directory
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(assembler._canonical_bytes(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise AssemblyError(
                    "output already exists; immutable checkpoint validation "
                    "refuses overwrite"
                ) from error
            raise
        os.unlink(staged)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--source-snapshot-path",
        "--source-snapshot",
        dest="source_snapshot_path",
        required=True,
    )
    parser.add_argument("--launcher-lock", required=True)
    parser.add_argument("--dependency-lock", required=True)
    parser.add_argument("--training-root", default="training_runs")
    parser.add_argument("--training-stderr-root", default="stderr/training")
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-analyzer-sha256", required=True)
    parser.add_argument("--expected-launcher-sha256", required=True)
    parser.add_argument("--expected-dependency-lock-sha256", required=True)
    parser.add_argument(
        "--output", required=True, help="root-relative checkpoint-stage report path"
    )
    parser.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_root = os.path.abspath(args.artifact_root)
    output_relative = assembler._normalized_relative(args.output, "output")
    output_path = assembler._inside_root(artifact_root, output_relative, "output")
    if os.path.lexists(output_path):
        raise AssemblyError(
            "output already exists; immutable checkpoint validation refuses overwrite"
        )
    expected_hashes = {
        "source_sha256": args.expected_source_sha256,
        "manifest_sha256": args.expected_manifest_sha256,
        "protocol_sha256": args.expected_protocol_sha256,
        "analyzer_sha256": args.expected_analyzer_sha256,
        "launcher_sha256": args.expected_launcher_sha256,
        "dependency_lock_sha256": args.expected_dependency_lock_sha256,
    }
    report = validate_checkpoint_stage(
        artifact_root=artifact_root,
        manifest_path=args.manifest,
        source_snapshot_path=args.source_snapshot_path,
        launcher_lock_path=args.launcher_lock,
        dependency_lock_path=args.dependency_lock,
        expected_hashes=expected_hashes,
        training_root=args.training_root,
        training_stderr_root=args.training_stderr_root,
        require_preregistered_manifest=not args.fixture_mode,
    )
    _atomic_json_write_once(output_path, report)
    readback = assembler._read_json(output_path, "checkpoint-stage report")
    if readback != report or _report_sha256(readback) != readback.get(
        "report_sha256"
    ):
        raise AssemblyError("atomic checkpoint-stage report readback changed its content")
    print(
        f"Validated {report['counts']['training_runs']} training runs and "
        f"{report['counts']['checkpoints']} checkpoints into {output_path} "
        f"(report_sha256={report['report_sha256']})"
    )


if __name__ == "__main__":
    main()

"""Focused tests for the lagged-study compute/storage disclosure."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest import mock

from experiments.lagged_subspace_study_lock import (
    STUDY_SOURCE_PATHS,
    compute_lagged_subspace_study_sha256,
)
from scripts import collect_lagged_subspace_compute_disclosure as disclosure


def _write(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(payload)


def _write_json(path: str, value: Any) -> None:
    _write(path, disclosure._canonical_bytes(value) + b"\n")


def _sha(path: str) -> str:
    return disclosure._sha256_file(path)


def _stamp_record(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = disclosure._record_sha256(result)
    return result


class DisclosureFixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = self.temporary.name
        self.artifact_root = os.path.join(self.root, "artifacts")
        self.snapshot_relative = "source_snapshot_fixture"
        self.snapshot = os.path.join(self.artifact_root, self.snapshot_relative)
        self.logs = os.path.join(self.root, "job_outputs")
        os.makedirs(self.snapshot)
        os.makedirs(self.logs)

        self.manifest = self._manifest()
        self.manifest_path = os.path.join(
            self.snapshot,
            "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
        )
        _write_json(self.manifest_path, self.manifest)
        self.manifest_sha = _sha(self.manifest_path)

        checkpoint_launcher = self._launcher(
            "lagckpt", "lagged_checkpoint", 2, 2, "1G", "01:00:00", False
        )
        diagnostic_launcher = self._launcher(
            "lagsubdiag", "lagged_diagnostic", 4, 2, "2G", "02:00:00", True
        )
        for relative, payload in (
            (disclosure.LAUNCHERS["checkpoint_generation"], checkpoint_launcher),
            (disclosure.LAUNCHERS["diagnostic"], diagnostic_launcher),
        ):
            _write(os.path.join(self.snapshot, relative), payload.encode("ascii"))
        launcher_bundle = {
            "schema_version": 1,
            "study": disclosure.STUDY,
            "kind": "launchers",
            "files": [
                {"path": relative, "sha256": _sha(os.path.join(self.snapshot, relative))}
                for relative in sorted(disclosure.LAUNCHERS.values())
            ],
        }
        launcher_bundle_path = os.path.join(self.snapshot, disclosure.LAUNCHER_BUNDLE)
        _write_json(launcher_bundle_path, launcher_bundle)
        self.launcher_sha = _sha(launcher_bundle_path)

        dependency_paths = ("environment.yml", "requirement.txt")
        for relative in dependency_paths:
            _write(os.path.join(self.snapshot, relative), f"fixture {relative}\n".encode())
        dependency_bundle = {
            "schema_version": 1,
            "study": disclosure.STUDY,
            "kind": "dependency_locks",
            "files": [
                {"path": relative, "sha256": _sha(os.path.join(self.snapshot, relative))}
                for relative in dependency_paths
            ],
        }
        dependency_bundle_path = os.path.join(self.snapshot, disclosure.DEPENDENCY_BUNDLE)
        _write_json(dependency_bundle_path, dependency_bundle)
        self.dependency_sha = _sha(dependency_bundle_path)

        study_lock = b'''RUNTIME_DEPENDENCY_VERSIONS = {"numpy": "1.26.4", "scipy": "1.15.3"}\n\n\ndef validate():\n    if tuple(sys.version_info[:3]) != (3, 10, 18):\n        raise RuntimeError\n'''
        self.study_lock_path = os.path.join(self.snapshot, disclosure.STUDY_LOCK)
        _write(self.study_lock_path, study_lock)
        self.study_lock_sha = _sha(self.study_lock_path)

        for relative in STUDY_SOURCE_PATHS:
            path = os.path.join(self.snapshot, relative)
            if not os.path.exists(path):
                _write(path, f"# fixture source: {relative}\n".encode("ascii"))
        for relative in (
            disclosure.CHECKPOINT_STAGE_VALIDATOR,
            disclosure.DIRECT_SNAPSHOT_LOCKS["assembler_sha256"],
        ):
            path = os.path.join(self.snapshot, relative)
            if not os.path.exists(path):
                _write(path, f"# fixture tool: {relative}\n".encode("ascii"))

        protocol_path = os.path.join(
            self.snapshot, disclosure.DIRECT_SNAPSHOT_LOCKS["protocol_sha256"]
        )
        analyzer_path = os.path.join(
            self.snapshot, disclosure.DIRECT_SNAPSHOT_LOCKS["analyzer_sha256"]
        )
        validator_path = os.path.join(
            self.snapshot, disclosure.CHECKPOINT_STAGE_VALIDATOR
        )
        assembler_path = os.path.join(
            self.snapshot, disclosure.DIRECT_SNAPSHOT_LOCKS["assembler_sha256"]
        )

        self.locks = {
            "source_sha256": compute_lagged_subspace_study_sha256(self.snapshot),
            "manifest_sha256": self.manifest_sha,
            "protocol_sha256": _sha(protocol_path),
            "analyzer_sha256": _sha(analyzer_path),
            "launcher_sha256": self.launcher_sha,
            "dependency_lock_sha256": self.dependency_sha,
        }
        self.validator_sha = _sha(validator_path)
        self.validator_path = validator_path
        self.assembler_sha = _sha(assembler_path)
        self.training_records = self._training_artifacts()
        self._diagnostic_artifacts()
        self.stage_path = self._stage_report()
        self.stage_sha = _sha(self.stage_path)
        self.audit_path = self._audit_index()
        self.audit_sha = _sha(self.audit_path)
        self._slurm_logs()
        self.checkpoint_accounting = self._accounting("checkpoint.sacct", "111", 2, 40)
        self.diagnostic_accounting = self._accounting("diagnostic.sacct", "222", 4, 80)

    def cleanup(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _manifest() -> dict[str, Any]:
        return {
            "schema_version": 1,
            "study": disclosure.STUDY,
            "designation": "preregistered_mechanism_diagnostic",
            "protocol_status": "final_locked_before_environment_outcomes",
            "tasks": [{"task_index": 0, "env_name": "Fixture-v0"}],
            "training_seeds": [1, 2],
            "checkpoint_generations": [5, 10],
            "dimensions": {
                "training_updates": 2,
                "population_size": 4,
                "calibration_episodes": 1,
                "banks": ["A", "B"],
                "pairs_per_bank": 5,
                "bank_b_partition_count": 2,
                "locality_q": [0.5],
                "endpoint_arms": ["structured", "isotropic"],
                "endpoint_episodes": 3,
            },
            "budget": {
                "checkpoint_training_candidate_rollouts": 16,
                "normalization_calibration_rollouts": 2,
                "bank_candidate_rollouts": 80,
                "endpoint_arm_rollouts": 48,
                "checkpoint_center_rollouts": 12,
                "total_policy_rollouts": 158,
                "environment_transitions_are_separate": True,
            },
        }

    @staticmethod
    def _launcher(
        job_name: str,
        prefix: str,
        tasks: int,
        concurrency: int,
        memory: str,
        walltime: str,
        diagnostic: bool,
    ) -> str:
        extra = "CHUNK_PAIRS=32\n" if diagnostic else ""
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=job_outputs/{prefix}_%A_%a.out
#SBATCH --error=job_outputs/{prefix}_%A_%a.err
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem={memory}
#SBATCH --time={walltime}
#SBATCH --array=0-{tasks - 1}%{concurrency}
WORKERS=3
{extra}python -m experiments.lagged_subspace_study_lock validate-runtime
"""

    def _training_artifacts(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for training_id in range(2):
            relative = f"training_runs/training_{training_id:06d}"
            directory = os.path.join(self.artifact_root, relative)
            os.makedirs(directory)
            rows = [
                {
                    "iteration": iteration,
                    "iteration_compute_seconds": float(training_id + iteration + 1),
                    "best_fitness_so_far": "SECRET_OUTCOME",
                }
                for iteration in range(2)
            ]
            history = os.path.join(directory, "history.jsonl")
            _write(
                history,
                b"".join(disclosure._canonical_bytes(row) + b"\n" for row in rows),
            )
            record = _stamp_record(
                {
                    "training_id": training_id,
                    "task_index": 0,
                    "env_name": "Fixture-v0",
                    "training_seed": training_id + 1,
                    "updates": 2,
                    "population_size": 4,
                    "candidate_rollouts": 8,
                    "calibration_rollouts": 1,
                    "online_evaluation_rollouts": 0,
                    "checkpoint_generations": [5, 10],
                    "training_transitions": 50,
                    "calibration_transitions": 1,
                    "training_log_sha256": _sha(history),
                    "training_log_path": f"{relative}/history.jsonl",
                    "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                    "stderr_empty": True,
                }
            )
            records.append(record)
        stderr_root = os.path.join(self.artifact_root, "stderr/training")
        os.makedirs(stderr_root)
        for training_id in range(2):
            _write(os.path.join(stderr_root, f"training_{training_id:06d}.stderr"), b"")
        return records

    def _diagnostic_artifacts(self) -> None:
        root = os.path.join(self.artifact_root, "checkpoint_artifacts")
        stderr_root = os.path.join(self.artifact_root, "stderr/diagnostic")
        os.makedirs(root)
        os.makedirs(stderr_root)
        for checkpoint_id in range(4):
            directory = os.path.join(root, f"checkpoint_{checkpoint_id:06d}")
            os.makedirs(directory)
            _write(os.path.join(directory, "artifact.bin"), bytes([checkpoint_id]) * 7)
            _write(
                os.path.join(stderr_root, f"checkpoint_{checkpoint_id:06d}.stderr"),
                b"",
            )

    def _stage_report(self) -> str:
        provenance = _stamp_record(
            {
                **self.locks,
                "source_snapshot_path": self.snapshot_relative,
                "validator_sha256": self.validator_sha,
                "assembler_sha256": self.assembler_sha,
                "study_lock_sha256": self.study_lock_sha,
                "stderr_empty": True,
                "validation_mode": "fixture",
            }
        )
        checkpoints = []
        for checkpoint_id in range(4):
            training_id, generation_index = divmod(checkpoint_id, 2)
            checkpoints.append(
                _stamp_record(
                    {
                        "checkpoint_id": checkpoint_id,
                        "training_id": training_id,
                        "task_index": 0,
                        "env_name": "Fixture-v0",
                        "training_seed": training_id + 1,
                        "generation": [5, 10][generation_index],
                        "checkpoint_artifact_path": f"fixture_{checkpoint_id}.npz",
                        "checkpoint_artifact_sha256": "6" * 64,
                        "training_config_path": f"fixture_config_{training_id}.json",
                        "training_config_sha256": "7" * 64,
                        "capture_manifest_path": f"fixture_capture_{training_id}.json",
                        "capture_manifest_sha256": "8" * 64,
                        "strictly_prior_gradient_archive": True,
                        "reward_selection_used": False,
                    }
                )
            )
        report = {
            "schema_version": 1,
            "study": disclosure.STUDY,
            "stage": "checkpoint_generation",
            "status": "validated",
            "designation": self.manifest["designation"],
            "manifest_sha256": self.manifest_sha,
            "provenance": provenance,
            "counts": {
                "training_runs": 2,
                "checkpoints": 4,
                "training_stderr_files": 2,
            },
            "budget": {
                "checkpoint_training_candidate_rollouts": 16,
                "normalization_calibration_rollouts": 2,
                "total_policy_rollouts": 18,
                "checkpoint_training_transitions": 100,
                "normalization_calibration_transitions": 2,
                "total_environment_transitions": 102,
            },
            "training_runs": self.training_records,
            "checkpoints": checkpoints,
            "no_outcome_selection": True,
            "no_forbidden_controls": True,
        }
        report["report_sha256"] = disclosure._report_sha256(report)
        path = os.path.join(self.artifact_root, "checkpoint_stage_validation.json")
        _write_json(path, report)
        return path

    def _audit_index(self) -> str:
        provenance = _stamp_record(
            {
                **self.locks,
                "source_snapshot_path": self.snapshot_relative,
                "stderr_empty": True,
                "documented_infrastructure_failures": [],
            }
        )
        secret = {"return": "SECRET_OUTCOME"}
        value = {
            "schema_version": 1,
            "study": disclosure.STUDY,
            "designation": self.manifest["designation"],
            "manifest_sha256": self.manifest_sha,
            "provenance": provenance,
            "analysis_declaration": {"not_an_outcome": True},
            "training_runs": [secret] * 2,
            "checkpoints": [secret] * 4,
            "banks": [secret] * 8,
            "partitions": [secret] * 8,
            "checkpoint_metrics": [secret] * 4,
            "center_endpoints": [secret] * 12,
            "endpoints": [secret] * 48,
            "budget": {
                "checkpoint_training_candidate_rollouts": 16,
                "normalization_calibration_rollouts": 2,
                "bank_candidate_rollouts": 80,
                "endpoint_arm_rollouts": 48,
                "checkpoint_center_rollouts": 12,
                "total_policy_rollouts": 158,
                "checkpoint_training_transitions": 100,
                "normalization_calibration_transitions": 2,
                "bank_transitions": 160,
                "endpoint_arm_transitions": 96,
                "checkpoint_center_transitions": 24,
                "total_environment_transitions": 382,
            },
        }
        path = os.path.join(self.artifact_root, "audit_index.json")
        _write_json(path, value)
        return path

    def _slurm_logs(self) -> None:
        for prefix, job_id, stage, count in (
            ("lagged_checkpoint", "111", "checkpoint_generation", 2),
            ("lagged_diagnostic", "222", "diagnostic", 4),
        ):
            for task_id in range(count):
                if stage == "checkpoint_generation":
                    identity = (
                        f"task_id={task_id} training_id={task_id} task_index=0 "
                        f"env=Fixture-v0 seed={task_id + 1}"
                    )
                else:
                    training_id, generation_index = divmod(task_id, 2)
                    identity = (
                        f"task_id={task_id} checkpoint_id={task_id} "
                        f"training_id={training_id} task_index=0 env=Fixture-v0 "
                        f"seed={training_id + 1} generation={[5, 10][generation_index]}"
                    )
                stdout = (
                    f"study={disclosure.STUDY} stage={stage}\n"
                    f"source_sha256={self.locks['source_sha256']} snapshot={self.snapshot}\n"
                    f"{identity}\n"
                    "workers=3 dry_run=0\n"
                ).encode("ascii")
                _write(
                    os.path.join(self.logs, f"{prefix}_{job_id}_{task_id}.out"),
                    stdout,
                )
                _write(
                    os.path.join(self.logs, f"{prefix}_{job_id}_{task_id}.err"),
                    b"",
                )

    def _accounting(self, name: str, job_id: str, count: int, elapsed: int) -> str:
        header = (
            "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|"
            "ConsumedEnergyRaw|Start|End|Restarts|\n"
        )
        lines = [header]
        for task_id in range(count):
            start = f"2026-07-13T00:0{task_id}:00"
            end = f"2026-07-13T00:{task_id + 2:02d}:00"
            lines.append(
                f"{job_id}_{task_id}|COMPLETED|0:0|{elapsed + task_id}|4||"
                f"{100 + task_id}|{start}|{end}|0|\n"
            )
            lines.append(
                f"{job_id}_{task_id}.batch|COMPLETED|0:0|{elapsed}|4|"
                f"{task_id + 1}G|||||\n"
            )
        path = os.path.join(self.root, name)
        _write(path, "".join(lines).encode("ascii"))
        return path

    def collect(self, **overrides: Any) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "artifact_root": self.artifact_root,
            "manifest_path": self.manifest_path,
            "checkpoint_stage_report_path": self.stage_path,
            "expected_checkpoint_stage_sha256": self.stage_sha,
            "checkpoint_stage_validator_path": self.validator_path,
            "audit_index_path": self.audit_path,
            "expected_audit_sha256": self.audit_sha,
            "mode": "final",
            "slurm_log_root": self.logs,
            "checkpoint_job_ids": ["111"],
            "diagnostic_job_ids": ["222"],
            "checkpoint_accounting_paths": [self.checkpoint_accounting],
            "diagnostic_accounting_paths": [self.diagnostic_accounting],
            "require_production": False,
        }
        arguments.update(overrides)
        return disclosure.collect_disclosure(**arguments)


class ComputeDisclosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = DisclosureFixture()

    def tearDown(self) -> None:
        self.fixture.cleanup()

    def test_checkpoint_metric_cardinality_includes_every_q(self) -> None:
        manifest = dict(self.fixture.manifest)
        manifest["dimensions"] = dict(manifest["dimensions"])
        manifest["dimensions"]["locality_q"] = [0.25, 0.5, 1.0]
        expected = disclosure._expected_cardinalities(manifest)
        self.assertEqual(expected["checkpoints"], 4)
        self.assertEqual(expected["checkpoint_metrics"], 12)
        self.assertEqual(expected["endpoints"], 144)

    def test_budget_requires_explicit_separate_transition_semantics(self) -> None:
        manifest = dict(self.fixture.manifest)
        manifest["budget"] = dict(manifest["budget"])
        manifest["budget"]["environment_transitions_are_separate"] = False
        stage, _ = disclosure._validate_stage_report(
            self.fixture.stage_path, expected_sha256=self.fixture.stage_sha
        )
        with self.assertRaisesRegex(disclosure.DisclosureError, "transitions separate"):
            disclosure._validate_budget(
                manifest=manifest,
                stage_budget=stage["budget"],
                audit_budget=None,
            )

    def test_final_report_is_complete_hashed_and_never_decodes_outcomes(self) -> None:
        real_loads = json.loads

        def guarded_loads(value: Any, *args: Any, **kwargs: Any) -> Any:
            content = value if isinstance(value, bytes) else str(value).encode("utf-8")
            if b"SECRET_OUTCOME" in content:
                raise AssertionError("scientific outcome value was deserialized")
            return real_loads(value, *args, **kwargs)

        with mock.patch.object(disclosure.json, "loads", side_effect=guarded_loads):
            report = self.fixture.collect()
        self.assertEqual(report["report_sha256"], disclosure._report_sha256(report))
        self.assertEqual(
            report["scientific_cardinality"][
                "lexically_observed_array_element_counts"
            ],
            report["scientific_cardinality"]["expected"],
        )
        self.assertEqual(
            report["scientific_workload"]["policy_rollouts"]["total_policy_rollouts"],
            158,
        )
        self.assertEqual(
            report["scientific_workload"]["environment_transitions"][
                "total_environment_transitions"
            ],
            382,
        )
        requested = report["requested_allocations"]
        self.assertEqual(
            requested["checkpoint_generation"]["slurm_cpus_per_task"], 4
        )
        self.assertEqual(requested["diagnostic"]["memory_bytes_per_task"], 2 * 1024**3)
        self.assertEqual(requested["diagnostic"]["maximum_concurrent_tasks"], 2)
        training = report["observed_execution"]["checkpoint_generation"]
        self.assertEqual(
            training["application_runtime"]["instrumented_iteration_compute_seconds"]["sum"],
            8.0,
        )
        self.assertEqual(training["slurm_accounting"]["max_rss_bytes"]["value"], 2 * 1024**3)
        self.assertEqual(training["slurm_accounting"]["consumed_energy_joules"]["value"], 201)
        self.assertEqual(
            training["slurm_accounting"]["same_job_requeue_or_restart_count"],
            {
                "value": 0,
                "observed_task_records": 2,
                "source_fields": ["Restarts"],
            },
        )
        self.assertIsNone(training["slurm_accounting"]["co2e_kg"]["value"])
        self.assertEqual(report["locked_runtime"]["python"], "3.10.18")
        self.assertEqual(report["locked_runtime"]["distributions"]["numpy"], "1.26.4")
        self.assertGreater(report["storage"]["artifact_root"]["total_logical_bytes"], 0)
        self.assertNotIn("SECRET_OUTCOME", disclosure._canonical_bytes(report).decode())

    def test_unavailable_accounting_is_explicit_and_not_inferred(self) -> None:
        report = self.fixture.collect(
            slurm_log_root=None,
            checkpoint_job_ids=[],
            diagnostic_job_ids=[],
            checkpoint_accounting_paths=[],
            diagnostic_accounting_paths=[],
        )
        for stage in ("checkpoint_generation", "diagnostic"):
            accounting = report["observed_execution"][stage]["slurm_accounting"]
            self.assertFalse(accounting["available"])
            for field in ("max_rss_bytes", "consumed_energy_joules", "co2e_kg"):
                self.assertIsNone(accounting[field]["value"])
                self.assertTrue(accounting[field]["reason"])
            failures = report["failures_and_retries"][stage]
            self.assertIsNone(
                failures[
                    "accounted_failed_or_incomplete_distinct_job_task_records"
                ]
            )
            self.assertTrue(failures["scheduler_failure_claim_reason"])

    def test_checkpoint_stage_mode_reports_only_validated_stage_workload(self) -> None:
        report = self.fixture.collect(
            mode="checkpoint-stage",
            audit_index_path=None,
            expected_audit_sha256=None,
            slurm_log_root=None,
            checkpoint_job_ids=[],
            diagnostic_job_ids=[],
            checkpoint_accounting_paths=[],
            diagnostic_accounting_paths=[],
        )
        self.assertFalse(
            report["scientific_cardinality"]["final_lexical_counts_match_manifest"]
        )
        self.assertEqual(
            report["scientific_workload"]["scope"],
            "checkpoint_generation_only",
        )
        self.assertEqual(
            report["scientific_workload"]["policy_rollouts"],
            {
                "checkpoint_training_candidate_rollouts": 16,
                "normalization_calibration_rollouts": 2,
                "total_policy_rollouts": 18,
            },
        )
        diagnostic = report["observed_execution"]["diagnostic"]
        self.assertFalse(diagnostic["slurm_accounting"]["available"])
        self.assertEqual(
            diagnostic["slurm_accounting"]["reason"],
            "diagnostic_stage_not_included",
        )

    def test_final_mode_rejects_partial_scientific_cardinality(self) -> None:
        with open(self.fixture.audit_path, encoding="utf-8") as stream:
            audit = json.load(stream)
        audit["endpoints"].pop()
        _write_json(self.fixture.audit_path, audit)
        with self.assertRaisesRegex(disclosure.DisclosureError, "cardinality is partial"):
            self.fixture.collect(expected_audit_sha256=_sha(self.fixture.audit_path))

    def test_final_mode_requires_committed_audit_digest(self) -> None:
        with open(self.fixture.audit_path, "rb") as stream:
            payload = stream.read()
        self.assertIn(b"SECRET_OUTCOME", payload)
        _write(
            self.fixture.audit_path,
            payload.replace(b'"SECRET_OUTCOME"', b"null"),
        )
        with self.assertRaisesRegex(disclosure.DisclosureError, "committed digest"):
            self.fixture.collect()

    def test_all_modes_require_committed_checkpoint_stage_digest(self) -> None:
        with self.assertRaisesRegex(disclosure.DisclosureError, "checkpoint-stage"):
            self.fixture.collect(expected_checkpoint_stage_sha256="9" * 64)

        with open(self.fixture.stage_path, encoding="utf-8") as stream:
            stage = json.load(stream)
        stage["counts"]["training_stderr_files"] = 99
        stage["report_sha256"] = disclosure._report_sha256(stage)
        _write_json(self.fixture.stage_path, stage)
        with self.assertRaisesRegex(disclosure.DisclosureError, "committed digest"):
            self.fixture.collect()

    def test_composite_source_and_direct_stage_tools_are_revalidated(self) -> None:
        source_path = os.path.join(self.fixture.snapshot, "core/policies.py")
        with open(source_path, "ab") as stream:
            stream.write(b"# mutation\n")
        with self.assertRaisesRegex(disclosure.DisclosureError, "source snapshot digest"):
            self.fixture.collect()

    def test_validator_digest_is_checked_even_outside_source_inventory(self) -> None:
        validator = os.path.join(
            self.fixture.snapshot,
            disclosure.CHECKPOINT_STAGE_VALIDATOR,
        )
        with open(validator, "ab") as stream:
            stream.write(b"# mutation\n")
        with self.assertRaisesRegex(disclosure.DisclosureError, "stage-report digest"):
            self.fixture.collect()

    def test_production_mode_pins_exact_manifest_and_every_cardinality(self) -> None:
        repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        production_path = os.path.join(repository_root, disclosure.MANIFEST_RELATIVE)
        production = disclosure._validate_manifest(
            production_path,
            expected_sha256=disclosure.PRODUCTION_MANIFEST_SHA256,
            require_production=True,
        )
        self.assertEqual(
            disclosure._expected_cardinalities(production),
            disclosure.PRODUCTION_CARDINALITIES,
        )

        changed = dict(production)
        changed["dimensions"] = dict(production["dimensions"])
        changed["dimensions"]["banks"] = ["A"]
        changed_path = os.path.join(self.fixture.root, "changed_production.json")
        _write_json(changed_path, changed)
        with self.assertRaisesRegex(disclosure.DisclosureError, "exact immutable"):
            disclosure._validate_manifest(
                changed_path,
                expected_sha256=_sha(changed_path),
                require_production=True,
            )

    def test_nonempty_infrastructure_list_is_skipped_not_deserialized(self) -> None:
        with open(self.fixture.audit_path, encoding="utf-8") as stream:
            audit = json.load(stream)
        audit["provenance"]["documented_infrastructure_failures"] = [
            {"scientific_return": "SECRET_OUTCOME"}
        ]
        audit["provenance"]["record_sha256"] = disclosure._record_sha256(
            audit["provenance"]
        )
        _write_json(self.fixture.audit_path, audit)
        real_loads = json.loads

        def guarded_loads(value: Any, *args: Any, **kwargs: Any) -> Any:
            content = value if isinstance(value, bytes) else str(value).encode("utf-8")
            if b"SECRET_OUTCOME" in content:
                raise AssertionError("failure-list outcome was deserialized")
            return real_loads(value, *args, **kwargs)

        with mock.patch.object(disclosure.json, "loads", side_effect=guarded_loads):
            with self.assertRaisesRegex(
                disclosure.DisclosureError, "must be an empty array"
            ):
                self.fixture.collect(
                    expected_audit_sha256=_sha(self.fixture.audit_path)
                )

    def test_stage_report_rejects_unexpected_checkpoint_outcome_field(self) -> None:
        with open(self.fixture.stage_path, encoding="utf-8") as stream:
            stage = json.load(stream)
        stage["checkpoints"][0]["scientific_return"] = 123.0
        stage["checkpoints"][0]["record_sha256"] = disclosure._record_sha256(
            stage["checkpoints"][0]
        )
        stage["report_sha256"] = disclosure._report_sha256(stage)
        _write_json(self.fixture.stage_path, stage)
        with self.assertRaisesRegex(disclosure.DisclosureError, "schema is not exact"):
            self.fixture.collect(
                expected_checkpoint_stage_sha256=_sha(self.fixture.stage_path)
            )

    def test_scanner_rejects_invalid_escape_utf8_and_nonfinite_constant(self) -> None:
        for value in (b'"bad\\q"', b'"bad\xff"'):
            with self.subTest(value=value):
                with self.assertRaises(disclosure.DisclosureError):
                    disclosure._skip_value(value, 0, len(value))
        with self.assertRaisesRegex(disclosure.DisclosureError, "nonfinite"):
            disclosure._decode_slice(b"NaN", 0, 3, "fixture")

    def test_slurm_memory_context_and_gpu_directives_are_fail_closed(self) -> None:
        self.assertEqual(
            disclosure._parse_memory_bytes("32000", default_unit="M"),
            32000 * 1024**2,
        )
        self.assertIsNone(disclosure._parse_maxrss("32000"))
        launcher = os.path.join(self.fixture.root, "gpu_launcher.sh")
        source = self.fixture._launcher(
            "gpu", "gpu", 2, 2, "32000", "01:00:00", False
        ).replace(
            "#SBATCH --cpus-per-task=4\n",
            "#SBATCH --cpus-per-task=4\n#SBATCH --gpus-per-node=1\n",
        )
        _write(launcher, source.encode("ascii"))
        with self.assertRaisesRegex(disclosure.DisclosureError, "GPU allocation"):
            disclosure._parse_launcher(launcher, expected_tasks=2)

    def test_zero_cost_failed_accounting_attempt_is_preserved(self) -> None:
        failed = os.path.join(self.fixture.root, "failed_retry.sacct")
        _write(
            failed,
            (
                "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|"
                "ConsumedEnergyRaw|Start|End|Restarts|\n"
                "110_0|CANCELLED|1:0|0|0|||||1|\n"
            ).encode("ascii"),
        )
        accounting, attempts = disclosure._collect_accounting(
            [failed, self.fixture.checkpoint_accounting],
            job_ids=["110", "111"],
            expected_tasks=2,
            log_attempts=set(),
        )
        self.assertEqual(len(attempts), 3)
        self.assertEqual(accounting["distinct_job_task_records"], 3)
        self.assertEqual(accounting["distinct_job_resubmission_records"], 1)
        self.assertEqual(
            accounting["failed_or_incomplete_distinct_job_task_records"], 1
        )
        self.assertEqual(accounting["elapsed_seconds"]["minimum"], 0.0)
        self.assertIsNone(
            accounting["maximum_observed_concurrent_tasks"]["value"]
        )

    def test_orphan_accounting_step_is_rejected(self) -> None:
        orphan = os.path.join(self.fixture.root, "orphan_step.sacct")
        _write(
            orphan,
            (
                "JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|MaxRSS|"
                "ConsumedEnergyRaw|Start|End|Restarts|\n"
                "110_0|CANCELLED|1:0|0|0|||||0|\n"
                "110_1.batch|FAILED|1:0|1|4|1G|||||\n"
            ).encode("ascii"),
        )
        with self.assertRaisesRegex(disclosure.DisclosureError, "without matching root"):
            disclosure._collect_accounting(
                [orphan, self.fixture.checkpoint_accounting],
                job_ids=["110", "111"],
                expected_tasks=2,
                log_attempts=set(),
            )

    def test_zero_length_accounting_interval_cannot_claim_concurrency(self) -> None:
        moment = disclosure.dt.datetime(2026, 7, 13)
        self.assertIsNone(
            disclosure._observed_concurrency(
                [{"start": moment, "end": moment, "alloc_cpus": 4}]
            )
        )

    def test_hash_validated_training_runtime_rejects_mutation(self) -> None:
        path = os.path.join(
            self.fixture.artifact_root,
            self.fixture.training_records[0]["training_log_path"],
        )
        with open(path, "ab") as stream:
            stream.write(b" \n")
        with self.assertRaisesRegex(disclosure.DisclosureError, "stage-report hash"):
            self.fixture.collect()

    def test_partial_slurm_logs_fail_closed_when_log_root_is_supplied(self) -> None:
        os.unlink(os.path.join(self.fixture.logs, "lagged_diagnostic_222_3.err"))
        with self.assertRaisesRegex(disclosure.DisclosureError, "attempt pair is incomplete"):
            self.fixture.collect()

    def test_slurm_log_wrong_source_or_dry_run_is_not_identity_valid(self) -> None:
        path = os.path.join(self.fixture.logs, "lagged_checkpoint_111_0.out")
        with open(path, "rb") as stream:
            payload = stream.read()
        payload = payload.replace(
            b"source_sha256=" + self.fixture.locks["source_sha256"].encode("ascii"),
            b"source_sha256=" + b"9" * 64,
        )
        payload = payload.replace(b"dry_run=0", b"dry_run=1")
        _write(path, payload)
        with self.assertRaisesRegex(disclosure.DisclosureError, "valid launcher identity"):
            self.fixture.collect()

    def test_failed_retry_log_may_precede_identity_valid_success(self) -> None:
        _write(
            os.path.join(self.fixture.logs, "lagged_checkpoint_110_0.out"), b""
        )
        _write(
            os.path.join(self.fixture.logs, "lagged_checkpoint_110_0.err"),
            b"runtime gate failed\n",
        )
        report = self.fixture.collect(
            checkpoint_job_ids=["110", "111"],
            checkpoint_accounting_paths=[],
        )
        logs = report["observed_execution"]["checkpoint_generation"]["slurm_logs"]
        self.assertEqual(logs["distinct_job_task_records"], 3)
        self.assertEqual(logs["distinct_job_resubmission_records"], 1)
        self.assertEqual(logs["launcher_metadata_incomplete_or_invalid_attempts"], 1)
        self.assertEqual(logs["nonempty_stderr_attempts"], 1)
        failures = report["failures_and_retries"]["checkpoint_generation"]
        self.assertTrue(
            failures["upstream_validated_scientific_completion"]["value"]
        )
        self.assertIsNone(
            failures["upstream_validated_included_stage_failures"]["value"]
        )
        self.assertIn(
            "does not enumerate scheduler attempts",
            failures["upstream_validated_included_stage_failures"]["reason"],
        )

    def test_atomic_output_is_canonical_and_refuses_overwrite(self) -> None:
        report = self.fixture.collect(slurm_log_root=None)
        output = os.path.join(self.fixture.root, "disclosure.json")
        disclosure._atomic_json_write_once(output, report)
        with open(output, "rb") as stream:
            self.assertEqual(stream.read(), disclosure._canonical_bytes(report) + b"\n")
        with self.assertRaisesRegex(disclosure.DisclosureError, "refuses overwrite"):
            disclosure._atomic_json_write_once(output, report)

    def test_file_measurement_rejects_mutation_during_hashing(self) -> None:
        path = os.path.join(self.fixture.root, "racing_input.txt")
        _write(path, b"before\n")
        real_sha256_file = disclosure._sha256_file

        def mutate_after_hash(selected: str) -> str:
            digest = real_sha256_file(selected)
            with open(selected, "ab") as stream:
                stream.write(b"after\n")
            return digest

        with mock.patch.object(
            disclosure, "_sha256_file", side_effect=mutate_after_hash
        ):
            with self.assertRaisesRegex(disclosure.DisclosureError, "changed"):
                disclosure._stable_file_measurement(path, "racing fixture")

    def test_cli_writes_canonical_final_report_and_refuses_overwrite(self) -> None:
        command = [
            sys.executable,
            "scripts/collect_lagged_subspace_compute_disclosure.py",
            "--artifact-root",
            self.fixture.artifact_root,
            "--manifest",
            self.fixture.manifest_path,
            "--checkpoint-stage-report",
            self.fixture.stage_path,
            "--expected-checkpoint-stage-sha256",
            self.fixture.stage_sha,
            "--checkpoint-stage-validator",
            self.fixture.validator_path,
            "--audit-index",
            self.fixture.audit_path,
            "--expected-audit-sha256",
            self.fixture.audit_sha,
            "--slurm-log-root",
            self.fixture.logs,
            "--checkpoint-job-id",
            "111",
            "--diagnostic-job-id",
            "222",
            "--checkpoint-accounting",
            self.fixture.checkpoint_accounting,
            "--diagnostic-accounting",
            self.fixture.diagnostic_accounting,
            "--output",
            "compute_disclosure.json",
            "--fixture-mode",
        ]
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = os.path.join(
            self.fixture.artifact_root, "compute_disclosure.json"
        )
        with open(output, "rb") as stream:
            serialized = stream.read()
        report = disclosure._read_json(output, "fixture disclosure")
        self.assertEqual(serialized, disclosure._canonical_bytes(report) + b"\n")
        self.assertEqual(report["report_sha256"], disclosure._report_sha256(report))

        repeated = subprocess.run(
            command, check=False, capture_output=True, text=True
        )
        self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()

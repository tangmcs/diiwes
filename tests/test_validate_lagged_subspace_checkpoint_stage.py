"""Focused fail-closed tests for the checkpoint-stage validator."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Callable
from unittest import mock

from scripts import assemble_lagged_subspace_frozen_checkpoint as assembler
from scripts.validate_lagged_subspace_checkpoint_stage import (
    AssemblyError,
    _report_sha256,
    validate_checkpoint_stage,
)

try:
    from tests.test_assemble_lagged_subspace_frozen_checkpoint import (
        _build_fixture,
        _write_json,
    )
except ModuleNotFoundError as error:
    if error.name not in {
        "tests",
        "tests.test_assemble_lagged_subspace_frozen_checkpoint",
    }:
        raise
    from test_assemble_lagged_subspace_frozen_checkpoint import (  # type: ignore
        _build_fixture,
        _write_json,
    )


class CheckpointStageValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = tempfile.TemporaryDirectory()
        cls.manifest, cls.hashes, cls.launcher, cls.dependency = _build_fixture(
            cls.base.name
        )
        cls.manifest_path = os.path.join(cls.base.name, "fixture_manifest.json")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.base.cleanup()

    def _copy(self) -> tempfile.TemporaryDirectory:
        target = tempfile.TemporaryDirectory()
        shutil.copytree(self.base.name, target.name, dirs_exist_ok=True)
        return target

    def _validate(
        self,
        root: str | None = None,
        *,
        absolute_snapshot: bool = False,
    ) -> dict[str, Any]:
        root = self.base.name if root is None else root
        snapshot = (
            os.path.join(root, "source_snapshot")
            if absolute_snapshot
            else "source_snapshot"
        )
        return validate_checkpoint_stage(
            artifact_root=root,
            manifest_path=os.path.join(root, "fixture_manifest.json"),
            source_snapshot_path=snapshot,
            launcher_lock_path=os.path.join(root, "locks", "launcher.sh"),
            dependency_lock_path=os.path.join(root, "locks", "requirements.lock"),
            expected_hashes=self.hashes,
            require_preregistered_manifest=False,
        )

    def _command(self, root: str, output: str) -> list[str]:
        return [
            sys.executable,
            "scripts/validate_lagged_subspace_checkpoint_stage.py",
            "--artifact-root",
            root,
            "--manifest",
            os.path.join(root, "fixture_manifest.json"),
            "--source-snapshot",
            os.path.join(root, "source_snapshot"),
            "--launcher-lock",
            os.path.join(root, "locks", "launcher.sh"),
            "--dependency-lock",
            os.path.join(root, "locks", "requirements.lock"),
            "--expected-source-sha256",
            self.hashes["source_sha256"],
            "--expected-manifest-sha256",
            self.hashes["manifest_sha256"],
            "--expected-protocol-sha256",
            self.hashes["protocol_sha256"],
            "--expected-analyzer-sha256",
            self.hashes["analyzer_sha256"],
            "--expected-launcher-sha256",
            self.hashes["launcher_sha256"],
            "--expected-dependency-lock-sha256",
            self.hashes["dependency_lock_sha256"],
            "--output",
            output,
            "--fixture-mode",
        ]

    def test_report_is_deterministic_complete_and_hashed(self) -> None:
        relative = self._validate()
        absolute = self._validate(absolute_snapshot=True)
        self.assertEqual(relative, absolute)
        training_count = len(self.manifest["tasks"]) * len(
            self.manifest["training_seeds"]
        )
        checkpoint_count = training_count * len(
            self.manifest["checkpoint_generations"]
        )
        self.assertEqual(relative["status"], "validated")
        self.assertEqual(relative["counts"]["training_runs"], training_count)
        self.assertEqual(relative["counts"]["checkpoints"], checkpoint_count)
        self.assertEqual(len(relative["training_runs"]), training_count)
        self.assertEqual(len(relative["checkpoints"]), checkpoint_count)
        self.assertEqual(
            [record["checkpoint_id"] for record in relative["checkpoints"]],
            list(range(checkpoint_count)),
        )
        self.assertTrue(
            all(record["stderr_empty"] for record in relative["training_runs"])
        )
        self.assertTrue(
            all(
                record["strictly_prior_gradient_archive"]
                and not record["reward_selection_used"]
                for record in relative["checkpoints"]
            )
        )
        self.assertEqual(relative["report_sha256"], _report_sha256(relative))
        self.assertEqual(relative["provenance"]["validation_mode"], "fixture")

    def test_cli_writes_canonical_report_atomically_and_refuses_overwrite(
        self,
    ) -> None:
        copied = self._copy()
        try:
            output_relative = "validation/checkpoint_stage.json"
            command = self._command(copied.name, output_relative)
            completed = subprocess.run(
                command,
                cwd=assembler.analyzer.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output_path = os.path.join(copied.name, output_relative)
            with open(output_path, "rb") as stream:
                serialized = stream.read()
            with open(output_path, encoding="utf-8") as stream:
                report = json.load(stream)
            self.assertEqual(
                serialized, assembler._canonical_bytes(report) + b"\n"
            )
            self.assertEqual(report["report_sha256"], _report_sha256(report))
            original_sha = assembler._sha256_file(output_path)

            repeated = subprocess.run(
                command,
                cwd=assembler.analyzer.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(assembler._sha256_file(output_path), original_sha)
        finally:
            copied.cleanup()

    def test_discrepancies_fail_closed(self) -> None:
        def remove_training(root: str) -> None:
            shutil.rmtree(os.path.join(root, "training_runs", "training_000000"))

        def add_training(root: str) -> None:
            os.makedirs(os.path.join(root, "training_runs", "unexpected"))

        def symlink_training(root: str) -> None:
            path = os.path.join(root, "training_runs", "training_000000")
            shutil.rmtree(path)
            os.symlink("training_000001", path)

        def nonempty_stderr(root: str) -> None:
            with open(
                os.path.join(root, "stderr", "training", "training_000000.stderr"),
                "wb",
            ) as stream:
                stream.write(b"warning\n")

        def mixed_lock(root: str) -> None:
            path = os.path.join(
                root, "training_runs", "training_000000", "config.json"
            )
            with open(path, encoding="utf-8") as stream:
                value = json.load(stream)
            value["provenance"]["source_sha256"] = "0" * 64
            _write_json(path, value)

        def reward_selected(root: str) -> None:
            path = os.path.join(
                root, "training_runs", "training_000000", "checkpoint_capture.json"
            )
            with open(path, encoding="utf-8") as stream:
                value = json.load(stream)
            value["reward_selection_used"] = True
            _write_json(path, value)

        def incomplete(root: str) -> None:
            path = os.path.join(
                root, "training_runs", "training_000000", "status.json"
            )
            with open(path, encoding="utf-8") as stream:
                value = json.load(stream)
            value["status"] = "running"
            _write_json(path, value)

        def missing_checkpoint(root: str) -> None:
            generation = self.manifest["checkpoint_generations"][0]
            os.unlink(
                os.path.join(
                    root,
                    "training_runs",
                    "training_000000",
                    "checkpoints",
                    f"checkpoint_generation_{generation:06d}.npz",
                )
            )

        def forbidden_control(root: str) -> None:
            path = os.path.join(
                root, "training_runs", "training_000000", "config.json"
            )
            with open(path, encoding="utf-8") as stream:
                value = json.load(stream)
            value["trust_radius"] = 1.0
            _write_json(path, value)

        def corrupt_history(root: str) -> None:
            run = os.path.join(root, "training_runs", "training_000000")
            path = os.path.join(run, "history.json")
            with open(path, encoding="utf-8") as stream:
                value = json.load(stream)
            value[0]["n_reused"] = 1
            _write_json(path, value)
            with open(
                os.path.join(run, "history.jsonl"), "w", encoding="utf-8"
            ) as stream:
                for row in value:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")

        cases: list[tuple[str, Callable[[str], None]]] = [
            ("missing training directory", remove_training),
            ("extra training directory", add_training),
            ("symlinked training directory", symlink_training),
            ("nonempty stderr", nonempty_stderr),
            ("mixed source lock", mixed_lock),
            ("reward-selected checkpoint", reward_selected),
            ("incomplete run", incomplete),
            ("missing checkpoint", missing_checkpoint),
            ("forbidden trust control", forbidden_control),
            ("corrupt history", corrupt_history),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                copied = self._copy()
                try:
                    mutate(copied.name)
                    with self.assertRaises(AssemblyError):
                        self._validate(copied.name)
                finally:
                    copied.cleanup()

    def test_cli_discrepancy_exits_nonzero_without_committing_report(self) -> None:
        copied = self._copy()
        try:
            status_path = os.path.join(
                copied.name, "training_runs", "training_000000", "status.json"
            )
            with open(status_path, encoding="utf-8") as stream:
                status = json.load(stream)
            status["completed_iterations"] -= 1
            _write_json(status_path, status)
            output_relative = "checkpoint_stage_validation.json"
            completed = subprocess.run(
                self._command(copied.name, output_relative),
                cwd=assembler.analyzer.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(
                os.path.lexists(os.path.join(copied.name, output_relative))
            )
            self.assertFalse(
                any(
                    name.startswith(".checkpoint_stage_validation_")
                    for name in os.listdir(copied.name)
                )
            )
        finally:
            copied.cleanup()

    def test_production_mode_requires_exact_60_by_180_design(self) -> None:
        with mock.patch(
            "scripts.validate_lagged_subspace_checkpoint_stage._validate_provenance",
            return_value=(
                self.manifest,
                self.hashes["manifest_sha256"],
                "source_snapshot",
                "a" * 64,
                "b" * 64,
            ),
        ):
            with self.assertRaisesRegex(AssemblyError, "exactly 60.*180"):
                validate_checkpoint_stage(
                    artifact_root=self.base.name,
                    manifest_path=self.manifest_path,
                    source_snapshot_path="source_snapshot",
                    launcher_lock_path=self.launcher,
                    dependency_lock_path=self.dependency,
                    expected_hashes=self.hashes,
                    require_preregistered_manifest=True,
                )


if __name__ == "__main__":
    unittest.main()

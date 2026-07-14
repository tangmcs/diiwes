"""Source-lock and array-mapping tests for the frozen-checkpoint study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments import run_lagged_subspace_checkpoint_diagnostic as producer
from experiments.lagged_subspace_study_lock import (
    CHECKPOINT_TASK_COUNT,
    DEPENDENCY_BUNDLE_PATH,
    LAUNCHER_BUNDLE_PATH,
    RUNTIME_DEPENDENCY_VERSIONS,
    STUDY_CONFIG_PATHS,
    STUDY_SOURCE_PATHS,
    TRAINING_TASK_COUNT,
    StudySourceLockError,
    checkpoint_coordinates,
    compute_lagged_subspace_study_sha256,
    current_lagged_subspace_study_sha256,
    require_checkpoint_generation_provenance_locks,
    require_lagged_subspace_study_source_lock,
    study_sha256_for_checkpoint_config,
    training_coordinates,
    validate_hash_bundle,
    validate_manifest_mapping,
    validate_runtime_inventory,
)
from experiments.train import train


REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_locked_snapshot(destination: str) -> None:
    for relative in STUDY_SOURCE_PATHS:
        source = os.path.join(REPOSITORY_ROOT, relative)
        target = os.path.join(destination, relative)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copyfile(source, target)


class LaggedSubspaceStudyLockTests(unittest.TestCase):
    def test_all_three_configs_share_one_composite_digest(self) -> None:
        expected = current_lagged_subspace_study_sha256()
        found = {
            study_sha256_for_checkpoint_config(
                os.path.join(REPOSITORY_ROOT, relative)
            )
            for relative in STUDY_CONFIG_PATHS
        }
        self.assertEqual(found, {expected})
        with self.assertRaisesRegex(StudySourceLockError, "three locked"):
            study_sha256_for_checkpoint_config(
                os.path.join(REPOSITORY_ROOT, "README.md")
            )
        with self.assertRaisesRegex(StudySourceLockError, "environment"):
            study_sha256_for_checkpoint_config(
                os.path.join(REPOSITORY_ROOT, STUDY_CONFIG_PATHS[0]),
                expected_env_name="Hopper-v5",
            )

    def test_each_listed_file_and_config_mutation_changes_digest(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            _copy_locked_snapshot(snapshot)
            baseline = compute_lagged_subspace_study_sha256(snapshot)
            for relative in STUDY_SOURCE_PATHS:
                with self.subTest(relative=relative):
                    changed = os.path.join(snapshot, relative)
                    with open(changed, "rb") as stream:
                        original = stream.read()
                    with open(changed, "ab") as stream:
                        stream.write(b"\n# source-lock mutation probe\n")
                    self.assertNotEqual(
                        compute_lagged_subspace_study_sha256(snapshot),
                        baseline,
                    )
                    with open(changed, "wb") as stream:
                        stream.write(original)
                    self.assertEqual(
                        compute_lagged_subspace_study_sha256(snapshot),
                        baseline,
                    )

    def test_missing_symlinked_and_unlisted_runtime_files_fail(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            _copy_locked_snapshot(snapshot)
            missing = os.path.join(snapshot, STUDY_CONFIG_PATHS[0])
            os.unlink(missing)
            with self.assertRaisesRegex(StudySourceLockError, "missing"):
                compute_lagged_subspace_study_sha256(snapshot)
            os.symlink(
                os.path.join(REPOSITORY_ROOT, STUDY_CONFIG_PATHS[0]), missing
            )
            with self.assertRaisesRegex(StudySourceLockError, "symlink|escapes"):
                compute_lagged_subspace_study_sha256(snapshot)
        with self.assertRaisesRegex(StudySourceLockError, "absent"):
            validate_runtime_inventory(("README.md",))
        with tempfile.TemporaryDirectory() as snapshot:
            _copy_locked_snapshot(snapshot)
            unlisted = os.path.join(
                snapshot, "experiments", "unlisted_runtime_probe.py"
            )
            with open(unlisted, "w", encoding="utf-8") as stream:
                stream.write("VALUE = 1\n")
            importer = os.path.join(snapshot, "experiments", "train.py")
            with open(importer, "a", encoding="utf-8") as stream:
                stream.write("\nimport experiments.unlisted_runtime_probe\n")
            with self.assertRaisesRegex(
                StudySourceLockError, "imports unlisted runtime file"
            ):
                compute_lagged_subspace_study_sha256(snapshot)

    def test_launcher_and_dependency_bundles_lock_every_named_file(self) -> None:
        for relative, kind in (
            (LAUNCHER_BUNDLE_PATH, "launchers"),
            (DEPENDENCY_BUNDLE_PATH, "dependency_locks"),
        ):
            with self.subTest(kind=kind):
                expected = _sha256_file(os.path.join(REPOSITORY_ROOT, relative))
                self.assertEqual(
                    validate_hash_bundle(
                        REPOSITORY_ROOT,
                        relative,
                        expected_bundle_sha256=expected,
                        expected_kind=kind,
                    ),
                    expected,
                )

    def test_launchers_activate_pinned_interpreter_before_python(self) -> None:
        for relative in (
            "scripts/submit_lagged_subspace_checkpoint_generation.sh",
            "scripts/submit_lagged_subspace_diagnostic.sh",
        ):
            with self.subTest(relative=relative):
                with open(
                    os.path.join(REPOSITORY_ROOT, relative), encoding="utf-8"
                ) as stream:
                    source = stream.read()
                activation = source.index(
                    "source /hpc/home/rt239/miniconda3/bin/activate es_parallel"
                )
                first_python = source.index("python -m experiments")
                self.assertLess(activation, first_python)
                for setting in (
                    "export OMP_NUM_THREADS=1",
                    "export MKL_NUM_THREADS=1",
                    "export OPENBLAS_NUM_THREADS=1",
                    "export PYTHONHASHSEED=0",
                    "export PYTHONDONTWRITEBYTECODE=1",
                ):
                    self.assertLess(source.index(setting), first_python)
        with open(
            os.path.join(REPOSITORY_ROOT, "environment.yml"), encoding="utf-8"
        ) as stream:
            environment_source = stream.read()
        self.assertIn("python=3.10.18", environment_source)
        for distribution, version in RUNTIME_DEPENDENCY_VERSIONS.items():
            if distribution == "pip":
                self.assertIn(f"pip={version}", environment_source)
            else:
                self.assertIn(f"{distribution}=={version}", environment_source)

    def test_launcher_arrays_resources_and_paths_are_fixed(self) -> None:
        expected = {
            "scripts/submit_lagged_subspace_checkpoint_generation.sh": (
                "#SBATCH --array=0-59%6",
                "#SBATCH --mem=32G",
                "training-map \"$TASK_ID\"",
                "training_runs/training_%06d",
                "stderr/training/training_%06d.stderr",
            ),
            "scripts/submit_lagged_subspace_diagnostic.sh": (
                "#SBATCH --array=0-179%6",
                "#SBATCH --mem=64G",
                "checkpoint-map \"$TASK_ID\"",
                "checkpoint_artifacts",
                "stderr/diagnostic/checkpoint_%06d.stderr",
            ),
        }
        for relative, required in expected.items():
            with self.subTest(relative=relative):
                with open(
                    os.path.join(REPOSITORY_ROOT, relative), encoding="utf-8"
                ) as stream:
                    source = stream.read()
                for text in (
                    "#SBATCH --cpus-per-task=32",
                    "#SBATCH --time=24:00:00",
                    "WORKERS=30",
                    "PAPER_EXPECTED_SOURCE_SHA",
                    "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256",
                    "EXPECTED_SNAPSHOT=",
                    *required,
                ):
                    self.assertIn(text, source)

    def test_array_mappings_are_complete_bijections(self) -> None:
        validate_manifest_mapping(REPOSITORY_ROOT)
        training = [
            training_coordinates(index) for index in range(TRAINING_TASK_COUNT)
        ]
        checkpoints = [
            checkpoint_coordinates(index) for index in range(CHECKPOINT_TASK_COUNT)
        ]
        self.assertEqual(TRAINING_TASK_COUNT, 60)
        self.assertEqual(CHECKPOINT_TASK_COUNT, 180)
        self.assertEqual(len(set(training)), 60)
        self.assertEqual(len(set(checkpoints)), 180)
        self.assertEqual({row[3] for row in training}, set(range(300, 320)))
        self.assertEqual({row[5] for row in checkpoints}, {50, 150, 250})
        with self.assertRaises(StudySourceLockError):
            training_coordinates(60)
        with self.assertRaises(StudySourceLockError):
            checkpoint_coordinates(180)

    def test_manifest_rejects_draft_status_and_stale_protocol_digest(self) -> None:
        with tempfile.TemporaryDirectory() as snapshot:
            _copy_locked_snapshot(snapshot)
            manifest_path = os.path.join(
                snapshot,
                "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
            )
            with open(manifest_path, encoding="utf-8") as stream:
                manifest = json.load(stream)
            manifest["protocol_status"] = "draft_locked_before_environment_outcomes"
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(StudySourceLockError, "stale"):
                validate_manifest_mapping(snapshot)
            manifest["protocol_status"] = "final_locked_before_environment_outcomes"
            manifest["protocol"]["sha256"] = "0" * 64
            with open(manifest_path, "w", encoding="utf-8") as stream:
                json.dump(manifest, stream)
            with self.assertRaisesRegex(StudySourceLockError, "stale"):
                validate_manifest_mapping(snapshot)

    def test_environment_lock_is_mandatory_and_exact(self) -> None:
        actual = current_lagged_subspace_study_sha256()
        with self.assertRaisesRegex(StudySourceLockError, "mandatory"):
            require_lagged_subspace_study_source_lock(environ={})
        with self.assertRaisesRegex(StudySourceLockError, "mismatch"):
            require_lagged_subspace_study_source_lock(
                environ={"PAPER_EXPECTED_SOURCE_SHA": "0" * 64}
            )
        self.assertEqual(
            require_lagged_subspace_study_source_lock(
                environ={"PAPER_EXPECTED_SOURCE_SHA": actual}
            ),
            actual,
        )

    def test_checkpoint_provenance_bundle_is_mandatory_and_exact(self) -> None:
        paths = {
            "PAPER_EXPECTED_MANIFEST_SHA256": (
                "experiments/manifests/lagged_subspace_frozen_checkpoint.json"
            ),
            "PAPER_EXPECTED_PROTOCOL_SHA256": (
                "docs/lagged_subspace_frozen_checkpoint_protocol.md"
            ),
            "PAPER_EXPECTED_ANALYZER_SHA256": (
                "scripts/analyze_lagged_subspace_frozen_checkpoint.py"
            ),
            "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256": LAUNCHER_BUNDLE_PATH,
            "PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256": DEPENDENCY_BUNDLE_PATH,
        }
        environment = {
            "PAPER_EXPECTED_SOURCE_SHA": current_lagged_subspace_study_sha256(),
            **{
                variable: _sha256_file(os.path.join(REPOSITORY_ROOT, relative))
                for variable, relative in paths.items()
            },
        }
        locks = require_checkpoint_generation_provenance_locks(
            snapshot_root=REPOSITORY_ROOT, environ=environment
        )
        self.assertEqual(locks["source_sha256"], environment["PAPER_EXPECTED_SOURCE_SHA"])
        corrupted = dict(environment)
        corrupted["PAPER_EXPECTED_MANIFEST_SHA256"] = "0" * 64
        with self.assertRaisesRegex(StudySourceLockError, "manifest_sha256 mismatch"):
            require_checkpoint_generation_provenance_locks(
                snapshot_root=REPOSITORY_ROOT, environ=corrupted
            )

    def test_checkpoint_mismatch_stops_before_environment_or_pool(self) -> None:
        config = {
            "checkpoint_capture_protocol": "lagged_subspace_frozen_checkpoint_v1",
            "checkpoint_capture_generations": [10],
            "checkpoint_gradient_archive_length": 10,
            "n_iterations": 10,
        }
        with tempfile.TemporaryDirectory() as output, patch.dict(
            os.environ,
            {"PAPER_EXPECTED_SOURCE_SHA": "0" * 64},
        ), patch(
            "experiments.train._validate_lagged_subspace_checkpoint_protocol"
        ), patch("experiments.train._make_env") as make_env, patch(
            "experiments.train.Pool"
        ) as pool:
            with self.assertRaisesRegex(StudySourceLockError, "mismatch"):
                train(config, seed=300, output_dir=output, n_workers=1)
            make_env.assert_not_called()
            pool.assert_not_called()

    def test_diagnostic_mismatch_stops_before_checkpoint_environment_or_pool(
        self,
    ) -> None:
        args = argparse.Namespace(
            expected_source_sha256="0" * 64,
            manifest="unread-manifest.json",
            expected_manifest_sha256="1" * 64,
            checkpoint="unread-checkpoint.npz",
            checkpoint_capture_manifest="unread-capture.json",
            training_config="unread-config.json",
            task_index=0,
            training_seed=300,
            generation=50,
            artifact_root="unwritten-artifacts",
            n_workers=1,
            chunk_pairs=1,
        )
        with patch.dict(
            os.environ,
            {"PAPER_EXPECTED_SOURCE_SHA": "0" * 64},
        ), patch.object(producer, "_parse_args", return_value=args), patch.object(
            producer, "load_locked_manifest"
        ) as load_manifest, patch.object(
            producer, "MujocoBatchEvaluator"
        ) as evaluator:
            with self.assertRaisesRegex(StudySourceLockError, "mismatch"):
                producer.main()
            load_manifest.assert_not_called()
            evaluator.assert_not_called()


if __name__ == "__main__":
    unittest.main()

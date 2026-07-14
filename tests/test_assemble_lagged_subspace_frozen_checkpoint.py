"""Fail-closed tests for the frozen-checkpoint artifact assembler."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from typing import Any, Mapping, Sequence
from unittest import mock

import numpy as np

from experiments.run_lagged_subspace_checkpoint_diagnostic import (
    CheckpointState,
    _array_sha256,
    _canonical_bytes,
    _labeled_arrays_sha256,
    _sha256_file,
    produce_checkpoint_diagnostic,
)
from scripts import analyze_lagged_subspace_frozen_checkpoint as analysis
from scripts.assemble_lagged_subspace_frozen_checkpoint import (
    AssemblyError,
    assemble,
)
try:
    from tests.test_analyze_lagged_subspace_frozen_checkpoint import _fixture_manifest
except ModuleNotFoundError as error:
    if error.name not in {
        "tests",
        "tests.test_analyze_lagged_subspace_frozen_checkpoint",
    }:
        raise
    from test_analyze_lagged_subspace_frozen_checkpoint import _fixture_manifest


class DeterministicEvaluator:
    def __init__(self, dimension: int, blocks: Sequence[int]) -> None:
        self.dimension = dimension
        self.blocks = list(blocks)

    def validate_policy(
        self,
        expected_dimension: int,
        expected_block_sizes: Sequence[int],
        expected_observation_dim: int,
        expected_action_dim: int,
    ) -> None:
        if expected_dimension != self.dimension or list(expected_block_sizes) != self.blocks:
            raise ValueError("fixture policy contract changed")
        if expected_observation_dim != 1 or expected_action_dim != 1:
            raise ValueError("fixture environment contract changed")

    def evaluate_batch(
        self, parameters: np.ndarray, rollout_seeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float64)
        seeds = np.asarray(rollout_seeds, dtype=np.uint64)
        target = np.linspace(-0.2, 0.25, parameters.shape[1])
        offsets = np.asarray([int(seed) % 997 for seed in seeds]) * 1e-8
        returns = -np.sum(np.square(parameters - target), axis=1) + offsets
        transitions = 5 + np.asarray(
            [int(seed) % 3 for seed in seeds], dtype=np.int64
        )
        return returns, transitions


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def _checkpoint_config(
    manifest: Mapping[str, Any],
    task_index: int,
    seed: int,
    hashes: Mapping[str, str],
) -> dict[str, Any]:
    dims = manifest["dimensions"]
    return {
        "checkpoint_capture_protocol": "lagged_subspace_frozen_checkpoint_v1",
        "env_name": manifest["tasks"][task_index]["env_name"],
        "seed": seed,
        "population_size": dims["population_size"],
        "learning_rate": manifest["training"]["learning_rate"],
        "lr_schedule": manifest["training"]["learning_rate_schedule"],
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
        "provenance": {
            "source_sha256": hashes["source_sha256"],
            "expected_source_sha256": hashes["source_sha256"],
            "expected_manifest_sha256": hashes["manifest_sha256"],
            "expected_protocol_sha256": hashes["protocol_sha256"],
            "expected_analyzer_sha256": hashes["analyzer_sha256"],
            "expected_launcher_sha256": hashes["launcher_sha256"],
            "expected_dependency_lock_sha256": hashes[
                "dependency_lock_sha256"
            ],
        },
    }


def _history(dims: Mapping[str, Any]) -> list[dict[str, Any]]:
    calibration = 11
    cumulative = 0
    rows = []
    for iteration in range(dims["training_updates"]):
        increment = dims["population_size"] * (iteration + 3)
        cumulative += increment
        rows.append(
            {
                "iteration": iteration,
                "n_fresh": dims["population_size"],
                "n_reused": 0,
                "used_replay": False,
                "eval_reward": None,
                "best_reward": None,
                "eval_env_steps": 0,
                "eval_env_steps_iter": 0,
                "initial_eval_reward": None,
                "initial_eval_env_steps": 0,
                "training_env_steps_iter": increment,
                "training_env_steps": cumulative,
                "normalization_calibration_env_steps": calibration,
            }
        )
    return rows


def _build_fixture(root: str) -> tuple[dict[str, Any], dict[str, str], str, str]:
    manifest = _fixture_manifest()
    manifest["checkpoint_generations"] = [10, 12]
    manifest["dimensions"]["training_updates"] = 12
    manifest["dimensions"]["lagged_gradient_count"] = 10
    manifest["dimensions"]["pairs_per_bank"] = 20
    manifest["dimensions"]["pairs_per_partition"] = 10
    dims = manifest["dimensions"]
    task_count = len(manifest["tasks"])
    seed_count = len(manifest["training_seeds"])
    checkpoint_count = task_count * seed_count * len(
        manifest["checkpoint_generations"]
    )
    budget = {
        "checkpoint_training_candidate_rollouts": task_count
        * seed_count
        * dims["training_updates"]
        * dims["population_size"],
        "normalization_calibration_rollouts": task_count
        * seed_count
        * dims["calibration_episodes"],
        "bank_candidate_rollouts": checkpoint_count
        * len(dims["banks"])
        * dims["pairs_per_bank"]
        * 2,
        "endpoint_arm_rollouts": checkpoint_count
        * len(dims["locality_q"])
        * dims["bank_b_partition_count"]
        * len(dims["endpoint_arms"])
        * dims["endpoint_episodes"],
        "checkpoint_center_rollouts": checkpoint_count
        * dims["endpoint_episodes"],
    }
    budget["total_policy_rollouts"] = sum(budget.values())
    budget["environment_transitions_are_separate"] = True
    manifest["budget"] = budget
    manifest_path = os.path.join(root, "fixture_manifest.json")
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256_file(manifest_path)

    locks_dir = os.path.join(root, "locks")
    os.makedirs(locks_dir)
    launcher_path = os.path.join(locks_dir, "launcher.sh")
    dependency_path = os.path.join(locks_dir, "requirements.lock")
    with open(launcher_path, "wb") as stream:
        stream.write(b"#!/bin/sh\nexit 0\n")
    with open(dependency_path, "wb") as stream:
        stream.write(b"numpy==fixture\n")
    hashes = {
        "source_sha256": hashlib.sha256(b"fixture-source").hexdigest(),
        "manifest_sha256": manifest_sha,
        "protocol_sha256": _sha256_file(
            os.path.join(analysis.REPO_ROOT, manifest["protocol"]["path"])
        ),
        "analyzer_sha256": _sha256_file(analysis.__file__),
        "launcher_sha256": _sha256_file(launcher_path),
        "dependency_lock_sha256": _sha256_file(dependency_path),
    }
    if manifest["protocol"]["sha256"] != hashes["protocol_sha256"]:
        raise AssertionError("fixture inherited a stale protocol hash")

    os.makedirs(os.path.join(root, "source_snapshot"))
    with open(os.path.join(root, "source_snapshot", "SOURCE_SHA256"), "w") as stream:
        stream.write(hashes["source_sha256"] + "\n")
    training_root = os.path.join(root, "training_runs")
    os.makedirs(training_root)
    diagnostic_root = os.path.join(root, "checkpoint_artifacts")
    os.makedirs(diagnostic_root)
    stderr_training = os.path.join(root, "stderr", "training")
    stderr_diagnostic = os.path.join(root, "stderr", "diagnostic")
    os.makedirs(stderr_training)
    os.makedirs(stderr_diagnostic)

    generations = manifest["checkpoint_generations"]
    rng = np.random.default_rng(8921)
    for task in manifest["tasks"]:
        task_index = task["task_index"]
        block_sizes = task["policy_block_sizes"]
        dimension = task["parameter_count"]
        for seed in manifest["training_seeds"]:
            training_id = analysis.training_id_for(manifest, task_index, seed)
            run_dir = os.path.join(training_root, f"training_{training_id:06d}")
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            os.makedirs(checkpoint_dir)
            config = _checkpoint_config(manifest, task_index, seed, hashes)
            _write_json(os.path.join(run_dir, "config.json"), config)
            checkpoint_config_path = os.path.join(
                run_dir, "checkpoint_training_config.json"
            )
            checkpoint_config = {
                key: value for key, value in config.items() if key != "provenance"
            }
            with open(checkpoint_config_path, "wb") as stream:
                stream.write(_canonical_bytes(checkpoint_config))
            checkpoint_config_sha = _sha256_file(checkpoint_config_path)

            capture_artifacts = []
            states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, str]] = {}
            for generation in generations:
                center = rng.normal(scale=0.1, size=dimension)
                gradients = rng.normal(
                    size=(dims["lagged_gradient_count"], dimension)
                )
                gradient_generations = np.arange(
                    generation - dims["lagged_gradient_count"],
                    generation,
                    dtype=np.int64,
                )
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"checkpoint_generation_{generation:06d}.npz",
                )
                np.savez(
                    checkpoint_path,
                    schema_version=np.asarray(2, dtype=np.int64),
                    checkpoint_generation=np.asarray(generation, dtype=np.int64),
                    study_source_sha256=np.asarray(
                        hashes["source_sha256"].encode("ascii"), dtype="S64"
                    ),
                    training_config_sha256=np.asarray(
                        checkpoint_config_sha.encode("ascii"), dtype="S64"
                    ),
                    center_params=center,
                    obs_normalizer_enabled=np.asarray(True, dtype=np.bool_),
                    obs_mean=np.asarray([0.2], dtype=np.float64),
                    obs_var=np.asarray([1.3], dtype=np.float64),
                    obs_count=np.asarray(20.0, dtype=np.float64),
                    gradient_generations=gradient_generations,
                    proposal_gradients=gradients,
                )
                metadata = {
                    "checkpoint_generation": generation,
                    "artifact": os.path.join(
                        "checkpoints",
                        f"checkpoint_generation_{generation:06d}.npz",
                    ),
                    "artifact_sha256": _sha256_file(checkpoint_path),
                    "strictly_prior_gradient_archive": True,
                    "current_checkpoint_gradient_included": False,
                    "last_applied_gradient_generation": generation - 1,
                    "source_sha256": hashes["source_sha256"],
                    "training_config_sha256": checkpoint_config_sha,
                }
                capture_artifacts.append(metadata)
                states[generation] = (
                    center,
                    gradients,
                    gradient_generations,
                    checkpoint_path,
                )
            capture = {
                "schema_version": 1,
                "status": "complete",
                "requested_generations": generations,
                "captured_generations": generations,
                "expected_checkpoint_count": len(generations),
                "checkpoint_count": len(generations),
                "gradient_archive_length": dims["lagged_gradient_count"],
                "selection_policy": "fixed_config_generations_only",
                "reward_selection_used": False,
                "current_generation_gradient_excluded": True,
                "online_evaluation_enabled": False,
                "source_sha256": hashes["source_sha256"],
                "training_config_sha256": checkpoint_config_sha,
                "validated_generator_controls": {
                    "plain_standard_es": True,
                    "rank_fitness": True,
                    "antithetic": True,
                    "replay": False,
                    "importance_sampling": False,
                    "trust_region": False,
                    "picard_iteration": False,
                    "gradient_clipping": False,
                    "parameter_projection": False,
                    "curvature": False,
                    "curvature_clipping": False,
                    "l2": False,
                    "checkpoint_selection_by_reward": False,
                },
                "artifacts": capture_artifacts,
            }
            capture_path = os.path.join(run_dir, "checkpoint_capture.json")
            _write_json(capture_path, capture)
            history = _history(dims)
            _write_json(os.path.join(run_dir, "history.json"), history)
            with open(os.path.join(run_dir, "history.jsonl"), "w") as stream:
                for row in history:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            _write_json(
                os.path.join(run_dir, "status.json"),
                {
                    "status": "complete",
                    "expected_iterations": dims["training_updates"],
                    "completed_iterations": dims["training_updates"],
                    "best_reward": None,
                    "initial_eval_reward": None,
                    "normalization_calibration_env_steps": 11,
                },
            )
            np.save(os.path.join(run_dir, "final_params.npy"), center)
            np.savez(
                os.path.join(run_dir, "obs_norm.npz"),
                mean=np.asarray([0.2]),
                var=np.asarray([1.3]),
                count=np.asarray(20.0),
            )
            open(
                os.path.join(stderr_training, f"training_{training_id:06d}.stderr"),
                "wb",
            ).close()

            evaluator = DeterministicEvaluator(dimension, block_sizes)
            for generation in generations:
                center, gradients, gradient_generations, checkpoint_path = states[
                    generation
                ]
                state = CheckpointState(
                    center=center,
                    obs_mean=np.asarray([0.2], dtype=np.float64),
                    obs_var=np.asarray([1.3], dtype=np.float64),
                    obs_count=20.0,
                    gradient_generations=gradient_generations,
                    gradients=gradients,
                    checkpoint_sha256=_sha256_file(checkpoint_path),
                    capture_manifest_sha256=_sha256_file(capture_path),
                    training_config_sha256=checkpoint_config_sha,
                    source_sha256=hashes["source_sha256"],
                )
                checkpoint_id = analysis.checkpoint_id_for(
                    manifest, task_index, seed, generation
                )
                produce_checkpoint_diagnostic(
                    manifest=manifest,
                    manifest_sha256=hashes["manifest_sha256"],
                    state=state,
                    training_config_path=checkpoint_config_path,
                    checkpoint_path=checkpoint_path,
                    capture_manifest_path=capture_path,
                    task_index=task_index,
                    training_seed=seed,
                    generation=generation,
                    artifact_root=root,
                    evaluator=evaluator,
                    chunk_pairs=3,
                )
                open(
                    os.path.join(
                        stderr_diagnostic,
                        f"checkpoint_{checkpoint_id:06d}.stderr",
                    ),
                    "wb",
                ).close()
    return manifest, hashes, launcher_path, dependency_path


class ArtifactAssemblerTests(unittest.TestCase):
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

    def _assemble(self, root: str | None = None) -> dict[str, Any]:
        root = self.base.name if root is None else root
        return assemble(
            artifact_root=root,
            manifest_path=os.path.join(root, "fixture_manifest.json"),
            expected_hashes=self.hashes,
            source_snapshot_path="source_snapshot",
            launcher_lock_path=os.path.join(root, "locks", "launcher.sh"),
            dependency_lock_path=os.path.join(root, "locks", "requirements.lock"),
            require_preregistered_manifest=False,
        )

    def _copy(self) -> tempfile.TemporaryDirectory:
        target = tempfile.TemporaryDirectory()
        shutil.copytree(self.base.name, target.name, dirs_exist_ok=True)
        return target

    def test_end_to_end_assembly_validates_and_analyzes(self) -> None:
        artifact = self._assemble()
        validated = analysis.validate_artifact(
            copy.deepcopy(artifact),
            self.manifest,
            expected_hashes=self.hashes,
            artifact_root=self.base.name,
            require_preregistered_manifest=False,
        )
        result = analysis.analyze_validated(validated, self.manifest)
        self.assertEqual(result["study"], analysis.STUDY)
        self.assertEqual(len(result["task_results"]), len(self.manifest["tasks"]))
        self.assertEqual(
            len(artifact["training_runs"]),
            len(self.manifest["tasks"]) * len(self.manifest["training_seeds"]),
        )

    def test_release_artifacts_omit_regenerable_full_dimension_arrays(self) -> None:
        fragment_path = os.path.join(
            self.base.name,
            "checkpoint_artifacts",
            "checkpoint_000000",
            "checkpoint_index.json",
        )
        with open(fragment_path, encoding="utf-8") as stream:
            fragment = json.load(stream)
        raw_path = os.path.join(
            self.base.name, fragment["banks"][0]["raw_bank_path"]
        )
        with np.load(raw_path, allow_pickle=False) as archive:
            self.assertNotIn("signed_noise", archive.files)
            self.assertEqual(set(archive.files), analysis.RAW_BANK_NPZ_KEYS)
        diagnostic_records = [
            *fragment["banks"],
            *fragment["partitions"],
        ]
        full_dimension_names = {
            "gradient",
            "gradient_component_variance",
            *{
                f"step_q{q_index}_{arm}"
                for q_index in range(
                    len(self.manifest["dimensions"]["locality_q"])
                )
                for arm in self.manifest["dimensions"]["endpoint_arms"]
            },
        }
        for record in diagnostic_records:
            path = os.path.join(self.base.name, record["diagnostics_path"])
            with np.load(path, allow_pickle=False) as archive:
                self.assertTrue(full_dimension_names.isdisjoint(archive.files))
                self.assertEqual(
                    set(archive.files),
                    analysis._diagnostic_npz_keys(self.manifest),
                )
                self.assertEqual(archive["gradient_sha256"].dtype, np.dtype("S64"))
                self.assertEqual(
                    archive["gradient_component_variance_sha256"].dtype,
                    np.dtype("S64"),
                )
                self.assertEqual(
                    archive["step_sha256"].shape,
                    (
                        len(self.manifest["dimensions"]["locality_q"]),
                        len(self.manifest["dimensions"]["endpoint_arms"]),
                    ),
                )

    def test_cli_writes_atomically_and_refuses_overwrite(self) -> None:
        output_relative = "assembled/audit_index.json"
        command = [
            sys.executable,
            "scripts/assemble_lagged_subspace_frozen_checkpoint.py",
            "--artifact-root",
            self.base.name,
            "--manifest",
            self.manifest_path,
            "--source-snapshot-path",
            "source_snapshot",
            "--launcher-lock",
            self.launcher,
            "--dependency-lock",
            self.dependency,
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
            output_relative,
            "--fixture-mode",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=analysis.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output_path = os.path.join(self.base.name, output_relative)
            with open(output_path, "rb") as stream:
                serialized = stream.read()
            expected = self._assemble()
            self.assertEqual(serialized.count(b"\n"), 1)
            self.assertEqual(serialized[:-1], _canonical_bytes(expected))
            with open(output_path, encoding="utf-8") as stream:
                written = json.load(stream)
            self.assertEqual(written, expected)
            original_sha = _sha256_file(output_path)
            repeated = subprocess.run(
                command,
                cwd=analysis.REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(_sha256_file(output_path), original_sha)
        finally:
            shutil.rmtree(os.path.join(self.base.name, "assembled"), ignore_errors=True)

    def test_missing_input_fails_before_merge(self) -> None:
        copied = self._copy()
        try:
            shutil.rmtree(
                os.path.join(copied.name, "checkpoint_artifacts", "checkpoint_000000")
            )
            with self.assertRaisesRegex(AssemblyError, "partial"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_extra_input_fails_before_merge(self) -> None:
        copied = self._copy()
        try:
            os.makedirs(os.path.join(copied.name, "training_runs", "unexpected"))
            with self.assertRaisesRegex(AssemblyError, "extras"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_nonempty_stderr_and_evaluation_artifact_fail(self) -> None:
        copied = self._copy()
        try:
            with open(
                os.path.join(
                    copied.name,
                    "stderr",
                    "diagnostic",
                    "checkpoint_000000.stderr",
                ),
                "wb",
            ) as stream:
                stream.write(b"warning\n")
            with self.assertRaisesRegex(AssemblyError, "not empty"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

        copied = self._copy()
        try:
            np.save(
                os.path.join(
                    copied.name,
                    "training_runs",
                    "training_000000",
                    "best_params.npy",
                ),
                np.zeros(1),
            )
            with self.assertRaisesRegex(AssemblyError, "forbidden"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_mixed_source_lock_fails(self) -> None:
        copied = self._copy()
        try:
            config_path = os.path.join(
                copied.name,
                "training_runs",
                "training_000000",
                "config.json",
            )
            with open(config_path, encoding="utf-8") as stream:
                config = json.load(stream)
            config["provenance"]["source_sha256"] = "0" * 64
            _write_json(config_path, config)
            with self.assertRaisesRegex(AssemblyError, "lock mismatch"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_duplicate_fragment_records_fail_even_when_resigned(self) -> None:
        copied = self._copy()
        try:
            fragment_path = os.path.join(
                copied.name,
                "checkpoint_artifacts",
                "checkpoint_000000",
                "checkpoint_index.json",
            )
            with open(fragment_path, encoding="utf-8") as stream:
                fragment = json.load(stream)
            fragment["banks"][1] = copy.deepcopy(fragment["banks"][0])
            payload = dict(fragment)
            payload.pop("fragment_sha256")
            fragment["fragment_sha256"] = hashlib.sha256(
                _canonical_bytes(payload)
            ).hexdigest()
            _write_json(fragment_path, fragment)
            with self.assertRaisesRegex(AssemblyError, "bank order/identity"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_unsigned_fragment_tampering_fails(self) -> None:
        copied = self._copy()
        try:
            fragment_path = os.path.join(
                copied.name,
                "checkpoint_artifacts",
                "checkpoint_000000",
                "checkpoint_index.json",
            )
            with open(fragment_path, encoding="utf-8") as stream:
                fragment = json.load(stream)
            fragment["endpoints"][0]["return"] += 1.0
            _write_json(fragment_path, fragment)
            with self.assertRaisesRegex(AssemblyError, "fragment digest mismatch"):
                self._assemble(copied.name)
        finally:
            copied.cleanup()

    def test_production_snapshot_composite_digest_is_enforced(self) -> None:
        production_manifest = analysis.DEFAULT_MANIFEST_PATH
        with open(production_manifest, encoding="utf-8") as stream:
            manifest = json.load(stream)
        hashes = {
            "source_sha256": "b" * 64,
            "manifest_sha256": _sha256_file(production_manifest),
            "protocol_sha256": _sha256_file(
                os.path.join(analysis.REPO_ROOT, manifest["protocol"]["path"])
            ),
            "analyzer_sha256": _sha256_file(analysis.__file__),
            "launcher_sha256": _sha256_file(self.launcher),
            "dependency_lock_sha256": _sha256_file(self.dependency),
        }
        with mock.patch(
            "scripts.assemble_lagged_subspace_frozen_checkpoint."
            "compute_lagged_subspace_study_sha256",
            return_value="a" * 64,
        ):
            with self.assertRaisesRegex(AssemblyError, "composite digest"):
                assemble(
                    artifact_root=self.base.name,
                    manifest_path=production_manifest,
                    expected_hashes=hashes,
                    source_snapshot_path="source_snapshot",
                    launcher_lock_path=self.launcher,
                    dependency_lock_path=self.dependency,
                    require_preregistered_manifest=True,
                )


if __name__ == "__main__":
    unittest.main()

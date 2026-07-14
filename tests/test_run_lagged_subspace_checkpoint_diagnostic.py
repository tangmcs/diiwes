"""End-to-end tests for the frozen-checkpoint diagnostic producer."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from typing import Sequence

import numpy as np

from core.lagged_subspace_diagnostic import (
    BasisProvenance,
    build_lagged_bases,
    calibrate_locality_rate,
    estimate_lopo_population,
    frozen_endpoint_diagnostics,
)
from experiments.run_lagged_subspace_checkpoint_diagnostic import (
    CheckpointState,
    FrozenEndpointReference,
    _array_sha256,
    _canonical_bytes,
    _labeled_arrays_sha256,
    _population_product,
    _sha256_file,
    _stable_norm,
    bank_b_partitions,
    derive_producer_seed,
    load_and_validate_checkpoint,
    produce_checkpoint_diagnostic,
)
from scripts.analyze_lagged_subspace_frozen_checkpoint import (
    BANK_KEYS,
    CENTER_KEYS,
    CHECKPOINT_KEYS as ANALYZER_CHECKPOINT_KEYS,
    ENDPOINT_KEYS,
    METRIC_KEYS,
    PARTITION_KEYS,
    Q_SUMMARY_KEYS,
    _validate_diagnostic_data,
    _validate_record_hash,
    reconstruct_lagged_bases,
)


def _manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "study": "lagged_subspace_frozen_checkpoint",
        "tasks": [
            {
                "task_index": 0,
                "env_name": "MockContinuous-v0",
                "observation_dim": 1,
                "action_dim": 1,
                "parameter_count": 10,
                "policy_block_sizes": [4, 4, 2],
                "policy_block_ranges": [[0, 4], [4, 8], [8, 10]],
            }
        ],
        "training_seeds": [7],
        "checkpoint_generations": [10],
        "dimensions": {
            "policy_hidden_widths": [2, 2],
            "layer_blocks": 3,
            "population_size": 6,
            "antithetic_pairs_per_training_update": 3,
            "training_updates": 10,
            "lagged_gradient_count": 10,
            "lagged_decay": 0.9,
            "noise_std": 0.1,
            "banks": ["A", "B"],
            "pairs_per_bank": 6,
            "bank_b_partition_count": 2,
            "pairs_per_partition": 3,
            "locality_q": [0.25, 0.5, 1.0],
            "endpoint_arms": ["structured", "isotropic", "explicit", "random"],
            "endpoint_episodes": 2,
            "calibration_episodes": 3,
        },
        "training": {"learning_rate": 0.0001},
        "rng": {
            "scheme": "sha256_uint64_v1",
            "master_seed": 2026071201,
            "serialization": "canonical_json_utf8",
            "uint64_extraction": "first_8_digest_bytes_big_endian",
            "partition_permutation": "ascending_sha256_of_partition_seed_and_pair_index",
            "basis_gaussian": "numpy_generator_pcg64_conditional_standard_normal_by_zero_block_v1",
            "random_signed_permutation": "numpy_generator_pcg64_permutation_then_int64_rademacher_and_renormalize_by_block_v1",
            "perturbation_gaussian": "numpy_generator_pcg64_standard_normal_per_pair_v1",
            "namespaces": {
                "basis": 11,
                "random_control": 12,
                "bank_perturbation": 21,
                "bank_rollout": 22,
                "bank_b_partition": 23,
                "endpoint": 31,
                "cluster_bootstrap": 41,
            },
        },
        "analysis": {
            "machine_epsilon": float(np.finfo(np.float64).eps),
            "primary_q": 0.5,
            "zero_gradient_calibration": {
                "detection": "scaled_l2_exact_zero_only",
                "alpha_sentinel": 0.0,
                "unresolved_reason": "bank_a_gradient_exact_zero",
                "all_four_steps_zero": True,
                "gate_policy": "fail_all_affected_task_conditions",
            },
        },
        "prohibitions": {
            "replay": False,
            "importance_sampling": False,
            "trust_clipping": False,
            "picard_iteration": False,
        },
    }


def _training_config(manifest: dict[str, object]) -> dict[str, object]:
    dims = manifest["dimensions"]
    return {
        "checkpoint_capture_protocol": "lagged_subspace_frozen_checkpoint_v1",
        "env_name": "MockContinuous-v0",
        "seed": 7,
        "population_size": dims["population_size"],
        "learning_rate": 0.0001,
        "lr_schedule": "constant",
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
        "obs_norm_calibration_episodes": 3,
        "heldout_evaluation_enabled": False,
        "checkpoint_capture_generations": [10],
        "checkpoint_gradient_archive_length": 10,
        "replay_enabled": False,
        "buffer_size": 0,
        "reuse_fraction": 0.0,
        "common_rollout_seed": True,
        "evaluate_center_fitness": False,
        "condition": "standard_es",
        "algorithm": "standard_es",
    }


class RecordingEvaluator:
    def __init__(self, *, fail_call: int | None = None) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.fail_call = fail_call

    def validate_policy(
        self,
        expected_dimension: int,
        expected_block_sizes: Sequence[int],
        expected_observation_dim: int,
        expected_action_dim: int,
    ) -> None:
        if (
            expected_dimension != 10
            or list(expected_block_sizes) != [4, 4, 2]
            or expected_observation_dim != 1
            or expected_action_dim != 1
        ):
            raise ValueError("mock policy contract changed")

    def evaluate_batch(
        self, parameters: np.ndarray, rollout_seeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.fail_call is not None and len(self.calls) == self.fail_call:
            raise RuntimeError("injected evaluator failure")
        parameters = np.asarray(parameters, dtype=np.float64)
        seeds = np.asarray(rollout_seeds, dtype=np.uint64)
        self.calls.append((parameters.copy(), seeds.copy()))
        target = np.linspace(-0.2, 0.3, parameters.shape[1])
        seed_offsets = np.asarray([int(value) % 101 for value in seeds]) * 1e-5
        returns = -np.sum((parameters - target) ** 2, axis=1) + seed_offsets
        transitions = 5 + np.asarray([int(value) % 3 for value in seeds], dtype=np.int64)
        return returns, transitions


class ConstantReturnEvaluator(RecordingEvaluator):
    def evaluate_batch(
        self, parameters: np.ndarray, rollout_seeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float64)
        seeds = np.asarray(rollout_seeds, dtype=np.uint64)
        self.calls.append((parameters.copy(), seeds.copy()))
        return (
            np.zeros(parameters.shape[0], dtype=np.float64),
            np.full(parameters.shape[0], 7, dtype=np.int64),
        )


class CheckpointProducerTests(unittest.TestCase):
    def test_core_and_analyzer_basis_reconstruction_are_bitwise_identical(self) -> None:
        block_sizes = [2, 3, 4]
        blocks = (slice(0, 2), slice(2, 5), slice(5, 9))
        base = np.random.default_rng(103).normal(size=(10, 9))
        for zero_bits in range(8):
            gradients = base.copy()
            for block_index, block in enumerate(blocks):
                if zero_bits & (1 << block_index):
                    gradients[:, block] = 0.0
            core = build_lagged_bases(
                gradients,
                blocks,
                primary_fallback_seed=123456789012345,
                random_permutation_seed=987654321098765,
                primary_reference="test-chronological-archive",
                random_reference="test-locked-random-control",
            )
            analyzer = reconstruct_lagged_bases(
                gradients,
                block_sizes,
                lagged_decay=0.9,
                basis_seed_value=123456789012345,
                random_seed_value=987654321098765,
            )
            fallback_columns = np.zeros_like(core.primary)
            for block_index, used in enumerate(core.primary_fallback_blocks):
                if used:
                    fallback_columns[:, block_index] = core.primary[:, block_index]
            norms = np.asarray(
                [
                    _stable_norm(core.lagged_block_gradients[:, index])
                    for index in range(3)
                ]
            )
            self.assertTrue(np.array_equal(core.primary, analyzer["primary_basis"]))
            self.assertTrue(
                np.array_equal(core.random_control, analyzer["random_basis"])
            )
            self.assertTrue(
                np.array_equal(fallback_columns, analyzer["fallback_columns"])
            )
            self.assertTrue(
                np.array_equal(norms, analyzer["lagged_block_norms"])
            )
            self.assertEqual(
                list(core.primary_fallback_blocks),
                analyzer["lagged_block_exact_zero"].tolist(),
            )

    def _lineage_fixture(
        self, root: str
    ) -> tuple[dict[str, object], str, str, str, str, np.ndarray, np.ndarray]:
        manifest = _manifest()
        source_sha = "a" * 64
        config = _training_config(manifest)
        config_path = os.path.join(root, "checkpoint_training_config.json")
        with open(config_path, "wb") as stream:
            stream.write(_canonical_bytes(config))
        config_sha = _sha256_file(config_path)

        run_dir = os.path.join(root, "run")
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(
            checkpoint_dir, "checkpoint_generation_000010.npz"
        )
        center = np.linspace(-0.1, 0.1, 10)
        gradients = np.arange(100, dtype=np.float64).reshape(10, 10) / 100.0
        generations = np.arange(10, dtype=np.int64)
        np.savez(
            checkpoint_path,
            schema_version=np.asarray(2, dtype=np.int64),
            checkpoint_generation=np.asarray(10, dtype=np.int64),
            study_source_sha256=np.asarray(
                source_sha.encode("ascii"), dtype="S64"
            ),
            training_config_sha256=np.asarray(
                config_sha.encode("ascii"), dtype="S64"
            ),
            center_params=center,
            obs_normalizer_enabled=np.asarray(True, dtype=np.bool_),
            obs_mean=np.asarray([0.25]),
            obs_var=np.asarray([1.5]),
            obs_count=np.asarray(100.0),
            gradient_generations=generations,
            proposal_gradients=gradients,
        )
        artifact_metadata = {
            "checkpoint_generation": 10,
            "artifact": "checkpoints/checkpoint_generation_000010.npz",
            "artifact_sha256": _sha256_file(checkpoint_path),
            "center_params_sha256": _array_sha256(center),
            "observation_normalizer_state_sha256": _labeled_arrays_sha256(
                [
                    ("enabled", np.asarray(True, dtype=np.bool_)),
                    ("mean", np.asarray([0.25])),
                    ("var", np.asarray([1.5])),
                    ("count", np.asarray(100.0)),
                ]
            ),
            "indexed_gradient_archive_sha256": _labeled_arrays_sha256(
                [
                    ("gradient_generations", generations),
                    ("proposal_gradients", gradients),
                ]
            ),
            "proposal_gradient_hashes": [
                {"generation": row, "sha256": _array_sha256(gradients[row])}
                for row in range(10)
            ],
            "strictly_prior_gradient_archive": True,
            "current_checkpoint_gradient_included": False,
            "last_applied_gradient_generation": 9,
            "source_sha256": source_sha,
            "training_config_sha256": config_sha,
        }
        capture = {
            "schema_version": 1,
            "status": "complete",
            "requested_generations": [10],
            "captured_generations": [10],
            "checkpoint_count": 1,
            "gradient_archive_length": 10,
            "reward_selection_used": False,
            "current_generation_gradient_excluded": True,
            "online_evaluation_enabled": False,
            "source_sha256": source_sha,
            "training_config_sha256": config_sha,
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
            "artifacts": [artifact_metadata],
        }
        capture_path = os.path.join(run_dir, "checkpoint_capture.json")
        with open(capture_path, "w", encoding="utf-8") as stream:
            json.dump(capture, stream)
        return (
            manifest,
            source_sha,
            checkpoint_path,
            capture_path,
            config_path,
            center,
            gradients,
        )

    def test_checkpoint_validation_enforces_exact_chronological_archive(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            (
                manifest,
                source_sha,
                checkpoint_path,
                capture_path,
                config_path,
                center,
                gradients,
            ) = self._lineage_fixture(root)
            state, _ = load_and_validate_checkpoint(
                checkpoint_path=checkpoint_path,
                capture_manifest_path=capture_path,
                training_config_path=config_path,
                manifest=manifest,
                task_index=0,
                training_seed=7,
                generation=10,
                expected_source_sha256=source_sha,
            )
            self.assertTrue(np.array_equal(state.center, center))
            self.assertTrue(np.array_equal(state.gradients, gradients))
            self.assertEqual(state.gradient_generations.tolist(), list(range(10)))

            with np.load(checkpoint_path, allow_pickle=False) as archive:
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            arrays["gradient_generations"] = arrays["gradient_generations"][::-1]
            np.savez(checkpoint_path, **arrays)
            with open(capture_path, encoding="utf-8") as stream:
                capture = json.load(stream)
            capture["artifacts"][0]["artifact_sha256"] = _sha256_file(checkpoint_path)
            with open(capture_path, "w", encoding="utf-8") as stream:
                json.dump(capture, stream)
            with self.assertRaisesRegex(ValueError, "chronological prior gradients"):
                load_and_validate_checkpoint(
                    checkpoint_path=checkpoint_path,
                    capture_manifest_path=capture_path,
                    training_config_path=config_path,
                    manifest=manifest,
                    task_index=0,
                    training_seed=7,
                    generation=10,
                    expected_source_sha256=source_sha,
                )

    def _state_and_inputs(
        self, root: str
    ) -> tuple[dict[str, object], CheckpointState, str, str, str]:
        manifest = _manifest()
        rng = np.random.default_rng(19)
        center = rng.normal(scale=0.1, size=10)
        gradients = rng.normal(size=(10, 10))
        config_path = os.path.join(root, "config.json")
        checkpoint_path = os.path.join(root, "checkpoint.npz")
        capture_path = os.path.join(root, "capture.json")
        for path, payload in (
            (config_path, b"{\"mock\":true}"),
            (checkpoint_path, b"mock-checkpoint"),
            (capture_path, b"{\"mock_capture\":true}"),
        ):
            with open(path, "wb") as stream:
                stream.write(payload)
        state = CheckpointState(
            center=center,
            obs_mean=np.asarray([0.0]),
            obs_var=np.asarray([1.0]),
            obs_count=10.0,
            gradient_generations=np.arange(10, dtype=np.int64),
            gradients=gradients,
            checkpoint_sha256=_sha256_file(checkpoint_path),
            capture_manifest_sha256=_sha256_file(capture_path),
            training_config_sha256=_sha256_file(config_path),
            source_sha256="b" * 64,
        )
        return manifest, state, config_path, checkpoint_path, capture_path

    def test_complete_mapping_crn_norm_matching_and_deterministic_artifacts(self) -> None:
        inventories = []
        for repetition in range(2):
            with tempfile.TemporaryDirectory() as root:
                manifest, state, config_path, checkpoint_path, capture_path = (
                    self._state_and_inputs(root)
                )
                artifact_root = os.path.join(root, "artifacts")
                evaluator = RecordingEvaluator()
                index_path = produce_checkpoint_diagnostic(
                    manifest=manifest,
                    manifest_sha256="c" * 64,
                    state=state,
                    training_config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    capture_manifest_path=capture_path,
                    task_index=0,
                    training_seed=7,
                    generation=10,
                    artifact_root=artifact_root,
                    evaluator=evaluator,
                    chunk_pairs=2,
                )
                with open(index_path, encoding="utf-8") as stream:
                    index = json.load(stream)
                expected_fragment = dict(index)
                fragment_sha = expected_fragment.pop("fragment_sha256")
                self.assertEqual(
                    fragment_sha, hashlib.sha256(_canonical_bytes(expected_fragment)).hexdigest()
                )
                self.assertEqual(len(index["banks"]), 2)
                self.assertEqual(len(index["partitions"]), 2)
                self.assertEqual(len(index["checkpoint_metrics"]), 3)
                self.assertEqual(len(index["center_endpoints"]), 2)
                self.assertEqual(len(index["endpoints"]), 3 * 2 * 4 * 2)
                self.assertEqual(set(index["checkpoint"]), ANALYZER_CHECKPOINT_KEYS)
                self.assertTrue(all(set(row) == BANK_KEYS for row in index["banks"]))
                self.assertTrue(
                    all(set(row) == PARTITION_KEYS for row in index["partitions"])
                )
                self.assertTrue(
                    all(set(row) == METRIC_KEYS for row in index["checkpoint_metrics"])
                )
                self.assertTrue(
                    all(set(row) == CENTER_KEYS for row in index["center_endpoints"])
                )
                self.assertTrue(
                    all(set(row) == ENDPOINT_KEYS for row in index["endpoints"])
                )
                for row in index["banks"] + index["partitions"]:
                    self.assertTrue(
                        all(set(summary) == Q_SUMMARY_KEYS for summary in row["q_summaries"])
                    )
                record_hash_issues: list[str] = []
                _validate_record_hash(
                    index["checkpoint"], "producer checkpoint", record_hash_issues
                )
                for family in (
                    "banks",
                    "partitions",
                    "checkpoint_metrics",
                    "center_endpoints",
                    "endpoints",
                ):
                    for row_index, row in enumerate(index[family]):
                        _validate_record_hash(
                            row, f"producer {family}[{row_index}]", record_hash_issues
                        )
                self.assertEqual(record_hash_issues, [])

                partitions = [row["pair_indices"] for row in index["partitions"]]
                self.assertEqual(
                    partitions, bank_b_partitions(manifest, 0, 7, 10)
                )
                self.assertEqual(sorted(sum(partitions, [])), list(range(6)))
                for q_index, q in enumerate([0.25, 0.5, 1.0]):
                    alpha = index["banks"][0]["q_summaries"][q_index]["alpha"]
                    gradient_norm = index["banks"][0]["q_summaries"][q_index][
                        "gradient_norm"
                    ]
                    self.assertAlmostEqual(alpha, q * 0.1 / gradient_norm)
                    for record in index["banks"] + index["partitions"]:
                        summary = record["q_summaries"][q_index]
                        self.assertEqual(summary["alpha"], alpha)
                        self.assertLessEqual(
                            abs(summary["structured_norm"] - summary["isotropic_norm"])
                            / max(summary["structured_norm"], np.finfo(float).eps),
                            1e-12,
                        )
                        if summary["random_control_valid"]:
                            self.assertLessEqual(
                                abs(summary["structured_norm"] - summary["random_norm"])
                                / max(summary["structured_norm"], np.finfo(float).eps),
                                1e-12,
                            )

                raw_banks = []
                for bank in index["banks"]:
                    raw_path = os.path.join(artifact_root, bank["raw_bank_path"])
                    with np.load(raw_path, allow_pickle=False) as archive:
                        raw = {name: np.asarray(archive[name]) for name in archive.files}
                    self.assertNotIn("signed_noise", raw)
                    regenerated_plus = np.stack(
                        [
                            np.random.Generator(
                                np.random.PCG64(int(seed))
                            ).standard_normal(10)
                            for seed in raw["perturbation_seeds"]
                        ]
                    )
                    raw["signed_noise"] = np.stack(
                        (regenerated_plus, -regenerated_plus), axis=1
                    )
                    raw_banks.append(raw)
                    self.assertTrue(
                        np.array_equal(raw["signed_noise"][:, 1], -raw["signed_noise"][:, 0])
                    )
                    self.assertTrue(
                        np.array_equal(raw["rollout_seeds_plus"], raw["rollout_seeds_minus"])
                    )
                    for pair, seed in enumerate(raw["perturbation_seeds"]):
                        expected = np.random.Generator(
                            np.random.PCG64(int(seed))
                        ).standard_normal(10)
                        self.assertTrue(
                            np.array_equal(raw["signed_noise"][pair, 0], expected)
                        )
                basis_path = os.path.join(
                    artifact_root, index["checkpoint"]["basis_artifact_path"]
                )
                with np.load(basis_path, allow_pickle=False) as archive:
                    basis = np.asarray(archive["primary_basis"])
                    random_basis = np.asarray(archive["random_basis"])
                analyzer_issues: list[str] = []
                reference = None
                for bank_record, raw in zip(index["banks"], raw_banks, strict=True):
                    with np.load(
                        os.path.join(artifact_root, bank_record["diagnostics_path"]),
                        allow_pickle=False,
                    ) as archive:
                        diagnostic = {
                            name: np.asarray(archive[name]) for name in archive.files
                        }
                    recomputed = _validate_diagnostic_data(
                        diagnostic,
                        label=f"test bank {bank_record['bank']}",
                        manifest=manifest,
                        theta=state.center,
                        signed_noise=raw["signed_noise"],
                        paired_returns=raw["paired_returns"],
                        basis=basis,
                        random_basis=random_basis,
                        q_summaries=bank_record["q_summaries"],
                        reference_curvature=(
                            None if reference is None else reference["curvature"]
                        ),
                        reference_actions=(
                            None if reference is None else reference["actions"]
                        ),
                        issues=analyzer_issues,
                        endpoint_reference=(
                            None
                            if reference is None
                            else reference["endpoint_reference"]
                        ),
                    )
                    self.assertIsNotNone(recomputed)
                    if reference is None:
                        reference = recomputed
                self.assertIsNotNone(reference)
                for partition_record in index["partitions"]:
                    selected = np.asarray(partition_record["pair_indices"], dtype=np.int64)
                    with np.load(
                        os.path.join(
                            artifact_root, partition_record["diagnostics_path"]
                        ),
                        allow_pickle=False,
                    ) as archive:
                        diagnostic = {
                            name: np.asarray(archive[name]) for name in archive.files
                        }
                    recomputed = _validate_diagnostic_data(
                        diagnostic,
                        label=(
                            f"test partition {partition_record['partition_index']}"
                        ),
                        manifest=manifest,
                        theta=state.center,
                        signed_noise=raw_banks[1]["signed_noise"][selected],
                        paired_returns=raw_banks[1]["paired_returns"][selected],
                        basis=basis,
                        random_basis=random_basis,
                        q_summaries=partition_record["q_summaries"],
                        reference_curvature=reference["curvature"],
                        reference_actions=reference["actions"],
                        issues=analyzer_issues,
                        endpoint_reference=reference["endpoint_reference"],
                    )
                    self.assertIsNotNone(recomputed)
                self.assertEqual(analyzer_issues, [])
                a_seeds = set(map(int, raw_banks[0]["perturbation_seeds"])) | set(
                    map(int, raw_banks[0]["rollout_seeds_plus"])
                )
                b_seeds = set(map(int, raw_banks[1]["perturbation_seeds"])) | set(
                    map(int, raw_banks[1]["rollout_seeds_plus"])
                )
                self.assertFalse(a_seeds.intersection(b_seeds))

                endpoint_relative = index["center_endpoints"][0]["rollout_artifact_path"]
                with np.load(
                    os.path.join(artifact_root, endpoint_relative), allow_pickle=False
                ) as archive:
                    self.assertEqual(archive["endpoint_returns"].shape, (3, 2, 4, 2))
                    endpoint_seeds = np.asarray(archive["rollout_seeds"])
                expected_endpoint = np.asarray(
                    [
                        derive_producer_seed(
                            manifest, "endpoint", 0, 7, 10, episode
                        )
                        for episode in range(2)
                    ],
                    dtype=np.uint64,
                )
                self.assertTrue(np.array_equal(endpoint_seeds, expected_endpoint))
                for record in index["center_endpoints"] + index["endpoints"]:
                    self.assertEqual(
                        record["rollout_seed"], int(endpoint_seeds[record["episode_index"]])
                    )

                inventory = {
                    os.path.basename(row["path"]): row["sha256"]
                    for row in index["artifact_inventory"]
                }
                inventories.append(inventory)
                self.assertFalse(
                    any(name.startswith(".checkpoint_") for name in os.listdir(
                        os.path.join(artifact_root, "checkpoint_artifacts")
                    ))
                )
        self.assertEqual(inventories[0], inventories[1])

    def test_bank_b_frozen_map_diagnostics_use_complete_bank_a(self) -> None:
        sigma = 0.1
        theta = np.zeros(6, dtype=np.float64)
        basis = np.zeros((6, 3), dtype=np.float64)
        random_basis = np.zeros_like(basis)
        basis[[0, 2, 4], range(3)] = 1.0
        random_basis[[1, 3, 5], range(3)] = 1.0
        provenance = BasisProvenance.strictly_lagged("prior", "random-prior")
        rng_a = np.random.default_rng(701)
        rng_b = np.random.default_rng(907)
        plus_a = rng_a.normal(size=(8, theta.size))
        plus_b = rng_b.normal(size=(8, theta.size))
        signed_a = np.stack((plus_a, -plus_a), axis=1)
        signed_b = np.stack((plus_b, -plus_b), axis=1)
        returns_a = rng_a.normal(size=(8, 2))
        returns_b = rng_b.normal(size=(8, 2))
        estimate_a = estimate_lopo_population(
            theta,
            signed_a,
            returns_a,
            sigma,
            basis,
            random_basis,
            basis_provenance=provenance,
        )
        q = 0.5
        alpha = calibrate_locality_rate(estimate_a.gradient, sigma, q)
        common = {
            "theta": theta,
            "sigma": sigma,
            "basis": basis,
            "random_basis": random_basis,
            "q_values": (q,),
            "alphas": (alpha,),
            "provenance": provenance,
        }
        product_a = _population_product(
            signed_noise=signed_a,
            paired_returns=returns_a,
            reference_curvature=None,
            reference_actions=None,
            **common,
        )
        product_b_own = _population_product(
            signed_noise=signed_b,
            paired_returns=returns_b,
            reference_curvature=product_a.curvature,
            reference_actions=product_a.actions,
            **common,
        )
        reference = FrozenEndpointReference(
            signed_noise=signed_a,
            utilities=product_a.arrays["utilities"],
            gradient=product_a.gradient,
            curvature=product_a.curvature,
            origin_metrics={
                key: float(product_a.arrays[key])
                for key in (
                    "gradient_endpoint_relative_error",
                    "subspace_jacobian_relative_error",
                    "self_normalized_gradient_relative_error",
                    "self_normalized_jacobian_relative_error",
                )
            },
        )
        product_b = _population_product(
            signed_noise=signed_b,
            paired_returns=returns_b,
            reference_curvature=product_a.curvature,
            reference_actions=product_a.actions,
            endpoint_reference=reference,
            **common,
        )

        self.assertTrue(np.array_equal(product_b.gradient, product_b_own.gradient))
        self.assertTrue(np.array_equal(product_b.curvature, product_b_own.curvature))
        self.assertTrue(
            np.array_equal(
                product_b.steps[q]["structured"],
                product_b_own.steps[q]["structured"],
            )
        )
        expected = frozen_endpoint_diagnostics(
            signed_a.reshape(-1, theta.size),
            product_a.arrays["utilities"].reshape(-1),
            product_b.steps[q]["structured"],
            sigma,
            basis,
            product_a.curvature,
        )
        own = frozen_endpoint_diagnostics(
            signed_b.reshape(-1, theta.size),
            product_b.arrays["utilities"].reshape(-1),
            product_b.steps[q]["structured"],
            sigma,
            basis,
            product_b.curvature,
        )
        self.assertIsNotNone(expected.full_linearization_residual)
        self.assertIsNotNone(own.full_linearization_residual)
        self.assertAlmostEqual(
            float(product_b.arrays["r_full"][0, 0]),
            float(expected.full_linearization_residual),
            places=13,
        )
        self.assertGreater(
            abs(
                float(product_b.arrays["r_full"][0, 0])
                - float(own.full_linearization_residual)
            ),
            1e-6,
        )
        for key in reference.origin_metrics:
            self.assertEqual(float(product_b.arrays[key]), float(product_a.arrays[key]))

    def test_negative_count_uses_exact_stored_near_zero_eigensystem(self) -> None:
        theta = np.zeros(3, dtype=np.float64)
        plus = np.random.default_rng(3).normal(size=(3, 3))
        signed = np.stack((plus, -plus), axis=1)
        returns = np.asarray([[0.0, 2.0], [1.0, 4.0], [3.0, 5.0]])
        basis = np.eye(3)
        provenance = BasisProvenance.strictly_lagged("prior", "random-prior")
        estimate = estimate_lopo_population(
            theta,
            signed,
            returns,
            0.1,
            basis,
            basis,
            basis_provenance=provenance,
        )
        alpha = calibrate_locality_rate(estimate.gradient, 0.1, 0.5)
        product = _population_product(
            theta=theta,
            signed_noise=signed,
            paired_returns=returns,
            sigma=0.1,
            basis=basis,
            random_basis=basis,
            q_values=(0.5,),
            alphas=(alpha,),
            provenance=provenance,
            reference_curvature=None,
            reference_actions=None,
        )
        stored = np.asarray(product.arrays["curvature_eigenvalues"])[0]
        stored_count = int(np.count_nonzero(stored < 0.0))
        fresh_count = int(np.count_nonzero(np.linalg.eigvalsh(product.curvature) < 0.0))
        self.assertNotEqual(stored_count, fresh_count)
        self.assertEqual(
            int(np.asarray(product.arrays["negative_eigenvalue_count"])),
            stored_count,
        )
        self.assertTrue(bool(product.arrays["projection_boundary_unresolved"][0]))

    def test_all_tied_bank_a_is_retained_as_unresolved_zero_step_record(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manifest, state, config_path, checkpoint_path, capture_path = (
                self._state_and_inputs(root)
            )
            artifact_root = os.path.join(root, "artifacts")
            index_path = produce_checkpoint_diagnostic(
                manifest=manifest,
                manifest_sha256="c" * 64,
                state=state,
                training_config_path=config_path,
                checkpoint_path=checkpoint_path,
                capture_manifest_path=capture_path,
                task_index=0,
                training_seed=7,
                generation=10,
                artifact_root=artifact_root,
                evaluator=ConstantReturnEvaluator(),
                chunk_pairs=2,
            )
            with open(index_path, encoding="utf-8") as stream:
                index = json.load(stream)
            zero_hash = _array_sha256(np.zeros(10, dtype=np.float64))
            for record in index["banks"] + index["partitions"]:
                for summary in record["q_summaries"]:
                    self.assertEqual(summary["alpha"], 0.0)
                    self.assertFalse(summary["alpha_resolved"])
                    self.assertEqual(
                        summary["alpha_unresolved_reason"],
                        "bank_a_gradient_exact_zero",
                    )
                    for field in (
                        "structured_norm",
                        "isotropic_norm",
                        "explicit_norm",
                        "random_norm",
                        "random_raw_norm",
                        "anisotropic_action_norm",
                        "structured_step_over_sigma",
                    ):
                        self.assertEqual(summary[field], 0.0)
                    self.assertEqual(set(summary["action_sha256"].values()), {zero_hash})
            self.assertEqual(len(index["endpoints"]), 3 * 2 * 4 * 2)

            basis_path = os.path.join(
                artifact_root, index["checkpoint"]["basis_artifact_path"]
            )
            with np.load(basis_path, allow_pickle=False) as archive:
                basis = np.asarray(archive["primary_basis"])
                random_basis = np.asarray(archive["random_basis"])
            reference = None
            analyzer_issues: list[str] = []
            for bank_record in index["banks"]:
                with np.load(
                    os.path.join(artifact_root, bank_record["raw_bank_path"]),
                    allow_pickle=False,
                ) as archive:
                    raw = {name: np.asarray(archive[name]) for name in archive.files}
                self.assertNotIn("signed_noise", raw)
                plus = np.stack(
                    [
                        np.random.Generator(
                            np.random.PCG64(int(seed))
                        ).standard_normal(10)
                        for seed in raw["perturbation_seeds"]
                    ]
                )
                signed = np.stack((plus, -plus), axis=1)
                with np.load(
                    os.path.join(artifact_root, bank_record["diagnostics_path"]),
                    allow_pickle=False,
                ) as archive:
                    diagnostics = {
                        name: np.asarray(archive[name]) for name in archive.files
                    }
                recomputed = _validate_diagnostic_data(
                    diagnostics,
                    label=f"all-tied bank {bank_record['bank']}",
                    manifest=manifest,
                    theta=state.center,
                    signed_noise=signed,
                    paired_returns=raw["paired_returns"],
                    basis=basis,
                    random_basis=random_basis,
                    q_summaries=bank_record["q_summaries"],
                    reference_curvature=(
                        None if reference is None else reference["curvature"]
                    ),
                    reference_actions=(
                        None if reference is None else reference["actions"]
                    ),
                    issues=analyzer_issues,
                    endpoint_reference=(
                        None if reference is None else reference["endpoint_reference"]
                    ),
                )
                self.assertIsNotNone(recomputed)
                if reference is None:
                    reference = recomputed
            self.assertEqual(analyzer_issues, [])
            self.assertIsNotNone(reference)
            self.assertFalse(reference["primary_diagnostics_resolved"])

    def test_evaluator_failure_leaves_no_committed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            manifest, state, config_path, checkpoint_path, capture_path = (
                self._state_and_inputs(root)
            )
            artifact_root = os.path.join(root, "artifacts")
            with self.assertRaisesRegex(RuntimeError, "injected evaluator failure"):
                produce_checkpoint_diagnostic(
                    manifest=manifest,
                    manifest_sha256="c" * 64,
                    state=state,
                    training_config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    capture_manifest_path=capture_path,
                    task_index=0,
                    training_seed=7,
                    generation=10,
                    artifact_root=artifact_root,
                    evaluator=RecordingEvaluator(fail_call=2),
                    chunk_pairs=2,
                )
            parent = os.path.join(artifact_root, "checkpoint_artifacts")
            self.assertTrue(os.path.isdir(parent))
            self.assertEqual(os.listdir(parent), [])


if __name__ == "__main__":
    unittest.main()

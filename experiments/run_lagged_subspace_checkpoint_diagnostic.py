#!/usr/bin/env python3
"""Produce one frozen-checkpoint lagged-subspace diagnostic artifact.

The producer deliberately contains its own manifest seed derivation and raw
artifact hashing.  The final analyzer independently regenerates these values;
the producer never imports analyzer helpers or performs inferential tests.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.lagged_subspace_diagnostic import (
    BasisProvenance,
    LaggedSubspaceDiagnostic,
    analyze_lagged_subspace_population,
    build_lagged_bases,
    calibrate_locality_rate,
    compute_action_metrics,
    compute_four_steps,
    estimate_lopo_population,
    frozen_endpoint_diagnostics,
    recompute_eigen_action_jackknife,
    summarize_locality,
)
from experiments.lagged_subspace_study_lock import (
    require_lagged_subspace_study_source_lock,
    validate_manifest_mapping,
)


STUDY = "lagged_subspace_frozen_checkpoint"
CHECKPOINT_KEYS = {
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
ARMS = ("structured", "isotropic", "explicit", "random")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FLOAT_EPS = np.finfo(np.float64).eps


class BatchEvaluator(Protocol):
    """Dependency-injected policy evaluator used by production and tests."""

    def validate_policy(
        self,
        expected_dimension: int,
        expected_block_sizes: Sequence[int],
        expected_observation_dim: int,
        expected_action_dim: int,
    ) -> None: ...

    def evaluate_batch(
        self, parameters: np.ndarray, rollout_seeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class CheckpointState:
    center: np.ndarray
    obs_mean: np.ndarray
    obs_var: np.ndarray
    obs_count: float
    gradient_generations: np.ndarray
    gradients: np.ndarray
    checkpoint_sha256: str
    capture_manifest_sha256: str
    training_config_sha256: str
    source_sha256: str


@dataclass(frozen=True)
class PopulationProduct:
    arrays: Mapping[str, np.ndarray]
    q_summaries: tuple[Mapping[str, Any], ...]
    gradient: np.ndarray
    curvature: np.ndarray
    actions: Mapping[float, np.ndarray]
    steps: Mapping[float, Mapping[str, np.ndarray]]
    utility_sum: float
    utility_abs_sum: float


@dataclass(frozen=True)
class FrozenEndpointReference:
    """Complete Bank-A empirical map used for every frozen-map diagnostic."""

    signed_noise: np.ndarray
    utilities: np.ndarray
    gradient: np.ndarray
    curvature: np.ndarray
    origin_metrics: Mapping[str, float]


ORIGIN_DIAGNOSTIC_KEYS = (
    "gradient_endpoint_relative_error",
    "subspace_jacobian_relative_error",
    "self_normalized_gradient_relative_error",
    "self_normalized_jacobian_relative_error",
)


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _fixed_ascii_sha256_matches(value: np.ndarray, expected: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.dtype("S64"):
        return False
    try:
        decoded = array.item().decode("ascii")
    except (AttributeError, UnicodeDecodeError):
        return False
    return decoded == expected


def _labeled_arrays_sha256(values: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for label, value in values:
        encoded = str(label).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_array_sha256(np.asarray(value))))
    return digest.hexdigest()


def _record_with_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    def native(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [native(item) for item in value.tolist()]
        if isinstance(value, Mapping):
            return {str(key): native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        return value

    result = native(dict(record))
    if "record_sha256" in result:
        raise ValueError("record already contains record_sha256")
    result["record_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _write_bytes_atomic(path: str, payload: bytes) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: str, value: Any) -> None:
    _write_bytes_atomic(path, _canonical_bytes(value) + b"\n")


def _write_npz_atomic(path: str, **arrays: np.ndarray) -> None:
    temporary = f"{path}.tmp.{os.getpid()}.npz"
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            if "/" in name or "\\" in name:
                raise ValueError("NPZ array names cannot contain path separators")
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            entry = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            entry.compress_type = zipfile.ZIP_STORED
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())
    os.replace(temporary, path)


def derive_producer_seed(
    manifest: Mapping[str, Any], namespace: str, *coordinates: Any
) -> int:
    """Independent implementation of the locked SHA256 uint64 seed scheme."""

    rng = manifest["rng"]
    namespace_id = rng["namespaces"][namespace]
    payload = [manifest["study"], rng["master_seed"], namespace_id, *coordinates]
    digest = hashlib.sha256(_canonical_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def checkpoint_id_for(
    manifest: Mapping[str, Any], task_index: int, training_seed: int, generation: int
) -> int:
    seeds = list(manifest["training_seeds"])
    generations = list(manifest["checkpoint_generations"])
    if task_index not in range(len(manifest["tasks"])):
        raise ValueError("task index is outside the manifest")
    if training_seed not in seeds or generation not in generations:
        raise ValueError("training seed or generation is outside the manifest")
    training_id = task_index * len(seeds) + seeds.index(training_seed)
    return training_id * len(generations) + generations.index(generation)


def _checkpoint_seed(
    manifest: Mapping[str, Any], namespace: str, task: int, seed: int, generation: int
) -> int:
    return derive_producer_seed(manifest, namespace, task, seed, generation)


def _pair_seed(
    manifest: Mapping[str, Any],
    namespace: str,
    task: int,
    seed: int,
    generation: int,
    bank: str,
    pair_index: int,
) -> int:
    bank_index = list(manifest["dimensions"]["banks"]).index(bank)
    return derive_producer_seed(
        manifest,
        namespace,
        task,
        seed,
        generation,
        bank_index,
        pair_index,
    )


def _endpoint_seed(
    manifest: Mapping[str, Any],
    task: int,
    seed: int,
    generation: int,
    episode_index: int,
) -> int:
    return derive_producer_seed(
        manifest, "endpoint", task, seed, generation, episode_index
    )


def bank_b_partitions(
    manifest: Mapping[str, Any], task: int, seed: int, generation: int
) -> list[list[int]]:
    """Create the locked disjoint partition without importing analyzer code."""

    dims = manifest["dimensions"]
    partition_seed = _checkpoint_seed(
        manifest, "bank_b_partition", task, seed, generation
    )

    def key(pair_index: int) -> tuple[bytes, int]:
        return (
            hashlib.sha256(_canonical_bytes([partition_seed, pair_index])).digest(),
            pair_index,
        )

    permutation = sorted(range(int(dims["pairs_per_bank"])), key=key)
    width = int(dims["pairs_per_partition"])
    count = int(dims["bank_b_partition_count"])
    result = [permutation[index * width : (index + 1) * width] for index in range(count)]
    flattened = [item for row in result for item in row]
    if sorted(flattened) != list(range(int(dims["pairs_per_bank"]))) or len(
        flattened
    ) != len(set(flattened)):
        raise RuntimeError("Bank-B partitions are not complete and disjoint")
    return result


def load_locked_manifest(path: str, expected_sha256: str) -> tuple[dict[str, Any], str]:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"manifest SHA256 mismatch: expected {expected_sha256}, found {actual}"
        )
    with open(path, "r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != 1 or manifest.get("study") != STUDY:
        raise ValueError("manifest study or schema is invalid")
    if manifest.get("protocol_status") != "final_locked_before_environment_outcomes":
        raise ValueError("manifest protocol status is not final and pre-outcome locked")
    protocol = manifest.get("protocol", {})
    protocol_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        str(protocol.get("path", "")),
    )
    if (
        not os.path.isfile(protocol_path)
        or protocol.get("sha256") != _sha256_file(protocol_path)
    ):
        raise ValueError("manifest embedded protocol digest is stale")
    dims = manifest.get("dimensions", {})
    required = {
        "noise_std",
        "lagged_gradient_count",
        "banks",
        "pairs_per_bank",
        "bank_b_partition_count",
        "pairs_per_partition",
        "locality_q",
        "endpoint_arms",
        "endpoint_episodes",
        "layer_blocks",
    }
    if not required.issubset(dims):
        raise ValueError("manifest diagnostic dimensions are incomplete")
    if dims["banks"] != ["A", "B"] or dims["endpoint_arms"] != list(ARMS):
        raise ValueError("manifest bank or endpoint-arm mapping is invalid")
    if int(dims["pairs_per_bank"]) != int(dims["bank_b_partition_count"]) * int(
        dims["pairs_per_partition"]
    ):
        raise ValueError("manifest Bank-B partition sizes do not cover the bank")
    if int(dims["lagged_gradient_count"]) != 10 or int(dims["layer_blocks"]) != 3:
        raise ValueError("manifest lagged archive or layer-block count is invalid")
    if manifest.get("prohibitions") and set(manifest["prohibitions"].values()) != {False}:
        raise ValueError("a prohibited mechanism is enabled in the manifest")
    if manifest.get("analysis", {}).get("zero_gradient_calibration") != {
        "detection": "scaled_l2_exact_zero_only",
        "alpha_sentinel": 0.0,
        "unresolved_reason": "bank_a_gradient_exact_zero",
        "all_four_steps_zero": True,
        "gate_policy": "fail_all_affected_task_conditions",
    }:
        raise ValueError("manifest zero-gradient calibration policy is invalid")
    for index, task in enumerate(manifest.get("tasks", [])):
        block_sizes = task.get("policy_block_sizes")
        block_ranges = task.get("policy_block_ranges")
        if (
            task.get("task_index") != index
            or not isinstance(block_sizes, list)
            or len(block_sizes) != 3
            or not isinstance(block_ranges, list)
            or block_ranges
            != [
                [0, block_sizes[0]],
                [block_sizes[0], block_sizes[0] + block_sizes[1]],
                [sum(block_sizes[:2]), sum(block_sizes)],
            ]
            or task.get("parameter_count") != sum(block_sizes)
            or not isinstance(task.get("observation_dim"), int)
            or not isinstance(task.get("action_dim"), int)
        ):
            raise ValueError(f"manifest task {index} policy dimensions are invalid")
    rng = manifest.get("rng", {})
    if (
        rng.get("scheme") != "sha256_uint64_v1"
        or rng.get("serialization") != "canonical_json_utf8"
        or rng.get("uint64_extraction") != "first_8_digest_bytes_big_endian"
        or rng.get("basis_gaussian")
        != "numpy_generator_pcg64_conditional_standard_normal_by_zero_block_v1"
        or rng.get("random_signed_permutation")
        != (
            "numpy_generator_pcg64_permutation_then_int64_rademacher_"
            "and_renormalize_by_block_v1"
        )
        or rng.get("perturbation_gaussian")
        != "numpy_generator_pcg64_standard_normal_per_pair_v1"
    ):
        raise ValueError("manifest producer RNG contract is invalid")
    return manifest, actual


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _require_equal(config: Mapping[str, Any], key: str, expected: Any) -> None:
    if key not in config or config[key] != expected:
        raise ValueError(f"training config {key} is not locked to {expected!r}")


def _validate_training_config(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    task_index: int,
    training_seed: int,
) -> None:
    dims = manifest["dimensions"]
    task = manifest["tasks"][task_index]
    exact = {
        "checkpoint_capture_protocol": "lagged_subspace_frozen_checkpoint_v1",
        "env_name": task["env_name"],
        "seed": training_seed,
        "population_size": int(dims.get("population_size", 200)),
        "learning_rate": float(manifest["training"]["learning_rate"]),
        "lr_schedule": "constant",
        "noise_std": float(dims["noise_std"]),
        "l2_coeff": 0.0,
        "rank_fitness": True,
        "antithetic": True,
        "max_grad_norm": 0.0,
        "max_param_norm": None,
        "n_iterations": int(dims.get("training_updates", 250)),
        "online_evaluation_enabled": False,
        "eval_episodes": 0,
        "use_obs_norm": True,
        "obs_norm_mode": "frozen_after_calibration",
        "obs_norm_calibration_episodes": int(dims.get("calibration_episodes", 3)),
        "heldout_evaluation_enabled": False,
        "checkpoint_capture_generations": list(manifest["checkpoint_generations"]),
        "checkpoint_gradient_archive_length": int(dims["lagged_gradient_count"]),
        "replay_enabled": False,
        "buffer_size": 0,
        "reuse_fraction": 0.0,
        "common_rollout_seed": True,
        "evaluate_center_fitness": False,
    }
    for key, expected in exact.items():
        _require_equal(config, key, expected)
    if config.get("condition", "standard_es") != "standard_es":
        raise ValueError("checkpoint training condition is not Standard ES")
    if config.get("algorithm", "standard_es") != "standard_es":
        raise ValueError("checkpoint training algorithm is not Standard ES")
    forbidden = {
        "trust_radius",
        "use_trust_radius_for_standard_es",
        "picard_iteration",
        "importance_sampling",
        "gradient_clipping",
        "parameter_projection",
        "optimizer_momentum",
    }
    present = sorted(forbidden.intersection(config))
    if present:
        raise ValueError("forbidden checkpoint config keys are present: " + ", ".join(present))


def load_and_validate_checkpoint(
    *,
    checkpoint_path: str,
    capture_manifest_path: str,
    training_config_path: str,
    manifest: Mapping[str, Any],
    task_index: int,
    training_seed: int,
    generation: int,
    expected_source_sha256: str,
) -> tuple[CheckpointState, dict[str, Any]]:
    """Fail closed on checkpoint lineage before any current noise is generated."""

    capture = _load_json(capture_manifest_path)
    config = _load_json(training_config_path)
    _validate_training_config(config, manifest, task_index, training_seed)
    if capture.get("schema_version") != 1 or capture.get("status") != "complete":
        raise ValueError("checkpoint capture manifest is not complete schema version 1")
    generations = list(manifest["checkpoint_generations"])
    if (
        capture.get("requested_generations") != generations
        or capture.get("captured_generations") != generations
        or capture.get("checkpoint_count") != len(generations)
        or capture.get("gradient_archive_length") != 10
        or capture.get("reward_selection_used") is not False
        or capture.get("current_generation_gradient_excluded") is not True
        or capture.get("online_evaluation_enabled") is not False
    ):
        raise ValueError("checkpoint capture completeness or no-selection controls are invalid")
    controls = capture.get("validated_generator_controls")
    required_controls = {
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
    }
    if controls != required_controls:
        raise ValueError("checkpoint generator controls are not exactly the fresh Standard-ES design")
    if capture.get("source_sha256") != expected_source_sha256:
        raise ValueError("checkpoint source digest does not match the enforced source lock")
    training_config_sha256 = _sha256_file(training_config_path)
    if capture.get("training_config_sha256") != training_config_sha256:
        raise ValueError("checkpoint training-config digest is invalid")

    artifacts = capture.get("artifacts")
    matches = [
        item
        for item in artifacts if isinstance(artifacts, list) and isinstance(item, dict)
        and item.get("checkpoint_generation") == generation
    ] if isinstance(artifacts, list) else []
    if len(matches) != 1:
        raise ValueError("capture manifest does not contain exactly one requested checkpoint")
    metadata = matches[0]
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if metadata.get("artifact_sha256") != checkpoint_sha256:
        raise ValueError("checkpoint artifact digest does not match its capture manifest")
    captured_path = os.path.realpath(
        os.path.join(os.path.dirname(capture_manifest_path), str(metadata.get("artifact", "")))
    )
    if captured_path != os.path.realpath(checkpoint_path):
        raise ValueError("checkpoint path does not match capture-manifest lineage")

    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if set(archive.files) != CHECKPOINT_KEYS:
            raise ValueError("checkpoint NPZ schema is not exact")
        data = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if data["schema_version"].shape != () or int(data["schema_version"]) != 2:
        raise ValueError("checkpoint schema version is invalid")
    if data["checkpoint_generation"].shape != () or int(
        data["checkpoint_generation"]
    ) != generation:
        raise ValueError("checkpoint generation is invalid")
    archive_source = np.asarray(data["study_source_sha256"])
    archive_config = np.asarray(data["training_config_sha256"])
    if not _fixed_ascii_sha256_matches(
        archive_source, expected_source_sha256
    ) or not _fixed_ascii_sha256_matches(
        archive_config, training_config_sha256
    ):
        raise ValueError(
            "checkpoint-embedded source or training-config digest is invalid"
        )
    center = np.asarray(data["center_params"], dtype=np.float64)
    gradients = np.asarray(data["proposal_gradients"], dtype=np.float64)
    gradient_generations = np.asarray(data["gradient_generations"])
    expected_dimension = int(manifest["tasks"][task_index]["parameter_count"])
    expected_indices = np.arange(generation - 10, generation, dtype=np.int64)
    if (
        center.shape != (expected_dimension,)
        or gradients.shape != (10, expected_dimension)
        or gradient_generations.dtype.kind not in "iu"
        or not np.array_equal(gradient_generations, expected_indices)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(gradients))
    ):
        raise ValueError("checkpoint parameters or chronological prior gradients are invalid")
    enabled = np.asarray(data["obs_normalizer_enabled"])
    obs_mean = np.asarray(data["obs_mean"], dtype=np.float64)
    obs_var = np.asarray(data["obs_var"], dtype=np.float64)
    obs_count_array = np.asarray(data["obs_count"], dtype=np.float64)
    observation_dimension = int(manifest["tasks"][task_index]["observation_dim"])
    if (
        enabled.shape != ()
        or bool(enabled) is not True
        or obs_mean.shape != (observation_dimension,)
        or obs_var.shape != obs_mean.shape
        or obs_count_array.shape != ()
        or not np.all(np.isfinite(obs_mean))
        or not np.all(np.isfinite(obs_var))
        or np.any(obs_var < 0.0)
        or not np.isfinite(float(obs_count_array))
        or float(obs_count_array) <= 0.0
    ):
        raise ValueError("checkpoint frozen observation-normalizer state is invalid")

    expected_gradient_hashes = [_array_sha256(row) for row in gradients]
    expected_parameter_hash = _array_sha256(center)
    expected_obs_hash = _labeled_arrays_sha256(
        [
            ("enabled", enabled),
            ("mean", obs_mean),
            ("var", obs_var),
            ("count", obs_count_array),
        ]
    )
    expected_archive_hash = _labeled_arrays_sha256(
        [("gradient_generations", gradient_generations), ("proposal_gradients", gradients)]
    )
    if (
        metadata.get("center_params_sha256") != expected_parameter_hash
        or metadata.get("observation_normalizer_state_sha256") != expected_obs_hash
        or metadata.get("indexed_gradient_archive_sha256") != expected_archive_hash
        or metadata.get("proposal_gradient_hashes")
        != [
            {"generation": int(index), "sha256": digest}
            for index, digest in zip(expected_indices, expected_gradient_hashes, strict=True)
        ]
        or metadata.get("strictly_prior_gradient_archive") is not True
        or metadata.get("current_checkpoint_gradient_included") is not False
        or metadata.get("last_applied_gradient_generation") != generation - 1
        or metadata.get("source_sha256") != expected_source_sha256
        or metadata.get("training_config_sha256") != training_config_sha256
    ):
        raise ValueError("checkpoint content hashes or strict-lag metadata are invalid")
    state = CheckpointState(
        center=center,
        obs_mean=obs_mean,
        obs_var=obs_var,
        obs_count=float(obs_count_array),
        gradient_generations=gradient_generations.astype(np.int64, copy=False),
        gradients=gradients,
        checkpoint_sha256=checkpoint_sha256,
        capture_manifest_sha256=_sha256_file(capture_manifest_path),
        training_config_sha256=training_config_sha256,
        source_sha256=expected_source_sha256,
    )
    return state, config


class MujocoBatchEvaluator:
    """Existing repository rollout helper wrapped in a batched interface."""

    def __init__(
        self,
        config: Mapping[str, Any],
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
        *,
        n_workers: int,
    ) -> None:
        from experiments.train import _evaluate_params, _init_worker, _make_env, make_policy
        from core.policies import make_layer_slices

        self._evaluate_params = _evaluate_params
        self._obs_mean = np.asarray(obs_mean, dtype=np.float64).copy()
        self._obs_var = np.asarray(obs_var, dtype=np.float64).copy()
        self._obs_scale = float(config.get("obs_scale", 1.0))
        self._config = dict(config)
        probe = _make_env(
            self._config["env_name"],
            self._config.get("env_kwargs"),
            frame_stack=self._config.get("frame_stack", 1),
            fire_reset=self._config.get("fire_reset", False),
            fire_reset_steps=self._config.get("fire_reset_steps"),
            fire_on_life_loss=self._config.get("fire_on_life_loss", False),
            action_indices=self._config.get("action_indices"),
        )
        try:
            policy = make_policy(self._config, probe)
            self._dimension = int(policy.num_params)
            self._observation_dimension = int(policy.ob_dim)
            self._action_dimension = int(policy.ac_dim)
            self._block_sizes = tuple(
                int(block.stop - block.start) for block in make_layer_slices(policy)
            )
        finally:
            probe.close()
        self._pool: Pool | None
        if n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if n_workers == 1:
            _init_worker(self._config)
            self._pool = None
        else:
            self._pool = Pool(n_workers, initializer=_init_worker, initargs=(self._config,))

    def validate_policy(
        self,
        expected_dimension: int,
        expected_block_sizes: Sequence[int],
        expected_observation_dim: int,
        expected_action_dim: int,
    ) -> None:
        if self._dimension != expected_dimension or self._block_sizes != tuple(
            int(value) for value in expected_block_sizes
        ) or self._observation_dimension != expected_observation_dim or (
            self._action_dimension != expected_action_dim
        ):
            raise ValueError(
                "live policy parameterization does not match manifest layer blocks"
            )

    def evaluate_batch(
        self, parameters: np.ndarray, rollout_seeds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float64)
        rollout_seeds = np.asarray(rollout_seeds, dtype=np.uint64)
        if parameters.ndim != 2 or rollout_seeds.shape != (parameters.shape[0],):
            raise ValueError("evaluation parameters and seeds have inconsistent shapes")
        tasks = [
            (
                parameters[index],
                int(rollout_seeds[index]),
                self._obs_mean,
                self._obs_var,
                False,
                self._obs_scale,
            )
            for index in range(parameters.shape[0])
        ]
        if self._pool is None:
            results = [self._evaluate_params(task) for task in tasks]
        else:
            results = self._pool.map(self._evaluate_params, tasks)
        returns = np.asarray([item[0] for item in results], dtype=np.float64)
        transitions = np.asarray([item[2] for item in results], dtype=np.int64)
        if not np.all(np.isfinite(returns)) or np.any(transitions <= 0):
            raise FloatingPointError("rollout returned nonfinite reward or invalid transitions")
        return returns, transitions

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "MujocoBatchEvaluator":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _evaluate_bank(
    *,
    evaluator: BatchEvaluator,
    manifest: Mapping[str, Any],
    task_index: int,
    training_seed: int,
    generation: int,
    bank: str,
    center: np.ndarray,
    chunk_pairs: int,
) -> dict[str, np.ndarray]:
    dims = manifest["dimensions"]
    count = int(dims["pairs_per_bank"])
    dimension = center.size
    sigma = float(dims["noise_std"])
    perturbation_seeds = np.asarray(
        [
            _pair_seed(
                manifest,
                "bank_perturbation",
                task_index,
                training_seed,
                generation,
                bank,
                index,
            )
            for index in range(count)
        ],
        dtype=np.uint64,
    )
    rollout_seeds = np.asarray(
        [
            _pair_seed(
                manifest,
                "bank_rollout",
                task_index,
                training_seed,
                generation,
                bank,
                index,
            )
            for index in range(count)
        ],
        dtype=np.uint64,
    )
    if len(set(map(int, perturbation_seeds))) != count or len(
        set(map(int, rollout_seeds))
    ) != count:
        raise RuntimeError("bank seed derivation produced a collision")
    signed_noise = np.empty((count, 2, dimension), dtype=np.float64)
    for index, perturbation_seed in enumerate(perturbation_seeds):
        epsilon = np.random.Generator(
            np.random.PCG64(int(perturbation_seed))
        ).standard_normal(dimension)
        signed_noise[index, 0] = epsilon
        signed_noise[index, 1] = -epsilon
    if not np.array_equal(signed_noise[:, 1], -signed_noise[:, 0]):
        raise RuntimeError("constructed perturbations are not exactly antithetic")

    paired_returns = np.empty((count, 2), dtype=np.float64)
    paired_transitions = np.empty((count, 2), dtype=np.int64)
    if chunk_pairs <= 0:
        raise ValueError("chunk_pairs must be positive")
    for start in range(0, count, chunk_pairs):
        stop = min(start + chunk_pairs, count)
        parameters = center[None, :] + sigma * signed_noise[start:stop].reshape(
            -1, dimension
        )
        seeds = np.repeat(rollout_seeds[start:stop], 2)
        returns, transitions = evaluator.evaluate_batch(parameters, seeds)
        expected = 2 * (stop - start)
        if returns.shape != (expected,) or transitions.shape != (expected,):
            raise ValueError("evaluator returned an incomplete bank chunk")
        paired_returns[start:stop] = returns.reshape(-1, 2)
        paired_transitions[start:stop] = transitions.reshape(-1, 2)
    return {
        "signed_noise": signed_noise,
        "paired_returns": paired_returns,
        "paired_transitions": paired_transitions,
        "perturbation_seeds": perturbation_seeds,
        "rollout_seeds_plus": rollout_seeds,
        "rollout_seeds_minus": rollout_seeds.copy(),
    }


def _analysis_from_estimate(
    first: LaggedSubspaceDiagnostic,
    theta: np.ndarray,
    sigma: float,
    basis: np.ndarray,
    random_basis: np.ndarray,
    alpha: float,
    endpoint_reference: FrozenEndpointReference,
) -> LaggedSubspaceDiagnostic:
    estimate = first.estimate
    steps = compute_four_steps(
        theta,
        estimate.gradient,
        estimate.curvature,
        basis,
        estimate.random_curvature,
        random_basis,
        alpha,
    )
    step_map = {name: getattr(steps, name) for name in ARMS}
    utilities = np.asarray(endpoint_reference.utilities, dtype=np.float64).reshape(-1)
    flattened_noise = np.asarray(
        endpoint_reference.signed_noise, dtype=np.float64
    ).reshape(-1, theta.size)
    return LaggedSubspaceDiagnostic(
        estimate=estimate,
        steps=steps,
        locality={name: summarize_locality(step, sigma) for name, step in step_map.items()},
        action_metrics=compute_action_metrics(steps, alpha),
        nonlinear_jackknife=recompute_eigen_action_jackknife(
            estimate, theta, basis, random_basis, alpha
        ),
        endpoint_diagnostics={
            name: frozen_endpoint_diagnostics(
                flattened_noise,
                utilities,
                step,
                sigma,
                basis,
                endpoint_reference.curvature,
            )
            for name, step in step_map.items()
        },
        gradient_endpoint_relative_error=endpoint_reference.origin_metrics[
            "gradient_endpoint_relative_error"
        ],
        subspace_jacobian_relative_error=endpoint_reference.origin_metrics[
            "subspace_jacobian_relative_error"
        ],
        self_normalized_gradient_relative_error=endpoint_reference.origin_metrics[
            "self_normalized_gradient_relative_error"
        ],
        self_normalized_jacobian_relative_error=endpoint_reference.origin_metrics[
            "self_normalized_jacobian_relative_error"
        ],
        basis_provenance=first.basis_provenance,
        claim_metadata=first.claim_metadata,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= FLOAT_EPS:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _resolved_float(value: float | None) -> float:
    return 0.0 if value is None else float(value)


def _stable_norm(vector: np.ndarray) -> float:
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    return float(scale * np.linalg.norm(vector / scale))


def _population_product(
    *,
    theta: np.ndarray,
    signed_noise: np.ndarray,
    paired_returns: np.ndarray,
    sigma: float,
    basis: np.ndarray,
    random_basis: np.ndarray,
    q_values: Sequence[float],
    alphas: Sequence[float],
    provenance: BasisProvenance,
    reference_curvature: np.ndarray | None,
    reference_actions: Mapping[float, np.ndarray] | None,
    endpoint_reference: FrozenEndpointReference | None = None,
    alpha_resolved: bool = True,
    alpha_unresolved_reason: str | None = None,
) -> PopulationProduct:
    if len(q_values) != len(alphas) or not q_values:
        raise ValueError("q values and locked alphas are inconsistent")
    if not isinstance(alpha_resolved, (bool, np.bool_)):
        raise ValueError("alpha_resolved must be boolean")
    alpha_resolved = bool(alpha_resolved)
    if alpha_resolved:
        if alpha_unresolved_reason is not None or any(
            not np.isfinite(alpha) or float(alpha) <= 0.0 for alpha in alphas
        ):
            raise ValueError("resolved locality rates must be finite and positive")
    elif (
        not isinstance(alpha_unresolved_reason, str)
        or not alpha_unresolved_reason
        or any(float(alpha) != 0.0 for alpha in alphas)
    ):
        raise ValueError("unresolved locality rates require zero sentinels and a reason")
    own_endpoint_reference = endpoint_reference is None
    first = analyze_lagged_subspace_population(
        theta,
        signed_noise,
        paired_returns,
        sigma,
        basis,
        random_basis,
        float(alphas[0]),
        basis_provenance=provenance,
        endpoint_reference_noise=(
            None if endpoint_reference is None else endpoint_reference.signed_noise
        ),
        endpoint_reference_utilities=(
            None if endpoint_reference is None else endpoint_reference.utilities
        ),
        endpoint_reference_curvature=(
            None if endpoint_reference is None else endpoint_reference.curvature
        ),
        endpoint_reference_gradient=(
            None if endpoint_reference is None else endpoint_reference.gradient
        ),
    )
    estimate = first.estimate
    if reference_curvature is None:
        reference_curvature = estimate.curvature
    if endpoint_reference is None:
        endpoint_reference = FrozenEndpointReference(
            signed_noise=np.asarray(signed_noise),
            utilities=estimate.utilities,
            gradient=estimate.gradient,
            curvature=estimate.curvature,
            origin_metrics={
                key: float(getattr(first, key)) for key in ORIGIN_DIAGNOSTIC_KEYS
            },
        )
    reference_signed_noise = np.asarray(
        endpoint_reference.signed_noise, dtype=np.float64
    )
    reference_utilities = np.asarray(endpoint_reference.utilities, dtype=np.float64)
    reference_endpoint_curvature = np.asarray(
        endpoint_reference.curvature, dtype=np.float64
    )
    reference_endpoint_gradient = np.asarray(
        endpoint_reference.gradient, dtype=np.float64
    )
    if (
        reference_signed_noise.ndim != 3
        or reference_signed_noise.shape[1:] != (2, theta.size)
        or reference_utilities.shape != reference_signed_noise.shape[:2]
        or reference_endpoint_gradient.shape != theta.shape
        or reference_endpoint_curvature.shape != (3, 3)
        or set(endpoint_reference.origin_metrics) != set(ORIGIN_DIAGNOSTIC_KEYS)
        or any(
            not np.isfinite(endpoint_reference.origin_metrics[key])
            for key in ORIGIN_DIAGNOSTIC_KEYS
        )
    ):
        raise ValueError("frozen endpoint reference is incomplete or malformed")

    arrays: dict[str, np.ndarray] = {
        "utilities": estimate.utilities,
        "gradient": estimate.gradient,
        "curvature": estimate.curvature,
        "random_curvature": estimate.random_curvature,
        "gradient_component_variance": np.square(
            estimate.gradient_jackknife.standard_error
        ),
        "curvature_vech_covariance": np.asarray(
            estimate.curvature_vech_jackknife.covariance
        ),
    }
    q_summaries: list[Mapping[str, Any]] = []
    actions: dict[float, np.ndarray] = {}
    steps_by_q: dict[float, Mapping[str, np.ndarray]] = {}
    eigenvalues: list[np.ndarray] = []
    eigen_se: list[np.ndarray] = []
    step_covariance: list[float] = []
    action_covariance: list[float] = []
    aligned_variance: list[float] = []
    repeated: list[bool] = []
    boundary: list[bool] = []
    zero_action: list[bool] = []
    r_full: list[list[float]] = []
    r_sub: list[list[float]] = []
    r_sn: list[list[float]] = []
    r_full_unresolved: list[list[bool]] = []
    r_sub_unresolved: list[list[bool]] = []
    r_sn_unresolved: list[list[bool]] = []
    ess: list[list[float]] = []
    ratio_cv: list[list[float]] = []
    ratio_mean: list[list[float]] = []
    ratio_span: list[list[float]] = []
    alpha_curvature: list[float] = []
    angles: list[float] = []
    multiplier_sd: list[float] = []
    multiplier_range: list[float] = []

    for q_index, (q, alpha) in enumerate(zip(q_values, alphas, strict=True)):
        result = (
            first
            if q_index == 0 and own_endpoint_reference
            else _analysis_from_estimate(
                first,
                theta,
                sigma,
                basis,
                random_basis,
                float(alpha),
                endpoint_reference,
            )
        )
        steps = result.steps
        step_map = {name: np.asarray(getattr(steps, name)) for name in ARMS}
        if not alpha_resolved and any(np.any(step != 0.0) for step in step_map.values()):
            raise RuntimeError("unresolved alpha sentinel did not produce four zero steps")
        action = steps.structured - steps.isotropic
        actions[float(q)] = action.copy()
        steps_by_q[float(q)] = {name: value.copy() for name, value in step_map.items()}
        if reference_actions is None:
            reference_action = action
        else:
            reference_action = np.asarray(reference_actions[float(q)])
        action_hashes = {name: _array_sha256(value) for name, value in step_map.items()}
        structured_norm = float(np.linalg.norm(steps.structured))
        random_error = steps.random_norm_match_relative_error
        summary = {
            "q": float(q),
            "alpha": float(alpha),
            "alpha_resolved": alpha_resolved,
            "alpha_unresolved_reason": alpha_unresolved_reason,
            "gradient_norm": _stable_norm(estimate.gradient),
            "structured_norm": structured_norm,
            "isotropic_norm": float(np.linalg.norm(steps.isotropic)),
            "explicit_norm": float(np.linalg.norm(steps.explicit)),
            "random_norm": float(np.linalg.norm(steps.random)),
            "random_raw_norm": float(np.linalg.norm(steps.random_raw)),
            "anisotropic_action_norm": float(np.linalg.norm(action)),
            "anisotropic_minus_bank_a_norm": float(
                np.linalg.norm(action - reference_action)
            ),
            "structured_step_over_sigma": structured_norm / sigma,
            "structured_solve_residual": float(
                steps.structured_solve_relative_residual
            ),
            "random_solve_residual": float(steps.random_solve_relative_residual),
            "structured_isotropic_relative_norm_error": float(
                steps.isotropic_norm_match_relative_error
            ),
            "structured_random_relative_norm_error": _resolved_float(random_error),
            "random_control_valid": bool(steps.random_control_valid),
            "material_denominator_resolved": structured_norm > FLOAT_EPS,
            "finite": True,
            "action_sha256": action_hashes,
        }
        q_summaries.append(summary)
        for name, value in step_map.items():
            arrays[f"step_q{q_index}_{name}"] = value

        nonlinear = result.nonlinear_jackknife
        eigenvalues.append(steps.curvature_eigenvalues)
        eigen_se.append(
            np.sqrt(np.maximum(np.diag(nonlinear.eigenvalue_covariance), 0.0))
        )
        step_covariance.append(nonlinear.structured_action_covariance_trace)
        action_covariance.append(nonlinear.anisotropic_action_covariance_trace)
        aligned_variance.append(
            _resolved_float(nonlinear.anisotropic_action_aligned_variance)
        )
        repeated.append(nonlinear.repeated_eigenvalue_unresolved)
        boundary.append(nonlinear.projection_boundary_unresolved)
        zero_action.append(nonlinear.zero_anisotropic_action_unresolved)
        rows = [result.endpoint_diagnostics[name] for name in ARMS]
        r_full.append([_resolved_float(row.full_linearization_residual) for row in rows])
        r_sub.append(
            [_resolved_float(row.restricted_linearization_residual) for row in rows]
        )
        r_sn.append(
            [_resolved_float(row.self_normalized_linearization_residual) for row in rows]
        )
        r_full_unresolved.append(
            [row.full_linearization_residual is None for row in rows]
        )
        r_sub_unresolved.append(
            [row.restricted_linearization_residual is None for row in rows]
        )
        r_sn_unresolved.append(
            [row.self_normalized_linearization_residual is None for row in rows]
        )
        ess.append([float(row.normalized_ess_ratio) for row in rows])
        ratio_cv.append([float(row.ratio_coefficient_of_variation) for row in rows])
        ratio_mean.append(
            [float(row.mean_unnormalized_ratio_minus_one) for row in rows]
        )
        ratio_span.append([float(row.log_ratio_span) for row in rows])
        alpha_curvature.append(result.action_metrics.alpha_max_concave_eigenvalue)
        angles.append(_resolved_float(result.action_metrics.structured_explicit_angle_degrees))
        multiplier_sd.append(result.action_metrics.multiplier_standard_deviation)
        multiplier_range.append(result.action_metrics.multiplier_range)

    reference_signs = np.sign(np.linalg.eigvalsh(reference_curvature))
    current_signs = np.sign(np.linalg.eigvalsh(estimate.curvature))
    arrays.update(
        {
            "curvature_eigenvalues": np.asarray(eigenvalues),
            "negative_eigenvalue_count": np.asarray(
                np.count_nonzero(np.asarray(eigenvalues[0]) < 0.0),
                dtype=np.int64,
            ),
            "jackknife_eigenvalue_se": np.asarray(eigen_se),
            "structured_action_covariance_trace": np.asarray(step_covariance),
            "anisotropic_action_covariance_trace": np.asarray(action_covariance),
            "anisotropic_action_aligned_variance": np.asarray(aligned_variance),
            "repeated_eigenvalue_unresolved": np.asarray(repeated, dtype=np.bool_),
            "projection_boundary_unresolved": np.asarray(boundary, dtype=np.bool_),
            "zero_anisotropic_action_unresolved": np.asarray(
                zero_action, dtype=np.bool_
            ),
            "b_frobenius_absolute_error_to_bank_a": np.asarray(
                np.linalg.norm(estimate.curvature - reference_curvature)
            ),
            "negative_eigenvalue_sign_agreement_to_bank_a": np.asarray(
                np.mean(current_signs == reference_signs)
            ),
            "anisotropic_action_cosine_to_bank_a": np.asarray(
                [
                    _cosine(actions[float(q)], (reference_actions or actions)[float(q)])
                    for q in q_values
                ]
            ),
            "anisotropic_action_relative_error_to_bank_a": np.asarray(
                [
                    np.linalg.norm(
                        actions[float(q)] - (reference_actions or actions)[float(q)]
                    )
                    / max(
                        np.linalg.norm((reference_actions or actions)[float(q)]),
                        FLOAT_EPS,
                    )
                    for q in q_values
                ]
            ),
            "r_full": np.asarray(r_full),
            "r_sub": np.asarray(r_sub),
            "r_sn": np.asarray(r_sn),
            "r_full_unresolved": np.asarray(r_full_unresolved, dtype=np.bool_),
            "r_sub_unresolved": np.asarray(r_sub_unresolved, dtype=np.bool_),
            "r_sn_unresolved": np.asarray(r_sn_unresolved, dtype=np.bool_),
            "normalized_ess_ratio": np.asarray(ess),
            "ratio_coefficient_of_variation": np.asarray(ratio_cv),
            "mean_unnormalized_ratio_minus_one": np.asarray(ratio_mean),
            "log_ratio_span": np.asarray(ratio_span),
            "alpha_max_concave_eigenvalue": np.asarray(alpha_curvature),
            "structured_explicit_angle_degrees": np.asarray(angles),
            "multiplier_standard_deviation": np.asarray(multiplier_sd),
            "multiplier_range": np.asarray(multiplier_range),
            **{
                key: np.asarray(endpoint_reference.origin_metrics[key])
                for key in ORIGIN_DIAGNOSTIC_KEYS
            },
        }
    )
    if any(not np.all(np.isfinite(value)) for value in arrays.values() if value.dtype != np.bool_):
        raise FloatingPointError("population diagnostics contain nonfinite values")
    return PopulationProduct(
        arrays=arrays,
        q_summaries=tuple(q_summaries),
        gradient=estimate.gradient.copy(),
        curvature=estimate.curvature.copy(),
        actions=actions,
        steps=steps_by_q,
        utility_sum=float(np.sum(estimate.utilities)),
        utility_abs_sum=float(np.sum(np.abs(estimate.utilities))),
    )


def _diagnostic_jackknife_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    labels = (
        "gradient_component_variance",
        "curvature_vech_covariance",
        "jackknife_eigenvalue_se",
        "structured_action_covariance_trace",
        "anisotropic_action_covariance_trace",
        "anisotropic_action_aligned_variance",
    )
    return _labeled_arrays_sha256([(label, arrays[label]) for label in labels])


def _compact_diagnostic_arrays(
    arrays: Mapping[str, np.ndarray], q_count: int
) -> dict[str, np.ndarray]:
    """Replace regenerable full-dimensional vectors by fixed-width digests."""

    omitted = {"gradient", "gradient_component_variance"} | {
        f"step_q{q_index}_{arm}"
        for q_index in range(q_count)
        for arm in ARMS
    }
    compact = {name: value for name, value in arrays.items() if name not in omitted}
    compact["gradient_sha256"] = np.asarray(
        _array_sha256(arrays["gradient"]), dtype="S64"
    )
    compact["gradient_component_variance_sha256"] = np.asarray(
        _array_sha256(arrays["gradient_component_variance"]), dtype="S64"
    )
    compact["step_sha256"] = np.asarray(
        [
            [_array_sha256(arrays[f"step_q{q_index}_{arm}"]) for arm in ARMS]
            for q_index in range(q_count)
        ],
        dtype="S64",
    )
    return compact


def _bank_record(
    *,
    checkpoint_id: int,
    bank: str,
    raw: Mapping[str, np.ndarray],
    product: PopulationProduct,
    raw_path: str,
    raw_sha256: str,
    diagnostic_path: str,
    diagnostic_sha256: str,
) -> dict[str, Any]:
    pair_count = int(raw["paired_returns"].shape[0])
    return _record_with_hash(
        {
            "bank_id": checkpoint_id * 2 + (0 if bank == "A" else 1),
            "checkpoint_id": checkpoint_id,
            "bank": bank,
            "pair_count": pair_count,
            "candidate_rollouts": 2 * pair_count,
            "pair_indices": list(range(pair_count)),
            "perturbation_seeds": [int(value) for value in raw["perturbation_seeds"]],
            "rollout_seeds_plus": [int(value) for value in raw["rollout_seeds_plus"]],
            "rollout_seeds_minus": [int(value) for value in raw["rollout_seeds_minus"]],
            "perturbations_sha256": _array_sha256(raw["signed_noise"]),
            "returns_sha256": _array_sha256(raw["paired_returns"]),
            "transitions_sha256": _array_sha256(raw["paired_transitions"]),
            "jackknife_sha256": _diagnostic_jackknife_sha256(product.arrays),
            "raw_bank_path": raw_path,
            "raw_bank_sha256": raw_sha256,
            "diagnostics_path": diagnostic_path,
            "diagnostics_sha256": diagnostic_sha256,
            "candidate_transitions": int(np.sum(raw["paired_transitions"])),
            "antithetic_max_abs_error": float(
                np.max(np.abs(raw["signed_noise"][:, 0] + raw["signed_noise"][:, 1]))
            ),
            "exact_antithetic": True,
            "shared_rollout_seed_within_pair": True,
            "lopo_utility_sum": product.utility_sum,
            "lopo_utility_abs_sum": product.utility_abs_sum,
            "lopo_gradient_curvature_shared": True,
            "dsn_da_relative_error": float(
                product.arrays["self_normalized_gradient_relative_error"]
            ),
            "jsn_ja_relative_error": float(
                product.arrays["self_normalized_jacobian_relative_error"]
            ),
            "finite_u_statistic": True,
            "finite_jackknife": True,
            "finite_eigensystem": True,
            "q_summaries": list(product.q_summaries),
            "stderr_sha256": EMPTY_SHA256,
            "stderr_empty": True,
        }
    )


def _evaluate_endpoints(
    *,
    evaluator: BatchEvaluator,
    center: np.ndarray,
    endpoint_seeds: np.ndarray,
    q_values: Sequence[float],
    partition_steps: Sequence[Mapping[float, Mapping[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    center_returns, center_transitions = evaluator.evaluate_batch(
        np.repeat(center[None, :], len(endpoint_seeds), axis=0), endpoint_seeds
    )
    shape = (len(q_values), len(partition_steps), len(ARMS), len(endpoint_seeds))
    endpoint_returns = np.empty(shape, dtype=np.float64)
    endpoint_transitions = np.empty(shape, dtype=np.int64)
    for q_index, q in enumerate(q_values):
        for partition_index, by_q in enumerate(partition_steps):
            parameters = np.concatenate(
                [
                    np.repeat(
                        (center + by_q[float(q)][arm])[None, :],
                        len(endpoint_seeds),
                        axis=0,
                    )
                    for arm in ARMS
                ],
                axis=0,
            )
            seeds = np.tile(endpoint_seeds, len(ARMS))
            returns, transitions = evaluator.evaluate_batch(parameters, seeds)
            endpoint_returns[q_index, partition_index] = returns.reshape(
                len(ARMS), len(endpoint_seeds)
            )
            endpoint_transitions[q_index, partition_index] = transitions.reshape(
                len(ARMS), len(endpoint_seeds)
            )
    return {
        "center_returns": center_returns,
        "center_transitions": center_transitions,
        "endpoint_returns": endpoint_returns,
        "endpoint_transitions": endpoint_transitions,
        "rollout_seeds": endpoint_seeds,
    }


def _relative_path(prefix: str, filename: str) -> str:
    path = os.path.join(prefix, filename)
    if os.path.isabs(path) or os.path.normpath(path) != path or path.startswith("../"):
        raise ValueError("artifact path is not normalized and relative")
    return path


def produce_checkpoint_diagnostic(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    state: CheckpointState,
    training_config_path: str,
    checkpoint_path: str,
    capture_manifest_path: str,
    task_index: int,
    training_seed: int,
    generation: int,
    artifact_root: str,
    evaluator: BatchEvaluator,
    chunk_pairs: int = 32,
) -> str:
    """Run and atomically commit one complete checkpoint-level artifact."""

    checkpoint_id = checkpoint_id_for(
        manifest, task_index, training_seed, generation
    )
    task = manifest["tasks"][task_index]
    block_sizes = tuple(int(value) for value in task["policy_block_sizes"])
    dimension = int(task["parameter_count"])
    if dimension != sum(block_sizes):
        raise ValueError("manifest parameter count and layer blocks disagree")
    evaluator.validate_policy(
        dimension,
        block_sizes,
        int(task["observation_dim"]),
        int(task["action_dim"]),
    )
    if state.center.size != dimension:
        raise ValueError("validated checkpoint dimension changed before production")
    q_values = tuple(float(value) for value in manifest["dimensions"]["locality_q"])
    sigma = float(manifest["dimensions"]["noise_std"])
    prefix = os.path.join("checkpoint_artifacts", f"checkpoint_{checkpoint_id:06d}")
    final_directory = os.path.join(artifact_root, prefix)
    parent = os.path.dirname(final_directory)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(final_directory):
        raise FileExistsError(f"checkpoint artifact already exists: {final_directory}")
    staging = tempfile.mkdtemp(prefix=f".checkpoint_{checkpoint_id:06d}.", dir=parent)

    def stage(filename: str) -> tuple[str, str]:
        return os.path.join(staging, filename), _relative_path(prefix, filename)

    try:
        checkpoint_copy, checkpoint_relative = stage("checkpoint.npz")
        config_copy, config_relative = stage("checkpoint_training_config.json")
        capture_copy, capture_relative = stage("checkpoint_capture.json")
        with open(checkpoint_path, "rb") as stream:
            _write_bytes_atomic(checkpoint_copy, stream.read())
        with open(training_config_path, "rb") as stream:
            _write_bytes_atomic(config_copy, stream.read())
        with open(capture_manifest_path, "rb") as stream:
            _write_bytes_atomic(capture_copy, stream.read())
        if _sha256_file(checkpoint_copy) != state.checkpoint_sha256:
            raise RuntimeError("checkpoint input changed after lineage validation")
        if _sha256_file(config_copy) != state.training_config_sha256:
            raise RuntimeError("training-config input changed after lineage validation")
        if _sha256_file(capture_copy) != state.capture_manifest_sha256:
            raise RuntimeError("capture-manifest input changed after lineage validation")

        blocks: list[slice] = []
        cursor = 0
        for size in block_sizes:
            blocks.append(slice(cursor, cursor + size))
            cursor += size
        basis_seed = _checkpoint_seed(
            manifest, "basis", task_index, training_seed, generation
        )
        random_seed = _checkpoint_seed(
            manifest, "random_control", task_index, training_seed, generation
        )
        bases = build_lagged_bases(
            state.gradients,
            blocks,
            primary_fallback_seed=basis_seed,
            random_permutation_seed=random_seed,
            primary_reference=_array_sha256(state.gradients),
            random_reference=f"signed-permutation-seed:{random_seed}",
        )
        lagged_norms = np.asarray(
            [
                _stable_norm(bases.lagged_block_gradients[:, index])
                for index in range(3)
            ],
            dtype=np.float64,
        )
        exact_zero = lagged_norms == 0.0
        fallback_columns = np.zeros_like(bases.primary)
        for column, used in enumerate(exact_zero):
            if used:
                fallback_columns[:, column] = bases.primary[:, column]
        basis_path, basis_relative = stage("basis.npz")
        _write_npz_atomic(
            basis_path,
            primary_basis=bases.primary,
            random_basis=bases.random_control,
            lagged_block_norms=lagged_norms,
            lagged_block_exact_zero=exact_zero,
            primary_gaussian_fallback_used=np.asarray(
                bases.primary_fallback_blocks, dtype=np.bool_
            ),
            random_control_permuted_fallback_used=np.asarray(
                bases.random_uses_primary_fallback_blocks, dtype=np.bool_
            ),
            fallback_columns=fallback_columns,
        )
        # The basis artifact is durable in staging before any current bank seed
        # or perturbation is derived.
        basis_sha256 = _sha256_file(basis_path)
        provenance = BasisProvenance.strictly_lagged(
            primary_reference=basis_sha256,
            random_reference=basis_sha256,
        )

        checkpoint_record = _record_with_hash(
            {
                "checkpoint_id": checkpoint_id,
                "training_id": checkpoint_id // len(manifest["checkpoint_generations"]),
                "task_index": task_index,
                "env_name": task["env_name"],
                "training_seed": training_seed,
                "generation": generation,
                "parameter_sha256": _array_sha256(state.center),
                "observation_normalizer_sha256": _labeled_arrays_sha256(
                    [
                        ("enabled", np.asarray(True, dtype=np.bool_)),
                        ("mean", state.obs_mean),
                        ("var", state.obs_var),
                        ("count", np.asarray(state.obs_count, dtype=np.float64)),
                    ]
                ),
                "training_config_sha256": state.training_config_sha256,
                "source_sha256": state.source_sha256,
                "prior_gradient_indices": [int(value) for value in state.gradient_generations],
                "prior_gradient_sha256": [
                    _array_sha256(row) for row in state.gradients
                ],
                "lagged_block_norms": lagged_norms.tolist(),
                "lagged_block_exact_zero": exact_zero.tolist(),
                "primary_gaussian_fallback_used": list(
                    bases.primary_fallback_blocks
                ),
                "random_control_permuted_fallback_used": list(
                    bases.random_uses_primary_fallback_blocks
                ),
                "fallback_column_sha256": [
                    _array_sha256(fallback_columns[:, index]) if exact_zero[index] else None
                    for index in range(3)
                ],
                "basis_seed": basis_seed,
                "random_control_seed": random_seed,
                "basis_sha256": _array_sha256(bases.primary),
                "random_basis_sha256": _array_sha256(bases.random_control),
                "basis_locked_before_bank_sampling": True,
                "checkpoint_artifact_sha256": _sha256_file(checkpoint_copy),
                "checkpoint_artifact_path": checkpoint_relative,
                "training_config_path": config_relative,
                "basis_artifact_path": basis_relative,
                "basis_artifact_sha256": basis_sha256,
            }
        )

        bank_records: list[dict[str, Any]] = []
        partition_records: list[dict[str, Any]] = []
        bank_products: dict[str, PopulationProduct] = {}
        frozen_endpoint_reference: FrozenEndpointReference | None = None
        all_bank_seeds: set[int] = set()
        locked_alphas: tuple[float, ...] | None = None
        alpha_resolved = True
        alpha_unresolved_reason: str | None = None
        for bank in ("A", "B"):
            raw = _evaluate_bank(
                evaluator=evaluator,
                manifest=manifest,
                task_index=task_index,
                training_seed=training_seed,
                generation=generation,
                bank=bank,
                center=state.center,
                chunk_pairs=chunk_pairs,
            )
            current_seeds = set(map(int, raw["perturbation_seeds"])) | set(
                map(int, raw["rollout_seeds_plus"])
            )
            if set(map(int, raw["perturbation_seeds"])).intersection(
                map(int, raw["rollout_seeds_plus"])
            ):
                raise RuntimeError("perturbation and rollout seed streams overlap")
            if all_bank_seeds.intersection(current_seeds):
                raise RuntimeError("Bank A and Bank B seed streams overlap")
            all_bank_seeds.update(current_seeds)
            raw_path, raw_relative = stage(f"bank_{bank}_raw.npz")
            _write_npz_atomic(
                raw_path,
                **{name: value for name, value in raw.items() if name != "signed_noise"},
            )
            if bank == "A":
                preliminary = estimate_lopo_population(
                    state.center,
                    raw["signed_noise"],
                    raw["paired_returns"],
                    sigma,
                    bases.primary,
                    bases.random_control,
                    basis_provenance=provenance,
                )
                bank_a_gradient_norm = _stable_norm(preliminary.gradient)
                if bank_a_gradient_norm == 0.0:
                    zero_policy = manifest["analysis"]["zero_gradient_calibration"]
                    locked_alphas = tuple(
                        float(zero_policy["alpha_sentinel"]) for _ in q_values
                    )
                    alpha_resolved = False
                    alpha_unresolved_reason = str(
                        zero_policy["unresolved_reason"]
                    )
                else:
                    locked_alphas = tuple(
                        calibrate_locality_rate(preliminary.gradient, sigma, q)
                        for q in q_values
                    )
                    alpha_resolved = True
                    alpha_unresolved_reason = None
                del preliminary
            if locked_alphas is None:
                raise RuntimeError("Bank-A locality rates were not locked before Bank B")
            reference = bank_products.get("A")
            product = _population_product(
                theta=state.center,
                signed_noise=raw["signed_noise"],
                paired_returns=raw["paired_returns"],
                sigma=sigma,
                basis=bases.primary,
                random_basis=bases.random_control,
                q_values=q_values,
                alphas=locked_alphas,
                provenance=provenance,
                reference_curvature=None if reference is None else reference.curvature,
                reference_actions=None if reference is None else reference.actions,
                endpoint_reference=frozen_endpoint_reference,
                alpha_resolved=alpha_resolved,
                alpha_unresolved_reason=alpha_unresolved_reason,
            )
            diagnostics_path, diagnostics_relative = stage(f"bank_{bank}_diagnostics.npz")
            _write_npz_atomic(
                diagnostics_path,
                **_compact_diagnostic_arrays(product.arrays, len(q_values)),
            )
            bank_records.append(
                _bank_record(
                    checkpoint_id=checkpoint_id,
                    bank=bank,
                    raw=raw,
                    product=product,
                    raw_path=raw_relative,
                    raw_sha256=_sha256_file(raw_path),
                    diagnostic_path=diagnostics_relative,
                    diagnostic_sha256=_sha256_file(diagnostics_path),
                )
            )
            bank_products[bank] = product
            if bank == "A":
                frozen_endpoint_reference = FrozenEndpointReference(
                    signed_noise=raw["signed_noise"],
                    utilities=product.arrays["utilities"],
                    gradient=product.gradient,
                    curvature=product.curvature,
                    origin_metrics={
                        key: float(product.arrays[key])
                        for key in ORIGIN_DIAGNOSTIC_KEYS
                    },
                )
            else:
                bank_b_raw = raw

        reference = bank_products["A"]
        replication = bank_products["B"]
        if frozen_endpoint_reference is None:
            raise RuntimeError("complete Bank-A endpoint reference was not locked")
        partitions = bank_b_partitions(
            manifest, task_index, training_seed, generation
        )
        partition_products: list[PopulationProduct] = []
        partition_steps: list[Mapping[float, Mapping[str, np.ndarray]]] = []
        partition_seed = _checkpoint_seed(
            manifest, "bank_b_partition", task_index, training_seed, generation
        )
        for partition_index, pair_indices in enumerate(partitions):
            selected = np.asarray(pair_indices, dtype=np.int64)
            product = _population_product(
                theta=state.center,
                signed_noise=bank_b_raw["signed_noise"][selected],
                paired_returns=bank_b_raw["paired_returns"][selected],
                sigma=sigma,
                basis=bases.primary,
                random_basis=bases.random_control,
                q_values=q_values,
                alphas=locked_alphas,
                provenance=provenance,
                reference_curvature=reference.curvature,
                reference_actions=reference.actions,
                endpoint_reference=frozen_endpoint_reference,
                alpha_resolved=alpha_resolved,
                alpha_unresolved_reason=alpha_unresolved_reason,
            )
            diagnostic_path, diagnostic_relative = stage(
                f"partition_{partition_index:02d}_diagnostics.npz"
            )
            _write_npz_atomic(
                diagnostic_path,
                **_compact_diagnostic_arrays(product.arrays, len(q_values)),
            )
            partition_records.append(
                _record_with_hash(
                    {
                        "partition_id": checkpoint_id * len(partitions) + partition_index,
                        "checkpoint_id": checkpoint_id,
                        "partition_index": partition_index,
                        "partition_seed": partition_seed,
                        "pair_indices": pair_indices,
                        "pair_count": len(pair_indices),
                        "lopo_utility_sum": product.utility_sum,
                        "lopo_utility_abs_sum": product.utility_abs_sum,
                        "lopo_gradient_curvature_shared": True,
                        "finite_u_statistic": True,
                        "finite_jackknife": True,
                        "finite_eigensystem": True,
                        "q_summaries": list(product.q_summaries),
                        "diagnostics_path": diagnostic_relative,
                        "diagnostics_sha256": _sha256_file(diagnostic_path),
                    }
                )
            )
            partition_products.append(product)
            partition_steps.append(product.steps)

        checkpoint_metrics: list[dict[str, Any]] = []
        for q_index, q in enumerate(q_values):
            reference_action = reference.actions[q]
            replication_action = replication.actions[q]
            reference_norm = float(np.linalg.norm(reference_action))
            replication_norm = float(np.linalg.norm(replication_action))
            structured_norm = float(reference.q_summaries[q_index]["structured_norm"])
            differences = np.asarray(
                [
                    np.linalg.norm(product.actions[q] - reference_action)
                    for product in partition_products
                ],
                dtype=np.float64,
            )
            high_difference = float(np.linalg.norm(reference_action - replication_action))
            squared_mean = float(np.mean(np.square(differences)))
            checkpoint_metrics.append(
                _record_with_hash(
                    {
                        "metric_id": checkpoint_id * len(q_values) + q_index,
                        "checkpoint_id": checkpoint_id,
                        "q": q,
                        "d_material": reference_norm / max(structured_norm, FLOAT_EPS),
                        "e_high": high_difference
                        / max(0.5 * (reference_norm + replication_norm), FLOAT_EPS),
                        "e_100": np.sqrt(squared_mean) / max(reference_norm, FLOAT_EPS),
                        "material_resolved": structured_norm > FLOAT_EPS,
                        "high_sample_resolved": 0.5
                        * (reference_norm + replication_norm)
                        > FLOAT_EPS,
                        "operational_resolved": reference_norm > FLOAT_EPS,
                        "high_sample_action_difference_norm": high_difference,
                        "partition_action_sq_error_mean": squared_mean,
                    }
                )
            )

        endpoint_seeds = np.asarray(
            [
                _endpoint_seed(
                    manifest, task_index, training_seed, generation, episode
                )
                for episode in range(int(manifest["dimensions"]["endpoint_episodes"]))
            ],
            dtype=np.uint64,
        )
        if len(set(map(int, endpoint_seeds))) != len(endpoint_seeds):
            raise RuntimeError("endpoint seed derivation produced a collision")
        if all_bank_seeds.intersection(map(int, endpoint_seeds)):
            raise RuntimeError("endpoint seed stream overlaps a bank stream")
        endpoint_data = _evaluate_endpoints(
            evaluator=evaluator,
            center=state.center,
            endpoint_seeds=endpoint_seeds,
            q_values=q_values,
            partition_steps=partition_steps,
        )
        endpoint_path, endpoint_relative = stage("endpoints.npz")
        _write_npz_atomic(endpoint_path, **endpoint_data)
        endpoint_sha256 = _sha256_file(endpoint_path)
        center_records: list[dict[str, Any]] = []
        endpoint_records: list[dict[str, Any]] = []
        episodes = len(endpoint_seeds)
        for episode in range(episodes):
            center_records.append(
                _record_with_hash(
                    {
                        "center_endpoint_id": checkpoint_id * episodes + episode,
                        "checkpoint_id": checkpoint_id,
                        "episode_index": episode,
                        "rollout_seed": int(endpoint_seeds[episode]),
                        "return": float(endpoint_data["center_returns"][episode]),
                        "transitions": int(endpoint_data["center_transitions"][episode]),
                        "rollout_artifact_path": endpoint_relative,
                        "rollout_artifact_sha256": endpoint_sha256,
                    }
                )
            )
        arms = list(ARMS)
        partition_count = len(partitions)
        for q_index, q in enumerate(q_values):
            for partition_index, product in enumerate(partition_products):
                for arm_index, arm in enumerate(arms):
                    action_hash = product.q_summaries[q_index]["action_sha256"][arm]
                    for episode in range(episodes):
                        value = checkpoint_id * len(q_values) + q_index
                        value = value * partition_count + partition_index
                        value = value * len(arms) + arm_index
                        endpoint_id = value * episodes + episode
                        endpoint_records.append(
                            _record_with_hash(
                                {
                                    "endpoint_id": endpoint_id,
                                    "checkpoint_id": checkpoint_id,
                                    "partition_index": partition_index,
                                    "q": q,
                                    "arm": arm,
                                    "episode_index": episode,
                                    "rollout_seed": int(endpoint_seeds[episode]),
                                    "return": float(
                                        endpoint_data["endpoint_returns"][
                                            q_index, partition_index, arm_index, episode
                                        ]
                                    ),
                                    "transitions": int(
                                        endpoint_data["endpoint_transitions"][
                                            q_index, partition_index, arm_index, episode
                                        ]
                                    ),
                                    "action_sha256": action_hash,
                                    "rollout_artifact_path": endpoint_relative,
                                    "rollout_artifact_sha256": endpoint_sha256,
                                }
                            )
                        )

        artifact_files = sorted(
            filename
            for filename in os.listdir(staging)
            if os.path.isfile(os.path.join(staging, filename))
        )
        inventory = [
            {
                "path": _relative_path(prefix, filename),
                "sha256": _sha256_file(os.path.join(staging, filename)),
            }
            for filename in artifact_files
        ]
        fragment = {
            "schema_version": 1,
            "study": STUDY,
            "status": "complete",
            "manifest_sha256": manifest_sha256,
            "checkpoint_id": checkpoint_id,
            "task_index": task_index,
            "training_seed": training_seed,
            "generation": generation,
            "checkpoint": checkpoint_record,
            "banks": bank_records,
            "partitions": partition_records,
            "checkpoint_metrics": checkpoint_metrics,
            "center_endpoints": center_records,
            "endpoints": endpoint_records,
            "transition_totals": {
                "bank": int(
                    sum(record["candidate_transitions"] for record in bank_records)
                ),
                "center": int(np.sum(endpoint_data["center_transitions"])),
                "endpoint": int(np.sum(endpoint_data["endpoint_transitions"])),
            },
            "lineage_artifacts": {
                "capture_manifest_path": capture_relative,
                "capture_manifest_sha256": _sha256_file(capture_copy),
            },
            "artifact_inventory": inventory,
            "no_outcome_selection": True,
            "no_record_exclusion": True,
        }
        fragment["fragment_sha256"] = hashlib.sha256(_canonical_bytes(fragment)).hexdigest()
        index_path = os.path.join(staging, "checkpoint_index.json")
        _write_json_atomic(index_path, fragment)
        if _load_json(index_path).get("fragment_sha256") != fragment["fragment_sha256"]:
            raise RuntimeError("staged checkpoint index failed readback")
        os.replace(staging, final_directory)
        return os.path.join(final_directory, "checkpoint_index.json")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-capture-manifest", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--chunk-pairs", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Verify the current complete study snapshot before loading a checkpoint,
    # constructing a MuJoCo environment, creating a Pool, or sampling noise.
    source_sha256 = require_lagged_subspace_study_source_lock(
        args.expected_source_sha256
    )
    validate_manifest_mapping(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    manifest, manifest_sha256 = load_locked_manifest(
        args.manifest, args.expected_manifest_sha256
    )
    state, config = load_and_validate_checkpoint(
        checkpoint_path=args.checkpoint,
        capture_manifest_path=args.checkpoint_capture_manifest,
        training_config_path=args.training_config,
        manifest=manifest,
        task_index=args.task_index,
        training_seed=args.training_seed,
        generation=args.generation,
        expected_source_sha256=source_sha256,
    )
    with MujocoBatchEvaluator(
        config,
        state.obs_mean,
        state.obs_var,
        n_workers=args.n_workers,
    ) as evaluator:
        index_path = produce_checkpoint_diagnostic(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            state=state,
            training_config_path=args.training_config,
            checkpoint_path=args.checkpoint,
            capture_manifest_path=args.checkpoint_capture_manifest,
            task_index=args.task_index,
            training_seed=args.training_seed,
            generation=args.generation,
            artifact_root=args.artifact_root,
            evaluator=evaluator,
            chunk_pairs=args.chunk_pairs,
        )
    print(index_path)


if __name__ == "__main__":
    main()

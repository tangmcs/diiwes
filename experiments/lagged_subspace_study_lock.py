#!/usr/bin/env python3
"""Fail-closed source identity and array mappings for the lagged study.

The study source identity is deliberately independent of the selected task
configuration.  Every training configuration and every executable component
of the two-stage study contributes under a stable repository-relative label.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import sys
from typing import Iterable, Mapping, Sequence


STUDY = "lagged_subspace_frozen_checkpoint"
EXPECTED_SOURCE_ENV = "PAPER_EXPECTED_SOURCE_SHA"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
PROVENANCE_LOCK_ENVIRONMENTS = {
    "manifest_sha256": "PAPER_EXPECTED_MANIFEST_SHA256",
    "protocol_sha256": "PAPER_EXPECTED_PROTOCOL_SHA256",
    "analyzer_sha256": "PAPER_EXPECTED_ANALYZER_SHA256",
    "launcher_sha256": "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256",
    "dependency_lock_sha256": "PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256",
}

STUDY_CONFIG_PATHS = (
    "configs/mujuco/halfcheetah_lagged_subspace_checkpoints.yaml",
    "configs/mujuco/hopper_lagged_subspace_checkpoints.yaml",
    "configs/mujuco/walker2d_lagged_subspace_checkpoints.yaml",
)
LAUNCHER_BUNDLE_PATH = (
    "experiments/manifests/lagged_subspace_launcher_lock.json"
)
DEPENDENCY_BUNDLE_PATH = (
    "experiments/manifests/lagged_subspace_dependency_lock.json"
)
BUNDLE_FILES_BY_KIND = {
    "launchers": (
        "scripts/submit_lagged_subspace_checkpoint_generation.sh",
        "scripts/submit_lagged_subspace_diagnostic.sh",
    ),
    "dependency_locks": ("environment.yml", "requirement.txt"),
}

# Keep this tuple explicit. Adding a runtime dependency without adding it here
# must fail review and the focused inventory tests.
STUDY_SOURCE_PATHS = tuple(
    sorted(
        (
            *STUDY_CONFIG_PATHS,
            "core/__init__.py",
            "core/diiwes.py",
            "core/implicit_es.py",
            "core/lagged_subspace_diagnostic.py",
            "core/policies.py",
            "core/standard_es.py",
            "docs/lagged_subspace_frozen_checkpoint_protocol.md",
            "environment.yml",
            "experiments/__init__.py",
            "experiments/lagged_subspace_study_lock.py",
            "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
            LAUNCHER_BUNDLE_PATH,
            DEPENDENCY_BUNDLE_PATH,
            "experiments/run_lagged_subspace_checkpoint_diagnostic.py",
            "experiments/train.py",
            "requirement.txt",
            "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
            "scripts/assemble_lagged_subspace_frozen_checkpoint.py",
            "scripts/submit_lagged_subspace_checkpoint_generation.sh",
            "scripts/submit_lagged_subspace_diagnostic.sh",
            "tests/test_analyze_lagged_subspace_frozen_checkpoint.py",
            "tests/test_assemble_lagged_subspace_frozen_checkpoint.py",
            "tests/test_lagged_subspace_diagnostic.py",
            "tests/test_lagged_subspace_study_lock.py",
            "tests/test_optimizers.py",
            "tests/test_run_lagged_subspace_checkpoint_diagnostic.py",
            "utilities/__init__.py",
            "utilities/obs_norm.py",
        )
    )
)

RUNTIME_PATHS_BY_ENTRYPOINT: Mapping[str, tuple[str, ...]] = {
    "checkpoint_generation": (
        "experiments/lagged_subspace_study_lock.py",
        "experiments/train.py",
        "core/__init__.py",
        "core/diiwes.py",
        "core/implicit_es.py",
        "core/policies.py",
        "core/standard_es.py",
        "utilities/__init__.py",
        "utilities/obs_norm.py",
        "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
        "docs/lagged_subspace_frozen_checkpoint_protocol.md",
        "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
        LAUNCHER_BUNDLE_PATH,
        DEPENDENCY_BUNDLE_PATH,
        "environment.yml",
        "requirement.txt",
        "scripts/submit_lagged_subspace_checkpoint_generation.sh",
        *STUDY_CONFIG_PATHS,
    ),
    "diagnostic": (
        "experiments/lagged_subspace_study_lock.py",
        "experiments/run_lagged_subspace_checkpoint_diagnostic.py",
        "experiments/train.py",
        "core/__init__.py",
        "core/diiwes.py",
        "core/implicit_es.py",
        "core/lagged_subspace_diagnostic.py",
        "core/policies.py",
        "core/standard_es.py",
        "utilities/__init__.py",
        "utilities/obs_norm.py",
        "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
        "docs/lagged_subspace_frozen_checkpoint_protocol.md",
        "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
        LAUNCHER_BUNDLE_PATH,
        DEPENDENCY_BUNDLE_PATH,
        "environment.yml",
        "requirement.txt",
        "scripts/submit_lagged_subspace_diagnostic.sh",
    ),
    "assembly_and_analysis": (
        "experiments/lagged_subspace_study_lock.py",
        "scripts/assemble_lagged_subspace_frozen_checkpoint.py",
        "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
        "core/lagged_subspace_diagnostic.py",
        "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
        "docs/lagged_subspace_frozen_checkpoint_protocol.md",
        LAUNCHER_BUNDLE_PATH,
        DEPENDENCY_BUNDLE_PATH,
        "environment.yml",
        "requirement.txt",
    ),
}

TASKS = (
    (0, "Hopper-v5", STUDY_CONFIG_PATHS[1]),
    (1, "Walker2d-v5", STUDY_CONFIG_PATHS[2]),
    (2, "HalfCheetah-v5", STUDY_CONFIG_PATHS[0]),
)
TRAINING_SEEDS = tuple(range(300, 320))
CHECKPOINT_GENERATIONS = (50, 150, 250)
TRAINING_TASK_COUNT = len(TASKS) * len(TRAINING_SEEDS)
CHECKPOINT_TASK_COUNT = TRAINING_TASK_COUNT * len(CHECKPOINT_GENERATIONS)
RUNTIME_DEPENDENCY_VERSIONS = {
    "PyYAML": "6.0.2",
    "ale-py": "0.11.2",
    "gymnasium": "1.2.0",
    "matplotlib": "3.10.5",
    "mujoco": "3.3.5",
    "numpy": "1.26.4",
    "pip": "25.2",
    "scipy": "1.15.3",
}


class StudySourceLockError(RuntimeError):
    """The study source identity or mapping contract is invalid."""


def _repository_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _normalized_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise StudySourceLockError("source inventory labels must be nonempty strings")
    normalized = os.path.normpath(value)
    if (
        os.path.isabs(value)
        or normalized != value
        or value == ".."
        or value.startswith("../")
        or "\\" in value
    ):
        raise StudySourceLockError(
            f"source inventory label is not normalized and relative: {value!r}"
        )
    return value


def validate_runtime_inventory(
    runtime_paths: Iterable[str],
    *,
    source_paths: Sequence[str] = STUDY_SOURCE_PATHS,
) -> tuple[str, ...]:
    """Require every declared repository runtime file to be source-locked."""

    locked = tuple(_normalized_relative_path(path) for path in source_paths)
    if locked != tuple(sorted(set(locked))):
        raise StudySourceLockError(
            "study source inventory must be sorted, unique, and normalized"
        )
    requested = tuple(_normalized_relative_path(path) for path in runtime_paths)
    unlisted = sorted(set(requested).difference(locked))
    if unlisted:
        raise StudySourceLockError(
            "runtime files are absent from the source inventory: "
            + ", ".join(unlisted)
        )
    return requested


def _validated_source_paths() -> tuple[str, ...]:
    paths = validate_runtime_inventory(STUDY_SOURCE_PATHS)
    for entrypoint, runtime_paths in RUNTIME_PATHS_BY_ENTRYPOINT.items():
        try:
            validate_runtime_inventory(runtime_paths, source_paths=paths)
        except StudySourceLockError as error:
            raise StudySourceLockError(
                f"{entrypoint} runtime inventory is not fully locked: {error}"
            ) from error
    return paths


def _regular_locked_file(root: str, relative_path: str) -> str:
    root_real = os.path.realpath(root)
    path = os.path.join(root_real, relative_path)
    path_real = os.path.realpath(path)
    if os.path.commonpath([root_real, path_real]) != root_real:
        raise StudySourceLockError(
            f"locked source path escapes the repository root: {relative_path}"
        )
    if os.path.islink(path) or not os.path.isfile(path):
        raise StudySourceLockError(
            f"locked source path is missing, non-regular, or a symlink: {relative_path}"
        )
    return path


def _local_module_candidates(
    module: str, *, source_relative_path: str, level: int
) -> tuple[str, ...]:
    components = [] if not module else module.split(".")
    if level:
        package = os.path.dirname(source_relative_path).split(os.sep)
        keep = len(package) - (level - 1)
        if keep < 0:
            return ()
        components = package[:keep] + components
    if not components:
        return ()
    stem = "/".join(components)
    return (f"{stem}.py", f"{stem}/__init__.py")


def _validate_local_python_imports(root: str, source_paths: Sequence[str]) -> None:
    """Reject local imports whose implementation is absent from the lock."""

    locked = set(source_paths)
    for relative_path in source_paths:
        if not relative_path.endswith(".py"):
            continue
        path = _regular_locked_file(root, relative_path)
        try:
            with open(path, encoding="utf-8") as stream:
                tree = ast.parse(stream.read(), filename=relative_path)
        except (OSError, UnicodeError, SyntaxError) as error:
            raise StudySourceLockError(
                f"cannot parse locked Python source {relative_path}: {error}"
            ) from error
        for node in ast.walk(tree):
            modules: list[tuple[str, int]] = []
            if isinstance(node, ast.Import):
                modules.extend((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                modules.append((base, node.level))
                modules.extend(
                    (
                        f"{base}.{alias.name}" if base else alias.name,
                        node.level,
                    )
                    for alias in node.names
                    if alias.name != "*"
                )
            for module, level in modules:
                for candidate in _local_module_candidates(
                    module,
                    source_relative_path=relative_path,
                    level=level,
                ):
                    candidate_path = os.path.join(root, candidate)
                    if os.path.isfile(candidate_path) and candidate not in locked:
                        raise StudySourceLockError(
                            f"{relative_path} imports unlisted runtime file {candidate}"
                        )


def compute_lagged_subspace_study_sha256(snapshot_root: str) -> str:
    """Return the one composite digest for the complete two-stage study."""

    root = os.path.abspath(snapshot_root)
    if os.path.islink(root) or not os.path.isdir(root):
        raise StudySourceLockError(
            "study snapshot root must be an existing non-symlink directory"
        )
    source_paths = _validated_source_paths()
    _validate_local_python_imports(root, source_paths)
    digest = hashlib.sha256()
    digest.update(b"diiwes-lagged-subspace-study-source-v1\0")
    for relative_path in source_paths:
        path = _regular_locked_file(root, relative_path)
        label = relative_path.encode("utf-8")
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        file_digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(block)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def current_lagged_subspace_study_sha256() -> str:
    return compute_lagged_subspace_study_sha256(_repository_root())


def study_sha256_for_checkpoint_config(
    config_path: str,
    *,
    snapshot_root: str | None = None,
    expected_env_name: str | None = None,
) -> str:
    """Validate a selected task config while returning the global study lock."""

    root = os.path.abspath(
        _repository_root() if snapshot_root is None else snapshot_root
    )
    config_real = os.path.realpath(config_path)
    candidates = {
        os.path.realpath(os.path.join(root, relative))
        for relative in STUDY_CONFIG_PATHS
    }
    if config_real not in candidates:
        raise StudySourceLockError(
            "checkpoint configuration is not one of the three locked study configs"
        )
    if expected_env_name is not None:
        expected_relative = {
            env_name: relative for _, env_name, relative in TASKS
        }.get(expected_env_name)
        if (
            expected_relative is None
            or config_real
            != os.path.realpath(os.path.join(root, expected_relative))
        ):
            raise StudySourceLockError(
                "checkpoint configuration does not match the selected environment"
            )
    return compute_lagged_subspace_study_sha256(root)


def require_lagged_subspace_study_source_lock(
    expected_sha256: str | None = None,
    *,
    snapshot_root: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Require the environment lock and verify it against current source."""

    environment = os.environ if environ is None else environ
    from_environment = environment.get(EXPECTED_SOURCE_ENV)
    if not _is_sha256(from_environment):
        raise StudySourceLockError(
            f"{EXPECTED_SOURCE_ENV} is mandatory and must be a lowercase SHA-256"
        )
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise StudySourceLockError("explicit expected source digest is invalid")
        if expected_sha256 != from_environment:
            raise StudySourceLockError(
                "explicit expected source digest disagrees with "
                f"{EXPECTED_SOURCE_ENV}"
            )
    expected = from_environment
    actual = compute_lagged_subspace_study_sha256(
        _repository_root() if snapshot_root is None else snapshot_root
    )
    if actual != expected:
        raise StudySourceLockError(
            f"study source SHA-256 mismatch: expected {expected}, found {actual}"
        )
    return actual


def _load_json_file(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StudySourceLockError(f"cannot load locked JSON {path}: {error}") from error


def _sha256_regular_file(snapshot_root: str, relative_path: str) -> str:
    path = _regular_locked_file(snapshot_root, relative_path)
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_hash_bundle(
    snapshot_root: str,
    bundle_relative_path: str,
    *,
    expected_bundle_sha256: str,
    expected_kind: str,
) -> str:
    """Validate one deterministic bundle and every file digest it names."""

    if not _is_sha256(expected_bundle_sha256):
        raise StudySourceLockError("expected bundle digest is invalid")
    relative = _normalized_relative_path(bundle_relative_path)
    if relative not in STUDY_SOURCE_PATHS:
        raise StudySourceLockError("hash bundle itself is not source-locked")
    path = _regular_locked_file(snapshot_root, relative)
    with open(path, "rb") as stream:
        payload = stream.read()
    actual_bundle = hashlib.sha256(payload).hexdigest()
    if actual_bundle != expected_bundle_sha256:
        raise StudySourceLockError(
            f"{expected_kind} bundle SHA-256 mismatch: expected "
            f"{expected_bundle_sha256}, found {actual_bundle}"
        )
    value = _load_json_file(path)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", "study", "kind", "files"}
        or value["schema_version"] != 1
        or value["study"] != STUDY
        or value["kind"] != expected_kind
        or not isinstance(value["files"], list)
        or not value["files"]
    ):
        raise StudySourceLockError(f"{expected_kind} hash bundle schema is invalid")
    records: list[tuple[str, str]] = []
    for index, item in enumerate(value["files"]):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise StudySourceLockError(
                f"{expected_kind} hash bundle item {index} is invalid"
            )
        item_path = _normalized_relative_path(item["path"])
        item_sha = item["sha256"]
        if not _is_sha256(item_sha) or item_path not in STUDY_SOURCE_PATHS:
            raise StudySourceLockError(
                f"{expected_kind} hash bundle item {index} is not source-locked"
            )
        records.append((item_path, item_sha))
    if records != sorted(set(records)):
        raise StudySourceLockError(
            f"{expected_kind} hash bundle records are not sorted and unique"
        )
    required_paths = BUNDLE_FILES_BY_KIND.get(expected_kind)
    if required_paths is None or tuple(path for path, _ in records) != required_paths:
        raise StudySourceLockError(
            f"{expected_kind} hash bundle file inventory is not exact"
        )
    for item_path, expected_file_sha in records:
        item_file = _regular_locked_file(snapshot_root, item_path)
        digest = hashlib.sha256()
        with open(item_file, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_file_sha:
            raise StudySourceLockError(
                f"{expected_kind} bundle file digest mismatch: {item_path}"
            )
    return actual_bundle


def require_checkpoint_generation_provenance_locks(
    *,
    snapshot_root: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify every production lock before checkpoint environment creation."""

    root = _repository_root() if snapshot_root is None else snapshot_root
    environment = os.environ if environ is None else environ
    source_sha256 = require_lagged_subspace_study_source_lock(
        snapshot_root=root, environ=environment
    )
    expected: dict[str, str] = {}
    for label, variable in PROVENANCE_LOCK_ENVIRONMENTS.items():
        value = environment.get(variable)
        if not _is_sha256(value):
            raise StudySourceLockError(
                f"{variable} is mandatory and must be a lowercase SHA-256"
            )
        expected[label] = value
    direct_files = {
        "manifest_sha256": (
            "experiments/manifests/lagged_subspace_frozen_checkpoint.json"
        ),
        "protocol_sha256": "docs/lagged_subspace_frozen_checkpoint_protocol.md",
        "analyzer_sha256": "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
    }
    for label, relative_path in direct_files.items():
        actual = _sha256_regular_file(root, relative_path)
        if actual != expected[label]:
            raise StudySourceLockError(
                f"{label} mismatch: expected {expected[label]}, found {actual}"
            )
    validate_hash_bundle(
        root,
        LAUNCHER_BUNDLE_PATH,
        expected_bundle_sha256=expected["launcher_sha256"],
        expected_kind="launchers",
    )
    validate_hash_bundle(
        root,
        DEPENDENCY_BUNDLE_PATH,
        expected_bundle_sha256=expected["dependency_lock_sha256"],
        expected_kind="dependency_locks",
    )
    validate_manifest_mapping(root)
    return {"source_sha256": source_sha256, **expected}


def training_coordinates(array_task_id: int) -> tuple[int, int, str, int, str]:
    if isinstance(array_task_id, bool) or not 0 <= array_task_id < TRAINING_TASK_COUNT:
        raise StudySourceLockError(
            f"training array task must be in [0, {TRAINING_TASK_COUNT - 1}]"
        )
    task_index, seed_index = divmod(int(array_task_id), len(TRAINING_SEEDS))
    manifest_index, env_name, config_path = TASKS[task_index]
    if manifest_index != task_index:
        raise StudySourceLockError("task mapping is not contiguous")
    return array_task_id, task_index, env_name, TRAINING_SEEDS[seed_index], config_path


def checkpoint_coordinates(
    array_task_id: int,
) -> tuple[int, int, int, str, int, int, str]:
    if isinstance(array_task_id, bool) or not 0 <= array_task_id < CHECKPOINT_TASK_COUNT:
        raise StudySourceLockError(
            f"diagnostic array task must be in [0, {CHECKPOINT_TASK_COUNT - 1}]"
        )
    training_id, generation_index = divmod(
        int(array_task_id), len(CHECKPOINT_GENERATIONS)
    )
    _, task_index, env_name, seed, config_path = training_coordinates(training_id)
    return (
        array_task_id,
        training_id,
        task_index,
        env_name,
        seed,
        CHECKPOINT_GENERATIONS[generation_index],
        config_path,
    )


def validate_manifest_mapping(snapshot_root: str) -> None:
    """Require the checked-in manifest to match both array bijections exactly."""

    manifest_path = _regular_locked_file(
        snapshot_root,
        "experiments/manifests/lagged_subspace_frozen_checkpoint.json",
    )
    manifest = _load_json_file(manifest_path)
    if not isinstance(manifest, dict):
        raise StudySourceLockError("study manifest is not a JSON object")
    protocol_path = "docs/lagged_subspace_frozen_checkpoint_protocol.md"
    protocol = manifest.get("protocol")
    if (
        manifest.get("protocol_status")
        != "final_locked_before_environment_outcomes"
        or not isinstance(protocol, dict)
        or set(protocol) != {"path", "sha256"}
        or protocol.get("path") != protocol_path
        or protocol.get("sha256")
        != _sha256_regular_file(snapshot_root, protocol_path)
    ):
        raise StudySourceLockError(
            "manifest protocol status or embedded protocol digest is stale"
        )
    tasks = manifest.get("tasks")
    seeds = manifest.get("training_seeds")
    generations = manifest.get("checkpoint_generations")
    expected_tasks = [
        {"task_index": index, "env_name": env_name}
        for index, env_name, _ in TASKS
    ]
    if (
        not isinstance(tasks, list)
        or [
            {"task_index": item.get("task_index"), "env_name": item.get("env_name")}
            for item in tasks
            if isinstance(item, dict)
        ]
        != expected_tasks
        or seeds != list(TRAINING_SEEDS)
        or generations != list(CHECKPOINT_GENERATIONS)
    ):
        raise StudySourceLockError("manifest and array mapping contracts disagree")
    training_rows = [training_coordinates(index) for index in range(TRAINING_TASK_COUNT)]
    checkpoint_rows = [
        checkpoint_coordinates(index) for index in range(CHECKPOINT_TASK_COUNT)
    ]
    if (
        len(set(training_rows)) != TRAINING_TASK_COUNT
        or [row[0] for row in training_rows] != list(range(TRAINING_TASK_COUNT))
        or len(set(checkpoint_rows)) != CHECKPOINT_TASK_COUNT
        or [row[0] for row in checkpoint_rows] != list(range(CHECKPOINT_TASK_COUNT))
    ):
        raise StudySourceLockError("array mappings are not bijective")


def validate_runtime_dependency_versions() -> dict[str, str]:
    """Require the active interpreter to match the checked-in environment."""

    if tuple(sys.version_info[:3]) != (3, 10, 18):
        raise StudySourceLockError(
            "active Python must be exactly 3.10.18; found "
            + ".".join(map(str, sys.version_info[:3]))
        )
    found: dict[str, str] = {}
    for distribution, expected in RUNTIME_DEPENDENCY_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise StudySourceLockError(
                f"required distribution is missing: {distribution}"
            ) from error
        if actual != expected:
            raise StudySourceLockError(
                f"{distribution} version mismatch: expected {expected}, found {actual}"
            )
        found[distribution] = actual
    return found


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    digest_parser = subparsers.add_parser("digest")
    digest_parser.add_argument("--snapshot-root", default=_repository_root())
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--snapshot-root", default=_repository_root())
    verify_parser.add_argument("--expected", default=None)
    bundle_parser = subparsers.add_parser("verify-bundle")
    bundle_parser.add_argument("--snapshot-root", default=_repository_root())
    bundle_parser.add_argument("--bundle", required=True)
    bundle_parser.add_argument("--expected", required=True)
    bundle_parser.add_argument("--kind", required=True)
    training_parser = subparsers.add_parser("training-map")
    training_parser.add_argument("task_id", type=int)
    checkpoint_parser = subparsers.add_parser("checkpoint-map")
    checkpoint_parser.add_argument("task_id", type=int)
    mapping_parser = subparsers.add_parser("validate-mappings")
    mapping_parser.add_argument("--snapshot-root", default=_repository_root())
    subparsers.add_parser("validate-runtime")
    args = parser.parse_args()
    if args.command == "digest":
        print(compute_lagged_subspace_study_sha256(args.snapshot_root))
    elif args.command == "verify":
        print(
            require_lagged_subspace_study_source_lock(
                args.expected, snapshot_root=args.snapshot_root
            )
        )
    elif args.command == "verify-bundle":
        print(
            validate_hash_bundle(
                args.snapshot_root,
                args.bundle,
                expected_bundle_sha256=args.expected,
                expected_kind=args.kind,
            )
        )
    elif args.command == "training-map":
        print("\t".join(map(str, training_coordinates(args.task_id))))
    elif args.command == "checkpoint-map":
        print("\t".join(map(str, checkpoint_coordinates(args.task_id))))
    elif args.command == "validate-mappings":
        validate_manifest_mapping(args.snapshot_root)
        print(f"training={TRAINING_TASK_COUNT}\tdiagnostic={CHECKPOINT_TASK_COUNT}")
    else:
        print(json.dumps(validate_runtime_dependency_versions(), sort_keys=True))


if __name__ == "__main__":
    _main()

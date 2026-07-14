#!/usr/bin/env python3
"""Collect a fail-closed compute and storage disclosure for the lagged study.

The final audit index contains scientific outcomes.  This collector never
deserializes its scientific record arrays.  It lexically counts those arrays
and decodes only the small top-level provenance and budget objects required
for workload accounting.

Slurm accounting is optional because some clusters do not retain or expose
it.  When it is absent (or when a field is absent from every task record), the
report records a JSON null together with a reason.  Requested allocations are
always reported separately from observed usage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import errno
import hashlib
import json
import math
import mmap
import os
import re
import stat
import sys
import tempfile
from collections import defaultdict
from statistics import median
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.lagged_subspace_study_lock import (
    StudySourceLockError,
    compute_lagged_subspace_study_sha256,
)


STUDY = "lagged_subspace_frozen_checkpoint"
SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRODUCTION_MANIFEST_SHA256 = (
    "8081421fdd03d282b2febe33ffdc3b457115d8c4e98ca8eb2a702ac495d94087"
)
PRODUCTION_CARDINALITIES = {
    "training_runs": 60,
    "checkpoints": 180,
    "banks": 360,
    "partitions": 3_600,
    "checkpoint_metrics": 540,
    "center_endpoints": 1_800,
    "endpoints": 432_000,
}
PRODUCTION_POLICY_BUDGET = {
    "checkpoint_training_candidate_rollouts": 3_000_000,
    "normalization_calibration_rollouts": 180,
    "bank_candidate_rollouts": 1_440_000,
    "endpoint_arm_rollouts": 432_000,
    "checkpoint_center_rollouts": 1_800,
    "total_policy_rollouts": 4_873_980,
    "environment_transitions_are_separate": True,
}
STAGE_REPORT_KEYS = {
    "schema_version",
    "study",
    "stage",
    "status",
    "designation",
    "manifest_sha256",
    "provenance",
    "counts",
    "budget",
    "training_runs",
    "checkpoints",
    "no_outcome_selection",
    "no_forbidden_controls",
    "report_sha256",
}
LOCK_KEYS = {
    "source_sha256",
    "manifest_sha256",
    "protocol_sha256",
    "analyzer_sha256",
    "launcher_sha256",
    "dependency_lock_sha256",
}
STAGE_PROVENANCE_KEYS = LOCK_KEYS | {
    "source_snapshot_path",
    "validator_sha256",
    "assembler_sha256",
    "study_lock_sha256",
    "stderr_empty",
    "validation_mode",
    "record_sha256",
}
AUDIT_PROVENANCE_KEYS = LOCK_KEYS | {
    "source_snapshot_path",
    "stderr_empty",
    "documented_infrastructure_failures",
    "record_sha256",
}
AUDIT_TOP_LEVEL_KEYS = {
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
AUDIT_ARRAY_KEYS = {
    "training_runs",
    "checkpoints",
    "banks",
    "partitions",
    "checkpoint_metrics",
    "center_endpoints",
    "endpoints",
}
AUDIT_DECODE_KEYS = {
    "schema_version",
    "study",
    "designation",
    "manifest_sha256",
    "provenance",
    "budget",
}
LAUNCHERS = {
    "checkpoint_generation": "scripts/submit_lagged_subspace_checkpoint_generation.sh",
    "diagnostic": "scripts/submit_lagged_subspace_diagnostic.sh",
}
LAUNCHER_BUNDLE = "experiments/manifests/lagged_subspace_launcher_lock.json"
DEPENDENCY_BUNDLE = "experiments/manifests/lagged_subspace_dependency_lock.json"
STUDY_LOCK = "experiments/lagged_subspace_study_lock.py"
MANIFEST_RELATIVE = "experiments/manifests/lagged_subspace_frozen_checkpoint.json"
DIRECT_SNAPSHOT_LOCKS = {
    "manifest_sha256": MANIFEST_RELATIVE,
    "protocol_sha256": "docs/lagged_subspace_frozen_checkpoint_protocol.md",
    "analyzer_sha256": "scripts/analyze_lagged_subspace_frozen_checkpoint.py",
    "assembler_sha256": "scripts/assemble_lagged_subspace_frozen_checkpoint.py",
    "study_lock_sha256": STUDY_LOCK,
}
CHECKPOINT_STAGE_VALIDATOR = "scripts/validate_lagged_subspace_checkpoint_stage.py"
TRAINING_RECORD_KEYS = {
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
CHECKPOINT_STAGE_RECORD_KEYS = {
    "checkpoint_id",
    "training_id",
    "task_index",
    "env_name",
    "training_seed",
    "generation",
    "checkpoint_artifact_path",
    "checkpoint_artifact_sha256",
    "training_config_path",
    "training_config_sha256",
    "capture_manifest_path",
    "capture_manifest_sha256",
    "strictly_prior_gradient_archive",
    "reward_selection_used",
    "record_sha256",
}
POLICY_BUDGET_KEYS = {
    "checkpoint_training_candidate_rollouts",
    "normalization_calibration_rollouts",
    "bank_candidate_rollouts",
    "endpoint_arm_rollouts",
    "checkpoint_center_rollouts",
    "total_policy_rollouts",
}
TRANSITION_BUDGET_KEYS = {
    "checkpoint_training_transitions",
    "normalization_calibration_transitions",
    "bank_transitions",
    "endpoint_arm_transitions",
    "checkpoint_center_transitions",
    "total_environment_transitions",
}
SBATCH_RE = re.compile(r"^#SBATCH\s+--([A-Za-z0-9-]+)(?:=|\s+)(\S+)\s*$")
ARRAY_RE = re.compile(r"(\d+)-(\d+)(?:%(\d+))?")
MEMORY_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)([KMGTP]?)", re.IGNORECASE)
ROOT_JOB_RE = re.compile(r"([0-9]+)_([0-9]+)")
STEP_JOB_RE = re.compile(r"([0-9]+)_([0-9]+)\.(.+)")
UNAVAILABLE_ACCOUNTING = "no_slurm_accounting_export_supplied"


class DisclosureError(RuntimeError):
    """The disclosure inputs do not establish the requested claims."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_file_measurement(path: str, label: str) -> dict[str, Any]:
    """Hash one regular file while rejecting an in-place concurrent change."""

    _require_regular(path, label)
    before = os.stat(path, follow_symlinks=False)
    digest = _sha256_file(path)
    after = os.stat(path, follow_symlinks=False)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise DisclosureError(f"{label} changed while it was hashed")
    return {"bytes": after.st_size, "sha256": digest}


def _require_committed_file(
    path: str, *, expected_sha256: Any, label: str
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise DisclosureError(f"{label} requires a caller-supplied SHA-256 commitment")
    measurement = _stable_file_measurement(path, label)
    if measurement["sha256"] != expected_sha256:
        raise DisclosureError(f"{label} differs from its caller-supplied committed digest")
    return measurement


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DisclosureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise DisclosureError(f"nonfinite JSON constant is forbidden: {value}")


def _read_json(path: str, label: str) -> Any:
    _require_regular(path, label)
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DisclosureError(f"{label} is not valid strict JSON: {error}") from error


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _stamp_report(report: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(report)
    if "report_sha256" in result:
        raise DisclosureError("cannot stamp an already stamped disclosure")
    result["report_sha256"] = _report_sha256(result)
    return result


def _record_sha256(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _require_regular(path: str, label: str) -> str:
    if os.path.islink(path) or not os.path.isfile(path):
        raise DisclosureError(f"{label} must be a non-symlink regular file")
    return path


def _require_directory(path: str, label: str) -> str:
    if os.path.islink(path) or not os.path.isdir(path):
        raise DisclosureError(f"{label} must be a non-symlink directory")
    return path


def _normalized_relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DisclosureError(f"{label} must be a nonempty relative path")
    normalized = os.path.normpath(value)
    if (
        os.path.isabs(value)
        or normalized != value
        or value == ".."
        or value.startswith("../")
        or "\\" in value
    ):
        raise DisclosureError(f"{label} is not normalized and relative: {value!r}")
    return value


def _inside(root: str, relative: Any, label: str) -> str:
    relative = _normalized_relative(relative, label)
    root_real = os.path.realpath(root)
    path = os.path.join(root_real, relative)
    if os.path.commonpath([root_real, os.path.realpath(path)]) != root_real:
        raise DisclosureError(f"{label} escapes its root")
    return path


def _null_metric(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        raise DisclosureError("cannot summarize an empty numeric sequence")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise DisclosureError("runtime/accounting values must be finite")
    return {
        "count": len(numbers),
        "sum": float(sum(numbers)),
        "minimum": float(min(numbers)),
        "median": float(median(numbers)),
        "maximum": float(max(numbers)),
    }


# The scanner below deliberately does not deserialize skipped values.  In the
# final audit, those skipped values include every return and mechanism metric.
def _skip_ws(data: Any, index: int, limit: int) -> int:
    while index < limit and data[index] in b" \t\r\n":
        index += 1
    return index


def _string_end(data: Any, index: int, limit: int) -> int:
    if index >= limit or data[index] != ord('"'):
        raise DisclosureError("expected a JSON string")
    index += 1
    while index < limit:
        value = data[index]
        if value == ord('"'):
            return index + 1
        if value == ord("\\"):
            if index + 1 >= limit:
                raise DisclosureError("unterminated JSON escape")
            escaped = data[index + 1]
            if escaped in b'"\\/bfnrt':
                index += 2
                continue
            if escaped == ord("u"):
                stop = index + 6
                if stop > limit or not re.fullmatch(
                    rb"[0-9a-fA-F]{4}", bytes(data[index + 2 : stop])
                ):
                    raise DisclosureError("invalid JSON unicode escape")
                index = stop
                continue
            raise DisclosureError("invalid JSON string escape")
        if value < 0x20:
            raise DisclosureError("unescaped control byte in JSON string")
        if value < 0x80:
            index += 1
            continue
        # Validate UTF-8 without decoding or retaining the string value.
        if 0xC2 <= value <= 0xDF:
            widths = (2,)
        elif 0xE0 <= value <= 0xEF:
            widths = (3,)
        elif 0xF0 <= value <= 0xF4:
            widths = (4,)
        else:
            raise DisclosureError("invalid UTF-8 leading byte in JSON string")
        width = widths[0]
        stop = index + width
        if stop > limit:
            raise DisclosureError("truncated UTF-8 sequence in JSON string")
        continuation = list(data[index + 1 : stop])
        if any(not 0x80 <= byte <= 0xBF for byte in continuation):
            raise DisclosureError("invalid UTF-8 continuation byte in JSON string")
        second = continuation[0]
        if (
            (value == 0xE0 and second < 0xA0)
            or (value == 0xED and second > 0x9F)
            or (value == 0xF0 and second < 0x90)
            or (value == 0xF4 and second > 0x8F)
        ):
            raise DisclosureError("non-scalar or overlong UTF-8 in JSON string")
        index = stop
    raise DisclosureError("unterminated JSON string")


def _decode_slice(data: Any, start: int, stop: int, label: str) -> Any:
    try:
        return json.loads(
            bytes(data[start:stop]),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise DisclosureError(f"cannot decode selected {label}: {error}") from error


def _skip_primitive(data: Any, index: int, limit: int) -> int:
    start = index
    while index < limit and data[index] not in b",]} \t\r\n":
        index += 1
    token = bytes(data[start:index])
    if not token or not re.fullmatch(
        rb"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)",
        token,
    ):
        raise DisclosureError("invalid JSON primitive while scanning")
    return index


def _skip_value(data: Any, index: int, limit: int) -> int:
    index = _skip_ws(data, index, limit)
    if index >= limit:
        raise DisclosureError("missing JSON value")
    value = data[index]
    if value == ord('"'):
        return _string_end(data, index, limit)
    if value == ord("{"):
        index = _skip_ws(data, index + 1, limit)
        if index < limit and data[index] == ord("}"):
            return index + 1
        while True:
            key_stop = _string_end(data, index, limit)
            index = _skip_ws(data, key_stop, limit)
            if index >= limit or data[index] != ord(":"):
                raise DisclosureError("object key is not followed by a colon")
            index = _skip_value(data, index + 1, limit)
            index = _skip_ws(data, index, limit)
            if index < limit and data[index] == ord("}"):
                return index + 1
            if index >= limit or data[index] != ord(","):
                raise DisclosureError("object item is not followed by comma or close")
            index = _skip_ws(data, index + 1, limit)
    if value == ord("["):
        index = _skip_ws(data, index + 1, limit)
        if index < limit and data[index] == ord("]"):
            return index + 1
        while True:
            index = _skip_value(data, index, limit)
            index = _skip_ws(data, index, limit)
            if index < limit and data[index] == ord("]"):
                return index + 1
            if index >= limit or data[index] != ord(","):
                raise DisclosureError("array item is not followed by comma or close")
            index = _skip_ws(data, index + 1, limit)
    return _skip_primitive(data, index, limit)


def _count_array(data: Any, index: int, limit: int) -> tuple[int, int]:
    index = _skip_ws(data, index, limit)
    if index >= limit or data[index] != ord("["):
        raise DisclosureError("cardinality field is not an array")
    count = 0
    index = _skip_ws(data, index + 1, limit)
    if index < limit and data[index] == ord("]"):
        return 0, index + 1
    while True:
        index = _skip_value(data, index, limit)
        count += 1
        index = _skip_ws(data, index, limit)
        if index < limit and data[index] == ord("]"):
            return count, index + 1
        if index >= limit or data[index] != ord(","):
            raise DisclosureError("cardinality array is malformed")
        index = _skip_ws(data, index + 1, limit)


def _scan_object(
    data: Any,
    *,
    decode_keys: set[str],
    count_array_keys: set[str] = frozenset(),
    decode_handlers: Mapping[str, Callable[[Any, int, int], Any]] | None = None,
    expected_keys: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, int], set[str]]:
    handlers = {} if decode_handlers is None else decode_handlers
    limit = len(data)
    index = _skip_ws(data, 0, limit)
    if index >= limit or data[index] != ord("{"):
        raise DisclosureError("selected JSON input is not an object")
    index = _skip_ws(data, index + 1, limit)
    decoded: dict[str, Any] = {}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    if index < limit and data[index] == ord("}"):
        index += 1
    else:
        while True:
            key_start = index
            key_stop = _string_end(data, key_start, limit)
            key = _decode_slice(data, key_start, key_stop, "object key")
            if not isinstance(key, str) or key in seen:
                raise DisclosureError("top-level JSON key is invalid or duplicated")
            seen.add(key)
            index = _skip_ws(data, key_stop, limit)
            if index >= limit or data[index] != ord(":"):
                raise DisclosureError("top-level key lacks a colon")
            value_start = _skip_ws(data, index + 1, limit)
            if key in count_array_keys:
                count, value_stop = _count_array(data, value_start, limit)
                counts[key] = count
            else:
                value_stop = _skip_value(data, value_start, limit)
                if key in handlers:
                    decoded[key] = handlers[key](
                        data, value_start, value_stop
                    )
                elif key in decode_keys:
                    decoded[key] = _decode_slice(
                        data, value_start, value_stop, f"field {key}"
                    )
            index = _skip_ws(data, value_stop, limit)
            if index < limit and data[index] == ord("}"):
                index += 1
                break
            if index >= limit or data[index] != ord(","):
                raise DisclosureError("top-level item lacks comma or close")
            index = _skip_ws(data, index + 1, limit)
    if _skip_ws(data, index, limit) != limit:
        raise DisclosureError("trailing content follows the JSON object")
    if expected_keys is not None and seen != expected_keys:
        raise DisclosureError(
            "top-level audit schema mismatch: "
            f"missing={sorted(expected_keys - seen)}, extras={sorted(seen - expected_keys)}"
        )
    return decoded, counts, seen


def _decode_audit_provenance(data: Any, start: int, stop: int) -> dict[str, Any]:
    """Decode fixed provenance scalars while never decoding failure-list elements."""

    view = memoryview(data)[start:stop]
    try:
        decoded, counts, _ = _scan_object(
            view,
            decode_keys=AUDIT_PROVENANCE_KEYS
            - {"documented_infrastructure_failures"},
            count_array_keys={"documented_infrastructure_failures"},
            expected_keys=AUDIT_PROVENANCE_KEYS,
        )
    finally:
        view.release()
    if counts.get("documented_infrastructure_failures") != 0:
        raise DisclosureError(
            "final audit documented_infrastructure_failures must be an empty array"
        )
    decoded["documented_infrastructure_failures"] = []
    return decoded


def _scan_audit(path: str) -> tuple[dict[str, Any], dict[str, int]]:
    _require_regular(path, "final audit index")
    with open(path, "rb") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            raise DisclosureError("final audit index is empty")
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            decoded, counts, _ = _scan_object(
                mapped,
                decode_keys=AUDIT_DECODE_KEYS - {"provenance"},
                count_array_keys=AUDIT_ARRAY_KEYS,
                decode_handlers={"provenance": _decode_audit_provenance},
                expected_keys=AUDIT_TOP_LEVEL_KEYS,
            )
    return decoded, counts


def _selected_line_fields(line: bytes, keys: set[str]) -> dict[str, Any]:
    decoded, _, _ = _scan_object(line, decode_keys=keys)
    if set(decoded) != keys:
        raise DisclosureError(
            f"validated training log is missing runtime fields: {sorted(keys - set(decoded))}"
        )
    return decoded


def _validate_stage_report(
    path: str, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    committed = _require_committed_file(
        path,
        expected_sha256=expected_sha256,
        label="checkpoint-stage report",
    )
    report = _read_json(path, "checkpoint-stage report")
    if _stable_file_measurement(path, "checkpoint-stage report") != committed:
        raise DisclosureError("checkpoint-stage report changed while it was validated")
    if not isinstance(report, dict) or set(report) != STAGE_REPORT_KEYS:
        raise DisclosureError("checkpoint-stage report schema is not exact")
    if report.get("report_sha256") != _report_sha256(report):
        raise DisclosureError("checkpoint-stage report hash is invalid")
    if (
        report.get("schema_version") != 1
        or report.get("study") != STUDY
        or report.get("stage") != "checkpoint_generation"
        or report.get("status") != "validated"
        or report.get("no_outcome_selection") is not True
        or report.get("no_forbidden_controls") is not True
    ):
        raise DisclosureError("checkpoint-stage report is not a successful validation")
    provenance = report.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != STAGE_PROVENANCE_KEYS:
        raise DisclosureError("checkpoint-stage provenance is absent")
    if provenance.get("record_sha256") != _record_sha256(provenance):
        raise DisclosureError("checkpoint-stage provenance record hash is invalid")
    if not LOCK_KEYS.issubset(provenance) or any(
        not _is_sha256(provenance.get(key)) for key in LOCK_KEYS
    ):
        raise DisclosureError("checkpoint-stage provenance locks are incomplete")
    if report.get("manifest_sha256") != provenance["manifest_sha256"]:
        raise DisclosureError("checkpoint-stage manifest locks disagree")
    return report, committed


def _expected_cardinalities(manifest: Mapping[str, Any]) -> dict[str, int]:
    try:
        dimensions = manifest["dimensions"]
        training = len(manifest["tasks"]) * len(manifest["training_seeds"])
        checkpoints = training * len(manifest["checkpoint_generations"])
        banks = checkpoints * len(dimensions["banks"])
        partitions = checkpoints * int(dimensions["bank_b_partition_count"])
        centers = checkpoints * int(dimensions["endpoint_episodes"])
        endpoints = (
            checkpoints
            * len(dimensions["locality_q"])
            * int(dimensions["bank_b_partition_count"])
            * len(dimensions["endpoint_arms"])
            * int(dimensions["endpoint_episodes"])
        )
    except (KeyError, TypeError, ValueError) as error:
        raise DisclosureError(f"manifest cannot define cardinalities: {error}") from error
    values = {
        "training_runs": training,
        "checkpoints": checkpoints,
        "banks": banks,
        "partitions": partitions,
        "checkpoint_metrics": checkpoints * len(dimensions["locality_q"]),
        "center_endpoints": centers,
        "endpoints": endpoints,
    }
    if any(isinstance(value, bool) or value <= 0 for value in values.values()):
        raise DisclosureError("manifest cardinalities must all be positive")
    return values


def _validate_manifest(
    path: str,
    *,
    expected_sha256: str,
    require_production: bool,
) -> dict[str, Any]:
    measurement = _stable_file_measurement(path, "manifest")
    if measurement["sha256"] != expected_sha256:
        raise DisclosureError("manifest digest differs from the validated stage lock")
    manifest = _read_json(path, "manifest")
    if _stable_file_measurement(path, "manifest") != measurement:
        raise DisclosureError("manifest changed while it was validated")
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("study") != STUDY
        or manifest.get("protocol_status")
        != "final_locked_before_environment_outcomes"
    ):
        raise DisclosureError("manifest is not the final locked lagged-subspace manifest")
    expected = _expected_cardinalities(manifest)
    if require_production and (
        measurement["sha256"] != PRODUCTION_MANIFEST_SHA256
        or expected != PRODUCTION_CARDINALITIES
        or manifest.get("budget") != PRODUCTION_POLICY_BUDGET
    ):
        raise DisclosureError(
            "final production mode requires the exact immutable production manifest"
        )
    return manifest


def _validate_record_hash(record: Mapping[str, Any], label: str) -> None:
    if record.get("record_sha256") != _record_sha256(record):
        raise DisclosureError(f"{label} record hash is invalid")


def _validate_stage_cardinality(
    stage: Mapping[str, Any], expected: Mapping[str, int]
) -> None:
    counts = stage.get("counts")
    training = stage.get("training_runs")
    checkpoints = stage.get("checkpoints")
    if (
        not isinstance(counts, dict)
        or counts.get("training_runs") != expected["training_runs"]
        or counts.get("checkpoints") != expected["checkpoints"]
        or counts.get("training_stderr_files") != expected["training_runs"]
        or not isinstance(training, list)
        or len(training) != expected["training_runs"]
        or not isinstance(checkpoints, list)
        or len(checkpoints) != expected["checkpoints"]
    ):
        raise DisclosureError("checkpoint-stage scientific cardinality is partial")
    if [item.get("training_id") for item in training if isinstance(item, dict)] != list(
        range(expected["training_runs"])
    ):
        raise DisclosureError("training identities are not complete and contiguous")
    if [item.get("checkpoint_id") for item in checkpoints if isinstance(item, dict)] != list(
        range(expected["checkpoints"])
    ):
        raise DisclosureError("checkpoint identities are not complete and contiguous")
    for index, record in enumerate(training):
        if not isinstance(record, dict) or set(record) != TRAINING_RECORD_KEYS:
            raise DisclosureError(f"training record {index} schema is not exact")
        _validate_record_hash(record, f"training record {index}")
    for index, record in enumerate(checkpoints):
        if not isinstance(record, dict) or set(record) != CHECKPOINT_STAGE_RECORD_KEYS:
            raise DisclosureError(f"checkpoint-stage record {index} schema is not exact")
        _validate_record_hash(record, f"checkpoint-stage record {index}")
        if (
            record["checkpoint_id"] != index
            or record["strictly_prior_gradient_archive"] is not True
            or record["reward_selection_used"] is not False
        ):
            raise DisclosureError(
                f"checkpoint-stage record {index} identity/selection metadata is invalid"
            )


def _validate_audit_metadata(
    decoded: Mapping[str, Any],
    observed: Mapping[str, int],
    *,
    stage: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected: Mapping[str, int],
    final_mode: bool,
) -> None:
    if (
        decoded.get("schema_version") != 1
        or decoded.get("study") != STUDY
        or decoded.get("designation") != manifest.get("designation")
        or decoded.get("manifest_sha256") != stage.get("manifest_sha256")
    ):
        raise DisclosureError("final audit top-level identity disagrees with its locks")
    provenance = decoded.get("provenance")
    stage_provenance = stage["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != AUDIT_PROVENANCE_KEYS:
        raise DisclosureError("final audit provenance is missing")
    _validate_record_hash(provenance, "final audit provenance")
    for key in LOCK_KEYS:
        if provenance.get(key) != stage_provenance.get(key):
            raise DisclosureError(f"stage and final audit provenance disagree on {key}")
    if provenance.get("source_snapshot_path") != stage_provenance.get(
        "source_snapshot_path"
    ):
        raise DisclosureError("stage and audit source snapshots disagree")
    if provenance.get("stderr_empty") is not True:
        raise DisclosureError("final audit does not validate empty application stderr")
    failures = provenance.get("documented_infrastructure_failures")
    if failures != []:
        raise DisclosureError(
            "final audit documented_infrastructure_failures must be empty"
        )
    if final_mode and dict(observed) != dict(expected):
        mismatch = {
            key: {"expected": expected[key], "observed": observed.get(key)}
            for key in sorted(expected)
            if observed.get(key) != expected[key]
        }
        raise DisclosureError(f"final audit scientific cardinality is partial: {mismatch}")


def _validate_budget(
    *,
    manifest: Mapping[str, Any],
    stage_budget: Mapping[str, Any],
    audit_budget: Mapping[str, Any] | None,
) -> dict[str, Any]:
    expected_policy = dict(manifest.get("budget", {}))
    if expected_policy.get("environment_transitions_are_separate") is not True:
        raise DisclosureError(
            "manifest must declare environment transitions separate from rollouts"
        )
    expected_policy.pop("environment_transitions_are_separate", None)
    if set(expected_policy) != POLICY_BUDGET_KEYS or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in expected_policy.values()
    ):
        raise DisclosureError("manifest policy-rollout budget schema is invalid")
    if expected_policy["total_policy_rollouts"] != sum(
        expected_policy[key]
        for key in POLICY_BUDGET_KEYS
        if key != "total_policy_rollouts"
    ):
        raise DisclosureError("manifest policy-rollout budget does not sum")
    stage_required = {
        "checkpoint_training_candidate_rollouts",
        "normalization_calibration_rollouts",
        "total_policy_rollouts",
        "checkpoint_training_transitions",
        "normalization_calibration_transitions",
        "total_environment_transitions",
    }
    if set(stage_budget) != stage_required:
        raise DisclosureError("checkpoint-stage workload budget schema is invalid")
    if (
        stage_budget["checkpoint_training_candidate_rollouts"]
        != expected_policy["checkpoint_training_candidate_rollouts"]
        or stage_budget["normalization_calibration_rollouts"]
        != expected_policy["normalization_calibration_rollouts"]
        or stage_budget["total_policy_rollouts"]
        != expected_policy["checkpoint_training_candidate_rollouts"]
        + expected_policy["normalization_calibration_rollouts"]
    ):
        raise DisclosureError("checkpoint-stage rollout counts disagree with manifest")
    for key, value in stage_budget.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DisclosureError(f"checkpoint-stage budget field {key} is invalid")
    if stage_budget["total_environment_transitions"] != (
        stage_budget["checkpoint_training_transitions"]
        + stage_budget["normalization_calibration_transitions"]
    ):
        raise DisclosureError("checkpoint-stage transition count does not sum")
    if audit_budget is None:
        return {
            "policy_rollouts": {
                key: stage_budget[key]
                for key in (
                    "checkpoint_training_candidate_rollouts",
                    "normalization_calibration_rollouts",
                    "total_policy_rollouts",
                )
            },
            "environment_transitions": {
                key: stage_budget[key]
                for key in (
                    "checkpoint_training_transitions",
                    "normalization_calibration_transitions",
                    "total_environment_transitions",
                )
            },
            "scope": "checkpoint_generation_only",
        }
    if set(audit_budget) != POLICY_BUDGET_KEYS | TRANSITION_BUDGET_KEYS:
        raise DisclosureError("final audit workload budget schema is invalid")
    for key, value in audit_budget.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise DisclosureError(f"final audit budget field {key} is invalid")
    if {key: audit_budget[key] for key in POLICY_BUDGET_KEYS} != expected_policy:
        raise DisclosureError("final audit rollout counts disagree with manifest")
    if audit_budget["total_environment_transitions"] != sum(
        audit_budget[key]
        for key in TRANSITION_BUDGET_KEYS
        if key != "total_environment_transitions"
    ):
        raise DisclosureError("final audit transition count does not sum")
    for key in (
        "checkpoint_training_candidate_rollouts",
        "normalization_calibration_rollouts",
        "checkpoint_training_transitions",
        "normalization_calibration_transitions",
    ):
        if audit_budget[key] != stage_budget[key]:
            raise DisclosureError(f"stage and final audit budget disagree on {key}")
    return {
        "policy_rollouts": {
            key: audit_budget[key] for key in sorted(POLICY_BUDGET_KEYS)
        },
        "environment_transitions": {
            key: audit_budget[key] for key in sorted(TRANSITION_BUDGET_KEYS)
        },
        "scope": "complete_two_stage_study",
    }


def _resolve_snapshot(artifact_root: str, stage: Mapping[str, Any]) -> tuple[str, str]:
    relative = _normalized_relative(
        stage["provenance"].get("source_snapshot_path"), "source snapshot path"
    )
    path = _inside(artifact_root, relative, "source snapshot path")
    _require_directory(path, "source snapshot")
    return path, relative


def _validate_snapshot_source(snapshot: str, stage: Mapping[str, Any]) -> None:
    provenance = stage["provenance"]
    expected_source = provenance.get("source_sha256")
    if not _is_sha256(expected_source):
        raise DisclosureError("checkpoint-stage source digest is invalid")
    try:
        before = compute_lagged_subspace_study_sha256(snapshot)
    except StudySourceLockError as error:
        raise DisclosureError(f"cannot validate immutable source snapshot: {error}") from error
    if before != expected_source:
        raise DisclosureError(
            "immutable source snapshot digest differs from checkpoint-stage provenance"
        )
    for lock_name, relative in DIRECT_SNAPSHOT_LOCKS.items():
        expected = provenance.get(lock_name)
        if not _is_sha256(expected):
            raise DisclosureError(f"checkpoint-stage {lock_name} is invalid")
        path = _inside(snapshot, relative, f"locked {lock_name} file")
        measurement = _stable_file_measurement(path, f"locked {lock_name} file")
        if measurement["sha256"] != expected:
            raise DisclosureError(
                f"locked snapshot file differs from {lock_name}: {relative}"
            )
    try:
        after = compute_lagged_subspace_study_sha256(snapshot)
    except StudySourceLockError as error:
        raise DisclosureError(f"cannot revalidate immutable source snapshot: {error}") from error
    if after != before:
        raise DisclosureError("immutable source snapshot changed while it was validated")


def _validate_checkpoint_stage_validator(
    path: str, stage: Mapping[str, Any]
) -> dict[str, Any]:
    expected = stage["provenance"].get("validator_sha256")
    if not _is_sha256(expected):
        raise DisclosureError("checkpoint-stage validator digest is invalid")
    measurement = _stable_file_measurement(path, "checkpoint-stage validator")
    if measurement["sha256"] != expected:
        raise DisclosureError(
            "checkpoint-stage validator differs from its stage-report digest"
        )
    return measurement


def _validate_bundle(
    snapshot: str,
    relative: str,
    *,
    expected_sha256: str,
    expected_kind: str,
) -> dict[str, str]:
    path = _inside(snapshot, relative, f"{expected_kind} bundle")
    measurement = _stable_file_measurement(path, f"{expected_kind} bundle")
    if measurement["sha256"] != expected_sha256:
        raise DisclosureError(f"{expected_kind} bundle digest differs from stage lock")
    bundle = _read_json(path, f"{expected_kind} bundle")
    if _stable_file_measurement(path, f"{expected_kind} bundle") != measurement:
        raise DisclosureError(f"{expected_kind} bundle changed while it was validated")
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") != 1
        or bundle.get("study") != STUDY
        or bundle.get("kind") != expected_kind
        or set(bundle) != {"schema_version", "study", "kind", "files"}
        or not isinstance(bundle.get("files"), list)
    ):
        raise DisclosureError(f"{expected_kind} bundle schema is invalid")
    result: dict[str, str] = {}
    for item in bundle["files"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not _is_sha256(item.get("sha256"))
        ):
            raise DisclosureError(f"{expected_kind} bundle entry is invalid")
        item_relative = _normalized_relative(item["path"], "bundle file path")
        if item_relative in result:
            raise DisclosureError(f"duplicate {expected_kind} bundle path")
        item_path = _inside(snapshot, item_relative, "bundle file")
        if _stable_file_measurement(item_path, "bundle file")["sha256"] != item[
            "sha256"
        ]:
            raise DisclosureError(f"locked bundle file digest mismatch: {item_relative}")
        result[item_relative] = item["sha256"]
    return result


def _parse_walltime(value: str) -> int:
    match = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d+):(\d+)", value)
    if match is None:
        raise DisclosureError(f"unsupported Slurm walltime: {value}")
    days, hours, minutes, seconds = match.groups()
    result = ((int(days or 0) * 24 + int(hours)) * 60 + int(minutes)) * 60 + int(seconds)
    if result <= 0 or int(minutes) >= 60 or int(seconds) >= 60:
        raise DisclosureError(f"invalid Slurm walltime: {value}")
    return result


def _parse_memory_bytes(value: str, *, default_unit: str | None) -> int:
    match = MEMORY_RE.fullmatch(value)
    if match is None:
        raise DisclosureError(f"unsupported Slurm memory request: {value}")
    amount, unit = match.groups()
    if not unit:
        if default_unit is None:
            raise DisclosureError(
                f"memory value has no unit and no context default: {value}"
            )
        unit = default_unit
    multiplier = 1024 ** {"": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[unit.upper()]
    result = float(amount) * multiplier
    if not math.isfinite(result) or result <= 0 or not result.is_integer():
        raise DisclosureError(f"nonintegral Slurm memory request: {value}")
    return int(result)


def _parse_launcher(path: str, *, expected_tasks: int) -> dict[str, Any]:
    measurement = _stable_file_measurement(path, "locked launcher")
    directives: dict[str, str] = {}
    assignments: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as stream:
            for line in stream:
                match = SBATCH_RE.match(line.rstrip("\n"))
                if match:
                    key, value = match.groups()
                    if key in directives:
                        raise DisclosureError(f"duplicate SBATCH directive: {key}")
                    directives[key] = value
                assignment = re.fullmatch(
                    r"(WORKERS|CHUNK_PAIRS)=([0-9]+)\s*", line
                )
                if assignment:
                    assignments[assignment.group(1)] = int(assignment.group(2))
    except (OSError, UnicodeError) as error:
        raise DisclosureError(f"cannot read locked launcher: {error}") from error
    if _stable_file_measurement(path, "locked launcher") != measurement:
        raise DisclosureError("locked launcher changed while it was parsed")
    required = {"job-name", "output", "error", "partition", "cpus-per-task", "mem", "time", "array"}
    if not required.issubset(directives):
        raise DisclosureError(f"locked launcher lacks SBATCH fields: {sorted(required - set(directives))}")
    gpu_directives = {
        "gres",
        "gpus",
        "gpus-per-task",
        "gpus-per-node",
        "gpus-per-socket",
    }.intersection(directives)
    if "gpu" in directives.get("tres-per-task", "").lower():
        gpu_directives.add("tres-per-task")
    if gpu_directives:
        raise DisclosureError(
            f"unsupported GPU allocation in locked launcher: {sorted(gpu_directives)}"
        )
    array = ARRAY_RE.fullmatch(directives["array"])
    if array is None:
        raise DisclosureError("launcher array directive is unsupported")
    start, stop, concurrency = (int(value) if value is not None else None for value in array.groups())
    task_count = stop - start + 1
    if start != 0 or task_count != expected_tasks or concurrency is None or concurrency <= 0:
        raise DisclosureError("launcher array cardinality/concurrency differs from manifest")
    cpus = int(directives["cpus-per-task"])
    workers = assignments.get("WORKERS")
    if cpus <= 0 or workers is None or workers <= 0 or workers >= cpus:
        raise DisclosureError("launcher CPU/worker allocation is invalid")
    # Slurm interprets a suffixless --mem request in megabytes.  This is not
    # the same context as sacct's MaxRSS display field.
    memory = _parse_memory_bytes(directives["mem"], default_unit="M")
    walltime = _parse_walltime(directives["time"])
    return {
        "array_task_count": task_count,
        "array_index_start": start,
        "array_index_end": stop,
        "maximum_concurrent_tasks": concurrency,
        "slurm_cpus_per_task": cpus,
        "application_workers_per_task": workers,
        "slurm_cpus_not_assigned_to_application_workers": cpus - workers,
        "memory_bytes_per_task": memory,
        "walltime_seconds_per_task": walltime,
        "partition": directives["partition"],
        "job_name": directives["job-name"],
        "gpu_allocation_requested": False,
        "maximum_concurrent_slurm_cpus": concurrency * cpus,
        "maximum_concurrent_memory_bytes": concurrency * memory,
        "requested_slurm_cpu_seconds_upper_bound": task_count * cpus * walltime,
        "requested_memory_byte_seconds_upper_bound": task_count * memory * walltime,
        "requested_task_wallclock_seconds_upper_bound": task_count * walltime,
        "allocation_semantics": (
            "upper bounds from locked SBATCH directives, not observed usage; "
            "Slurm CPUs are not asserted to be physical cores"
        ),
        **(
            {"chunk_pairs_per_worker_batch": assignments["CHUNK_PAIRS"]}
            if "CHUNK_PAIRS" in assignments
            else {}
        ),
    }


def _locked_runtime_versions(snapshot: str, stage: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(snapshot, STUDY_LOCK, "study lock")
    measurement = _stable_file_measurement(path, "study lock")
    actual_sha = measurement["sha256"]
    expected_sha = stage["provenance"].get("study_lock_sha256")
    if not _is_sha256(expected_sha) or actual_sha != expected_sha:
        raise DisclosureError("study-lock digest differs from checkpoint-stage provenance")
    try:
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        tree = ast.parse(source, filename=STUDY_LOCK)
    except (OSError, UnicodeError, SyntaxError) as error:
        raise DisclosureError(f"cannot parse locked runtime declaration: {error}") from error
    if _stable_file_measurement(path, "study lock") != measurement:
        raise DisclosureError("study lock changed while runtime metadata was extracted")
    dependencies: Any = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RUNTIME_DEPENDENCY_VERSIONS"
            for target in node.targets
        ):
            dependencies = ast.literal_eval(node.value)
    version_match = re.search(
        r"tuple\(sys\.version_info\[:3\]\)\s*!=\s*\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)",
        source,
    )
    if (
        not isinstance(dependencies, dict)
        or not dependencies
        or any(not isinstance(key, str) or not isinstance(value, str) for key, value in dependencies.items())
        or version_match is None
    ):
        raise DisclosureError("locked runtime versions cannot be extracted exactly")
    return {
        "python": ".".join(version_match.groups()),
        "distributions": dict(sorted(dependencies.items())),
        "source_path": STUDY_LOCK,
        "source_sha256": actual_sha,
        "launcher_gate_declaration": (
            "validate-runtime command is present exactly once as a command line "
            "in each locked launcher"
        ),
    }


def _validate_launcher_runtime_gates(snapshot: str) -> None:
    marker = re.compile(
        r"python -m experiments\.lagged_subspace_study_lock "
        r"validate-runtime(?: >/dev/null)?"
    )
    for stage, relative in LAUNCHERS.items():
        path = _inside(snapshot, relative, f"{stage} launcher")
        measurement = _stable_file_measurement(path, f"{stage} launcher")
        try:
            with open(path, encoding="utf-8") as stream:
                source = stream.read()
        except (OSError, UnicodeError) as error:
            raise DisclosureError(f"cannot inspect {stage} runtime gate: {error}") from error
        if _stable_file_measurement(path, f"{stage} launcher") != measurement:
            raise DisclosureError(f"{stage} launcher changed while it was inspected")
        command_lines = [line.strip() for line in source.splitlines()]
        if sum(marker.fullmatch(line) is not None for line in command_lines) != 1:
            raise DisclosureError(
                f"{stage} launcher does not contain exactly one locked runtime gate"
            )


def _training_application_runtime(
    artifact_root: str, training_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    per_run: list[float] = []
    iteration_count = 0
    for record_index, record in enumerate(training_records):
        relative = _normalized_relative(record.get("training_log_path"), "training log path")
        path = _inside(artifact_root, relative, "training log")
        expected_sha = record.get("training_log_sha256")
        if not _is_sha256(expected_sha) or _sha256_file(_require_regular(path, "training log")) != expected_sha:
            raise DisclosureError(f"training log {record_index} differs from its stage-report hash")
        expected_updates = record.get("updates")
        if isinstance(expected_updates, bool) or not isinstance(expected_updates, int) or expected_updates <= 0:
            raise DisclosureError("training record has invalid update count")
        elapsed = 0.0
        rows = 0
        try:
            with open(path, "rb") as stream:
                for line in stream:
                    if not line.strip():
                        raise DisclosureError("blank training JSONL record")
                    fields = _selected_line_fields(
                        line, {"iteration", "iteration_compute_seconds"}
                    )
                    if fields["iteration"] != rows:
                        raise DisclosureError("training runtime rows are not contiguous")
                    seconds = fields["iteration_compute_seconds"]
                    if (
                        isinstance(seconds, bool)
                        or not isinstance(seconds, (int, float))
                        or not math.isfinite(float(seconds))
                        or float(seconds) < 0
                    ):
                        raise DisclosureError("training iteration runtime is invalid")
                    elapsed += float(seconds)
                    rows += 1
        except OSError as error:
            raise DisclosureError(f"cannot read validated training log: {error}") from error
        if rows != expected_updates:
            raise DisclosureError("training runtime log is partial")
        if _sha256_file(path) != expected_sha:
            raise DisclosureError(
                f"training log {record_index} changed while runtime metadata was read"
            )
        per_run.append(elapsed)
        iteration_count += rows
    return {
        "instrumented_iteration_compute_seconds": _numeric_summary(per_run),
        "covered_training_runs": len(per_run),
        "covered_iterations": iteration_count,
        "measurement_scope": (
            "sum of iteration_compute_seconds in hash-validated training JSONL; "
            "excludes launcher setup, calibration, and finalization"
        ),
    }


def _log_unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "distinct_job_task_records": None,
        "logical_tasks_observed": None,
        "distinct_job_resubmission_records": None,
        "same_job_requeue_or_restart_count": _null_metric(reason),
        "stdout": _null_metric(reason),
        "stderr": _null_metric(reason),
        "nonempty_stderr_attempts": None,
    }


def _parse_metadata_line(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in result:
            raise DisclosureError(f"duplicate launcher log metadata key: {key}")
        result[key] = value
    return result


def _validated_job_ids(job_ids: Sequence[str]) -> tuple[str, ...]:
    if not job_ids or len(set(job_ids)) != len(job_ids):
        raise DisclosureError("Slurm job ids must be nonempty and unique")
    if any(not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+", job_id) for job_id in job_ids):
        raise DisclosureError("Slurm job ids must contain only decimal digits")
    return tuple(sorted(job_ids, key=int))


def _collect_slurm_logs(
    log_root: str | None,
    *,
    stage: str,
    prefix: str,
    job_ids: Sequence[str],
    expected_tasks: int,
    expected_source_sha256: str,
    expected_snapshot: str,
    expected_workers: int,
    expected_task_metadata: Sequence[Mapping[str, str]],
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    if log_root is None:
        return _log_unavailable("no_slurm_log_root_supplied"), set()
    _require_directory(log_root, "Slurm log root")
    job_ids = _validated_job_ids(job_ids)
    if len(expected_task_metadata) != expected_tasks:
        raise DisclosureError("expected Slurm task metadata cardinality is invalid")
    attempts: set[tuple[str, int]] = set()
    stdout_inventory: list[dict[str, Any]] = []
    stderr_inventory: list[dict[str, Any]] = []
    nonempty_stderr = 0
    valid_metadata_attempts = 0
    logical_with_valid_metadata: set[int] = set()
    expected_stage = "checkpoint_generation" if stage == "checkpoint_generation" else "diagnostic"
    for job_id in job_ids:
        pattern = re.compile(rf"{re.escape(prefix)}_{re.escape(job_id)}_([0-9]+)\.(out|err)")
        matched: dict[tuple[int, str], str] = {}
        for name in os.listdir(log_root):
            match = pattern.fullmatch(name)
            if match is None:
                continue
            task_id, kind = int(match.group(1)), match.group(2)
            if task_id >= expected_tasks or (task_id, kind) in matched:
                raise DisclosureError("Slurm log task id is out of range or duplicated")
            matched[(task_id, kind)] = os.path.join(log_root, name)
        task_ids = {task_id for task_id, _ in matched}
        if not task_ids:
            raise DisclosureError(f"declared Slurm job {job_id} has no log attempts")
        for task_id in sorted(task_ids):
            if (task_id, "out") not in matched or (task_id, "err") not in matched:
                raise DisclosureError("Slurm stdout/stderr attempt pair is incomplete")
            key = (job_id, task_id)
            attempts.add(key)
            stdout_path = _require_regular(matched[(task_id, "out")], "Slurm stdout")
            stderr_path = _require_regular(matched[(task_id, "err")], "Slurm stderr")
            stdout_before = _stable_file_measurement(stdout_path, "Slurm stdout")
            try:
                with open(stdout_path, encoding="utf-8") as stream:
                    header = [stream.readline() for _ in range(4)]
            except (OSError, UnicodeError) as error:
                raise DisclosureError(f"cannot read Slurm stdout metadata: {error}") from error
            stdout_after = _stable_file_measurement(stdout_path, "Slurm stdout")
            if stdout_after != stdout_before:
                raise DisclosureError("Slurm stdout changed while metadata was read")
            first = _parse_metadata_line(header[0])
            source = _parse_metadata_line(header[1])
            identity = _parse_metadata_line(header[2])
            runtime = _parse_metadata_line(header[3])
            expected_identity = expected_task_metadata[task_id]
            metadata_valid = not (
                first.get("study") != STUDY
                or first.get("stage") != expected_stage
                or source.get("source_sha256") != expected_source_sha256
                or source.get("snapshot") != expected_snapshot
                or runtime.get("workers") != str(expected_workers)
                or runtime.get("dry_run") != "0"
                or any(identity.get(key) != value for key, value in expected_identity.items())
            )
            if metadata_valid:
                valid_metadata_attempts += 1
                logical_with_valid_metadata.add(task_id)
            stdout_inventory.append(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    **stdout_after,
                }
            )
            stderr_measurement = _stable_file_measurement(stderr_path, "Slurm stderr")
            stderr_size = stderr_measurement["bytes"]
            nonempty_stderr += int(stderr_size != 0)
            stderr_inventory.append(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    **stderr_measurement,
                }
            )
    logical = {task_id for _, task_id in attempts}
    if logical != set(range(expected_tasks)):
        raise DisclosureError(
            f"Slurm logs are partial for {stage}: expected {expected_tasks}, found {len(logical)}"
        )
    if logical_with_valid_metadata != logical:
        raise DisclosureError(
            f"Slurm logs lack a valid launcher identity for "
            f"{len(logical - logical_with_valid_metadata)} logical tasks"
        )
    return (
        {
            "available": True,
            "job_ids": list(job_ids),
            "job_id_scope": (
                "only explicitly supplied job ids; attempts from undisclosed job ids "
                "cannot be counted"
            ),
            "distinct_job_task_records": len(attempts),
            "logical_tasks_observed": len(logical),
            "distinct_job_resubmission_records": len(attempts) - len(logical),
            "same_job_requeue_or_restart_count": _null_metric(
                "Slurm stdout/stderr filenames do not expose Slurm restart-count fields"
            ),
            "launcher_metadata_valid_attempts": valid_metadata_attempts,
            "launcher_metadata_incomplete_or_invalid_attempts": (
                len(attempts) - valid_metadata_attempts
            ),
            "stdout": {
                "files": len(stdout_inventory),
                "bytes": sum(item["bytes"] for item in stdout_inventory),
                "inventory_sha256": hashlib.sha256(
                    _canonical_bytes(stdout_inventory)
                ).hexdigest(),
            },
            "stderr": {
                "files": len(stderr_inventory),
                "bytes": sum(item["bytes"] for item in stderr_inventory),
                "inventory_sha256": hashlib.sha256(
                    _canonical_bytes(stderr_inventory)
                ).hexdigest(),
            },
            "nonempty_stderr_attempts": nonempty_stderr,
            "contents_embedded_in_disclosure": False,
        },
        attempts,
    )


def _accounting_unavailable(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "distinct_job_task_records": None,
        "successful_distinct_job_task_records": None,
        "failed_or_incomplete_distinct_job_task_records": None,
        "distinct_job_resubmission_records": None,
        "same_job_requeue_or_restart_count": _null_metric(reason),
        "elapsed_seconds": _null_metric(reason),
        "allocated_slurm_cpu_seconds": _null_metric(reason),
        "maximum_observed_concurrent_tasks": _null_metric(reason),
        "maximum_observed_concurrent_slurm_cpus": _null_metric(reason),
        "max_rss_bytes": _null_metric(reason),
        "consumed_energy_joules": _null_metric(reason),
        "co2e_kg": _null_metric(
            "not_measured_and_no_energy_to_co2e_conversion_method_is_declared"
        ),
    }


def _clean_accounting_row(row: Mapping[str | None, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized = key.strip().lower()
        if normalized:
            result[normalized] = "" if value is None else str(value).strip()
    return result


def _field(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name.lower())
        if value is not None:
            return value
    return ""


def _field_with_name(
    row: Mapping[str, str], *names: str
) -> tuple[str, str | None]:
    for name in names:
        normalized = name.lower()
        if normalized in row:
            return row[normalized], name
    return "", None


def _parse_positive_int(value: str) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _parse_nonnegative_int(value: str) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _parse_maxrss(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() in {"unknown", "n/a", "none", "notset"}:
        return None
    try:
        # MaxRSS normally carries an explicit K/M/G/T/P display suffix.  A
        # suffixless export is ambiguous unless the sacct --units option is
        # recorded separately, so it is unavailable rather than guessed.
        result = _parse_memory_bytes(value, default_unit=None)
    except DisclosureError:
        return None
    return result if result > 0 else None


def _parse_timestamp(value: str) -> dt.datetime | None:
    value = value.strip()
    if not value or value.lower() in {"unknown", "n/a", "none"}:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return parsed


def _observed_concurrency(records: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    events: list[tuple[dt.datetime, int, int, int]] = []
    for record in records:
        start = record.get("start")
        end = record.get("end")
        cpus = record.get("alloc_cpus")
        if (
            not isinstance(start, dt.datetime)
            or not isinstance(end, dt.datetime)
            or end <= start
            or not isinstance(cpus, int)
        ):
            return None
        events.append((end, 0, -1, -cpus))
        events.append((start, 1, 1, cpus))
    current_tasks = 0
    current_cpus = 0
    max_tasks = 0
    max_cpus = 0
    for _, _, task_delta, cpu_delta in sorted(events):
        current_tasks += task_delta
        current_cpus += cpu_delta
        if current_tasks < 0 or current_cpus < 0:
            return None
        max_tasks = max(max_tasks, current_tasks)
        max_cpus = max(max_cpus, current_cpus)
    if current_tasks != 0 or current_cpus != 0:
        return None
    return max_tasks, max_cpus


def _collect_accounting(
    paths: Sequence[str],
    *,
    job_ids: Sequence[str],
    expected_tasks: int,
    log_attempts: set[tuple[str, int]],
) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    if not paths:
        return _accounting_unavailable(UNAVAILABLE_ACCOUNTING), set()
    job_ids = _validated_job_ids(job_ids)
    allowed_jobs = set(job_ids)
    roots: dict[tuple[str, int], dict[str, Any]] = {}
    steps: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    source_inventory: list[dict[str, Any]] = []
    for path in paths:
        before = _stable_file_measurement(path, "Slurm accounting export")
        source_inventory.append(
            {"name": os.path.basename(path), **before}
        )
        try:
            with open(path, newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream, delimiter="|")
                if reader.fieldnames is None:
                    raise DisclosureError("Slurm accounting export lacks a header")
                headers = [
                    header.strip().lower()
                    for header in reader.fieldnames
                    if header is not None and header.strip()
                ]
                if len(headers) != len(set(headers)):
                    raise DisclosureError("Slurm accounting export has duplicate headers")
                required_headers = {
                    "state",
                    "exitcode",
                    "elapsedraw",
                    "alloccpus",
                }
                if not required_headers.issubset(headers) or not {
                    "jobidraw",
                    "jobid",
                }.intersection(headers):
                    raise DisclosureError(
                        "Slurm accounting export lacks required root-task fields"
                    )
                for raw in reader:
                    row = _clean_accounting_row(raw)
                    job_raw = _field(row, "jobidraw", "jobid")
                    root_match = ROOT_JOB_RE.fullmatch(job_raw)
                    step_match = STEP_JOB_RE.fullmatch(job_raw)
                    if root_match:
                        job_id, task_text = root_match.groups()
                        if job_id not in allowed_jobs:
                            raise DisclosureError(f"accounting contains undeclared job {job_id}")
                        task_id = int(task_text)
                        key = (job_id, task_id)
                        if task_id >= expected_tasks or key in roots:
                            raise DisclosureError("accounting root task is out of range or duplicated")
                        elapsed = _parse_nonnegative_int(_field(row, "elapsedraw"))
                        cpus = _parse_nonnegative_int(_field(row, "alloccpus", "ncpus"))
                        if elapsed is None or cpus is None:
                            raise DisclosureError("accounting root task lacks elapsed/allocated CPU data")
                        state = _field(row, "state").split()[0].rstrip("+")
                        exit_code = _field(row, "exitcode")
                        completed_successfully = state == "COMPLETED" and exit_code == "0:0"
                        if completed_successfully and (elapsed <= 0 or cpus <= 0):
                            raise DisclosureError(
                                "successful accounting task has zero elapsed time or CPUs"
                            )
                        restart_value, restart_field = _field_with_name(
                            row, "Restarts", "RestartCnt"
                        )
                        roots[key] = {
                            "job_id": job_id,
                            "task_id": task_id,
                            "elapsed": elapsed,
                            "alloc_cpus": cpus,
                            "state": state,
                            "exit_code": exit_code,
                            "max_rss": _parse_maxrss(_field(row, "maxrss")),
                            "energy": _parse_positive_int(_field(row, "consumedenergyraw")),
                            "start": _parse_timestamp(_field(row, "start")),
                            "end": _parse_timestamp(_field(row, "end")),
                            "restart_count": _parse_nonnegative_int(
                                restart_value
                            ),
                            "restart_source_field": restart_field,
                        }
                    elif step_match:
                        job_id, task_text, _ = step_match.groups()
                        if job_id not in allowed_jobs:
                            raise DisclosureError(f"accounting contains undeclared job {job_id}")
                        task_id = int(task_text)
                        if task_id >= expected_tasks:
                            raise DisclosureError("accounting step task is out of range")
                        steps[(job_id, task_id)].append(row)
                    elif job_raw and job_raw not in allowed_jobs:
                        raise DisclosureError(f"unsupported accounting JobIDRaw: {job_raw}")
        except (OSError, UnicodeError, csv.Error) as error:
            raise DisclosureError(f"cannot parse Slurm accounting export: {error}") from error
        if _stable_file_measurement(path, "Slurm accounting export") != before:
            raise DisclosureError("Slurm accounting export changed while it was parsed")
    logical = {task_id for _, task_id in roots}
    jobs_with_roots = {job_id for job_id, _ in roots}
    if jobs_with_roots != allowed_jobs:
        raise DisclosureError(
            "at least one declared Slurm job has no root task accounting record"
        )
    if logical != set(range(expected_tasks)):
        raise DisclosureError(
            f"Slurm accounting is partial: expected {expected_tasks} logical tasks, found {len(logical)}"
        )
    orphan_steps = set(steps).difference(roots)
    if orphan_steps:
        raise DisclosureError(
            "Slurm accounting contains step records without matching root task records"
        )
    if log_attempts and set(roots) != log_attempts:
        raise DisclosureError("Slurm accounting attempts disagree with stdout/stderr attempts")
    records = [roots[key] for key in sorted(roots, key=lambda item: (int(item[0]), item[1]))]
    for record in records:
        key = (record["job_id"], record["task_id"])
        rss_values = [record["max_rss"]]
        rss_values.extend(_parse_maxrss(_field(row, "maxrss")) for row in steps.get(key, []))
        available_rss = [value for value in rss_values if value is not None]
        record["max_rss"] = max(available_rss) if available_rss else None
    success = sum(
        record["state"] == "COMPLETED" and record["exit_code"] == "0:0"
        for record in records
    )
    elapsed = [record["elapsed"] for record in records]
    slurm_cpu_seconds = sum(
        record["elapsed"] * record["alloc_cpus"] for record in records
    )
    rss = [record["max_rss"] for record in records if record["max_rss"] is not None]
    energy = [record["energy"] for record in records if record["energy"] is not None]
    restart_counts = [
        record["restart_count"]
        for record in records
        if record["restart_count"] is not None
    ]
    restart_source_fields = sorted(
        {
            record["restart_source_field"]
            for record in records
            if record["restart_count"] is not None
            and record["restart_source_field"] is not None
        }
    )
    concurrency = _observed_concurrency(records)
    rss_metric: dict[str, Any]
    if len(rss) == len(records):
        rss_metric = {
            "value": max(rss),
            "observed_distinct_job_task_records": len(rss),
            "aggregation": "maximum over root and step MaxRSS, then maximum over task attempts",
        }
    else:
        rss_metric = _null_metric(
            f"MaxRSS unavailable for {len(records) - len(rss)} of {len(records)} "
            "distinct job/task records"
        )
        rss_metric["observed_distinct_job_task_records"] = len(rss)
    energy_metric: dict[str, Any]
    if len(energy) == len(records):
        energy_metric = {
            "value": sum(energy),
            "observed_distinct_job_task_records": len(energy),
            "aggregation": "sum of positive ConsumedEnergyRaw values on root task records",
        }
    else:
        energy_metric = _null_metric(
            f"ConsumedEnergyRaw unavailable or zero for "
            f"{len(records) - len(energy)} of {len(records)} distinct job/task records"
        )
        energy_metric["observed_distinct_job_task_records"] = len(energy)
    return (
        {
            "available": True,
            "source_files": sorted(source_inventory, key=lambda item: item["name"]),
            "job_ids": list(job_ids),
            "job_id_scope": (
                "only explicitly supplied job ids; records from undisclosed job ids "
                "cannot be counted"
            ),
            "distinct_job_task_records": len(records),
            "logical_tasks_observed": len(logical),
            "successful_distinct_job_task_records": success,
            "failed_or_incomplete_distinct_job_task_records": len(records) - success,
            "distinct_job_resubmission_records": len(records) - len(logical),
            "same_job_requeue_or_restart_count": (
                {
                    "value": sum(restart_counts),
                    "observed_task_records": len(restart_counts),
                    "source_fields": restart_source_fields,
                }
                if len(restart_counts) == len(records)
                else {
                    **_null_metric(
                        "Slurm restart-count field unavailable for at least one root task record"
                    ),
                    "observed_task_records": len(restart_counts),
                }
            ),
            "elapsed_seconds": _numeric_summary(elapsed),
            "allocated_slurm_cpus_per_distinct_job_task_record": _numeric_summary(
                [record["alloc_cpus"] for record in records]
            ),
            "allocated_slurm_cpu_seconds": {
                "value": slurm_cpu_seconds,
                "method": (
                    "sum(ElapsedRaw * AllocCPUS) across distinct root job/task records; "
                    "Slurm CPUs are not asserted to be physical cores"
                ),
            },
            "maximum_observed_concurrent_tasks": (
                {"value": concurrency[0], "method": "half-open intervals from Slurm Start/End"}
                if concurrency is not None
                else _null_metric("Start/End unavailable or invalid for at least one task attempt")
            ),
            "maximum_observed_concurrent_slurm_cpus": (
                {
                    "value": concurrency[1],
                    "method": (
                        "half-open intervals weighted by AllocCPUS; Slurm CPUs are not "
                        "asserted to be physical cores"
                    ),
                }
                if concurrency is not None
                else _null_metric("Start/End unavailable or invalid for at least one task attempt")
            ),
            "max_rss_bytes": rss_metric,
            "consumed_energy_joules": energy_metric,
            "co2e_kg": _null_metric(
                "not_measured_and_no_energy_to_co2e_conversion_method_is_declared"
            ),
            "elapsed_scope": "Slurm task elapsed includes launcher setup and application",
        },
        set(roots),
    )


def _validate_final_filesystem_cardinality(
    artifact_root: str, expected: Mapping[str, int]
) -> None:
    directory_specs = {
        "training_runs": ("training", expected["training_runs"]),
        "checkpoint_artifacts": ("checkpoint", expected["checkpoints"]),
    }
    for relative, (prefix, count) in directory_specs.items():
        path = _inside(artifact_root, relative, relative)
        _require_directory(path, relative)
        expected_names = {f"{prefix}_{index:06d}" for index in range(count)}
        if set(os.listdir(path)) != expected_names:
            raise DisclosureError(f"{relative} filesystem cardinality is partial or has extras")
        for name in expected_names:
            _require_directory(os.path.join(path, name), f"{relative}/{name}")
    stderr_specs = {
        "stderr/training": ("training", expected["training_runs"]),
        "stderr/diagnostic": ("checkpoint", expected["checkpoints"]),
    }
    for relative, (prefix, count) in stderr_specs.items():
        path = _inside(artifact_root, relative, relative)
        _require_directory(path, relative)
        names = {f"{prefix}_{index:06d}.stderr" for index in range(count)}
        if set(os.listdir(path)) != names:
            raise DisclosureError(f"{relative} inventory is partial or has extras")
        for name in names:
            file_path = _require_regular(os.path.join(path, name), f"{relative}/{name}")
            if os.path.getsize(file_path) != 0:
                raise DisclosureError(f"validated application stderr became nonempty: {relative}/{name}")


def _storage_category(relative: str) -> str:
    first = relative.split(os.sep, 1)[0]
    if first == "training_runs":
        return "training_runs"
    if first == "checkpoint_artifacts":
        return "diagnostic_artifacts"
    if relative.startswith(f"stderr{os.sep}training{os.sep}"):
        return "training_application_stderr"
    if relative.startswith(f"stderr{os.sep}diagnostic{os.sep}"):
        return "diagnostic_application_stderr"
    if first.startswith("source_snapshot_"):
        return "immutable_source_snapshot"
    return "other_metadata"


def _storage_inventory(artifact_root: str, *, exclude: str | None) -> dict[str, Any]:
    categories: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "logical_bytes": 0, "allocated_bytes": 0}
    )
    root_real = os.path.realpath(artifact_root)
    exclude_real = os.path.realpath(exclude) if exclude is not None else None
    for directory, names, files in os.walk(root_real, followlinks=False):
        for name in names:
            path = os.path.join(directory, name)
            if os.path.islink(path):
                raise DisclosureError(f"artifact storage contains a symlink: {path}")
        for name in files:
            path = os.path.join(directory, name)
            if exclude_real is not None and os.path.realpath(path) == exclude_real:
                continue
            info = os.stat(path, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise DisclosureError(f"artifact storage contains a nonregular file: {path}")
            relative = os.path.relpath(path, root_real)
            category = categories[_storage_category(relative)]
            category["files"] += 1
            category["logical_bytes"] += info.st_size
            category["allocated_bytes"] += getattr(info, "st_blocks", 0) * 512
    ordered = {key: categories[key] for key in sorted(categories)}
    return {
        "categories": ordered,
        "total_files": sum(item["files"] for item in ordered.values()),
        "total_logical_bytes": sum(item["logical_bytes"] for item in ordered.values()),
        "total_allocated_bytes": sum(item["allocated_bytes"] for item in ordered.values()),
        "logical_size_method": "sum of regular-file st_size",
        "allocated_size_method": "sum of st_blocks * 512 where supplied by the filesystem",
        "disclosure_output_excluded": True,
    }


def _attempt_summary(
    logs: Mapping[str, Any],
    accounting: Mapping[str, Any],
    *,
    included: bool,
    upstream_evidence: str | None,
) -> dict[str, Any]:
    return {
        "upstream_validated_scientific_completion": (
            {"value": True, "evidence": upstream_evidence}
            if included
            else _null_metric("stage_not_included_in_disclosure_mode")
        ),
        "upstream_validated_included_stage_failures": (
            _null_metric(
                "scientific artifact validation does not enumerate scheduler attempts"
            )
            if included
            else _null_metric("stage_not_included_in_disclosure_mode")
        ),
        "slurm_log_distinct_job_resubmission_records": logs.get(
            "distinct_job_resubmission_records"
        ),
        "slurm_log_same_job_requeue_or_restart_count": logs.get(
            "same_job_requeue_or_restart_count"
        ),
        "slurm_nonempty_outer_stderr_attempts": logs.get("nonempty_stderr_attempts"),
        "accounted_distinct_job_resubmission_records": accounting.get(
            "distinct_job_resubmission_records"
        ),
        "accounted_same_job_requeue_or_restart_count": accounting.get(
            "same_job_requeue_or_restart_count"
        ),
        "accounted_failed_or_incomplete_distinct_job_task_records": accounting.get(
            "failed_or_incomplete_distinct_job_task_records"
        ),
        "scheduler_failure_claim_when_accounting_unavailable": None,
        "scheduler_failure_claim_reason": (
            None
            if accounting.get("available") is True
            else "scheduler failures are not inferred from validated scientific completion"
        ),
    }


def collect_disclosure(
    *,
    artifact_root: str,
    manifest_path: str,
    checkpoint_stage_report_path: str,
    expected_checkpoint_stage_sha256: str,
    checkpoint_stage_validator_path: str,
    audit_index_path: str | None,
    expected_audit_sha256: str | None,
    mode: str,
    slurm_log_root: str | None = None,
    checkpoint_job_ids: Sequence[str] = (),
    diagnostic_job_ids: Sequence[str] = (),
    checkpoint_accounting_paths: Sequence[str] = (),
    diagnostic_accounting_paths: Sequence[str] = (),
    output_path: str | None = None,
    require_production: bool = True,
) -> dict[str, Any]:
    """Build the deterministic disclosure without writing it."""

    if mode not in {"checkpoint-stage", "final"}:
        raise DisclosureError("mode must be 'checkpoint-stage' or 'final'")
    if not _is_sha256(expected_checkpoint_stage_sha256):
        raise DisclosureError(
            "all modes require a caller-supplied checkpoint-stage SHA-256"
        )
    final_mode = mode == "final"
    if final_mode != (audit_index_path is not None):
        raise DisclosureError("final mode requires one audit index; stage mode forbids it")
    if final_mode and not _is_sha256(expected_audit_sha256):
        raise DisclosureError(
            "final mode requires a caller-supplied expected audit SHA-256"
        )
    if not final_mode and expected_audit_sha256 is not None:
        raise DisclosureError("checkpoint-stage mode forbids an expected audit digest")
    artifact_root = os.path.abspath(artifact_root)
    _require_directory(artifact_root, "artifact root")
    stage, stage_measurement = _validate_stage_report(
        checkpoint_stage_report_path,
        expected_sha256=expected_checkpoint_stage_sha256,
    )
    snapshot, snapshot_relative = _resolve_snapshot(artifact_root, stage)
    _validate_snapshot_source(snapshot, stage)
    validator_measurement = _validate_checkpoint_stage_validator(
        checkpoint_stage_validator_path, stage
    )
    snapshot_manifest = _inside(snapshot, MANIFEST_RELATIVE, "snapshot manifest")
    _require_regular(manifest_path, "manifest input")
    if os.path.abspath(manifest_path) != os.path.abspath(snapshot_manifest):
        raise DisclosureError("manifest input must be the locked snapshot manifest")
    manifest = _validate_manifest(
        snapshot_manifest,
        expected_sha256=stage["manifest_sha256"],
        require_production=require_production,
    )
    expected = _expected_cardinalities(manifest)
    _validate_stage_cardinality(stage, expected)
    launcher_files = _validate_bundle(
        snapshot,
        LAUNCHER_BUNDLE,
        expected_sha256=stage["provenance"]["launcher_sha256"],
        expected_kind="launchers",
    )
    dependency_files = _validate_bundle(
        snapshot,
        DEPENDENCY_BUNDLE,
        expected_sha256=stage["provenance"]["dependency_lock_sha256"],
        expected_kind="dependency_locks",
    )
    if set(launcher_files) != set(LAUNCHERS.values()):
        raise DisclosureError("launcher bundle does not contain exactly both study launchers")
    if set(dependency_files) != {"environment.yml", "requirement.txt"}:
        raise DisclosureError(
            "dependency bundle does not contain exactly both locked environment files"
        )
    _validate_launcher_runtime_gates(snapshot)
    requested = {
        "checkpoint_generation": _parse_launcher(
            _inside(snapshot, LAUNCHERS["checkpoint_generation"], "checkpoint launcher"),
            expected_tasks=expected["training_runs"],
        ),
        "diagnostic": _parse_launcher(
            _inside(snapshot, LAUNCHERS["diagnostic"], "diagnostic launcher"),
            expected_tasks=expected["checkpoints"],
        ),
    }
    audit_metadata: dict[str, Any] | None = None
    observed_cardinality: dict[str, int] = {
        "training_runs": expected["training_runs"],
        "checkpoints": expected["checkpoints"],
    }
    audit_sha: str | None = None
    if final_mode:
        assert audit_index_path is not None
        audit_sha = _sha256_file(_require_regular(audit_index_path, "final audit index"))
        if audit_sha != expected_audit_sha256:
            raise DisclosureError(
                "final audit index differs from the caller-supplied committed digest"
            )
        audit_metadata, observed_cardinality = _scan_audit(audit_index_path)
        _validate_audit_metadata(
            audit_metadata,
            observed_cardinality,
            stage=stage,
            manifest=manifest,
            expected=expected,
            final_mode=True,
        )
        if _sha256_file(audit_index_path) != audit_sha:
            raise DisclosureError("final audit index changed while it was scanned")
        _validate_final_filesystem_cardinality(artifact_root, expected)
    workload = _validate_budget(
        manifest=manifest,
        stage_budget=stage["budget"],
        audit_budget=None if audit_metadata is None else audit_metadata["budget"],
    )
    training_task_metadata = [
        {
            "task_id": str(record["training_id"]),
            "training_id": str(record["training_id"]),
            "task_index": str(record["task_index"]),
            "env": str(record["env_name"]),
            "seed": str(record["training_seed"]),
        }
        for record in stage["training_runs"]
    ]
    diagnostic_task_metadata = [
        {
            "task_id": str(record["checkpoint_id"]),
            "checkpoint_id": str(record["checkpoint_id"]),
            "training_id": str(record["training_id"]),
            "task_index": str(record["task_index"]),
            "env": str(record["env_name"]),
            "seed": str(record["training_seed"]),
            "generation": str(record["generation"]),
        }
        for record in stage["checkpoints"]
    ]
    training_logs, training_log_attempts = _collect_slurm_logs(
        slurm_log_root,
        stage="checkpoint_generation",
        prefix="lagged_checkpoint",
        job_ids=checkpoint_job_ids,
        expected_tasks=expected["training_runs"],
        expected_source_sha256=stage["provenance"]["source_sha256"],
        expected_snapshot=snapshot,
        expected_workers=requested["checkpoint_generation"][
            "application_workers_per_task"
        ],
        expected_task_metadata=training_task_metadata,
    )
    training_accounting, _ = _collect_accounting(
        checkpoint_accounting_paths,
        job_ids=checkpoint_job_ids,
        expected_tasks=expected["training_runs"],
        log_attempts=training_log_attempts,
    )
    if final_mode:
        diagnostic_logs, diagnostic_log_attempts = _collect_slurm_logs(
            slurm_log_root,
            stage="diagnostic",
            prefix="lagged_diagnostic",
            job_ids=diagnostic_job_ids,
            expected_tasks=expected["checkpoints"],
            expected_source_sha256=stage["provenance"]["source_sha256"],
            expected_snapshot=snapshot,
            expected_workers=requested["diagnostic"][
                "application_workers_per_task"
            ],
            expected_task_metadata=diagnostic_task_metadata,
        )
        diagnostic_accounting, _ = _collect_accounting(
            diagnostic_accounting_paths,
            job_ids=diagnostic_job_ids,
            expected_tasks=expected["checkpoints"],
            log_attempts=diagnostic_log_attempts,
        )
    else:
        diagnostic_logs = _log_unavailable("diagnostic_stage_not_included")
        diagnostic_accounting = _accounting_unavailable("diagnostic_stage_not_included")
    application_runtime = {
        "checkpoint_generation": _training_application_runtime(
            artifact_root, stage["training_runs"]
        ),
        "diagnostic": {
            "instrumented_application_seconds": _null_metric(
                "diagnostic application did not persist an internal wallclock field"
            ),
            "slurm_task_elapsed_seconds": (
                diagnostic_accounting["elapsed_seconds"]
                if diagnostic_accounting.get("available") is True
                else _null_metric(diagnostic_accounting["reason"])
            ),
            "slurm_elapsed_scope": (
                "launcher setup plus diagnostic application"
                if diagnostic_accounting.get("available") is True
                else None
            ),
        },
    }
    infrastructure_failures = (
        []
        if audit_metadata is None
        else audit_metadata["provenance"]["documented_infrastructure_failures"]
    )
    output_real = os.path.abspath(output_path) if output_path is not None else None
    report = _stamp_report(
        {
            "schema_version": SCHEMA_VERSION,
            "study": STUDY,
            "disclosure": "compute_and_storage",
            "mode": mode,
            "outcome_access": {
                "final_audit_scientific_array_elements_deserialized": False,
                "audit_method": (
                    "top-level selective lexical scan; scientific arrays are counted "
                    "without deserializing their elements"
                ),
                "training_runtime_method": (
                    "select only iteration and iteration_compute_seconds from each "
                    "hash-validated JSONL row"
                ),
            },
            "provenance": {
                "manifest_sha256": stage["manifest_sha256"],
                "source_sha256": stage["provenance"]["source_sha256"],
                "source_snapshot_path": snapshot_relative,
                "checkpoint_stage_report": {
                    "sha256": stage_measurement["sha256"],
                    "expected_sha256": expected_checkpoint_stage_sha256,
                    "digest_match": True,
                    "report_sha256": stage["report_sha256"],
                },
                "checkpoint_stage_validator": {
                    **validator_measurement,
                    "expected_sha256": stage["provenance"]["validator_sha256"],
                    "digest_match": True,
                },
                "final_audit_index": (
                    {
                        "sha256": audit_sha,
                        "expected_sha256": expected_audit_sha256,
                        "digest_match": True,
                    }
                    if audit_sha is not None
                    else _null_metric("not_in_checkpoint_stage_mode")
                ),
                "launcher_bundle_sha256": stage["provenance"]["launcher_sha256"],
                "dependency_lock_sha256": stage["provenance"]["dependency_lock_sha256"],
                "collector_sha256": _sha256_file(__file__),
            },
            "scientific_cardinality": {
                "expected": dict(expected),
                "lexically_observed_array_element_counts": observed_cardinality,
                "final_lexical_counts_match_manifest": final_mode,
                "scientific_record_contents_validated_by_collector": False,
                "scientific_record_validation_boundary": (
                    "upstream validated audit input, bound here by the caller-supplied "
                    "expected audit digest"
                    if final_mode
                    else "checkpoint-stage validation report only"
                ),
            },
            "scientific_workload": workload,
            "locked_runtime": _locked_runtime_versions(snapshot, stage),
            "requested_allocations": requested,
            "observed_execution": {
                "checkpoint_generation": {
                    "application_runtime": application_runtime["checkpoint_generation"],
                    "slurm_logs": training_logs,
                    "slurm_accounting": training_accounting,
                },
                "diagnostic": {
                    "application_runtime": application_runtime["diagnostic"],
                    "slurm_logs": diagnostic_logs,
                    "slurm_accounting": diagnostic_accounting,
                },
            },
            "failures_and_retries": {
                "checkpoint_generation": _attempt_summary(
                    training_logs,
                    training_accounting,
                    included=True,
                    upstream_evidence="checkpoint-stage validation report",
                ),
                "diagnostic": _attempt_summary(
                    diagnostic_logs,
                    diagnostic_accounting,
                    included=final_mode,
                    upstream_evidence=(
                        "caller-attested final audit digest"
                        if final_mode
                        else None
                    ),
                ),
                "documented_infrastructure_failures": infrastructure_failures,
                "documented_infrastructure_failure_count": len(
                    infrastructure_failures
                ),
            },
            "storage": {
                "artifact_root": _storage_inventory(
                    artifact_root, exclude=output_real
                ),
                "slurm_log_files": {
                    "checkpoint_generation": (
                        {
                            "logical_bytes": training_logs["stdout"]["bytes"]
                            + training_logs["stderr"]["bytes"],
                            "files": training_logs["stdout"]["files"]
                            + training_logs["stderr"]["files"],
                        }
                        if training_logs.get("available") is True
                        else _null_metric(training_logs["reason"])
                    ),
                    "diagnostic": (
                        {
                            "logical_bytes": diagnostic_logs["stdout"]["bytes"]
                            + diagnostic_logs["stderr"]["bytes"],
                            "files": diagnostic_logs["stdout"]["files"]
                            + diagnostic_logs["stderr"]["files"],
                        }
                        if diagnostic_logs.get("available") is True
                        else _null_metric(diagnostic_logs["reason"])
                    ),
                },
            },
            "telemetry_limitations": {
                "max_rss": (
                    {"value": "complete_for_all_included_stages", "reason": None}
                    if training_accounting.get("max_rss_bytes", {}).get("value") is not None
                    and (
                        not final_mode
                        or diagnostic_accounting.get("max_rss_bytes", {}).get("value") is not None
                    )
                    else _null_metric(
                        "at least one included stage lacks complete MaxRSS telemetry"
                    )
                ),
                "energy": (
                    {"value": "complete_for_all_included_stages", "reason": None}
                    if training_accounting.get("consumed_energy_joules", {}).get("value") is not None
                    and (
                        not final_mode
                        or diagnostic_accounting.get("consumed_energy_joules", {}).get("value") is not None
                    )
                    else _null_metric(
                        "at least one included stage lacks complete positive "
                        "ConsumedEnergyRaw telemetry"
                    )
                ),
                "co2e_kg": _null_metric(
                    "not_measured_and_no_energy_to_co2e_conversion_method_is_declared"
                ),
            },
        }
    )
    if _stable_file_measurement(
        checkpoint_stage_report_path, "checkpoint-stage report"
    ) != stage_measurement:
        raise DisclosureError("checkpoint-stage report changed during collection")
    if _stable_file_measurement(
        checkpoint_stage_validator_path, "checkpoint-stage validator"
    ) != validator_measurement:
        raise DisclosureError("checkpoint-stage validator changed during collection")
    try:
        final_source_sha = compute_lagged_subspace_study_sha256(snapshot)
    except StudySourceLockError as error:
        raise DisclosureError(f"cannot revalidate immutable source snapshot: {error}") from error
    if final_source_sha != stage["provenance"]["source_sha256"]:
        raise DisclosureError("immutable source snapshot changed during collection")
    if audit_index_path is not None and _stable_file_measurement(
        audit_index_path, "final audit index"
    )["sha256"] != audit_sha:
        raise DisclosureError("final audit index changed during collection")
    return report


def _atomic_json_write_once(path: str, value: Any) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    if os.path.lexists(path):
        raise DisclosureError("output already exists; compute disclosure refuses overwrite")
    descriptor, staged = tempfile.mkstemp(prefix=".compute_disclosure_", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_bytes(value))
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except OSError as error:
            if error.errno == errno.EEXIST:
                raise DisclosureError(
                    "output already exists; compute disclosure refuses overwrite"
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
    parser.add_argument("--checkpoint-stage-report", required=True)
    parser.add_argument(
        "--expected-checkpoint-stage-sha256",
        required=True,
        help="digest committed by the checkpoint-stage validation workflow",
    )
    parser.add_argument(
        "--checkpoint-stage-validator",
        required=True,
        help="validator source whose digest is committed in the stage report",
    )
    parser.add_argument("--audit-index")
    parser.add_argument(
        "--expected-audit-sha256",
        help="digest committed by the upstream audit-validation workflow",
    )
    parser.add_argument(
        "--mode", choices=("checkpoint-stage", "final"), default="final"
    )
    parser.add_argument("--slurm-log-root")
    parser.add_argument("--checkpoint-job-id", action="append", default=[])
    parser.add_argument("--diagnostic-job-id", action="append", default=[])
    parser.add_argument("--checkpoint-accounting", action="append", default=[])
    parser.add_argument("--diagnostic-accounting", action="append", default=[])
    parser.add_argument(
        "--output", required=True, help="artifact-root-relative disclosure JSON path"
    )
    parser.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact_root = os.path.abspath(args.artifact_root)
    output_path = _inside(artifact_root, args.output, "output")
    if os.path.lexists(output_path):
        raise DisclosureError("output already exists; compute disclosure refuses overwrite")
    report = collect_disclosure(
        artifact_root=artifact_root,
        manifest_path=args.manifest,
        checkpoint_stage_report_path=args.checkpoint_stage_report,
        expected_checkpoint_stage_sha256=args.expected_checkpoint_stage_sha256,
        checkpoint_stage_validator_path=args.checkpoint_stage_validator,
        audit_index_path=args.audit_index,
        expected_audit_sha256=args.expected_audit_sha256,
        mode=args.mode,
        slurm_log_root=args.slurm_log_root,
        checkpoint_job_ids=args.checkpoint_job_id,
        diagnostic_job_ids=args.diagnostic_job_id,
        checkpoint_accounting_paths=args.checkpoint_accounting,
        diagnostic_accounting_paths=args.diagnostic_accounting,
        output_path=output_path,
        require_production=not args.fixture_mode,
    )
    _atomic_json_write_once(output_path, report)
    readback = _read_json(output_path, "compute disclosure readback")
    if readback != report or readback.get("report_sha256") != _report_sha256(readback):
        raise DisclosureError("compute disclosure changed during atomic readback")
    print(f"Wrote compute/storage disclosure to {output_path}")


if __name__ == "__main__":
    main()

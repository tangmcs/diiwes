"""Focused tests for deterministic lagged-subspace paper outputs."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import render_lagged_subspace_paper_outputs as renderer


def _sign_p(wins: int, trials: int = 20) -> float:
    return renderer._binomial_upper_tail(wins, trials)


def _analysis_fixture() -> dict[str, object]:
    task_results = []
    raw = [_sign_p(20), _sign_p(20), _sign_p(20)]
    adjusted = renderer._holm_adjust(raw)
    metric_values = {"L": 0.5, "D": 0.02, "H": 0.1, "E": 0.2}
    for task_position, (task_index, env_name) in enumerate(renderer.TASKS):
        seed_statistics = {
            metric: [value] * 20 for metric, value in metric_values.items()
        }
        simultaneous_bounds = {
            metric: {
                "estimate": value,
                "one_sided_bound": value,
                "resolved": True,
                "order_index_zero_based": 3 if metric == "D" else 16,
            }
            for metric, value in metric_values.items()
        }
        gates = {key: True for key in renderer.GATE_KEYS}
        task_results.append(
            {
                "task_index": task_index,
                "env_name": env_name,
                "seed_mean_contrast": float(task_position + 1),
                "strict_positive_seed_count": 20,
                "strict_tie_seed_count": 0,
                "seed_count": 20,
                "seed_level_probability_of_improvement": 1.0,
                "raw_one_sided_sign_p": raw[task_position],
                "holm_adjusted_one_sided_sign_p": adjusted[task_position],
                "seed_statistics": seed_statistics,
                "simultaneous_bounds": simultaneous_bounds,
                "gate_conditions": gates,
                "task_pass": True,
            }
        )

    locality = []
    for task_index, env_name in renderer.TASKS:
        for q in renderer.Q_VALUES:
            for arm_index, arm in enumerate(renderer.ARMS):
                mean = q + 0.1 * arm_index + 0.01 * task_index
                locality.append(
                    {
                        "task_index": task_index,
                        "env_name": env_name,
                        "q": q,
                        "arm": arm,
                        "repeated_measure_count": 1200,
                        "first_step_over_sigma": mean,
                        "mean_step_over_sigma": mean,
                        "median_step_over_sigma": mean,
                        "percentile_95_step_over_sigma": mean + 0.2,
                        "maximum_step_over_sigma": mean + 0.3,
                        "fraction_at_or_below_0_25": 0.1,
                        "fraction_at_or_below_0_5": 0.2,
                        "fraction_at_or_below_1_0": 0.9,
                        "inference": "descriptive_repeated_measures_only",
                    }
                )

    return_contrasts = []
    for task_index, env_name in renderer.TASKS:
        for q in renderer.Q_VALUES:
            for control_index, control in enumerate(renderer.CONTROLS):
                mean = (
                    float(task_index + 1)
                    if q == renderer.PRIMARY_Q and control == "isotropic"
                    else float(task_index + 1) + q - 0.2 * control_index
                )
                role = (
                    "primary_holm_family"
                    if q == renderer.PRIMARY_Q and control == "isotropic"
                    else (
                        "secondary_no_p_value_reported"
                        if q == renderer.PRIMARY_Q
                        else "descriptive_sensitivity_no_p_value"
                    )
                )
                return_contrasts.append(
                    {
                        "task_index": task_index,
                        "env_name": env_name,
                        "q": q,
                        "contrast": f"structured_minus_{control}",
                        "paired_difference_count": 12000,
                        "training_seed_cluster_count": 20,
                        "paired_mean": mean,
                        "paired_median": mean - 0.05,
                        "paired_interquartile_mean": mean - 0.02,
                        "paired_checkpoint_partition_episode_probability_of_improvement": 0.6,
                        "seed_cluster_bootstrap_mean_interval_95": [
                            mean - 0.25,
                            mean + 0.25,
                        ],
                        "bootstrap_seed": 424242,
                        "multiplicity_role": role,
                    }
                )
    return {
        "schema_version": 1,
        "study": renderer.STUDY,
        "analysis_designation": renderer.ANALYSIS_DESIGNATION,
        "primary_q": renderer.PRIMARY_Q,
        "mechanism_bound_method": renderer.MECHANISM_BOUND_METHOD,
        "mechanism_familywise_error_upper_bound": (
            renderer.MECHANISM_FAMILYWISE_ERROR_UPPER_BOUND
        ),
        "endpoint_family_alpha": renderer.ENDPOINT_FAMILY_ALPHA,
        "combined_false_advance_upper_bound": (
            renderer.COMBINED_FALSE_ADVANCE_UPPER_BOUND
        ),
        "top_level_unit": "training_seed",
        "task_results": task_results,
        "descriptive_locality": locality,
        "descriptive_return_contrasts": return_contrasts,
        "passing_task_count": 3,
        "required_passing_task_count": renderer.REQUIRED_PASSING_TASK_COUNT,
        "mechanism_advances_to_optimizer_pilot": True,
        "claim_boundary": renderer.CLAIM_BOUNDARY,
    }


def _write_analysis(root: str, value: dict[str, object]) -> str:
    path = os.path.join(root, "analysis.json")
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def _input_sha(path: str) -> str:
    return renderer._sha256_file(path)


def _render(analysis_path: str, output_path: str) -> dict[str, object]:
    return renderer.render_paper_outputs(
        analysis_path,
        output_path,
        expected_analysis_sha256=_input_sha(analysis_path),
    )


def _refresh_decision(analysis: dict[str, object]) -> None:
    task_results = analysis["task_results"]  # type: ignore[assignment]
    passing = sum(bool(task["task_pass"]) for task in task_results)
    analysis["passing_task_count"] = passing
    analysis["mechanism_advances_to_optimizer_pilot"] = passing >= 2


def _set_metric(
    analysis: dict[str, object], task_index: int, metric: str, value: float
) -> None:
    gate_key = {
        "L": "locality",
        "D": "material_action",
        "H": "high_sample_replication",
        "E": "operational_reliability",
    }[metric]
    task = analysis["task_results"][task_index]  # type: ignore[index]
    task["seed_statistics"][metric] = [value] * 20
    task["simultaneous_bounds"][metric].update(
        {"estimate": value, "one_sided_bound": value}
    )
    threshold = renderer.GATE_THRESHOLDS[metric]
    gate_value = value > threshold if metric == "D" else (
        value <= threshold if metric == "L" else value < threshold
    )
    task["gate_conditions"][gate_key] = gate_value
    task["task_pass"] = all(task["gate_conditions"].values())
    _refresh_decision(analysis)


class PaperOutputRendererTests(unittest.TestCase):
    def test_complete_outputs_are_byte_deterministic_and_provenanced(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            analysis = _analysis_fixture()
            first_input = _write_analysis(first, analysis)
            second_input = _write_analysis(second, analysis)
            first_output = os.path.join(first, "paper_outputs")
            second_output = os.path.join(second, "paper_outputs")
            first_manifest = _render(first_input, first_output)
            second_manifest = _render(second_input, second_output)

            expected_names = sorted(renderer.OUTPUT_SPECS) + [
                "paper_output_manifest.json"
            ]
            self.assertEqual(sorted(os.listdir(first_output)), sorted(expected_names))
            self.assertEqual(first_manifest, second_manifest)
            for name in expected_names:
                with open(os.path.join(first_output, name), "rb") as stream:
                    first_bytes = stream.read()
                with open(os.path.join(second_output, name), "rb") as stream:
                    second_bytes = stream.read()
                self.assertEqual(first_bytes, second_bytes, name)

            self.assertEqual(len(first_manifest["outputs"]), 10)
            self.assertEqual(
                first_manifest["validated_protocol_invariants"][
                    "descriptive_bootstrap"
                ],
                "one_shared_seed_required_for_one_common_resample_index_matrix",
            )
            self.assertEqual(
                [entry["output_path"] for entry in first_manifest["outputs"]],
                [f"paper_outputs/{name}" for name in sorted(renderer.OUTPUT_SPECS)],
            )
            with open(first_input, "rb") as stream:
                input_sha = hashlib.sha256(stream.read()).hexdigest()
            generator_sha = renderer._sha256_file(renderer.__file__)
            for entry in first_manifest["outputs"]:
                output_name = entry["output_path"].split("/", 1)[1]
                self.assertEqual(
                    entry["output_sha256"],
                    renderer._sha256_file(os.path.join(first_output, output_name)),
                )
                self.assertEqual(entry["input_sha256"], input_sha)
                self.assertEqual(entry["generator_sha256"], generator_sha)
                self.assertEqual(entry["claim_boundary"], renderer.CLAIM_BOUNDARY)
                self.assertEqual(
                    entry["input_sha256_expectation"], "explicit_required_argument"
                )
                self.assertEqual(
                    entry["command_working_directory"], "artifact_root"
                )
                self.assertIn("REPOSITORY_ROOT", entry["command"])
                self.assertIn(input_sha, entry["command"])
                self.assertEqual(
                    entry["runtime_provenance"]["dejavu_sans_font_sha256"],
                    renderer._runtime_provenance()["dejavu_sans_font_sha256"],
                )
                self.assertTrue(entry["json_selectors"])
                self.assertNotIn(first, entry["command"])

            for name in (
                "figure_mechanism_bounds.pdf",
                "figure_endpoint_contrasts.pdf",
                "figure_locality_sensitivity.pdf",
            ):
                with open(os.path.join(first_output, name), "rb") as stream:
                    payload = stream.read()
                self.assertTrue(payload.startswith(b"%PDF"))
                for forbidden in (
                    b"/CreationDate",
                    b"/ModDate",
                    b"/Author",
                    b"/Creator",
                    b"/Producer",
                ):
                    self.assertNotIn(forbidden, payload)
            for name in (
                "figure_mechanism_bounds.png",
                "figure_endpoint_contrasts.png",
                "figure_locality_sensitivity.png",
            ):
                with open(os.path.join(first_output, name), "rb") as stream:
                    payload = stream.read()
                self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
                for forbidden in (b"Creation Time", b"Author", b"Software"):
                    self.assertNotIn(forbidden, payload)

            with open(
                os.path.join(first_output, "table_mechanism_gates.csv"),
                newline="",
                encoding="utf-8",
            ) as stream:
                gate_rows = list(csv.DictReader(stream))
            self.assertEqual(len(gate_rows), 3)
            self.assertEqual(gate_rows[0]["D_one_sided_bound"], "0.02")
            self.assertEqual(gate_rows[0]["gate_directional_endpoint"], "true")

    def test_cli_is_byte_deterministic_across_fresh_processes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis = _analysis_fixture()
            roots = [os.path.join(root, "first"), os.path.join(root, "second")]
            for child in roots:
                os.makedirs(child)
                _write_analysis(child, analysis)
            script = os.path.abspath(renderer.__file__)
            for index, child in enumerate(roots):
                environment = dict(os.environ)
                environment["MPLCONFIGDIR"] = os.path.join(root, f"mpl-{index}")
                completed = subprocess.run(
                    [
                        sys.executable,
                        script,
                        os.path.join(child, "analysis.json"),
                        "--output-dir",
                        os.path.join(child, "paper_outputs"),
                        "--expected-analysis-sha256",
                        _input_sha(os.path.join(child, "analysis.json")),
                    ],
                    cwd=os.path.dirname(os.path.dirname(script)),
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("Rendered 10 deterministic outputs", completed.stdout)
            first_names = sorted(os.listdir(os.path.join(roots[0], "paper_outputs")))
            second_names = sorted(os.listdir(os.path.join(roots[1], "paper_outputs")))
            self.assertEqual(first_names, second_names)
            for name in first_names:
                with open(os.path.join(roots[0], "paper_outputs", name), "rb") as stream:
                    first_payload = stream.read()
                with open(os.path.join(roots[1], "paper_outputs", name), "rb") as stream:
                    second_payload = stream.read()
                self.assertEqual(first_payload, second_payload, name)

    def test_recorded_command_reproduces_from_declared_artifact_root(self) -> None:
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as target:
            analysis = _analysis_fixture()
            source_input = _write_analysis(source, analysis)
            manifest = _render(source_input, os.path.join(source, "paper_outputs"))
            _write_analysis(target, analysis)
            environment = dict(os.environ)
            environment.update(
                {
                    "PYTHON": sys.executable,
                    "REPOSITORY_ROOT": os.path.dirname(
                        os.path.dirname(os.path.abspath(renderer.__file__))
                    ),
                    "MPLCONFIGDIR": os.path.join(target, ".mplconfig"),
                }
            )
            completed = subprocess.run(
                manifest["outputs"][0]["command"],
                cwd=target,
                env=environment,
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Rendered 10 deterministic outputs", completed.stdout)
            self.assertEqual(
                sorted(os.listdir(os.path.join(target, "paper_outputs"))),
                sorted(
                    list(renderer.OUTPUT_SPECS) + ["paper_output_manifest.json"]
                ),
            )

    def test_explicit_analysis_hash_is_required_and_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis_path = _write_analysis(root, _analysis_fixture())
            output = os.path.join(root, "paper_outputs")
            with self.assertRaisesRegex(renderer.PaperOutputError, "SHA-256"):
                renderer.render_paper_outputs(
                    analysis_path,
                    output,
                    expected_analysis_sha256="0" * 64,
                )
            self.assertFalse(os.path.lexists(output))
            with self.assertRaises(TypeError):
                renderer.render_paper_outputs(analysis_path, output)  # type: ignore[call-arg]

    def test_single_open_binds_parsed_bytes_to_the_verified_hash(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis_path = _write_analysis(root, _analysis_fixture())
            with open(analysis_path, "rb") as stream:
                original_payload = stream.read()
            original_sha = hashlib.sha256(original_payload).hexdigest()
            replacement = os.path.join(root, "replacement.json")
            with open(replacement, "wb") as stream:
                stream.write(original_payload + b" ")
            replacement_sha = renderer._sha256_file(replacement)
            real_open = os.open
            swapped = False

            def open_then_replace(path: str, flags: int) -> int:
                nonlocal swapped
                descriptor = real_open(path, flags)
                if not swapped:
                    os.replace(replacement, analysis_path)
                    swapped = True
                return descriptor

            with mock.patch.object(renderer.os, "open", side_effect=open_then_replace):
                parsed, actual_sha = renderer._read_analysis(
                    analysis_path, original_sha
                )
            self.assertEqual(parsed["study"], renderer.STUDY)
            self.assertEqual(actual_sha, original_sha)
            self.assertEqual(renderer._sha256_file(analysis_path), replacement_sha)

    def test_all_bootstrap_cells_must_use_one_shared_seed(self) -> None:
        analysis = _analysis_fixture()
        analysis["descriptive_return_contrasts"][0]["bootstrap_seed"] = 7  # type: ignore[index]
        with self.assertRaisesRegex(renderer.PaperOutputError, "shared bootstrap seed"):
            renderer._validate_analysis(analysis)

    def test_exact_and_adjacent_gate_thresholds_use_recomputed_statistics(self) -> None:
        cases = (
            ("L", 1.0, True),
            ("L", math.nextafter(1.0, math.inf), False),
            ("D", 0.01, False),
            ("D", math.nextafter(0.01, math.inf), True),
            ("H", 0.25, False),
            ("H", math.nextafter(0.25, -math.inf), True),
            ("E", 0.5, False),
            ("E", math.nextafter(0.5, -math.inf), True),
        )
        gate_names = {
            "L": "locality",
            "D": "material_action",
            "H": "high_sample_replication",
            "E": "operational_reliability",
        }
        for metric, value, expected in cases:
            with self.subTest(metric=metric, value=value):
                analysis = _analysis_fixture()
                _set_metric(analysis, 0, metric, value)
                normalized = renderer._validate_analysis(analysis)
                self.assertEqual(
                    normalized["task_results"][0]["gate_conditions"][
                        gate_names[metric]
                    ],
                    expected,
                )

        analysis = _analysis_fixture()
        _set_metric(analysis, 0, "D", 0.01)
        task = analysis["task_results"][0]  # type: ignore[index]
        supplied = math.nextafter(0.01, math.inf)
        task["simultaneous_bounds"]["D"].update(
            {"estimate": supplied, "one_sided_bound": supplied}
        )
        normalized = renderer._validate_analysis(analysis)
        self.assertEqual(
            normalized["task_results"][0]["simultaneous_bounds"]["D"][
                "one_sided_bound"
            ],
            0.01,
        )
        task["gate_conditions"]["material_action"] = True
        task["task_pass"] = True
        _refresh_decision(analysis)
        with self.assertRaisesRegex(renderer.PaperOutputError, "material_action"):
            renderer._validate_analysis(analysis)

    def test_non_enclosing_bootstrap_interval_is_drawn_at_true_endpoints(self) -> None:
        analysis = _analysis_fixture()
        row = next(
            value
            for value in analysis["descriptive_return_contrasts"]  # type: ignore[union-attr]
            if value["task_index"] == 0
            and value["q"] == renderer.PRIMARY_Q
            and value["contrast"] == "structured_minus_explicit"
        )
        mean = row["paired_mean"]
        expected_interval = [mean + 1.0, mean + 2.0]
        row["seed_cluster_bootstrap_mean_interval_95"] = expected_interval
        normalized = renderer._validate_analysis(analysis)
        figure = renderer._endpoint_contrasts_figure(normalized)
        try:
            vertical_segments = []
            for collection in figure.axes[0].collections:
                if not hasattr(collection, "get_segments"):
                    continue
                for segment in collection.get_segments():
                    if len(segment) == 2 and segment[0][0] == segment[1][0] == 0.0:
                        vertical_segments.append(
                            sorted([float(segment[0][1]), float(segment[1][1])])
                        )
            self.assertIn(expected_interval, vertical_segments)
        finally:
            renderer.plt.close(figure)

    def test_refuses_overwrite_without_modifying_existing_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis_path = _write_analysis(root, _analysis_fixture())
            output = os.path.join(root, "paper_outputs")
            _render(analysis_path, output)
            before = {
                name: renderer._sha256_file(os.path.join(output, name))
                for name in os.listdir(output)
            }
            with self.assertRaisesRegex(renderer.PaperOutputError, "refusing overwrite"):
                _render(analysis_path, output)
            after = {
                name: renderer._sha256_file(os.path.join(output, name))
                for name in os.listdir(output)
            }
            self.assertEqual(before, after)

    def test_valid_negative_result_is_rendered_without_relabeling(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis = _analysis_fixture()
            for task_index in (0, 1):
                task = analysis["task_results"][task_index]  # type: ignore[index]
                task["seed_statistics"]["L"] = [2.0] * 20
                task["simultaneous_bounds"]["L"].update(
                    {"estimate": 2.0, "one_sided_bound": 2.0}
                )
                task["gate_conditions"]["locality"] = False
                task["task_pass"] = False
            analysis["passing_task_count"] = 1
            analysis["mechanism_advances_to_optimizer_pilot"] = False
            analysis_path = _write_analysis(root, analysis)
            manifest = _render(analysis_path, os.path.join(root, "paper_outputs"))
            self.assertEqual(len(manifest["outputs"]), 10)
            with open(
                os.path.join(root, "paper_outputs", "table_mechanism_gates.csv"),
                newline="",
                encoding="utf-8",
            ) as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["task_pass"] for row in rows], ["false", "false", "true"])

    def test_rejects_identity_claim_boundary_and_unsupported_schema(self) -> None:
        mutations = (
            ("study", "other_study"),
            ("analysis_designation", "exploratory"),
            ("claim_boundary", "optimizer_superiority"),
            ("schema_version", 2),
        )
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                analysis = _analysis_fixture()
                analysis[key] = value
                analysis_path = _write_analysis(root, analysis)
                output = os.path.join(root, "paper_outputs")
                with self.assertRaises(renderer.PaperOutputError):
                    _render(analysis_path, output)
                self.assertFalse(os.path.exists(output))

    def test_rejects_partial_and_internally_inconsistent_analyses(self) -> None:
        def remove_task(value: dict[str, object]) -> None:
            value["task_results"].pop()  # type: ignore[union-attr]

        def remove_locality(value: dict[str, object]) -> None:
            value["descriptive_locality"].pop()  # type: ignore[union-attr]

        def remove_contrast(value: dict[str, object]) -> None:
            value["descriptive_return_contrasts"].pop()  # type: ignore[union-attr]

        def corrupt_p(value: dict[str, object]) -> None:
            value["task_results"][0]["raw_one_sided_sign_p"] = 0.5  # type: ignore[index]

        def corrupt_gate(value: dict[str, object]) -> None:
            value["task_results"][0]["gate_conditions"]["locality"] = False  # type: ignore[index]

        for mutation in (
            remove_task,
            remove_locality,
            remove_contrast,
            corrupt_p,
            corrupt_gate,
        ):
            with self.subTest(mutation=mutation.__name__), tempfile.TemporaryDirectory() as root:
                analysis = _analysis_fixture()
                mutation(analysis)
                analysis_path = _write_analysis(root, analysis)
                with self.assertRaises(renderer.PaperOutputError):
                    _render(analysis_path, os.path.join(root, "paper_outputs"))

    def test_render_failure_leaves_no_partial_output_or_staging_tree(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis_path = _write_analysis(root, _analysis_fixture())
            output = os.path.join(root, "paper_outputs")
            with mock.patch.object(
                renderer, "_save_figure_pair", side_effect=RuntimeError("fixture failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    _render(analysis_path, output)
            self.assertFalse(os.path.lexists(output))
            self.assertFalse(
                any(name.startswith(".paper_outputs.staging.") for name in os.listdir(root))
            )

    def test_refuses_wrong_input_or_output_names_and_symlink_input(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            analysis_path = _write_analysis(root, _analysis_fixture())
            with self.assertRaisesRegex(renderer.PaperOutputError, "must be named"):
                _render(analysis_path, os.path.join(root, "other_outputs"))
            renamed = os.path.join(root, "fixture.json")
            os.rename(analysis_path, renamed)
            with self.assertRaisesRegex(renderer.PaperOutputError, "must be named analysis.json"):
                _render(renamed, os.path.join(root, "paper_outputs"))
            os.rename(renamed, analysis_path)
            target = os.path.join(root, "target.json")
            os.rename(analysis_path, target)
            os.symlink(target, analysis_path)
            with self.assertRaisesRegex(renderer.PaperOutputError, "non-symlink"):
                _render(analysis_path, os.path.join(root, "paper_outputs"))


if __name__ == "__main__":
    unittest.main()

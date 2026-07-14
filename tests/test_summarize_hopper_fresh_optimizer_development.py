#!/usr/bin/env python3
"""Focused tests for the manifest-driven Hopper development validator."""

from __future__ import annotations

import inspect
import json
import os
import tempfile
import unittest

import numpy as np

from experiments.train import _source_digest
import scripts.summarize_hopper_fresh_optimizer_development as summary


class HopperFreshDevelopmentSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_sha = summary._sha256_file(summary.DEFAULT_MANIFEST_PATH)
        cls.launcher_sha = summary._sha256_file(summary.DEFAULT_LAUNCHER_PATH)
        cls.source_sha = _source_digest(summary.CONFIG_PATH)
        cls.manifest, _ = summary.load_and_validate_manifest(
            summary.DEFAULT_MANIFEST_PATH,
            expected_sha256=cls.manifest_sha,
        )

    @staticmethod
    def _serialized(record: dict[str, object], key: str, values: list[float]) -> None:
        record[key] = values
        record[f"{key}_length"] = len(values)
        record[f"{key}_omitted"] = False
        record[f"{key}_serialization"] = "persisted"

    @classmethod
    def _record(
        cls,
        cell: dict[str, object],
        index: int,
        previous_eval_reward: float,
        best_reward: float,
    ) -> tuple[dict[str, object], float, float]:
        actual_eval = index % 10 == 0 or index == 249
        eval_reward = float(index + 1) if actual_eval else previous_eval_reward
        best_reward = max(best_reward, eval_reward)
        eval_count = sum(1 for value in range(index + 1) if value % 10 == 0 or value == 249)
        lr = float(cell["learning_rate"])
        if cell["lr_schedule"] == "inverse_sqrt":
            lr /= np.sqrt(index + 1.0)
        condition = str(cell["condition"])
        step_norm = float(cell.get("clipup_max_speed", 1.0)) if condition == summary.CLIPUP else 1.0
        record: dict[str, object] = {
            "iteration": index,
            "lr": lr,
            "learning_rate": lr,
            "eval_reward": eval_reward,
            "best_reward": best_reward,
            "initial_eval_reward": 0.5,
            "initial_eval_env_steps": 5,
            "normalization_calibration_env_steps": 3,
            "training_env_steps_iter": 200,
            "training_env_steps": 200 * (index + 1),
            "train_env_steps_iter": 200,
            "train_env_steps": 200 * (index + 1),
            "env_steps_iter": 200,
            "env_steps": 200 * (index + 1),
            "eval_env_steps_iter": 5 if actual_eval else 0,
            "eval_env_steps": 5 + 5 * eval_count,
            "total_env_steps_iter": 205 if actual_eval else 200,
            "total_env_steps": 3 + 200 * (index + 1) + 5 + 5 * eval_count,
            "n_fresh": 200,
            "n_reused": 0,
            "used_replay": False,
            "replay_weight_mass": 0.0,
            "fresh_weight_mass": 1.0,
            "buffer_size": 0,
            "ess": 200.0,
            "ess_ratio": 1.0,
            "ess_normalized": 1.0,
            "importance_weight_mean": 1.0,
            "importance_weight_min": 1.0,
            "importance_weight_max": 1.0,
            "clip_frac": 0.0,
            "clip_fraction": 0.0,
            "parameter_projection_active": False,
            "curvature_clip_frac": 0.0,
            "sigma": 0.02,
            "eval_count": 200 * (index + 1),
            "grad_norm": 2.0,
            "grad_norm_before_clip": 2.0,
            "param_norm": 3.0,
            "param_change": step_norm,
            "step_norm": step_norm,
            "proposed_step_norm": step_norm,
            "explicit_step_norm": step_norm,
            "explicit_gradient_step_norm": step_norm,
            "step_norm_ratio": step_norm / (step_norm + 1e-12),
            "time": 0.1,
            "iteration_compute_seconds": 0.1,
            "mean_fitness": 1.0,
            "max_fitness": 2.0,
            "min_fitness": 0.0,
        }
        if condition in summary.CURVATURE_CONDITIONS:
            attenuation = "isotropic_norm_matched" if condition == summary.ISOTROPIC else "structured"
            estimator = "block_joint_ols" if condition == summary.OLS_GATE else "stein_moment"
            record.update(
                {
                    "solver_type": (
                        "concave_projected_block_isotropic_attenuation_control"
                        if condition == summary.ISOTROPIC
                        else "concave_projected_block"
                    ),
                    "solve_success": True,
                    "implicit_damping": 0.0,
                    "fitness_transform": "centered_rank",
                    "curvature_estimator": estimator,
                    "curvature_attenuation_mode": attenuation,
                    "curvature_fitness": "matched",
                    "curvature_matches_gradient": True,
                    "curvature_mode": "block",
                    "curvature_structure": "block",
                    "curvature_beta": 0.9,
                    "curvature_same_generation": False,
                    "curvature_components": 3,
                    "curvature_component_count": 3,
                    "hessian_pairs": 100,
                    "hessian_ema_count": index + 1,
                    "curvature_step_state": "bias_corrected_ema",
                    "curvature_same_generation_se_available": True,
                    "h_split_available": True,
                    "h_split_rank_semantics": "independent_centered_ranks_per_disjoint_pair_half",
                    "h_split_pair_partition": "first_vs_second_antithetic_pair_halves",
                    "h_split_first_pair_count": 50,
                    "h_split_second_pair_count": 50,
                    "h_temporal_available": index > 0,
                    "curvature_confidence_gate_enabled": condition == summary.OLS_GATE,
                    "curvature_confidence_z": 1.645 if condition == summary.OLS_GATE else None,
                    "h_split_correlation": 1.0,
                    "h_split_sign_agreement": 1.0,
                    "h_split_relative_disagreement": 0.0,
                    "h_temporal_correlation": 0.0 if index == 0 else 1.0,
                    "h_temporal_sign_agreement": 0.0 if index == 0 else 1.0,
                    "h_temporal_relative_disagreement": 0.0,
                    "curvature_projection_frac": 0.5,
                    "curvature_projection_parameter_frac": 0.5,
                    "curvature_active_frac": 0.5,
                    "curvature_confidence_pass_frac": 0.5,
                    "curvature_confidence_gate_frac": 0.25 if condition == summary.OLS_GATE else 0.0,
                    "attenuation_norm_match_relative_error": 0.0,
                    "linear_relative_residual": 0.0,
                    "structured_reference_relative_residual": 0.0,
                    "denominator_min": 1.0,
                    "denominator_max": 2.0,
                    "denominator_condition": 2.0,
                    "h_raw_std": 1.0,
                    "h_raw_mean": -1.0,
                    "isotropic_attenuation_scale": 0.5 if condition == summary.ISOTROPIC else 1.0,
                    "structured_reference_step_norm": step_norm,
                }
            )
            vectors = {
                "curvature_block_sizes": [768.0, 4160.0, 195.0],
                "h_split_first_components": [1.0, 2.0, 3.0],
                "h_split_second_components": [1.0, 2.0, 3.0],
                "curvature_same_generation_components": [-1.0, 0.0, 1.0],
                "curvature_same_generation_se_components": [0.1, 0.1, 0.1],
                "curvature_step_state_components": [-1.0, 0.0, 1.0],
                "curvature_step_state_se_components": [0.1, 0.1, 0.1],
                "curvature_confidence_upper_components": [-0.5, 0.5, 1.5],
                "curvature_raw_components": [-1.0, 0.0, 1.0],
                "curvature_ema_components": [-1.0, 0.0, 1.0],
                "curvature_ema_variance_components": [0.1, 0.1, 0.1],
                "curvature_bias_corrected_ema_components": [-1.0, 0.0, 1.0],
                "curvature_step_components": [-1.0, 0.0, 1.0],
                "concave_curvature_components": [1.0, 0.0, 0.0],
                "denominator_components": [2.0, 1.0, 1.0],
            }
            for key, values in vectors.items():
                cls._serialized(record, key, values)
            if condition == summary.OLS_GATE:
                record.update(
                    {
                        "regression_rank": 4,
                        "regression_parameters": 4,
                        "regression_residual_dof": 96,
                        "regression_residual_std": 0.1,
                        "regression_r_squared": 0.5,
                        "regression_design_condition": 2.0,
                    }
                )
        else:
            record["solver_type"] = "none"
            if condition == summary.MOMENTUM:
                record.update(
                    {
                        "optimizer_type": "momentum",
                        "momentum_beta": cell["momentum_beta"],
                        "momentum_iteration": index + 1,
                        "momentum_buffer_norm": 1.0,
                    }
                )
            elif condition == summary.ADAM:
                record.update(
                    {
                        "optimizer_type": "adam",
                        "adam_beta1": cell["adam_beta1"],
                        "adam_beta2": cell["adam_beta2"],
                        "adam_epsilon": cell["adam_epsilon"],
                        "adam_iteration": index + 1,
                        "adam_first_moment_norm": 1.0,
                        "adam_second_moment_norm": 1.0,
                        "adam_first_moment_bias_correction": 1.0 - float(cell["adam_beta1"]) ** (index + 1),
                        "adam_second_moment_bias_correction": 1.0 - float(cell["adam_beta2"]) ** (index + 1),
                    }
                )
            elif condition == summary.CLIPUP:
                vmax = float(cell["clipup_max_speed"])
                record.update(
                    {
                        "optimizer_type": "clipup",
                        "clipup_momentum": cell["clipup_momentum"],
                        "clipup_max_speed": vmax,
                        "clipup_iteration": index + 1,
                        "clipup_step_size": lr,
                        "clipup_input_gradient_norm": 2.0,
                        "clipup_zero_gradient": False,
                        "clipup_normalized_gradient_step_norm": lr,
                        "clipup_velocity_norm_before_clip": 2.0 * vmax,
                        "clipup_velocity_norm": vmax,
                        "clipup_velocity_clip_scale": 0.5,
                        "clipup_velocity_clipped": True,
                    }
                )
        return record, eval_reward, best_reward

    @classmethod
    def _history(cls, cell: dict[str, object]) -> list[dict[str, object]]:
        records = []
        previous = 0.5
        best = 0.5
        for index in range(250):
            record, previous, best = cls._record(cell, index, previous, best)
            records.append(record)
        return records

    def test_manifest_and_rotated_mapping_are_exact_and_bijective(self) -> None:
        self.assertEqual(len(self.manifest["cells"]), 33)
        mappings = {
            summary.mapping_for_task(task_id) for task_id in range(99)
        }
        self.assertEqual(len(mappings), 99)
        for cell_id, seed_index in mappings:
            task_id = summary.task_id_for(cell_id, seed_index)
            self.assertEqual(summary.mapping_for_task(task_id), (cell_id, seed_index))

    def test_generation_auc_ignores_carried_forward_non_eval_reward(self) -> None:
        history = [
            {"initial_eval_reward": 0.0, "eval_env_steps_iter": 1, "eval_reward": 1.0},
            {"initial_eval_reward": 0.0, "eval_env_steps_iter": 0, "eval_reward": 999.0},
            {"initial_eval_reward": 0.0, "eval_env_steps_iter": 1, "eval_reward": 3.0},
        ]
        metrics = summary.evaluation_generation_metrics(
            history, expected_iterations=3
        )
        self.assertEqual(metrics["evaluation_generations"], [0, 1, 3])
        self.assertAlmostEqual(metrics["evaluation_generation_auc"], 1.5)
        self.assertEqual(metrics["best_return"], 3.0)

    def test_each_optimizer_history_schema_validates_and_summarizes(self) -> None:
        representatives = {}
        for cell in self.manifest["cells"]:
            representatives.setdefault(cell["condition"], cell)
        for condition, cell in representatives.items():
            with self.subTest(condition=condition):
                history = self._history(cell)
                issues: list[str] = []
                metrics = summary._validate_history(
                    history, "/tmp/run", cell, issues
                )
                self.assertEqual(issues, [])
                self.assertIsNotNone(metrics)
                self.assertEqual(metrics["evaluation_point_count"], 27)
                self.assertEqual(metrics["training_env_steps"], 50_000)
                expected_step = (
                    float(cell["clipup_max_speed"])
                    if condition == summary.CLIPUP
                    else 1.0
                )
                expected_over_sigma = expected_step / 0.02
                self.assertAlmostEqual(
                    metrics["first_step_over_sigma"], expected_over_sigma
                )
                self.assertAlmostEqual(
                    metrics["mean_step_over_sigma"], expected_over_sigma
                )
                self.assertAlmostEqual(
                    metrics["max_step_over_sigma"], expected_over_sigma
                )
                self.assertEqual(
                    metrics["local_step_fraction"],
                    float(expected_over_sigma <= 1.0),
                )

    def test_provenance_requires_all_three_enforced_hash_locks(self) -> None:
        cell = self.manifest["cells"][0]
        seed = 200
        task_id = summary.task_id_for(0, 0)
        run_dir = f"/tmp/cell0_{cell['label']}_seed200_job123_task{task_id}"
        source_sha = "a" * 64
        config = {
            "provenance": {
                "source_sha256": source_sha,
                "expected_source_sha256": source_sha,
                "expected_manifest_sha256": self.manifest_sha,
                "expected_launcher_sha256": self.launcher_sha,
                "dependencies": {
                    "gymnasium": "1",
                    "mujoco": "1",
                    "PyYAML": "1",
                },
                "rng_scheme": {
                    "optimizer": "numpy.RandomState(run_seed)",
                    "parameter_initialization": "numpy.default_rng(SeedSequence([run_seed, 1]))",
                    "rollout": "injective Cantor encoding of (run_seed, stream, iteration, index)",
                },
                "slurm_array_job_id": "123",
                "slurm_array_task_id": str(task_id),
                "argv": summary._expected_argv(cell, seed, run_dir, "2"),
            }
        }
        issues: list[str] = []
        summary._validate_provenance(
            config,
            run_dir,
            cell,
            seed,
            task_id,
            source_sha,
            self.manifest_sha,
            self.launcher_sha,
            issues,
        )
        self.assertEqual(issues, [])
        del config["provenance"]["expected_launcher_sha256"]
        issues = []
        summary._validate_provenance(
            config,
            run_dir,
            cell,
            seed,
            task_id,
            source_sha,
            self.manifest_sha,
            self.launcher_sha,
            issues,
        )
        self.assertTrue(any("expected_launcher_sha256" in issue for issue in issues))

    def test_snapshot_config_path_is_accepted_but_wrong_protocol_is_rejected(self) -> None:
        cell = self.manifest["cells"][0]
        config = dict(summary.EXPECTED_COMMON_CONFIG)
        config.update(
            {
                "_config_path": (
                    "/immutable/source-snapshot/"
                    + summary.CONFIG_RELATIVE_PATH
                ),
                "condition": cell["condition"],
                "seed": 200,
                "learning_rate": cell["learning_rate"],
                "lr_schedule": cell["lr_schedule"],
                "provenance": {},
                "resolved_optimizer": summary._resolved_optimizer_values(cell),
                **summary._condition_config_values(cell),
            }
        )
        issues: list[str] = []
        summary._validate_config(config, "/tmp/run", cell, 200, issues)
        self.assertEqual(issues, [])

        config["_config_path"] = "/immutable/source-snapshot/configs/mujuco/hopper.yaml"
        issues = []
        summary._validate_config(config, "/tmp/run", cell, 200, issues)
        self.assertTrue(any("protocol path" in issue for issue in issues))

    def test_grouping_outputs_and_contrasts_remain_exploratory(self) -> None:
        rows = []
        for cell in self.manifest["cells"]:
            for seed in summary.SEEDS:
                row = {
                    "cell_id": cell["cell_id"],
                    "label": cell["label"],
                    "condition": cell["condition"],
                    "seed": seed,
                    "learning_rate": cell["learning_rate"],
                    "lr_schedule": cell["lr_schedule"],
                    "momentum_beta": cell.get("momentum_beta"),
                    "adam_beta1": cell.get("adam_beta1"),
                    "adam_beta2": cell.get("adam_beta2"),
                    "adam_epsilon": cell.get("adam_epsilon"),
                    "clipup_momentum": cell.get("clipup_momentum"),
                    "clipup_max_speed": cell.get("clipup_max_speed"),
                }
                for metric in summary.GROUP_METRICS:
                    if metric.startswith("clipup_"):
                        value = 0.5 if cell["condition"] == summary.CLIPUP else None
                    elif metric.startswith("mean_h_") or metric.startswith("mean_curvature_") or metric.startswith("mean_denominator") or metric.startswith("mean_isotropic") or metric.startswith("mean_attenuation") or metric.startswith("mean_regression") or metric == "solve_success_fraction":
                        value = 0.5 if cell["condition"] in summary.CURVATURE_CONDITIONS else None
                        if metric.startswith("mean_regression") and cell["condition"] != summary.OLS_GATE:
                            value = None
                    else:
                        value = float(cell["cell_id"] + seed)
                    row[metric] = value
                row["best_return"] = float(cell["cell_id"] + seed + 1)
                rows.append(row)
        groups = summary.aggregate_runs(rows, self.manifest)
        contrasts = summary.paired_structured_minus_isotropic(rows)
        self.assertEqual(len(groups), 33)
        self.assertTrue(all(group["runs"] == 3 for group in groups))
        self.assertEqual(len(contrasts), 6)
        self.assertTrue(all(item["exploratory"] for item in contrasts))
        serialized = json.dumps(contrasts)
        self.assertNotIn("p_value", serialized)

    def test_source_digest_freezes_development_protocol_and_analyzer(self) -> None:
        source = inspect.getsource(_source_digest)
        self.assertIn("docs/hopper_fresh_optimizer_development_protocol.md", source)
        self.assertIn("scripts/summarize_hopper_fresh_optimizer_development.py", source)
        self.assertIn("environment.yml", source)
        self.assertIn("requirement.txt", source)

    def test_collector_rejects_an_unlocked_analyzer_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(
                summary.DevelopmentValidationError,
                "analyzer checkout source digest mismatch",
            ):
                summary.validate_and_collect(
                    root,
                    expected_source_sha="a" * 64,
                    expected_manifest_sha=self.manifest_sha,
                    expected_launcher_sha=self.launcher_sha,
                )

    def test_incomplete_matrix_fails_closed_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_output = os.path.join(root, "runs.csv")
            group_output = os.path.join(root, "groups.csv")
            json_output = os.path.join(root, "summary.json")
            with self.assertRaises(summary.DevelopmentValidationError):
                summary.summarize(
                    root,
                    manifest_path=summary.DEFAULT_MANIFEST_PATH,
                    launcher_path=summary.DEFAULT_LAUNCHER_PATH,
                    expected_source_sha=self.source_sha,
                    expected_manifest_sha=self.manifest_sha,
                    expected_launcher_sha=self.launcher_sha,
                    run_output=run_output,
                    group_output=group_output,
                    json_output=json_output,
                )
            self.assertFalse(os.path.exists(run_output))
            self.assertFalse(os.path.exists(group_output))
            self.assertFalse(os.path.exists(json_output))


if __name__ == "__main__":
    unittest.main()

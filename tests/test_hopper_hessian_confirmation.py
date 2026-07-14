#!/usr/bin/env python3
"""Regression tests for the strict Hopper Hessian confirmation analyzer."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from scripts.summarize_hopper_hessian_confirmation import (
    CONDITIONS,
    CONFIG_PATH,
    CONFIG_RELATIVE_PATH,
    EXPECTED_COMMON_CONFIG,
    HELDOUT_EPISODES,
    INITIAL_LEARNING_RATE,
    ISOTROPIC,
    LR_SCHEDULE,
    SEEDS,
    STANDARD,
    STRUCTURED,
    TRAINING_STEP_BUDGET,
    ConfirmationValidationError,
    _heldout_seed_bank,
    _metrics_at_budget,
    _task_id,
    summarize,
    validate_and_collect,
)


SOURCE_SHA = "a" * 64
JOB_ID = "98765"


class ConfirmationFixture:
    @staticmethod
    def _resolved_heldout() -> dict[str, object]:
        return {
            "enabled": True,
            "artifact": "heldout_evaluation.json",
            "training_step_budget": TRAINING_STEP_BUDGET,
            "episodes_per_checkpoint": HELDOUT_EPISODES,
            "checkpoint_selection": "initial_and_every_center_through_first_budget_crossing",
            "execution_phase": "post_training",
            "rollout_seed_stream": 4,
            "common_seed_bank_across_checkpoints": True,
            "optimizer_or_checkpoint_selection_uses_heldout_results": False,
            "observation_normalizer_state": "frozen_per_checkpoint",
        }

    @staticmethod
    def _resolved_optimizer(condition: str) -> dict[str, object]:
        resolved: dict[str, object] = {
            "type": "StandardES" if condition == STANDARD else "ConcaveCurvatureES",
            "population_size": 200,
            "initial_learning_rate": 10.0,
            "noise_std": 0.02,
            "rank_fitness": True,
            "l2_coeff": 0.0,
            "antithetic": True,
            "max_grad_norm": 0.0,
            "max_param_norm": None,
            "trust_region": False,
            "replay_enabled": False,
        }
        if condition != STANDARD:
            attenuation = "structured" if condition == STRUCTURED else "isotropic_norm_matched"
            resolved.update(
                {
                    "method": "concave_curvature",
                    "implicit_damping": 0.0,
                    "curvature_fitness": "matched",
                    "curvature_mode": "block",
                    "curvature_structure": "block",
                    "curvature_beta": 0.9,
                    "curvature_same_generation": False,
                    "curvature_estimator": "stein_moment",
                    "curvature_confidence_z": None,
                    "curvature_attenuation_mode": attenuation,
                    "curvature_clipping": False,
                    "curvature_projection": "concave",
                    "curvature_components": 3,
                    "solver_type": (
                        "concave_projected_block"
                        if condition == STRUCTURED
                        else "concave_projected_block_isotropic_attenuation_control"
                    ),
                }
            )
        return resolved

    @staticmethod
    def _history(condition: str, seed: int) -> list[dict[str, object]]:
        history: list[dict[str, object]] = []
        for iteration in range(500):
            cumulative_steps = 10_000 * (iteration + 1)
            record: dict[str, object] = {
                "iteration": iteration,
                "lr": INITIAL_LEARNING_RATE / (iteration + 1),
                "eval_reward": float(seed),
                "best_reward": float(seed),
                "initial_eval_reward": float(seed),
                "training_env_steps": cumulative_steps,
                "training_env_steps_iter": 10_000,
                "normalization_calibration_env_steps": 300,
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
                "parameter_projection_active": False,
                "curvature_clip_frac": 0.0,
                "grad_norm": 1.0,
                "grad_norm_before_clip": 1.0,
                "step_norm": 1.0,
                "proposed_step_norm": 1.0,
                "explicit_step_norm": 1.0 if condition == STANDARD else 2.0,
                "step_norm_ratio": (
                    1.0 / (1.0 + 1e-12) if condition == STANDARD else 0.5
                ),
                "solver_type": "none",
            }
            if condition != STANDARD:
                attenuation = "structured" if condition == STRUCTURED else "isotropic_norm_matched"
                record.update(
                    {
                        "solver_type": (
                            "concave_projected_block"
                            if condition == STRUCTURED
                            else "concave_projected_block_isotropic_attenuation_control"
                        ),
                        "solve_success": True,
                        "implicit_damping": 0.0,
                        "curvature_beta": 0.9,
                        "curvature_same_generation": False,
                        "curvature_step_state": "bias_corrected_ema",
                        "curvature_estimator": "stein_moment",
                        "curvature_attenuation_mode": attenuation,
                        "curvature_confidence_gate_enabled": False,
                        "curvature_confidence_gate_frac": 0.0,
                        "curvature_fitness": "matched",
                        "curvature_matches_gradient": True,
                        "curvature_mode": "block",
                        "curvature_structure": "block",
                        "curvature_components": 3,
                        "curvature_block_size_min": 195,
                        "curvature_block_size_max": 4160,
                        "hessian_pairs": 100,
                        "hessian_ema_count": iteration + 1,
                        "curvature_same_generation_se_available": True,
                        "linear_nonpositive_diagonal_frac": 0.0,
                        "linear_relative_residual": 1e-16,
                        "structured_reference_relative_residual": 1e-16,
                        "attenuation_norm_match_relative_error": 0.0,
                        "denominator_min": 1.0,
                        "denominator_max": 2.0,
                        "linear_condition_estimate": 2.0,
                        "curvature_projection_frac": 0.5,
                        "curvature_active_frac": 0.5,
                        "structured_reference_step_norm": 1.0,
                        "isotropic_attenuation_scale": (
                            1.0 if condition == STRUCTURED else 0.5
                        ),
                        "h_raw_mean": -0.5,
                        "h_raw_std": 0.2,
                    }
                )
            history.append(record)
        return history

    @staticmethod
    def _heldout(condition: str, seed: int) -> dict[str, object]:
        checkpoint_steps = [0] + [10_000 * index for index in range(1, 9)]
        records: list[dict[str, object]] = []
        for checkpoint_index, step in enumerate(checkpoint_steps):
            effect = 0.0
            if checkpoint_index > 0:
                effect = 3.0 if condition == STRUCTURED else (1.0 if condition == ISOTROPIC else 0.0)
            episode_returns = [
                seed * 0.01 + checkpoint_index + effect + episode * 0.01
                for episode in range(HELDOUT_EPISODES)
            ]
            records.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "source_iteration": None if checkpoint_index == 0 else checkpoint_index - 1,
                    "training_env_steps": step,
                    "mean_return": sum(episode_returns) / HELDOUT_EPISODES,
                    "episode_returns": episode_returns,
                    "episode_env_steps": [100] * HELDOUT_EPISODES,
                }
            )
        means = [float(record["mean_return"]) for record in records]
        auc, at_budget = _metrics_at_budget(checkpoint_steps, means)
        return {
            "schema_version": 1,
            "training_step_budget": TRAINING_STEP_BUDGET,
            "episodes_per_checkpoint": HELDOUT_EPISODES,
            "checkpoint_selection": "initial_and_every_center_through_first_budget_crossing",
            "rollout_seed_stream": 4,
            "rollout_seeds": _heldout_seed_bank(seed),
            "common_seed_bank_across_checkpoints": True,
            "optimizer_or_checkpoint_selection_uses_heldout_results": False,
            "observation_normalizer_state": "frozen_per_checkpoint",
            "checkpoint_count": len(records),
            "heldout_evaluation_env_steps": len(records) * HELDOUT_EPISODES * 100,
            "normalized_auc_at_budget": auc,
            "return_at_budget": at_budget,
            "checkpoints": records,
        }

    @classmethod
    def write_run(cls, root: str, condition: str, seed: int) -> str:
        task_id = _task_id(condition, seed)
        run_dir = os.path.join(
            root,
            f"{condition}_inverse_linear_a10_seed{seed}_job{JOB_ID}_task{task_id}",
        )
        os.makedirs(run_dir)
        config = dict(EXPECTED_COMMON_CONFIG)
        config.update(
            {
                "_config_path": CONFIG_PATH,
                "condition": condition,
                "seed": seed,
                "curvature_beta": 0.0 if condition == STANDARD else 0.9,
                "algorithm": "standard_es" if condition == STANDARD else "concave_curvature_es",
                "use_curvature": condition != STANDARD,
                "resolved_optimizer": cls._resolved_optimizer(condition),
                "resolved_heldout_evaluation": cls._resolved_heldout(),
            }
        )
        if condition != STANDARD:
            config.update(
                {
                    "curvature_fitness": "matched",
                    "curvature_mode": "block",
                    "curvature_estimator": "stein_moment",
                    "curvature_confidence_z": None,
                    "curvature_attenuation_mode": (
                        "structured" if condition == STRUCTURED else "isotropic_norm_matched"
                    ),
                }
            )
        config["provenance"] = {
            "git_revision": "deadbeef",
            "source_sha256": SOURCE_SHA,
            "expected_source_sha256": SOURCE_SHA,
            "argv": [
                "experiments/train.py",
                "--config",
                CONFIG_RELATIVE_PATH,
                "--condition",
                condition,
                "--learning-rate",
                "10",
                "--lr-schedule",
                LR_SCHEDULE,
                "--reuse-fraction",
                "0",
                "--seed",
                str(seed),
                "--workers",
                "30",
                "--output",
                run_dir,
            ],
            "hostname": "test-host",
            "python": "3.test",
            "numpy": "2.test",
            "dependencies": {
                "gymnasium": "1.test",
                "mujoco": "3.test",
                "PyYAML": "6.test",
            },
            "slurm_job_id": JOB_ID,
            "slurm_array_job_id": JOB_ID,
            "slurm_array_task_id": str(task_id),
            "started_at": "2026-01-01T00:00:00+00:00",
            "rng_scheme": {
                "heldout_evaluation": "stream=4 with fixed (run_seed, episode_index) bank"
            },
        }
        history = cls._history(condition, seed)
        heldout = cls._heldout(condition, seed)
        status = {
            "status": "complete",
            "expected_iterations": 500,
            "completed_iterations": 500,
            "history_records": "history.jsonl",
            "initial_eval_reward": float(seed),
            "best_reward": float(seed),
            "normalization_calibration_env_steps": 300,
            "heldout_evaluation": {
                "status": "complete",
                "artifact": "heldout_evaluation.json",
                "training_step_budget": TRAINING_STEP_BUDGET,
                "episodes_per_checkpoint": HELDOUT_EPISODES,
                "checkpoint_count": heldout["checkpoint_count"],
                "normalized_auc_at_budget": heldout["normalized_auc_at_budget"],
                "return_at_budget": heldout["return_at_budget"],
            },
        }
        for filename, value in (
            ("config.json", config),
            ("history.json", history),
            ("status.json", status),
            ("heldout_evaluation.json", heldout),
        ):
            with open(os.path.join(run_dir, filename), "w", encoding="utf-8") as stream:
                json.dump(value, stream)
        with open(os.path.join(run_dir, "history.jsonl"), "w", encoding="utf-8") as stream:
            for record in history:
                stream.write(json.dumps(record, separators=(",", ":")))
                stream.write("\n")
        return run_dir

    @classmethod
    def write_matrix(cls, root: str) -> dict[tuple[str, int], str]:
        runs = {
            (condition, seed): cls.write_run(root, condition, seed)
            for seed in SEEDS
            for condition in CONDITIONS
        }
        job_output_dir = os.path.join(root, "job_outputs")
        os.makedirs(job_output_dir)
        for condition, seed in runs:
            task_id = _task_id(condition, seed)
            with open(
                os.path.join(job_output_dir, f"hopper_hconf_{JOB_ID}_{task_id}.err"),
                "w",
                encoding="utf-8",
            ):
                pass
        return runs


class HopperHessianConfirmationTests(unittest.TestCase):
    def test_complete_matrix_validates_and_produces_preregistered_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            ConfirmationFixture.write_matrix(root)
            run_output = os.path.join(root, "validated.csv")
            contrast_output = os.path.join(root, "contrasts.json")
            rows, analysis = summarize(
                root,
                expected_source_sha=SOURCE_SHA,
                run_output=run_output,
                contrast_output=contrast_output,
                job_output_dir=os.path.join(root, "job_outputs"),
            )
            self.assertEqual(len(rows), 30)
            self.assertEqual(len(analysis["contrasts"]), 2)
            self.assertTrue(analysis["confirmation_claim_supported"])
            self.assertEqual(
                [record["sign_flip_p_raw"] for record in analysis["contrasts"]],
                [2 / 1024, 2 / 1024],
            )
            self.assertEqual(
                [record["holm_adjusted_p"] for record in analysis["contrasts"]],
                [4 / 1024, 4 / 1024],
            )
            self.assertTrue(os.path.isfile(run_output))
            self.assertTrue(os.path.isfile(contrast_output))

    def test_relocated_archived_absolute_config_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = ConfirmationFixture.write_matrix(root)
            archived_path = os.path.join(
                os.sep,
                "archived",
                "original_checkout",
                "scratch",
                os.pardir,
                CONFIG_RELATIVE_PATH,
            )
            for run_dir in runs.values():
                path = os.path.join(run_dir, "config.json")
                with open(path, "r", encoding="utf-8") as stream:
                    config = json.load(stream)
                config["_config_path"] = archived_path
                with open(path, "w", encoding="utf-8") as stream:
                    json.dump(config, stream)

            rows = validate_and_collect(
                root,
                expected_source_sha=SOURCE_SHA,
                job_output_dir=os.path.join(root, "job_outputs"),
            )
            self.assertEqual(len(rows), 30)

    def test_wrong_relocated_config_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = ConfirmationFixture.write_matrix(root)
            path = os.path.join(runs[(STANDARD, 100)], "config.json")
            with open(path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            config["_config_path"] = os.path.join(
                os.sep,
                "archived",
                "original_checkout",
                "configs",
                "mujuco",
                "hopper_hessian_fix_no_replay.yaml",
            )
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)

            with self.assertRaises(ConfirmationValidationError) as raised:
                validate_and_collect(
                    root,
                    expected_source_sha=SOURCE_SHA,
                    job_output_dir=os.path.join(root, "job_outputs"),
                )
            self.assertIn(
                "config path is not the locked confirmation config",
                "\n".join(raised.exception.issues),
            )

    def test_corrupt_heldout_mean_is_rejected_before_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = ConfirmationFixture.write_matrix(root)
            path = os.path.join(runs[(STRUCTURED, 100)], "heldout_evaluation.json")
            with open(path, "r", encoding="utf-8") as stream:
                heldout = json.load(stream)
            heldout["checkpoints"][1]["mean_return"] += 1.0
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(heldout, stream)
            run_output = os.path.join(root, "should_not_exist.csv")
            contrast_output = os.path.join(root, "should_not_exist.json")
            with self.assertRaises(ConfirmationValidationError):
                summarize(
                    root,
                    expected_source_sha=SOURCE_SHA,
                    run_output=run_output,
                    contrast_output=contrast_output,
                    job_output_dir=os.path.join(root, "job_outputs"),
                )
            self.assertFalse(os.path.exists(run_output))
            self.assertFalse(os.path.exists(contrast_output))

    def test_rotated_task_mapping_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runs = ConfirmationFixture.write_matrix(root)
            path = os.path.join(runs[(STANDARD, 101)], "config.json")
            with open(path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            config["provenance"]["slurm_array_task_id"] = "3"
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)
            with self.assertRaises(ConfirmationValidationError) as raised:
                validate_and_collect(
                    root,
                    expected_source_sha=SOURCE_SHA,
                    job_output_dir=os.path.join(root, "job_outputs"),
                )
            self.assertIn("rotated mapping", "\n".join(raised.exception.issues))

    def test_nonempty_slurm_stderr_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            ConfirmationFixture.write_matrix(root)
            stderr_path = os.path.join(
                root,
                "job_outputs",
                f"hopper_hconf_{JOB_ID}_{_task_id(STANDARD, 100)}.err",
            )
            with open(stderr_path, "w", encoding="utf-8") as stream:
                stream.write("failure\n")
            with self.assertRaises(ConfirmationValidationError) as raised:
                validate_and_collect(
                    root,
                    expected_source_sha=SOURCE_SHA,
                    job_output_dir=os.path.join(root, "job_outputs"),
                )
            self.assertIn("stderr artifact is nonempty", "\n".join(raised.exception.issues))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the no-Picard Hopper Hessian sweep validator."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from scripts.summarize_hopper_implicit_sweep import (
    COMMON_CONFIG,
    CONDITIONS,
    HESSIAN_FIX_CONDITIONS,
    HESSIAN_FIX_SEEDS,
    INITIAL_LEARNING_RATES,
    LR_SCHEDULES,
    SEEDS,
    ValidationError,
    aggregate,
    paired_contrasts,
    validate_and_collect,
)


class HopperHessianSummaryTests(unittest.TestCase):
    @staticmethod
    def _learning_rate(initial: float, iteration: int, schedule: str) -> float:
        if schedule == "inverse_sqrt":
            return initial / (iteration + 1) ** 0.5
        if schedule == "inverse_linear":
            return initial / (iteration + 1)
        raise ValueError(schedule)

    @classmethod
    def _record(
        cls, condition: str, iteration: int, learning_rate: float, schedule: str, seed: int
    ) -> dict[str, object]:
        record: dict[str, object] = {
            "iteration": iteration,
            "lr": cls._learning_rate(learning_rate, iteration, schedule),
            "eval_reward": float(iteration + seed + 1),
            "initial_eval_reward": float(seed) + 0.5,
            "training_env_steps": 100 * (iteration + 1),
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
            "solver_type": "none",
        }
        if condition == "linearized_implicit_es":
            record.update(
                {
                    "solver_type": "signed_diagonal_linearized_implicit",
                    "solve_success": True,
                    "implicit_converged": True,
                    "implicit_relative_residual": 1e-16,
                    "linear_relative_residual": 1e-16,
                    "linear_min_abs_diagonal": 0.2,
                    "linear_condition_estimate": 100.0,
                    "linear_nonpositive_diagonal_frac": 0.4,
                    "hessian_pairs": 100,
                    "h_raw_std": 20.0,
                    "h_split_correlation": 0.05,
                    "h_split_sign_agreement": 0.51,
                    "h_temporal_correlation": 0.01,
                    "h_temporal_sign_agreement": 0.5,
                    "step_norm_ratio": 2.0,
                }
            )
        elif condition in {
            "concave_diagonal_curvature_es",
            "concave_block_curvature_es",
            "concave_block_ema_curvature_es",
        }:
            structure = (
                "diag"
                if condition == "concave_diagonal_curvature_es"
                else "block"
            )
            beta = 0.9 if condition == "concave_block_ema_curvature_es" else 0.0
            components = 5123 if structure == "diag" else 3
            record.update(
                {
                    "solver_type": f"concave_projected_{structure}",
                    "solve_success": True,
                    "linear_relative_residual": 1e-16,
                    "linear_min_abs_diagonal": 1.0,
                    "linear_condition_estimate": 2.0,
                    "linear_nonpositive_diagonal_frac": 0.0,
                    "hessian_pairs": 100,
                    "h_raw_std": 2.0,
                    "h_split_correlation": 0.5,
                    "h_split_sign_agreement": 0.8,
                    "h_split_relative_disagreement": 0.2,
                    "h_temporal_correlation": 0.4,
                    "h_temporal_sign_agreement": 0.7,
                    "h_temporal_relative_disagreement": 0.3,
                    "curvature_projection_frac": 0.2,
                    "curvature_clip_frac": 0.0,
                    "curvature_components": components,
                    "curvature_block_size_min": 1 if structure == "diag" else 195,
                    "curvature_block_size_max": 1 if structure == "diag" else 4160,
                    "curvature_beta": beta,
                    "curvature_mode": structure,
                    "curvature_matches_gradient": True,
                    "curvature_same_generation": beta == 0.0,
                    "curvature_active_frac": 0.8,
                    "hessian_ema_count": iteration + 1,
                    "step_norm_ratio": 0.5,
                }
            )
        return record

    @classmethod
    def _write_run(
        cls,
        root: str,
        *,
        condition: str,
        schedule: str,
        learning_rate: float,
        seed: int,
        task_id: int,
        expected_iterations: int = 2,
        job_id: str = "1234",
    ) -> str:
        name = (
            f"{condition}_{schedule}_a{learning_rate:g}_seed{seed}_"
            f"job{job_id}_task{task_id}"
        )
        run_dir = os.path.join(root, name)
        os.makedirs(run_dir)
        config = dict(COMMON_CONFIG)
        is_concave = condition.startswith("concave_")
        structure = "diag" if condition == "concave_diagonal_curvature_es" else "block"
        beta = 0.9 if condition == "concave_block_ema_curvature_es" else 0.0
        config.update(
            {
                "condition": condition,
                "algorithm": "concave_curvature_es" if is_concave else condition,
                "learning_rate": learning_rate,
                "lr_schedule": schedule,
                "n_iterations": expected_iterations,
                "seed": seed,
                "diagnostic_schema_version": 2,
                "use_curvature": condition != "standard_es",
                "resolved_optimizer": {
                    "type": (
                        "StandardES"
                        if condition == "standard_es"
                        else (
                            "ConcaveCurvatureES"
                            if is_concave
                            else "LinearizedImplicitES"
                        )
                    ),
                    "population_size": 200,
                    "initial_learning_rate": learning_rate,
                    "noise_std": 0.02,
                    "rank_fitness": True,
                    "l2_coeff": 0.0,
                    "antithetic": True,
                    "max_grad_norm": 0.0,
                    "max_param_norm": None,
                    "trust_region": False,
                    "replay_enabled": False,
                },
                "provenance": {
                    "source_sha256": "a" * 64,
                    "expected_source_sha256": "a" * 64,
                    "dependencies": {
                        "gymnasium": "1.0.0",
                        "mujoco": "3.0.0",
                        "PyYAML": "6.0.0",
                    },
                    "slurm_array_job_id": job_id,
                    "slurm_array_task_id": str(task_id),
                },
            }
        )
        if is_concave:
            config.update(
                {
                    "curvature_fitness": "matched",
                    "curvature_mode": structure,
                    "curvature_beta": beta,
                    "implicit_damping": 0.0,
                }
            )
        resolved = config["resolved_optimizer"]
        if condition == "linearized_implicit_es":
            resolved.update(
                {
                    "solver_type": "signed_diagonal_linearized_implicit",
                    "implicit_damping": 0.0,
                    "curvature_fitness": "matched",
                    "curvature_beta": 0.0,
                    "curvature_clipping": False,
                }
            )
        elif is_concave:
            resolved.update(
                {
                    "solver_type": f"concave_projected_{structure}",
                    "implicit_damping": 0.0,
                    "curvature_fitness": "matched",
                    "curvature_mode": structure,
                    "curvature_structure": structure,
                    "curvature_beta": beta,
                    "curvature_clipping": False,
                    "curvature_projection": "concave",
                    "curvature_same_generation": beta == 0.0,
                    "curvature_components": 5123 if structure == "diag" else 3,
                }
            )
        history = [
            cls._record(condition, iteration, learning_rate, schedule, seed)
            for iteration in range(expected_iterations)
        ]
        status = {
            "status": "complete",
            "completed_iterations": expected_iterations,
            "history_records": "history.jsonl",
        }
        for filename, value in (
            ("config.json", config),
            ("history.json", history),
            ("status.json", status),
        ):
            with open(os.path.join(run_dir, filename), "w", encoding="utf-8") as stream:
                json.dump(value, stream)
        with open(os.path.join(run_dir, "history.jsonl"), "w", encoding="utf-8") as stream:
            for record in history:
                stream.write(json.dumps(record, separators=(",", ":")))
                stream.write("\n")
        return run_dir

    @staticmethod
    def _task_id(
        condition_index: int,
        schedule_index: int,
        learning_rate_index: int,
        seed_index: int,
        *,
        n_schedules: int,
        n_learning_rates: int,
        n_seeds: int,
    ) -> int:
        return (
            condition_index * n_schedules * n_learning_rates * n_seeds
            + schedule_index * n_learning_rates * n_seeds
            + learning_rate_index * n_seeds
            + seed_index
        )

    @staticmethod
    def _rewrite_history(run_dir: str, history: list[dict[str, object]]) -> None:
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as stream:
            json.dump(history, stream)
        with open(os.path.join(run_dir, "history.jsonl"), "w", encoding="utf-8") as stream:
            for record in history:
                stream.write(json.dumps(record, separators=(",", ":")))
                stream.write("\n")

    @staticmethod
    def _validate_single_run(root: str, condition: str) -> None:
        validate_and_collect(
            root,
            conditions=(condition,),
            lr_schedules=("inverse_sqrt",),
            learning_rates=(10.0,),
            seeds=(0,),
            expected_iterations=2,
            budget=100,
            expected_source_sha="a" * 64,
        )

    def test_production_grid_is_no_picard_and_has_80_cells(self) -> None:
        self.assertEqual(CONDITIONS, ("standard_es", "linearized_implicit_es"))
        self.assertNotIn("endpoint_implicit_es", CONDITIONS)
        self.assertEqual(LR_SCHEDULES, ("inverse_sqrt", "inverse_linear"))
        self.assertEqual(INITIAL_LEARNING_RATES, (10.0, 30.0))
        self.assertEqual(SEEDS, tuple(range(10)))
        self.assertEqual(
            len(CONDITIONS)
            * len(LR_SCHEDULES)
            * len(INITIAL_LEARNING_RATES)
            * len(SEEDS),
            80,
        )

    def test_hessian_fix_grid_is_no_picard_and_has_100_cells(self) -> None:
        self.assertNotIn("endpoint_implicit_es", HESSIAN_FIX_CONDITIONS)
        self.assertEqual(HESSIAN_FIX_SEEDS, tuple(range(5)))
        self.assertEqual(
            len(HESSIAN_FIX_CONDITIONS)
            * len(LR_SCHEDULES)
            * len(INITIAL_LEARNING_RATES)
            * len(HESSIAN_FIX_SEEDS),
            100,
        )

    def test_complete_two_arm_two_schedule_matrix_validates(self) -> None:
        conditions = ("standard_es", "linearized_implicit_es")
        schedules = ("inverse_sqrt", "inverse_linear")
        learning_rates = (10.0, 30.0)
        seeds = (0, 1)
        with tempfile.TemporaryDirectory() as root:
            for condition_index, condition in enumerate(conditions):
                for schedule_index, schedule in enumerate(schedules):
                    for learning_rate_index, learning_rate in enumerate(learning_rates):
                        for seed_index, seed in enumerate(seeds):
                            task_id = self._task_id(
                                condition_index,
                                schedule_index,
                                learning_rate_index,
                                seed_index,
                                n_schedules=len(schedules),
                                n_learning_rates=len(learning_rates),
                                n_seeds=len(seeds),
                            )
                            self._write_run(
                                root,
                                condition=condition,
                                schedule=schedule,
                                learning_rate=learning_rate,
                                seed=seed,
                                task_id=task_id,
                            )
            rows = validate_and_collect(
                root,
                conditions=conditions,
                lr_schedules=schedules,
                learning_rates=learning_rates,
                seeds=seeds,
                expected_iterations=2,
                budget=100,
                expected_source_sha="a" * 64,
            )
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(aggregate(rows)), 8)

    def test_complete_hessian_fix_matrix_validates(self) -> None:
        conditions = HESSIAN_FIX_CONDITIONS
        schedules = ("inverse_sqrt",)
        learning_rates = (10.0,)
        seeds = (0,)
        with tempfile.TemporaryDirectory() as root:
            for condition_index, condition in enumerate(conditions):
                self._write_run(
                    root,
                    condition=condition,
                    schedule=schedules[0],
                    learning_rate=learning_rates[0],
                    seed=seeds[0],
                    task_id=condition_index,
                )
            rows = validate_and_collect(
                root,
                conditions=conditions,
                lr_schedules=schedules,
                learning_rates=learning_rates,
                seeds=seeds,
                expected_iterations=2,
                budget=100,
                expected_source_sha="a" * 64,
            )
            self.assertEqual(len(rows), 5)
            self.assertEqual(len(aggregate(rows)), 5)

    def test_paired_contrasts_preserve_seed_differences_and_exact_test(self) -> None:
        rows: list[dict[str, object]] = []
        for condition, values in (
            ("standard_es", (0.0, 0.0, 0.0)),
            ("linearized_implicit_es", (3.0, 2.0, 1.0)),
        ):
            for seed, value in enumerate(values):
                row: dict[str, object] = {
                    "condition": condition,
                    "lr_schedule": "inverse_sqrt",
                    "initial_learning_rate": 30.0,
                    "seed": seed,
                }
                for field in (
                    "training_step_auc",
                    "return_at_training_step_budget",
                    "final_return",
                    "best_return",
                ):
                    row[field] = value
                rows.append(row)

        result = paired_contrasts(rows)
        self.assertEqual(result["difference_direction"], "condition_minus_standard_es")
        self.assertEqual(len(result["cells"]), 1)
        metric = result["cells"][0]["metrics"]["training_step_auc"]
        self.assertEqual(metric["paired_mean_difference"], 2.0)
        self.assertEqual(metric["paired_median_difference"], 2.0)
        self.assertEqual(metric["paired_sample_sd"], 1.0)
        self.assertEqual((metric["wins"], metric["losses"], metric["ties"]), (3, 0, 0))
        self.assertEqual(metric["exact_two_sided_sign_flip_p"], 0.25)
        self.assertEqual(
            metric["differences_by_seed"],
            [
                {"seed": 0, "difference": 3.0},
                {"seed": 1, "difference": 2.0},
                {"seed": 2, "difference": 1.0},
            ],
        )

    def test_fresh_only_runtime_drift_is_rejected_for_every_condition(self) -> None:
        drift_values: dict[str, object] = {
            "n_fresh": 199,
            "n_reused": 1,
            "used_replay": True,
            "replay_weight_mass": 0.01,
            "fresh_weight_mass": 0.99,
            "buffer_size": 1,
            "ess": 199.0,
            "ess_ratio": 0.995,
            "ess_normalized": 0.995,
            "importance_weight_mean": 0.99,
            "importance_weight_min": 0.98,
            "importance_weight_max": 1.01,
        }
        for condition in HESSIAN_FIX_CONDITIONS:
            for field, bad_value in drift_values.items():
                with self.subTest(condition=condition, field=field):
                    with tempfile.TemporaryDirectory() as root:
                        run_dir = self._write_run(
                            root,
                            condition=condition,
                            schedule="inverse_sqrt",
                            learning_rate=10.0,
                            seed=0,
                            task_id=0,
                        )
                        with open(
                            os.path.join(run_dir, "history.json"),
                            "r",
                            encoding="utf-8",
                        ) as stream:
                            history = json.load(stream)
                        history[0][field] = bad_value
                        self._rewrite_history(run_dir, history)
                        with self.assertRaises(ValidationError):
                            self._validate_single_run(root, condition)

    def test_missing_or_aliased_importance_weight_evidence_is_rejected(self) -> None:
        cases = (
            ("importance_weight_min", None, True),
            ("mean_importance_weight", 0.9, False),
            ("max_importance_weight", 1.1, False),
        )
        for field, bad_value, delete_field in cases:
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as root:
                    run_dir = self._write_run(
                        root,
                        condition="standard_es",
                        schedule="inverse_sqrt",
                        learning_rate=10.0,
                        seed=0,
                        task_id=0,
                    )
                    with open(
                        os.path.join(run_dir, "history.json"),
                        "r",
                        encoding="utf-8",
                    ) as stream:
                        history = json.load(stream)
                    if delete_field:
                        del history[0][field]
                    else:
                        history[0][field] = bad_value
                    self._rewrite_history(run_dir, history)
                    with self.assertRaises(ValidationError):
                        self._validate_single_run(root, "standard_es")

    def test_schema_v2_certifies_same_generation_curvature_metadata(self) -> None:
        expected = {
            "concave_diagonal_curvature_es": True,
            "concave_block_curvature_es": True,
            "concave_block_ema_curvature_es": False,
        }
        for condition, expected_value in expected.items():
            with self.subTest(condition=condition):
                with tempfile.TemporaryDirectory() as root:
                    run_dir = self._write_run(
                        root,
                        condition=condition,
                        schedule="inverse_sqrt",
                        learning_rate=10.0,
                        seed=0,
                        task_id=0,
                    )
                    self._validate_single_run(root, condition)
                    with open(
                        os.path.join(run_dir, "history.json"),
                        "r",
                        encoding="utf-8",
                    ) as stream:
                        history = json.load(stream)
                    self.assertIs(
                        history[0]["curvature_same_generation"], expected_value
                    )
                    history[0]["curvature_same_generation"] = not expected_value
                    self._rewrite_history(run_dir, history)
                    with self.assertRaises(ValidationError) as raised:
                        self._validate_single_run(root, condition)
                    self.assertIn(
                        "same-generation metadata is wrong",
                        "\n".join(raised.exception.issues),
                    )

    def test_legacy_concave_records_without_same_generation_metadata_validate(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            condition = "concave_block_ema_curvature_es"
            run_dir = self._write_run(
                root,
                condition=condition,
                schedule="inverse_sqrt",
                learning_rate=10.0,
                seed=0,
                task_id=0,
            )
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            del config["diagnostic_schema_version"]
            del config["resolved_optimizer"]["curvature_same_generation"]
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)
            with open(
                os.path.join(run_dir, "history.json"), "r", encoding="utf-8"
            ) as stream:
                history = json.load(stream)
            for record in history:
                del record["curvature_same_generation"]
            self._rewrite_history(run_dir, history)
            self._validate_single_run(root, condition)

    def test_endpoint_cell_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._write_run(
                root,
                condition="standard_es",
                schedule="inverse_sqrt",
                learning_rate=10.0,
                seed=0,
                task_id=0,
            )
            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            config["condition"] = "endpoint_implicit_es"
            config["algorithm"] = "endpoint_implicit_es"
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)
            with self.assertRaises(ValidationError) as raised:
                validate_and_collect(
                    root,
                    conditions=("standard_es",),
                    lr_schedules=("inverse_sqrt",),
                    learning_rates=(10.0,),
                    seeds=(0,),
                    expected_iterations=2,
                    budget=100,
                    expected_source_sha="a" * 64,
                )
            self.assertIn("unexpected cell", "\n".join(raised.exception.issues))

    def test_wrong_task_mapping_and_schedule_value_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._write_run(
                root,
                condition="standard_es",
                schedule="inverse_linear",
                learning_rate=30.0,
                seed=0,
                task_id=1,
            )
            history_path = os.path.join(run_dir, "history.json")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)
            history[1]["lr"] = 99.0
            with open(history_path, "w", encoding="utf-8") as stream:
                json.dump(history, stream)
            with self.assertRaises(ValidationError) as raised:
                validate_and_collect(
                    root,
                    conditions=("standard_es",),
                    lr_schedules=("inverse_linear",),
                    learning_rates=(30.0,),
                    seeds=(0,),
                    expected_iterations=2,
                    budget=100,
                    expected_source_sha="a" * 64,
                )
            report = "\n".join(raised.exception.issues)
            self.assertIn("task id 1 does not match cell mapping 0", report)
            self.assertIn("deviates from inverse_linear schedule", report)

    def test_history_jsonl_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._write_run(
                root,
                condition="standard_es",
                schedule="inverse_sqrt",
                learning_rate=10.0,
                seed=0,
                task_id=0,
            )
            jsonl_path = os.path.join(run_dir, "history.jsonl")
            with open(jsonl_path, "r", encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream]
            records[-1]["eval_reward"] = -1.0
            with open(jsonl_path, "w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, separators=(",", ":")))
                    stream.write("\n")

            with self.assertRaises(ValidationError) as raised:
                validate_and_collect(
                    root,
                    conditions=("standard_es",),
                    lr_schedules=("inverse_sqrt",),
                    learning_rates=(10.0,),
                    seeds=(0,),
                    expected_iterations=2,
                    budget=100,
                    expected_source_sha="a" * 64,
                )
            self.assertIn(
                "history.jsonl does not match history.json",
                "\n".join(raised.exception.issues),
            )

    def test_nonhex_source_lock_and_unexpected_config_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = self._write_run(
                root,
                condition="standard_es",
                schedule="inverse_sqrt",
                learning_rate=10.0,
                seed=0,
                task_id=0,
            )
            path = os.path.join(run_dir, "config.json")
            with open(path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            config["provenance"]["source_sha256"] = "z" * 64
            config["provenance"]["expected_source_sha256"] = "z" * 64
            config["allow_overwrite"] = True
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)
            with self.assertRaises(ValidationError):
                validate_and_collect(
                    root,
                    conditions=("standard_es",),
                    lr_schedules=("inverse_sqrt",),
                    learning_rates=(10.0,),
                    seeds=(0,),
                    expected_iterations=2,
                    budget=100,
                )


if __name__ == "__main__":
    unittest.main()

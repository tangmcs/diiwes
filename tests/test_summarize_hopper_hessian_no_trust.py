"""Focused tests for the main-branch Hessian/no-trust summarizer."""

from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest

import numpy as np

from core.diiwes import DIIWES
from experiments.train import _history_record

from scripts.summarize_hopper_hessian_no_trust import (
    CONDITIONS,
    EXPECTED_COMMON_CONFIG,
    EXPECTED_DIAG_CONFIG,
    EXPECTED_POPULATION_SIZE,
    INITIAL_LEARNING_RATES,
    LR_SCHEDULES,
    SEEDS,
    HessianSweepValidationError,
    _production_task_id,
    aggregate,
    learning_rate_at_iteration,
    main,
    paired_diag_minus_standard,
    validate_and_collect,
)


SOURCE_SHA = "a" * 64


class SweepFixture:
    @staticmethod
    def config(
        condition: str,
        schedule: str,
        alpha0: float,
        seed: int,
        iterations: int,
    ) -> dict[str, object]:
        config: dict[str, object] = dict(EXPECTED_COMMON_CONFIG)
        config.update(
            {
                "condition": condition,
                "algorithm": (
                    "standard_es"
                    if condition == "standard_es"
                    else "semi_implicit_curvature_es"
                ),
                "seed": seed,
                "learning_rate": alpha0,
                "initial_learning_rate": alpha0,
                "lr_schedule": schedule,
                "n_iterations": iterations,
                "trust_radius": None,
                "source_sha256": SOURCE_SHA,
            }
        )
        if condition == "standard_es":
            config["use_trust_radius_for_standard_es"] = False
        else:
            config.update(EXPECTED_DIAG_CONFIG)
        return config

    @staticmethod
    def history(
        condition: str,
        schedule: str,
        alpha0: float,
        seed: int,
        iterations: int,
    ) -> list[dict[str, object]]:
        offset = 5.0 if condition == "diag_curvature" else 0.0
        records: list[dict[str, object]] = []
        best = float("-inf")
        for iteration in range(iterations):
            evaluation = float(iteration + seed + 1) + offset
            best = max(best, evaluation)
            lr = learning_rate_at_iteration(alpha0, iteration, schedule)
            record: dict[str, object] = {
                "iteration": iteration,
                "lr": lr,
                "learning_rate": lr,
                "eval_reward": evaluation,
                "best_reward": best,
                "mean_fitness": evaluation - 1.0,
                "max_fitness": evaluation + 1.0,
                "grad_norm": 2.0,
                "step_norm": 1.0,
                "n_fresh": 500,
                "n_reused": 0,
                "sigma": 0.02,
                "train_env_steps": 10_000 * (iteration + 1),
                "trust_active": False,
                "trust_scale": 1.0,
            }
            if condition == "diag_curvature":
                record.update(
                    {
                        "curvature_mode": "diag",
                        "curvature_step_mode": "dampen",
                        "curvature_fitness": "raw",
                        "lambda": 0.0,
                        "reuse_fraction": 0.0,
                        "buffer_size": 0,
                        "used_replay": False,
                        "replay_weight_mass": 0.0,
                        "fresh_weight_mass": 1.0,
                        "importance_weight_mean": 1.0,
                        "importance_weight_min": 1.0,
                        "importance_weight_max": 1.0,
                        "w_min": 0.002,
                        "w_max": 0.002,
                        "clip_frac": 0.0,
                        "hessian_pairs": 250,
                        "h_split_correlation": 0.25 + 0.1 * iteration,
                        "h_split_sign_agreement": 0.6,
                        "h_split_relative_disagreement": 1.2,
                        "division_relative_residual": 1e-16,
                        "applied_relative_residual": 0.1 * iteration,
                        "linear_condition_estimate": 2.0 + iteration,
                        "linear_min_abs_diagonal": 1.0,
                        "linear_max_abs_diagonal": 2.0 + iteration,
                        "multiplier_floor_frac": 0.25 if iteration == 2 else 0.0,
                    }
                )
                if iteration > 0:
                    record.update(
                        {
                            "h_temporal_correlation": 0.4,
                            "h_temporal_sign_agreement": 0.7,
                        }
                    )
            records.append(record)
        return records

    @classmethod
    def write_run(
        cls,
        root: str,
        *,
        condition: str,
        schedule: str,
        alpha0: float,
        seed: int,
        iterations: int = 3,
        job_id: str = "1234",
    ) -> str:
        task_id = _production_task_id(condition, schedule, alpha0, seed)
        run_dir = os.path.join(
            root,
            f"{condition}_{schedule}_a{alpha0:g}_seed{seed}_"
            f"job{job_id}_task{task_id}",
        )
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as stream:
            json.dump(cls.config(condition, schedule, alpha0, seed, iterations), stream)
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as stream:
            json.dump(cls.history(condition, schedule, alpha0, seed, iterations), stream)
        return run_dir

    @classmethod
    def write_matrix(
        cls,
        root: str,
        *,
        seeds: tuple[int, ...] = (0, 1),
        iterations: int = 3,
    ) -> None:
        for condition in CONDITIONS:
            for schedule in LR_SCHEDULES:
                for alpha0 in INITIAL_LEARNING_RATES:
                    for seed in seeds:
                        cls.write_run(
                            root,
                            condition=condition,
                            schedule=schedule,
                            alpha0=alpha0,
                            seed=seed,
                            iterations=iterations,
                        )

    @staticmethod
    def read(path: str) -> object:
        with open(path, "r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def write(path: str, value: object) -> None:
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(value, stream)


class HopperHessianNoTrustSummaryTests(unittest.TestCase):
    def _validate_one(self, root: str, condition: str = "diag_curvature"):
        return validate_and_collect(
            root,
            conditions=(condition,),
            lr_schedules=("inverse_sqrt",),
            initial_learning_rates=(10.0,),
            seeds=(0,),
            expected_iterations=3,
            expected_source_sha=SOURCE_SHA,
        )

    def test_production_grid_and_main_diag_defaults_are_locked(self) -> None:
        self.assertEqual(CONDITIONS, ("standard_es", "diag_curvature"))
        self.assertEqual(LR_SCHEDULES, ("inverse_sqrt", "inverse_linear"))
        self.assertEqual(INITIAL_LEARNING_RATES, (10.0, 30.0))
        self.assertEqual(SEEDS, tuple(range(10)))
        self.assertEqual(EXPECTED_POPULATION_SIZE, 500)
        self.assertEqual(
            len(CONDITIONS)
            * len(LR_SCHEDULES)
            * len(INITIAL_LEARNING_RATES)
            * len(SEEDS),
            80,
        )
        self.assertEqual(EXPECTED_DIAG_CONFIG["reuse_fraction"], 0.0)
        self.assertEqual(EXPECTED_DIAG_CONFIG["buffer_size"], 0)
        self.assertEqual(EXPECTED_DIAG_CONFIG["curvature_fitness"], "raw")
        self.assertEqual(EXPECTED_DIAG_CONFIG["implicit_damping"], 0.0)
        self.assertEqual(EXPECTED_DIAG_CONFIG["curvature_beta"], 0.99)
        self.assertEqual(EXPECTED_DIAG_CONFIG["curvature_clip"], 1000.0)
        self.assertEqual(EXPECTED_DIAG_CONFIG["min_step_multiplier"], 0.05)

    def test_complete_matrix_produces_runs_groups_and_paired_differences(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            SweepFixture.write_matrix(root)
            rows = validate_and_collect(
                root,
                seeds=(0, 1),
                expected_iterations=3,
                expected_source_sha=SOURCE_SHA,
            )
            self.assertEqual(len(rows), 16)
            groups = aggregate(rows)
            self.assertEqual(len(groups), 8)
            paired = paired_diag_minus_standard(rows)
            self.assertEqual(paired["difference_direction"], "diag_curvature_minus_standard_es")
            self.assertEqual(len(paired["cells"]), 4)
            for cell in paired["cells"]:
                self.assertEqual(cell["paired_runs"], 2)
                self.assertAlmostEqual(
                    cell["metrics"]["final_eval_return"]["paired_mean_difference"],
                    5.0,
                )
                self.assertAlmostEqual(
                    cell["diag_curvature_mechanism"][
                        "mean_h_temporal_correlation"
                    ]["mean_across_runs"],
                    0.4,
                )

    def test_live_optimizer_serialization_allows_undefined_correlations(self) -> None:
        """Degenerate Hessians serialize without Pearson-correlation keys."""

        with tempfile.TemporaryDirectory() as root:
            run_dir = SweepFixture.write_run(
                root,
                condition="diag_curvature",
                schedule="inverse_sqrt",
                alpha0=10.0,
                seed=0,
            )
            optimizer = DIIWES(
                num_params=2,
                population_size=500,
                learning_rate=10.0,
                noise_std=0.02,
                buffer_size=0,
                reuse_fraction=0.0,
                implicit_damping=0.0,
                trust_radius=None,
                seed=0,
            )
            params = np.zeros(2, dtype=np.float64)
            optimizer.current_params = params.copy()
            history: list[dict[str, object]] = []
            for iteration in range(3):
                optimizer.learning_rate = learning_rate_at_iteration(
                    10.0, iteration, "inverse_sqrt"
                )
                noise, ask_info = optimizer.ask()
                fresh_count = int(np.sum(~np.asarray(ask_info["is_reused"], dtype=bool)))
                fresh_fitness = np.zeros(fresh_count, dtype=np.float64)
                params, info = optimizer.tell(
                    params, noise, fresh_fitness, ask_info
                )
                optimizer.current_params = params.copy()
                record = _history_record(
                    iteration,
                    float(iteration + 1),
                    float(iteration + 1),
                    0.0,
                    0.0,
                    fresh_fitness,
                    info,
                    optimizer.learning_rate,
                    0.01,
                    100 * (iteration + 1),
                    100,
                    10 * (iteration + 1),
                    10,
                )
                history.append(record)
            self.assertNotIn("h_split_correlation", history[0])
            self.assertNotIn("h_temporal_correlation", history[1])
            SweepFixture.write(os.path.join(run_dir, "history.json"), history)

            rows = self._validate_one(root)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["mean_h_split_correlation"])
            self.assertEqual(rows[0]["h_split_correlation_available_fraction"], 0.0)
            self.assertIsNone(rows[0]["mean_h_temporal_correlation"])
            self.assertEqual(rows[0]["h_temporal_correlation_available_fraction"], 0.0)

    def test_rejects_trust_activation_and_nonunit_scale(self) -> None:
        for field, value, expected_text in (
            ("trust_active", True, "trust region activated"),
            ("trust_scale", 0.5, "trust_scale is not one"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                path = os.path.join(run_dir, "history.json")
                history = SweepFixture.read(path)
                history[1][field] = value
                SweepFixture.write(path, history)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(expected_text in issue for issue in caught.exception.issues)
                )

    def test_rejects_any_replay_or_nonuniform_fresh_weight(self) -> None:
        for field, value in (
            ("used_replay", True),
            ("replay_weight_mass", 0.1),
            ("w_max", 0.01),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                path = os.path.join(run_dir, "history.json")
                history = SweepFixture.read(path)
                history[1][field] = value
                SweepFixture.write(path, history)
                with self.assertRaises(HessianSweepValidationError):
                    self._validate_one(root)

    def test_rejects_schedule_default_and_source_deviations(self) -> None:
        mutations = (
            ("history", lambda history: history[1].__setitem__("lr", 10.0), "lr deviates"),
            ("config", lambda config: config.__setitem__("reuse_fraction", 0.2), "reuse_fraction"),
            ("config", lambda config: config.__setitem__("implicit_damping", 0.1), "implicit_damping"),
            ("config", lambda config: config.__setitem__("source_sha256", "b" * 64), "source digest"),
            ("config", lambda config: config.pop("trust_radius"), "explicit null"),
        )
        for artifact, mutate, expected_text in mutations:
            with self.subTest(artifact=artifact, text=expected_text), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                path = os.path.join(run_dir, f"{artifact}.json")
                value = SweepFixture.read(path)
                mutate(value)
                SweepFixture.write(path, value)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(expected_text in issue for issue in caught.exception.issues),
                    caught.exception.issues,
                )

    def test_rejects_incomplete_and_nonfinite_history(self) -> None:
        for mutation, expected_text in (
            (lambda history: history.pop(), "incomplete history"),
            (
                lambda history: history[1].__setitem__("eval_reward", float("nan")),
                "non-finite history",
            ),
        ):
            with self.subTest(expected_text=expected_text), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                path = os.path.join(run_dir, "history.json")
                history = SweepFixture.read(path)
                mutation(history)
                SweepFixture.write(path, history)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(expected_text in issue for issue in caught.exception.issues)
                )

    def test_cli_writes_all_three_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            SweepFixture.write_matrix(root)
            run_output = os.path.join(root, "out", "runs.csv")
            group_output = os.path.join(root, "out", "groups.csv")
            paired_output = os.path.join(root, "out", "paired.json")
            exit_code = main(
                [
                    root,
                    "--expected-source-sha",
                    SOURCE_SHA,
                    "--seeds",
                    "0",
                    "1",
                    "--expected-iterations",
                    "3",
                    "--run-output",
                    run_output,
                    "--group-output",
                    group_output,
                    "--paired-output",
                    paired_output,
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.isfile(run_output))
            self.assertTrue(os.path.isfile(group_output))
            self.assertTrue(os.path.isfile(paired_output))
            with open(run_output, "r", newline="", encoding="utf-8") as stream:
                run_rows = list(csv.DictReader(stream))
            with open(group_output, "r", newline="", encoding="utf-8") as stream:
                group_rows = list(csv.DictReader(stream))
            paired = SweepFixture.read(paired_output)
            self.assertEqual(len(run_rows), 16)
            self.assertEqual(len(group_rows), 8)
            self.assertIn("max_applied_relative_residual", run_rows[0])
            self.assertEqual(paired["source_sha256"], SOURCE_SHA)
            self.assertEqual(paired["environment"], "Hopper-v5")
            self.assertEqual(paired["iterations"], 3)


if __name__ == "__main__":
    unittest.main()

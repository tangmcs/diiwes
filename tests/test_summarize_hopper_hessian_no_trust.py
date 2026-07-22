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

from scripts.analysis.summarize_hopper_hessian_no_trust import (
    CONDITIONS,
    EXPECTED_COMMON_CONFIG,
    EXPECTED_DIAG_CONFIG,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_POPULATION_SIZE,
    HESSIAN_FOR_STEP_HISTORY_FILENAME,
    INITIAL_LEARNING_RATES,
    LR_SCHEDULES,
    SEEDS,
    STEP_MULTIPLIER_HISTORY_FILENAME,
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
    def coordinate_histories(
        schedule: str,
        alpha0: float,
        iterations: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        hessian_history = np.zeros(
            (iterations, EXPECTED_PARAMETER_COUNT), dtype=np.float64
        )
        multiplier_history = np.ones_like(hessian_history)
        for iteration in range(iterations):
            preclip = np.zeros(EXPECTED_PARAMETER_COUNT, dtype=np.float64)
            preclip[:2000] = 0.1 * (iteration + 1)
            if iteration == 2:
                excess = np.full(10, 150.0 / 9.0, dtype=np.float64)
                excess[0] = 50.0
                preclip[:10] = 1000.0 + excess
            hessian_history[iteration] = -preclip
            lr = learning_rate_at_iteration(alpha0, iteration, schedule)
            curvature = np.clip(preclip, 0.0, 1000.0)
            raw_multiplier = 1.0 / (1.0 + lr * curvature)
            multiplier_history[iteration] = np.clip(raw_multiplier, 0.05, 1.0)
        return hessian_history, multiplier_history

    @staticmethod
    def coordinate_diagnostics(
        hessian_for_step: np.ndarray,
        learning_rate: float,
    ) -> dict[str, object]:
        preclip = np.maximum(-hessian_for_step, 0.0)
        active_mask = preclip > 0.0
        clip_mask = preclip > 1000.0
        curvature = np.clip(preclip, 0.0, 1000.0)
        linear_diagonal = 1.0 + learning_rate * curvature
        raw_multiplier = 1.0 / linear_diagonal
        multiplier = np.clip(raw_multiplier, 0.05, 1.0)

        clip_count = int(np.count_nonzero(clip_mask))
        if clip_count:
            excess = preclip[clip_mask] - 1000.0
            excess_mean = float(np.mean(excess))
            excess_max = float(np.max(excess))
        else:
            excess_mean = 0.0
            excess_max = 0.0

        floor_mask = raw_multiplier < 0.05
        floor_count = int(np.count_nonzero(floor_mask))
        if floor_count:
            deficit = 0.05 - raw_multiplier[floor_mask]
            deficit_mean = float(np.mean(deficit))
            deficit_max = float(np.max(deficit))
        else:
            deficit_mean = 0.0
            deficit_max = 0.0

        ceiling_mask = raw_multiplier > 1.0
        ceiling_count = int(np.count_nonzero(ceiling_mask))
        if ceiling_count:
            ceiling_excess = raw_multiplier[ceiling_mask] - 1.0
            ceiling_excess_mean = float(np.mean(ceiling_excess))
            ceiling_excess_max = float(np.max(ceiling_excess))
        else:
            ceiling_excess_mean = 0.0
            ceiling_excess_max = 0.0

        active_count = int(np.count_nonzero(active_mask))
        multiplier_mean = float(np.mean(multiplier))
        multiplier_std = float(np.std(multiplier))
        return {
            "h_step_mean": float(np.mean(hessian_for_step)),
            "h_step_min": float(np.min(hessian_for_step)),
            "h_step_max": float(np.max(hessian_for_step)),
            "linear_condition_estimate": float(
                np.max(linear_diagonal) / np.min(linear_diagonal)
            ),
            "linear_min_abs_diagonal": float(np.min(linear_diagonal)),
            "linear_max_abs_diagonal": float(np.max(linear_diagonal)),
            "curvature_coordinate_count": EXPECTED_PARAMETER_COUNT,
            "curvature_active_count": active_count,
            "curvature_active_frac": active_count / EXPECTED_PARAMETER_COUNT,
            "curvature_preclip_mean": float(np.mean(preclip)),
            "curvature_preclip_max": float(np.max(preclip)),
            "curvature_clip_count": clip_count,
            "curvature_clip_frac": clip_count / EXPECTED_PARAMETER_COUNT,
            "curvature_clip_active": clip_count > 0,
            "curvature_clip_excess_mean": excess_mean,
            "curvature_clip_excess_max": excess_max,
            "curv_mean": float(np.mean(curvature)),
            "curv_min": float(np.min(curvature)),
            "curv_max": float(np.max(curvature)),
            "multiplier_coordinate_count": EXPECTED_PARAMETER_COUNT,
            "raw_step_multiplier_min": float(np.min(raw_multiplier)),
            "raw_step_multiplier_max": float(np.max(raw_multiplier)),
            "multiplier_clipping_diagnostics_exact": True,
            "multiplier_floor_clip_count": floor_count,
            "multiplier_floor_clip_frac": floor_count / EXPECTED_PARAMETER_COUNT,
            "multiplier_floor_clip_active": floor_count > 0,
            "multiplier_floor_clip_deficit_mean": deficit_mean,
            "multiplier_floor_clip_deficit_max": deficit_max,
            "multiplier_ceiling_clip_count": ceiling_count,
            "multiplier_ceiling_clip_frac": ceiling_count / EXPECTED_PARAMETER_COUNT,
            "multiplier_ceiling_clip_active": ceiling_count > 0,
            "multiplier_ceiling_clip_excess_mean": ceiling_excess_mean,
            "multiplier_ceiling_clip_excess_max": ceiling_excess_max,
            "step_multiplier_min": float(np.min(multiplier)),
            "step_multiplier_max": float(np.max(multiplier)),
            "step_multiplier_mean": multiplier_mean,
            "step_multiplier_std": multiplier_std,
            "step_multiplier_cv": multiplier_std / (multiplier_mean + 1e-12),
            "hessian_shrinkage_median": float(np.median(multiplier)),
            "hessian_shrinkage_p90": float(np.percentile(multiplier, 90.0)),
            "hessian_shrinkage_max": float(np.max(multiplier)),
            "multiplier_floor_frac": float(np.mean(multiplier <= 0.05 + 1e-12)),
        }

    @classmethod
    def history(
        cls,
        condition: str,
        schedule: str,
        alpha0: float,
        seed: int,
        iterations: int,
        *,
        hessian_history: np.ndarray | None = None,
    ) -> list[dict[str, object]]:
        if condition == "diag_curvature" and hessian_history is None:
            hessian_history, _ = cls.coordinate_histories(
                schedule, alpha0, iterations
            )
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
                assert hessian_history is not None
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
                    }
                )
                record.update(
                    cls.coordinate_diagnostics(hessian_history[iteration], lr)
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
        hessian_history: np.ndarray | None = None
        multiplier_history: np.ndarray | None = None
        if condition == "diag_curvature":
            hessian_history, multiplier_history = cls.coordinate_histories(
                schedule, alpha0, iterations
            )
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as stream:
            json.dump(cls.config(condition, schedule, alpha0, seed, iterations), stream)
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as stream:
            json.dump(
                cls.history(
                    condition,
                    schedule,
                    alpha0,
                    seed,
                    iterations,
                    hessian_history=hessian_history,
                ),
                stream,
            )
        if hessian_history is not None and multiplier_history is not None:
            np.save(
                os.path.join(run_dir, HESSIAN_FOR_STEP_HISTORY_FILENAME),
                hessian_history,
            )
            np.save(
                os.path.join(run_dir, STEP_MULTIPLIER_HISTORY_FILENAME),
                multiplier_history,
            )
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
        self.assertEqual(EXPECTED_PARAMETER_COUNT, 5123)
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

    def test_coordinate_artifacts_are_required_only_for_diagonal_runs(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            SweepFixture.write_run(
                root,
                condition="standard_es",
                schedule="inverse_sqrt",
                alpha0=10.0,
                seed=0,
            )
            rows = self._validate_one(root, condition="standard_es")
            self.assertEqual(len(rows), 1)

        for filename in (
            HESSIAN_FOR_STEP_HISTORY_FILENAME,
            STEP_MULTIPLIER_HISTORY_FILENAME,
        ):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                os.unlink(os.path.join(run_dir, filename))
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(
                        filename in issue and "coordinate artifact is missing" in issue
                        for issue in caught.exception.issues
                    ),
                    caught.exception.issues,
                )

    def test_rejects_malformed_coordinate_artifacts(self) -> None:
        cases = (
            (HESSIAN_FOR_STEP_HISTORY_FILENAME, "dtype", "expected float64"),
            (STEP_MULTIPLIER_HISTORY_FILENAME, "shape", "expected (3, 5123)"),
            (
                HESSIAN_FOR_STEP_HISTORY_FILENAME,
                "nonfinite",
                "contains non-finite values",
            ),
        )
        for filename, mutation, expected_text in cases:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                run_dir = SweepFixture.write_run(
                    root,
                    condition="diag_curvature",
                    schedule="inverse_sqrt",
                    alpha0=10.0,
                    seed=0,
                )
                path = os.path.join(run_dir, filename)
                values = np.load(path)
                if mutation == "dtype":
                    values = values.astype(np.float32)
                elif mutation == "shape":
                    values = values[:, :-1]
                else:
                    values[1, 10] = np.nan
                np.save(path, values)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(expected_text in issue for issue in caught.exception.issues),
                    caught.exception.issues,
                )

    def test_rejects_any_saved_multiplier_coordinate_drift(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = SweepFixture.write_run(
                root,
                condition="diag_curvature",
                schedule="inverse_sqrt",
                alpha0=10.0,
                seed=0,
            )
            path = os.path.join(run_dir, STEP_MULTIPLIER_HISTORY_FILENAME)
            multipliers = np.load(path)
            multipliers[2, 0] = np.nextafter(multipliers[2, 0], 1.0)
            np.save(path, multipliers)
            with self.assertRaises(HessianSweepValidationError) as caught:
                self._validate_one(root)
            self.assertTrue(
                any(
                    "differs from the exactly reconstructed applied multiplier"
                    in issue
                    for issue in caught.exception.issues
                ),
                caught.exception.issues,
            )

    def test_coordinate_artifacts_crosscheck_scalar_history_diagnostics(self) -> None:
        mutations = (
            ("h_step_max", -999.0),
            ("curvature_active_count", 1999),
            ("curvature_clip_count", 9),
            ("curvature_clip_frac", 9.0 / EXPECTED_PARAMETER_COUNT),
            ("curvature_preclip_max", 1049.0),
            ("curvature_clip_excess_mean", 19.0),
            ("raw_step_multiplier_min", 0.001),
            ("multiplier_floor_clip_count", 9),
            ("multiplier_floor_clip_frac", 9.0 / EXPECTED_PARAMETER_COUNT),
            ("multiplier_floor_clip_deficit_mean", 0.01),
            ("step_multiplier_min", 0.051),
            ("step_multiplier_mean", 0.5),
        )
        for field, value in mutations:
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
                history[2][field] = value
                SweepFixture.write(path, history)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(
                        f".{field}=" in issue
                        and "reconstructed coordinate artifacts" in issue
                        for issue in caught.exception.issues
                    ),
                    caught.exception.issues,
                )

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
            diag_row = next(
                row
                for row in rows
                if row["condition"] == "diag_curvature"
                and row["lr_schedule"] == "inverse_sqrt"
                and row["initial_learning_rate"] == 10.0
                and row["seed"] == 0
            )
            self.assertEqual(diag_row["curvature_coordinate_count"], 5123)
            self.assertEqual(diag_row["total_curvature_clip_count"], 10)
            self.assertAlmostEqual(diag_row["mean_curvature_clip_count"], 10.0 / 3.0)
            self.assertEqual(diag_row["max_curvature_clip_count"], 10)
            self.assertAlmostEqual(
                diag_row["curvature_clip_active_iteration_fraction"], 1.0 / 3.0
            )
            self.assertAlmostEqual(
                diag_row["mean_curvature_clip_excess_per_clipped_coordinate"],
                20.0,
            )
            self.assertEqual(diag_row["total_multiplier_floor_clip_count"], 10)
            self.assertAlmostEqual(
                diag_row["mean_multiplier_floor_clip_frac"], 10.0 / 5123.0 / 3.0
            )
            self.assertAlmostEqual(
                diag_row["multiplier_floor_clip_active_iteration_fraction"],
                1.0 / 3.0,
            )
            expected_floor_deficit = 0.05 - 1.0 / (
                1.0
                + learning_rate_at_iteration(10.0, 2, "inverse_sqrt") * 1000.0
            )
            self.assertAlmostEqual(
                diag_row[
                    "mean_multiplier_floor_clip_deficit_per_clipped_coordinate"
                ],
                expected_floor_deficit,
            )
            self.assertEqual(diag_row["total_multiplier_ceiling_clip_count"], 0)
            self.assertEqual(
                diag_row["multiplier_ceiling_clip_active_iteration_fraction"],
                0.0,
            )
            diag_group = next(
                group
                for group in groups
                if group["condition"] == "diag_curvature"
                and group["lr_schedule"] == "inverse_sqrt"
                and group["initial_learning_rate"] == 10.0
            )
            self.assertEqual(diag_group["total_curvature_clip_count_mean"], 10.0)
            self.assertEqual(
                diag_group["total_multiplier_floor_clip_count_mean"], 10.0
            )
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
                self.assertEqual(
                    cell["diag_curvature_mechanism"][
                        "total_curvature_clip_count"
                    ]["mean_across_runs"],
                    10.0,
                )
                self.assertEqual(
                    cell["diag_curvature_mechanism"][
                        "total_multiplier_floor_clip_count"
                    ]["mean_across_runs"],
                    10.0,
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
                num_params=5123,
                population_size=500,
                learning_rate=10.0,
                noise_std=0.02,
                buffer_size=0,
                reuse_fraction=0.0,
                implicit_damping=0.0,
                trust_radius=None,
                seed=0,
            )
            params = np.zeros(5123, dtype=np.float64)
            optimizer.current_params = params.copy()
            history: list[dict[str, object]] = []
            hessian_history: list[np.ndarray] = []
            multiplier_history: list[np.ndarray] = []
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
                hessian_for_step = optimizer.hessian_for_step_vector
                step_multiplier = optimizer.step_multiplier_vector
                self.assertIsNotNone(hessian_for_step)
                self.assertIsNotNone(step_multiplier)
                hessian_history.append(hessian_for_step)
                multiplier_history.append(step_multiplier)
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
            np.save(
                os.path.join(run_dir, HESSIAN_FOR_STEP_HISTORY_FILENAME),
                np.asarray(hessian_history, dtype=np.float64),
            )
            np.save(
                os.path.join(run_dir, STEP_MULTIPLIER_HISTORY_FILENAME),
                np.asarray(multiplier_history, dtype=np.float64),
            )

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

    def test_strict_clip_masks_accept_values_exactly_at_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            run_dir = SweepFixture.write_run(
                root,
                condition="diag_curvature",
                schedule="inverse_sqrt",
                alpha0=10.0,
                seed=0,
            )
            path = os.path.join(run_dir, "history.json")
            history = SweepFixture.read(path)
            hessian_path = os.path.join(
                run_dir, HESSIAN_FOR_STEP_HISTORY_FILENAME
            )
            multiplier_path = os.path.join(
                run_dir, STEP_MULTIPLIER_HISTORY_FILENAME
            )
            hessian_history = np.load(hessian_path)
            multiplier_history = np.load(multiplier_path)
            hessian_history[0, 0] = -1000.0
            floor_boundary_curvature = 19.0 / float(history[1]["lr"])
            hessian_history[1, 0] = -floor_boundary_curvature
            for iteration in (0, 1):
                lr = float(history[iteration]["lr"])
                history[iteration].update(
                    SweepFixture.coordinate_diagnostics(
                        hessian_history[iteration], lr
                    )
                )
                curvature = np.clip(
                    np.maximum(-hessian_history[iteration], 0.0),
                    0.0,
                    1000.0,
                )
                multiplier_history[iteration] = np.clip(
                    1.0 / (1.0 + lr * curvature), 0.05, 1.0
                )
            SweepFixture.write(path, history)
            np.save(hessian_path, hessian_history)
            np.save(multiplier_path, multiplier_history)

            rows = self._validate_one(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["total_curvature_clip_count"], 10)
            self.assertEqual(rows[0]["total_multiplier_floor_clip_count"], 11)

    def test_rejects_inconsistent_exact_clipping_diagnostics(self) -> None:
        mutations = (
            ("curvature_active_frac", 0.3, "curvature_active_frac disagrees"),
            ("curvature_clip_frac", 0.2, "curvature_clip_frac disagrees"),
            ("curvature_clip_active", False, "curvature_clip_active disagrees"),
            ("curvature_clip_excess_max", 40.0, "maximum/excess arithmetic"),
            (
                "multiplier_coordinate_count",
                99,
                "curvature/multiplier coordinate counts disagree",
            ),
            (
                "multiplier_floor_clip_frac",
                0.1,
                "multiplier_floor_clip_frac disagrees",
            ),
            (
                "multiplier_floor_clip_active",
                False,
                "multiplier_floor_clip_active disagrees",
            ),
            (
                "multiplier_floor_clip_deficit_max",
                0.02,
                "minimum/deficit arithmetic",
            ),
            (
                "multiplier_ceiling_clip_active",
                True,
                "must be false in the locked protocol",
            ),
            ("multiplier_floor_frac", 0.001, "legacy at-floor fraction"),
        )
        for field, value, expected_text in mutations:
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
                history[2][field] = value
                SweepFixture.write(path, history)
                with self.assertRaises(HessianSweepValidationError) as caught:
                    self._validate_one(root)
                self.assertTrue(
                    any(expected_text in issue for issue in caught.exception.issues),
                    caught.exception.issues,
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

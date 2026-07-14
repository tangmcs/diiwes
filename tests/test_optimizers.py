#!/usr/bin/env python3
"""Deterministic regression tests for the ES optimizers and trainer helpers."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import numpy as np

from core import (
    AdamES,
    ClipUpES,
    ConcaveCurvatureES,
    DIIWES,
    EndpointImplicitES,
    LinearizedImplicitES,
    LOPOGradientES,
    MomentumES,
    SNES,
    StandardES,
)
from core.standard_es import (
    centered_ranks,
    centered_ranks_from_reference,
    snes_utilities,
)
from core.policies import MLPPolicy
from experiments.lagged_subspace_study_lock import (
    DEPENDENCY_BUNDLE_PATH,
    LAUNCHER_BUNDLE_PATH,
    current_lagged_subspace_study_sha256,
)
from experiments.train import (
    CHECKPOINT_CAPTURE_MANIFEST,
    CHECKPOINT_TRAINING_CONFIG_ARTIFACT,
    LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL,
    _apply_optimizer_cli_overrides,
    _array_sha256,
    _calibration_rollout_seed,
    _center_rollout_seed,
    _condition_config,
    _evaluation_rollout_seed,
    _history_record,
    _labeled_arrays_sha256,
    _learning_rate_at,
    _resolved_optimizer_config,
    _resolve_checkpoint_capture_config,
    _resolve_training_env_step_budget,
    _source_digest,
    _sha256_file,
    _training_rollout_seed,
    _validate_lagged_subspace_checkpoint_protocol,
    _validate_no_replay_protocol,
    make_optimizer,
    train,
)
from experiments.train import main as train_main


class CenteredRanksTests(unittest.TestCase):
    def test_all_ties_are_zero(self) -> None:
        self.assertTrue(np.allclose(centered_ranks(np.full(6, -21.0)), 0.0))

    def test_ties_share_rank(self) -> None:
        ranks = centered_ranks(np.asarray([1.0, 1.0, 3.0, 3.0]))
        self.assertAlmostEqual(ranks[0], ranks[1])
        self.assertAlmostEqual(ranks[2], ranks[3])
        self.assertLess(ranks[0], ranks[2])
        self.assertAlmostEqual(float(np.mean(ranks)), 0.0)

    def test_reference_ranks_equal_centered_ranks_on_the_reference_batch(self) -> None:
        values = np.asarray([3.0, 1.0, 7.0, 1.0, 3.0])
        actual = centered_ranks_from_reference(values, values)
        self.assertTrue(np.allclose(actual, centered_ranks(values)))


class SNESUtilityTests(unittest.TestCase):
    def test_canonical_log_rank_weights_match_closed_form(self) -> None:
        fitness = np.asarray([40.0, 10.0, 30.0, 20.0])
        positive = np.maximum(
            0.0,
            np.log(3.0) - np.log(np.arange(1, 5, dtype=np.float64)),
        )
        ranked = positive / np.sum(positive) - 0.25
        expected = np.asarray([ranked[0], ranked[3], ranked[1], ranked[2]])
        actual = snes_utilities(fitness)
        self.assertTrue(np.allclose(actual, expected))
        self.assertAlmostEqual(float(np.sum(actual)), 0.0)

    def test_exact_ties_average_the_occupied_rank_utilities(self) -> None:
        utilities = snes_utilities(np.asarray([7.0, 7.0, -2.0, -2.0]))
        self.assertAlmostEqual(utilities[0], utilities[1])
        self.assertAlmostEqual(utilities[2], utilities[3])
        self.assertGreater(utilities[0], utilities[2])
        self.assertAlmostEqual(float(np.sum(utilities)), 0.0)
        self.assertTrue(np.array_equal(snes_utilities(np.ones(6)), np.zeros(6)))


class ScheduleTests(unittest.TestCase):
    def test_inverse_sqrt_matches_requested_sequence(self) -> None:
        actual = [_learning_rate_at(30.0, i, "inverse_sqrt") for i in range(100)]
        expected = [30.0 / np.sqrt(i + 1.0) for i in range(100)]
        self.assertTrue(np.allclose(actual, expected))
        self.assertTrue(np.all(np.diff(actual) < 0.0))

    def test_inverse_linear_matches_requested_sequence(self) -> None:
        actual = [_learning_rate_at(30.0, i, "inverse_linear") for i in range(100)]
        expected = [30.0 / (i + 1.0) for i in range(100)]
        self.assertTrue(np.allclose(actual, expected))
        self.assertTrue(np.all(np.diff(actual) < 0.0))

    def test_exponential_schedule_is_backward_compatible(self) -> None:
        self.assertAlmostEqual(_learning_rate_at(2.0, 3, "exponential", exponential_decay=0.5), 0.25)

    def test_common_training_seed_is_shared_only_with_antithetic_partner(self) -> None:
        self.assertEqual(
            _training_rollout_seed(7, 2, 0, True, 10, True),
            _training_rollout_seed(7, 2, 5, True, 10, True),
        )
        self.assertNotEqual(
            _training_rollout_seed(7, 2, 0, True, 10, True),
            _training_rollout_seed(7, 2, 1, True, 10, True),
        )
        self.assertNotEqual(
            _training_rollout_seed(7, 2, 0, True, 10, True),
            _training_rollout_seed(7, 3, 0, True, 10, True),
        )

    def test_nonantithetic_samples_do_not_share_common_seeds(self) -> None:
        seeds = [_training_rollout_seed(7, 2, i, True, 10, False) for i in range(10)]
        self.assertEqual(len(set(seeds)), 10)

    def test_planned_seed_grid_is_unique_across_streams(self) -> None:
        train_seeds = {
            _training_rollout_seed(run_seed, iteration, pair_index, True, 200, True)
            for run_seed in range(5)
            for iteration in range(500)
            for pair_index in range(100)
        }
        eval_seeds = {
            _evaluation_rollout_seed(run_seed, eval_index)
            for run_seed in range(5)
            for eval_index in range(3)
        }
        center_seeds = {
            _center_rollout_seed(run_seed, iteration)
            for run_seed in range(5)
            for iteration in range(500)
        }
        calibration_seeds = {
            _calibration_rollout_seed(run_seed, calibration_index)
            for run_seed in range(5)
            for calibration_index in range(3)
        }

        self.assertEqual(len(train_seeds), 5 * 500 * 100)
        self.assertEqual(len(eval_seeds), 5 * 3)
        self.assertEqual(len(center_seeds), 5 * 500)
        self.assertEqual(len(calibration_seeds), 5 * 3)
        self.assertTrue(train_seeds.isdisjoint(eval_seeds))
        self.assertTrue(train_seeds.isdisjoint(center_seeds))
        self.assertTrue(eval_seeds.isdisjoint(center_seeds))
        self.assertTrue(calibration_seeds.isdisjoint(train_seeds))
        self.assertTrue(calibration_seeds.isdisjoint(eval_seeds))
        self.assertTrue(calibration_seeds.isdisjoint(center_seeds))

    def test_evaluation_episodes_use_distinct_fixed_seeds(self) -> None:
        seeds = [_evaluation_rollout_seed(7, i) for i in range(3)]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(seeds, [_evaluation_rollout_seed(7, i) for i in range(3)])

    def test_source_digest_is_independent_of_config_checkout_path(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_config = os.path.join(first, "hopper.yaml")
            second_config = os.path.join(second, "renamed.yaml")
            for path in (first_config, second_config):
                with open(path, "w", encoding="utf-8") as stream:
                    stream.write("env_name: Hopper-v5\n")
            self.assertEqual(_source_digest(first_config), _source_digest(second_config))


class HistorySerializationTests(unittest.TestCase):
    @staticmethod
    def _record(info: dict) -> dict:
        return _history_record(
            iteration=0,
            eval_reward=1.0,
            best_reward=1.0,
            best_fitness_iter=1.0,
            best_fitness_so_far=1.0,
            fresh_fitness=np.asarray([1.0, 2.0]),
            info=info,
            learning_rate=0.1,
            elapsed=0.0,
            train_env_steps=10,
            train_env_steps_iter=10,
            eval_env_steps=2,
            eval_env_steps_iter=2,
            initial_eval_reward=0.0,
            initial_eval_env_steps=2,
            normalization_calibration_env_steps=0,
        )

    def test_short_whitelisted_curvature_vectors_are_persisted(self) -> None:
        record = self._record(
            {
                "curvature_raw_components": np.asarray([1.0, -2.0, 3.0]),
                "h_split_first_components": [0.25, -0.5],
                "denominator_components": np.arange(64, dtype=np.float64),
                "non_whitelisted_vector": [1.0, 2.0],
            }
        )
        self.assertEqual(record["curvature_raw_components"], [1.0, -2.0, 3.0])
        self.assertEqual(record["curvature_raw_components_length"], 3)
        self.assertFalse(record["curvature_raw_components_omitted"])
        self.assertEqual(
            record["curvature_raw_components_serialization"], "persisted"
        )
        self.assertEqual(record["h_split_first_components"], [0.25, -0.5])
        self.assertEqual(len(record["denominator_components"]), 64)
        self.assertFalse(record["denominator_components_omitted"])
        self.assertNotIn("non_whitelisted_vector", record)

    def test_oversized_diagonal_vectors_are_explicitly_omitted(self) -> None:
        diagonal = np.arange(5123, dtype=np.float64)
        record = self._record(
            {
                "curvature_raw_components": diagonal,
                "h_split_first_components": diagonal,
            }
        )
        for field in (
            "curvature_raw_components",
            "h_split_first_components",
        ):
            self.assertNotIn(field, record)
            self.assertEqual(record[f"{field}_length"], 5123)
            self.assertTrue(record[f"{field}_omitted"])
            self.assertEqual(
                record[f"{field}_serialization"],
                "omitted_length_exceeds_64",
            )
        self.assertLess(len(json.dumps(record)), 5000)


class NoReplayCliTests(unittest.TestCase):
    def test_cli_revalidates_reuse_fraction_after_overrides(self) -> None:
        base_config = {
            "replay_enabled": False,
            "reuse_fraction": 0.0,
            "buffer_size": 0,
        }
        common_argv = [
            "train.py",
            "--config",
            "unused.yaml",
            "--condition",
            "standard_es",
            "--workers",
            "1",
            "--output",
            "/tmp/unused_fresh_only_cli_test",
        ]
        with patch(
            "experiments.train.load_config", return_value=base_config
        ), patch("experiments.train.train") as mocked_train, patch.object(
            sys,
            "argv",
            common_argv + ["--reuse-fraction", "0.2"],
        ):
            with self.assertRaisesRegex(ValueError, "reuse_fraction must be zero"):
                train_main()
            mocked_train.assert_not_called()

        with patch(
            "experiments.train.load_config", return_value=base_config
        ), patch("experiments.train.train") as mocked_train, patch.object(
            sys,
            "argv",
            common_argv + ["--reuse-fraction", "0"],
        ):
            train_main()
            mocked_train.assert_called_once()
            self.assertEqual(mocked_train.call_args.args[0]["reuse_fraction"], 0.0)


class AdaptiveOptimizerCliTests(unittest.TestCase):
    @staticmethod
    def _argv(condition: str, *extra: str) -> list[str]:
        return [
            "train.py",
            "--config",
            "unused.yaml",
            "--condition",
            condition,
            "--workers",
            "1",
            "--output",
            "/tmp/unused_adaptive_cli_test",
            *extra,
        ]

    @staticmethod
    def _base_config() -> dict[str, object]:
        return {
            "replay_enabled": False,
            "reuse_fraction": 0.0,
            "buffer_size": 0,
        }

    def test_cli_applies_each_optimizer_namespace_and_training_budget(self) -> None:
        cases = (
            (
                "momentum_es",
                ("--momentum-beta", "0.7"),
                {"momentum_beta": 0.7},
            ),
            (
                "adam_es",
                (
                    "--adam-beta1",
                    "0.8",
                    "--adam-beta2",
                    "0.95",
                    "--adam-epsilon",
                    "1e-6",
                ),
                {
                    "adam_beta1": 0.8,
                    "adam_beta2": 0.95,
                    "adam_epsilon": 1e-6,
                },
            ),
            (
                "clipup_es",
                (
                    "--clipup-momentum",
                    "0.6",
                    "--clipup-max-speed",
                    "0.25",
                    "--training-env-step-budget",
                    "123",
                ),
                {
                    "clipup_momentum": 0.6,
                    "clipup_max_speed": 0.25,
                    "training_env_step_budget": 123,
                },
            ),
            (
                "snes",
                ("--snes-sigma-learning-rate", "0.12"),
                {"snes_sigma_learning_rate": 0.12},
            ),
        )
        for condition, flags, expected in cases:
            with self.subTest(condition=condition), patch(
                "experiments.train.load_config",
                return_value=self._base_config(),
            ), patch("experiments.train.train") as mocked_train, patch.object(
                sys,
                "argv",
                self._argv(condition, *flags),
            ):
                train_main()
                passed_config = mocked_train.call_args.args[0]
                for key, value in expected.items():
                    self.assertEqual(passed_config[key], value)

    def test_cli_rejects_optimizer_flag_for_wrong_condition(self) -> None:
        with patch(
            "experiments.train.load_config", return_value=self._base_config()
        ), patch("experiments.train.train") as mocked_train, patch.object(
            sys,
            "argv",
            self._argv("standard_es", "--adam-beta1", "0.8"),
        ), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                train_main()
        self.assertEqual(raised.exception.code, 2)
        mocked_train.assert_not_called()

    def test_cli_rejects_invalid_optimizer_values_and_budget(self) -> None:
        cases = (
            ("momentum_es", ("--momentum-beta", "1.0")),
            ("adam_es", ("--adam-epsilon", "0")),
            ("clipup_es", ("--clipup-max-speed", "nan")),
            ("snes", ("--snes-sigma-learning-rate", "0")),
            ("standard_es", ("--training-env-step-budget", "0")),
        )
        for condition, flags in cases:
            with self.subTest(condition=condition, flags=flags), patch(
                "experiments.train.load_config",
                return_value=self._base_config(),
            ), patch("experiments.train.train") as mocked_train, patch.object(
                sys,
                "argv",
                self._argv(condition, *flags),
            ), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    train_main()
                self.assertEqual(raised.exception.code, 2)
                mocked_train.assert_not_called()

    def test_pure_override_validator_does_not_mutate_on_failure(self) -> None:
        config = {"condition": "standard_es"}
        with self.assertRaisesRegex(ValueError, "only valid"):
            _apply_optimizer_cli_overrides(
                config,
                "standard_es",
                {"clipup_max_speed": 0.2},
            )
        self.assertEqual(config, {"condition": "standard_es"})

    def test_named_lopo_conditions_reject_incompatible_config_semantics(self) -> None:
        incompatible = {
            "antithetic": False,
            "l2_coeff": 0.01,
            "implicit_damping": 0.1,
            "scalar_damping": 0.1,
            "min_replay_weight_mass": 0.1,
            "curvature_beta": 0.9,
            "curvature_clip": 1000.0,
            "curvature_fitness": "raw",
            "curvature_mode": "diag",
            "curvature_estimator": "block_joint_ols",
            "curvature_confidence_z": 1.645,
            "curvature_rank_utility_mode": "pooled_centered_ranks",
            "curvature_attenuation_mode": "isotropic_norm_matched",
            "rank_fitness": False,
            "evaluate_center_fitness": True,
            "use_leave_one_out_curvature_baseline": True,
            "bias_correct_curvature_ema": True,
        }
        for key, value in incompatible.items():
            config = self._base_config()
            config[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, rf"locks {key}="
            ):
                _condition_config(config, "concave_block_lopo_u_stat")

        condition_specific = (
            (
                "lopo_gradient_only_es",
                "curvature_fitness",
                "matched",
            ),
            (
                "concave_block_lopo_u_stat_isotropic_control",
                "curvature_attenuation_mode",
                "structured",
            ),
        )
        for condition, key, value in condition_specific:
            config = self._base_config()
            config[key] = value
            with self.subTest(condition=condition), self.assertRaisesRegex(
                ValueError, rf"locks {key}="
            ):
                _condition_config(config, condition)

    def test_named_lopo_cli_rejects_semantic_and_irrelevant_overrides(self) -> None:
        cases = (
            ("--curvature-beta", "0.5"),
            ("--curvature-fitness", "raw"),
            ("--curvature-mode", "diag"),
            ("--rank-fitness", "false"),
            ("--bias-correct-curvature-ema", "true"),
            ("--leave-one-out-curvature-baseline", "true"),
            ("--scalar-damping", "0.1"),
            ("--min-replay-weight-mass", "0.1"),
            ("--evaluate-center-fitness", "true"),
        )
        for flags in cases:
            with self.subTest(flags=flags), patch(
                "experiments.train.load_config",
                return_value=self._base_config(),
            ), patch("experiments.train.train") as mocked_train, patch.object(
                sys,
                "argv",
                self._argv("concave_block_lopo_u_stat", *flags),
            ), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    train_main()
                self.assertEqual(raised.exception.code, 2)
                mocked_train.assert_not_called()

    def test_pooled_condition_cli_overrides_remain_available(self) -> None:
        flags = (
            "--curvature-beta",
            "0.5",
            "--curvature-fitness",
            "raw",
            "--leave-one-out-curvature-baseline",
            "true",
            "--scalar-damping",
            "0.2",
        )
        with patch(
            "experiments.train.load_config",
            return_value=self._base_config(),
        ), patch("experiments.train.train") as mocked_train, patch.object(
            sys,
            "argv",
            self._argv("concave_block_ema_curvature_es", *flags),
        ):
            train_main()
        passed = mocked_train.call_args.args[0]
        self.assertEqual(passed["curvature_beta"], 0.5)
        self.assertEqual(passed["curvature_fitness"], "raw")
        self.assertTrue(passed["use_leave_one_out_curvature_baseline"])
        self.assertEqual(passed["scalar_damping"], 0.2)


class TrainingEnvironmentStepBudgetTests(unittest.TestCase):
    class FakeObservationSpace:
        shape = (2,)

    class FakeActionSpace:
        shape = (1,)
        low = np.asarray([-1.0])
        high = np.asarray([1.0])

        def seed(self, seed: int) -> None:
            self.last_seed = seed

    class FakeEnv:
        observation_space = None
        action_space = None

        def __init__(self) -> None:
            self.observation_space = (
                TrainingEnvironmentStepBudgetTests.FakeObservationSpace()
            )
            self.action_space = TrainingEnvironmentStepBudgetTests.FakeActionSpace()

        def reset(self, *, seed: int | None = None):
            self.steps = 0
            return np.zeros(2, dtype=np.float64), {}

        def step(self, action: np.ndarray):
            self.steps += 1
            action_value = float(np.asarray(action).ravel()[0])
            return (
                np.zeros(2, dtype=np.float64),
                1.0 + action_value,
                self.steps >= 2,
                False,
                {},
            )

        def close(self) -> None:
            pass

    class SynchronousPool:
        def __init__(self, *, processes, initializer, initargs):
            self.processes = processes
            initializer(*initargs)

        def map(self, function, tasks):
            return [function(task) for task in tasks]

        def close(self) -> None:
            pass

        def join(self) -> None:
            pass

    @staticmethod
    def _config(**overrides: object) -> dict[str, object]:
        config: dict[str, object] = {
            "env_name": "FakeContinuous-v0",
            "population_size": 2,
            "learning_rate": 0.05,
            "noise_std": 0.02,
            "l2_coeff": 0.0,
            "rank_fitness": True,
            "antithetic": True,
            "max_grad_norm": 0.0,
            "max_param_norm": None,
            "hidden_dims": [],
            "activation": "tanh",
            "output_activation": "tanh",
            "init_param_std": 0.1,
            "n_iterations": 5,
            "eval_episodes": 1,
            "eval_interval": 100,
            "log_interval": 100,
            "max_episode_steps": 2,
            "use_obs_norm": False,
            "replay_enabled": False,
            "buffer_size": 0,
            "reuse_fraction": 0.0,
            "common_rollout_seed": True,
        }
        config.update(overrides)
        return _condition_config(config, "standard_es")

    @staticmethod
    def _production_checkpoint_config() -> dict[str, object]:
        return TrainingEnvironmentStepBudgetTests._config(
            checkpoint_capture_protocol=LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL,
            env_name="Hopper-v5",
            population_size=200,
            learning_rate=1e-4,
            lr_schedule="constant",
            noise_std=0.02,
            hidden_dims=[64, 64],
            init_param_std=0.1,
            n_iterations=250,
            online_evaluation_enabled=False,
            eval_episodes=0,
            max_episode_steps=1000,
            use_obs_norm=True,
            obs_norm_mode="frozen_after_calibration",
            obs_norm_calibration_episodes=3,
            heldout_evaluation_enabled=False,
            checkpoint_capture_generations=[50, 150, 250],
            checkpoint_gradient_archive_length=10,
            min_replay_weight_mass=0.0,
            implicit_damping=0.0,
            scalar_damping=0.0,
            curvature_beta=0.0,
            curvature_clip=0.0,
            evaluate_center_fitness=False,
        )

    @staticmethod
    def _read_json(output: str, name: str):
        with open(os.path.join(output, name), encoding="utf-8") as stream:
            return json.load(stream)

    def _train(self, config: dict[str, object], output: str) -> None:
        with patch(
            "experiments.train._make_env",
            side_effect=lambda *args, **kwargs: self.FakeEnv(),
        ), patch("experiments.train.Pool", self.SynchronousPool), redirect_stdout(
            io.StringIO()
        ):
            train(config, seed=17, output_dir=output, n_workers=1)

    @staticmethod
    def _indexed_gradient(
        optimizer: StandardES,
        noise: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        del noise, fitness
        return np.full(
            optimizer.num_params,
            float(optimizer.iteration),
            dtype=np.float64,
        )

    @staticmethod
    def _load_npz(path: str) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as artifact:
            return {name: artifact[name].copy() for name in artifact.files}

    def test_budget_stops_at_complete_generation_and_records_overshoot(self) -> None:
        config = self._config(
            training_env_step_budget=10,
            heldout_evaluation_enabled=True,
            heldout_training_step_budget=8,
            heldout_eval_episodes=1,
        )
        with tempfile.TemporaryDirectory() as output:
            self._train(config, output)
            saved_config = self._read_json(output, "config.json")
            status = self._read_json(output, "status.json")
            summary = self._read_json(output, "summary.json")
            history = self._read_json(output, "history.json")
            heldout = self._read_json(output, "heldout_evaluation.json")

        expected_budget = {
            "target": 10,
            "reached": 12,
            "overshoot": 2,
            "unit": "training_environment_steps",
            "stopping_reason": "training_env_step_budget_reached",
            "generation_boundary": (
                "first_complete_generation_at_or_above_target"
            ),
            "max_iterations_safety_cap": 5,
        }
        self.assertEqual(saved_config["training_budget"], expected_budget)
        self.assertEqual(status["training_budget"], expected_budget)
        self.assertEqual(summary["training_budget"], expected_budget)
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["completed_iterations"], 3)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[-1]["train_env_steps"], 12)
        self.assertGreater(history[-1]["eval_env_steps_iter"], 0)
        heldout_steps = [
            record["training_env_steps"] for record in heldout["checkpoints"]
        ]
        self.assertEqual(heldout_steps, [0, 4, 8])

    def test_safety_cap_exhaustion_fails_with_auditable_metadata(self) -> None:
        config = self._config(
            training_env_step_budget=20,
            n_iterations=2,
        )
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(
                RuntimeError,
                "safety cap exhausted.*reached 8 of 20",
            ):
                self._train(config, output)
            saved_config = self._read_json(output, "config.json")
            status = self._read_json(output, "status.json")
            summary = self._read_json(output, "summary.json")
            history = self._read_json(output, "history.json")

        for artifact in (saved_config, status, summary):
            self.assertEqual(artifact["training_budget"]["target"], 20)
            self.assertEqual(artifact["training_budget"]["reached"], 8)
            self.assertEqual(artifact["training_budget"]["overshoot"], 0)
            self.assertEqual(
                artifact["training_budget"]["stopping_reason"],
                "max_iterations_exhausted_before_training_env_step_budget",
            )
        self.assertEqual(status["status"], "failed")
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(len(history), 2)

    def test_absent_budget_preserves_fixed_iteration_artifact_shape(self) -> None:
        config = self._config(n_iterations=3)
        with tempfile.TemporaryDirectory() as output:
            self._train(config, output)
            saved_config = self._read_json(output, "config.json")
            status = self._read_json(output, "status.json")
            history = self._read_json(output, "history.json")
            summary_exists = os.path.exists(os.path.join(output, "summary.json"))
            checkpoint_manifest_exists = os.path.exists(
                os.path.join(output, CHECKPOINT_CAPTURE_MANIFEST)
            )
            checkpoint_config_exists = os.path.exists(
                os.path.join(output, CHECKPOINT_TRAINING_CONFIG_ARTIFACT)
            )

        self.assertEqual(len(history), 3)
        self.assertEqual(status["completed_iterations"], 3)
        self.assertEqual(status["expected_iterations"], 3)
        self.assertNotIn("training_budget", saved_config)
        self.assertNotIn("training_budget", status)
        self.assertFalse(summary_exists)
        self.assertNotIn("resolved_checkpoint_capture", saved_config)
        self.assertNotIn("checkpoint_capture", status)
        self.assertFalse(checkpoint_manifest_exists)
        self.assertFalse(checkpoint_config_exists)

    def test_checkpoint_capture_uses_exact_prior_gradient_indices_and_no_evaluation(
        self,
    ) -> None:
        config = self._config(
            n_iterations=12,
            learning_rate=0.05,
            online_evaluation_enabled=False,
            checkpoint_capture_generations=[10, 12],
            checkpoint_gradient_archive_length=10,
            use_obs_norm=True,
            obs_norm_mode="frozen_after_calibration",
            obs_norm_calibration_episodes=1,
        )
        with tempfile.TemporaryDirectory() as output, patch.object(
            StandardES,
            "_gradient",
            self._indexed_gradient,
        ):
            self._train(config, output)
            manifest = self._read_json(output, CHECKPOINT_CAPTURE_MANIFEST)
            saved_config = self._read_json(output, "config.json")
            status = self._read_json(output, "status.json")
            history = self._read_json(output, "history.json")
            training_config_path = os.path.join(
                output, CHECKPOINT_TRAINING_CONFIG_ARTIFACT
            )
            checkpoint_training_config = self._read_json(
                output, CHECKPOINT_TRAINING_CONFIG_ARTIFACT
            )
            checkpoint_paths = [
                os.path.join(
                    output,
                    "checkpoints",
                    f"checkpoint_generation_{generation:06d}.npz",
                )
                for generation in (10, 12)
            ]
            checkpoints = [
                self._load_npz(path) for path in checkpoint_paths
            ]
            training_config_file_hash = _sha256_file(training_config_path)
            checkpoint_file_hashes = [
                _sha256_file(path) for path in checkpoint_paths
            ]
            final_params = np.load(
                os.path.join(output, "final_params.npy"),
                allow_pickle=False,
            )
            with np.load(
                os.path.join(output, "obs_norm.npz"), allow_pickle=False
            ) as obs_norm:
                final_obs_state = {
                    name: np.asarray(obs_norm[name]).copy()
                    for name in obs_norm.files
                }
            temporary_files = [
                filename
                for directory, _, filenames in os.walk(output)
                for filename in filenames
                if ".tmp." in filename
            ]

            self.assertTrue(os.path.exists(training_config_path))
            self.assertFalse(
                os.path.exists(os.path.join(output, "best_params.npy"))
            )
            self.assertFalse(
                os.path.exists(os.path.join(output, "best_obs_norm.npz"))
            )

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["requested_generations"], [10, 12])
        self.assertEqual(manifest["captured_generations"], [10, 12])
        self.assertEqual(manifest["checkpoint_count"], 2)
        self.assertEqual(manifest["gradient_archive_length"], 10)
        self.assertEqual(manifest["selection_policy"], "fixed_config_generations_only")
        self.assertFalse(manifest["reward_selection_used"])
        self.assertFalse(manifest["online_evaluation_enabled"])
        self.assertEqual(saved_config["n_workers"], 1)
        self.assertEqual(checkpoint_training_config["n_workers"], 1)
        self.assertEqual(
            manifest["training_config_sha256"],
            training_config_file_hash,
        )
        self.assertEqual(temporary_files, [])

        expected_ranges = (list(range(10)), list(range(2, 12)))
        for generation, checkpoint, artifact_hash, metadata, expected_indices in zip(
            (10, 12),
            checkpoints,
            checkpoint_file_hashes,
            manifest["artifacts"],
            expected_ranges,
        ):
            self.assertEqual(int(checkpoint["schema_version"]), 2)
            self.assertEqual(
                int(checkpoint["checkpoint_generation"]), generation
            )
            self.assertEqual(
                checkpoint["study_source_sha256"].item().decode("ascii"),
                manifest["source_sha256"],
            )
            self.assertEqual(
                checkpoint["training_config_sha256"].item().decode("ascii"),
                manifest["training_config_sha256"],
            )
            self.assertEqual(
                checkpoint["gradient_generations"].tolist(),
                expected_indices,
            )
            self.assertNotIn(generation, expected_indices)
            self.assertTrue(
                np.array_equal(
                    checkpoint["proposal_gradients"],
                    np.repeat(
                        np.asarray(expected_indices, dtype=np.float64)[:, None],
                        checkpoint["center_params"].size,
                        axis=1,
                    ),
                )
            )
            self.assertEqual(metadata["checkpoint_generation"], generation)
            self.assertEqual(metadata["gradient_generations"], expected_indices)
            self.assertEqual(metadata["artifact_sha256"], artifact_hash)
            self.assertEqual(
                metadata["center_params_sha256"],
                _array_sha256(checkpoint["center_params"]),
            )
            self.assertFalse(metadata["current_checkpoint_gradient_included"])
            self.assertTrue(metadata["strictly_prior_gradient_archive"])
            self.assertEqual(
                metadata["proposal_gradient_hashes"],
                [
                    {
                        "generation": index,
                        "sha256": _array_sha256(
                            checkpoint["proposal_gradients"][row]
                        ),
                    }
                    for row, index in enumerate(expected_indices)
                ],
            )
            self.assertEqual(
                metadata["observation_normalizer_state_sha256"],
                _labeled_arrays_sha256(
                    [
                        ("enabled", checkpoint["obs_normalizer_enabled"]),
                        ("mean", checkpoint["obs_mean"]),
                        ("var", checkpoint["obs_var"]),
                        ("count", checkpoint["obs_count"]),
                    ]
                ),
            )
            self.assertTrue(bool(checkpoint["obs_normalizer_enabled"]))
            self.assertTrue(
                np.array_equal(checkpoint["obs_mean"], final_obs_state["mean"])
            )
            self.assertTrue(
                np.array_equal(checkpoint["obs_var"], final_obs_state["var"])
            )
            self.assertEqual(
                float(checkpoint["obs_count"]),
                float(final_obs_state["count"]),
            )

        self.assertTrue(
            np.allclose(
                checkpoints[1]["center_params"]
                - checkpoints[0]["center_params"],
                0.05 * (10.0 + 11.0),
            )
        )
        self.assertTrue(
            np.array_equal(final_params, checkpoints[1]["center_params"])
        )
        self.assertEqual(status["status"], "complete")
        self.assertEqual(status["checkpoint_capture"]["status"], "complete")
        self.assertEqual(
            status["checkpoint_capture"]["captured_generations"], [10, 12]
        )
        self.assertFalse(saved_config["resolved_online_evaluation"]["enabled"])
        self.assertFalse(
            saved_config["resolved_online_evaluation"][
                "best_policy_selection_by_return"
            ]
        )
        self.assertEqual(len(history), 12)
        self.assertTrue(all(record["eval_reward"] is None for record in history))
        self.assertTrue(all(record["best_reward"] is None for record in history))
        self.assertTrue(all(record["eval_env_steps"] == 0 for record in history))
        self.assertTrue(
            all(record["eval_env_steps_iter"] == 0 for record in history)
        )
        self.assertEqual(
            manifest["validated_generator_controls"],
            {
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
        )

    def test_checkpoint_capture_hashes_are_deterministic_across_identical_runs(
        self,
    ) -> None:
        config = self._config(
            n_iterations=10,
            online_evaluation_enabled=False,
            checkpoint_capture_generations=[10],
        )
        with tempfile.TemporaryDirectory() as root, patch.object(
            StandardES,
            "_gradient",
            self._indexed_gradient,
        ):
            outputs = [os.path.join(root, name) for name in ("first", "second")]
            for output in outputs:
                self._train(config, output)
            manifests = [
                self._read_json(output, CHECKPOINT_CAPTURE_MANIFEST)
                for output in outputs
            ]
            training_configs = []
            checkpoint_payloads = []
            for output in outputs:
                with open(
                    os.path.join(output, CHECKPOINT_TRAINING_CONFIG_ARTIFACT),
                    "rb",
                ) as stream:
                    training_configs.append(stream.read())
                with open(
                    os.path.join(
                        output,
                        "checkpoints",
                        "checkpoint_generation_000010.npz",
                    ),
                    "rb",
                ) as stream:
                    checkpoint_payloads.append(stream.read())

        self.assertEqual(manifests[0], manifests[1])
        self.assertEqual(training_configs[0], training_configs[1])
        self.assertEqual(checkpoint_payloads[0], checkpoint_payloads[1])

    def test_checkpoint_capture_failure_records_partial_artifacts_and_status(
        self,
    ) -> None:
        config = self._config(
            n_iterations=12,
            online_evaluation_enabled=False,
            checkpoint_capture_generations=[10, 12],
        )
        original_tell = StandardES.tell

        def fail_after_first_checkpoint(
            optimizer: StandardES, *args: object, **kwargs: object
        ):
            if optimizer.iteration == 10:
                raise RuntimeError("injected checkpoint generator failure")
            return original_tell(optimizer, *args, **kwargs)

        with tempfile.TemporaryDirectory() as output, patch.object(
            StandardES,
            "_gradient",
            self._indexed_gradient,
        ), patch.object(
            StandardES,
            "tell",
            fail_after_first_checkpoint,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "injected checkpoint generator failure"
            ):
                self._train(config, output)
            manifest = self._read_json(output, CHECKPOINT_CAPTURE_MANIFEST)
            status = self._read_json(output, "status.json")
            history = self._read_json(output, "history.json")
            first_exists = os.path.exists(
                os.path.join(
                    output,
                    "checkpoints",
                    "checkpoint_generation_000010.npz",
                )
            )
            second_exists = os.path.exists(
                os.path.join(
                    output,
                    "checkpoints",
                    "checkpoint_generation_000012.npz",
                )
            )

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failure_phase"], "training")
        self.assertEqual(manifest["captured_generations"], [10])
        self.assertEqual(manifest["completed_iterations"], 10)
        self.assertEqual(len(manifest["artifacts"]), 1)
        self.assertEqual(manifest["error_type"], "RuntimeError")
        self.assertIn("injected checkpoint generator failure", manifest["error"])
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_iterations"], 10)
        self.assertEqual(status["checkpoint_capture"]["status"], "failed")
        self.assertEqual(
            status["checkpoint_capture"]["captured_generations"], [10]
        )
        self.assertEqual(len(history), 10)
        self.assertTrue(first_exists)
        self.assertFalse(second_exists)

    def test_checkpoint_capture_configuration_fails_closed(self) -> None:
        invalid_cases = (
            (
                {"checkpoint_capture_generations": [10, 10]},
                12,
                "unique and strictly increasing",
            ),
            (
                {"checkpoint_capture_generations": [9]},
                12,
                "ten strictly prior gradients",
            ),
            (
                {"checkpoint_capture_generations": [13]},
                12,
                "cannot exceed configured iterations",
            ),
            (
                {
                    "checkpoint_capture_generations": [10],
                    "checkpoint_gradient_archive_length": 9,
                },
                12,
                "must equal 10",
            ),
            (
                {"checkpoint_gradient_archive_length": 10},
                12,
                "checkpoint_capture_generations is required",
            ),
        )
        for config, n_iterations, pattern in invalid_cases:
            with self.subTest(config=config), self.assertRaisesRegex(
                ValueError, pattern
            ):
                _resolve_checkpoint_capture_config(config, n_iterations)

        config = self._config(
            n_iterations=10,
            checkpoint_capture_generations=[10],
        )
        with tempfile.TemporaryDirectory() as output, self.assertRaisesRegex(
            ValueError,
            "online_evaluation_enabled=false",
        ):
            self._train(config, output)

    def test_checked_in_production_checkpoint_configs_opt_into_protocol(self) -> None:
        repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = (
            ("hopper_lagged_subspace_checkpoints.yaml", "Hopper-v5"),
            ("walker2d_lagged_subspace_checkpoints.yaml", "Walker2d-v5"),
            ("halfcheetah_lagged_subspace_checkpoints.yaml", "HalfCheetah-v5"),
        )
        for filename, env_name in expected:
            with self.subTest(filename=filename):
                with open(
                    os.path.join(repository_root, "configs", "mujuco", filename),
                    encoding="utf-8",
                ) as stream:
                    lines = {line.strip() for line in stream}
                self.assertIn(
                    "checkpoint_capture_protocol: "
                    + LAGGED_SUBSPACE_CHECKPOINT_PROTOCOL,
                    lines,
                )
                self.assertIn(f"env_name: {env_name}", lines)

        config = self._production_checkpoint_config()
        settings = _resolve_checkpoint_capture_config(config, 250)
        for seed in (300, 319):
            _validate_lagged_subspace_checkpoint_protocol(
                config,
                seed=seed,
                checkpoint_settings=settings,
            )

    def test_production_checkpoint_serializes_all_study_locks(self) -> None:
        repository_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        lock_paths = {
            "PAPER_EXPECTED_MANIFEST_SHA256": (
                "experiments/manifests/lagged_subspace_frozen_checkpoint.json"
            ),
            "PAPER_EXPECTED_PROTOCOL_SHA256": (
                "docs/lagged_subspace_frozen_checkpoint_protocol.md"
            ),
            "PAPER_EXPECTED_ANALYZER_SHA256": (
                "scripts/analyze_lagged_subspace_frozen_checkpoint.py"
            ),
            "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256": LAUNCHER_BUNDLE_PATH,
            "PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256": DEPENDENCY_BUNDLE_PATH,
        }
        environment = {
            "PAPER_EXPECTED_SOURCE_SHA": (
                current_lagged_subspace_study_sha256()
            ),
            "PAPER_REPO_DIR": repository_root,
            **{
                key: _sha256_file(os.path.join(repository_root, relative))
                for key, relative in lock_paths.items()
            },
        }
        config = self._production_checkpoint_config()
        config["_config_path"] = os.path.join(
            repository_root,
            "configs/mujuco/hopper_lagged_subspace_checkpoints.yaml",
        )
        tiny_policy = MLPPolicy(2, 1, hidden_dims=[])
        with tempfile.TemporaryDirectory() as output, patch.dict(
            os.environ, environment, clear=False
        ), patch(
            "experiments.train._make_env",
            side_effect=lambda *args, **kwargs: self.FakeEnv(),
        ), patch(
            "experiments.train.make_policy", return_value=tiny_policy
        ), patch(
            "experiments.train.Pool", self.SynchronousPool
        ), redirect_stdout(io.StringIO()):
            train(config, seed=300, output_dir=output, n_workers=1)
            saved = self._read_json(output, "config.json")
            capture = self._read_json(output, CHECKPOINT_CAPTURE_MANIFEST)
            checkpoint = self._load_npz(
                os.path.join(
                    output,
                    "checkpoints",
                    "checkpoint_generation_000050.npz",
                )
            )

        provenance = saved["provenance"]
        expected = {
            "expected_source_sha256": environment[
                "PAPER_EXPECTED_SOURCE_SHA"
            ],
            "expected_manifest_sha256": environment[
                "PAPER_EXPECTED_MANIFEST_SHA256"
            ],
            "expected_protocol_sha256": environment[
                "PAPER_EXPECTED_PROTOCOL_SHA256"
            ],
            "expected_analyzer_sha256": environment[
                "PAPER_EXPECTED_ANALYZER_SHA256"
            ],
            "expected_launcher_sha256": environment[
                "PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256"
            ],
            "expected_dependency_lock_sha256": environment[
                "PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256"
            ],
        }
        for key, value in expected.items():
            self.assertEqual(provenance[key], value)
        self.assertEqual(capture["source_sha256"], expected["expected_source_sha256"])
        self.assertEqual(
            checkpoint["study_source_sha256"].item().decode("ascii"),
            expected["expected_source_sha256"],
        )
        self.assertEqual(
            checkpoint["training_config_sha256"].item().decode("ascii"),
            capture["training_config_sha256"],
        )

    def test_production_checkpoint_protocol_rejects_every_drift_category(self) -> None:
        base = self._production_checkpoint_config()
        settings = _resolve_checkpoint_capture_config(base, 250)
        self.assertIsNotNone(settings)
        drift_cases = (
            ("env_name", "Ant-v5"),
            ("population_size", 198),
            ("noise_std", 0.03),
            ("learning_rate", 3e-4),
            ("lr_schedule", "inverse_sqrt"),
            ("n_iterations", 251),
            ("checkpoint_capture_generations", [50, 149, 250]),
            ("checkpoint_gradient_archive_length", 11),
            ("hidden_dims", [32, 64]),
            ("activation", "relu"),
            ("output_activation", "identity"),
            ("init_param_std", 0.2),
            ("use_obs_norm", False),
            ("obs_norm_mode", "online"),
            ("obs_norm_calibration_episodes", 2),
            ("common_rollout_seed", False),
            ("antithetic", False),
            ("max_episode_steps", 999),
            ("l2_coeff", 1e-4),
            ("max_grad_norm", 1.0),
            ("max_param_norm", 10.0),
            ("replay_enabled", True),
            ("buffer_size", 1),
            ("reuse_fraction", 0.1),
            ("min_replay_weight_mass", 0.1),
            ("use_curvature", True),
            ("curvature_beta", 0.9),
            ("curvature_clip", 1.0),
            ("implicit_damping", 1.0),
            ("scalar_damping", 1.0),
            ("evaluate_center_fitness", True),
            ("online_evaluation_enabled", True),
            ("heldout_evaluation_enabled", True),
            ("training_env_step_budget", 1000),
        )
        for key, bad_value in drift_cases:
            with self.subTest(key=key, bad_value=bad_value):
                config = dict(base)
                config[key] = bad_value
                with self.assertRaisesRegex(ValueError, key):
                    _validate_lagged_subspace_checkpoint_protocol(
                        config,
                        seed=300,
                        checkpoint_settings=settings,
                    )

        for forbidden_key in (
            "trust_radius",
            "implicit_iterations",
            "min_importance_weight",
            "momentum_beta",
            "curvature_estimator",
        ):
            with self.subTest(forbidden_key=forbidden_key):
                config = dict(base)
                config[forbidden_key] = 1.0
                with self.assertRaisesRegex(
                    ValueError, "forbidden mechanism-specific settings"
                ):
                    _validate_lagged_subspace_checkpoint_protocol(
                        config,
                        seed=300,
                        checkpoint_settings=settings,
                    )

        for invalid_seed in (299, 320, 300.0, "300", True):
            with self.subTest(seed=invalid_seed), self.assertRaisesRegex(
                ValueError, "seed"
            ):
                _validate_lagged_subspace_checkpoint_protocol(
                    base,
                    seed=invalid_seed,
                    checkpoint_settings=settings,
                )

    def test_cli_equivalent_checkpoint_drift_fails_before_environment_or_pool(
        self,
    ) -> None:
        cases = (
            ("learning_rate", 3e-4, 300),
            ("lr_schedule", "inverse_sqrt", 300),
            ("n_iterations", 251, 300),
            (None, None, 299),
        )
        for key, value, seed in cases:
            with self.subTest(key=key, value=value, seed=seed):
                config = self._production_checkpoint_config()
                if key is not None:
                    config[key] = value
                with tempfile.TemporaryDirectory() as output, patch(
                    "experiments.train._make_env"
                ) as make_env, patch("experiments.train.Pool") as pool:
                    with self.assertRaisesRegex(
                        ValueError,
                        "configuration drift",
                    ):
                        train(
                            config,
                            seed=seed,
                            output_dir=output,
                            n_workers=1,
                        )
                    make_env.assert_not_called()
                    pool.assert_not_called()

    def test_snes_training_persists_dynamic_search_distribution(self) -> None:
        config = _condition_config(
            self._config(population_size=4, n_iterations=2),
            "snes",
        )
        with tempfile.TemporaryDirectory() as output:
            self._train(config, output)
            saved_config = self._read_json(output, "config.json")
            history = self._read_json(output, "history.json")
            final_search_std = np.load(
                os.path.join(output, "snes_search_std.npy")
            )

        resolved = saved_config["resolved_optimizer"]
        self.assertEqual(resolved["method"], "snes")
        self.assertEqual(resolved["search_covariance"], "learned_diagonal")
        self.assertEqual(
            resolved["final_search_std_artifact"], "snes_search_std.npy"
        )
        self.assertEqual(
            resolved["final_search_std_artifact_semantics"],
            "final_optimizer_state_for_audit_not_a_resume_checkpoint",
        )
        self.assertEqual(len(history), 2)
        self.assertEqual(final_search_std.ndim, 1)
        self.assertTrue(np.all(np.isfinite(final_search_std)))
        self.assertTrue(np.all(final_search_std > 0.0))
        self.assertAlmostEqual(
            float(np.min(final_search_std)), history[-1]["snes_sigma_min_after"]
        )
        self.assertAlmostEqual(
            float(np.max(final_search_std)), history[-1]["snes_sigma_max_after"]
        )
        for record in history:
            self.assertEqual(record["optimizer_type"], "snes")
            self.assertGreater(record["snes_sigma_min_after"], 0.0)
            self.assertGreaterEqual(
                record["snes_sigma_max_after"],
                record["snes_sigma_min_after"],
            )
        self.assertAlmostEqual(
            history[1]["snes_sigma_min_before"],
            history[0]["snes_sigma_min_after"],
        )
        self.assertAlmostEqual(
            history[1]["snes_sigma_max_before"],
            history[0]["snes_sigma_max_after"],
        )

    def test_snes_second_generation_uses_first_updated_coordinate_scale(self) -> None:
        evaluated_populations: list[tuple[np.ndarray, np.ndarray]] = []

        class RecordingPool(self.SynchronousPool):
            def map(inner_self, function, tasks):
                results = super().map(function, tasks)
                if getattr(function, "__name__", "") == "_evaluate_params" and len(tasks) == 4:
                    evaluated_populations.append(
                        (
                            np.stack([np.asarray(task[0]) for task in tasks]),
                            np.asarray([result[0] for result in results]),
                        )
                    )
                return results

        config = _condition_config(
            self._config(
                population_size=4,
                n_iterations=2,
                learning_rate=0.4,
                snes_sigma_learning_rate=0.2,
            ),
            "snes",
        )
        with tempfile.TemporaryDirectory() as output, patch(
            "experiments.train._make_env",
            side_effect=lambda *args, **kwargs: self.FakeEnv(),
        ), patch("experiments.train.Pool", RecordingPool), redirect_stdout(
            io.StringIO()
        ):
            train(config, seed=17, output_dir=output, n_workers=1)

        self.assertEqual(len(evaluated_populations), 2)
        first_candidates, first_fitness = evaluated_populations[0]
        second_candidates, _ = evaluated_populations[1]
        num_params = first_candidates.shape[1]
        reference = SNES(
            num_params=num_params,
            population_size=4,
            learning_rate=0.4,
            noise_std=0.02,
            antithetic=True,
            snes_sigma_learning_rate=0.2,
            seed=17,
        )
        first_noise, _ = reference.ask()
        second_noise, _ = reference.ask()
        first_center = np.mean(first_candidates, axis=0)
        self.assertTrue(
            np.allclose(first_candidates, first_center + 0.02 * first_noise)
        )
        utilities = snes_utilities(first_fitness)
        expected_second_std = 0.02 * np.exp(
            0.1
            * np.sum(
                utilities[:, None] * (np.square(first_noise) - 1.0),
                axis=0,
            )
        )
        self.assertGreater(float(np.ptp(expected_second_std)), 1e-8)
        second_center = np.mean(second_candidates, axis=0)
        self.assertTrue(
            np.allclose(
                second_candidates,
                second_center + expected_second_std * second_noise,
            )
        )

    def test_lopo_trainer_condition_is_fresh_and_resolved(self) -> None:
        cases = (
            (
                "lopo_gradient_only_es",
                "explicit_lopo_rank_gradient",
                "lopo_gradient_without_curvature",
                False,
            ),
            (
                "concave_block_lopo_u_stat",
                "concave_projected_block_lopo_rank_u_statistic",
                "lopo_structured_block_curvature",
                True,
            ),
            (
                "concave_block_lopo_u_stat_isotropic_control",
                "concave_projected_block_lopo_rank_u_statistic_"
                "isotropic_norm_control",
                "lopo_isotropic_norm_matched_attenuation_control",
                True,
            ),
        )
        for condition, solver, role, uses_curvature in cases:
            with self.subTest(condition=condition):
                config = _condition_config(
                    self._config(population_size=6, n_iterations=1),
                    condition,
                )
                with tempfile.TemporaryDirectory() as output:
                    self._train(config, output)
                    saved_config = self._read_json(output, "config.json")
                    history = self._read_json(output, "history.json")
                    artifacts = set(os.listdir(output))

                resolved = saved_config["resolved_optimizer"]
                self.assertEqual(saved_config["l2_coeff"], 0.0)
                self.assertEqual(saved_config["scalar_damping"], 0.0)
                self.assertEqual(saved_config["curvature_clip"], 0.0)
                self.assertEqual(saved_config["min_replay_weight_mass"], 0.0)
                self.assertFalse(saved_config["evaluate_center_fitness"])
                self.assertFalse(
                    saved_config["use_leave_one_out_curvature_baseline"]
                )
                self.assertEqual(
                    resolved["curvature_rank_utility_mode"],
                    "lopo_rank_u_statistic",
                )
                self.assertEqual(resolved["solver_type"], solver)
                self.assertEqual(resolved["attribution_role"], role)
                self.assertFalse(resolved["replay_enabled"])
                self.assertFalse(resolved["sample_reuse"])
                self.assertFalse(resolved["importance_weighting"])
                self.assertEqual(
                    resolved["persists_hessian_ema_artifact"],
                    uses_curvature,
                )
                self.assertEqual(
                    resolved["hessian_ema_artifact"],
                    "hessian_ema.npy" if uses_curvature else None,
                )
                self.assertEqual(
                    "hessian_ema.npy" in artifacts,
                    uses_curvature,
                )
                self.assertEqual(len(history), 1)
                record = history[0]
                self.assertEqual(record["n_fresh"], 6)
                self.assertEqual(record["n_reused"], 0)
                self.assertFalse(record["used_replay"])
                self.assertEqual(
                    record["rank_utility_mode"],
                    "lopo_rank_u_statistic",
                )
                self.assertTrue(record["lopo_zero_sum_identity_verified"])
                self.assertFalse(record["lopo_centering_operation_applied"])
                self.assertTrue(
                    record[
                        "lopo_at_proposal_sn_unnormalized_identity_verified"
                    ]
                )
                self.assertLessEqual(
                    record[
                        "lopo_at_proposal_unnormalized_minus_sn_block_gap_max_abs"
                    ],
                    record["lopo_at_proposal_endpoint_identity_tolerance"],
                )
                if uses_curvature:
                    self.assertTrue(
                        resolved[
                            "raw_lopo_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average"
                        ]
                    )
                    self.assertTrue(record["lopo_jackknife_computation_valid"])
                    self.assertEqual(
                        record["lopo_jackknife_validity"],
                        "asymptotic_if_iid_nondegenerate_pair_clusters",
                    )
                    self.assertFalse(
                        record[
                            "lopo_jackknife_inference_assumptions_runtime_verified"
                        ]
                    )
                    self.assertTrue(record["lopo_raw_identities_verified"])
                else:
                    self.assertFalse(resolved["curvature_used"])
                    self.assertNotIn(
                        "lopo_jackknife_computation_valid", record
                    )
                    self.assertEqual(
                        record[
                            "lopo_raw_block_moment_endpoint_jacobian_claim_applicability"
                        ],
                        "not_applicable_no_curvature_operator",
                    )

    def test_diiwes_training_still_persists_hessian_ema_artifact(self) -> None:
        config = _condition_config(
            self._config(population_size=6, n_iterations=1),
            "diag_curvature",
        )
        with tempfile.TemporaryDirectory() as output:
            self._train(config, output)
            saved_config = self._read_json(output, "config.json")
            state_path = os.path.join(output, "hessian_ema.npy")
            self.assertTrue(os.path.isfile(state_path))
            state = np.load(state_path)

        resolved = saved_config["resolved_optimizer"]
        self.assertEqual(resolved["type"], "DIIWES")
        self.assertTrue(resolved["persists_hessian_ema_artifact"])
        self.assertEqual(
            resolved["hessian_ema_artifact"], "hessian_ema.npy"
        )
        self.assertEqual(state.ndim, 1)
        self.assertTrue(np.all(np.isfinite(state)))

    def test_diiwes_fresh_tolerance_does_not_hide_replay(self) -> None:
        config = _condition_config(
            self._config(population_size=6, n_iterations=1),
            "diag_curvature",
        )
        original_tell = DIIWES.tell

        def report_replay(optimizer, *args, **kwargs):
            updated, info = original_tell(optimizer, *args, **kwargs)
            info = dict(info)
            info["n_reused"] = 1
            info["used_replay"] = True
            return updated, info

        with tempfile.TemporaryDirectory() as output, patch.object(
            DIIWES, "tell", report_replay
        ), self.assertRaisesRegex(
            RuntimeError, "no-replay protocol detected replay"
        ):
            self._train(config, output)

    def test_heldout_endpoint_cannot_exceed_training_budget(self) -> None:
        config = self._config(
            training_env_step_budget=8,
            heldout_evaluation_enabled=True,
            heldout_training_step_budget=9,
            heldout_eval_episodes=1,
        )
        with tempfile.TemporaryDirectory() as output, self.assertRaisesRegex(
            ValueError,
            "heldout_training_step_budget cannot exceed",
        ):
            train(config, seed=17, output_dir=output, n_workers=1)

    def test_budget_resolver_rejects_nonpositive_or_nonintegral_values(self) -> None:
        for value in (True, 0, -1, 1.5, float("inf"), "not-an-integer"):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError,
                "positive integer",
            ):
                _resolve_training_env_step_budget(
                    {"training_env_step_budget": value}
                )
        self.assertIsNone(_resolve_training_env_step_budget({}))
        self.assertEqual(
            _resolve_training_env_step_budget(
                {"training_env_step_budget": "12"}
            ),
            12,
        )


class StandardESTests(unittest.TestCase):
    def test_step_is_finite_and_has_no_trust_state(self) -> None:
        opt = StandardES(num_params=5, population_size=8, learning_rate=1.0, noise_std=0.1, seed=0)
        params = np.zeros(5)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - 0.2 * noise[:, 1]
        new_params, step_info = opt.tell(params, noise, fitness, ask_info)

        self.assertTrue(np.all(np.isfinite(new_params)))
        self.assertGreater(step_info["step_norm"], 0.0)
        self.assertFalse(hasattr(opt, "trust_radius"))
        self.assertNotIn("trust_active", step_info)
        self.assertNotIn("pre_trust_step_norm", step_info)

    def test_nonfinite_fitness_is_rejected(self) -> None:
        opt = StandardES(num_params=1, population_size=2, noise_std=0.1)
        params = np.zeros(1)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        with self.assertRaisesRegex(ValueError, "fitness"):
            opt.tell(params, noise, np.asarray([0.0, np.nan]), ask_info)

    def test_empty_and_nonvector_fitness_are_rejected(self) -> None:
        opt = StandardES(num_params=2, population_size=2)
        with self.assertRaisesRegex(ValueError, "at least one"):
            opt.tell(np.zeros(2), np.empty((0, 2)), np.empty(0))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            opt.tell(np.zeros(2), np.zeros((2, 2)), np.zeros((2, 1)))

    def test_antithetic_pair_metadata_matches_noise(self) -> None:
        opt = StandardES(num_params=3, population_size=8, seed=9)
        noise, ask_info = opt.ask()
        plus = ask_info["fresh_pair_plus"]
        minus = ask_info["fresh_pair_minus"]
        self.assertEqual(len(plus), 4)
        self.assertTrue(np.array_equal(noise[minus], -noise[plus]))


class AdaptiveESBaselineTests(unittest.TestCase):
    @staticmethod
    def _axis_batch(axis: int) -> tuple[np.ndarray, np.ndarray]:
        noise = np.zeros((2, 2), dtype=np.float64)
        noise[0, axis] = 1.0
        noise[1, axis] = -1.0
        return noise, np.asarray([1.0, -1.0], dtype=np.float64)

    def test_momentum_closed_form_first_and_second_updates(self) -> None:
        optimizer = MomentumES(
            num_params=2,
            population_size=2,
            learning_rate=2.0,
            noise_std=1.0,
            momentum_beta=0.5,
            antithetic=False,
        )
        params = np.zeros(2, dtype=np.float64)
        first_noise, fitness = self._axis_batch(0)
        first_gradient = optimizer._gradient(first_noise, fitness)
        first_expected_buffer = first_gradient
        first_params, first_info = optimizer.tell(
            params, first_noise, fitness
        )
        self.assertTrue(
            np.allclose(first_params, 2.0 * first_expected_buffer)
        )
        self.assertTrue(
            np.allclose(optimizer.momentum_buffer, first_expected_buffer)
        )
        self.assertEqual(first_info["momentum_iteration"], 1)

        second_noise, fitness = self._axis_batch(1)
        second_gradient = optimizer._gradient(second_noise, fitness)
        second_expected_buffer = 0.5 * first_expected_buffer + second_gradient
        second_params, second_info = optimizer.tell(
            first_params, second_noise, fitness
        )
        self.assertTrue(
            np.allclose(
                second_params - first_params,
                2.0 * second_expected_buffer,
            )
        )
        self.assertTrue(
            np.allclose(optimizer.momentum_buffer, second_expected_buffer)
        )
        self.assertEqual(second_info["optimizer_type"], "momentum")
        self.assertEqual(second_info["momentum_iteration"], 2)

    def test_adam_closed_form_first_and_second_updates(self) -> None:
        beta1 = 0.5
        beta2 = 0.25
        epsilon = 1e-6
        learning_rate = 2.0
        optimizer = AdamES(
            num_params=2,
            population_size=2,
            learning_rate=learning_rate,
            noise_std=1.0,
            adam_beta1=beta1,
            adam_beta2=beta2,
            adam_epsilon=epsilon,
            antithetic=False,
        )
        params = np.zeros(2, dtype=np.float64)
        first_noise, fitness = self._axis_batch(0)
        first_gradient = optimizer._gradient(first_noise, fitness)
        first_moment = (1.0 - beta1) * first_gradient
        first_second_moment = (1.0 - beta2) * np.square(first_gradient)
        first_corrected = first_moment / (1.0 - beta1)
        first_second_corrected = first_second_moment / (1.0 - beta2)
        first_expected_step = learning_rate * first_corrected / (
            np.sqrt(first_second_corrected) + epsilon
        )
        first_params, first_info = optimizer.tell(
            params, first_noise, fitness
        )
        self.assertTrue(np.allclose(first_params, first_expected_step))
        self.assertEqual(first_info["adam_iteration"], 1)

        second_noise, fitness = self._axis_batch(1)
        second_gradient = optimizer._gradient(second_noise, fitness)
        second_moment = beta1 * first_moment + (1.0 - beta1) * second_gradient
        second_second_moment = (
            beta2 * first_second_moment
            + (1.0 - beta2) * np.square(second_gradient)
        )
        second_corrected = second_moment / (1.0 - beta1**2)
        second_second_corrected = second_second_moment / (1.0 - beta2**2)
        second_expected_step = learning_rate * second_corrected / (
            np.sqrt(second_second_corrected) + epsilon
        )
        second_params, second_info = optimizer.tell(
            first_params, second_noise, fitness
        )
        self.assertTrue(
            np.allclose(second_params - first_params, second_expected_step)
        )
        self.assertTrue(np.allclose(optimizer.adam_first_moment, second_moment))
        self.assertTrue(
            np.allclose(optimizer.adam_second_moment, second_second_moment)
        )
        self.assertEqual(second_info["optimizer_type"], "adam")
        self.assertEqual(second_info["adam_iteration"], 2)

    def test_zero_gradient_leaves_fresh_optimizer_at_same_parameters(self) -> None:
        tied_fitness = np.ones(2, dtype=np.float64)
        for optimizer in (
            MomentumES(num_params=2, population_size=2),
            AdamES(num_params=2, population_size=2),
            ClipUpES(num_params=2, population_size=2),
            SNES(num_params=2, population_size=2),
        ):
            params = np.asarray([0.25, -0.75])
            optimizer.current_params = params.copy()
            noise, ask_info = optimizer.ask()
            updated, info = optimizer.tell(
                params, noise, tied_fitness, ask_info
            )
            self.assertTrue(np.array_equal(updated, params))
            self.assertEqual(info["grad_norm"], 0.0)
            self.assertEqual(info["step_norm"], 0.0)

    def test_adaptive_baselines_share_noise_and_gradient_with_standard_es(self) -> None:
        optimizers = [
            StandardES(num_params=3, population_size=8, seed=37),
            MomentumES(num_params=3, population_size=8, seed=37),
            AdamES(num_params=3, population_size=8, seed=37),
            ClipUpES(num_params=3, population_size=8, seed=37),
        ]
        for _ in range(3):
            batches = [optimizer.ask() for optimizer in optimizers]
            noises = [batch[0] for batch in batches]
            self.assertTrue(np.array_equal(noises[0], noises[1]))
            self.assertTrue(np.array_equal(noises[0], noises[2]))
            self.assertTrue(np.array_equal(noises[0], noises[3]))
            fitness = noises[0][:, 0] - 0.2 * noises[0][:, 1]
            gradients = [optimizer._gradient(noise, fitness) for optimizer, noise in zip(optimizers, noises)]
            self.assertTrue(np.array_equal(gradients[0], gradients[1]))
            self.assertTrue(np.array_equal(gradients[0], gradients[2]))
            self.assertTrue(np.array_equal(gradients[0], gradients[3]))

    def test_adaptive_baselines_report_exact_fresh_only_diagnostics(self) -> None:
        for optimizer in (
            MomentumES(num_params=3, population_size=8, seed=41),
            AdamES(num_params=3, population_size=8, seed=41),
            ClipUpES(num_params=3, population_size=8, seed=41),
            SNES(num_params=3, population_size=8, seed=41),
        ):
            params = np.zeros(3, dtype=np.float64)
            optimizer.current_params = params.copy()
            noise, ask_info = optimizer.ask()
            self.assertEqual(ask_info["n_fresh"], 8)
            self.assertEqual(ask_info["n_reused"], 0)
            self.assertFalse(np.any(ask_info["is_reused"]))
            fitness = noise[:, 0] - noise[:, 1]
            _, info = optimizer.tell(params, noise, fitness, ask_info)
            exact = {
                "n_fresh": 8,
                "n_reused": 0,
                "buffer_size": 0,
                "used_replay": False,
                "replay_weight_mass": 0.0,
                "fresh_weight_mass": 1.0,
                "ess": 8.0,
                "ess_ratio": 1.0,
                "ess_normalized": 1.0,
                "importance_weight_min": 1.0,
                "importance_weight_mean": 1.0,
                "importance_weight_max": 1.0,
            }
            for key, expected in exact.items():
                self.assertEqual(info[key], expected)
            self.assertFalse(hasattr(optimizer, "sample_buffer"))
            replay_noise, replay_info = optimizer.ask()
            replay_fitness = replay_noise[:, 0] - replay_noise[:, 1]
            replay_info["is_reused"] = np.asarray(
                [True] + [False] * (optimizer.population_size - 1)
            )
            with self.assertRaisesRegex(ValueError, "replayed samples"):
                optimizer.tell(
                    optimizer.current_params,
                    replay_noise,
                    replay_fitness,
                    replay_info,
                )

    def test_adaptive_conditions_resolve_exact_optimizer_metadata(self) -> None:
        cases = {
            "momentum_es": (
                MomentumES,
                {"momentum_beta": 0.75},
                {
                    "method": "momentum_es",
                    "update_rule": "heavy_ball_momentum",
                    "momentum_beta": 0.75,
                },
            ),
            "adam_es": (
                AdamES,
                {
                    "adam_beta1": 0.8,
                    "adam_beta2": 0.95,
                    "adam_epsilon": 1e-6,
                },
                {
                    "method": "adam_es",
                    "update_rule": "bias_corrected_adam",
                    "adam_beta1": 0.8,
                    "adam_beta2": 0.95,
                    "adam_epsilon": 1e-6,
                    "adam_bias_correction": True,
                },
            ),
            "clipup_es": (
                ClipUpES,
                {
                    "clipup_momentum": 0.8,
                    "clipup_max_speed": 0.25,
                },
                {
                    "method": "clipup_es",
                    "update_rule": "normalized_gradient_momentum_velocity_clip",
                    "clipup_momentum": 0.8,
                    "clipup_max_speed": 0.25,
                    "clipup_step_size_source": "learning_rate_schedule",
                    "clipup_gradient_normalization": True,
                    "clipup_velocity_clipping": True,
                },
            ),
            "snes": (
                SNES,
                {"snes_sigma_learning_rate": 0.125},
                {
                    "method": "snes",
                    "update_rule": "separable_gaussian_natural_gradient",
                    "initial_coordinate_sigma": 0.2,
                    "sigma_learning_rate": 0.125,
                    "uses_default_sigma_learning_rate": False,
                    "utility_shaping": "canonical_log_rank_tie_averaged",
                    "search_covariance": "learned_diagonal",
                    "sigma_clipping": False,
                },
            ),
        }
        for condition, (optimizer_type, overrides, expected) in cases.items():
            config = _condition_config(
                {
                    "replay_enabled": False,
                    "reuse_fraction": 0.0,
                    "buffer_size": 0,
                    "population_size": 8,
                    "noise_std": 0.2,
                    **overrides,
                },
                condition,
            )
            optimizer = make_optimizer(config, 3, None, seed=43)
            self.assertIsInstance(optimizer, optimizer_type)
            resolved = _resolved_optimizer_config(optimizer)
            for key, value in expected.items():
                self.assertEqual(resolved[key], value)
            self.assertFalse(resolved["trust_region"])
            self.assertFalse(resolved["replay_enabled"])
            self.assertEqual(resolved["max_grad_norm"], 0.0)
            self.assertIsNone(resolved["max_param_norm"])


class SNESTests(unittest.TestCase):
    def test_mean_and_diagonal_sigma_updates_match_snes_equations(self) -> None:
        optimizer = SNES(
            num_params=2,
            population_size=4,
            learning_rate=0.4,
            noise_std=0.5,
            antithetic=False,
            snes_sigma_learning_rate=0.2,
        )
        params = np.asarray([1.0, -1.0])
        noise = np.asarray(
            [
                [1.0, 2.0],
                [-1.0, -2.0],
                [0.5, -0.5],
                [-0.5, 0.5],
            ]
        )
        fitness = np.asarray([4.0, 1.0, 3.0, 2.0])
        utilities = snes_utilities(fitness)
        old_std = np.full(2, 0.5)
        mean_natural_gradient = old_std * np.sum(
            utilities[:, None] * noise, axis=0
        )
        log_sigma_gradient = np.sum(
            utilities[:, None] * (np.square(noise) - 1.0), axis=0
        )
        expected_step = 0.4 * mean_natural_gradient
        expected_std = old_std * np.exp(0.1 * log_sigma_gradient)

        optimizer.current_params = params.copy()
        _, ask_info = optimizer.ask()
        updated, info = optimizer.tell(params, noise, fitness, ask_info)

        self.assertTrue(np.allclose(updated, params + expected_step))
        self.assertTrue(np.allclose(optimizer.search_std, expected_std))
        self.assertEqual(info["optimizer_type"], "snes")
        self.assertEqual(info["solver_type"], "separable_natural_gradient")
        self.assertEqual(info["snes_iteration"], 1)
        self.assertAlmostEqual(
            info["snes_standardized_mean_gradient_norm"],
            float(np.linalg.norm(np.sum(utilities[:, None] * noise, axis=0))),
        )
        self.assertAlmostEqual(
            info["snes_parameter_space_mean_direction_norm"],
            float(np.linalg.norm(mean_natural_gradient)),
        )
        self.assertAlmostEqual(
            info["snes_mean_natural_gradient_norm"],
            float(np.linalg.norm(mean_natural_gradient)),
        )
        self.assertAlmostEqual(
            info["snes_log_sigma_natural_gradient_norm"],
            float(np.linalg.norm(log_sigma_gradient)),
        )
        candidate = optimizer.candidate_params(updated, np.ones(2))
        self.assertTrue(np.allclose(candidate, updated + expected_std))

    def test_default_sigma_rate_matches_reference_dimension_rule(self) -> None:
        optimizer = SNES(num_params=25, population_size=4)
        expected = (3.0 + np.log(25.0)) / (5.0 * np.sqrt(25.0))
        self.assertAlmostEqual(optimizer.snes_sigma_learning_rate, expected)
        self.assertTrue(optimizer.snes_uses_default_sigma_learning_rate)

    def test_ask_records_diagonal_sampling_scale_and_rejects_stale_state(self) -> None:
        optimizer = SNES(num_params=3, population_size=4, noise_std=0.2, seed=7)
        params = np.asarray([0.1, -0.2, 0.3])
        optimizer.current_params = params.copy()
        noise, ask_info = optimizer.ask()
        self.assertTrue(
            np.array_equal(ask_info["snes_sampling_std"], np.full(3, 0.2))
        )
        self.assertEqual(ask_info["snes_generation_token"], 0)
        candidates = optimizer.candidate_params(params, noise)
        self.assertTrue(np.allclose(candidates, params + 0.2 * noise))

        optimizer.search_std[0] *= 2.0
        with self.assertRaisesRegex(ValueError, "search scale"):
            optimizer.tell(params, noise, noise[:, 0], ask_info)

    def test_tell_requires_complete_current_ask_metadata(self) -> None:
        optimizer = SNES(num_params=2, population_size=4, noise_std=0.2, seed=5)
        params = np.asarray([0.1, -0.2])
        optimizer.current_params = params.copy()
        noise, ask_info = optimizer.ask()
        fitness = noise[:, 0]

        with self.assertRaisesRegex(ValueError, "requires ask_info"):
            optimizer.tell(params, noise, fitness)

        missing_scale = dict(ask_info)
        del missing_scale["snes_sampling_std"]
        with self.assertRaisesRegex(ValueError, "snes_sampling_std"):
            optimizer.tell(params, noise, fitness, missing_scale)

        missing_token = dict(ask_info)
        del missing_token["snes_generation_token"]
        with self.assertRaisesRegex(ValueError, "snes_generation_token"):
            optimizer.tell(params, noise, fitness, missing_token)

        stale_token = dict(ask_info)
        stale_token["snes_generation_token"] = -1
        with self.assertRaisesRegex(ValueError, "generation token"):
            optimizer.tell(params, noise, fitness, stale_token)

        updated, _ = optimizer.tell(params, noise, fitness, ask_info)
        with self.assertRaisesRegex(ValueError, "generation token"):
            optimizer.tell(updated, noise, fitness, ask_info)

    def test_tied_fitness_leaves_mean_and_search_scale_unchanged(self) -> None:
        optimizer = SNES(
            num_params=2,
            population_size=4,
            noise_std=0.3,
            seed=11,
        )
        params = np.asarray([0.4, -0.7])
        optimizer.current_params = params.copy()
        noise, ask_info = optimizer.ask()
        updated, info = optimizer.tell(
            params, noise, np.ones(4), ask_info
        )
        self.assertTrue(np.array_equal(updated, params))
        self.assertTrue(np.array_equal(optimizer.search_std, np.full(2, 0.3)))
        self.assertEqual(info["snes_utility_l1_norm"], 0.0)

    def test_rejects_noncanonical_controls_and_invalid_sigma_rate(self) -> None:
        cases = (
            ({"rank_fitness": False}, "rank-based"),
            ({"l2_coeff": 0.1}, "l2_coeff"),
            ({"max_grad_norm": 1.0}, "gradient clipping"),
            ({"max_param_norm": 1.0}, "parameter projection"),
            ({"snes_sigma_learning_rate": 0.0}, "finite and positive"),
        )
        for kwargs, pattern in cases:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, pattern
            ):
                SNES(num_params=2, population_size=4, **kwargs)


class ClipUpESTests(unittest.TestCase):
    @staticmethod
    def _axis_batch(axis: int) -> tuple[np.ndarray, np.ndarray]:
        noise = np.zeros((2, 2), dtype=np.float64)
        noise[0, axis] = 1.0
        noise[1, axis] = -1.0
        return noise, np.asarray([1.0, -1.0], dtype=np.float64)

    def test_primary_update_first_and_second_steps_without_clipping(self) -> None:
        optimizer = ClipUpES(
            num_params=2,
            population_size=2,
            learning_rate=0.2,
            noise_std=1.0,
            antithetic=False,
            clipup_momentum=0.5,
            clipup_max_speed=1.0,
        )
        params = np.zeros(2, dtype=np.float64)
        first_noise, fitness = self._axis_batch(0)
        first_params, first_info = optimizer.tell(
            params, first_noise, fitness
        )
        first_velocity = np.asarray([0.2, 0.0])
        self.assertTrue(np.allclose(first_params, first_velocity))
        self.assertTrue(np.allclose(optimizer.clipup_velocity, first_velocity))
        self.assertFalse(first_info["clipup_velocity_clipped"])
        self.assertAlmostEqual(
            first_info["clipup_normalized_gradient_step_norm"], 0.2
        )

        second_noise, fitness = self._axis_batch(1)
        second_params, second_info = optimizer.tell(
            first_params, second_noise, fitness
        )
        second_velocity = np.asarray([0.1, 0.2])
        self.assertTrue(
            np.allclose(second_params - first_params, second_velocity)
        )
        self.assertTrue(np.allclose(optimizer.clipup_velocity, second_velocity))
        self.assertFalse(second_info["clipup_velocity_clipped"])
        self.assertEqual(second_info["clipup_iteration"], 2)

    def test_velocity_is_clipped_after_momentum_accumulation(self) -> None:
        optimizer = ClipUpES(
            num_params=2,
            population_size=2,
            learning_rate=0.4,
            noise_std=1.0,
            antithetic=False,
            clipup_momentum=0.9,
            clipup_max_speed=0.25,
        )
        noise, fitness = self._axis_batch(0)
        params, first_info = optimizer.tell(np.zeros(2), noise, fitness)
        self.assertTrue(np.allclose(params, [0.25, 0.0]))
        self.assertTrue(first_info["clipup_velocity_clipped"])
        self.assertAlmostEqual(first_info["clipup_velocity_clip_scale"], 0.625)
        self.assertAlmostEqual(first_info["clipup_velocity_norm"], 0.25)

        updated, second_info = optimizer.tell(params, noise, fitness)
        self.assertTrue(np.allclose(updated - params, [0.25, 0.0]))
        self.assertAlmostEqual(
            second_info["clipup_velocity_norm_before_clip"], 0.625
        )
        self.assertAlmostEqual(second_info["clipup_velocity_clip_scale"], 0.4)
        self.assertAlmostEqual(np.linalg.norm(updated - params), 0.25)

    def test_gradient_normalization_is_scale_invariant_and_handles_zero(self) -> None:
        first = ClipUpES(
            num_params=2,
            learning_rate=0.1,
            clipup_momentum=0.5,
            clipup_max_speed=1.0,
        )
        second = ClipUpES(
            num_params=2,
            learning_rate=0.1,
            clipup_momentum=0.5,
            clipup_max_speed=1.0,
        )
        step, _ = first._optimizer_step(np.asarray([3.0, 4.0]))
        scaled_step, _ = second._optimizer_step(np.asarray([30.0, 40.0]))
        self.assertTrue(np.allclose(step, scaled_step))
        self.assertAlmostEqual(np.linalg.norm(step), 0.1)

        zero_step, zero_info = first._optimizer_step(np.zeros(2))
        self.assertTrue(np.allclose(zero_step, 0.5 * step))
        self.assertTrue(zero_info["clipup_zero_gradient"])
        self.assertEqual(
            zero_info["clipup_normalized_gradient_step_norm"], 0.0
        )

    def test_primary_clipup_rejects_conflicting_update_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "l2_coeff"):
            ClipUpES(num_params=2, l2_coeff=0.1)
        with self.assertRaisesRegex(ValueError, "gradient clipping"):
            ClipUpES(num_params=2, max_grad_norm=1.0)
        with self.assertRaisesRegex(ValueError, "parameter projection"):
            ClipUpES(num_params=2, max_param_norm=1.0)

    def test_one_iteration_trainer_smoke_is_fresh_and_velocity_bounded(self) -> None:
        class FakeObservationSpace:
            shape = (2,)

        class FakeActionSpace:
            shape = (1,)
            low = np.asarray([-1.0])
            high = np.asarray([1.0])

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            observation_space = FakeObservationSpace()
            action_space = FakeActionSpace()

            def reset(self, *, seed: int | None = None):
                self.steps = 0
                return np.zeros(2, dtype=np.float64), {}

            def step(self, action: np.ndarray):
                self.steps += 1
                action_value = float(np.asarray(action).ravel()[0])
                return (
                    np.zeros(2, dtype=np.float64),
                    1.0 + action_value,
                    self.steps >= 2,
                    False,
                    {},
                )

            def close(self) -> None:
                pass

        class SynchronousPool:
            def __init__(self, *, processes, initializer, initargs):
                self.processes = processes
                initializer(*initargs)

            def map(self, function, tasks):
                return [function(task) for task in tasks]

            def close(self) -> None:
                pass

            def join(self) -> None:
                pass

        config = _condition_config(
            {
                "env_name": "FakeContinuous-v0",
                "population_size": 2,
                "learning_rate": 0.05,
                "noise_std": 0.02,
                "l2_coeff": 0.0,
                "rank_fitness": True,
                "antithetic": True,
                "max_grad_norm": 0.0,
                "max_param_norm": None,
                "hidden_dims": [],
                "activation": "tanh",
                "output_activation": "tanh",
                "init_param_std": 0.1,
                "n_iterations": 1,
                "eval_episodes": 1,
                "eval_interval": 1,
                "log_interval": 1,
                "max_episode_steps": 2,
                "use_obs_norm": False,
                "replay_enabled": False,
                "buffer_size": 0,
                "reuse_fraction": 0.0,
                "common_rollout_seed": True,
                "clipup_momentum": 0.9,
                "clipup_max_speed": 0.1,
            },
            "clipup_es",
        )
        with tempfile.TemporaryDirectory() as output, patch(
            "experiments.train._make_env", side_effect=lambda *args, **kwargs: FakeEnv()
        ), patch("experiments.train.Pool", SynchronousPool):
            with redirect_stdout(io.StringIO()):
                train(config, seed=5, output_dir=output, n_workers=1)
            with open(os.path.join(output, "config.json"), encoding="utf-8") as stream:
                saved_config = json.load(stream)
            with open(os.path.join(output, "history.json"), encoding="utf-8") as stream:
                history = json.load(stream)

        self.assertEqual(saved_config["resolved_optimizer"]["method"], "clipup_es")
        self.assertEqual(len(history), 1)
        record = history[0]
        self.assertEqual(record["n_fresh"], 2)
        self.assertEqual(record["n_reused"], 0)
        self.assertFalse(record["used_replay"])
        self.assertLessEqual(record["clipup_velocity_norm"], 0.1 + 1e-12)
        self.assertLessEqual(record["step_norm"], 0.1 + 1e-12)


class ImplicitESTests(unittest.TestCase):
    def test_primary_conditions_resolve_to_distinct_certified_optimizers(self) -> None:
        expected = {
            "standard_es": (StandardES, None),
            "endpoint_implicit_es": (EndpointImplicitES, "picard_endpoint_implicit"),
            "linearized_implicit_es": (
                LinearizedImplicitES,
                "signed_diagonal_linearized_implicit",
            ),
        }
        for condition, (optimizer_type, solver_type) in expected.items():
            config = _condition_config(
                {
                    "replay_enabled": False,
                    "reuse_fraction": 0.0,
                    "buffer_size": 0,
                    "population_size": 8,
                    "noise_std": 0.2,
                },
                condition,
            )
            optimizer = make_optimizer(config, 3, None, 11)
            self.assertIsInstance(optimizer, optimizer_type)
            resolved = _resolved_optimizer_config(optimizer)
            self.assertFalse(resolved["trust_region"])
            self.assertFalse(resolved["replay_enabled"])
            if solver_type is not None:
                self.assertEqual(resolved["solver_type"], solver_type)
                self.assertEqual(resolved["implicit_damping"], 0.0)

    def test_primary_conditions_receive_identical_noise_streams(self) -> None:
        optimizers = [
            StandardES(num_params=3, population_size=8, seed=12),
            EndpointImplicitES(num_params=3, population_size=8, seed=12),
            LinearizedImplicitES(num_params=3, population_size=8, seed=12),
        ]
        for _ in range(3):
            batches = [optimizer.ask()[0] for optimizer in optimizers]
            self.assertTrue(np.array_equal(batches[0], batches[1]))
            self.assertTrue(np.array_equal(batches[0], batches[2]))

    def test_concave_conditions_resolve_structure_and_ema(self) -> None:
        policy = MLPPolicy(3, 2, hidden_dims=(4,))
        expected = {
            "concave_diagonal_curvature_es": (
                "diag", 0.0, policy.num_params, "stein_moment", "structured", None,
                "pooled_centered_ranks",
            ),
            "concave_block_curvature_es": (
                "block", 0.0, 2, "stein_moment", "structured", None,
                "pooled_centered_ranks",
            ),
            "concave_block_ema_curvature_es": (
                "block", 0.9, 2, "stein_moment", "structured", None,
                "pooled_centered_ranks",
            ),
            "concave_block_ema_isotropic_control_es": (
                "block", 0.9, 2, "stein_moment", "isotropic_norm_matched", None,
                "pooled_centered_ranks",
            ),
            "concave_block_ols_ema_curvature_es": (
                "block", 0.9, 2, "block_joint_ols", "structured", 1.645,
                "pooled_centered_ranks",
            ),
            "concave_block_lopo_u_stat": (
                "block", 0.0, 2, "stein_moment", "structured", None,
                "lopo_rank_u_statistic",
            ),
            "concave_block_lopo_u_stat_isotropic_control": (
                "block", 0.0, 2, "stein_moment", "isotropic_norm_matched", None,
                "lopo_rank_u_statistic",
            ),
        }
        for condition, values in expected.items():
            (
                structure,
                beta,
                components,
                estimator,
                attenuation,
                confidence,
                utility_mode,
            ) = values
            config = _condition_config(
                {
                    "replay_enabled": False,
                    "reuse_fraction": 0.0,
                    "buffer_size": 0,
                    "population_size": 16,
                    "noise_std": 0.2,
                },
                condition,
            )
            optimizer = make_optimizer(
                config, policy.num_params, policy, seed=13
            )
            self.assertIsInstance(optimizer, ConcaveCurvatureES)
            self.assertEqual(optimizer.curvature_structure, structure)
            self.assertEqual(optimizer.curvature_beta, beta)
            self.assertEqual(optimizer.curvature_estimator, estimator)
            self.assertEqual(optimizer.rank_utility_mode, utility_mode)
            self.assertEqual(optimizer.attenuation_mode, attenuation)
            self.assertEqual(optimizer.curvature_confidence_z, confidence)
            resolved = _resolved_optimizer_config(optimizer)
            self.assertEqual(resolved["curvature_projection"], "concave")
            self.assertFalse(resolved["curvature_clipping"])
            self.assertEqual(resolved["curvature_components"], components)
            self.assertEqual(resolved["curvature_estimator"], estimator)
            self.assertEqual(
                resolved["curvature_rank_utility_mode"], utility_mode
            )
            self.assertEqual(resolved["curvature_attenuation_mode"], attenuation)
            if utility_mode == "lopo_rank_u_statistic":
                self.assertEqual(
                    resolved["solver_type"],
                    (
                        "concave_projected_block_lopo_rank_u_statistic"
                        if attenuation == "structured"
                        else (
                            "concave_projected_block_lopo_rank_u_statistic_"
                            "isotropic_norm_control"
                        )
                    ),
                )
                self.assertEqual(
                    resolved["curvature_standard_error_method"],
                    "delete_one_antithetic_pair_order_two_u_statistic",
                )
                self.assertEqual(
                    resolved["curvature_standard_error_target"],
                    "raw_same_generation_block_u_statistic",
                )
                self.assertFalse(
                    resolved[
                        "curvature_standard_error_optimization_coverage_calibrated"
                    ]
                )
                self.assertFalse(
                    resolved[
                        "curvature_inference_assumptions_runtime_verified"
                    ]
                )
                self.assertTrue(
                    resolved[
                        "raw_lopo_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average"
                    ]
                )
                self.assertEqual(
                    resolved["raw_lopo_block_moment_endpoint_jacobian_scope"],
                    "at_proposal_frozen_lopo_utility_self_normalized_map_"
                    "raw_preprojection_block_average_of_diagonal",
                )
                self.assertFalse(resolved["full_endpoint_jacobian_operator_claim"])
                self.assertFalse(
                    resolved[
                        "projected_curvature_operator_endpoint_jacobian_claim"
                    ]
                )
                self.assertFalse(resolved["off_proposal_endpoint_jacobian_claim"])

        pooled_config = _condition_config(
            {
                "replay_enabled": False,
                "reuse_fraction": 0.0,
                "buffer_size": 0,
                "curvature_rank_utility_mode": "lopo_rank_u_statistic",
            },
            "concave_block_curvature_es",
        )
        self.assertEqual(
            pooled_config["curvature_rank_utility_mode"],
            "pooled_centered_ranks",
        )

        gradient_config = _condition_config(
            {
                "replay_enabled": False,
                "reuse_fraction": 0.0,
                "buffer_size": 0,
                "population_size": 16,
                "noise_std": 0.2,
                "rank_fitness": True,
                "curvature_beta": 0.0,
            },
            "lopo_gradient_only_es",
        )
        gradient_optimizer = make_optimizer(
            gradient_config, policy.num_params, policy, seed=13
        )
        self.assertIsInstance(gradient_optimizer, LOPOGradientES)
        gradient_resolved = _resolved_optimizer_config(gradient_optimizer)
        self.assertEqual(gradient_resolved["method"], "lopo_gradient_only")
        self.assertFalse(gradient_resolved["curvature_used"])
        self.assertEqual(gradient_resolved["curvature_components"], 0)
        self.assertEqual(
            gradient_resolved["solver_type"], "explicit_lopo_rank_gradient"
        )
        self.assertEqual(
            gradient_resolved["attribution_role"],
            "lopo_gradient_without_curvature",
        )
        self.assertEqual(
            gradient_resolved[
                "raw_lopo_block_moment_endpoint_jacobian_claim_applicability"
            ],
            "not_applicable_no_curvature_operator",
        )
        self.assertFalse(
            gradient_resolved["projected_curvature_operator_endpoint_jacobian_claim"]
        )

    def test_endpoint_log_ratio_matches_gaussian_density_difference(self) -> None:
        sigma = 0.3
        opt = EndpointImplicitES(
            num_params=2,
            population_size=4,
            noise_std=sigma,
            diagnostic_ratio_floor=1e-12,
            diagnostic_ratio_cap=1e12,
        )
        noise = np.asarray([[1.0, -0.5], [-1.0, 0.5], [0.2, 1.3], [-0.2, -1.3]])
        delta = np.asarray([0.07, -0.11])
        points = sigma * noise
        expected = (
            np.sum(points * points, axis=1)
            - np.sum((points - delta) ** 2, axis=1)
        ) / (2.0 * sigma**2)
        self.assertTrue(np.allclose(opt._endpoint_log_ratios(noise, delta), expected))

    def test_endpoint_gradient_at_proposal_matches_standard_es(self) -> None:
        noise = np.asarray(
            [[0.5, -1.0], [1.5, 0.2], [-0.5, 1.0], [-1.5, -0.2]],
            dtype=np.float64,
        )
        fitness = np.asarray([3.0, 1.0, -2.0, 0.5])
        endpoint = EndpointImplicitES(
            num_params=2,
            population_size=4,
            noise_std=0.2,
            rank_fitness=True,
        )
        standard = StandardES(
            num_params=2,
            population_size=4,
            noise_std=0.2,
            rank_fitness=True,
        )
        utilities, _ = endpoint._utilities(fitness)
        actual, stats = endpoint._endpoint_gradient(
            noise, utilities, np.zeros(2)
        )
        proposal_gradient, _ = endpoint._proposal_gradient(
            np.zeros(2), noise, utilities
        )
        self.assertTrue(
            np.array_equal(
                proposal_gradient,
                standard._gradient_from_utilities(noise, utilities),
            )
        )
        self.assertTrue(np.allclose(actual, standard._gradient(noise, fitness)))
        self.assertAlmostEqual(stats["endpoint_ess_ratio"], 1.0, places=10)
        self.assertEqual(stats["endpoint_clip_frac"], 0.0)

    def test_endpoint_gradient_recomputes_weighted_utility_center_off_center(self) -> None:
        noise = np.asarray(
            [[0.5, -1.0], [1.5, 0.2], [-0.5, 1.0], [-1.5, -0.2]],
            dtype=np.float64,
        )
        fitness = np.asarray([3.0, 1.0, -2.0, 0.5])
        sigma = 0.2
        delta = np.asarray([0.03, -0.02])
        opt = EndpointImplicitES(
            num_params=2, population_size=4, noise_std=sigma, rank_fitness=True
        )
        utilities, _ = opt._utilities(fitness)
        logits = noise @ (delta / sigma)
        weights = np.exp(logits - np.max(logits))
        weights /= np.sum(weights)
        weighted_mean = float(np.dot(weights, utilities))
        expected = (
            (weights * (utilities - weighted_mean))
            @ (noise - delta[None, :] / sigma)
            / sigma
        )
        uncentered = (
            (weights * utilities) @ (noise - delta[None, :] / sigma) / sigma
        )
        actual, stats = opt._endpoint_gradient(noise, utilities, delta)
        self.assertTrue(np.allclose(actual, expected))
        self.assertFalse(np.allclose(actual, uncentered))
        self.assertAlmostEqual(stats["endpoint_weighted_utility_mean"], weighted_mean)

    def test_matched_diagonal_is_endpoint_jacobian_diagonal(self) -> None:
        noise_half = np.asarray(
            [[0.5, -1.0], [1.5, 0.2], [-0.3, 0.7]], dtype=np.float64
        )
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        fitness = np.asarray([3.0, 1.0, -2.0, 0.5, 2.0, -1.0])
        endpoint = EndpointImplicitES(
            num_params=2,
            population_size=6,
            noise_std=0.2,
            diagnostic_ratio_floor=1e-100,
            diagnostic_ratio_cap=1e100,
        )
        linearized = LinearizedImplicitES(
            num_params=2, population_size=6, noise_std=0.2
        )
        utilities, _ = endpoint._utilities(fitness)
        ask_info = {
            "fresh_pair_plus": np.arange(3),
            "fresh_pair_minus": np.arange(3, 6),
        }
        diagonal, _ = linearized._matched_diagonal_hessian(
            noise, utilities, ask_info
        )
        finite_difference = np.zeros(2)
        step = 1e-6
        for coordinate in range(2):
            offset = np.zeros(2)
            offset[coordinate] = step
            grad_plus, _ = endpoint._endpoint_gradient(noise, utilities, offset)
            grad_minus, _ = endpoint._endpoint_gradient(noise, utilities, -offset)
            finite_difference[coordinate] = (
                grad_plus[coordinate] - grad_minus[coordinate]
            ) / (2.0 * step)
        self.assertTrue(np.allclose(diagonal, finite_difference, rtol=1e-5, atol=1e-5))

    def test_split_curvature_ranks_are_independent_between_pair_halves(self) -> None:
        noise_half = np.asarray(
            [
                [0.5, -1.0, 0.2],
                [1.5, 0.2, -0.8],
                [-0.3, 0.7, 1.1],
                [0.9, -0.6, -1.3],
            ],
            dtype=np.float64,
        )
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        ask_info = {
            "fresh_pair_plus": np.arange(4),
            "fresh_pair_minus": np.arange(4, 8),
        }
        fitness = np.asarray([3.0, 1.0, -2.0, 0.5, 2.0, -1.0, 1.5, -0.3])
        changed_second_half = fitness.copy()
        changed_second_half[[2, 3, 6, 7]] = [10.0, -10.0, -9.0, 9.0]

        linearized = LinearizedImplicitES(
            num_params=3,
            population_size=8,
            learning_rate=1e-4,
            noise_std=0.2,
        )
        linear_first, linear_second = (
            linearized._independent_split_diagonal_hessians(
                noise, fitness, ask_info
            )
        )
        changed_linear_first, changed_linear_second = (
            linearized._independent_split_diagonal_hessians(
                noise, changed_second_half, ask_info
            )
        )
        np.testing.assert_array_equal(linear_first, changed_linear_first)
        self.assertFalse(np.allclose(linear_second, changed_linear_second))
        _, linear_info = linearized.tell(
            np.zeros(3), noise, fitness, ask_info
        )
        self.assertEqual(
            linear_info["h_split_rank_semantics"],
            "independent_centered_ranks_per_disjoint_pair_half",
        )

        block = ConcaveCurvatureES(
            num_params=3,
            population_size=8,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 1), (1, 3)],
        )
        block_first, block_second = (
            block._independent_split_curvature_estimates(
                noise, fitness, ask_info
            )
        )
        changed_block_first, changed_block_second = (
            block._independent_split_curvature_estimates(
                noise, changed_second_half, ask_info
            )
        )
        np.testing.assert_array_equal(block_first, changed_block_first)
        self.assertFalse(np.allclose(block_second, changed_block_second))
        _, block_info = block.tell(np.zeros(3), noise, fitness, ask_info)
        self.assertEqual(
            block_info["h_split_rank_semantics"],
            "independent_centered_ranks_per_disjoint_pair_half",
        )
        self.assertEqual(len(block_info["h_split_first_components"]), 2)

    def test_picard_reports_actual_nonconvergence_without_trust_fallback(self) -> None:
        opt = EndpointImplicitES(
            num_params=3,
            population_size=8,
            learning_rate=30.0,
            noise_std=0.02,
            implicit_iterations=1,
            implicit_tolerance=1e-12,
            seed=4,
        )
        params = np.zeros(3)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - 0.2 * noise[:, 1]
        new_params, info = opt.tell(params, noise, fitness, ask_info)
        self.assertTrue(np.all(np.isfinite(new_params)))
        self.assertFalse(info["implicit_converged"])
        self.assertFalse(info["solve_success"])
        self.assertEqual(info["implicit_iterations"], 1)
        self.assertFalse(info["endpoint_ratio_clipping_enabled"])
        self.assertEqual(info["endpoint_clip_frac"], 0.0)
        self.assertLessEqual(info["implicit_endpoint_ess_ratio_min"], 1.0)
        self.assertNotIn("trust_active", info)
        self.assertNotIn("trust_scale", info)
        utilities, _ = opt._utilities(fitness)
        _, far_stats = opt._endpoint_gradient(noise, utilities, 1000.0 * noise[0])
        self.assertFalse(far_stats["endpoint_ratio_clipping_enabled"])
        self.assertEqual(far_stats["endpoint_clip_frac"], 0.0)
        self.assertAlmostEqual(far_stats["endpoint_weight_max"], 1.0)
        self.assertAlmostEqual(
            far_stats["endpoint_ess_ratio"], 1.0 / opt.population_size, places=10
        )

    def test_small_alpha_picard_converges_to_recomputed_equation(self) -> None:
        opt = EndpointImplicitES(
            num_params=3,
            population_size=8,
            learning_rate=1e-5,
            noise_std=0.2,
            implicit_iterations=100,
            implicit_tolerance=1e-10,
            seed=7,
        )
        params = np.zeros(3)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - 0.3 * noise[:, 1]
        utilities, _ = opt._utilities(fitness)
        new_params, info = opt.tell(params, noise, fitness, ask_info)
        delta = new_params - params
        endpoint_gradient, _ = opt._endpoint_gradient(noise, utilities, delta)
        residual = delta - opt.learning_rate * endpoint_gradient
        relative = np.linalg.norm(residual) / max(
            np.linalg.norm(delta),
            np.linalg.norm(opt.learning_rate * endpoint_gradient),
            1e-12,
        )
        self.assertTrue(info["implicit_converged"])
        self.assertTrue(info["solve_success"])
        self.assertLess(relative, opt.implicit_tolerance)
        self.assertAlmostEqual(relative, info["implicit_relative_residual"])

    def test_linearized_update_applies_signed_same_batch_system(self) -> None:
        opt = LinearizedImplicitES(
            num_params=3,
            population_size=8,
            learning_rate=2.0,
            noise_std=0.2,
            rank_fitness=True,
            seed=8,
        )
        params = np.asarray([0.2, -0.1, 0.3])
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - 0.5 * noise[:, 1] + 0.1 * noise[:, 2]
        utilities, _ = opt._utilities(fitness)
        _, gradient = opt._proposal_gradient(params, noise, utilities)
        hessian, _ = opt._matched_diagonal_hessian(noise, utilities, ask_info)
        diagonal = 1.0 - opt.learning_rate * hessian
        expected_step = opt.learning_rate * gradient / diagonal
        new_params, info = opt.tell(params, noise, fitness, ask_info)
        self.assertTrue(np.allclose(new_params - params, expected_step))
        self.assertEqual(info["solver_type"], "signed_diagonal_linearized_implicit")
        self.assertLess(info["linear_relative_residual"], 1e-12)
        self.assertEqual(info["hessian_pairs"], 4)
        self.assertNotIn("trust_active", info)

    def test_linearized_update_rejects_diagonal_below_numerical_threshold(self) -> None:
        opt = LinearizedImplicitES(
            num_params=2,
            population_size=4,
            learning_rate=1.0,
            noise_std=0.2,
            min_abs_diagonal=1e9,
            seed=3,
        )
        params = np.zeros(2)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - noise[:, 1]
        with self.assertRaisesRegex(FloatingPointError, "numerically singular"):
            opt.tell(params, noise, fitness, ask_info)

    def test_concave_block_curvature_is_mean_of_diagonal_curvature(self) -> None:
        noise_half = np.asarray(
            [
                [0.5, -1.0, 0.2, 1.4],
                [1.5, 0.2, -0.8, 0.7],
                [-0.3, 0.7, 1.1, -0.4],
                [0.9, -0.6, -1.3, 0.1],
            ],
            dtype=np.float64,
        )
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        fitness = np.asarray([3.0, 1.0, -2.0, 0.5, 2.0, -1.0, 1.5, -0.3])
        ask_info = {
            "fresh_pair_plus": np.arange(4),
            "fresh_pair_minus": np.arange(4, 8),
        }
        diagonal = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            noise_std=0.2,
            curvature_structure="diag",
        )
        block = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
        )
        utilities, _ = diagonal._utilities(fitness)
        diagonal_raw, diagonal_pairs = diagonal._matched_curvature_components(
            noise, utilities, ask_info
        )
        block_raw, block_pairs = block._matched_curvature_components(
            noise, utilities, ask_info
        )

        expected_raw = np.asarray(
            [np.mean(diagonal_raw[:2]), np.mean(diagonal_raw[2:])]
        )
        expected_pairs = np.column_stack(
            [np.mean(diagonal_pairs[:, :2], axis=1), np.mean(diagonal_pairs[:, 2:], axis=1)]
        )
        self.assertTrue(np.allclose(block_raw, expected_raw))
        self.assertTrue(np.allclose(block_pairs, expected_pairs))

    def test_concave_update_has_positive_denominator_and_no_amplification(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            learning_rate=3.0,
            noise_std=0.2,
            l2_coeff=0.1,
            implicit_damping=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_beta=0.0,
            seed=8,
        )
        params = np.asarray([0.2, -0.1, 0.3, 0.4])
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - 0.5 * noise[:, 1] + 0.1 * noise[:, 2]
        utilities, _ = opt._utilities(fitness)
        _, gradient = opt._proposal_gradient(params, noise, utilities)
        raw, _ = opt._matched_curvature_components(noise, utilities, ask_info)
        concave = np.maximum(-raw, 0.0)
        denominator_components = 1.0 + opt.learning_rate * (
            opt.implicit_damping + opt.l2_coeff + concave
        )
        denominator = opt._expand_components(denominator_components)
        expected_step = opt.learning_rate * gradient / denominator

        new_params, info = opt.tell(params, noise, fitness, ask_info)
        actual_step = new_params - params
        self.assertTrue(np.all(denominator >= 1.0))
        self.assertTrue(np.allclose(actual_step, expected_step))
        self.assertTrue(
            np.all(np.abs(actual_step) <= np.abs(opt.learning_rate * gradient) + 1e-12)
        )
        self.assertGreaterEqual(info["denominator_min"], 1.0)
        self.assertEqual(info["solver_type"], "concave_projected_block")
        self.assertLess(info["linear_relative_residual"], 1e-12)
        self.assertNotIn("implicit_converged", info)
        self.assertNotIn("trust_active", info)

    def test_concave_curvature_ema_is_bias_corrected(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_beta=0.9,
        )
        observation = np.asarray([-3.0, 2.0])
        for count in range(1, 5):
            corrected = opt._update_curvature_ema(observation)
            self.assertTrue(np.allclose(corrected, observation))
            self.assertEqual(opt.hessian_ema_count, count)

    def test_lopo_utilities_match_literal_pair_deletion_with_ties(self) -> None:
        pair_fitness = np.asarray(
            [
                [2.0, 2.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [-1.0, -1.0],
            ]
        )
        pair_count = len(pair_fitness)
        fitness = np.concatenate([pair_fitness[:, 0], pair_fitness[:, 1]])
        eps = np.asarray(
            [
                [0.5, -1.0, 0.3, 0.2],
                [1.2, 0.4, -0.7, 0.9],
                [-0.3, 0.8, 1.1, -0.4],
                [0.7, -0.2, 0.5, -1.3],
                [-1.1, 0.6, -0.9, 0.1],
            ],
            dtype=np.float64,
        )
        noise = np.concatenate([eps, -eps], axis=0)
        ask_info = {
            "fresh_pair_plus": np.arange(pair_count),
            "fresh_pair_minus": np.arange(pair_count, 2 * pair_count),
        }
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            rank_utility_mode="lopo_rank_u_statistic",
        )

        lopo, pooled, within_pair_sign, c_m = opt._lopo_rank_utilities(
            fitness, noise, ask_info
        )
        literal = np.empty_like(lopo)
        for pair_index in range(pair_count):
            own = [pair_index, pair_count + pair_index]
            reference = np.delete(fitness, own)
            for sample_index in own:
                less = np.sum(reference < fitness[sample_index])
                equal = np.sum(reference == fitness[sample_index])
                literal[sample_index] = (
                    (less + 0.5 * equal) / len(reference) - 0.5
                )

        np.testing.assert_allclose(lopo, literal, rtol=0.0, atol=1e-15)
        self.assertLessEqual(abs(float(np.sum(lopo))), 2e-15)
        expected_within_pair_sign = (
            (pair_fitness[:, 0] > pair_fitness[:, 1]).astype(float)
            - (pair_fitness[:, 0] < pair_fitness[:, 1]).astype(float)
        )
        np.testing.assert_array_equal(
            within_pair_sign, expected_within_pair_sign
        )
        max_float = np.finfo(np.float64).max
        with np.errstate(over="raise", invalid="raise"):
            extreme_sign = opt._comparison_sign(
                np.asarray([max_float, -max_float, max_float]),
                np.asarray([-max_float, max_float, max_float]),
            )
        np.testing.assert_array_equal(extreme_sign, [1, -1, 0])
        mate = np.concatenate(
            [expected_within_pair_sign, -expected_within_pair_sign]
        )
        np.testing.assert_allclose(
            lopo,
            (pooled - mate / (2.0 * (2.0 * pair_count - 1.0))) / c_m,
            rtol=0.0,
            atol=1e-15,
        )

        pooled_gradient = opt._gradient_from_utilities(noise, pooled)
        lopo_gradient = opt._gradient_from_utilities(noise, lopo)
        actual_remainder = pooled_gradient - c_m * lopo_gradient
        expected_remainder = np.sum(
            eps * expected_within_pair_sign[:, None], axis=0
        ) / (
            2.0
            * pair_count
            * opt.noise_std
            * (2.0 * pair_count - 1.0)
        )
        np.testing.assert_allclose(
            actual_remainder, expected_remainder, rtol=0.0, atol=1e-15
        )

        diagnostics = opt._lopo_gradient_estimate(
            np.zeros(4), noise, fitness, ask_info
        )[-1]
        self.assertTrue(diagnostics["lopo_zero_sum_identity_verified"])
        self.assertFalse(diagnostics["lopo_centering_operation_applied"])
        self.assertLessEqual(
            diagnostics["lopo_zero_sum_abs_sum"],
            diagnostics["lopo_zero_sum_tolerance"],
        )
        self.assertLessEqual(
            diagnostics[
                "lopo_at_proposal_unnormalized_minus_sn_block_gap_max_abs"
            ],
            diagnostics["lopo_at_proposal_endpoint_identity_tolerance"],
        )
        self.assertTrue(
            diagnostics[
                "lopo_raw_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average"
            ]
        )

    def test_lopo_raw_block_moment_matches_at_proposal_frozen_utility_sn_jacobian(self) -> None:
        eps = np.asarray(
            [
                [0.5, -1.0, 0.3, 0.2],
                [1.2, 0.4, -0.7, 0.9],
                [-0.3, 0.8, 1.1, -0.4],
                [0.7, -0.2, 0.5, -1.3],
                [-1.1, 0.6, -0.9, 0.1],
            ],
            dtype=np.float64,
        )
        pair_fitness = np.asarray(
            [
                [2.0, 2.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [-1.0, -1.0],
            ]
        )
        pair_count = len(eps)
        noise = np.concatenate([eps, -eps], axis=0)
        fitness = np.concatenate([pair_fitness[:, 0], pair_fitness[:, 1]])
        ask_info = {
            "fresh_pair_plus": np.arange(pair_count),
            "fresh_pair_minus": np.arange(pair_count, 2 * pair_count),
        }
        sigma = 0.3
        blocks = [slice(0, 2), slice(2, 4)]
        curvature = ConcaveCurvatureES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=sigma,
            curvature_structure="block",
            block_slices=blocks,
            rank_utility_mode="lopo_rank_u_statistic",
        )
        utilities, pooled, _, c_m = curvature._lopo_rank_utilities(
            fitness, noise, ask_info
        )
        raw_components = curvature._lopo_u_stat_curvature_estimate(
            noise,
            fitness,
            utilities,
            pooled,
            ask_info,
            c_m,
        )[0]
        endpoint = EndpointImplicitES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=sigma,
        )
        finite_difference_diagonal = np.zeros(4)
        finite_difference_step = 1e-6
        for coordinate in range(4):
            offset = np.zeros(4)
            offset[coordinate] = finite_difference_step
            plus_gradient, _ = endpoint._endpoint_gradient(
                noise, utilities, offset
            )
            minus_gradient, _ = endpoint._endpoint_gradient(
                noise, utilities, -offset
            )
            finite_difference_diagonal[coordinate] = (
                plus_gradient[coordinate] - minus_gradient[coordinate]
            ) / (2.0 * finite_difference_step)
        finite_difference_blocks = np.asarray(
            [np.mean(finite_difference_diagonal[block]) for block in blocks]
        )
        np.testing.assert_allclose(
            raw_components,
            finite_difference_blocks,
            rtol=2e-6,
            atol=2e-8,
        )

    def test_constant_utility_shift_exposes_sn_unnormalized_origin_gap(self) -> None:
        eps = np.asarray(
            [
                [0.5, -1.0, 0.3, 0.2],
                [1.2, 0.4, -0.7, 0.9],
                [-0.3, 0.8, 1.1, -0.4],
                [0.7, -0.2, 0.5, -1.3],
                [-1.1, 0.6, -0.9, 0.1],
            ],
            dtype=np.float64,
        )
        noise = np.concatenate([eps, -eps], axis=0)
        fitness = np.asarray(
            [2.0, 1.0, 1.0, 0.0, -1.0, 2.0, -1.0, 1.0, 2.0, -1.0]
        )
        pair_count = len(eps)
        ask_info = {
            "fresh_pair_plus": np.arange(pair_count),
            "fresh_pair_minus": np.arange(pair_count, 2 * pair_count),
        }
        sigma = 0.3
        blocks = [slice(0, 2), slice(2, 4)]
        curvature = ConcaveCurvatureES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=sigma,
            curvature_structure="block",
            block_slices=blocks,
            rank_utility_mode="lopo_rank_u_statistic",
        )
        utilities = curvature._lopo_rank_utilities(
            fitness, noise, ask_info
        )[0]
        shifted = utilities + 0.37
        endpoint = EndpointImplicitES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=sigma,
        )
        finite_difference_diagonal = np.zeros(4)
        finite_difference_step = 1e-6
        for coordinate in range(4):
            offset = np.zeros(4)
            offset[coordinate] = finite_difference_step
            plus_gradient, _ = endpoint._endpoint_gradient(
                noise, shifted, offset
            )
            minus_gradient, _ = endpoint._endpoint_gradient(
                noise, shifted, -offset
            )
            finite_difference_diagonal[coordinate] = (
                plus_gradient[coordinate] - minus_gradient[coordinate]
            ) / (2.0 * finite_difference_step)

        unnormalized_diagonal = np.mean(
            shifted[:, None] * (noise * noise - 1.0), axis=0
        ) / sigma**2
        second_moment = np.mean(noise * noise, axis=0)
        predicted_diagonal_gap = (
            float(np.mean(shifted)) * (second_moment - 1.0) / sigma**2
        )
        actual_gap = np.asarray(
            [
                np.mean(
                    unnormalized_diagonal[block]
                    - finite_difference_diagonal[block]
                )
                for block in blocks
            ]
        )
        predicted_gap = np.asarray(
            [np.mean(predicted_diagonal_gap[block]) for block in blocks]
        )
        np.testing.assert_allclose(
            actual_gap,
            predicted_gap,
            rtol=2e-6,
            atol=2e-8,
        )
        self.assertGreater(float(np.linalg.norm(predicted_gap)), 0.1)

    def test_lopo_u_stat_jackknife_matches_literal_pair_deletion(self) -> None:
        rng = np.random.RandomState(71)
        pair_count = 6
        pair_fitness = np.asarray(
            [
                [2.0, 1.0],
                [1.0, 1.0],
                [-1.0, 2.0],
                [0.0, -1.0],
                [2.0, 0.0],
                [-1.0, -1.0],
            ]
        )
        eps = rng.randn(pair_count, 4)
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=2 * pair_count,
            noise_std=0.3,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            rank_utility_mode="lopo_rank_u_statistic",
        )
        block_features = opt._block_quadratic_features(eps)
        (
            estimate,
            standard_error,
            leave_one_pair_out,
            row_kernel_sums,
            total_kernel_sum,
            a_matrix,
        ) = opt._lopo_order_two_u_statistic_jackknife(
            pair_fitness, block_features
        )

        noise = np.concatenate([eps, -eps], axis=0)
        fitness = np.concatenate([pair_fitness[:, 0], pair_fitness[:, 1]])
        ask_info = {
            "fresh_pair_plus": np.arange(pair_count),
            "fresh_pair_minus": np.arange(pair_count, 2 * pair_count),
        }
        lopo, pooled, _, c_m = opt._lopo_rank_utilities(
            fitness, noise, ask_info
        )
        lopo_raw = np.mean(
            (lopo[:pair_count] + lopo[pair_count:])[:, None]
            * block_features
            / (2.0 * opt.noise_std**2),
            axis=0,
        )
        pooled_raw = np.mean(
            (pooled[:pair_count] + pooled[pair_count:])[:, None]
            * block_features
            / (2.0 * opt.noise_std**2),
            axis=0,
        )
        np.testing.assert_allclose(estimate, lopo_raw, atol=2e-14)
        np.testing.assert_allclose(lopo_raw, pooled_raw / c_m, atol=2e-14)

        direct_leaves = []
        for deleted in range(pair_count):
            retained = np.arange(pair_count) != deleted
            retained_fitness = pair_fitness[retained]
            retained_features = block_features[retained]
            retained_count = len(retained_fitness)
            flat_fitness = np.concatenate(
                [retained_fitness[:, 0], retained_fitness[:, 1]]
            )
            pair_utilities = []
            for pair_index in range(retained_count):
                own = [pair_index, retained_count + pair_index]
                reference = np.delete(flat_fitness, own)
                utilities = []
                for sample_index in own:
                    less = np.sum(reference < flat_fitness[sample_index])
                    equal = np.sum(reference == flat_fitness[sample_index])
                    utilities.append(
                        (less + 0.5 * equal) / len(reference) - 0.5
                    )
                pair_utilities.append(sum(utilities))
            direct_leaves.append(
                np.mean(
                    np.asarray(pair_utilities)[:, None]
                    * retained_features
                    / (2.0 * opt.noise_std**2),
                    axis=0,
                )
            )
        direct_leaves = np.asarray(direct_leaves)
        np.testing.assert_allclose(
            leave_one_pair_out, direct_leaves, rtol=0.0, atol=2e-14
        )
        np.testing.assert_allclose(
            np.mean(leave_one_pair_out, axis=0),
            estimate,
            rtol=0.0,
            atol=2e-14,
        )
        expected_se = np.sqrt(
            (pair_count - 1.0)
            / pair_count
            * np.sum((direct_leaves - estimate) ** 2, axis=0)
        )
        np.testing.assert_allclose(
            standard_error, expected_se, rtol=0.0, atol=2e-14
        )

        upper_i, upper_j = np.triu_indices(pair_count, 1)
        upper_kernels = (
            a_matrix[upper_i, upper_j, None]
            * (block_features[upper_i] - block_features[upper_j])
            / (16.0 * opt.noise_std**2)
        )
        direct_rows = np.zeros_like(row_kernel_sums)
        np.add.at(direct_rows, upper_i, upper_kernels)
        np.add.at(direct_rows, upper_j, upper_kernels)
        np.testing.assert_allclose(row_kernel_sums, direct_rows, atol=1e-14)
        np.testing.assert_allclose(
            total_kernel_sum, np.sum(upper_kernels, axis=0), atol=1e-14
        )
        np.testing.assert_array_equal(a_matrix, -a_matrix.T)
        np.testing.assert_array_equal(np.diag(a_matrix), np.zeros(pair_count))

    def test_lopo_all_equal_returns_have_zero_curvature_and_se(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            rank_utility_mode="lopo_rank_u_statistic",
            seed=19,
        )
        params = np.asarray([0.2, -0.1, 0.3, -0.4])
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        updated, info = opt.tell(
            params, noise, np.ones(opt.population_size), ask_info
        )
        np.testing.assert_array_equal(updated, params)
        np.testing.assert_array_equal(
            info["curvature_raw_components"], [0.0, 0.0]
        )
        np.testing.assert_array_equal(
            info["curvature_same_generation_se_components"], [0.0, 0.0]
        )
        self.assertEqual(info["lopo_rank_utility_sum"], 0.0)
        self.assertEqual(info["lopo_rank_utility_mean"], 0.0)
        self.assertTrue(info["lopo_zero_sum_identity_verified"])
        self.assertFalse(info["lopo_centering_operation_applied"])
        self.assertEqual(
            info[
                "lopo_at_proposal_unnormalized_minus_sn_block_gap_max_abs"
            ],
            0.0,
        )
        self.assertTrue(
            info["lopo_at_proposal_sn_unnormalized_identity_verified"]
        )
        self.assertTrue(info["lopo_raw_identities_verified"])

    def test_lopo_attribution_arms_share_noise_utility_and_gradient(self) -> None:
        common = {
            "num_params": 4,
            "population_size": 80,
            "learning_rate": 0.5,
            "noise_std": 0.25,
            "block_slices": [(0, 2), (2, 4)],
            "seed": 91,
        }
        gradient_only = LOPOGradientES(**common)
        structured = ConcaveCurvatureES(
            **common,
            curvature_structure="block",
            rank_utility_mode="lopo_rank_u_statistic",
            attenuation_mode="structured",
        )
        isotropic = ConcaveCurvatureES(
            **common,
            curvature_structure="block",
            rank_utility_mode="lopo_rank_u_statistic",
            attenuation_mode="isotropic_norm_matched",
        )
        optimizers = (gradient_only, structured, isotropic)
        params = np.asarray([0.6, -0.4, 0.3, -0.2])
        batches = []
        ask_infos = []
        for optimizer in optimizers:
            optimizer.current_params = params.copy()
            noise, ask_info = optimizer.ask()
            batches.append(noise)
            ask_infos.append(ask_info)
        np.testing.assert_array_equal(batches[0], batches[1])
        np.testing.assert_array_equal(batches[0], batches[2])

        candidates = params + common["noise_std"] * batches[0]
        fitness = -(
            0.5
            * np.sum(
                (candidates[:, :2] - np.asarray([0.1, -0.2])) ** 2,
                axis=1,
            )
            + 5.0
            * np.sum(
                (candidates[:, 2:] - np.asarray([-0.2, 0.15])) ** 2,
                axis=1,
            )
        )
        lopo_states = [
            optimizer._lopo_gradient_estimate(
                params, noise, fitness, ask_info
            )
            for optimizer, noise, ask_info in zip(
                optimizers, batches, ask_infos, strict=True
            )
        ]
        for state in lopo_states[1:]:
            np.testing.assert_array_equal(state[0], lopo_states[0][0])
            np.testing.assert_array_equal(state[4], lopo_states[0][4])
            self.assertEqual(
                state[5]["lopo_utility_semantics"],
                lopo_states[0][5]["lopo_utility_semantics"],
            )

        updates = []
        infos = []
        for optimizer, noise, ask_info in zip(
            optimizers, batches, ask_infos, strict=True
        ):
            updated, info = optimizer.tell(
                params, noise, fitness, ask_info
            )
            updates.append(updated - params)
            infos.append(info)
            self.assertFalse(info["used_replay"])
            self.assertEqual(info["n_reused"], 0)
            self.assertTrue(info["lopo_zero_sum_identity_verified"])

        explicit_step = common["learning_rate"] * lopo_states[0][4]
        np.testing.assert_allclose(updates[0], explicit_step, atol=1e-15)
        self.assertEqual(infos[0]["solver_type"], "explicit_lopo_rank_gradient")
        self.assertFalse(infos[0]["curvature_used"])

        self.assertGreater(
            float(np.ptp(infos[1]["denominator_components"])), 0.1
        )
        self.assertGreater(
            float(np.linalg.norm(updates[1] - updates[2])), 1e-3
        )
        np.testing.assert_array_equal(
            infos[1]["curvature_raw_components"],
            infos[2]["curvature_raw_components"],
        )
        np.testing.assert_array_equal(
            infos[1]["curvature_same_generation_se_components"],
            infos[2]["curvature_same_generation_se_components"],
        )
        self.assertAlmostEqual(
            float(np.linalg.norm(updates[1])),
            float(np.linalg.norm(updates[2])),
            places=14,
        )
        np.testing.assert_allclose(
            updates[2],
            infos[2]["isotropic_attenuation_scale"] * explicit_step,
            rtol=1e-14,
            atol=1e-15,
        )
        self.assertEqual(
            infos[1]["solver_type"],
            "concave_projected_block_lopo_rank_u_statistic",
        )
        self.assertEqual(
            infos[2]["solver_type"],
            "concave_projected_block_lopo_rank_u_statistic_"
            "isotropic_norm_control",
        )

    def test_lopo_quadratic_smoke_is_fresh_stable_and_auditable(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=40,
            learning_rate=0.5,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_beta=0.0,
            curvature_confidence_z=1.0,
            rank_utility_mode="lopo_rank_u_statistic",
            seed=23,
        )
        params = np.asarray([0.4, -0.3, 0.2, -0.1])
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        candidates = params + opt.noise_std * noise
        fitness = -(
            0.5 * np.sum(candidates[:, :2] ** 2, axis=1)
            + 2.0 * np.sum(candidates[:, 2:] ** 2, axis=1)
        )
        lopo, _, _, _ = opt._lopo_rank_utilities(fitness, noise, ask_info)
        explicit_gradient = opt._gradient_from_utilities(noise, lopo)

        updated, info = opt.tell(params, noise, fitness, ask_info)
        actual_step = updated - params
        self.assertTrue(np.all(np.isfinite(updated)))
        self.assertTrue(np.all(np.isfinite(info["curvature_raw_components"])))
        self.assertTrue(
            np.all(np.isfinite(info["curvature_same_generation_se_components"]))
        )
        self.assertGreaterEqual(info["denominator_min"], 1.0)
        self.assertLessEqual(
            np.linalg.norm(actual_step),
            np.linalg.norm(opt.learning_rate * explicit_gradient) + 1e-12,
        )
        self.assertEqual(
            info["solver_type"],
            "concave_projected_block_lopo_rank_u_statistic",
        )
        self.assertEqual(
            info["lopo_standard_error_scope"],
            "componentwise_asymptotic_non_simultaneous",
        )
        self.assertEqual(
            info["lopo_standard_error_target"],
            "raw_same_generation_block_u_statistic",
        )
        self.assertFalse(
            info["lopo_standard_error_optimization_coverage_calibrated"]
        )
        self.assertFalse(info["used_replay"])
        self.assertEqual(info["n_reused"], 0)
        self.assertTrue(info["lopo_raw_identities_verified"])

    def test_lopo_rejects_replay_and_invalid_pair_metadata(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=8,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            rank_utility_mode="lopo_rank_u_statistic",
            seed=29,
        )
        params = np.zeros(4)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        replay_info = dict(ask_info)
        replay_info["is_reused"] = np.asarray(
            [True] + [False] * (opt.population_size - 1)
        )
        with self.assertRaisesRegex(ValueError, "replayed samples"):
            opt.tell(params, noise, noise[:, 0], replay_info)

        incomplete_info = dict(ask_info)
        incomplete_info["fresh_pair_minus"] = ask_info["fresh_pair_minus"][:-1]
        with self.assertRaisesRegex(ValueError, "complete antithetic pairs"):
            opt.tell(params, noise, noise[:, 0], incomplete_info)

        broken_noise = noise.copy()
        broken_noise[int(ask_info["fresh_pair_minus"][0]), 0] += 0.1
        with self.assertRaisesRegex(ValueError, "does not match"):
            opt.tell(params, broken_noise, noise[:, 0], ask_info)

    def test_block_joint_ols_recovers_layer_isotropic_quadratic(self) -> None:
        rng = np.random.RandomState(23)
        sigma = 0.2
        eps = rng.randn(400, 5)
        block_curvature = np.asarray([-3.5, 1.25])
        pair_utility = (
            4.0
            + sigma**2
            * (
                block_curvature[0] * np.sum(eps[:, :2] ** 2, axis=1)
                + block_curvature[1] * np.sum(eps[:, 2:] ** 2, axis=1)
            )
        )
        opt = ConcaveCurvatureES(
            num_params=5,
            population_size=40,
            noise_std=sigma,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 5)],
            curvature_estimator="block_joint_ols",
        )

        estimate, standard_error, diagnostics = opt._fit_block_joint_ols(
            eps, pair_utility
        )

        self.assertTrue(np.allclose(estimate, block_curvature, atol=1e-12))
        self.assertTrue(np.all(standard_error < 1e-12))
        self.assertEqual(diagnostics["regression_rank"], 3)
        self.assertEqual(diagnostics["regression_parameters"], 3)
        self.assertAlmostEqual(diagnostics["regression_r_squared"], 1.0)

    def test_block_joint_ols_split_diagnostics_refit_each_half(self) -> None:
        rng = np.random.RandomState(29)
        eps = rng.randn(40, 4)
        pair_utility = rng.randn(40)
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=40,
            noise_std=0.15,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_estimator="block_joint_ols",
        )

        split_estimates = opt._split_curvature_estimates(
            eps, pair_utility, None
        )
        self.assertIsNotNone(split_estimates)
        first, second = split_estimates
        expected_first, _, _ = opt._fit_block_joint_ols(
            eps[:20], pair_utility[:20]
        )
        expected_second, _, _ = opt._fit_block_joint_ols(
            eps[20:], pair_utility[20:]
        )
        self.assertTrue(np.allclose(first, expected_first))
        self.assertTrue(np.allclose(second, expected_second))

    def test_block_joint_ols_confidence_gate_rejects_null_curvature(self) -> None:
        rng = np.random.RandomState(31)
        eps = rng.randn(500, 4)
        pair_utility = rng.randn(500)
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=40,
            noise_std=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_estimator="block_joint_ols",
            curvature_confidence_z=5.0,
        )
        estimate, standard_error, _ = opt._fit_block_joint_ols(
            eps, pair_utility
        )
        concave, upper_bound = opt._confidence_adjusted_concave(
            estimate, standard_error
        )

        self.assertTrue(np.all(upper_bound > 0.0))
        self.assertTrue(np.allclose(concave, 0.0))

    def test_block_joint_ols_update_is_stable_and_reports_state(self) -> None:
        opt = ConcaveCurvatureES(
            num_params=4,
            population_size=40,
            learning_rate=5.0,
            noise_std=0.2,
            l2_coeff=0.1,
            implicit_damping=0.2,
            curvature_structure="block",
            block_slices=[(0, 2), (2, 4)],
            curvature_beta=0.9,
            curvature_estimator="block_joint_ols",
            curvature_confidence_z=1.0,
            seed=17,
        )
        params = np.asarray([0.2, -0.1, 0.3, 0.4])
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = (
            noise[:, 0]
            - 0.4 * noise[:, 2]
            - 0.2 * np.sum(noise[:, :2] ** 2, axis=1)
        )
        utilities, _ = opt._utilities(fitness)
        _, gradient = opt._proposal_gradient(params, noise, utilities)

        new_params, info = opt.tell(params, noise, fitness, ask_info)
        actual_step = new_params - params
        explicit_step = opt.learning_rate * gradient

        self.assertTrue(np.all(np.isfinite(new_params)))
        self.assertTrue(
            np.all(np.abs(actual_step) <= np.abs(explicit_step) + 1e-12)
        )
        self.assertGreaterEqual(info["denominator_min"], 1.0)
        self.assertLessEqual(info["step_norm_ratio"], 1.0 + 1e-12)
        self.assertEqual(info["curvature_estimator"], "block_joint_ols")
        self.assertEqual(info["curvature_step_state"], "bias_corrected_ema")
        self.assertFalse(opt.curvature_same_generation)
        self.assertFalse(info["curvature_same_generation"])
        self.assertTrue(info["curvature_confidence_gate_enabled"])
        self.assertEqual(len(info["curvature_same_generation_components"]), 2)
        self.assertEqual(len(info["curvature_same_generation_se_components"]), 2)
        self.assertEqual(len(info["curvature_step_state_components"]), 2)
        self.assertGreater(info["regression_residual_dof"], 0)
        self.assertTrue(info["h_split_available"])

    def test_isotropic_control_matches_structured_norm_not_direction(self) -> None:
        common = {
            "num_params": 4,
            "population_size": 8,
            "learning_rate": 0.5,
            "noise_std": 0.2,
            "rank_fitness": True,
            "antithetic": True,
            "block_slices": [slice(0, 2), slice(2, 4)],
            "curvature_structure": "block",
            "curvature_beta": 0.0,
            "seed": 17,
        }
        structured = ConcaveCurvatureES(**common)
        control = ConcaveCurvatureES(
            **common, attenuation_mode="isotropic_norm_matched"
        )
        params = np.asarray([0.2, -0.1, 0.3, -0.4], dtype=np.float64)
        structured.current_params = params.copy()
        control.current_params = params.copy()
        noise, ask_info = structured.ask()
        control.current_params = params.copy()
        control_noise, control_ask_info = control.ask()
        np.testing.assert_array_equal(control_noise, noise)
        fitness = (
            noise[:, 0]
            + 0.5 * noise[:, 2]
            - 0.3 * noise[:, 0] ** 2
            - noise[:, 2] ** 2
        )

        structured_params, structured_info = structured.tell(
            params, noise, fitness, ask_info
        )
        control_params, control_info = control.tell(
            params, control_noise, fitness, control_ask_info
        )
        structured_step = structured_params - params
        control_step = control_params - params
        explicit_step = control.learning_rate * control._gradient(
            control_noise, fitness
        )

        self.assertAlmostEqual(
            np.linalg.norm(control_step), np.linalg.norm(structured_step), places=12
        )
        scale = control_info["isotropic_attenuation_scale"]
        np.testing.assert_allclose(control_step, scale * explicit_step)
        self.assertLessEqual(scale, 1.0)
        self.assertGreater(scale, 0.0)
        self.assertLessEqual(
            control_info["attenuation_norm_match_relative_error"], 1e-12
        )
        self.assertEqual(
            control_info["solver_type"],
            "concave_projected_block_isotropic_attenuation_control",
        )
        self.assertEqual(structured_info["curvature_attenuation_mode"], "structured")
        self.assertGreater(
            np.ptp(np.asarray(structured_info["denominator_components"])), 1e-6
        )
        self.assertFalse(np.allclose(control_step, structured_step))

    def test_concave_curvature_validates_structure_blocks_and_beta(self) -> None:
        common = {"num_params": 4, "population_size": 8}
        for beta in (-0.1, 1.0, np.nan):
            with self.assertRaisesRegex(ValueError, "curvature_beta"):
                ConcaveCurvatureES(**common, curvature_beta=beta)
        with self.assertRaisesRegex(ValueError, "curvature_structure"):
            ConcaveCurvatureES(**common, curvature_structure="global")
        with self.assertRaisesRegex(ValueError, "curvature_estimator"):
            ConcaveCurvatureES(**common, curvature_estimator="unknown")
        with self.assertRaisesRegex(ValueError, "attenuation_mode"):
            ConcaveCurvatureES(**common, attenuation_mode="unknown")
        with self.assertRaisesRegex(ValueError, "requires block structure"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="diag",
                attenuation_mode="isotropic_norm_matched",
            )
        with self.assertRaisesRegex(ValueError, "requires block structure"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="diag",
                curvature_estimator="block_joint_ols",
            )
        with self.assertRaisesRegex(ValueError, "requires block_joint_ols"):
            ConcaveCurvatureES(
                **common,
                curvature_confidence_z=1.96,
            )
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                curvature_estimator="block_joint_ols",
                curvature_confidence_z=-1.0,
            )
        with self.assertRaisesRegex(ValueError, "rank_utility_mode"):
            ConcaveCurvatureES(**common, rank_utility_mode="unknown")
        with self.assertRaisesRegex(ValueError, "requires block curvature"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="diag",
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "requires stein_moment"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                curvature_estimator="block_joint_ols",
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "at least three"):
            ConcaveCurvatureES(
                num_params=4,
                population_size=4,
                curvature_structure="block",
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "even antithetic population"):
            ConcaveCurvatureES(
                **common,
                antithetic=False,
                curvature_structure="block",
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "rank_fitness"):
            ConcaveCurvatureES(
                **common,
                rank_fitness=False,
                curvature_structure="block",
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "curvature_beta=0"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                curvature_beta=0.9,
                curvature_confidence_z=1.0,
                rank_utility_mode="lopo_rank_u_statistic",
            )
        with self.assertRaisesRegex(ValueError, "each split half"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                block_slices=[(0, 2), (2, 4)],
                curvature_estimator="block_joint_ols",
            )
        with self.assertRaisesRegex(ValueError, "full partition"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                block_slices=[(0, 3), (2, 4)],
            )
        with self.assertRaisesRegex(ValueError, "full partition"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                block_slices=[(0, 2), (3, 4)],
            )
        with self.assertRaisesRegex(ValueError, "slice steps"):
            ConcaveCurvatureES(
                **common,
                curvature_structure="block",
                block_slices=[slice(0, 4, 2)],
            )

        normalized = ConcaveCurvatureES(
            **common,
            curvature_structure="block",
            block_slices=[(2, 4), (0, 2)],
        )
        self.assertEqual(
            [(item.start, item.stop) for item in normalized.block_slices],
            [(0, 2), (2, 4)],
        )

    def test_implicit_methods_reject_replay_metadata(self) -> None:
        opt = EndpointImplicitES(num_params=2, population_size=4)
        noise = np.zeros((4, 2))
        ask_info = {
            "ask_params": np.zeros(2),
            "is_reused": np.asarray([True, False, False, False]),
            "n_reused": 1,
        }
        with self.assertRaisesRegex(ValueError, "replayed"):
            opt.tell(np.zeros(2), noise, np.zeros(4), ask_info)


class DIIWESTests(unittest.TestCase):
    @staticmethod
    def _one_dimensional_step(alpha: float) -> tuple[float, dict[str, object]]:
        opt = DIIWES(
            num_params=1,
            population_size=2,
            learning_rate=alpha,
            noise_std=0.1,
            reuse_fraction=0.0,
            scalar_damping=0.5,
            use_curvature=True,
            curvature_beta=0.0,
            rank_fitness=False,
            seed=0,
        )
        opt.hessian_ema[:] = -2.0
        opt.hessian_ema_count = 1
        params = np.zeros(1)
        opt.current_params = params.copy()
        noise = np.asarray([[1.0], [-1.0]])
        fitness = np.asarray([1.0, -1.0])
        ask_info = {
            "ask_params": params.copy(),
            "is_reused": np.zeros(2, dtype=bool),
            "fresh_pair_plus": np.asarray([], dtype=int),
            "fresh_pair_minus": np.asarray([], dtype=int),
        }
        new_params, info = opt.tell(params, noise, fitness, ask_info)
        return float(new_params[0]), info

    def test_closed_form_solve_is_exact_and_saturates(self) -> None:
        steps = []
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            step, info = self._one_dimensional_step(alpha)
            steps.append(step)
            self.assertTrue(info["solve_success"])
            self.assertLess(float(info["linear_relative_residual"]), 1e-12)
            self.assertEqual(info["solver_type"], "projected_diagonal_closed_form")
            self.assertTrue(info["signed_system_positive"])
            self.assertNotIn("trust_active", info)
            self.assertNotIn("multiplier_floor_frac", info)

        self.assertTrue(np.all(np.diff(steps) > 0.0))
        self.assertLess(steps[-1], 4.0)
        self.assertAlmostEqual(steps[-1], 4.0, delta=0.01)

    def test_signed_system_instability_is_logged_without_applying_it(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=2,
            learning_rate=1.0,
            noise_std=0.1,
            reuse_fraction=0.0,
            scalar_damping=0.5,
            use_curvature=True,
            curvature_beta=0.0,
            rank_fitness=False,
        )
        opt.hessian_ema[:] = 2.0
        opt.hessian_ema_count = 1
        params = np.zeros(1)
        ask_info = {
            "ask_params": params.copy(),
            "is_reused": np.zeros(2, dtype=bool),
            "fresh_pair_plus": np.asarray([], dtype=int),
            "fresh_pair_minus": np.asarray([], dtype=int),
        }
        _, info = opt.tell(
            params,
            np.asarray([[1.0], [-1.0]]),
            np.asarray([1.0, -1.0]),
            ask_info,
        )
        self.assertTrue(info["solve_success"])
        self.assertFalse(info["signed_system_positive"])
        self.assertEqual(info["signed_linear_nonpositive_diagonal_frac"], 1.0)
        self.assertLess(info["signed_linear_diagonal_min"], 0.0)

    def test_production_diagonal_curvature_recovers_quadratic(self) -> None:
        rng = np.random.RandomState(7)
        hessian_diag = np.asarray([-4.0, -1.0, 2.0])
        theta = np.asarray([0.3, -0.2, 0.4])
        sigma = 0.2
        eps = rng.randn(100_000, 3)

        def objective(points: np.ndarray) -> np.ndarray:
            return 0.5 * np.sum(hessian_diag[None, :] * points * points, axis=1)

        plus = objective(theta[None, :] + sigma * eps)
        minus = objective(theta[None, :] - sigma * eps)
        noise = np.concatenate([eps, -eps], axis=0)
        fitness = np.concatenate([plus, minus], axis=0)
        n_pairs = len(eps)
        ask_info = {
            "fresh_pair_plus": np.arange(n_pairs),
            "fresh_pair_minus": np.arange(n_pairs, 2 * n_pairs),
        }
        opt = DIIWES(num_params=3, noise_std=sigma, reuse_fraction=0.0, curvature_beta=0.0)
        estimate, count = opt._estimate_fresh_curvature(
            noise,
            fitness,
            ask_info,
            sigma,
            center_f_for_curv=float(objective(theta[None, :])[0]),
        )

        self.assertEqual(count, n_pairs)
        self.assertIsNotNone(estimate)
        self.assertTrue(np.allclose(estimate, hessian_diag, rtol=0.06, atol=0.08))

    def test_bias_corrected_ema_recovers_constant_first_observation(self) -> None:
        opt = DIIWES(num_params=3, curvature_beta=0.9)
        expected = np.asarray([-2.0, 0.5, 4.0])
        opt.hessian_ema = (1.0 - opt.curvature_beta) * expected
        opt.hessian_ema_count = 1
        self.assertTrue(np.allclose(opt._hessian_for_step(), expected))

    def test_matched_rank_curvature_is_reward_scale_invariant(self) -> None:
        def run(scale: float, offset: float) -> tuple[np.ndarray, dict[str, object]]:
            opt = DIIWES(
                num_params=3,
                population_size=20,
                learning_rate=2.0,
                noise_std=0.1,
                reuse_fraction=0.0,
                curvature_beta=0.0,
                curvature_fitness="matched",
                rank_fitness=True,
                seed=4,
            )
            params = np.zeros(3)
            opt.current_params = params.copy()
            noise, ask_info = opt.ask()
            base_fitness = noise[:, 0] - 0.3 * noise[:, 1] + 0.1 * noise[:, 2]
            return opt.tell(params, noise, scale * base_fitness + offset, ask_info)

        theta_a, info_a = run(1.0, 0.0)
        theta_b, info_b = run(100.0, 17.0)
        self.assertTrue(np.allclose(theta_a, theta_b))
        self.assertAlmostEqual(float(info_a["curv_mean"]), float(info_b["curv_mean"]))
        self.assertTrue(info_a["curvature_matches_gradient"])

    def test_ess_fallback_matches_fresh_only_rank_gradient(self) -> None:
        params = np.zeros(1)
        fresh_noise = np.asarray([[-1.0], [0.0], [1.0]])
        fresh_fitness = np.asarray([0.0, 1.0, 2.0])

        with_replay = DIIWES(
            num_params=1,
            population_size=4,
            learning_rate=0.1,
            noise_std=0.1,
            buffer_size=8,
            reuse_fraction=0.0,
            scalar_damping=0.0,
            use_curvature=False,
            rank_fitness=True,
            ess_min_ratio=0.9,
        )
        replay_noise = np.concatenate([np.asarray([[10.0]]), fresh_noise], axis=0)
        replay_info = {
            "ask_params": params.copy(),
            "is_reused": np.asarray([True, False, False, False]),
            "buffer_fitness": np.asarray([100.0]),
            "buffer_sigma_old": np.asarray([0.1]),
            "buffer_dist_old_sq": np.asarray([0.0]),
            "fresh_pair_plus": np.asarray([], dtype=int),
            "fresh_pair_minus": np.asarray([], dtype=int),
        }
        replay_theta, replay_step = with_replay.tell(
            params,
            replay_noise,
            fresh_fitness,
            replay_info,
        )

        fresh_only = DIIWES(
            num_params=1,
            population_size=3,
            learning_rate=0.1,
            noise_std=0.1,
            buffer_size=0,
            reuse_fraction=0.0,
            scalar_damping=0.0,
            use_curvature=False,
            rank_fitness=True,
            ess_min_ratio=0.9,
        )
        fresh_info = {
            "ask_params": params.copy(),
            "is_reused": np.zeros(3, dtype=bool),
            "fresh_pair_plus": np.asarray([], dtype=int),
            "fresh_pair_minus": np.asarray([], dtype=int),
        }
        fresh_theta, fresh_step = fresh_only.tell(
            params,
            fresh_noise,
            fresh_fitness,
            fresh_info,
        )

        self.assertFalse(replay_step["used_replay"])
        self.assertAlmostEqual(float(replay_step["replay_weight_mass"]), 0.0)
        self.assertTrue(np.allclose(replay_theta, fresh_theta))
        self.assertAlmostEqual(float(replay_step["grad_norm"]), float(fresh_step["grad_norm"]))

    def test_replay_mass_gate_rejects_low_mass_overlapping_candidates(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=4,
            noise_std=0.1,
            reuse_fraction=0.5,
            min_replay_weight_mass=0.1,
            buffer_sampling="random",
            seed=2,
        )
        opt._add_fresh_to_buffer(
            np.zeros(1),
            np.zeros((2, 1)),
            np.asarray([0.0, 1.0]),
        )
        target_ratio = 0.05
        opt.current_params = np.asarray([opt.noise_std * np.sqrt(-2.0 * np.log(target_ratio))])
        _, ask_info = opt.ask()

        expected_mass = 2.0 * target_ratio / (2.0 * target_ratio + 2.0)
        self.assertEqual(ask_info["n_replay_candidates"], 2)
        self.assertEqual(ask_info["n_replay_overlapping"], 2)
        self.assertTrue(ask_info["replay_mass_rejected"])
        self.assertEqual(ask_info["n_reused"], 0)
        self.assertEqual(ask_info["n_fresh"], 4)
        self.assertAlmostEqual(float(ask_info["predicted_replay_weight_mass"]), expected_mass)

    def test_accepted_replay_predicted_mass_matches_realized_mass(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=4,
            learning_rate=0.1,
            noise_std=0.1,
            buffer_size=8,
            reuse_fraction=0.5,
            min_replay_weight_mass=0.1,
            scalar_damping=0.0,
            use_curvature=False,
            buffer_sampling="random",
            seed=3,
        )
        params = np.zeros(1)
        opt._add_fresh_to_buffer(
            params,
            np.zeros((2, 1)),
            np.asarray([0.0, 1.0]),
        )
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        _, step_info = opt.tell(
            params,
            noise,
            np.asarray([2.0, 3.0]),
            ask_info,
        )

        self.assertFalse(ask_info["replay_mass_rejected"])
        self.assertEqual(ask_info["n_reused"], 2)
        self.assertTrue(step_info["used_replay"])
        self.assertAlmostEqual(float(ask_info["predicted_replay_weight_mass"]), 0.5)
        self.assertAlmostEqual(
            float(step_info["replay_weight_mass"]),
            float(ask_info["predicted_replay_weight_mass"]),
        )

    def test_low_predicted_ess_replay_is_replaced_before_evaluation(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=4,
            noise_std=0.1,
            buffer_size=8,
            reuse_fraction=0.5,
            max_importance_weight=10.0,
            min_replay_weight_mass=0.0,
            ess_min_ratio=0.9,
            seed=3,
        )
        high_ratio_noise = np.sqrt(2.0 * np.log(10.0))
        opt._add_fresh_to_buffer(
            np.zeros(1),
            np.full((2, 1), high_ratio_noise),
            np.asarray([0.0, 1.0]),
        )
        opt.current_params = np.asarray([opt.noise_std * high_ratio_noise])
        _, ask_info = opt.ask()

        self.assertTrue(ask_info["replay_ess_rejected"])
        self.assertEqual(ask_info["n_reused"], 0)
        self.assertEqual(ask_info["n_fresh"], 4)

    def test_curvature_match_diagnostic_uses_actual_fitness_transforms(self) -> None:
        def run(rank_fitness: bool) -> dict[str, object]:
            opt = DIIWES(
                num_params=2,
                population_size=4,
                noise_std=0.1,
                buffer_size=0,
                reuse_fraction=0.0,
                rank_fitness=rank_fitness,
                curvature_fitness="standardized",
                seed=4,
            )
            params = np.zeros(2)
            opt.current_params = params.copy()
            noise, ask_info = opt.ask()
            fitness = noise[:, 0] - 0.2 * noise[:, 1]
            _, info = opt.tell(params, noise, fitness, ask_info)
            return info

        rank_info = run(True)
        standardized_info = run(False)
        self.assertFalse(rank_info["curvature_matches_gradient"])
        self.assertTrue(standardized_info["curvature_matches_gradient"])
        self.assertEqual(rank_info["fitness_transform"], "rank_gradient_standardized_curvature")

    def test_matched_rank_curvature_rejects_standalone_center_fitness(self) -> None:
        opt = DIIWES(
            num_params=2,
            population_size=4,
            noise_std=0.1,
            buffer_size=8,
            reuse_fraction=0.0,
            rank_fitness=True,
            curvature_fitness="matched",
            seed=5,
        )
        params = np.zeros(2)
        opt.current_params = params.copy()
        noise, ask_info = opt.ask()
        fitness = noise[:, 0] - noise[:, 1]
        with self.assertRaisesRegex(ValueError, "matched rank curvature"):
            opt.tell(params, noise, fitness, ask_info, center_fitness=0.0)
        self.assertEqual(len(opt.sample_buffer), 0)

    def test_nonempty_fitness_with_empty_noise_is_rejected(self) -> None:
        opt = DIIWES(num_params=2)
        with self.assertRaisesRegex(ValueError, "fitness must be empty"):
            opt.tell(np.zeros(2), np.empty((0, 2)), np.asarray([1.0]))
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            opt.tell(np.zeros(2), np.zeros((2, 2)), np.zeros((2, 1)))
        with self.assertRaisesRegex(ValueError, "is_reused must have shape"):
            opt.tell(
                np.zeros(2),
                np.zeros((2, 2)),
                np.zeros(2),
                {"is_reused": np.zeros((2, 1), dtype=bool)},
            )

    def test_nonoverlapping_replay_is_replaced_with_fresh_samples(self) -> None:
        opt = DIIWES(
            num_params=2,
            population_size=4,
            noise_std=0.1,
            buffer_size=8,
            reuse_fraction=0.5,
            seed=2,
        )
        old_center = np.zeros(2)
        opt._add_fresh_to_buffer(
            old_center,
            np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]),
            np.arange(4.0),
        )
        opt.current_params = np.full(2, 10.0)
        _, ask_info = opt.ask()

        self.assertEqual(ask_info["n_replay_candidates"], 4)
        self.assertEqual(ask_info["n_replay_overlapping"], 0)
        self.assertEqual(ask_info["n_replay_selected_below_ratio_floor"], 0)
        self.assertEqual(ask_info["n_reused"], 0)
        self.assertEqual(ask_info["n_fresh"], 4)

    def test_elite_selection_handles_an_empty_overlap_set(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=4,
            noise_std=0.1,
            reuse_fraction=0.5,
            buffer_sampling="elite_distance",
            seed=2,
        )
        opt._add_fresh_to_buffer(
            np.zeros(1), np.asarray([[1.0], [-1.0]]), np.asarray([0.0, 1.0])
        )
        opt.current_params = np.asarray([10.0])
        _, ask_info = opt.ask()
        self.assertEqual(ask_info["n_reused"], 0)
        self.assertEqual(ask_info["n_fresh"], 4)

    def test_random_replay_samples_only_overlap_qualified_entries(self) -> None:
        opt = DIIWES(
            num_params=1,
            population_size=8,
            noise_std=0.1,
            reuse_fraction=0.5,
            min_replay_weight_mass=0.0,
            ess_min_ratio=0.0,
            seed=2,
        )
        opt._add_fresh_to_buffer(
            np.zeros(1), np.zeros((2, 1)), np.asarray([0.0, 1.0])
        )
        opt._add_fresh_to_buffer(
            np.asarray([10.0]), np.zeros((2, 1)), np.asarray([2.0, 3.0])
        )
        opt.current_params = np.zeros(1)
        _, ask_info = opt.ask()

        self.assertEqual(ask_info["n_replay_candidates"], 4)
        self.assertEqual(ask_info["n_replay_overlapping"], 2)
        self.assertEqual(ask_info["n_replay_selected_below_ratio_floor"], 0)
        self.assertEqual(set(ask_info["selected_indices"]), {0, 1})
        self.assertTrue(ask_info["replay_selection_uniform_within_overlap"])
        self.assertFalse(ask_info["replay_selection_unbiased"])
        self.assertEqual(ask_info["n_reused"], 2)
        self.assertEqual(ask_info["n_fresh"], 6)

    def test_conditions_reject_retired_and_norm_control_settings(self) -> None:
        stale = {"trust_radius": 1.0, "min_step_multiplier": 0.05}
        with self.assertRaisesRegex(ValueError, "retired"):
            _condition_config(stale, "standard_es")
        with self.assertRaisesRegex(ValueError, "max_grad_norm"):
            _condition_config({"max_grad_norm": 1.0}, "standard_es")
        with self.assertRaisesRegex(ValueError, "max_param_norm"):
            _condition_config({"max_param_norm": 1.0}, "standard_es")
        for bad_config in (
            {"replay_enabled": True, "reuse_fraction": 0.0, "buffer_size": 0},
            {"reuse_fraction": 0.2, "buffer_size": 0},
            {"reuse_fraction": 0.0, "buffer_size": 1024},
        ):
            with self.assertRaisesRegex(ValueError, "no-replay"):
                _condition_config(bad_config, "standard_es")
            with self.assertRaisesRegex(ValueError, "no-replay"):
                _validate_no_replay_protocol(bad_config)

    def test_scalar_damped_condition_is_explicitly_no_replay(self) -> None:
        config = _condition_config(
            {"reuse_fraction": 0.0, "buffer_size": 0}, "scalar_damped_es"
        )
        self.assertEqual(config["algorithm"], "curvature_preconditioned_es")
        self.assertFalse(config["use_curvature"])
        self.assertEqual(config["reuse_fraction"], 0.0)
        self.assertEqual(config["buffer_size"], 0)

    def test_replay_free_diiwes_matches_standard_es_when_damping_is_zero(self) -> None:
        population_size = 8
        params_standard = np.zeros(3)
        params_diiwes = np.zeros(3)
        standard = StandardES(
            num_params=3,
            population_size=population_size,
            learning_rate=0.2,
            noise_std=0.1,
            seed=19,
        )
        diiwes = DIIWES(
            num_params=3,
            population_size=population_size,
            learning_rate=0.2,
            noise_std=0.1,
            buffer_size=0,
            reuse_fraction=0.0,
            scalar_damping=0.0,
            use_curvature=False,
            seed=19,
        )

        for _ in range(3):
            standard.current_params = params_standard.copy()
            diiwes.current_params = params_diiwes.copy()
            standard_noise, standard_ask = standard.ask()
            diiwes_noise, diiwes_ask = diiwes.ask()
            self.assertTrue(np.array_equal(standard_noise, diiwes_noise))
            self.assertEqual(diiwes_ask["n_fresh"], population_size)
            self.assertEqual(diiwes_ask["n_reused"], 0)
            fitness = standard_noise[:, 0] - 0.4 * standard_noise[:, 1]
            params_standard, _ = standard.tell(
                params_standard, standard_noise, fitness, standard_ask
            )
            params_diiwes, info = diiwes.tell(
                params_diiwes, diiwes_noise, fitness, diiwes_ask
            )
            self.assertTrue(np.allclose(params_standard, params_diiwes))
            self.assertEqual(info["n_fresh"], population_size)
            self.assertEqual(info["n_reused"], 0)
            self.assertFalse(info["used_replay"])
            self.assertEqual(len(diiwes.sample_buffer), 0)

    def test_raw_and_matched_rank_conditions_remain_separate(self) -> None:
        raw = _condition_config({}, "diag_curvature_raw")
        matched = _condition_config({}, "diag_curvature_matched_rank")
        self.assertEqual(raw["curvature_fitness"], "raw")
        self.assertEqual(matched["curvature_fitness"], "matched")


if __name__ == "__main__":
    unittest.main()

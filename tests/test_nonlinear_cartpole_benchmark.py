#!/usr/bin/env python3
"""Tests for the nonlinear CartPole warm-start benchmark."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.nonlinear_cartpole.benchmark import (
    INITIALIZATIONS,
    METHODS,
    BenchmarkConfig,
    CartPoleDynamics,
    batched_greedy_returns,
    evaluate_policy,
    log_policy_gradient,
    make_policy,
    policy_probabilities,
    reinforce_pretrain,
    rollout,
    run_benchmark,
    write_outputs,
)


class NonlinearCartPoleBenchmarkTests(unittest.TestCase):
    def test_default_protocol_uses_300_updates_and_250_pairs(self) -> None:
        config = BenchmarkConfig()
        self.assertEqual(config.es_updates, 300)
        self.assertEqual(config.population_size, 500)
        self.assertEqual(config.antithetic_pairs, 250)

    def test_slurm_launcher_is_locked_and_direct_execution_is_dry(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = root / "scripts" / "slurm" / "submit_nonlinear_cartpole_300.sh"
        environment = os.environ.copy()
        for name in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "PAPER_EXPECTED_SOURCE_SHA"):
            environment.pop(name, None)
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd=root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in (
            "Job ID: local",
            "ES updates: 300",
            "Antithetic pairs per update: 250",
            "Candidate population per update: 500",
            "Dry run complete; no benchmark launched.",
            "--es-updates 300",
            "--antithetic-pairs 250",
        ):
            self.assertIn(fragment, result.stdout)

    def test_dynamics_are_seeded_and_nonlinear(self) -> None:
        first = CartPoleDynamics(max_steps=10)
        second = CartPoleDynamics(max_steps=10)
        np.testing.assert_array_equal(first.reset(13), second.reset(13))
        next_first, reward_first, done_first = first.step(1)
        next_second, reward_second, done_second = second.step(1)
        np.testing.assert_array_equal(next_first, next_second)
        self.assertEqual((reward_first, done_first), (reward_second, done_second))

        environment = CartPoleDynamics(max_steps=10)
        environment.reset(0)
        environment.state = np.asarray([0.0, 0.0, 0.1, 0.2])
        positive, _, _ = environment.step(1)
        environment.state = np.asarray([0.0, 0.0, -0.1, -0.2])
        environment.steps = 0
        negative, _, _ = environment.step(1)
        # The same rightward force breaks odd symmetry through sin/cos dynamics.
        self.assertGreater(np.linalg.norm(positive + negative), 0.01)

    def test_log_policy_gradient_matches_finite_difference(self) -> None:
        config = BenchmarkConfig(seeds=(0,), reinforce_updates=1, es_updates=1)
        policy = make_policy(config)
        rng = np.random.default_rng(17)
        params = rng.normal(scale=0.1, size=policy.num_params)
        state = np.asarray([0.02, -0.1, 0.04, 0.2])
        analytic = log_policy_gradient(policy, params, state, action=1)
        selected = rng.choice(policy.num_params, size=12, replace=False)
        step = 1e-6
        for index in selected:
            plus = params.copy()
            minus = params.copy()
            plus[index] += step
            minus[index] -= step
            numerical = (
                np.log(policy_probabilities(policy, plus, state)[1])
                - np.log(policy_probabilities(policy, minus, state)[1])
            ) / (2.0 * step)
            self.assertAlmostEqual(analytic[index], numerical, places=6)

    def test_batched_evaluation_matches_scalar_rollouts(self) -> None:
        config = BenchmarkConfig(seeds=(0,), reinforce_updates=1, es_updates=1)
        policy = make_policy(config)
        rng = np.random.default_rng(23)
        parameter_matrix = rng.normal(
            scale=0.1, size=(3, policy.num_params)
        )
        seeds = (5, 6, 7)
        expected = np.asarray(
            [
                rollout(
                    policy,
                    params,
                    seed,
                    config.max_episode_steps,
                    stochastic=False,
                )[0]
                for params, seed in zip(parameter_matrix, seeds, strict=True)
            ]
        )
        actual = batched_greedy_returns(
            policy, parameter_matrix, seeds, config.max_episode_steps
        )
        np.testing.assert_array_equal(actual, expected)

    def test_reinforce_pretraining_is_finite_and_auditable(self) -> None:
        config = BenchmarkConfig(
            seeds=(0,),
            eval_episodes=2,
            reinforce_updates=2,
            reinforce_batch_episodes=2,
            reinforce_eval_interval=1,
            reinforce_target_return=200.0,
            es_updates=1,
            population_size=4,
        )
        policy = make_policy(config)
        initial = np.zeros(policy.num_params, dtype=np.float64)
        params, history = reinforce_pretrain(policy, initial, config, seed=0)
        self.assertEqual(len(history), config.reinforce_updates + 1)
        self.assertTrue(np.all(np.isfinite(params)))
        self.assertGreater(np.linalg.norm(params - initial), 0.0)
        self.assertTrue(all(row["seed"] == 0 for row in history))
        evaluation = evaluate_policy(
            policy, params, seeds=(1, 2), max_steps=config.max_episode_steps
        )
        self.assertTrue(np.isfinite(evaluation))

    def test_small_benchmark_uses_both_core_optimizers_and_matched_starts(self) -> None:
        config = BenchmarkConfig(
            hidden_dims=(2,),
            max_episode_steps=25,
            seeds=(0,),
            eval_episodes=2,
            solve_return=24.0,
            reinforce_updates=1,
            reinforce_batch_episodes=2,
            reinforce_eval_interval=1,
            reinforce_target_return=25.0,
            es_updates=2,
            population_size=4,
        )
        result = run_benchmark(config)
        expected_rows = len(INITIALIZATIONS) * len(METHODS) * (config.es_updates + 1)
        self.assertEqual(len(result.trajectories), expected_rows)
        self.assertEqual(len(result.run_summaries), len(INITIALIZATIONS) * len(METHODS))
        self.assertEqual(len(result.aggregates), len(INITIALIZATIONS) * len(METHODS))
        for initialization in INITIALIZATIONS:
            starting = [
                row
                for row in result.trajectories
                if row["initialization"] == initialization and row["update"] == 0
            ]
            self.assertEqual({row["method"] for row in starting}, set(METHODS))
            self.assertEqual(len({row["eval_return"] for row in starting}), 1)

        diiwes_rows = [
            row
            for row in result.trajectories
            if row["method"] == "diiwes" and row["update"] > 0
        ]
        self.assertTrue(diiwes_rows)
        self.assertTrue(
            all(0.0 <= row["mean_step_multiplier"] <= 1.0 for row in diiwes_rows)
        )

    def test_output_directory_contract(self) -> None:
        config = BenchmarkConfig(
            hidden_dims=(2,),
            max_episode_steps=20,
            seeds=(0,),
            eval_episodes=1,
            solve_return=19.0,
            reinforce_updates=1,
            reinforce_batch_episodes=1,
            reinforce_eval_interval=1,
            reinforce_target_return=20.0,
            es_updates=1,
            population_size=4,
        )
        result = run_benchmark(config)
        with tempfile.TemporaryDirectory() as directory:
            outputs = write_outputs(result, config, directory)
            for path in outputs.values():
                self.assertTrue(Path(path).is_file(), path)
            manifest = json.loads(Path(outputs["manifest.json"]).read_text())
            self.assertEqual(manifest["experiment_version"], "1.2.0")
            self.assertEqual(manifest["config"]["seeds"], [0])
            self.assertEqual(
                manifest["optimizer_contract"]["antithetic_pairs_per_update"],
                2,
            )
            with Path(outputs["aggregate.csv"]).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(INITIALIZATIONS) * len(METHODS))
            report = Path(outputs["report.md"]).read_text()
            self.assertIn("REINFORCE", report)
            self.assertIn("DIIWES", report)


if __name__ == "__main__":
    unittest.main()

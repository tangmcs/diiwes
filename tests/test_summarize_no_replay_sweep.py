import contextlib
import io
import json
import os
import tempfile
import unittest

from scripts.summarize_no_replay_sweep import (
    DEFAULT_CONDITIONS,
    DEFAULT_AUC_TRAIN_STEP_BUDGET,
    DEFAULT_EXPECTED_ITERATIONS,
    DEFAULT_INITIAL_LEARNING_RATES,
    DEFAULT_SEEDS,
    EXPECTED_COMMON_CONFIG,
    EXPECTED_DIIWES_RESOLVED,
    EXPECTED_RESOLVED_COMMON,
    SweepValidationError,
    _return_at_training_env_step_budget,
    _training_env_step_auc,
    aggregate_runs,
    main,
    validate_and_collect_runs,
)


def _history(training_steps, rewards, initial_reward=0.0, learning_rate=1.0):
    return [
        {
            "iteration": index,
            "training_env_steps": step,
            "eval_reward": reward,
            "step_norm": 1.0,
            "solve_success": True,
            "initial_eval_reward": initial_reward,
            "lr": learning_rate / (index + 1) ** 0.5,
            "n_reused": 0,
            "n_fresh": 200,
            "used_replay": False,
            "replay_weight_mass": 0.0,
            "n_replay_candidates": 0,
            "n_replay_overlapping": 0,
            "buffer_size": 0,
        }
        for index, (step, reward) in enumerate(zip(training_steps, rewards))
    ]


class SweepFixture:
    def __init__(self, root):
        self.root = root

    def add_run(
        self,
        name,
        *,
        condition="standard_es",
        learning_rate=1.0,
        seed=0,
        history=None,
        status="complete",
        expected_iterations=3,
        completed_iterations=None,
    ):
        run_dir = os.path.join(self.root, name)
        os.makedirs(run_dir)
        history = (
            history
            if history is not None
            else _history(
                [4, 8, 12], [1.0, 3.0, 5.0], learning_rate=learning_rate
            )
        )
        if condition == "standard_es":
            resolved_optimizer = {
                **EXPECTED_RESOLVED_COMMON,
                "type": "StandardES",
                "initial_learning_rate": learning_rate,
            }
        else:
            use_curvature = condition not in {"no_curvature", "scalar_damped_es"}
            matched_rank = condition == "diag_curvature_matched_rank"
            resolved_optimizer = {
                **EXPECTED_RESOLVED_COMMON,
                **EXPECTED_DIIWES_RESOLVED,
                "type": "DIIWES",
                "initial_learning_rate": learning_rate,
                "use_curvature": use_curvature,
                "curvature_fitness": "matched" if matched_rank else "raw",
            }
            for record in history:
                record.update(
                    {
                        "solve_success": True,
                        "solver_type": "projected_diagonal_closed_form",
                        "linear_relative_residual": 0.0,
                    }
                )
                if use_curvature:
                    record.update(
                        {
                            "signed_linear_diagonal_min": 1.0,
                            "signed_linear_min_abs_diagonal": 1.0,
                            "signed_linear_condition_estimate": 1.0,
                            "signed_linear_nonpositive_diagonal_frac": 0.0,
                            "signed_system_finite": True,
                            "signed_system_invertible": True,
                            "signed_system_positive": True,
                        }
                    )
        config = {
            **EXPECTED_COMMON_CONFIG,
            "env_name": "Hopper-v5",
            "condition": condition,
            "algorithm": condition,
            "seed": seed,
            "learning_rate": learning_rate,
            "lr_schedule": "inverse_sqrt",
            "n_iterations": expected_iterations,
            "use_obs_norm": True,
            "obs_norm_mode": "frozen_after_calibration",
            "obs_norm_calibration_episodes": 3,
            "resolved_optimizer": resolved_optimizer,
            "provenance": {"source_sha256": "a" * 64},
        }
        status_record = {
            "status": status,
            "expected_iterations": expected_iterations,
            "completed_iterations": (
                len(history) if completed_iterations is None else completed_iterations
            ),
        }
        for filename, value in (
            ("config.json", config),
            ("history.json", history),
            ("status.json", status_record),
        ):
            with open(os.path.join(run_dir, filename), "w", encoding="utf-8") as stream:
                json.dump(value, stream)
        return run_dir


class SummarizeNoTrustSweepTests(unittest.TestCase):
    def test_production_defaults_define_preregistered_100_run_matrix(self):
        self.assertEqual(
            DEFAULT_CONDITIONS,
            (
                "standard_es",
                "scalar_damped_es",
                "diag_curvature_raw",
                "diag_curvature_matched_rank",
            ),
        )
        self.assertEqual(DEFAULT_INITIAL_LEARNING_RATES, (0.25, 1.0, 3.0, 10.0, 30.0))
        self.assertEqual(DEFAULT_SEEDS, (0, 1, 2, 3, 4))
        self.assertEqual(DEFAULT_EXPECTED_ITERATIONS, 500)
        self.assertEqual(DEFAULT_AUC_TRAIN_STEP_BUDGET, 75_000)
        self.assertEqual(
            len(DEFAULT_CONDITIONS)
            * len(DEFAULT_INITIAL_LEARNING_RATES)
            * len(DEFAULT_SEEDS),
            100,
        )

    def test_common_budget_auc_interpolates_and_ignores_later_endpoint(self):
        shorter = _history([4, 8, 12], [1.0, 3.0, 5.0])
        longer = _history([4, 8, 20], [1.0, 3.0, 9.0])

        self.assertAlmostEqual(_training_env_step_auc(shorter, 10), 1.7)
        self.assertAlmostEqual(_training_env_step_auc(longer, 10), 1.7)
        self.assertAlmostEqual(_return_at_training_env_step_budget(shorter, 10), 4.0)
        self.assertAlmostEqual(_return_at_training_env_step_budget(longer, 10), 4.0)

    def test_valid_complete_matrix_is_collected_and_aggregated(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SweepFixture(root)
            for condition in ("standard_es", "diag_curvature_raw"):
                for learning_rate in (1.0, 3.0):
                    for seed in (0, 1):
                        fixture.add_run(
                            f"{condition}_{learning_rate}_{seed}",
                            condition=condition,
                            learning_rate=learning_rate,
                            seed=seed,
                        )

            rows = validate_and_collect_runs(
                root,
                conditions=("standard_es", "diag_curvature_raw"),
                initial_learning_rates=(1.0, 3.0),
                seeds=(0, 1),
                expected_iterations=3,
                auc_train_step_budget=10,
            )
            summaries = aggregate_runs(rows)

            self.assertEqual(len(rows), 8)
            self.assertTrue(all(row["iterations"] == 3 for row in rows))
            self.assertTrue(all(row["training_env_step_auc"] == 1.7 for row in rows))
            self.assertEqual(len(summaries), 4)
            self.assertTrue(all(summary["runs"] == 2 for summary in summaries))

    def test_missing_and_duplicate_cells_are_both_reported(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SweepFixture(root)
            fixture.add_run("first")
            fixture.add_run("duplicate")

            with self.assertRaises(SweepValidationError) as raised:
                validate_and_collect_runs(
                    root,
                    conditions=("standard_es",),
                    initial_learning_rates=(1.0,),
                    seeds=(0, 1),
                    expected_iterations=3,
                    auc_train_step_budget=10,
                )

            report = "\n".join(raised.exception.issues)
            self.assertIn("duplicate runs", report)
            self.assertIn("seed=1", report)
            self.assertIn("missing run", report)

    def test_failed_nonfinite_incomplete_and_under_budget_run_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SweepFixture(root)
            fixture.add_run(
                "bad",
                history=_history([4, 8], [1.0, float("nan")]),
                status="failed",
                expected_iterations=3,
                completed_iterations=2,
            )

            with self.assertRaises(SweepValidationError) as raised:
                validate_and_collect_runs(
                    root,
                    conditions=("standard_es",),
                    initial_learning_rates=(1.0,),
                    seeds=(0,),
                    expected_iterations=3,
                    auc_train_step_budget=10,
                )

            report = "\n".join(raised.exception.issues)
            self.assertIn("run status is 'failed'", report)
            self.assertIn("nonfinite history value", report)
            self.assertIn("incomplete history", report)
            self.assertIn("completed_iterations=2", report)
            self.assertIn("insufficient training-step coverage", report)

    def test_protocol_drift_realized_schedule_and_missing_solve_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = SweepFixture(root)
            run_dir = fixture.add_run("bad", condition="scalar_damped_es")

            config_path = os.path.join(run_dir, "config.json")
            with open(config_path, "r", encoding="utf-8") as stream:
                config = json.load(stream)
            config["population_size"] = 201
            config["replay_enabled"] = True
            config["reuse_fraction"] = 0.2
            config["obs_scale"] = 2.0
            config["unexpected_override"] = True
            with open(config_path, "w", encoding="utf-8") as stream:
                json.dump(config, stream)

            history_path = os.path.join(run_dir, "history.json")
            with open(history_path, "r", encoding="utf-8") as stream:
                history = json.load(stream)
            history[1]["lr"] = 99.0
            history[1]["n_reused"] = 1
            history[1]["n_fresh"] = 199
            history[1]["used_replay"] = True
            history[1]["replay_weight_mass"] = 0.1
            history[1]["buffer_size"] = 1
            history[1].pop("linear_relative_residual")
            with open(history_path, "w", encoding="utf-8") as stream:
                json.dump(history, stream)

            with self.assertRaises(SweepValidationError) as raised:
                validate_and_collect_runs(
                    root,
                    conditions=("scalar_damped_es",),
                    initial_learning_rates=(1.0,),
                    seeds=(0,),
                    expected_iterations=3,
                    auc_train_step_budget=10,
                )

            report = "\n".join(raised.exception.issues)
            self.assertIn("config.population_size=201", report)
            self.assertIn("config.replay_enabled=True", report)
            self.assertIn("config.reuse_fraction=0.2", report)
            self.assertIn("config.obs_scale=2.0", report)
            self.assertIn("unexpected config key(s): unexpected_override", report)
            self.assertIn("expected 0.707106", report)
            self.assertIn("lacks a numeric solve residual", report)
            self.assertIn("reused samples in a no-replay run", report)
            self.assertIn("reports active replay", report)

    def test_cli_returns_nonzero_and_does_not_write_survivorship_outputs(self):
        with tempfile.TemporaryDirectory() as root:
            run_output = os.path.join(root, "runs.csv")
            summary_output = os.path.join(root, "groups.csv")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                exit_code = main(
                    [
                        root,
                        "--conditions",
                        "standard_es",
                        "--initial-learning-rates",
                        "1",
                        "--seeds",
                        "0",
                        "--expected-iterations",
                        "3",
                        "--auc-train-step-budget",
                        "10",
                        "--run-output",
                        run_output,
                        "--summary-output",
                        summary_output,
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertIn("missing run", stderr.getvalue())
            self.assertFalse(os.path.exists(run_output))
            self.assertFalse(os.path.exists(summary_output))


if __name__ == "__main__":
    unittest.main()

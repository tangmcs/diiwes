"""Tests for the automatic Hopper comparison report."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.analysis.summarize_hopper_pginit_noema import summarize


class TestHopperPGComparisonSummary(unittest.TestCase):
    def test_summarizer_validates_pairing_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warmstart = root / "warmstart"
            warmstart.mkdir()
            params = warmstart / "policy_params.npy"
            obs_norm = warmstart / "obs_norm.npz"
            params.touch()
            obs_norm.touch()
            (warmstart / "manifest.json").write_text(
                json.dumps({"best_eval_return": 321.0}), encoding="utf-8"
            )
            for task, condition in enumerate(("standard_es", "diag_curvature")):
                run = root / f"{condition}_seed0_task{task}"
                run.mkdir()
                config = {
                    "condition": condition,
                    "seed": 0,
                    "env_name": "Hopper-v5",
                    "population_size": 2000,
                    "n_iterations": 300,
                    "buffer_size": 0,
                    "reuse_fraction": 0.0,
                    "implicit_damping": 0.0,
                    "lr_schedule": "constant",
                    "learning_rate": 0.16,
                    "parameter_initialization": "policy_gradient_checkpoint",
                    "initial_params_path": str(params),
                    "initial_obs_norm_path": str(obs_norm),
                    "initial_params_sha256": "a" * 64,
                    "initial_obs_norm_sha256": "b" * 64,
                }
                if condition == "diag_curvature":
                    config.update(
                        curvature_beta=0.0,
                        bias_correct_curvature_ema=False,
                    )
                (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
                offset = 10.0 if condition == "diag_curvature" else 0.0
                history = []
                for iteration in range(300):
                    record = {
                        "iteration": iteration,
                        "eval_reward": iteration + offset,
                        "total_env_steps": (iteration + 1) * 100,
                    }
                    if condition == "diag_curvature":
                        record.update(
                            hessian_pairs=1000,
                            h_raw_mean=0.25,
                            h_step_mean=0.25,
                            curvature_active_frac=0.4,
                            step_multiplier_mean=0.8,
                        )
                    history.append(record)
                (run / "history.json").write_text(json.dumps(history), encoding="utf-8")

            result = summarize(root)
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["protocol"]["ema_enabled"])
            self.assertEqual(result["protocol"]["learning_rate"], 0.16)
            self.assertEqual(
                result["diag_curvature_minus_standard_es"]["final_eval_return"],
                10.0,
            )
            self.assertEqual(
                result["runs"]["diag_curvature"]["max_abs_raw_vs_step_curvature_mean"],
                0.0,
            )
            self.assertTrue((root / "comparison.json").is_file())
            self.assertTrue((root / "comparison.csv").is_file())
            self.assertTrue((root / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()

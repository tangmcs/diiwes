#!/usr/bin/env python3
"""Protocol tests for the new low-rate Hopper and convex launchers."""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HOPPER = ROOT / "scripts" / "slurm" / "submit_hopper_low_lr_long_sweep.sh"
CONVEX = ROOT / "scripts" / "slurm" / "submit_convex_implicit_step_sweep.sh"


class LauncherProtocolTests(unittest.TestCase):
    def run_hopper_task(self, task_id: int) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "PAPER_DRY_RUN": "1",
                "SLURM_ARRAY_TASK_ID": str(task_id),
            }
        )
        for name in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID"):
            environment.pop(name, None)
        return subprocess.run(
            ["bash", str(HOPPER)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_hopper_task_mapping_and_locked_flags(self) -> None:
        expected = {
            0: ("standard_es", "0.1", "0", "50"),
            49: ("standard_es", "2", "9", "99"),
            50: ("diag_curvature", "0.1", "0", "0"),
            99: ("diag_curvature", "2", "9", "49"),
        }
        for task_id, (condition, alpha0, seed, paired) in expected.items():
            with self.subTest(task_id=task_id):
                result = self.run_hopper_task(task_id)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"Condition: {condition}", result.stdout)
                self.assertIn(f"Initial learning rate: {alpha0}", result.stdout)
                self.assertIn(f"Seed: {seed}", result.stdout)
                self.assertIn(f"Paired task ID: {paired}", result.stdout)
                self.assertIn("Updates: 2000", result.stdout)
                self.assertIn("Predeclared prefix horizons: 500 1000 2000", result.stdout)
                command = next(
                    line for line in result.stdout.splitlines() if line.startswith("Command:")
                )
                for fragment in (
                    "--population-size 500",
                    "--buffer-size 0",
                    "--reuse-fraction 0",
                    "--implicit-damping 0",
                    "--lr-schedule inverse_sqrt",
                    "--iterations 2000",
                    "--trust-radius none",
                ):
                    self.assertIn(fragment, command)

    def test_hopper_rejects_out_of_range_task(self) -> None:
        result = self.run_hopper_task(100)
        self.assertEqual(result.returncode, 2)
        self.assertIn("outside the locked range 0-99", result.stderr)

    def test_convex_dry_run_is_locked(self) -> None:
        environment = os.environ.copy()
        environment["PAPER_DRY_RUN"] = "1"
        environment.pop("SLURM_JOB_ID", None)
        result = subprocess.run(
            ["bash", str(CONVEX)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for fragment in (
            "Population: 500",
            "Iterations: 500",
            "Predeclared checkpoints: 10,30,100,300,500",
            "Constant step sizes: 0.05,0.1,0.25,0.5,0.75,1,1.5,2",
            "Fitness transform: raw",
            "Trust / replay / additive scalar damping: not present",
            "equal-norm isotropic comparators are included separately",
            "Dry run complete; no benchmark launched.",
        ):
            self.assertIn(fragment, result.stdout)


if __name__ == "__main__":
    unittest.main()

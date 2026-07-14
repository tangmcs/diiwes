#!/usr/bin/env python3
"""Focused tests for post-training held-out evaluation infrastructure."""

from __future__ import annotations

import unittest

from experiments.train import (
    _calibration_rollout_seed,
    _center_rollout_seed,
    _evaluation_rollout_seed,
    _heldout_evaluation_rollout_seed,
    _heldout_metrics_at_budget,
    _resolve_heldout_evaluation_config,
    _select_heldout_checkpoint_steps,
    _training_rollout_seed,
)


class HeldoutSeedTests(unittest.TestCase):
    def test_heldout_seed_stream_is_fixed_and_separate(self) -> None:
        heldout = {_heldout_evaluation_rollout_seed(100, index) for index in range(20)}
        other = {
            _evaluation_rollout_seed(100, index) for index in range(20)
        } | {
            _calibration_rollout_seed(100, index) for index in range(20)
        } | {
            _center_rollout_seed(100, iteration) for iteration in range(20)
        } | {
            _training_rollout_seed(100, iteration, 0, True, 200, True)
            for iteration in range(20)
        }

        self.assertEqual(len(heldout), 20)
        self.assertTrue(heldout.isdisjoint(other))
        self.assertEqual(
            sorted(heldout),
            sorted(_heldout_evaluation_rollout_seed(100, index) for index in range(20)),
        )

    def test_heldout_seed_bank_depends_on_run_seed(self) -> None:
        first = {_heldout_evaluation_rollout_seed(100, index) for index in range(5)}
        second = {_heldout_evaluation_rollout_seed(101, index) for index in range(5)}
        self.assertTrue(first.isdisjoint(second))


class HeldoutCheckpointTests(unittest.TestCase):
    def test_selects_every_center_through_first_crossing(self) -> None:
        self.assertEqual(
            _select_heldout_checkpoint_steps([0, 8, 17, 31, 49], 30),
            [0, 8, 17, 31],
        )

    def test_exact_budget_center_is_included(self) -> None:
        self.assertEqual(
            _select_heldout_checkpoint_steps([0, 10, 20, 30, 40], 30),
            [0, 10, 20, 30],
        )

    def test_rejects_nonincreasing_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _select_heldout_checkpoint_steps([0, 10, 10], 30)


class HeldoutMetricTests(unittest.TestCase):
    def test_interpolates_return_and_auc_exactly_at_budget(self) -> None:
        auc, return_at_budget = _heldout_metrics_at_budget(
            [0, 40, 100],
            [0.0, 4.0, 10.0],
            75,
        )
        self.assertAlmostEqual(return_at_budget, 7.5)
        self.assertAlmostEqual(auc, 3.75)

    def test_requires_budget_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not cover"):
            _heldout_metrics_at_budget([0, 25, 50], [1.0, 2.0, 3.0], 75)


class HeldoutConfigTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertIsNone(_resolve_heldout_evaluation_config({}))

    def test_enabled_config_is_resolved(self) -> None:
        self.assertEqual(
            _resolve_heldout_evaluation_config(
                {
                    "heldout_evaluation_enabled": True,
                    "heldout_training_step_budget": 75_000,
                    "heldout_eval_episodes": 20,
                }
            ),
            {
                "heldout_training_step_budget": 75_000,
                "heldout_eval_episodes": 20,
            },
        )

    def test_enabled_config_requires_positive_integer_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "heldout_eval_episodes"):
            _resolve_heldout_evaluation_config(
                {
                    "heldout_evaluation_enabled": True,
                    "heldout_training_step_budget": 75_000,
                    "heldout_eval_episodes": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "heldout_training_step_budget"):
            _resolve_heldout_evaluation_config(
                {
                    "heldout_evaluation_enabled": True,
                    "heldout_training_step_budget": 1.5,
                    "heldout_eval_episodes": 20,
                }
            )


if __name__ == "__main__":
    unittest.main()

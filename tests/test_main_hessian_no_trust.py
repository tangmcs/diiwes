"""Focused regression tests for the mentor-requested main Hessian rerun."""

from __future__ import annotations

import unittest

import numpy as np

from core.diiwes import DIIWES
from experiments.train import learning_rate_at_iteration


class LearningRateScheduleTests(unittest.TestCase):
    def test_inverse_sqrt_matches_mentor_sequence(self) -> None:
        for iteration in (0, 1, 3, 99, 499):
            expected = 30.0 / np.sqrt(iteration + 1.0)
            self.assertEqual(
                learning_rate_at_iteration(30.0, iteration, "inverse_sqrt"),
                expected,
            )

    def test_inverse_linear_and_legacy_exponential(self) -> None:
        self.assertEqual(learning_rate_at_iteration(30.0, 2, "inverse_linear"), 10.0)
        self.assertEqual(learning_rate_at_iteration(2.0, 3, "exponential", 0.5), 0.25)
        self.assertEqual(learning_rate_at_iteration(2.0, 3, "constant"), 2.0)


class MainHessianDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def _optimizer(*, trust_radius: float | None, use_curvature: bool = False) -> DIIWES:
        return DIIWES(
            num_params=2,
            population_size=4,
            learning_rate=10.0,
            noise_std=1.0,
            buffer_size=0,
            reuse_fraction=0.0,
            implicit_damping=0.0,
            rank_fitness=False,
            use_curvature=use_curvature,
            curvature_beta=0.0,
            curvature_clip=1000.0,
            min_step_multiplier=0.05,
            trust_radius=trust_radius,
            bias_correct_curvature_ema=False,
            seed=0,
        )

    @staticmethod
    def _batch() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
        params = np.zeros(2, dtype=np.float64)
        noise = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float64,
        )
        fitness = np.asarray([10.0, 10.0, -10.0, -10.0], dtype=np.float64)
        ask_info: dict[str, object] = {
            "ask_params": params.copy(),
            "is_reused": np.zeros(4, dtype=bool),
        }
        return params, noise, fitness, ask_info

    def test_none_disables_only_trust_rescaling(self) -> None:
        optimizer = self._optimizer(trust_radius=None)
        params, noise, fitness, ask_info = self._batch()
        updated, info = optimizer.tell(params, noise, fitness, ask_info)

        self.assertGreater(info["pre_trust_step_norm"], 1.0)
        self.assertFalse(info["trust_active"])
        self.assertEqual(info["trust_scale"], 1.0)
        self.assertAlmostEqual(
            np.linalg.norm(updated - params),
            info["pre_trust_step_norm"],
            places=12,
        )

    def test_radius_one_reproduces_old_fixed_norm_rescaling(self) -> None:
        optimizer = self._optimizer(trust_radius=1.0)
        params, noise, fitness, ask_info = self._batch()
        updated, info = optimizer.tell(params, noise, fitness, ask_info)

        self.assertTrue(info["trust_active"])
        self.assertAlmostEqual(np.linalg.norm(updated - params), 1.0, places=12)

    def test_division_and_safeguard_residuals_are_separate(self) -> None:
        optimizer = self._optimizer(trust_radius=None, use_curvature=True)
        optimizer.hessian_ema[:] = -1000.0
        optimizer.hessian_ema_count = 1
        params, noise, fitness, ask_info = self._batch()
        _, info = optimizer.tell(params, noise, fitness, ask_info)

        self.assertLessEqual(info["division_relative_residual"], 1e-12)
        self.assertGreater(info["applied_relative_residual"], 1.0)
        self.assertEqual(info["multiplier_floor_frac"], 1.0)
        self.assertGreater(info["linear_condition_estimate"], 0.0)

    def test_raw_stein_diagonal_recovers_quadratic_curvature(self) -> None:
        rng = np.random.RandomState(7)
        n_pairs = 100_000
        sigma = 0.2
        diagonal = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        eps = rng.randn(n_pairs, len(diagonal))
        noise = np.concatenate([eps, -eps], axis=0)
        values = 0.5 * np.sum(diagonal * (sigma * eps) ** 2, axis=1)
        fitness = np.concatenate([values, values], axis=0)
        ask_info = {
            "fresh_pair_plus": np.arange(n_pairs),
            "fresh_pair_minus": np.arange(n_pairs, 2 * n_pairs),
        }
        optimizer = DIIWES(
            num_params=len(diagonal),
            population_size=2 * n_pairs,
            noise_std=sigma,
            reuse_fraction=0.0,
            curvature_mode="diag",
            seed=7,
        )

        estimate, observed_pairs = optimizer._estimate_fresh_curvature(
            noise,
            fitness,
            ask_info,
            sigma,
        )

        self.assertEqual(observed_pairs, n_pairs)
        np.testing.assert_allclose(estimate, diagonal, rtol=0.08, atol=0.08)
        self.assertIsNotNone(optimizer._last_h_split_correlation)
        self.assertGreater(optimizer._last_h_split_correlation, 0.9)


if __name__ == "__main__":
    unittest.main()

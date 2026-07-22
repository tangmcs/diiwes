"""Validation tests for the PPO-initialized Hopper comparison."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core import DIIWES
from core.policies import MLPPolicy
from experiments.hopper_policy_gradient.warmstart import (
    gaussian_log_probability,
    initialize_network,
    network_forward,
    network_gradient,
    ppo_actor_gradient,
)
from experiments.train import _load_initial_state, load_config
from utilities import ObsNormalizer


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mujoco" / "hopper_pginit_noema_1000pairs.yaml"
HIGH_LR_CONFIG = ROOT / "configs" / "mujoco" / "hopper_pginit_noema_lr0p5_1000pairs.yaml"


class TestHopperPGInitialization(unittest.TestCase):
    def test_locked_protocol_has_1000_pairs_300_updates_and_no_ema(self) -> None:
        config = load_config(str(CONFIG))
        self.assertEqual(config["env_name"], "Hopper-v5")
        self.assertEqual(config["population_size"], 2000)
        self.assertTrue(config["antithetic"])
        self.assertEqual(config["population_size"] // 2, 1000)
        self.assertEqual(config["n_iterations"], 300)
        self.assertEqual(config["buffer_size"], 0)
        self.assertEqual(config["reuse_fraction"], 0.0)
        self.assertEqual(config["implicit_damping"], 0.0)
        self.assertEqual(config["curvature_beta"], 0.0)
        self.assertFalse(config["bias_correct_curvature_ema"])

    def test_high_rate_protocol_changes_only_the_intended_step_scale(self) -> None:
        baseline = load_config(str(CONFIG))
        high_rate = load_config(str(HIGH_LR_CONFIG))
        self.assertEqual(baseline["learning_rate"], 0.16)
        self.assertEqual(high_rate["learning_rate"], 0.5)
        ignored = {"learning_rate"}
        self.assertEqual(
            {key: value for key, value in baseline.items() if key not in ignored},
            {key: value for key, value in high_rate.items() if key not in ignored},
        )

    def test_network_backpropagation_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(4)
        dimensions = (3, 4, 2)
        params = initialize_network(dimensions, rng)
        inputs = rng.normal(size=(5, 3))
        output_weights = rng.normal(size=(5, 2))
        analytic = network_gradient(params, inputs, output_weights, dimensions)

        def objective(values: np.ndarray) -> float:
            output, _ = network_forward(values, inputs, dimensions)
            return float(np.sum(output * output_weights))

        epsilon = 1e-6
        numeric = np.empty_like(params)
        for index in range(len(params)):
            plus = params.copy()
            minus = params.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)

    def test_ppo_actor_gradient_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(7)
        dimensions = (3, 4, 2)
        params = initialize_network(dimensions, rng)
        observations = rng.normal(size=(6, 3))
        means, _ = network_forward(params, observations, dimensions)
        action_std = 0.5
        latent_actions = means + action_std * rng.normal(size=means.shape)
        old_log_probabilities = gaussian_log_probability(
            latent_actions, means, action_std
        )
        advantages = rng.normal(size=len(observations))
        analytic, _ = ppo_actor_gradient(
            params,
            observations,
            latent_actions,
            old_log_probabilities,
            advantages,
            action_std,
            0.2,
            dimensions,
        )

        def objective(values: np.ndarray) -> float:
            new_means, _ = network_forward(values, observations, dimensions)
            new_log_probabilities = gaussian_log_probability(
                latent_actions, new_means, action_std
            )
            ratios = np.exp(np.clip(new_log_probabilities - old_log_probabilities, -20, 20))
            clipped = np.clip(ratios, 0.8, 1.2)
            return float(np.mean(np.minimum(ratios * advantages, clipped * advantages)))

        epsilon = 1e-6
        numeric = np.empty_like(params)
        for index in range(len(params)):
            plus = params.copy()
            minus = params.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            numeric[index] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=3e-6, atol=3e-8)

    def test_checkpoint_and_normalizer_are_loaded_exactly(self) -> None:
        policy = MLPPolicy(3, 2, hidden_dims=(4,))
        normalizer = ObsNormalizer((3,))
        params = np.linspace(-0.2, 0.3, policy.num_params)
        mean = np.array([1.0, -2.0, 0.5])
        var = np.array([0.4, 1.2, 2.5])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params_path = root / "params.npy"
            obs_path = root / "obs.npz"
            np.save(params_path, params)
            np.savez(obs_path, mean=mean, var=var, count=1234.0)
            loaded, metadata = _load_initial_state(
                {
                    "initial_params_path": str(params_path),
                    "initial_obs_norm_path": str(obs_path),
                },
                policy,
                normalizer,
                seed=0,
            )
        np.testing.assert_array_equal(loaded, params)
        np.testing.assert_array_equal(normalizer.mean, mean)
        np.testing.assert_array_equal(normalizer.var, var)
        self.assertEqual(normalizer.count, 1234.0)
        self.assertEqual(metadata["parameter_initialization"], "policy_gradient_checkpoint")
        self.assertEqual(len(metadata["initial_params_sha256"]), 64)
        self.assertEqual(len(metadata["initial_obs_norm_sha256"]), 64)

    def test_beta_zero_uses_the_current_batch_curvature_exactly(self) -> None:
        optimizer = DIIWES(
            num_params=3,
            population_size=40,
            learning_rate=0.1,
            noise_std=0.2,
            buffer_size=0,
            reuse_fraction=0.0,
            implicit_damping=0.0,
            rank_fitness=True,
            use_curvature=True,
            curvature_fitness="raw",
            curvature_mode="diag",
            curvature_beta=0.0,
            bias_correct_curvature_ema=False,
            seed=11,
        )
        center = np.array([0.1, -0.2, 0.3])
        optimizer.current_params = center.copy()
        noise, ask_info = optimizer.ask()
        candidates = center + optimizer.noise_std * noise
        fitness = -np.sum((candidates - np.array([0.4, 0.2, -0.5])) ** 2, axis=1)
        expected, pair_count = optimizer._estimate_fresh_curvature(
            noise=noise,
            f=fitness,
            ask_info=ask_info,
            sigma=optimizer.noise_std,
        )
        self.assertEqual(pair_count, 20)
        optimizer.tell(center, noise, fitness, ask_info=ask_info)
        np.testing.assert_array_equal(optimizer.hessian_for_step_vector, expected)


if __name__ == "__main__":
    unittest.main()

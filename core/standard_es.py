"""Standard evolution strategies baseline."""

from __future__ import annotations

from typing import Any

import numpy as np


def centered_ranks(x: np.ndarray) -> np.ndarray:
    """Return tie-aware centered ranks in [-0.5, 0.5]."""
    values = np.asarray(x)
    flat = values.ravel()
    n_values = len(flat)
    if n_values == 0:
        return flat.astype(np.float64).reshape(values.shape)

    order = np.argsort(flat, kind="mergesort")
    sorted_values = flat[order]
    ranks = np.empty(n_values, dtype=np.float64)

    start = 0
    while start < n_values:
        end = start + 1
        while end < n_values and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end

    if n_values > 1:
        ranks = ranks / float(n_values - 1) - 0.5
    else:
        ranks.fill(0.0)
    return ranks.reshape(values.shape)


class StandardES:
    """OpenAI-style ES baseline with rank or standardized fitness."""

    def __init__(
        self,
        num_params: int,
        population_size: int = 200,
        learning_rate: float = 0.02,
        noise_std: float = 0.02,
        l2_coeff: float = 0.0,
        antithetic: bool = True,
        rank_fitness: bool = True,
        max_grad_norm: float = 0.0,
        max_param_norm: float | None = None,
        trust_radius: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.num_params = int(num_params)
        self.population_size = int(population_size)
        self.learning_rate = float(learning_rate)
        self.noise_std = float(noise_std)
        self.l2_coeff = float(l2_coeff)
        self.antithetic = bool(antithetic)
        self.rank_fitness = bool(rank_fitness)
        self.max_grad_norm = float(max_grad_norm)
        self.max_param_norm = None if max_param_norm is None else float(max_param_norm)
        self.trust_radius = None if trust_radius is None else float(trust_radius)
        if self.num_params <= 0:
            raise ValueError("num_params must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if self.noise_std <= 0.0:
            raise ValueError("noise_std must be positive")
        if self.max_param_norm is not None and self.max_param_norm <= 0.0:
            raise ValueError("max_param_norm must be positive when provided")
        if self.trust_radius is not None and self.trust_radius <= 0.0:
            raise ValueError("trust_radius must be positive when provided")

        self.rng = np.random.RandomState(seed)
        self.iteration = 0
        self.eval_count = 0
        self.current_params = np.zeros(self.num_params, dtype=np.float64)

    def _sample_noise(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            return np.zeros((0, self.num_params), dtype=np.float64)
        if not self.antithetic or n_samples == 1:
            return self.rng.randn(n_samples, self.num_params)

        half = n_samples // 2
        noise_half = self.rng.randn(half, self.num_params)
        noise = np.concatenate([noise_half, -noise_half], axis=0)
        if n_samples % 2 == 1:
            noise = np.concatenate([noise, self.rng.randn(1, self.num_params)], axis=0)
        return noise.astype(np.float64, copy=False)

    def _gradient(self, noise: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        if self.rank_fitness:
            f = centered_ranks(fitness)
        else:
            f = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)
        grad = np.mean(f[:, None] * noise, axis=0) / (self.noise_std + 1e-10)
        return grad.astype(np.float64, copy=False)

    def _sgd_step(self, grad: np.ndarray) -> np.ndarray:
        if self.max_grad_norm > 0.0:
            norm = float(np.linalg.norm(grad))
            if norm > self.max_grad_norm and norm > 1e-12:
                grad = grad * (self.max_grad_norm / norm)
        return self.learning_rate * grad

    def ask(self) -> tuple[np.ndarray, dict[str, Any]]:
        noise = self._sample_noise(self.population_size)
        return noise, {
            "ask_params": self.current_params.copy(),
            "n_fresh": int(len(noise)),
            "n_reused": 0,
            "is_reused": np.zeros(len(noise), dtype=bool),
        }

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del center_fitness
        theta_t = np.asarray(params, dtype=np.float64)
        noise = np.asarray(noise, dtype=np.float64)
        fitness = np.asarray(fitness, dtype=np.float64)
        if theta_t.shape != (self.num_params,):
            raise ValueError(f"params must have shape ({self.num_params},), got {theta_t.shape}")
        if noise.ndim != 2 or noise.shape[1] != self.num_params:
            raise ValueError(f"noise must have shape (n, {self.num_params}), got {noise.shape}")
        if len(fitness) != len(noise):
            raise ValueError(f"fitness length must match noise rows, got {len(fitness)} and {len(noise)}")
        if ask_info is not None and "ask_params" in ask_info:
            if not np.allclose(theta_t, np.asarray(ask_info["ask_params"]), rtol=1e-7, atol=1e-9):
                raise ValueError("tell() params must match the params used by ask().")

        grad = self._gradient(noise, fitness)
        grad_norm_before_clip = float(np.linalg.norm(grad))
        step = self._sgd_step(grad)
        if self.l2_coeff > 0.0:
            step = step - self.learning_rate * self.l2_coeff * theta_t
        pre_trust_step_norm = float(np.linalg.norm(step))
        trust_active = False
        trust_scale = 1.0
        if (
            self.trust_radius is not None
            and pre_trust_step_norm > self.trust_radius
            and pre_trust_step_norm > 1e-12
        ):
            trust_active = True
            trust_scale = float(self.trust_radius / pre_trust_step_norm)
            step = step * trust_scale

        theta = theta_t + step
        if self.max_param_norm is not None:
            param_norm = float(np.linalg.norm(theta))
            if param_norm > self.max_param_norm and param_norm > 1e-12:
                theta = theta * (float(self.max_param_norm) / param_norm)

        self.iteration += 1
        self.eval_count += len(fitness)
        self.current_params = theta.copy()
        info = {
            "grad_norm": float(np.linalg.norm(grad)),
            "grad_norm_before_clip": grad_norm_before_clip,
            "param_norm": float(np.linalg.norm(theta)),
            "param_change": float(np.linalg.norm(theta - theta_t)),
            "step_norm": float(np.linalg.norm(theta - theta_t)),
            "pre_trust_step_norm": pre_trust_step_norm,
            "trust_active": bool(trust_active),
            "trust_scale": trust_scale,
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "max_fitness": float(np.max(fitness)),
            "min_fitness": float(np.min(fitness)),
            "iteration": int(self.iteration),
            "eval_count": int(self.eval_count),
            "n_fresh": int(len(fitness)),
            "n_reused": 0,
            "ess": float(len(fitness)),
            "ess_ratio": 1.0,
            "ess_normalized": 1.0,
            "clip_frac": 0.0,
            "clip_fraction": 0.0,
            "mean_importance_weight": 1.0,
            "max_importance_weight": 1.0,
            "importance_weight_mean": 1.0,
            "importance_weight_min": 1.0,
            "importance_weight_max": 1.0,
            "explicit_step_norm": float(np.linalg.norm(theta - theta_t)),
            "step_norm_ratio": 1.0,
            "sigma": float(self.noise_std),
            "learning_rate": float(self.learning_rate),
            "used_replay": False,
        }
        return theta.astype(np.float64, copy=True), info

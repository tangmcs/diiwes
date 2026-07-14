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


def centered_ranks_from_reference(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map values to centered empirical ranks defined only by a reference batch."""
    values = np.asarray(values)
    reference = np.asarray(reference).ravel()
    if len(reference) == 0:
        raise ValueError("reference must contain at least one value")
    if len(reference) == 1:
        return np.zeros_like(values, dtype=np.float64)

    sorted_reference = np.sort(reference, kind="mergesort")
    flat_values = values.ravel()
    left = np.searchsorted(sorted_reference, flat_values, side="left")
    right = np.searchsorted(sorted_reference, flat_values, side="right")
    positions = np.where(right > left, 0.5 * (left + right - 1), left).astype(np.float64)
    ranks = positions / float(len(reference) - 1) - 0.5
    return np.clip(ranks, -0.5, 0.5).reshape(values.shape)


def snes_utilities(fitness: np.ndarray) -> np.ndarray:
    """Return the canonical zero-sum SNES log-rank utilities.

    The best observation receives rank one. Exact fitness ties receive the
    average utility of the ranks occupied by the tied group, avoiding a
    dependence on input ordering that is not part of the SNES update.
    """
    values = np.asarray(fitness, dtype=np.float64)
    flat = values.ravel()
    population_size = len(flat)
    if population_size == 0:
        raise ValueError("fitness must contain at least one observation")
    if not np.all(np.isfinite(flat)):
        raise ValueError("fitness must contain only finite values")

    ranks = np.arange(1, population_size + 1, dtype=np.float64)
    positive = np.maximum(
        0.0,
        np.log(population_size / 2.0 + 1.0) - np.log(ranks),
    )
    utilities_by_rank = (
        positive / float(np.sum(positive)) - 1.0 / population_size
    )

    order = np.argsort(-flat, kind="mergesort")
    sorted_fitness = flat[order]
    utilities = np.empty(population_size, dtype=np.float64)
    start = 0
    while start < population_size:
        end = start + 1
        while (
            end < population_size
            and sorted_fitness[end] == sorted_fitness[start]
        ):
            end += 1
        utilities[order[start:end]] = float(np.mean(utilities_by_rank[start:end]))
        start = end
    utilities -= float(np.mean(utilities))
    return utilities.reshape(values.shape)


class StandardES:
    """Plain SGD-style ES baseline with rank or standardized fitness."""

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
        if self.num_params <= 0:
            raise ValueError("num_params must be positive")
        if self.population_size <= 0:
            raise ValueError("population_size must be positive")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.noise_std) or self.noise_std <= 0.0:
            raise ValueError("noise_std must be finite and positive")
        if not np.isfinite(self.l2_coeff) or self.l2_coeff < 0.0:
            raise ValueError("l2_coeff must be finite and nonnegative")
        if not np.isfinite(self.max_grad_norm) or self.max_grad_norm < 0.0:
            raise ValueError("max_grad_norm must be finite and nonnegative")
        if self.max_param_norm is not None and (
            not np.isfinite(self.max_param_norm) or self.max_param_norm <= 0.0
        ):
            raise ValueError("max_param_norm must be finite and positive when provided")

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
        f = f - float(np.mean(f))
        return self._gradient_from_utilities(noise, f)

    def _gradient_from_utilities(
        self, noise: np.ndarray, utilities: np.ndarray
    ) -> np.ndarray:
        """Compute the shared ES score gradient from centered utilities."""
        grad = np.mean(utilities[:, None] * noise, axis=0) / self.noise_std
        return grad.astype(np.float64, copy=False)

    def _sgd_step(self, grad: np.ndarray) -> np.ndarray:
        if self.max_grad_norm > 0.0:
            norm = float(np.linalg.norm(grad))
            if norm > self.max_grad_norm and norm > 1e-12:
                grad = grad * (self.max_grad_norm / norm)
        return self.learning_rate * grad

    def _optimizer_step(
        self, grad: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the optimizer-specific ascent step and scalar diagnostics."""
        return self.learning_rate * grad, {}

    def candidate_params(
        self, params: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        """Map standardized search noise to one or more candidate vectors."""
        theta = np.asarray(params, dtype=np.float64)
        standardized_noise = np.asarray(noise, dtype=np.float64)
        if theta.shape != (self.num_params,):
            raise ValueError(
                f"params must have shape ({self.num_params},), got {theta.shape}"
            )
        if (
            standardized_noise.ndim not in {1, 2}
            or standardized_noise.shape[-1] != self.num_params
        ):
            raise ValueError(
                "noise must have shape "
                f"({self.num_params},) or (n, {self.num_params}), got "
                f"{standardized_noise.shape}"
            )
        return theta + self.noise_std * standardized_noise

    def ask(self) -> tuple[np.ndarray, dict[str, Any]]:
        noise = self._sample_noise(self.population_size)
        if self.antithetic and self.population_size > 1:
            pair_count = self.population_size // 2
            pair_plus = np.arange(pair_count, dtype=int)
            pair_minus = np.arange(pair_count, 2 * pair_count, dtype=int)
        else:
            pair_plus = np.asarray([], dtype=int)
            pair_minus = np.asarray([], dtype=int)
        return noise, {
            "ask_params": self.current_params.copy(),
            "n_fresh": int(len(noise)),
            "n_reused": 0,
            "is_reused": np.zeros(len(noise), dtype=bool),
            "fresh_pair_plus": pair_plus,
            "fresh_pair_minus": pair_minus,
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
        if fitness.ndim != 1:
            raise ValueError(f"fitness must be one-dimensional, got shape {fitness.shape}")
        if len(fitness) != len(noise):
            raise ValueError(f"fitness length must match noise rows, got {len(fitness)} and {len(noise)}")
        if len(fitness) == 0:
            raise ValueError("fitness must contain at least one observation")
        if not np.all(np.isfinite(theta_t)):
            raise ValueError("params must contain only finite values")
        if not np.all(np.isfinite(noise)):
            raise ValueError("noise must contain only finite values")
        if not np.all(np.isfinite(fitness)):
            raise ValueError("fitness must contain only finite values")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if ask_info is not None:
            if "ask_params" in ask_info and not np.allclose(
                theta_t,
                np.asarray(ask_info["ask_params"]),
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError("tell() params must match the params used by ask().")
            is_reused = np.asarray(
                ask_info.get("is_reused", np.zeros(len(noise), dtype=bool)),
                dtype=bool,
            )
            if is_reused.shape != (len(noise),) or np.any(is_reused):
                raise ValueError("Standard ES variants do not accept replayed samples")
            if int(ask_info.get("n_reused", 0)) != 0:
                raise ValueError("Standard ES variants require n_reused=0")
            if int(ask_info.get("n_fresh", len(noise))) != len(noise):
                raise ValueError("Standard ES variants require a fully fresh batch")

        grad = self._gradient(noise, fitness)
        grad_norm_before_clip = float(np.linalg.norm(grad))
        if self.max_grad_norm > 0.0 and grad_norm_before_clip > self.max_grad_norm:
            grad = grad * (self.max_grad_norm / (grad_norm_before_clip + 1e-12))
        step, optimizer_info = self._optimizer_step(grad)
        step = np.asarray(step, dtype=np.float64)
        if step.shape != (self.num_params,) or not np.all(np.isfinite(step)):
            raise FloatingPointError("optimizer produced a non-finite step")
        if self.l2_coeff > 0.0:
            step = step - self.learning_rate * self.l2_coeff * theta_t

        theta = theta_t + step
        proposed_step_norm = float(np.linalg.norm(step))
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
            "proposed_step_norm": proposed_step_norm,
            "parameter_projection_active": bool(
                abs(float(np.linalg.norm(theta - theta_t)) - proposed_step_norm) > 1e-10
            ),
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "max_fitness": float(np.max(fitness)),
            "min_fitness": float(np.min(fitness)),
            "iteration": int(self.iteration),
            "eval_count": int(self.eval_count),
            "n_fresh": int(len(fitness)),
            "n_reused": 0,
            "buffer_size": 0,
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
            "explicit_step_norm": proposed_step_norm,
            "explicit_gradient_step_norm": proposed_step_norm,
            "step_norm_ratio": float(
                np.linalg.norm(theta - theta_t) / (proposed_step_norm + 1e-12)
            ),
            "sigma": float(self.noise_std),
            "learning_rate": float(self.learning_rate),
            "used_replay": False,
            "replay_weight_mass": 0.0,
            "fresh_weight_mass": 1.0,
            "curvature_clip_frac": 0.0,
            "solver_type": "none",
        }
        info.update(optimizer_info)
        return theta.astype(np.float64, copy=True), info


class SNES(StandardES):
    """Separable NES with a learned diagonal Gaussian search scale.

    ``learning_rate`` is the mean natural-gradient rate ``eta_mu``. The
    per-coordinate standard deviations use the canonical exponential update
    with ``eta_sigma``. No replay, clipping, projection, or weight decay is
    permitted because those mechanisms are not part of the SNES baseline.
    """

    def __init__(
        self,
        *args: Any,
        snes_sigma_learning_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not self.rank_fitness:
            raise ValueError("SNES requires canonical rank-based fitness utilities")
        if self.l2_coeff != 0.0:
            raise ValueError("SNES requires l2_coeff=0")
        if self.max_grad_norm != 0.0:
            raise ValueError("SNES does not permit gradient clipping")
        if self.max_param_norm is not None:
            raise ValueError("SNES does not permit parameter projection")

        default_sigma_rate = (
            3.0 + np.log(float(self.num_params))
        ) / (5.0 * np.sqrt(float(self.num_params)))
        if snes_sigma_learning_rate is None:
            sigma_rate = default_sigma_rate
            uses_default_sigma_rate = True
        else:
            sigma_rate = float(snes_sigma_learning_rate)
            uses_default_sigma_rate = False
        if not np.isfinite(sigma_rate) or sigma_rate <= 0.0:
            raise ValueError(
                "snes_sigma_learning_rate must be finite and positive"
            )

        self.snes_sigma_learning_rate = float(sigma_rate)
        self.snes_default_sigma_learning_rate = float(default_sigma_rate)
        self.snes_uses_default_sigma_learning_rate = uses_default_sigma_rate
        self.search_std = np.full(
            self.num_params, self.noise_std, dtype=np.float64
        )
        self.snes_iteration = 0
        self._snes_pending_standardized_mean_gradient: np.ndarray | None = None
        self._snes_pending_log_sigma_gradient: np.ndarray | None = None
        self._snes_pending_sampling_std: np.ndarray | None = None
        self._snes_pending_utilities: np.ndarray | None = None

    def candidate_params(
        self, params: np.ndarray, noise: np.ndarray
    ) -> np.ndarray:
        """Map standardized noise through the current diagonal search scale."""
        theta = np.asarray(params, dtype=np.float64)
        standardized_noise = np.asarray(noise, dtype=np.float64)
        if theta.shape != (self.num_params,):
            raise ValueError(
                f"params must have shape ({self.num_params},), got {theta.shape}"
            )
        if (
            standardized_noise.ndim not in {1, 2}
            or standardized_noise.shape[-1] != self.num_params
        ):
            raise ValueError(
                "noise must have shape "
                f"({self.num_params},) or (n, {self.num_params}), got "
                f"{standardized_noise.shape}"
            )
        return theta + self.search_std * standardized_noise

    def ask(self) -> tuple[np.ndarray, dict[str, Any]]:
        noise, ask_info = super().ask()
        ask_info["snes_sampling_std"] = self.search_std.copy()
        ask_info["snes_generation_token"] = int(self.snes_iteration)
        return noise, ask_info

    def _gradient(self, noise: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        utilities = np.asarray(snes_utilities(fitness), dtype=np.float64)
        mean_score = np.sum(utilities[:, None] * noise, axis=0)
        log_sigma_gradient = np.sum(
            utilities[:, None] * (np.square(noise) - 1.0),
            axis=0,
        )
        sampling_std = self.search_std.copy()
        self._snes_pending_standardized_mean_gradient = mean_score
        self._snes_pending_log_sigma_gradient = log_sigma_gradient
        self._snes_pending_sampling_std = sampling_std
        self._snes_pending_utilities = utilities
        return sampling_std * mean_score

    def _optimizer_step(
        self, grad: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if (
            self._snes_pending_standardized_mean_gradient is None
            or self._snes_pending_log_sigma_gradient is None
            or self._snes_pending_sampling_std is None
            or self._snes_pending_utilities is None
        ):
            raise RuntimeError("SNES update requires a preceding fitness gradient")

        old_std = self._snes_pending_sampling_std
        standardized_mean_gradient = self._snes_pending_standardized_mean_gradient
        log_sigma_gradient = self._snes_pending_log_sigma_gradient
        utilities = self._snes_pending_utilities
        log_sigma_step = (
            0.5 * self.snes_sigma_learning_rate * log_sigma_gradient
        )
        next_log_std = np.log(old_std) + log_sigma_step
        if not np.all(np.isfinite(next_log_std)):
            raise FloatingPointError("SNES produced a non-finite log standard deviation")
        with np.errstate(over="ignore", under="ignore"):
            next_std = old_std * np.exp(log_sigma_step)
        if not np.all(np.isfinite(next_std)) or np.any(next_std <= 0.0):
            raise FloatingPointError("SNES produced an invalid search standard deviation")

        step = self.learning_rate * grad
        if step.shape != (self.num_params,) or not np.all(np.isfinite(step)):
            raise FloatingPointError("SNES produced a non-finite mean step")
        self.search_std = next_std
        self.snes_iteration += 1
        self._snes_pending_standardized_mean_gradient = None
        self._snes_pending_log_sigma_gradient = None
        self._snes_pending_sampling_std = None
        self._snes_pending_utilities = None
        return step, {
            "optimizer_type": "snes",
            "solver_type": "separable_natural_gradient",
            "snes_iteration": self.snes_iteration,
            "snes_generation_token": self.snes_iteration - 1,
            "snes_mean_learning_rate": float(self.learning_rate),
            "snes_sigma_learning_rate": self.snes_sigma_learning_rate,
            "snes_standardized_mean_gradient_norm": float(
                np.linalg.norm(standardized_mean_gradient)
            ),
            "snes_parameter_space_mean_direction_norm": float(
                np.linalg.norm(grad)
            ),
            # Backward-compatible alias for pre-hardening exploratory artifacts.
            "snes_mean_natural_gradient_norm": float(np.linalg.norm(grad)),
            "snes_log_sigma_natural_gradient_norm": float(
                np.linalg.norm(log_sigma_gradient)
            ),
            "snes_log_sigma_step_norm": float(np.linalg.norm(log_sigma_step)),
            "snes_log_sigma_step_max_abs": float(
                np.max(np.abs(log_sigma_step))
            ),
            "snes_sigma_min_before": float(np.min(old_std)),
            "snes_sigma_max_before": float(np.max(old_std)),
            "snes_sigma_geometric_mean_before": float(
                np.exp(np.mean(np.log(old_std)))
            ),
            "snes_sigma_min_after": float(np.min(next_std)),
            "snes_sigma_max_after": float(np.max(next_std)),
            "snes_sigma_geometric_mean_after": float(
                np.exp(np.mean(np.log(next_std)))
            ),
            "snes_mean_step_mahalanobis_norm": float(
                np.linalg.norm(step / old_std)
            ),
            "snes_utility_sum": float(np.sum(utilities)),
            "snes_utility_l1_norm": float(np.sum(np.abs(utilities))),
            "snes_positive_utility_count": int(np.sum(utilities > 0.0)),
            "snes_antithetic_sampling": bool(self.antithetic),
            "sigma": float(np.exp(np.mean(np.log(old_std)))),
        }

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if ask_info is None:
            raise ValueError("SNES tell() requires ask_info returned by ask()")
        required_metadata = {"snes_sampling_std", "snes_generation_token"}
        missing_metadata = sorted(required_metadata.difference(ask_info))
        if missing_metadata:
            raise ValueError(
                "SNES tell() ask_info is missing required metadata: "
                + ", ".join(missing_metadata)
            )

        generation_token = ask_info["snes_generation_token"]
        if isinstance(generation_token, (bool, np.bool_)) or not isinstance(
            generation_token, (int, np.integer)
        ):
            raise ValueError("SNES generation token must be an integer")
        if int(generation_token) != self.snes_iteration:
            raise ValueError(
                "tell() SNES generation token does not match the active generation"
            )

        try:
            sampled_std = np.asarray(
                ask_info["snes_sampling_std"], dtype=np.float64
            )
        except (TypeError, ValueError) as error:
            raise ValueError("SNES sampling scale must be a numeric vector") from error
        if sampled_std.shape != (self.num_params,) or not np.array_equal(
            sampled_std, self.search_std
        ):
            raise ValueError(
                "tell() SNES search scale must exactly match the scale used by ask()"
            )
        return super().tell(
            params,
            noise,
            fitness,
            ask_info,
            center_fitness=center_fitness,
        )


class MomentumES(StandardES):
    """Fresh-population ES with a heavy-ball momentum ascent update."""

    def __init__(
        self,
        *args: Any,
        momentum_beta: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(momentum_beta) or not 0.0 <= momentum_beta < 1.0:
            raise ValueError("momentum_beta must be finite and in [0, 1)")
        self.momentum_beta = float(momentum_beta)
        self.momentum_buffer = np.zeros(self.num_params, dtype=np.float64)
        self.momentum_iteration = 0

    def _optimizer_step(
        self, grad: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.momentum_buffer = (
            self.momentum_beta * self.momentum_buffer + grad
        )
        self.momentum_iteration += 1
        step = self.learning_rate * self.momentum_buffer
        return step, {
            "optimizer_type": "momentum",
            "momentum_beta": self.momentum_beta,
            "momentum_iteration": self.momentum_iteration,
            "momentum_buffer_norm": float(np.linalg.norm(self.momentum_buffer)),
        }


class ClipUpES(StandardES):
    """Fresh-population ES with the primary ClipUp ascent update."""

    def __init__(
        self,
        *args: Any,
        clipup_momentum: float = 0.9,
        clipup_max_speed: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(clipup_momentum) or not 0.0 <= clipup_momentum < 1.0:
            raise ValueError("clipup_momentum must be finite and in [0, 1)")
        if not np.isfinite(clipup_max_speed) or clipup_max_speed <= 0.0:
            raise ValueError("clipup_max_speed must be finite and positive")
        if self.l2_coeff != 0.0:
            raise ValueError("primary ClipUp ES requires l2_coeff=0")
        if self.max_grad_norm != 0.0:
            raise ValueError("primary ClipUp ES does not permit gradient clipping")
        if self.max_param_norm is not None:
            raise ValueError("primary ClipUp ES does not permit parameter projection")
        self.clipup_momentum = float(clipup_momentum)
        self.clipup_max_speed = float(clipup_max_speed)
        self.clipup_velocity = np.zeros(self.num_params, dtype=np.float64)
        self.clipup_iteration = 0

    def _optimizer_step(
        self, grad: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gradient_norm = float(np.linalg.norm(grad))
        if gradient_norm > 0.0:
            normalized_gradient_step = (
                self.learning_rate * grad / gradient_norm
            )
        else:
            normalized_gradient_step = np.zeros_like(grad)

        unclipped_velocity = (
            self.clipup_momentum * self.clipup_velocity
            + normalized_gradient_step
        )
        unclipped_norm = float(np.linalg.norm(unclipped_velocity))
        if unclipped_norm > self.clipup_max_speed:
            clip_scale = self.clipup_max_speed / unclipped_norm
            velocity = unclipped_velocity * clip_scale
            velocity_clipped = True
        else:
            clip_scale = 1.0
            velocity = unclipped_velocity
            velocity_clipped = False

        self.clipup_velocity = velocity
        self.clipup_iteration += 1
        return velocity, {
            "optimizer_type": "clipup",
            "clipup_momentum": self.clipup_momentum,
            "clipup_max_speed": self.clipup_max_speed,
            "clipup_step_size": float(self.learning_rate),
            "clipup_iteration": self.clipup_iteration,
            "clipup_input_gradient_norm": gradient_norm,
            "clipup_zero_gradient": gradient_norm == 0.0,
            "clipup_normalized_gradient_step_norm": float(
                np.linalg.norm(normalized_gradient_step)
            ),
            "clipup_velocity_norm_before_clip": unclipped_norm,
            "clipup_velocity_norm": float(np.linalg.norm(velocity)),
            "clipup_velocity_clip_scale": float(clip_scale),
            "clipup_velocity_clipped": velocity_clipped,
        }


class AdamES(StandardES):
    """Fresh-population ES with a bias-corrected Adam ascent update."""

    def __init__(
        self,
        *args: Any,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(adam_beta1) or not 0.0 <= adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must be finite and in [0, 1)")
        if not np.isfinite(adam_beta2) or not 0.0 <= adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must be finite and in [0, 1)")
        if not np.isfinite(adam_epsilon) or adam_epsilon <= 0.0:
            raise ValueError("adam_epsilon must be finite and positive")
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)
        self.adam_first_moment = np.zeros(self.num_params, dtype=np.float64)
        self.adam_second_moment = np.zeros(self.num_params, dtype=np.float64)
        self.adam_iteration = 0

    def _optimizer_step(
        self, grad: np.ndarray
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.adam_iteration += 1
        self.adam_first_moment = (
            self.adam_beta1 * self.adam_first_moment
            + (1.0 - self.adam_beta1) * grad
        )
        self.adam_second_moment = (
            self.adam_beta2 * self.adam_second_moment
            + (1.0 - self.adam_beta2) * np.square(grad)
        )
        first_correction = 1.0 - self.adam_beta1**self.adam_iteration
        second_correction = 1.0 - self.adam_beta2**self.adam_iteration
        corrected_first = self.adam_first_moment / first_correction
        corrected_second = self.adam_second_moment / second_correction
        step = self.learning_rate * corrected_first / (
            np.sqrt(corrected_second) + self.adam_epsilon
        )
        return step, {
            "optimizer_type": "adam",
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "adam_iteration": self.adam_iteration,
            "adam_first_moment_norm": float(
                np.linalg.norm(self.adam_first_moment)
            ),
            "adam_second_moment_norm": float(
                np.linalg.norm(self.adam_second_moment)
            ),
            "adam_first_moment_bias_correction": first_correction,
            "adam_second_moment_bias_correction": second_correction,
        }

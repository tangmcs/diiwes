"""Fresh-population implicit evolution-strategy updates.

These optimizers intentionally exclude replay and step-norm controls.  The
endpoint method implements the paper's Picard map by recomputing both Gaussian
importance ratios and transformed perturbations at every candidate endpoint.
The linearized method applies the signed diagonal system obtained by
linearizing that same fresh-batch map at the current search center.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .standard_es import StandardES, centered_ranks


class _FreshImplicitBase(StandardES):
    """Shared validation and bookkeeping for fresh-only implicit ES methods."""

    def __init__(self, *args: Any, implicit_damping: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(implicit_damping) or implicit_damping < 0.0:
            raise ValueError("implicit_damping must be finite and nonnegative")
        if self.max_grad_norm != 0.0:
            raise ValueError("implicit ES does not permit gradient clipping")
        if self.max_param_norm is not None:
            raise ValueError("implicit ES does not permit parameter projection")
        self.implicit_damping = float(implicit_damping)

    def _validate_fresh_batch(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None,
        center_fitness: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta_t = np.asarray(params, dtype=np.float64)
        noise = np.asarray(noise, dtype=np.float64)
        fitness = np.asarray(fitness, dtype=np.float64)
        if theta_t.shape != (self.num_params,):
            raise ValueError(
                f"params must have shape ({self.num_params},), got {theta_t.shape}"
            )
        if noise.shape != (self.population_size, self.num_params):
            raise ValueError(
                "implicit ES requires one complete fresh population with shape "
                f"({self.population_size}, {self.num_params}), got {noise.shape}"
            )
        if fitness.shape != (self.population_size,):
            raise ValueError(
                f"fitness must have shape ({self.population_size},), got {fitness.shape}"
            )
        if not np.all(np.isfinite(theta_t)):
            raise ValueError("params must contain only finite values")
        if not np.all(np.isfinite(noise)):
            raise ValueError("noise must contain only finite values")
        if not np.all(np.isfinite(fitness)):
            raise ValueError("fitness must contain only finite values")
        if center_fitness is not None and not np.isfinite(center_fitness):
            raise ValueError("center_fitness must be finite when provided")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if ask_info is not None:
            if "ask_params" in ask_info and not np.allclose(
                theta_t,
                np.asarray(ask_info["ask_params"], dtype=np.float64),
                rtol=1e-7,
                atol=1e-9,
            ):
                raise ValueError("tell() params must match the params used by ask()")
            is_reused = np.asarray(
                ask_info.get("is_reused", np.zeros(self.population_size, dtype=bool)),
                dtype=bool,
            )
            if is_reused.shape != (self.population_size,) or np.any(is_reused):
                raise ValueError("implicit ES does not accept replayed samples")
            if int(ask_info.get("n_reused", 0)) != 0:
                raise ValueError("implicit ES requires n_reused=0")
        return theta_t, noise, fitness

    def _utilities(self, fitness: np.ndarray) -> tuple[np.ndarray, str]:
        if self.rank_fitness:
            utilities = centered_ranks(fitness).astype(np.float64, copy=False)
            transform = "centered_rank"
        else:
            utilities = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)
            transform = "standardized"
        # The implicit and linearized formulas assume the fixed utilities are
        # exactly centered at the proposal distribution.
        utilities = utilities - float(np.mean(utilities))
        return utilities, transform

    def _proposal_gradient(
        self, theta_t: np.ndarray, noise: np.ndarray, utilities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        data_gradient = self._gradient_from_utilities(noise, utilities)
        total_gradient = data_gradient - self.l2_coeff * theta_t
        return data_gradient, total_gradient

    @staticmethod
    def _correlation(left: np.ndarray, right: np.ndarray) -> float:
        left = np.asarray(left, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        left_centered = left - float(np.mean(left))
        right_centered = right - float(np.mean(right))
        denominator = float(
            np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
        )
        if denominator <= 1e-15:
            return 1.0 if np.allclose(left, right, rtol=1e-12, atol=1e-12) else 0.0
        return float(np.dot(left_centered, right_centered) / denominator)

    def _finish_update(
        self,
        theta_t: np.ndarray,
        fitness: np.ndarray,
        proposal_gradient: np.ndarray,
        step: np.ndarray,
        *,
        solver_type: str,
        solve_success: bool,
        fitness_transform: str,
        extra_info: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        step = np.asarray(step, dtype=np.float64)
        if step.shape != (self.num_params,) or not np.all(np.isfinite(step)):
            raise FloatingPointError("implicit update produced a non-finite step")
        theta = theta_t + step
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("implicit update produced non-finite parameters")

        alpha = float(self.learning_rate)
        explicit_step_norm = float(np.linalg.norm(alpha * proposal_gradient))
        step_norm = float(np.linalg.norm(step))
        self.iteration += 1
        self.eval_count += len(fitness)
        self.current_params = theta.copy()

        info: dict[str, Any] = {
            "grad_norm": float(np.linalg.norm(proposal_gradient)),
            "grad_norm_before_clip": float(np.linalg.norm(proposal_gradient)),
            "param_norm": float(np.linalg.norm(theta)),
            "param_change": step_norm,
            "step_norm": step_norm,
            "proposed_step_norm": step_norm,
            "parameter_projection_active": False,
            "mean_fitness": float(np.mean(fitness)),
            "std_fitness": float(np.std(fitness)),
            "max_fitness": float(np.max(fitness)),
            "min_fitness": float(np.min(fitness)),
            "iteration": int(self.iteration),
            "eval_count": int(self.eval_count),
            "n_fresh": int(len(fitness)),
            "n_reused": 0,
            "buffer_size": 0,
            "used_replay": False,
            "replay_weight_mass": 0.0,
            "fresh_weight_mass": 1.0,
            "ess": float(len(fitness)),
            "ess_ratio": 1.0,
            "ess_normalized": 1.0,
            "clip_frac": 0.0,
            "clip_fraction": 0.0,
            "importance_weight_mean": 1.0,
            "importance_weight_min": 1.0,
            "importance_weight_max": 1.0,
            "explicit_step_norm": explicit_step_norm,
            "explicit_gradient_step_norm": explicit_step_norm,
            "step_norm_ratio": float(step_norm / (explicit_step_norm + 1e-12)),
            "sigma": float(self.noise_std),
            "learning_rate": alpha,
            "implicit_damping": float(self.implicit_damping),
            "fitness_transform": fitness_transform,
            "solver_type": solver_type,
            "solve_success": bool(solve_success),
            "curvature_clip_frac": 0.0,
        }
        info.update(extra_info)
        return theta.astype(np.float64, copy=True), info


class EndpointImplicitES(_FreshImplicitBase):
    """Paper endpoint update solved by unrelaxed Picard iteration.

    A generation's evaluated points and transformed utilities remain fixed,
    while target-distribution weights and score vectors are recomputed at each
    candidate endpoint.  This within-update reweighting is not replay.
    """

    def __init__(
        self,
        *args: Any,
        implicit_iterations: int = 10,
        implicit_tolerance: float = 1e-5,
        diagnostic_ratio_floor: float = 1e-3,
        diagnostic_ratio_cap: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(implicit_iterations) <= 0:
            raise ValueError("implicit_iterations must be positive")
        if not np.isfinite(implicit_tolerance) or implicit_tolerance <= 0.0:
            raise ValueError("implicit_tolerance must be finite and positive")
        if (
            not np.isfinite(diagnostic_ratio_floor)
            or not np.isfinite(diagnostic_ratio_cap)
            or diagnostic_ratio_floor <= 0.0
            or diagnostic_ratio_cap <= 0.0
            or diagnostic_ratio_floor > diagnostic_ratio_cap
        ):
            raise ValueError("diagnostic ratio bounds must be finite, positive, and ordered")
        self.implicit_iterations = int(implicit_iterations)
        self.implicit_tolerance = float(implicit_tolerance)
        self.diagnostic_ratio_floor = float(diagnostic_ratio_floor)
        self.diagnostic_ratio_cap = float(diagnostic_ratio_cap)
        self._endpoint_log_ratio_floor = float(np.log(self.diagnostic_ratio_floor))
        self._endpoint_log_ratio_cap = float(np.log(self.diagnostic_ratio_cap))
        self.use_curvature = False
        self.curvature_fitness = "none"
        self.curvature_mode = "none"

    def _endpoint_gradient(
        self,
        noise: np.ndarray,
        utilities: np.ndarray,
        delta: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        delta_scaled = np.asarray(delta, dtype=np.float64) / self.noise_std
        log_ratio_raw = self._endpoint_log_ratios(noise, delta)
        if not np.all(np.isfinite(log_ratio_raw)):
            raise FloatingPointError("endpoint log importance ratios are non-finite")
        # The quadratic term in the Gaussian log ratio is common to every
        # sample and cancels under self-normalization.  Using relative logits
        # avoids underflow without the absolute ratio floor that would turn a
        # far endpoint into uniform proposal weights.
        relative_logits = noise @ delta_scaled
        unnormalized = np.exp(relative_logits - float(np.max(relative_logits)))
        weight_sum = float(np.sum(unnormalized))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise FloatingPointError("endpoint importance weights cannot be normalized")
        weights = unnormalized / weight_sum

        transformed_noise = noise - delta_scaled[None, :]
        weighted_utility_mean = float(np.dot(weights, utilities))
        centered_utilities = utilities - weighted_utility_mean
        gradient = (weights * centered_utilities) @ transformed_noise
        gradient /= self.noise_std
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError("endpoint gradient is non-finite")

        below_floor = log_ratio_raw <= self._endpoint_log_ratio_floor
        above_cap = log_ratio_raw >= self._endpoint_log_ratio_cap
        ess = float(1.0 / (np.sum(weights * weights) + 1e-12))
        stats: dict[str, Any] = {
            "endpoint_ess": ess,
            "endpoint_ess_ratio": float(ess / len(weights)),
            "endpoint_ratio_clipping_enabled": False,
            "endpoint_clip_frac": 0.0,
            "endpoint_ratio_outside_diagnostic_bounds_frac": float(
                np.mean(below_floor | above_cap)
            ),
            "endpoint_ratio_below_floor_frac": float(np.mean(below_floor)),
            "endpoint_ratio_above_cap_frac": float(np.mean(above_cap)),
            "endpoint_all_ratios_below_floor": bool(np.all(below_floor)),
            "endpoint_all_ratios_above_cap": bool(np.all(above_cap)),
            "endpoint_weight_min": float(np.min(weights)),
            "endpoint_weight_max": float(np.max(weights)),
            "endpoint_weight_underflow_frac": float(np.mean(weights == 0.0)),
            "endpoint_relative_logit_span": float(
                np.max(relative_logits) - np.min(relative_logits)
            ),
            "endpoint_log_ratio_raw_min": float(np.min(log_ratio_raw)),
            "endpoint_log_ratio_raw_max": float(np.max(log_ratio_raw)),
            "endpoint_log_ratio_raw_mean": float(np.mean(log_ratio_raw)),
            "endpoint_weighted_utility_mean": weighted_utility_mean,
            "endpoint_gradient_norm": float(np.linalg.norm(gradient)),
        }
        return gradient.astype(np.float64, copy=False), stats

    def _endpoint_log_ratios(
        self, noise: np.ndarray, delta: np.ndarray
    ) -> np.ndarray:
        """Return log p_theta+delta(w) - log p_theta(w) for fresh points."""
        delta_scaled = np.asarray(delta, dtype=np.float64) / self.noise_std
        return noise @ delta_scaled - 0.5 * float(
            np.dot(delta_scaled, delta_scaled)
        )

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta_t, noise, fitness = self._validate_fresh_batch(
            params, noise, fitness, ask_info, center_fitness
        )
        utilities, transform = self._utilities(fitness)
        _, proposal_gradient = self._proposal_gradient(theta_t, noise, utilities)
        alpha = float(self.learning_rate)

        delta = np.zeros(self.num_params, dtype=np.float64)
        initial_endpoint_gradient: np.ndarray | None = None
        initial_relative_residual = 0.0
        iterations_used = 0
        residual_values: list[float] = []
        iterate_norms: list[float] = []
        gradient_changes: list[float] = []
        endpoint_ess_ratios: list[float] = []
        endpoint_weight_maxima: list[float] = []
        endpoint_logit_spans: list[float] = []
        target_history: list[np.ndarray] = []
        for iteration in range(self.implicit_iterations):
            data_gradient, iteration_stats = self._endpoint_gradient(
                noise, utilities, delta
            )
            if initial_endpoint_gradient is None:
                initial_endpoint_gradient = data_gradient.copy()
            gradient_changes.append(
                float(
                    np.linalg.norm(data_gradient - initial_endpoint_gradient)
                    / max(float(np.linalg.norm(initial_endpoint_gradient)), 1e-12)
                )
            )
            endpoint_ess_ratios.append(float(iteration_stats["endpoint_ess_ratio"]))
            endpoint_weight_maxima.append(float(iteration_stats["endpoint_weight_max"]))
            endpoint_logit_spans.append(
                float(iteration_stats["endpoint_relative_logit_span"])
            )
            total_gradient = data_gradient - self.l2_coeff * (theta_t + delta)
            target_delta = alpha * (
                total_gradient - self.implicit_damping * delta
            )
            if not np.all(np.isfinite(target_delta)):
                raise FloatingPointError("Picard iteration produced a non-finite iterate")
            residual = target_delta - delta
            relative_residual = float(
                np.linalg.norm(residual)
                / max(
                    float(np.linalg.norm(target_delta)),
                    float(np.linalg.norm(delta)),
                    1e-12,
                )
            )
            if iteration == 0:
                initial_relative_residual = relative_residual
            residual_values.append(relative_residual)
            iterate_norms.append(float(np.linalg.norm(target_delta)))
            target_history.append(target_delta.copy())
            delta = target_delta
            iterations_used = iteration + 1
            if relative_residual <= self.implicit_tolerance:
                break

        final_data_gradient, endpoint_stats = self._endpoint_gradient(
            noise, utilities, delta
        )
        final_total_gradient = final_data_gradient - self.l2_coeff * (theta_t + delta)
        final_target = alpha * (
            final_total_gradient - self.implicit_damping * delta
        )
        equation_residual = delta - final_target
        absolute_residual = float(np.linalg.norm(equation_residual))
        relative_residual = float(
            absolute_residual
            / max(
                float(np.linalg.norm(delta)),
                float(np.linalg.norm(alpha * final_total_gradient)),
                1e-12,
            )
        )
        converged = bool(relative_residual <= self.implicit_tolerance)
        gradient_change = float(
            np.linalg.norm(final_data_gradient - initial_endpoint_gradient)
            / max(float(np.linalg.norm(initial_endpoint_gradient)), 1e-12)
        )
        if len(target_history) >= 3:
            two_cycle_error = float(
                np.linalg.norm(target_history[-1] - target_history[-3])
                / max(
                    float(np.linalg.norm(target_history[-1])),
                    float(np.linalg.norm(target_history[-3])),
                    1e-12,
                )
            )
        else:
            two_cycle_error = 0.0
        extra_info = {
            "implicit_converged": converged,
            "implicit_iterations": int(iterations_used),
            "implicit_max_iterations": int(self.implicit_iterations),
            "implicit_tolerance": float(self.implicit_tolerance),
            "implicit_initial_relative_residual": initial_relative_residual,
            "implicit_absolute_residual": absolute_residual,
            "implicit_relative_residual": relative_residual,
            "implicit_gradient_relative_change": gradient_change,
            "implicit_gradient_relative_change_max": float(max(gradient_changes)),
            "implicit_relative_residual_min": float(min(residual_values)),
            "implicit_relative_residual_max": float(max(residual_values)),
            "implicit_iterate_norm_min": float(min(iterate_norms)),
            "implicit_iterate_norm_max": float(max(iterate_norms)),
            "implicit_endpoint_ess_ratio_min": float(min(endpoint_ess_ratios)),
            "implicit_endpoint_weight_max_max": float(max(endpoint_weight_maxima)),
            "implicit_endpoint_logit_span_max": float(max(endpoint_logit_spans)),
            "implicit_final_two_cycle_relative_error": two_cycle_error,
            "step_over_sigma": float(np.linalg.norm(delta) / self.noise_std),
            "endpoint_weighting_is_replay": False,
            **endpoint_stats,
        }
        return self._finish_update(
            theta_t,
            fitness,
            proposal_gradient,
            delta,
            solver_type="picard_endpoint_implicit",
            solve_success=converged,
            fitness_transform=transform,
            extra_info=extra_info,
        )


class LinearizedImplicitES(_FreshImplicitBase):
    """Signed same-generation diagonal linearization of endpoint implicit ES."""

    def __init__(
        self,
        *args: Any,
        min_abs_diagonal: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not np.isfinite(min_abs_diagonal) or min_abs_diagonal <= 0.0:
            raise ValueError("min_abs_diagonal must be finite and positive")
        self.min_abs_diagonal = float(min_abs_diagonal)
        self._previous_hessian: np.ndarray | None = None
        self.use_curvature = True
        self.curvature_fitness = "matched"
        self.curvature_mode = "diag"

    def _matched_diagonal_hessian(
        self,
        noise: np.ndarray,
        utilities: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ask_info is None:
            raise ValueError("linearized implicit ES requires antithetic pair metadata")
        plus = np.asarray(ask_info.get("fresh_pair_plus", []), dtype=int)
        minus = np.asarray(ask_info.get("fresh_pair_minus", []), dtype=int)
        if len(plus) == 0 or len(plus) != len(minus):
            raise ValueError("linearized implicit ES requires complete antithetic pairs")
        if len(plus) * 2 != len(noise):
            raise ValueError("every sample must belong to an antithetic pair")
        if not np.allclose(noise[minus], -noise[plus], rtol=1e-12, atol=1e-12):
            raise ValueError("antithetic pair metadata does not match the sampled noise")

        pair_utility = utilities[plus] + utilities[minus]
        pair_contributions = pair_utility[:, None] * (noise[plus] ** 2 - 1.0)
        pair_contributions /= 2.0 * self.noise_std**2
        hessian = np.mean(pair_contributions, axis=0)
        if not np.all(np.isfinite(hessian)):
            raise FloatingPointError("matched diagonal Hessian is non-finite")
        return hessian.astype(np.float64, copy=False), pair_contributions

    def _independent_split_diagonal_hessians(
        self,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Estimate each pair half using utilities ranked only within that half."""
        if ask_info is None:
            raise ValueError("split diagnostics require antithetic pair metadata")
        plus = np.asarray(ask_info.get("fresh_pair_plus", []), dtype=int)
        minus = np.asarray(ask_info.get("fresh_pair_minus", []), dtype=int)
        if len(plus) == 0 or len(plus) != len(minus):
            raise ValueError("split diagnostics require complete antithetic pairs")
        split = len(plus) // 2
        if split <= 0 or split >= len(plus):
            return None

        estimates: list[np.ndarray] = []
        for pair_indices in (np.arange(split), np.arange(split, len(plus))):
            half_plus = plus[pair_indices]
            half_minus = minus[pair_indices]
            sample_indices = np.concatenate([half_plus, half_minus])
            half_utilities, _ = self._utilities(fitness[sample_indices])
            pair_count = len(pair_indices)
            pair_utility = (
                half_utilities[:pair_count]
                + half_utilities[pair_count:]
            )
            contributions = pair_utility[:, None] * (
                noise[half_plus] ** 2 - 1.0
            )
            contributions /= 2.0 * self.noise_std**2
            estimate = np.mean(contributions, axis=0)
            if not np.all(np.isfinite(estimate)):
                raise FloatingPointError(
                    "independent split diagonal Hessian is non-finite"
                )
            estimates.append(estimate.astype(np.float64, copy=False))
        return estimates[0], estimates[1]

    def _split_rank_semantics(self) -> str:
        if self.rank_fitness:
            return "independent_centered_ranks_per_disjoint_pair_half"
        return "independent_standardization_per_disjoint_pair_half"

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta_t, noise, fitness = self._validate_fresh_batch(
            params, noise, fitness, ask_info, center_fitness
        )
        utilities, transform = self._utilities(fitness)
        _, proposal_gradient = self._proposal_gradient(theta_t, noise, utilities)
        hessian, pair_contributions = self._matched_diagonal_hessian(
            noise, utilities, ask_info
        )

        alpha = float(self.learning_rate)
        diagonal = (
            1.0
            + alpha * (self.implicit_damping + self.l2_coeff)
            - alpha * hessian
        )
        if not np.all(np.isfinite(diagonal)):
            raise FloatingPointError("signed diagonal system is non-finite")
        absolute_diagonal = np.abs(diagonal)
        minimum_absolute = float(np.min(absolute_diagonal))
        if minimum_absolute < self.min_abs_diagonal:
            raise FloatingPointError(
                "signed diagonal system is numerically singular: "
                f"min abs diagonal {minimum_absolute:.6e}"
            )

        rhs = alpha * proposal_gradient
        step = rhs / diagonal
        residual = diagonal * step - rhs
        relative_residual = float(
            np.linalg.norm(residual) / max(float(np.linalg.norm(rhs)), 1e-12)
        )
        solve_success = bool(np.isfinite(relative_residual) and relative_residual <= 1e-10)

        split_estimates = self._independent_split_diagonal_hessians(
            noise, fitness, ask_info
        )
        if split_estimates is not None:
            first_half, second_half = split_estimates
            split_correlation = self._correlation(first_half, second_half)
            split_sign_agreement = float(
                np.mean(np.sign(first_half) == np.sign(second_half))
            )
            split_first_components = first_half.tolist()
            split_second_components = second_half.tolist()
        else:
            split_correlation = 0.0
            split_sign_agreement = 0.0
            split_first_components = []
            split_second_components = []

        if self._previous_hessian is None:
            temporal_correlation = 0.0
            temporal_sign_agreement = 0.0
        else:
            temporal_correlation = self._correlation(
                self._previous_hessian, hessian
            )
            temporal_sign_agreement = float(
                np.mean(np.sign(self._previous_hessian) == np.sign(hessian))
            )
        self._previous_hessian = hessian.copy()

        maximum_absolute = float(np.max(absolute_diagonal))
        extra_info = {
            "implicit_converged": solve_success,
            "implicit_iterations": 1,
            "implicit_relative_residual": relative_residual,
            "linear_relative_residual": relative_residual,
            "linear_diagonal_min": float(np.min(diagonal)),
            "linear_diagonal_max": float(np.max(diagonal)),
            "linear_min_abs_diagonal": minimum_absolute,
            "linear_condition_estimate": float(maximum_absolute / minimum_absolute),
            "linear_nonpositive_diagonal_frac": float(np.mean(diagonal <= 0.0)),
            "linear_near_singular_frac": float(
                np.mean(absolute_diagonal < 1e-6)
            ),
            "hessian_pairs": int(len(pair_contributions)),
            "h_raw_mean": float(np.mean(hessian)),
            "h_raw_std": float(np.std(hessian)),
            "h_raw_min": float(np.min(hessian)),
            "h_raw_max": float(np.max(hessian)),
            "h_raw_norm": float(np.linalg.norm(hessian)),
            "h_positive_frac": float(np.mean(hessian > 0.0)),
            "h_negative_frac": float(np.mean(hessian < 0.0)),
            "h_split_correlation": split_correlation,
            "h_split_sign_agreement": split_sign_agreement,
            "h_split_first_components": split_first_components,
            "h_split_second_components": split_second_components,
            "h_split_rank_semantics": self._split_rank_semantics(),
            "h_split_pair_partition": "first_vs_second_antithetic_pair_halves",
            "h_split_first_pair_count": int(len(pair_contributions) // 2),
            "h_split_second_pair_count": int(
                len(pair_contributions) - len(pair_contributions) // 2
            ),
            "h_temporal_correlation": temporal_correlation,
            "h_temporal_sign_agreement": temporal_sign_agreement,
            "curvature_fitness": "matched",
            "curvature_matches_gradient": True,
            "curvature_mode": "diag",
            "curvature_beta": 0.0,
            "step_over_sigma": float(np.linalg.norm(step) / self.noise_std),
        }
        return self._finish_update(
            theta_t,
            fitness,
            proposal_gradient,
            step,
            solver_type="signed_diagonal_linearized_implicit",
            solve_success=solve_success,
            fitness_transform=transform,
            extra_info=extra_info,
        )


class ConcaveCurvatureES(_FreshImplicitBase):
    """Fresh ES damped only by transform-matched concave curvature.

    The signed linearized system can amplify noise whenever an estimated
    Hessian coordinate is positive or makes its denominator nearly zero.  This
    variant retains only the concave part of the estimate, so every diagonal
    denominator is at least one.  Block structure and a bias-corrected EMA are
    available to reduce estimator variance without clipping the curvature. A
    separate LOPO rank mode matches the gradient and block moment as order-two
    pair-cluster U-statistics and supplies a delete-pair jackknife diagnostic.
    """

    def __init__(
        self,
        *args: Any,
        curvature_structure: str = "diag",
        block_slices: Sequence[slice | Sequence[int]] | None = None,
        curvature_beta: float = 0.0,
        curvature_estimator: str = "stein_moment",
        curvature_confidence_z: float | None = None,
        rank_utility_mode: str = "pooled_centered_ranks",
        attenuation_mode: str = "structured",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        structure = str(curvature_structure).lower()
        estimator = str(curvature_estimator).lower()
        utility_mode = str(rank_utility_mode).lower()
        attenuation = str(attenuation_mode).lower()
        if structure not in {"diag", "block"}:
            raise ValueError("curvature_structure must be one of diag, block")
        if estimator not in {"stein_moment", "block_joint_ols"}:
            raise ValueError(
                "curvature_estimator must be one of stein_moment, block_joint_ols"
            )
        if estimator == "block_joint_ols" and structure != "block":
            raise ValueError(
                "block_joint_ols curvature_estimator requires block structure"
            )
        if utility_mode not in {
            "pooled_centered_ranks",
            "lopo_rank_u_statistic",
        }:
            raise ValueError(
                "rank_utility_mode must be one of pooled_centered_ranks, "
                "lopo_rank_u_statistic"
            )
        if utility_mode == "lopo_rank_u_statistic" and structure != "block":
            raise ValueError(
                "lopo_rank_u_statistic requires block curvature structure"
            )
        if utility_mode == "lopo_rank_u_statistic" and estimator != "stein_moment":
            raise ValueError(
                "lopo_rank_u_statistic requires stein_moment curvature_estimator"
            )
        if attenuation not in {"structured", "isotropic_norm_matched"}:
            raise ValueError(
                "attenuation_mode must be one of structured, isotropic_norm_matched"
            )
        if attenuation == "isotropic_norm_matched" and structure != "block":
            raise ValueError(
                "isotropic_norm_matched attenuation requires block structure"
            )
        if not np.isfinite(curvature_beta) or not 0.0 <= curvature_beta < 1.0:
            raise ValueError("curvature_beta must be finite and in [0, 1)")
        if curvature_confidence_z is not None and (
            not np.isfinite(curvature_confidence_z)
            or float(curvature_confidence_z) < 0.0
        ):
            raise ValueError(
                "curvature_confidence_z must be finite and nonnegative when provided"
            )
        confidence_se_available = (
            estimator == "block_joint_ols"
            or utility_mode == "lopo_rank_u_statistic"
        )
        if curvature_confidence_z is not None and not confidence_se_available:
            raise ValueError(
                "curvature_confidence_z requires block_joint_ols or "
                "lopo_rank_u_statistic"
            )
        if (
            curvature_confidence_z is not None
            and utility_mode == "lopo_rank_u_statistic"
            and float(curvature_beta) != 0.0
        ):
            raise ValueError(
                "LOPO U-statistic confidence adjustment requires curvature_beta=0"
            )
        if not self.antithetic or self.population_size < 2 or self.population_size % 2:
            raise ValueError(
                "ConcaveCurvatureES requires an even antithetic population"
            )
        if not self.rank_fitness:
            raise ValueError("ConcaveCurvatureES requires rank_fitness=True")
        if (
            utility_mode == "lopo_rank_u_statistic"
            and self.population_size // 2 < 3
        ):
            raise ValueError(
                "lopo_rank_u_statistic requires at least three antithetic pairs"
            )

        self.curvature_structure = structure
        self.curvature_mode = structure
        self.curvature_fitness = "matched"
        self.curvature_beta = float(curvature_beta)
        self.curvature_same_generation = self.curvature_beta == 0.0
        self.curvature_estimator = estimator
        self.rank_utility_mode = utility_mode
        self.curvature_confidence_z = (
            None
            if curvature_confidence_z is None
            else float(curvature_confidence_z)
        )
        self.attenuation_mode = attenuation
        self.use_curvature = True
        self.persist_hessian_ema_artifact = True
        self.block_slices = self._normalize_block_slices(block_slices)
        if structure == "diag":
            self._component_slices = tuple(
                slice(index, index + 1) for index in range(self.num_params)
            )
        else:
            self._component_slices = self.block_slices
        self.num_curvature_components = len(self._component_slices)
        if self.curvature_estimator == "block_joint_ols":
            regression_parameters = self.num_curvature_components + 1
            pair_count = self.population_size // 2
            split = pair_count // 2
            if min(split, pair_count - split) <= regression_parameters:
                raise ValueError(
                    "block_joint_ols requires each split half to contain more "
                    "pairs than regression parameters"
                )
        self.hessian_ema = np.zeros(
            self.num_curvature_components, dtype=np.float64
        )
        self.hessian_ema_variance = np.zeros(
            self.num_curvature_components, dtype=np.float64
        )
        self.hessian_ema_count = 0
        self._previous_raw_curvature: np.ndarray | None = None

    def _normalize_block_slices(
        self,
        block_slices: Sequence[slice | Sequence[int]] | None,
    ) -> tuple[slice, ...]:
        if block_slices is None:
            return (slice(0, self.num_params),)

        normalized: list[slice] = []
        coverage = np.zeros(self.num_params, dtype=np.int8)
        for item in block_slices:
            if isinstance(item, slice):
                if item.step not in (None, 1):
                    raise ValueError("block_slices do not support slice steps")
                start = 0 if item.start is None else int(item.start)
                stop = self.num_params if item.stop is None else int(item.stop)
            else:
                if len(item) != 2:
                    raise ValueError(
                        "block_slices entries must be slices or [start, stop] pairs"
                    )
                start, stop = int(item[0]), int(item[1])
            if start < 0 or stop > self.num_params or start >= stop:
                raise ValueError(
                    f"invalid block slice [{start}, {stop}) for d={self.num_params}"
                )
            coverage[start:stop] += 1
            normalized.append(slice(start, stop))

        if not normalized or np.any(coverage != 1):
            raise ValueError(
                "block_slices must form a non-overlapping full partition"
            )
        normalized.sort(key=lambda item: int(item.start))
        return tuple(normalized)

    def _antithetic_pairs(
        self,
        noise: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if ask_info is None:
            raise ValueError(
                "ConcaveCurvatureES requires antithetic pair metadata"
            )
        plus = np.asarray(ask_info.get("fresh_pair_plus", []), dtype=int)
        minus = np.asarray(ask_info.get("fresh_pair_minus", []), dtype=int)
        expected_pairs = len(noise) // 2
        if (
            plus.ndim != 1
            or minus.ndim != 1
            or len(plus) != expected_pairs
            or len(minus) != expected_pairs
        ):
            raise ValueError(
                "ConcaveCurvatureES requires complete antithetic pairs"
            )
        combined = np.concatenate([plus, minus])
        if (
            np.any(combined < 0)
            or np.any(combined >= len(noise))
            or len(np.unique(combined)) != len(noise)
        ):
            raise ValueError("antithetic pair metadata must partition the population")
        if not np.array_equal(noise[minus], -noise[plus]):
            raise ValueError(
                "antithetic pair metadata does not match the sampled noise"
            )
        return plus, minus

    def _curvature_pair_data(
        self,
        noise: np.ndarray,
        utilities: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        plus, minus = self._antithetic_pairs(noise, ask_info)
        eps = noise[plus]
        pair_utility = utilities[plus] + utilities[minus]
        return eps, pair_utility

    @staticmethod
    def _comparison_sign(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return sign(left-right) without subtracting possibly large returns."""
        return (np.asarray(left) > np.asarray(right)).astype(np.int8) - (
            np.asarray(left) < np.asarray(right)
        ).astype(np.int8)

    def _lopo_rank_utilities(
        self,
        fitness: np.ndarray,
        noise: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Compute exact leave-own-pair-out empirical midrank utilities."""
        plus, minus = self._antithetic_pairs(noise, ask_info)
        pair_count = len(plus)
        if pair_count < 3:
            raise ValueError(
                "lopo_rank_u_statistic requires at least three antithetic pairs"
            )
        pooled = centered_ranks(fitness).astype(np.float64, copy=False)
        within_pair_sign = self._comparison_sign(
            fitness[plus], fitness[minus]
        ).astype(np.float64)
        mate = np.zeros(len(fitness), dtype=np.float64)
        mate[plus] = within_pair_sign
        mate[minus] = -within_pair_sign
        c_m = 2.0 * (pair_count - 1.0) / (2.0 * pair_count - 1.0)
        lopo = (
            (2.0 * pair_count - 1.0) * pooled - 0.5 * mate
        ) / (2.0 * (pair_count - 1.0))
        if not np.all(np.isfinite(lopo)):
            raise FloatingPointError("LOPO rank utilities are non-finite")
        return lopo, pooled, within_pair_sign, float(c_m)

    def _lopo_gradient_estimate(
        self,
        theta_t: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        """Return the shared matched LOPO gradient state and identities."""
        (
            utilities,
            pooled_utilities,
            within_pair_sign,
            c_m,
        ) = self._lopo_rank_utilities(fitness, noise, ask_info)
        utility_sum = float(np.sum(utilities, dtype=np.float64))
        utility_mean = float(utility_sum / len(utilities))
        zero_sum_tolerance = float(
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(utilities), dtype=np.float64)))
        )
        if abs(utility_sum) > zero_sum_tolerance:
            raise FloatingPointError(
                "LOPO utilities violated the structural zero-sum identity"
            )
        data_gradient, proposal_gradient = self._proposal_gradient(
            theta_t, noise, utilities
        )
        plus, _ = self._antithetic_pairs(noise, ask_info)
        pooled_data_gradient = self._gradient_from_utilities(
            noise, pooled_utilities
        )
        within_pair_gradient_remainder = (
            pooled_data_gradient - c_m * data_gradient
        )
        expected_remainder = np.sum(
            noise[plus] * within_pair_sign[:, None], axis=0
        ) / (
            2.0
            * len(plus)
            * self.noise_std
            * (2.0 * len(plus) - 1.0)
        )
        remainder_identity_error = float(
            np.max(
                np.abs(within_pair_gradient_remainder - expected_remainder)
            )
        )
        remainder_scale = max(
            1.0,
            float(np.max(np.abs(within_pair_gradient_remainder))),
            float(np.max(np.abs(expected_remainder))),
        )
        if remainder_identity_error > 5e-12 * remainder_scale:
            raise FloatingPointError("LOPO within-pair gradient identity failed")

        # With fixed utilities, the self-normalized endpoint-gradient
        # Jacobian differs from the unnormalized Gaussian Stein moment by a
        # utility-mean term. Exact antithetic noise and the structural LOPO
        # zero sum make that difference zero at the proposal center. Keep the
        # full finite-sample calculation here so the scoped identity is
        # checked rather than inferred from a rounded utility mean.
        noise_mean = np.mean(noise, axis=0, dtype=np.float64)
        noise_second_moment = np.mean(noise * noise, axis=0, dtype=np.float64)
        utility_noise_mean = np.mean(
            utilities[:, None] * noise, axis=0, dtype=np.float64
        )
        utility_noise_second_moment = np.mean(
            utilities[:, None] * noise * noise,
            axis=0,
            dtype=np.float64,
        )
        inverse_variance = 1.0 / self.noise_std**2
        unnormalized_diagonal = inverse_variance * (
            utility_noise_second_moment - utility_mean
        )
        self_normalized_diagonal = inverse_variance * (
            utility_noise_second_moment
            - 2.0 * utility_noise_mean * noise_mean
            - utility_mean * noise_second_moment
            + 2.0 * utility_mean * noise_mean * noise_mean
        )
        general_gap_diagonal = inverse_variance * (
            2.0 * utility_noise_mean * noise_mean
            + utility_mean * (noise_second_moment - 1.0)
            - 2.0 * utility_mean * noise_mean * noise_mean
        )
        antithetic_gap_diagonal = (
            inverse_variance
            * utility_mean
            * (noise_second_moment - 1.0)
        )
        unnormalized_components = np.asarray(
            [
                np.mean(unnormalized_diagonal[block])
                for block in self._component_slices
            ],
            dtype=np.float64,
        )
        self_normalized_components = np.asarray(
            [
                np.mean(self_normalized_diagonal[block])
                for block in self._component_slices
            ],
            dtype=np.float64,
        )
        general_gap_components = np.asarray(
            [
                np.mean(general_gap_diagonal[block])
                for block in self._component_slices
            ],
            dtype=np.float64,
        )
        antithetic_gap_components = np.asarray(
            [
                np.mean(antithetic_gap_diagonal[block])
                for block in self._component_slices
            ],
            dtype=np.float64,
        )
        observed_gap_components = (
            unnormalized_components - self_normalized_components
        )
        gap_identity_error = float(
            np.max(np.abs(observed_gap_components - general_gap_components))
        )
        antithetic_reduction_error = float(
            np.max(np.abs(general_gap_components - antithetic_gap_components))
        )
        endpoint_identity_scale = max(
            1.0,
            float(np.max(np.abs(unnormalized_components))),
            float(np.max(np.abs(self_normalized_components))),
        )
        endpoint_identity_tolerance = 5e-12 * endpoint_identity_scale
        endpoint_gap_max_abs = float(
            np.max(np.abs(observed_gap_components))
        )
        if (
            gap_identity_error > endpoint_identity_tolerance
            or antithetic_reduction_error > endpoint_identity_tolerance
            or endpoint_gap_max_abs > endpoint_identity_tolerance
        ):
            raise FloatingPointError(
                "LOPO at-proposal self-normalized Jacobian identity failed"
            )
        diagnostics: dict[str, Any] = {
            "lopo_c_m": float(c_m),
            "lopo_utility_semantics": (
                "exact_leave_own_antithetic_pair_out_midranks_no_recentering"
            ),
            "lopo_centering_operation_applied": False,
            "lopo_zero_sum_identity_basis": (
                "ordered_cross_pair_comparison_cancellation"
            ),
            "lopo_rank_utility_sum": utility_sum,
            "lopo_rank_utility_mean": utility_mean,
            "lopo_zero_sum_abs_sum": abs(utility_sum),
            "lopo_zero_sum_abs_mean": abs(utility_mean),
            "lopo_zero_sum_tolerance": zero_sum_tolerance,
            "lopo_zero_mean_tolerance": (
                zero_sum_tolerance / len(utilities)
            ),
            "lopo_zero_sum_identity_verified": True,
            "sample_reuse": False,
            "importance_weighting": False,
            "lopo_within_pair_gradient_remainder_norm": float(
                np.linalg.norm(within_pair_gradient_remainder)
            ),
            "lopo_within_pair_gradient_remainder_max_abs": float(
                np.max(np.abs(within_pair_gradient_remainder))
            ),
            "lopo_within_pair_gradient_identity_max_abs_error": (
                remainder_identity_error
            ),
            "lopo_gradient_identity_verified": True,
            "lopo_utility_population_target": (
                "current_mid_cdf_stop_gradient"
            ),
            "lopo_raw_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average": True,
            "lopo_raw_block_moment_endpoint_jacobian_scope": (
                "at_proposal_frozen_lopo_utility_self_normalized_map_"
                "raw_preprojection_block_average_of_diagonal"
            ),
            "lopo_full_endpoint_jacobian_operator_claim": False,
            "lopo_projected_curvature_operator_endpoint_jacobian_claim": False,
            "lopo_off_proposal_endpoint_jacobian_claim": False,
            "lopo_global_adaptive_rank_hessian_claim": False,
            "lopo_raw_return_hessian_claim": False,
            "lopo_at_proposal_unnormalized_minus_sn_block_gap_max_abs": (
                endpoint_gap_max_abs
            ),
            "lopo_at_proposal_unnormalized_minus_sn_block_gap_norm": float(
                np.linalg.norm(observed_gap_components)
            ),
            "lopo_at_proposal_gap_identity_max_abs_error": (
                gap_identity_error
            ),
            "lopo_at_proposal_antithetic_gap_reduction_max_abs_error": (
                antithetic_reduction_error
            ),
            "lopo_at_proposal_endpoint_identity_tolerance": (
                endpoint_identity_tolerance
            ),
            "lopo_at_proposal_sn_unnormalized_identity_verified": True,
        }
        return (
            utilities,
            pooled_utilities,
            c_m,
            data_gradient,
            proposal_gradient,
            diagnostics,
        )

    def _block_quadratic_features(self, eps: np.ndarray) -> np.ndarray:
        if self.curvature_structure != "block":
            raise ValueError("block quadratic features require block structure")
        eps = np.asarray(eps, dtype=np.float64)
        if eps.ndim != 2 or eps.shape[1] != self.num_params:
            raise ValueError("block perturbations have the wrong shape")
        return np.column_stack(
            [
                np.mean(eps[:, block] ** 2, axis=1) - 1.0
                for block in self._component_slices
            ]
        )

    def _lopo_order_two_u_statistic_jackknife(
        self,
        pair_fitness: np.ndarray,
        block_features: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return the order-two U-statistic and delete-pair jackknife state."""
        pair_fitness = np.asarray(pair_fitness, dtype=np.float64)
        block_features = np.asarray(block_features, dtype=np.float64)
        pair_count = len(pair_fitness)
        if pair_fitness.shape != (pair_count, 2):
            raise ValueError("pair_fitness must have shape (m, 2)")
        if block_features.shape != (
            pair_count,
            self.num_curvature_components,
        ):
            raise ValueError("block_features have the wrong shape")
        if pair_count < 3:
            raise ValueError(
                "delete-pair U-statistic jackknife requires at least three pairs"
            )
        if not np.all(np.isfinite(pair_fitness)) or not np.all(
            np.isfinite(block_features)
        ):
            raise ValueError("U-statistic inputs must contain only finite values")

        comparisons = self._comparison_sign(
            pair_fitness[:, None, :, None],
            pair_fitness[None, :, None, :],
        )
        a_matrix = np.sum(comparisons, axis=(2, 3), dtype=np.int64).astype(
            np.float64
        )
        row_sums = a_matrix.sum(axis=1)
        scale = 16.0 * self.noise_std**2
        row_kernel_sums = (
            block_features * row_sums[:, None]
            - a_matrix @ block_features
        ) / scale
        total_kernel_sum = 0.5 * np.sum(row_kernel_sums, axis=0)
        pair_combinations = pair_count * (pair_count - 1.0) / 2.0
        estimate = total_kernel_sum / pair_combinations

        leave_pair_combinations = (
            (pair_count - 1.0) * (pair_count - 2.0) / 2.0
        )
        leave_one_pair_out = (
            total_kernel_sum[None, :] - row_kernel_sums
        ) / leave_pair_combinations
        jackknife_variance = (pair_count - 1.0) / pair_count * np.sum(
            (leave_one_pair_out - estimate[None, :]) ** 2,
            axis=0,
        )
        standard_error = np.sqrt(np.maximum(jackknife_variance, 0.0))
        outputs = (
            estimate,
            standard_error,
            leave_one_pair_out,
            row_kernel_sums,
            total_kernel_sum,
            a_matrix,
        )
        if not all(np.all(np.isfinite(value)) for value in outputs):
            raise FloatingPointError("LOPO U-statistic jackknife is non-finite")
        return outputs

    def _lopo_u_stat_curvature_estimate(
        self,
        noise: np.ndarray,
        fitness: np.ndarray,
        lopo_utilities: np.ndarray,
        pooled_utilities: np.ndarray,
        ask_info: dict[str, Any] | None,
        c_m: float,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        bool,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        dict[str, Any],
    ]:
        plus, minus = self._antithetic_pairs(noise, ask_info)
        eps = noise[plus]
        block_features = self._block_quadratic_features(eps)
        pair_utility = lopo_utilities[plus] + lopo_utilities[minus]
        pooled_pair_utility = pooled_utilities[plus] + pooled_utilities[minus]
        pair_contributions = (
            pair_utility[:, None]
            * block_features
            / (2.0 * self.noise_std**2)
        )
        raw_curvature = np.mean(pair_contributions, axis=0)
        pooled_raw_curvature = np.mean(
            pooled_pair_utility[:, None]
            * block_features
            / (2.0 * self.noise_std**2),
            axis=0,
        )
        pair_fitness = np.column_stack([fitness[plus], fitness[minus]])
        (
            u_statistic,
            standard_error,
            leave_one_pair_out,
            _,
            _,
            _,
        ) = self._lopo_order_two_u_statistic_jackknife(
            pair_fitness, block_features
        )

        pooled_identity_error = float(
            np.max(np.abs(raw_curvature - pooled_raw_curvature / c_m))
        )
        u_statistic_identity_error = float(
            np.max(np.abs(raw_curvature - u_statistic))
        )
        jackknife_mean_identity_error = float(
            np.max(
                np.abs(
                    np.mean(leave_one_pair_out, axis=0) - u_statistic
                )
            )
        )
        scale = max(
            1.0,
            float(np.max(np.abs(raw_curvature))),
            float(np.max(np.abs(pooled_raw_curvature / c_m))),
        )
        tolerance = 5e-12 * scale
        if (
            pooled_identity_error > tolerance
            or u_statistic_identity_error > tolerance
            or jackknife_mean_identity_error > tolerance
        ):
            raise FloatingPointError(
                "LOPO curvature identities failed beyond floating-point tolerance"
            )
        diagnostics: dict[str, Any] = {
            "lopo_c_m": float(c_m),
            "lopo_utility_semantics": (
                "exact_leave_own_antithetic_pair_out_midranks_no_recentering"
            ),
            "lopo_u_statistic_order": 2,
            "lopo_jackknife_method": (
                "delete_one_antithetic_pair_order_two_u_statistic"
            ),
            "lopo_jackknife_computation_valid": True,
            "lopo_jackknife_validity": (
                "asymptotic_if_iid_nondegenerate_pair_clusters"
            ),
            "lopo_jackknife_inference_assumptions_runtime_verified": False,
            "lopo_standard_error_scope": (
                "componentwise_asymptotic_non_simultaneous"
            ),
            "lopo_standard_error_target": (
                "raw_same_generation_block_u_statistic"
            ),
            "lopo_standard_error_optimization_coverage_calibrated": False,
            "lopo_across_pair_assumption": (
                "iid_nondegenerate_pair_clusters_for_asymptotic_inference"
            ),
            "lopo_within_pair_dependence_allowed": True,
            "lopo_raw_pooled_rescaling_max_abs_error": pooled_identity_error,
            "lopo_raw_u_statistic_max_abs_error": u_statistic_identity_error,
            "lopo_jackknife_mean_identity_max_abs_error": (
                jackknife_mean_identity_error
            ),
            "lopo_raw_identities_verified": True,
            "lopo_matching_scope": (
                "population_current_mid_cdf_stop_gradient_and_block_curvature"
            ),
        }
        return (
            raw_curvature.astype(np.float64, copy=False),
            standard_error.astype(np.float64, copy=False),
            True,
            eps,
            pair_utility,
            pair_contributions.astype(np.float64, copy=False),
            diagnostics,
        )

    def _stein_moment_estimate(
        self,
        eps: np.ndarray,
        pair_utility: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        scale = 2.0 * self.noise_std**2

        if self.curvature_structure == "diag":
            centered_squared_noise = eps * eps - 1.0
        else:
            centered_squared_noise = np.column_stack(
                [
                    np.mean(eps[:, block] ** 2, axis=1) - 1.0
                    for block in self._component_slices
                ]
            )
        pair_contributions = (
            pair_utility[:, None] * centered_squared_noise / scale
        )
        raw_curvature = np.mean(pair_contributions, axis=0)
        if not np.all(np.isfinite(raw_curvature)):
            raise FloatingPointError("matched concave curvature is non-finite")
        if len(pair_contributions) > 1:
            standard_error = np.std(
                pair_contributions, axis=0, ddof=1
            ) / np.sqrt(len(pair_contributions))
            standard_error_available = True
        else:
            standard_error = np.zeros_like(raw_curvature)
            standard_error_available = False
        return (
            raw_curvature.astype(np.float64, copy=False),
            pair_contributions.astype(np.float64, copy=False),
            standard_error.astype(np.float64, copy=False),
            standard_error_available,
        )

    def _fit_block_joint_ols(
        self,
        eps: np.ndarray,
        pair_utility: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Fit layer-isotropic quadratic curvature from all blocks jointly.

        For a block-isotropic quadratic with curvature ``h_B``, an antithetic
        pair sum has coefficient ``sigma**2 * d_B * h_B`` on
        ``mean(epsilon_B**2) - 1``.  The intercept absorbs the shared level.
        """
        if self.curvature_structure != "block":
            raise ValueError("block joint OLS requires block curvature structure")
        eps = np.asarray(eps, dtype=np.float64)
        pair_utility = np.asarray(pair_utility, dtype=np.float64)
        if eps.ndim != 2 or eps.shape[1] != self.num_params:
            raise ValueError("OLS perturbations have the wrong shape")
        if pair_utility.shape != (len(eps),):
            raise ValueError("OLS pair utilities have the wrong shape")

        features = np.column_stack(
            [
                np.mean(eps[:, block] ** 2, axis=1) - 1.0
                for block in self._component_slices
            ]
        )
        design = np.column_stack([np.ones(len(eps), dtype=np.float64), features])
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            design, pair_utility, rcond=None
        )
        parameter_count = design.shape[1]
        if rank != parameter_count:
            raise FloatingPointError(
                "block joint OLS design is rank deficient: "
                f"rank {rank}, expected {parameter_count}"
            )
        residual = pair_utility - design @ coefficients
        residual_dof = len(pair_utility) - parameter_count
        if residual_dof <= 0:
            raise FloatingPointError(
                "block joint OLS has no residual degrees of freedom"
            )
        residual_variance = float(np.dot(residual, residual) / residual_dof)
        covariance = residual_variance * np.linalg.pinv(
            design.T @ design, rcond=1e-12
        )
        coefficient_variance = np.maximum(np.diag(covariance)[1:], 0.0)
        block_sizes = np.asarray(
            [block.stop - block.start for block in self._component_slices],
            dtype=np.float64,
        )
        curvature_scale = self.noise_std**2 * block_sizes
        curvature = coefficients[1:] / curvature_scale
        standard_error = np.sqrt(coefficient_variance) / curvature_scale
        if not np.all(np.isfinite(curvature)) or not np.all(
            np.isfinite(standard_error)
        ):
            raise FloatingPointError("block joint OLS curvature is non-finite")

        centered_response = pair_utility - float(np.mean(pair_utility))
        total_sum_squares = float(np.dot(centered_response, centered_response))
        residual_sum_squares = float(np.dot(residual, residual))
        if total_sum_squares <= 1e-30:
            r_squared = 1.0 if residual_sum_squares <= 1e-30 else 0.0
        else:
            r_squared = 1.0 - residual_sum_squares / total_sum_squares
        smallest_singular = float(np.min(singular_values))
        condition = float(
            np.max(singular_values) / max(smallest_singular, 1e-300)
        )
        diagnostics: dict[str, Any] = {
            "regression_intercept": float(coefficients[0]),
            "regression_rank": int(rank),
            "regression_parameters": int(parameter_count),
            "regression_residual_dof": int(residual_dof),
            "regression_residual_std": float(np.sqrt(residual_variance)),
            "regression_r_squared": float(r_squared),
            "regression_design_condition": condition,
        }
        return (
            curvature.astype(np.float64, copy=False),
            standard_error.astype(np.float64, copy=False),
            diagnostics,
        )

    def _matched_curvature_estimate(
        self,
        noise: np.ndarray,
        utilities: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        bool,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        dict[str, Any],
    ]:
        eps, pair_utility = self._curvature_pair_data(
            noise, utilities, ask_info
        )
        if self.curvature_estimator == "stein_moment":
            raw, pair_contributions, standard_error, se_available = (
                self._stein_moment_estimate(eps, pair_utility)
            )
            fit_diagnostics: dict[str, Any] = {}
        else:
            raw, standard_error, fit_diagnostics = self._fit_block_joint_ols(
                eps, pair_utility
            )
            pair_contributions = None
            se_available = True
        return (
            raw,
            standard_error,
            se_available,
            eps,
            pair_utility,
            pair_contributions,
            fit_diagnostics,
        )

    def _matched_curvature_components(
        self,
        noise: np.ndarray,
        utilities: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the legacy Stein estimate and its pair contributions."""
        eps, pair_utility = self._curvature_pair_data(
            noise, utilities, ask_info
        )
        raw, pair_contributions, _, _ = self._stein_moment_estimate(
            eps, pair_utility
        )
        return raw, pair_contributions

    def _split_curvature_estimates(
        self,
        eps: np.ndarray,
        pair_utility: np.ndarray,
        pair_contributions: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.rank_utility_mode == "lopo_rank_u_statistic":
            return None
        split = len(pair_utility) // 2
        if split <= 0 or split >= len(pair_utility):
            return None
        if self.curvature_estimator == "stein_moment":
            if pair_contributions is None:
                raise RuntimeError("Stein split diagnostics require pair contributions")
            return (
                np.mean(pair_contributions[:split], axis=0),
                np.mean(pair_contributions[split:], axis=0),
            )
        first_half, _, _ = self._fit_block_joint_ols(
            eps[:split], pair_utility[:split]
        )
        second_half, _, _ = self._fit_block_joint_ols(
            eps[split:], pair_utility[split:]
        )
        return first_half, second_half

    def _independent_split_curvature_estimates(
        self,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Refit each split after independently ranking its own pair samples."""
        if self.rank_utility_mode == "lopo_rank_u_statistic":
            return None
        plus, minus = self._antithetic_pairs(noise, ask_info)
        split = len(plus) // 2
        if split <= 0 or split >= len(plus):
            return None

        estimates: list[np.ndarray] = []
        for pair_indices in (np.arange(split), np.arange(split, len(plus))):
            half_plus = plus[pair_indices]
            half_minus = minus[pair_indices]
            sample_indices = np.concatenate([half_plus, half_minus])
            half_utilities, _ = self._utilities(fitness[sample_indices])
            pair_count = len(pair_indices)
            half_pair_utility = (
                half_utilities[:pair_count]
                + half_utilities[pair_count:]
            )
            half_eps = noise[half_plus]
            if self.curvature_estimator == "stein_moment":
                estimate, _, _, _ = self._stein_moment_estimate(
                    half_eps, half_pair_utility
                )
            else:
                estimate, _, _ = self._fit_block_joint_ols(
                    half_eps, half_pair_utility
                )
            estimates.append(estimate)
        return estimates[0], estimates[1]

    def _split_rank_semantics(self) -> str:
        if self.rank_utility_mode == "lopo_rank_u_statistic":
            return "not_applicable_delete_pair_jackknife_replaces_split_rank"
        return "independent_centered_ranks_per_disjoint_pair_half"

    def _update_curvature_ema(
        self,
        raw_curvature: np.ndarray,
        standard_error: np.ndarray | None = None,
    ) -> np.ndarray:
        raw_curvature = np.asarray(raw_curvature, dtype=np.float64)
        if raw_curvature.shape != self.hessian_ema.shape:
            raise ValueError(
                "raw curvature shape does not match curvature components"
            )
        if not np.all(np.isfinite(raw_curvature)):
            raise ValueError("raw curvature must contain only finite values")
        if standard_error is None:
            standard_error = np.zeros_like(raw_curvature)
        standard_error = np.asarray(standard_error, dtype=np.float64)
        if standard_error.shape != raw_curvature.shape or not np.all(
            np.isfinite(standard_error)
        ) or np.any(standard_error < 0.0):
            raise ValueError(
                "curvature standard error must be finite, nonnegative, and match shape"
            )
        beta = self.curvature_beta
        self.hessian_ema = (
            beta * self.hessian_ema + (1.0 - beta) * raw_curvature
        )
        self.hessian_ema_variance = (
            beta**2 * self.hessian_ema_variance
            + (1.0 - beta) ** 2 * standard_error**2
        )
        self.hessian_ema_count += 1
        return self._hessian_for_step().copy()

    def _hessian_for_step(self) -> np.ndarray:
        if self.hessian_ema_count <= 0:
            return self.hessian_ema
        beta = self.curvature_beta
        correction = 1.0 - beta**self.hessian_ema_count
        return self.hessian_ema / correction

    def _hessian_standard_error_for_step(self) -> np.ndarray:
        if self.hessian_ema_count <= 0:
            return np.zeros_like(self.hessian_ema)
        correction = 1.0 - self.curvature_beta**self.hessian_ema_count
        return np.sqrt(np.maximum(self.hessian_ema_variance, 0.0)) / correction

    def _confidence_adjusted_concave(
        self,
        step_curvature: np.ndarray,
        step_standard_error: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        step_curvature = np.asarray(step_curvature, dtype=np.float64)
        step_standard_error = np.asarray(step_standard_error, dtype=np.float64)
        if step_curvature.shape != step_standard_error.shape:
            raise ValueError("curvature and standard error shapes must match")
        if self.curvature_confidence_z is None:
            upper_bound = step_curvature.copy()
        else:
            upper_bound = (
                step_curvature
                + self.curvature_confidence_z * step_standard_error
            )
        return np.maximum(-upper_bound, 0.0), upper_bound

    def _expand_components(self, components: np.ndarray) -> np.ndarray:
        components = np.asarray(components, dtype=np.float64)
        if components.shape != (len(self._component_slices),):
            raise ValueError("curvature components have the wrong shape")
        expanded = np.empty(self.num_params, dtype=np.float64)
        for value, block in zip(components, self._component_slices, strict=True):
            expanded[block] = value
        return expanded

    @staticmethod
    def _relative_disagreement(left: np.ndarray, right: np.ndarray) -> float:
        return float(
            np.linalg.norm(left - right)
            / max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-12)
        )

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta_t, noise, fitness = self._validate_fresh_batch(
            params, noise, fitness, ask_info, center_fitness
        )
        pooled_utilities, transform = self._utilities(fitness)
        if self.rank_utility_mode == "lopo_rank_u_statistic":
            (
                utilities,
                pooled_utilities,
                c_m,
                _,
                proposal_gradient,
                gradient_diagnostics,
            ) = self._lopo_gradient_estimate(
                theta_t, noise, fitness, ask_info
            )
            transform = "exact_leave_own_pair_out_midrank"
        else:
            utilities = pooled_utilities
            c_m = 1.0
            _, proposal_gradient = self._proposal_gradient(
                theta_t, noise, utilities
            )
            gradient_diagnostics = {}
        if self.rank_utility_mode == "lopo_rank_u_statistic":
            (
                raw_components,
                raw_standard_error,
                raw_standard_error_available,
                pair_eps,
                pair_utility,
                pair_contributions,
                fit_diagnostics,
            ) = self._lopo_u_stat_curvature_estimate(
                noise,
                fitness,
                utilities,
                pooled_utilities,
                ask_info,
                c_m,
            )
            fit_diagnostics.update(gradient_diagnostics)
        else:
            (
                raw_components,
                raw_standard_error,
                raw_standard_error_available,
                pair_eps,
                pair_utility,
                pair_contributions,
                fit_diagnostics,
            ) = self._matched_curvature_estimate(
                noise, utilities, ask_info
            )
        step_components = self._update_curvature_ema(
            raw_components, raw_standard_error
        )
        step_standard_error = self._hessian_standard_error_for_step()
        concave_components, confidence_upper = (
            self._confidence_adjusted_concave(
                step_components, step_standard_error
            )
        )

        alpha = float(self.learning_rate)
        denominator_components = 1.0 + alpha * (
            self.implicit_damping + self.l2_coeff + concave_components
        )
        denominator = self._expand_components(denominator_components)
        if not np.all(np.isfinite(denominator)) or np.any(denominator < 1.0):
            raise FloatingPointError(
                "concave-projected denominator must be finite and at least one"
            )
        rhs = alpha * proposal_gradient
        structured_step = rhs / denominator
        structured_residual = denominator * structured_step - rhs
        structured_relative_residual = float(
            np.linalg.norm(structured_residual)
            / max(float(np.linalg.norm(rhs)), 1e-12)
        )
        rhs_norm = float(np.linalg.norm(rhs))
        structured_step_norm = float(np.linalg.norm(structured_step))
        if self.attenuation_mode == "isotropic_norm_matched":
            attenuation_scale = (
                1.0 if rhs_norm <= 1e-15 else structured_step_norm / rhs_norm
            )
            if (
                not np.isfinite(attenuation_scale)
                or attenuation_scale <= 0.0
                or attenuation_scale > 1.0 + 1e-12
            ):
                raise FloatingPointError(
                    "isotropic attenuation scale must be finite and in (0, 1]"
                )
            attenuation_scale = min(attenuation_scale, 1.0)
            step = attenuation_scale * rhs
            residual = step - attenuation_scale * rhs
            residual_scale = max(structured_step_norm, 1e-12)
            norm_match_relative_error = float(
                abs(float(np.linalg.norm(step)) - structured_step_norm)
                / residual_scale
            )
        else:
            attenuation_scale = 1.0
            step = structured_step
            residual = structured_residual
            residual_scale = max(rhs_norm, 1e-12)
            norm_match_relative_error = 0.0
        relative_residual = float(np.linalg.norm(residual) / residual_scale)
        solve_success = bool(
            np.isfinite(relative_residual)
            and relative_residual <= 1e-10
            and np.isfinite(structured_relative_residual)
            and structured_relative_residual <= 1e-10
            and norm_match_relative_error <= 1e-10
        )

        split_estimates = self._independent_split_curvature_estimates(
            noise, fitness, ask_info
        )
        if split_estimates is not None:
            first_half, second_half = split_estimates
            split_correlation = self._correlation(first_half, second_half)
            split_sign_agreement = float(
                np.mean(np.sign(first_half) == np.sign(second_half))
            )
            split_relative_disagreement = self._relative_disagreement(
                first_half, second_half
            )
            split_available = True
            split_first_components = first_half.tolist()
            split_second_components = second_half.tolist()
        else:
            split_correlation = 0.0
            split_sign_agreement = 0.0
            split_relative_disagreement = 0.0
            split_available = False
            split_first_components = []
            split_second_components = []

        if self._previous_raw_curvature is None:
            temporal_correlation = 0.0
            temporal_sign_agreement = 0.0
            temporal_relative_disagreement = 0.0
            temporal_available = False
        else:
            temporal_correlation = self._correlation(
                self._previous_raw_curvature, raw_components
            )
            temporal_sign_agreement = float(
                np.mean(
                    np.sign(self._previous_raw_curvature)
                    == np.sign(raw_components)
                )
            )
            temporal_relative_disagreement = self._relative_disagreement(
                self._previous_raw_curvature, raw_components
            )
            temporal_available = True
        self._previous_raw_curvature = raw_components.copy()

        raw_curvature = self._expand_components(raw_components)
        step_curvature = self._expand_components(step_components)
        concave_curvature = self._expand_components(concave_components)
        ema_curvature = self._expand_components(self.hessian_ema)
        denominator_min = float(np.min(denominator))
        denominator_max = float(np.max(denominator))
        projection_component_frac = float(np.mean(step_components > 0.0))
        projection_parameter_frac = float(np.mean(step_curvature > 0.0))
        confidence_gate_enabled = self.curvature_confidence_z is not None
        legacy_active = step_components < 0.0
        confidence_pass = confidence_upper < 0.0
        confidence_blocked = legacy_active & ~confidence_pass
        block_sizes = [
            int(block.stop - block.start) for block in self._component_slices
        ]
        extra_info: dict[str, Any] = {
            "linear_relative_residual": relative_residual,
            "solve_relative_residual": relative_residual,
            "structured_reference_relative_residual": (
                structured_relative_residual
            ),
            "linear_diagonal_min": denominator_min,
            "linear_diagonal_max": denominator_max,
            "linear_min_abs_diagonal": denominator_min,
            "linear_condition_estimate": float(denominator_max / denominator_min),
            "linear_nonpositive_diagonal_frac": 0.0,
            "denominator_min": denominator_min,
            "denominator_max": denominator_max,
            "denominator_condition": float(denominator_max / denominator_min),
            "hessian_pairs": int(len(pair_utility)),
            "h_raw_mean": float(np.mean(raw_curvature)),
            "h_raw_std": float(np.std(raw_curvature)),
            "h_raw_min": float(np.min(raw_curvature)),
            "h_raw_max": float(np.max(raw_curvature)),
            "h_raw_norm": float(np.linalg.norm(raw_curvature)),
            "h_ema_mean": float(np.mean(ema_curvature)),
            "h_ema_min": float(np.min(ema_curvature)),
            "h_ema_max": float(np.max(ema_curvature)),
            "h_step_mean": float(np.mean(step_curvature)),
            "h_step_std": float(np.std(step_curvature)),
            "h_step_min": float(np.min(step_curvature)),
            "h_step_max": float(np.max(step_curvature)),
            "h_step_norm": float(np.linalg.norm(step_curvature)),
            "curv_mean": float(np.mean(concave_curvature)),
            "curv_min": float(np.min(concave_curvature)),
            "curv_max": float(np.max(concave_curvature)),
            "curvature_active_frac": float(np.mean(concave_curvature > 0.0)),
            "curvature_projection_frac": projection_component_frac,
            "curvature_projection_parameter_frac": projection_parameter_frac,
            "curvature_clip_frac": 0.0,
            "curvature_estimator": self.curvature_estimator,
            "rank_utility_mode": self.rank_utility_mode,
            "curvature_attenuation_mode": self.attenuation_mode,
            "isotropic_attenuation_scale": attenuation_scale,
            "structured_reference_step_norm": structured_step_norm,
            "attenuation_norm_match_relative_error": (
                norm_match_relative_error
            ),
            "curvature_same_generation_components": raw_components.tolist(),
            "curvature_same_generation_se_components": raw_standard_error.tolist(),
            "curvature_same_generation_se_available": bool(
                raw_standard_error_available
            ),
            "curvature_same_generation_se_mean": float(
                np.mean(raw_standard_error)
            ),
            "curvature_same_generation_se_max": float(
                np.max(raw_standard_error)
            ),
            "curvature_step_state": (
                "same_generation"
                if self.curvature_beta == 0.0
                else "bias_corrected_ema"
            ),
            "curvature_step_state_components": step_components.tolist(),
            "curvature_step_state_se_components": step_standard_error.tolist(),
            "curvature_step_state_se_mean": float(
                np.mean(step_standard_error)
            ),
            "curvature_step_state_se_max": float(
                np.max(step_standard_error)
            ),
            "curvature_confidence_gate_enabled": confidence_gate_enabled,
            "curvature_confidence_z": self.curvature_confidence_z,
            "curvature_confidence_upper_components": confidence_upper.tolist(),
            "curvature_confidence_pass_frac": float(np.mean(confidence_pass)),
            "curvature_confidence_gate_frac": float(
                np.mean(confidence_blocked)
                if confidence_gate_enabled
                else 0.0
            ),
            "h_split_available": split_available,
            "h_split_correlation": split_correlation,
            "h_split_sign_agreement": split_sign_agreement,
            "h_split_relative_disagreement": split_relative_disagreement,
            "h_split_first_components": split_first_components,
            "h_split_second_components": split_second_components,
            "h_split_rank_semantics": self._split_rank_semantics(),
            "h_split_pair_partition": "first_vs_second_antithetic_pair_halves",
            "h_split_first_pair_count": int(len(pair_utility) // 2),
            "h_split_second_pair_count": int(
                len(pair_utility) - len(pair_utility) // 2
            ),
            "h_temporal_available": temporal_available,
            "h_temporal_correlation": temporal_correlation,
            "h_temporal_sign_agreement": temporal_sign_agreement,
            "h_temporal_relative_disagreement": temporal_relative_disagreement,
            "hessian_ema_count": int(self.hessian_ema_count),
            "curvature_beta": float(self.curvature_beta),
            "curvature_same_generation": self.curvature_same_generation,
            "curvature_components": int(self.num_curvature_components),
            "curvature_component_count": int(self.num_curvature_components),
            "curvature_block_sizes": block_sizes,
            "curvature_block_size_min": int(min(block_sizes)),
            "curvature_block_size_mean": float(np.mean(block_sizes)),
            "curvature_block_size_max": int(max(block_sizes)),
            "curvature_raw_components": raw_components.tolist(),
            "curvature_ema_components": self.hessian_ema.tolist(),
            "curvature_ema_variance_components": (
                self.hessian_ema_variance.tolist()
            ),
            "curvature_bias_corrected_ema_components": step_components.tolist(),
            "curvature_step_components": step_components.tolist(),
            "concave_curvature_components": concave_components.tolist(),
            "denominator_components": denominator_components.tolist(),
            "curvature_fitness": "matched",
            "curvature_matches_gradient": True,
            "curvature_mode": self.curvature_structure,
            "curvature_structure": self.curvature_structure,
            "step_over_sigma": float(np.linalg.norm(step) / self.noise_std),
            **fit_diagnostics,
        }
        return self._finish_update(
            theta_t,
            fitness,
            proposal_gradient,
            step,
            solver_type=(
                (
                    "concave_projected_block_lopo_rank_u_statistic"
                    if self.attenuation_mode == "structured"
                    else (
                        "concave_projected_block_lopo_rank_u_statistic_"
                        "isotropic_norm_control"
                    )
                )
                if self.rank_utility_mode == "lopo_rank_u_statistic"
                else (
                    f"concave_projected_{self.curvature_structure}"
                    if self.attenuation_mode == "structured"
                    else "concave_projected_block_isotropic_attenuation_control"
                )
            ),
            solve_success=solve_success,
            fitness_transform=transform,
            extra_info=extra_info,
        )


class LOPOGradientES(ConcaveCurvatureES):
    """Fresh explicit ES using exact LOPO rank utilities and no curvature."""

    def __init__(
        self,
        *args: Any,
        block_slices: Sequence[slice | Sequence[int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            curvature_structure="block",
            block_slices=block_slices,
            curvature_beta=0.0,
            curvature_estimator="stein_moment",
            curvature_confidence_z=None,
            rank_utility_mode="lopo_rank_u_statistic",
            attenuation_mode="structured",
            **kwargs,
        )
        self.use_curvature = False
        self.curvature_fitness = "none"
        self.curvature_mode = "none"
        self.persist_hessian_ema_artifact = False

    def tell(
        self,
        params: np.ndarray,
        noise: np.ndarray,
        fitness: np.ndarray,
        ask_info: dict[str, Any] | None = None,
        center_fitness: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        theta_t, noise, fitness = self._validate_fresh_batch(
            params, noise, fitness, ask_info, center_fitness
        )
        (
            _,
            _,
            _,
            _,
            proposal_gradient,
            diagnostics,
        ) = self._lopo_gradient_estimate(
            theta_t, noise, fitness, ask_info
        )
        diagnostics.pop(
            "lopo_raw_block_moment_is_at_proposal_frozen_utility_sn_jacobian_diagonal_block_average"
        )
        diagnostics.pop("lopo_raw_block_moment_endpoint_jacobian_scope")
        step = float(self.learning_rate) * proposal_gradient
        diagnostics.update(
            {
                "optimizer_type": "lopo_gradient_es",
                "rank_utility_mode": "lopo_rank_u_statistic",
                "curvature_used": False,
                "curvature_estimator": "none",
                "curvature_fitness": "none",
                "curvature_mode": "none",
                "curvature_structure": "none",
                "curvature_attenuation_mode": "none",
                "curvature_components": 0,
                "curvature_matches_gradient": False,
                "lopo_matching_scope": (
                    "population_current_mid_cdf_stop_gradient"
                ),
                "lopo_raw_block_moment_endpoint_jacobian_claim_applicability": (
                    "not_applicable_no_curvature_operator"
                ),
                "step_over_sigma": float(np.linalg.norm(step) / self.noise_std),
            }
        )
        return self._finish_update(
            theta_t,
            fitness,
            proposal_gradient,
            step,
            solver_type="explicit_lopo_rank_gradient",
            solve_success=True,
            fitness_transform="exact_leave_own_pair_out_midrank",
            extra_info=diagnostics,
        )

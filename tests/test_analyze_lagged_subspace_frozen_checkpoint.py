#!/usr/bin/env python3
"""Adversarial fixtures for the lagged-subspace audit-index validator."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

import scripts.analyze_lagged_subspace_frozen_checkpoint as analysis
from core.lagged_subspace_diagnostic import (
    BasisProvenance,
    analyze_lagged_subspace_population,
    build_lagged_bases,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _stamp(record: dict[str, object]) -> dict[str, object]:
    record["record_sha256"] = analysis._record_sha256(record)
    return record


def _fixture_manifest() -> dict[str, object]:
    with open(analysis.DEFAULT_MANIFEST_PATH, encoding="utf-8") as stream:
        manifest = copy.deepcopy(json.load(stream))
    manifest["training_seeds"] = [300, 301]
    manifest["checkpoint_generations"] = [2, 4]
    for task in manifest["tasks"]:
        task.update(
            {
                "observation_dim": 1,
                "action_dim": 1,
                "parameter_count": 6,
                "policy_block_sizes": [2, 2, 2],
                "policy_block_ranges": [[0, 2], [2, 4], [4, 6]],
            }
        )
    dims = manifest["dimensions"]
    dims.update(
        {
            "population_size": 4,
            "antithetic_pairs_per_training_update": 2,
            "training_updates": 4,
            "lagged_gradient_count": 2,
            "pairs_per_bank": 6,
            "bank_b_partition_count": 2,
            "pairs_per_partition": 3,
            "endpoint_episodes": 2,
        }
    )
    manifest["analysis"]["bootstrap_resamples"] = 200
    fixture_delta = 1.0 / 4.0
    manifest["analysis"].update(
        {
            "mechanism_seed_count": 2,
            "lower_order_index_zero_based": 0,
            "upper_order_index_zero_based": 1,
            "per_bound_error_numerator": 1,
            "per_bound_error_denominator": 4,
            "per_bound_error": fixture_delta,
            "simultaneous_coverage_lower_bound": 1.0
            - 12.0 * fixture_delta,
            "endpoint_family_alpha": 0.05 - 12.0 * fixture_delta,
        }
    )
    manifest["analysis"]["gate_thresholds"][
        "endpoint_adjusted_one_sided_p_upper"
    ] = manifest["analysis"]["endpoint_family_alpha"]
    task_count = len(manifest["tasks"])
    seed_count = len(manifest["training_seeds"])
    checkpoint_count = (
        task_count * seed_count * len(manifest["checkpoint_generations"])
    )
    budget = {
        "checkpoint_training_candidate_rollouts": task_count
        * seed_count
        * dims["training_updates"]
        * dims["population_size"],
        "normalization_calibration_rollouts": task_count
        * seed_count
        * dims["calibration_episodes"],
        "bank_candidate_rollouts": checkpoint_count
        * len(dims["banks"])
        * dims["pairs_per_bank"]
        * 2,
        "endpoint_arm_rollouts": checkpoint_count
        * len(dims["locality_q"])
        * dims["bank_b_partition_count"]
        * len(dims["endpoint_arms"])
        * dims["endpoint_episodes"],
        "checkpoint_center_rollouts": checkpoint_count
        * dims["endpoint_episodes"],
    }
    budget["total_policy_rollouts"] = sum(budget.values())
    budget["environment_transitions_are_separate"] = True
    manifest["budget"] = budget
    return manifest


def _q_summary(
    manifest: dict[str, object],
    checkpoint_id: int,
    q: float,
    *,
    label: str,
    gradient_norm: float,
    structured_norm: float,
    anisotropic_norm: float,
    distance_to_a: float,
    alpha: float,
) -> dict[str, object]:
    epsilon = manifest["analysis"]["machine_epsilon"]
    action_hashes = {
        arm: _digest(f"{label}-{arm}")
        for arm in manifest["dimensions"]["endpoint_arms"]
    }
    return {
        "q": q,
        "alpha": alpha,
        "alpha_resolved": True,
        "alpha_unresolved_reason": None,
        "gradient_norm": gradient_norm,
        "structured_norm": structured_norm,
        "isotropic_norm": structured_norm,
        "explicit_norm": alpha * gradient_norm,
        "random_norm": structured_norm,
        "random_raw_norm": structured_norm * 1.1,
        "anisotropic_action_norm": anisotropic_norm,
        "anisotropic_minus_bank_a_norm": distance_to_a,
        "structured_step_over_sigma": structured_norm
        / manifest["dimensions"]["noise_std"],
        "structured_solve_residual": 0.0,
        "random_solve_residual": 0.0,
        "structured_isotropic_relative_norm_error": analysis._relative_error(
            structured_norm, structured_norm, epsilon
        ),
        "structured_random_relative_norm_error": analysis._relative_error(
            structured_norm, structured_norm, epsilon
        ),
        "random_control_valid": True,
        "material_denominator_resolved": True,
        "finite": True,
        "action_sha256": action_hashes,
    }


def _write_npz(root: str, relative_path: str, arrays: dict[str, np.ndarray]) -> tuple[str, str]:
    path = os.path.join(root, relative_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    return relative_path, analysis._sha256_file(path)


def _diagnostic_payload(
    manifest: dict[str, object],
    theta: np.ndarray,
    signed_noise: np.ndarray,
    paired_returns: np.ndarray,
    basis: np.ndarray,
    random_basis: np.ndarray,
    alphas: list[float],
    reference_results: list[object] | None,
    endpoint_reference: dict[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[object]]:
    provenance = BasisProvenance.strictly_lagged("fixture-primary", "fixture-random")
    endpoint_kwargs = (
        {}
        if endpoint_reference is None
        else {
            "endpoint_reference_noise": endpoint_reference["signed_noise"],
            "endpoint_reference_utilities": endpoint_reference["utilities"],
            "endpoint_reference_gradient": endpoint_reference["gradient"],
            "endpoint_reference_curvature": endpoint_reference["curvature"],
        }
    )
    results = [
        analyze_lagged_subspace_population(
            theta,
            signed_noise,
            paired_returns,
            manifest["dimensions"]["noise_std"],
            basis,
            random_basis,
            alpha,
            basis_provenance=provenance,
            **endpoint_kwargs,
        )
        for alpha in alphas
    ]
    if reference_results is None:
        reference_results = results
    first = results[0]
    payload: dict[str, np.ndarray] = {
        "utilities": first.estimate.utilities,
        "gradient": first.estimate.gradient,
        "curvature": first.estimate.curvature,
        "random_curvature": first.estimate.random_curvature,
        "gradient_component_variance": np.square(
            first.estimate.gradient_jackknife.standard_error
        ),
        "curvature_vech_covariance": first.estimate.curvature_vech_jackknife.covariance,
        "curvature_eigenvalues": np.stack(
            [result.steps.curvature_eigenvalues for result in results]
        ),
        "negative_eigenvalue_count": np.asarray(
            np.sum(first.steps.curvature_eigenvalues < 0.0), dtype=np.int64
        ),
        "jackknife_eigenvalue_se": np.stack(
            [
                np.sqrt(
                    np.maximum(
                        np.diag(result.nonlinear_jackknife.eigenvalue_covariance),
                        0.0,
                    )
                )
                for result in results
            ]
        ),
        "structured_action_covariance_trace": np.asarray(
            [
                result.nonlinear_jackknife.structured_action_covariance_trace
                for result in results
            ]
        ),
        "anisotropic_action_covariance_trace": np.asarray(
            [
                result.nonlinear_jackknife.anisotropic_action_covariance_trace
                for result in results
            ]
        ),
        "anisotropic_action_aligned_variance": np.asarray(
            [
                0.0
                if result.nonlinear_jackknife.anisotropic_action_aligned_variance
                is None
                else result.nonlinear_jackknife.anisotropic_action_aligned_variance
                for result in results
            ]
        ),
        "repeated_eigenvalue_unresolved": np.asarray(
            [
                result.nonlinear_jackknife.repeated_eigenvalue_unresolved
                for result in results
            ],
            dtype=np.bool_,
        ),
        "projection_boundary_unresolved": np.asarray(
            [
                result.nonlinear_jackknife.projection_boundary_unresolved
                for result in results
            ],
            dtype=np.bool_,
        ),
        "zero_anisotropic_action_unresolved": np.asarray(
            [
                result.nonlinear_jackknife.zero_anisotropic_action_unresolved
                for result in results
            ],
            dtype=np.bool_,
        ),
        "gradient_endpoint_relative_error": np.asarray(
            first.gradient_endpoint_relative_error
        ),
        "subspace_jacobian_relative_error": np.asarray(
            first.subspace_jacobian_relative_error
        ),
        "self_normalized_gradient_relative_error": np.asarray(
            first.self_normalized_gradient_relative_error
        ),
        "self_normalized_jacobian_relative_error": np.asarray(
            first.self_normalized_jacobian_relative_error
        ),
    }
    reference_curvature = reference_results[0].estimate.curvature
    payload["b_frobenius_absolute_error_to_bank_a"] = np.asarray(
        np.linalg.norm(first.estimate.curvature - reference_curvature)
    )
    payload["negative_eigenvalue_sign_agreement_to_bank_a"] = np.asarray(
        np.mean(
            np.sign(np.linalg.eigvalsh(first.estimate.curvature))
            == np.sign(np.linalg.eigvalsh(reference_curvature))
        )
    )
    reference_actions = [
        result.steps.structured - result.steps.isotropic
        for result in reference_results
    ]
    actions = [result.steps.structured - result.steps.isotropic for result in results]
    payload["anisotropic_action_cosine_to_bank_a"] = np.asarray(
        [analysis._cosine(action, reference) for action, reference in zip(actions, reference_actions)]
    )
    payload["anisotropic_action_relative_error_to_bank_a"] = np.asarray(
        [
            np.linalg.norm(action - reference)
            / max(np.linalg.norm(reference), np.finfo(np.float64).eps)
            for action, reference in zip(actions, reference_actions)
        ]
    )
    arms = manifest["dimensions"]["endpoint_arms"]
    endpoint_fields = {
        "r_full": "full_linearization_residual",
        "r_sub": "restricted_linearization_residual",
        "r_sn": "self_normalized_linearization_residual",
        "normalized_ess_ratio": "normalized_ess_ratio",
        "ratio_coefficient_of_variation": "ratio_coefficient_of_variation",
        "mean_unnormalized_ratio_minus_one": "mean_unnormalized_ratio_minus_one",
        "log_ratio_span": "log_ratio_span",
    }
    for output, attribute in endpoint_fields.items():
        values = [
            [getattr(result.endpoint_diagnostics[arm], attribute) for arm in arms]
            for result in results
        ]
        payload[output] = np.asarray(
            [[0.0 if value is None else value for value in row] for row in values]
        )
        if output in {"r_full", "r_sub", "r_sn"}:
            payload[f"{output}_unresolved"] = np.asarray(
                [[value is None for value in row] for row in values],
                dtype=np.bool_,
            )
    payload["alpha_max_concave_eigenvalue"] = np.asarray(
        [result.action_metrics.alpha_max_concave_eigenvalue for result in results]
    )
    payload["structured_explicit_angle_degrees"] = np.asarray(
        [
            0.0
            if result.action_metrics.structured_explicit_angle_degrees is None
            else result.action_metrics.structured_explicit_angle_degrees
            for result in results
        ]
    )
    payload["multiplier_standard_deviation"] = np.asarray(
        [result.action_metrics.multiplier_standard_deviation for result in results]
    )
    payload["multiplier_range"] = np.asarray(
        [result.action_metrics.multiplier_range for result in results]
    )
    summaries = []
    for q_index, (q, alpha, result, reference_action) in enumerate(
        zip(manifest["dimensions"]["locality_q"], alphas, results, reference_actions)
    ):
        steps = result.steps
        action = actions[q_index]
        step_map = {
            "structured": steps.structured,
            "isotropic": steps.isotropic,
            "explicit": steps.explicit,
            "random": steps.random,
        }
        for arm, step in step_map.items():
            payload[f"step_q{q_index}_{arm}"] = step
        summaries.append(
            {
                "q": q,
                "alpha": alpha,
                "alpha_resolved": True,
                "alpha_unresolved_reason": None,
                "gradient_norm": analysis._stable_norm(result.estimate.gradient),
                "structured_norm": float(np.linalg.norm(steps.structured)),
                "isotropic_norm": float(np.linalg.norm(steps.isotropic)),
                "explicit_norm": float(np.linalg.norm(steps.explicit)),
                "random_norm": float(np.linalg.norm(steps.random)),
                "random_raw_norm": float(np.linalg.norm(steps.random_raw)),
                "anisotropic_action_norm": float(np.linalg.norm(action)),
                "anisotropic_minus_bank_a_norm": float(
                    np.linalg.norm(action - reference_action)
                ),
                "structured_step_over_sigma": float(
                    np.linalg.norm(steps.structured)
                    / manifest["dimensions"]["noise_std"]
                ),
                "structured_solve_residual": steps.structured_solve_relative_residual,
                "random_solve_residual": steps.random_solve_relative_residual,
                "structured_isotropic_relative_norm_error": steps.isotropic_norm_match_relative_error,
                "structured_random_relative_norm_error": (
                    0.0
                    if steps.random_norm_match_relative_error is None
                    else steps.random_norm_match_relative_error
                ),
                "random_control_valid": steps.random_control_valid,
                "material_denominator_resolved": bool(
                    np.linalg.norm(steps.structured) > np.finfo(np.float64).eps
                ),
                "finite": True,
                "action_sha256": {
                    arm: analysis._array_sha256(step)
                    for arm, step in step_map.items()
                },
            }
        )
    omitted = {"gradient", "gradient_component_variance"} | {
        f"step_q{q_index}_{arm}"
        for q_index in range(len(manifest["dimensions"]["locality_q"]))
        for arm in arms
    }
    compact = {name: value for name, value in payload.items() if name not in omitted}
    compact["gradient_sha256"] = np.asarray(
        analysis._array_sha256(payload["gradient"]), dtype="S64"
    )
    compact["gradient_component_variance_sha256"] = np.asarray(
        analysis._array_sha256(payload["gradient_component_variance"]), dtype="S64"
    )
    compact["step_sha256"] = np.asarray(
        [
            [
                analysis._array_sha256(payload[f"step_q{q_index}_{arm}"])
                for arm in arms
            ]
            for q_index in range(len(manifest["dimensions"]["locality_q"]))
        ],
        dtype="S64",
    )
    return compact, summaries, results


def _fixture_jackknife_sha256(
    payload: dict[str, np.ndarray], first: object
) -> str:
    return analysis._labeled_arrays_sha256(
        [
            (
                "gradient_component_variance",
                np.square(first.estimate.gradient_jackknife.standard_error),
            ),
            *[
                (name, payload[name])
                for name in (
                    "curvature_vech_covariance",
                    "jackknife_eigenvalue_se",
                    "structured_action_covariance_trace",
                    "anisotropic_action_covariance_trace",
                    "anisotropic_action_aligned_variance",
                )
            ],
        ]
    )


def _fixture(
    artifact_root: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, str]]:
    manifest = _fixture_manifest()
    dims = manifest["dimensions"]
    hashes = {name: _digest(name) for name in manifest["required_hash_locks"]}
    hashes["protocol_sha256"] = manifest["protocol"]["sha256"]
    training_runs = []
    checkpoints = []
    banks = []
    partitions = []
    metrics = []
    centers = []
    endpoints = []

    for task in manifest["tasks"]:
        for seed in manifest["training_seeds"]:
            training_id = analysis.training_id_for(manifest, task["task_index"], seed)
            training_runs.append(
                _stamp(
                    {
                        "training_id": training_id,
                        "task_index": task["task_index"],
                        "env_name": task["env_name"],
                        "training_seed": seed,
                        "updates": dims["training_updates"],
                        "population_size": dims["population_size"],
                        "candidate_rollouts": dims["training_updates"]
                        * dims["population_size"],
                        "calibration_rollouts": dims["calibration_episodes"],
                        "online_evaluation_rollouts": 0,
                        "checkpoint_generations": manifest[
                            "checkpoint_generations"
                        ],
                        "training_transitions": 100 + training_id,
                        "calibration_transitions": 10 + training_id,
                        "training_log_sha256": _digest(f"training-{training_id}"),
                        "stderr_sha256": analysis.EMPTY_SHA256,
                        "stderr_empty": True,
                    }
                )
            )
            for generation in manifest["checkpoint_generations"]:
                checkpoint_id = analysis.checkpoint_id_for(
                    manifest, task["task_index"], seed, generation
                )
                checkpoints.append(
                    _stamp(
                        {
                            "checkpoint_id": checkpoint_id,
                            "training_id": training_id,
                            "task_index": task["task_index"],
                            "env_name": task["env_name"],
                            "training_seed": seed,
                            "generation": generation,
                            "parameter_sha256": _digest(f"parameter-{checkpoint_id}"),
                            "observation_normalizer_sha256": _digest(
                                f"normalizer-{checkpoint_id}"
                            ),
                            "training_config_sha256": _digest("training-config"),
                            "source_sha256": hashes["source_sha256"],
                            "prior_gradient_indices": list(
                                range(
                                    generation - dims["lagged_gradient_count"],
                                    generation,
                                )
                            ),
                            "prior_gradient_sha256": [
                                _digest(f"gradient-{checkpoint_id}-{index}")
                                for index in range(dims["lagged_gradient_count"])
                            ],
                            "lagged_block_norms": [1.0, 2.0, 3.0],
                            "lagged_block_exact_zero": [False, False, False],
                            "primary_gaussian_fallback_used": [False, False, False],
                            "random_control_permuted_fallback_used": [
                                False,
                                False,
                                False,
                            ],
                            "fallback_column_sha256": [None, None, None],
                            "basis_seed": analysis.checkpoint_seed(
                                manifest, "basis", checkpoint_id
                            ),
                            "random_control_seed": analysis.checkpoint_seed(
                                manifest, "random_control", checkpoint_id
                            ),
                            "basis_sha256": _digest(f"basis-{checkpoint_id}"),
                            "random_basis_sha256": _digest(
                                f"random-basis-{checkpoint_id}"
                            ),
                            "basis_locked_before_bank_sampling": True,
                            "checkpoint_artifact_sha256": _digest(
                                f"checkpoint-{checkpoint_id}"
                            ),
                        }
                    )
                )

                seed_factor = 1.0 + 0.01 * (seed - 300)
                a_by_q: dict[float, dict[str, float]] = {}
                for bank_index, bank in enumerate(dims["banks"]):
                    pair_indices = list(range(dims["pairs_per_bank"]))
                    gradient_norm = 2.0 if bank == "A" else 2.2
                    q_summaries = []
                    for q in dims["locality_q"]:
                        alpha = q * dims["noise_std"] / 2.0
                        structured_norm = alpha * 1.5 * seed_factor
                        anisotropic_norm = structured_norm * (
                            0.10 if bank == "A" else 0.105
                        )
                        if bank == "A":
                            a_by_q[q] = {
                                "structured": structured_norm,
                                "anisotropic": anisotropic_norm,
                            }
                        q_summaries.append(
                            _q_summary(
                                manifest,
                                checkpoint_id,
                                q,
                                label=f"bank-{checkpoint_id}-{bank}-{q}",
                                gradient_norm=gradient_norm,
                                structured_norm=structured_norm,
                                anisotropic_norm=anisotropic_norm,
                                distance_to_a=0.0,
                                alpha=alpha,
                            )
                        )
                    perturbation_seeds = [
                        analysis.pair_seed(
                            manifest,
                            "bank_perturbation",
                            checkpoint_id,
                            bank,
                            pair_index,
                        )
                        for pair_index in pair_indices
                    ]
                    rollout_seeds = [
                        analysis.pair_seed(
                            manifest,
                            "bank_rollout",
                            checkpoint_id,
                            bank,
                            pair_index,
                        )
                        for pair_index in pair_indices
                    ]
                    banks.append(
                        _stamp(
                            {
                                "bank_id": analysis.bank_id_for(
                                    manifest, checkpoint_id, bank
                                ),
                                "checkpoint_id": checkpoint_id,
                                "bank": bank,
                                "pair_count": dims["pairs_per_bank"],
                                "candidate_rollouts": 2
                                * dims["pairs_per_bank"],
                                "pair_indices": pair_indices,
                                "perturbation_seeds": perturbation_seeds,
                                "rollout_seeds_plus": rollout_seeds,
                                "rollout_seeds_minus": rollout_seeds,
                                "perturbations_sha256": _digest(
                                    f"perturbations-{checkpoint_id}-{bank}"
                                ),
                                "returns_sha256": _digest(
                                    f"returns-{checkpoint_id}-{bank}"
                                ),
                                "transitions_sha256": _digest(
                                    f"bank-transitions-{checkpoint_id}-{bank}"
                                ),
                                "jackknife_sha256": _digest(
                                    f"jackknife-{checkpoint_id}-{bank}"
                                ),
                                "candidate_transitions": 20
                                + 2 * checkpoint_id
                                + bank_index,
                                "antithetic_max_abs_error": 0.0,
                                "exact_antithetic": True,
                                "shared_rollout_seed_within_pair": True,
                                "lopo_utility_sum": 0.0,
                                "lopo_utility_abs_sum": 10.0,
                                "lopo_gradient_curvature_shared": True,
                                "dsn_da_relative_error": 0.0,
                                "jsn_ja_relative_error": 0.0,
                                "finite_u_statistic": True,
                                "finite_jackknife": True,
                                "finite_eigensystem": True,
                                "q_summaries": q_summaries,
                                "stderr_sha256": analysis.EMPTY_SHA256,
                                "stderr_empty": True,
                            }
                        )
                    )

                partition_rows = analysis.bank_b_partition(manifest, checkpoint_id)
                partition_q: dict[float, list[dict[str, object]]] = {
                    q: [] for q in dims["locality_q"]
                }
                for partition_index, pair_indices in enumerate(partition_rows):
                    q_summaries = []
                    for q in dims["locality_q"]:
                        alpha = q * dims["noise_std"] / 2.0
                        structured_norm = a_by_q[q]["structured"] * (
                            1.0 + 0.01 * partition_index
                        )
                        anisotropic_norm = a_by_q[q]["anisotropic"] * 1.02
                        distance = a_by_q[q]["anisotropic"] * (
                            0.18 + 0.02 * partition_index
                        )
                        summary = _q_summary(
                            manifest,
                            checkpoint_id,
                            q,
                            label=f"partition-{checkpoint_id}-{partition_index}-{q}",
                            gradient_norm=2.1,
                            structured_norm=structured_norm,
                            anisotropic_norm=anisotropic_norm,
                            distance_to_a=distance,
                            alpha=alpha,
                        )
                        q_summaries.append(summary)
                        partition_q[q].append(summary)
                    partitions.append(
                        _stamp(
                            {
                                "partition_id": analysis.partition_id_for(
                                    manifest, checkpoint_id, partition_index
                                ),
                                "checkpoint_id": checkpoint_id,
                                "partition_index": partition_index,
                                "partition_seed": analysis.checkpoint_seed(
                                    manifest, "bank_b_partition", checkpoint_id
                                ),
                                "pair_indices": pair_indices,
                                "pair_count": dims["pairs_per_partition"],
                                "lopo_utility_sum": 0.0,
                                "lopo_utility_abs_sum": 5.0,
                                "lopo_gradient_curvature_shared": True,
                                "finite_u_statistic": True,
                                "finite_jackknife": True,
                                "finite_eigensystem": True,
                                "q_summaries": q_summaries,
                            }
                        )
                    )

                bank_a = next(
                    bank
                    for bank in banks
                    if bank["checkpoint_id"] == checkpoint_id and bank["bank"] == "A"
                )
                bank_b = next(
                    bank
                    for bank in banks
                    if bank["checkpoint_id"] == checkpoint_id and bank["bank"] == "B"
                )
                for q_index, q in enumerate(dims["locality_q"]):
                    a_norm = bank_a["q_summaries"][q_index][
                        "anisotropic_action_norm"
                    ]
                    b_norm = bank_b["q_summaries"][q_index][
                        "anisotropic_action_norm"
                    ]
                    difference = 0.05 * a_norm
                    distances = [
                        row["anisotropic_minus_bank_a_norm"]
                        for row in partition_q[q]
                    ]
                    sq_mean = float(np.mean(np.square(distances)))
                    metrics.append(
                        _stamp(
                            {
                                "metric_id": analysis.metric_id_for(
                                    manifest, checkpoint_id, q
                                ),
                                "checkpoint_id": checkpoint_id,
                                "q": q,
                                "d_material": a_norm
                                / bank_a["q_summaries"][q_index][
                                    "structured_norm"
                                ],
                                "e_high": difference
                                / (0.5 * (a_norm + b_norm)),
                                "e_100": np.sqrt(sq_mean) / a_norm,
                                "material_resolved": True,
                                "high_sample_resolved": True,
                                "operational_resolved": True,
                                "high_sample_action_difference_norm": difference,
                                "partition_action_sq_error_mean": sq_mean,
                            }
                        )
                    )

                for episode_index in range(dims["endpoint_episodes"]):
                    rollout_seed = analysis.endpoint_seed(
                        manifest, checkpoint_id, episode_index
                    )
                    centers.append(
                        _stamp(
                            {
                                "center_endpoint_id": analysis.center_endpoint_id_for(
                                    manifest, checkpoint_id, episode_index
                                ),
                                "checkpoint_id": checkpoint_id,
                                "episode_index": episode_index,
                                "rollout_seed": rollout_seed,
                                "return": float(checkpoint_id + episode_index),
                                "transitions": 5,
                            }
                        )
                    )
                for partition_index in range(dims["bank_b_partition_count"]):
                    partition = next(
                        row
                        for row in partitions
                        if row["checkpoint_id"] == checkpoint_id
                        and row["partition_index"] == partition_index
                    )
                    for q_index, q in enumerate(dims["locality_q"]):
                        for arm_index, arm in enumerate(dims["endpoint_arms"]):
                            for episode_index in range(dims["endpoint_episodes"]):
                                base = float(checkpoint_id + partition_index + episode_index)
                                arm_offset = {
                                    "structured": 1.0 + 0.01 * (seed - 300),
                                    "isotropic": 0.0,
                                    "explicit": 0.4,
                                    "random": 0.2,
                                }[arm]
                                endpoints.append(
                                    _stamp(
                                        {
                                            "endpoint_id": analysis.endpoint_id_for(
                                                manifest,
                                                checkpoint_id,
                                                q,
                                                partition_index,
                                                arm,
                                                episode_index,
                                            ),
                                            "checkpoint_id": checkpoint_id,
                                            "partition_index": partition_index,
                                            "q": q,
                                            "arm": arm,
                                            "episode_index": episode_index,
                                            "rollout_seed": analysis.endpoint_seed(
                                                manifest,
                                                checkpoint_id,
                                                episode_index,
                                            ),
                                            "return": base + arm_offset,
                                            "transitions": 5 + arm_index,
                                            "action_sha256": partition[
                                                "q_summaries"
                                            ][q_index]["action_sha256"][arm],
                                        }
                                    )
                                )

    # Replace the lightweight construction values with independently readable
    # checkpoint, bank, diagnostic, and endpoint artifacts.
    config_relative = "artifacts/training_config.json"
    config_path = os.path.join(artifact_root, config_relative)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "wb") as stream:
        stream.write(b'{"fixture":"lagged-subspace"}')
    config_sha = analysis._sha256_file(config_path)
    for row in training_runs:
        relative = f"artifacts/training_{row['training_id']:03d}.json"
        path = os.path.join(artifact_root, relative)
        with open(path, "wb") as stream:
            stream.write(
                json.dumps(
                    {"training_id": row["training_id"]},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
        row["training_log_path"] = relative
        row["training_log_sha256"] = analysis._sha256_file(path)

    checkpoint_arrays: dict[int, dict[str, np.ndarray]] = {}
    for row in checkpoints:
        checkpoint_id = row["checkpoint_id"]
        generation = row["generation"]
        theta = np.asarray(
            [
                0.1 + 0.001 * checkpoint_id,
                -0.2,
                0.3,
                -0.1,
                0.2,
                -0.3,
            ],
            dtype=np.float64,
        )
        gradients = np.asarray(
            [
                [
                    1.0 + 0.01 * checkpoint_id,
                    0.5,
                    -0.25,
                    0.2,
                    -0.4,
                    0.3,
                ],
                [
                    0.8 + 0.01 * checkpoint_id,
                    0.4,
                    -0.20,
                    0.15,
                    -0.35,
                    0.25,
                ],
            ],
            dtype=np.float64,
        )
        obs_enabled = np.asarray(True, dtype=np.bool_)
        obs_mean = np.asarray([0.0, 0.1], dtype=np.float64)
        obs_var = np.asarray([1.0, 1.1], dtype=np.float64)
        obs_count = np.asarray(100.0, dtype=np.float64)
        checkpoint_payload = {
            "schema_version": np.asarray(2, dtype=np.int64),
            "checkpoint_generation": np.asarray(generation, dtype=np.int64),
            "study_source_sha256": np.asarray(
                hashes["source_sha256"].encode("ascii"), dtype="S64"
            ),
            "training_config_sha256": np.asarray(
                config_sha.encode("ascii"), dtype="S64"
            ),
            "center_params": theta,
            "obs_normalizer_enabled": obs_enabled,
            "obs_mean": obs_mean,
            "obs_var": obs_var,
            "obs_count": obs_count,
            "gradient_generations": np.asarray(
                row["prior_gradient_indices"], dtype=np.int64
            ),
            "proposal_gradients": gradients,
        }
        checkpoint_relative, checkpoint_sha = _write_npz(
            artifact_root,
            f"artifacts/checkpoint_{checkpoint_id:03d}.npz",
            checkpoint_payload,
        )
        reconstructed_basis = analysis.reconstruct_lagged_bases(
            gradients,
            [2, 2, 2],
            lagged_decay=dims["lagged_decay"],
            basis_seed_value=row["basis_seed"],
            random_seed_value=row["random_control_seed"],
        )
        basis = reconstructed_basis["primary_basis"]
        random_basis = reconstructed_basis["random_basis"]
        block_norms = reconstructed_basis["lagged_block_norms"]
        basis_payload = reconstructed_basis
        basis_relative, basis_file_sha = _write_npz(
            artifact_root,
            f"artifacts/basis_{checkpoint_id:03d}.npz",
            basis_payload,
        )
        row.update(
            {
                "parameter_sha256": analysis._array_sha256(theta),
                "observation_normalizer_sha256": analysis._labeled_arrays_sha256(
                    [
                        ("enabled", obs_enabled),
                        ("mean", obs_mean),
                        ("var", obs_var),
                        ("count", obs_count),
                    ]
                ),
                "training_config_sha256": config_sha,
                "prior_gradient_sha256": [
                    analysis._array_sha256(gradient) for gradient in gradients
                ],
                "lagged_block_norms": block_norms.tolist(),
                "lagged_block_exact_zero": [False, False, False],
                "primary_gaussian_fallback_used": [False, False, False],
                "random_control_permuted_fallback_used": [False, False, False],
                "fallback_column_sha256": [None, None, None],
                "basis_sha256": analysis._array_sha256(basis),
                "random_basis_sha256": analysis._array_sha256(random_basis),
                "checkpoint_artifact_path": checkpoint_relative,
                "checkpoint_artifact_sha256": checkpoint_sha,
                "training_config_path": config_relative,
                "basis_artifact_path": basis_relative,
                "basis_artifact_sha256": basis_file_sha,
            }
        )
        checkpoint_arrays[checkpoint_id] = {
            "theta": theta,
            "basis": basis,
            "random_basis": random_basis,
        }

    raw_banks: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    for row in banks:
        checkpoint_id = row["checkpoint_id"]
        bank = row["bank"]
        pair_noise = np.stack(
            [
                np.random.Generator(np.random.PCG64(int(seed))).standard_normal(6)
                for seed in row["perturbation_seeds"]
            ]
        )
        signed_noise = np.stack((pair_noise, -pair_noise), axis=1)
        direction = np.asarray([1.0, 0.4, -0.25, 0.3, -0.2, 0.15])
        score = pair_noise @ direction
        offsets = 0.013 * np.arange(dims["pairs_per_bank"])
        paired_returns = np.column_stack((score + offsets, -score + offsets))
        paired_transitions = np.full(
            (dims["pairs_per_bank"], 2),
            5 + row["bank_id"],
            dtype=np.int64,
        )
        raw_payload = {
            "signed_noise": signed_noise,
            "paired_returns": paired_returns,
            "paired_transitions": paired_transitions,
            "perturbation_seeds": np.asarray(
                row["perturbation_seeds"], dtype=np.uint64
            ),
            "rollout_seeds_plus": np.asarray(
                row["rollout_seeds_plus"], dtype=np.uint64
            ),
            "rollout_seeds_minus": np.asarray(
                row["rollout_seeds_minus"], dtype=np.uint64
            ),
        }
        raw_relative, raw_sha = _write_npz(
            artifact_root,
            f"artifacts/bank_{checkpoint_id:03d}_{bank}.npz",
            {
                name: value
                for name, value in raw_payload.items()
                if name != "signed_noise"
            },
        )
        row.update(
            {
                "raw_bank_path": raw_relative,
                "raw_bank_sha256": raw_sha,
                "perturbations_sha256": analysis._array_sha256(signed_noise),
                "returns_sha256": analysis._array_sha256(paired_returns),
                "transitions_sha256": analysis._array_sha256(
                    paired_transitions
                ),
                "candidate_transitions": int(np.sum(paired_transitions)),
            }
        )
        raw_banks[(checkpoint_id, bank)] = raw_payload

    reference_results_by_checkpoint: dict[int, list[object]] = {}
    bank_results_by_checkpoint: dict[tuple[int, str], list[object]] = {}
    partition_results: dict[tuple[int, int], list[object]] = {}
    for checkpoint_id in range(len(checkpoints)):
        state = checkpoint_arrays[checkpoint_id]
        bank_a_row = next(
            row
            for row in banks
            if row["checkpoint_id"] == checkpoint_id and row["bank"] == "A"
        )
        a_raw = raw_banks[(checkpoint_id, "A")]
        probe = analyze_lagged_subspace_population(
            state["theta"],
            a_raw["signed_noise"],
            a_raw["paired_returns"],
            dims["noise_std"],
            state["basis"],
            state["random_basis"],
            1e-4,
            basis_provenance=BasisProvenance.strictly_lagged("probe", "probe-r"),
        )
        gradient_norm = float(np.linalg.norm(probe.estimate.gradient))
        alphas = [q * dims["noise_std"] / gradient_norm for q in dims["locality_q"]]
        reference_results: list[object] | None = None
        endpoint_reference: dict[str, np.ndarray] | None = None
        for bank in dims["banks"]:
            row = next(
                item
                for item in banks
                if item["checkpoint_id"] == checkpoint_id and item["bank"] == bank
            )
            raw = raw_banks[(checkpoint_id, bank)]
            payload, summaries, results = _diagnostic_payload(
                manifest,
                state["theta"],
                raw["signed_noise"],
                raw["paired_returns"],
                state["basis"],
                state["random_basis"],
                alphas,
                reference_results,
                endpoint_reference,
            )
            if bank == "A":
                reference_results = results
                reference_results_by_checkpoint[checkpoint_id] = results
                endpoint_reference = {
                    "signed_noise": raw["signed_noise"],
                    "utilities": results[0].estimate.utilities,
                    "gradient": results[0].estimate.gradient,
                    "curvature": results[0].estimate.curvature,
                }
            bank_results_by_checkpoint[(checkpoint_id, bank)] = results
            diag_relative, diag_sha = _write_npz(
                artifact_root,
                f"artifacts/diagnostics_{checkpoint_id:03d}_{bank}.npz",
                payload,
            )
            row.update(
                {
                    "q_summaries": summaries,
                    "diagnostics_path": diag_relative,
                    "diagnostics_sha256": diag_sha,
                    "jackknife_sha256": _fixture_jackknife_sha256(
                        payload, results[0]
                    ),
                    "lopo_utility_sum": float(np.sum(payload["utilities"])),
                    "lopo_utility_abs_sum": float(
                        np.sum(np.abs(payload["utilities"]))
                    ),
                    "dsn_da_relative_error": float(
                        payload["self_normalized_gradient_relative_error"]
                    ),
                    "jsn_ja_relative_error": float(
                        payload["self_normalized_jacobian_relative_error"]
                    ),
                }
            )
        assert reference_results is not None
        b_raw = raw_banks[(checkpoint_id, "B")]
        for partition_index, pair_indices in enumerate(
            analysis.bank_b_partition(manifest, checkpoint_id)
        ):
            row = next(
                item
                for item in partitions
                if item["checkpoint_id"] == checkpoint_id
                and item["partition_index"] == partition_index
            )
            selected = np.asarray(pair_indices, dtype=np.int64)
            payload, summaries, results = _diagnostic_payload(
                manifest,
                state["theta"],
                b_raw["signed_noise"][selected],
                b_raw["paired_returns"][selected],
                state["basis"],
                state["random_basis"],
                alphas,
                reference_results,
                endpoint_reference,
            )
            diag_relative, diag_sha = _write_npz(
                artifact_root,
                f"artifacts/partition_{checkpoint_id:03d}_{partition_index:02d}.npz",
                payload,
            )
            row.update(
                {
                    "q_summaries": summaries,
                    "diagnostics_path": diag_relative,
                    "diagnostics_sha256": diag_sha,
                    "lopo_utility_sum": float(np.sum(payload["utilities"])),
                    "lopo_utility_abs_sum": float(
                        np.sum(np.abs(payload["utilities"]))
                    ),
                }
            )
            partition_results[(checkpoint_id, partition_index)] = results

        bank_a_results = reference_results_by_checkpoint[checkpoint_id]
        bank_b_results = bank_results_by_checkpoint[(checkpoint_id, "B")]
        for q_index, q in enumerate(dims["locality_q"]):
            a_action = (
                bank_a_results[q_index].steps.structured
                - bank_a_results[q_index].steps.isotropic
            )
            b_action = np.asarray(
                bank_b_results[q_index].steps.structured
                - bank_b_results[q_index].steps.isotropic
            )
            distances = [
                float(
                    next(
                        row
                        for row in partitions
                        if row["checkpoint_id"] == checkpoint_id
                        and row["partition_index"] == partition_index
                    )["q_summaries"][q_index][
                        "anisotropic_minus_bank_a_norm"
                    ]
                )
                for partition_index in range(dims["bank_b_partition_count"])
            ]
            sq_mean = float(np.mean(np.square(distances)))
            metric = next(
                item
                for item in metrics
                if item["checkpoint_id"] == checkpoint_id and item["q"] == q
            )
            structured_norm = float(
                np.linalg.norm(bank_a_results[q_index].steps.structured)
            )
            a_norm = float(np.linalg.norm(a_action))
            b_norm = float(np.linalg.norm(b_action))
            high_difference = float(np.linalg.norm(a_action - b_action))
            epsilon = manifest["analysis"]["machine_epsilon"]
            metric.update(
                {
                    "d_material": a_norm / max(structured_norm, epsilon),
                    "e_high": high_difference
                    / max(0.5 * (a_norm + b_norm), epsilon),
                    "e_100": np.sqrt(sq_mean) / max(a_norm, epsilon),
                    "material_resolved": structured_norm > epsilon,
                    "high_sample_resolved": 0.5 * (a_norm + b_norm) > epsilon,
                    "operational_resolved": a_norm > epsilon,
                    "high_sample_action_difference_norm": high_difference,
                    "partition_action_sq_error_mean": sq_mean,
                }
            )

        center_returns = np.zeros(dims["endpoint_episodes"], dtype=np.float64)
        center_transitions = np.zeros(dims["endpoint_episodes"], dtype=np.int64)
        endpoint_shape = (
            len(dims["locality_q"]),
            dims["bank_b_partition_count"],
            len(dims["endpoint_arms"]),
            dims["endpoint_episodes"],
        )
        endpoint_returns = np.zeros(endpoint_shape, dtype=np.float64)
        endpoint_transitions = np.zeros(endpoint_shape, dtype=np.int64)
        for row in centers:
            if row["checkpoint_id"] == checkpoint_id:
                center_returns[row["episode_index"]] = row["return"]
                center_transitions[row["episode_index"]] = row["transitions"]
        for row in endpoints:
            if row["checkpoint_id"] != checkpoint_id:
                continue
            q_index = dims["locality_q"].index(row["q"])
            arm_index = dims["endpoint_arms"].index(row["arm"])
            coords = (
                q_index,
                row["partition_index"],
                arm_index,
                row["episode_index"],
            )
            endpoint_returns[coords] = row["return"]
            endpoint_transitions[coords] = row["transitions"]
            partition = next(
                item
                for item in partitions
                if item["checkpoint_id"] == checkpoint_id
                and item["partition_index"] == row["partition_index"]
            )
            row["action_sha256"] = partition["q_summaries"][q_index][
                "action_sha256"
            ][row["arm"]]
        endpoint_relative, endpoint_sha = _write_npz(
            artifact_root,
            f"artifacts/endpoints_{checkpoint_id:03d}.npz",
            {
                "center_returns": center_returns,
                "center_transitions": center_transitions,
                "endpoint_returns": endpoint_returns,
                "endpoint_transitions": endpoint_transitions,
                "rollout_seeds": np.asarray(
                    [
                        analysis.endpoint_seed(manifest, checkpoint_id, episode)
                        for episode in range(dims["endpoint_episodes"])
                    ],
                    dtype=np.uint64,
                ),
            },
        )
        for row in centers + endpoints:
            if row["checkpoint_id"] == checkpoint_id:
                row["rollout_artifact_path"] = endpoint_relative
                row["rollout_artifact_sha256"] = endpoint_sha

    for collection in (
        training_runs,
        checkpoints,
        banks,
        partitions,
        metrics,
        centers,
        endpoints,
    ):
        for row in collection:
            _stamp(row)

    provenance = _stamp(
        {
            **hashes,
            "source_snapshot_path": "/immutable/fixture",
            "stderr_empty": True,
            "documented_infrastructure_failures": [],
        }
    )
    budget = {
        key: value
        for key, value in manifest["budget"].items()
        if key != "environment_transitions_are_separate"
    }
    budget.update(
        {
            "checkpoint_training_transitions": sum(
                row["training_transitions"] for row in training_runs
            ),
            "normalization_calibration_transitions": sum(
                row["calibration_transitions"] for row in training_runs
            ),
            "bank_transitions": sum(row["candidate_transitions"] for row in banks),
            "endpoint_arm_transitions": sum(
                row["transitions"] for row in endpoints
            ),
            "checkpoint_center_transitions": sum(
                row["transitions"] for row in centers
            ),
        }
    )
    budget["total_environment_transitions"] = sum(
        budget[key]
        for key in (
            "checkpoint_training_transitions",
            "normalization_calibration_transitions",
            "bank_transitions",
            "endpoint_arm_transitions",
            "checkpoint_center_transitions",
        )
    )
    artifact = {
        "schema_version": 1,
        "study": analysis.STUDY,
        "designation": manifest["designation"],
        "manifest_sha256": hashes["manifest_sha256"],
        "provenance": provenance,
        "analysis_declaration": analysis._expected_analysis_declaration(manifest),
        "training_runs": training_runs,
        "checkpoints": checkpoints,
        "banks": banks,
        "partitions": partitions,
        "checkpoint_metrics": metrics,
        "center_endpoints": centers,
        "endpoints": endpoints,
        "budget": budget,
    }
    return manifest, artifact, hashes


class LaggedSubspaceAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifact_directory = tempfile.TemporaryDirectory()
        cls.manifest, cls.artifact, cls.hashes = _fixture(
            cls.artifact_directory.name
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.artifact_directory.cleanup()

    def _validate(self, artifact: dict[str, object]) -> dict[str, object]:
        return analysis.validate_artifact(
            artifact,
            self.manifest,
            expected_hashes=self.hashes,
            artifact_root=self.artifact_directory.name,
            require_preregistered_manifest=False,
        )

    def test_compact_mappings_are_bijective_and_fixture_validates(self) -> None:
        validated = self._validate(copy.deepcopy(self.artifact))
        expected_checkpoints = (
            len(self.manifest["tasks"])
            * len(self.manifest["training_seeds"])
            * len(self.manifest["checkpoint_generations"])
        )
        self.assertEqual(len(validated["checkpoint_by_id"]), expected_checkpoints)
        for checkpoint_id in range(expected_checkpoints):
            task_index, seed, generation = analysis.checkpoint_coordinates(
                self.manifest, checkpoint_id
            )
            self.assertEqual(
                analysis.checkpoint_id_for(
                    self.manifest, task_index, seed, generation
                ),
                checkpoint_id,
            )
            flattened = sum(
                analysis.bank_b_partition(self.manifest, checkpoint_id), []
            )
            self.assertEqual(
                sorted(flattened),
                list(range(self.manifest["dimensions"]["pairs_per_bank"])),
            )

    def test_missing_and_duplicate_endpoint_records_fail_closed(self) -> None:
        missing = copy.deepcopy(self.artifact)
        missing["endpoints"].pop()
        with self.assertRaisesRegex(
            analysis.SubspaceValidationError, "validation failed"
        ) as caught:
            self._validate(missing)
        self.assertTrue(any("expected" in issue for issue in caught.exception.issues))

        duplicate = copy.deepcopy(self.artifact)
        duplicate["endpoints"][1] = copy.deepcopy(duplicate["endpoints"][0])
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(duplicate)
        self.assertTrue(any("duplicate endpoint" in issue for issue in caught.exception.issues))

    def test_hash_corruption_and_independently_resigned_provenance_fail(self) -> None:
        stale = copy.deepcopy(self.artifact)
        stale["endpoints"][0]["return"] += 1.0
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(stale)
        self.assertTrue(any("record SHA-256 mismatch" in issue for issue in caught.exception.issues))

        resigned = copy.deepcopy(self.artifact)
        resigned["provenance"]["source_sha256"] = _digest("wrong-source")
        _stamp(resigned["provenance"])
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(resigned)
        self.assertTrue(any("source_sha256" in issue for issue in caught.exception.issues))

    def test_prior_gradient_and_bank_disjointness_corruption_fail(self) -> None:
        prior = copy.deepcopy(self.artifact)
        prior["checkpoints"][0]["prior_gradient_indices"] = [0, 2]
        _stamp(prior["checkpoints"][0])
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(prior)
        self.assertTrue(any("prior_gradient_indices" in issue for issue in caught.exception.issues))

        overlap = copy.deepcopy(self.artifact)
        first_a = next(row for row in overlap["banks"] if row["bank"] == "A")
        first_b = next(
            row
            for row in overlap["banks"]
            if row["bank"] == "B" and row["checkpoint_id"] == first_a["checkpoint_id"]
        )
        first_b["perturbation_seeds"] = list(first_a["perturbation_seeds"])
        _stamp(first_b)
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(overlap)
        self.assertTrue(
            any("perturbation_seeds" in issue for issue in caught.exception.issues)
        )

    def test_raw_basis_and_perturbations_are_reconstructed_from_locked_seeds(self) -> None:
        basis_case = copy.deepcopy(self.artifact)
        checkpoint = basis_case["checkpoints"][0]
        with np.load(
            os.path.join(
                self.artifact_directory.name, checkpoint["basis_artifact_path"]
            ),
            allow_pickle=False,
        ) as source:
            basis_payload = {name: np.asarray(source[name]) for name in source.files}
        basis_payload["primary_basis"] = basis_payload["primary_basis"].copy()
        basis_payload["primary_basis"][:, 0] *= -1.0
        relative, digest = _write_npz(
            self.artifact_directory.name,
            "corrupt/re_signed_basis.npz",
            basis_payload,
        )
        checkpoint["basis_artifact_path"] = relative
        checkpoint["basis_artifact_sha256"] = digest
        checkpoint["basis_sha256"] = analysis._array_sha256(
            basis_payload["primary_basis"]
        )
        _stamp(checkpoint)
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(basis_case)
        self.assertTrue(
            any("independent lagged reconstruction" in issue for issue in caught.exception.issues)
        )

        noise_case = copy.deepcopy(self.artifact)
        bank = noise_case["banks"][0]
        with np.load(
            os.path.join(self.artifact_directory.name, bank["raw_bank_path"]),
            allow_pickle=False,
        ) as source:
            raw_payload = {name: np.asarray(source[name]) for name in source.files}
        raw_payload["perturbation_seeds"] = raw_payload["perturbation_seeds"].copy()
        raw_payload["perturbation_seeds"][0] ^= np.uint64(1)
        relative, digest = _write_npz(
            self.artifact_directory.name,
            "corrupt/re_signed_seed_map.npz",
            raw_payload,
        )
        bank["raw_bank_path"] = relative
        bank["raw_bank_sha256"] = digest
        _stamp(bank)
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(noise_case)
        self.assertTrue(
            any("seed arrays disagree" in issue for issue in caught.exception.issues)
        )

        hash_case = copy.deepcopy(self.artifact)
        hash_bank = hash_case["banks"][0]
        hash_bank["perturbations_sha256"] = _digest("wrong-regenerated-noise")
        _stamp(hash_bank)
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(hash_case)
        self.assertTrue(
            any("perturbations_sha256" in issue for issue in caught.exception.issues)
        )

    def test_basis_reconstruction_matches_core_bitwise_for_all_zero_masks(self) -> None:
        blocks = [slice(0, 2), slice(2, 5), slice(5, 6)]
        block_sizes = [2, 3, 1]
        for mask in range(8):
            gradients = np.arange(60, dtype=np.float64).reshape(10, 6) / 17.0
            for block_index, block in enumerate(blocks):
                if mask & (1 << block_index):
                    gradients[:, block] = 0.0
            core_result = build_lagged_bases(
                gradients,
                blocks,
                primary_fallback_seed=123456789,
                random_permutation_seed=987654321,
                primary_reference="test-primary",
                random_reference="test-random",
            )
            reconstructed = analysis.reconstruct_lagged_bases(
                gradients,
                block_sizes,
                lagged_decay=0.9,
                basis_seed_value=123456789,
                random_seed_value=987654321,
            )
            with self.subTest(mask=mask):
                self.assertTrue(
                    np.array_equal(reconstructed["primary_basis"], core_result.primary)
                )
                self.assertTrue(
                    np.array_equal(
                        reconstructed["random_basis"], core_result.random_control
                    )
                )
                self.assertEqual(
                    reconstructed["lagged_block_exact_zero"].tolist(),
                    list(core_result.primary_fallback_blocks),
                )
                self.assertEqual(
                    reconstructed[
                        "random_control_permuted_fallback_used"
                    ].tolist(),
                    list(core_result.random_uses_primary_fallback_blocks),
                )

    def test_artifact_path_traversal_is_rejected_even_with_valid_digest(self) -> None:
        case = copy.deepcopy(self.artifact)
        case["training_runs"][0]["training_log_path"] = "../outside.json"
        _stamp(case["training_runs"][0])
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(case)
        self.assertTrue(any("normalized relative path" in issue for issue in caught.exception.issues))

    def test_norm_budget_and_inference_contract_corruption_fail(self) -> None:
        norm = copy.deepcopy(self.artifact)
        norm["partitions"][0]["q_summaries"][0]["isotropic_norm"] *= 1.1
        _stamp(norm["partitions"][0])
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(norm)
        self.assertTrue(any("norm match" in issue for issue in caught.exception.issues))

        budget = copy.deepcopy(self.artifact)
        budget["budget"]["total_policy_rollouts"] += 1
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(budget)
        self.assertTrue(any("total_policy_rollouts" in issue for issue in caught.exception.issues))

        inference = copy.deepcopy(self.artifact)
        inference["analysis_declaration"]["primary_multiplicity"] = "none"
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(inference)
        self.assertTrue(any("multiplicity" in issue for issue in caught.exception.issues))

        leakage = copy.deepcopy(self.artifact)
        leakage["endpoints"][0]["p_value"] = 0.01
        with self.assertRaises(analysis.SubspaceValidationError) as caught:
            self._validate(leakage)
        self.assertTrue(any("inferential result" in issue for issue in caught.exception.issues))

    def test_analysis_uses_training_seed_clusters_and_holm_family(self) -> None:
        validated = self._validate(copy.deepcopy(self.artifact))
        result = analysis.analyze_validated(validated, self.manifest)
        self.assertEqual(result["top_level_unit"], "training_seed")
        self.assertEqual(len(result["task_results"]), 3)
        self.assertTrue(
            all(len(task["seed_statistics"]["L"]) == 2 for task in result["task_results"])
        )
        raw = [task["raw_one_sided_sign_p"] for task in result["task_results"]]
        adjusted = [
            task["holm_adjusted_one_sided_sign_p"]
            for task in result["task_results"]
        ]
        self.assertTrue(all(right >= left for left, right in zip(raw, adjusted)))

    def test_unresolved_alpha_fails_every_affected_task_gate_condition(self) -> None:
        validated = self._validate(copy.deepcopy(self.artifact))
        for (checkpoint_id, _), record in validated["partition_by_key"].items():
            task_index, _, _ = analysis.checkpoint_coordinates(
                self.manifest, checkpoint_id
            )
            if task_index == 0:
                for summary in record["q_summaries"]:
                    summary["alpha_resolved"] = False
                    summary["alpha_unresolved_reason"] = (
                        "bank_a_gradient_exact_zero"
                    )
        result = analysis.analyze_validated(validated, self.manifest)
        first = result["task_results"][0]
        self.assertFalse(first["task_pass"])
        self.assertTrue(
            all(value is False for value in first["gate_conditions"].values())
        )

    def test_distribution_free_order_bounds_handle_indices_ties_and_nonfinite(self) -> None:
        with open(analysis.DEFAULT_MANIFEST_PATH, encoding="utf-8") as stream:
            production = json.load(stream)
        statistics = {}
        for task in production["tasks"]:
            task_index = task["task_index"]
            statistics[(task_index, "L")] = np.arange(20, dtype=np.float64)
            statistics[(task_index, "D")] = np.arange(20, dtype=np.float64)
            statistics[(task_index, "H")] = np.ones(20, dtype=np.float64) * 7.0
            statistics[(task_index, "E")] = np.asarray(
                [0.0] * 10 + [1.0] * 10, dtype=np.float64
            )
        bounds = analysis._simultaneous_bounds(statistics, production)
        self.assertEqual(bounds[(0, "D")]["one_sided_bound"], 3.0)
        self.assertEqual(bounds[(0, "D")]["order_index_zero_based"], 3)
        self.assertEqual(bounds[(0, "L")]["one_sided_bound"], 16.0)
        self.assertEqual(bounds[(0, "L")]["order_index_zero_based"], 16)
        self.assertEqual(bounds[(0, "H")]["one_sided_bound"], 7.0)
        self.assertTrue(bounds[(0, "H")]["resolved"])
        self.assertEqual(bounds[(0, "E")]["one_sided_bound"], 1.0)

        statistics[(0, "L")] = np.asarray([0.0] * 19 + [np.nan])
        unresolved = analysis._simultaneous_bounds(statistics, production)
        self.assertFalse(unresolved[(0, "L")]["resolved"])
        self.assertIsNone(unresolved[(0, "L")]["one_sided_bound"])

    def test_exact_sign_gate_distinguishes_sixteen_from_fifteen_wins(self) -> None:
        with open(analysis.DEFAULT_MANIFEST_PATH, encoding="utf-8") as stream:
            production = json.load(stream)
        alpha = production["analysis"]["endpoint_family_alpha"]
        p16 = analysis._binomial_upper_tail(16, 20)
        p15 = analysis._binomial_upper_tail(15, 20)
        self.assertAlmostEqual(p16, 0.005908966064453125)
        adjusted16 = analysis._holm_adjust([p16, p16, p16])
        adjusted15 = analysis._holm_adjust([p15, p15, p15])
        self.assertTrue(all(value < alpha for value in adjusted16))
        self.assertTrue(all(value >= alpha for value in adjusted15))


if __name__ == "__main__":
    unittest.main()

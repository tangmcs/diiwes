# Documentation map

This directory contains the durable scientific record: protocols, theory,
audits, compact benchmark tables, and validated experiment summaries. Raw run
directories and scheduler logs are local-only under `results/` and
`job_outputs/`; they are not part of the repository documentation.

- [`README.md`](README.md) is this navigation index.

## Direction, readiness, and claim audits

- [`conference_readiness_and_paper_plan.md`](conference_readiness_and_paper_plan.md) sets the paper direction and evidence threshold.
- [`novelty_and_claims_audit.md`](novelty_and_claims_audit.md) separates supported claims from open novelty work.
- [`baseline_readiness_audit.md`](baseline_readiness_audit.md) inventories strong ES comparators and their evidence status.
- [`issues_document_review.md`](issues_document_review.md) reviews the earlier algorithm-issues document.

## Theory and method contracts

- [`theory_rank_curvature_surrogate.md`](theory_rank_curvature_surrogate.md) defines the implemented rank-curvature estimand.
- [`theory_resonance_sample_complexity.md`](theory_resonance_sample_complexity.md) gives concentration and resonance results.
- [`lopo_u_stat_curvature.md`](lopo_u_stat_curvature.md) documents the leave-own-pair-out U-statistic construction.
- [`snes_baseline.md`](snes_baseline.md) specifies the separable NES baseline.

## Hopper experiment record

- [`experiment_diagnosis.md`](experiment_diagnosis.md) diagnoses the trust-free, no-replay experiments.
- [`hessian_fix_ablation.md`](hessian_fix_ablation.md) separates estimator instability from solve instability.
- [`hopper_implicit_job_49648326.md`](hopper_implicit_job_49648326.md) records the original no-replay implicit sweep.
- [`hopper_hessian_job_49678516.md`](hopper_hessian_job_49678516.md) records Standard ES versus linearized Hessian ES.
- [`hopper_hessian_fix_job_49678999.md`](hopper_hessian_fix_job_49678999.md) records the curvature-stabilization sweep.
- [`hopper_hessian_confirmation_preregistration.md`](hopper_hessian_confirmation_preregistration.md) locks the untouched-seed confirmation design.
- [`hopper_hessian_confirmation_job_49681345.md`](hopper_hessian_confirmation_job_49681345.md) records that confirmation result.
- [`hopper_fresh_optimizer_development_protocol.md`](hopper_fresh_optimizer_development_protocol.md) defines the fresh-only optimizer screen.
- [`hopper_fresh_optimizer_development_job_49685417.md`](hopper_fresh_optimizer_development_job_49685417.md) records the optimizer-screen job.

## Controlled estimator benchmarks

- [`implicit_quadratic_optimization_benchmark.md`](implicit_quadratic_optimization_benchmark.md) studies implicit updates on controlled quadratics.
- [`curvature_estimator_diagnostic.csv`](curvature_estimator_diagnostic.csv) contains the high-dimensional diagonal-estimator diagnostic.
- [`structured_curvature_diagnostic.csv`](structured_curvature_diagnostic.csv) compares diagonal and block curvature stability.
- [`curvature_reliability_benchmark.csv`](curvature_reliability_benchmark.csv) and [`curvature_reliability_benchmark.json`](curvature_reliability_benchmark.json) are the schema-v1 controlled reliability table and metadata.
- [`rank_surrogate_reliability_benchmark.md`](rank_surrogate_reliability_benchmark.md) explains the locked rank-surrogate benchmark; [`rank_surrogate_reliability_benchmark.csv`](rank_surrogate_reliability_benchmark.csv) and [`rank_surrogate_reliability_benchmark.json`](rank_surrogate_reliability_benchmark.json) are its table and metadata.

## Lagged-subspace study

- [`lagged_subspace_frozen_checkpoint_protocol.md`](lagged_subspace_frozen_checkpoint_protocol.md) locks the mechanism-diagnostic design.
- [`lagged_subspace_postlaunch_audit.md`](lagged_subspace_postlaunch_audit.md) records the post-launch integrity audit.
- [`reproducibility_and_artifact_plan.md`](reproducibility_and_artifact_plan.md) specifies artifact acceptance and remaining work.

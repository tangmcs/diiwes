# Script map

The canonical training entry point is [`experiments/train.py`](../experiments/train.py).
This directory holds cluster launchers, strict result validators, controlled
diagnostics, and reporting utilities. Generated run data and scheduler output
remain local-only under `results/` and `job_outputs/`.

- [`README.md`](README.md) is this navigation index.

## Source-locked cluster launchers

These launchers enforce protocol fields and, where required, source or
manifest hashes. Some historical launcher names and cluster-specific paths are
retained for compatibility with archived artifacts and source-lock checks; use
their documented environment overrides rather than casually rewriting a locked
launcher.

- [`submit_hopper_no_replay_sweep.sh`](submit_hopper_no_replay_sweep.sh) launches the trust-free Standard ES versus linearized-Hessian sweep.
- [`submit_hopper_hessian_fix_sweep.sh`](submit_hopper_hessian_fix_sweep.sh) launches the curvature-stabilization ablation.
- [`submit_hopper_hessian_confirmation.sh`](submit_hopper_hessian_confirmation.sh) launches the locked untouched-seed confirmation.
- [`submit_hopper_fresh_optimizer_development.sh`](submit_hopper_fresh_optimizer_development.sh) launches the fresh-only optimizer screen.
- [`submit_lagged_subspace_checkpoint_generation.sh`](submit_lagged_subspace_checkpoint_generation.sh) launches locked checkpoint training.
- [`submit_lagged_subspace_diagnostic.sh`](submit_lagged_subspace_diagnostic.sh) launches the frozen-checkpoint diagnostic array.

## Hopper validation and analysis

- [`summarize_hopper_implicit_sweep.py`](summarize_hopper_implicit_sweep.py) validates and compares the main Hessian and stabilization sweeps.
- [`summarize_no_replay_sweep.py`](summarize_no_replay_sweep.py) validates the earlier no-replay diagnostic sweep.
- [`summarize_hopper_hessian_confirmation.py`](summarize_hopper_hessian_confirmation.py) performs the preregistered paired confirmation analysis.
- [`summarize_hopper_fresh_optimizer_development.py`](summarize_hopper_fresh_optimizer_development.py) validates the optimizer-development screen.

## Lagged-subspace artifact pipeline

- [`validate_lagged_subspace_checkpoint_stage.py`](validate_lagged_subspace_checkpoint_stage.py) gates the completed checkpoint stage.
- [`assemble_lagged_subspace_frozen_checkpoint.py`](assemble_lagged_subspace_frozen_checkpoint.py) assembles the provenance-preserving audit index.
- [`analyze_lagged_subspace_frozen_checkpoint.py`](analyze_lagged_subspace_frozen_checkpoint.py) validates and analyzes the locked diagnostic.
- [`collect_lagged_subspace_compute_disclosure.py`](collect_lagged_subspace_compute_disclosure.py) records compute and storage disclosure.
- [`render_lagged_subspace_paper_outputs.py`](render_lagged_subspace_paper_outputs.py) renders deterministic paper outputs from validated analysis.

## Controlled diagnostics

- [`diagnose_curvature_estimator.py`](diagnose_curvature_estimator.py) tests production Stein-curvature behavior on known quadratics.
- [`diagnose_structured_curvature.py`](diagnose_structured_curvature.py) compares diagonal and layer-block stability.

## General reporting

- [`export_plot_table.py`](export_plot_table.py) exports histories to a long-format table.
- [`plot_mujoco_lr_robustness.py`](plot_mujoco_lr_robustness.py) reproduces the historical trust-confounded learning-rate plot.

The retired `summarize.py` and `analyze_mujoco_results.py` helpers were moved to
the ignored local archive `analysis/legacy_tools/`. They target the historical
trust-confounded result format and are not dependencies of the current study.

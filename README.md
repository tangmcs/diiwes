# DIIWES Paper Code

This repository contains the reference implementation for the DIIWES paper experiments. It includes optimizer implementations, experiment configurations, and training entry points intended to support reproducibility.

## Repository Layout

- `core/`: optimizer and policy code.
  - `diiwes.py`: final `DIIWES` optimizer with importance weighting, raw-fitness Stein curvature, leave-one-out curvature baseline, optional EMA bias correction, and trust-radius clipping.
  - `standard_es.py`: OpenAI-style ES baseline with rank fitness.
  - `policies.py`: NumPy MLP policies and layer-slice helpers.
- `utilities/`: shared utilities such as observation normalization.
- `experiments/`: runnable training entry points.
- `configs/`: experiment configuration files, split into `configs/mujuco/` and `configs/atari/`.
- `plots/`: reserved for paper figure-generation scripts.

Generated outputs live in `results/` and `job_outputs/`; both are ignored by git.

## Experimental Conditions

The trainer exposes the optimizer comparisons used by the experiments:

- `standard_es`
- `no_curvature`
- `diag_curvature`

The clean comparisons are:

- `diag_curvature` vs `no_curvature`: evaluates whether diagonal curvature improves the semi-implicit replay/trust optimizer.
- `standard_es`: plain ES baseline.

## Paper Result Artifacts

The current balanced MuJoCo learning-rate sweep is exported as a long-format
table and then summarized into LaTeX fragments and figures for the paper.

```bash
python scripts/export_plot_table.py \
  results/mujoco_lr_sweep_46638567 \
  --output plots/mujoco_lr_sweep_46638567_plot_table.csv

python scripts/analyze_mujoco_results.py \
  --input plots/mujoco_lr_sweep_46638567_plot_table.csv \
  --out-dir plots \
  --env-step-lr 0.02
```

The analysis command writes:

- `plots/mujoco_experiments_section_draft.tex`
- `plots/mujoco_best_return_table.tex`
- `plots/mujoco_robustness_table.tex`
- `plots/mujoco_diagnostics_summary.tex`
- `plots/mujoco_env_step_learning_curves_lr0p02.{png,pdf}`

The no-curvature ablation needed to isolate the diagonal Stein-curvature term
can be launched as a separate Slurm array:

```bash
sbatch slurm/mujoco_no_curvature_lr_sweep.sh
```

Direct shell execution of that Slurm script defaults to a dry run; set
`PAPER_DRY_RUN=0` only when intentionally running one local task.

After the array finishes, export its histories with the existing collector:

```bash
PAPER_OUTPUT_ROOT=results/mujoco_no_curvature_lr_sweep_<jobid> \
PAPER_PLOT_OUTPUT=plots/mujoco_no_curvature_lr_sweep_<jobid>_plot_table.csv \
sbatch slurm/collect_mujoco_lr_sweep.sh
```

Then regenerate the paper artifacts by merging the main sweep and the
no-curvature ablation sweep:

```bash
python scripts/analyze_mujoco_results.py \
  --input plots/mujoco_lr_sweep_46638567_plot_table.csv \
  --input plots/mujoco_no_curvature_lr_sweep_<jobid>_plot_table.csv \
  --out-dir plots \
  --env-step-lr 0.02
```

When the merged table contains `DIIWES-no-H`, the analysis script also writes
`plots/mujoco_hessian_ablation_lr_robustness.{png,pdf}` and includes the
Hessian-ablation figure in `plots/mujoco_experiments_section_draft.tex`.

## Local Execution

All commands below assume they are run from this repository root.

```bash
python experiments/train.py \
  --config configs/mujuco/hopper.yaml \
  --condition diag_curvature \
  --learning-rate 0.02 \
  --seed 0 \
  --workers 8 \
  --output results/debug_diag_curvature_seed0
```

Add `--verbose` to print per-iteration optimizer diagnostics such as step norm, pre-trust norm, curvature, fresh/reused counts, and iteration time. Without `--verbose`, training prints only the run header and compact progress lines.

## Reproducing A Grid

The same training entry point can be used to run multiple seeds and conditions locally. For example:

```bash
for condition in standard_es no_curvature diag_curvature; do
  for seed in 0 1 2; do
    python experiments/train.py \
      --config configs/mujuco/hopper.yaml \
      --condition "$condition" \
      --learning-rate 0.02 \
      --seed "$seed" \
      --workers 8 \
      --output "results/hopper_lr002/hopper_${condition}_seed${seed}"
  done
done
```

Atari RAM configs live under `configs/atari/` and use `obs_scale: 255.0` with a lower ES learning rate:

```bash
python experiments/train.py \
  --config configs/atari/pong.yaml \
  --condition diag_curvature \
  --learning-rate 0.02 \
  --seed 0 \
  --workers 8 \
  --output results/pong_diag_curvature_seed0
```

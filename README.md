# DIIWES Paper Code

This repository contains the reference implementation for the DIIWES paper experiments. It includes optimizer implementations, experiment configurations, and training entry points intended to support reproducibility.

## Repository Layout

- `core/`: optimizer and policy code.
  - `diiwes.py`: final `DIIWES` optimizer with importance weighting, raw-return Stein curvature by default, optional standardized-curvature checks, leave-one-out curvature baseline, optional EMA bias correction, and trust-radius clipping.
  - `standard_es.py`: OpenAI-style ES baseline with rank fitness and optional trust-radius clipping.
  - `policies.py`: NumPy MLP policies and layer-slice helpers.
- `utilities/`: shared utilities such as observation normalization.
- `experiments/`: runnable training entry points.
  - `nonlinear_cartpole/`: policy-gradient warm-start benchmark on nonlinear
    CartPole dynamics, followed by matched Standard ES and DIIWES fine-tuning.
  - `curvature_sample_size/`: controlled linear and nonlinear validation of
    diagonal curvature-estimation error versus antithetic sample size.
- `configs/`: experiment configuration files, split into `configs/mujoco/` and `configs/atari/`.
- `scripts/`: standalone tooling, separated into `analysis/`, `plotting/`, and
  `slurm/`; see [`scripts/README.md`](scripts/README.md).
- `reports/`: generated analysis packages stored on DCC `/work` and ignored by
  Git; compatibility links preserve each report's relative figure paths.
- `figures/`: the central collection of standalone, presentation-ready, and
  migrated report figures, grouped by study and stored on DCC `/work`.

Raw experiment outputs live in `results/`, and scheduler logs live in
`job_outputs/`. Both paths point to DCC `/work` storage and are ignored by git.
Historical material lives in the ignored, `/work`-backed `archive/` directory.
There is intentionally no top-level `plots/` directory: maintained plotting
code is in `scripts/plotting/`, while standalone generated images are under
`figures/`. Current paths such as `reports/<study>/figures/` are compatibility
links into that central figure tree so existing report HTML and TeX continue
to resolve. Plotting embedded in a new self-contained experiment may initially
write inside its report package.

On a fresh DCC clone, initialize the unversioned storage directories and links
with `bash scripts/maintenance/setup_dcc_storage.sh`.

DCC `/work` is scratch storage: it is not backed up, and files older than 75
days are automatically purged. Copy irreplaceable final results to persistent
storage; see Duke's [DCC storage documentation](https://oit-rc.pages.oit.duke.edu/rcsupportdocs/storage/).

## Experimental Conditions

The trainer exposes the optimizer comparisons used by the experiments:

- `standard_es`
- `standard_es_trust`
- `no_curvature`
- `diag_curvature`
- `global_curvature`
- `block_curvature`
- `directional_curvature`
- `normalized_diag_curvature`
- `normalized_block_curvature`

The clean comparisons are:

- `diag_curvature` vs `no_curvature`: evaluates whether diagonal curvature improves the semi-implicit replay/trust optimizer.
- `standard_es`: plain ES baseline.
- `standard_es_trust`: standard ES with the same trust-radius clipping interface.

## Mentor-requested no-trust Hessian rerun

The focused Hopper rerun compares only the original `main` conditions
`standard_es` and `diag_curvature`. It explicitly disables the trust radius
and applies the decreasing sequences `alpha_0 / sqrt(t + 1)` and
`alpha_0 / (t + 1)` for `alpha_0` in `{10, 30}`. The Standard ES arm is kept
as the required matched control. The population is increased from 200 to 500
to improve Hessian-estimate stability. Replay is fully disabled
(`reuse_fraction=0`, `buffer_size=0`), and geometry-free scalar damping is
removed (`implicit_damping=0`). Picard, replacement Hessian solvers, trust
variants, and optimizer-development arms are excluded.

The active entry points are the Slurm launcher
`scripts/slurm/submit_hopper_main_hessian_no_trust.sh` and the strict validator
`scripts/analysis/summarize_hopper_hessian_no_trust.py`.

By default, DIIWES estimates Stein curvature on the raw return scale. Use
`--curvature-fitness standardized` only for compatibility checks against the
older standardized-return estimator.

## Reports and plots

Standalone analysis, plotting, and Slurm entry points have one maintained home
under `scripts/`. New generated material should be written beneath
`reports/<study>/` or `figures/<study>/`. Report-facing figure links remain at
`reports/<study>/figures/`; for the migrated report packages, their image files
live under
`figures/reports/<study>/`. This preserves report-relative links without
duplicating the images.

The former top-level MuJoCo plot snapshots and their retired generation tools
are retained under `archive/analysis/` for provenance. Those figures
come from a trust-confounded sweep: they are historical evidence, not the
canonical Hessian comparison and not an active reproduction workflow.

## Local Execution

All commands below assume they are run from this repository root.

```bash
python experiments/train.py \
  --config configs/mujoco/hopper.yaml \
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
      --config configs/mujoco/hopper.yaml \
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

## Nonlinear policy-gradient initialization benchmark

The low-cost nonlinear benchmark adapts the PPO-initialization idea from
[Wang, Zhang, and Ying (2026)](https://arxiv.org/abs/2604.17747) without its
human-feedback or federated layers.
It compares random initialization with a REINFORCE/Adam first stage, then runs
the checked-out Standard ES and diagonal-curvature DIIWES implementations from
the same matched initial policies. The locked nonlinear protocol uses 300 ES
updates and 250 antithetic perturbation pairs (500 candidate policies) per
update:

```bash
python -m experiments.nonlinear_cartpole.benchmark
```

See `experiments/nonlinear_cartpole/README.md` for the protocol and smoke-run
options. The default report is written to
`reports/nonlinear_cartpole_warm_start/`.

## Curvature sample-size validation

The isolated curvature study calls the current DIIWES estimator directly and
compares it with analytic Gaussian-smoothed Hessian targets on nonlinear and
linear functions:

```bash
python -m experiments.curvature_sample_size.benchmark
```

Raw repetitions, aggregate error curves, validation checks, and the report
payload are kept separately in `reports/curvature_sample_size/`. See
`experiments/curvature_sample_size/README.md` for the protocol.

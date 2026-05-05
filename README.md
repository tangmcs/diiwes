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

## Local Execution

All commands below assume they are run from this repository root.

```bash
python experiments/train.py \
  --config configs/mujuco/hopper.yaml \
  --condition diag_curvature \
  --learning-rate 0.16 \
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

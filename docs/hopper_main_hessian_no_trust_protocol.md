# Hopper Main Hessian Without Trust Clipping

## Question

This sweep answers the mentor-requested question using the algorithms on
`main`: does the repository's `diag_curvature` condition improve on its
`standard_es` condition under large, decreasing learning rates when trust
clipping is disabled? The goal is improvement over Standard ES and diagnosis
of the Hessian path, not obtaining an optimal Hopper policy.

Job `49678516` does not answer this question. It ran the later experimental
`linearized_implicit_es` condition with
`configs/mujuco/hopper_implicit_no_replay.yaml`, not `main`'s
`diag_curvature` condition with `configs/mujuco/hopper.yaml`. Among other
differences, that job used fresh-only samples and a signed frozen-rank linear
system while disabling the replay, scalar damping, curvature
projection/clipping, curvature EMA, and multiplier-floor behavior present in
the main implementation. Its result therefore cannot be presented as a
result for the main Hessian algorithm.

## Locked protocol

- Environment/config: the original `configs/mujuco/hopper.yaml`.
- Conditions: `standard_es` and `diag_curvature` only; no Picard or additional
  curvature arm.
- Learning rates: `alpha_t = alpha_0 / sqrt(t + 1)` and
  `alpha_t = alpha_0 / (t + 1)`, with `alpha_0` in `{10, 30}`.
- Seeds: `0` through `9`, paired by condition within each schedule/rate cell.
- Updates: 500.
- Trust radius: explicitly `none` for every run.

The launcher changes only condition, seed, initial learning rate, learning-rate
schedule, and the requested trust setting. All other algorithm and experiment
settings retain `main` behavior from the original Hopper config. In
particular, `diag_curvature` still has the config's replay/importance-weighting,
raw-return diagonal curvature, damping, clipping, EMA, and multiplier-floor
semantics. Accordingly, this is a comparison of the two named main conditions,
not a fresh-only ablation that isolates curvature from every other difference.

The array has exactly 80 tasks. Seed is the fastest-changing index: tasks
`0-39` are Standard ES, tasks `40-79` are diagonal curvature, and matched
condition pairs differ by 40. Within each condition, tasks `0-19` relative to
the condition block use inverse square root and tasks `20-39` use inverse
linear; each schedule contains rate 10 (seeds 0-9) followed by rate 30 (seeds
0-9).

## Launch guard

Every Slurm task must receive `PAPER_EXPECTED_SOURCE_SHA` and independently
match it against the digest computed from the training source and locked
config. Compute the digest from the exact checkout to be submitted, then pass
it through Slurm's environment. Running the launcher directly is always a dry
run; `SLURM_ARRAY_TASK_ID=<id>` can be used locally to inspect any mapping
without starting training.

The task log records the protocol, full source revision and digest, matrix
indices, paired task ID, exact schedule formula, explicit no-trust setting,
output path, and fully quoted training command.

From the repository root, launch the locked array with:

```bash
SOURCE_SHA=$(python -c "from experiments.train import _source_digest; print(_source_digest('configs/mujuco/hopper.yaml'))")
PAPER_EXPECTED_SOURCE_SHA="$SOURCE_SHA" \
  sbatch scripts/submit_hopper_main_hessian_no_trust.sh
```

Do not edit `core/`, `experiments/train.py`, or the Hopper config while tasks
from the array are queued: every task recomputes and checks the digest before
training. After all 80 tasks finish, validate and summarize the matched matrix:

```bash
python scripts/summarize_hopper_hessian_no_trust.py \
  results/hopper_main_hessian_no_trust_<jobid> \
  --expected-source-sha "$SOURCE_SHA"
```

The validator writes run-level and grouped CSV files plus paired
`diag_curvature - standard_es` contrasts. It rejects partial grids, source or
configuration drift, an incorrect learning-rate sequence, any trust
rescaling, non-finite histories, and missing Hessian/linear-system diagnostics.

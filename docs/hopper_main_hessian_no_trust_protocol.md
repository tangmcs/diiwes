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
differences, that job used population 200 and a signed frozen-rank linear
system while disabling scalar damping, curvature
projection/clipping, curvature EMA, and multiplier-floor behavior present in
the main implementation. Its result therefore cannot be presented as a
result for the main Hessian algorithm.

## Locked protocol

- Environment/config: the original `configs/mujuco/hopper.yaml`.
- Conditions: `standard_es` and `diag_curvature` only; no Picard or additional
  curvature arm.
- Population: 500 candidates per update, increased from the main config's 200
  to improve the stability of the diagonal Hessian estimate. This yields 250
  fresh antithetic pairs at every update.
- Replay: disabled with both `reuse_fraction: 0` and `buffer_size: 0`. No old
  candidate receives an importance weight or enters the gradient/Hessian
  estimate.
- Scalar damping: disabled with `implicit_damping: 0`. The update therefore
  uses geometry only through `1 / (1 + alpha_t c_{t,j})`, before the existing
  multiplier clipping.
- Learning rates: `alpha_t = alpha_0 / sqrt(t + 1)` and
  `alpha_t = alpha_0 / (t + 1)`, with `alpha_0` in `{10, 30}`.
- Seeds: `0` through `9`, paired by condition within each schedule/rate cell.
- Updates: 500.
- Trust radius: explicitly `none` for every run.

The launcher changes only condition, seed, population size, replay settings,
scalar damping, initial learning rate, learning-rate schedule, and the
requested trust setting. All other algorithm and experiment settings retain
`main` behavior from the original Hopper config. In particular,
`diag_curvature` still has the main implementation's raw-return diagonal
curvature, curvature clipping, EMA, and multiplier-floor semantics; replay,
trust, and the geometry-free scalar damping term are removed. Accordingly,
this is a fresh-only comparison using the main Hessian estimator/update form,
not a newly substituted Hessian solver.

The array has exactly 80 tasks. Seed is the fastest-changing index: tasks
`0-39` are Standard ES, tasks `40-79` are diagonal curvature, and matched
condition pairs differ by 40. Within each condition, tasks `0-19` relative to
the condition block use inverse square root and tasks `20-39` use inverse
linear; each schedule contains rate 10 (seeds 0-9) followed by rate 30 (seeds
0-9).

## Clipping diagnostics

Every `diag_curvature` update records exact coordinate-level intervention
telemetry in both history artifacts. `curvature_coordinate_count`,
`curvature_active_count`, and `curvature_active_frac` distinguish coordinates
with active projected geometry from coordinates that actually saturate the
upper curvature cap. `curvature_preclip_mean` and
`curvature_preclip_max` describe that projected geometry before the cap.
Using the strict mask `curvature_preclip > curvature_clip`, the fields
`curvature_clip_count`, `curvature_clip_frac`, and
`curvature_clip_active` record the intervention exactly;
`curvature_clip_excess_mean` and `curvature_clip_excess_max` measure severity
among clipped coordinates and are zero when the cap is inactive.

The multiplier diagnostics likewise preserve the unclipped values.
`multiplier_coordinate_count`, `raw_step_multiplier_min`, and
`raw_step_multiplier_max` describe the solve before the multiplier bounds.
`multiplier_clipping_diagnostics_exact` must be true; this explicitly limits
the contract to the locked `dampen` update rather than the separately clipped
normalized-curvature variants.
Using the strict mask `raw_step_multiplier < min_step_multiplier`,
`multiplier_floor_clip_count`, `multiplier_floor_clip_frac`, and
`multiplier_floor_clip_active` give the exact lower-floor intervention, while
`multiplier_floor_clip_deficit_mean` and
`multiplier_floor_clip_deficit_max` measure the deficit among floored
coordinates and are zero when inactive. The legacy `multiplier_floor_frac`
remains as a post-clipping at-floor occupancy measure (so exact equality may
be included), but the strict pre-clipping `*_clip_*` fields are canonical.
The analogous `multiplier_ceiling_clip_*` fields record any intervention at
the upper bound of one. They should remain zero in this locked protocol because
zero scalar damping and nonnegative projected curvature imply a raw multiplier
no greater than one.

Each diagonal run also writes two exact coordinate artifacts.
`hessian_for_step_history.npy` stores the signed, bias-corrected Hessian vector
actually used by the update, and `step_multiplier_history.npy` stores the final
post-clipping multiplier applied to each gradient coordinate. Both are
`float64` arrays with shape `(500, 5123)`, indexed by update and policy
coordinate. For update `t`, reconstruct the intervention path as

```text
H[t]                       = hessian_for_step_history[t]
c_pre[t]                   = maximum(-H[t], 0)
c[t]                       = clip(c_pre[t], 0, curvature_clip)
alpha[t]                   = alpha_0 / sqrt(t + 1)  or  alpha_0 / (t + 1)
m_raw[t]                   = 1 / (1 + alpha[t] * c[t])
m[t]                       = clip(m_raw[t], min_step_multiplier, 1)
step_multiplier_history[t] = m[t]
```

Thus `c_pre[t] > curvature_clip` exactly reconstructs the upper-cap mask and
`m_raw[t] < min_step_multiplier` exactly reconstructs the multiplier-floor
mask. Equality is not an intervention. These formulas rely on this protocol's
zero scalar damping and zero L2 coefficient; the saved signed Hessian and
final multiplier remain the authoritative values.

Each array has 20,492,000 bytes of numeric payload. With the 128-byte `.npy`
header produced in this environment, that is 20,492,128 bytes per file and
40,984,256 bytes per diagonal run. Across 40 diagonal runs, the exact projected
increment is 1,639,370,240 bytes (1.639 GB, or 1.527 GiB).
Standard ES runs do not create either coordinate array. Consumers should use
memory mapping, for example `np.load(path, mmap_mode="r")`, when full in-memory
loading is unnecessary.

The coordinate arrays are preallocated and each completed update row is
written and flushed incrementally. After an interruption, only the prefix
corresponding to complete records in `history.jsonl` is eligible for diagnosis;
the remaining preallocated rows are not observations and must be ignored. A
valid completed run has all 500 coordinate rows plus the canonical
`history.json`; interrupted artifacts cannot be promoted to a matrix cell.

Counts and active flags can be summed to obtain exact clipped-coordinate
events and active iterations across a run or the full matrix; fractions and
pre-clipping excess/deficit values retain intervention prevalence and
severity. Run summaries include both zero-inclusive iteration means and
count-weighted mean excess/deficit per clipped coordinate-update event. The
validator summarizes these at run, schedule/rate group, and paired-cell
levels. Normal progress lines also show `CurvCap` and `MultFloor`
percentages, but the periodic console output is not the source for exact
totals.

`history.json` is the canonical artifact for a completed 500-update run.
`history.jsonl` is flushed after every completed update and preserves the same
records if a run is interrupted; it supports failure diagnosis but does not
turn an incomplete run into a valid matrix cell. Training refuses to overwrite
an output directory that already contains a completed `history.json`.

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
  results/hopper_main_hessian_fresh_no_trust_no_scalar_damping_pop500_<jobid> \
  --expected-source-sha "$SOURCE_SHA"
```

The validator writes run-level and grouped CSV files plus paired
`diag_curvature - standard_es` contrasts. It rejects partial grids, source or
configuration drift, an incorrect learning-rate sequence, any trust
rescaling, non-finite histories, and missing Hessian/linear-system diagnostics.

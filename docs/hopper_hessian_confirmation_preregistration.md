# Hopper Hessian Confirmation Preregistration

## Selection Evidence

Job `49678999` was an exploratory five-seed, four-cell mechanism screen. All
100 runs passed strict validation. Concave layer-block curvature with a
bias-corrected `beta=0.9` EMA was the only arm with a positive mean AUC@75k
difference from matched Standard ES in every learning-rate cell. At the
selected `alpha_t = 10 / (t + 1)` cell, the paired AUC differences were
`+22.36`, `+37.79`, `+0.01`, `+19.28`, and `+39.27`.

This screen is selection evidence only. Its seeds and repeated comparisons are
not part of the confirmatory inference. Raw split-half and temporal curvature
agreement remained near chance, so improved performance cannot yet be
attributed to useful layer-specific curvature.

## Fixed Question

Does concave-projected layer-block EMA curvature improve held-out AUC over
Standard ES, and is any improvement larger than an isotropic control with the
same curvature-derived step attenuation?

## Fixed Protocol

- Environment: `Hopper-v5`
- Training seeds: `100` through `109`, paired across all arms
- Updates: 500, with no early or interim stopping
- Population: 200 fresh candidates, 100 antithetic pairs
- Noise scale: `0.02`
- Fitness transform: centered ranks for both gradient and curvature surrogate
- Learning rate: `alpha_t = 10 / (t + 1)`
- Observation normalization: frozen after three calibration episodes
- Held-out endpoint: post-training evaluation of the initial center and every
  successive center through the first crossing of 75,000 training steps
- Held-out episodes: 20 per checkpoint from seed stream 4, unused during
  training and common across arms within each paired training seed

Replay, sample importance weighting, Picard iteration, trust clipping, scalar
damping, gradient clipping, parameter projection, curvature clipping, L2, and
fixed-radius or fixed-norm updates are disabled. Held-out returns cannot affect
optimization, stopping, checkpoint selection, reruns, or exclusions.

## Arms

1. `standard_es`: the matched explicit ES baseline.
2. `concave_block_ema_curvature_es`: the selected structured method.
3. `concave_block_ema_isotropic_control_es`: computes the same hypothetical
   structured step on its own current batch/state, sets
   `q = ||Delta_structured|| / ||alpha_t g||`, and applies `q alpha_t g`.
   It retains the Standard ES direction while matching the structured
   reference step norm. There is no target radius or trust parameter.

The array contains 30 tasks. Conditions rotate within each adjacent
three-task seed block so that no condition is always launched first.

## Primary Endpoint And Inference

The primary endpoint is normalized trapezoidal held-out return AUC through
exactly 75,000 actual training environment steps. There are exactly two primary
paired contrasts:

1. structured block-EMA minus Standard ES;
2. structured block-EMA minus isotropic attenuation control.

For each contrast, report the paired mean, sample standard deviation, median,
95% paired-mean t interval, wins out of ten, and the exact two-sided paired
sign-flip p-value. Apply Holm correction across exactly these two p-values.

A useful layer-curvature result requires both paired means to be positive and
both Holm-adjusted p-values to be below `0.05`. Return@75k and existing online
evaluation metrics are secondary and cannot rescue a failed primary endpoint.

## Validation And Failures

Analysis begins only after all 30 runs pass the locked source/config digest,
rotated task mapping, 500-record JSON/JSONL equality, finite histories, 200
fresh and zero replayed samples per update, unit fresh/importance weights,
empty stderr, and complete held-out checkpoint/seed-bank validation.

Structured curvature must have denominator at least one, nonpositive-denominator
fraction zero, residual at most `1e-10`, and step ratio at most one. The
isotropic control must additionally have `0 < q <= 1` and relative norm-match
error at most `1e-10`. Infrastructure failures may be rerun under the identical
digest and seed before analysis; no seed may be excluded after results are
observed.

## Separate Pilot

The joint block-OLS estimator and one-sided confidence gate are not a
confirmatory arm. They change the estimator and gate after the selection job
and therefore remain separately labeled pilot work. Any promotion requires a
new untouched seed set and a new preregistration.

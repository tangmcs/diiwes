# Review of "Issues of the Current DIIWES Algorithm"

## Verdict

The main ES, Gaussian-ratio, trust-cancellation, and Stein-estimator signs are
correct. The document should not be presented as a description of the current
implementation. It analyzes the July 6 trust/replay implementation, while the
current experiment protocol disables replay, trust clipping, and the multiplier
floor.

A clearer title is `Audit of the July 6 Historical DIIWES Implementation`.
The historical `core/diiwes.py` path remains a projected preconditioner and is
not an endpoint implicit solve. The mentor-requested launcher now uses the
separate `EndpointImplicitES` and `LinearizedImplicitES` implementations in
`core/implicit_es.py`; replay, trust clipping, scalar damping, curvature
projection, curvature clipping, and curvature EMA are disabled in that matrix.

## Current Applied Update

The current code forms a bias-corrected curvature EMA, then applies

```text
h_bar_t = bias_corrected_ema(h_hat_t)
c_tj = min(c_max, max(-h_bar_tj, 0))
g_t = g_data,t - lambda_2 * theta_t
delta_tj = alpha_t * g_tj /
            (1 + alpha_t * (lambda_scalar + lambda_2 + c_tj))
```

For the Hopper configuration, `lambda_2 = 0`. The signed denominator

```text
1 + alpha_t * (lambda_scalar + lambda_2) - alpha_t * h_bar_tj
```

is logged only as a diagnostic. It is not applied. The method should be called
`projected diagonal curvature-preconditioned ES`, not implicit ES.

## Required Corrections

1. **Framework scope**

   Move the replay, importance-weighting, ESS, trust-radius, and multiplier-floor
   subsections into a section explicitly labeled historical. The active protocol
   has `replay_enabled=false`, `reuse_fraction=0`, and `buffer_size=0`.
   If the optional replay code is discussed, note that it excludes candidates
   below the ratio floor before batching and uses a fresh-only empirical rank
   reference; it does not lower-clip every old sample in a combined-rank batch.

2. **Gradient identity versus implemented pseudo-gradient**

   The raw Stein identity for `grad F_sigma` is correct. Rank shaping does not
   estimate that raw gradient. Clipped self-normalized importance sampling,
   an estimated baseline, and data-dependent sample selection also introduce
   finite-sample bias. Describe the implemented quantity as a shaped ES
   pseudo-gradient.

3. **Historical multiplier equation**

   Even for the retired code, the multiplier equation omitted scalar damping
   and L2:

   ```text
   m_tj = clip(1 / (1 + alpha_t *
                    (lambda_scalar + lambda_2 + c_tj)), m_min, 1)
   ```

   The present code has no `m_min` and no trust-radius rescaling.

4. **Gaussian overlap calculation**

   The log-ratio mean, standard deviation, and KL calculations are correct.
   With `sigma=0.02` and a unit center displacement, the mean log ratio is
   `-1250`, its standard deviation is `50`, and the Gaussian KL is `1250`.
   Label the 99 percent clipping observation as historical evidence.

5. **ESS wording**

   Global ESS is not mathematically misleading; it measures total weight
   concentration rather than replay usefulness. In the stated example,
   `ESS/B` is approximately `0.8004` while replay mass is approximately
   `2.499e-4`. State that a separate replay-mass diagnostic is required.

6. **Curvature estimator**

   The antithetic Stein factor `1/(2*sigma^2)` and leave-one-out baseline are
   correct when other pairs and rollout seeds are independent. The baseline is
   intended to reduce variance, but reduction is not guaranteed. Constant
   reward offsets are removed by the baseline; arbitrary reward scaling still
   changes both true curvature and its estimate, making fixed clipping
   thresholds scale-dependent.

   With replay disabled, population size 200 gives exactly 100 fresh
   antithetic curvature pairs, not 80.

   For ascent, a negative diagonal Hessian estimate motivates coordinate
   damping. Individual diagonal signs are basis-dependent and do not by
   themselves establish that the full objective is concave or convex.

7. **Gradient-curvature mismatch**

   Problem 5 is correct and should be strengthened. The rank gradient and raw
   Hessian are not derivatives of one objective. The matched-rank estimate is
   also a batch-defined surrogate rather than a literal Hessian. The current
   code has no coherent raw-gradient/raw-Hessian arm because `rank_fitness=false`
   uses standardized rather than raw fitness.

8. **Ablation matrix**

   The proposed trust-centered table conflicts with the no-trust question.
   `Normalized ES` and `trust-only DIIWES` are also redundant when clipping is
   active. For the immediate diagnostic, use no replay and no trust. Compare:

   | Method | Interpretation |
   | --- | --- |
   | Standard ES | Fresh explicit baseline |
   | Scalar-damped ES | Pure effective-learning-rate control |
   | Projected curvature ES | Estimator/preconditioner diagnostic |

   This historical table still does not test implicit ES. The completed
   three-arm diagnostic included endpoint Picard, but the active advisor-facing
   follow-up excludes Picard and compares Standard ES with the matched signed
   diagonal linearization under two decreasing learning-rate laws.

9. **Rank shaping and unbiasedness**

   Problem 8 should say rank shaping adds a batch-dependent target mismatch.
   It is not the only source of bias: self-normalization and ratio clipping are
   already finite-sample biased.

10. **Novelty language**

    Present the final section as literature-positioning risks requiring
    citations, not definitive novelty conclusions. Fixed norm clipping,
    importance-weighted ES, and Hessian-estimation ES all have close prior art.

## Historical Results

The four single-seed Hopper values in the document match the saved histories
after rounding. They came from separate job families and source snapshots with
no content digest, so they should be labeled exploratory evidence rather than a
controlled causal ablation.

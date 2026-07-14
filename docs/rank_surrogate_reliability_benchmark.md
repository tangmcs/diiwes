# Rank-Surrogate Covariance-Score Reliability Benchmark

Status: locked diagnostic artifact, July 12, 2026. Schema version 2.

> **Post-lock scope clarification.** The schema-v2 JSON phrase "not used by
> the production optimizer" refers specifically to the pooled-rank optimizer
> evaluated in this locked historical benchmark. The repository now also
> contains the separate diagnostic condition `concave_block_lopo_u_stat`,
> which uses matched LOPO gradient and curvature. It was not evaluated by this
> benchmark, and no locked CSV/JSON field or numeric result has been changed.

## Executive conclusion

This benchmark establishes three facts that must remain separate:

1. The raw production same-batch statistic is implemented correctly as the
   diagonal of the **conditional Jacobian of the frozen-utility endpoint
   gradient**. The production diagonal agrees with an independently assembled
   matrix to `7.77e-16`, and the full matrix agrees with a central finite
   difference to relative Frobenius error `4.81e-11`.
2. Under iid antithetic pair clusters, the raw curvature statistic has the
   exact population relation

   ```text
   E[J_D] = c_m H_stop,    c_m = 2 (m - 1) / (2m - 1).
   ```

   Here `H_stop` is the Hessian of a current-return-mid-CDF transformed
   objective with that CDF held fixed. `J_D / c_m` is exactly the matched
   leave-one-pair-out (LOPO) rank order-two U-statistic. It is not a raw-return
   Hessian or the total Hessian of a globally adaptive rank objective.
3. Correct arithmetic and a precise population target do not imply a reliable
   curvature signal. On clean synthetic quadratics, pooled block estimates
   have useful target agreement. With additive observation noise and
   `sigma = 0.02`, target correlation and independent split agreement are near
   zero. Fixed-reference and cross-fitted transforms do not repair that regime.

The pooled-rank production gradient retains a finite-`m` within-pair comparison
term. Therefore dividing curvature by `c_m` alone does not produce a matched
population implicit method; a future population-targeted method must use LOPO
utilities for both gradient and curvature. No optimizer behavior is changed by
this benchmark.

The defensible conclusion remains diagnostic: at the small perturbation scale
used in Hopper, curvature signal-to-noise is a primary failure mechanism in
this controlled model. The benchmark does not establish policy improvement or
superiority to standard ES.

## Quantities being compared

### Raw production conditional Jacobian

For a frozen antithetic batch `D = {(epsilon_i, u_i)}_{i=1}^B`, the endpoint
gradient is

```math
G_D(\delta)
=
\frac{1}{\sigma}
\sum_i p_i(\delta)
\left(u_i-\bar u_{p(\delta)}\right)
\left(\epsilon_i-\frac{\delta}{\sigma}\right),
\qquad
p_i(\delta)
=
\frac{\exp(\epsilon_i^\top\delta/\sigma)}
{\sum_k\exp(\epsilon_k^\top\delta/\sigma)}.
```

Holding the observed utilities fixed,

```math
J_D
=
\nabla_\delta G_D(0)
=
\frac{1}{B\sigma^2}
\sum_i u_i(\epsilon_i\epsilon_i^\top-I).
```

Centered ranks sum to zero, so the `-I` term has no numerical effect. This is
an exact conditional statement for the production code.

### Matching same-batch population target

Let

```text
K(y,y') = 1{y > y'} - 1{y < y'},
A(X,X') = sum_s,t K(Y_s,Y'_t),
S(X) = epsilon epsilon^T - I.
```

The comparison is zero on ties, exactly matching midrank semantics. For two
independent antithetic pair clusters define

```text
h_H(X,X') = A(X,X') [S(X) - S(X')] / (16 sigma^2).
```

For every realized batch with `m >= 2`, including ties,

```text
J_D / c_m
  = choose(m, 2)^(-1) sum_{k < l} h_H(X_k, X_l)
  = J_LOPO.
```

This corrected statistic is unbiased for `H_stop`. The benchmark estimates the
matching target independently by averaging `h_H` over 50,000 independent
pair-of-pairs kernel draws. This removes the previous empirical-CDF estimand
mismatch from same-batch rows.

The corresponding gradient identity is

```text
g_current
  = c_m g_LOPO
    + [1 / (2m sigma (2m - 1))]
      sum_k epsilon_k K(Y_k,+,Y_k,-).
```

Thus corrected curvature and the current pooled-rank gradient are not a matched
finite-population pair.

### Other transform targets

Independent-reference and cross-fitted rows retain the original comparison to
a fixed empirical-reference-CDF covariance score. An independent reference of
size 50,000 defines `U_ref`; a target sample of size 100,000 estimates

```math
H_{\mathrm{ref}}
=
\frac{1}{\sigma^2}
\mathbb E[U_{\mathrm{ref}}(Y)(\epsilon\epsilon^\top-I)].
```

These finite-batch estimates are centered on their current batch, so their
`relative_rmse` still mixes sampling variation, current-batch centering, and
transform mismatch. Cross-fitted folds are also mutually dependent.

Accordingly, `relative_rmse` has transform-specific semantics:

| Transform | Reported estimate and target |
| --- | --- |
| Same-batch centered rank | `J_D / c_m = J_LOPO` versus an independent matching pair-of-pairs estimate of `H_stop`. |
| Independent reference CDF | Batch-centered fixed-CDF statistic versus `H_ref`; a fixed-reference discrepancy. |
| Cross-fitted rank | Reciprocal-fold batch statistic versus `H_ref`; a fixed-reference discrepancy with estimand mismatch. |

Values across transforms remain useful diagnostics, but they are not errors of
three unbiased estimators for one common finite-sample estimand.

## Protocol

The benchmark has no optimizer loop or RL environment. Its Cartesian grid is:

- surfaces: diagonal concave, block-isotropic concave, rotated concave, and
  block saddle;
- dimensions: 8 and 32;
- antithetic populations: 40 and 200;
- perturbation scales: `0.02` and `0.1`;
- linear-term scales: 0 and 1;
- observation noise: none, independent Gaussian, or one Gaussian draw shared
  within each plus/minus pair (`paired_crn`);
- transforms: same-batch, independent reference CDF, and pair-preserving
  two-fold cross-fit;
- structures: coordinate diagonal and four contiguous block means;
- repetitions: 100 per cell; seed: `20260712`.

This gives 1,152 unique summary cells. Additive noise has standard deviation
0.1. Matching same-batch targets use independent plus/minus noise marginals;
their expectation is invariant to the within-pair coupling used by a finite
estimate. `paired_crn` is only a controlled coupling model and does not claim
to reproduce MuJoCo rollout noise.

The `correlation` field pools repetition-by-component estimates against the
repeated target. Split correlations are computed within each repetition and
then summarized. Tables report medians of scalar cell summaries, giving every
design cell equal weight.

## Exact validation

The locked metadata records a separate six-dimensional, population-20 check:

| Check | Result |
| --- | ---: |
| Production diagonal vs. independent analytic diagonal, max absolute error | `7.7716e-16` |
| Full analytic matrix vs. central finite difference, max absolute error | `3.4081e-10` |
| Full matrix relative Frobenius error | `4.8137e-11` |
| Production endpoint gradient vs. independent implementation, max absolute error | `0` |
| Finite-difference relative step | `1e-5` |

Tests additionally verify the exact tied-return identity
`J_D = c_m J_LOPO` and the pooled-gradient within-pair remainder. On the
analytic one-dimensional case `Y = epsilon^2`, where `H_stop = 1 / pi`, a
fixed 200,000-batch Monte Carlo check agrees with `c_m / pi` within `0.002`.

These checks validate arithmetic and estimand identities. They do not validate
statistical reliability, the diagonal/block approximation, the implicit solve,
or policy-learning performance.

## Results

### Transform comparison

The following medians use all 64 block cells for each noise-transform pair.
`D` is `relative_rmse` with the transform-specific target stated above.

| Noise | Transform | `D` | Target corr. | Split corr. | Split sign agreement |
| --- | --- | ---: | ---: | ---: | ---: |
| None | Same batch / LOPO | 0.478 | 0.785 | 0.656 | 0.799 |
| None | Independent reference | 0.479 | 0.766 | 0.688 | 0.808 |
| None | Cross-fit | 0.521 | 0.748 | 0.657 | 0.793 |
| Independent | Same batch / LOPO | 3.686 | 0.107 | 0.031 | 0.538 |
| Independent | Independent reference | 3.540 | 0.104 | 0.026 | 0.532 |
| Independent | Cross-fit | 3.738 | 0.102 | 0.117 | 0.549 |
| Paired shared | Same batch / LOPO | 4.968 | 0.088 | 0.033 | 0.532 |
| Paired shared | Independent reference | 4.741 | 0.079 | 0.032 | 0.532 |
| Paired shared | Cross-fit | 5.056 | 0.080 | 0.094 | 0.538 |

The clean results show broadly similar agreement across transforms. Under
either additive-noise coupling, all three lose target and split agreement.
This does not support same-batch ranking as the primary source of the noisy
failure, and cross-fitting does not recover the missing signal.

Same-batch split diagnostics rerank two disjoint pair halves and divide each by
its own finite-population factor. They are independent under the iid-pair
model. Independent-reference splits are independent conditional on their
shared reference CDF. Reciprocal cross-fit split diagnostics are dependent.

### Perturbation-scale stress test

These medians retain corrected same-batch/LOPO block rows, with 32 cells per
line.

| Noise | `sigma` | `D` | Target corr. | Split corr. | Split sign agreement | Fully resolved target cells |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 0.02 | 0.479 | 0.770 | 0.656 | 0.800 | 32/32 |
| None | 0.10 | 0.460 | 0.787 | 0.718 | 0.799 | 32/32 |
| Independent | 0.02 | 13.358 | 0.032 | -0.064 | 0.493 | 4/32 |
| Independent | 0.10 | 0.766 | 0.607 | 0.367 | 0.673 | 32/32 |
| Paired shared | 0.02 | 18.653 | 0.036 | -0.012 | 0.499 | 4/32 |
| Paired shared | 0.10 | 0.931 | 0.539 | 0.298 | 0.635 | 32/32 |

A target cell is fully resolved only when every component exceeds `1.96` times
its target Monte Carlo standard error in magnitude. At noisy `sigma = 0.02`,
only 4 of 32 matching targets are fully resolved. Independent split agreement
is also near chance, which does not rely on target resolution.

The failure is consistent with the covariance score's `1 / sigma^2` scale and
fixed observation noise dominating return order as `sigma` shrinks. This is a
controlled association, not a theorem about every RL noise process.

### Pooling and population size

Across all other factors, block pooling lowers median same-batch/LOPO target
error relative to the coordinate diagonal:

| Noise | Diagonal `D` | Block `D` | Diagonal target corr. | Block target corr. |
| --- | ---: | ---: | ---: | ---: |
| None | 0.795 | 0.478 | 0.555 | 0.785 |
| Independent | 7.398 | 3.686 | 0.067 | 0.107 |
| Paired shared | 10.069 | 4.968 | 0.070 | 0.088 |

The diagonal and block rows target different resolutions, so this is not a
direct estimator-efficiency comparison. It also does not establish that four
architecture blocks approximate a neural-network Hessian.

Increasing population from 40 to 200 helps but does not make the noisy
aggregate reliable. Under independent noise, block `D` falls from 7.514 to
3.199 and correlation rises from 0.066 to 0.133. Under paired-shared noise,
the corresponding changes are 10.325 to 4.387 and 0.067 to 0.096. These
medians mix both perturbation scales; the scale table remains essential.

## Limitations and claim boundary

- The benchmark evaluates controlled quadratics, not policy learning,
  nonstationary training, or an RL environment.
- Same-batch results target a current-CDF **stop-gradient** Hessian. This is not
  a raw-return Hessian or the total Hessian of an adaptive rank objective.
- Correcting curvature alone does not correct the production gradient's
  within-pair term. A matched LOPO optimizer is not evaluated in this benchmark.
- Matching targets use 50,000 Monte Carlo kernel draws. Their uncertainty is
  material in the noisy `sigma = 0.02` regime.
- Relative error across transforms has different estimand semantics and cannot
  rank all transforms as estimators of one common finite-sample target.
- Split statistics use half populations. Cross-fit splits remain dependent;
  four-component block correlations are individually unstable.
- Four contiguous blocks are a coarse structural assumption. Off-diagonal
  recovery is not evaluated.
- The additive noise models omit state dependence, heteroskedasticity, and
  temporal correlation.
- Cell medians are descriptive, without multiplicity-adjusted tests.
- The moment estimator's logged `std(pair_contributions)/sqrt(m)` value is not
  a valid same-batch-rank standard error because pair contributions are
  dependent. The active Stein method does not use it as a confidence gate; a
  U-statistic jackknife is required for inferential use.
- No optimizer comparison appears here, so the artifact cannot establish that
  curvature improves standard ES.

## Reproduction and provenance

The locked artifacts are:

| File | SHA-256 |
| --- | --- |
| `docs/rank_surrogate_reliability_benchmark.csv` | `f897b3f52577de93157f69ba4209a6f9fdf245291b89ca3d29c0e648c4676d67` |
| `docs/rank_surrogate_reliability_benchmark.json` | `990bb68632666da8b86e43e92cccc1934708e996b5e2cb1bc242fe78096f7435` |
| `experiments/rank_surrogate_reliability_benchmark.py` | `8edf8644db77c7db3de349d7dfe26b63584472aae19d6d9ad226a3f60c8c0434` |
| `tests/test_rank_surrogate_reliability_benchmark.py` | `255887db60a4914fa16562588878205b7f2ed1b7eb5a1ac2d404122c8c3c22b2` |

The JSON records the full configuration, scope, conditional-Jacobian check,
and all 1,152 rows. Validation found 1,152 unique grid keys, no missing cells,
no non-finite numeric values, and exact CSV/JSON row agreement.

Reproduction command:

```bash
source /hpc/home/rt239/miniconda3/bin/activate es_parallel
PYTHONPATH=. python experiments/rank_surrogate_reliability_benchmark.py \
  --output /tmp/rank_surrogate_reliability_benchmark_rerun.csv \
  --metadata-output /tmp/rank_surrogate_reliability_benchmark_rerun.json
```

The audit uses Python 3.10.18 and NumPy 1.26.4. A fresh default rerun matched
all categorical and numeric CSV fields exactly in this environment. Locked
hashes identify the reported artifact; cross-platform reruns should still use
`rtol=1e-12`, `atol=1e-12` for floating-point comparisons.

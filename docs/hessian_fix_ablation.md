# Hessian Estimator and Solve-Stability Ablation

## Motivation

The completed no-Picard job `49678516` showed that the signed diagonal method
does not robustly improve Standard ES. Across all four schedule cells, the
direct division residual was approximately `4e-17`, but the estimated Hessian
had split and temporal correlations near zero and coordinate sign agreement
near `0.5`. The smallest observed absolute denominator was `6.29e-9`, and the
largest step amplification was `1,064,336` times the explicit-step norm.

This rules out inaccurate elementwise division. The failure is the combination
of an unidentifiable 5,123-coordinate estimate and the resonance in
`1 - alpha_t * h_j`.

## Structured Curvature

For a layer block `B`, the new estimator pools the matched-rank diagonal
surrogate:

```text
kappa_B = (1 / (2 m sigma^2)) * sum_k s_k
          * (||epsilon_k,B||^2 / |B| - 1),
```

where `s_k = u_k,+ + u_k,-`. Hopper's three MLP layer blocks contain 768,
4,160, and 195 parameters. Pooling reduces the Gaussian factor variance from
`2` for one coordinate to `2 / |B|` for a block while retaining layer-specific
preconditioning.

Pooling imposes layer-isotropic curvature, so it trades within-layer curvature
bias for lower estimator variance. The controlled diagnostic below verifies
the intended mechanics on a quadratic; it is not evidence that block curvature
is accurate in Hopper. That question is reserved for the paired environment
ablation.

For ascent, the stabilized update retains only concave curvature:

```text
c_B = max(-kappa_B, 0)
delta_B = alpha_t * g_B / (1 + alpha_t * c_B).
```

Every denominator is at least one. Therefore, the Hessian term cannot reverse
a coordinate, create a singular solve, or amplify the explicit step. This is
not fixed-radius clipping: no update norm is prescribed, and `alpha_t` remains
in both the numerator and denominator. The method is labeled concave-projected
curvature rather than an exact signed implicit solve.

## Controlled Diagnostic

On a 5,123-dimensional concave quadratic with 100 antithetic pairs, 50
replicates, and the Hopper layer partition:

| Statistic | Diagonal | Layer block |
| --- | ---: | ---: |
| Independent-estimate correlation | 0.0065 | 0.9729 |
| Coordinate/component sign agreement | 0.5064 | 0.8600 |
| Median signed condition, alpha=10 | 25,545 | 9.46 |
| Median signed condition, alpha=30 | 28,102 | 9.47 |

The concave projection has minimum denominator `1` and maximum amplification
`1` by construction. Full diagnostic output is in
`docs/structured_curvature_diagnostic.csv`.

## Experiment Matrix

The follow-up contains five arms:

1. `standard_es`: explicit control.
2. `linearized_implicit_es`: current signed diagonal failure control.
3. `concave_diagonal_curvature_es`: solve stabilization only.
4. `concave_block_curvature_es`: stabilization plus layer pooling.
5. `concave_block_ema_curvature_es`: layer pooling plus bias-corrected
   `beta=0.9` temporal averaging.

Each arm uses `alpha_0` in `{10, 30}`, inverse-square-root and inverse-linear
schedules, and paired seeds `0` through `4`: 100 runs. The primary comparison
is paired return AUC over the first 75,000 training environment steps. Replay,
Picard, trust clipping, scalar damping, gradient clipping, parameter
projection, curvature clipping, and an update-norm fallback are disabled.

The first five-seed stage is a mechanism ablation. An arm should be promoted to
ten seeds only if it is mechanically stable and shows consistent paired gains;
a single favorable mean is insufficient.

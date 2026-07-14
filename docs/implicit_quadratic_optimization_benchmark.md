# Controlled Implicit Quadratic Optimization Benchmark

## Purpose

This benchmark turns the curvature-surrogate theory into deterministic,
falsifiable optimization trajectories. It is a synthetic mechanism study, not
evidence of reinforcement-learning performance.

The benchmark separates:

1. exact-gradient dynamics with known full Hessians;
2. raw-fitness Monte Carlo ES, for which the Gaussian Stein statistic has a
   literal quadratic-Hessian target; and
3. same-batch centered-rank Monte Carlo ES, for which the statistic is labeled
   a frozen-rank covariance-score surrogate and has no literal-Hessian RMSE.

Implementation:
[`experiments/implicit_quadratic_optimization_benchmark.py`](../experiments/implicit_quadratic_optimization_benchmark.py)

Tests:
[`tests/test_implicit_quadratic_optimization_benchmark.py`](../tests/test_implicit_quadratic_optimization_benchmark.py)

## Fixed Protocol

| Setting | Value |
| --- | --- |
| Dimension | 12 |
| Coordinate blocks | 3 equal blocks |
| Population | 96 candidates / 48 antithetic pairs |
| Updates | 30 |
| Perturbation scale | `sigma = 0.1` |
| Learning rates | `0.1`, `1`, `10` |
| Monte Carlo seeds | `0` through `4` |
| Additive-noise SD | `0.05` in the noise case |
| Parameter-norm divergence boundary | `1e6` |
| Finite-optimum gap-ratio boundary | `1e6` times initial gap |

Every method within a case, learning rate, seed, and iteration uses the same
Gaussian perturbations and additive-noise draws. The raw curvature estimator
uses a leave-one-antithetic-pair-out pair-sum baseline. Post-divergence metrics
carry the applicable boundary state forward to the fixed horizon, so failed
runs cannot disappear from AUC calculations.

The four cases are:

- `block_aligned_concave`: negative block-isotropic Hessian, so the block model
  is correctly specified;
- `rotated_concave`: the same negative spectrum after a dense rotation, so the
  architecture-style coordinate blocks are misspecified;
- `rotated_indefinite`: a dense saddle with positive and negative eigenvalues;
  it has no finite maximum and therefore reports no objective gap; and
- `block_aligned_additive_noise`: the aligned concave case with independent
  additive observation noise.

The compared updates are explicit ES, full-Hessian oracle implicit, signed
sampled diagonal, signed sampled block, concave-projected structured block,
and norm-matched isotropic attenuation. In the rank regime, the full-Hessian
oracle is replaced by the full frozen-batch signed Jacobian control because a
raw Hessian would be an objective-mismatched oracle.

## Metrics

- True quadratic objective and, only for strictly concave cases, objective gap.
- Trapezoidal mean objective AUC and initial-gap-normalized gap AUC.
- Parameter/gap divergence and the first divergence iteration.
- Step amplification relative to `alpha * gradient`.
- Minimum absolute system denominator or eigenvalue margin.
- Nonpositive denominator fraction.
- Raw diagonal and block curvature RMSE against the known Hessian.
- True one-step structured-versus-isotropic directional benefit at exactly
  matched step norm.

The directional comparison is evaluated from the same state using the same
sampled gradient and curvature statistic. It measures direction, not a norm
advantage, and does not affect either trajectory.

## Validated Artifact

Run:

```bash
python experiments/implicit_quadratic_optimization_benchmark.py \
  --output-dir results/implicit_quadratic_optimization_benchmark
```

Validated output:

```text
Validated implicit quadratic benchmark: summaries=774, trajectories=23994
```

The run used source SHA-256
`9abc719d8a4206a664250da12281b3abd6497b3d0cad4374d7da39e3d2cf171b`.
The manifest is
`results/implicit_quadratic_optimization_benchmark/benchmark_manifest.json`
with SHA-256
`50cb92e8cb42b25dfa4ee8e62e8574a38e8cecc7f1a58d7ac8956970094b5785`.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `aggregate_summary.csv` | 198 | `d91a7093c00f067b336260e878978c7fc8dadb0ec2dabc3a5c3b71f2f4d0309b` |
| `directional_aggregate.csv` | 33 | `98e74419329286d41febcc36b80746d11c21adf935ef72602602d373c6b0ec3a` |
| `directional_comparison.csv` | 3,263 | `fa697733c0df8bfcbc036cf280299055a0d8e4442e27a7e1dde7ea68867bf14b` |
| `run_summary.csv` | 774 | `40b31eb8de5c703ab2d915e79d93b57e7de8fb612ced0cc820425c46c2ae4a81` |
| `trajectories.csv` | 23,994 | `91c1c4670c6fc5981715beb7a75356e0bda687b47a6ed662f125df1d872add00` |

The validator confirmed a complete matrix, unique run/update keys, finite
machine-readable numbers, no amplification by either safe attenuation method,
exact norm matching by the isotropic control, and suppression of nonexistent
gap metrics in the indefinite case.

`aggregate_summary.csv` reports divergence counts and rates before any
performance field. Boundary-inclusive means exist only to audit the fixed
horizon and must not be read as ordinary performance. The separately labeled
nondiverged means exclude failed runs and are survivor descriptives, not
complete-method estimates. When divergence is nonzero, divergence is the
primary result.

## Exact Sanity Results

On the correctly specified block-aligned concave case at `alpha = 10`:

| Method | Divergence | Nondiverged gap AUC | Final gap ratio |
| --- | ---: | ---: | ---: |
| Explicit exact gradient | 100% | -- | boundary-capped |
| Full oracle implicit | 0% | 0.0177306 | `5.74434e-20` |
| Oracle block-approximation signed | 0% | 0.0177306 | `5.74434e-20` |
| Concave-projected block | 0% | 0.0177306 | `5.74434e-20` |
| Norm-matched isotropic | 0% | 0.0564451 | 0.000483159 |

This is an arithmetic and specification sanity check: when the Hessian is
exactly block isotropic and concave, full, diagonal-approximation,
block-approximation, and projected implicit updates coincide.

The dense rotation falsifies any claim that denominator safety alone is
sufficient. At `alpha = 10`:

| Method | Divergence | Nondiverged gap AUC | Final gap ratio |
| --- | ---: | ---: | ---: |
| Full oracle implicit | 0% | 0.0171193 | `2.42280e-20` |
| Oracle block-approximation signed | 100% | -- | boundary-capped |
| Concave-projected block | 100% | -- | boundary-capped |
| Norm-matched isotropic | 0% | 6,778.423016 | 75,946.253867 |

The concave-projected block method never amplified the explicit step, yet its
misspecified direction reached the gap-divergence boundary. The guarantee is
mechanical, not an objective-improvement theorem.

## Monte Carlo Results

The tables below report divergence first and then five-seed mean normalized
gap AUC among nondiverged runs at `alpha = 1`; lower AUC is better. These are
descriptive synthetic outcomes, not hypothesis tests. Survivor means must not
be compared as if they represented all runs when divergence is nonzero.

### Raw-Fitness ES

| Method | Block aligned: div; AUC | Rotated: div; AUC | Additive noise: div; AUC |
| --- | ---: | ---: | ---: |
| Explicit ES | 0%; 6,395.667240 | 0%; 784.946198 | 0%; 80.113338 |
| Full-Hessian oracle implicit | 0%; 0.0437730 | 0%; 0.0328164 | 0%; 0.0516833 |
| Sampled signed diagonal | 0%; 0.0841253 | 0%; 0.0646844 | 0%; 4,316.280626 |
| Sampled signed block | 0%; 0.0428994 | 0%; 0.0356793 | 0%; 0.0526996 |
| Concave-projected block | 0%; 0.0438817 | 0%; 0.0356794 | 0%; 0.0526825 |
| Norm-matched isotropic | 0%; 0.0482741 | 0%; 0.0385063 | 0%; 0.0591268 |

The sampled diagonal method reached mean maximum amplification factors of
`12.03`, `3.07`, and `36.20` in these three cells. Pooling was much less
variable here, but its signed version still lacks a no-amplification guarantee
and becomes unreliable in several `alpha = 10` cells.

### Same-Batch Rank ES

| Method | Block aligned: div; AUC | Rotated: div; AUC | Additive noise: div; AUC |
| --- | ---: | ---: | ---: |
| Explicit ES | 0%; 2.203543 | 0%; 1.208250 | 0%; 2.092407 |
| Full frozen-batch signed | 0%; 1,062.580840 | 20%; 663.371582 | 0%; 117,141.154861 |
| Sampled signed diagonal | 0%; 131.942503 | 0%; 8.524266 | 0%; 2,334.282573 |
| Sampled signed block | 0%; 0.330696 | 0%; 0.342042 | 0%; 22.928714 |
| Concave-projected block | 0%; 0.0414809 | 0%; 0.331107 | 0%; 0.0871637 |
| Norm-matched isotropic | 0%; 0.227235 | 0%; 0.294036 | 0%; 0.234472 |

The structured method is not uniformly better than its equal-norm scalar
control. In the rotated rank case at `alpha = 1`, isotropic attenuation has
the lower mean gap AUC (`0.294036` versus `0.331107`). The common-state
directional artifact also contains both positive and negative benefits across
cases and steps. This is why a structured-curvature claim requires the
norm-matched control rather than comparison with explicit ES alone.

## Equal-Norm Directional Result

The primary directional diagnostic uses the explicit trajectory as a common
reference state and compares structured versus isotropic true one-step
improvement at identical norm. At `alpha = 1`, every displayed reference path
completed all 30 updates with zero divergence. Median benefit below is
structured minus isotropic; positive favors the structured direction.

| Transform | Case | Reference divergence | Completion | Median benefit | Structured win fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw | Block aligned | 0% | 100% | -0.250986 | 22.0% |
| Raw | Rotated | 0% | 100% | 0.008256 | 56.7% |
| Raw | Additive noise | 0% | 100% | -0.085465 | 17.3% |
| Same-batch rank | Block aligned | 0% | 100% | -0.031855 | 39.3% |
| Same-batch rank | Rotated | 0% | 100% | 0.005082 | 56.0% |
| Same-batch rank | Additive noise | 0% | 100% | -0.032415 | 40.0% |

The sign changes across cases. In particular, better structured trajectory
AUC in some aligned cells cannot be summarized as uniformly better immediate
direction at equal norm. Large-step directional rows must additionally be read
with `reference_run_divergence_rate` and
`reference_horizon_completion_fraction`; the artifact does not report an
unqualified full-horizon mean when the explicit reference terminates early.

## What This Establishes

- The exact full implicit equation is implemented correctly and behaves as
  predicted on concave quadratics.
- Signed sampled systems can amplify severely even when their arithmetic solve
  is exact.
- Concave projection enforces the no-amplification invariant on every tested
  step.
- Block pooling can repair sampling variance when its structural assumption is
  appropriate, but rotation exposes consequential misspecification.
- Equal-norm directional controls can reverse the apparent conclusion about
  whether structured attenuation is useful.

## What This Does Not Establish

- It does not show that the same-batch rank statistic is a Hessian estimator.
- It does not establish superiority on Hopper or any other environment.
- It does not provide a convergence theorem for noisy or rank-based ES.
- It does not validate architecture-defined blocks in policy networks.
- It does not study EMA, reference-rank, or cross-fitted estimators.
- Five synthetic seeds and descriptive means are not confirmatory inference.

The main paper use is as a controlled mechanism figure and falsification
suite. Any optimizer-performance claim still requires strong ES baselines,
multiple real tasks, untouched seeds, and a separately frozen protocol.

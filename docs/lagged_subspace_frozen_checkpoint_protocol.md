# Lagged-Subspace Frozen-Checkpoint Diagnostic Protocol

Status: final preregistered protocol locked before environment outcomes.
Checkpoint, estimator, manifest, runner, and analyzer code exist, but no
scientific diagnostic outcome has been observed. Any later implementation
change requires a new source lock and protocol revision before execution.

## 1. Designation And Claim Boundary

This is a high-sample **mechanism diagnostic**, not an optimizer confirmation
and not a sample-efficiency experiment. It asks whether a low-rank curvature
action can be estimated reproducibly from 100 antithetic pairs, remains local
relative to the ES search scale, and changes one-step policy return beyond the
effect of taking an equal-norm step in the gradient direction.

The study cannot establish:

- that the method improves a training trajectory;
- that it is better than Standard ES, SNES, CMA-ES, or another optimizer;
- that its rank curvature is a raw-return Hessian;
- that an `m = 100` frozen-checkpoint result transfers to a training trajectory,
  a new task, or a different policy class;
- environment-transition sample efficiency; or
- a general claim outside the three MuJoCo tasks named below.

Any later optimizer pilot or confirmation requires a new protocol, untouched
seeds, and a separately locked source snapshot.

## 2. Fixed Scientific Questions

The protocol answers four ordered questions:

1. Does a lagged, search-relevant rank-three subspace contain a reproducible
   current-CDF stop-gradient curvature signal?
2. Do estimates formed from `m = 100` pairs reproduce the high-sample
   anisotropic action closely enough to be operationally distinguishable from
   estimator noise?
3. Is that anisotropic action material while the proposed update remains
   within one search-distribution noise scale?
4. At equal update norm, does the lagged-subspace direction improve paired
   one-step return relative to the explicit gradient direction?

The questions are evaluated in this order. A return difference cannot rescue
an unresolved estimator, an immaterial action, or a nonlocal step.

## 3. Tasks, Policy, And Standard-ES Checkpoint Runs

The fixed task set is:

- `Hopper-v5`;
- `Walker2d-v5`; and
- `HalfCheetah-v5`.

All tasks use the repository MLP policy with hidden widths `[64, 64]`, frozen
observation normalization after three calibration episodes, population 200,
100 exact antithetic pairs, `sigma = 0.02`, and centered-rank Standard ES.
Replay, cross-generation importance sampling, trust clipping, Picard
iteration, gradient clipping, parameter projection, curvature clipping, L2,
and optimizer momentum are disabled.

Checkpoint-generating runs use:

- training seeds `300` through `319`, paired by seed number across tasks;
- 250 updates with no early stopping;
- constant Standard-ES learning rate `1e-4`; and
- no online policy evaluation used for selection.

The common rate is fixed because it produced local Hopper updates in the prior
development screen and avoids outcome-based task-specific rate selection.
Checkpoint quality is not an endpoint of this study.

Index centers by `theta_0` before the first update. Generation `j` samples at
`theta_j`, records `g_j`, and moves to `theta_{j+1}`. For every task and seed,
retain `theta_50`, `theta_150`, and `theta_250`. All 180 resulting checkpoints
enter the diagnostic. A checkpoint is never
chosen because of its rollout return, best-so-far return, curvature estimate,
or apparent learning progress. Failed or low-return policies remain included
if their artifacts are technically valid.

For checkpoint `theta_t`, the lagged archive is exactly
`g_{t-10}, ..., g_{t-1}`; `g_{t-1}` is the gradient that moved
`theta_{t-1}` to `theta_t`. The checkpoint artifact must hash those ten indexed
gradients, the parameters, observation-normalization state, training config,
and source. Checkpoint selection is complete before any high-sample curvature
or endpoint return is observed.

## 4. Lagged And Random-Control Bases

Let `P_b` select the parameters of affine policy layer `b`, including its bias.
There are three disjoint blocks. At checkpoint generation `t`, define the
strictly lagged block gradient

```text
g_bar[t-1,b]
  = sum_{h=1}^{10} 0.9^(h-1) P_b g[t-h]
    / sum_{h=1}^{10} 0.9^(h-1).
```

The primary basis has one unit vector per layer:

```text
v[t,b] = g_bar[t-1,b] / ||g_bar[t-1,b]||,
V[t]   = [v[t,1], v[t,2], v[t,3]].
```

The disjoint supports make the columns orthonormal. If a lagged block gradient
is exactly zero, replace only that column by a deterministic Gaussian unit
vector generated from the predeclared basis seed stream. No tolerance other
than exact zero is used.

`V[t]` is computed and serialized before either current checkpoint population
is sampled. Current returns, perturbations, gradients, and curvature estimates
cannot alter it.

The lagged-random-subspace control applies one independently seeded signed
coordinate permutation within each block to `g_bar[t-1,b]` before
normalization. It preserves block support, vector norm, the number and
magnitudes of zero/nonzero entries, and lagged construction, but not the
original locations of sparse entries. It destroys the intended coordinate
alignment. If `g_bar[t-1,b]` is exactly zero, first generate the locked
deterministic Gaussian fallback used by the primary column and then apply the
independently seeded signed coordinate permutation to that fallback. This
exception is recorded explicitly; preserving the all-zero sparsity pattern is
impossible for a unit control column. The permutations are generated and
locked before current checkpoint sampling.

The machine-readable manifest fixes the exact NumPy `PCG64` draw convention.
The primary-basis generator advances only when an exact-zero block requires a
`standard_normal` fallback, in layer order. For every random-control block,
the independent generator draws `permutation(block_size)` first, then
`integers(0, 2, size=block_size, dtype=int64)`, maps those bits to `{-1,+1}`,
applies the signs to the permuted primary values, and normalizes the resulting
column. Producer and analyzer implementations must agree bit-for-bit for every
possible three-block exact-zero mask.

## 5. Independent High-Sample Banks

At each checkpoint collect two independent banks, `A` and `B`. Each bank has:

- `M = 2,000` iid antithetic pair clusters;
- 4,000 candidate-policy rollouts;
- full-dimensional `epsilon_k ~ N(0, I_d)` perturbations; and
- one rollout seed shared by the plus/minus members of a pair, with distinct
  keyed streams across pairs and banks.

Common randomness inside a pair is allowed by the LOPO U-statistic theory.
There is no shared rollout randomness across different pair clusters. Bank A
is the high-sample reference and determines the controlled locality rates.
Bank B is an independent replication bank and is partitioned, after a fixed
seeded permutation, into twenty disjoint populations of exactly `m = 100`
pairs. Ranks and LOPO utilities are recomputed independently within each
100-pair population.

Bank A and the basis are fixed before Bank B is revealed. No Bank-B partition
may be replaced, pooled selectively, or excluded based on its estimate or
endpoint return.

`M = 2,000` is fixed because it supplies twenty disjoint operational-size
populations and reduces the Monte Carlo standard error of a bank-wide mean to
approximately `1 / sqrt(20) = 22.4%` of an `m = 100` mean under ordinary
root-sample-size scaling. Two independent banks test whether that nominal
precision is realized instead of treating one large bank as truth. This
rationale is fixed before subspace outcomes are available.

## 6. Matched LOPO Gradient And Subspace Hessian

For pair `k`, sign `s`, and tie-aware comparison

```text
K(y,y') = 1{y > y'} - 1{y < y'},
```

define the leave-own-pair-out utility

```text
u[k,s]^(-k)
  = [1 / (4 (m - 1))]
    sum_{l != k} sum_{r in {+,-}} K(Y[k,s], Y[l,r]).
```

Tie comparisons contribute zero, and the utility is not recentered using the
target population.
The same utilities must be used for gradient and curvature:

```text
g_hat
  = [1 / (2 m sigma)] sum_k
      (u[k,+]^(-k) - u[k,-]^(-k)) epsilon_k,

z_k = V^T epsilon_k,

B_hat
  = [1 / (2 m sigma^2)] sum_k
      (u[k,+]^(-k) + u[k,-]^(-k)) (z_k z_k^T - I_3).
```

`B_hat` is symmetrized numerically. Conditional on the lagged basis,

```text
E[g_hat] = G_stop(theta),
E[B_hat] = V^T H_stop(theta) V,
```

where the current return mid-CDF is held fixed in the derivatives. These are
not raw-return derivatives and not total derivatives of a globally adaptive
rank objective.

The lagged-random control uses the identical equations with `V_random` and the
same population. It does not get an independent or more favorable batch.

## 7. U-Statistic Jackknife

For independent pair clusters `X_k` and `X_l`, define

```text
A[k,l] = sum_{s,r in {+,-}} K(Y[k,s], Y[l,r]),

h_H[k,l]
  = A[k,l]
    { (z_k z_k^T - I_3) - (z_l z_l^T - I_3) }
    / (16 sigma^2),

h_G[k,l]
  = [1 / (16 sigma)] sum_{s,r}
      K(Y[k,s],Y[l,r]) (s epsilon_k - r epsilon_l).
```

The LOPO estimates are order-two U-statistic averages of these kernels. For
each bank and each 100-pair partition, compute delete-one-pair estimates of
`g_hat`, `vech(B_hat)`, every reported eigenvalue, and the final structured
action. Full parameter-space covariance matrices need not be materialized:
record componentwise gradient variance, `vech(B_hat)` covariance, the trace of
action covariance, and covariance projected onto the high-sample anisotropic
action. The generic jackknife covariance is

```text
Cov_JK(T)
  = [(m - 1) / m] sum_k
      (T[-k] - mean_l T[-l]) (T[-k] - mean_l T[-l])^T.
```

Eigenvalue and action jackknives recompute the complete nonlinear quantity
after deletion; entrywise SEs are not propagated through a fixed eigensystem.
Near repeated eigenvalues, eigenvector-specific intervals are nonregular, so
the primary reliability object is the resulting action vector, not a named
eigenvector. Zero projected curvature, zero anisotropic action, and eigenvalues
at the concave-projection boundary are also nonregular and are reported as
unresolved rather than assigned a zero standard error.

The jackknife is componentwise/asymptotic under iid pair clusters. It is not a
confidence statement for a training trajectory, repeated model selection,
EMA, or nonlinear projection across generations. No EMA is used in this
frozen-checkpoint study.

## 8. Locality Grid And Four Endpoint Arms

For Bank A, define three controlled explicit-step locality levels

```text
q in {0.25, 0.50, 1.00},
alpha_q = q sigma / ||g_A||.
```

The levels are geometric halvings of the one-noise-scale boundary. `q = 1`
tests the boundary itself; `q = 0.5` is the primary stratum and leaves a
factor-of-two sampling margin; `q = 0.25` diagnoses whether conclusions persist
deeper in the local regime. This rate calibration is an experimental control,
not a proposed adaptive optimizer or a claim of step-size robustness.

Each `alpha_q` is computed once from the complete Bank-A gradient and then
frozen for the complete Bank-B estimate, every 100-pair partition, every arm,
and every delete-pair action recomputation. Bank-A jackknives therefore target
action uncertainty conditional on the locked rate; they do not claim coverage
for the Bank-A rate-calibration noise. Bank A and Bank B use the same locked
rate.

The Bank-A gradient norm is evaluated with a scaled finite L2 calculation,
`max(abs(g_A)) * ||g_A / max(abs(g_A))||`. Exact zero is the only unresolved
zero-gradient case; no small-gradient tolerance is used. If every component is
exactly zero, all three rates use the finite sentinel `alpha_q = 0`, every arm
step is exactly zero, and evaluation and artifact production continue. Every q
summary records `alpha_resolved = false` and
`alpha_unresolved_reason = bank_a_gradient_exact_zero`. The affected task fails
every gate condition. This is a retained scientific no-go record, not an
infrastructure failure or a reason to replace a checkpoint. A nonfinite norm or
rate remains an invalid artifact.

For a given population, eigendecompose

```text
B_hat = Q diag(lambda) Q^T,
C_hat = Q diag(max(-lambda, 0)) Q^T.
```

The four endpoint steps are:

```text
Delta_struct
  = alpha_q (I - V V^T) g_hat
    + alpha_q V (I_3 + alpha_q C_hat)^(-1) V^T g_hat,

Delta_iso
  = ||Delta_struct|| g_hat / ||g_hat||,

Delta_explicit
  = alpha_q g_hat,

Delta_random_raw
  = the same subspace equation using V_random,

Delta_random
  = ||Delta_struct|| Delta_random_raw / ||Delta_random_raw||.
```

Zero-vector cases are retained and reported; the corresponding normalized
direction metric is undefined rather than replaced by zero. If
`Delta_random_raw = 0` while `Delta_struct != 0`, that checkpoint-partition is
an invalid random-control action and the task fails the gate; it is not dropped
or replaced. If both are zero, both remain zero and the material-action gate is
unresolved. Concave projection
makes the structured subspace solve nonamplifying, but there is no trust radius,
step clipping, or parameter projection.

`Delta_iso` exactly matches the structured step norm while retaining the
gradient direction. `Delta_random` matches the same norm while controlling for
an arbitrary rank-three directional transformation. The explicit arm shows
the attenuation relative to ordinary ES but is not norm matched.

## 9. Common-Random Paired Endpoint Evaluation

For every Bank-B 100-pair partition and every `q`, evaluate

```text
theta + Delta_struct,
theta + Delta_iso,
theta + Delta_explicit,
theta + Delta_random
```

on exactly ten rollout seeds unused by checkpoint generation or either
curvature bank. The same ten seeds are used for all four arms. The checkpoint
center is evaluated once on the same seed bank and reused across the twenty
partitions and three locality levels.

The primary return contrast is structured minus equal-norm isotropic. Explicit
and lagged-random contrasts are secondary mechanism controls and cannot rescue
a failed primary contrast. Endpoint return is a one-step counterfactual at a
frozen policy, not a training-return or AUC endpoint.

## 10. Prespecified Metrics

### Locality

For every arm report:

- first, mean, median, 95th percentile, and maximum `||Delta|| / sigma`;
- fraction at or below `0.25`, `0.5`, and `1.0`;
- `alpha_q * max_eigenvalue(C_hat)`; and
- any numerical solve residual or nonfinite value.

### Curvature and action reliability

Let

```text
a_A = Delta_struct,A - Delta_iso,A,
a_B = Delta_struct,B - Delta_iso,B
```

be the anisotropic actions formed from the complete 2,000-pair Bank A and the
independent complete 2,000-pair Bank B, respectively. Both use the Bank-A
`alpha_q`. Report:

```text
D_material = ||a_A|| / ||Delta_struct,A||,

E_high
  = ||a_A - a_B||
    / max(0.5 (||a_A|| + ||a_B||), epsilon),

E_100
  = sqrt(mean_r ||a_100,r - a_A||^2)
    / max(||a_A||, epsilon).
```

Also report:

- cosine and relative error between each 100-pair structured action and the
  independent Bank-A action;
- Frobenius error of `B_100` relative to `B_A`;
- negative-eigenvalue count and sign agreement;
- jackknife SEs, action-covariance trace, and action-aligned variance;
- the angle between structured and explicit steps; and
- multiplier dispersion across the three subspace modes.

`epsilon` is machine-safe numerical protection only. Any metric whose true
denominator is numerically unresolved is classified as unresolved and fails
the gate.

### Frozen-batch endpoint linearization

Raw episodic return and the frozen-rank Gaussian score do not share one Taylor
objective, so raw endpoint change is not used as a Taylor remainder. Instead,
hold Bank-A candidates and LOPO utilities fixed. For each signed candidate
noise vector `epsilon_i`, define the unnormalized Gaussian importance ratio

```text
ell_i(Delta)
  = epsilon_i^T Delta / sigma - ||Delta||^2 / (2 sigma^2),

r_i(Delta) = exp(ell_i(Delta)).
```

The matching frozen-utility empirical objective and endpoint gradient are

```text
Phi_A(Delta)
  = [1 / (2M)] sum_i u_i r_i(Delta),

D_A(Delta)
  = [1 / (2M sigma)] sum_i
      u_i r_i(Delta) (epsilon_i - Delta / sigma).
```

This map is deliberately unnormalized: its derivatives match the LOPO score
moments exactly. At zero,

```text
D_A(0) = g_A,

J_A
  = derivative of D_A at zero
  = [1 / (2M sigma^2)] sum_i
      u_i (epsilon_i epsilon_i^T - I).
```

The full matrix need not be materialized. For every arm, compute `J_A Delta`
directly as a weighted sum of score-vector products and report

```text
R_full(Delta)
  = ||D_A(Delta) - D_A(0) - J_A Delta||
    / max(||D_A(Delta) - D_A(0)||, ||J_A Delta||, epsilon).
```

Also report the restricted-model residual with
`J_sub = V B_A V^T`:

```text
R_sub(Delta)
  = ||D_A(Delta) - D_A(0) - J_sub Delta||
    / max(||D_A(Delta) - D_A(0)||, ||J_sub Delta||, epsilon).
```

`R_full` isolates endpoint nonlinearity for the frozen empirical objective.
`R_sub` additionally includes rank-three model misspecification. For numerical
and overlap diagnostics, normalize `r_i` only after it has been defined and
report the normalized effective-sample-size ratio, coefficient of variation,
mean unnormalized ratio minus one, and

```text
max_i ell_i(Delta) - min_i ell_i(Delta).
```

These calculations reuse evaluated Bank-A candidates and add no policy
rollouts. Raw paired one-step return remains a separate outcome below and is
never called a Taylor remainder.

As a secondary diagnostic of the repository's self-normalized endpoint map,
also define

```text
a_i(Delta) = r_i(Delta) / sum_j r_j(Delta),
u_bar_a(Delta) = sum_i a_i(Delta) u_i,

D_SN,A(Delta)
  = [1 / sigma] sum_i a_i(Delta)
      (u_i - u_bar_a(Delta)) (epsilon_i - Delta / sigma).
```

Complete antithetic noise has zero empirical mean. Exact LOPO utilities also
have zero sample sum even though no recentering operation is applied. Indeed,

```text
sum_{k,s} u[k,s]^(-k)
  = [1 / (4 (m - 1))]
    sum_{k != l} sum_{s,r} K(Y[k,s], Y[l,r])
  = 0,
```

because every ordered cross-pair term is paired with
`K(Y[l,r],Y[k,s]) = -K(Y[k,s],Y[l,r])`; ties contribute zero in both orders.
Consequently, `D_SN,A(0) = D_A(0)`.

For reference, with arbitrary frozen utilities

```text
u_bar = mean_i u_i,
S_epsilon = mean_i epsilon_i epsilon_i^T,
```

direct differentiation at zero gives

```text
J_SN,A
  = [mean_i u_i epsilon_i epsilon_i^T - u_bar S_epsilon]
    / sigma^2,

J_A
  = [mean_i u_i epsilon_i epsilon_i^T - u_bar I]
    / sigma^2,

J_SN,A - J_A = u_bar (I - S_epsilon) / sigma^2.
```

This general identity is a negative-control check: antithetic zero mean alone
would not remove the difference for an arbitrary noncentered utility. For the
exact LOPO utilities in this protocol, `u_bar = 0` structurally, so
`J_SN,A = J_A` at `Delta = 0` in real arithmetic. Report the numerical utility
sum and Frobenius/action-relative Jacobian mismatch to verify that only
floating-point residue remains. Away from zero, self-normalization changes the
empirical map even though the at-origin Jacobians agree. Report

```text
R_SN(Delta)
  = ||D_SN,A(Delta) - D_SN,A(0) - J_SN,A Delta||
    / max(||D_SN,A(Delta) - D_SN,A(0)||,
          ||J_SN,A Delta||,
          epsilon).
```

`R_SN` is secondary and diagnoses the implemented endpoint-map geometry.
`R_full` remains primary because its frozen empirical objective has the LOPO
score Hessian exactly. The two residuals must not be pooled or described as
the same endpoint objective.

### Return metrics

Report the paired mean, median, interquartile mean, probability of improvement,
and a descriptive 95% percentile interval from 20,000 training-seed cluster
bootstrap resamples for every arm contrast. Use one locked analysis seed and
the same resampled seed indices across arms and tasks. These descriptive
intervals are not used by the go/no-go gate. Preserve the ten episode-level
paired differences; do not analyze unpaired arm means.

## 11. Go/No-Go Gate

Only `q = 0.5` is used for the gate. The other locality levels are fixed
sensitivity analyses and cannot rescue failure at `q = 0.5`.

A task passes only if all conditions below hold:

1. **Locality:** the simultaneous one-sided upper confidence bound for the
   task median of seed-level 95th percentiles of 100-pair structured
   `||Delta|| / sigma` is at most `1`.
2. **Material action:** the simultaneous one-sided lower confidence bound for
   the task median of seed-level `D_material` summaries is greater than
   `0.01`.
3. **High-sample replication:** the simultaneous upper confidence bound for
   the task median of seed-level `E_high` summaries is less than `0.25`.
4. **Operational `m = 100` reliability:** the simultaneous upper confidence
   bound for the task median of seed-level `E_100` summaries is less than
   `0.50`.
5. **Directional endpoint evidence:** the task-level structured-minus-isotropic
   seed mean is positive and its Holm-adjusted one-sided exact strict-positive
   binomial-test p-value is below
   `alpha_endpoint = 0.034539031982421875`.

The thresholds have the following fixed rationale:

| Threshold | Rationale |
| --- | --- |
| `||Delta|| / sigma <= 1` | One perturbation scale is the declared boundary of the local Gaussian neighborhood; no radius is imposed by the algorithm. |
| `D_material > 0.01` | One percent was fixed as the material-effect boundary in the July 12, 2026 read-only locality/effect audit of development job `49685417`, before this subspace design or any subspace result existed. Smaller directional changes are not a useful mechanism claim. |
| `E_high < 0.25` | The independent high-sample reference disagreement must consume at most half of the operational error allowance. |
| `E_100 < 0.50` | RMS action error below half the anisotropic signal is a signal-to-noise ratio above two; larger error cannot reliably distinguish structure from estimation noise. |
| adjusted `p < alpha_endpoint` | The endpoint-family level uses the remainder of the overall `0.05` false-advance budget after accounting for the exact twelve-bound mechanism family. |

The complete mechanism advances to a new multi-step optimizer pilot only if at
least two of the three tasks pass all five conditions. Requiring replication
on a majority of the fixed task set prevents a Hopper-only result from being
presented as a general locomotion mechanism. Any passing task may still be
reported diagnostically if the majority gate fails, but no optimizer pilot is
authorized by this protocol.

A no-go result is scientifically interpretable: it means either the subspace
signal is immaterial locally, the high-sample target is unresolved, `m = 100`
is inadequate, or equal-norm endpoint evidence does not support the direction.
It does not prove that every low-rank ES method is impossible.

## 12. Seeds, Clustering, And Multiplicity

Training seed is the independent top-level unit. Checkpoints, Bank-B
partitions, locality levels, and endpoint rollout seeds are repeated measures,
not additional training seeds.

At primary `q = 0.5`, reduce each task/seed to exactly four mechanism
statistics before cross-seed analysis:

```text
L_seed = 95th percentile of the 60 structured locality values
         (3 checkpoints * 20 partitions),

D_seed = median of the 3 checkpoint D_material values,

H_seed = median of the 3 checkpoint E_high values,

E_seed = median over checkpoints of
         sqrt(mean_{20 partitions} ||a_100 - a_A||^2) / ||a_A||.
```

The task estimands for conditions 1 through 4 are the medians of the twenty
corresponding seed-level statistics. No checkpoint or partition is directly
treated as an independent replicate.

Use finite-sample, distribution-free one-sided bounds for the twelve fixed
task-level seed medians. Let `X_(j)` denote the one-based `j`th order statistic
of the twenty seed-level values for one task metric. The material-action lower
bound is `X_(4)`, implemented as `sorted_values[3]`. The locality,
high-sample-replication, and operational-reliability upper bounds are `X_(17)`,
implemented as `sorted_values[16]`.

For any population median `m`, including a median of a discrete distribution
with atoms or ties,

```text
P{X_(4) > m}, P{X_(17) < m}
  <= delta
  = sum_{j=0}^3 choose(20,j) / 2^20
  = 1351 / 1048576
  = 0.0012884140014648438.
```

Bonferroni's inequality therefore gives simultaneous coverage over all twelve
one-sided bounds, under arbitrary dependence across tasks and metrics, of at
least

```text
1 - 12 delta = 0.9845390319824219.
```

The only sampling assumption is that the twenty training-seed clusters are
independent or exchangeable within each task. No bootstrap calibration,
distributional family, scale, copula, correlation grid, standard-error
estimate, or post-result interval-method choice is used. Constant finite seed
arrays remain valid. A nonfinite value, an unresolved scientific denominator,
or an incomplete seed set fails the affected gate rather than changing the
order-statistic rule.

For condition 5, average the paired endpoint contrast over the three fixed
checkpoints, twenty Bank-B partitions, and ten rollout seeds within each
training seed. This gives twenty paired seed-level differences per task.
Report their mean, but test the number of strictly positive differences using
the exact one-sided binomial sign test with null win probability `0.5`; ties
count as half a win only in the descriptive probability-of-improvement value
and count as failures in the conservative exact test. Apply Holm correction
across exactly the three task p-values at family level
`alpha_endpoint = 0.05 - 12(1351/1048576) = 0.034539031982421875`. Unlike a
sign-flip mean test, this test does not assume a symmetric distribution of
seed-level effect magnitudes. It tests whether the probability of a strictly
positive seed-level contrast exceeds one half; it is not a test of the
population mean. The additional positive observed seed mean is a descriptive
guard, not confidence evidence for a positive expected return.

Twenty seeds were fixed before outcome collection. For the exact sign gate,
sixteen wins give raw `p = 0.005908966064453125` and remain sufficient at the
worst-case first Holm threshold `alpha_endpoint / 3`; the chance of at least
sixteen wins is approximately `0.829847` when the true per-seed win probability
is `0.85`. Each mechanism order-statistic condition effectively requires at
least seventeen of twenty seeds on the favorable side. At favorable-seed
probability `0.85`, one such condition passes with probability approximately
`0.6477`, so the mechanism gate is deliberately conservative and may have low
power when several weakly correlated conditions are near their boundaries.
This is accepted as the price of a distribution-free no-selection gate; no
threshold or seed count may change after outcomes are generated.

The twelve mechanism bounds have familywise error at most
`12 delta = 0.015460968017578125`. The primary endpoint family has Holm level
`alpha_endpoint = 0.034539031982421875`. By a union bound, the complete
false-advance probability is therefore at most `0.05`, even when a different
component is false on each task. Every criterion must pass, and secondary
analyses cannot inflate or replace the primary decision. Explicit and
random-subspace return contrasts form one separate secondary family; if
p-values are reported, Holm-correct them across the six task-by-control
comparisons and label them secondary. All `q = 0.25` and `q = 1.0` analyses are
descriptive sensitivity results.

No new seed, checkpoint, partition, endpoint episode, or metric may be added
after any Bank-A, Bank-B, or endpoint result is inspected.

## 13. Fixed Rollout And Transition Budgets

The planned policy-rollout budget is:

| Stage | Calculation | Rollouts |
| --- | ---: | ---: |
| Standard-ES checkpoint generation | `3 tasks * 20 seeds * 250 updates * 200 candidates` | 3,000,000 |
| Observation-normalization calibration | `3 tasks * 20 seeds * 3 episodes` | 180 |
| Frozen Bank A and Bank B | `180 checkpoints * 2 banks * 2,000 pairs * 2 signs` | 1,440,000 |
| Four operational endpoint arms | `180 * 3 q-levels * 20 partitions * 4 arms * 10 episodes` | 432,000 |
| Checkpoint-center endpoint bank | `180 checkpoints * 10 episodes` | 1,800 |
| **Total** |  | **4,873,980** |

Checkpoint generation performs no online evaluation beyond the three
normalization episodes. If the eventual harness requires additional technical
evaluation episodes, their exact number must be added to the manifest and this
budget before submission; they cannot affect checkpoint selection.

The study fixes policy rollouts, not environment transitions. Record actual
transitions separately for checkpoint training, Bank A, Bank B, center
evaluation, and every endpoint arm. Unequal episode lengths cannot be used to
claim transition-level efficiency or to exclude a rollout.

## 14. Validation, Failures, And Locking

Before execution, create and hash a machine-readable manifest containing every
task, seed, checkpoint generation, bank, pair index, partition, locality level,
basis seed, endpoint seed, and arm. Freeze source, dependency environment,
protocol, analyzer, manifest, and launcher digests before any diagnostic
result is available.

Checkpoint archives use schema version 2 and embed scalar fixed-width ASCII
`study_source_sha256` and `training_config_sha256` values; the producer and
independent analyzer enforce both. Full bank perturbations are not duplicated
on disk: the raw bank stores their locked per-pair seeds, returns, transitions,
and the digest of the exact regenerated antithetic array. Likewise,
full-dimensional gradients, componentwise gradient variances, and endpoint
step vectors are represented by fixed-width content digests in diagnostic
archives. The analyzer independently regenerates these arrays from locked
seeds and returns and verifies every digest before aggregation.

Validation requires:

- exactly 180 parameter and observation-normalization checkpoint hashes;
- checkpoint generations exactly 50, 150, and 250;
- ten strictly prior gradients for every basis;
- two disjoint 2,000-pair banks per checkpoint;
- twenty complete, disjoint 100-pair Bank-B partitions;
- exact antithetic perturbations and intended within-pair rollout seeds;
- LOPO utilities shared by gradient and curvature;
- LOPO utility sum at most `1e-12 * max(1, sum_i |u_i|)` in absolute value;
- at-origin agreement of `D_SN,A` with `D_A` and of `J_SN,A` with `J_A` to
  `1e-10` relative error, together with the recorded floating-point residue;
- finite U-statistic, jackknife, eigensystem, step, and endpoint records;
- a resolved positive Bank-A rate or the exact-zero retained sentinel with its
  prescribed reason, four exact zero steps, and a failed affected-task gate;
- exact structured/isotropic and structured/random norm matching up to
  `1e-10` relative error;
- no replay, trust clipping, Picard iteration, gradient clipping, parameter
  projection, or EMA; and
- empty stderr or a documented infrastructure failure.

An infrastructure failure may be rerun only under the identical source,
manifest, task, seed, checkpoint, bank, and endpoint seed mapping. No policy,
seed, pair, partition, episode, or task may be excluded because of return,
curvature magnitude, locality, disagreement, or unfavorable direction.

The final analyzer must refuse aggregate output until every planned artifact
passes validation. Interim job progress may be monitored, but no estimator,
threshold, task, checkpoint, or endpoint decision may change in response.

## 15. Reporting And Exact Allowed Claims

All 180 checkpoints, both high-sample banks, all 3,600 operational 100-pair
partitions, and every endpoint rollout must be released or represented in the
validated artifact.

If the gate passes, the strongest allowed conclusion is:

> On at least two of three fixed MuJoCo tasks, a lagged rank-three LOPO
> subspace produced a local, reproducible `m = 100` anisotropic one-step action
> and outperformed its equal-norm gradient-direction endpoint control in this
> frozen-checkpoint diagnostic.

This still does not establish multi-step optimization benefit, raw-return
Hessian estimation, novelty over prior Hessian-aware zeroth-order methods, or
general RL performance.

If the gate fails, report the failed condition without substituting a locality
level, task subset, checkpoint subset, larger population, different basis,
different transform, or unadjusted return comparison. A revised estimator may
be studied only under a new diagnostic protocol.

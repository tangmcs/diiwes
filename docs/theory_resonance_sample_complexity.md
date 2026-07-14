# Concentration, Resonance, and Sample Complexity for Covariance-Score Curvature

## Scope

This note gives a finite-sample analysis of the diagonal and block
covariance-score estimators used in the project. The first result applies to
**independent antithetic pairs with a fixed bounded utility transform**. An
independent empirical reference CDF, conditioned on its reference sample, is
the main example.

Section 5 then uses the exact order-two U-statistic representation of
same-batch antithetic ranks to give a dependence-aware coordinate bound. The
raw pair contributions remain dependent and must not be inserted directly
into the independent-pair theorem. The note does not establish that either
covariance-score target is the Hessian of expected raw return. Estimator
semantics, the finite-population factor, and the matched LOPO gradient are
developed separately in
[`theory_rank_curvature_surrogate.md`](theory_rank_curvature_surrogate.md).

All constants denoted by `c`, `C`, or `C_i` below are positive universal
constants. Their values may change between displays. They do not depend on
dimension, pair count, perturbation scale, or confidence level.
The results are concentration upper bounds and sufficient sample-size
conditions, not minimax lower bounds.

## 1. Setup and assumptions

Let `epsilon_k ~ N(0, I_d)` independently for `k = 1, ..., m`. Pair `k`
evaluates the two perturbed points

```text
theta + sigma epsilon_k,    theta - sigma epsilon_k,
```

where `sigma > 0`. Let `Y_k,+` and `Y_k,-` denote their possibly stochastic
observations. Randomness may be coupled inside an antithetic pair, but pairs
must be independent across `k`.

Let `T` be a fixed pointwise utility transform and let `mu` be a fixed
centering constant. Define

```text
U_k,+ = T(Y_k,+) - mu,
U_k,- = T(Y_k,-) - mu,
S_k   = U_k,+ + U_k,-.
```

The assumptions are:

1. **Iid pairs.** The tuples `(epsilon_k, Y_k,+, Y_k,-)` are independent and
   identically distributed across `k`.
2. **Fixed transform.** `T` and `mu` are fixed independently of the `m` target
   pairs. They may come from a separate reference sample.
3. **Bounded utility.** There is a deterministic `U > 0` such that
   `|U_k,+| <= U` and `|U_k,-| <= U` almost surely. Thus `|S_k| <= 2U`.
4. **Fixed update quantities.** `sigma > 0` and the learning rate `alpha > 0`
   are fixed while the estimator is formed.

For a reference-CDF transform scaled to `[-1/2, 1/2]`, one may take `U = 1/2`
if the centering constant is zero. More generally, `U` must include the fixed
centering shift.

All expectations below are conditional on any independently generated
reference object. Conditioning on an independent empirical reference CDF
makes the transform deterministic, so the theorem applies conditionally and
therefore also unconditionally. Re-centering utilities by their mean on the
current target batch is different: that mean depends on every target pair and
is not covered by the theorem.

### Orlicz convention

For a random variable `X`, define

```text
||X||_psi1 = inf { a > 0 : E exp(|X| / a) <= 2 }.
```

This is the sub-exponential Orlicz norm. For `G = Z^2 - 1` with
`Z ~ N(0, 1)`,

```text
||G||_psi1 <= C_chi
```

for a universal `C_chi`. Equivalently, for every integer `p >= 2`,

```text
(E |G|^p)^(1/p) <= C p.
```

The bounded multiplier `S_k` may depend on `epsilon_k`; independence between
the utility and Gaussian score is not required for the coordinate bound.

## 2. Coordinate estimator

For coordinate `j`, define the pair contribution and its mean target by

```text
X_k,j = S_k (epsilon_k,j^2 - 1) / (2 sigma^2),
kappa_j = E[X_k,j].
```

The estimator is

```text
kappa_hat_j = (1 / m) sum_k X_k,j.
```

This is a covariance-score target induced by the fixed utility transform. It
need not equal raw-return curvature.

### Lemma 1: contribution tails

Under the assumptions above,

```text
||X_k,j||_psi1 <= C U / sigma^2,
||X_k,j - kappa_j||_psi1 <= C U / sigma^2.
```

#### Proof

The pointwise bound `|S_k| <= 2U` gives

```text
|X_k,j| <= (U / sigma^2) |epsilon_k,j^2 - 1|.
```

Monotonicity of the Orlicz norm and the Gaussian-square bound give the first
claim. Centering increases a `psi1` norm by at most another universal factor,
which gives the second claim. No independence between `S_k` and
`epsilon_k,j` is used. QED.

### Theorem 1: uniform coordinate concentration

Let `delta in (0, 1)` and define

```text
L_d(delta) = log(2 d / delta),
A_sigma    = U / sigma^2.
```

With probability at least `1 - delta`,

```text
max_j |kappa_hat_j - kappa_j|
  <= C A_sigma
       [ sqrt(L_d(delta) / m) + L_d(delta) / m ].        (1)
```

#### Proof

The centered variables `X_k,j - kappa_j` are independent across `k`, have
mean zero, and have `psi1` norm at most `C A_sigma`. Sub-exponential Bernstein
gives, for each fixed coordinate and every `t > 0`,

```text
P(|kappa_hat_j - kappa_j| >= t)
  <= 2 exp[-c m min(t^2 / A_sigma^2, t / A_sigma)].
```

Apply a union bound over the `d` coordinates and invert the tail probability.
QED.

The leading, moderate-deviation term is

```text
(U / sigma^2) sqrt(log(d / delta) / m).
```

This displays the two main difficulties directly:

- shrinking `sigma` by a factor `a` enlarges the absolute estimation error by
  a factor `a^2`; and
- controlling every coordinate costs `sqrt(log d)` beyond the usual
  `1 / sqrt(m)` Monte Carlo rate.

The linear `L_d / m` term matters for small samples or far tails. It should not
be silently removed when deriving a pair requirement.

### Corollary 1: coordinate pair requirement

For `eta > 0`, to guarantee

```text
max_j |kappa_hat_j - kappa_j| <= eta
```

with probability at least `1 - delta`, it is sufficient that

```text
m >= C L_d(delta) max {
       1,
       U^2 / (sigma^4 eta^2),
       U   / (sigma^2 eta)
     }.                                                   (2)
```

The population size is `2m`. In the variance-dominated regime, the leading
requirement is

```text
m = O(U^2 log(d / delta) / (sigma^4 eta^2)).
```

This is an absolute-error statement. Relative error also depends on the size
of `kappa_j`. A bounded rank transform controls contribution tails but may
make the target itself very small under strong observation noise. In that
case, a good absolute bound can still be useless for sign recovery.

## 3. Signed denominators and resonance

Consider the ideal signed coordinate update

```text
Delta_j = alpha g_j / D_hat_j,
D_hat_j = 1 - alpha kappa_hat_j.
```

Define the population denominator and its uniform margin by

```text
D_j     = 1 - alpha kappa_j,
gamma   = min_j |D_j|.
```

The resonance statements below assume `gamma > 0`. If `gamma = 0`, the
population signed system is already singular.

If scalar damping `lambda >= 0` is present, replace both denominators by
`1 + alpha lambda - alpha kappa`; the argument is unchanged.

### Proposition 2: denominator-margin condition

On any event where

```text
max_j |kappa_hat_j - kappa_j| <= eta,
```

we have

```text
max_j |D_hat_j - D_j| <= alpha eta.
```

Therefore, for any desired estimated margin `tau` satisfying
`0 < tau < gamma`, the condition

```text
alpha eta <= gamma - tau                              (3)
```

implies

```text
min_j |D_hat_j| >= tau.
```

It also preserves the sign of every denominator. The coordinate-wise
amplification relative to the explicit step is then at most `1 / tau`.

A convenient choice is `tau = gamma / 2`, which requires

```text
eta <= gamma / (2 alpha)                              (4)
```

and yields `min_j |D_hat_j| >= gamma / 2`.

### Corollary 2: pair requirement for avoiding signed resonance

Combining (2) and (4), a sufficient condition for a `gamma / 2` estimated
denominator margin with probability at least `1 - delta` is

```text
m >= C L_d(delta) max {
       1,
       alpha^2 U^2 / (sigma^4 gamma^2),
       alpha U     / (sigma^2 gamma)
     }.                                                   (5)
```

Factors of two from (4) are absorbed into `C`. More generally, to guarantee a
margin `tau`, replace `gamma` in (5) by `gamma - tau`.

Equation (5) separates two failure modes:

1. **Estimator-induced resonance.** `gamma` is moderate, but the uniform
   curvature error is too large relative to `gamma / alpha`.
2. **Structural resonance.** The population denominator itself is close to
   zero, so `gamma` is small. No feasible estimator accuracy can give a useful
   signed margin without an enormous pair count.

The direct arithmetic residual of the elementwise solve says nothing about
either margin.

For the concave-projected denominator

```text
1 + alpha [-kappa_hat_j]_+,
```

the denominator is at least one deterministically. Concentration is still
needed to argue that the attenuation is directionally meaningful, but not to
prevent resonance.

## 4. Block pooling

Let `B_1, ..., B_q` be a fixed partition of the coordinates. Write
`r_B = |B|` and `r_min = min_B r_B`. Define the normalized block score

```text
Q_k,B = (1 / r_B) sum_{j in B} (epsilon_k,j^2 - 1)
```

and the block contribution, target, and estimator

```text
X_k,B       = S_k Q_k,B / (2 sigma^2),
kappa_B     = E[X_k,B],
kappa_hat_B = (1 / m) sum_k X_k,B.
```

By linearity,

```text
kappa_B = (1 / r_B) sum_{j in B} kappa_j.              (6)
```

Thus block pooling targets the block-average diagonal covariance score. It
does not recover every coordinate without an additional block-isotropy
assumption.

### Lemma 2: block-score moments

For `p >= 2`, the centered chi-square average satisfies

```text
(E |Q_k,B|^p)^(1/p)
  <= C [ sqrt(p / r_B) + p / r_B ].                    (7)
```

One way to obtain (7) is from the exact moment-generating function

```text
E exp(lambda Q_k,B)
  = exp(-lambda) (1 - 2 lambda / r_B)^(-r_B / 2),
  lambda < r_B / 2.
```

Since `|S_k| <= 2U`, (7) implies the general Orlicz bound

```text
||X_k,B - kappa_B||_psi1
  <= C U / (sigma^2 sqrt(r_B)).                         (8)
```

The dependence of `S_k` on the perturbation does not invalidate the displayed
upper bounds: pointwise multiplication by a bounded variable cannot enlarge
the absolute moments beyond the corresponding `2U |Q_k,B|` moments. Centering
changes only universal constants.

Boundedness alone does **not** imply the sharper Bernstein absolute-moment
condition with scale `U / (sigma^2 r_B)`: the `p = 3` absolute moment still has
a variance-scale contribution of order `r_B^(-3/2)`. A sharper far-tail term
therefore requires an additional assumption, stated below.

### Theorem 3: uniform block concentration

Let

```text
L_q(delta) = log(2 q / delta).
```

With probability at least `1 - delta`,

```text
max_B |kappa_hat_B - kappa_B|
  <= C (U / sigma^2)
       [ sqrt(L_q(delta) / (m r_min))
         + L_q(delta) / (m sqrt(r_min)) ].              (9)
```

#### Proof

Apply sub-exponential Bernstein using (8) to each average of independent pair
contributions. For a fixed block,

```text
P(|kappa_hat_B - kappa_B| >= t)
  <= 2 exp[-c m min {
       t^2 sigma^4 r_B / U^2,
       t sigma^2 sqrt(r_B) / U
     }].
```

Union bound over the `q` blocks and use `r_B >= r_min`. QED.

Consequently, a sufficient pair requirement for block error at most `eta` is

```text
m >= C L_q(delta) max {
       1,
       U^2 / (sigma^4 r_min eta^2),
       U   / (sigma^2 sqrt(r_min) eta)
     }.                                                  (10)
```

The leading variance-regime term improves by `r_min`; the generic
sub-exponential far-tail term improves by `sqrt(r_min)`.

### Optional sharper block-tail assumption

Suppose, in addition, that each centered block contribution
`Z_k,B = X_k,B - kappa_B` satisfies the two-sided sub-gamma mgf bound

```text
log E exp(lambda Z_k,B)
  <= lambda^2 nu_B^2 / [2 (1 - b_B |lambda|)]
```

for `|lambda| < 1 / b_B`, with

```text
nu_B^2 <= C U^2 / (sigma^4 r_B),
b_B    <= C U   / (sigma^2 r_B).                        (8-sharp)
```

This assumption holds for the Gaussian block score `Q_k,B` itself, up to
scaling. It is **not** implied merely by allowing an arbitrary bounded,
score-dependent multiplier `S_k`; it must be justified for the utility model
being analyzed.

Under this additional assumption, (9) sharpens to

```text
max_B |kappa_hat_B - kappa_B|
  <= C (U / sigma^2)
       [ sqrt(L_q(delta) / (m r_min))
         + L_q(delta) / (m r_min) ],                    (9-sharp)
```

and the sufficient pair requirement becomes

```text
m r_min >= C L_q(delta) max {
               1,
               U^2 / (sigma^4 eta^2),
               U   / (sigma^2 eta)
             }.                                         (10-sharp)
```

For equal blocks of size `r = d / q`, the leading variance-regime pair count
is smaller than the coordinate-wise requirement by approximately

```text
r * log(2 d / delta) / log(2 q / delta).
```

The improvement comes from two sources:

1. averaging `r` Gaussian-square scores reduces the variance scale by `r`;
2. uniform control covers `q` block targets rather than `d` coordinates.

### Structural approximation error

If a block estimate is expanded to every coordinate in the block, define

```text
rho_B = max_{j in B} |kappa_j - kappa_B|,
rho   = max_B rho_B.
```

Then

```text
max_{B, j in B} |kappa_hat_B - kappa_j|
  <= max_B |kappa_hat_B - kappa_B| + rho.               (11)
```

The sampling advantage does not reduce `rho`. Exact block isotropy sets
`rho = 0`; otherwise pooling trades estimation variance for structural bias.

For coordinate population denominators with margin `gamma`, a block-expanded
signed update is protected at estimated margin `tau` only if

```text
alpha (eta + rho) <= gamma - tau.                       (12)
```

Thus the block pair requirement uses the positive error budget

```text
eta <= (gamma - tau) / alpha - rho.
```

If the right-hand side is nonpositive, increasing `m` cannot establish the
desired coordinate-level denominator guarantee. The block model itself is too
coarse.

### Corollary 3: block pair requirement for a denominator margin

Let `gamma > tau > 0` be the coordinate population-denominator margin and the
desired estimated margin. Define the remaining statistical error budget

```text
eta_star = (gamma - tau) / alpha - rho.
```

If `eta_star > 0`, a sufficient condition for the block-expanded denominators
to have margin at least `tau` with probability at least `1 - delta` is

```text
m >= C L_q(delta) max {
       1,
       U^2 / (sigma^4 r_min eta_star^2),
       U   / (sigma^2 sqrt(r_min) eta_star)
     }.                                                  (13)
```

Under the optional sub-gamma assumption `(8-sharp)`, condition (13) can be
replaced by `(10-sharp)` with `eta = eta_star`.

For a genuinely block-level target and update, define
`gamma_block = min_B |1 - alpha kappa_B|`, set `rho = 0`, and use
`eta_star_block = (gamma_block - tau) / alpha` in (13).

## 5. Same-Batch Rank U-Statistic Concentration

Same-batch ranks couple all apparent pair contributions, so applying Theorem 1
to those contributions as if they were independent is invalid. Antithetic
pairing nevertheless gives a sharper representation.

Let `K(y,y') = 1{y > y'} - 1{y < y'}`, with value zero on ties, and let

```text
A(X, X') = sum_s,t K(Y_s, Y'_t),
S(X)     = epsilon epsilon^T - I_d.
```

For `m >= 2`, define

```text
c_m = 2 (m - 1) / (2m - 1)
```

and the corrected coordinate statistic

```text
kappa_tilde_j = [J_D]_jj / c_m.
```

The exact rank algebra in the estimator-semantics note gives

```text
kappa_tilde_j
  = choose(m, 2)^(-1) sum_{k < l} h_j(X_k, X_l),

h_j(X, X')
  = A(X, X') (epsilon_j^2 - epsilon_j'^2) / (16 sigma^2).
```

This is an order-two U-statistic over iid pair clusters. It is identical to
the covariance-score statistic constructed from leave-one-pair-out rank
utilities. If `kappa_stop,j = E[h_j(X_1, X_2)]`, then

```text
E[kappa_tilde_j] = kappa_stop,j,
E[[J_D]_jj]      = c_m kappa_stop,j.
```

The target is a current-CDF stop-gradient transformed-objective curvature, not
raw-return curvature. Common rollout randomness inside one antithetic pair is
allowed; uncontrolled dependence across pairs is not. Ties are covered by the
definition of `K`.

### Theorem 4: uniform same-batch coordinate concentration

Let `r = floor(m / 2)`, `delta in (0,1)`, and

```text
L_d(delta) = log(2 d / delta).
```

There are universal constants `c,C > 0` such that

```text
P(|kappa_tilde_j - kappa_stop,j| >= t)
  <= 2 exp[-c r min(sigma^4 t^2, sigma^2 t)]            (14)
```

for each coordinate. Consequently, with probability at least `1 - delta`,

```text
max_j |kappa_tilde_j - kappa_stop,j|
  <= (C / sigma^2)
       [sqrt(L_d(delta) / r) + L_d(delta) / r].          (15)
```

#### Proof

The comparison sum satisfies `|A| <= 4`. Both `epsilon_j^2 - 1` and its
independent copy have universal sub-exponential norm, so

```text
||h_j(X, X') - E[h_j]||_psi1 <= C / sigma^2.
```

For any permutation of the `m` clusters, average the kernel over `r` disjoint
pairs. Those `r` kernel values are independent, so sub-exponential Bernstein
applies. The complete U-statistic is the average of these disjoint-pair
averages over permutations. Convexity of the exponential transfers the same
moment-generating-function bound to the complete U-statistic. This gives
(14). A union bound over `d` coordinates and inversion of the tail gives
(15). QED.

The corrected statistic therefore has the same leading scale

```text
sigma^(-2) sqrt(log(d / delta) / m)
```

as the fixed-transform bound. The dependence changes the proof and constants,
not that leading order. For the uncorrected production statistic,

```text
J_D - H_stop
  = c_m (kappa_tilde - H_stop) + (c_m - 1) H_stop.
```

Thus comparison with `H_stop` must include both stochastic error and the exact
finite-population bias. Comparing `J_D` with `c_m H_stop` removes only the
second term.

### Population-matched denominator warning

Dividing `J_D` by `c_m` alone does not make the current implicit system a
population-coherent approximation. The pooled-rank gradient obeys

```text
g_hat
  = c_m g_LOPO
    + [1 / (2m sigma (2m - 1))]
      sum_k epsilon_k K(Y_k,+, Y_k,-).
```

A population-targeted linearized system must use the matched LOPO gradient and
LOPO curvature together. The production optimizer instead leaves both raw
quantities unscaled because they are matched derivatives of its conditional
self-normalized endpoint map. Bound (15) diagnoses population curvature
estimation; it is not, by itself, a guarantee for the current complete update.

## 6. What Remains Outside The Theorems

### Current-batch centering and cross-fitting

Even with an independent reference CDF, subtracting the current target-batch
mean introduces dependence across pairs. Theorems 1 and 3 instead assume a
fixed centering constant. Theorem 4 handles the specific same-batch rank
dependence through its exact LOPO U-statistic identity; it does not justify
arbitrary data-dependent centering. Two reciprocal cross-fit folds are also
dependent because each fold supplies the other fold's reference distribution.
One-way sample splitting with a separately frozen reference may recover
conditional independence, but it changes the evaluation budget and estimand.

### Moving EMA targets

An EMA combines estimators formed at different policy centers. Their targets
can drift, their noise can be temporally dependent, and the bias correction
does not remove target drift. A subsequent concave projection is nonlinear and
does not preserve unbiasedness. The one-generation proofs above do not give a
confidence interval for a projected moving EMA. That requires a martingale or
mixing argument plus an explicit drift bound.

### Arbitrary rollout noise

Bounded fixed utilities make the marginal contribution tails manageable even
when raw returns are heavy-tailed. They do not repair arbitrary dependence or
nonstationarity. The theorem permits coupling inside one antithetic pair, but
not uncontrolled dependence across pairs, shared global simulator shocks,
or adaptive reruns. A transform fitted on the target population is covered
only when it has a separately proved structure, such as the exact LOPO
U-statistic identity above. Unbounded raw-return utilities require their own
moment or Orlicz assumptions.

### Iteration-wise reuse of the guarantee

The results are for one fixed center and one generation. A uniform statement
over `T` independently generated, fixed-target generations can replace
`delta` by `delta / T`, adding `log T`. Adaptive trajectories, shared random
states, refreshed current-CDF transforms, and moving targets require additional
conditioning and dependence arguments.

## 7. Connection to the controlled evidence

The bounds make several qualitative predictions that appear in the controlled
benchmarks:

1. **Perturbation-scale sensitivity.** In the raw centered-second-difference
   benchmark, reducing `sigma` from `0.1` to `0.02` increased independent-noise
   RMSE by roughly `24.5-24.9`, close to the predicted absolute-error factor
   `(0.1 / 0.02)^2 = 25`.
2. **Monte Carlo scaling.** Increasing population from 64 to 200 improved raw
   estimator RMSE by roughly `1.8-2.0`, consistent with the variance-regime
   prediction `sqrt(200 / 64) = 1.77` up to finite-sample and model effects.
3. **Pooling.** On noiseless rank-surrogate cells, the production-semantics
   block estimator had lower median relative RMSE and higher correlation than
   the diagonal estimator. This matches the direction predicted by (9), while
   not proving that real policy layers are block-isotropic.
4. **Low-signal small-sigma regime.** With controlled observation noise and
   `sigma = 0.02`, the high-sample rank target itself was often weak relative
   to its Monte Carlo uncertainty, and finite-batch split agreement was near
   zero. The absolute concentration scale alone cannot guarantee relative or
   sign accuracy when the target shrinks.

The raw benchmark in item 1 uses unbounded additive Gaussian observations, so
it is not an instance of the bounded-utility theorem as stated. Gaussian
noise supports a related Orlicz calculation with additional noise-scale
parameters; the comparison above is only a check of the shared `sigma^-2`
scaling. The corrected same-batch rank cells are covered qualitatively by the
U-statistic structure in Theorem 4, but the displayed unknown constants do not
turn their descriptive errors into calibrated confidence intervals.

A scale-only substitution illustrates why the coordinate problem can be
severe. With centered-reference utilities bounded by `U = 1/2`, `sigma =
0.02`, `m = 100`, `d = 5123`, and `delta = 0.05`, the leading term in (1),
before its unknown universal constant, is approximately

```text
(0.5 / 0.02^2) sqrt(log(2 * 5123 / 0.05) / 100)
  approximately 437.
```

For three blocks and a smallest block size of 195, the analogous leading block
scale from (9) is approximately

```text
(0.5 / 0.02^2) sqrt(log(2 * 3 / 0.05) / (100 * 195))
  approximately 20.
```

These are not calibrated confidence radii: the constants are unspecified, and
the production Hopper method adds block projection and a moving EMA to the raw
same-batch statistic. The substitution only illustrates the scaling pressure
and the potential variance reduction from pooling.

The controlled results are therefore consistent with the idealized theory,
but they do not prove that the theorem explains Hopper. In particular, the
theorem does not convert the failed Hopper confirmation into a positive method
claim.

## 8. Practical implications

The analysis suggests the following auditable checks before a signed
curvature update is trusted:

1. report the empirical denominator margin, not only the linear-solve residual;
2. estimate whether the target magnitude exceeds its uncertainty at the
   intended `sigma`;
3. use independent rank splits for reliability diagnostics;
4. treat block pooling as a bias-variance assumption and test wrong-block or
   shuffled-block controls;
5. separate deterministic no-resonance projection from claims of curvature
   accuracy; and
6. use the U-statistic or jackknife variance for same-batch ranks, and do not
   transfer that confidence label to a nonlinear projection or moving EMA.

The main theoretical conclusion is narrow: under iid pair clusters, both the
fixed-transform estimator and the corrected same-batch LOPO U-statistic have
leading coordinate scale `sigma^-2 sqrt(log d / m)`. A bound certifies
avoidance of signed resonance only when the resulting error radius is smaller
than the population denominator margin divided by the learning rate. Block
pooling replaces coordinate complexity by block complexity and gains an
effective factor of block size in the leading variance term, at the cost of an
explicit structural approximation error. Its generic far-tail improvement is
only a factor of `sqrt(r_min)` unless the sharper sub-gamma assumption is
justified. None of these one-generation results covers the projected moving
EMA used by the stabilized optimizer.

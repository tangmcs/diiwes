# Theory Note: Rank-Based Curvature Surrogates In Evolution Strategies

## Scope And Claim Boundary

This note formalizes the fresh-population curvature calculations used by the
repository. It separates three objects that must not be conflated:

1. the Hessian of a Gaussian-smoothed **fixed raw-fitness function**;
2. the Jacobian of a **finite-batch endpoint-gradient map with frozen
   utilities**; and
3. the Gaussian covariance-score moment formed from **same-batch centered
   ranks**.

The raw-fitness object has a standard unbiased Stein estimator. The
same-batch rank statistic is exactly a diagonal Jacobian of the repository's
batch-dependent endpoint map after the ranks have been observed and frozen.
Under iid antithetic pair clusters, its expectation is exactly `c_m` times a
current-CDF stop-gradient transformed-objective Hessian, and division by
`c_m` gives the unbiased LOPO U-statistic. Neither result makes it a Hessian of
raw expected return or the total Hessian of a global adaptive rank objective.

The paper-facing term for the implemented statistic is **frozen-rank
covariance-score curvature surrogate**. Historical identifiers containing
`hessian` are implementation names, not an estimand claim.

The analysis below excludes replay, importance sampling across generations,
Picard iteration, trust clipping, and fixed-norm updates. Those mechanisms are
not needed for any result in this note.

## 1. Notation

Let:

- `d` be the number of policy parameters;
- `theta` be the current search-distribution mean in `R^d`;
- `sigma > 0` be the fixed isotropic perturbation scale;
- `Z ~ N(0, I_d)`;
- `W = theta + sigma Z` be a sampled policy parameter vector; and
- `Y(w, xi)` be a possibly noisy rollout return, with rollout randomness
  `xi`.

Define the expected raw return

```text
f(w) = E_xi[Y(w, xi)]
```

and its Gaussian smoothing

```text
F_sigma(theta) = E_Z[f(theta + sigma Z)].
```

For antithetic sampling, draw `epsilon_k ~ N(0, I_d)` independently for
`k = 1, ..., m` and evaluate

```text
W_k,+ = theta + sigma epsilon_k,
W_k,- = theta - sigma epsilon_k.
```

The population size is `n = 2m`. A subscript `j` denotes a coordinate. For a
parameter block `B`, `|B|` denotes its number of coordinates.

All expectations and derivatives below require the displayed quantities to
exist. A sufficient working condition for the score identities is local
integrability of

```text
|Y(theta + sigma Z, xi)| (1 + ||Z||_2^2).
```

Bounded rank transforms automatically satisfy the corresponding moment
condition.

## 2. The Fixed Raw-Fitness Stein Identity

### Proposition 1: Gaussian-smoothed gradient and Hessian

Suppose `f` is measurable and differentiation may be exchanged with the
Gaussian convolution. Then `F_sigma` is smooth and

```text
grad F_sigma(theta)
  = (1 / sigma) E[f(theta + sigma Z) Z],

H_sigma(theta) := Hessian F_sigma(theta)
  = (1 / sigma^2)
    E[f(theta + sigma Z) (Z Z^T - I_d)].
```

Consequently,

```text
[H_sigma(theta)]_jj
  = (1 / sigma^2)
    E[f(theta + sigma Z) (Z_j^2 - 1)].
```

#### Proof

Write the smoothing as a convolution with the Gaussian density
`p_sigma(w - theta)`. Direct differentiation of that density gives

```text
partial_theta_j p_sigma(w - theta)
  = ((w_j - theta_j) / sigma^2) p_sigma(w - theta),

partial_theta_j partial_theta_l p_sigma(w - theta)
  = (((w_j - theta_j)(w_l - theta_l) / sigma^4)
     - 1{j=l} / sigma^2) p_sigma(w - theta).
```

Substituting `w = theta + sigma z` gives the two identities. This is the
standard second-order Gaussian Stein identity. A primary reference is
[Zhu (2022)](https://proceedings.mlr.press/v145/zhu22c.html). QED.

### Stochastic rollout returns

The same identities hold with `Y(theta + sigma Z, xi)` in place of `f` after
taking expectation over `xi`. The target is then the Gaussian smoothing of
`E_xi[Y(w, xi)]`. It is enough that each rollout has the correct marginal
conditional mean. Common random numbers may change variance and dependence
between the `+` and `-` evaluations, but do not change the expectation when
their marginals remain correct.

If a fixed pointwise transform `T` is applied to the noisy return, the identity
instead targets

```text
F_sigma,T(theta)
  = E_Z,xi[T(Y(theta + sigma Z, xi))].
```

This is generally not `T(F_sigma(theta))` and not the raw-return objective.

## 3. Antithetic Pair Estimation

### Proposition 2: Unbiased raw-fitness antithetic estimator

Under Proposition 1, define

```text
H_hat_raw
  = (1 / (2 m sigma^2)) sum_{k=1}^m
      (Y_k,+ + Y_k,-) (epsilon_k epsilon_k^T - I_d).
```

Then

```text
E[H_hat_raw] = H_sigma(theta).
```

In particular,

```text
[H_hat_raw]_jj
  = (1 / (2 m sigma^2)) sum_{k=1}^m
      (Y_k,+ + Y_k,-) (epsilon_k,j^2 - 1)
```

is unbiased for `[H_sigma(theta)]_jj`.

#### Proof

The distribution of `epsilon_k` is invariant under sign reversal, and
`epsilon_k epsilon_k^T - I_d` is even. Therefore the expectations of the `+`
and `-` terms are equal, and each equals the Stein expectation in Proposition
1. Averaging independent pairs preserves that expectation. QED.

Antithetic sampling can reduce variance by canceling odd local terms, but the
variance reduction is problem-dependent. It does not create an unbiasedness
property for a data-dependent fitness transform.

### Baselines

A scalar baseline `b_k` may be subtracted from the pair sum without changing
the expectation if `b_k` is independent of the target pair's perturbation:

```text
E[b_k (epsilon_k epsilon_k^T - I_d)] = 0.
```

A current-pair or current-batch baseline is not automatically independent.
For Hessian-score estimation, subtracting a sample mean computed from the same
target pairs can introduce bias because the finite-sample average of
`epsilon_j^2 - 1` is not identically zero. An independent reference baseline
or a leave-one-pair-out baseline is the safer construction.

The scaling must be stated. If `s_k = Y_k,+ + Y_k,-` and
`b_k = (m - 1)^(-1) sum_{l != k} s_l`, then

```text
s_k - b_k = [m / (m - 1)] (s_k - s_bar).
```

The leave-one-pair-out baseline is unbiased because it excludes the target
pair; the globally sample-centered version is smaller by `(m - 1) / m` in
expectation. For already centered same-batch rank pair sums `p_k = U_k,+ +
U_k,-`, `sum_k p_k = 0`, so applying the same old baseline would simply
multiply every `p_k` by `m / (m - 1)`. That is not the population correction
`1 / c_m = (2m - 1) / (2m - 2)` derived below, and it would change the exact
conditional-Jacobian scale.

## 4. The Implemented Same-Batch Rank Statistic

Let `z_1, ..., z_n` denote the complete antithetic population, so every
`epsilon_k` appears once with each sign and

```text
(1 / n) sum_i z_i = 0.
```

Let `U_i` be the tie-aware centered rank of the observed return `Y_i` among
all `n` returns. The implementation scales ranks into `[-1/2, 1/2]`. Thus,
for every realized batch,

```text
sum_i U_i = 0.
```

After computing the ranks, the update holds them fixed. Its proposal-gradient
estimate is

```text
g_hat = (1 / (n sigma)) sum_i U_i z_i
      = (1 / (2 m sigma)) sum_k (U_k,+ - U_k,-) epsilon_k.
```

The coordinate statistic used in the signed linearization is

```text
kappa_hat_j
  = (1 / (n sigma^2)) sum_i U_i (z_i,j^2 - 1)
  = (1 / (2 m sigma^2)) sum_k
      (U_k,+ + U_k,-) (epsilon_k,j^2 - 1).
```

The second equality uses antithetic pairing. Because `sum_i U_i = 0`, the
`-1` has no numerical effect in the first expression, but it exposes the
Gaussian covariance-score form.

### Proposition 3: Exact conditional finite-batch interpretation

Fix one realized antithetic population `D = {(z_i, U_i)}_{i=1}^n`; in
particular, do not recompute ranks as the candidate endpoint changes. Define
self-normalized endpoint weights

```text
q_i(delta)
  = exp(z_i^T delta / sigma)
    / sum_l exp(z_l^T delta / sigma)
```

and the weighted utility mean

```text
U_bar_q(delta) = sum_i q_i(delta) U_i.
```

The repository's empirical endpoint data-gradient map is

```text
G_D(delta)
  = (1 / sigma) sum_i q_i(delta)
      (U_i - U_bar_q(delta)) (z_i - delta / sigma).
```

Conditional on `D`,

```text
G_D(0) = g_hat
```

and

```text
diag(J_D) = kappa_hat,
J_D := partial G_D(delta) / partial delta^T evaluated at delta = 0.
```

The full conditional Jacobian is

```text
J_D = (1 / (n sigma^2)) sum_i U_i z_i z_i^T
    = (1 / (n sigma^2)) sum_i
        U_i (z_i z_i^T - I_d).
```

#### Proof

The term involving `-delta / sigma` cancels because

```text
sum_i q_i(delta) (U_i - U_bar_q(delta)) = 0.
```

Equivalently, `G_D` is the gradient of the finite-batch softmax interpolation

```text
Phi_D(delta) = sum_i q_i(delta) U_i.
```

At `delta = 0`, all `q_i = 1/n`. Antithetic sampling gives zero empirical
mean noise, and centered ranks give zero empirical mean utility.
Differentiating the softmax weights then yields the displayed `J_D`. The two
matrix forms agree because `sum_i U_i = 0`. QED.

### What Proposition 3 does and does not say

Proposition 3 is an exact algebraic statement **conditional on one realized
batch and frozen utilities**. It shows that the signed code is a linearization
of its own finite-batch, self-normalized endpoint map. Equivalently, `J_D` is
the second derivative at zero of a batch-specific softmax interpolation.

The next subsection gives a separate unconditional interpretation under iid
pair assumptions. Neither interpretation identifies `J_D` with the Hessian of
Gaussian-smoothed raw return or with the total Hessian of a global adaptive
rank objective.

## 5. Exact Population Target Of Same-Batch Antithetic Ranks

### Assumptions and tie convention

Let the pair clusters

```text
X_k = (epsilon_k, Y_k,+, Y_k,-),    k = 1, ..., m,
```

be iid across `k`, with `epsilon_k ~ N(0, I_d)`. The two returns inside one
pair may be arbitrarily dependent. In particular, the plus and minus rollouts
may use a common rollout seed. Each sign must nevertheless have the intended
marginal rollout law, and different pair-level rollout streams must be
independent. The implementation's `common_rollout_seed` convention uses one
seed within a pair and distinct keyed seeds across pairs, which matches this
model when those keyed streams are treated as iid draws. Conditional on one
fixed deterministic seed bank, literal identical distribution is an
additional modeling assumption.

Define the antisymmetric, tie-aware comparison

```text
K(y, y') = 1{y > y'} - 1{y < y'}.
```

It is zero on a tie. Let `F_theta^mid` be the mid-CDF of the current pooled
candidate-return distribution and define

```text
T_theta(y) = F_theta^mid(y) - 1/2.
```

For the exact tie-aware centered ranks used by the code,

```text
U_k,s
  = [1 / (2 (2m - 1))]
    sum_{(l,t) != (k,s)} K(Y_k,s, Y_l,t).
```

This representation covers both continuous returns and atoms.
The constants below are specific to this centered-linear rank convention,
whose empirical CDF denominator is `2m - 1`; they do not transfer unchanged to
an arbitrary rank-shaping utility.

Define the local frozen-transform objective

```text
L_theta(v)
  = E_Z,xi[T_theta(Y(v + sigma Z, xi))],
```

where `T_theta` is held fixed while differentiating `v`. Its score gradient
and Hessian at the current center are

```text
G_stop(theta)
  = (1 / sigma) E[T_theta(Y(theta + sigma Z, xi)) Z],

H_stop(theta)
  = (1 / sigma^2) E[T_theta(Y(theta + sigma Z, xi))
                           (Z Z^T - I_d)].
```

### Proposition 3a: finite-population curvature factor

Let

```text
c_m = 2 (m - 1) / (2m - 1)
    = 1 - 1 / (2m - 1).
```

Then

```text
E[J_D] = c_m H_stop(theta).
```

#### Proof

For a focal sign, separate the cross-pair and mate comparisons:

```text
C_k,s = sum_{l != k} sum_t K(Y_k,s, Y_l,t),
M_k,s = K(Y_k,s, Y_k,-s).
```

The definitions give

```text
U_k,s      = [C_k,s + M_k,s] / [2 (2m - 1)],
U_k,s^(-k) = C_k,s / [4 (m - 1)],

U_k,s = c_m U_k,s^(-k) + M_k,s / [2 (2m - 1)].
```

Antisymmetry gives `M_k,+ + M_k,- = 0`, so the pair-sum relation

```text
U_k,+ + U_k,-
  = c_m [U_k,+^(-k) + U_k,-^(-k)]
```

holds samplewise, including ties. Conditional on `X_k`, one return from the
pooled marginal satisfies

```text
E[K(Y_k,s, Y) | X_k] = 2 T_theta(Y_k,s).
```

Each of the `m - 1` other pairs contributes two such marginals. Hence

```text
E[U_k,s^(-k) | X_k] = T_theta(Y_k,s),

E[U_k,+ + U_k,- | X_k]
  = c_m [T_theta(Y_k,+) + T_theta(Y_k,-)].
```

Finally,

```text
J_D = [1 / (2m sigma^2)] sum_k (U_k,+ + U_k,-) S_k.
```

The plus and minus candidates have the intended Gaussian marginal laws, so
the antithetic score identity gives

```text
E[(T_theta(Y_+) + T_theta(Y_-)) S] / (2 sigma^2)
  = H_stop(theta).
```

Substitution proves `E[J_D] = c_m H_stop(theta)`. Common random numbers within
a pair do not alter the samplewise mate cancellation, but the stated
cross-pair independence and marginal-law assumptions remain necessary. QED.

For population `n = 2m`, `c_m = (n - 2) / (n - 1)`. It is `198 / 199` at the
population 200 used in Hopper. This finite-population factor is too small to
explain the observed instability.

### Leave-one-pair-out utilities and exact U-statistics

For each focal pair define the leave-one-pair-out (LOPO) utility

```text
U_k,s^(-k)
  = [1 / (4 (m - 1))]
    sum_{l != k} sum_t K(Y_k,s, Y_l,t).
```

This is the empirical mid-CDF score of the focal return using only the other
pairs as its reference. It is not recentered using the target batch. The
samplewise relation to the pooled centered rank is

```text
U_k,s
  = c_m U_k,s^(-k)
    + K(Y_k,s, Y_k,-s) / [2 (2m - 1)].                 (A)
```

Define

```text
g_LOPO
  = (1 / (2m sigma)) sum_k
      (U_k,+^(-k) - U_k,-^(-k)) epsilon_k,

J_LOPO
  = (1 / (2m sigma^2)) sum_k
      (U_k,+^(-k) + U_k,-^(-k)) S_k.
```

Equation (A) gives the exact batch identities

```text
J_D = c_m J_LOPO,

g_hat
  = c_m g_LOPO
    + [1 / (2m sigma (2m - 1))]
      sum_k epsilon_k K(Y_k,+, Y_k,-).                (B)
```

Thus rescaling curvature alone as `J_D / c_m` does **not** make the current
finite-`m` implicit system population matched. The pooled-rank gradient still
contains the within-pair comparison in (B). The existing pooled-rank optimizer
remains unchanged because its unscaled `g_hat` and `J_D` are exactly matched to
its conditional self-normalized endpoint map. The separate locked diagnostic
optimizer `concave_block_lopo_u_stat` now uses matched `g_LOPO` and `J_LOPO`;
its implementation and restricted claim boundary are documented in
[`lopo_u_stat_curvature.md`](lopo_u_stat_curvature.md).

### Proposition 3b: order-two U-statistic representation

For two independent pair clusters `X` and `X'`, write

```text
A(X, X') = sum_s,t K(Y_s, Y'_t),
S(X)     = epsilon epsilon^T - I_d,
z_s      = s epsilon.
```

Define the symmetric kernels

```text
h_H(X, X')
  = A(X, X') [S(X) - S(X')] / (16 sigma^2),

h_G(X, X')
  = [1 / (16 sigma)] sum_s,t
      K(Y_s, Y'_t) (z_s - z'_t).
```

For every realized batch with `m >= 2`, including tied returns,

```text
J_LOPO = J_D / c_m
       = choose(m, 2)^(-1) sum_{k < l} h_H(X_k, X_l),

g_LOPO = choose(m, 2)^(-1) sum_{k < l} h_G(X_k, X_l).
```

Moreover,

```text
E[g_LOPO] = G_stop(theta),
E[J_LOPO] = H_stop(theta).
```

#### Proof

Substituting `U_k,s^(-k)` into `g_LOPO` produces ordered cross-pair terms with
coefficient `1 / [8m(m-1)sigma]`. For one unordered pair `{k,l}`, group the
`(k,l)` and `(l,k)` terms. Antisymmetry,
`K(Y_l,t,Y_k,s) = -K(Y_k,s,Y_l,t)`, turns their sum into

```text
sum_s,t K(Y_k,s, Y_l,t) (z_k,s - z_l,t).
```

Because

```text
choose(m, 2)^(-1) / (16 sigma)
  = 1 / [8m(m-1)sigma],
```

the unordered-pair average is exactly the displayed `h_G` U-statistic.

For curvature, the same substitution gives ordered terms proportional to
`K(Y_k,s,Y_l,t) S(X_k)`. Grouping both orders yields

```text
A(X_k, X_l) [S(X_k) - S(X_l)].
```

The identical coefficient calculation with `sigma^2` gives the displayed
`h_H` U-statistic. This proves the two samplewise identities, not merely their
expectations. Conditional on one focal pair, the other pair's two return
marginals give `E[U_k,s^(-k) | X_k] = T_theta(Y_k,s)`. The Gaussian score
identities then give `E[g_LOPO] = G_stop(theta)` and
`E[J_LOPO] = H_stop(theta)`. QED.

Because these are ordinary order-two U-statistics over iid pair clusters,
finite second moments imply strong consistency. For a fixed finite
vectorization dimension and a nondegenerate first-order projection, either
vectorized kernel `h` obeys

```text
sqrt(m) (U_m(h) - E[h])
  => N(0, 4 Var(E[h(X_1, X_2) | X_1])).
```

The bounded comparison kernel and Gaussian quadratic score give all required
moments. This supplies a dependence-aware route to same-batch rank standard
errors and concentration; treating the original pair contributions as iid is
still invalid.

### Proposition 3c: expected pooled-rank gradient

Taking expectation in (B) gives

```text
E[g_hat]
  = c_m G_stop(theta) + b_pair / (2m - 1),

b_pair
  = (1 / (2 sigma)) E[epsilon K(Y_+, Y_-)].
```

The term `b_pair` depends on the within-pair joint return law and can therefore
change under common random numbers. It is generally nonzero. It vanishes at
rate `1 / m`, but at finite population it prevents `E[J_D]` from being the
Hessian corresponding to the complete expected pooled-rank gradient.

Under dominated differentiation, `E[J_D]` is still literally the derivative
with respect to endpoint displacement of the expected **frozen-batch endpoint
map**. It is not generally `d E[g_hat(theta)] / d theta` when a new batch is
sampled and reranked as `theta` changes.

## 6. Stop-Gradient And Global-Objective Boundaries

The transform `T_theta` depends on the current search distribution. The score
identities above differentiate the first argument of

```text
L(v, theta) = E[T_theta(Y(v + sigma Z, xi))]
```

while holding the second argument fixed. They omit derivatives of the CDF as
the search center changes. Indeed, the mid-CDF symmetry identity gives

```text
L(theta, theta) = E[T_theta(Y_theta)] = 0
```

for every `theta`. The total adaptive objective along the diagonal is
identically zero, even though its stop-gradient vector field can be nonzero.
Thus neither `G_stop` nor `H_stop` is generally the gradient or Hessian of one
global scalar objective refreshed with current ranks. This is the IGO-style
adaptive-flow interpretation, not ordinary optimization of rank values.

### Corollary: Rank invariance rules out a general raw-Hessian identity

Let `c > 0` and replace every realized return by `c Y`. Centered ranks, and
therefore `kappa_hat`, are unchanged sample by sample. In contrast, the raw
smoothed objective and Hessian satisfy

```text
F_sigma,c(theta) = c F_sigma(theta),
H_sigma,c(theta) = c H_sigma(theta).
```

Whenever `H_sigma(theta) != 0` and `c != 1`, one unchanged rank statistic
cannot be unbiased for both raw Hessians. Thus no general equality between the
same-batch rank surrogate and the raw-return Hessian is possible. This is an
invariance argument, independent of sample size or numerical conditioning.

Accordingly, the exact expectation derived above is a current-CDF
**stop-gradient transformed-objective Hessian**, up to `c_m`. It is not the
raw-return Hessian.

This adaptive-transform viewpoint is consistent with
[Information-Geometric Optimization](https://www.jmlr.org/papers/v18/14-467.html),
which explicitly formulates quantile weighting as an adaptive, time-dependent
transformation used to define a search-distribution flow.

## 7. The Signed Linearized System

Let:

- `lambda_2 >= 0` be the L2 coefficient;
- `gamma >= 0` be the additional implicit damping coefficient; and
- `lambda = lambda_2 + gamma`.

For one frozen batch, the intended endpoint fixed-point equation is

```text
delta
  = alpha [G_D(delta)
           - lambda_2 (theta + delta)
           - gamma delta].
```

Define the current total gradient

```text
g_t = G_D(0) - lambda_2 theta.
```

The first-order expansion

```text
G_D(delta) approximately G_D(0) + J_D delta
```

gives the full signed linearized system

```text
[I_d + alpha lambda I_d - alpha J_D] delta = alpha g_t.
```

The implementation replaces `J_D` by its diagonal surrogate `kappa_hat`, so

```text
D_j delta_j = alpha g_t,j,
D_j = 1 + alpha lambda - alpha kappa_hat_j.
```

In the locked mentor experiments, `lambda_2 = gamma = 0`, so
`D_j = 1 - alpha kappa_hat_j`.

### Proposition 4: Coordinate resonance and amplification

For a coordinate with `g_t,j != 0` and `D_j != 0`, the signed diagonal update
has

```text
delta_j / (alpha g_t,j) = 1 / D_j.
```

Therefore:

1. the coordinate system is singular when
   `kappa_hat_j = lambda + 1/alpha`;
2. its amplification relative to the explicit coordinate step is
   `1 / |D_j|`;
3. `0 < D_j < 1` amplifies without reversing the coordinate;
4. `D_j < 0` reverses the coordinate; and
5. when every `D_j` is nonzero, the operator norm of the inverse diagonal
   system is `1 / min_j |D_j|`.

#### Proof

All statements follow directly by solving the scalar equation for each
coordinate. QED.

This is a resonance condition, not a failure of numerical linear algebra. A
division residual near zero establishes that the displayed sampled system was
solved accurately. It says nothing about whether `kappa_hat` estimates a
population curvature, whether dropping off-diagonal terms is appropriate, or
whether the sampled system is well-conditioned.

## 8. Concave Projection And The No-Amplification Guarantee

Let `kappa_tilde_j` be any finite coordinate or block surrogate used for the
current step. It may be a same-generation value or an EMA. Define

```text
c_j = max(-kappa_tilde_j, 0),
d_j = 1 + alpha (lambda + c_j),
delta_safe,j = alpha g_t,j / d_j.
```

Equivalently, this replaces the signed surrogate by its nonpositive part
`min(kappa_tilde_j, 0)` before forming the implicit denominator.

### Proposition 5: Conditional no-amplification

For `alpha > 0`, `lambda >= 0`, and finite `c_j >= 0`:

1. `d_j >= 1` for every coordinate;
2. `delta_safe,j` has the same sign as `g_t,j`;
3. `|delta_safe,j| <= alpha |g_t,j|`;
4. for every `p` in `[1, infinity]`,
   `||delta_safe||_p <= alpha ||g_t||_p`; and
5. if `g_t != 0`, then `g_t^T delta_safe > 0`.

#### Proof

The first statement follows from nonnegativity of `alpha`, `lambda`, and
`c_j`. Division by a positive number at least one preserves sign and cannot
increase coordinate magnitude, proving statements 2 and 3. Monotonicity of
all `p`-norms under coordinate-wise magnitude reduction proves statement 4.
Finally,

```text
g_t^T delta_safe
  = alpha sum_j g_t,j^2 / d_j > 0
```

when at least one gradient coordinate is nonzero. QED.

### Limits of the proposition

This guarantee is conditional and mechanical:

- It compares against the explicit sampled step `alpha g_t`, not against an
  oracle or an optimal update.
- It does not show that `g_t` is an accurate ascent direction for expected
  return.
- It does not guarantee improvement of a nonlinear or noisy objective after a
  finite step.
- The projection discards all positive surrogate values and is therefore a
  biased nonlinear transformation of any signed estimand.
- EMA bias correction removes zero-initialization shrinkage under a stationary
  mean; it does not remove lag when the policy and estimand change.

In particular, neither operation inherits the unbiased LOPO result from
Proposition 3b. Generally `E[max(-J_LOPO, 0)]` is not
`max(-H_stop, 0)`. A bias-corrected EMA is unbiased only for one stationary
fixed target under the usual noise assumptions; across changing policy
centers it estimates a weighted history, not current curvature.

The rule is also not a trust radius. It does not prescribe `||delta||`. With
`c_j` fixed, `alpha` remains in both numerator and denominator; coordinates
with positive damping saturate as `alpha` grows, while undamped coordinates
remain proportional to `alpha`.

## 9. Block Pooling: Estimand And Misspecification

Let the parameter coordinates be partitioned into blocks. The implemented
moment estimator for block `B` is

```text
kappa_hat_B
  = (1 / (2 m sigma^2)) sum_k
      (U_k,+ + U_k,-)
      [(1 / |B|) sum_{j in B} (epsilon_k,j^2 - 1)].
```

### Proposition 6: A block statistic is an average of coordinate statistics

For every realized batch,

```text
kappa_hat_B = (1 / |B|) sum_{j in B} kappa_hat_j.
```

For comparison, replace the same-batch rank `U_i` by a fixed pointwise utility
`V_i = T(Y_i)` that satisfies Proposition 1, and denote the analogous block
statistic by `kappa_hat_B^T`. Then

```text
E[kappa_hat_B^T]
  = (1 / |B|) trace([H_sigma,T(theta)]_BB),
```

where `H_sigma,T` is the Hessian of the Gaussian smoothing of the fixed
transformed fitness. Raw return is the special case `T(y) = y`.

#### Proof

Exchange the finite sums over pairs and coordinates. The fixed-transform
expectation then follows coordinate by coordinate from Proposition 2 and the
fixed-transform observation after Proposition 1. QED.

For independent standard-normal coordinates, the Gaussian block feature has

```text
Var[(1 / |B|) sum_{j in B} (Z_j^2 - 1)] = 2 / |B|.
```

This explains why pooling can make the score feature less variable. It does
not imply that the complete product with utility has variance exactly reduced
by the same factor, because utility and the score feature are dependent.

### When the block solve is exact

Replacing every coordinate in `B` by one scalar `kappa_B` matches the full
linearized system only under a strong structural model, for example:

```text
J_BB = kappa_B I_|B|
```

for each block and `J_BC = 0` for distinct blocks. Otherwise:

- heterogeneous diagonal values are replaced by their average;
- within-block off-diagonal interactions are ignored;
- cross-block interactions are ignored; and
- architecture-defined blocks need not align with curvature eigenspaces.

Thus pooling trades resolution and possible misspecification for lower
dimensional estimation. Under same-batch ranks, Proposition 6 remains an exact
algebraic averaging statement. Its expectation is `c_m` times the corresponding
block trace of `H_stop`; after LOPO correction it is unbiased for that
stop-gradient trace. The raw-Hessian trace interpretation does not carry over.

The separate joint-OLS pilot is a different estimator. Its coefficient has a
block-isotropic quadratic interpretation only when its regression model is
correct; it is not part of the locked confirmation and should not be merged
with the moment-estimator theory.

## 10. Independent Reference Ranks And Cross-Fitting

The dependence problem can be reduced by constructing the fitness transform
from data independent of the target perturbation pair.

Let `A` be a reference dataset independent of a new target population. From
`A`, construct a bounded measurable mapping `T_A(y)`, such as a centered
empirical-CDF score. Conditional on `A`, define the frozen-reference objective

```text
F_A,sigma(theta)
  = E_Z,xi[T_A(Y(theta + sigma Z, xi))].
```

### Proposition 7: Conditional interpretation with an independent reference

Assume:

1. `A` is independent of every target antithetic pair and its rollout noise;
2. `T_A` is held fixed when differentiating with respect to `theta`;
3. target perturbations are Gaussian and have the intended marginal rollout
   distribution; and
4. the required score moments are integrable.

Then, conditional on `A`,

```text
(1 / (2 m sigma^2)) sum_k
  [T_A(Y_k,+) + T_A(Y_k,-)]
  (epsilon_k epsilon_k^T - I_d)
```

is unbiased for `Hessian F_A,sigma(theta)`.

#### Proof

Conditional on `A`, `T_A` is a fixed pointwise transform independent of every
target pair. Proposition 2 therefore applies to the transformed rollout
return. QED.

This proposition yields a precise, limited interpretation: a Hessian of a
**random frozen-reference transformed objective**, conditional on the
reference data. It is not the Hessian of raw return. If `A` was sampled from a
distribution centered at the current `theta`, then refreshing `A` as `theta`
changes again creates a sequence of local frozen objectives, not automatically
one global scalar objective.

### Pair-level cross-fitting

A data-efficient version can split independent antithetic pairs into folds.
For each target fold `r`:

1. construct `T_-r` using only pairs outside fold `r`;
2. transform returns in fold `r` with `T_-r`; and
3. compute the score moment on fold `r`.

Excluding the **entire antithetic pair** is necessary. Excluding only one
member while using its mate in the reference does not give pair-level
independence. Each fold contribution then has the conditional interpretation
in Proposition 7. Averaging folds estimates the average of fold-specific
random frozen-reference objectives.

Cross-fitted contributions are mutually dependent because a fold used for
estimation also appears in other folds' reference sets. Standard errors and
confidence intervals must account for that dependence; treating all
pair-fold contributions as iid is not justified.

### Additional requirements for a defensible estimator

- Do not recenter or standardize target utilities using their own current
  fold unless the induced baseline is proven harmless. Use a reference-derived
  constant or a baseline independent of the target pair.
- Freeze the reference mapping in the derivative and state whether it is
  refreshed between optimization iterations.
- State whether the target is expected transformed noisy return
  `E[T_A(Y)]` or a transform of expected return; they differ.
- Handle ties by a predeclared empirical-CDF convention.
- Use independent reference and target rollout seed streams.
- If a deterministic limiting objective is claimed, provide an asymptotic
  argument as reference size grows and specify the reference distribution.
- If a single global objective is claimed across iterations, hold `T_A` fixed
  across those iterations or prove that the adaptive vector field is the
  gradient of a scalar potential. Local stop-gradient semantics are not
  sufficient.

## 11. NES And IGO Novelty Boundary

The Gaussian factor

```text
Z_j^2 - 1
```

is the score for a coordinate log scale of a Gaussian search distribution.
Multiplying this score by fitness utilities, including rank utilities, is an
established covariance/scale-update construction in
[Natural Evolution Strategies](https://www.jmlr.org/papers/v15/wierstra14a.html).
IGO likewise places adaptive quantile weights inside search-distribution score
updates and explains the resulting invariances
([Ollivier et al., 2017](https://www.jmlr.org/papers/v18/14-467.html)).

Therefore, the following are not defensible novelty claims:

- discovering the statistic `U (Z_j^2 - 1)`;
- establishing rank-based monotone invariance in ES;
- identifying the statistic as a Gaussian covariance score; or
- obtaining a raw-fitness Stein Hessian identity.

The repository's narrower, potentially publishable contribution is the
diagnosis of what happens when a frozen-rank covariance-score statistic is
rescaled into curvature units and inserted into a signed implicit **mean**
update:

- the exact batch-conditional linearization;
- high-dimensional estimator unreliability;
- signed-denominator resonance despite an accurate arithmetic solve;
- the distinction between structured direction change and generic
  attenuation; and
- a locked negative confirmation that prevents a superiority claim.

Concave projection, architecture-level pooling, and EMA are stabilization
devices. Their combination should not be presented as novel without a much
more specific comparison to existing curvature and ES methods.

## 12. Recommended Terminology And Allowed Claims

| Object | Recommended term | Claim that is justified |
| --- | --- | --- |
| Raw-return Stein moment | Gaussian-smoothed raw-fitness Hessian estimator | Unbiased under Propositions 1 and 2 |
| Same-batch rank moment | Frozen-rank covariance-score curvature surrogate | Exact conditional Jacobian; expectation is `c_m H_stop` under iid pairs |
| LOPO rank moment | Current-CDF stop-gradient U-statistic | Unbiased and consistent for `G_stop` or `H_stop`, not raw-return derivatives |
| Signed update | Signed diagonal surrogate linearization | Exact solve of the sampled diagonal system when residual is small |
| Projected update | Concave-projected structured attenuation | Cannot amplify or reverse the explicit sampled step |
| Block statistic | Block-averaged covariance-score surrogate | Exact average of coordinate statistics; raw trace meaning only for a fixed transform |
| Independent-reference statistic | Frozen-reference transformed-objective Hessian estimator | Conditionally unbiased for the stated random transformed objective |

Avoid unqualified phrases such as "the Hessian method," "unbiased rank
Hessian," "implicit Hessian of ES," or "second-order ES" when referring to the
same-batch rank implementation.

## 13. Remaining Theoretical Limitations

This note does not provide:

- a convergence rate for the complete adaptive optimizer trajectory;
- calibrated finite-sample constants for policy-network dimensions;
- a guarantee that the concave projection improves expected return;
- a justification for architecture-defined blocks;
- an analysis of nonstationary EMA error; or
- a proof that the adaptive rank vector field is conservative.

Those are open requirements, not details to assume. The dependence-aware
coordinate concentration scale for the LOPO U-statistic is developed in the
separate resonance note. The next optimizer theory must additionally cover
the changing current-CDF target, structural projection, EMA drift, and
block-model misspecification.

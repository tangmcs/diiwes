# Leave-own-pair-out rank U-statistic curvature

The LOPO attribution study has three fresh-population conditions. It does not
change the pooled-rank, EMA, or block-OLS conditions. Replay, importance
weighting, trust clipping, gradient clipping, parameter projection, L2
regularization, and scalar damping are disabled.

| Condition | Utility and gradient | Curvature statistic | Applied update | Attribution role |
| --- | --- | --- | --- | --- |
| `lopo_gradient_only_es` | exact LOPO | none | explicit `alpha g_LOPO` | LOPO-gradient baseline |
| `concave_block_lopo_u_stat` | exact LOPO | raw LOPO block moment | structured block attenuation | directional curvature treatment |
| `concave_block_lopo_u_stat_isotropic_control` | exact LOPO | the same raw LOPO block moment | isotropic attenuation with the structured-step norm | norm-only control |

The gradient-only condition does not persist `hessian_ema.npy` or advertise a
curvature-state artifact. The two curvature conditions persist that state for
auditability.

At an identical center with the same evaluated population, all three
conditions compute the same utility vector and proposal gradient. The two
curvature arms also compute the same raw curvature; the isotropic arm matches
the Euclidean norm of the structured arm's counterfactual step on that batch.
The regression test checks this exact one-generation decomposition with
bit-identical perturbations and fitness.

Independent multi-step training runs diverge after their first different
update. A common seed continues to pair the standardized perturbation draws,
but it does not keep candidate policies, returns, utilities, gradients, raw
curvature, or realized step norms equal across the resulting trajectories.
Consequently, the three conditions are the correct training ablation, while a
frozen-checkpoint paired evaluation is required for exact one-step causal
attribution of direction versus norm.

## Utilities

Let an even population contain `m >= 3` exact antithetic pairs with returns
`Y[k,+]` and `Y[k,-]`. Let `U` be `centered_ranks` over all `2m` returns and

```text
K[k]       = (Y[k,+] > Y[k,-]) - (Y[k,+] < Y[k,-])
mate       = [K; -K]
c_m        = 2(m - 1) / (2m - 1)
U_LOPO     = ((2m - 1) U - 0.5 mate) / (2(m - 1)).
```

Comparisons, rather than subtraction followed by `sign`, preserve exact tie
zeros and avoid overflow for large finite returns. No recentering operation is
applied to `U_LOPO`. Nevertheless, it is structurally zero-sum: every ordered
cross-pair comparison has an equal and opposite comparison. This cancellation
also holds with ties because both directions contribute zero. The runtime
checks the sum and mean against floating-point tolerances. `U_LOPO` equals the
empirical midrank of each return against all observations outside its own
antithetic pair.

The same `U_LOPO` vector is used in both estimates:

```text
g_LOPO = mean_i(U_LOPO[i] epsilon[i]) / sigma

q[k,B] = mean_{j in B}(epsilon[k,j]^2) - 1
H_LOPO[B] = mean_k((U_LOPO[k,+] + U_LOPO[k,-]) q[k,B]
                        / (2 sigma^2)).
```

For the existing pooled-rank block moment `H_pooled`, the implementation checks
the finite-sample identity `H_LOPO = H_pooled / c_m`. It also records the norm
of the within-pair term in
`g_pooled = c_m g_LOPO + within_pair_remainder`. Rescaling only curvature while
leaving the pooled gradient unchanged is not this matched estimator.

Under iid antithetic pair clusters, these order-two statistics target the
gradient and block covariance-score curvature of the current-return mid-CDF
transform with that population CDF held fixed. They are not raw-return
derivatives and not the total derivatives of a globally adaptive rank
objective.

## Exact at-proposal endpoint identity

Hold the observed LOPO utility vector fixed and define the self-normalized
endpoint gradient map using Gaussian relative weights. At the proposal center,
exact antithetic sampling gives `mean(epsilon) = 0`, while the structural LOPO
identity gives `mean(U_LOPO) = 0`. Therefore the raw, preprojection LOPO block
moment is exactly the block average of diagonal entries of the Jacobian of that
fixed-utility self-normalized map at the proposal center.

This is deliberately narrower than saying that the optimizer uses the endpoint
Jacobian. The implementation estimates only diagonal entries averaged within
blocks, then applies concave projection and either structured or isotropic
attenuation. It does not estimate the full Jacobian, and the projected
curvature operator is not itself an endpoint Jacobian. The identity does not
extend to an off-proposal endpoint, a globally adaptive rank objective, or a
raw-return Hessian.

The self-normalization detail is testable rather than cosmetic. For utilities
with mean `ubar` and block empirical second moment `S_B`, exact antithetic
sampling gives

```text
J_unnormalized[B] - J_self_normalized[B]
    = ubar (S_B - 1) / sigma^2.
```

The gap is floating-point zero for structural LOPO utilities. A regression test
adds a constant to the utility vector and verifies by finite differences that
the resulting nonzero gap has exactly this value. A separate finite-difference
test verifies the raw LOPO block moment against the self-normalized endpoint
map without the artificial shift.

## Delete-pair jackknife

For pair clusters `k` and `l`, define

```text
A[k,l]   = sum_{s,t in {+,-}} sign(Y[k,s] - Y[l,t])
h[k,l,B] = A[k,l] (q[k,B] - q[l,B]) / (16 sigma^2).
```

`A` is antisymmetric and `h` is symmetric. With

```text
R[k,B] = sum_{l != k} h[k,l,B]
T[B]   = 0.5 sum_k R[k,B]
H[B]   = T[B] / choose(m,2),
```

the delete-pair estimates are

```text
H[-k,B] = (T[B] - R[k,B]) / choose(m-1,2)
SE[B]^2 = (m-1)/m sum_k (H[-k,B] - H[B])^2.
```

The production calculation checks that the U-statistic, the LOPO rank moment,
the pooled rescaling, and the mean delete-pair estimate agree up to floating
point roundoff. Tests also compare every fast row-sum deletion with literal
removal and reranking of that pair.

## Inference boundary

The jackknife SE targets the raw same-generation block U-statistic and is
labeled `componentwise_asymptotic_non_simultaneous`. Its usual interpretation
requires independent, identically distributed,
nondegenerate pair clusters. Arbitrary dependence and common random numbers
within an antithetic pair are allowed; dependence across pairs is not covered.
Ties are allowed and use zero comparison kernels.

The SE is not simultaneous across blocks and is not calibrated for the moving
optimizer, nonlinear concave projection, model selection, or repeated use over
training. If `curvature_confidence_z` is enabled for this mode,
`curvature_beta` must be zero. The resulting shrinkage remains a heuristic
componentwise screen, not a confidence or optimization-coverage guarantee.

The default condition leaves `curvature_confidence_z` disabled. It records the
utility definition, `c_m`, jackknife method and validity, inference assumptions,
within-pair gradient remainder, and all checked finite-sample identities.

## Named-condition locks

The three names above are attribution contracts, not loose presets. Config and
CLI validation reject changes to rank shaping, LOPO utility mode, curvature
fitness/structure/estimator, attenuation role, nonzero curvature EMA or
confidence adjustment, L2 or implicit/scalar damping, curvature clipping,
center evaluation, leave-one-out legacy baselines, replay-weight thresholds,
or non-antithetic sampling. The generic pooled-rank conditions retain their
existing sweep overrides.

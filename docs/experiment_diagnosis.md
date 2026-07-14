# No-Replay, Trust-Free DIIWES Experiment Diagnosis

## What the previous plots measured

The MuJoCo plot documented in the previous README used
`results/mujoco_lr_sweep_46638567`. Those DIIWES runs inherited a trust radius
of 1.0 from the environment YAML files, while the Standard ES condition did
not. Trust clipping was active on 99.8-100% of DIIWES iterations. The applied
step norm was therefore almost always exactly 1.0 even though the pre-clipping
norm changed substantially with the configured learning rate.

For the no-curvature update, clipping reduces the update to

```text
delta = R * gradient / ||gradient||.
```

The configured learning rate and scalar damping cancel from this expression.
The old learning-rate sweep therefore measured a radius-normalized method
rather than implicit step-size robustness.

The trust contribution is also visible in the saved Hopper debug runs at
learning rate 0.25. Across three seeds, Standard ES plus trust and no-curvature
DIIWES plus trust both reached median best returns around 3,650. Their
trust-free counterparts were both around 1,050. These runs should be treated as
a diagnosis of the confound, not as evidence for the proposed method.

## Why the Hessian path was failing

There was no endpoint fixed-point iteration or general linear-system solve in
the implementation.
The code applied an elementwise diagonal denominator and then reported
`converged=true` and `residual=0` unconditionally. The old diagnostics therefore
could not distinguish estimator instability from solver instability.

The saved trust-free Hopper runs point to the estimator and update heuristics:

- A 5,123-parameter policy estimated one diagonal value per parameter from
  about 80 fresh antithetic pairs after replay warmup.
- Raw-return Hessian estimates reached coordinate extrema above 100,000 at
  learning rate 0.25 and above 400,000 at learning rate 1.0.
- The curvature clip was saturated broadly, and 27-50% of coordinates reached
  the old multiplier floor.
- The multiplier floor changed the intended large-alpha limit. Once the exact
  inverse multiplier fell below 0.05, the coordinate update became
  `0.05 * alpha * gradient` and grew linearly with alpha again.

The closed-form diagonal solve itself is not singular in the implemented
negative-curvature projection because every denominator is positive. The
dominant observed problem was a high-variance curvature estimate followed by
clipping and flooring, not an unstable `numpy.linalg.solve` call.

The production estimator check in `scripts/diagnose_curvature_estimator.py`
reproduces this failure on known diagonal quadratics. With 80 pairs, raw-Hessian
relative error rises from 0.58 at dimension 5 to 323.5 at Hopper's dimension
5,123. At dimension 5,123, independent rank-curvature batches have mean
coordinate correlation 0.006 and sign agreement 0.505, effectively chance. The
resulting `alpha_0=30` update has relative error 3.74 against the exact-Hessian
update. The sampled signed system has nonpositive diagonals on 51.7% of
coordinates and median condition about 72,099. Direct diagonal arithmetic still
has residual below `6e-17`; the failure is estimator-induced system geometry,
not inaccurate division. The exact results are saved in
`docs/curvature_estimator_diagnostic.csv`.

The raw path mixes a rank-shaped policy gradient with raw-return
curvature. Those are not derivatives of the same transformed objective, and
rescaling rewards changed the curvature step while leaving the rank gradient
unchanged. A matched-rank arm is therefore included as a separate scale-invariant
ablation. Population ranks are batch-dependent rather than a fixed smooth
objective, so that arm is labeled as a surrogate, not a Hessian or exact
implicit step.

More fundamentally, none of the current arms solves
`delta = alpha * g(theta_t + delta)`. The applied update evaluates the gradient
once at `theta_t` and applies a projected diagonal curvature preconditioner.
The curvature-free case is exactly a scalar effective-learning-rate
reparameterization.

## Changes in the corrected implementation

- Trust-radius rescaling and its CLI/condition variants were removed.
- The multiplier floor and norm-preserving curvature variants were removed.
- Replay is disabled with `replay_enabled: false`, `reuse_fraction: 0`, and
  `buffer_size: 0`; runtime and result validators require a fully fresh batch.
- Every curvature update now has exactly 100 fresh antithetic pairs at
  population size 200.
- Raw-return Hessian and matched-rank surrogate arms have separate condition
  labels and are not pooled.
- The applied projected diagonal update and the unprojected signed system have
  separate residual, diagonal, positivity, and condition diagnostics.
- `inverse_sqrt` scheduling implements
  `alpha_t = alpha_0 / sqrt(t + 1)`, including `alpha_0 = 30`.
- Observation normalization is calibrated once and frozen, so buffered returns
  would remain attached to a fixed policy mapping if replay is studied later.
- Common random numbers are shared within each antithetic pair but change
  between pairs and generations. Evaluation episodes use distinct fixed seeds.
- The initial policy is evaluated at zero training steps, and matched-seed runs
  must have the same initial return across every condition and learning rate.
- Exported tables separate initial learning rate from the effective per-step
  learning rate, so scheduled runs are grouped correctly.

## Completed three-arm diagnostic

The completed diagnostic job `49648326` contained:

1. `standard_es`
2. `endpoint_implicit_es`
3. `linearized_implicit_es`

The endpoint arm recomputes weights and transformed score vectors for the
current fresh population at every Picard iterate and evaluates the actual
fixed-point residual at the returned point. The linearized arm applies the
signed same-generation diagonal system instead of the historical
negative-curvature projection. Both primary implicit arms set scalar damping to
zero.

That historical matrix used no replay or trust mechanism, `alpha_0` values
0.25, 1, 3, 10, and 30, and the inverse-square-root schedule. The active
advisor-facing launcher no longer includes the endpoint Picard arm. It compares
only Standard ES and the signed diagonal linearized method for `alpha_0` values
10 and 30 under inverse-square-root and inverse-linear schedules, with ten
paired seeds.

An initial two-update Hopper smoke run at `alpha_0=30` exposed a defect in the
historical absolute ratio floor: every endpoint ratio reached the lower floor,
giving uniform weights and an endpoint gradient numerically identical to
Standard ES. The production endpoint arm therefore normalizes relative
Gaussian logits without absolute ratio clipping; the old bounds are diagnostics
only. In the same smoke run, the signed diagonal division had relative residual
below `5e-17`, while split-half Hessian correlation was about `0.039` and sign
agreement about `0.521`.

After removing absolute ratio clipping and adding endpoint-weighted utility
centering, the corrected `alpha_0=30` smoke run exposed the actual Picard
failure. The ten unrelaxed iterations formed a two-cycle between the explicit
step (norm about `2493.5`) and zero. Inner endpoint ESS fell to one sample
(`ESS/B=0.005`), the maximum endpoint weight reached one, the relative-logit
span exceeded `3.9e6`, and the final equation residual remained `1.0`. Because
ten is even, the returned diagnostic iterate was zero. These trajectory fields
are now logged explicitly; final-iterate ESS alone would have hidden the
failure. The full matrix is still needed before drawing a performance
conclusion.

The controlled quadratic diagnostic already separates the estimator from the
solve at the training population size. It records raw relative error,
split-batch rank correlation, coordinate sign agreement, update error, signed
nonpositive-diagonal rate, signed condition, and arithmetic residual.

Reproduce the current estimator check with:

```bash
python scripts/diagnose_curvature_estimator.py \
  --output docs/curvature_estimator_diagnostic.csv
```

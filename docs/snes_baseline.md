# Separable NES baseline

The `snes` condition implements separable natural evolution strategies (SNES)
as a fresh-population, diagonal-Gaussian baseline. Its reference is Wierstra et
al., [Natural Evolution Strategies](https://www.jmlr.org/papers/v15/wierstra14a.html),
especially the separable Gaussian update.

For the search distribution

```text
x_k = mu + sigma * s_k,       s_k ~ N(0, I),
```

where `*` is coordinate-wise multiplication, the implementation sorts returns
from best to worst and assigns the canonical utilities

```text
u_k = max(0, log(lambda / 2 + 1) - log(k)) / Z - 1 / lambda.
```

It then applies both natural-gradient updates from the same pre-update search
distribution:

```text
g_mu       = sum_k u_k s_k
g_logsigma = sum_k u_k (s_k^2 - 1)

mu    <- mu + eta_mu * sigma * g_mu
sigma <- sigma * exp((eta_sigma / 2) * g_logsigma).
```

`learning_rate` is `eta_mu`. `snes_sigma_learning_rate` is `eta_sigma`; when
omitted, it uses the SNES default `(3 + log(d)) / (5 sqrt(d))`. `noise_std`
initializes every coordinate of `sigma`. There is no replay, importance
weighting, trust radius, gradient clipping, parameter projection, weight decay,
or sigma clipping.

## Recorded conventions and deviations

- Exact return ties receive the average utility across their occupied ranks.
  This removes arbitrary dependence on input order.
- The configured population size is retained for equal-budget experiments;
  the reference default `4 + floor(3 log(d))` is not forced.
- Antithetic sampling follows the experiment config. It is a variance-reduction
  modification rather than part of the reference SNES pseudocode.
- The common trainer learning-rate schedule may vary `eta_mu`; canonical SNES
  uses the constant default `eta_mu = 1`. `eta_sigma` remains constant.
- Adaptation sampling and restart logic from broader NES implementations are
  not included.

The generic experiment configuration does not silently replace `learning_rate`
with the reference value. Any future SNES comparison must therefore define a
separate, preregistered `eta_mu` grid that includes the canonical constant value
`1`; reusing the Standard-ES learning-rate grid alone is not a valid tuning
protocol.

## Ask/tell state contract

`ask()` records the exact coordinate-wise sampling standard deviation and the
current SNES generation token. `tell()` requires both fields and rejects a
missing, stale, or modified value before updating either the mean or search
scale. This prevents delayed evaluations from being interpreted as samples
from a newer diagonal Gaussian.

The final coordinate-wise standard deviation is written to
`snes_search_std.npy`. The resolved optimizer metadata names this artifact and
marks it as final optimizer state for audit. It is not a resume checkpoint;
the trainer does not claim resume support.

The resolved run config records all of these choices. Per-generation history
records the pre/post sigma range and geometric mean, `||g_mu||` in standardized
coordinates, `||sigma * g_mu||` in parameter space, the log-sigma gradient and
step norms, and the mean step's Mahalanobis norm under the sampled distribution.
The old `snes_mean_natural_gradient_norm` field remains as a compatibility alias
for `snes_parameter_space_mean_direction_norm`; new analysis should use the two
explicitly named fields.

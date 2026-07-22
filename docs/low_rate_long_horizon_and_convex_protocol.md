# Low-Rate Long-Horizon and Convex Implicit-Update Protocol

## Purpose

This protocol separates two questions:

1. Does the implicit equation have the predicted large-step stability benefit
   on a quadratic with a known Hessian?
2. Does the checked-out main diagonal-Hessian implementation improve Standard
   ES on Hopper when the initial learning rate is at most 2 and training lasts
   longer?

Neither experiment uses replay, a trust region, or geometry-free scalar
damping.

## Controlled convex mechanism experiment

The benchmark maximizes a concave quadratic reward

\[
J(x)=\tfrac12x^\top Hx,
\]

which is reported equivalently as minimizing the strongly convex loss
\(f(x)=-J(x)\). The aligned case has curvature magnitudes 0.1,
\(\sqrt{0.2}\), and 2 in three coordinate blocks. Therefore the exact explicit
method is stable only for a constant step below \(2/L=1\), whereas the exact
implicit method

\[
(I-\alpha H)\Delta=\alpha\nabla J
\]

is stable for every positive step on this concave quadratic.

Locked grid:

- cases: aligned strongly convex loss, a rotated positive-definite loss with
  the same spectrum, and an aligned additive-observation-noise case;
- constant steps: `0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2`;
- 500 updates, with predeclared readouts at 10, 30, 100, 300, and 500;
- population 500 (250 antithetic pairs), perturbation scale 0.1;
- Monte Carlo seeds 0 through 9;
- raw-fitness gradient and Stein-curvature regime only.

The benchmark includes exact explicit and full-implicit controls, a
sampled-gradient/known-diagonal control, signed and concave-projected diagonal
Stein-Hessian arms, block-pooled variance-reduction ablations, and separate
equal-norm isotropic controls. Common random numbers are used inside each cell.
The equal-norm controls are analysis comparators, not additive scalar damping
inside the Hessian method. The synthetic harness is isolated from
`core/diiwes.py`; it tests the implicit mechanism and estimator behavior, not
the exact Hopper software path.

The presentation postprocessor derives prefix metrics at the five predeclared
checkpoints from the saved per-update trajectories. It also deterministically
replays the explicit-ES reference states to archive coordinatewise diagonal
Stein estimates and their implied projected multipliers. Convex-loss figures
plot `-H_hat`, because the optimizer internally maximizes the negative
quadratic reward and therefore estimates its negative Hessian.

Launcher: `scripts/submit_convex_implicit_step_sweep.sh`.

## Hopper low-rate long-horizon experiment

Final locked grid:

- environment: Hopper-v5;
- methods: `standard_es` and `diag_curvature`;
- schedule: \(\alpha_t=\alpha_0/\sqrt{t+1}\);
- initial rates: `0.1, 0.25, 0.5, 1, 2`;
- seeds: 0 through 9, paired by method and rate;
- population: 500 fresh candidates (250 antithetic pairs) per update;
- maximum horizon: 2,000 updates;
- prefix-horizon readouts: 500, 1,000, and 2,000 updates;
- replay buffer and reuse fraction: zero;
- trust radius: disabled;
- scalar implicit damping and L2 coefficient: zero.

Only inverse-square-root decay is repeated because it was the directionally
favorable schedule in job 49811294. At these smaller initial rates,
inverse-linear decay would reduce the update by factors of 500--2,000 at the
analysis horizons and would primarily test early freezing again.

The diagonal-Hessian arm retains the main implementation's raw-return
leave-one-out Stein curvature, EMA beta 0.99 with bias correction, curvature
cap 1,000, and multiplier floor 0.05. Exact signed Hessian and multiplier
vectors are saved at every update, along with cap/floor counts, split and
temporal agreement, condition estimates, and linear-system residuals.

The 2,000-update trajectory is the source for every prefix readout; separate
shorter jobs would duplicate identical seeded prefixes. Prefix comparisons
are correlated and must not be counted as independent experiments.

Primary performance metric: paired, iteration-normalized AUC at each prefix.
Secondary metrics: last-50-update mean, final return, best return, and
time-to-threshold. Mechanism interpretation must include multiplier-floor and
curvature-cap occupancy, Hessian repeatability, and applied step norms.

Launcher: `scripts/submit_hopper_low_lr_long_sweep.sh`.

The matrix contains 100 tasks and 100 million fresh candidate evaluations.
With exact coordinate histories retained, expected storage is approximately
8 GiB for the two arrays alone and roughly 10--12 GiB overall.

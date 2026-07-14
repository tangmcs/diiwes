# Hopper Fresh-Only Optimizer Development Protocol

## Purpose

This is an exploratory calibration screen on development seeds. It does not
test a frozen superiority claim and must not be reported as confirmation.
Its purpose is to replace the weak plain-SGD baseline with tuned fresh-only ES
optimizers and to determine whether structured curvature changes direction in
a useful way beyond generic attenuation.

## Fixed Scope

- Environment: `Hopper-v5`
- Development seeds: `200`, `201`, and `202`
- Population: 200 fresh candidates, or 100 antithetic pairs
- Updates: 250
- Candidate-policy rollout budget: exactly 50,000 rollouts per run
- Perturbation scale: `sigma=0.02`
- Policy: MLP with hidden widths `[64, 64]`
- Observation normalization: frozen after three calibration episodes
- Online evaluation: five fixed-seed episodes every ten updates
- Replay and cross-generation importance sampling: disabled
- Picard iteration: excluded
- Trust radius and parameter/update norm projection: disabled for the
  curvature and plain ES arms
- Gradient, parameter, and curvature clipping: disabled for the curvature and
  plain ES arms

ClipUp is included as an established strong baseline and uses its own published
velocity clipping rule. It is not part of the proposed curvature method and
must not be described as evidence for implicit stabilization.

This development screen fixes the number of candidate-policy rollouts, not the
number of environment transitions. Hopper episodes terminate at different
lengths, so the 250-generation evaluation AUC is a tuning metric and must not
be described as transition-level sample efficiency. The validator reports the
actual training and evaluation transition counts for every run, together with
the first, mean, and maximum update norm divided by `sigma` and the fraction of
updates with `||Delta_t|| <= sigma`. Any subsequent
confirmatory comparison must instead use one prospectively fixed training-step
budget, stop after the first complete generation that crosses it, and evaluate
held-out return and AUC at the common budget. Fixed-rollout and fixed-transition
results answer different questions and must be reported separately.

## Matrix

The versioned JSON manifest is
`experiments/manifests/hopper_fresh_optimizer_development.json`. It contains 33
hyperparameter cells:

- 7 plain SGD-style ES cells;
- 3 Momentum ES cells;
- 4 Adam ES cells;
- 4 ClipUp ES cells;
- 6 structured block-EMA curvature cells;
- 6 norm-matched isotropic attenuation controls; and
- 3 exploratory block-OLS confidence-adjusted-shrinkage cells.

The OLS adjustment is a screening heuristic, not a confidence interval or a
hypothesis test. It replaces `[-h]_+` by `[-(h + z SE)]_+`, so it both suppresses
uncertain components and shrinks the magnitude retained for active components;
it is not a binary gate. Its classical homoskedastic standard error does not
account for dependence induced by same-batch ranks, changing policy parameters,
or the EMA state. The `z=1.645` threshold may reduce activation frequency, but
it is not calibrated to 95% coverage in this experiment and cannot support an
inferential claim.

Before freezing this exploratory manifest, a one-generation scale calibration
on development seeds `200`, `201`, and `202` measured Standard ES gradient
norms `77.6491` to `97.5638`. At `alpha=0.001`, the corresponding first update
was `3.8825` to `4.8782` times `sigma`. These measurements, not a comparative
performance outcome, motivated adding sub-`sigma` and near-`sigma` cells. The
calibration seeds are therefore development data and cannot be reused as
untouched confirmation seeds.

The Standard ES grid is `3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2`.
Structured and isotropic controls use the matched subset
`3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 3e-2`. Momentum, Adam, ClipUp, and OLS
confidence-adjusted shrinkage each span local and nonlocal candidate settings
specified exactly in the manifest. The four ClipUp settings retain the primary
paper's starting relationship `alpha = v_max / 2`.

All cells in this screen use constant learning rates. The advisor-requested
decreasing sequences were already completed separately for Standard ES and the
signed/concave curvature variants: `10 / sqrt(t+1)`, `30 / sqrt(t+1)`,
`10 / (t+1)`, and `30 / (t+1)`. They are not repeated here because this screen
tests the locality and optimizer-baseline gaps those large-step jobs exposed.

## Interpretation Rules

1. Use this screen only to choose development hyperparameters and identify
   gross failures.
2. Compare structured curvature with the norm-matched isotropic control at the
   same learning rate and schedule before attributing a gain to direction.
3. Compare all curvature arms with tuned Adam, ClipUp, Momentum, and plain ES;
   beating only the old large-step plain-SGD runs is insufficient.
4. Report all 99 runs, including failed or low-return cells.
5. Do not use seeds `100` through `109` for tuning; those were already consumed
   by the locked confirmation.
6. Any new positive claim requires a new preregistration, untouched tasks and
   seeds, a source snapshot, and a larger paired sample.
7. Do not compare this screen as an environment-transition sample-efficiency
   result; its budget is 50,000 policy rollouts, and transition counts vary.

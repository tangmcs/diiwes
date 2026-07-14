# Hopper No-Replay, No-Trust Implicit ES Results

> Historical result: job `49648326` included an endpoint Picard arm. The active
> advisor-facing launcher excludes Picard and does not extend this job.

## Assessment

**Ready to share with caveats.** Job `49648326` cleanly answers the mentor's
mechanism question, but it does not support a performance-improvement claim.

- Plain endpoint Picard iteration did not solve the implicit equation in any
  generation.
- The diagonal linear-system arithmetic was accurate, but its matched-rank
  Jacobian/Hessian surrogate was non-repeatable and produced an indefinite,
  badly conditioned system.
- Neither method provides a robust improvement over Standard ES under the
  tested protocol.

## Protocol And Integrity

- Environment: `Hopper-v5`
- Conditions: Standard ES, endpoint Picard implicit ES, signed diagonal
  linearized implicit ES
- Initial learning rates: `0.25, 1, 3, 10, 30`
- Schedule: `alpha_t = alpha_0 / sqrt(t + 1)`
- Seeds: `0, 1, 2, 3, 4`
- Population: 200 fresh candidates per update, 500 updates per run
- Replay, trust clipping, scalar damping, gradient clipping, parameter
  projection, curvature clipping, and curvature EMA: disabled
- Completed cells: 75/75; Slurm exit code: 0 for every cell
- Validated history records: 37,500
- Launch source digest:
  `0bf68707382138dd24482c5396705ff817e7af4ae1442e990f50bc73e8cc443c`
- Post-run source digest after the validator-only fix and before the no-Picard
  follow-up:
  `de8f47c2fc396479309c979ea00529b096b8cdb81222191abe6e605fd2951074`

Every history record has 200 fresh and zero reused candidates, zero replay
mass, an empty buffer, no parameter projection, no trust fields, and the exact
inverse-square-root learning rate. Initial-return differences within each seed
were at most `4.1e-13`, a numerically negligible floating-point variation.
The JSON and JSONL histories agree record for record, all 75 status files say
`complete`, and every Slurm stderr file is empty.

The post-run validator changed one equality check from bitwise equality to a
`1e-12` numerical tolerance. This did not change optimizer, trainer, or run
artifacts. Every task verified the launch digest, but the launch tree included
modified or untracked source not reconstructible from its recorded Git
revision, and no byte-for-byte source snapshot was archived. The active
no-Picard follow-up subsequently changed the launcher and validator again, so
the current tree is also not byte-identical to this historical job.

## What "Implicit" Means Here

Standard ES applies the explicit step `delta = alpha_t * g_t(0)`. The endpoint
arm instead defines `g_t(delta)` by re-centering the same 200 evaluated
candidate parameters at `theta_t + delta`, recomputing their Gaussian weights
and score vectors, and attempting to solve

```text
delta = alpha_t * g_t(delta).
```

This is within-generation endpoint reweighting: it performs no replay and no
additional environment rollout. The signed diagonal arm linearizes that same
equation at zero and solves

```text
(I - alpha_t * diag(H_t)) delta = alpha_t * g_t(0).
```

Thus, "implicit" refers to the update appearing on both sides of its defining
equation, not to the historical curvature multiplier or trust-radius code.

## Primary Performance

The primary metric is evaluation-return AUC over the first 75,000 training
environment steps, with linear interpolation between evaluation checkpoints.
Evaluation rollouts are excluded from that training-step axis. Values are
five-seed mean +/- sample standard deviation.

| Initial alpha | Standard ES | Endpoint Picard | Signed diagonal |
| ---: | ---: | ---: | ---: |
| 0.25 | 89.18 +/- 27.18 | 26.49 +/- 33.89 | 12.45 +/- 6.60 |
| 1 | 41.08 +/- 39.71 | 26.50 +/- 33.89 | 10.04 +/- 5.18 |
| 3 | 22.61 +/- 15.98 | 26.50 +/- 33.89 | 14.21 +/- 5.71 |
| 10 | 14.56 +/- 5.61 | 26.50 +/- 33.89 | 19.02 +/- 7.93 |
| 30 | 12.59 +/- 8.07 | 26.50 +/- 33.89 | 10.94 +/- 3.21 |

The endpoint values near `26.50` are the mean initial return. They do not
represent learning: the even-length Picard iteration usually returned the
near-zero branch of a two-cycle during the early budget. Its apparent advantage
at some large learning rates is mainly that Standard ES damaged the initial
policy while Picard made almost no update.

For the signed diagonal method, the matched AUC difference versus Standard ES
was:

| Initial alpha | Paired difference | Wins |
| ---: | ---: | ---: |
| 0.25 | -76.72 +/- 24.40 | 0/5 |
| 1 | -31.04 +/- 41.33 | 0/5 |
| 3 | -8.40 +/- 18.09 | 2/5 |
| 10 | +4.46 +/- 11.98 | 3/5 |
| 30 | -1.66 +/- 9.90 | 2/5 |

The `alpha_0=10` mean gain is not robust: its paired standard deviation exceeds
the gain, the exact two-sided sign-flip p-value is `0.4375`, and all five final
returns are worse. Across all 25 matched cells, signed diagonal wins 7/25 on
AUC, 6/25 on return at 75,000 steps, and 1/25 on final return.

With five seeds, the smallest attainable nonzero two-sided sign-flip p-value is
`0.0625`. The experiment should be interpreted through effect consistency and
mechanism diagnostics rather than thresholded significance claims.
The return after 500 updates is not a matched-interaction endpoint: depending
on policy performance and resulting rollout lengths, runs accumulated between
778,478 and 82,750,774 training environment steps. Final-return win counts are
secondary diagnostics only.

## Why Picard Fails

Across 25 runs and 12,500 attempted generation solves:

- Converged solves: `0/12,500`
- Mean final relative residual: `1.00092`; tolerance: `1e-5`
- Solves exhausting all 10 iterations: `100%`
- Exact final period-two recurrence: `91.584%`
- Two-cycle error at most `1e-5`: `93.816%`
- Returned step at most `0.001` of the explicit step: `97.072%`
- Mean minimum endpoint `ESS/B`: `0.0050000017`, essentially one effective
  sample out of 200
- Mean maximum endpoint weight: `0.99999983`
- Median maximum relative-logit span: `26,314.7`

The mechanism is:

1. Starting from zero displacement, the first Picard map produces the large
   explicit ES step.
2. At that endpoint, importance weights collapse onto approximately one
   candidate.
3. Endpoint-weighted utility centering then makes the gradient nearly zero.
4. The next map returns near zero, after which the cycle repeats.
5. Ten iterations are even, so the applied update is usually the
   near-zero branch.

Occasional high final policies occur in nonconverged trajectories that depart
from the exact two-cycle. Because no solve met tolerance, they cannot be
credited to a solved implicit update.

## Diagonal Surrogate Versus Linear Solve

The signed diagonal arm separates estimated-system instability from arithmetic
solve failure. Its batch-rank quantity is a matched-rank diagonal
Jacobian/Hessian surrogate, not literally the Hessian of a fixed smooth
objective.

### Arithmetic solve

- Successful divisions: `12,500/12,500`
- Mean relative residual: `4.02e-17`
- Maximum relative residual: `4.91e-17`
- Non-finite records or hard singular failures: 0

The elementwise division itself is accurate.

### Estimator repeatability

- Overall split-half Hessian correlation: `0.000083`
- Overall split-half sign agreement: `0.499998`
- Consecutive-generation correlation: `0.000251`
- Consecutive-generation sign agreement: `0.500065`

Two disjoint 50-pair subsets of each batch showed essentially zero agreement.
Because rank utilities are formed from the full 200-sample population, this is
a repeatability diagnostic rather than a claim of statistically independent
estimates. Consecutive-generation agreement is also affected by changing
parameters and, in some runs, extreme updates.

### Resulting system geometry

- Median signed condition estimate: `24,797.6`
- 95th percentile: `323,039`
- Maximum: `1.171e8`
- Updates with condition above `1e6`: `217/12,500`
- Nonpositive diagonal entries: `43.524%`
- Every update contained both positive and negative diagonal entries
- Minimum absolute diagonal: `1.337e-7`

The signed system is therefore highly sensitive even though it is divided
accurately. Step norms reflect this: median `19.65`, 95th percentile `251.76`,
and maximum `91,679`; the maximum ratio to the explicit-step norm was `141,967`.

**Conclusion:** the job rules out inaccurate elementwise division and shows
unstable, ill-conditioned estimated-system geometry consistent with estimator
noise. Near-zero estimated denominators amplify the otherwise accurate
solution into extreme updates; this experiment does not prove that sampling
noise is the only source of variation.

## Recommended Next Work

1. Do not add trust clipping to these comparisons. When fixed-radius clipping
   is active, it fixes the update norm at `R` and can dominate the scalar
   learning rate, obscuring both failure mechanisms.
2. Treat the current Picard arm as a solver-failure diagnostic. Before another
   environment sweep, test a genuine root solver on controlled quadratics and
   require the final implicit residual to pass. Solver relaxation or
   globalization must be labeled as solver machinery, not as an optimizer
   trust radius.
3. Do not repeat the coordinatewise diagonal Hessian at the same
   dimension/population ratio. First test lower-variance structured estimates
   such as directional or layer-block curvature, or substantially increase the
   independent-pair count. Require split-half agreement materially above zero
   before interpreting return curves.
4. Keep the gradient and Hessian objective matched. Raw-return curvature should
   remain a separately labeled estimator diagnostic when the gradient uses
   rank utilities.
5. Preserve Standard ES and the full step-size sequence as controls. The goal
   is improvement over Standard ES, not reaching an optimal Hopper score.

## Artifacts

- Run-level validated metrics: `results/hopper_implicit_no_replay_no_trust_inverse_sqrt_49648326/validated_runs.csv`
- Grouped validated metrics: `results/hopper_implicit_no_replay_no_trust_inverse_sqrt_49648326/validated_summary.csv`
- Validator: `scripts/summarize_hopper_implicit_sweep.py`
- Endpoint and signed update implementation: `core/implicit_es.py`

# Hopper Standard ES Versus Linearized Hessian Job 49678516

## Status

- Slurm array: `49678516`
- Submitted: July 12, 2026
- Array size: 80 tasks, at most 4 concurrent
- Completed cells: 80/80; every run contains 500 updates
- Validation: passed; all 80 Slurm stderr files are empty
- Picard: excluded from the launcher and rejected by the validator
- Source digest:
  `8f0ca464d481b0010fc7c3f992a3f943db8d510624409e64c44061c92bd968b0`
- Source archive:
  `job_outputs/hopper_hessian_source_8f0ca464d481b0010fc7c3f992a3f943db8d510624409e64c44061c92bd968b0.tar.gz`
- Archive SHA-256:
  `b39acc797eec4bff939c5aaade6f5e880cc9c69ddff2b30f4a9fe7ba96249afe`

Digest-covered source files must not be changed until every queued task has
started, because each task independently checks the digest.

## Protocol

- Environment: `Hopper-v5`
- Conditions: `standard_es`, `linearized_implicit_es`
- Initial learning rates: `10`, `30`
- Schedules: `alpha_0 / sqrt(t + 1)`, `alpha_0 / (t + 1)`
- Seeds: `0` through `9`
- Population: 200 fresh candidates, 100 antithetic pairs
- Updates: 500
- Primary metric: evaluation-return AUC over the first 75,000 training
  environment steps
- Replay, Picard, trust radius, scalar damping, gradient clipping, parameter
  projection, curvature projection, curvature clipping, and curvature EMA:
  disabled

## Task Map

| Tasks | Condition | Schedule | Initial rates and seeds |
| --- | --- | --- | --- |
| 0-19 | Standard ES | inverse square root | rate 10, seeds 0-9; rate 30, seeds 0-9 |
| 20-39 | Standard ES | inverse linear | rate 10, seeds 0-9; rate 30, seeds 0-9 |
| 40-59 | Linearized Hessian | inverse square root | rate 10, seeds 0-9; rate 30, seeds 0-9 |
| 60-79 | Linearized Hessian | inverse linear | rate 10, seeds 0-9; rate 30, seeds 0-9 |

Within every condition/schedule block, the initial-rate index changes every ten
tasks and the seed is the fastest-changing index.

## Validation

After all 80 tasks finish, run:

```bash
python scripts/summarize_hopper_implicit_sweep.py \
  results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_49678516 \
  --expected-source-sha 8f0ca464d481b0010fc7c3f992a3f943db8d510624409e64c44061c92bd968b0
```

The validator requires the complete matrix and rejects endpoint/Picard cells,
wrong task mappings, source drift, either schedule drifting from its formula,
replay, trust or norm controls, nonfinite records, incomplete histories, and
invalid signed-system arithmetic.

The completed validation prints:

```text
Validated 80 runs; wrote .../validated_runs.csv, .../validated_summary.csv,
and .../paired_contrasts.json
```

## Final Result: No Robust Improvement Over Standard ES

The primary metric is mean evaluation-return AUC through 75,000 actual
training environment steps. Values before the paired difference are the
ten-seed mean +/- sample standard deviation. The paired difference is signed
diagonal curvature minus Standard ES, also reported as mean +/- sample
standard deviation.

| Schedule | Initial alpha | Standard ES | Signed diagonal curvature | Paired difference | Wins-losses | Exact sign-flip p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alpha_0 / sqrt(t + 1)` | 10 | 18.480 +/- 8.572 | 26.802 +/- 35.536 | +8.322 +/- 41.019 | 4-6 | 0.8223 |
| `alpha_0 / sqrt(t + 1)` | 30 | 16.260 +/- 9.434 | 16.822 +/- 12.958 | +0.563 +/- 10.984 | 6-4 | 0.8945 |
| `alpha_0 / (t + 1)` | 10 | 22.879 +/- 18.155 | 21.898 +/- 16.620 | -0.981 +/- 28.812 | 4-6 | 0.9219 |
| `alpha_0 / (t + 1)` | 30 | 21.241 +/- 12.535 | 16.798 +/- 9.582 | -4.444 +/- 15.313 | 5-5 | 0.3691 |

No cell shows a consistent paired improvement. In particular, the largest
positive mean, `+8.322` at `alpha_0 = 10` under the inverse-square-root
schedule, has a paired median of `-8.820`, only four wins in ten seeds, and an
exact two-sided sign-flip p-value of `0.8223`. Seed 9 contributes a `+119.661`
outlier to that mean. The experiment therefore does not establish that signed
diagonal curvature improves Standard ES for either decreasing sequence.

The exact paired seed differences for every performance metric are saved in
`results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_49678516/paired_contrasts.json`.

## Why the Curvature Arm Fails

The elementwise solve is arithmetically stable, but the estimated system is
not statistically stable:

| Diagnostic across the 40 curvature runs | Result |
| --- | ---: |
| Maximum relative division residual | `4.94e-17` |
| Mean split-half curvature correlation | `-4.53e-5` |
| Mean split-half coordinate sign agreement | `0.49989` |
| Mean temporal curvature correlation | `-6.85e-5` |
| Mean temporal coordinate sign agreement | `0.49903` |
| Smallest observed absolute denominator | `6.29e-9` |
| Median run-level signed condition estimate | `24,903` |
| Largest update/explicit-step norm ratio | `1,064,336` |

Correlations near zero and sign agreement near one half mean that independent
halves of the same population and adjacent generations effectively disagree
at chance level. Near-zero values of `1 - alpha_t * h_j` then amplify the
accurately divided update. This supports an estimator-induced
ill-conditioning diagnosis, not an inaccurate linear-system arithmetic
diagnosis.

The paper-facing name for this arm is **signed diagonal frozen-rank
curvature surrogate**. It is not a raw-return Hessian, and the negative result
does not show that all curvature-informed ES methods fail.

## Mentor-Facing Conclusion

The old trust-region plots should not be used as evidence because active
fixed-radius clipping largely reparameterized the learning rate as the radius.
This job is the clean replacement: it uses decreasing learning rates without
trust rescaling and directly compares the basic curvature arm to Standard ES.
It shows no robust performance gain and locates the immediate failure in the
high-variance curvature estimate and resulting system geometry. The goal here
is comparison to Standard ES and mechanism diagnosis, not reaching an optimal
Hopper return.

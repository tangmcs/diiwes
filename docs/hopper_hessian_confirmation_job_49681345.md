# Hopper Hessian Confirmation Job 49681345

## Status

- Slurm array: `49681345`
- Submitted: July 12, 2026
- Array size: 30 tasks, at most 6 concurrent
- Initial state: tasks `0` through `5` launched under the expected source lock
- Initial stderr: all six files empty
- Source digest:
  `16638ba846490cd2e3673813f8514d0a7f2b7b9b5ac5f632ffe77e2d11b959f2`
- Source archive:
  `job_outputs/hopper_hessian_confirmation_source_16638ba846490cd2e3673813f8514d0a7f2b7b9b5ac5f632ffe77e2d11b959f2.tar.gz`
- Archive SHA-256:
  `1c6234ead44f67ce8a215bf8706ed8875f3e1bff15f0d1db8f669e62c1d5f6e0`

The source digest includes the optimizer, trainer, locked config, launcher,
validator/analyzer, and preregistration. Covered files must remain unchanged
until every queued task has started.

## Protocol

- Arms: Standard ES, structured concave block-EMA curvature, and isotropic
  curvature-derived attenuation control
- Seeds: untouched paired seeds `100` through `109`
- Learning rate: `10 / (t + 1)`
- Population: 200 fresh candidates, 100 antithetic pairs
- Updates: 500, without early stopping
- Primary endpoint: post-training held-out AUC through 75,000 actual training
  steps, using 20 independent episodes per saved center
- Replay, sample importance weighting, Picard, trust, scalar damping, L2,
  gradient/parameter clipping or projection, and curvature clipping: disabled

Within every adjacent three-task seed block, the condition order rotates:

| Seed-index modulo 3 | Task-slot order |
| --- | --- |
| 0 | Standard, structured, isotropic |
| 1 | Structured, isotropic, Standard |
| 2 | Isotropic, Standard, structured |

The complete scientific decision rule is frozen in
`docs/hopper_hessian_confirmation_preregistration.md` and covered by the source
digest.

## Validation

After all 30 tasks finish, run:

```bash
python scripts/summarize_hopper_hessian_confirmation.py \
  results/hopper_hessian_confirmation_49681345 \
  --expected-source-sha 16638ba846490cd2e3673813f8514d0a7f2b7b9b5ac5f632ffe77e2d11b959f2 \
  --job-output-dir job_outputs
```

The validator emits results only after the complete matrix, source/config/task
locks, zero-byte stderr, fresh-only diagnostics, solver/control invariants, and
held-out artifacts all pass. It then computes the two preregistered exact
paired sign-flip tests, applies Holm correction, and reports the joint claim
flag.

## Final Validated Result

All 30 tasks passed the locked validator. The validator reported:

```text
Validated 30 confirmation runs; claim_supported=False
```

The exact preregistered primary contrasts were:

| Contrast | Mean | Median | Sample SD | 95% paired-mean t interval | Wins-losses-ties | Raw exact sign-flip p | Holm p | Reject at 0.05? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Structured block-EMA minus Standard ES | 1.41483102745371 | 2.274387169850158 | 11.714193351766125 | [-6.964998084395898, 9.794660139303318] | 7-3-0 | 0.70703125 | 0.70703125 | No |
| Structured block-EMA minus isotropic attenuation | 2.751514037559521 | 1.3020385453028478 | 7.6215581311726766 | [-2.7006202056938333, 8.203648280812875] | 6-4-0 | 0.27734375 | 0.5546875 | No |

The paired differences in seed order `100` through `109` are:

```text
block_ema_minus_standard =
[3.567281444559516, 0.9814928951408, 15.180461984930275,
 0.8126197570633567, 4.251738414622443, 7.958975132590186,
 -9.857979244110984, -11.869835458930982, 20.08000095354186,
 -16.95644560486937]

block_ema_minus_isotropic =
[-2.0755027344835515, 6.035236768914816, 13.111280185481428,
 -1.893396241682428, 0.8971058642049794, 1.7069712264007162,
 -10.864103836201933, 14.477983923882586, 7.071973944952873,
 -0.9524087258742746]
```

The two-sided sign-flip tests counted `724 / 1024` and `284 / 1024`
assignments at least as extreme, respectively.

The decision rule required both paired means to be positive and both
Holm-adjusted p-values to be below `0.05`. Although both means were positive,
neither adjusted test rejected and both confidence intervals include zero.
Therefore, the preregistered joint positive claim is not supported.

This result does not prove that the structured treatment has no effect. It
does show that the exploratory five-seed advantage did not establish a
replicable improvement under the frozen confirmation, and it provides no
confirmatory basis for claiming that layer-specific attenuation is superior
to scalar norm-matched attenuation. The stable denominator invariant remains
a mechanical result, not an optimization-superiority result.

The paper-facing term for the matched-rank statistic should be
`frozen-rank covariance-score curvature surrogate`, not a literal Hessian.
Historical artifact and code names retain `hessian` for compatibility.

The complete machine-readable outputs are:

- `results/hopper_hessian_confirmation_49681345/validated_confirmation_runs.csv`
- `results/hopper_hessian_confirmation_49681345/confirmation_primary_contrasts.json`

The paired differences are retained verbatim in the JSON. No secondary metric
can rescue the failed primary decision under the preregistration.

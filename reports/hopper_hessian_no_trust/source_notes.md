# Mentor Report Source Notes

## Reporting job

- Question: Does the basic signed diagonal curvature update improve Standard ES
  when trust-radius rescaling is removed and large decreasing learning rates are
  used?
- Audience: technical (mentor/advisor).
- Scope: Hopper job `49678516`, completed July 12, 2026.
- Baseline: Standard ES under the identical schedule, initial rate, and seed.
- Primary metric: online evaluation-return AUC through 75,000 actual training
  environment steps.
- Decision rule for this diagnostic report: look for a seed-consistent paired
  improvement and use mechanism diagnostics to distinguish estimator failure
  from arithmetic solve failure. This was not a preregistered confirmatory
  family, so cellwise p-values remain descriptive.

## Evidence inventory

- `results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_49678516/validated_runs.csv`
  contains the 80 validated run rows and run-level mechanism diagnostics.
- `results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_49678516/validated_summary.csv`
  contains the eight condition/schedule/rate group summaries.
- `results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_49678516/paired_contrasts.json`
  contains exact seed-paired differences and sign-flip tests produced by the
  repository validator/summarizer.
- `docs/hopper_hessian_job_49678516.md` records the locked protocol, source
  digest, final result, and interpretation.
- `docs/experiment_diagnosis.md` records why the earlier trust-region plots are
  excluded.

The report is a post-run presentation artifact. The training result remains
tied to archived launch digest
`8f0ca464d481b0010fc7c3f992a3f943db8d510624409e64c44061c92bd968b0`.

## Required-structure mapping

The technical-report specification maps to visible sections as follows:

1. Title: report title block.
2. Technical summary: `Technical summary` plus the headline metric strip.
3. Key findings with visual evidence: the mean-AUC chart, paired-distribution
   chart, exact paired table, and mechanism table.
4. Scope, data, and definitions: `Scope and metric definitions`.
5. Methodology: `Experimental controls isolate curvature from step-norm
   rescaling`.
6. Limitations and robustness: `Limitations and robustness boundary`.
7. Recommended next steps: `Recommended next steps`.
8. Further questions: `Further questions`.

No required role is omitted or merged away.

## Chart map and contracts

### Mean AUC comparison

- Analytical question: How do mean primary-metric values compare across method,
  schedule, and initial rate?
- Takeaway: curvature is higher in the two inverse-square-root cells and lower
  in the two inverse-linear cells, with no consistent pattern.
- Family/type: comparison, grouped `bar`.
- Data: eight aggregate rows; four schedule/rate categories and two methods.
- Fields: category=`cell`, value=`auc_mean`, meaningful second
  dimension=`method`; `auc_sd` and `n_seeds` remain in tooltips.
- Scale: zero-based absolute magnitude.
- Palette policy: hard two-root cap for focal method versus baseline, with
  method labels and grouped position providing non-color distinction.
- Final surface: full-width native chart in the portable HTML report.

### Paired AUC distribution

- Analytical question: Are method differences consistent across paired seeds,
  or are means driven by outliers?
- Takeaway: all cells straddle zero; the largest positive mean is driven by one
  extreme seed.
- Family/type: distribution across groups, `boxPlot`.
- Data: 40 seed-level paired differences, ten per schedule/rate cell.
- Fields: group=`cell`, value=`paired_auc_difference`; seed retained for
  tooltip/audit context.
- Scale: signed common scale; positive favors curvature and negative favors
  Standard ES.
- Palette policy: single-root preferred; sign and zero context, not color,
  carry the comparison.
- Final surface: full-width native chart in the portable HTML report.

The repeated categorical x-axis is intentional: the two charts answer
different questions (absolute method means versus paired-seed distribution).

## Calculation and validation notes

- The validator independently re-read 80 configurations, 80 status files,
  40,000 JSON history records and matching JSONL records before regenerating
  the CSV/JSON summaries.
- Paired differences are always `curvature - Standard ES` for the same seed,
  schedule, and initial rate.
- The exact two-sided sign-flip test enumerates all signs of the nonzero paired
  differences and uses the absolute paired sum as its statistic.
- Across-seed spread uses sample standard deviation (`ddof=1`).
- `best_return` is not a primary result because it selects over repeated online
  evaluations. Final return is also secondary because 500 updates correspond
  to unequal environment-step totals. The report therefore leads with AUC at a
  fixed training-step budget.
- No new positive performance claim is made. The strongest supported claim is
  a failure diagnosis: division arithmetic is accurate, while curvature
  repeatability and signed-system conditioning are poor.

## Remaining limitations

- Online evaluation uses three fixed episodes rather than independent held-out
  evaluation.
- The study covers one environment and one policy dimension/population ratio.
- Very large updates confound estimator noise with local-linearization error.
- The HTML report is a snapshot, not a live view of future experiment outputs.

## Delivery QA

The portable-report builder passed artifact validation, packaging, exact
payload equality, required reader-root checks, and semantic-fallback checks.
Its receipt reported `verification=structural_only` because no compatible
Chromium headless-shell executable is installed on this host. Consequently,
chart SVG extraction, source-dialog interaction, and desktop/narrow viewport
browser checks were not run. The delivered HTML retains the builder-generated
semantic chart tables and all narrative/table evidence.

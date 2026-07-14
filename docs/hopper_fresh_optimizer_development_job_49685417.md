# Hopper Fresh-Only Optimizer Development Job 49685417

## Status

- Slurm array: `49685417`
- Submitted: July 12, 2026
- Array size: 99 tasks, at most 6 concurrent
- Designation: exploratory development and locality calibration only
- Final status: all 99 tasks completed 250 updates; all stderr files were empty
- Strict analyzer: passed all 99 task/cell/seed mappings and invariants from
  the immutable source snapshot

This job cannot support a confirmatory or environment-transition
sample-efficiency claim. It fixes 50,000 candidate-policy rollouts per run;
actual training-transition counts range from `1,059,423` to `41,715,971`
(`39.38x`) because episode lengths vary.

## Immutable Source

- Git revision recorded for context:
  `6177d480521971b52372e016f9d4bb9c2cefbdca`
- Source digest:
  `5b467f867e071ebecfbc89e8e39417f8bfbb45230267ff88b1aa8e177f66d7bb`
- Manifest digest:
  `68df7a8e3d86cb800f08712652f22f8ad07b684e214256b0088fd2f1d0237f57`
- Launcher digest:
  `fa686122681315241878bdce1eeaf0678f7abcb52cfd77af5a42756fddeb574f`
- Analyzer digest:
  `1c00d5a3a02df1af6af3d549815a364a795788e743064e027665c006100dc4af`
- Source archive:
  `job_outputs/hopper_fresh_optimizer_development_source_5b467f867e071ebecfbc89e8e39417f8bfbb45230267ff88b1aa8e177f66d7bb.tar.gz`
- Archive SHA-256:
  `7864af92c56058cd4e4f4e102885acd7f1e1c8115f2f9d6e5f4b5b5790dcf62e`

The extracted snapshot reproduced all four study hashes and passed all 148
tests before submission. Representative snapshot dry runs covered all seven
optimizer families. Live one-generation MuJoCo smoke tests completed for
ClipUp and OLS confidence-adjusted shrinkage with fresh-only diagnostics.

## Protocol

- Environment: `Hopper-v5`
- Development seeds: `200`, `201`, `202`
- Population: 200 fresh candidates, 100 antithetic pairs
- Updates: 250
- Search scale: `sigma=0.02`
- Arms: plain rank ES, Momentum ES, Adam ES, ClipUp, structured block-EMA
  attenuation, norm-matched isotropic attenuation, and exploratory block-OLS
  confidence-adjusted shrinkage
- Replay, cross-generation importance sampling, Picard iteration, trust
  radius, gradient clipping, parameter projection, curvature clipping, scalar
  damping, and L2: disabled
- ClipUp retains its published internal velocity clipping as an external
  baseline, not as part of the curvature method

The manifest spans sub-`sigma`, near-`sigma`, and nonlocal update settings.
The analyzer reports first, mean, and maximum `||Delta_t||/sigma` plus the
fraction of updates with `||Delta_t|| <= sigma`.

## Final Validation Command

Run the analyzer from the immutable snapshot after every task completes:

```bash
SNAPSHOT=job_outputs/source_snapshots/hopper_fresh_optimizer_development_5b467f867e071ebecfbc89e8e39417f8bfbb45230267ff88b1aa8e177f66d7bb

python "$SNAPSHOT/scripts/summarize_hopper_fresh_optimizer_development.py" \
  results/hopper_fresh_optimizer_development_49685417 \
  --manifest "$SNAPSHOT/experiments/manifests/hopper_fresh_optimizer_development.json" \
  --launcher "$SNAPSHOT/scripts/submit_hopper_fresh_optimizer_development.sh" \
  --expected-source-sha 5b467f867e071ebecfbc89e8e39417f8bfbb45230267ff88b1aa8e177f66d7bb \
  --expected-manifest-sha 68df7a8e3d86cb800f08712652f22f8ad07b684e214256b0088fd2f1d0237f57 \
  --expected-launcher-sha fa686122681315241878bdce1eeaf0678f7abcb52cfd77af5a42756fddeb574f
```

The analyzer must validate all 99 mappings before writing any aggregate. Its
paired structured-minus-isotropic contrasts are descriptive and contain no
p-values or claim-selection rule.

## Final Validated Outputs

- Per-run CSV: `results/hopper_fresh_optimizer_development_49685417/validated_development_runs.csv`
  (`ab5be68904f1d7c8dbc65d4defb3d902f6edaae2e88704705d7682083477d7de`)
- Grouped CSV: `results/hopper_fresh_optimizer_development_49685417/development_grouped_summary.csv`
  (`3f750c6e59c70db9f2ba3944b7a4959c3ed928ce3549646107560f98aeeccf7e`)
- Summary JSON: `results/hopper_fresh_optimizer_development_49685417/development_summary.json`
  (`4c872523ae439a27c21b8c117f0396abc43c3b7d1d6cf4310fbdc2e6dc8d4f0f`)

Two analyzer runs produced the same three hashes. The JSON records
`confirmatory_analysis_performed=false`, `p_values_computed=false`, and
`claim_selection_performed=false`.

## Exploratory Result

The block-curvature method did not identify a regime that was both local and
materially different from explicit ES:

- At learning rates `3e-5` and `1e-4`, every structured update was within one
  `sigma`, but mean step attenuation was only `0.000709%` and `0.002337%`.
- At rates `3e-4`, `1e-3`, `3e-3`, and `3e-2`, mean update sizes were `1.05`,
  `3.20`, `9.65`, and `124.64` times `sigma`; mean attenuation was still only
  `0.00739%`, `0.0307%`, `0.1018%`, and `0.3479%`.
- Across the six structured cells, mean split-half curvature correlation was
  between `-0.048` and `0.078`, sign agreement was between `0.489` and
  `0.522`, and relative disagreement exceeded `1.05`.

The paired structured-minus-isotropic contrast was not robust. At `alpha=.003`,
best-return differences were `[+0.436, -0.485, +509.523]`; the positive mean
was driven by one seed while updates averaged `9.65 sigma`. At `alpha=.03`,
structured curvature lost on all three seeds: mean best-return difference
`-654.0` and mean final-return difference `-991.9`, with updates averaging
`124.64 sigma`.

Against Standard ES at `alpha=.003`, the median paired differences were only
`+0.168` AUC, `+0.739` final return, and `-0.011` best return; the group mean
again came from one large seed-specific deviation. At `alpha=.03`, structured
curvature's final return was lower on every seed by `246` to `1277` points.

The highest mean AUC in this screen was Momentum ES at `alpha=.003`
(`1648.1`, final return `2420.7`), but its updates averaged `24.38 sigma` and
the tuning budgets differed by optimizer family. This is a calibration signal,
not confirmatory evidence.

The defensible conclusion is negative: arithmetic solves were stable, but the
three-block curvature statistic was unreliable and too weak to mitigate step
size in the local regime. This job does not support a curvature-superiority,
optimizer-superiority, or transition-sample-efficiency claim.

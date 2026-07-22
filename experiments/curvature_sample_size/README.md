# Curvature estimator sample-size study

This folder isolates the relationship between antithetic sample size and the
diagonal curvature estimate returned by the repository's current DIIWES
implementation.

The three controlled cases are:

- `nonlinear_nonconvex`: sine + quartic + adjacent-coordinate coupling, with
  an exact Gaussian-smoothed diagonal Hessian;
- `linear_deterministic`: a noiseless linear function with exactly zero
  curvature;
- `linear_noisy`: the same zero-curvature linear function with independent
  observation noise.

Run the locked default protocol from the repository root:

```bash
python -m experiments.curvature_sample_size.benchmark
```

This writes raw repetitions, coordinate estimates, aggregates, convergence
rates, a decision summary, provenance, and the canonical report payload to
`reports/curvature_sample_size/`. Sample size means antithetic pairs, so `N`
pairs require `2N` function evaluations.

For a fast code-path check:

```bash
python -m experiments.curvature_sample_size.benchmark --quick \
  --output-dir /tmp/curvature_sample_size_smoke
```

The benchmark intentionally calls
`DIIWES._estimate_fresh_curvature`; it does not copy the estimator formula.

## Dimension sweep

To compare the sample-size relationship at 100, 1,000, and 2,000
dimensions, run:

```bash
python -m experiments.curvature_sample_size.dimension_sweep
```

The default sweep uses the same pair-count grid as the original benchmark and
20 independent repetitions. It writes compact run metrics, aggregates,
convergence fits, and a two-plot report payload to
`reports/curvature_sample_size/dimension_sweep/`. Coordinate bias is
accumulated online instead of writing a per-coordinate CSV, which keeps the
2,000-dimensional run memory-safe.

The report contains:

- absolute nonlinear curvature RMSE versus antithetic pair count, with one
  curve per dimension; and
- within-dimension normalized RMSE versus pair count, including an
  inverse-square-root reference.

Use `--dimensions`, `--pair-counts`, and `--repetitions` to change the
protocol, or `--quick` for a small code-path check.

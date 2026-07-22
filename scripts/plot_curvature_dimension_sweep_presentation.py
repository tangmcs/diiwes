#!/usr/bin/env python3
"""Create PPT-ready figures and source material for the dimension sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / f"diiwes-mpl-{os.getuid()}"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402


COLORS = {
    100: "#2563EB",
    1000: "#D97706",
    2000: "#6B7F2A",
}
MARKERS = {100: "o", 1000: "s", 2000: "^"}
NEUTRAL = "#6B7280"
INK = "#1F2937"
GRID = "#D1D5DB"


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_figure(fig: Any, path: Path, *, dpi: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )


def _style_axis(ax: Any, pair_counts: Sequence[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(pair_counts)
    ax.set_xticklabels([f"{value:,}" for value in pair_counts])
    ax.grid(True, which="major", axis="both", color=GRID, linewidth=0.8, alpha=0.7)
    ax.grid(False, which="minor")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=INK, labelsize=10)
    ax.set_axisbelow(True)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            "lines.linewidth": 2.4,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _nonlinear_rows(aggregate_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    selected = []
    for row in aggregate_rows:
        if row["case"] != "nonlinear_nonconvex":
            continue
        selected.append(
            {
                "dimension": int(row["dimension"]),
                "pair_count": int(row["pair_count"]),
                "function_evaluations": int(row["function_evaluations"]),
                "pairs_per_dimension": float(row["pairs_per_dimension"]),
                "repetitions": int(row["repetitions"]),
                "median_rmse": float(row["median_rmse"]),
                "q25_rmse": float(row["q25_rmse"]),
                "q75_rmse": float(row["q75_rmse"]),
                "median_relative_l2_error": float(row["median_relative_l2_error"]),
                "mean_sign_accuracy": float(row["mean_sign_accuracy"]),
                "target_rms": float(row["target_rms"]),
            }
        )
    return sorted(selected, key=lambda row: (row["dimension"], row["pair_count"]))


def _plot_absolute(
    rows: Sequence[dict[str, Any]],
    model: dict[str, Any],
    output_dir: Path,
) -> list[Path]:
    pair_counts = sorted({row["pair_count"] for row in rows})
    fig, ax = plt.subplots(figsize=(10, 5.625))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.82)
    for dimension in sorted({row["dimension"] for row in rows}):
        selected = [row for row in rows if row["dimension"] == dimension]
        x = [row["pair_count"] for row in selected]
        y = [row["median_rmse"] for row in selected]
        low = [row["q25_rmse"] for row in selected]
        high = [row["q75_rmse"] for row in selected]
        color = COLORS[dimension]
        ax.fill_between(x, low, high, color=color, alpha=0.13, linewidth=0)
        ax.plot(
            x,
            y,
            color=color,
            marker=MARKERS[dimension],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"d = {dimension:,}",
        )
    _style_axis(ax, pair_counts)
    ax.set_yscale("log")
    ax.set_xlabel("Antithetic pair count, N (log scale)")
    ax.set_ylabel("Median coordinate RMSE (log scale)")
    ax.set_title("Curvature-estimation error by dimension", loc="left", pad=18)
    ax.text(
        0.0,
        1.01,
        "Nonlinear objective; shaded bands show the interquartile range across 20 repetitions",
        transform=ax.transAxes,
        fontsize=10.5,
        color=NEUTRAL,
        va="bottom",
    )
    alpha = model["dimension_exponent_alpha"]
    beta = model["sample_size_exponent_beta"]
    r_squared = model["r_squared"]
    ax.text(
        0.025,
        0.055,
        rf"Pooled fit: RMSE $\propto d^{{{alpha:.2f}}}N^{{{beta:.2f}}}$   ($R^2={r_squared:.4f}$)",
        transform=ax.transAxes,
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": GRID},
    )
    ax.legend(title="Dimension", loc="upper right", frameon=False, ncol=1)
    fig.text(
        0.01,
        0.025,
        "Source: DIIWES diagonal Stein estimator; antithetic pairs; sigma = 0.1.",
        fontsize=9,
        color=NEUTRAL,
    )
    png = output_dir / "figures" / "curvature_rmse_by_dimension.png"
    svg = output_dir / "figures" / "curvature_rmse_by_dimension.svg"
    _save_figure(fig, png, dpi=300)
    _save_figure(fig, svg)
    plt.close(fig)
    return [png, svg]


def _plot_normalized(rows: Sequence[dict[str, Any]], output_dir: Path) -> list[Path]:
    pair_counts = sorted({row["pair_count"] for row in rows})
    first_pairs = min(pair_counts)
    baselines = {
        row["dimension"]: row["median_rmse"]
        for row in rows
        if row["pair_count"] == first_pairs
    }
    fig, ax = plt.subplots(figsize=(10, 5.625))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.19, top=0.82)
    for dimension in sorted({row["dimension"] for row in rows}):
        selected = [row for row in rows if row["dimension"] == dimension]
        x = [row["pair_count"] for row in selected]
        y = [row["median_rmse"] / baselines[dimension] for row in selected]
        ax.plot(
            x,
            y,
            color=COLORS[dimension],
            marker=MARKERS[dimension],
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.5,
            label=f"d = {dimension:,}",
        )
    reference = [math.sqrt(first_pairs / pair_count) for pair_count in pair_counts]
    ax.plot(
        pair_counts,
        reference,
        color=NEUTRAL,
        linestyle="--",
        linewidth=2,
        label=r"$N^{-1/2}$ reference",
    )
    _style_axis(ax, pair_counts)
    ax.set_yscale("log")
    ax.set_xlabel("Antithetic pair count, N (log scale)")
    ax.set_ylabel(f"RMSE / RMSE at N = {first_pairs} (log scale)")
    ax.set_title("Normalized curvature-estimation error", loc="left", pad=18)
    ax.text(
        0.0,
        1.01,
        "All dimensions follow nearly the same inverse-square-root sample-size trend",
        transform=ax.transAxes,
        fontsize=10.5,
        color=NEUTRAL,
        va="bottom",
    )
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.text(
        0.01,
        0.025,
        "Source: within-dimension normalization of median coordinate RMSE; 20 repetitions.",
        fontsize=9,
        color=NEUTRAL,
    )
    png = output_dir / "figures" / "curvature_rmse_normalized.png"
    svg = output_dir / "figures" / "curvature_rmse_normalized.svg"
    _save_figure(fig, png, dpi=300)
    _save_figure(fig, svg)
    plt.close(fig)
    return [png, svg]


def _write_plot_data(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "dimension",
        "pair_count",
        "function_evaluations",
        "pairs_per_dimension",
        "repetitions",
        "median_rmse",
        "q25_rmse",
        "q75_rmse",
        "median_relative_l2_error",
        "mean_sign_accuracy",
        "target_rms",
        "normalized_rmse",
    )
    baselines = {
        row["dimension"]: row["median_rmse"]
        for row in rows
        if row["pair_count"] == min(item["pair_count"] for item in rows)
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "normalized_rmse": row["median_rmse"] / baselines[row["dimension"]],
                }
            )


def _slide_source(
    rows: Sequence[dict[str, Any]],
    rates: Sequence[dict[str, str]],
    model: dict[str, Any],
) -> str:
    dimensions = sorted({row["dimension"] for row in rows})
    first_pairs = min(row["pair_count"] for row in rows)
    last_pairs = max(row["pair_count"] for row in rows)
    rate_by_dimension = {
        int(row["dimension"]): float(row["log_log_slope"])
        for row in rates
        if row["case"] == "nonlinear_nonconvex"
    }
    by_key = {(row["dimension"], row["pair_count"]): row for row in rows}
    alpha = model["dimension_exponent_alpha"]
    beta = model["sample_size_exponent_beta"]
    constant_error_exponent = alpha / -beta

    table_rows = []
    for dimension in dimensions:
        first = by_key[(dimension, first_pairs)]
        last = by_key[(dimension, last_pairs)]
        table_rows.append(
            f"| {dimension:,} | {first['median_rmse']:.3f} | "
            f"{last['median_rmse']:.3f} | {last['median_relative_l2_error']:.3f} | "
            f"{last['mean_sign_accuracy']:.1%} | {rate_by_dimension[dimension]:.3f} |"
        )
    rows_text = "\n".join(table_rows)
    return f"""# PPT source: curvature sample size across dimensions

## One-sentence result

Curvature-estimation RMSE scales approximately as
**dimension^0.48 / sqrt(sample size)**, so maintaining comparable accuracy
requires increasing antithetic pair count roughly in proportion to dimension.

## Slide 1 — Main result

**Suggested title:** Curvature estimation follows an approximately sqrt(d/N) law

**Suggested bullets:**

- Pooled fit: RMSE is proportional to d^{alpha:.3f} N^{beta:.3f} (R²={model['r_squared']:.4f}).
- Holding error constant implies N is proportional to d^{constant_error_exponent:.3f}, approximately linear scaling.
- Per-dimension sample-size slopes are all close to the Monte Carlo benchmark of -0.5.

**Figure:** `figures/curvature_rmse_by_dimension.svg` for editable vector use,
or the matching 300-DPI PNG.

## Slide 2 — Same convergence rate, different error level

**Suggested title:** More samples help equally, but high dimension starts noisier

**Suggested bullets:**

- After normalization at N={first_pairs}, all three dimensions track the inverse-square-root reference.
- At a fixed N, absolute RMSE increases strongly with dimension.
- A fixed curvature population should therefore not be reused unchanged across model sizes.

**Figure:** `figures/curvature_rmse_normalized.svg` for editable vector use,
or the matching 300-DPI PNG.

## Slide 3 — Exact endpoint values

| Dimension | RMSE at N={first_pairs} | RMSE at N={last_pairs:,} | Relative error at N={last_pairs:,} | Sign accuracy | Fitted N slope |
| ---: | ---: | ---: | ---: | ---: | ---: |
{rows_text}

**Interpretation:** d=100 reaches relative L2 error below 1 at N=128. At
N=1,000, d=1,000 is approximately at relative error 1, while d=2,000 remains
at 1.43. This is consistent with pair count needing to scale approximately
with dimension.

## Protocol footnote

- Nonlinear non-convex synthetic objective with exact Gaussian-smoothed diagonal Hessian.
- Dimensions: 100, 1,000, and 2,000.
- Antithetic pair counts: 4, 8, 16, 32, 64, 128, 250, 500, and 1,000.
- 20 independent repetitions; sigma=0.1; each pair costs two function evaluations.
- Repository implementation: `DIIWES._estimate_fresh_curvature`, raw fitness,
  diagonal mode, leave-one-out pair baseline.

## Caveat wording for slides

This is a controlled local estimator study on one synthetic objective, not a
universal complexity law or an end-to-end optimizer result.
"""


def _presentation_brief(
    rows: Sequence[dict[str, Any]],
    rates: Sequence[dict[str, str]],
    model: dict[str, Any],
) -> str:
    """Return the canonical context document for a future PPT-building agent."""
    dimensions = sorted({row["dimension"] for row in rows})
    first_pairs = min(row["pair_count"] for row in rows)
    last_pairs = max(row["pair_count"] for row in rows)
    by_key = {(row["dimension"], row["pair_count"]): row for row in rows}
    rate_by_dimension = {
        int(row["dimension"]): float(row["log_log_slope"])
        for row in rates
        if row["case"] == "nonlinear_nonconvex"
    }
    noisy_rate_by_dimension = {
        int(row["dimension"]): float(row["log_log_slope"])
        for row in rates
        if row["case"] == "linear_noisy"
    }
    alpha = float(model["dimension_exponent_alpha"])
    beta = float(model["sample_size_exponent_beta"])
    r_squared = float(model["r_squared"])
    constant_error_exponent = alpha / -beta
    coefficient = math.exp(float(model["intercept"]))

    table_rows = []
    for dimension in dimensions:
        first = by_key[(dimension, first_pairs)]
        last = by_key[(dimension, last_pairs)]
        table_rows.append(
            f"| {dimension:,} | {first['median_rmse']:.3f} | "
            f"{last['median_rmse']:.3f} | {last['median_relative_l2_error']:.3f} | "
            f"{last['mean_sign_accuracy']:.1%} | {rate_by_dimension[dimension]:.3f} |"
        )
    rows_text = "\n".join(table_rows)
    nonlinear_slopes = ", ".join(
        f"d={dimension:,}: {rate_by_dimension[dimension]:.3f}"
        for dimension in dimensions
    )
    noisy_slopes = ", ".join(
        f"d={dimension:,}: {noisy_rate_by_dimension[dimension]:.3f}"
        for dimension in dimensions
    )

    context = r"""# Presentation brief: curvature sample size across dimensions

## How to use this document

This is the canonical context for creating presentation slides about the
curvature sample-size experiment. Build the deck from this document together
with the figures in `figures/`. Use the supplied values and wording; do not
recompute numbers from the plotted pixels. Prefer the SVG files for editable
PowerPoint graphics and the PNG files when a raster image is needed.

## Research context

DIIWES is an evolution-strategy optimizer that augments the usual ES gradient
estimate with a diagonal Stein curvature estimate. The curvature estimate is
used by the broader optimizer to form curvature-aware, semi-implicit updates.
Before interpreting optimizer-level results, we need to understand how many
random perturbation pairs are required for the diagonal curvature estimate to
be reliable, especially when the parameter dimension is large.

This study isolates the estimator from the rest of the optimizer. It asks a
single controlled question:

> How does diagonal curvature-estimation error change with antithetic sample
> size as dimension increases from 100 to 1,000 and 2,000?

The intended presentation takeaway is not that one fixed population size is
universally sufficient. The intended takeaway is that the sample-size rate is
stable across dimensions while the absolute error level grows with dimension.

## Controlled objective and exact target

The nonlinear test objective is

```text
f(x) = sum_i [sin(omega_i x_i) + 0.05 x_i^4]
       + 0.08 sum_i x_i x_(i+1).
```

- The evaluation point is `theta = linspace(-1, 1, dimension)`.
- Frequencies are `omega = linspace(0.7, 1.9, dimension)`.
- Gaussian perturbation scale is `sigma = 0.1`.
- The adjacent-coordinate term makes the objective non-separable.
- The sine and quartic terms make it nonlinear and locally non-convex.

The exact diagonal Hessian after isotropic Gaussian smoothing is available in
closed form:

```text
h_i = -omega_i^2 exp(-0.5 omega_i^2 sigma^2) sin(omega_i theta_i)
      + 0.6 (theta_i^2 + sigma^2).
```

This analytic target lets us measure estimator error without substituting a
numerical approximation for the truth.

## Curvature estimator being tested

For antithetic direction `epsilon_k`, define the paired objective sum

```text
s_k = f(theta + sigma epsilon_k) + f(theta - sigma epsilon_k).
```

The leave-one-out baseline is

```text
b_k = mean of s_l over all l != k.
```

The tested diagonal Stein estimate is

```text
h_hat_j = (1/N) sum_k [(s_k - b_k)(epsilon_(k,j)^2 - 1)] / (2 sigma^2).
```

The benchmark calls the checked-out repository implementation directly:
`DIIWES._estimate_fresh_curvature`. It uses raw objective values, antithetic
pairs, diagonal curvature mode, and the leave-one-out pair baseline. The
experiment does not duplicate or replace the estimator formula.

## Experimental setting

| Item | Setting |
| --- | --- |
| Dimensions | 100, 1,000, 2,000 |
| Antithetic pair counts N | 4, 8, 16, 32, 64, 128, 250, 500, 1,000 |
| Function evaluations | 2N per estimate |
| Independent repetitions | 20 per dimension |
| Perturbation distribution | Standard Gaussian directions |
| Perturbation scale | sigma = 0.1 |
| Curvature fitness | Raw objective value |
| Curvature mode | Diagonal |
| Baseline | Leave-one-out paired-sum baseline |
| Rate-fit range | N = 32 through 1,000 |
| Master seed | 20260721 |

Within a repetition, the largest perturbation batch is generated once and
smaller sample sizes use prefixes of that batch. This common-random-number
design makes comparisons across pair counts less noisy. Repetitions remain
independent.

## Controls

The nonlinear objective is the primary accuracy test. Two linear controls
separate estimator behavior from spurious curvature:

1. **Noiseless linear control:** the true curvature is zero, and antithetic
   evaluation cancels the linear term exactly. The observed median error is
   zero at every tested dimension and sample size.
2. **Noisy linear control:** the true curvature is still zero, but independent
   Gaussian observation noise with standard deviation 0.05 breaks exact pair
   cancellation. Its error should decrease with sample size if averaging is
   working correctly.

## What we compute

For every dimension, repetition, case, and pair count:

1. Generate Gaussian antithetic perturbations.
2. Evaluate the nonlinear or linear objective at the positive and negative
   perturbations.
3. Call the repository curvature estimator.
4. Compare the estimated diagonal with the exact smoothed diagonal Hessian.
5. Record coordinate RMSE, mean absolute error, maximum absolute error,
   relative L2 error, and curvature-sign accuracy.

Across repetitions, the primary curve is the median coordinate RMSE. The
shaded bands in the absolute-error figure show the 25th to 75th percentile
range. For each dimension, we fit

```text
log(median RMSE) = intercept + slope * log(N)
```

over N >= 32. We also fit the pooled dimension/sample-size model

```text
log(median RMSE) = intercept + alpha log(dimension) + beta log(N).
```
"""

    results = f"""

## Main numerical result

The pooled nonlinear fit is

```text
RMSE approximately {coefficient:.3f} * dimension^{alpha:.3f} * N^{beta:.3f}
```

with R-squared = {r_squared:.4f}. The exponents are close to +0.5 for
dimension and -0.5 for sample size, giving the compact interpretation

```text
RMSE is approximately proportional to sqrt(dimension / N).
```

Holding RMSE constant implies `N proportional to dimension^{constant_error_exponent:.3f}`,
which is approximately linear scaling of antithetic pair count with dimension.

The nonlinear per-dimension sample-size slopes are {nonlinear_slopes}. The
noisy-linear slopes are {noisy_slopes}. Both sets are close to the Monte Carlo
reference slope of -0.5.

| Dimension | RMSE at N={first_pairs} | RMSE at N={last_pairs:,} | Relative L2 error at N={last_pairs:,} | Sign accuracy at N={last_pairs:,} | Fitted N slope |
| ---: | ---: | ---: | ---: | ---: | ---: |
{rows_text}

Operationally, d=100 first reaches median relative L2 error below 1 at N=128.
At N=1,000, d=1,000 is approximately at relative error 1.01, while d=2,000
remains at 1.43. These values reinforce the approximately proportional
dimension/sample-size relationship.
"""

    guidance = r"""

## Figure guide

### `figures/curvature_rmse_by_dimension.svg`

- **What it plots:** absolute median coordinate RMSE versus antithetic pair
  count on log-log axes, with one series per dimension.
- **Uncertainty:** shaded bands are interquartile ranges across repetitions.
- **What to say:** error decreases with sample size at every dimension, but
  the same absolute sample count is less accurate at higher dimension.
- **Visual annotation:** the pooled fitted scaling law and R-squared appear
  directly on the figure.

### `figures/curvature_rmse_normalized.svg`

- **What it plots:** each dimension's RMSE divided by its RMSE at N=4,
  compared with a neutral inverse-square-root reference.
- **What to say:** once the different starting levels are removed, all three
  dimensions have nearly the same sample-size convergence shape.
- **Why both figures are needed:** the absolute plot shows the dimension cost;
  the normalized plot shows that the convergence rate itself remains stable.

The matching PNG files are 300-DPI fallbacks. Do not use both SVG and PNG for
the same visual in one slide.

## Recommended presentation narrative

1. **Motivation:** DIIWES relies on estimated diagonal curvature; the required
   perturbation population may grow with model dimension.
2. **Controlled setup:** analytic smoothed-Hessian target, direct call to the
   repository estimator, dimensions 100/1,000/2,000, and antithetic pair sweep.
3. **Absolute result:** show `curvature_rmse_by_dimension.svg` and explain that
   higher dimension has higher error at fixed N.
4. **Rate result:** show `curvature_rmse_normalized.svg` and explain that all
   dimensions retain an approximately N^-1/2 convergence rate.
5. **Implication:** report pairs per dimension and scale curvature population
   approximately with parameter dimension when comparable estimator accuracy
   is required.
6. **Boundary:** connect this estimator result to optimizer design as a next
   question, not as an already demonstrated optimizer-performance gain.

## Claims that are supported

- Increasing antithetic pair count reduces curvature-estimation error.
- The observed sample-size slopes are approximately -0.5 at all three
  dimensions.
- Absolute error at fixed N increases approximately with the square root of
  dimension in this controlled setting.
- A roughly proportional increase in N with dimension is needed to maintain
  comparable error under the fitted relationship.
- The noiseless linear control does not manufacture curvature, and noisy
  linear error decreases at the expected averaging rate.

## Claims to avoid

- Do not call the fitted relationship a universal complexity theorem.
- Do not claim that N equal to dimension is always sufficient.
- Do not claim full-Hessian recovery; only the diagonal is estimated.
- Do not claim improved optimizer return or wall-clock efficiency from this
  experiment alone.
- Do not claim the result covers replay, importance weighting, curvature EMA,
  trust regions, damping, or policy-return noise.

## Limitations and next experiments

- One nonlinear synthetic objective and one evaluation point.
- One Gaussian smoothing scale, sigma=0.1.
- Pair counts stop at 1,000, below the largest tested dimension.
- Twenty repetitions are sufficient for the comparative curves but less
  precise for estimating small residual bias than the original 200-repetition
  12-dimensional study.
- The next useful experiments are N above 2,000, multiple smoothing scales,
  multiple evaluation points, heavy-tailed or policy-return noise, and an
  optimizer-level cost/accuracy study.

## Files available to the PPT-building agent

- `PRESENTATION_BRIEF.md`: this complete context and claim guide.
- `ppt_source.md`: compact slide-ready titles, bullets, and exact endpoint table.
- `figures/curvature_rmse_by_dimension.svg`: editable absolute-error figure.
- `figures/curvature_rmse_normalized.svg`: editable normalized-error figure.
- Matching `*.png`: 300-DPI raster fallbacks.
- `plot_data.csv`: the 27 plotted nonlinear aggregate rows.
- `aggregate.csv`: all nonlinear and linear-control aggregates.
- `convergence_rates.csv`: fitted rates for every dimension and case.
- `summary.json` and `scaling_model.json`: machine-readable headline results.
- `manifest.json`: provenance, hashes, row counts, and plotting details.
"""
    return context + results + guidance


def _readme() -> str:
    return """# Curvature sample-size presentation sources

This folder contains PPT-ready sources for the 100/1,000/2,000-dimensional
curvature sample-size sweep.

- `PRESENTATION_BRIEF.md`: canonical experimental context, methods, results,
  figure guide, supported claims, caveats, and recommended presentation flow.
- `ppt_source.md`: slide-ready result wording, exact table, protocol, and caveat.
- `figures/*.svg`: editable vector figures recommended for PowerPoint.
- `figures/*.png`: 300-DPI raster fallbacks.
- `plot_data.csv`: the 27 nonlinear aggregate rows used by both figures.
- `aggregate.csv`, `convergence_rates.csv`, `summary.json`, and
  `scaling_model.json`: copied source evidence from the validated experiment.
- `manifest.json`: hashes, sizes, row counts, source paths, and plotting runtime.

Regenerate from the repository root with:

```bash
python scripts/plot_curvature_dimension_sweep_presentation.py
```
"""


def build_package(input_dir: Path, output_dir: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    source_names = (
        "aggregate.csv",
        "convergence_rates.csv",
        "summary.json",
        "scaling_model.json",
    )
    for name in source_names:
        source = input_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing required input: {source}")
        shutil.copy2(source, output_dir / name)

    aggregate_rows = _load_csv(input_dir / "aggregate.csv")
    rates = _load_csv(input_dir / "convergence_rates.csv")
    model = _load_json(input_dir / "scaling_model.json")
    nonlinear_rows = _nonlinear_rows(aggregate_rows)
    if len(nonlinear_rows) != 27:
        raise ValueError(f"expected 27 nonlinear aggregate rows, found {len(nonlinear_rows)}")
    dimensions = sorted({row["dimension"] for row in nonlinear_rows})
    if dimensions != [100, 1000, 2000]:
        raise ValueError(f"unexpected dimensions: {dimensions}")

    _configure_plot_style()
    figure_paths = [
        *_plot_absolute(nonlinear_rows, model, output_dir),
        *_plot_normalized(nonlinear_rows, output_dir),
    ]
    plot_data = output_dir / "plot_data.csv"
    _write_plot_data(plot_data, nonlinear_rows)
    ppt_source = output_dir / "ppt_source.md"
    ppt_source.write_text(_slide_source(nonlinear_rows, rates, model), encoding="utf-8")
    presentation_brief = output_dir / "PRESENTATION_BRIEF.md"
    presentation_brief.write_text(
        _presentation_brief(nonlinear_rows, rates, model), encoding="utf-8"
    )
    readme = output_dir / "README.md"
    readme.write_text(_readme(), encoding="utf-8")

    tracked = [
        *(output_dir / name for name in source_names),
        plot_data,
        ppt_source,
        presentation_brief,
        readme,
        *figure_paths,
    ]
    file_records: dict[str, Any] = {}
    for path in tracked:
        record: dict[str, Any] = {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".csv":
            record["rows"] = len(_load_csv(path))
        file_records[path.stem if path.parent == output_dir else path.name] = record

    manifest = {
        "package_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "purpose": "PPT-ready sources for curvature sample size across dimensions",
        "source_directory": str(input_dir.relative_to(REPO_ROOT)),
        "source_experiment_manifest": str(
            (input_dir / "experiment_manifest.json").relative_to(REPO_ROOT)
        ),
        "plot_contract": {
            "question": "How does curvature-estimation error vary with pair count and dimension?",
            "takeaway": "RMSE is approximately proportional to sqrt(dimension / pair_count).",
            "chart_family": "log-log multi-series line",
            "rows": len(nonlinear_rows),
            "grain": "dimension by antithetic pair count",
            "palette": {str(key): value for key, value in COLORS.items()},
            "non_color_encoding": "distinct markers plus a dashed neutral reference",
            "footprint_inches": [10, 5.625],
            "exports": ["300-DPI PNG", "SVG"],
        },
        "plotting_runtime": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
        },
        "plot_script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "files": file_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "reports" / "curvature_sample_size" / "dimension_sweep",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "presentation" / "curvature_sample_size",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_package(args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "files": len(manifest["files"]) + 1,
                "dimensions": [100, 1000, 2000],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

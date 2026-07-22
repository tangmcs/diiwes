#!/usr/bin/env python3
"""Plot the completed LR=0.16 PPO-initialized Hopper comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diiwes_matplotlib_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


STANDARD_COLOR = "#2563EB"
CURVATURE_COLOR = "#D97706"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
BACKGROUND = "#FFFFFF"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _rolling_mean(values: np.ndarray, window: int = 20) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result[index] = (
            cumulative[index + 1] - cumulative[start]
        ) / (index + 1 - start)
    return result


def _validate_and_load(
    root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    comparison = _load_json(root / "comparison.json")
    if comparison.get("status") != "complete":
        raise ValueError("comparison report is not complete")
    protocol = comparison.get("protocol", {})
    expected = {
        "environment": "Hopper-v5",
        "seed": 0,
        "iterations": 300,
        "pairs_per_iteration": 1000,
        "population_size": 2000,
        "ema_enabled": False,
        "learning_rate": 0.16,
    }
    mismatches = {
        key: (protocol.get(key), value)
        for key, value in expected.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"protocol mismatch: {mismatches}")

    histories: dict[str, list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for condition in ("standard_es", "diag_curvature"):
        matches = list(root.glob(f"{condition}_seed0_task*"))
        if len(matches) != 1:
            raise ValueError(f"expected one {condition} run, found {len(matches)}")
        history = _load_json(matches[0] / "history.json")
        config = _load_json(matches[0] / "config.json")
        if len(history) != 300:
            raise ValueError(f"{condition} does not have 300 records")
        if [int(row["iteration"]) for row in history] != list(range(300)):
            raise ValueError(f"{condition} iteration sequence is incomplete")
        if config.get("learning_rate") != 0.16 or config.get("seed") != 0:
            raise ValueError(f"{condition} config is not the locked run")
        histories[condition] = history
        configs[condition] = config
    if configs["standard_es"]["initial_params_sha256"] != configs["diag_curvature"]["initial_params_sha256"]:
        raise ValueError("the methods used different initial policy checkpoints")
    if configs["standard_es"]["initial_obs_norm_sha256"] != configs["diag_curvature"]["initial_obs_norm_sha256"]:
        raise ValueError("the methods used different observation normalizers")
    curvature = histories["diag_curvature"]
    if any(int(row.get("hessian_pairs", -1)) != 1000 for row in curvature):
        raise ValueError("curvature pair count is not exactly 1000 at every iteration")
    if any(float(row["h_raw_mean"]) != float(row["h_step_mean"]) for row in curvature):
        raise ValueError("raw and applied no-EMA curvature differ")
    return comparison, histories


def _write_source_data(
    output: Path,
    histories: dict[str, list[dict[str, Any]]],
) -> None:
    standard = histories["standard_es"]
    curvature = histories["diag_curvature"]
    standard_returns = np.asarray([row["eval_reward"] for row in standard], dtype=float)
    curvature_returns = np.asarray([row["eval_reward"] for row in curvature], dtype=float)
    rows = []
    standard_rolling = _rolling_mean(standard_returns)
    curvature_rolling = _rolling_mean(curvature_returns)
    for index in range(300):
        rows.append(
            {
                "iteration": index,
                "standard_eval_return": standard_returns[index],
                "standard_return_rolling20": standard_rolling[index],
                "standard_step_norm": standard[index]["step_norm"],
                "curvature_eval_return": curvature_returns[index],
                "curvature_return_rolling20": curvature_rolling[index],
                "curvature_step_norm": curvature[index]["step_norm"],
                "curvature_multiplier_mean": curvature[index]["step_multiplier_mean"],
                "curvature_multiplier_floor_fraction": curvature[index]["multiplier_floor_clip_frac"],
                "curvature_active_fraction": curvature[index]["curvature_active_frac"],
                "curvature_split_correlation": curvature[index].get("h_split_correlation"),
                "curvature_split_sign_agreement": curvature[index].get("h_split_sign_agreement"),
            }
        )
    with (output / "lr0p16_plot_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(root: Path, output: Path) -> dict[str, Any]:
    comparison, histories = _validate_and_load(root)
    output.mkdir(parents=True, exist_ok=True)
    _write_source_data(output, histories)

    standard = histories["standard_es"]
    curvature = histories["diag_curvature"]
    iterations = np.arange(300)
    standard_return = np.asarray([row["eval_reward"] for row in standard], dtype=float)
    curvature_return = np.asarray([row["eval_reward"] for row in curvature], dtype=float)
    standard_rolling = _rolling_mean(standard_return)
    curvature_rolling = _rolling_mean(curvature_return)
    multiplier = np.asarray([row["step_multiplier_mean"] for row in curvature], dtype=float)
    floor_fraction = np.asarray([row["multiplier_floor_clip_frac"] for row in curvature], dtype=float)

    summary = {
        "standard_es": {
            "final": float(standard_return[-1]),
            "best": float(np.max(standard_return)),
            "last20": float(np.mean(standard_return[-20:])),
        },
        "diag_curvature": {
            "final": float(curvature_return[-1]),
            "best": float(np.max(curvature_return)),
            "last20": float(np.mean(curvature_return[-20:])),
        },
    }
    reported = comparison["runs"]
    for condition in summary:
        checks = {
            "final": reported[condition]["final_eval_return"],
            "best": reported[condition]["best_eval_return"],
            "last20": reported[condition]["last_20_mean_eval_return"],
        }
        for metric, value in checks.items():
            if not np.isclose(summary[condition][metric], value, rtol=0.0, atol=1e-9):
                raise ValueError(f"plotted {condition} {metric} disagrees with report")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
        }
    )
    figure = plt.figure(figsize=(14.4, 8.1), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(2.15, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.065,
        right=0.975,
        bottom=0.105,
        top=0.81,
        wspace=0.28,
        hspace=0.48,
    )
    returns_axis = figure.add_subplot(grid[:, 0])
    summary_axis = figure.add_subplot(grid[0, 1])
    damping_axis = figure.add_subplot(grid[1, 1])

    returns_axis.plot(iterations, standard_return, color=STANDARD_COLOR, alpha=0.18, linewidth=0.9)
    returns_axis.plot(iterations, curvature_return, color=CURVATURE_COLOR, alpha=0.20, linewidth=0.9)
    returns_axis.plot(
        iterations,
        standard_rolling,
        color=STANDARD_COLOR,
        linewidth=2.6,
        label="Standard ES · 20-iteration mean",
    )
    returns_axis.plot(
        iterations,
        curvature_rolling,
        color=CURVATURE_COLOR,
        linewidth=2.6,
        linestyle="--",
        label="Diagonal curvature · 20-iteration mean",
    )
    returns_axis.scatter([299], [standard_return[-1]], color=STANDARD_COLOR, s=42, zorder=5)
    returns_axis.scatter(
        [299], [curvature_return[-1]], facecolor=BACKGROUND, edgecolor=CURVATURE_COLOR, linewidth=1.8, s=48, zorder=5
    )
    returns_axis.annotate(
        f"Final {standard_return[-1]:,.0f}",
        (299, standard_return[-1]),
        xytext=(-70, 12),
        textcoords="offset points",
        color=STANDARD_COLOR,
        fontsize=9,
        fontweight="bold",
    )
    returns_axis.annotate(
        f"Final {curvature_return[-1]:,.0f}",
        (299, curvature_return[-1]),
        xytext=(-78, -19),
        textcoords="offset points",
        color=CURVATURE_COLOR,
        fontsize=9,
        fontweight="bold",
    )
    returns_axis.set_title("Evaluation return over ES iterations", loc="left", fontweight="bold")
    returns_axis.set_xlabel("ES iteration")
    returns_axis.set_ylabel("Mean episodic return")
    returns_axis.set_xlim(0, 299)
    returns_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    returns_axis.spines[["top", "right"]].set_visible(False)
    returns_axis.legend(loc="upper left", frameon=False, fontsize=9)

    metrics = ("Best", "Final", "Last-20 mean")
    standard_values = [summary["standard_es"][key] for key in ("best", "final", "last20")]
    curvature_values = [summary["diag_curvature"][key] for key in ("best", "final", "last20")]
    positions = np.arange(len(metrics))
    width = 0.36
    bars_standard = summary_axis.bar(
        positions - width / 2,
        standard_values,
        width,
        color=STANDARD_COLOR,
        label="Standard ES",
    )
    bars_curvature = summary_axis.bar(
        positions + width / 2,
        curvature_values,
        width,
        color="#F7E7D0",
        edgecolor=CURVATURE_COLOR,
        linewidth=1.4,
        hatch="//",
        label="Diagonal curvature",
    )
    summary_axis.bar_label(bars_standard, fmt="{:,.0f}", padding=3, fontsize=8, color=STANDARD_COLOR)
    summary_axis.bar_label(bars_curvature, fmt="{:,.0f}", padding=3, fontsize=8, color=CURVATURE_COLOR)
    summary_axis.set_title("Completed-run summary", loc="left", fontweight="bold")
    summary_axis.set_xticks(positions, metrics)
    summary_axis.set_ylim(0, max(standard_values + curvature_values) * 1.17)
    summary_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    summary_axis.spines[["top", "right"]].set_visible(False)
    summary_axis.legend(frameon=False, fontsize=8, loc="upper right")
    summary_axis.tick_params(axis="y", labelsize=8)

    damping_axis.plot(
        iterations,
        multiplier,
        color=CURVATURE_COLOR,
        linewidth=1.8,
        label="Mean multiplier",
    )
    damping_axis.plot(
        iterations,
        floor_fraction,
        color=INK,
        linewidth=1.5,
        linestyle="--",
        label="Fraction at 0.05 floor",
    )
    damping_axis.axhline(0.525, color=MUTED, linestyle=":", linewidth=1.0, label="Half 0.05 / half 1.0")
    damping_axis.set_title("Curvature damping diagnostics", loc="left", fontweight="bold")
    damping_axis.set_xlabel("ES iteration")
    damping_axis.set_ylabel("Fraction or multiplier")
    damping_axis.set_xlim(0, 299)
    damping_axis.set_ylim(0.35, 0.65)
    damping_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    damping_axis.spines[["top", "right"]].set_visible(False)
    damping_axis.legend(frameon=False, fontsize=8, loc="upper right")

    figure.suptitle(
        "Hopper-v5: standard ES and diagonal-curvature ES (learning rate 0.16)",
        x=0.065,
        y=0.955,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    figure.text(
        0.065,
        0.902,
        "Seed 0 · shared PPO checkpoint · 1,000 antithetic pairs (2,000 evaluations) per iteration · 300 iterations · EMA disabled",
        ha="left",
        fontsize=10.5,
        color=MUTED,
    )
    figure.text(
        0.065,
        0.035,
        "Thin lines show per-iteration evaluation returns; bold lines show trailing 20-iteration means. Source: Duke DCC jobs 50410849 and 50410861.",
        ha="left",
        fontsize=8.5,
        color=MUTED,
    )

    png_path = output / "lr0p16_finished_comparison.png"
    svg_path = output / "lr0p16_finished_comparison.svg"
    figure.savefig(png_path, dpi=180, facecolor=BACKGROUND)
    figure.savefig(svg_path, facecolor=BACKGROUND)
    plt.close(figure)

    summary_rows = []
    for condition, label in (("standard_es", "Standard ES"), ("diag_curvature", "Diagonal curvature")):
        summary_rows.append(
            {
                "condition": condition,
                "label": label,
                "best_return": summary[condition]["best"],
                "final_return": summary[condition]["final"],
                "last20_mean_return": summary[condition]["last20"],
            }
        )
    with (output / "lr0p16_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "status": "complete",
        "source_root": str(root.resolve()),
        "figure_png": png_path.name,
        "figure_svg": svg_path.name,
        "plot_data_csv": "lr0p16_plot_data.csv",
        "summary_csv": "lr0p16_summary.csv",
        "summary": summary,
        "comparison_deltas": comparison["diag_curvature_minus_standard_es"],
    }
    (output / "lr0p16_figure_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/hopper_pginit_noema_pairs1000_iter300_seed0_job50410861"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("presentation/hopper_pginit_noema/figures"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = plot(args.root, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

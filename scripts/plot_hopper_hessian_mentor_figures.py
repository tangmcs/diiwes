#!/usr/bin/env python3
"""Create mentor-ready fitness and paired-AUC figures for Slurm job 49811294."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diiwes-matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


CONDITION_LABELS = {
    "standard_es": "Standard ES",
    "diag_curvature": "Diagonal Hessian",
}
CONDITION_ORDER = ("standard_es", "diag_curvature")
CELLS = (
    ("inverse_sqrt", 10.0),
    ("inverse_sqrt", 30.0),
    ("inverse_linear", 10.0),
    ("inverse_linear", 30.0),
)
CELL_TITLES = {
    ("inverse_sqrt", 10.0): r"$\alpha_t = 10 / \sqrt{t+1}$",
    ("inverse_sqrt", 30.0): r"$\alpha_t = 30 / \sqrt{t+1}$",
    ("inverse_linear", 10.0): r"$\alpha_t = 10 / (t+1)$",
    ("inverse_linear", 30.0): r"$\alpha_t = 30 / (t+1)$",
}
CELL_CSV_LABELS = {
    ("inverse_sqrt", 10.0): "sqrt / 10",
    ("inverse_sqrt", 30.0): "sqrt / 30",
    ("inverse_linear", 10.0): "linear / 10",
    ("inverse_linear", 30.0): "linear / 30",
}

INK = "#172033"
NEUTRAL = "#596579"
NEUTRAL_LIGHT = "#B8C0CC"
GRID = "#DDE2E8"
BLUE = "#2563EB"
BLUE_LIGHT = "#AFC8FA"
ORANGE = "#D97706"
WHITE = "#FFFFFF"
T_CRITICAL_95_DF9 = 2.2621571627409915


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-csv",
        type=Path,
        default=Path("reports/hopper_main_hessian_no_trust_49811294/runs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/hopper_main_hessian_no_trust_49811294/figures"),
    )
    parser.add_argument("--smooth-window", type=int, default=25)
    return parser.parse_args()


def centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("The smoothing window must be odd.")
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def exact_two_sided_sign_flip_p(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        statistic = abs(float(np.mean(differences * np.asarray(signs))))
        extreme += statistic >= observed - 1e-12
    return extreme / (2 ** len(differences))


def load_runs(runs_csv: Path) -> dict[tuple[str, float, str], list[dict]]:
    grouped: dict[tuple[str, float, str], list[dict]] = defaultdict(list)
    with runs_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 80:
        raise ValueError(f"Expected 80 validated runs, found {len(rows)}.")

    for row in rows:
        schedule = row["lr_schedule"]
        alpha0 = float(row["initial_learning_rate"])
        condition = row["condition"]
        seed = int(row["seed"])
        history_path = Path(row["run_dir"]) / "history.json"
        with history_path.open(encoding="utf-8") as handle:
            history = json.load(handle)
        if len(history) != 500:
            raise ValueError(f"{history_path} has {len(history)} records, expected 500.")
        iterations = np.asarray([entry["iteration"] for entry in history], dtype=int)
        if not np.array_equal(iterations, np.arange(500)):
            raise ValueError(f"Unexpected iteration sequence in {history_path}.")
        fitness = np.asarray([entry["eval_reward"] for entry in history], dtype=float)
        population_fitness = np.asarray(
            [entry["mean_fitness"] for entry in history], dtype=float
        )
        if not np.all(np.isfinite(fitness)):
            raise ValueError(f"Non-finite evaluation fitness in {history_path}.")
        if not np.all(np.isfinite(population_fitness)):
            raise ValueError(f"Non-finite population fitness in {history_path}.")
        auc = float(np.trapz(fitness, dx=1.0) / (len(fitness) - 1))
        expected_auc = float(row["iteration_auc"])
        if not math.isclose(auc, expected_auc, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(
                f"AUC mismatch in {history_path}: computed {auc}, expected {expected_auc}."
            )
        grouped[(schedule, alpha0, condition)].append(
            {
                "seed": seed,
                "fitness": fitness,
                "population_fitness": population_fitness,
                "auc": auc,
            }
        )

    for cell in CELLS:
        for condition in CONDITION_ORDER:
            key = (*cell, condition)
            grouped[key].sort(key=lambda run: run["seed"])
            seeds = [run["seed"] for run in grouped[key]]
            if seeds != list(range(10)):
                raise ValueError(f"Expected seeds 0..9 for {key}, found {seeds}.")
    return grouped


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def trajectory_summary(
    runs: list[dict], window: int, field: str = "fitness"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([run[field] for run in runs])
    smooth = np.stack(
        [centered_moving_average(values, window) for values in raw]
    )
    raw_mean = raw.mean(axis=0)
    mean = smooth.mean(axis=0)
    half_width = T_CRITICAL_95_DF9 * smooth.std(axis=0, ddof=1) / np.sqrt(10)
    return raw_mean, mean, mean - half_width, mean + half_width, smooth


def make_fitness_figure(grouped: dict, window: int) -> tuple[plt.Figure, list[dict]]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.12, top=0.82, hspace=0.39, wspace=0.24)
    iterations = np.arange(1, 501)
    summary_rows: list[dict] = []

    panel_uppers: list[float] = []
    for ax, cell in zip(axes.flat, CELLS):
        panel_upper = 0.0
        for condition in CONDITION_ORDER:
            raw_mean, mean, low, high, _ = trajectory_summary(
                grouped[(*cell, condition)], window
            )
            color = NEUTRAL if condition == "standard_es" else BLUE
            fill = NEUTRAL_LIGHT if condition == "standard_es" else BLUE_LIGHT
            linestyle = "--" if condition == "standard_es" else "-"
            ax.fill_between(iterations, low, high, color=fill, alpha=0.28, linewidth=0)
            ax.plot(
                iterations,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=2.1,
                label=CONDITION_LABELS[condition],
            )
            panel_upper = max(panel_upper, float(np.max(high)))
            for i in range(500):
                summary_rows.append(
                    {
                        "iteration": i,
                        "schedule": cell[0],
                        "initial_learning_rate": cell[1],
                        "cell": CELL_CSV_LABELS[cell],
                        "condition": condition,
                        "n_seeds": 10,
                        "raw_mean_eval_fitness": raw_mean[i],
                        "smoothed_mean_eval_fitness": mean[i],
                        "smoothed_ci95_low": low[i],
                        "smoothed_ci95_high": high[i],
                    }
                )

        ax.set_title(CELL_TITLES[cell], loc="left", pad=8)
        ax.set_xlim(1, 500)
        panel_uppers.append(max(10.0, panel_upper * 1.08))
        ax.set_xticks([1, 100, 200, 300, 400, 500])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    sqrt_upper = max(panel_uppers[:2])
    linear_upper = max(panel_uppers[2:])
    for ax in axes[0, :]:
        ax.set_ylim(0, sqrt_upper)
    for ax in axes[1, :]:
        ax.set_ylim(0, linear_upper)

    fig.supylabel("Evaluation return", x=0.018, fontsize=11)
    fig.supxlabel("Update", y=0.045, fontsize=11)

    fig.suptitle(
        "Hopper-v5 Evaluation Fitness",
        x=0.075,
        y=0.955,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.900,
        "10 seeds · mean ± 95% CI · 25-update moving average",
        ha="left",
        fontsize=10.8,
        color=NEUTRAL,
    )
    handles = [
        Line2D([0], [0], color=NEUTRAL, linestyle="--", linewidth=2.1, label="Standard ES"),
        Line2D([0], [0], color=BLUE, linestyle="-", linewidth=2.1, label="Diagonal Hessian"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.976, 0.946),
        ncol=2,
        handlelength=2.8,
        columnspacing=1.4,
    )
    return fig, summary_rows


def make_population_fitness_figure(grouped: dict, window: int) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.12, top=0.82, hspace=0.35, wspace=0.24)
    iterations = np.arange(1, 501)
    panel_uppers: list[float] = []

    for ax, cell in zip(axes.flat, CELLS):
        panel_upper = 0.0
        for condition in CONDITION_ORDER:
            _, mean, low, high, _ = trajectory_summary(
                grouped[(*cell, condition)], window, field="population_fitness"
            )
            color = NEUTRAL if condition == "standard_es" else BLUE
            fill = NEUTRAL_LIGHT if condition == "standard_es" else BLUE_LIGHT
            linestyle = "--" if condition == "standard_es" else "-"
            ax.fill_between(iterations, low, high, color=fill, alpha=0.28, linewidth=0)
            ax.plot(iterations, mean, color=color, linestyle=linestyle, linewidth=2.1)
            panel_upper = max(panel_upper, float(np.max(high)))
        ax.set_title(CELL_TITLES[cell], loc="left", pad=8)
        ax.set_xlim(1, 500)
        panel_uppers.append(max(10.0, panel_upper * 1.08))
        ax.set_xticks([1, 100, 200, 300, 400, 500])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    for ax in axes[0, :]:
        ax.set_ylim(0, max(panel_uppers[:2]))
    for ax in axes[1, :]:
        ax.set_ylim(0, max(panel_uppers[2:]))
    axes[0, 0].set_ylabel("Population mean fitness")
    axes[1, 0].set_ylabel("Population mean fitness")
    axes[1, 0].set_xlabel("Update")
    axes[1, 1].set_xlabel("Update")

    fig.suptitle(
        "Hopper-v5 training-population mean fitness",
        x=0.075,
        y=0.963,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.908,
        "Mean return of the 500 fresh perturbed candidates per update; this is not the center-policy evaluation and is not used for the reported AUC. "
        "Lines average 10 seeds after 25-update smoothing; bands are pointwise 95% t intervals.",
        ha="left",
        fontsize=10.2,
        color=NEUTRAL,
    )
    handles = [
        Line2D([0], [0], color=NEUTRAL, linestyle="--", linewidth=2.1, label="Standard ES"),
        Line2D([0], [0], color=BLUE, linestyle="-", linewidth=2.1, label="Diagonal Hessian"),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.976, 0.953),
        ncol=2,
        handlelength=2.8,
        columnspacing=1.4,
    )
    fig.text(
        0.075,
        0.035,
        "Job 49811294 · population 500 · 500 updates · no replay · no trust region · zero scalar damping. "
        "Y scales are shared within each schedule row; smoothing is display-only.",
        ha="left",
        fontsize=8.6,
        color=NEUTRAL,
    )
    return fig


def make_auc_figure(grouped: dict) -> tuple[plt.Figure, list[dict]]:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.12, top=0.82, hspace=0.42, wspace=0.28)
    auc_rows: list[dict] = []

    for ax, cell in zip(axes.flat, CELLS):
        standard_runs = grouped[(*cell, "standard_es")]
        hessian_runs = grouped[(*cell, "diag_curvature")]
        standard = np.asarray([run["auc"] for run in standard_runs])
        hessian = np.asarray([run["auc"] for run in hessian_runs])
        differences = hessian - standard
        mean_difference = float(differences.mean())
        ci_half = T_CRITICAL_95_DF9 * float(differences.std(ddof=1)) / np.sqrt(10)
        ci_low, ci_high = mean_difference - ci_half, mean_difference + ci_half
        p_value = exact_two_sided_sign_flip_p(differences)
        wins = int(np.sum(differences > 0))

        for seed, std_auc, hess_auc, difference in zip(
            range(10), standard, hessian, differences
        ):
            line_color = BLUE if difference > 0 else ORANGE
            ax.plot([0, 1], [std_auc, hess_auc], color=line_color, alpha=0.40, linewidth=1.15)
            ax.scatter(0, std_auc, s=30, facecolor=WHITE, edgecolor=NEUTRAL, linewidth=1.15, zorder=3)
            ax.scatter(1, hess_auc, s=30, facecolor=BLUE, edgecolor=BLUE, linewidth=1.0, zorder=3)
            auc_rows.append(
                {
                    "schedule": cell[0],
                    "initial_learning_rate": cell[1],
                    "cell": CELL_CSV_LABELS[cell],
                    "seed": seed,
                    "standard_auc": std_auc,
                    "hessian_auc": hess_auc,
                    "hessian_minus_standard": difference,
                }
            )

        ax.scatter(
            [0, 1],
            [standard.mean(), hessian.mean()],
            marker="D",
            s=78,
            facecolors=[WHITE, BLUE],
            edgecolors=[INK, INK],
            linewidths=1.4,
            zorder=5,
        )
        ax.set_title(
            CELL_TITLES[cell]
            + f"\nΔ = {mean_difference:+.2f}  [95% CI {ci_low:+.2f}, {ci_high:+.2f}]\n"
            + f"{wins}/10 seeds higher; exact p = {p_value:.4f}",
            loc="left",
            pad=8,
        )
        ax.set_xlim(-0.28, 1.28)
        panel_max = float(max(np.max(standard), np.max(hessian)))
        ax.set_ylim(0, max(10.0, panel_max * 1.12))
        ax.set_xticks([0, 1], ["Standard ES", "Diagonal Hessian"])
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    axes[0, 0].set_ylabel("Normalized iteration AUC")
    axes[1, 0].set_ylabel("Normalized iteration AUC")

    fig.suptitle(
        "Hopper-v5 paired training-curve AUC by seed",
        x=0.075,
        y=0.963,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        0.908,
        "Each line connects the same seed; diamonds show method means. Blue lines favor Hessian and orange lines favor Standard ES.",
        ha="left",
        fontsize=10.5,
        color=NEUTRAL,
    )
    fig.text(
        0.075,
        0.035,
        "AUC is the normalized trapezoidal area under the 500 unsmoothed per-update evaluation returns. "
        "All paired 95% confidence intervals cross zero; exact p-values are unadjusted. Three evaluation episodes repeat the same seed within a run.",
        ha="left",
        fontsize=8.6,
        color=NEUTRAL,
    )
    return fig, auc_rows


def make_auc_difference_figure(grouped: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.975, bottom=0.20, top=0.72, wspace=0.27)
    panel_specs = (
        ("inverse_sqrt", r"Inverse-square-root decay", (-50.0, 85.0)),
        ("inverse_linear", r"Inverse-linear decay", (-250.0, 225.0)),
    )

    for ax, (schedule, panel_title, x_limits) in zip(axes, panel_specs):
        for row_index, alpha0 in enumerate((10.0, 30.0)):
            y = 1 - row_index
            standard = np.asarray(
                [run["auc"] for run in grouped[(schedule, alpha0, "standard_es")]]
            )
            hessian = np.asarray(
                [run["auc"] for run in grouped[(schedule, alpha0, "diag_curvature")]]
            )
            differences = hessian - standard
            mean_difference = float(differences.mean())
            ci_half = T_CRITICAL_95_DF9 * float(differences.std(ddof=1)) / np.sqrt(10)
            p_value = exact_two_sided_sign_flip_p(differences)
            wins = int(np.sum(differences > 0))
            jitter = np.linspace(-0.105, 0.105, len(differences))
            positive = differences > 0
            ax.scatter(
                differences[positive],
                y + jitter[positive],
                s=28,
                color=BLUE,
                alpha=0.62,
                edgecolor=WHITE,
                linewidth=0.35,
                zorder=3,
            )
            ax.scatter(
                differences[~positive],
                y + jitter[~positive],
                s=28,
                color=ORANGE,
                alpha=0.62,
                edgecolor=WHITE,
                linewidth=0.35,
                zorder=3,
            )
            ax.errorbar(
                mean_difference,
                y,
                xerr=ci_half,
                fmt="s",
                markersize=7,
                markerfacecolor=INK,
                markeredgecolor=INK,
                ecolor=INK,
                elinewidth=2.0,
                capsize=4,
                zorder=5,
            )
            ax.text(
                0.01,
                y + 0.24,
                f"Standard {standard.mean():.1f} → Hessian {hessian.mean():.1f}; "
                f"Δ {mean_difference:+.1f}; {wins}/10 higher; p={p_value:.4f}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8.8,
                color=NEUTRAL,
            )

        ax.axvline(0, color=INK, linewidth=1.2, linestyle="--", zorder=1)
        ax.set_xlim(*x_limits)
        ax.set_ylim(-0.35, 1.46)
        ax.set_yticks([1, 0], [r"$\alpha_0=10$", r"$\alpha_0=30$"])
        ax.set_title(panel_title, loc="left", pad=9, fontweight="bold")
        ax.set_xlabel("Hessian − Standard iteration AUC")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.8)

    fig.suptitle(
        "Paired difference in Hopper-v5 training-curve AUC",
        x=0.10,
        y=0.955,
        ha="left",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.10,
        0.865,
        "Positive values favor the diagonal Hessian method. Each circle is one matched seed; squares and horizontal lines show the paired mean and 95% t interval.",
        ha="left",
        fontsize=10.3,
        color=NEUTRAL,
    )
    fig.text(
        0.10,
        0.075,
        "AUC is normalized by 499 update intervals and uses raw online evaluation returns. Panel x-scales differ because inverse-linear/10 has extreme seed variation. "
        "Every interval crosses zero; smallest exact p=.0566 (Holm-adjusted p=.2266).",
        ha="left",
        fontsize=8.7,
        color=NEUTRAL,
    )
    return fig


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.smooth_window <= 0 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    grouped = load_runs(args.runs_csv)

    fitness_fig, curve_rows = make_fitness_figure(grouped, args.smooth_window)
    auc_fig, auc_rows = make_auc_figure(grouped)
    auc_difference_fig = make_auc_difference_figure(grouped)
    population_fitness_fig = make_population_fitness_figure(
        grouped, args.smooth_window
    )

    fitness_png = args.output_dir / "fitness_curves.png"
    fitness_svg = args.output_dir / "fitness_curves.svg"
    auc_png = args.output_dir / "auc_paired_by_seed.png"
    auc_svg = args.output_dir / "auc_paired_by_seed.svg"
    auc_difference_png = args.output_dir / "auc_difference_forest.png"
    auc_difference_svg = args.output_dir / "auc_difference_forest.svg"
    population_fitness_png = args.output_dir / "population_mean_fitness_curves.png"
    population_fitness_svg = args.output_dir / "population_mean_fitness_curves.svg"
    combined_pdf = args.output_dir / "mentor_fitness_and_auc.pdf"

    fitness_fig.savefig(fitness_png, dpi=300, bbox_inches="tight")
    fitness_fig.savefig(fitness_svg, bbox_inches="tight")
    auc_fig.savefig(auc_png, dpi=300, bbox_inches="tight")
    auc_fig.savefig(auc_svg, bbox_inches="tight")
    auc_difference_fig.savefig(auc_difference_png, dpi=300, bbox_inches="tight")
    auc_difference_fig.savefig(auc_difference_svg, bbox_inches="tight")
    population_fitness_fig.savefig(population_fitness_png, dpi=300, bbox_inches="tight")
    population_fitness_fig.savefig(population_fitness_svg, bbox_inches="tight")
    with PdfPages(combined_pdf) as pdf:
        pdf.savefig(fitness_fig, bbox_inches="tight")
        pdf.savefig(auc_difference_fig, bbox_inches="tight")
        pdf.savefig(auc_fig, bbox_inches="tight")
        pdf.savefig(population_fitness_fig, bbox_inches="tight")

    write_csv(args.output_dir / "fitness_curve_summary.csv", curve_rows)
    write_csv(args.output_dir / "auc_pairs.csv", auc_rows)
    plt.close(fitness_fig)
    plt.close(auc_fig)
    plt.close(auc_difference_fig)
    plt.close(population_fitness_fig)

    print(f"Validated {sum(len(runs) for runs in grouped.values())} runs.")
    print(f"Wrote {fitness_png}")
    print(f"Wrote {auc_difference_png}")
    print(f"Wrote {auc_png}")
    print(f"Wrote {population_fitness_png}")
    print(f"Wrote {combined_pdf}")


if __name__ == "__main__":
    main()

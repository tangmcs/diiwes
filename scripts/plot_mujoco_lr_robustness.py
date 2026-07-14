#!/usr/bin/env python3
"""Plot the historical, trust-confounded MuJoCo learning-rate sweep."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENV_ORDER = ["Ant-v5", "HalfCheetah-v5", "Hopper-v5", "Walker2d-v5"]
LR_ORDER = [0.02, 0.25, 0.5, 1.0]
METHOD_ORDER = ["DIIWES-H", "Standard ES"]
METHOD_COLORS = {"DIIWES-H": "#0072B2", "Standard ES": "#D55E00"}
METHOD_MARKERS = {"DIIWES-H": "o", "Standard ES": "s"}
LR_COLORS = {
    0.02: "#0072B2",
    0.25: "#E69F00",
    0.5: "#009E73",
    1.0: "#CC79A7",
}
LR_MARKERS = {
    0.02: "o",
    0.25: "s",
    0.5: "^",
    1.0: "D",
}


@dataclass
class RunSummary:
    env: str
    method: str
    lr: float
    seed: int
    final_return: float
    best_return: float
    mean_return: float
    env_steps: int


def _to_float(value: str) -> float:
    return float(value) if value not in ("", "nan", "None") else float("nan")


def _sem(values: Iterable[float]) -> float:
    xs = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if len(xs) <= 1:
        return 0.0
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs)))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.22, linewidth=0.7)
    ax.grid(True, axis="x", alpha=0.10, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3, width=0.7)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def load_table(path: str) -> tuple[list[dict[str, str]], list[RunSummary]]:
    rows: list[dict[str, str]] = []
    grouped: dict[tuple[str, str, float, int], list[dict[str, str]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("suite") != "mujoco":
                continue
            rows.append(row)
            grouped[
                (
                    row["env"],
                    row["method"],
                    float(row["learning_rate"]),
                    int(row["seed"]),
                )
            ].append(row)

    summaries: list[RunSummary] = []
    for (env, method, lr, seed), run_rows in grouped.items():
        run_rows.sort(key=lambda r: int(r["iteration"]))
        returns = np.asarray([_to_float(r["eval_return"]) for r in run_rows], dtype=np.float64)
        last = run_rows[-1]
        summaries.append(
            RunSummary(
                env=env,
                method=method,
                lr=lr,
                seed=seed,
                final_return=float(returns[-1]),
                best_return=_to_float(last["best_eval_return_so_far"]),
                mean_return=float(np.nanmean(returns)),
                env_steps=int(float(last["env_steps"])),
            )
        )
    return rows, summaries


def _format_lr(lr: float) -> str:
    return f"{lr:g}"


def _save(fig: plt.Figure, out_dir: str, stem: str) -> None:
    png = os.path.join(out_dir, f"{stem}.png")
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(png)
    print(pdf)


def plot_best_vs_lr(summaries: list[RunSummary], out_dir: str) -> None:
    by_group: dict[tuple[str, str, float], list[RunSummary]] = defaultdict(list)
    for item in summaries:
        by_group[(item.env, item.method, item.lr)].append(item)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.18, hspace=0.36, wspace=0.28)
    axes = axes.ravel()
    x = np.arange(len(LR_ORDER), dtype=np.float64)
    offset = {"DIIWES-H": -0.04, "Standard ES": 0.04}
    for panel, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        for method in METHOD_ORDER:
            means = []
            errs = []
            for lr in LR_ORDER:
                values = [r.best_return for r in by_group.get((env, method, lr), [])]
                means.append(float(np.mean(values)) if values else np.nan)
                errs.append(_sem(values))
            ax.errorbar(
                x + offset[method],
                means,
                yerr=errs,
                marker=METHOD_MARKERS[method],
                markersize=4.5,
                linewidth=1.7,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
                color=METHOD_COLORS[method],
                label=method,
            )
        ax.set_title(env)
        ax.set_xticks(x, [_format_lr(lr) for lr in LR_ORDER])
        _style_axis(ax)
        ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.45)
        ax.set_ylabel("Best eval return")
        _add_panel_label(ax, chr(ord("a") + panel))
    axes[-2].set_xlabel("Learning rate")
    axes[-1].set_xlabel("Learning rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handlelength=2.2,
    )
    fig.suptitle("MuJoCo learning-rate robustness", y=0.965, fontsize=12)
    _save(fig, out_dir, "mujoco_lr_robustness_best_return")


def plot_normalized_implicit(summaries: list[RunSummary], out_dir: str) -> None:
    values_by_env_lr: dict[tuple[str, float], list[float]] = defaultdict(list)
    for item in summaries:
        if item.method == "DIIWES-H":
            values_by_env_lr[(item.env, item.lr)].append(item.best_return)

    normalized: dict[str, list[float]] = {}
    for env in ENV_ORDER:
        means = np.asarray(
            [np.mean(values_by_env_lr[(env, lr)]) for lr in LR_ORDER],
            dtype=np.float64,
        )
        denom = float(np.max(means)) if len(means) else 1.0
        normalized[env] = (means / max(denom, 1e-12)).tolist()

    x = np.arange(len(LR_ORDER), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    fig.subplots_adjust(left=0.12, right=0.72, top=0.88, bottom=0.16)
    for env in ENV_ORDER:
        ax.plot(x, normalized[env], marker="o", markersize=4.3, linewidth=1.5, alpha=0.9, label=env)

    aggregate = np.mean(np.asarray([normalized[env] for env in ENV_ORDER]), axis=0)
    ax.plot(x, aggregate, marker="o", markersize=4.8, linewidth=2.4, color="#111111", label="Mean across envs")
    ax.set_xticks(x, [_format_lr(lr) for lr in LR_ORDER])
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("DIIWES-H best return / env max")
    ax.set_title("DIIWES-H robustness across learning rates")
    _style_axis(ax)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        handlelength=2.0,
    )
    _save(fig, out_dir, "mujoco_diiwes_lr_robustness_normalized")


def plot_learning_curves(rows: list[dict[str, str]], out_dir: str) -> None:
    grouped: dict[tuple[str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] != "DIIWES-H":
            continue
        grouped[(row["env"], float(row["learning_rate"]), int(row["seed"]))].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.19, hspace=0.36, wspace=0.28)
    axes = axes.ravel()
    for panel, (ax, env) in enumerate(zip(axes, ENV_ORDER)):
        for lr in LR_ORDER:
            seed_curves = []
            iterations = None
            for seed in range(5):
                run_rows = grouped.get((env, lr, seed), [])
                if not run_rows:
                    continue
                run_rows.sort(key=lambda r: int(r["iteration"]))
                iterations = np.asarray([int(r["iteration"]) for r in run_rows], dtype=np.int64)
                seed_curves.append([_to_float(r["eval_return"]) for r in run_rows])
            if not seed_curves or iterations is None:
                continue
            curves = np.asarray(seed_curves, dtype=np.float64)
            mean = np.nanmean(curves, axis=0)
            sem = np.nanstd(curves, axis=0, ddof=1) / math.sqrt(curves.shape[0])
            ax.plot(
                iterations,
                mean,
                color=LR_COLORS[lr],
                linewidth=1.5,
                marker=LR_MARKERS[lr],
                markevery=80,
                markersize=3.4,
                label=f"lr={_format_lr(lr)}",
            )
            ax.fill_between(iterations, mean - sem, mean + sem, color=LR_COLORS[lr], alpha=0.15, linewidth=0)
        ax.set_title(env)
        ax.set_ylabel("Eval return")
        _style_axis(ax)
        ax.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.45)
        _add_panel_label(ax, chr(ord("a") + panel))
    axes[-2].set_xlabel("Iteration")
    axes[-1].set_xlabel("Iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
    )
    fig.suptitle("DIIWES-H learning curves by learning rate", y=0.965, fontsize=12)
    _save(fig, out_dir, "mujoco_diiwes_lr_learning_curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="plots/mujoco_lr_sweep_46638567_plot_table.csv",
        help="Long-format CSV produced by scripts/export_plot_table.py.",
    )
    parser.add_argument("--out-dir", default="plots", help="Directory for generated figures.")
    parser.add_argument(
        "--allow-confounded-historical-results",
        action="store_true",
        help="Acknowledge that the input sweep used trust clipping and is not evidence of robustness.",
    )
    args = parser.parse_args()
    if not args.allow_confounded_historical_results:
        parser.error(
            "this script only plots the retired trust-confounded sweep; pass "
            "--allow-confounded-historical-results only for historical diagnosis"
        )

    configure_style()
    os.makedirs(args.out_dir, exist_ok=True)
    rows, summaries = load_table(args.input)
    plot_best_vs_lr(summaries, args.out_dir)
    plot_normalized_implicit(summaries, args.out_dir)
    plot_learning_curves(rows, args.out_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export experiment histories to the long-format plotting table."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any


METHOD_LABELS = {
    "standard_es": "Standard ES",
    "standard_es_trust": "ES+trust",
    "no_curvature": "DIIWES-no-H",
    "diag_curvature": "DIIWES-H",
    "global_curvature": "DIIWES-global-H",
    "block_curvature": "DIIWES-block-H",
    "directional_curvature": "DIIWES-dir-H",
    "normalized_diag_curvature": "DIIWES-norm-diag-H",
    "normalized_block_curvature": "DIIWES-norm-block-H",
}

BASE_FIELDS = [
    "suite",
    "env",
    "method",
    "condition",
    "seed",
    "learning_rate",
    "iteration",
    "env_steps",
    "eval_return",
    "train_population_mean",
    "train_population_best",
    "best_eval_return_so_far",
    "wall_time_sec",
    "iteration_time_sec",
    "train_env_steps",
    "train_env_steps_iter",
    "eval_env_steps",
    "eval_env_steps_iter",
    "total_env_steps",
    "total_env_steps_iter",
    "eval_count",
    "run_dir",
]

DIAGNOSTIC_FIELDS = [
    "n_fresh",
    "n_reused",
    "buffer_size",
    "used_replay",
    "ess",
    "ess_ratio",
    "ess_normalized",
    "clip_frac",
    "clip_fraction",
    "w_max",
    "w_min",
    "max_weight_ratio",
    "mean_importance_weight",
    "max_importance_weight",
    "importance_weight_mean",
    "importance_weight_min",
    "importance_weight_max",
    "grad_norm",
    "step_norm",
    "pre_trust_step_norm",
    "no_curv_pre_trust_step_norm",
    "curv_norm_shrink",
    "trust_active",
    "trust_scale",
    "explicit_step_norm",
    "step_norm_ratio",
    "step_multiplier_mean",
    "step_multiplier_std",
    "step_multiplier_min",
    "step_multiplier_max",
    "multiplier_floor_frac",
    "hessian_shrinkage_median",
    "hessian_shrinkage_p90",
    "hessian_shrinkage_max",
    "curv_mean",
    "curv_max",
    "curv_min",
    "curvature_active_frac",
    "curvature_mode",
    "curvature_step_mode",
    "curvature_baseline",
    "hessian_pairs",
    "h_raw_mean",
    "h_raw_std",
    "h_raw_min",
    "h_raw_max",
    "h_ema_mean",
    "h_ema_min",
    "h_ema_max",
    "h_step_mean",
    "h_step_min",
    "h_step_max",
    "lambda",
    "sigma",
    "reuse_fraction",
]


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _suite_for_env(env_name: str) -> str:
    return "atari" if env_name.startswith("ALE/") or env_name.startswith("ALE_") else "mujoco"


def _value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return ""


def export_rows(root: str) -> list[dict[str, Any]]:
    histories = sorted(glob.glob(os.path.join(root, "**", "history.json"), recursive=True))
    rows: list[dict[str, Any]] = []
    for history_path in histories:
        run_dir = os.path.dirname(history_path)
        config_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_path):
            continue

        config = _read_json(config_path)
        history = _read_json(history_path)
        env_name = str(config.get("env_name", ""))
        condition = str(config.get("condition", config.get("algorithm", "")))
        method = METHOD_LABELS.get(condition, condition)
        seed = config.get("seed", "")
        config_lr = config.get("learning_rate", "")

        cumulative_wall_time = 0.0
        for record in history:
            iteration_time = float(record.get("time", 0.0))
            cumulative_wall_time += iteration_time
            row = {
                "suite": _suite_for_env(env_name),
                "env": env_name,
                "method": method,
                "condition": condition,
                "seed": seed,
                "learning_rate": _value(record, "lr", "learning_rate") or config_lr,
                "iteration": record.get("iteration", ""),
                "env_steps": _value(record, "env_steps", "train_env_steps"),
                "eval_return": record.get("eval_reward", ""),
                "train_population_mean": record.get("mean_fitness", ""),
                "train_population_best": record.get("max_fitness", ""),
                "best_eval_return_so_far": record.get("best_reward", ""),
                "wall_time_sec": cumulative_wall_time,
                "iteration_time_sec": iteration_time,
                "train_env_steps": record.get("train_env_steps", ""),
                "train_env_steps_iter": record.get("train_env_steps_iter", ""),
                "eval_env_steps": record.get("eval_env_steps", ""),
                "eval_env_steps_iter": record.get("eval_env_steps_iter", ""),
                "total_env_steps": record.get("total_env_steps", ""),
                "total_env_steps_iter": record.get("total_env_steps_iter", ""),
                "eval_count": record.get("eval_count", ""),
                "run_dir": run_dir,
            }
            for field in DIAGNOSTIC_FIELDS:
                row[field] = record.get(field, "")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Result root to scan recursively.")
    parser.add_argument(
        "--output",
        default="plots/plot_table.csv",
        help="Destination CSV path. Parent directories are created automatically.",
    )
    args = parser.parse_args()

    rows = export_rows(args.root)
    fieldnames = BASE_FIELDS + DIAGNOSTIC_FIELDS
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

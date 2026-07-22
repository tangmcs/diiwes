#!/usr/bin/env python3
"""Validate and summarize the matched PPO-initialized Hopper comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CONDITIONS = ("standard_es", "diag_curvature")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_run(path: Path, condition: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_json(path / "config.json")
    history = _load_json(path / "history.json")
    if not isinstance(history, list) or len(history) != 300:
        raise ValueError(f"{path}: expected exactly 300 completed iterations")
    expected = {
        "condition": condition,
        "seed": 0,
        "env_name": "Hopper-v5",
        "population_size": 2000,
        "n_iterations": 300,
        "buffer_size": 0,
        "reuse_fraction": 0.0,
        "implicit_damping": 0.0,
        "lr_schedule": "constant",
        "parameter_initialization": "policy_gradient_checkpoint",
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{path}: locked protocol mismatch: {mismatches}")
    if [int(row["iteration"]) for row in history] != list(range(300)):
        raise ValueError(f"{path}: iteration sequence is incomplete")
    if condition == "diag_curvature":
        if config.get("curvature_beta") != 0.0:
            raise ValueError(f"{path}: curvature_beta must be zero")
        if config.get("bias_correct_curvature_ema") is not False:
            raise ValueError(f"{path}: EMA bias correction must be disabled")
        if any(int(row.get("hessian_pairs", -1)) != 1000 for row in history):
            raise ValueError(f"{path}: every update must use exactly 1000 curvature pairs")
    return config, history


def summarize(root: Path) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        matches = sorted(root.glob(f"{condition}_seed0_task*"))
        if len(matches) != 1:
            raise ValueError(f"expected one {condition} directory in {root}, found {len(matches)}")
        config, history = _validate_run(matches[0], condition)
        warmstart_manifest = _load_json(
            Path(config["initial_params_path"]).parent / "manifest.json"
        )
        rewards = np.asarray([float(row["eval_reward"]) for row in history])
        row: dict[str, Any] = {
            "condition": condition,
            "directory": str(matches[0].resolve()),
            "initial_params_sha256": config["initial_params_sha256"],
            "initial_obs_norm_sha256": config["initial_obs_norm_sha256"],
            "learning_rate": float(config["learning_rate"]),
            "initial_eval_return": float(warmstart_manifest["best_eval_return"]),
            "final_eval_return": float(rewards[-1]),
            "best_eval_return": float(np.max(rewards)),
            "best_iteration": int(np.argmax(rewards)),
            "mean_eval_return": float(np.mean(rewards)),
            "last_20_mean_eval_return": float(np.mean(rewards[-20:])),
            "total_environment_steps": int(history[-1]["total_env_steps"]),
        }
        if condition == "diag_curvature":
            raw = np.asarray([float(record["h_raw_mean"]) for record in history])
            used = np.asarray([float(record["h_step_mean"]) for record in history])
            row.update(
                {
                    "curvature_beta": float(config["curvature_beta"]),
                    "curvature_pairs_per_update": 1000,
                    "max_abs_raw_vs_step_curvature_mean": float(np.max(np.abs(raw - used))),
                    "mean_curvature_active_fraction": float(
                        np.mean([float(record["curvature_active_frac"]) for record in history])
                    ),
                    "mean_step_multiplier": float(
                        np.mean([float(record["step_multiplier_mean"]) for record in history])
                    ),
                }
            )
        runs[condition] = row

    es = runs["standard_es"]
    curvature = runs["diag_curvature"]
    if es["initial_params_sha256"] != curvature["initial_params_sha256"]:
        raise ValueError("the two arms did not use the same policy checkpoint")
    if es["initial_obs_norm_sha256"] != curvature["initial_obs_norm_sha256"]:
        raise ValueError("the two arms did not use the same observation normalizer")
    if es["learning_rate"] != curvature["learning_rate"]:
        raise ValueError("the two arms did not use the same learning rate")
    deltas = {
        metric: float(curvature[metric] - es[metric])
        for metric in (
            "final_eval_return",
            "best_eval_return",
            "mean_eval_return",
            "last_20_mean_eval_return",
        )
    }
    result = {
        "status": "complete",
        "protocol": {
            "environment": "Hopper-v5",
            "seed": 0,
            "iterations": 300,
            "pairs_per_iteration": 1000,
            "population_size": 2000,
            "policy_gradient_initialization": True,
            "ema_enabled": False,
            "learning_rate": es["learning_rate"],
        },
        "runs": runs,
        "diag_curvature_minus_standard_es": deltas,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted(set().union(*(row.keys() for row in runs.values())))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs.values())
    report = [
        "# Hopper PPO-initialized comparison",
        "",
        f"Both arms use seed 0, constant learning rate {es['learning_rate']:g}, "
        "the same policy/normalizer checkpoint, 1,000 "
        "antithetic pairs per update, and 300 updates. The curvature arm uses "
        "the fresh estimate directly (`curvature_beta=0`); EMA is disabled.",
        "",
        "| Method | Final | Best | Mean | Last-20 mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = runs[condition]
        report.append(
            f"| {condition} | {row['final_eval_return']:.3f} | "
            f"{row['best_eval_return']:.3f} | {row['mean_eval_return']:.3f} | "
            f"{row['last_20_mean_eval_return']:.3f} |"
        )
    report.extend(
        (
            "",
            "Reported differences are diagonal-curvature minus standard ES.",
            "",
            *(
                f"- {metric}: {value:+.3f}"
                for metric, value in deltas.items()
            ),
            "",
        )
    )
    (root / "report.md").write_text("\n".join(report), encoding="utf-8")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result = summarize(parse_args(argv).root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

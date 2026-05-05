# Plotting

This folder is reserved for paper figure-generation scripts and plot-specific
notes. Do not use this README for general project documentation; the top-level
README is intentionally separate.

## Publication Figure Data Readiness

Use one method name in all figure legends:

- `DIIWES-H`: DIIWES with the semi-implicit diagonal Hessian approximation.

Snapshot date: 2026-05-05. The inventory below is based on the current
`../results` and `../job_outputs` directories.

## Current Completed Result Inventory

The completed histories currently cover diagonal-curvature DIIWES-H-style runs,
almost entirely with `seed=0`.

Completed MuJoCo histories:

| Environment | Completed DIIWES-H data | Notes |
| --- | --- | --- |
| Humanoid-v5 | seed 0, 500 iter, pop 200, lr 0.16; seed 0, 1000 iter, pop 500, lr 0.02 | A tuned 3000-iteration seed-0 run is still running in job `46611731`. Choose one canonical budget/config before plotting. |
| HalfCheetah-v5 | seed 0, 500 iter, pop 200, lr 0.16 | DIIWES-H only. |
| Ant-v5 | seed 0, 500 iter, pop 200, lr 0.16 | DIIWES-H only. |
| Hopper-v5 | seed 0, 500 iter, pop 200, lr 0.16 | DIIWES-H only. |
| Walker2d-v5 | seed 0, 500 iter, pop 200, lr 0.16 | DIIWES-H only. |

Completed Atari histories:

| Environment | Completed DIIWES-H data | Notes |
| --- | --- | --- |
| Boxing | seed 0, 200 iter; seed 0, 200 iter with `reuse_fraction=0.0` | Replay-off run still uses curvature, so it is not a no-Hessian ablation. |
| Freeway | seed 0, 200 iter; seed 0, 500 iter with obs norm; seed 0, 1000 iter with obs norm | The 500/1000-iteration obs-norm runs are stronger candidates for a canonical setup. |
| SpaceInvaders | seed 0, 200 iter | Fire-reset/normalized 1500-iteration variants are still running in jobs `46610008` and `46612096`. |
| Pong | seed 0, several 200-iteration variants; seed 0, 200 iter with `reuse_fraction=0.0` | A 900-iteration normalized run is still running in job `46608499`; the raw-fitness variant is not directly comparable unless the same setting is used for all methods. |

The existing histories include useful fields for plotting learning curves and
diagnostics: `iteration`, `eval_reward`, `best_reward`, `mean_fitness`,
`max_fitness`, `eval_count`, `ess`, `ess_ratio`, `clip_frac`, `w_max`, `w_min`,
`max_weight_ratio`, `grad_norm`, `step_norm`, `pre_trust_step_norm`,
`no_curv_pre_trust_step_norm`, `curv_norm_shrink`, `step_multiplier_mean`,
`step_multiplier_min`, `step_multiplier_max`, and wall-clock `time`.

Important limitation: `eval_count` is a count of fresh training rollout
evaluations, not true environment interactions. For sample-efficiency x-axes,
collect or reconstruct actual environment steps per rollout. Multiplying by
`max_episode_steps` is only an upper bound because episodes can terminate early.

## Data Still Needed Before Publication Figures

Priority collection items:

1. Collect a canonical multi-seed matrix for all benchmark plots.
   - Tasks: Humanoid, HalfCheetah, Ant, Hopper, Walker2d, Boxing, Freeway, SpaceInvaders, Pong.
   - Methods: Standard ES, Explicit IW-ES, DIIWES without Hessian, DIIWES-H.
   - Recommended seeds: at least 3 seeds per task/method; 5 seeds is preferred for credible uncertainty intervals.
   - Use one canonical preprocessing and budget per environment before collecting baselines. Current Atari reruns mix obs norm, frame stack, action subsets, raw-fitness, fire-reset, and different iteration counts.

2. Collect baselines and ablations.
   - No completed Standard ES histories are present in `../results`.
   - No completed Explicit IW-ES histories are present.
   - No completed `no_curvature` DIIWES histories are present.
   - The current `reuse_fraction=0.0` Boxing/Pong runs are replay-off curvature runs, not the requested no-Hessian ablation.
   - Optional blockwise Hessian runs are not present.

3. Collect normalization references.
   - For MuJoCo aggregate plots, collect random-policy returns and the chosen Standard ES final score for each task, or define another task-wise normalization baseline.
   - For Atari aggregate plots, store `random_score`, `human_score`, and `human_normalized_score` for Boxing, Freeway, SpaceInvaders, and Pong.
   - Do not average raw MuJoCo returns across tasks without normalization.

4. Collect true environment-interaction counts.
   - Current histories have `eval_count`, but Figure 2 and Figure 3 should use environment interactions.
   - Store per-iteration `env_steps` for training rollouts and, if needed, separate `eval_env_steps` for center-policy evaluations.

5. Standardize the long-format plotting table.
   Each plotting row should be convertible to:

   ```text
   suite, env, method, seed, iteration, env_steps, eval_return,
   train_population_mean, train_population_best,
   best_eval_return_so_far, wall_time_sec
   ```

   Current mappings are mostly available but need normalization:

   | Required column | Current source |
   | --- | --- |
   | `suite` | infer from `env_name` in `config.json` |
   | `env` | `env_name` in `config.json` |
   | `method` | `condition`/`algorithm` in `config.json`; relabel `diag_curvature` as `DIIWES-H` |
   | `seed` | `seed` in `config.json` |
   | `iteration` | `history.json` `iteration` |
   | `env_steps` | missing; do not substitute raw `eval_count` without marking it as rollout-count or upper-bound steps |
   | `eval_return` | `history.json` `eval_reward` |
   | `train_population_mean` | `history.json` `mean_fitness` |
   | `train_population_best` | `history.json` `max_fitness` |
   | `best_eval_return_so_far` | `history.json` `best_reward` |
   | `wall_time_sec` | `history.json` `time` |

6. Add DIIWES-H diagnostic fields needed for Figure 6.
   The current logs are enough for approximate ESS, clipping fraction, and broad
   shrinkage/update-norm diagnostics, but the following fields should be logged
   directly for clean plots:

   ```text
   ess_normalized
   clip_fraction
   max_importance_weight
   mean_importance_weight
   explicit_step_norm
   step_norm_ratio
   hessian_shrinkage_median
   hessian_shrinkage_p90
   hessian_shrinkage_max
   lambda
   sigma
   learning_rate
   reuse_fraction
   ```

   Existing approximate mappings:

   | Desired diagnostic | Current source or gap |
   | --- | --- |
   | `ess_normalized` | `ess_ratio` |
   | `clip_fraction` | `clip_frac` |
   | `max_importance_weight` | `w_max`, but confirm whether normalized weight or unclipped ratio is desired |
   | `mean_importance_weight` | missing |
   | `explicit_step_norm` | approximate with `no_curv_pre_trust_step_norm`, but this is pre-trust and no-curvature, not necessarily the explicit baseline step |
   | `step_norm_ratio` | can derive from `step_norm / no_curv_pre_trust_step_norm`; log directly for clarity |
   | `hessian_shrinkage_median` | missing |
   | `hessian_shrinkage_p90` | missing |
   | `hessian_shrinkage_max` | approximate with `step_multiplier_max` |
   | `lambda` | use `l2_coeff`/implicit damping config only after confirming the paper's notation |
   | `sigma` | `noise_std` in `config.json` |
   | `learning_rate` | `lr` in history or `learning_rate` in `config.json` |
   | `reuse_fraction` | `reuse_fraction` in `config.json` |

## Figure-by-Figure Collection Checklist

| Figure | Can plot from current data? | Still needed |
| --- | --- | --- |
| Figure 1: method schematic | Yes; no experiment data required. | Create a vector schematic and use `DIIWES-H` in the caption. |
| Figure 2: aggregate benchmark summary | Not publication-ready. | Multi-seed Standard ES, Explicit IW-ES, no-Hessian DIIWES, and DIIWES-H across all 9 tasks; task normalization references; human-normalized Atari scores; true `env_steps`; IQM/bootstrap table. |
| Figure 3: MuJoCo learning curves | DIIWES-H seed-0 draft only. | Standard ES and Explicit IW-ES baselines, 3-5 seeds, one canonical Humanoid config, true environment-interaction x-axis. |
| Figure 4: Atari learning curves | DIIWES-H seed-0 draft only, with mixed variants. | Canonical Atari preprocessing/budget for all four games, baselines, 3-5 seeds, random/human reference lines, true interaction counts. |
| Figure 5: Hessian ablation | No. | Standard ES, Explicit IW-ES, DIIWES without Hessian, DIIWES-H diagonal Hessian, optional blockwise Hessian; same replay/clipping/lr family; normalized AUC values with seed dots. |
| Figure 6: stability diagnostics | Partial for DIIWES-H only. | Representative-env multi-seed diagnostics plus direct logging for shrinkage median/p90, explicit step norm, update norm ratio, and raw/mean importance-weight statistics. |

## Minimum Next Collection Batch

For the next useful batch, collect a balanced matrix rather than more
single-method variants:

```text
tasks = Humanoid, HalfCheetah, Ant, Hopper, Walker2d,
        Boxing, Freeway, SpaceInvaders, Pong
methods = Standard ES, Explicit IW-ES, DIIWES without Hessian, DIIWES-H
seeds = 0, 1, 2, 3, 4
```

If compute is limited, prioritize:

1. Standard ES and DIIWES-H, 5 seeds, all 9 tasks.
2. Add Explicit IW-ES and no-Hessian DIIWES on Humanoid, Hopper, HalfCheetah, Freeway, and Boxing.
3. Add full ablations across all 9 tasks only after the main comparison is stable.

Do not use the current seed-0 result set as the final publication evidence for
uncertainty intervals, probability of improvement, or aggregate IQM claims.

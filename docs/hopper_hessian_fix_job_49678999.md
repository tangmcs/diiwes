# Hopper Hessian Stabilization Job 49678999

## Status

- Slurm array: `49678999`
- Submitted: July 12, 2026
- Array size: 100 tasks, at most 4 concurrent
- Initial status: tasks `0` through `3` running with empty stderr
- Source digest:
  `895c5370d330d495bb66ad391382c1969d8d1720248093c6a2093d3720f7342f`
- Source archive:
  `job_outputs/hopper_hessian_fix_source_895c5370d330d495bb66ad391382c1969d8d1720248093c6a2093d3720f7342f.tar.gz`
- Archive SHA-256:
  `d311663898eeb58d5c9b283012487112b7d98c6a397684bff55d782c061620ed`

Digest-covered source files must remain unchanged until every queued task has
started because each task independently enforces the source lock.

## Protocol

- Environment: `Hopper-v5`
- Conditions: Standard ES, signed diagonal Hessian, concave-projected diagonal
  Hessian, concave-projected layer-block Hessian, and the block method with a
  bias-corrected `beta=0.9` curvature EMA
- Initial learning rates: `10`, `30`
- Schedules: `alpha_0 / sqrt(t + 1)`, `alpha_0 / (t + 1)`
- Paired seeds: `0` through `4`
- Population: 200 fresh candidates, 100 antithetic pairs
- Updates: 500
- Primary metric: evaluation-return AUC over the first 75,000 training
  environment steps
- Replay, Picard iteration, trust radius, scalar damping, gradient clipping,
  parameter projection, curvature clipping, and fixed update-norm controls:
  disabled

This is a five-seed mechanism screen. A stable arm with consistent paired gains
must be promoted to ten seeds before making a strong performance claim.

## Task Map

| Tasks | Condition |
| --- | --- |
| 0-19 | Standard ES |
| 20-39 | Signed diagonal Hessian control |
| 40-59 | Concave-projected diagonal Hessian |
| 60-79 | Concave-projected layer-block Hessian |
| 80-99 | Concave-projected layer-block Hessian with EMA |

Within each 20-task condition block, tasks `0-4` use inverse square root with
rate 10, `5-9` use inverse square root with rate 30, `10-14` use inverse linear
with rate 10, and `15-19` use inverse linear with rate 30. The seed is the
fastest-changing index.

## Validation

After all 100 tasks finish, run:

```bash
python scripts/summarize_hopper_implicit_sweep.py \
  results/hopper_hessian_fix_ablation_49678999 \
  --protocol hessian_fix \
  --expected-source-sha 895c5370d330d495bb66ad391382c1969d8d1720248093c6a2093d3720f7342f
```

The validator requires the complete matrix, exact task mapping, source and
JSON/JSONL agreement, complete 500-update histories, exact learning-rate
schedules, and condition-specific solver, structure, EMA, denominator,
residual, and no-amplification invariants. It rejects replay, trust controls,
Picard labels, norm controls, and nonfinite records.

#!/bin/bash
#SBATCH --job-name=hop_optdev
#SBATCH --output=job_outputs/hopper_optdev_%A_%a.out
#SBATCH --error=job_outputs/hopper_optdev_%A_%a.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-98%6

set -euo pipefail

source /hpc/home/rt239/miniconda3/bin/activate es_parallel

WORKSPACE_DIR=${PAPER_WORKSPACE_DIR:-/hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes}
REPO_DIR=${PAPER_REPO_DIR:-$WORKSPACE_DIR}
cd "$REPO_DIR"
mkdir -p "$WORKSPACE_DIR/job_outputs"

if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-local}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
SLURMD_NODENAME=${SLURMD_NODENAME:-$(hostname)}

case "${PAPER_DRY_RUN:-}" in
  "") REQUESTED_DRY_RUN=0 ;;
  1|true|TRUE|yes|YES) REQUESTED_DRY_RUN=1 ;;
  0|false|FALSE|no|NO) REQUESTED_DRY_RUN=0 ;;
  *)
    echo "PAPER_DRY_RUN must be a boolean (0/1, false/true, or no/yes)." >&2
    exit 2
    ;;
esac

if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  DRY_RUN=1
else
  DRY_RUN=$REQUESTED_DRY_RUN
fi

if [ -n "${PAPER_ITERATIONS:-}" ]; then
  echo "PAPER_ITERATIONS is forbidden: this development screen is fixed at 250 updates." >&2
  exit 2
fi

CONFIG=configs/mujuco/hopper_fresh_optimizer_development.yaml
MANIFEST=experiments/manifests/hopper_fresh_optimizer_development.json
if [ -n "${PAPER_CONFIG:-}" ] && [ "$PAPER_CONFIG" != "$CONFIG" ]; then
  echo "PAPER_CONFIG cannot override the development protocol." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER}_matplotlib_hopper_optdev_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

N_CELLS=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["cells"]))' "$MANIFEST")
SEEDS=(200 201 202)
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((N_CELLS * N_SEEDS))
TASK_ID=$SLURM_ARRAY_TASK_ID
if [ "$N_CELLS" -ne 33 ] || [ "$TOTAL_TASKS" -ne 99 ]; then
  echo "Manifest must contain exactly 33 cells and 99 paired tasks." >&2
  exit 2
fi
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "Task $TASK_ID is outside matrix size $TOTAL_TASKS." >&2
  exit 2
fi

SLOT_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))
CELL_INDEX=$(((SLOT_INDEX + 11 * SEED_INDEX) % N_CELLS))
SEED=${SEEDS[$SEED_INDEX]}

CELL_FIELDS=$(python -c '
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
index = int(sys.argv[2])
cell = manifest["cells"][index]
if cell["cell_id"] != index:
    raise SystemExit(f"cell_id mismatch at index {index}")
keys = (
    "condition", "label", "learning_rate", "lr_schedule",
    "momentum_beta", "adam_beta1", "adam_beta2", "adam_epsilon",
    "clipup_momentum", "clipup_max_speed",
)
print("\t".join(str(cell.get(key, "NA")) for key in keys))
' "$MANIFEST" "$CELL_INDEX")
IFS=$'\t' read -r CONDITION CELL_LABEL INITIAL_LR LR_SCHEDULE MOMENTUM_BETA ADAM_BETA1 ADAM_BETA2 ADAM_EPSILON CLIPUP_MOMENTUM CLIPUP_MAX_SPEED <<< "$CELL_FIELDS"

WORKERS=${PAPER_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
if [ "$WORKERS" -le 0 ] || [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; then
  echo "PAPER_WORKERS must be between 1 and SLURM_CPUS_PER_TASK-1." >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(python -c "import sys; from experiments.train import _source_digest; print(_source_digest(sys.argv[1]))" "$CONFIG")
ACTUAL_MANIFEST_SHA=$(sha256sum "$MANIFEST" | cut -d' ' -f1)
ACTUAL_LAUNCHER_SHA=$(sha256sum scripts/submit_hopper_fresh_optimizer_development.sh | cut -d' ' -f1)
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
EXPECTED_MANIFEST_SHA=${PAPER_EXPECTED_MANIFEST_SHA256:-}
EXPECTED_LAUNCHER_SHA=${PAPER_EXPECTED_LAUNCHER_SHA256:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && {
  [ -z "$EXPECTED_SOURCE_SHA" ] || [ -z "$EXPECTED_MANIFEST_SHA" ] || [ -z "$EXPECTED_LAUNCHER_SHA" ];
}; then
  echo "Expected source, manifest, and launcher SHA-256 values are required under Slurm." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA" >&2
  exit 2
fi
if [ -n "$EXPECTED_MANIFEST_SHA" ] && [ "$ACTUAL_MANIFEST_SHA" != "$EXPECTED_MANIFEST_SHA" ]; then
  echo "Manifest hash mismatch: expected $EXPECTED_MANIFEST_SHA, found $ACTUAL_MANIFEST_SHA" >&2
  exit 2
fi
if [ -n "$EXPECTED_LAUNCHER_SHA" ] && [ "$ACTUAL_LAUNCHER_SHA" != "$EXPECTED_LAUNCHER_SHA" ]; then
  echo "Launcher hash mismatch: expected $EXPECTED_LAUNCHER_SHA, found $ACTUAL_LAUNCHER_SHA" >&2
  exit 2
fi

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-$WORKSPACE_DIR/results/hopper_fresh_optimizer_development_${SLURM_ARRAY_JOB_ID}}
OUTPUT_DIR="${OUTPUT_ROOT}/cell${CELL_INDEX}_${CELL_LABEL}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
SOURCE_REVISION=${PAPER_SOURCE_GIT_REVISION:-$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')}

TRAIN_ARGS=(
  --config "$CONFIG"
  --condition "$CONDITION"
  --learning-rate "$INITIAL_LR"
  --lr-schedule "$LR_SCHEDULE"
  --reuse-fraction 0
  --seed "$SEED"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR"
)
if [ "$MOMENTUM_BETA" != "NA" ]; then
  TRAIN_ARGS+=(--momentum-beta "$MOMENTUM_BETA")
fi
if [ "$ADAM_BETA1" != "NA" ]; then
  TRAIN_ARGS+=(--adam-beta1 "$ADAM_BETA1" --adam-beta2 "$ADAM_BETA2" --adam-epsilon "$ADAM_EPSILON")
fi
if [ "$CLIPUP_MOMENTUM" != "NA" ]; then
  TRAIN_ARGS+=(--clipup-momentum "$CLIPUP_MOMENTUM" --clipup-max-speed "$CLIPUP_MAX_SPEED")
fi

echo "========================================"
echo "Hopper fresh-only optimizer development"
echo "========================================"
echo "Exploratory: yes; no confirmation claim"
echo "Repository snapshot: $REPO_DIR"
echo "Workspace: $WORKSPACE_DIR"
echo "Source revision: $SOURCE_REVISION"
echo "Source SHA-256: $ACTUAL_SOURCE_SHA"
echo "Manifest SHA-256: $ACTUAL_MANIFEST_SHA"
echo "Launcher SHA-256: $ACTUAL_LAUNCHER_SHA"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $TASK_ID / $((TOTAL_TASKS - 1))"
echo "Slot/cell/seed indices: $SLOT_INDEX / $CELL_INDEX / $SEED_INDEX"
echo "Node: $SLURMD_NODENAME"
echo "Workers: $WORKERS"
echo "Condition: $CONDITION"
echo "Cell label: $CELL_LABEL"
echo "Initial learning rate: $INITIAL_LR"
echo "Learning-rate schedule: $LR_SCHEDULE"
echo "Seed: $SEED"
echo "Updates: 250"
echo "Replay/importance sampling/Picard/trust: disabled"
echo "Dry run: $DRY_RUN"
echo "Output: $OUTPUT_DIR"
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  printf 'Command:'
  printf ' %q' python experiments/train.py "${TRAIN_ARGS[@]}"
  printf '\n'
  echo "Dry run complete; no training launched."
  exit 0
fi

python experiments/train.py "${TRAIN_ARGS[@]}"

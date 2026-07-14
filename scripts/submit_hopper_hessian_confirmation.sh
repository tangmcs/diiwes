#!/bin/bash
#SBATCH --job-name=hop_hconf
#SBATCH --output=job_outputs/hopper_hconf_%A_%a.out
#SBATCH --error=job_outputs/hopper_hconf_%A_%a.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-29%6

set -euo pipefail

source /hpc/home/rt239/miniconda3/bin/activate es_parallel

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

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
  echo "PAPER_ITERATIONS is forbidden: confirmation is fixed at 500 updates." >&2
  exit 2
fi

CONFIG=configs/mujuco/hopper_hessian_confirmation_no_replay.yaml
if [ -n "${PAPER_CONFIG:-}" ] && [ "$PAPER_CONFIG" != "$CONFIG" ]; then
  echo "PAPER_CONFIG cannot override the locked confirmation protocol." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER}_matplotlib_hopper_hconf_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

CONDITIONS=(
  standard_es
  concave_block_ema_curvature_es
  concave_block_ema_isotropic_control_es
)
SEEDS=(100 101 102 103 104 105 106 107 108 109)

N_CONDITIONS=${#CONDITIONS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((N_CONDITIONS * N_SEEDS))
TASK_ID=$SLURM_ARRAY_TASK_ID
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "Task $TASK_ID is outside matrix size $TOTAL_TASKS." >&2
  exit 2
fi

SEED_INDEX=$((TASK_ID / N_CONDITIONS))
SLOT_INDEX=$((TASK_ID % N_CONDITIONS))
CONDITION_INDEX=$(((SLOT_INDEX + SEED_INDEX) % N_CONDITIONS))
SEED=${SEEDS[$SEED_INDEX]}
CONDITION=${CONDITIONS[$CONDITION_INDEX]}

WORKERS=${PAPER_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
if [ "$WORKERS" -le 0 ] || [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; then
  echo "PAPER_WORKERS must be between 1 and SLURM_CPUS_PER_TASK-1." >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(
  python -c "import sys; from experiments.train import _source_digest; print(_source_digest(sys.argv[1]))" "$CONFIG"
)
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_SOURCE_SHA is required for every Slurm task." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA" >&2
  exit 2
fi

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/hopper_hessian_confirmation_${SLURM_ARRAY_JOB_ID}}
OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}_inverse_linear_a10_seed${SEED}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
SOURCE_REVISION=$(git rev-parse --short HEAD 2>/dev/null || printf 'unavailable')
PROTOCOL="Standard ES, structured block-EMA curvature, and isotropic attenuation control; 10/t; fresh-only held-out confirmation"

TRAIN_ARGS=(
  --config "$CONFIG"
  --condition "$CONDITION"
  --learning-rate 10
  --lr-schedule inverse_linear
  --reuse-fraction 0
  --seed "$SEED"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR"
)

echo "========================================"
echo "Hopper Hessian held-out confirmation"
echo "========================================"
echo "Protocol: $PROTOCOL"
echo "Source repository: $REPO_DIR"
echo "Source revision: $SOURCE_REVISION"
echo "Source SHA256: $ACTUAL_SOURCE_SHA"
echo "Expected source SHA256: ${EXPECTED_SOURCE_SHA:-<local dry run>}"
echo "Config: $CONFIG"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / $((TOTAL_TASKS - 1))"
echo "Condition slot: $SLOT_INDEX; rotated condition index: $CONDITION_INDEX"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "Condition: $CONDITION"
echo "Seed: $SEED"
echo "Learning rate: 10/(t+1)"
echo "Updates: 500"
echo "Held-out endpoint: AUC through 75000 training steps, 20 episodes/checkpoint"
echo "Replay/importance sampling/Picard/trust/norm clipping: disabled"
echo "Dry run: $DRY_RUN"
echo "Output: $OUTPUT_DIR"
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete; no training launched."
  exit 0
fi

python experiments/train.py "${TRAIN_ARGS[@]}"

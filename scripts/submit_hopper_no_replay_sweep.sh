#!/bin/bash
#SBATCH --job-name=hop_hessian
#SBATCH --output=job_outputs/hopper_hessian_%A_%a.out
#SBATCH --error=job_outputs/hopper_hessian_%A_%a.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-79%4

set -euo pipefail

source /hpc/home/rt239/miniconda3/bin/activate es_parallel

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
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
  echo "PAPER_ITERATIONS is forbidden: production runs are fixed at 500 updates." >&2
  exit 2
fi

CONFIG=configs/mujuco/hopper_implicit_no_replay.yaml
if [ -n "${PAPER_CONFIG:-}" ] && [ "$PAPER_CONFIG" != "$CONFIG" ]; then
  echo "PAPER_CONFIG cannot override the locked production protocol." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER}_matplotlib_hopper_hessian_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

CONDITIONS=(standard_es linearized_implicit_es)
LR_SCHEDULES=(inverse_sqrt inverse_linear)
INITIAL_LRS=(10 30)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

N_CONDITIONS=${#CONDITIONS[@]}
N_SCHEDULES=${#LR_SCHEDULES[@]}
N_LRS=${#INITIAL_LRS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((N_CONDITIONS * N_SCHEDULES * N_LRS * N_SEEDS))

TASK_ID=$SLURM_ARRAY_TASK_ID
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "Task $TASK_ID is outside matrix size $TOTAL_TASKS." >&2
  exit 2
fi

SEED_INDEX=$((TASK_ID % N_SEEDS))
LR_INDEX=$(((TASK_ID / N_SEEDS) % N_LRS))
SCHEDULE_INDEX=$(((TASK_ID / (N_SEEDS * N_LRS)) % N_SCHEDULES))
CONDITION_INDEX=$((TASK_ID / (N_SEEDS * N_LRS * N_SCHEDULES)))

CONDITION=${CONDITIONS[$CONDITION_INDEX]}
LR_SCHEDULE=${LR_SCHEDULES[$SCHEDULE_INDEX]}
INITIAL_LR=${INITIAL_LRS[$LR_INDEX]}
SEED=${SEEDS[$SEED_INDEX]}
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

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/hopper_hessian_no_picard_no_replay_no_trust_power_schedules_${SLURM_ARRAY_JOB_ID}}
OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}_${LR_SCHEDULE}_a${INITIAL_LR}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
SOURCE_REVISION=$(git rev-parse --short HEAD 2>/dev/null || printf 'unavailable')
PROTOCOL="fresh-only Standard ES vs signed diagonal linearized implicit ES; alpha_t in {alpha_0/sqrt(t+1), alpha_0/(t+1)} with alpha_0 in {10,30}; no Picard/trust/replay/norm control/damping"

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

echo "========================================"
echo "Hopper Hessian ES mentor-requested sweep"
echo "========================================"
echo "Protocol: $PROTOCOL"
echo "Source repository: $REPO_DIR"
echo "Source revision: $SOURCE_REVISION"
echo "Source SHA256: $ACTUAL_SOURCE_SHA"
echo "Expected source SHA256: ${EXPECTED_SOURCE_SHA:-<local dry run>}"
echo "Config: $CONFIG"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID / $((TOTAL_TASKS - 1))"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "Condition: $CONDITION"
echo "Picard endpoint iteration: disabled (not in task matrix)"
echo "Initial learning rate: $INITIAL_LR"
echo "Learning-rate schedule: $LR_SCHEDULE"
echo "Seed: $SEED"
echo "Updates: 500"
echo "Dry run: $DRY_RUN"
echo "Output: $OUTPUT_DIR"
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete; no training launched."
  exit 0
fi

python experiments/train.py "${TRAIN_ARGS[@]}"

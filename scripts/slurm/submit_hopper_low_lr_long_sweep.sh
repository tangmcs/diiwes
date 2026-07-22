#!/bin/bash
#SBATCH --job-name=hop_low_lr_long
#SBATCH --output=job_outputs/hopper_low_lr_long_%A_%a.out
#SBATCH --error=job_outputs/hopper_low_lr_long_%A_%a.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-99%4

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

if [ -n "${SLURM_JOB_ID:-}" ] && \
   [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && \
   [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
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

# A direct/local invocation is always inspection-only. Only Slurm can launch
# training, and Slurm additionally requires a matching expected source digest.
if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  DRY_RUN=1
else
  DRY_RUN=$REQUESTED_DRY_RUN
fi

if [ -f /hpc/home/rt239/miniconda3/bin/activate ]; then
  source /hpc/home/rt239/miniconda3/bin/activate es_parallel
elif [ "$RUNNING_UNDER_SLURM" = "1" ]; then
  echo "The es_parallel Conda activation script is unavailable." >&2
  exit 2
fi

CONFIG=configs/mujoco/hopper.yaml
POPULATION_SIZE=500
BUFFER_SIZE=0
REUSE_FRACTION=0
IMPLICIT_DAMPING=0
N_ITERATIONS=2000
if [ -n "${PAPER_CONFIG:-}" ] && [ "$PAPER_CONFIG" != "$CONFIG" ]; then
  echo "PAPER_CONFIG cannot override the locked main-branch Hopper config." >&2
  exit 2
fi
if [ -n "${PAPER_ITERATIONS:-}" ]; then
  echo "PAPER_ITERATIONS is forbidden: this protocol is fixed at $N_ITERATIONS updates." >&2
  exit 2
fi

CONFIG_DEFAULT_UPDATES=$(python -c \
  'import sys; from experiments.train import load_config; print(load_config(sys.argv[1]).get("n_iterations"))' \
  "$CONFIG")
if [ "$CONFIG_DEFAULT_UPDATES" != "500" ]; then
  echo "$CONFIG must retain its audited 500-update default; found $CONFIG_DEFAULT_UPDATES." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER:-user}_matplotlib_hopper_low_lr_long_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

# Locked 2 x 1 x 5 x 10 matrix. Seed is the fastest-changing index.
CONDITIONS=(standard_es diag_curvature)
LR_SCHEDULES=(inverse_sqrt)
INITIAL_LRS=(0.1 0.25 0.5 1 2)
SEEDS=(0 1 2 3 4 5 6 7 8 9)

N_CONDITIONS=${#CONDITIONS[@]}
N_SCHEDULES=${#LR_SCHEDULES[@]}
N_LRS=${#INITIAL_LRS[@]}
N_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((N_CONDITIONS * N_SCHEDULES * N_LRS * N_SEEDS))
if [ "$TOTAL_TASKS" -ne 100 ]; then
  echo "Internal protocol error: expected exactly 100 tasks, found $TOTAL_TASKS." >&2
  exit 2
fi

TASK_ID=$SLURM_ARRAY_TASK_ID
if ! [[ "$TASK_ID" =~ ^[0-9]+$ ]]; then
  echo "Task ID must be a nonnegative integer; found $TASK_ID." >&2
  exit 2
fi
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
  echo "Task $TASK_ID is outside the locked range 0-$((TOTAL_TASKS - 1))." >&2
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
if [ "$CONDITION_INDEX" -eq 0 ]; then
  PAIRED_TASK_ID=$((TASK_ID + N_SCHEDULES * N_LRS * N_SEEDS))
else
  PAIRED_TASK_ID=$((TASK_ID - N_SCHEDULES * N_LRS * N_SEEDS))
fi

if [ "$LR_SCHEDULE" = "inverse_sqrt" ]; then
  LR_FORMULA="${INITIAL_LR}/sqrt(t+1)"
else
  LR_FORMULA="${INITIAL_LR}/(t+1)"
fi

if ! [[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [ "$SLURM_CPUS_PER_TASK" -le 1 ]; then
  echo "SLURM_CPUS_PER_TASK must be an integer greater than one." >&2
  exit 2
fi
WORKERS=${PAPER_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || \
   [ "$WORKERS" -le 0 ] || \
   [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; then
  echo "PAPER_WORKERS must be an integer from 1 to SLURM_CPUS_PER_TASK-1." >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(
  python -c \
    'import sys; from experiments.train import _source_digest; print(_source_digest(sys.argv[1]))' \
    "$CONFIG"
)
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_SOURCE_SHA is required for every Slurm task." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && ! [[ "$EXPECTED_SOURCE_SHA" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "PAPER_EXPECTED_SOURCE_SHA must be a 64-character SHA-256 digest." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && \
   [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA." >&2
  exit 2
fi

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/hopper_main_hessian_low_lr_long_fresh_no_trust_pop500_${SLURM_ARRAY_JOB_ID}}
OUTPUT_DIR="${OUTPUT_ROOT}/${CONDITION}_${LR_SCHEDULE}_a${INITIAL_LR}_seed${SEED}_job${SLURM_ARRAY_JOB_ID}_task${TASK_ID}"
SOURCE_REVISION=$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')
PROTOCOL="main standard_es vs main diag_curvature; population 500; 2000 updates with prefix horizons 500/1000/2000; fresh-only/no replay; zero scalar damping; inverse-square-root low-rate sweep; trust explicitly disabled"

TRAIN_ARGS=(
  --config "$CONFIG"
  --condition "$CONDITION"
  --population-size "$POPULATION_SIZE"
  --buffer-size "$BUFFER_SIZE"
  --reuse-fraction "$REUSE_FRACTION"
  --implicit-damping "$IMPLICIT_DAMPING"
  --learning-rate "$INITIAL_LR"
  --lr-schedule "$LR_SCHEDULE"
  --iterations "$N_ITERATIONS"
  --trust-radius none
  --seed "$SEED"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR"
)

echo "========================================"
echo "Hopper low-initial-rate, long-horizon Hessian sweep"
echo "========================================"
echo "Protocol: $PROTOCOL"
echo "Source repository: $REPO_DIR"
echo "Source revision: $SOURCE_REVISION"
echo "Source SHA-256: $ACTUAL_SOURCE_SHA"
echo "Expected source SHA-256: ${EXPECTED_SOURCE_SHA:-<local dry run; not required>}"
echo "Config: $CONFIG"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $TASK_ID / $((TOTAL_TASKS - 1))"
echo "Paired task ID: $PAIRED_TASK_ID"
echo "Matrix indices (condition/schedule/rate/seed): $CONDITION_INDEX/$SCHEDULE_INDEX/$LR_INDEX/$SEED_INDEX"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "Condition: $CONDITION"
echo "Population size: $POPULATION_SIZE"
echo "Replay buffer size: $BUFFER_SIZE"
echo "Reuse fraction: $REUSE_FRACTION"
echo "Scalar implicit damping: $IMPLICIT_DAMPING"
echo "Initial learning rate: $INITIAL_LR"
echo "Learning-rate schedule: $LR_SCHEDULE ($LR_FORMULA, t=0,...,$((N_ITERATIONS - 1)))"
echo "Seed: $SEED"
echo "Updates: $N_ITERATIONS"
echo "Predeclared prefix horizons: 500 1000 2000"
echo "Trust radius: none"
echo "Additional requested overrides: population_size=$POPULATION_SIZE, buffer_size=$BUFFER_SIZE, reuse_fraction=$REUSE_FRACTION, implicit_damping=$IMPLICIT_DAMPING"
echo "Dry run: $DRY_RUN"
echo "Output: $OUTPUT_DIR"
printf 'Command:'
printf ' %q' python experiments/train.py "${TRAIN_ARGS[@]}"
printf '\n'
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete; no training launched."
  exit 0
fi

python experiments/train.py "${TRAIN_ARGS[@]}"

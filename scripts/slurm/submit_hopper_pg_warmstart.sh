#!/bin/bash
#SBATCH --job-name=hopper_pg_init
#SBATCH --output=job_outputs/hopper_pg_init_%j.out
#SBATCH --error=job_outputs/hopper_pg_init_%j.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "hopper_pg_init" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

SLURM_JOB_ID=${SLURM_JOB_ID:-local}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
SLURMD_NODENAME=${SLURMD_NODENAME:-$(hostname)}
if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  SLURM_JOB_ID=local
  SLURM_CPUS_PER_TASK=32
fi

case "${PAPER_DRY_RUN:-}" in
  "") REQUESTED_DRY_RUN=0 ;;
  1|true|TRUE|yes|YES) REQUESTED_DRY_RUN=1 ;;
  0|false|FALSE|no|NO) REQUESTED_DRY_RUN=0 ;;
  *)
    echo "PAPER_DRY_RUN must be a boolean." >&2
    exit 2
    ;;
esac
if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  DRY_RUN=1
else
  DRY_RUN=$REQUESTED_DRY_RUN
fi

if [ -f /hpc/home/rt239/miniconda3/bin/activate ]; then
  source /hpc/home/rt239/miniconda3/bin/activate es_parallel
elif [ "$RUNNING_UNDER_SLURM" = "1" ]; then
  echo "The es_parallel Conda environment is unavailable." >&2
  exit 2
fi

if ! [[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [ "$SLURM_CPUS_PER_TASK" -lt 32 ]; then
  echo "The PPO warm start requires at least 32 allocated CPUs." >&2
  exit 2
fi
WORKERS=${PAPER_WORKERS:-30}
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; then
  echo "PAPER_WORKERS must be between 1 and SLURM_CPUS_PER_TASK-1." >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(python -c 'from experiments.hopper_policy_gradient.warmstart import source_digest; print(source_digest())')
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_PG_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_PG_SOURCE_SHA is required for a Slurm run." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "PPO source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA." >&2
  exit 2
fi

OUTPUT_DIR=${PAPER_PG_OUTPUT_DIR:-results/hopper_policy_gradient_init_job${SLURM_JOB_ID}}
if [ -e "$OUTPUT_DIR/manifest.json" ]; then
  echo "Refusing to overwrite completed warm start: $OUTPUT_DIR/manifest.json" >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

COMMAND=(
  python -m experiments.hopper_policy_gradient.warmstart
  --output-dir "$OUTPUT_DIR"
  --workers "$WORKERS"
  --updates 300
  --batch-episodes 32
  --ppo-epochs 6
  --eval-episodes 10
  --master-seed 0
)

echo "Hopper policy-gradient initialization"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Resources: ${SLURM_CPUS_PER_TASK} CPUs, 64G requested, 24 hours"
echo "Workers: $WORKERS"
echo "Protocol: one seed (0), NumPy PPO, up to 300 updates"
echo "Output: $OUTPUT_DIR"
echo "Source SHA-256: $ACTUAL_SOURCE_SHA"

if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY RUN command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"

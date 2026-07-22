#!/bin/bash
#SBATCH --job-name=nl_cartpole_300
#SBATCH --output=job_outputs/nonlinear_cartpole_300_%j.out
#SBATCH --error=job_outputs/nonlinear_cartpole_300_%j.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "nl_cartpole_300" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi
if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  SLURM_JOB_ID=local
fi
SLURMD_NODENAME=${SLURMD_NODENAME:-$(hostname)}

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
  echo "The es_parallel Conda activation script is unavailable." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SOURCE=experiments/nonlinear_cartpole/benchmark.py
ES_UPDATES=300
ANTITHETIC_PAIRS=250
SEEDS=0,1,2,3,4
EVAL_EPISODES=20
OUTPUT_DIR=${PAPER_OUTPUT_ROOT:-reports/nonlinear_cartpole_warm_start_300iter_job${SLURM_JOB_ID}}

CONFIG_DEFAULTS=$(python -c \
  'from experiments.nonlinear_cartpole.benchmark import BenchmarkConfig; c=BenchmarkConfig(); print(f"{c.es_updates},{c.antithetic_pairs},{len(c.seeds)},{c.eval_episodes}")')
if [ "$CONFIG_DEFAULTS" != "300,250,5,20" ]; then
  echo "Locked nonlinear defaults changed unexpectedly: $CONFIG_DEFAULTS" >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(python -c \
  'from experiments.nonlinear_cartpole.benchmark import _source_digest; print(_source_digest())')
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_SOURCE_SHA is required for the Slurm job." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && ! [[ "$EXPECTED_SOURCE_SHA" =~ ^[[:xdigit:]]{64}$ ]]; then
  echo "PAPER_EXPECTED_SOURCE_SHA must be a 64-character SHA-256 digest." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA." >&2
  exit 2
fi

ARGS=(
  --output-dir "$OUTPUT_DIR"
  --seeds "$SEEDS"
  --es-updates "$ES_UPDATES"
  --antithetic-pairs "$ANTITHETIC_PAIRS"
  --eval-episodes "$EVAL_EPISODES"
)

echo "========================================"
echo "Nonlinear CartPole warm-start benchmark"
echo "========================================"
echo "Node: $SLURMD_NODENAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Source: $SOURCE"
echo "Source SHA-256: $ACTUAL_SOURCE_SHA"
echo "Expected SHA-256: ${EXPECTED_SOURCE_SHA:-<local dry run; not required>}"
echo "Seeds: $SEEDS"
echo "ES updates: $ES_UPDATES"
echo "Antithetic pairs per update: $ANTITHETIC_PAIRS"
echo "Candidate population per update: $((2 * ANTITHETIC_PAIRS))"
echo "Evaluation episodes: $EVAL_EPISODES"
echo "Output: $OUTPUT_DIR"
echo "Dry run: $DRY_RUN"
printf 'Command:'
printf ' %q' python -m experiments.nonlinear_cartpole.benchmark "${ARGS[@]}"
printf '\n'
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete; no benchmark launched."
  exit 0
fi

python -m experiments.nonlinear_cartpole.benchmark "${ARGS[@]}"

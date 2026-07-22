#!/bin/bash
#SBATCH --job-name=hop_pg_compare
#SBATCH --output=job_outputs/hop_pg_compare_%j.out
#SBATCH --error=job_outputs/hop_pg_compare_%j.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-}
if [ -z "$OUTPUT_ROOT" ]; then
  echo "PAPER_OUTPUT_ROOT is required." >&2
  exit 2
fi

if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "hop_pg_compare" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

if [ -f /hpc/home/rt239/miniconda3/bin/activate ]; then
  source /hpc/home/rt239/miniconda3/bin/activate es_parallel
elif [ "$RUNNING_UNDER_SLURM" = "1" ]; then
  echo "The es_parallel Conda environment is unavailable." >&2
  exit 2
fi

COMMAND=(python scripts/summarize_hopper_pginit_noema.py --root "$OUTPUT_ROOT")
echo "Comparison input/output root: $OUTPUT_ROOT"
if [ "$RUNNING_UNDER_SLURM" = "0" ] || [ "${PAPER_DRY_RUN:-0}" = "1" ]; then
  printf 'DRY RUN command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"

#!/bin/bash
#SBATCH --job-name=convex_impl
#SBATCH --output=job_outputs/convex_implicit_%j.out
#SBATCH --error=job_outputs/convex_implicit_%j.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

if [ -n "${SLURM_JOB_ID:-}" ] && [ "${SLURM_JOB_NAME:-}" = "convex_impl" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  SLURM_JOB_ID=local
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
  echo "The es_parallel Conda activation script is unavailable." >&2
  exit 2
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

SOURCE=experiments/implicit_quadratic_optimization_benchmark.py
DIMENSION=12
NUM_BLOCKS=3
POPULATION_SIZE=500
ITERATIONS=500
SIGMA=0.1
ALPHAS=0.05,0.1,0.25,0.5,0.75,1,1.5,2
MC_SEEDS=0,1,2,3,4,5,6,7,8,9
CASES=block_aligned_concave,rotated_concave,block_aligned_additive_noise
FITNESS_TRANSFORMS=raw
CHECKPOINTS=10,30,100,300,500
OUTPUT_DIR=${PAPER_OUTPUT_ROOT:-results/implicit_quadratic_low_step_sweep_${SLURM_JOB_ID}}

ACTUAL_SOURCE_SHA=$(python -c \
  'import hashlib, pathlib, sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
  "$SOURCE")
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_SOURCE_SHA is required for the Slurm job." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && \
   [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA." >&2
  exit 2
fi

ARGS=(
  --output-dir "$OUTPUT_DIR"
  --dimension "$DIMENSION"
  --num-blocks "$NUM_BLOCKS"
  --population-size "$POPULATION_SIZE"
  --iterations "$ITERATIONS"
  --sigma "$SIGMA"
  --alphas "$ALPHAS"
  --mc-seeds "$MC_SEEDS"
  --cases "$CASES"
  --fitness-transforms "$FITNESS_TRANSFORMS"
)

echo "========================================"
echo "Controlled convex implicit-step sweep"
echo "========================================"
echo "Interpretation: minimize a strongly convex quadratic by maximizing its negative reward"
echo "Source: $SOURCE"
echo "Source SHA-256: $ACTUAL_SOURCE_SHA"
echo "Expected SHA-256: ${EXPECTED_SOURCE_SHA:-<local dry run; not required>}"
echo "Dimension / blocks: $DIMENSION / $NUM_BLOCKS"
echo "Cases: $CASES"
echo "Population: $POPULATION_SIZE"
echo "Iterations: $ITERATIONS"
echo "Predeclared checkpoints: $CHECKPOINTS"
echo "Constant step sizes: $ALPHAS"
echo "Monte Carlo seeds: $MC_SEEDS"
echo "Fitness transform: $FITNESS_TRANSFORMS"
echo "Trust / replay / additive scalar damping: not present"
echo "Analysis control: equal-norm isotropic comparators are included separately"
echo "Output: $OUTPUT_DIR"
echo "Dry run: $DRY_RUN"
printf 'Command:'
printf ' %q' python "$SOURCE" "${ARGS[@]}"
printf '\n'
date
echo "========================================"

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry run complete; no benchmark launched."
  exit 0
fi

python "$SOURCE" "${ARGS[@]}"

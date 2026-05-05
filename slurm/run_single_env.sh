#!/bin/bash
#SBATCH --job-name=diiwes_single
#SBATCH --output=job_outputs/diiwes_single_%j.out
#SBATCH --error=job_outputs/diiwes_single_%j.err
#SBATCH --partition=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

source /hpc/home/rt239/miniconda3/bin/activate es_parallel
cd /hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes

mkdir -p job_outputs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER}_matplotlib_diiwes_single_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

CONFIG=${PAPER_CONFIG:?Set PAPER_CONFIG to a config path, e.g. configs/mujuco/halfcheetah.yaml}
CONDITION=${PAPER_CONDITION:-diag_curvature}
LR_VALUE=${PAPER_LR:-}
SEED_VALUE=${PAPER_SEED:-0}
WORKERS=${PAPER_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
VERBOSE=${PAPER_VERBOSE:-0}
REUSE_FRACTION=${PAPER_REUSE_FRACTION:-}
RANK_FITNESS=${PAPER_RANK_FITNESS:-}

CONFIG_LR=$(python -c "import yaml; cfg=yaml.safe_load(open('$CONFIG', 'r', encoding='utf-8')); print(cfg.get('learning_rate', 0.02))")
LR_EFFECTIVE=${LR_VALUE:-$CONFIG_LR}
LR_LABEL=${LR_EFFECTIVE//./p}
ENV_LABEL=${PAPER_ENV_LABEL:-$(python -c "import re, yaml; cfg=yaml.safe_load(open('$CONFIG', 'r', encoding='utf-8')); print(re.sub(r'[^A-Za-z0-9]+', '_', cfg['env_name']).strip('_'))")}
REUSE_LABEL=""
if [ -n "$REUSE_FRACTION" ]; then
    REUSE_LABEL="_reuse${REUSE_FRACTION//./p}"
fi
OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/${ENV_LABEL}_${CONDITION}_lr${LR_LABEL}${REUSE_LABEL}}
OUTPUT_DIR="${OUTPUT_ROOT}/${ENV_LABEL}_${CONDITION}_lr${LR_LABEL}_seed${SEED_VALUE}_job${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_ROOT"

echo "========================================"
echo "DIIWES single environment run"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "Config: $CONFIG"
echo "Environment: $ENV_LABEL"
echo "Condition: $CONDITION"
echo "Learning rate: $LR_EFFECTIVE"
echo "Learning rate override: ${LR_VALUE:-<config>}"
echo "Seed: $SEED_VALUE"
echo "Reuse fraction override: ${REUSE_FRACTION:-<config>}"
echo "Rank fitness override: ${RANK_FITNESS:-<config>}"
echo "Verbose: $VERBOSE"
echo "Output: $OUTPUT_DIR"
date
echo "========================================"

TRAIN_ARGS=(
    --config "$CONFIG"
    --condition "$CONDITION"
    --seed "$SEED_VALUE"
    --workers "$WORKERS"
    --output "$OUTPUT_DIR"
)

if [ -n "$LR_VALUE" ]; then
    TRAIN_ARGS+=(--learning-rate "$LR_VALUE")
fi

if [ -n "$REUSE_FRACTION" ]; then
    TRAIN_ARGS+=(--reuse-fraction "$REUSE_FRACTION")
fi

if [ -n "$RANK_FITNESS" ]; then
    TRAIN_ARGS+=(--rank-fitness "$RANK_FITNESS")
fi

if [ "$VERBOSE" = "1" ]; then
    TRAIN_ARGS+=(--verbose)
fi

python experiments/train.py "${TRAIN_ARGS[@]}"

echo "========================================"
echo "DIIWES single environment run complete"
date
echo "Output: $OUTPUT_DIR"
echo "========================================"

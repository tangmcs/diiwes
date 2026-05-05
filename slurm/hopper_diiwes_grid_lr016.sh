#!/bin/bash
#SBATCH --job-name=diiwes_hopper
#SBATCH --output=job_outputs/diiwes_hopper_lr016_%A_%a.out
#SBATCH --error=job_outputs/diiwes_hopper_lr016_%A_%a.err
#SBATCH --partition=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-8%3

set -e

source /hpc/home/rt239/miniconda3/bin/activate es_parallel
cd /hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes

mkdir -p job_outputs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER}_matplotlib_diiwes_hopper_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

CONFIG=${PAPER_CONFIG:-configs/mujuco/hopper.yaml}
LR_VALUE=${PAPER_LR:-0.16}
WORKERS=${PAPER_WORKERS:-$((SLURM_CPUS_PER_TASK - 2))}
LR_LABEL=${LR_VALUE//./p}
ENV_LABEL=${PAPER_ENV_LABEL:-$(python -c "import re, yaml; cfg=yaml.safe_load(open('$CONFIG', 'r', encoding='utf-8')); print(re.sub(r'[^A-Za-z0-9]+', '_', cfg['env_name']).strip('_'))")}
OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/${ENV_LABEL}_lr${LR_LABEL}}
mkdir -p "$OUTPUT_ROOT"

CONDITIONS=(
  "standard_es"
  "no_curvature"
  "diag_curvature"
)
SEEDS=("0" "1" "2")
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
CONDITION_INDEX=$((TASK_ID / N_SEEDS))
SEED_INDEX=$((TASK_ID % N_SEEDS))

CONDITION=${CONDITIONS[$CONDITION_INDEX]}
SEED_VALUE=${SEEDS[$SEED_INDEX]}

OUTPUT_DIR="${OUTPUT_ROOT}/${ENV_LABEL}_${CONDITION}_lr${LR_LABEL}_seed${SEED_VALUE}_job${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "========================================"
echo "DIIWES ${ENV_LABEL} grid"
echo "========================================"
echo "Array job ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Workers: $WORKERS"
echo "Config: $CONFIG"
echo "Condition: $CONDITION"
echo "Learning rate: $LR_VALUE"
echo "Seed: $SEED_VALUE"
echo "Output: $OUTPUT_DIR"
date
echo "========================================"

python experiments/train.py \
    --config "$CONFIG" \
    --condition "$CONDITION" \
    --learning-rate "$LR_VALUE" \
    --seed "$SEED_VALUE" \
    --workers "$WORKERS" \
    --output "$OUTPUT_DIR"

echo "========================================"
echo "DIIWES ${ENV_LABEL} grid task complete"
date
echo "Output: $OUTPUT_DIR"
echo "========================================"

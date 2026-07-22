#!/bin/bash
#SBATCH --job-name=hop_pg_noema
#SBATCH --output=job_outputs/hop_pg_noema_%A_%a.out
#SBATCH --error=job_outputs/hop_pg_noema_%A_%a.err
#SBATCH --partition=common
#SBATCH --chdir=/hpc/home/rt239/diiwes
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2

set -euo pipefail

REPO_DIR=${PAPER_REPO_DIR:-/hpc/home/rt239/diiwes}
cd "$REPO_DIR"
mkdir -p job_outputs

if [ -n "${SLURM_JOB_ID:-}" ] && \
   { [ "${SLURM_JOB_NAME:-}" = "hop_pg_noema" ] || [ "${SLURM_JOB_NAME:-}" = "hop_pg_hi05" ]; } && \
   [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-local}
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-64}
SLURMD_NODENAME=${SLURMD_NODENAME:-$(hostname)}
if [ "$RUNNING_UNDER_SLURM" = "0" ]; then
  SLURM_CPUS_PER_TASK=64
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

PROTOCOL_VARIANT=${PAPER_PROTOCOL_VARIANT:-baseline}
case "$PROTOCOL_VARIANT" in
  baseline)
    CONFIG=configs/mujoco/hopper_pginit_noema_1000pairs.yaml
    LEARNING_RATE=0.16
    OUTPUT_PREFIX=hopper_pginit_noema_pairs1000_iter300_seed0
    ;;
  high_lr0p5)
    CONFIG=configs/mujoco/hopper_pginit_noema_lr0p5_1000pairs.yaml
    LEARNING_RATE=0.5
    OUTPUT_PREFIX=hopper_pginit_noema_lr0p5_pairs1000_iter300_seed0
    ;;
  *)
    echo "PAPER_PROTOCOL_VARIANT must be baseline or high_lr0p5." >&2
    exit 2
    ;;
esac
CONFIG_CHECK=$(python -c 'import sys; from experiments.train import load_config; c=load_config(sys.argv[1]); lr=float(sys.argv[2]); required={"env_name":"Hopper-v5","population_size":2000,"learning_rate":lr,"n_iterations":300,"buffer_size":0,"reuse_fraction":0.0,"implicit_damping":0.0,"curvature_beta":0.0,"bias_correct_curvature_ema":False,"antithetic":True}; bad={k:(c.get(k),v) for k,v in required.items() if c.get(k)!=v}; print("ok" if not bad else repr(bad))' "$CONFIG" "$LEARNING_RATE")
if [ "$CONFIG_CHECK" != "ok" ]; then
  echo "Locked Hopper config validation failed: $CONFIG_CHECK" >&2
  exit 2
fi

if ! [[ "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]; then
  echo "Array task must be 0 or 1." >&2
  exit 2
fi
CONDITIONS=(standard_es diag_curvature)
CONDITION=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}
SEED=0

if ! [[ "$SLURM_CPUS_PER_TASK" =~ ^[0-9]+$ ]] || [ "$SLURM_CPUS_PER_TASK" -lt 64 ]; then
  echo "Each Hopper arm requires at least 64 allocated CPUs." >&2
  exit 2
fi
WORKERS=${PAPER_WORKERS:-62}
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [ "$WORKERS" -lt 1 ] || [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; then
  echo "PAPER_WORKERS must be between 1 and SLURM_CPUS_PER_TASK-1." >&2
  exit 2
fi

WARMSTART_DIR=${PAPER_WARMSTART_DIR:-}
if [ -z "$WARMSTART_DIR" ]; then
  echo "PAPER_WARMSTART_DIR is required." >&2
  exit 2
fi
MANIFEST="$WARMSTART_DIR/manifest.json"
PARAMS="$WARMSTART_DIR/policy_params.npy"
OBS_NORM="$WARMSTART_DIR/obs_norm.npz"
if [ ! -f "$MANIFEST" ] || [ ! -f "$PARAMS" ] || [ ! -f "$OBS_NORM" ]; then
  echo "The policy-gradient checkpoint is incomplete in $WARMSTART_DIR." >&2
  exit 2
fi

MANIFEST_CHECK=$(python -c 'import json,sys; m=json.load(open(sys.argv[1])); expected={"status":"complete","method":"numpy_ppo_policy_gradient","environment":"Hopper-v5","actor_parameter_count":5123,"actor_dimensions":[11,64,64,3]}; bad={k:(m.get(k),v) for k,v in expected.items() if m.get(k)!=v}; print("ok" if not bad else repr(bad))' "$MANIFEST")
if [ "$MANIFEST_CHECK" != "ok" ]; then
  echo "Warm-start manifest validation failed: $MANIFEST_CHECK" >&2
  exit 2
fi
PARAMS_SHA=$(sha256sum "$PARAMS" | awk '{print $1}')
OBS_SHA=$(sha256sum "$OBS_NORM" | awk '{print $1}')
EXPECTED_PARAMS_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["policy_params_sha256"])' "$MANIFEST")
EXPECTED_OBS_SHA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["obs_norm_sha256"])' "$MANIFEST")
if [ "$PARAMS_SHA" != "$EXPECTED_PARAMS_SHA" ] || [ "$OBS_SHA" != "$EXPECTED_OBS_SHA" ]; then
  echo "Warm-start artifact digest mismatch." >&2
  exit 2
fi

ACTUAL_SOURCE_SHA=$(python -c 'import sys; from experiments.train import _source_digest; print(_source_digest(sys.argv[1]))' "$CONFIG")
EXPECTED_SOURCE_SHA=${PAPER_EXPECTED_SOURCE_SHA:-}
if [ "$RUNNING_UNDER_SLURM" = "1" ] && [ -z "$EXPECTED_SOURCE_SHA" ]; then
  echo "PAPER_EXPECTED_SOURCE_SHA is required for a Slurm run." >&2
  exit 2
fi
if [ -n "$EXPECTED_SOURCE_SHA" ] && [ "$ACTUAL_SOURCE_SHA" != "$EXPECTED_SOURCE_SHA" ]; then
  echo "Trainer source hash mismatch: expected $EXPECTED_SOURCE_SHA, found $ACTUAL_SOURCE_SHA." >&2
  exit 2
fi

OUTPUT_ROOT=${PAPER_OUTPUT_ROOT:-results/${OUTPUT_PREFIX}_job${SLURM_ARRAY_JOB_ID}}
OUTPUT_DIR="$OUTPUT_ROOT/${CONDITION}_seed0_task${SLURM_ARRAY_TASK_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/${USER:-user}_hopper_pg_noema_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

COMMAND=(
  python experiments/train.py
  --config "$CONFIG"
  --condition "$CONDITION"
  --population-size 2000
  --buffer-size 0
  --reuse-fraction 0
  --implicit-damping 0
  --learning-rate "$LEARNING_RATE"
  --lr-schedule constant
  --trust-radius none
  --curvature-beta 0
  --bias-correct-curvature-ema false
  --iterations 300
  --initial-params "$PARAMS"
  --initial-obs-norm "$OBS_NORM"
  --initial-params-sha256 "$PARAMS_SHA"
  --initial-obs-norm-sha256 "$OBS_SHA"
  --seed "$SEED"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR"
)

echo "Hopper PPO-initialized ES comparison"
echo "Array job/task: ${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}"
echo "Node: $SLURMD_NODENAME"
echo "Resources per arm: ${SLURM_CPUS_PER_TASK} CPUs, 96G requested, 36 hours"
echo "Workers: $WORKERS"
echo "Condition: $CONDITION"
echo "Seed: $SEED (the only seed)"
echo "Protocol variant: $PROTOCOL_VARIANT"
echo "Constant learning rate: $LEARNING_RATE"
echo "Budget: 1000 antithetic pairs (2000 policies) x 300 iterations"
echo "EMA: disabled (curvature_beta=0, bias correction=false)"
echo "Replay/scalar damping/trust radius: disabled"
echo "Warm start: $WARMSTART_DIR"
echo "Warm-start parameter SHA-256: $PARAMS_SHA"
echo "Trainer source SHA-256: $ACTUAL_SOURCE_SHA"
echo "Output: $OUTPUT_DIR"

if [ "$DRY_RUN" = "1" ]; then
  printf 'DRY RUN command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

"${COMMAND[@]}"

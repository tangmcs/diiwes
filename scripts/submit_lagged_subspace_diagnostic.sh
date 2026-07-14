#!/bin/bash
#SBATCH --job-name=lagsubdiag
#SBATCH --output=job_outputs/lagged_diagnostic_%A_%a.out
#SBATCH --error=job_outputs/lagged_diagnostic_%A_%a.err
#SBATCH --partition=common
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-179%6

set -euo pipefail

WORKSPACE_DIR=${PAPER_WORKSPACE_DIR:-/hpc/home/rt239/clean_implementation/importance_sampling_es/diiwes}
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  RUNNING_UNDER_SLURM=1
else
  RUNNING_UNDER_SLURM=0
fi

case "${PAPER_DRY_RUN:-}" in
  "") REQUESTED_DRY_RUN=0 ;;
  1|true|TRUE|yes|YES) REQUESTED_DRY_RUN=1 ;;
  0|false|FALSE|no|NO) REQUESTED_DRY_RUN=0 ;;
  *) echo "PAPER_DRY_RUN must be boolean." >&2; exit 2 ;;
esac
if [ "$RUNNING_UNDER_SLURM" = 0 ]; then
  DRY_RUN=1
else
  DRY_RUN=$REQUESTED_DRY_RUN
fi

if [ "$RUNNING_UNDER_SLURM" = 1 ] && [ -z "${PAPER_REPO_DIR:-}" ]; then
  echo "PAPER_REPO_DIR must identify the immutable source snapshot under Slurm." >&2
  exit 2
fi
REPO_DIR=${PAPER_REPO_DIR:-$(pwd)}
if [ ! -d "$REPO_DIR" ] || [ -L "$REPO_DIR" ]; then
  echo "PAPER_REPO_DIR must be an existing non-symlink directory." >&2
  exit 2
fi
REPO_DIR=$(cd "$REPO_DIR" && pwd -P)
cd "$REPO_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
source /hpc/home/rt239/miniconda3/bin/activate es_parallel
python -m experiments.lagged_subspace_study_lock validate-runtime >/dev/null

for variable in \
  PAPER_EXPECTED_SOURCE_SHA \
  PAPER_EXPECTED_MANIFEST_SHA256 \
  PAPER_EXPECTED_PROTOCOL_SHA256 \
  PAPER_EXPECTED_ANALYZER_SHA256 \
  PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256 \
  PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256; do
  if [ -z "${!variable:-}" ]; then
    echo "$variable is mandatory." >&2
    exit 2
  fi
done
for forbidden in PAPER_CONFIG PAPER_SEED PAPER_GENERATION PAPER_CONDITION PAPER_LEARNING_RATE PAPER_REUSE_FRACTION PAPER_CHUNK_PAIRS; do
  if [ "${!forbidden+x}" = x ]; then
    echo "$forbidden is forbidden by the preregistered diagnostic protocol." >&2
    exit 2
  fi
done

MANIFEST=experiments/manifests/lagged_subspace_frozen_checkpoint.json
PROTOCOL=docs/lagged_subspace_frozen_checkpoint_protocol.md
ANALYZER=scripts/analyze_lagged_subspace_frozen_checkpoint.py
LAUNCHER_BUNDLE=experiments/manifests/lagged_subspace_launcher_lock.json
DEPENDENCY_BUNDLE=experiments/manifests/lagged_subspace_dependency_lock.json

export PAPER_EXPECTED_SOURCE_SHA
python -m experiments.lagged_subspace_study_lock verify \
  --snapshot-root "$REPO_DIR" \
  --expected "$PAPER_EXPECTED_SOURCE_SHA" >/dev/null
python -m experiments.lagged_subspace_study_lock validate-mappings \
  --snapshot-root "$REPO_DIR" >/dev/null
python -m experiments.lagged_subspace_study_lock verify-bundle \
  --snapshot-root "$REPO_DIR" --bundle "$LAUNCHER_BUNDLE" \
  --kind launchers --expected "$PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256" >/dev/null
python -m experiments.lagged_subspace_study_lock verify-bundle \
  --snapshot-root "$REPO_DIR" --bundle "$DEPENDENCY_BUNDLE" \
  --kind dependency_locks --expected "$PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256" >/dev/null

verify_file_lock() {
  local path=$1
  local expected=$2
  local label=$3
  local actual
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  if [ "$actual" != "$expected" ]; then
    echo "$label SHA-256 mismatch: expected $expected, found $actual" >&2
    exit 2
  fi
}
verify_file_lock "$MANIFEST" "$PAPER_EXPECTED_MANIFEST_SHA256" manifest
verify_file_lock "$PROTOCOL" "$PAPER_EXPECTED_PROTOCOL_SHA256" protocol
verify_file_lock "$ANALYZER" "$PAPER_EXPECTED_ANALYZER_SHA256" analyzer

ARTIFACT_ROOT=${PAPER_ARTIFACT_ROOT:-$WORKSPACE_DIR/results/lagged_subspace_frozen_checkpoint_${PAPER_EXPECTED_SOURCE_SHA:0:16}}
ARTIFACT_ROOT=$(realpath -m "$ARTIFACT_ROOT")
EXPECTED_SNAPSHOT="$ARTIFACT_ROOT/source_snapshot_${PAPER_EXPECTED_SOURCE_SHA}"
if [ "$RUNNING_UNDER_SLURM" = 1 ] && [ "$REPO_DIR" != "$EXPECTED_SNAPSHOT" ]; then
  echo "PAPER_REPO_DIR must equal the immutable study snapshot $EXPECTED_SNAPSHOT" >&2
  exit 2
fi

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
IFS=$'\t' read -r CHECKPOINT_ID TRAINING_ID TASK_INDEX ENV_NAME SEED GENERATION CONFIG < <(
  python -m experiments.lagged_subspace_study_lock checkpoint-map "$TASK_ID"
)
if [ "$CHECKPOINT_ID" -ne "$TASK_ID" ]; then
  echo "Diagnostic array mapping is not identity-indexed." >&2
  exit 2
fi

SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}
WORKERS=30
CHUNK_PAIRS=32
if [ "$DRY_RUN" = 0 ] && { [ "$SLURM_CPUS_PER_TASK" -lt 32 ] || [ "$WORKERS" -ge "$SLURM_CPUS_PER_TASK" ]; }; then
  echo "Diagnostic generation requires at least 32 CPUs and reserves two CPUs." >&2
  exit 2
fi

TRAINING_DIR=$(printf '%s/training_runs/training_%06d' "$ARTIFACT_ROOT" "$TRAINING_ID")
CHECKPOINT=$(printf '%s/checkpoints/checkpoint_generation_%06d.npz' "$TRAINING_DIR" "$GENERATION")
CAPTURE_MANIFEST="$TRAINING_DIR/checkpoint_capture.json"
TRAINING_CONFIG="$TRAINING_DIR/checkpoint_training_config.json"
STDERR_PATH=$(printf '%s/stderr/diagnostic/checkpoint_%06d.stderr' "$ARTIFACT_ROOT" "$CHECKPOINT_ID")

if [ "$DRY_RUN" = 0 ]; then
  for required in "$CHECKPOINT" "$CAPTURE_MANIFEST" "$TRAINING_CONFIG"; do
    if [ ! -f "$required" ] || [ -L "$required" ]; then
      echo "Required checkpoint-lineage file is missing or a symlink: $required" >&2
      exit 2
    fi
  done
fi

export MPLCONFIGDIR="/tmp/${USER}_lagged_diagnostic_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "study=lagged_subspace_frozen_checkpoint stage=diagnostic"
echo "source_sha256=$PAPER_EXPECTED_SOURCE_SHA snapshot=$REPO_DIR"
echo "task_id=$TASK_ID checkpoint_id=$CHECKPOINT_ID training_id=$TRAINING_ID task_index=$TASK_INDEX env=$ENV_NAME seed=$SEED generation=$GENERATION"
echo "workers=$WORKERS chunk_pairs=$CHUNK_PAIRS artifact_root=$ARTIFACT_ROOT stderr=$STDERR_PATH dry_run=$DRY_RUN"

COMMAND=(
  python experiments/run_lagged_subspace_checkpoint_diagnostic.py
  --manifest "$MANIFEST"
  --expected-manifest-sha256 "$PAPER_EXPECTED_MANIFEST_SHA256"
  --expected-source-sha256 "$PAPER_EXPECTED_SOURCE_SHA"
  --checkpoint "$CHECKPOINT"
  --checkpoint-capture-manifest "$CAPTURE_MANIFEST"
  --training-config "$TRAINING_CONFIG"
  --task-index "$TASK_INDEX"
  --training-seed "$SEED"
  --generation "$GENERATION"
  --artifact-root "$ARTIFACT_ROOT"
  --n-workers "$WORKERS"
  --chunk-pairs "$CHUNK_PAIRS"
)
if [ "$DRY_RUN" = 1 ]; then
  printf 'Command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$ARTIFACT_ROOT/checkpoint_artifacts" "$ARTIFACT_ROOT/stderr/diagnostic" "$WORKSPACE_DIR/job_outputs"
: > "$STDERR_PATH"
"${COMMAND[@]}" 2> "$STDERR_PATH"

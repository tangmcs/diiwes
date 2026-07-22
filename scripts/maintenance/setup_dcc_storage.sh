#!/usr/bin/env bash

# Create the repository's unversioned DCC storage directories and links.
# Existing real files or directories are never moved or overwritten.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
DIIWES_REPO_ROOT=${DIIWES_REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd -P)}
DIIWES_NETID=${DIIWES_NETID:-${USER:?USER is required when DIIWES_NETID is unset}}
DIIWES_WORK_ROOT=${DIIWES_WORK_ROOT:-/work/${DIIWES_NETID}/diiwes}

ensure_link() {
  local link_path=$1
  local target_path=$2

  if [ -L "$link_path" ]; then
    if [ "$(readlink -f -- "$link_path")" = "$(readlink -f -- "$target_path")" ]; then
      printf 'ok: %s -> %s\n' "$link_path" "$target_path"
      return
    fi
    printf 'Refusing to replace symlink with a different target: %s\n' "$link_path" >&2
    exit 2
  fi

  if [ -e "$link_path" ]; then
    printf 'Refusing to replace existing path: %s\n' "$link_path" >&2
    exit 2
  fi

  ln -s -- "$target_path" "$link_path"
  printf 'created: %s -> %s\n' "$link_path" "$target_path"
}

mkdir -p -- \
  "$DIIWES_WORK_ROOT/results" \
  "$DIIWES_WORK_ROOT/job_outputs" \
  "$DIIWES_WORK_ROOT/reports" \
  "$DIIWES_WORK_ROOT/figures" \
  "$DIIWES_WORK_ROOT/archive"

ensure_link "$DIIWES_REPO_ROOT/results" "$DIIWES_WORK_ROOT/results"
ensure_link "$DIIWES_REPO_ROOT/job_outputs" "$DIIWES_WORK_ROOT/job_outputs"
ensure_link "$DIIWES_REPO_ROOT/reports" "$DIIWES_WORK_ROOT/reports"
ensure_link "$DIIWES_REPO_ROOT/figures" "$DIIWES_WORK_ROOT/figures"
ensure_link "$DIIWES_REPO_ROOT/archive" "$DIIWES_WORK_ROOT/archive"

printf 'DCC storage root ready: %s\n' "$DIIWES_WORK_ROOT"

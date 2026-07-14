# Reproducibility And Artifact Plan

Status: implementation checklist for the locked lagged-subspace
frozen-checkpoint study. This document inventories what exists on July 13,
2026 and what must still be produced. It is not evidence that the mechanism
works, an optimizer result, or an artifact-availability statement.

## 1. Scope And Acceptance Meaning

The artifact supports one preregistered **frozen-checkpoint mechanism
diagnostic** on `Hopper-v5`, `Walker2d-v5`, and `HalfCheetah-v5`. Its claim
boundary is fixed in
`docs/lagged_subspace_frozen_checkpoint_protocol.md`: it cannot establish
training-trajectory improvement, optimizer superiority, or sample efficiency.
Replay, importance sampling, trust clipping, Picard iteration, gradient and
curvature clipping, parameter projection, L2, momentum, and curvature EMA are
absent from this study. The prespecified structured arm does apply spectral
concave projection inside its rank-three local operator; that mechanism is
part of the tested method, not an omitted control.

Artifact acceptance and scientific advancement are separate decisions:

- **Artifact accepted** means that the archive, source locks, lineage,
  numerical records, preregistered analysis, and paper outputs all pass the
  checks below.
- **Mechanism advances** means that the validated analysis sets
  `mechanism_advances_to_optimizer_pilot=true`, which requires at least two of
  three tasks to pass every locked gate.
- A valid negative result must pass artifact acceptance. It must not be
  relabeled as an infrastructure failure or removed.
- A positive mechanism diagnostic is not an optimizer claim. It permits a new
  optimizer pilot followed by a separately locked, untouched-seed
  confirmation.

## 2. Current Inventory

Use the status labels `COMPLETE`, `RUNNING`, and `MISSING` literally. Do not
promote an item based on a Slurm state, a plausible file count, or an
unvalidated partial output.

| Item | Status on July 13, 2026 | Authoritative evidence |
| --- | --- | --- |
| Final preregistered protocol | COMPLETE | Protocol hash in Section 3 and manifest `protocol_status` |
| Machine-readable design and budget | COMPLETE | `experiments/manifests/lagged_subspace_frozen_checkpoint.json` |
| Composite source snapshot | COMPLETE | Read-only 31-file snapshot; no symlinks; composite hash in Section 3 |
| Launcher and dependency bundles | COMPLETE | Both bundle manifests and hashes in Section 3 |
| Checkpoint-generating runs | COMPLETE | `60/60` `status.json` records say `complete`; each has generations 50, 150, and 250 |
| Frozen checkpoints | COMPLETE | 180 predeclared checkpoints from 3 tasks x 20 seeds x 3 generations |
| Checkpoint-stage validator | COMPLETE in mutable release layer | `scripts/validate_lagged_subspace_checkpoint_stage.py`; focused tests pass and read-only production validation accepted 60 runs, 180 checkpoints, and 60 empty stderr files |
| Durable checkpoint-stage report | MISSING | The live artifact was deliberately not modified after launch; generate the write-once report in the release or independent-rerun root |
| Diagnostic array | COMPLETE | All 180 original elements of Slurm array `49720838` committed their predeclared artifacts; the four infrastructure-held elements completed in place |
| Complete diagnostic fragments | COMPLETE | The locked assembler accepted 180 checkpoint artifact directories, 180 empty diagnostic stderr files, and no temporary directories |
| Immutable audit index | COMPLETE | `audit_index.json`, 292,664,873 bytes, SHA-256 `4fd609b08a3bc78731494572145102951bf8da5389ea10b0aa11abc6eafc1d19` |
| Preregistered analysis JSON | MISSING | Must be created from the validated audit index by the locked analyzer |
| Paper figures and tables | MISSING | The deterministic renderer exists, but no outputs may be generated before the preregistered analysis artifact exists |
| Full compute disclosure | MISSING | The collector exists, but final wall time, CPU-hours, MaxRSS, retries, hardware, energy method, and final storage have not been assembled |
| Clean-machine reproduction record | MISSING | No independent environment build and rerun report exists |
| Anonymous release archive | MISSING | Current files contain cluster paths and host metadata and must not be uploaded as-is |
| License and citation metadata | MISSING | No top-level `LICENSE` or `CITATION.cff` exists |
| Container or full transitive lock | MISSING | `environment.yml` pins direct packages only; OS, libc, driver, and transitive packages are not fully locked |

The live artifact root is:

```text
results/lagged_subspace_frozen_checkpoint_7120047c6891def1
```

Its required final scientific subtrees are:

```text
source_snapshot_7120047c6891def192309ecba8eea37b09ea01314a2ba7d2a958bcd7fc97ac48/
training_runs/training_000000 ... training_000059/
checkpoint_artifacts/checkpoint_000000 ... checkpoint_000179/
stderr/training/training_000000.stderr ... training_000059.stderr
stderr/diagnostic/checkpoint_000000.stderr ... checkpoint_000179.stderr
```

Temporary directories such as `checkpoint_artifacts/.checkpoint_*` are not
committed results. None may remain in the final archive.

Each training directory currently contains the training configuration,
history, final parameters, observation-normalization state, status, capture
manifest, checkpoint-training configuration, and three checkpoint NPZ files.
The audit assembler, rather than this prose inventory, is authoritative about
the exact allowed schema and lineage.

## 3. Immutable Identities

These are the locks used for the submitted study. A mismatch is a hard error;
do not update a hash to accommodate a changed file.

| Object | SHA-256 |
| --- | --- |
| Composite study source | `7120047c6891def192309ecba8eea37b09ea01314a2ba7d2a958bcd7fc97ac48` |
| Manifest | `8081421fdd03d282b2febe33ffdc3b457115d8c4e98ca8eb2a702ac495d94087` |
| Protocol | `5c04b957e3fd8cee005626e2e112e8372571dae9372a1a87cfa66e32dde3ad38` |
| Analyzer | `290fd258d61531a1dc7f9cb9f4debffe5dfefba1b909b4ca2d67d68ec3428154` |
| Launcher bundle manifest | `6b485c9c6e49c28abfed1ee20d033a80e717aedfa1595b3919189de471901261` |
| Dependency bundle manifest | `30afcb40adfc5877c7a094719260bb0e8a02ffb39e29923be0c50fb55691f08b` |
| Checkpoint launcher | `98c2b3e165d24de25cc48a5640f8ad2d383d68de961ac60b34fe72ac801fbedf` |
| Diagnostic launcher | `c2bfe1a89a53b8e99723fae7c5af253638983a915955797d1fe31cc1ad8bcaca` |

The composite source inventory is the sorted explicit tuple
`STUDY_SOURCE_PATHS` in `experiments/lagged_subspace_study_lock.py`. The
existing snapshot contains exactly those 31 regular files, has no symlinks,
and has no write bits. The launcher bundle covers both Slurm launchers. The
dependency bundle covers `environment.yml` and `requirement.txt`.

The locked direct runtime versions are Python 3.10.18, pip 25.2, NumPy 1.26.4,
PyYAML 6.0.2, Gymnasium 1.2.0, MuJoCo 3.3.5, SciPy 1.15.3, Matplotlib 3.10.5,
and ale-py 0.11.2.

## 4. Existing Validation Surface

The latest full mutable-worktree run passed 268 unit tests in 232.487 seconds.
The immutable 31-file study snapshot contains 164 tests. Its five
study-specific modules contain 53 tests. Later worktree coverage includes the
overflow-safe comparison fix, exhaustive small-`m` identities, the
release-layer checkpoint validator, deterministic paper-output rendering, and
compute-disclosure collection. The snapshot's five modules are:

- `tests/test_lagged_subspace_diagnostic.py` checks previsible bases,
  chronological weighting, exact tie-safe LOPO statistics, brute-force kernel
  agreement, delete-pair jackknife recomputation, quadratic signs,
  endpoint-Jacobian finite differences, controls, locality, and explicit
  failure on degeneracy.
- `tests/test_run_lagged_subspace_checkpoint_diagnostic.py` checks checkpoint
  chronology, deterministic mappings and artifacts, common random numbers,
  norm matching, use of complete Bank A, all-tied retention, and atomic
  failure behavior.
- `tests/test_analyze_lagged_subspace_frozen_checkpoint.py` checks complete
  bijections, hash and lineage corruption, independent seed reconstruction,
  path traversal, budget and inference contracts, seed clustering, Holm
  correction, unresolved-gate failure, exact order-statistic bounds, and the
  exact sign gate.
- `tests/test_assemble_lagged_subspace_frozen_checkpoint.py` checks end-to-end
  assembly, compact release records, atomic write-once behavior, missing and
  extra inputs, nonempty stderr, mixed source locks, tampering, and the
  production composite digest.
- `tests/test_lagged_subspace_study_lock.py` checks source sensitivity,
  runtime-file inventory, bundle locks, launcher resources, bijective arrays,
  final protocol status, exact runtime versions, and fail-before-environment
  provenance enforcement.

The full worktree suite was last run successfully before study launch, but a
durable machine-readable test report is not yet part of the artifact. The
release must rerun the snapshot suite in a clean environment and retain the
command, interpreter inventory, start/end timestamps, exit status, and full
log. A test count alone is not a pass record.

## 5. Clean-Environment Verification

The commands below verify the complete integrity package without rerunning
4.87 million policy rollouts. Run them internally before submission and in
the deblinded archive from a newly extracted tree and a new Conda environment.
The anonymous review archive cannot expose the two identity-bearing original
launchers, so it uses the commitment-and-sanitization verification in Section
10 instead of claiming that reviewers recomputed the composite source hash.
`RELEASE_SHA256SUMS` and the release verification scripts named below are
required deliverables; they do not exist yet.

```bash
set -euo pipefail
sha256sum -c RELEASE_SHA256SUMS
conda env create -f environment.yml
conda activate diiwes-repro
export ARCHIVE_ROOT="$(pwd -P)"
```

Set the immutable paths and identities:

```bash
export ARTIFACT_ROOT="$ARCHIVE_ROOT/artifacts/lagged_subspace_frozen_checkpoint_7120047c6891def1"
export SOURCE_SHA=7120047c6891def192309ecba8eea37b09ea01314a2ba7d2a958bcd7fc97ac48
export MANIFEST_SHA=8081421fdd03d282b2febe33ffdc3b457115d8c4e98ca8eb2a702ac495d94087
export PROTOCOL_SHA=5c04b957e3fd8cee005626e2e112e8372571dae9372a1a87cfa66e32dde3ad38
export ANALYZER_SHA=290fd258d61531a1dc7f9cb9f4debffe5dfefba1b909b4ca2d67d68ec3428154
export LAUNCHER_SHA=6b485c9c6e49c28abfed1ee20d033a80e717aedfa1595b3919189de471901261
export DEPENDENCY_SHA=30afcb40adfc5877c7a094719260bb0e8a02ffb39e29923be0c50fb55691f08b
export SNAP="$ARTIFACT_ROOT/source_snapshot_$SOURCE_SHA"
export PAPER_EXPECTED_SOURCE_SHA="$SOURCE_SHA"
```

Verify source, bundles, mappings, and direct dependencies:

```bash
cd "$SNAP"
python -m experiments.lagged_subspace_study_lock validate-runtime
python -m experiments.lagged_subspace_study_lock verify \
  --snapshot-root . --expected "$SOURCE_SHA"
python -m experiments.lagged_subspace_study_lock verify-bundle \
  --snapshot-root . \
  --bundle experiments/manifests/lagged_subspace_launcher_lock.json \
  --kind launchers --expected "$LAUNCHER_SHA"
python -m experiments.lagged_subspace_study_lock verify-bundle \
  --snapshot-root . \
  --bundle experiments/manifests/lagged_subspace_dependency_lock.json \
  --kind dependency_locks --expected "$DEPENDENCY_SHA"
python -m experiments.lagged_subspace_study_lock validate-mappings \
  --snapshot-root .
```

Expected mapping output is exactly:

```text
training=60	diagnostic=180
```

Run the exact snapshot tests, not a mutable checkout:

```bash
cd "$SNAP"
python -m unittest discover -s tests -p '*lagged_subspace*.py' -v
python -m unittest discover -s tests -v
```

The expected discovered counts for this snapshot are 53 focused tests and 164
total tests. The release verifier must fail if a test is skipped unexpectedly,
if discovery reports a different count, or if either command exits nonzero.

Perform a minimal environment smoke test before any independent rerun:

```bash
python - <<'PY'
import gymnasium as gym
for name in ("Hopper-v5", "Walker2d-v5", "HalfCheetah-v5"):
    env = gym.make(name)
    observation, _ = env.reset(seed=0)
    assert observation.shape == env.observation_space.shape
    env.close()
print("mujoco-smoke=ok")
PY
```

## 6. Exact Full-Study Reproduction

### 6.1 Existing Original-Infrastructure Path

The locked launchers are exact records of the run, but they activate
`/hpc/home/rt239/miniconda3/bin/activate es_parallel`. Therefore, they are not
portable to a clean external cluster as written. On the original
infrastructure, the checkpoint stage uses 60 array tasks at concurrency 6,
32 CPUs and 32 GiB per task, 30 rollout workers, and a 24-hour task limit. The
diagnostic stage uses 180 array tasks at concurrency 6, 32 CPUs and 64 GiB per
task, 30 rollout workers, a pair chunk size of 32, and a 24-hour task limit.

For an exact same-infrastructure rerun into a new empty artifact root, export
all locks to both stages:

```bash
export WORKSPACE=/absolute/path/to/new/workspace
export REFERENCE_SNAP="$ARTIFACT_ROOT/source_snapshot_$SOURCE_SHA"
export NEW_ROOT="$WORKSPACE/results/independent_lagged_subspace_$SOURCE_SHA"
test ! -e "$NEW_ROOT"
mkdir -p "$NEW_ROOT"
cp -a "$REFERENCE_SNAP" "$NEW_ROOT/source_snapshot_$SOURCE_SHA"
export SNAP="$NEW_ROOT/source_snapshot_$SOURCE_SHA"
cd "$SNAP"
python -m experiments.lagged_subspace_study_lock verify \
  --snapshot-root . --expected "$SOURCE_SHA"
export EXPORTS="ALL,PAPER_EXPECTED_SOURCE_SHA=$SOURCE_SHA,PAPER_EXPECTED_MANIFEST_SHA256=$MANIFEST_SHA,PAPER_EXPECTED_PROTOCOL_SHA256=$PROTOCOL_SHA,PAPER_EXPECTED_ANALYZER_SHA256=$ANALYZER_SHA,PAPER_EXPECTED_LAUNCHER_BUNDLE_SHA256=$LAUNCHER_SHA,PAPER_EXPECTED_DEPENDENCY_LOCK_SHA256=$DEPENDENCY_SHA,PAPER_REPO_DIR=$SNAP,PAPER_ARTIFACT_ROOT=$NEW_ROOT,PAPER_WORKSPACE_DIR=$WORKSPACE"
sbatch --parsable --export="$EXPORTS" \
  "$SNAP/scripts/submit_lagged_subspace_checkpoint_generation.sh"
```

Do not submit the diagnostic array merely because the checkpoint array leaves
the queue. Require all 60 status records, all 180 checkpoints, exact lineage,
and empty stderr. Run the release-layer validator, which pins its imported
assembler, analyzer, and study-lock modules to the immutable snapshot and
delegates the run-level contract to the final assembler:

```bash
python "$WORKSPACE/scripts/validate_lagged_subspace_checkpoint_stage.py" \
  --artifact-root "$NEW_ROOT" \
  --manifest "$SNAP/experiments/manifests/lagged_subspace_frozen_checkpoint.json" \
  --source-snapshot-path "$SNAP" \
  --launcher-lock "$SNAP/experiments/manifests/lagged_subspace_launcher_lock.json" \
  --dependency-lock "$SNAP/experiments/manifests/lagged_subspace_dependency_lock.json" \
  --expected-source-sha256 "$SOURCE_SHA" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-analyzer-sha256 "$ANALYZER_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --expected-dependency-lock-sha256 "$DEPENDENCY_SHA" \
  --output checkpoint_stage_validation.json
```

The validator writes canonical JSON with a top-level `report_sha256`, refuses
overwrite, and exits nonzero on missing, extra, symlinked, mixed-lock,
nonempty-stderr, reward-selected, forbidden-control, or incomplete input. Only
after that report passes:

```bash
sbatch --parsable --export="$EXPORTS" \
  "$SNAP/scripts/submit_lagged_subspace_diagnostic.sh"
```

### 6.2 Required Portable Path

Before artifact release, add a packaging-layer runner that accepts the Conda
activation path or an already-active interpreter rather than embedding a user
home directory. This runner must:

- invoke the unchanged immutable Python entry points and locked arguments;
- preserve all six expected hashes and the array mappings;
- reject overrides of seeds, tasks, generations, population sizes, bank sizes,
  partitions, endpoint arms, and chunk semantics;
- record the portable runner's own digest separately from the experimental
  source digest;
- support Slurm and a documented sequential/local mode;
- provide `--dry-run` mapping checks for task 0, task 59, checkpoint 0, and
  checkpoint 179; and
- pass an independent clean-cluster run before being called supported.

Changing the locked launchers now would create a different experimental source
identity. Treat the portable runner as a transparent release wrapper, retain
the original source hash, and document every wrapper-to-entry-point argument
mapping.

Raw byte identity is required when verifying the supplied archive. It is not a
portable claim for an independent MuJoCo rerun on different CPUs, libc, or OS.
The independent rerun report must distinguish exact seed/config equivalence,
validator success, numerical differences, gate-decision agreement, and any
predeclared tolerance. Do not silently replace this with a claim of bitwise
cross-platform determinism.

## 7. Assembly And Preregistered Analysis

Run assembly only after both stages are complete and all stage stderr files
are empty. The assembler refuses to overwrite its output and validates exact
records, hashes, paths, identities, budgets, and absence of inference fields.

```bash
set -euo pipefail
cd "$SNAP"
test ! -e "$ARTIFACT_ROOT/audit_index.json"
python scripts/assemble_lagged_subspace_frozen_checkpoint.py \
  --artifact-root "$ARTIFACT_ROOT" \
  --manifest experiments/manifests/lagged_subspace_frozen_checkpoint.json \
  --source-snapshot-path "source_snapshot_$SOURCE_SHA" \
  --launcher-lock experiments/manifests/lagged_subspace_launcher_lock.json \
  --dependency-lock experiments/manifests/lagged_subspace_dependency_lock.json \
  --expected-source-sha256 "$SOURCE_SHA" \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-analyzer-sha256 "$ANALYZER_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --expected-dependency-lock-sha256 "$DEPENDENCY_SHA" \
  --output audit_index.json
```

The final audit index must contain exactly:

| Record family | Count |
| --- | ---: |
| Training runs | 60 |
| Checkpoints | 180 |
| Banks | 360 |
| Bank-B partitions | 3,600 |
| Checkpoint-by-q metrics | 540 |
| Center endpoint episodes | 1,800 |
| Arm endpoint episodes | 432,000 |

The recomputed policy-rollout budget must be exactly:

| Component | Rollouts |
| --- | ---: |
| Checkpoint training candidates | 3,000,000 |
| Observation-normalization calibration | 180 |
| Curvature-bank candidates | 1,440,000 |
| Endpoint arms | 432,000 |
| Checkpoint centers | 1,800 |
| Total | 4,873,980 |

Environment transitions are separate and must equal the sum of validated raw
records; do not infer them from the rollout count.

The analyzer validates the entire audit index before calculating the locked
gates. Unlike the assembler, the immutable analyzer does not refuse an
existing output internally. The explicit existence check below is therefore a
mandatory write-once guard and must be run immediately before the analyzer:

```bash
test ! -e "$ARTIFACT_ROOT/analysis.json"
python scripts/analyze_lagged_subspace_frozen_checkpoint.py \
  "$ARTIFACT_ROOT/audit_index.json" \
  --artifact-root "$ARTIFACT_ROOT" \
  --manifest experiments/manifests/lagged_subspace_frozen_checkpoint.json \
  --expected-manifest-sha256 "$MANIFEST_SHA" \
  --expected-source-sha256 "$SOURCE_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-analyzer-sha256 "$ANALYZER_SHA" \
  --expected-launcher-sha256 "$LAUNCHER_SHA" \
  --expected-dependency-lock-sha256 "$DEPENDENCY_SHA" \
  --output "$ARTIFACT_ROOT/analysis.json"
```

Re-run the analyzer to a new temporary path and require byte identity with the
released `analysis.json`. Record SHA-256 digests for the audit index, analysis,
analyzer, and manifest. The analysis is valid whether
`mechanism_advances_to_optimizer_pilot` is true or false. Report all task gate
conditions, unresolved cases, exact sign-test counts, raw p-values,
Holm-adjusted p-values, and descriptive sensitivity results without selective
omission.

## 8. Figure And Table Provenance

No paper-output generator exists yet. Add one deterministic script, for
example `scripts/render_lagged_subspace_paper_outputs.py`, whose only
scientific input is the validated `analysis.json`. It must refuse an analysis
whose embedded study identity or claim boundary differs from the locks above.

Minimum outputs are:

- `table_mechanism_gates.csv` and `.tex`: one row per task with the L, D, H,
  and E simultaneous bounds, every Boolean gate, and `task_pass`;
- `table_endpoint_sign_tests.csv` and `.tex`: seed mean contrast, strict wins,
  ties, seed count, raw one-sided sign p-value, Holm-adjusted p-value, and the
  locked alpha threshold;
- `figure_mechanism_bounds.pdf` and `.png`: task-level simultaneous bounds
  against their preregistered thresholds;
- `figure_endpoint_contrasts.pdf` and `.png`: seed-cluster endpoint summaries
  for structured-minus-isotropic at primary `q=0.5`, clearly distinguished
  from descriptive secondary controls; and
- `figure_locality_sensitivity.pdf` and `.png`: descriptive step-over-sigma
  summaries for every arm and q, with no inferential annotation.

Every output must have one entry in `paper_output_manifest.json` with:

```json
{
  "output_path": "paper_outputs/example.pdf",
  "output_sha256": "<sha256>",
  "generator_path": "scripts/render_lagged_subspace_paper_outputs.py",
  "generator_sha256": "<sha256>",
  "input_path": "analysis.json",
  "input_sha256": "<sha256>",
  "json_selectors": ["task_results[*].simultaneous_bounds"],
  "transformation_id": "<versioned exact transformation>",
  "command": "<exact argv without machine-specific absolute paths>",
  "software_versions": {"matplotlib": "3.10.5", "numpy": "1.26.4"},
  "caption_id": "<paper label>",
  "claim_boundary": "frozen_checkpoint_mechanism_only_not_optimizer_or_sample_efficiency"
}
```

Acceptance requires deterministic regeneration into a new directory and
byte-identical CSV/TeX. PDF byte identity is required only after fixing PDF
metadata and timestamps; otherwise compare a canonicalized PDF plus rendered
pixel hashes. No manuscript table may be edited by hand after generation.
Every number in the manuscript must map to a JSON selector and transformation
entry.

## 9. Compute And Storage Disclosure

Create `compute_disclosure.json` and a human-readable table from it. Unknown
values must be `null` with a reason; allocation limits must not be reported as
actual consumption. Populate at least:

- execution dates in UTC and scheduler type/version;
- anonymous cluster description, OS release, kernel, libc, CPU vendor/model,
  sockets, cores, memory, and accelerator count (`0` for this CPU study);
- Python and all locked direct package versions;
- stage name, task count, maximum concurrency, CPUs per task, worker count,
  requested memory, requested wall limit, actual application wall time,
  allocated CPU-hours, and MaxRSS distribution;
- scheduler retries, preemptions, timeouts, failed attempts, exclusions, and
  the disposition of every attempt;
- exact policy rollout counts and validated environment-transition counts by
  stage;
- final uncompressed artifact bytes, archive bytes, peak temporary bytes,
  source bytes, training bytes, diagnostic bytes, and paper-output bytes;
- energy estimate in kWh with measurement/model and assumptions, or an
  explicit explanation that it is unavailable; and
- CO2e estimate with region, time window, intensity source, and method, or an
  explicit unavailable statement.

The Slurm launchers establish only these allocations:

| Stage | Tasks | Concurrency | CPUs/task | Workers | Memory/task | Time limit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Checkpoint generation | 60 | 6 | 32 | 30 | 32 GiB | 24 h |
| Frozen diagnostic | 180 | 6 | 32 | 30 | 64 GiB | 24 h |

Application start and finish timestamps are present in run status/provenance
records. Scheduler accounting should be collected separately for CPU-hours
and MaxRSS. If scheduler accounting remains unavailable, disclose that fact
and report application-time bounds without inventing utilization. The
pre-run storage estimate of roughly 1.0--1.1 GB is not a final measurement;
use `du -sb` on the accepted tree and archive.

## 10. Anonymous Archive

### 10.1 Current Anonymity Blocker

The immutable experimental records include `/hpc/home/rt239/...`, hostnames,
Slurm identifiers, and a cluster-specific Conda activation path. Publishing
the current artifact root or source snapshot directly would violate a
double-blind anonymity check. Do not alter the only scientific copy in place,
because that would invalidate record and file hashes.

Implement a versioned, tested release transformation that starts from a
successfully validated audit index and copies only files referenced by that
index. This compact set should exclude runtime `config.json`, status, capture,
and scheduler records when the locked analyzer does not consume them. Do not
rewrite a referenced scientific artifact: preserve its original file and
array/content digests. If a required referenced artifact itself leaks an
identity, treat anonymous release as blocked until a separately tested
canonical release schema exists.

For source, retain exact non-identifying scientific files, replace only
identity-bearing launcher defaults in the anonymous copy, and emit
`source_file_commitments.json` with the original per-file hashes plus a
machine-checkable allowlist of redactions. The original composite hash remains
the pre-outcome commitment, but the anonymous verifier must say explicitly
that it cannot reconstruct identity-bearing original launcher bytes. Keep the
untouched full integrity package under access control until de-anonymization.
The transformation must prove that parameters, returns, transitions, seeds,
mappings, q summaries, actions, metrics, and analysis values are unchanged.

### 10.2 Required Archive Contents

The anonymous archive must contain only an allowlisted tree:

```text
ARTIFACT_README.md
LICENSE
environment.yml
requirement.txt
RELEASE_SHA256SUMS
release_inventory.json
compute_disclosure.json
source/                         # anonymous release wrapper and validators
locked_source/                  # sanitized scientific snapshot representation
protocol/                       # protocol, manifest, and all lock manifests
artifacts/
  lagged_subspace_frozen_checkpoint_7120047c6891def1/
    training_runs/              # only analyzer-referenced logs/checkpoints
    checkpoint_artifacts/       # only analyzer-referenced scientific files
    stderr/                     # the 240 empty stage stderr files
    audit_index.json
    analysis.json
paper_outputs/
  paper_output_manifest.json
  *.csv
  *.tex
  *.pdf
  *.png
verification/
  test_report.json
  clean_environment_report.json
  reproduction_report.json
```

Do not include `.git`, caches, editor files, temporary diagnostic directories,
core dumps, scheduler stdout containing host paths, unrelated historical
results, exploratory plots, credentials, home-directory paths, email
addresses, author names, institution names, or PDF author metadata. Use an
anonymous temporary license/citation presentation consistent with the venue;
add final `CITATION.cff` only after de-anonymization.

Create the archive reproducibly only after the anonymity and acceptance
verifiers pass. Normalize entry order, numeric owner/group, permissions, and
timestamps. Publish the archive SHA-256 and byte size outside the archive as
well as in the submission form.

## 11. Fail-Closed Release Acceptance

Implement one command, for example
`python source/verify_release_artifact.py --root . --mode anonymous`, that
performs all checks below and emits `verification/acceptance_report.json`.
Every check is mandatory unless explicitly labeled disclosure-only.

1. **Archive integrity:** every allowlisted regular file except the checksum
   manifest itself matches `RELEASE_SHA256SUMS`; the manifest matches its
   separately published digest. No unlisted path, symlink, device, FIFO,
   hard-linked regular file, path traversal, or case-colliding name exists.
2. **Anonymity:** recursive text and binary-string scans find no usernames,
   home/HPC paths, cluster hostnames, job IDs, git remotes, emails, author or
   institution metadata, credentials, or hidden revision history.
3. **Environment:** Python and the eight direct distributions match exact
   versions; OS/transitive/container identity is recorded; all three MuJoCo
   smoke tests pass.
4. **Source:** in deblinded mode, the composite source identity, manifest,
   protocol, analyzer, launcher bundle, dependency bundle, and 60/180 mappings
   match Section 3 exactly. In anonymous mode, every non-redacted file matches
   its original commitment, the only redactions are allowlisted
   machine-specific launcher defaults, the portable replacements pass mapping
   tests, and the report states that the original composite digest was not
   recomputed from redacted bytes.
5. **Tests:** 53 focused and 164 snapshot tests are discovered and pass with
   no unexpected skips; logs and timestamps are present and hashed.
6. **Training completeness:** exactly 60 training identities exist, all status
   records are complete, all 250 updates are present, all 180 fixed-generation
   checkpoints exist, no online evaluation/best-policy selection occurred,
   and all 60 training stderr files are zero bytes.
7. **Diagnostic completeness:** exactly 180 committed checkpoint identities
   exist, no temporary directory remains, all 180 diagnostic stderr files are
   zero bytes, and no post-outcome record exclusion occurred.
8. **Lineage and numerics:** the locked analyzer independently reconstructs
   bases, perturbations, utilities, maps, actions, jackknives, endpoints,
   digests, seed disjointness, norm matches, and budgets; all arrays and JSON
   values are finite where required.
9. **Record cardinality:** all seven counts in Section 7 are exact, identities
   are contiguous/bijective, and missing, duplicate, extra, resigned, or
   path-escaping records fail.
10. **Budget:** the five rollout components total exactly 4,873,980; transition
    totals equal the validated records and are not substituted with rollout
    counts.
11. **Analysis:** a clean rerun is byte-identical to released `analysis.json`;
    clustering, order bounds, thresholds, Holm family, exact sign tests,
    unresolved policy, and claim boundary match the preregistration.
12. **Paper outputs:** every figure and table regenerates from `analysis.json`,
    matches its provenance manifest, and contains no unsupported optimizer or
    sample-efficiency label.
13. **Compute disclosure:** all required fields are populated or carry an
    explicit reason; requested resources are distinguished from actual use.
14. **Independent reproduction:** environment-only verification passes on a
    clean machine. A full rerun report is included if the submission claims
    full experimental reproducibility; otherwise the artifact README states
    clearly that supplied-output verification, not an independent 4.87-million
    rollout rerun, was completed.

The verifier must exit nonzero and must not create an `accepted` marker when
any mandatory check fails. A failed scientific gate is not an artifact
validation failure; a changed, missing, or selectively omitted gate is.

## 12. Release Milestones

- [x] Freeze protocol, manifest, executable source, launchers, dependencies,
  mappings, and analysis plan before diagnostic outcomes.
- [x] Complete and validate 60 checkpoint-generating runs and 180 fixed
  checkpoints.
- [x] Add a strict write-once checkpoint-stage validator and focused tests.
- [x] Complete all 180 diagnostic tasks with no nonempty stderr or excluded
  record.
- [x] Assemble the write-once audit index and validate exact counts, hashes,
  lineage, and budgets.
- [ ] Run the locked analyzer and report the result without changing gates.
- [ ] Add deterministic figure/table generation and cell-level provenance.
- [ ] Capture hardware, runtime, memory, failures, transitions, storage,
  energy-method, and carbon-method disclosures.
- [ ] Add a portable release runner, stage validator, release verifier,
  artifact README, license, complete environment/container record, and test
  reports.
- [ ] Build a sanitized anonymous archive and pass identity-leakage review.
- [ ] Verify supplied artifacts in a clean environment and document exactly
  what was and was not independently rerun.
- [ ] After a positive mechanism result only, preregister a new multi-step
  optimizer pilot; after a successful pilot, run a separately locked
  untouched-seed confirmation with strong baselines.

Until every applicable unchecked item is supported by a generated artifact
and verifier output, describe the repository as an in-progress reproducible
mechanism study, not a conference-ready optimizer package.

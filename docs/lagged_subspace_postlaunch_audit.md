# Lagged-Subspace Post-Launch Audit

Status: blinded operational and deviation ledger for the locked
lagged-subspace frozen-checkpoint study. This document contains no scientific
metrics and must not be used to infer a curvature result.

## 1. Blinding Boundary

The permitted post-launch observations are scheduler state, task identities,
file and directory inventories, hashes, source lineage, exact schemas, and
empty versus nonempty stderr. The following remain blinded until all 180
fragments pass the locked assembler:

- returns, curvature estimates, gradients, eigenvalues, and update vectors;
- per-bank, per-partition, per-checkpoint, or per-task scientific summaries;
- endpoint-arm comparisons, statistical gates, aggregate analysis, and plots.

No partial-fragment analysis has been run. No checkpoint or record has been
selected, excluded, replaced, or reordered using an outcome. At the
observation below there was no assembled audit index, `analysis.json`, or
study-specific figure in the live artifact root. Structural validation of the
first fragment read its schema and provenance but did not print or inspect
scientific values.

## 2. Locked Identity

The launched jobs execute the read-only 31-file source snapshot under the live
artifact root, not the mutable worktree.

| Object | Locked identity |
| --- | --- |
| Composite source SHA-256 | `7120047c6891def192309ecba8eea37b09ea01314a2ba7d2a958bcd7fc97ac48` |
| Manifest | `experiments/manifests/lagged_subspace_frozen_checkpoint.json` |
| Manifest SHA-256 | `8081421fdd03d282b2febe33ffdc3b457115d8c4e98ca8eb2a702ac495d94087` |
| Artifact root | `results/lagged_subspace_frozen_checkpoint_7120047c6891def1` |
| Source snapshot | `results/lagged_subspace_frozen_checkpoint_7120047c6891def1/source_snapshot_7120047c6891def192309ecba8eea37b09ea01314a2ba7d2a958bcd7fc97ac48` |
| Checkpoint-generation Slurm array | `49719081`, tasks `0-59%6` |
| Diagnostic Slurm array | `49720838`, tasks `0-179%6` |

The mutable and snapshot manifest files independently hash to the locked
manifest digest above. The snapshot remains read-only; the post-launch fix in
Section 6 is not part of either launched job.

## 3. Checkpoint-Stage Gate

The checkpoint-generation array completed before diagnostic submission. A
read-only production validation using
`scripts/validate_lagged_subspace_checkpoint_stage.py` and the immutable
assembler contracts accepted:

- the exact directory set `training_000000` through `training_000059`;
- 60 complete training status records and the three predeclared generations
  50, 150, and 250 for every run;
- exactly 180 frozen checkpoints with matching task, seed, generation, source,
  configuration, capture-manifest, and artifact hashes;
- exactly 60 empty captured training-stderr files;
- the absence of replay, importance sampling, trust clipping, Picard
  iteration, online evaluation, and forbidden selection artifacts.

This was a launch gate, not an outcome test. A durable write-once copy of the
checkpoint-stage validation report remains a release-layer deliverable; its
absence does not authorize reconstructing or modifying the live training
records.

## 4. Diagnostic Submission And First Fragment

Diagnostic array `49720838` was submitted only after the checkpoint-stage gate
passed. Launch log `job_outputs/lagged_diagnostic_49720838_0.out` records the
predeclared mapping:

```text
task_id=0 checkpoint_id=0 training_id=0 task_index=0
env=Hopper-v5 seed=300 generation=50
```

The first committed fragment,
`checkpoint_artifacts/checkpoint_000000/checkpoint_index.json`, passed
`_training_record` and `_fragment_records` from the immutable source snapshot.
That check covered the exact artifact inventory, hashes, lineage, deterministic
identities, seed-stream contracts, record counts, and no-selection/no-exclusion
flags. Both its captured diagnostic stderr and Slurm stderr were empty. The
validator emitted only a structural pass statement and record counts; it did
not emit scientific values.

## 5. Provisional Operational Observation

**PROVISIONAL, observed 2026-07-13 01:10:04 EDT.** These counts are a live
operational snapshot, not a completion statement and not scientific evidence.

- `squeue` showed six running array tasks, `34-39`.
- It showed the remaining compressed pending set as
  `28,32-33,40-179`, with the scheduler reporting
  `JobArrayTaskLimit,JobHeldUser`. This ledger does not infer why a scheduler
  reason was assigned.
- The artifact root contained 31 committed fragment indexes. Temporary
  `.checkpoint_*` directories were present and are not completed records.
- All captured and Slurm diagnostic-stderr files present at that observation
  were empty. In particular, every then-committed fragment had both required
  stderr files present and empty.

Counts and scheduler states can change after this timestamp. Empty stderr for
a running, pending, held, or temporary record is not completion evidence. No
claim about the overflow defect's impact is made for any unfinished record.

## 6. Post-Launch Comparison-Overflow Deviation

After launch, review found a finite-extreme robustness defect in the immutable
snapshot's `core/lagged_subspace_diagnostic.py`. Two rank-comparison paths used
subtraction before taking a sign:

```python
np.sign(flat - flat[mate_indices])
np.sign(returns[:, 0] - returns[:, 1])
```

Both inputs are validated as finite, but two finite `float64` values near
opposite limits can have a mathematical difference outside the finite
`float64` range.

The impact is narrow:

- Under ordinary NumPy overflow handling, the subtraction becomes signed
  infinity and `np.sign` still returns the mathematically correct comparison
  sign for finite operands. A runtime warning may be emitted; the ordering is
  not reversed.
- Under strict `np.errstate(over="raise")` or an equivalent global setting,
  the subtraction can raise `FloatingPointError` before a fragment commits.
- The locked production path does not explicitly wrap these comparisons in a
  strict overflow context. All stderr for fragments committed at the
  observation in Section 5 was empty. These facts are execution evidence, not
  a proof that every possible finite return is safe and not evidence about
  unfinished records.

The immutable snapshot and live study were not edited. The mutable release
source fixes the defect in `core/lagged_subspace_diagnostic.py` by adding
`_comparison_sign(left, right)`, implemented as `greater - less`, and using it
in both `_lopo_utility_numerators` and
`gradient_u_statistic_row_sums`.

Named regression coverage is:

- `tests/test_lagged_subspace_diagnostic.py::LaggedSubspaceDiagnosticTests.test_lopo_comparisons_are_safe_for_extreme_finite_returns`,
  which exercises both affected diagnostic paths with finite `float64`
  extremes inside `np.errstate(over="raise", invalid="raise")` and compares
  them with rank-equivalent ordinary values;
- `tests/test_optimizers.py::ImplicitESTests.test_lopo_utilities_match_literal_pair_deletion_with_ties`,
  which checks the optimizer's comparison helper on the same finite-extreme
  ordering cases and provides parity with the already overflow-safe optimizer
  implementation.

The two named targeted tests passed together in the mutable worktree. That
test result validates the release fix; it does not retroactively change the
locked source hash.

## 7. Completion And Update Rules

This ledger is append-only in substance. Later observations must retain the
timestamped provisional record and add a new dated entry rather than rewrite
history.

1. Do not run the scientific analyzer, generate plots, or report scientific
   metrics from a partial array.
2. Completion requires exactly 180 committed fragment directories, exactly
   180 captured diagnostic-stderr files, no temporary fragment directories,
   and the exact locked source, manifest, launcher, dependency, and artifact
   hashes. Scheduler state alone is insufficient.
3. Validate every fragment and stderr with the locked assembler. Commit the
   immutable audit index once, then run the locked analyzer once against that
   index. Preserve a negative result unchanged.
4. Record the overflow audit separately from the scientific result. Empty
   stderr and successful assembly may establish that the locked execution
   completed, but they do not establish the general finite-extreme safety that
   the snapshot lacked.
5. If any task fails because of this defect, preserve its logs and mark the
   locked execution incomplete or invalid. Do not drop the task, replace its
   record, or mix a mutable-source rerun into this artifact root.
6. Any rerun using the fix must be a new, fully locked study with a new source
   digest, manifest identity, artifact root, and complete task set. It must not
   inherit successful fragments from `49720838`.
7. Updates to this operational ledger must remain free of returns, curvature
   values, endpoint comparisons, p-values, pass/fail mechanism gates, and
   plots. Scientific results belong only in the validated analysis artifact
   and the evidence ledger after full unblinding.

## 8. Infrastructure-Held Elements And In-Place Release

**Observed and acted on 2026-07-13 04:30-04:31 EDT.** After every other array
element had left the queue, the artifact root contained exactly 176 committed
fragments, no temporary fragment directories, 176 empty captured diagnostic
stderr files, and 176 empty Slurm stderr files. The missing identities were
exactly `28`, `32`, `33`, and `56`.

`squeue` reported those same four original array elements as pending with the
reason `user env retrieval failed requeued held`. None had a launcher stdout,
Slurm stderr, captured stderr, staging directory, or committed fragment. This
is consistent with failure before the locked launcher began; it is not a
scientific record and no outcome existed to inspect.

The four original elements were released in place with:

```text
scontrol release 49720838_28 49720838_32 49720838_33 49720838_56
```

Immediately afterward, `squeue` showed all four as running under diagnostic
array `49720838`. Their task IDs, checkpoint mappings, source snapshot,
manifest, seeds, and output paths were not changed. No successful fragment was
replaced, no task was excluded, and no new array or seed was introduced. The
compute disclosure must retain this as a same-job infrastructure requeue and
hold/release event. Completion still requires all four original elements to
commit and pass the same locked validation as the other 176.

## 9. Complete Array And Immutable Audit Index

**Observed on 2026-07-13 after the in-place release.** All four held elements
committed their original predeclared checkpoint artifacts. The final artifact
root contained 60 training-run directories, 180 checkpoint-artifact
directories, 60 empty captured training stderr files, 180 empty captured
diagnostic stderr files, and no temporary fragment directory. A later scheduler
check showed no remaining jobs for the user.

The locked structural assembler accepted the complete artifact and created the
write-once `audit_index.json` at 05:02:07 EDT. Its immutable identity is:

```text
size = 292,664,873 bytes
sha256 = 4fd609b08a3bc78731494572145102951bf8da5389ea10b0aa11abc6eafc1d19
```

This establishes complete structural and provenance assembly only. The
preregistered `analysis.json` does not yet exist, the scientific analyzer has
not been run from this record, and no mechanism result or paper figure is
reported here.

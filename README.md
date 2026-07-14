# DIIWES — Isolated Testing Branch

> **Branch:** `testing/current-run-20260713`
> **Base:** upstream [`main` at `6177d48`](https://github.com/tangmcs/diiwes/commit/6177d480521971b52372e016f9d4bb9c2cefbdca)
> **Scope:** experimental research snapshot; do not merge this branch wholesale into `main`.

This branch preserves the original DIIWES framework while testing rank-based
curvature mechanisms, stronger ES baselines, locked validation workflows, and
reproducible scientific artifacts. The upstream project remains available at
[`tangmcs/diiwes`](https://github.com/tangmcs/diiwes); this README documents
the testing branch specifically.

The goal is diagnostic: determine what the implemented curvature statistics
estimate, when their implicit updates become unstable, whether stabilization
changes anything beyond generic step attenuation, and whether a frozen,
low-rank mechanism survives independent high-sample evaluation. This branch
does **not** currently support an optimizer-superiority or
transition-sample-efficiency claim.

## Scientific Status — July 13, 2026

> The 30-run Hopper confirmation did not meet its preregistered joint
> superiority rule. The separate three-task lagged-subspace study is
> structurally complete, but its preregistered scientific analysis has not
> been run. There is no lagged-subspace mechanism result or paper figure yet.

| Workstream | Status | Evidence or next gate |
| --- | --- | --- |
| Signed-diagonal diagnostic, job `49678516` | Complete | The signed solve was arithmetically correct but numerically unstable; no robust improvement was established. |
| Curvature-repair screen, job `49678999` | Complete, exploratory | Concave projection removed amplification by construction; pooling and EMA supplied an exploratory candidate. |
| Untouched-seed confirmation, job `49681345` | Complete, confirmatory | The structured treatment did not satisfy the frozen joint superiority rule. |
| Fresh optimizer screen, job `49685417` | Complete, exploratory | All 99 runs validated; no structured-curvature regime was both local and materially different from explicit ES. |
| Lagged-subspace execution, jobs `49719081` and `49720838` | Structurally complete | 60 training records and 180 diagnostic fragments were accepted into the immutable audit index. |
| Lagged-subspace scientific analysis | Pending | `analysis.json` is absent. No outcomes, gates, or figures may be reported before the locked analyzer runs. |
| Portable release package | In progress | Compute disclosure, clean-environment record, anonymous archive, license/citation metadata, and final paper outputs remain incomplete. |

## Active Workstream: Frozen-Checkpoint Lagged Subspace

The active study is a preregistered **mechanism diagnostic**, not an optimizer
training comparison. It covers:

- `Hopper-v5`, `Walker2d-v5`, and `HalfCheetah-v5`;
- training seeds `300` through `319`;
- checkpoints at generations `50`, `150`, and `250`, for 180 fixed
  checkpoints;
- strictly lagged rank-three, layer-aligned subspaces and randomized subspace
  controls;
- two independent banks of 2,000 antithetic pairs per checkpoint; and
- locality, materiality, estimator-reliability, and equal-norm one-step
  direction gates fixed before outcomes.

The complete artifact passed structural and provenance assembly:

- `60/60` training records and `180/180` diagnostic fragments are present;
- all required captured stderr files are empty and no temporary fragment
  directory remains;
- `audit_index.json` is 292,664,873 bytes; and
- its SHA-256 is
  `4fd609b08a3bc78731494572145102951bf8da5389ea10b0aa11abc6eafc1d19`.

Structural completion is not a scientific result. The write-once
`analysis.json` is still missing, so returns, curvature values, mechanism
gates, endpoint tests, and paper plots remain unreported. The next scientific
step must follow the immutable-source procedure in
[the reproducibility plan](docs/reproducibility_and_artifact_plan.md#7-assembly-and-preregistered-analysis);
the analyzer must not be run casually from the mutable checkout.

The launched immutable snapshot retains a finite-extreme subtraction-overflow
edge case discovered after launch. The mutable branch fixes the comparison
operation and tests the fix, but the locked snapshot was not rewritten.
Complete assembly and empty stderr establish that this execution finished;
they do not establish general finite-extreme safety. See the
[post-launch audit](docs/lagged_subspace_postlaunch_audit.md) for the preserved
deviation record.

## What the Completed Evidence Says

### Untouched-seed confirmation

The confirmation used Hopper-v5, paired seeds `100` through `109`, 500
updates, 200 fresh candidates per update, `alpha_t = 10 / (t + 1)`, and 20
held-out seed-bank episodes per saved center. Those episodes were unused in
training and shared across arms within each paired seed. Replay, importance
weighting, Picard iteration, trust clipping, fixed-norm updates, scalar
damping, L2, and gradient, parameter, and curvature clipping were disabled.

The primary endpoint was normalized held-out AUC through 75,000 actual
training environment steps.

| Preregistered contrast | Mean difference | 95% paired-mean interval | Holm-adjusted p | Decision |
| --- | ---: | ---: | ---: | --- |
| Structured block-EMA minus Standard ES | +1.415 | [-6.965, 9.795] | 0.707 | Not significant |
| Structured block-EMA minus isotropic attenuation | +2.752 | [-2.701, 8.204] | 0.555 | Not significant |

The rule required both means to be positive and both Holm-adjusted,
two-sided exact sign-flip tests to reject at `0.05`. Neither test rejected and
both intervals include zero, so `confirmation_claim_supported=false`. This
does not prove a zero effect; it means the planned experiment did not
establish improvement or a benefit of layer-specific attenuation over its
norm-matched isotropic control.

The first confirmation update was also far outside a local perturbation
regime: `||Delta_0|| / sigma` ranged from 1,665.52 to 49,957.39, with median
38,758.32. The result therefore cannot cleanly separate estimator error from
linearization error at a distant endpoint.

Exact statistics, paired differences, and the frozen decision rule are in:

- [the confirmation record](docs/hopper_hessian_confirmation_job_49681345.md);
- [the preregistration](docs/hopper_hessian_confirmation_preregistration.md);
  and
- the local-only
  `results/hopper_hessian_confirmation_49681345/confirmation_primary_contrasts.json`.

### Failure and development diagnostics

The signed 5,123-coordinate diagnostic reached a smallest absolute
denominator of `6.29e-9` and maximum step amplification of `1,064,336x`.
This rules in a resonance failure mode despite an accurate arithmetic solve.

The 99-run fresh-only development screen found no regime in which structured
curvature was both local and material. At learning rates `3e-5` and `1e-4`,
all structured updates were within one `sigma`, but mean attenuation was only
`0.000709%` and `0.002337%`. Larger-rate cells became nonlocal while estimator
agreement remained weak. The screen was exploratory: it performed no
confirmatory test or claim selection and cannot support transition-sample
efficiency because actual transition counts varied by `39.38x`.

The stable projected update is mechanically non-amplifying:

```text
c_B = max(-kappa_B, 0)
delta_B = alpha_t * g_B / (1 + alpha_t * c_B)
```

Concave projection alone makes the denominator at least one. Pooling and EMA
change the estimator's structure and state; they do not by themselves prove
accuracy or learning benefit.

See the [diagnostic record](docs/hopper_hessian_job_49678516.md), the
[repair analysis](docs/hessian_fix_ablation.md), the
[fresh-development record](docs/hopper_fresh_optimizer_development_job_49685417.md),
and the self-contained
[mentor report](reports/hopper_hessian_no_trust/mentor_report.html).

## Estimator Terminology and Claim Boundary

For block `B`, `kappa_B` is a scalar block-average frozen-rank
covariance-score statistic, not a full Hessian. Under iid antithetic pair
clusters with the current-return mid-CDF held fixed,

```text
E[kappa_B] = c_m * H_stop,B^avg
c_m = 2 (m - 1) / (2m - 1)
H_stop,B^avg = mean_j-in-B (H_stop)_{jj}
```

The bias-corrected sample form is the corresponding leave-own-pair-out block
U-statistic. Rescaling curvature alone does not make the pooled-rank gradient
and curvature a matched finite-population implicit system.

Use **frozen-rank covariance-score curvature surrogate** in scientific text.
Historical class names, fields, and filenames retain `Hessian` or `hessian`
for artifact compatibility.

The three named fresh-only LOPO training arms are implemented and
code-audited, but they have not been empirically evaluated. The executed
lagged-subspace study uses matched LOPO estimates at frozen checkpoints; its
scientific analysis remains pending.

This branch must not be used to claim that:

- the statistic is an unbiased raw-return Hessian or the total Hessian of a
  globally adaptive rank objective;
- block-EMA curvature is better than Standard ES or scalar attenuation;
- stable denominators prove estimator accuracy;
- the exploratory screens are confirmatory evidence; or
- the method is transition-sample efficient.

Detailed derivations and claim limits are in
[the theory note](docs/theory_rank_curvature_surrogate.md),
[the LOPO note](docs/lopo_u_stat_curvature.md), and
[the claims audit](docs/novelty_and_claims_audit.md).

## Quick Start

All commands assume the repository root.

For a fresh clone, switch to this branch before creating the environment:

```bash
git fetch origin
git switch --track origin/testing/current-run-20260713
conda env create -f environment.yml
conda activate diiwes-repro
```

For an existing checkout with the local branch already present, use
`git switch testing/current-run-20260713`.

Run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

The latest branch verification passed all 284 discovered tests with the pinned
Python 3.10.18, NumPy 1.26.4, and Matplotlib 3.10.5 environment. Atari ROMs are
not bundled and are not required for the MuJoCo studies. The environment pins
direct dependencies; it is not a container or a full transitive system lock.

Inspect the trainer interface:

```bash
python experiments/train.py --help
```

A one-generation local smoke run can be launched with:

```bash
python experiments/train.py \
  --config configs/mujuco/hopper.yaml \
  --condition standard_es \
  --learning-rate 1e-4 \
  --iterations 1 \
  --seed 0 \
  --workers 2 \
  --output results/smoke_hopper_standard_es
```

New exploratory runs are not confirmatory evidence. Any new empirical claim
requires a frozen protocol, new source identity, and untouched evaluation
seeds.

## Artifact-Dependent Validation

A fresh clone contains source, tests, compact documentation, and reports, but
not the large historical run trees. Commands that validate completed jobs
require the ignored local `results/` and `job_outputs/` artifacts.

The lagged-subspace analyzer is also an intentional scientific unblinding and
write-once operation. Before running it, verify the immutable source,
manifest, protocol, analyzer, launcher, and dependency hashes; use the locked
source snapshot; confirm that `analysis.json` does not exist; and follow
[Section 7 of the reproducibility plan](docs/reproducibility_and_artifact_plan.md#7-assembly-and-preregistered-analysis)
exactly.

## Repository Organization

| Path | Role |
| --- | --- |
| `core/` | Original optimizer/policy framework, extended with implicit ES, adaptive ES baselines, LOPO estimators, and lagged-subspace diagnostics. |
| `utilities/` | Observation normalization and shared helpers. |
| `experiments/` | Training entry point, controlled benchmarks, frozen-checkpoint producer, and machine-readable manifests. |
| `configs/` | Original Atari and MuJoCo configuration hierarchy plus source-locked study configs. |
| `scripts/` | Plotting utilities, Slurm launchers, strict validators, analyzers, disclosure collection, and deterministic rendering. |
| `plots/` | Six historical tracked upstream plot assets and the generated-output policy. Newly generated products are ignored. |
| `tests/` | Unit, regression, provenance, artifact-validation, and deterministic-rendering tests. |
| `docs/` | Protocols, theory, audits, exact experiment records, and release planning. |
| `reports/` | Compact self-contained review artifacts derived from validated local results. |
| `results/`, `job_outputs/` | Ignored local training artifacts and scheduler logs. |
| `analysis/`, `figures/` | Ignored local archives and generated analysis products. |

The upstream spellings `configs/mujuco/` and `requirement.txt` are preserved
because completed artifacts and source manifests lock those exact paths.
Do not rename them.

## Evidence Map

- [Documentation index](docs/README.md)
- [Configuration index](configs/README.md)
- [Script index](scripts/README.md)
- [Lagged-subspace protocol](docs/lagged_subspace_frozen_checkpoint_protocol.md)
- [Lagged-subspace post-launch audit](docs/lagged_subspace_postlaunch_audit.md)
- [Reproducibility and artifact plan](docs/reproducibility_and_artifact_plan.md)
- [Confirmation result](docs/hopper_hessian_confirmation_job_49681345.md)
- [Fresh optimizer development result](docs/hopper_fresh_optimizer_development_job_49685417.md)
- [Rank-surrogate theory](docs/theory_rank_curvature_surrogate.md)
- [Novelty and claims audit](docs/novelty_and_claims_audit.md)
- [Report index](reports/README.md)

## Preservation Policy

- Keep this branch separate from `main`; do not merge it wholesale.
- Preserve the original framework directories and upstream naming.
- Do not rewrite immutable source snapshots, manifests, completed run records,
  or provenance hashes.
- Do not commit or delete ignored `results/`, `job_outputs/`, `analysis/`, or
  source archives as part of routine cleanup.
- Preserve negative and unresolved outcomes without relabeling them.
- Treat any rerun after a code or protocol change as a new study with a new
  source lock and artifact root.

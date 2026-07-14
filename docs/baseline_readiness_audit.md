# Strong ES Baseline Readiness Audit

## Status And Scope

This audit records the implementation, trainer integration, tests, and existing
scientific artifacts for Momentum ES, Adam ES, ClipUp, and SNES as of July 13,
2026. It also fixes the next-comparator work order. The audit covers the current
worktree and the immutable Hopper optimizer-development snapshot used by Slurm
job `49685417`.

In this document, "equation-ready" means that the implemented update agrees
with the documented update equation and has focused deterministic tests. It
does not mean that the method has reproduced a published benchmark, received a
fair tuning budget, or supported an optimizer-superiority claim.

The conference-readiness plan requires ES+Adam, ClipUp, SNES or separable NES,
separable CMA-ES where feasible, and a directly comparable Hessian-aware
black-box method. It also requires a documented comparable tuning budget and a
published or independently expected benchmark trend. See
`docs/conference_readiness_and_paper_plan.md`, lines 252-268 and 318-323.
Those requirements are not yet met.

## Executive Finding

The four baseline implementations audited here are equation-ready under their
documented assumptions. Momentum ES, Adam ES, and ClipUp have only exploratory
Hopper artifacts. SNES has strong unit and trainer coverage but no completed
scientific run artifact. Therefore the next baseline experiment must be an
SNES development calibration, not a new sep-CMA-ES experiment. After SNES has
a validated artifact, sep-CMA-ES is the safest new policy-scale comparator to
add. Exact HE-ES should first be evaluated on analytic and BBOB/COCO problems,
where its dense covariance machinery and extra center query are tractable.

## Implementation Audit

### Shared Fresh-Only Contract

`StandardES` supplies tie-aware centered ranks and the shared score-gradient
estimate

```text
g = mean_i[u_i epsilon_i] / sigma.
```

The implementation is in `core/standard_es.py`, lines 10-34 and 156-169.
Candidate construction is explicit at lines 184-203. `tell()` validates finite
parameters, noise, and fitness and rejects replay metadata at lines 223-270.
The optimizer-specific update is applied at lines 272-283.

The trainer maps the four baseline conditions to distinct classes in
`experiments/train.py`, lines 531-552 and 695-721. It requires a complete fresh
population at every generation and rejects any replay or non-unit fresh-weight
diagnostic at lines 2697-2847. Optimizer-specific CLI namespaces are checked at
lines 3320-3359. Resolved run metadata records the exact method and update rule
at lines 1787-1885.

This integration excludes replay and cross-generation importance sampling for
all four methods. It also sets `max_grad_norm=0` and `max_param_norm=None` in
the no-norm-control protocol at `experiments/train.py`, lines 515-528.

### Momentum ES

The implementation uses the heavy-ball ascent recurrence

```text
v_t = beta v_{t-1} + g_t
Delta_t = alpha_t v_t.
```

The class is in `core/standard_es.py`, lines 557-586. Deterministic first- and
second-generation closed-form checks are in `tests/test_optimizers.py`, lines
1706-1746. Shared-noise and shared-gradient checks against Standard ES are at
lines 1822-1839. Fresh-only diagnostics and replay rejection are at lines
1841-1884. Condition construction and resolved metadata are checked at lines
1886-1964.

Verdict: equation-ready for heavy-ball Momentum ES when `l2_coeff=0`. It has no
confirmatory or multi-task evidence.

### Adam ES

The implementation uses first and second exponential moments, both standard
bias corrections, and the ascent step

```text
Delta_t = alpha_t m_hat_t / (sqrt(v_hat_t) + epsilon).
```

The class is in `core/standard_es.py`, lines 660-718. Exact first- and
second-generation tests are in `tests/test_optimizers.py`, lines 1748-1802.
The shared-noise, fresh-only, replay-rejection, condition-construction, and
resolved-metadata coverage is shared with Momentum ES at lines 1822-1964.

Verdict: equation-ready for bias-corrected Adam ES when `l2_coeff=0`. The
nonzero-L2 behavior is a separately documented hazard below. It has no
confirmatory or multi-task evidence.

### ClipUp ES

The implementation applies the primary ClipUp recurrence

```text
z_t = alpha_t g_t / ||g_t||
v_raw,t = momentum * v_{t-1} + z_t
v_t = v_raw,t * min(1, v_max / ||v_raw,t||)
Delta_t = v_t.
```

The class is in `core/standard_es.py`, lines 589-657. It rejects external L2,
gradient clipping, and parameter projection at lines 604-609. Tests cover the
first two updates, clipping after momentum accumulation, reward-scale
invariance, zero-gradient behavior, incompatible controls, and a trainer run at
`tests/test_optimizers.py`, lines 2109-2305.

The development protocol retains the published starting relation
`alpha=v_max/2`; see `docs/hopper_fresh_optimizer_development_protocol.md`,
lines 76-81. Its internal velocity clipping is correctly identified as ClipUp
baseline machinery, not evidence for implicit or curvature stabilization; see
the same document, lines 29-31.

Verdict: equation-ready for the primary ClipUp update. Its clipping is part of
the comparator and must not be described as a trust-region-free property of the
proposed curvature method. It has no confirmatory or multi-task evidence.

### SNES

The implementation uses the canonical zero-sum log-rank utility, with exact
fitness ties receiving the average utility of their occupied ranks. See
`core/standard_es.py`, lines 55-93. For coordinate-wise search scale `s`, it
applies

```text
g_mu = sum_i u_i epsilon_i
g_log_s = sum_i u_i (epsilon_i^2 - 1)
mu <- mu + eta_mu * s * g_mu
s <- s * exp((eta_s / 2) * g_log_s).
```

The class and update are in `core/standard_es.py`, lines 339-506. `ask()`
records the sampled scale and generation token, and `tell()` rejects missing,
stale, or changed search-distribution state at lines 508-554. The trainer saves
the final coordinate scale to `snes_search_std.npy` at
`experiments/train.py`, lines 3121-3134.

Tests cover utility equations and ties at `tests/test_optimizers.py`, lines
84-103; two-generation trainer sampling and final-state persistence at lines
1354-1469; and the mean update, diagonal-scale update, default scale rate,
state contract, tied fitness, and prohibited controls at lines 1967-2106.

SNES intentionally differs from a single canonical reference setup in four
recorded ways: the experiment config retains population 200, antithetic
sampling is optional, `eta_mu` can use the trainer schedule, and adaptation
sampling and restarts are absent. See `docs/snes_baseline.md`, lines 38-55.
These are declared deviations, not hidden implementation changes.

Verdict: equation-ready and trainer-ready, but empirically uncalibrated in this
repository.

## Focused Verification

The following focused suite passed 28 of 28 tests on July 13, 2026:

```bash
cd tests
PYTHONPATH=.. /hpc/home/rt239/miniconda3/envs/es_parallel/bin/python -m unittest \
  test_optimizers.SNESUtilityTests \
  test_optimizers.AdaptiveOptimizerCliTests \
  test_optimizers.AdaptiveESBaselineTests \
  test_optimizers.SNESTests \
  test_optimizers.ClipUpESTests \
  test_optimizers.TrainingEnvironmentStepBudgetTests.test_snes_training_persists_dynamic_search_distribution \
  test_optimizers.TrainingEnvironmentStepBudgetTests.test_snes_second_generation_uses_first_updated_coordinate_scale
```

The immutable job-`49685417` copy of `core/standard_es.py` was also diffed
against the current file. The Momentum ES, Adam ES, and ClipUp class bodies are
unchanged. Current additions are the common `candidate_params()` mapper and the
SNES utility, class, and state contract. Consequently, the completed adaptive
artifacts remain evidence for the current Momentum, Adam, and ClipUp equations,
but they cannot provide evidence for SNES.

## Scientific Artifact Inventory

The authoritative development matrix is
`experiments/manifests/hopper_fresh_optimizer_development.json`. It uses three
development seeds, 250 generations, and 33 cells; see lines 1-7. The relevant
cells are:

| Method | Cells | Seeds | Validated runs | Manifest evidence |
| --- | ---: | ---: | ---: | --- |
| Momentum ES | 3 | 3 | 9 | lines 58-80 |
| Adam ES | 4 | 3 | 12 | lines 82-120 |
| ClipUp | 4 | 3 | 12 | lines 122-156 |
| SNES | 0 | 0 | 0 | absent from all 33 cells |

A recursive audit of result `config.json` files on July 13, 2026 independently
found the same counts: `momentum_es=9`, `adam_es=12`, `clipup_es=12`, and
`snes=0`.

All 99 tasks in job `49685417` completed and passed the immutable analyzer; see
`docs/hopper_fresh_optimizer_development_job_49685417.md`, lines 3-16. The
study used only Hopper-v5 and seeds 200-202. It fixed 50,000 candidate-policy
rollouts per run, while actual training-transition counts ranged from
`1,059,423` to `41,715,971`, a `39.38x` range. It therefore does not establish
transition-level sample efficiency.

The highest descriptive mean AUC was Momentum ES at `alpha=.003`, but its
updates averaged `24.38 sigma`, and optimizer families received unequal tuning
grids. The job report explicitly classifies this as calibration rather than
confirmatory evidence; see
`docs/hopper_fresh_optimizer_development_job_49685417.md`, lines 119-127.

## Cross-Method Hazards

### Learning-Rate Semantics

The shared `learning_rate` field represents different quantities:

| Method | Meaning of `learning_rate` |
| --- | --- |
| Standard ES | scalar multiplier on the score-gradient estimate |
| Momentum ES | multiplier on the heavy-ball buffer |
| Adam ES | multiplier on the coordinate-normalized Adam direction |
| ClipUp | norm of the new normalized-gradient velocity contribution |
| SNES | mean-distribution natural-gradient rate `eta_mu` |

One shared numerical grid is therefore not a fair tuning protocol. In
particular, canonical SNES uses constant `eta_mu=1`. The repository explicitly
requires any SNES study to include that value rather than reuse the Standard-ES
grid; see `docs/snes_baseline.md`, lines 46-55.

Every future comparison must assign the same number of prospectively specified
development trials to each optimizer while using method-appropriate parameter
ranges. Report both the tuning budget and the selected setting for every method.

### Nonzero L2 Changes The Adam And Momentum Methods

`AdamES` and `MomentumES` form their internal state from the data gradient.
After the optimizer-specific step, `StandardES.tell()` subtracts
`alpha*l2_coeff*theta`; see `core/standard_es.py`, lines 272-282. For Adam,
this is decoupled weight decay, not L2 included inside the Adam moments.

All completed Hopper development artifacts use `l2_coeff=0` and are unaffected.
However, `configs/mujuco/humanoid.yaml`, line 15, sets `l2_coeff=0.005`.
ClipUp and SNES reject nonzero L2 in their constructors, while Momentum and Adam
currently accept it. Running all four directly from that config would therefore
either fail or compare differently regularized methods.

Future baseline protocols must enforce `l2_coeff=0` for the primary comparison.
Any weight-decayed optimizer must be a separately named, separately tuned arm
whose update semantics are recorded explicitly.

### Final Optimizer State

SNES persists its final coordinate scale, but Momentum ES, Adam ES, and ClipUp
currently persist only final policy parameters and scalar history diagnostics,
not their vector optimizer state. This does not invalidate deterministic
from-scratch reruns, and the trainer does not claim resume support. A new
conference experiment should nevertheless persist final momentum, Adam moments,
ClipUp velocity, RNG state, and any distribution-adaptation state as audit-only
artifacts with explicit non-resume semantics.

## Required Work Order

### 1. Produce The Missing SNES Artifact

SNES must run before sep-CMA-ES is added to the scientific matrix because its
implementation and trainer integration already exist, its empirical evidence
count is exactly zero, and the conference plan already names it as a mandatory
strong baseline. Adding another uncalibrated optimizer first would increase the
number of unvalidated methods without closing the existing milestone.

Create a development-only SNES manifest with:

- fresh populations only;
- no replay, importance sampling, trust radius, L2, gradient clipping, or
  parameter projection;
- initial coordinate scale matched to the comparison protocol;
- constant `eta_mu` settings that include the canonical value `1`;
- the canonical default `eta_sigma=(3+log(d))/(5 sqrt(d))` plus only
  prospectively justified alternatives;
- a tuning-cell count equal to the other optimizer families;
- development seeds disjoint from all future confirmation seeds;
- actual environment-transition accounting and a common prospective stopping
  budget; and
- immutable source, config, manifest, launcher, analyzer, and dependency hashes.

The artifact is a development calibration. It cannot support superiority,
sample-efficiency, or generality claims.

### 2. Add sep-CMA-ES As The Next New Policy-Scale Comparator

After SNES validation, add separable CMA-ES. Ros and Hansen's sep-CMA-ES
restricts covariance adaptation to a diagonal matrix and therefore has linear
internal time and memory in dimension. This is feasible for the repository's
roughly 5,000-parameter policies, whereas dense CMA adaptation is not a safe
default.

Prefer a thin wrapper around a pinned maintained CMA implementation over an
unverified hand-written approximation. If `pycma` is selected, pin its exact
version in `environment.yml` and `requirement.txt` and freeze all behavior that
would otherwise change the named method. The current dependency specifications
contain no CMA package; see `requirement.txt`, lines 1-9, and
`environment.yml`, lines 1-14.

The reference condition must explicitly fix:

- diagonal covariance for every generation, not only an initial diagonal phase;
- active covariance updates on or off, with the choice matched to the named
  reference method;
- population size, parent weights, mean learning rate, and initial global
  scale;
- cumulative step-size adaptation;
- mirrored sampling policy;
- elitism, bounds, restarts, and termination behavior;
- minimization-to-return-maximization sign convention; and
- library version and every nondefault option in resolved run metadata.

Add a `sep_cma_es` trainer condition with an ask/tell generation token and an
exact hash of the sampled mean, global scale, and diagonal covariance. Candidate
evaluation must remain fresh-only. Reject stale or modified ask state, replay,
importance weighting, trust controls, external clipping, parameter projection,
and L2. Persist the final mean, global scale, diagonal covariance, evolution
paths, RNG state, and library-resolved options.

The sep-CMA test gate is:

1. Closed-form recombination weights and algorithm constants match the selected
   primary reference.
2. Deterministic one- and two-generation state transitions match a separately
   computed oracle in low dimension.
3. Candidate reconstruction exactly uses the sampled mean, global scale, and
   diagonal covariance from the active generation.
4. Stale, missing, duplicated, or changed ask/tell state is rejected.
5. Covariance and scale remain finite and strictly positive, with failures
   reported rather than silently clipped unless clipping belongs to the frozen
   reference algorithm.
6. A trainer integration test verifies exact fresh counts, no replay or trust,
   transition-budget accounting, persisted state, and resolved metadata.
7. Multi-seed Sphere, axis-scaled ellipsoid, and Rosenbrock tests reproduce a
   preregistered published or independently expected trend before any MuJoCo
   experiment.
8. The development matrix gives sep-CMA-ES the same number of tuning trials and
   the same actual-transition budget as every other strong baseline.

The main sep-CMA risks are silent drift to a library's active-CMA or temporary
diagonal default, unfair reuse of the Standard-ES learning-rate grid, hidden
restart or termination behavior, coordinate-scale collapse, and comparing
different numbers of environment transitions. All must be prevented by config
validation and artifact checks.

### 3. Use Exact HE-ES On Analytic And BBOB Problems First

The published HE-ES uses mirrored orthogonal directions, a dense covariance
factor, finite-difference curvature relative to a center evaluation, and a
specialized cumulative step-size rule. It evaluates the center once per
generation in addition to the offspring. See Glasmachers and Krause,
[The Hessian Estimation Evolution Strategy](https://arxiv.org/abs/2003.13256).

An exact dense HE-ES policy run is not the safest next addition at dimensions
around 5,000. Its state and transforms are quadratic in dimension, and its
extra center query changes the evaluation budget. Use the authors' exact method
first on the analytic and BBOB/COCO suite, count every center evaluation, and
reproduce a reference trend. Only then decide whether policy-scale compute is
justified.

A diagonal, block, or lagged-subspace rewrite is not exact HE-ES. It may be a
useful adapted comparator, but it must be named `adapted_diagonal_he_es` or an
equally explicit name, derived separately, tested against exact HE-ES where
both are feasible, and never cited as if it were the published algorithm.

HiZOO is likewise not a drop-in policy-ES baseline. Its published setting uses
minibatch zeroth-order fine-tuning and an additional function query for Hessian
information. See the official
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference.html).
Any RL adaptation must be identified as an adaptation and receive an exact
query-budget accounting.

## No-Claim Boundary

This audit supports only the following statements:

- the four audited baseline cores agree with their documented update equations
  under the stated controls;
- focused deterministic and trainer tests pass;
- Momentum ES, Adam ES, and ClipUp have completed exploratory Hopper runs; and
- SNES has no scientific run artifact in the repository.

This audit does not establish:

- that any baseline is optimally tuned;
- that any method is better than Standard ES or the curvature method;
- that the curvature method is better than any strong optimizer;
- transition-level sample efficiency;
- robustness across tasks, seeds, dimensions, or domains;
- reproduction of a published optimizer benchmark trend;
- a faithful policy-scale HE-ES or HiZOO comparison; or
- an optimizer contribution suitable for a top-conference superiority claim.

The project ledger already marks multi-step optimizer improvement as pending
and unauthorized unless the locked mechanism study advances; see
`docs/novelty_and_claims_audit.md`, lines 516-540. Even a positive frozen-
checkpoint mechanism result licenses only a new development pilot followed by
a separately locked untouched-seed confirmation. Until those gates pass, the
scientific result remains a mechanism and failure-mode study, not evidence that
the proposed curvature method is a superior optimizer.

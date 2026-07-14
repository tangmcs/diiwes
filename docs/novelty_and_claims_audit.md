# Novelty And Claims Audit

## Status And Use

This is the paper-facing source of truth for claims about the fresh-only,
trust-free curvature work in this repository. It separates mathematical truth,
novelty, and empirical support. Those are different questions:

- a proved identity is not automatically novel;
- a potentially novel identity is not automatically important;
- a stable implementation is not evidence that an estimator is accurate; and
- a mechanism result at frozen checkpoints is not an optimizer result.

The audit is current as of July 13, 2026. The lagged-subspace study is still a
locked, pending diagnostic until its complete artifact is assembled and the
independent analyzer accepts every planned record. No result from that study is
used below.

The safest current paper direction is a mechanistic and negative-results paper
about the semantics and failure modes of same-batch rank curvature surrogates.
An optimizer-superiority paper is not supported by the present evidence.

## Scope And Vocabulary

The in-scope estimator uses a fresh antithetic population and tie-aware rank
utilities. Replay, cross-generation importance sampling, trust-radius
clipping, Picard iteration, gradient clipping, parameter projection, and a
moving curvature EMA are absent from the active frozen-checkpoint study.

Use the following terms in the paper:

| Code or informal term | Paper term | Reason |
| --- | --- | --- |
| `hessian`, rank Hessian | frozen-rank covariance-score curvature surrogate | The statistic is not generally a raw-return Hessian. |
| implicit Hessian step | signed surrogate linearization | The code solves a local linearized system, not the nonlinear endpoint equation. |
| fixed curvature | concave-projected structured attenuation | Projection guarantees attenuation but does not certify curvature accuracy. |
| scalar control | isotropic norm-matched attenuation control | It isolates direction from generic step shrinkage. |
| subspace Hessian | LOPO rank-curvature operator restricted to a prespecified lagged subspace | The basis and estimand must both be named. |

Historical filenames may retain `hessian`, but equations, captions, tables,
abstracts, and conclusions must use the paper terms above.

## Claim Status Key

| Label | Meaning |
| --- | --- |
| **PROVED-IN-REPO** | An exact derivation is present. It has not been externally peer reviewed. |
| **KNOWN** | Primary prior work already contains the core idea or mathematical tool. |
| **POTENTIALLY-NOVEL** | A narrow claim was not found in the completed audit, but novelty is not established. |
| **EMPIRICALLY-SUPPORTED** | A completed, validated artifact supports only the stated scope. |
| **PENDING** | The required experiment or validation is incomplete. |
| **PROHIBITED** | Current theory, evidence, or prior art rules out this wording. |

## A. Proven Mathematical Claims

These statements may be presented as propositions after an independent proof
check. Their proof status does not imply novelty.

### M1. Fixed-fitness Gaussian score identity

**Status:** PROVED-IN-REPO, KNOWN.

For a fixed measurable fitness function `f`, fixed `sigma > 0`, and conditions
that permit differentiation through the Gaussian convolution,

```text
grad F_sigma(theta)
  = sigma^(-1) E[f(theta + sigma Z) Z],

Hessian F_sigma(theta)
  = sigma^(-2) E[f(theta + sigma Z) (Z Z^T - I)].
```

The antithetic raw-fitness estimator is unbiased for this Hessian when pair
clusters are iid and every rollout has the intended conditional marginal mean.
Common random numbers may be used inside a pair but not as uncontrolled shared
noise across pairs.

**Repo proof:** `docs/theory_rank_curvature_surrogate.md`, Propositions 1 and 2.

**Prior art:** Zhu, *Hessian Estimation via Stein's Identity in Black-Box
Problems* ([arXiv:2104.01317](https://arxiv.org/abs/2104.01317)).

**Boundary:** Replacing `f` with a data-dependent same-batch rank does not
inherit this raw-fitness identity.

### M2. Exact conditional endpoint-Jacobian identity

**Status:** PROVED-IN-REPO.

Conditional on one realized antithetic batch and after holding its centered
rank utilities fixed, the empirical self-normalized endpoint map satisfies

```text
G_D(0) = g_hat,
J_D = d G_D(delta) / d delta^T at delta = 0
    = (n sigma^2)^(-1) sum_i U_i z_i z_i^T,
diag(J_D) = kappa_hat.
```

**Assumptions:** exact antithetic noise, exact zero-sum centered utilities, and
utilities frozen as `delta` changes. This is a conditional finite-batch
identity, not an expectation over a fixed global objective.

**Repo proof and tests:** `docs/theory_rank_curvature_surrogate.md`, Proposition
3; `docs/lopo_u_stat_curvature.md`; `tests/test_optimizers.py` and
`tests/test_lagged_subspace_diagnostic.py` for the finite-difference and origin
identities.

**Boundary:** `J_D` is not the raw-return Hessian and not the total derivative
of ranks recomputed at the endpoint.

### M3. Exact antithetic finite-population curvature factor

**Status:** PROVED-IN-REPO, POTENTIALLY-NOVEL.

For `m >= 2` iid antithetic pair clusters and tie-aware centered-linear ranks,

```text
c_m = 2 (m - 1) / (2m - 1),
E[J_D] = c_m H_stop(theta),
J_D = c_m J_LOPO.
```

Here `H_stop` is the Gaussian score Hessian of the current-return mid-CDF
transform while that transform is held fixed. At population 200, `c_m =
198/199`.

**Assumptions:** pair clusters are iid; within-pair returns may be dependent;
each sign has the intended marginal law; ties use zero pairwise comparison;
the current mid-CDF transform is held fixed during differentiation.

**Repo proof and executable identities:**
`docs/theory_rank_curvature_surrogate.md`, Proposition 3a;
`docs/lopo_u_stat_curvature.md`; `core/implicit_es.py`; and
`tests/test_optimizers.py`. The worktree regression
`tests/test_lagged_subspace_diagnostic.py::LaggedSubspaceDiagnosticTests::test_exhaustive_small_m_lopo_scaling_and_kernel_identities`
enumerates all 81 tied three-level return arrays at `m=2` and all 729 at
`m=3`, checking the factor, gradient remainder, curvature relation, and LOPO
kernel reconstruction.

**Novelty boundary:** NES/IGO already establish rank utilities and Gaussian
covariance scores. Only this exact finite-`m`, antithetic, centered-linear-rank
factor is a candidate new detail. See N1.

### M4. Exact pooled-gradient within-pair remainder

**Status:** PROVED-IN-REPO, POTENTIALLY-NOVEL.

The pooled centered-rank gradient does not receive the same simple scaling as
curvature:

```text
g_hat
  = c_m g_LOPO
    + [1 / (2 m sigma (2m - 1))]
      sum_k epsilon_k K(Y_k,+, Y_k,-).
```

Therefore dividing only `J_D` by `c_m` does not produce a population-matched
finite-sample linearized system.

**Assumptions:** the same iid pair-cluster and tie convention as M3.

**Repo proof and checks:** `docs/theory_rank_curvature_surrogate.md`, equation
(B) and Proposition 3c; `docs/lopo_u_stat_curvature.md`; `core/implicit_es.py`;
and `tests/test_optimizers.py`.

### M5. Matched pair-cluster LOPO U-statistics

**Status:** PROVED-IN-REPO, POTENTIALLY-NOVEL AS A SYNTHESIS/APPLICATION.

The matched `g_LOPO` and `J_LOPO` can be written as ordinary order-two
U-statistics over iid antithetic pair clusters with symmetric pair-comparison
kernels. They are unbiased for `G_stop(theta)` and `H_stop(theta)` under the M3
assumptions. Finite second moments give the standard consistency result; the
usual nondegenerate U-statistic central-limit result applies to vectorized
kernels.

**Repo proof and implementation:** `docs/theory_rank_curvature_surrogate.md`,
Proposition 3b; `docs/lopo_u_stat_curvature.md`; `core/implicit_es.py`;
`core/lagged_subspace_diagnostic.py`; and the corresponding tests under
`tests/`.

**Known mathematical machinery:** Hoeffding, *A Class of Statistics with
Asymptotically Normal Distribution*
([DOI:10.1214/aoms/1177730196](https://doi.org/10.1214/aoms/1177730196)).
The U-statistic theory itself is not new.

### M6. Delete-pair jackknife for the LOPO kernel

**Status:** PROVED-IN-REPO, KNOWN TOOL APPLIED TO M5.

The row-sum deletion formula exactly reproduces literal deletion and reranking
for the pair-cluster LOPO U-statistic. The resulting jackknife is a
componentwise asymptotic variance estimator under iid, nondegenerate pair
clusters.

**Repo derivation and tests:** `docs/lopo_u_stat_curvature.md`, section
"Delete-pair jackknife"; `core/implicit_es.py`;
`core/lagged_subspace_diagnostic.py`; and tests under `tests/` that compare fast
deletion with literal reranking.

**Prior art:** Arvesen, *Jackknifing U-Statistics*
([DOI:10.1214/aoms/1177697287](https://doi.org/10.1214/aoms/1177697287)).

**Boundary:** The jackknife is not simultaneous across coordinates or blocks.
It does not cover a projected action, an adaptive trajectory, model selection,
or a moving EMA without additional analysis.

### M7. One-generation concentration and resonance conditions

**Status:** PROVED-IN-REPO; novelty of the tailored bounds is unassessed.

For fixed bounded utilities and iid pair clusters, the coordinate estimator has
leading uniform error scale

```text
(U / sigma^2) sqrt(log(d / delta) / m).
```

The corrected same-batch LOPO U-statistic has the same leading
`sigma^(-2) sqrt(log(d/delta)/m)` scale. Block averaging changes the leading
variance scale by block size but introduces explicit structural approximation
error. A signed denominator margin can be certified only when the statistical
error, multiplied by the learning rate, is smaller than the population margin.

**Assumptions:** See Sections 1, 4, 5, and 6 of
`docs/theory_resonance_sample_complexity.md`. In particular, fixed-transform
bounds require a transform independent of the target pairs; same-batch ranks
require the exact LOPO U-statistic representation; raw unbounded returns need
additional tail assumptions; all results are one-center, one-generation
statements.

**Repo proof:** `docs/theory_resonance_sample_complexity.md`, Theorems 1, 3,
and 4 and Corollaries 1-3.

**Boundary:** These theorems do not cover moving EMA targets, nonlinear
projection across generations, adaptive policy trajectories, or expected
return improvement.

### M8. Signed resonance and projected no-amplification

**Status:** PROVED-IN-REPO, elementary algebra, not a novelty claim.

For a diagonal signed linearization, a coordinate is amplified when
`|1 - alpha kappa_j| < 1`, reverses when the denominator is negative, and is
arbitrarily sensitive near zero. Under concave projection,

```text
c_j = max(-kappa_j, 0),
Delta_j = alpha g_j / (1 + alpha c_j),
```

so `0 < 1/(1 + alpha c_j) <= 1` for `alpha >= 0`. The projected step cannot
amplify or reverse a coordinate relative to the explicit step.

**Repo proof and exact quadratic checks:**
`docs/theory_rank_curvature_surrogate.md`, Propositions 4 and 5;
`docs/implicit_quadratic_optimization_benchmark.md`; and
`experiments/implicit_quadratic_optimization_benchmark.py`.

**Boundary:** No amplification is a deterministic safety property. It does not
show that `kappa` is accurate, that the direction is useful, or that expected
return improves.

### M9. Block statistic estimand

**Status:** PROVED-IN-REPO, elementary linearity, not a novelty claim.

A block statistic is the block average of diagonal score moments. Expanding it
to coordinates is exact only under block isotropy; otherwise it trades sampling
variance for structural approximation error.

**Repo proof:** `docs/theory_rank_curvature_surrogate.md`, Proposition 6, and
`docs/theory_resonance_sample_complexity.md`, Section 4.

## B. Potentially Novel Narrow Claims

The completed audit did not locate the three exact results below in the
primary sources reviewed. This is not enough to say "first" or "novel." The
only acceptable pre-search wording is **"potentially novel pending a human
citation-chain search."** After that search, the paper may say "to our
knowledge" only if the claim is qualified exactly as written here.

### N1. Exact finite-`m` antithetic centered-rank factor

**Candidate claim:** For tie-aware centered-linear ranks over `m` antithetic
pairs, the pooled covariance-score curvature has the exact multiplicative
factor `2(m-1)/(2m-1)` relative to the current-mid-CDF stop-gradient target,
and LOPO removes that factor samplewise.

**Why it may be distinctive:** NES and IGO contain rank utilities and Gaussian
covariance scores, but the audit did not find this exact antithetic
finite-population correction.

**Evidence already present:** M3.

**Evidence required before an ICLR claim:**

1. A human backward-and-forward citation search from NES, IGO, rank-based
   covariance adaptation, antithetic ES, and U-statistic rank estimators.
2. An independent rederivation that covers ties and arbitrary within-pair
   dependence.
3. Retain and independently run the checked-in exhaustive `m=2,3` test in the
   released test suite.
4. A theorem stated with the exact utility scaling; no silent switch to another
   rank convention.
5. Evidence that the factor or its matched construction matters scientifically;
   the factor is only `198/199` at population 200 and does not explain the
   observed instability by itself.

### N2. Exact gradient-curvature finite-sample mismatch

**Candidate claim:** The pooled gradient contains an exact within-pair
comparison remainder while pooled curvature has only the factor in N1;
therefore curvature-only rescaling cannot yield a matched finite-`m`
population system.

**Why it may be distinctive:** The audit did not find the exact remainder and
the resulting mismatch stated for antithetic centered-linear ranks.

**Evidence already present:** M4.

**Evidence required before an ICLR claim:** The N1 literature and proof checks,
plus a direct comparison against alternate rank conventions and a controlled
experiment showing when the mismatch is material. If it is numerically
negligible at practical `m`, present it as a semantic correction rather than a
performance contribution.

### N3. Matched LOPO pair-cluster kernels for rank ES curvature

**Candidate claim:** The repository combines leave-own-antithetic-pair-out
utilities with matched gradient and curvature kernels, yielding exact
order-two pair-cluster U-statistics and a dependence-aware delete-pair
jackknife for the same rank-transformed target.

**Why it may be distinctive:** LOPO, U-statistics, jackknife inference, rank
fitness shaping, and covariance scores are each known. Only their exact matched
construction for this antithetic rank-ES setting is a candidate synthesis.

**Evidence already present:** M5 and M6.

**Evidence required before an ICLR claim:**

1. The citation-chain search in N1, extended to cross-fitting, leave-cluster-out
   ranks, rank U-statistics, and clustered jackknife literature.
2. Independent code and proof audit of the kernels, scaling, ties, and literal
   deletion equivalence.
3. Calibrated synthetic coverage experiments over nondegenerate and nearly
   degenerate kernels; do not call the jackknife simultaneous.
4. A mechanism result showing reproducible anisotropic information beyond an
   equal-norm isotropic control. The locked lagged-subspace diagnostic is
   designed for this question but is currently pending.

### Novelty decision

N1-N3 could support a focused estimator-semantics contribution. They do not
support a claim of a new class of ES optimizer. If the human citation search
finds the same formulas, the remaining contribution must be framed as an
audited synthesis and empirical failure analysis, not theorem novelty.

## C. Known Prior Art And The Boundary It Imposes

All citations below are primary papers. Their inclusion does not imply that
the present method is identical; it identifies concepts that cannot be claimed
as new.

| Established area | Primary source | Boundary for this project |
| --- | --- | --- |
| Gaussian search-distribution scores, natural-gradient ES, covariance adaptation, and rank fitness shaping | Wierstra et al., *Natural Evolution Strategies* ([arXiv:1106.4487](https://arxiv.org/abs/1106.4487)) | The `u(epsilon^2-1)` covariance score, rank shaping, and Gaussian search-distribution updates are not new. |
| Adaptive quantile/rank objectives and invariance | Ollivier et al., *Information-Geometric Optimization Algorithms* ([arXiv:1106.3708](https://arxiv.org/abs/1106.3708)) | A time-varying quantile transform and monotone-fitness invariance are not new; a same-batch rank is not automatically a fixed raw objective. |
| Rank-selected covariance/Hessian relations | Shir and Yehudayoff, *On the Covariance-Hessian Relation in Evolution Strategies* ([arXiv:1806.03674](https://arxiv.org/abs/1806.03674)) | Broad claims that rank-selected Gaussian ES has a newly discovered relation to Hessian structure are unavailable. |
| Raw zeroth-order Stein Hessian estimation | Zhu, *Hessian Estimation via Stein's Identity in Black-Box Problems* ([arXiv:2104.01317](https://arxiv.org/abs/2104.01317)) | The Gaussian second-order score identity and raw-fitness Hessian estimator are known. |
| Implicit/proximal stochastic updates for stability | Toulis, Horel, and Airoldi, *The Proximal Robbins-Monro Method* ([arXiv:1510.00967](https://arxiv.org/abs/1510.00967); [journal DOI](https://doi.org/10.1111/rssb.12405)) | Solving an update with the new point on the right-hand side, and motivating it by stability, are not new. |
| Surrogate-gradient guiding subspaces | Maheswaranathan et al., *Guided Evolutionary Strategies* ([arXiv:1806.10230](https://arxiv.org/abs/1806.10230)) | Restricting or biasing ES exploration using a low-dimensional gradient-informed subspace is known. |
| Adaptive ES active subspaces | Choromanski et al., *ASEBO* ([arXiv:1903.04268](https://arxiv.org/abs/1903.04268)) | Learning a low-dimensional historical gradient subspace for ES is known. |
| Reusing past descent directions in ES | Meier et al., *Improving Gradient Estimation in Evolutionary Strategies With Past Descent Directions* ([arXiv:1910.05268](https://arxiv.org/abs/1910.05268)) | A subspace or estimator based on prior ES gradients is not new. |
| Limited-memory evolution paths/subspaces | Loshchilov, *A Computationally Efficient Limited Memory CMA-ES* ([DOI:10.1145/2576768.2598294](https://doi.org/10.1145/2576768.2598294)) | Limited-memory directional structure from previous iterations is known in ES. |
| Hessian-aware zeroth-order preconditioning | Ye et al., *ZO-HessAware* ([arXiv:1812.11377](https://arxiv.org/abs/1812.11377)) | Structured-Hessian approximation and zeroth-order preconditioning are not new. |
| Direct Hessian estimation in an ES | Glasmachers and Krause, *The Hessian Estimation Evolution Strategy* ([arXiv:2003.13256](https://arxiv.org/abs/2003.13256)) | "Hessian-estimating ES" and curvature-adapted search distributions are not new. |
| Diagonal Hessian-informed zeroth-order optimization | Zhao et al., *HiZOO* ([arXiv:2402.15173](https://arxiv.org/abs/2402.15173)) | Diagonal Hessian-informed ZO scaling and related convergence claims are not new in general. |
| Quasi-Newton ES | Glasmachers, *A Superlinearly Convergent Evolution Strategy* ([arXiv:2505.10987](https://arxiv.org/abs/2505.10987)) | Quasi-Newton steps and superlinear convergence in an ES are prior art; this repository has no comparable theorem. |
| Subspace approximate Hessians in ZO optimization | Kim et al., *Subspace-based Approximate Hessian Method for Zeroth-Order Optimization* ([arXiv:2507.06125](https://arxiv.org/abs/2507.06125)) | Estimating and applying Hessians in small subspaces is not new. |
| U-statistic representation, consistency, and asymptotics | Hoeffding, *A Class of Statistics with Asymptotically Normal Distribution* ([DOI:10.1214/aoms/1177730196](https://doi.org/10.1214/aoms/1177730196)) | Calling the LOPO kernels U-statistics and invoking classical asymptotics are not novel. |
| Jackknife inference for U-statistics | Arvesen, *Jackknifing U-Statistics* ([DOI:10.1214/aoms/1177697287](https://doi.org/10.1214/aoms/1177697287)) | The delete-cluster jackknife principle is known. Only the exact application and fast identity may be implementation contributions. |

Trust-region ES and sample reuse are also established and are outside the
active claim: see *Trust Region Evolution Strategies*
([AAAI primary paper](https://ojs.aaai.org/index.php/AAAI/article/view/4345)),
*Importance Weighted Evolution Strategies*
([arXiv:1811.04624](https://arxiv.org/abs/1811.04624)), and *Sample Reuse in
Information-Geometric Optimization*
([arXiv:1805.12388](https://arxiv.org/abs/1805.12388)). Their absence is an
experimental control, not a novelty contribution.

## D. Empirical Claims Supported Now

### E1. The historical trust-region plots do not test step-size robustness

**Status:** EMPIRICALLY-SUPPORTED diagnosis.

In the historical MuJoCo sweep, trust clipping was active on almost every
DIIWES update. When active, the scalar learning rate cancels from the norm of a
trust-clipped explicit gradient step. Those plots therefore compare a
fixed-radius normalized method, not trust-free implicit robustness.

**Evidence:** `docs/experiment_diagnosis.md` and
`docs/issues_document_review.md`, with the historical result directories named
there.

**Allowed wording:** "The historical result was confounded by trust-radius
normalization and is excluded from evidence for the curvature mechanism."

### E2. The tested signed solve is arithmetically stable while its estimator is unreliable

**Status:** EMPIRICALLY-SUPPORTED in the stated diagnostics.

The diagonal division has tiny relative residual, while independent
finite-population curvature estimates at Hopper-like dimension have near-zero
correlation, chance-level sign agreement, and poor update accuracy. This
separates arithmetic solve error from estimator-induced system geometry in
those tests.

**Evidence:** `docs/curvature_estimator_diagnostic.csv`,
`docs/hopper_implicit_job_49648326.md`, and
`docs/experiment_diagnosis.md`.

**Boundary:** This does not prove that every Hessian estimator or every linear
solver is unstable or stable. It applies to the implemented diagonal
rank/raw-score estimators and tested regimes.

### E3. Concave projection removes denominator resonance mechanically

**Status:** EMPIRICALLY-SUPPORTED exact sanity check plus M8.

Controlled quadratics reproduce the signed-denominator resonance and confirm
the projected no-amplification identity to numerical precision. Dense rotated
quadratics also show that diagonal safety alone does not recover the full
implicit direction.

**Evidence:** `docs/implicit_quadratic_optimization_benchmark.md` and
`results/implicit_quadratic_optimization_benchmark/benchmark_manifest.json`.

**Boundary:** This is a mechanical validation, not evidence of policy-return
improvement.

### E4. The selected structured method failed its untouched-seed Hopper confirmation

**Status:** EMPIRICALLY-SUPPORTED negative result.

On 30 validated runs, structured block-EMA did not pass either preregistered
Holm-adjusted primary contrast: versus Standard ES or versus isotropic
norm-matched attenuation. Both confidence intervals included zero. This is
insufficient evidence of improvement, not proof of identical performance.

**Evidence:** `docs/hopper_hessian_confirmation_preregistration.md`,
`docs/hopper_hessian_confirmation_job_49681345.md`, and
`results/hopper_hessian_confirmation_49681345/confirmation_primary_contrasts.json`.

### E5. The completed exploratory screen found no local, material structured-curvature regime

**Status:** EMPIRICALLY-SUPPORTED descriptive result, not confirmatory.

The 99/99 validated fresh-only development runs found negligible attenuation
at the fully local rates and poor split/temporal curvature agreement. At larger
rates, locality failed or structured curvature did not reliably beat the
isotropic control.

**Evidence:** `docs/hopper_fresh_optimizer_development_job_49685417.md`,
`results/hopper_fresh_optimizer_development_49685417/development_summary.json`,
and `results/hopper_fresh_optimizer_development_49685417/validated_development_runs.csv`.

**Boundary:** The screen used exploratory selection and cannot support a
superiority p-value or a general no-effect conclusion.

## E. Pending Empirical Claims

### E6. Reproducible directional information in a lagged rank-three subspace

**Status:** PENDING.

The locked frozen-checkpoint study asks whether matched LOPO curvature contains
local, material, reproducible anisotropic action information beyond isotropic
norm attenuation on Hopper-v5, Walker2d-v5, and HalfCheetah-v5. Its unit of
replication is the training seed, not checkpoint, partition, pair, or endpoint
episode.

**Protocol and locked implementation:**

- `docs/lagged_subspace_frozen_checkpoint_protocol.md`
- `experiments/manifests/lagged_subspace_frozen_checkpoint.json`
- `experiments/lagged_subspace_study_lock.py`
- `core/lagged_subspace_diagnostic.py`
- `experiments/run_lagged_subspace_checkpoint_diagnostic.py`
- `scripts/assemble_lagged_subspace_frozen_checkpoint.py`
- `scripts/analyze_lagged_subspace_frozen_checkpoint.py`
- `results/lagged_subspace_frozen_checkpoint_7120047c6891def1/`

**Evidence required to promote the claim:** exactly 180 accepted checkpoint
fragments, all 60 checkpoint training records, all planned banks, partitions,
and endpoint rollouts, empty or adjudicated stderr, a complete immutable audit
index, and a successful independent analyzer result under the locked hashes.
Only the prespecified `q = 0.5` gate may determine advancement.

**Maximum wording if the gate passes:**

> On at least two of three fixed MuJoCo tasks, a lagged rank-three LOPO
> subspace produced a local, reproducible `m = 100` anisotropic one-step action
> and outperformed its equal-norm gradient-direction endpoint control in this
> frozen-checkpoint diagnostic.

This wording is copied from the locked protocol. It still does not establish a
multi-step optimizer benefit, a raw-return Hessian, novelty over Hessian-aware
ZO methods, or general RL performance.

**Required wording if the gate fails:** State which prespecified conditions
failed. Do not switch locality level, task subset, checkpoint subset,
population, basis, transform, test, or endpoint after observing the result.

### E7. Multi-step optimizer improvement

**Status:** PENDING and not authorized unless E6 advances.

No current artifact shows that matched LOPO structured attenuation improves a
multi-step ES trajectory. Even a passing E6 result only licenses a new,
separately locked optimizer pilot.

**Evidence required for an ICLR-level claim:**

1. A development pilot that fixes one update rule and tuning budget without
   using final confirmation seeds.
2. A separately preregistered, untouched-seed confirmation across multiple
   tasks with equal candidate-rollout budgets.
3. Standard antithetic rank ES, Adam/momentum ES, ClipUp, SNES or separable NES,
   separable CMA-ES where feasible, and at least one direct Hessian-aware ZO/ES
   comparator such as HE-ES or a faithful diagonal/subspace analogue.
4. Isotropic norm-matched, random-subspace, wrong-subspace or shuffled-block,
   explicit-step, and oracle-curvature controls.
5. Seed-clustered effect sizes and uncertainty, multiplicity control, learning
   curves against both policy rollouts and actual environment transitions, and
   all seeds reported.
6. Analytic quadratic and BBOB/COCO experiments spanning dimension,
   conditioning, rotation, noise, population, and `sigma` before a general ES
   claim.
7. At least one non-MuJoCo domain before claiming broad black-box behavior.

### E8. Step-size robustness, sample efficiency, and convergence

**Status:** PENDING; no present evidence supports these claims.

An ICLR-level step-size claim requires a prespecified learning-rate/schedule
grid, matched tuning budgets, area-under-curve and failure-rate summaries, and
an interaction analysis showing that the method degrades less than strong
baselines without trust or fixed-norm clipping. Sample-efficiency claims
require equal rollout budgets and transition accounting. A convergence claim
requires a theorem for the actual adaptive rank, projection, and trajectory
setting, not the one-generation bounds in M7.

## F. Explicitly Prohibited Broad Claims

The following statements must not appear in the title, abstract, main text,
captions, conclusion, talk, or repository summary unless new evidence changes
the ledger.

| ID | Prohibited statement | Why it is false or unsupported | Permitted replacement |
| --- | --- | --- | --- |
| P1 | "Our rank statistic is an unbiased Hessian of expected return." | It targets a current-mid-CDF stop-gradient transform under M3, not raw return. | "Frozen-rank covariance-score curvature surrogate." |
| P2 | "It is the Hessian of the rank objective." | Same-batch ranks are adaptive and batch-dependent; only the conditional frozen map and stop-gradient target are derived. | State the conditional or stop-gradient estimand explicitly. |
| P3 | "We solve the implicit ES fixed point." | The active method uses one local linearization. Picard is disabled. | "Signed surrogate linearization" or "linearly implicit local update." |
| P4 | "Implicit curvature removes learning-rate sensitivity." | No trust-free multi-task schedule study establishes this, and historical plots were trust-confounded. | Report the exact tested rates and outcomes without a robustness claim. |
| P5 | "Our curvature method outperforms Standard ES." | The untouched Hopper confirmation failed; E7 is pending. | Report the negative confirmation and any later locked result separately. |
| P6 | "Structured curvature beats scalar damping." | The locked confirmation did not separate from isotropic norm matching; E6 is pending. | Say the directional mechanism remains unresolved. |
| P7 | "A stable denominator or tiny residual validates the curvature." | Those facts show arithmetic/safety only, not estimator accuracy. | Report residual, agreement, uncertainty, locality, and endpoint evidence separately. |
| P8 | "The method is the first Hessian-aware, second-order, implicit, proximal, subspace, or curvature ES/ZO method." | Section C contains direct prior art for every broad category. | Limit novelty discussion to N1-N3 after human search. |
| P9 | "Lagged gradient subspaces are novel." | Guided ES, ASEBO, past-descent ES, and LM-CMA-ES predate this work. | Treat the lagged basis as a controlled design choice. |
| P10 | "Projection proves accurate or beneficial curvature." | Projection only gives M8's no-amplification property. | "Concave projection removes signed resonance mechanically." |
| P11 | "The method is more sample efficient." | No equal-budget multi-task optimizer confirmation exists; the frozen study is a diagnostic. | Report exact rollout and transition budgets only. |
| P12 | "Replay or importance weighting improves this method." | Replay and importance sampling are disabled and historically confounded. | State that all active evidence is fresh-only. |
| P13 | "Trust clipping contributes to the proposed method." | Trust is a separate established mechanism and is disabled. | State `trust_radius = None` and discuss old results only as a confound. |
| P14 | "The method converges" or "has an ICLR-level convergence guarantee." | Current proofs are one-generation estimator bounds and algebraic safety results. | State exactly which fixed-center proposition is proved. |
| P15 | "The Hopper confirmation proves no effect." | Failure to reject is not equivalence. | "The preregistered superiority criterion was not met." |
| P16 | "The exploratory best cell is confirmatory." | It was selected on development seeds and did not replicate under the locked rule. | Label it exploratory and report the untouched confirmation. |
| P17 | "One-step frozen-checkpoint evidence proves training benefit." | A fixed local endpoint is not an adaptive multi-step trajectory. | Use the exact E6 wording, if and only if its gate passes. |
| P18 | "Results generalize to ES, RL, or black-box optimization broadly." | Current completed RL evidence is Hopper-only and negative; the three-task diagnostic is pending. | Name every tested task and restrict the conclusion to that design. |

## G. ICLR-Level Promotion Gates

No claim moves into the abstract merely because its code runs or its point
estimate is favorable. Promotion requires all applicable gates below.

### Theory gate

- The theorem states the exact estimand, rank convention, antithetic sampling
  scheme, tie rule, stochastic-rollout assumptions, and whether the result is
  conditional, stop-gradient, or unconditional.
- A second researcher independently checks every proof and scaling constant.
- Tests cover small `m`, ties, common random numbers within a pair, literal
  LOPO reranking, and finite-difference endpoint identities.
- Claims about confidence distinguish componentwise/asymptotic jackknife
  inference from simultaneous or adaptive-trajectory guarantees.
- The final notation never identifies projected curvature, a moving EMA, or a
  block average with the exact raw Hessian.

### Novelty gate

- Conduct and record a human citation-chain search from every source in
  Section C, including references and citing papers through the submission
  cutoff.
- Search exact formulas and equivalent rank normalizations, not only method
  names.
- Compare N1-N3 theorem-by-theorem with the closest primary papers in the
  related-work section.
- Remove "first" entirely. Use "to our knowledge" only for an exact narrow
  statement that survives the search.
- Explain why the surviving result changes understanding or practice; an
  algebraic correction too small to matter is not a standalone top-conference
  contribution.

### Mechanism gate

- Complete and independently validate E6 without changing the locked protocol.
- Establish locality, material action, high-sample replication, operational
  reliability, and equal-norm endpoint direction under the prespecified
  multiple-testing rule.
- Release passing and failing tasks, all seeds, all endpoint arms, and the exact
  gate decision.
- If E6 fails, make the failure mode the result; do not rescue the claim with a
  secondary analysis.

### Optimizer gate

- E6 must first authorize a pilot.
- Freeze one optimizer after development, then use untouched seeds for a
  multi-task confirmation.
- Match environment interactions and hyperparameter-search budgets against
  strong ES and Hessian-aware baselines.
- Demonstrate benefit beyond generic attenuation and historical-gradient
  subspace effects.
- Report robust aggregate metrics, per-task intervals, probability of
  improvement, failure rates, and sensitivity to `sigma`, population, and
  learning-rate schedule.

### Reproducibility gate

- Provide immutable source, manifest, protocol, analyzer, launcher, and
  dependency hashes.
- Make a clean environment reproduce validation, analysis tables, and figures
  from raw artifacts without manual edits.
- Publish exact seeds, failed runs, stderr adjudications, rollout counts,
  transition counts, hardware, wall-clock time, and compute budget.
- Preserve the negative Hopper confirmation and exploratory history rather
  than replacing them with the final selected result.

## H. Current Defensible Paper Claim

Before E6 completes, the strongest defensible research statement is:

> We audit a same-batch rank covariance-score statistic used in a signed
> linearized ES update. We derive its exact conditional endpoint semantics,
> its antithetic finite-population stop-gradient target, the matched LOPO
> pair-cluster U-statistics, and the pooled-gradient mismatch. Controlled and
> preregistered evidence shows that arithmetic stabilization is not evidence of
> reliable directional curvature, and the selected structured method did not
> satisfy its untouched-seed Hopper superiority criterion.

This is a candidate paper statement, not a novelty declaration. "Novel" may be
attached only to a surviving N1-N3 theorem after the novelty gate. A positive
optimizer claim requires E6 and E7; neither is currently established.

## I. Evidence Index

| Evidence role | Authoritative repository path |
| --- | --- |
| Estimator semantics and exact identities | `docs/theory_rank_curvature_surrogate.md` |
| Concentration, pooling, and resonance analysis | `docs/theory_resonance_sample_complexity.md` |
| LOPO implementation and inference boundary | `docs/lopo_u_stat_curvature.md` |
| Controlled quadratic mechanism check | `docs/implicit_quadratic_optimization_benchmark.md` |
| High-dimensional estimator diagnostic | `docs/curvature_estimator_diagnostic.csv` |
| Trust/replay/Picard diagnosis | `docs/experiment_diagnosis.md` |
| Historical trust-free implicit job | `docs/hopper_implicit_job_49648326.md` |
| Fresh-only development screen | `docs/hopper_fresh_optimizer_development_job_49685417.md` |
| Untouched-seed preregistration | `docs/hopper_hessian_confirmation_preregistration.md` |
| Untouched-seed result | `docs/hopper_hessian_confirmation_job_49681345.md` |
| Machine-readable confirmation contrasts | `results/hopper_hessian_confirmation_49681345/confirmation_primary_contrasts.json` |
| Conference-readiness program | `docs/conference_readiness_and_paper_plan.md` |
| Locked next mechanism protocol | `docs/lagged_subspace_frozen_checkpoint_protocol.md` |
| Locked study manifest | `experiments/manifests/lagged_subspace_frozen_checkpoint.json` |
| Locked study artifact root | `results/lagged_subspace_frozen_checkpoint_7120047c6891def1/` |

## Maintenance Rule

Update this ledger only after a theorem audit, a completed validated artifact,
or a documented primary-literature finding. Every promotion must name the new
evidence path and delete or narrow any now-inconsistent wording. A favorable
interim scheduler status, an incomplete task subset, or an unvalidated plot is
never claim evidence.

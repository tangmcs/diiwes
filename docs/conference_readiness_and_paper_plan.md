# Conference Readiness And Paper Plan

## Executive Decision

The repository is not ready for an ICLR-level algorithm-superiority
submission. The strongest exploratory cell did not pass its untouched-seed,
preregistered Hopper confirmation. The responsible paper direction is a
diagnostic and mechanistic study of when rank-based covariance-score curvature
surrogates fail, what stabilization changes mechanically, and which apparent
gains are explained by generic step attenuation.

Working paper title:

> **When Curvature Surrogates Fail in Rank-Based Evolution Strategies: Noise,
> Resonance, and Attenuation Controls**

This title is a direction, not a final novelty claim. A broad literature check
and a multi-task benchmark must precede submission.

## Locked Evidence

The completed confirmation job `49681345` used 30 validated runs: three arms
on paired untouched seeds `100` through `109`. Its primary endpoint was
normalized held-out AUC through 75,000 actual training steps.

| Preregistered contrast | Mean difference | 95% paired-mean t interval | Raw exact p | Holm p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Structured block-EMA minus Standard ES | 1.41483102745371 | [-6.964998084395898, 9.794660139303318] | 0.70703125 | 0.70703125 | Not significant |
| Structured block-EMA minus isotropic attenuation | 2.751514037559521 | [-2.7006202056938333, 8.203648280812875] | 0.27734375 | 0.5546875 | Not significant |

The joint claim flag is `false`. Both means are positive, but neither
Holm-adjusted test rejects and both intervals include zero. This is
insufficient evidence of improvement, not proof of a zero effect.

The result changes the paper story in two ways:

1. The five-seed screen was useful for selecting a hypothesis, but its positive
   pattern did not replicate under the frozen decision rule.
2. The structured method has not separated itself from a scalar, norm-matched
   attenuation control. Any mechanism claim about useful layer-specific
   curvature remains open.

There is also a locality confound. Across the 30 locked runs, the first applied
update had `||Delta_0||/sigma` between `1,665.52` and `49,957.39` (median
`38,758.32`). The signed update is a local linearization, so the existing RL
result does not isolate covariance-score estimation error from Taylor-remainder
error at a distant endpoint. Both must be measured in future mechanism tests.

The exact result, source digest, source archive hash, paired differences, and
decision rule are preserved in
[`hopper_hessian_confirmation_job_49681345.md`](hopper_hessian_confirmation_job_49681345.md),
[`hopper_hessian_confirmation_preregistration.md`](hopper_hessian_confirmation_preregistration.md),
and `results/hopper_hessian_confirmation_49681345/confirmation_primary_contrasts.json`.

## Validated Exploratory Development Evidence

Development job `49685417` completed and validated all `99/99` planned runs.
Its immutable source, manifest, launcher, analyzer, and archive SHA-256 values
are `5b467f867e071ebecfbc89e8e39417f8bfbb45230267ff88b1aa8e177f66d7bb`,
`68df7a8e3d86cb800f08712652f22f8ad07b684e214256b0088fd2f1d0237f57`,
`fa686122681315241878bdce1eeaf0678f7abcb52cfd77af5a42756fddeb574f`,
`1c00d5a3a02df1af6af3d549815a364a795788e743064e027665c006100dc4af`,
and `7864af92c56058cd4e4f4e102885acd7f1e1c8115f2f9d6e5f4b5b5790dcf62e`.
The validated outputs and their hashes are recorded in
[`hopper_fresh_optimizer_development_job_49685417.md`](hopper_fresh_optimizer_development_job_49685417.md).

This screen found no structured-curvature regime that was both local and
material. At the two fully local rates, mean norm attenuation was only
`0.000709%` and `0.002337%`. Split-half and lag-one curvature correlations
were near zero, sign agreement was near `0.5`, and relative disagreement
exceeded `1`. At `alpha=.003`, the structured-minus-isotropic best-return
differences were `[+0.436, -0.485, +509.523]`, so one seed drove the positive
mean while updates averaged `9.65 sigma`. At `alpha=.03`, structured
curvature lost to isotropic on every seed for AUC, final return, and best
return.

Momentum ES at `alpha=.003` was the descriptive mean-AUC winner, but its
updates averaged `24.38 sigma` and optimizer families had unequal tuning-grid
sizes. The job is an exploratory calibration and negative mechanism
diagnostic only: it computed no p-values, performed no claim selection, and
supports no optimizer-superiority or transition-sample-efficiency claim.

## Technical Naming

For antithetic pair `k` and parameter block `B`, the implementation computes

```text
kappa_B = (1 / (2 m sigma^2)) * sum_k s_k
          * (mean(epsilon_k,B^2) - 1),
s_k = u_k,+ + u_k,-,
```

where `u` is a centered-rank utility computed from the current population. The
utilities are held fixed when forming the local linearized update. The factor
`epsilon^2 - 1` is the Gaussian covariance score; the same statistic appears
in natural evolution strategies when updating search-distribution scales.

Under a fixed raw fitness function, a related Stein identity connects this
moment to diagonal curvature of a Gaussian-smoothed objective. Here, ranks are
functions of the entire sampled batch. For `m` iid antithetic pairs, the exact
result is `E[kappa] = c_m H_stop`, where
`c_m = 2(m - 1)/(2m - 1)` and `H_stop` is the Hessian of the current-return
mid-CDF transformed objective with that CDF frozen. Equivalently,
`kappa / c_m` is the LOPO rank order-two U-statistic. The pooled-rank gradient,
however, retains an `O(1/(2m-1))` within-pair comparison term, so rescaling
curvature alone is not a matched finite-population implicit method. This target
is neither a raw-return Hessian nor the total Hessian of a global adaptive rank
objective. Paper text should use:

- **frozen-rank covariance-score curvature surrogate** for `kappa`;
- **signed surrogate linearization** for `(I - alpha diag(kappa)) delta = alpha g`;
- **concave-projected structured attenuation** for the repaired update; and
- **isotropic norm-matched attenuation control** for the scalar control.

Historical identifiers containing `Hessian` can remain for reproducibility,
but equations, captions, claims, and talks must state the distinction.

The current moment-estimator diagnostic named
`curvature_same_generation_se_*` computes the naive dispersion of dependent
same-batch pair contributions divided by `sqrt(m)`. It is **not** a valid
standard error for pooled ranks. The active Stein-moment arms do not use it as
a confidence gate. Paper diagnostics must call it a naive contribution-scale
proxy or replace it with a U-statistic jackknife; inferential SE language is
reserved for a dependence-aware estimator. The separate OLS adjustment is
already labeled a screening heuristic rather than a confidence interval.

## Claims Ledger

### Supported By Current Evidence

- The signed elementwise division is implemented consistently: a tiny residual
  rules out arithmetic division error as the main observed failure.
- A high-dimensional signed diagonal surrogate can place denominators close to
  zero and produce extreme update amplification in the tested Hopper setup.
- Concave projection changes the denominator to `1 + alpha max(-kappa, 0)`, so
  it is at least one and cannot amplify or reverse the explicit step.
- Block pooling and EMA reduce estimator dimensionality and temporal noise by
  construction.
- The selected structured intervention did not satisfy its preregistered
  superiority rule on the untouched confirmation seeds.
- The corrected same-batch curvature has an exact LOPO U-statistic target and
  dependence-aware concentration rate under iid pair clusters. This does not
  validate the projected moving EMA or establish optimizer improvement.
- For exact zero-sum LOPO utilities, the raw preprojection block moment at the
  proposal center equals the corresponding block averages of diagonal entries
  of the Jacobian of the fixed-utility self-normalized endpoint map. This
  identity is not a full-Jacobian result, does not apply to the projected
  curvature operator or an off-proposal endpoint, and is not a raw-return or
  globally adaptive rank-objective Hessian. See
  [`lopo_u_stat_curvature.md`](lopo_u_stat_curvature.md).

### Open Hypotheses

- A matched LOPO gradient-and-curvature method may be more interpretable than
  the current pooled-rank conditional endpoint system. It is implemented and
  code-audited as three locked fresh-only attribution arms in
  [`lopo_u_stat_curvature.md`](lopo_u_stat_curvature.md), but it has not yet
  been empirically evaluated.
- The next empirical mechanism gate is the unrun high-sample, lagged rank-three
  subspace frozen-checkpoint protocol in
  [`lagged_subspace_frozen_checkpoint_protocol.md`](lagged_subspace_frozen_checkpoint_protocol.md).
  It is a mechanism diagnostic, not optimizer confirmation.
- Structured attenuation may help on problems with persistent block
  anisotropy, even though it was not established on the current Hopper test.
- Confidence-gated or shrinkage estimates may separate useful curvature signal
  from generic attenuation.
- Failure severity may be predictable from effective dimension, perturbation
  scale, pair count, denominator margins, and estimator agreement.

### Claims That Must Not Be Made

- The current unscaled statistic is an unbiased raw-return Hessian or a total
  Hessian of a globally adaptive rank objective.
- Dividing only curvature by `c_m` makes the current pooled-rank gradient and
  curvature a matched population implicit system.
- The block-EMA method is better than Standard ES or better than scalar
  attenuation.
- A stable denominator proves that estimated curvature is accurate.
- The exploratory five-seed cell is confirmatory evidence.
- Replay, importance sampling, trust clipping, or Picard iteration contributed
  to the locked result; all were disabled.
- The method is the first Hessian-aware, curvature-informed, implicit, trust
  region, or sample-reuse ES algorithm.

## Closest Prior Work

The novelty boundary must be written against primary sources, not only against
the repository's earlier framing.

| Area | Closest work | Relevance to this project |
| --- | --- | --- |
| Rank utilities and covariance scores in ES | [Natural Evolution Strategies, JMLR 2014](https://www.jmlr.org/papers/v15/wierstra14a.html); [Information-Geometric Optimization, JMLR 2017](https://www.jmlr.org/papers/v18/14-467.html) | The `u(epsilon^2-1)` statistic and rank-based invariance are established ES ideas. The block statistic is a pooled version, not a new Hessian identity. |
| Hessian estimation in ES | [Hessian Estimation Evolution Strategy](https://arxiv.org/abs/2003.13256); [Quasi-Newton Evolution Strategies](https://arxiv.org/abs/2505.10987) | Directly limits any broad claim that second-order structure in ES is new. |
| Hessian-aware zeroth-order optimization | [ZO-HessAware](https://arxiv.org/abs/1812.11377); [HiZOO, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6bf82cc56a5fa0287c438baa8be65a70-Abstract-Conference.html); [ZoVH](https://arxiv.org/abs/2605.30960) | Provides modern diagonal/structured Hessian-aware black-box baselines and theory. |
| Stein Hessian estimation | [A General Method for Estimating Hessian-Based Metrics Using Stein's Identity](https://proceedings.mlr.press/v145/zhu22c.html) | Establishes the raw-fitness Stein connection; it also sharpens why batch ranks require separate justification. |
| Diagonal, block, and EMA curvature | [AdaHessian](https://ojs.aaai.org/index.php/AAAI/article/view/17275); [Sophia](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06960915ba8674c7a898ec0b472b80ff-Abstract-Conference.html); [HesScale](https://proceedings.mlr.press/v235/elsayed24a.html) | Spatial averaging, clipping/projection, and temporal averaging of curvature are known stabilization patterns. |
| Implicit and proximal updates | [Stable Robbins-Monro Approximations Through Stochastic Proximal Updates](https://proceedings.mlr.press/v51/toulis16.html) | Relevant to the signed implicit motivation and stability claims. |
| Trust-region ES | [Trust Region Evolution Strategies](https://ojs.aaai.org/index.php/AAAI/article/view/4345); [TR-CMA-ES](https://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2017/GECCO/proceedings/proceedings_files/pap348s3-file1.pdf) | Confirms that trust control is a separate established mechanism and must not be folded into the curvature claim. |
| ES sample reuse | [Efficient Natural Evolution Strategies](https://arxiv.org/abs/1209.5853); [Importance Weighted Evolution Strategies](https://arxiv.org/abs/1811.04624); [Sample Reuse in Information-Geometric Optimization](https://arxiv.org/abs/1805.12388) | Replay and importance weighting are prior directions and are disabled in the current study. |
| Strong ES optimizers | [ClipUp](https://arxiv.org/abs/2008.02387) | Plain SGD-style ES is not a sufficient state-of-practice baseline. |
| Variance reduction | [Structured Evolution with Compact Architectures for Scalable Policy Optimization](https://proceedings.mlr.press/v80/choromanski18a.html); [ASEBO](https://proceedings.neurips.cc/paper/2019/hash/88bade49e98db8790df275fcebb37a13-Abstract.html) | Alternative ways to improve black-box gradient quality should be represented in the empirical comparison. |

The potentially distinctive contribution is narrow: an audited failure
analysis of applying a frozen-rank covariance-score statistic inside a signed
implicit mean update, together with controls that distinguish structured
direction changes from generic attenuation. That contribution still needs
multi-problem evidence and a formal account of the statistic being estimated.

## Proposed Diagnostic Contributions

A credible paper should target four contributions:

1. **Estimator semantics.** Derive exactly what is and is not estimated when
   population ranks are frozen inside a local ES update. Contrast raw-fitness,
   reference-rank, cross-fitted-rank, same-batch-rank, and matched LOPO
   versions.
2. **Failure mechanism.** Show how high-dimensional covariance-score noise and
   signed denominators interact to create resonance even when the linear solve
   residual is essentially zero.
3. **Controlled stabilization.** Separate concave projection, spatial pooling,
   EMA, confidence gating, and isotropic attenuation. Treat these as
   interventions to test mechanisms, not automatically as a new optimizer.
4. **Reproducible negative result.** Report the exploratory selection and the
   failed untouched-seed confirmation without replacing the endpoint or
   hiding unfavorable seeds.

The central scientific question is not "can one setting make a nice Hopper
plot?" It is "when does the structured statistic contain reproducible
directional information beyond the benefit of taking a smaller step?"

## Mandatory Experimental Program

### Estimator And Mechanism Tests

- Use analytic quadratics with known diagonal, block, rotated, indefinite, and
  ill-conditioned Hessians; add controlled observation noise.
- Measure bias, variance, mean squared error, sign recovery, subspace recovery,
  split-sample agreement, temporal agreement, and denominator margins.
- Sweep dimension, pairs per dimension, perturbation scale, reward noise,
  curvature spectrum, block misspecification, and rank transform.
- Sweep and report `||Delta||/sigma`; compare the linearized prediction with
  the actual endpoint map and measure the Taylor remainder before assigning a
  failure to curvature estimation.
- Include same-batch ranks, cross-fitted ranks, a fixed reference empirical CDF,
  standardized raw fitness, and raw fitness where scales permit.
- Test one-step counterfactual improvement using common perturbations and
  evaluation seeds before relying on long training curves.
- Include isotropic, randomly shuffled block, wrong-block, and oracle-curvature
  controls. Structured curvature must outperform these controls before a
  directional-curvature mechanism is claimed.

### Optimization Baselines

At minimum, compare against:

- plain antithetic rank ES with the same evaluation budget;
- ES with Adam or the repository's strongest standard adaptive optimizer;
- ClipUp;
- separable NES or SNES;
- separable CMA-ES where parameter scale permits;
- the isotropic norm-matched attenuation control;
- HE-ES, HiZOO, or another directly comparable Hessian-aware black-box method
  where implementation and compute permit; and
- an oracle or high-sample curvature condition on synthetic problems.

Every baseline must receive a documented, comparable tuning budget. A method
is not conference-ready merely because it beats a poorly tuned plain-SGD ES
that fails to learn a useful policy.

### Tasks And Scale

- Synthetic analytic suite first, because it supplies ground truth.
- Standard noiseless/noisy black-box functions or BBOB-style tasks next.
- At least Hopper, Walker2d, HalfCheetah, Ant, and one higher-dimensional or
  qualitatively different control task for RL.
- Multiple policy widths or parameter dimensions to test scaling.
- At least one domain outside MuJoCo if the paper claims general ES behavior.

The task list should be reduced only by a preregistered compute argument, not
after observing results. Claims must match the final scope.

### Statistics And Reporting

- Split exploratory tuning seeds from final confirmation seeds and never reuse
  confirmation seeds for model selection.
- Use at least 10 paired seeds per task or a prospectively justified power
  calculation; use more when variance makes 10 inconclusive.
- Count actual environment interactions, including normalization, center
  evaluation, and held-out evaluation; also report wall-clock time and memory.
- Report return distributions, interquartile mean, median, probability of
  improvement, and performance profiles with stratified bootstrap intervals,
  following [RLiable](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
- Freeze primary endpoints and comparison families; correct for multiplicity.
- Publish every planned seed and failure, paired differences, exact configs,
  source digests, and exclusion/rerun logs.
- Distinguish online evaluation used during development from held-out
  evaluation used for inference.
- Report tuning budgets and compute for every method, not only final runs.

## Artifact Milestones And Gates

### M0: Freeze The Existing Evidence

- Preserve the confirmation result JSON, validated-run CSV, source digest,
  source archive, archive SHA-256, Slurm logs, and preregistration.
- Generate the paper table directly from the locked JSON.
- Gate: a clean checkout reproduces `confirmation_claim_supported=false`.

### M1: Establish Estimator Semantics

- Write a derivation for raw, standardized, reference-rank, cross-fitted-rank,
  and same-batch-rank statistics.
- Add unit tests against analytic Gaussian-smoothed quadratics.
- Gate: notation and code agree on the estimand; no literal-Hessian claim is
  attached to same-batch ranks beyond the proved current-CDF stop-gradient
  target and finite-population factor.

### M2: Build Strong Baselines

- Add ES+Adam, ClipUp, SNES/sep-CMA, and the selected Hessian-aware comparator.
- Give each a fixed tuning budget on development seeds.
- Gate: baseline implementations pass analytic tests and reproduce a published
  or independently expected benchmark trend.

### M3: Complete The Mechanism Suite

- Run the synthetic and black-box experiments with ground-truth diagnostics.
- Gate: predeclare which observable distinguishes structured signal from
  scalar attenuation, and show it is measurable.

### M4: Run Multi-Task Exploration

- Tune hypotheses on development tasks/seeds only.
- Gate: choose one method and one primary claim before touching final seeds.
  If no method consistently beats strong controls, retain the diagnostic paper
  and do not manufacture an optimizer-superiority claim.

### M5: Preregister And Confirm

- Freeze tasks, seeds, budgets, metrics, exclusions, source digest, and
  multiplicity correction.
- Gate: complete every planned cell and validate before aggregation.

### M6: Reproducibility Package

- Provide an environment lock, one-command tests, launch manifests, validators,
  raw/processed result schemas, deterministic plotting, and an anonymous source
  archive.
- Add compute, limitations, broader-impact, and negative-result reporting.
- Gate: a second person reproduces all main tables and figures from artifacts.

### M7: Submission Audit

- Recheck the target conference's current author, reproducibility, ethics, and
  LLM policies.
- Verify every citation against the primary source and every number against a
  generated artifact.
- Gate: the abstract and conclusion contain only claims in the supported
  claims ledger.

## Suggested Paper Structure

1. Problem: implicit curvature correction is attractive when ES step-size
   sensitivity is severe, but the plug-in statistic can be noisy and resonant.
2. Estimator semantics: derive the covariance-score statistic and explain the
   effect of rank coupling.
3. Failure analysis: arithmetic residual, estimator reliability, denominator
   geometry, and amplification.
4. Controlled interventions: concave projection, pooling, EMA, confidence
   gates, and isotropic/shuffled controls.
5. Ground-truth synthetic experiments.
6. Multi-task black-box and RL experiments with strong baselines.
7. Locked negative confirmation and limitations.
8. Reproducibility, compute, and LLM-use disclosure.

## LLM-Use Disclosure

The current official [ICLR 2026 Author Guide](https://iclr.cc/Conferences/2026/AuthorGuide)
and [ICLR LLM FAQ](https://iclr.cc/FAQ/LLM) require disclosure of LLM use and
leave human authors fully responsible for content. Because an LLM coding agent
has materially assisted this repository's code review, experiment harness,
analysis, literature discovery, and documentation, omission would be
inaccurate. The target year's policy must be checked again at submission.

Maintain a contemporaneous log containing tool/product, model when available,
dates, tasks, affected files, human verification, and whether generated text
was retained. A paper section can follow this factual template after replacing
the bracketed fields with verified details:

> **Use of LLMs.** The authors used [product/model/version] as an assistive
> research tool for [code review and implementation, experiment validation,
> literature discovery, data analysis, and drafting]. Human authors specified
> the hypotheses, approved experimental protocols, inspected code changes,
> reran validators, checked citations and numerical outputs against primary
> artifacts, and take full responsibility for the work. The LLM was not an
> author.

Do not state a model name, version, or scope from memory. Fill the disclosure
from actual logs, disclose it in both the manuscript and submission form when
required, and retain enough detail for an audit.

## Immediate Work Order

1. Freeze and back up job `49681345` artifacts; do not rerun or filter its
   seeds based on the observed result.
2. Complete the estimator-semantics derivation and analytic tests before new
   MuJoCo tuning.
3. Implement and validate the strong baseline set.
4. Run the ground-truth synthetic mechanism suite.
5. Decide whether the evidence supports a diagnostic paper only or justifies a
   new, separately preregistered multi-task superiority test.

The project can still produce a valuable paper, but credibility now depends
on treating the failed confirmation as a result and on narrowing every claim
to what the controls actually identify.

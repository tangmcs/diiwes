# Hopper policy-gradient initialization and no-EMA comparison

This protocol trains one NumPy PPO actor for `Hopper-v5`, saves the actor and
observation-normalization state, and then starts two matched ES runs from that
exact checkpoint:

- `standard_es`;
- `diag_curvature` using raw-return diagonal Stein curvature with
  `curvature_beta=0` and EMA bias correction disabled.

The comparison uses seed 0 only, 300 ES updates, and 1,000 antithetic pairs
(2,000 candidate policies) per update. Replay, scalar implicit damping, trust
clipping, and center-fitness evaluation are disabled so the difference is the
fresh diagonal-curvature step.

The actor layout is exactly the repository's 11-64-64-3 tanh MLP. The PPO
implementation is in NumPy because the DCC `es_parallel` environment contains
Gymnasium/MuJoCo but not PyTorch or Stable-Baselines3.

A separate stress comparison uses the same checkpoint and protocol with a
constant learning rate of `0.5` (3.125 times the baseline `0.16`). It is
selected by setting `PAPER_PROTOCOL_VARIANT=high_lr0p5` in the shared launcher.

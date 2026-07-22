# Nonlinear CartPole warm-start benchmark

This directory tests the initialization idea used in Wang, Zhang, and Ying,
[*Efficient Federated RLHF via Zeroth-Order Policy
Optimization*](https://arxiv.org/abs/2604.17747) (2026). Their MuJoCo actors
are pretrained with PPO before zeroth-order fine-tuning. Here the same
experimental idea is adapted to DIIWES without the paper's human-feedback or
federated components:

1. initialize a two-hidden-layer actor randomly;
2. optionally pretrain it with batched REINFORCE, Adam, and gradient clipping;
3. fine-tune the identical initial actor with either `core.StandardES` or
   `core.DIIWES`;
4. compare random and policy-gradient initializations under matched seeds.

CartPole is implemented locally so this mechanism test needs only NumPy. The
nonlinear `sin`/`cos` dynamics and tanh neural actor make this a
non-convex policy-optimization problem rather than another quadratic test.

Run the locked default protocol from the repository root:

```bash
python -m experiments.nonlinear_cartpole.benchmark
```

The locked protocol uses 300 zeroth-order updates and 250 antithetic
perturbation pairs (500 candidate policies) per update. Override the pair
count explicitly with `--antithetic-pairs N` when running a sensitivity check.

For a quick smoke run:

```bash
python -m experiments.nonlinear_cartpole.benchmark \
  --seeds 0 \
  --reinforce-updates 2 \
  --es-updates 2 \
  --antithetic-pairs 4 \
  --eval-episodes 2 \
  --output-dir /tmp/nonlinear_cartpole_smoke
```

Outputs are grouped in one report directory: a manifest, raw policy-gradient
and ES trajectories, per-run and aggregate summaries, a Markdown report, and
an SVG learning-curve figure.

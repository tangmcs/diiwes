# Configuration index

Configuration files are YAML inputs to `experiments/train.py`. The historical
directory name `mujuco/` is misspelled, but it is retained because completed
artifacts and source manifests lock those exact paths.

## General environment configurations

- `atari/`: Boxing, Freeway, Pong, and Space Invaders RAM-policy diagnostics.
- `mujuco/ant.yaml`, `halfcheetah.yaml`, `hopper.yaml`, `humanoid.yaml`, and
  `walker2d.yaml`: general continuous-control diagnostics.

## Locked or study-specific configurations

- `mujuco/hopper_implicit_no_replay.yaml`: mentor-requested trust-free Standard
  ES versus signed-curvature comparison.
- `mujuco/hopper_hessian_fix_no_replay.yaml`: exploratory curvature repair
  screen.
- `mujuco/hopper_hessian_confirmation_no_replay.yaml`: untouched-seed
  confirmation.
- `mujuco/hopper_fresh_optimizer_development.yaml`: fresh-only optimizer
  development screen.
- `mujuco/{hopper,walker2d,halfcheetah}_lagged_subspace_checkpoints.yaml`:
  frozen-checkpoint mechanism study.

Treat study-specific files as protocol inputs. Changing one requires a new
source digest and, for preregistered studies, a versioned manifest rather than
an in-place reinterpretation of completed runs.

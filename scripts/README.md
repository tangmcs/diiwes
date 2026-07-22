# Repository scripts

Standalone tooling is grouped by purpose:

- `analysis/`: validate completed runs and produce machine-readable summaries;
- `plotting/`: render figures and presentation packages from validated data;
- `slurm/`: submit DCC jobs or collect their outputs;
- `maintenance/`: perform safe, repository-level setup tasks.

Run commands from the repository root so configuration, result, and output
paths resolve consistently. Raw runs belong under `results/`; scheduler output
belongs under `job_outputs/`; generated report packages belong under `reports/`,
and standalone or presentation-ready figure packages belong under `figures/`.

Plotting that is an inseparable part of an experiment remains with that
experiment. Retired scripts for the historical, trust-confounded MuJoCo sweep
are preserved under `archive/analysis/legacy_tools/`; they are not an
active reproduction workflow.

On a fresh DCC clone, create the unversioned `/work` directories and repository
links with:

```bash
bash scripts/maintenance/setup_dcc_storage.sh
```

The setup helper is idempotent and refuses to replace an existing real path or
a symlink that points somewhere else.

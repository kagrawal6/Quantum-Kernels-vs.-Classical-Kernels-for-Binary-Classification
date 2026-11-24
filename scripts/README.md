# scripts/

Thin command-line entry points. They read a config from `configs/`, call code in `src/`, and write CSVs to `results/`. No experiment logic should live only in a script.

## Scripts

| Script | What it does | Output |
|--------|--------------|--------|
| `check_data.py` | Data sanity checks + PCA explained-variance table | `results/pca_explained_variance.csv` |
| `compare_reduction.py` | PCA vs SelectKBest on a validation split (why we use PCA) | printed table |
| `check_kernels.py [classical\|quantum]` | Shape / symmetry / PSD checks; classical kernels must match `SVC(kernel=...)`, quantum must match direct statevector overlaps | printed |
| `run_experiment.py --sweep <name>` | One sweep: `baseline`, `features`, `train_size`, `params`, `shots` | `results/<name>.csv` |
| `summarize.py <csv>` | Mean ± std over seeds per setting | `results/summary_<name>.csv` |
| `run_hardware.py [--local aer \| --smoke-test \| --submit]` | Tiny QSVM run on AWS Braket with spending caps (see `src/quantum/README.md`) | `results/hardware/` |
| `make_figures` | Plots listed in `figures/README.md` | TODO |

`run_experiment.py` options: `--models linear rbf`, `--datasets breast_cancer`, `--seeds 0` (quick test), `--dry-run` (list runs), `--config <yaml>`, `--out <csv>`.
It appends to the CSV and skips runs already there, so an interrupted sweep can be restarted. Models whose kernel is still a TODO are skipped with a message.
For `train_size`, QSVM stops growing once one run exceeds `sweeps.time_budget_minutes` in the config, and classical models also run once on the full training pool (`n_train` = all non-test data).

## TODO
- [x] Baseline + all sweeps (`run_experiment.py`)
- [ ] `make_figures`

# scripts/

Thin command-line entry points. They read a config from `configs/`, call code in `src/`, and write CSVs to `results/`. No experiment logic should live only in a script.

## Scripts to write

| Script | What it runs | Output |
|--------|--------------|--------|
| `run_baseline` | All three models at the defaults, both datasets, 5 seeds | `results/baseline.csv` |
| `run_sweep_features` | Feature-count sweep, all models | `results/sweep_features.csv` |
| `run_sweep_train_size` | Training-size sweep, all models | `results/sweep_train_size.csv` |
| `run_sweep_params` | `gamma` (RBF) and feature map / reps / entanglement / bandwidth (QSVM) | `results/sweep_params.csv` |
| `run_sweep_shots` | Shots sweep, QSVM only (secondary) | `results/sweep_shots.csv` |
| `make_figures` | Plots listed in `figures/README.md` from the CSVs | `figures/*.png` |

Each script should accept a config path, e.g. `python scripts/run_baseline.py --config configs/default.yaml`.

## TODO
- [ ] `run_baseline` (do this first — it tests the whole pipeline end to end)
- [ ] The three priority sweeps
- [ ] Shots sweep
- [ ] `make_figures`

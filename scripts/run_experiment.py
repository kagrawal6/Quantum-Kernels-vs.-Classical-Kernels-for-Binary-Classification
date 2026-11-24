"""Run one sweep for all models and seeds; one CSV row per run.

Usage (from the repo root):
    python scripts/run_experiment.py --sweep baseline
    python scripts/run_experiment.py --sweep features
    python scripts/run_experiment.py --sweep train_size
    python scripts/run_experiment.py --sweep params
    python scripts/run_experiment.py --sweep shots

Useful options:
    --models linear rbf          only some models
    --datasets breast_cancer     only some datasets
    --seeds 0                    quick test with one seed
    --dry-run                    list the runs without running them
    --config configs/default.yaml

Output: results/<sweep>.csv (appends). Runs already in the CSV are skipped,
so an interrupted sweep can simply be restarted.
Models whose kernel is still a TODO are skipped with a message.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_data  # noqa: E402
from src.evaluation.runner import (  # noqa: E402
    append_row, existing_keys, row_key, run_one, write_machine_info)
from src.evaluation.sweeps import ALL_MODELS, SWEEPS, load_config, make_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, choices=SWEEPS)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS)
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--out", help="CSV path (default results/<sweep>.csv)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["data"]["seeds"]
    n_test = cfg["data"]["n_test"]
    C_grid, cv_folds = cfg["svm"]["C_grid"], cfg["svm"]["cv_folds"]
    out = Path(args.out) if args.out else ROOT / "results" / f"{args.sweep}.csv"
    budget_min = cfg["sweeps"].get("time_budget_minutes")

    jobs = make_jobs(cfg, args.sweep, args.datasets, args.models)
    print(f"{args.sweep}: {len(jobs)} settings x {len(seeds)} seeds -> {_rel(out)}")
    if args.dry_run:
        for j in jobs:
            print(f"  {j['dataset']:13s} {j['model']:6s} f={j['n_features']} "
                  f"n={j['n_train']} {j['params']}")
        return

    write_machine_info(ROOT / "results" / "machine.txt")
    done = existing_keys(out)
    todo_models = set()   # models whose kernel is not implemented yet
    over_budget = set()   # datasets where one QSVM run exceeded the time budget
    n_new = 0

    for job in jobs:
        ds, model = job["dataset"], job["model"]
        if model in todo_models:
            continue
        if model == "qsvm" and args.sweep == "train_size" and ds in over_budget:
            continue
        n_train = None if job["n_train"] == "full" else job["n_train"]
        for seed in seeds:
            data = get_data(ds, job["n_features"], n_train, seed, n_test=n_test)
            probe = {**job["params"], "sweep": args.sweep, "dataset": ds,
                     "model": model, "seed": seed, "n_features": job["n_features"],
                     "n_train": len(data[2]), "n_test": len(data[3])}
            if row_key(probe) in done:
                continue
            try:
                t0 = time.perf_counter()
                row = run_one(ds, model, seed, job["n_features"], n_train, n_test,
                              job["params"], C_grid, cv_folds, sweep=args.sweep,
                              data=data)
            except NotImplementedError as e:
                print(f"  SKIP {model}: {e}")
                todo_models.add(model)
                break
            append_row(out, row)
            done.add(row_key(probe))  # the default setting appears in several sub-sweeps; run it once
            n_new += 1
            print(f"  {ds:13s} {model:6s} seed={seed} f={job['n_features']} "
                  f"n={row['n_train']:<5} {_short(job['params'])}  "
                  f"acc={row['accuracy']:.3f} f1={row['f1']:.3f}  "
                  f"kernel={row['kernel_train_time'] + row['kernel_test_time']:.3f}s "
                  f"({time.perf_counter() - t0:.1f}s incl. tuning)")
            if (model == "qsvm" and args.sweep == "train_size" and budget_min
                    and row["total_time"] > budget_min * 60):
                print(f"  QSVM exceeded the {budget_min}-min budget at n_train={row['n_train']} "
                      f"on {ds}; stopping larger QSVM runs for this dataset.")
                over_budget.add(ds)
    print(f"Done: {n_new} new rows in {_rel(out)}")


def _rel(path):
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def _short(params):
    return " ".join(f"{k}={v}" for k, v in params.items() if v is not None)


if __name__ == "__main__":
    main()

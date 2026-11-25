"""Mean +/- std over seeds for every setting in a results CSV.

Usage:  python scripts/summarize.py results/baseline.csv
Writes results/summary_<name>.csv and prints the main columns.
"""

import sys
from pathlib import Path

import pandas as pd

METRICS = ["accuracy", "f1", "precision", "recall",
           "train_accuracy", "train_f1", "val_accuracy", "val_f1",
           "train_hinge_loss", "val_hinge_loss", "test_hinge_loss",
           "svm_objective", "n_support_vectors",
           "kernel_train_time", "kernel_test_time", "fit_time", "predict_time",
           "total_time", "kernel_offdiag_std", "n_circuits",
           "peak_memory_mb", "kernel_matrix_mb", "statevector_mb"]
SETTING = ["sweep", "dataset", "model", "n_features", "n_train", "gamma",
           "feature_map", "reps", "entanglement", "bandwidth", "shots"]


def summarize(path):
    df = pd.read_csv(path)
    keys = [c for c in SETTING if c in df]
    metrics = [m for m in METRICS if m in df and df[m].notna().any()]
    g = df.groupby(keys, dropna=False)[metrics].agg(["mean", "std"])
    g.columns = [f"{m}_{s}" for m, s in g.columns]
    g["n_seeds"] = df.groupby(keys, dropna=False).size()
    return g.reset_index()


def main():
    path = Path(sys.argv[1])
    s = summarize(path)
    out = path.with_name(f"summary_{path.stem}.csv")
    s.to_csv(out, index=False)
    show = [c for c in ["dataset", "model", "n_features", "n_train", "gamma", "feature_map",
                        "reps", "entanglement", "bandwidth", "shots"]
            if c in s and s[c].nunique(dropna=False) > 1 or c in ("dataset", "model")]
    view = s[show].copy()
    for m in ["accuracy", "f1"]:
        view[m] = s[f"{m}_mean"].map("{:.3f}".format) + " ± " + s[f"{m}_std"].fillna(0).map("{:.3f}".format)
    for m in ["train_f1", "val_f1", "test_hinge_loss", "peak_memory_mb"]:
        if f"{m}_mean" in s:
            view[m] = s[f"{m}_mean"].map("{:.3f}".format)
    view["total_time_s"] = s["total_time_mean"].map("{:.4f}".format)
    view["seeds"] = s["n_seeds"]
    print(view.to_string(index=False))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()

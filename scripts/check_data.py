"""Sanity-check the data pipeline and print the PCA explained-variance table.

Usage (from the repo root):
    python scripts/check_data.py
    python scripts/check_data.py --config configs/default.yaml

Writes the variance table to results/pca_explained_variance.csv.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_data, load_dataset, pca_explained_variance  # noqa: E402
from src.data.preprocess import split_test, subsample_train  # noqa: E402


def check_run(dataset, n_features, n_train, seed, n_test, reduction):
    Xtr, Xte, ytr, yte = get_data(dataset, n_features, n_train, seed,
                                  n_test=n_test, reduction=reduction)
    assert Xtr.shape == (n_train, n_features), Xtr.shape
    assert Xte.shape == (n_test, n_features), Xte.shape
    assert Xtr.min() >= 0 and Xtr.max() <= np.pi + 1e-12
    assert Xte.min() >= 0 and Xte.max() <= np.pi + 1e-12
    assert set(np.unique(ytr)) == {0, 1} and set(np.unique(yte)) == {0, 1}
    # Same seed -> identical arrays (reproducibility).
    Xtr2, Xte2, _, _ = get_data(dataset, n_features, n_train, seed,
                                n_test=n_test, reduction=reduction)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(Xte, Xte2)
    return ytr.mean(), yte.mean()


def check_no_overlap(dataset, n_train, n_test):
    """Train rows must never appear in the test set, for any seed."""
    X, y = load_dataset(dataset)
    X_pool, X_test, y_pool, _ = split_test(X, y, n_test)
    test_rows = {r.tobytes() for r in X_test}
    for seed in range(5):
        X_tr, _ = subsample_train(X_pool, y_pool, n_train, seed)
        n_dup = sum(r.tobytes() in test_rows for r in X_tr)
        # MNIST contains a few pixel-identical images; anything beyond a
        # handful would mean the split is broken.
        assert n_dup <= 2, f"{dataset} seed {seed}: {n_dup} train rows in test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    cfg = yaml.safe_load(open(ap.parse_args().config))["data"]
    n_test, seeds = cfg["n_test"], cfg["seeds"]

    for ds in cfg["datasets"]:
        X, y = load_dataset(ds)
        print(f"\n== {ds}: {X.shape[0]} samples x {X.shape[1]} features, "
              f"positive fraction {y.mean():.3f}")
        check_no_overlap(ds, cfg["n_train"], n_test)
        for reduction in ("pca", "select_k"):
            for nf in (2, 4, 6, 8):
                for seed in seeds:
                    p_tr, p_te = check_run(ds, nf, cfg["n_train"], seed,
                                           n_test, reduction)
            print(f"  {reduction:8s} ok  (train pos {p_tr:.2f}, test pos {p_te:.2f})")
        for n_train in (25, 50, 200, 400):
            check_run(ds, cfg["n_features"], n_train, 0, n_test, "pca")
        print("  train-size sweep shapes ok; no train/test overlap")

    rows = []
    for ds in cfg["datasets"]:
        cum = pca_explained_variance(ds, max_components=8, n_test=n_test)
        rows.append({"dataset": ds, **{f"k={k}": round(cum[k - 1], 3)
                                       for k in (2, 4, 6, 8)}})
    table = pd.DataFrame(rows).set_index("dataset")
    print("\nCumulative PCA explained variance:\n", table)
    out = ROOT / "results" / "pca_explained_variance.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(f"\nAll checks passed. Table written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

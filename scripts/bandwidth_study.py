"""Mini study: why the QSVM bandwidth default is 0.05 (and hardware uses 0.25).

Same validation protocol as pick_bandwidth.py: train/validation points come
from the non-test pool, so the test set is never touched. PCA is fit on the
pool, the [0, pi] scaler on the training points, C is tuned by 5-fold CV (F1).

Parts (results/bandwidth_study/<part>.csv, figure figures/bandwidth_study.png):
  grid      4 features, ZZ reps=2 linear, fine bandwidth grid. Per bandwidth:
            validation F1, kernel off-diagonal mean/std, collapse diagnostics
            (share predicted positive, support-vector fraction, chosen C) and
            how RBF-like the kernel is (best centered alignment with an RBF
            kernel). Also an exact check: Z map, reps=1 == RBF-like cos^2 product.
  settings  Best bandwidth vs number of qubits (2/4/6/8) and reps (1/2/3):
            fixed 0.05 vs bandwidth picked by CV on the training set only.
  shots     Bandwidth x shots (exact/256/1024/4096), plus the hardware run's
            setting (2 features, reps=1, n_train=8, C=1, 200 shots).

Usage: python scripts/bandwidth_study.py [--parts grid settings shots] [--seeds 5]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.data.loaders import load_dataset
from src.data.preprocess import make_reducer, split_test
from src.quantum.kernels import quantum_kernel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "bandwidth_study"
FIG = ROOT / "figures" / "bandwidth_study.png"
DATASETS = ["breast_cancer", "mnist_3v5"]
C_GRID = [0.01, 0.1, 1, 10, 100]
FINE_BW = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.25, 0.5, 1.0]
BW = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
RBF_GAMMAS = [0.01, 0.1, 1, 10]
ZZ = dict(feature_map="zz", reps=2, entanglement="linear")

_pool = {}


def split(ds, n_features, n_train, n_val, seed):
    """Train / validation arrays in [0, pi] drawn from the non-test pool."""
    if (ds, n_features) not in _pool:
        X, y = load_dataset(ds)
        Xp, _, yp, _ = split_test(X.astype(float), y, 100)
        _pool[ds, n_features] = (Xp, yp, make_reducer(ds, n_features, "pca").fit(Xp))
    Xp, yp, red = _pool[ds, n_features]
    Xtr, Xr, ytr, yr = train_test_split(Xp, yp, train_size=n_train, stratify=yp, random_state=seed)
    Xva, _, yva, _ = train_test_split(Xr, yr, train_size=n_val, stratify=yr, random_state=100 + seed)
    sc = MinMaxScaler((0, np.pi)).fit(red.transform(Xtr))
    return (sc.transform(red.transform(Xtr)), np.clip(sc.transform(red.transform(Xva)), 0, np.pi),
            ytr, yva)


def cv_search(Ktr, ytr, seed):
    """Best (CV F1, C) on the training kernel; same CV as runner.tune_C."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    s = GridSearchCV(SVC(kernel="precomputed"), {"C": C_GRID}, cv=cv,
                     scoring=make_scorer(f1_score, zero_division=0)).fit(Ktr, ytr)
    return s.best_score_, s.best_params_["C"]


def evaluate(Ktr, ytr, Kva, yva, seed, C=None):
    """Fit on Ktr (C tuned unless given), score on validation, plus diagnostics."""
    cv_f1, C = cv_search(Ktr, ytr, seed) if C is None else (np.nan, C)
    svm = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
    pred = svm.predict(Kva)
    off = Ktr[~np.eye(len(Ktr), dtype=bool)]
    return dict(val_f1=f1_score(yva, pred, zero_division=0), cv_f1=cv_f1, C=C,
                pos_rate=pred.mean(), sv_frac=len(svm.support_) / len(ytr),
                offdiag_mean=off.mean(), offdiag_std=off.std())


def cka(K1, K2):
    """Centered kernel alignment: 1 = same kernel up to scale and shift."""
    H = np.eye(len(K1)) - 1.0 / len(K1)
    A, B = H @ K1 @ H, H @ K2 @ H
    return float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum()))


def rbf_reference(Ztr, ytr, Zva, yva, seed):
    """RBF val F1 with gamma='scale' and with gamma picked by CV on train."""
    g_scale = 1.0 / (Ztr.shape[1] * Ztr.var())
    scale = evaluate(rbf_kernel(Ztr, gamma=g_scale), ytr, rbf_kernel(Zva, Ztr, gamma=g_scale), yva, seed)
    best = max(RBF_GAMMAS, key=lambda g: cv_search(rbf_kernel(Ztr, gamma=g), ytr, seed)[0])
    tuned = evaluate(rbf_kernel(Ztr, gamma=best), ytr, rbf_kernel(Zva, Ztr, gamma=best), yva, seed)
    return scale["val_f1"], tuned["val_f1"], g_scale


def part_grid(seeds):
    rows, gammas = [], np.logspace(-4, 1, 51)
    for ds in DATASETS:
        for s in seeds:
            Ztr, Zva, ytr, yva = split(ds, 4, 100, 200, s)
            rbf_scale, rbf_tuned, g_scale = rbf_reference(Ztr, ytr, Zva, yva, s)
            rbf_tr = [rbf_kernel(Ztr, gamma=g) for g in gammas]
            for bw in FINE_BW:
                Ktr, Kva = quantum_kernel(Ztr, bandwidth=bw, **ZZ), quantum_kernel(Zva, Ztr, bandwidth=bw, **ZZ)
                al = [cka(Ktr, R) for R in rbf_tr]
                rows.append(dict(dataset=ds, seed=s, bandwidth=bw, **evaluate(Ktr, ytr, Kva, yva, s),
                                 rbf_cka=max(al), rbf_cka_gamma=gammas[int(np.argmax(al))],
                                 rbf_scale_f1=rbf_scale, rbf_tuned_f1=rbf_tuned, rbf_scale_gamma=g_scale,
                                 majority_f1=2 * ytr.mean() / (1 + ytr.mean())))

    # Exact check: the Z map with reps=1 is a product of cos^2 terms, which
    # is ~exp(-bw^2 * ||x - x'||^2), i.e. an RBF kernel with gamma = bw^2.
    Ztr = split("breast_cancer", 4, 100, 200, 0)[0]
    bw = 0.05
    Kz = quantum_kernel(Ztr, feature_map="z", reps=1, bandwidth=bw)
    D = Ztr[:, None, :] - Ztr[None, :, :]
    print(f"Z map reps=1 vs prod cos^2(bw*dx): max diff {np.abs(Kz - np.prod(np.cos(bw * D) ** 2, -1)).max():.1e};"
          f" vs RBF(gamma=bw^2): max diff {np.abs(Kz - rbf_kernel(Ztr, gamma=bw ** 2)).max():.1e}")
    return pd.DataFrame(rows)


def part_settings(seeds):
    rows = []
    settings = [dict(n_features=q, reps=2) for q in (2, 4, 6, 8)] + \
               [dict(n_features=4, reps=r) for r in (1, 3)]
    for ds in DATASETS:
        for st in settings:
            for s in seeds:
                Ztr, Zva, ytr, yva = split(ds, st["n_features"], 100, 200, s)
                rbf_scale, rbf_tuned, _ = rbf_reference(Ztr, ytr, Zva, yva, s)
                for bw in BW:
                    kw = dict(ZZ, reps=st["reps"], bandwidth=bw)
                    Ktr, Kva = quantum_kernel(Ztr, **kw), quantum_kernel(Zva, Ztr, **kw)
                    rows.append(dict(dataset=ds, seed=s, **st, bandwidth=bw,
                                     **evaluate(Ktr, ytr, Kva, yva, s),
                                     rbf_scale_f1=rbf_scale, rbf_tuned_f1=rbf_tuned))
    return pd.DataFrame(rows)


def part_shots(seeds):
    rows = []
    for ds in DATASETS:
        for s in seeds:
            Ztr, Zva, ytr, yva = split(ds, 4, 100, 200, s)
            for bw in BW:
                for shots in (None, 256, 1024, 4096):
                    kw = dict(ZZ, bandwidth=bw, shots=shots, seed=s)
                    Ktr, Kva = quantum_kernel(Ztr, **kw), quantum_kernel(Zva, Ztr, **kw)
                    rows.append(dict(setting="default", dataset=ds, seed=s, bandwidth=bw,
                                     shots=shots or "exact", **evaluate(Ktr, ytr, Kva, yva, s)))
    # configs/default.yaml hardware run: tiny n_train, fixed C=1, 200 shots.
    # More seeds since 8 training points make single runs very noisy.
    for s in range(30):
        Ztr, Zva, ytr, yva = split("breast_cancer", 2, 8, 200, s)
        for bw in BW:
            for shots in (None, 200):
                kw = dict(feature_map="zz", reps=1, entanglement="linear", bandwidth=bw, shots=shots, seed=s)
                Ktr, Kva = quantum_kernel(Ztr, **kw), quantum_kernel(Zva, Ztr, **kw)
                rows.append(dict(setting="hardware", dataset="breast_cancer", seed=s, bandwidth=bw,
                                 shots=shots or "exact", **evaluate(Ktr, ytr, Kva, yva, s, C=1.0)))
    return pd.DataFrame(rows)


def report(part, df):
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.precision", 3)
    if part == "grid":
        cols = ["val_f1", "offdiag_mean", "offdiag_std", "pos_rate", "sv_frac", "C", "rbf_cka", "rbf_cka_gamma"]
        print(df.groupby(["dataset", "bandwidth"])[cols].mean())
        print(df.groupby("dataset")[["rbf_scale_f1", "rbf_tuned_f1", "rbf_scale_gamma", "majority_f1"]].mean())
    elif part == "settings":
        g = df.groupby(["dataset", "n_features", "reps"])
        mean_f1 = df.groupby(["dataset", "n_features", "reps", "bandwidth"]).val_f1.mean()
        # Bandwidth a team member could pick without validation data: best CV F1 on train.
        cv_pick = df.loc[df.groupby(["dataset", "n_features", "reps", "seed"]).cv_f1.idxmax()]
        print(pd.DataFrame({
            "best_bw_val": mean_f1.groupby(level=[0, 1, 2]).idxmax().map(lambda t: t[-1]),
            "f1_best_bw": mean_f1.groupby(level=[0, 1, 2]).max(),
            "f1_bw_0.05": df[df.bandwidth == 0.05].groupby(["dataset", "n_features", "reps"]).val_f1.mean(),
            "f1_cv_picked": cv_pick.groupby(["dataset", "n_features", "reps"]).val_f1.mean(),
            "cv_picked_bw_median": cv_pick.groupby(["dataset", "n_features", "reps"]).bandwidth.median(),
            "offdiag_std_0.05": df[df.bandwidth == 0.05].groupby(["dataset", "n_features", "reps"]).offdiag_std.mean(),
            "rbf_scale_f1": g.rbf_scale_f1.mean(),
            "rbf_tuned_f1": g.rbf_tuned_f1.mean(),
        }))
    else:
        print(df.pivot_table(index=["setting", "dataset", "bandwidth"], columns="shots",
                             values="val_f1", aggfunc="mean", sort=False))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = pd.read_csv(OUT / "grid.csv")
    shots = pd.read_csv(OUT / "shots.csv")
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for ds, d in grid.groupby("dataset"):
        m = d.groupby("bandwidth").mean(numeric_only=True)
        sd = d.groupby("bandwidth").val_f1.std()
        line = ax[0].errorbar(m.index, m.val_f1, sd, marker="o", capsize=3, label=f"{ds} QSVM")[0]
        ax[0].axhline(m.rbf_tuned_f1.iloc[0], ls="--", color=line.get_color(), label=f"{ds} RBF (tuned)")
        ax[1].plot(m.index, m.offdiag_std, marker="o", label=f"{ds} std")
        ax[1].plot(m.index, m.offdiag_mean, marker="s", ls=":", color=line.get_color(), label=f"{ds} mean")
        ax[2].plot(m.index, m.pos_rate, marker="o", label=f"{ds} predicted positive")
        ax[2].plot(m.index, m.sv_frac, marker="s", ls=":", color=line.get_color(), label=f"{ds} support vectors")
        ax[3].plot(m.index, m.rbf_cka, marker="o", label=ds)
    for sh, d in shots[(shots.setting == "default") & (shots.dataset == "breast_cancer")].groupby("shots"):
        m = d.groupby("bandwidth").val_f1.mean()
        ax[0].plot(m.index, m.values, lw=1, alpha=0.6, label=f"BC shots={sh}")
    titles = ["Validation F1", "Kernel off-diagonal values", "Collapse check (fractions)",
              "Alignment with best RBF kernel (CKA)"]
    for a, t in zip(ax, titles):
        a.set_xscale("log"); a.set_xlabel("bandwidth λ"); a.set_title(t); a.legend(fontsize=7)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"saved {FIG}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["grid", "settings", "shots"])
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"grid": part_grid, "settings": part_settings, "shots": part_shots}
    for part in args.parts:
        print(f"\n=== {part} ===")
        df = parts[part](range(args.seeds))
        df.to_csv(OUT / f"{part}.csv", index=False)
        report(part, df)
    if (OUT / "grid.csv").exists() and (OUT / "shots.csv").exists():
        plot()

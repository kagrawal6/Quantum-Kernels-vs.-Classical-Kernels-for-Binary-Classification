"""Figures for the classical SVMs, and classical vs QSVM (results/*.csv -> figures/).

Usage:  python scripts/plot_classical_results.py
Needs the classical sweeps (scripts/run_experiment.py --models linear rbf) and,
for the comparison figures, the QSVM sweeps.
Every point = mean over 5 seeds; error bars / bands = std over seeds.

    classical_gamma.png               RBF gamma sweep: F1 + kernel concentration
    classical_learning_curves.png     train vs test F1 as training size grows (over-fitting)
    compare_baseline.png              linear vs RBF vs QSVM at the default setting
    compare_features.png              F1 vs feature count (= qubits for QSVM)
    compare_train_size.png            F1 vs training size (classical also on ALL data)
    compare_decision_boundaries.png   2-feature boundaries: linear, RBF, QSVM

Style (colors, fonts, helpers) is shared with scripts/plot_qsvm_results.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from plot_qsvm_results import (  # noqa: E402  (also applies the shared rcParams)
    CLASSES, COLOR, DATASETS, DEFAULT, GRID, INK, INK2, MARKER, MODEL_STYLE, NAME, RES,
    SURFACE, plot_mean_std, plt, save, title)

MODELS = ["linear", "rbf", "mlp", "qsvm"]


def load(name, models=MODELS):
    d = pd.read_csv(RES / f"{name}.csv")
    return d[d.model.isin(models)]


def qsvm_default(d):
    """QSVM rows at the default circuit (the params sweep also has other circuits)."""
    m = d.model == "qsvm"
    for k, v in DEFAULT.items():
        if k in d:
            m &= d[k] == v
    return d[(d.model != "qsvm") | m]


def model_line(ax, d, by, m, col="f1", **kw):
    label, color, mk, ls = MODEL_STYLE[m]
    g = d[d.model == m].groupby(by)[col].agg(["mean", "std"])
    if g.empty:
        return g
    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=color, marker=mk, ls=kw.pop("ls", ls), capsize=3,
                elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=kw.pop("label", label), **kw)
    return g


# 1. RBF gamma: F1 and kernel concentration (the classical twin of qsvm_bandwidth) ----
def fig_gamma():
    d = load("params", ["rbf"])
    fixed, scale = d[d.gamma != "scale"].copy(), d[d.gamma == "scale"]
    fixed["g"] = fixed.gamma.astype(float)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.3})
    for ds in DATASETS:
        dd = fixed[fixed.dataset == ds]
        plot_mean_std(a, dd, "g", ds, label=NAME[ds])
        s = scale[scale.dataset == ds].f1.mean()
        a.axhline(s, color=COLOR[ds], ls="--", lw=1.2, label=f"{NAME[ds]}, gamma = scale: {s:.3f}")
        m = dd.groupby("g").kernel_offdiag_mean.mean()
        sd = dd.groupby("g").kernel_offdiag_std.mean()
        b.fill_between(m.index, m - sd, m + sd, color=COLOR[ds], alpha=0.12, lw=0)
        b.plot(m.index, m, color=COLOR[ds], marker=MARKER[ds], markeredgecolor=SURFACE, label=NAME[ds])
    a.set(xscale="log", xlabel="gamma (larger = each point only 'sees' close neighbours)",
          ylabel="test F1", ylim=(0.4, 1.0), title="RBF SVM F1 vs gamma")
    a.legend(loc="lower left", fontsize=8.5)
    b.set(xscale="log", xlabel="gamma", ylabel="off-diagonal kernel value", ylim=(0, 1.05),
          title="Kernel values between different points")
    b.legend(loc="upper right")
    title(fig, "RBF gamma plays the same role as the QSVM bandwidth: too large concentrates the kernel")
    fig.text(0.01, -0.11, "Dashed = gamma 'scale' (sklearn default, 1 / (features × variance of the training data), "
             "about 0.6 for Breast Cancer and 0.5 for MNIST), which the other sweeps use. Right: line = mean, band = ± std of kernel values "
             "between different training points.", fontsize=8.5, color=INK2, wrap=True)
    save(fig, "classical_gamma.png")


# 2. Learning curves: train vs test F1 (over-fitting) --------------------------------
def fig_learning_curves():
    d = load("train_size", ["linear", "rbf"])
    if "train_f1" not in d or d.train_f1.isna().all():
        print("  skip classical_learning_curves.png (train_size has no train_f1 yet)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, ds in zip(axes, DATASETS):
        dd = d[d.dataset == ds]
        for m in ["linear", "rbf"]:
            label, color, mk, _ = MODEL_STYLE[m]
            model_line(a, dd, "n_train", m, "train_f1", label=f"{label}: train", ls=":")
            model_line(a, dd, "n_train", m, "f1", label=f"{label}: test", ls="-")
        a.set(xscale="log", xlabel="training points (log scale; last point = all data)",
              title=NAME[ds], ylim=(0.6, 1.02))
    axes[0].set_ylabel("F1")
    axes[1].legend(loc="lower right", fontsize=8.5)
    title(fig, "Classical SVMs: the train-test gap closes as training data grows")
    fig.text(0.01, -0.06, "Dotted = F1 on the points the SVM was trained on; solid = F1 on the fixed test set. "
             "A big gap = over-fitting. 4 PCA features.", fontsize=8.5, color=INK2)
    save(fig, "classical_learning_curves.png")


# 3. Baseline: one number per model ------------------------------------------------
def fig_baseline():
    d = load("baseline")
    fig, a = plt.subplots(figsize=(7.4, 4.2))
    y = np.arange(len(MODELS))
    for k, ds in enumerate(DATASETS):
        g = d[d.dataset == ds].groupby("model").f1.agg(["mean", "std"]).reindex(MODELS)
        a.errorbar(g["mean"], y + (k - 0.5) * 0.28, xerr=g["std"], fmt=MARKER[ds], color=COLOR[ds],
                   capsize=3, elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=1.5,
                   label=NAME[ds], ms=8)
        for yi, (mu, sd) in zip(y, g.itertuples(index=False)):
            a.annotate(f"{mu:.3f}", (mu + sd, yi + (k - 0.5) * 0.28), xytext=(6, 0),
                       textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a.set_yticks(y, [MODEL_STYLE[m][0] for m in MODELS])
    a.invert_yaxis()
    a.set(xlabel="test F1 (4 features, 100 training points)", xlim=(0.78, 0.99))
    a.grid(axis="y", visible=False)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    title(fig, "At the default setting all four models are within one standard deviation")
    save(fig, "compare_baseline.png")


# 4. Feature count ----------------------------------------------------------------
def fig_features():
    d = qsvm_default(load("features"))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, ds in zip(axes, DATASETS):
        for m in MODELS:
            model_line(a, d[d.dataset == ds], "n_features", m)
        a.set(xlabel="features (= qubits for QSVM)", xticks=[2, 4, 6, 8], title=NAME[ds], ylim=(0.6, 1.0))
    axes[0].set_ylabel("test F1")
    axes[1].legend(loc="lower right")
    title(fig, "F1 vs feature count: QSVM tracks the classical models, it does not pull ahead")
    save(fig, "compare_features.png")


# 5. Training size ----------------------------------------------------------------
def fig_train_size():
    d = load("train_size")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True, gridspec_kw={"wspace": 0.08})
    for a, ds in zip(axes, DATASETS):
        dd = d[d.dataset == ds]
        for m in MODELS:
            model_line(a, dd, "n_train", m)
        n_all = int(dd.n_train.max()) if len(dd) else None
        if n_all:
            a.axvline(n_all, color=GRID, lw=1, zorder=0)
            a.annotate(f"all data\n({n_all:,})", (n_all, 0.62), xytext=(-4, 0), textcoords="offset points",
                       ha="right", fontsize=8, color=INK2)
        a.set(xscale="log", xlabel="training points (log scale)", title=NAME[ds], ylim=(0.6, 1.0))
    axes[0].set_ylabel("test F1")
    axes[1].legend(loc="lower center")
    title(fig, "F1 vs training size: on the simulator QSVM tracks the classical models all the way to the full dataset")
    fig.text(0.01, -0.06, "Simulator only. On real hardware the QSVM kernel needs one circuit per pair of points: ~80,000 circuits "
             "at 400 points, ~90 million for all of MNIST (see qsvm_hardware_cost.png). 4 features.", fontsize=8.5, color=INK2)
    save(fig, "compare_train_size.png")


# 6. Decision boundaries: linear vs RBF vs QSVM (2 features) -------------------------
def fig_boundaries():
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from src.classical.kernels import linear_kernel, rbf_kernel
    from src.data import get_data
    from src.evaluation.runner import tune_C
    from src.quantum.kernels import quantum_kernel

    CLASS_COLOR = ("#1baf7a", "#6250d6")
    CLASS_MARK = ("o", "^")
    cmap = matplotlib.colors.ListedColormap(["#d6f0e6", "#e2defa"])
    models = [("Linear SVM", linear_kernel), ("RBF SVM (gamma = scale)", rbf_kernel),
              ("QSVM (ZZ, bandwidth 0.05)", lambda A, B=None: quantum_kernel(A, B, bandwidth=0.05))]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.4), gridspec_kw={"hspace": 0.42, "wspace": 0.12})
    g = np.linspace(0, np.pi, 90)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()]
    for row, ds in zip(axes, DATASETS):
        Xtr, Xte, ytr, yte = get_data(ds, 2, 100, 0)
        for a, (label, kern) in zip(row, models):
            Ktr = kern(Xtr)
            C = tune_C(Ktr, ytr, [0.01, 0.1, 1, 10, 100], 5, 0)
            clf = SVC(kernel="precomputed", C=C).fit(Ktr, ytr)
            f1 = f1_score(yte, clf.predict(kern(Xte, Xtr)))
            z = clf.decision_function(kern(grid, Xtr)).reshape(gx.shape)
            a.contourf(gx, gy, (z > 0).astype(int), levels=[-0.5, 0.5, 1.5], cmap=cmap)
            a.contour(gx, gy, z, levels=[0], colors=INK, linewidths=1.6)
            for c in (0, 1):
                m = ytr == c
                a.scatter(Xtr[m, 0], Xtr[m, 1], s=22, c=CLASS_COLOR[c], marker=CLASS_MARK[c],
                          edgecolors=SURFACE, linewidths=0.6, label=CLASSES[ds][c], zorder=3)
            a.set_title(f"{label}\n{NAME[ds]}: test F1 {f1:.3f}", fontsize=9.5, loc="left")
            a.set(xlim=(0, np.pi), ylim=(0, np.pi), xticks=[], yticks=[])
            a.grid(False)
            a.set_aspect("equal")
        row[0].set_ylabel("PCA feature 2")
        for a in row:
            a.set_xlabel("PCA feature 1", fontsize=8.5)
        row[-1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=9.5)
    title(fig, "Decision boundaries with 2 features: at bandwidth 0.05 the QSVM boundary is almost a straight line, like the linear SVM")
    fig.text(0.01, -0.01, "Each panel: 100 training points (seed 0), 2 PCA features scaled to [0, π]. "
             "Shading = class the SVM predicts; black line = decision boundary. C tuned by 5-fold CV.\n"
             "One seed only, for illustration: compare boundary shapes, not the F1 values.",
             fontsize=8.5, color=INK2)
    save(fig, "compare_decision_boundaries.png")


if __name__ == "__main__":
    for f in [fig_gamma, fig_learning_curves, fig_baseline, fig_features, fig_train_size,
              fig_boundaries]:
        f()

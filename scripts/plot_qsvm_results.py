"""Figures for the QSVM simulator results (results/*.csv -> figures/qsvm_*.png).

Usage:  python scripts/plot_qsvm_results.py
Needs the QSVM sweeps to have been run (scripts/run_experiment.py --models qsvm).
Every point = mean over 5 seeds; error bars / bands = std over seeds.

Both datasets share each figure (color + marker + legend); hardware cost does
not depend on the dataset.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, FIG = ROOT / "results", ROOT / "figures"

DATASETS = ["breast_cancer", "mnist_3v5"]
# Dataset color kept consistent across all figures (validated pair, dataviz slots 1-2).
COLOR = {"breast_cancer": "#2a78d6", "mnist_3v5": "#eb6834"}
MARKER = {"breast_cancer": "o", "mnist_3v5": "s"}
NAME = {"breast_cancer": "Breast Cancer", "mnist_3v5": "MNIST 3 vs 5"}
CLASSES = {"breast_cancer": ("benign", "malignant"), "mnist_3v5": ("digit 3", "digit 5")}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
DEFAULT = {"feature_map": "zz", "reps": 2, "entanglement": "linear", "bandwidth": 0.05}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 7,
})


def load(name, ds=None):
    d = pd.read_csv(RES / f"{name}.csv")
    d = d[d.model == "qsvm"]
    return d if ds is None else d[d.dataset == ds]


def plot_mean_std(ax, d, by, ds, col="f1", label=None):
    """One line: mean +/- std over seeds of `col` against `by`."""
    g = d.groupby(by)[col].agg(["mean", "std"])
    ax.errorbar(g.index, g["mean"], yerr=g["std"], color=COLOR[ds], marker=MARKER[ds],
                capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=label)
    return g


def one_factor(d, p):
    """Rows of the params sweep where only `p` differs from the defaults."""
    m = np.ones(len(d), bool)
    for k, v in DEFAULT.items():
        if k != p:
            m &= (d[k] == v).to_numpy()
    return d[m]


def title(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures/{name}")


# 1. Bandwidth: F1 and kernel concentration -------------------------------------
def fig_bandwidth():
    d = one_factor(load("params"), "bandwidth")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"wspace": 0.3})
    for ds in DATASETS:
        dd = d[d.dataset == ds]
        plot_mean_std(a, dd, "bandwidth", ds, label=NAME[ds])
        m = dd.groupby("bandwidth").kernel_offdiag_mean.mean()
        s = dd.groupby("bandwidth").kernel_offdiag_std.mean()
        b.fill_between(m.index, m - s, m + s, color=COLOR[ds], alpha=0.12, lw=0)
        b.plot(m.index, m, color=COLOR[ds], marker=MARKER[ds], markeredgecolor=SURFACE, label=NAME[ds])
    a.axvline(0.05, color=INK2, lw=1, ls=":")
    a.annotate("default 0.05", (0.05, 0.985), xytext=(4, 0), textcoords="offset points",
               fontsize=8.5, color=INK2, va="top")
    a.set(xscale="log", xlabel="bandwidth λ (inputs scaled by λ before encoding)",
          ylabel="test F1", ylim=(0.4, 1.0), title="QSVM F1 vs bandwidth")
    a.legend(loc="lower left")
    b.set(xscale="log", xlabel="bandwidth λ", ylabel="off-diagonal kernel value",
          ylim=(0, 1.05), title="Kernel values between different points")
    b.legend(loc="upper right")
    title(fig, "Large bandwidth concentrates the quantum kernel and destroys accuracy")
    fig.text(0.01, -0.11, "Right: line = mean, band = ± spread (std) of kernel values between different "
             "training points. Near 0 = points look unrelated (concentrated); near 1 = points look alike, "
             "so their small differences need many shots to resolve.", fontsize=8.5, color=INK2, wrap=True)
    save(fig, "qsvm_bandwidth.png")


# 2. Qubits (= feature count) --------------------------------------------------
def fig_qubits():
    d = load("features")
    fig, a = plt.subplots(figsize=(6.2, 3.8))
    for ds in DATASETS:
        plot_mean_std(a, d[d.dataset == ds], "n_features", ds, label=NAME[ds])
    a.set(xlabel="qubits (= PCA features)", ylabel="test F1", ylim=(0.6, 1.0),
          xticks=[2, 4, 6, 8], title="QSVM F1 vs qubits")
    a.legend(loc="lower right")
    save(fig, "qsvm_qubits.png")


# 3. Shots -----------------------------------------------------------------------
def fig_shots():
    d = load("shots")
    fig, a = plt.subplots(figsize=(6.2, 3.8))
    for ds in DATASETS:
        dd = d[d.dataset == ds]
        exact = dd[dd.shots.isna()].f1.mean()
        plot_mean_std(a, dd[dd.shots.notna()], "shots", ds, label=f"{NAME[ds]} (shots)")
        a.axhline(exact, color=COLOR[ds], ls="--", lw=1.2, label=f"{NAME[ds]} exact: {exact:.3f}")
    ticks = [256, 1024, 4096, 16384, 65536, 262144]
    a.set(xscale="log", xticks=ticks, xticklabels=["256", "1k", "4k", "16k", "65k", "262k"],
          xlabel="shots per kernel entry", ylabel="test F1", ylim=(0.6, 1.0),
          title="QSVM F1 approaches the exact result only near 10⁵ shots")
    a.legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, -0.04, "Dashed = exact simulation (infinite shots). Shot noise only; no hardware noise simulated.",
             fontsize=8.5, color=INK2)
    save(fig, "qsvm_shots.png")


# 4. Training size ----------------------------------------------------------------
def fig_train_size():
    d = load("train_size")
    fig, a = plt.subplots(figsize=(6.2, 3.8))
    for ds in DATASETS:
        plot_mean_std(a, d[d.dataset == ds], "n_train", ds, label=NAME[ds])
    a.set(xscale="log", xticks=[25, 50, 100, 200, 400],
          xticklabels=["25", "50", "100", "200", "400"], xlabel="training points",
          ylabel="test F1", ylim=(0.6, 1.0), title="QSVM F1 vs training size")
    a.legend(loc="lower right")
    save(fig, "qsvm_train_size.png")


# 5. Circuit design: feature map, reps, entanglement ------------------------------
def fig_design():
    d = load("params")
    configs = [("Z map (no entanglement)", {"feature_map": "z"}),
               ("ZZ, reps 1", {"reps": 1}),
               ("ZZ, reps 2  (default)", {}),
               ("ZZ, reps 3", {"reps": 3}),
               ("ZZ, reps 2, full entanglement", {"entanglement": "full"})]
    fig, a = plt.subplots(figsize=(8, 3.9))
    y = np.arange(len(configs))[::-1]
    cx = []
    for k, ds in enumerate(DATASETS):
        f1, sd = [], []
        for label, over in configs:
            want = {**DEFAULT, **over}
            m = (d.dataset == ds).to_numpy()
            for key, v in want.items():
                m &= (d[key] == v).to_numpy()
            s = d[m]
            f1.append(s.f1.mean()); sd.append(s.f1.std())
            if k == 0:
                cx.append(s.n_2q_gates.iloc[0])
        a.errorbar(f1, y + (0.12 if k == 0 else -0.12), xerr=sd, fmt=MARKER[ds], color=COLOR[ds],
                   capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5, label=NAME[ds])
    a.set_yticks(y, [f"{c}\n{n} CX gates" for (c, _), n in zip(configs, cx)])
    a.set(xlabel="test F1 (4 qubits, 100 training points, bandwidth 0.05)", xlim=(0.78, 0.97),
          title="Entanglement does not help: the classically easy Z map matches ZZ")
    a.grid(axis="y", visible=False)
    a.legend(loc="lower left", ncols=2, bbox_to_anchor=(0, -0.32))
    save(fig, "qsvm_circuit_design.png")


# 6. Kernel heatmaps (one row per dataset) ------------------------------------------
def fig_heatmaps():
    """Each panel gets its OWN color range (2nd-98th percentile of its off-diagonal
    values), so class structure is visible at every bandwidth. On one shared 0-1
    scale the best setting (0.05) looks like a flat dark square, which misleads.
    Absolute values are stated in each title instead."""
    from src.data import get_data
    from src.quantum.kernels import quantum_kernel
    f1_by_bw = (one_factor(load("params"), "bandwidth")
                .groupby(["dataset", "bandwidth"]).f1.mean())
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4), gridspec_kw={"hspace": 0.5, "wspace": 0.35})
    for row, ds in zip(axes, DATASETS):
        Xtr, _, ytr, _ = get_data(ds, 4, 100, 0)
        order = np.argsort(ytr, kind="stable")
        X, y = Xtr[order], ytr[order]
        n0 = int((y == 0).sum())
        c0, c1 = CLASSES[ds]
        for a, bw in zip(row, [1.0, 0.25, 0.05]):
            K = quantum_kernel(X, bandwidth=bw)
            off = ~np.eye(len(K), dtype=bool)
            same = (y[:, None] == y[None, :]) & off
            diff = y[:, None] != y[None, :]
            lo, hi = np.percentile(K[off], [2, 98])
            im = a.imshow(K, cmap="Blues", vmin=lo, vmax=hi)
            a.set_title(f"{NAME[ds]}, bandwidth {bw}   (F1 {f1_by_bw[(ds, bw)]:.3f})\n"
                        f"same-class {K[same].mean():.3f}  vs  different-class {K[diff].mean():.3f}",
                        fontsize=9.5, loc="center")
            a.axhline(n0 - 0.5, color=SURFACE, lw=1.5)
            a.axvline(n0 - 0.5, color=SURFACE, lw=1.5)
            ticks = [n0 / 2, n0 + (len(K) - n0) / 2]
            a.set_xticks(ticks, [c0, c1])
            a.set_yticks(ticks, [c0, c1], rotation=90, va="center")
            a.grid(False)
            for s_ in a.spines.values():
                s_.set_visible(False)
            cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03, ticks=[lo, hi])
            cb.ax.set_yticklabels([f"{lo:.2f}", f"{hi:.2f}"], fontsize=8)
            cb.outline.set_visible(False)
    title(fig, "Training kernel at three bandwidths (100 points sorted by class, seed 0)")
    fig.text(0.01, 0.02, "Color range differs per panel (its own 2nd–98th percentile, shown on each colorbar) so the "
             "class structure is visible at every bandwidth; compare absolute levels using the titles.\n"
             "λ = 1.0: all values near 0, every point looks unrelated (concentration). λ = 0.05: all values near 1, but "
             "same-class pairs are consistently a little higher. F1 = mean over 5 seeds from the bandwidth sweep.",
             fontsize=8.5, color=INK2)
    save(fig, "qsvm_kernel_heatmaps.png")


# 7. Real-hardware price of one run ----------------------------------------------
def fig_cost():
    import yaml
    hw = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))["hardware"]
    devices = [  # (config key, label, color, marker, linestyle) - validated set + markers + direct labels
        ("Cepheus-1-108Q", "Rigetti Cepheus", "#1baf7a", "o", "-"),
        ("Garnet", "IQM Garnet", "#6250d6", "s", "--"),
        ("Forte-1", "IonQ Forte", "#e87ba4", "D", ":"),
    ]

    def cost(n_circuits, shots, dev):
        pr = hw["prices"][dev]
        return n_circuits * (pr["task_fee"] + shots * pr["per_shot"])

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.45})

    # (a) project-scale run: n_train training points + 100 test points, 1024 shots
    n = np.array([25, 50, 100, 200, 400])
    circ = n * (n - 1) // 2 + 100 * n
    nudge = {"Cepheus-1-108Q": -12, "Garnet": 12, "Forte-1": 0}
    for dev, label, col, mk, ls in devices:
        c = cost(circ, 1024, dev)
        a.plot(n, c, color=col, marker=mk, ls=ls, markeredgecolor=SURFACE, label=label)
        a.annotate(f"{label}\n${c[-1]:,.0f}", (n[-1], c[-1]), xytext=(8, nudge[dev]),
                   textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    a.axhline(100, color=INK, lw=1.2, ls="--")
    a.annotate("our $100 AWS credit", (25, 100), xytext=(2, 5), textcoords="offset points",
               fontsize=8.5, color=INK2)
    a.set(xscale="log", yscale="log", xticks=list(n), xticklabels=[str(v) for v in n],
          xlabel="training points (+100 test points)", ylabel="cost of one run (USD, log)",
          title="One full-size QSVM run, 1024 shots")
    a.set_xlim(20, 1500)

    # (b) our planned hardware run (config): n_train x n_test, vary shots
    nt, ne = hw["n_train"], hw["n_test"]
    circ_hw = nt * (nt - 1) // 2 + ne * nt
    shots = np.array([100, 200, 500, 1000, 2000])
    for dev, label, col, mk, ls in devices[:2]:  # IonQ is off the chart (> $1,000)
        c = cost(circ_hw, shots, dev)
        b.plot(shots, c, color=col, marker=mk, ls=ls, markeredgecolor=SURFACE, label=label)
        k = len(shots) - 1 if c[-1] < 130 else int(np.argmax(c > 100)) - 1  # last on-chart point
        b.annotate(label, (shots[k], c[k]), xytext=(8, 0) if k == len(shots) - 1 else (-8, 4),
                   ha="left" if k == len(shots) - 1 else "right", textcoords="offset points",
                   va="center", fontsize=8.5, color=INK2)
    sel = cost(circ_hw, hw["shots"], hw["device"])
    b.plot([hw["shots"]], [sel], marker="*", markersize=16, color=INK, ls="none", zorder=5)
    b.annotate(f"planned run: {hw['shots']} shots\n${sel:.2f} on Rigetti", (hw["shots"], sel),
               xytext=(10, -26), textcoords="offset points", fontsize=8.5, color=INK)
    for y, txt in [(100, "$100 total credit"), (hw["max_run_usd"], f"${hw['max_run_usd']} per-run cap\n(script refuses above)")]:
        b.axhline(y, color=INK, lw=1.2, ls="--" if y == 100 else ":")
        below = y != 100
        b.annotate(txt, (4400, y), xytext=(0, -4 if below else 4), textcoords="offset points",
                   ha="right", va="top" if below else "bottom", fontsize=8.5, color=INK2)
    ionq = cost(circ_hw, hw["shots"], "Forte-1")
    b.text(0.98, 0.04, f"IonQ Forte at {hw['shots']} shots: ${ionq:,.0f} (off chart)",
           transform=b.transAxes, ha="right", fontsize=8.5, color=INK2)
    b.set(xscale="log", xticks=list(shots), xticklabels=[str(s) for s in shots], ylim=(0, 130),
          xlabel="shots per circuit", ylabel="cost of the run (USD)",
          title=f"Our planned hardware run ({nt} train, {ne} test, {circ_hw} circuits)")
    b.set_xlim(80, 4500)

    fig.suptitle("Real quantum hardware: what one QSVM run costs on AWS Braket",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", y=1.03)
    fig.text(0.01, -0.06, "Cost = circuits × (\\$0.30 per task + shots × per-shot price); one circuit per kernel entry, "
             "circuits = n_train(n_train−1)/2 + n_test·n_train. Prices from configs/default.yaml "
             "(third-party listings, Nov 2026): verify at aws.amazon.com/braket/pricing.",
             fontsize=8, color=INK2, wrap=True)
    save(fig, "qsvm_hardware_cost.png")


# 8. Decision boundaries in 2D (2 qubits) -----------------------------------------
def fig_boundaries():
    """2 PCA features -> 2 qubits, so the input space is a plane we can draw."""
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC
    from src.data import get_data
    from src.evaluation.runner import tune_C
    from src.quantum.kernels import quantum_kernel

    CLASS_COLOR = ("#1baf7a", "#6250d6")   # validated pair; class markers differ too
    CLASS_MARK = ("o", "^")
    cmap = matplotlib.colors.ListedColormap(["#d6f0e6", "#e2defa"])  # light tints of the two
    models = [("QSVM, bandwidth 1.0", 1.0), ("QSVM, bandwidth 0.25", 0.25),
              ("QSVM, bandwidth 0.05", 0.05)]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.4), gridspec_kw={"hspace": 0.42, "wspace": 0.12})
    g = np.linspace(0, np.pi, 90)
    gx, gy = np.meshgrid(g, g)
    grid = np.c_[gx.ravel(), gy.ravel()]
    for row, ds in zip(axes, DATASETS):
        Xtr, Xte, ytr, yte = get_data(ds, 2, 100, 0)
        for a, (label, bw) in zip(row, models):
            kern = lambda A, B=None, bw=bw: quantum_kernel(A, B, bandwidth=bw)  # noqa: E731
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
        row[0].set_ylabel("PCA feature 2  (qubit 2 angle)")
        for a in row:
            a.set_xlabel("PCA feature 1  (qubit 1 angle)", fontsize=8.5)
        row[-1].legend(loc="center left", bbox_to_anchor=(1.03, 0.5), fontsize=9.5)
    title(fig, "Decision boundaries with 2 qubits: larger bandwidth makes the QSVM boundary more irregular")
    fig.text(0.01, -0.01, "Each panel: 100 training points (seed 0), 2 PCA features scaled to [0, π]. "
             "Shading = class the SVM predicts; black line = decision boundary. ZZ map, reps 2; C tuned by 5-fold CV.\n"
             "One seed only, for illustration: compare boundary shapes, not the F1 values. With 2 qubits concentration "
             "is mild, so bandwidth 1.0 does not collapse here as it does at 4 qubits.",
             fontsize=8.5, color=INK2)
    save(fig, "qsvm_decision_boundaries.png")


# 9. Timing: kernel computation, SVM training, inference --------------------------
MODEL_STYLE = {  # (label, color, marker, linestyle); models are the entities here
    "linear": ("Linear SVM", "#8a8984", "s", "--"),
    "rbf": ("RBF SVM", INK, "^", "-."),
    "qsvm": ("QSVM (simulator)", "#6250d6", "o", "-"),
}


def fig_timing():
    """Every model found in the CSVs is drawn; linear/RBF appear once their runs exist.
    Averaged over both datasets (timing barely depends on the data)."""
    ts = pd.read_csv(RES / "train_size.csv")
    ft = pd.read_csv(RES / "features.csv")
    for d in (ts, ft):
        d["kernel_time"] = d.kernel_train_time + d.kernel_test_time
    sizes = [25, 50, 100, 200, 400]
    ts = ts[ts.n_train.isin(sizes)]
    panels = [(ts, "n_train", "kernel_time", "Kernel computation vs training size", "training points"),
              (ft, "n_features", "kernel_time", "Kernel computation vs qubits", "qubits (= features)"),
              (ts, "n_train", "fit_time", "SVM training (fit) vs training size", "training points"),
              (ts, "n_train", "predict_time", "Inference (predict) vs training size", "training points")]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), gridspec_kw={"hspace": 0.45, "wspace": 0.28})
    present = [m for m in MODEL_STYLE if m in set(ts.model)]
    for a, (d, by, col, ttl, xl) in zip(axes.ravel(), panels):
        for m in present:
            label, color, mk, ls = MODEL_STYLE[m]
            g = d[d.model == m].groupby(by)[col].agg(["mean", "std"])
            a.errorbar(g.index, g["mean"], yerr=g["std"], color=color, marker=mk, ls=ls, capsize=3,
                       elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
        a.set(yscale="log", ylabel="time (log scale)", xlabel=xl, title=ttl, ylim=(1e-5, 3))
        a.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                     ["10 µs", "100 µs", "1 ms", "10 ms", "0.1 s", "1 s"])
        a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if by == "n_train":
            a.set_xscale("log")
            a.set_xticks(sizes, [str(s) for s in sizes])
        else:
            a.set_xticks([2, 4, 6, 8])
        a.legend(loc="upper left", fontsize=8.5)
    missing = [MODEL_STYLE[m][0] for m in MODEL_STYLE if m not in present]
    title(fig, "Where the time goes: kernel computation is ~1000× slower than SVM training or inference")
    note = ("All panels share one time scale. Mean over 5 seeds and both datasets, one laptop CPU (results/machine.txt). Simulator time is a classical "
            "computer imitating the circuit, NOT quantum-hardware time:\na real-device run needs one circuit per "
            "kernel entry (~15,000 for 100 training points) and takes hours with queueing; see qsvm_hardware_cost.png.")
    if missing:
        note += f"\nNot yet run: {', '.join(missing)}; they will appear here automatically once their sweeps are in the CSVs."
    fig.text(0.01, -0.02, note, fontsize=8.5, color=INK2)
    save(fig, "timing.png")


if __name__ == "__main__":
    for f in [fig_bandwidth, fig_qubits, fig_shots, fig_train_size,
              fig_design, fig_heatmaps, fig_cost, fig_boundaries, fig_timing]:
        f()

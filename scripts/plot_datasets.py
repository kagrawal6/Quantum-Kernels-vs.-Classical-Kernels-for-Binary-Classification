"""What the two datasets actually look like (for the Dataset Preparation slide).

    python scripts/plot_datasets.py
    -> figures/dataset_breast_cancer.png   a few real rows of the 30 measurements
    -> figures/dataset_mnist.png           real 3s and 5s from MNIST
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.loaders import load_dataset  # noqa: E402

FIG = ROOT / "figures"
INK, INK2, RULE, SURFACE, HEAD = "#1B3C42", "#5B6E72", "#C9D3D2", "#F8F8F7", "#ECEFEE"
COLS = ["mean radius", "mean texture", "mean area"]


def breast_cancer():
    from sklearn.datasets import load_breast_cancer
    X, y = load_dataset("breast_cancer")
    names = list(load_breast_cancer().feature_names)
    idx = [names.index(c) for c in COLS]
    rng = np.random.default_rng(3)
    rows = list(rng.choice(np.where(y == 0)[0], 3, replace=False)) + \
        list(rng.choice(np.where(y == 1)[0], 3, replace=False))
    rng.shuffle(rows)
    header = ["sample"] + [c.replace("mean ", "") for c in COLS] + ["+27 more", "label"]
    cells = [[f"#{r}"] + [f"{X[r, i]:.2f}" if X[r, i] < 100 else f"{X[r, i]:.0f}" for i in idx]
             + ["…", "malignant" if y[r] == 1 else "benign"] for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=header, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(13)
    t.scale(1, 2.6)
    t.auto_set_column_width(list(range(len(header))))
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(RULE)
        cell.set_text_props(color=INK)
        cell.set_facecolor(HEAD if r == 0 else SURFACE)
        if r == 0:
            cell.set_text_props(color=INK, fontweight="bold")
        if c == len(header) - 1 and r > 0:
            cell.set_text_props(fontweight="bold",
                                color="#B5462C" if cells[r - 1][-1] == "malignant" else "#2A6F8F")
    fig.savefig(FIG / "dataset_breast_cancer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def mnist():
    X, y = load_dataset("mnist_3v5")
    rng = np.random.default_rng(0)
    n = 3
    fig, axes = plt.subplots(2, n, figsize=(5.4, 3.6), gridspec_kw={"wspace": 0.08, "hspace": 0.12})
    fig.patch.set_facecolor(SURFACE)
    for row, (lab, name) in enumerate([(0, "3"), (1, "5")]):
        pick = rng.choice(np.where(y == lab)[0], n, replace=False)
        for a, i in zip(axes[row], pick):
            a.imshow(X[i].reshape(28, 28), cmap="Greys", vmin=0, vmax=1)
            a.set_xticks([]), a.set_yticks([])
            for s in a.spines.values():
                s.set_edgecolor(RULE)
        axes[row, 0].set_ylabel(f"digit {name}", color=INK, fontsize=13, rotation=0,
                                ha="right", va="center", labelpad=12)
    fig.savefig(FIG / "dataset_mnist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    breast_cancer()
    mnist()
    print("  figures/dataset_breast_cancer.png\n  figures/dataset_mnist.png")

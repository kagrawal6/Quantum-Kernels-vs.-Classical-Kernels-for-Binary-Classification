"""Load the raw datasets as (X, y) with the positive class labeled 1.

Both loaders return the FULL dataset. Splitting, reduction and scaling
happen in preprocess.py so that every model sees identical arrays.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer

# <repo>/data/raw  (git-ignored; only the README is committed)
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = ("breast_cancer", "mnist_3v5")


def load_breast_cancer_data():
    """Breast Cancer Wisconsin (Diagnostic): 569 samples x 30 features.

    sklearn encodes malignant = 0 and benign = 1. We flip it so that
    malignant = 1, because malignant is our positive class (the class
    precision / recall / F1 are computed for).
    """
    ds = load_breast_cancer()
    X = ds.data.astype(np.float64)
    y = (ds.target == 0).astype(int)  # 1 = malignant (212), 0 = benign (357)
    return X, y


def load_mnist_3v5():
    """MNIST digits 3 vs 5: ~13.4k samples x 784 pixel features.

    Positive class = digit 5. Pixels are rescaled from 0..255 to 0..1.
    The first call downloads MNIST from OpenML (~50 MB) into data/raw/;
    the filtered 3-vs-5 subset is then cached as data/raw/mnist_3v5.npz
    so later calls are instant and work offline.

    We pool the official 60k train + 10k test images and make our own
    split later, so the split rules are identical for both datasets.
    """
    cache = RAW_DIR / "mnist_3v5.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # parser="liac-arff": sklearn's pandas ARFF parser crashes on pandas 3.x.
    X, labels = fetch_openml(
        "mnist_784", version=1, as_frame=False, parser="liac-arff", return_X_y=True,
        data_home=str(RAW_DIR),
    )
    labels = labels.astype(str)  # OpenML gives the digits as strings
    keep = (labels == "3") | (labels == "5")
    X = (X[keep] / 255.0).astype(np.float32)
    y = (labels[keep] == "5").astype(int)  # 1 = digit 5, 0 = digit 3

    np.savez_compressed(cache, X=X, y=y)
    return X, y


def load_dataset(name):
    """Return the full (X, y) for a dataset name in DATASETS."""
    if name == "breast_cancer":
        return load_breast_cancer_data()
    if name == "mnist_3v5":
        return load_mnist_3v5()
    raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")

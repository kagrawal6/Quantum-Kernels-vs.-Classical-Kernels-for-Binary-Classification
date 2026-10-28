"""Dataset loaders.

Owner: Steve Cho
Datasets: Breast Cancer Wisconsin [2], MNIST binary subset [1]
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a binary classification dataset.

    Parameters
    ----------
    name:
        One of: "breast_cancer", "mnist_binary" (extend as needed).

    Returns
    -------
    X, y : features and binary labels
    """
    name = name.lower()
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer()
        return data.data, data.target

    if name == "mnist_binary":
        # TODO: load MNIST and keep two digits (e.g. 0 vs 1)
        raise NotImplementedError("MNIST binary loader not implemented yet.")

    raise ValueError(f"Unknown dataset: {name}")

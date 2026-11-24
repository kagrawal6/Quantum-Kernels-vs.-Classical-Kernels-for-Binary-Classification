"""Quality metrics and kernel diagnostics (EXPERIMENTS.md Sections 2 and 6)."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_metrics(y_true, y_pred):
    """Accuracy, precision, recall, F1. Positive class = 1 (malignant / digit 5).

    zero_division=0: if a model never predicts the positive class, precision
    is reported as 0 instead of raising a warning.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def kernel_diagnostics(K_train):
    """Statistics of the off-diagonal entries of the training kernel.

    Computed identically for all three kernels so they can be compared.
    A very small std means the kernel values are all nearly the same
    ("exponential concentration", Thanasilp et al. [4]): the kernel no
    longer tells points apart and the SVM cannot learn much.
    """
    K = np.asarray(K_train)
    off = K[~np.eye(K.shape[0], dtype=bool)]
    return {
        "kernel_offdiag_mean": off.mean(),
        "kernel_offdiag_std": off.std(),
        "kernel_offdiag_min": off.min(),
        "kernel_offdiag_max": off.max(),
    }

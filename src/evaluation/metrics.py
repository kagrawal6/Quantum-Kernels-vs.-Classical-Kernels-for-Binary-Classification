"""Quality metrics and kernel diagnostics (EXPERIMENTS.md Sections 2 and 6)."""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, hinge_loss,
                             precision_score, recall_score)


def svm_hinge_loss(y_true, decision):
    """Mean hinge loss  mean(max(0, 1 - y * f(x)))  with y in {-1, +1}.

    This is the loss an SVM actually minimizes. Unlike accuracy it also
    penalizes points that are on the right side of the boundary but inside
    the margin, so it tells us how confident the model is, not just whether
    it is right. `decision` is SVC.decision_function (signed distance to the
    boundary). Labels 0/1 are mapped to -1/+1.
    """
    return float(hinge_loss(y_true, decision, labels=[0, 1]))


def svm_objective(clf, K_train, y_train):
    """The SVM cost function (primal objective) at the fitted solution.

        J = 1/2 ||w||^2  +  C * sum_i max(0, 1 - y_i f(x_i))
            ^ margin term      ^ total training hinge loss

    With a precomputed kernel w is never formed explicitly, but
    ||w||^2 = a^T K_SV a, where a = clf.dual_coef_ (alpha_i * y_i of the
    support vectors) and K_SV is the kernel between support vectors.
    Lower J = a wider margin and/or fewer margin violations. Values are
    only comparable between runs with the same C and n_train.
    """
    K_train = np.asarray(K_train)
    sv = clf.support_
    a = clf.dual_coef_.ravel()
    w_sq = float(a @ K_train[np.ix_(sv, sv)] @ a)
    y_pm = np.where(np.asarray(y_train) == 1, 1.0, -1.0)
    hinge = np.maximum(0.0, 1.0 - y_pm * clf.decision_function(K_train))
    return 0.5 * w_sq + clf.C * float(hinge.sum())


def split_metrics(prefix, y_true, y_pred, decision):
    """Accuracy, F1 and hinge loss for one split, with column prefix."""
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_hinge_loss": svm_hinge_loss(y_true, decision),
    }


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

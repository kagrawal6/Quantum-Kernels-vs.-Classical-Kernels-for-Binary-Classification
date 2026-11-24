"""Classical kernels — TEMPLATE for the classical-SVM owner (Kushal).

Fill in the two TODOs. The signatures are fixed: the shared runner
(src/evaluation/runner.py) calls them like this:

    K_train = kernel(X_train)            # XB=None  -> shape (n_train, n_train)
    K_test  = kernel(X_test, X_train)    # XB given -> shape (n_test,  n_train)

Inputs are already reduced and scaled to [0, pi] by src/data — do NOT
rescale here (all three models must see the same features).
C tuning, timing, metrics and the CSV row are all done by the runner;
these functions should ONLY build the matrix (everything inside them is timed).

Check your work:  python scripts/check_kernels.py
"""

import numpy as np


def linear_kernel(XA, XB=None):
    """K[i, j] = <XA[i], XB[j]>   (XB=None means XB = XA).

    TODO(classical):
      - use sklearn.metrics.pairwise.linear_kernel(XA, XB)
    """
    raise NotImplementedError("TODO(classical): linear_kernel")


def resolve_gamma(gamma, X_train):
    """Turn gamma="scale" into a number, exactly like sklearn's SVC does.

    Precomputed kernels bypass SVC's own handling of "scale", so we compute
    it ourselves: gamma = 1 / (n_features * X_train.var()).
    IMPORTANT: always computed from the TRAINING set, for both K_train and
    K_test, otherwise train and test kernels would use different gammas.
    """
    if gamma == "scale":
        return 1.0 / (X_train.shape[1] * X_train.var())
    return float(gamma)


def rbf_kernel(XA, XB=None, gamma="scale"):
    """K[i, j] = exp(-gamma * ||XA[i] - XB[j]||^2)   (XB=None means XB = XA).

    TODO(classical):
      - work out which array is the training set: XA if XB is None, else XB
      - g = resolve_gamma(gamma, <training set>)
      - use sklearn.metrics.pairwise.rbf_kernel(XA, XB, gamma=g)
    """
    raise NotImplementedError("TODO(classical): rbf_kernel")

"""Classical kernels: linear and RBF (owners: Jerry + Collin).

The signatures are fixed: the shared runner
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
from sklearn.metrics.pairwise import linear_kernel as _sk_linear
from sklearn.metrics.pairwise import rbf_kernel as _sk_rbf


def linear_kernel(XA, XB=None):
    """K[i, j] = <XA[i], XB[j]>   (XB=None means XB = XA)."""
    return _sk_linear(XA, XB)


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

    The training set is XA for K_train (XB=None) and XB for K_test, so
    "scale" gamma always comes from the training set.
    """
    X_train = XA if XB is None else XB
    g = resolve_gamma(gamma, np.asarray(X_train))
    return _sk_rbf(XA, XB, gamma=g)

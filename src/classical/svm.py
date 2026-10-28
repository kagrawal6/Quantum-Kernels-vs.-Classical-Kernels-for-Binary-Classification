"""Classical SVM baselines (linear + RBF).

Owner: Kushal Agrawal
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from sklearn.svm import SVC


def train_linear_svm(
    X_train,
    y_train,
    C: float = 1.0,
    **kwargs,
) -> SVC:
    """Train a linear-kernel SVM."""
    model = SVC(kernel="linear", C=C, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_rbf_svm(
    X_train,
    y_train,
    C: float = 1.0,
    gamma: Any = "scale",
    **kwargs,
) -> SVC:
    """Train an RBF-kernel SVM."""
    model = SVC(kernel="rbf", C=C, gamma=gamma, **kwargs)
    model.fit(X_train, y_train)
    return model


def get_default_params() -> Dict[str, Dict[str, Any]]:
    """Placeholder for grid-search / tuned defaults."""
    return {
        "linear": {"C": 1.0},
        "rbf": {"C": 1.0, "gamma": "scale"},
    }

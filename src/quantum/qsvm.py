"""Quantum Kernel SVM (classical SVM + quantum kernel matrix).

Owner: Shunzhe Yang (+ team help on Qiskit / hardware)
Reference: Havlíček et al. [3]
"""

from __future__ import annotations

from typing import Any, Optional

from sklearn.svm import SVC

from .kernel import compute_quantum_kernel


def train_qsvm(
    X_train,
    y_train,
    feature_map=None,
    shots: int = 1024,
    C: float = 1.0,
    backend_name: str = "aer_simulator",
    **kwargs,
) -> SVC:
    """Train an SVM using a precomputed quantum kernel."""
    K_train = compute_quantum_kernel(
        X_train,
        X_test=None,
        feature_map=feature_map,
        shots=shots,
        backend_name=backend_name,
    )
    model = SVC(kernel="precomputed", C=C, **kwargs)
    model.fit(K_train, y_train)
    # Attach training data so callers can build test kernels later.
    model.X_train_ = X_train
    model.feature_map_ = feature_map
    model.shots_ = shots
    model.backend_name_ = backend_name
    return model


def predict_qsvm(model: SVC, X_test) -> Any:
    """Predict with a quantum-kernel SVM (needs test-train kernel)."""
    K_test = compute_quantum_kernel(
        model.X_train_,
        X_test=X_test,
        feature_map=getattr(model, "feature_map_", None),
        shots=getattr(model, "shots_", 1024),
        backend_name=getattr(model, "backend_name_", "aer_simulator"),
    )
    # For sklearn precomputed kernel, predict expects shape (n_test, n_train).
    # compute_quantum_kernel should return that layout once implemented.
    return model.predict(K_test)

"""Quantum kernel estimation.

Owner: Shunzhe Yang
References:
  [3] Havlíček et al. — quantum kernel estimator (QKE)
  [4] Thanasilp et al. — watch for kernel concentration under noise / expressivity
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def compute_quantum_kernel(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    feature_map=None,
    shots: int = 1024,
    backend_name: str = "aer_simulator",
) -> np.ndarray:
    """Estimate a quantum kernel matrix.

    If X_test is None, returns the train-train Gram matrix.
    Otherwise returns the test-train kernel used for prediction.

    TODO:
      - Wire up Qiskit QuantumKernel / FidelityQuantumKernel
      - Support noisy simulator and (later) IBM hardware
      - Log / inspect kernel value spread (concentration check; see [4])
    """
    raise NotImplementedError(
        "Quantum kernel estimation not implemented yet. "
        "Use Qiskit Machine Learning fidelity/quantum kernel APIs."
    )


def kernel_diagnostics(K: np.ndarray) -> dict:
    """Simple checks for concentration / degeneracy (see Thanasilp et al. [4])."""
    off_diag = K[~np.eye(K.shape[0], dtype=bool)] if K.ndim == 2 else K.ravel()
    return {
        "mean": float(np.mean(off_diag)),
        "std": float(np.std(off_diag)),
        "min": float(np.min(off_diag)),
        "max": float(np.max(off_diag)),
    }

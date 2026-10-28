"""Quantum feature maps (Z / ZZ style).

Reference: Havlíček et al. [3] — quantum-enhanced feature spaces / QKE.
"""

from __future__ import annotations


def get_feature_map(name: str = "zz", n_qubits: int = 4, reps: int = 1):
    """Return a Qiskit feature map.

    name: "z" | "zz"
    """
    # Import lazily so classical-only work does not require Qiskit installed yet.
    from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap

    name = name.lower()
    if name == "z":
        return ZFeatureMap(feature_dimension=n_qubits, reps=reps)
    if name == "zz":
        return ZZFeatureMap(feature_dimension=n_qubits, reps=reps)
    if name == "pauli":
        return PauliFeatureMap(feature_dimension=n_qubits, reps=reps)
    raise ValueError(f"Unknown feature map: {name}")

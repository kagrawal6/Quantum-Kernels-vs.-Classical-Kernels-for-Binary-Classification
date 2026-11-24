"""Quantum (fidelity) kernel — TEMPLATE for the quantum-SVM owner (Jerry).

Fill in the TODOs. The signatures are fixed: the shared runner
(src/evaluation/runner.py) calls them like this:

    K_train = quantum_kernel(X_train, **params, seed=seed)           # (n_train, n_train)
    K_test  = quantum_kernel(X_test, X_train, **params, seed=seed)   # (n_test,  n_train)
    cost    = quantum_cost(n_features=..., n_train=..., n_test=..., **params, seed=seed)

params come from configs/default.yaml:
    feature_map  "z" | "zz"
    reps         1, 2, 3            (circuit depth / repetitions)
    entanglement "linear" | "full"
    bandwidth    0.1 ... 1.0        (multiply inputs by this before encoding)
    shots        None = exact statevector; int = simulate that many shots

Inputs are already reduced and scaled to [0, pi] by src/data, and
n_features = number of qubits (angle encoding).
C tuning, timing, metrics and the CSV row are all done by the runner;
quantum_kernel should ONLY build the matrix (everything inside it is timed).

Kernel definition (Havlicek et al. [3]):  K(x, x') = |<phi(x)|phi(x')>|^2,
so K is in [0, 1] and the diagonal is exactly 1.

Check your work:  python scripts/check_kernels.py
"""

import numpy as np


def build_feature_map(n_features, feature_map="zz", reps=2, entanglement="linear"):
    """Return the Qiskit feature-map circuit U_phi(x) with n_features qubits.

    TODO(quantum):
      - from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
      - "z"  -> ZFeatureMap(feature_dimension=n_features, reps=reps)
               (no entanglement: the Z map is a product state, so it is
               classically easy — a useful "is entanglement helping?" control)
      - "zz" -> ZZFeatureMap(feature_dimension=n_features, reps=reps,
                             entanglement=entanglement)
    """
    raise NotImplementedError("TODO(quantum): build_feature_map")


def add_shot_noise(K_exact, shots, seed, symmetric):
    """Simulate estimating each kernel entry from a finite number of shots.

    On hardware each entry is estimated as (# times we measure all-zeros) /
    shots, i.e. a Binomial(shots, K_exact) / shots sample. Drawing that
    directly is statistically the same as simulating the circuits with
    shots (noiseless), but costs nothing extra.

    For the training kernel (symmetric=True) only the upper triangle is
    sampled and mirrored, and the diagonal stays exactly 1 (K(x, x) = 1 is
    known without measuring, so no circuit is run for it).
    """
    rng = np.random.default_rng(seed)
    K = rng.binomial(shots, np.clip(K_exact, 0.0, 1.0)) / shots
    if symmetric:
        K = np.triu(K, 1)
        K = K + K.T + np.eye(K.shape[0])
    return K


def quantum_kernel(XA, XB=None, feature_map="zz", reps=2, entanglement="linear",
                   bandwidth=1.0, shots=None, seed=0):
    """Fidelity kernel matrix K[i, j] = |<phi(XA[i])|phi(XB[j])>|^2.

    TODO(quantum):
      1. fm = build_feature_map(XA.shape[1], feature_map, reps, entanglement)
      2. scale inputs by the bandwidth:  XA_b = bandwidth * XA  (same for XB)
      3. exact kernel with
            from qiskit_machine_learning.kernels import FidelityStatevectorKernel
            qk = FidelityStatevectorKernel(feature_map=fm)
            K  = qk.evaluate(x_vec=XA_b, y_vec=XB_b)   # y_vec=None for training
         (simulates one statevector per point, not per pair -> fast)
      4. if shots is not None:
            K = add_shot_noise(K, shots, seed, symmetric=(XB is None))
         Use a different seed for the test kernel than for the training kernel
         (e.g. seed + 1 when XB is not None) so their noise is independent.
      5. return K as a numpy array
    """
    raise NotImplementedError("TODO(quantum): quantum_kernel")


def quantum_cost(n_features, n_train, n_test, feature_map="zz", reps=2,
                 entanglement="linear", shots=None, **_):
    """Hardware-independent cost of one QSVM run (EXPERIMENTS.md Section 6).

    n_circuits / total_shots are what a REAL device would need with the
    compute-uncompute circuit U(x')^dagger U(x): one circuit per kernel entry
    (training kernel: only pairs i < j, the diagonal is 1 by definition).
    This is the number that sets the AWS budget, not simulator time.
    """
    n_circuits = n_train * (n_train - 1) // 2 + n_test * n_train
    cost = {
        "n_circuits": n_circuits,
        "total_shots": "" if shots is None else n_circuits * shots,
        "circuit_depth": "",
        "n_2q_gates": "",
    }
    # TODO(quantum): fill circuit_depth and n_2q_gates for ONE kernel circuit:
    #   fm = build_feature_map(n_features, feature_map, reps, entanglement)
    #   x, y = random parameter values (or zeros) of length n_features
    #   circ = fm.assign_parameters(x).compose(fm.assign_parameters(y).inverse())
    #   from qiskit import transpile
    #   t = transpile(circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    #   cost["circuit_depth"] = t.depth()
    #   cost["n_2q_gates"] = t.count_ops().get("cx", 0)
    return cost

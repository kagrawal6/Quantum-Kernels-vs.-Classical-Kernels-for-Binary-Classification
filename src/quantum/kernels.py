"""Quantum (fidelity) kernel for the QSVM.

The signatures are fixed: the shared runner
(src/evaluation/runner.py) calls them like this:

    K_train = quantum_kernel(X_train, **params, seed=seed)           # (n_train, n_train)
    K_test  = quantum_kernel(X_test, X_train, **params, seed=seed)   # (n_test,  n_train)
    cost    = quantum_cost(n_features=..., n_train=..., n_test=..., **params, seed=seed)

params come from configs/default.yaml:
    feature_map  "z" | "zz"
    reps         1, 2, 3            (circuit depth / repetitions)
    entanglement "linear" | "full"
    bandwidth    default 0.05; swept 0.025 ... 1.0  (multiply inputs by this
                 before encoding; see scripts/bandwidth_study.py)
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

    "z"  -> Z feature map: no entanglement, so the state is a product state
            and the kernel is classically easy to compute (our "is
            entanglement helping?" control). `entanglement` is ignored.
    "zz" -> ZZ feature map (Havlicek et al. [3]): adds two-qubit
            (pi - x_i)(pi - x_j) phases between the qubit pairs given by
            `entanglement`.

    Uses the function forms (z_feature_map / zz_feature_map); the
    ZFeatureMap / ZZFeatureMap classes are deprecated since Qiskit 2.1.
    """
    from qiskit.circuit.library import z_feature_map, zz_feature_map

    if feature_map == "z":
        return z_feature_map(n_features, reps=reps)
    if feature_map == "zz":
        return zz_feature_map(n_features, reps=reps, entanglement=entanglement)
    raise ValueError(f"feature_map must be 'z' or 'zz', got {feature_map!r}")


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

    XB=None -> training kernel (n, n); otherwise test kernel (m, n).
    Inputs are multiplied by `bandwidth` before encoding. The exact kernel
    is computed from statevectors (one simulation per point, not per pair);
    with `shots`, binomial shot noise is added on top of the exact values.
    """
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    XA = np.asarray(XA, dtype=float)
    fm = build_feature_map(XA.shape[1], feature_map, reps, entanglement)
    # The exact fidelity kernel is a Gram matrix, so it is already PSD;
    # skipping enforce_psd keeps an extra eigendecomposition out of the timing.
    qk = FidelityStatevectorKernel(feature_map=fm, enforce_psd=False)
    XB_b = None if XB is None else bandwidth * np.asarray(XB, dtype=float)
    K = np.asarray(qk.evaluate(x_vec=bandwidth * XA, y_vec=XB_b), dtype=float)

    if shots is not None:
        symmetric = XB is None
        # Separate stream for the test kernel: independent of K_train, and unlike
        # seed + 1 it cannot coincide with the next run's training-kernel noise.
        K = add_shot_noise(K, int(shots), seed if symmetric else [seed, 1], symmetric)
    return K


def quantum_cost(n_features, n_train, n_test, feature_map="zz", reps=2,
                 entanglement="linear", shots=None, **_):
    """Hardware-independent cost of one QSVM run (EXPERIMENTS.md Section 6).

    n_circuits / total_shots are what a REAL device would need with the
    compute-uncompute circuit U(x')^dagger U(x): one circuit per kernel entry
    (training kernel: only pairs i < j, the diagonal is 1 by definition).
    This is the number that sets the AWS budget, not simulator time.
    """
    from qiskit import transpile

    n_circuits = n_train * (n_train - 1) // 2 + n_test * n_train
    # One compute-uncompute kernel circuit U(x')^dagger U(x). Random angles,
    # not zeros: zero angles let the transpiler delete gates and undercount.
    fm = build_feature_map(n_features, feature_map, reps, entanglement)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0.1, 3.0, size=(2, n_features))
    circ = fm.assign_parameters(x).compose(fm.assign_parameters(y).inverse())
    t = transpile(circ, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    return {
        "n_circuits": n_circuits,
        "total_shots": "" if shots is None else n_circuits * shots,
        "circuit_depth": t.depth(),
        "n_2q_gates": t.count_ops().get("cx", 0),
    }

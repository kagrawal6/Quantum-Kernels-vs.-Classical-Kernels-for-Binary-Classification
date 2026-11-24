"""Check the kernel implementations before running experiments.

Usage:  python scripts/check_kernels.py            # all kernels
        python scripts/check_kernels.py classical  # just linear + RBF
        python scripts/check_kernels.py quantum

Kernels that are still TODO (NotImplementedError) are reported and skipped.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical.kernels import linear_kernel, rbf_kernel  # noqa: E402
from src.data import get_data  # noqa: E402
from src.quantum.kernels import (  # noqa: E402
    build_feature_map, quantum_cost, quantum_kernel)

N_TRAIN, N_TEST = 40, 20


def basic_checks(name, kfn, Xtr, Xte, unit_diag, **kw):
    Ktr, Kte = kfn(Xtr, **kw), kfn(Xte, Xtr, **kw)
    Ktr, Kte = np.asarray(Ktr), np.asarray(Kte)
    assert Ktr.shape == (len(Xtr), len(Xtr)), f"{name}: K_train shape {Ktr.shape}"
    assert Kte.shape == (len(Xte), len(Xtr)), f"{name}: K_test shape {Kte.shape}"
    assert np.allclose(Ktr, Ktr.T, atol=1e-8), f"{name}: K_train not symmetric"
    min_eig = np.linalg.eigvalsh(Ktr).min()
    assert min_eig > -1e-6, f"{name}: K_train not PSD (min eigenvalue {min_eig:.2e})"
    if unit_diag:
        assert np.allclose(np.diag(Ktr), 1, atol=1e-8), f"{name}: diagonal != 1"
        assert Ktr.min() >= -1e-9 and Ktr.max() <= 1 + 1e-9, f"{name}: values outside [0, 1]"
    return Ktr, Kte


def check_classical(Xtr, Xte, ytr):
    Ktr, Kte = basic_checks("linear", linear_kernel, Xtr, Xte, unit_diag=False)
    ref = SVC(kernel="linear", C=1.0).fit(Xtr, ytr).predict(Xte)
    ours = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr).predict(Kte)
    assert np.array_equal(ref, ours), "linear: predictions differ from SVC(kernel='linear')"
    print("  linear: ok (matches SVC(kernel='linear'))")

    Ktr, Kte = basic_checks("rbf", rbf_kernel, Xtr, Xte, unit_diag=True, gamma="scale")
    ref = SVC(kernel="rbf", gamma="scale", C=1.0).fit(Xtr, ytr).predict(Xte)
    ours = SVC(kernel="precomputed", C=1.0).fit(Ktr, ytr).predict(Kte)
    assert np.array_equal(ref, ours), ("rbf: predictions differ from SVC(kernel='rbf', "
                                       "gamma='scale') - is gamma computed from the TRAINING set?")
    print("  rbf: ok (matches SVC(kernel='rbf', gamma='scale'))")


def check_quantum(Xtr, Xte):
    from qiskit.quantum_info import Statevector

    for fm_name, ent in [("z", "linear"), ("zz", "linear"), ("zz", "full")]:
        kw = dict(feature_map=fm_name, reps=2, entanglement=ent, bandwidth=0.5)
        Ktr, Kte = basic_checks(f"qsvm[{fm_name},{ent}]", quantum_kernel, Xtr, Xte,
                                unit_diag=True, **kw)
        # Independent reference: overlap of statevectors, computed directly.
        fm = build_feature_map(Xtr.shape[1], fm_name, 2, ent)
        sv = lambda x: Statevector(fm.assign_parameters(0.5 * x)).data  # noqa: E731
        for i, j in [(0, 1), (2, 5)]:
            ref = abs(np.vdot(sv(Xte[i]), sv(Xtr[j]))) ** 2
            assert np.isclose(Kte[i, j], ref, atol=1e-8), (
                f"qsvm[{fm_name},{ent}]: K_test[{i},{j}]={Kte[i, j]:.6f} but direct "
                f"overlap is {ref:.6f} - is the bandwidth applied?")
        print(f"  qsvm {fm_name}/{ent}: ok (matches direct statevector overlaps)")

    # Shot noise: unbiased, shrinks with more shots.
    Kex = quantum_kernel(Xtr, bandwidth=0.5)
    errs = []
    for shots in (256, 4096):
        Ksh = quantum_kernel(Xtr, bandwidth=0.5, shots=shots, seed=0)
        assert np.allclose(np.diag(Ksh), 1) and np.allclose(Ksh, Ksh.T)
        errs.append(np.abs(Ksh - Kex).mean())
    assert errs[1] < errs[0], "shot noise should shrink with more shots"
    print(f"  shots: ok (mean |error| {errs[0]:.4f} @256 -> {errs[1]:.4f} @4096)")

    cost = quantum_cost(n_features=Xtr.shape[1], n_train=100, n_test=100, shots=1024)
    assert cost["circuit_depth"] != "" and cost["n_2q_gates"] != "", "quantum_cost: depth / 2q gates still TODO"
    print(f"  cost: ok {cost}")


def run(name, fn, *args):
    try:
        fn(*args)
    except NotImplementedError as e:
        print(f"  {name}: SKIPPED - {e}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    Xtr, Xte, ytr, _ = get_data("breast_cancer", 4, N_TRAIN, 0, n_test=N_TEST)
    if which in ("all", "classical"):
        print("classical kernels")
        run("classical", check_classical, Xtr, Xte, ytr)
    if which in ("all", "quantum"):
        print("quantum kernel")
        run("quantum", check_quantum, Xtr, Xte)


if __name__ == "__main__":
    main()

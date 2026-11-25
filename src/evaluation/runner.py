"""Run ONE setting end to end and turn it into ONE CSV row.

    data -> K_train, K_test -> tune C (5-fold CV) -> fit -> predict -> metrics

The same code path is used for linear, RBF and quantum kernels, so tuning,
timing and metrics are identical for all three models by construction.

Timed (EXPERIMENTS.md Section 2), with time.perf_counter():
    kernel_train_time  building K_train
    kernel_test_time   building K_test
    fit_time           SVC.fit with the chosen C
    predict_time       SVC.predict
NOT timed: loading / PCA / scaling, and the C search (that is tuning).

Train / validation / test (each run reports all three):
    train       metrics of the final SVM on the n_train points it was fit on
    validation  mean over the 5 CV folds at the chosen C (held-out folds of
                the training set; this is what picks C)
    test        the fixed test set, never used for any choice
Comparing them shows over-fitting (train >> validation/test) and whether
the validation score used for tuning predicts the test score.

Loss / cost: hinge loss (what an SVM minimizes) on every split, plus the
SVM primal objective  1/2||w||^2 + C * sum(hinge)  on the training set.

Memory (measure_memory=True): the two kernel builds are repeated once more,
untimed, under tracemalloc to get peak_memory_mb, so tracing overhead never
leaks into the timings. kernel_matrix_mb / statevector_mb are the
analytic sizes of the n^2 kernel matrices and the 2^qubits statevectors.
"""

import csv
import os
import platform
import time
import tracemalloc
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from src.data import get_data
from src.evaluation.metrics import (classification_metrics, kernel_diagnostics,
                                    split_metrics, svm_hinge_loss, svm_objective)
from src.evaluation.models import MODELS

# EXPERIMENTS.md Section 7, in order. Every CSV row has exactly these columns.
COLUMNS = [
    # run settings
    "run_id", "timestamp", "sweep", "dataset", "model", "seed",
    "n_features", "n_train", "n_test",
    # model parameters
    "C", "gamma", "feature_map", "reps", "entanglement", "bandwidth",
    "shots", "backend",
    # quality (test set)
    "accuracy", "precision", "recall", "f1",
    # train / validation / test comparison and loss (hinge = SVM loss)
    "train_accuracy", "train_f1", "train_hinge_loss",
    "val_accuracy", "val_f1", "val_hinge_loss",
    "test_hinge_loss", "svm_objective", "n_support_vectors",
    # time
    "kernel_train_time", "kernel_test_time", "fit_time", "predict_time",
    "total_time",
    # quantum cost
    "n_circuits", "total_shots", "circuit_depth", "n_2q_gates",
    # kernel diagnostics
    "kernel_offdiag_mean", "kernel_offdiag_std", "kernel_offdiag_min",
    "kernel_offdiag_max",
    # memory
    "peak_memory_mb", "kernel_matrix_mb", "statevector_mb",
]

# Columns that identify a setting. Two rows with the same values here are the
# same run, which lets an interrupted sweep resume without redoing work.
KEY_COLUMNS = [
    "sweep", "dataset", "model", "seed", "n_features", "n_train", "n_test",
    "gamma", "feature_map", "reps", "entanglement", "bandwidth", "shots",
]


def cv_search(K_train, y_train, C_grid, cv_folds, seed):
    """Pick C by stratified k-fold CV on the precomputed training kernel.

    GridSearchCV knows SVC(kernel="precomputed") takes a kernel matrix, so
    for each fold it slices K_train[fold_train][:, fold_train] (fit) and
    K_train[fold_val][:, fold_train] (score) correctly. The kernel is
    computed once and reused for every C; the test set is never touched.
    C is chosen by F1, our main metric; validation accuracy and hinge loss
    at that C are recorded too.

    Returns (C, {"val_f1": ..., "val_accuracy": ..., "val_hinge_loss": ...}).
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = GridSearchCV(
        SVC(kernel="precomputed"),
        {"C": list(C_grid)},
        cv=cv,
        scoring={
            "f1": make_scorer(f1_score, zero_division=0),
            "accuracy": "accuracy",
            "hinge": make_scorer(svm_hinge_loss, greater_is_better=False,
                                 response_method="decision_function"),
        },
        refit="f1",
    )
    search.fit(K_train, y_train)
    i, r = search.best_index_, search.cv_results_
    val = {
        "val_f1": r["mean_test_f1"][i],
        "val_accuracy": r["mean_test_accuracy"][i],
        "val_hinge_loss": -r["mean_test_hinge"][i],  # scorer is negated
    }
    return search.best_params_["C"], val


def tune_C(K_train, y_train, C_grid, cv_folds, seed):
    """Just the chosen C (see cv_search). Used by the bandwidth scripts."""
    return cv_search(K_train, y_train, C_grid, cv_folds, seed)[0]


def peak_kernel_memory_mb(kernel, X_train, X_test, kparams):
    """Peak Python-heap memory (MB) while building K_train and K_test.

    Uses tracemalloc, which sees NumPy arrays (kernel matrices,
    statevectors) but not allocations made inside native C++/Rust code.
    Run separately from the timed builds because tracing slows Python down.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    K_train = kernel(X_train, **kparams)
    K_test = kernel(X_test, X_train, **kparams)
    peak = tracemalloc.get_traced_memory()[1]
    del K_train, K_test
    if not was_tracing:
        tracemalloc.stop()
    return (peak - base) / 1e6


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def run_one(dataset, model, seed, n_features, n_train, n_test, params,
            C_grid, cv_folds, sweep="adhoc", data=None, measure_memory=True):
    """Run one (dataset, model, setting, seed) and return the CSV row (dict).

    params: kernel parameters for this run, e.g. {"gamma": "scale"} or
            {"feature_map": "zz", "reps": 2, ...}. Keys the model does not
            use are ignored. shots=None means exact (statevector).
    data:   optional precomputed (X_train, X_test, y_train, y_test).
    measure_memory: also record peak_memory_mb (one extra, untimed kernel build).
    """
    spec = MODELS[model]
    if data is None:
        data = get_data(dataset, n_features, n_train, seed, n_test=n_test)
    X_train, X_test, y_train, y_test = data

    kparams = {k: params[k] for k in spec["params"] if k in params}
    if spec["seeded"]:
        kparams["seed"] = seed

    # 1-2. kernel matrices (timed separately)
    K_train, t_ktrain = _timed(spec["kernel"], X_train, **kparams)
    K_test, t_ktest = _timed(spec["kernel"], X_test, X_train, **kparams)
    _check_shapes(model, K_train, K_test, len(X_train), len(X_test))

    # 3. choose C (not timed), then 4-5. fit and predict (timed)
    C, val = cv_search(K_train, y_train, C_grid, cv_folds, seed)
    clf = SVC(kernel="precomputed", C=C)
    _, t_fit = _timed(clf.fit, K_train, y_train)
    y_pred, t_pred = _timed(clf.predict, K_test)

    # Not timed: train-set metrics, losses, objective, memory.
    train = split_metrics("train", y_train, clf.predict(K_train),
                          clf.decision_function(K_train))
    n_tr, n_te = len(y_train), len(y_test)
    extra = {
        **train, **val,
        "test_hinge_loss": svm_hinge_loss(y_test, clf.decision_function(K_test)),
        "svm_objective": svm_objective(clf, K_train, y_train),
        "n_support_vectors": int(len(clf.support_)),
        "kernel_matrix_mb": (n_tr * n_tr + n_te * n_tr) * 8 / 1e6,
    }
    if model == "qsvm":
        # One complex128 statevector of 2^qubits amplitudes per data point.
        extra["statevector_mb"] = (n_tr + n_te) * (2 ** n_features) * 16 / 1e6
    if measure_memory:
        extra["peak_memory_mb"] = peak_kernel_memory_mb(
            spec["kernel"], X_train, X_test, kparams)

    row = {c: "" for c in COLUMNS}
    row.update({
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "sweep": sweep, "dataset": dataset, "model": model, "seed": seed,
        "n_features": n_features, "n_train": len(y_train),
        "n_test": len(y_test), "C": C,
        "kernel_train_time": t_ktrain, "kernel_test_time": t_ktest,
        "fit_time": t_fit, "predict_time": t_pred,
        "total_time": t_ktrain + t_ktest + t_fit + t_pred,
    })
    for k in spec["params"]:
        if k in params and params[k] is not None:
            row[k] = params[k]
    if model == "qsvm":
        row["backend"] = "statevector" if params.get("shots") is None else "shots"
    row.update(classification_metrics(y_test, y_pred))
    row.update(extra)
    row.update(kernel_diagnostics(K_train))
    if spec["cost"] is not None:
        row.update(spec["cost"](n_features=n_features, n_train=len(y_train),
                                n_test=len(y_test), **kparams))
    return row


def _check_shapes(model, K_train, K_test, n_train, n_test):
    if np.shape(K_train) != (n_train, n_train):
        raise ValueError(f"{model}: K_train has shape {np.shape(K_train)}, "
                         f"expected {(n_train, n_train)}")
    if np.shape(K_test) != (n_test, n_train):
        raise ValueError(f"{model}: K_test has shape {np.shape(K_test)}, "
                         f"expected {(n_test, n_train)}")


# ---------------------------------------------------------------- CSV I/O

def row_key(row):
    """String key identifying a setting (see KEY_COLUMNS)."""
    return "|".join(_fmt(row.get(c, "")) for c in KEY_COLUMNS)


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(v)  # keep 1.0 as "1.0" so it matches what csv wrote
    return str(v)


def existing_keys(csv_path):
    """Keys of runs already in a results CSV (empty set if no file)."""
    path = Path(csv_path)
    if not path.exists():
        return set()
    with open(path, newline="") as f:
        return {row_key(r) for r in csv.DictReader(f)}


def append_row(csv_path, row):
    """Append one row, writing the header first if the file is new.

    If the file was written with an older column list, it is first rewritten
    with the current COLUMNS (old rows keep their values; new columns are
    left empty), so every row in a file always lines up with the header.
    """
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _upgrade_header(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({c: _fmt(row.get(c, "")) for c in COLUMNS})


def _upgrade_header(path):
    if not path.exists():
        return
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == COLUMNS:
            return
        rows = list(reader)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def write_machine_info(path):
    """Record the machine used for timing (EXPERIMENTS.md ground rule 8)."""
    import sklearn
    lines = [
        f"date: {datetime.now().isoformat(timespec='seconds')}",
        f"os: {platform.platform()}",
        f"cpu: {platform.processor() or platform.machine()}",
        f"cpu_count: {os.cpu_count()}",
        f"python: {platform.python_version()}",
        f"numpy: {np.__version__}",
        f"scikit-learn: {sklearn.__version__}",
    ]
    try:
        lines.append(f"ram_gb: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9:.1f}")
    except (ValueError, OSError, AttributeError):
        pass
    for mod in ("qiskit", "qiskit_machine_learning", "qiskit_aer"):
        try:
            lines.append(f"{mod}: {__import__(mod).__version__}")
        except ImportError:
            lines.append(f"{mod}: not installed")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")

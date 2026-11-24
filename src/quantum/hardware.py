"""Estimate the quantum kernel on real hardware (AWS Braket) WITHOUT overspending.

How a kernel entry is measured on hardware (compute-uncompute, Havlicek [3]):
    circuit = U(x_i) followed by U(x_j)^dagger, then measure every qubit
    K[i, j] = fraction of shots that return all zeros  (estimates |<phi(x_j)|phi(x_i)>|^2)
One circuit per entry. The training kernel is symmetric with K[i, i] = 1, so
only pairs i < j are run. Braket bills every circuit as one "task":

    cost = n_circuits * (task_fee + shots * price_per_shot)

Money safety (all in this file):
  * estimate_cost()   - computed from the config BEFORE anything is sent
  * check_budget()    - refuses if this run or the total spent would pass a cap
  * Ledger            - every batch is written to the spend ledger BEFORE it is
                        submitted (pessimistic: a crash still counts as spent)
  * EntryStore        - every measured entry and every submitted job id is saved
                        immediately, so a restart re-uses results / re-attaches to
                        jobs instead of paying for the same circuits twice
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from src.quantum.kernels import build_feature_map

# ------------------------------------------------------------------ circuits


def kernel_circuit(fm, xi, xj, bandwidth):
    """U(bandwidth*xi) then U(bandwidth*xj)^dagger, measured on all qubits."""
    circ = fm.assign_parameters(bandwidth * np.asarray(xi)).compose(
        fm.assign_parameters(bandwidth * np.asarray(xj)).inverse())
    circ.measure_all()
    return circ


def entry_list(n_train, n_test):
    """All kernel entries a device must measure, as (kind, i, j)."""
    train = [("train", i, j) for i in range(n_train) for j in range(i + 1, n_train)]
    test = [("test", i, j) for i in range(n_test) for j in range(n_train)]
    return train + test


def zero_fraction(counts, n_qubits, shots):
    """Fraction of shots that measured |00...0>."""
    return counts.get("0" * n_qubits, 0) / shots


def assemble(entries, n_train, n_test):
    """Build K_train (diag = 1, mirrored) and K_test from measured entries."""
    K_train, K_test = np.eye(n_train), np.zeros((n_test, n_train))
    for key, val in entries.items():
        kind, i, j = key.split(":")
        i, j = int(i), int(j)
        if kind == "train":
            K_train[i, j] = K_train[j, i] = val
        else:
            K_test[i, j] = val
    return K_train, K_test


# ---------------------------------------------------------------- budget


def device_price(hw_cfg):
    dev = hw_cfg["device"]
    prices = hw_cfg["prices"]
    if dev not in prices:
        raise ValueError(f"No price for device {dev!r} in config hardware.prices; "
                         "add it (from aws.amazon.com/braket/pricing) before running.")
    return prices[dev]["task_fee"], prices[dev]["per_shot"]


def estimate_cost(n_circuits, shots, hw_cfg):
    task_fee, per_shot = device_price(hw_cfg)
    return n_circuits * (task_fee + shots * per_shot)


class Ledger:
    """Append-only CSV of estimated spend. results/ is git-ignored, so the
    team should let ONE person submit (or commit this file with git add -f)."""

    FIELDS = ["timestamp", "run_name", "device", "n_circuits", "shots", "est_cost_usd", "note"]

    def __init__(self, path):
        self.path = Path(path)

    def total(self):
        if not self.path.exists():
            return 0.0
        with open(self.path, newline="") as f:
            return sum(float(r["est_cost_usd"]) for r in csv.DictReader(f))

    def record(self, run_name, device, n_circuits, shots, cost, note=""):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            if new:
                w.writeheader()
            w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"),
                        "run_name": run_name, "device": device, "n_circuits": n_circuits,
                        "shots": shots, "est_cost_usd": f"{cost:.4f}", "note": note})


def check_budget(cost, ledger, hw_cfg):
    """Raise if this run (with safety margin) would break either cap."""
    margin = hw_cfg["safety_margin"]
    spent = ledger.total()
    problems = []
    if cost * margin > hw_cfg["max_run_usd"]:
        problems.append(f"this run ${cost:.2f} x margin {margin} > max_run_usd ${hw_cfg['max_run_usd']}")
    if spent + cost * margin > hw_cfg["max_total_usd"]:
        problems.append(f"already spent ${spent:.2f} + this run ${cost:.2f} x margin {margin} "
                        f"> max_total_usd ${hw_cfg['max_total_usd']}")
    if problems:
        raise RuntimeError("BUDGET GUARD - refusing to submit:\n  " + "\n  ".join(problems)
                           + "\nReduce n_train / n_test / shots, or pick a cheaper device.")
    return spent


# ---------------------------------------------------------------- storage


class EntryStore:
    """Measured entries + pending job ids for one run, saved after every batch."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = (json.loads(self.path.read_text()) if self.path.exists()
                     else {"entries": {}, "pending": []})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))

    @property
    def entries(self):
        return self.data["entries"]


# ---------------------------------------------------------------- backends


def get_backend(kind, device=None):
    """'aer'          - free local Qiskit Aer simulator (with shots)
       'braket_local' - free local Braket simulator (tests the Braket path)
       'braket'       - REAL device on AWS (costs money)"""
    if kind == "aer":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    from qiskit_braket_provider import BraketLocalBackend, BraketProvider
    if kind == "braket_local":
        return BraketLocalBackend()
    if kind == "braket":
        return BraketProvider().get_backend(device)
    raise ValueError(kind)


def measure_entries(X_train, X_test, hw_cfg, backend, backend_kind, store, ledger,
                    run_name, log=print):
    """Measure every missing kernel entry in batches, saving after each batch."""
    from qiskit import transpile

    n_q = X_train.shape[1]
    fm = build_feature_map(n_q, hw_cfg["feature_map"], hw_cfg["reps"], hw_cfg["entanglement"])
    bw, shots, batch = hw_cfg["bandwidth"], hw_cfg["shots"], hw_cfg["batch_size"]
    paid = backend_kind == "braket"
    X = {"train": X_train, "test": X_test}

    def finish(keys, result):
        for k, key in enumerate(keys):
            store.entries[key] = zero_fraction(result.get_counts(k), n_q, shots)

    # 1. Re-attach to jobs submitted before a crash (no new spending).
    for pend in list(store.data["pending"]):
        log(f"  re-attaching to job {pend['job_id'][:40]}... ({len(pend['keys'])} circuits)")
        finish(pend["keys"], backend.retrieve_job(pend["job_id"]).result())
        store.data["pending"].remove(pend)
        store.save()

    # 2. Submit what is still missing.
    all_keys = [f"{k}:{i}:{j}" for k, i, j in entry_list(len(X_train), len(X_test))]
    n_total = len(all_keys)
    todo = [key for key in all_keys if key not in store.entries]
    for start in range(0, len(todo), batch):
        keys = todo[start:start + batch]
        circs = []
        for key in keys:
            kind, i, j = key.split(":")
            # train entry (i, j): x_i vs x_j from the training set
            # test entry  (i, j): test x_i vs training x_j
            circs.append(kernel_circuit(fm, X[kind][int(i)], X_train[int(j)], bw))
        circs = transpile(circs, backend=backend, optimization_level=3)
        if paid:
            cost = estimate_cost(len(circs), shots, hw_cfg)
            ledger.record(run_name, hw_cfg["device"], len(circs), shots, cost)  # BEFORE submitting
        job = backend.run(circs, shots=shots)
        if paid:
            store.data["pending"].append({"job_id": job.job_id(), "keys": keys})
            store.save()
        finish(keys, job.result())
        if paid:
            store.data["pending"] = [p for p in store.data["pending"] if p["keys"] != keys]
        store.save()
        log(f"  measured {len(store.entries)}/{n_total} entries")
    return assemble(store.entries, len(X_train), len(X_test))

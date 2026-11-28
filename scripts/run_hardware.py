"""Small QSVM run on real quantum hardware (AWS Braket), with hard spending caps.

Settings: `hardware:` section of configs/default.yaml.

    python scripts/run_hardware.py --list-devices       # 0. free: which devices exist and are online
    python scripts/run_hardware.py                      # 1. PLAN: cost estimate only, spends nothing
    python scripts/run_hardware.py --local aer          # 2. free: full pipeline on a local simulator
    python scripts/run_hardware.py --local braket_local # 3. free: same, through the Braket SDK
    python scripts/run_hardware.py --smoke-test         # 4. ONE circuit on the device (~$0.39)
    python scripts/run_hardware.py --submit             # 5. the real run (asks you to confirm the $)

Money safety:
  * nothing is ever sent to AWS without --smoke-test / --submit AND typing the amount
  * refuses if the run (x safety_margin) > max_run_usd, or total spent would pass max_total_usd
  * spend is logged to results/hardware/spend_ledger.csv BEFORE each batch is submitted
  * progress + job ids are saved after each batch: re-running after a crash re-attaches
    to submitted jobs and only pays for circuits that were never sent
  * ALSO set an AWS Budgets alert in the console - the ledger only knows about this script

Output: results/hardware/<run>/  entries.json (raw), kernels.npz, summary.csv
The summary compares, on the SAME tiny dataset: exact simulation, simulated shot noise,
and the device - so hardware noise can be separated from finite-shot noise.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_data  # noqa: E402
from src.evaluation.metrics import classification_metrics, kernel_diagnostics  # noqa: E402
from src.quantum.hardware import (  # noqa: E402
    EntryStore, Ledger, check_budget, entry_list, estimate_cost, get_backend,
    kernel_circuit, measure_entries, zero_fraction)
from src.quantum.kernels import add_shot_noise, build_feature_map, quantum_kernel  # noqa: E402

HW_DIR = ROOT / "results" / "hardware"


def run_name(hw, backend_kind):
    where = hw["device"] if backend_kind == "braket" else backend_kind
    return (f"{where}_{hw['dataset']}_f{hw['n_features']}_n{hw['n_train']}_t{hw['n_test']}"
            f"_s{hw['shots']}_{hw['feature_map']}{hw['reps']}{hw['entanglement'][0]}_bw{hw['bandwidth']}")


def print_plan(hw, n_circuits, ledger):
    print(f"Plan: {hw['dataset']}, {hw['n_features']} qubits, n_train={hw['n_train']}, "
          f"n_test={hw['n_test']}, {hw['shots']} shots")
    print(f"  circuits = {hw['n_train']}*{hw['n_train'] - 1}/2 + {hw['n_test']}*{hw['n_train']} "
          f"= {n_circuits}")
    print("  estimated cost by device:")
    for dev in hw["prices"]:
        c = estimate_cost(n_circuits, hw["shots"], {**hw, "device": dev})
        mark = " <- selected" if dev == hw["device"] else ""
        print(f"    {dev:16s} ${c:8.2f}{mark}")
    cost = estimate_cost(n_circuits, hw["shots"], hw)
    spent = ledger.total()
    print(f"  caps: run ${cost:.2f} x {hw['safety_margin']} = ${cost * hw['safety_margin']:.2f} "
          f"(max ${hw['max_run_usd']}); spent so far ${spent:.2f} (max total ${hw['max_total_usd']})")
    try:
        check_budget(cost, ledger, hw)
        print("  budget check: OK")
    except RuntimeError as e:
        print("  " + str(e).replace("\n", "\n  "))
    return cost


def confirm(cost):
    answer = input(f"\nThis will spend about ${cost:.2f} of real AWS credit. "
                   f"Type the amount ({cost:.2f}) to continue: ").strip().lstrip("$")
    if answer != f"{cost:.2f}":
        sys.exit("Not confirmed - nothing was submitted.")


def smoke_test(hw, data, ledger):
    """One circuit on the device: checks credentials, device name and parsing."""
    from qiskit import transpile
    cost = estimate_cost(1, hw["shots"], hw)
    check_budget(cost, ledger, hw)
    confirm(cost)
    Xtr = data[0]
    fm = build_feature_map(Xtr.shape[1], hw["feature_map"], hw["reps"], hw["entanglement"])
    backend = get_backend("braket", hw["device"])
    circ = transpile(kernel_circuit(fm, Xtr[0], Xtr[1], hw["bandwidth"]), backend=backend,
                     optimization_level=3)
    ledger.record("smoke_test", hw["device"], 1, hw["shots"], cost, "smoke test")
    print("  submitted; waiting for the device queue (can take minutes to hours)...")
    counts = backend.run(circ, shots=hw["shots"]).result().get_counts()
    exact = quantum_kernel(Xtr[:2], feature_map=hw["feature_map"], reps=hw["reps"],
                           entanglement=hw["entanglement"], bandwidth=hw["bandwidth"])[0, 1]
    print(f"  K[0,1]: device {zero_fraction(counts, Xtr.shape[1], hw['shots']):.3f}  "
          f"vs exact simulation {exact:.3f}")


def compare(hw, data, K_measured, label, out_dir):
    """Evaluate exact / shot-noise / measured kernels with the same SVC and data."""
    Xtr, Xte, ytr, yte = data
    qp = dict(feature_map=hw["feature_map"], reps=hw["reps"],
              entanglement=hw["entanglement"], bandwidth=hw["bandwidth"])
    kernels = {
        "exact_sim": (quantum_kernel(Xtr, **qp), quantum_kernel(Xte, Xtr, **qp)),
        f"shot_sim_{hw['shots']}": (quantum_kernel(Xtr, **qp, shots=hw["shots"], seed=hw["seed"]),
                                    quantum_kernel(Xte, Xtr, **qp, shots=hw["shots"], seed=hw["seed"])),
        label: K_measured,
    }
    try:  # classical reference, if the classical kernels are implemented
        from src.classical.kernels import rbf_kernel
        kernels["rbf_classical"] = (rbf_kernel(Xtr), rbf_kernel(Xte, Xtr))
    except NotImplementedError:
        pass

    Kex_tr, Kex_te = kernels["exact_sim"]
    pred_exact = SVC(kernel="precomputed", C=hw["C"]).fit(Kex_tr, ytr).predict(Kex_te)
    rows = []
    for name, (Ktr, Kte) in kernels.items():
        pred = SVC(kernel="precomputed", C=hw["C"]).fit(Ktr, ytr).predict(Kte)
        rows.append({"kernel": name, **classification_metrics(yte, pred),
                     "agree_with_exact": np.mean(pred == pred_exact),
                     "kernel_mae_vs_exact": (np.abs(Ktr - Kex_tr).sum() + np.abs(Kte - Kex_te).sum())
                     / (Ktr.size - len(Ktr) + Kte.size) if "classical" not in name else np.nan,
                     **kernel_diagnostics(Ktr)})
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary.csv", index=False)
    np.savez(out_dir / "kernels.npz", **{f"{k}_{s}": v for k, (a, b) in kernels.items()
                                         for s, v in (("train", a), ("test", b))},
             y_train=ytr, y_test=yte)
    cols = ["kernel", "accuracy", "f1", "agree_with_exact", "kernel_mae_vs_exact",
            "kernel_offdiag_mean", "kernel_offdiag_std"]
    print("\n" + df[cols].to_string(index=False, float_format="{:.3f}".format))
    print(f"\n(test set is only {len(yte)} points: compare the KERNELS (mae, std) more than accuracy)")
    plot_kernels(kernels, label, hw, out_dir)
    print(f"Saved to {out_dir.relative_to(ROOT)}/  (summary.csv, kernels.npz, kernel_vs_exact.png)")


def plot_kernels(kernels, label, hw, out_dir):
    """Scatter of every measured kernel entry against its exact simulated value."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def entries(name):
        Ktr, Kte = kernels[name]
        return np.concatenate([np.asarray(Ktr)[np.triu_indices(len(Ktr), 1)], np.asarray(Kte).ravel()])

    exact = entries("exact_sim")
    shot = f"shot_sim_{hw['shots']}"
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color="#8a8984", lw=1, ls="--", label="perfect agreement")
    for name, lab, color, mk in [(shot, f"simulated shot noise only ({hw['shots']} shots)", "#2a78d6", "o"),
                                 (label, "measured on " + label.replace("measured_", ""), "#eb6834", "s")]:
        mae = np.abs(entries(name) - exact).mean()
        ax.scatter(exact, entries(name), s=26, color=color, marker=mk, edgecolors="#fcfcfb",
                   linewidths=0.6, label=f"{lab}: mean error {mae:.3f}", zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="exact kernel value (statevector simulation)",
           ylabel="estimated kernel value")
    ax.set_title("Each dot = one kernel entry (one circuit)", loc="left", fontsize=11, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3de", lw=0.8)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    fig.text(0.01, -0.02, f"{hw['dataset']}, {hw['n_features']} qubits, {hw['n_train']} train + "
             f"{hw['n_test']} test points, {hw['feature_map'].upper()} map, bandwidth {hw['bandwidth']}. "
             "Spread beyond the blue dots = hardware noise.", fontsize=8, color="#52514e")
    fig.savefig(out_dir / "kernel_vs_exact.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def list_devices():
    """Free: list AWS Braket quantum devices, their status and qubit counts."""
    from braket.aws import AwsDevice
    print(f"{'name':18s} {'provider':12s} {'status':10s} {'qubits':>6s}  arn")
    for d in AwsDevice.get_devices(types=["QPU"]):
        try:
            q = d.properties.paradigm.qubitCount
        except Exception:
            q = "?"
        print(f"{d.name:18s} {d.provider_name:12s} {d.status:10s} {q!s:>6s}  {d.arn}")
    print("\nThe `device:` in configs/default.yaml must match a name above (and a key in `prices`).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local", choices=["aer", "braket_local"], help="free local simulator run")
    g.add_argument("--smoke-test", action="store_true", help="one circuit on the real device")
    g.add_argument("--submit", action="store_true", help="the full run on the real device")
    g.add_argument("--list-devices", action="store_true",
                   help="free: list AWS Braket devices and their status")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()
    if args.list_devices:
        return list_devices()

    hw = yaml.safe_load(open(args.config))["hardware"]
    ledger = Ledger(HW_DIR / "spend_ledger.csv")
    data = get_data(hw["dataset"], hw["n_features"], hw["n_train"], hw["seed"], n_test=hw["n_test"])
    n_circuits = len(entry_list(hw["n_train"], hw["n_test"]))
    cost = print_plan(hw, n_circuits, ledger)

    if args.smoke_test:
        return smoke_test(hw, data, ledger)
    if not (args.local or args.submit):
        print("\nPlan only - nothing was run. Use --local aer for a free test run.")
        return

    kind = args.local or "braket"
    name = run_name(hw, kind)
    store = EntryStore(HW_DIR / name / "entries.json")
    if kind == "braket":
        remaining = n_circuits - len(store.entries) - sum(len(p["keys"]) for p in store.data["pending"])
        cost = estimate_cost(remaining, hw["shots"], hw)
        if remaining < n_circuits:
            print(f"  resuming: {n_circuits - remaining} circuits already done/submitted, "
                  f"{remaining} left (${cost:.2f})")
        check_budget(cost, ledger, hw)
        if remaining:
            confirm(cost)
    print(f"\nRunning on {kind} -> {name}")
    backend = get_backend(kind, hw["device"])
    K = measure_entries(data[0], data[1], hw, backend, kind, store, ledger, name)
    compare(hw, data, K, f"measured_{kind}", HW_DIR / name)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:  # budget guard
        sys.exit(str(e))

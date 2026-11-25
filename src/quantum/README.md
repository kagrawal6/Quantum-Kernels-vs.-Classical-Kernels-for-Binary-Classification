# src/quantum/

Quantum kernel SVM (QSVM) using the quantum kernel estimator (QKE) approach from Havlíček et al. [3]: encode data into quantum states, estimate pairwise overlaps to build the kernel matrix, then train a classical SVM.

## Responsibilities
- Feature maps: Z and ZZ (Qiskit `z_feature_map`, `zz_feature_map`; the `ZFeatureMap` / `ZZFeatureMap` classes are deprecated since Qiskit 2.1) with `reps`, entanglement, and bandwidth λ (multiply inputs by λ before encoding).
- Kernel matrices: return `K_train` `(n_train, n_train)` and `K_test` `(n_test, n_train)` (see [src/README.md](../README.md)).
  - Default: exact statevector simulation (e.g. Qiskit Machine Learning `FidelityStatevectorKernel`).
  - Shots mode: sampler-based kernel with a set number of shots (for the shots sweep).
  - Optional: noisy simulator.
- Quantum cost counts per run: `n_circuits`, `total_shots`, `circuit_depth`, `n_2q_gates` (definitions in [EXPERIMENTS.md](../../EXPERIMENTS.md) Section 6).
- Kernel diagnostics: off-diagonal mean / std / min / max of `K_train`. A very small std means kernel concentration [4].
- Tune `C` like the classical models; the kernel does not depend on `C`, so compute it once and reuse it.

Watch for concentration [4]: if off-diagonal values bunch together, try fewer `reps`, linear entanglement, or a smaller bandwidth.

## Start here
[kernels.py](kernels.py) is implemented. `python scripts/check_kernels.py quantum` checks it against statevector overlaps computed directly, checks the shot-noise behaviour, and checks the cost counts.
Then `python scripts/run_experiment.py --sweep baseline --models qsvm` (defaults, test set: F1 0.93 Breast Cancer / 0.87 MNIST, ~0.1 s per kernel; one kernel circuit at 4 qubits = depth 39, 22 CNOTs).
C tuning, timing, metrics and diagnostics are done by `src/evaluation/runner.py`.

## Why bandwidth 0.05
See [BANDWIDTH.md](BANDWIDTH.md). Full study: `python scripts/bandwidth_study.py` (figure in `figures/bandwidth_study.png`, per-seed rows in `results/bandwidth_study/`).

## Real hardware (AWS Braket) — `scripts/run_hardware.py`
Settings in the `hardware:` section of `configs/default.yaml` (default: Breast Cancer, 2 qubits, 8 train / 6 test, 200 shots → 76 circuits ≈ **$29** on Rigetti Cepheus, $45 on IQM Garnet; IonQ Forte would be ≈ $1,239).
Uses `build_feature_map` from `kernels.py`, so it works once that TODO is filled.

1. `python scripts/run_hardware.py` — cost plan only, spends nothing.
2. `python scripts/run_hardware.py --local aer` — free; the whole pipeline on a local simulator.
3. `pip install qiskit-braket-provider`, set up AWS credentials, then `--local braket_local` (free, tests the Braket path).
4. `--smoke-test` — ONE circuit on the device (≈ $0.39): checks credentials and the device name.
5. `--submit` — the real run; you must type the dollar amount to confirm.

Budget guards: refuses if the run × 1.2 > `max_run_usd` ($50) or total spend would pass `max_total_usd` ($80); every batch is logged to `results/hardware/spend_ledger.csv` **before** it is submitted; progress and job ids are saved after each batch, so re-running after a crash re-attaches to submitted jobs instead of paying again.
**Only one person should submit** (the ledger is local and git-ignored), and set an **AWS Budgets alert** in the console as a second safety net. Prices in the config came from third-party listings (Nov 2026) — check https://aws.amazon.com/braket/pricing first.

Output `results/hardware/<run>/summary.csv` compares exact simulation, simulated shot noise, and the device on the same data. The test set is only 6 points, so compare the kernels (`kernel_mae_vs_exact`, off-diagonal std) more than accuracy.

## TODO
- [x] Set up Qiskit + Qiskit Machine Learning + Aer (installed: qiskit 2.6.1, qiskit-machine-learning 0.9.1, qiskit-aer 0.18.2)
- [x] Z and ZZ feature maps with reps / entanglement / bandwidth
- [x] Exact (statevector) kernel builder
- [x] Shot-based kernel builder
- [x] Cost counts (circuits, shots, depth, two-qubit gates)
- [x] Kernel diagnostics
- [x] First QSVM result at the defaults (both datasets)
- [x] Bandwidth study (why 0.05; why 0.25 on hardware)
- [x] Measure how long one run takes: exact simulation is not the bottleneck (4 qubits, `n_train` = 400: K_train 0.31 s, K_test 0.18 s). On hardware the same run is 119,800 circuits, so the budget is set by circuits, not simulator time.
- [ ] Optional: noisy simulator; optional: IBM hardware

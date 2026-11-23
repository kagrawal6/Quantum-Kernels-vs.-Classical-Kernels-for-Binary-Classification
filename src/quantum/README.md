# src/quantum/

Quantum kernel SVM (QSVM) using the quantum kernel estimator (QKE) approach from Havlíček et al. [3]: encode data into quantum states, estimate pairwise overlaps to build the kernel matrix, then train a classical SVM.

## Responsibilities
- Feature maps: Z and ZZ (Qiskit `ZFeatureMap`, `ZZFeatureMap`) with `reps`, entanglement, and bandwidth λ (multiply inputs by λ before encoding).
- Kernel matrices: return `K_train` `(n_train, n_train)` and `K_test` `(n_test, n_train)` (see [src/README.md](../README.md)).
  - Default: exact statevector simulation (e.g. Qiskit Machine Learning `FidelityStatevectorKernel`).
  - Shots mode: sampler-based kernel with a set number of shots (for the shots sweep).
  - Optional: noisy simulator.
- Quantum cost counts per run: `n_circuits`, `total_shots`, `circuit_depth`, `n_2q_gates` (definitions in [EXPERIMENTS.md](../../EXPERIMENTS.md) Section 6).
- Kernel diagnostics: off-diagonal mean / std / min / max of `K_train`. A very small std means kernel concentration [4].
- Tune `C` like the classical models; the kernel does not depend on `C`, so compute it once and reuse it.

Watch for concentration [4]: if off-diagonal values bunch together, try fewer `reps`, linear entanglement, or a smaller bandwidth.

## TODO
- [ ] Set up Qiskit + Qiskit Machine Learning + Aer
- [ ] Z and ZZ feature maps with reps / entanglement / bandwidth
- [ ] Exact (statevector) kernel builder
- [ ] Shot-based kernel builder
- [ ] Cost counts (circuits, shots, depth, two-qubit gates)
- [ ] Kernel diagnostics
- [ ] First QSVM result at the defaults on Breast Cancer
- [ ] Measure how long one run takes at `n_train` = 100, 200 to help set the time budget
- [ ] Optional: noisy simulator; optional: IBM hardware

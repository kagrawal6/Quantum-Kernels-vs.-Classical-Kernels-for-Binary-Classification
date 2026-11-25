# Experiment Specification

This is the single source of truth for **what we measure, how we measure it, and what we sweep**.
Every folder README links here. If something here changes, update it here first.

"Model" (or "machine") always means one of the three classifiers below — not a physical computer.

| Model | Kernel | Kernel parameters |
|-------|--------|-------------------|
| Linear SVM | linear | none |
| RBF SVM | RBF | `gamma` |
| QSVM | quantum fidelity kernel (QKE, [3]) | feature map, `reps`, entanglement, bandwidth, shots |

---

## 1. Ground rules (fairness)

1. **Same data for all models.** Same dataset, same reduced features, same train/test split, same seed.
2. **All models use a precomputed kernel.** Build the kernel matrix first (linear, RBF, or quantum), then train `SVC(kernel="precomputed")`. This gives one identical pipeline and lets us time kernel computation separately for every model.
3. **Test set is never used for tuning.** Tune with cross-validation on the training set only.
4. **Equal tuning budget.** Every model gets the same `C` grid and comparable effort on its kernel parameters.
5. **Repeat with seeds.** Every setting runs over **5 seeds**; report mean ± std.
6. **One factor at a time.** Sweeps change one variable and hold everything else at the defaults (Section 5).
7. **Fixed test set across training-size sweeps.** Only the training subset grows.
8. **Same machine for timing.** All timed runs on one computer, single process; record its specs with the results.

---

## 2. Priority measurements (logged for every run)

These are the measurements we prioritize. Every run must produce all of them.

| Metric | Definition |
|--------|------------|
| **F1-score** | Harmonic mean of precision and recall on the test set (positive class defined in `data/README.md`). Log **precision** and **recall** separately too. |
| **Test accuracy** | Fraction of correct test predictions. |
| **SVM training time** | Wall-clock time of `SVC.fit` only (kernel already computed). |
| **Kernel computation time** | Wall-clock time to build the kernel matrices, split into `kernel_train_time` (train × train) and `kernel_test_time` (test × train). |
| **Inference time** | Wall-clock time of `SVC.predict` only (test kernel already computed). |

`total_time = kernel_train_time + kernel_test_time + fit_time + predict_time`

**Train / validation / test and loss (logged for every run, not timed).**
Every run reports the same quantities on three splits so over-fitting and tuning quality are visible:

| Split | What it is | Columns |
|-------|------------|---------|
| Train | the `n_train` points the final SVM was fit on | `train_accuracy, train_f1, train_hinge_loss` |
| Validation | mean over the 5 CV folds at the chosen `C` (held-out parts of the training set; this is what picks `C`) | `val_accuracy, val_f1, val_hinge_loss` |
| Test | the fixed test set, never used for any choice | `accuracy, f1, …, test_hinge_loss` |

- **Loss = hinge loss** `mean(max(0, 1 − y·f(x)))`, `y ∈ {−1, +1}`, `f` = `SVC.decision_function`. This is the loss an SVM minimizes; unlike accuracy it also penalizes correct points inside the margin.
- **Cost function = SVM primal objective** on the training set: `svm_objective = ½‖w‖² + C·Σ hinge`, with `‖w‖² = αᵀ K_SV α` from the dual coefficients (exact for precomputed kernels; checked against `SVC(kernel="linear")`). Only comparable between runs with the same `C` and `n_train`.
- `n_support_vectors`: fewer support vectors = simpler boundary and faster inference.

Preprocessing (PCA, scaling) is **not** included in any timer.

---

## 3. Priority sweeps (run for every model)

| Sweep | What changes | Record |
|-------|--------------|--------|
| **Feature count** | `n_features` ∈ {2, 4, 6, 8} | all Section 2 metrics |
| **Training set size** | `n_train` ∈ {25, 50, 100, 200, 400, …} — keep doubling until one QSVM run exceeds the agreed time budget (this is our practical "what NISQ can handle" limit). Classical models also run at the full training set as a reference. | all Section 2 metrics |
| **Model parameters** | see Section 4 | all Section 2 metrics |

With angle encoding, **qubits = features**, so the feature-count sweep is also our qubit sweep.

---

## 4. How to vary model parameters

Two groups of parameters:

**a) SVM `C` — tuned, not swept.**
Grid-search `C` ∈ {0.01, 0.1, 1, 10, 100} with 5-fold CV on the training set, for all three models.
For QSVM the kernel does not depend on `C`, so compute the kernel once and reuse it for every `C`.

**b) Kernel parameters — swept one at a time around the defaults.**

| Model | Parameter | Values |
|-------|-----------|--------|
| RBF | `gamma` | {0.01, 0.1, 1, 10} (also try `"scale"`) |
| QSVM | feature map | {Z, ZZ} |
| QSVM | `reps` (circuit depth) | {1, 2, 3} |
| QSVM | entanglement | {linear, full} |
| QSVM | bandwidth λ (scale inputs by λ before encoding) | {0.025, 0.05, 0.1, 0.25, 0.5, 1.0} |

Bandwidth is the quantum counterpart of RBF `gamma`, and smaller bandwidth is a known way to reduce kernel concentration [4] (Shaydulin & Wild 2022; Canatar et al. 2023 — verify before citing).

---

## 5. Defaults (held fixed unless a sweep changes them)

| Setting | Default |
|---------|---------|
| Datasets | Breast Cancer Wisconsin; MNIST digits 3 vs 5 |
| Reduction | PCA, fit on all non-test data (chosen over SelectKBest: `scripts/compare_reduction.py`) |
| `n_features` | 4 |
| `n_train` | 100 (stratified subsample) |
| `n_test` | 100 (fixed, stratified) |
| Seeds | 0, 1, 2, 3, 4 |
| QSVM feature map | ZZ, `reps=2`, linear entanglement, bandwidth 0.05 (chosen on a validation split, test set untouched: `scripts/pick_bandwidth.py`) |
| QSVM backend | exact statevector simulation (no shot noise) |
| RBF `gamma` | `"scale"` |

Machine-readable copy: `configs/default.yaml`. Keep both in sync.

---

## 6. Secondary measurements (log if feasible)

Not the main focus, but they make the comparison honest and help explain results.

**Shots sweep (QSVM only).** Shots are the key NISQ constraint.
`shots` ∈ {exact, 256, 1024, 4096, 16384, 65536, 262144} — record all Section 2 metrics. (Extended past 4096 because F1 was still well below exact there; shots are simulated by binomial sampling, so large counts cost no extra time.)

**Hardware-independent quantum cost.** Simulator time is our laptop simulating circuits, which grows exponentially with qubits and does not reflect real quantum hardware. So for QSVM also record:
- `n_circuits` — train kernel needs n_train(n_train−1)/2, test kernel needs n_test × n_train
- `total_shots` — `n_circuits × shots`
- `circuit_depth` and `n_2q_gates` (two-qubit gate count) of one kernel circuit

Example: n_train = 200, n_test = 100 → 19,900 + 20,000 ≈ 40k circuits; at 1024 shots ≈ 40M shots.

**Kernel diagnostics (explains *why* QSVM fails, ref [4]).** For every kernel (all three models), on the off-diagonal entries of `K_train`: mean, std, min, max. Very small std = kernel concentration.

**Peak memory (low priority).** Peak memory during kernel computation. Kernel matrix grows as n²; statevector simulation grows as 2^qubits.
- `peak_memory_mb`: measured with `tracemalloc` over building `K_train` + `K_test`, in a **separate untimed repeat** of the kernel build (tracing slows Python, so it never touches the timers). Sees NumPy arrays (kernel matrices, statevectors) but not memory allocated inside native C++/Rust code. Turn off with `--no-memory` or `evaluation.measure_memory: false`.
- `kernel_matrix_mb`: analytic size of both kernel matrices, `(n_train² + n_test·n_train) × 8 bytes`.
- `statevector_mb` (QSVM only): analytic size of one complex128 statevector per point, `(n_train + n_test) × 2^qubits × 16 bytes`.

**Optional:** noisy simulator runs for QSVM; real IBM hardware is out of scope unless time allows.

---

## 7. Results schema (one CSV row per run)

All experiments append rows to CSVs in `results/` with these columns. Leave a column empty if it does not apply (e.g. `shots` for Linear SVM).

| Group | Columns |
|-------|---------|
| Run settings | `run_id, timestamp, sweep, dataset, model, seed, n_features, n_train, n_test` |
| Model parameters | `C, gamma, feature_map, reps, entanglement, bandwidth, shots, backend` |
| Quality (priority) | `accuracy, precision, recall, f1` |
| Train / val / test and loss | `train_accuracy, train_f1, train_hinge_loss, val_accuracy, val_f1, val_hinge_loss, test_hinge_loss, svm_objective, n_support_vectors` |
| Time (priority) | `kernel_train_time, kernel_test_time, fit_time, predict_time, total_time` |
| Quantum cost (secondary) | `n_circuits, total_shots, circuit_depth, n_2q_gates` |
| Kernel diagnostics (secondary) | `kernel_offdiag_mean, kernel_offdiag_std, kernel_offdiag_min, kernel_offdiag_max` |
| Memory (low priority) | `peak_memory_mb, kernel_matrix_mb, statevector_mb` |

---

## 8. Required outputs for the report

- Table: all three models at the defaults (mean ± std over seeds), both datasets.
- Per sweep: accuracy and F1 vs the swept variable, one line per model.
- Per sweep: kernel computation time and SVM training time vs the swept variable (log scale), one line per model.
- QSVM: accuracy vs shots; kernel off-diagonal std vs features / reps / bandwidth.
- Short interpretation: if QSVM wins, is the gain worth its cost? If it loses, which factor explains it (data size, features, shots, concentration)?

---

## References

- [1] LeCun, Cortes, Burges, "MNIST handwritten digit database," 2010.
- [2] Wolberg et al., "Breast Cancer Wisconsin (Diagnostic)," UCI ML Repository, 1993. https://doi.org/10.24432/C5DW2B
- [3] Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature* 567, 209–212, 2019. https://doi.org/10.1038/s41586-019-0980-2
- [4] Thanasilp et al., "Exponential concentration in quantum kernel methods," *Nature Communications* 15, 2024. https://doi.org/10.1038/s41467-024-49287-w

# Project TODO

**Project:** Classical vs Quantum Kernels for Binary Classification and Their Practicality  
**Team:** Steve Cho, Kushal Agrawal, Shunzhe Yang, Prateek Tandon

---

## References (keep cited while implementing)

| ID | Paper / dataset | Why we care |
|----|-----------------|-------------|
| [1] | LeCun et al., MNIST handwritten digit database (2010) | Digit recognition dataset |
| [2] | Wolberg et al., Breast Cancer Wisconsin (Diagnostic), UCI (1993) https://doi.org/10.24432/C5DW2B | Medical binary classification dataset |
| [3] | Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” *Nature* 567, 209–212 (2019) https://doi.org/10.1038/s41586-019-0980-2 | QVC + **QKE** (quantum kernel estimator) → our QSVM approach |
| [4] | Thanasilp et al., “Exponential concentration in quantum kernel methods,” *Nature Communications* (2024) https://doi.org/10.1038/s41467-024-49287-w | Kernel values can collapse → check kernel spread; watch shots / noise / expressivity |

---

## Phase 0 — Setup (all)

- [ ] Create / activate shared Python env
- [ ] `pip install -r requirements.txt`
- [ ] Confirm `python scripts/run_classical.py` runs on Breast Cancer
- [ ] Agree on default config in `configs/default.yaml` (start with `n_features=4`)

---

## Phase 1 — Dataset preparation (Steve) · Nov 3–15

- [ ] Finalize Breast Cancer loader (`src/data/load.py`) — [2]
- [ ] Add MNIST binary loader (pick digit pair, e.g. 0 vs 1) — [1]
- [ ] Implement / verify PCA + SelectKBest (`src/data/preprocess.py`)
- [ ] Save processed arrays under `data/processed/` (optional caching)
- [ ] Report how accuracy of a simple baseline changes with `n_features ∈ {2,4,6,8}`
- [ ] Document train sizes to sweep (e.g. 20% / 50% / 100% of training split)

---

## Phase 2 — Classical SVMs (Kushal) · Nov 3–15

- [ ] Linear SVM working end-to-end (`src/classical/svm.py`)
- [ ] RBF SVM working end-to-end
- [ ] Light hyperparameter search (`C`, `gamma`) on validation / CV
- [ ] Record **accuracy + F1** baselines for each dataset / feature count
- [ ] Save baseline numbers to `results/classical_baseline.csv`
- [ ] Keep classical pipeline API stable so quantum side can plug in later

---

## Phase 3 — Quantum kernel SVM (Shunzhe + team) · Nov 3–15

Inspired by Havlíček et al. [3] (quantum kernel estimator path, not full QVC focus).

- [ ] Set up Qiskit / Aer simulator
- [ ] Implement Z and ZZ feature maps (`src/quantum/feature_maps.py`) — [3]
- [ ] Implement kernel estimation (`src/quantum/kernel.py`)
- [ ] Train SVM with `kernel="precomputed"` (`src/quantum/qsvm.py`)
- [ ] Get a first Aer simulator result on Breast Cancer (`n_features=4`, few samples if slow)
- [ ] Add kernel diagnostics (`kernel_diagnostics`) — mean/std of off-diagonal entries — [4]
- [ ] If kernel values look “flat,” reduce `reps`, simplify map, or increase `shots` — [4]
- [ ] (Stretch) noisy simulator; (stretch) limited IBM hardware run

---

## Phase 4 — Experiments & analysis (Prateek + all) · Nov 16–22

- [ ] Unified runner: linear vs RBF vs QSVM (`src/evaluation/experiments.py`, `scripts/run_experiments.py`)
- [ ] Sweep: feature count, train size, qubits / map reps, shots
- [ ] Log accuracy, F1, runtime / shot cost → `results/`
- [ ] Plots in `figures/` (bar charts + sweep curves)
- [ ] Write short interpretation:
  - If QSVM wins: is the gain worth quantum cost?
  - If QSVM loses: link causes to data size / features / shots / concentration [4]

---

## Phase 5 — Report & presentation · Nov 23–Dec 2

- [ ] Finalize tables / figures
- [ ] Write methods + results in final report (cite [1]–[4])
- [ ] Prepare presentation slides
- [ ] Dry run as a team

---

## Repo map (who owns what)

```
configs/default.yaml          # shared experiment knobs
src/data/                     # Steve — load + PCA / feature select
src/classical/                # Kushal — linear + RBF SVM
src/quantum/                  # Shunzhe — feature maps, kernel, QSVM
src/evaluation/               # Prateek — metrics, sweeps
scripts/                      # thin CLIs to run pieces
results/  figures/            # outputs
notebooks/                    # optional EDA / demos
```

---

## Immediate next 3 actions

1. **Kushal / anyone:** run `python scripts/run_classical.py` and paste first Breast Cancer linear/RBF scores in chat or `results/`.
2. **Steve:** implement MNIST binary loader + confirm PCA shapes `(n_samples, n_features)`.
3. **Shunzhe:** fill in `compute_quantum_kernel` with Qiskit fidelity kernel (simulator first).

# TODO

Phased plan. Details for each area are in that folder's `README.md`; measurement definitions are in [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Phase 0 — Agree as a team (before coding)

- [ ] Read [EXPERIMENTS.md](EXPERIMENTS.md) and agree on the defaults
- [ ] Confirm MNIST digit pair (default 3 vs 5)
- [ ] Set the time budget for one QSVM run (caps the training-size sweep) → `configs/default.yaml`
- [ ] Agree on the interfaces in [src/README.md](src/README.md)
- [ ] Pick one machine for all timed runs
- [ ] Everyone: set up the environment from `requirements.txt`

---

## Phase 1 — Build the pipeline · Nov 3–15

**Data** — [src/data/](src/data/README.md)
- [x] Loaders for Breast Cancer and MNIST 3 vs 5 (positive class = 1)
- [x] Fixed stratified test set + training subsample by seed
- [x] PCA / SelectKBest + train-only scaling to `[0, π]`

**Classical** — [src/classical/](src/classical/README.md)
- [x] Linear and RBF kernel builders (precomputed)
- [x] CV tuning of `C`

**Quantum** — [src/quantum/](src/quantum/README.md)
- [x] Z / ZZ feature maps with reps, entanglement, bandwidth
- [x] Exact (statevector) kernel builder
- [x] Cost counts + kernel diagnostics
- [x] Bandwidth default justified (0.05; see [src/quantum/BANDWIDTH.md](src/quantum/BANDWIDTH.md))

**Evaluation** — [src/evaluation/](src/evaluation/README.md)
- [x] Metrics + timers + single-run function writing one CSV row
- [x] `scripts/run_experiment.py --sweep baseline` working end to end (harness done; needs the kernel TODOs filled)

**Milestone:** baseline table at the defaults for both datasets.

---

## Phase 2 — Experiments · Nov 16–22

**Priority sweeps** (all models, 5 seeds)
- [ ] Feature count {2, 4, 6, 8}
- [ ] Training set size (double until QSVM hits the time budget)
- [ ] Model parameters: RBF `gamma`; QSVM feature map, reps, entanglement, bandwidth

**Secondary**
- [ ] Shots sweep (QSVM)
- [ ] Kernel diagnostics across sweeps (concentration check [4])
- [x] Peak memory (low priority) — `peak_memory_mb` via tracemalloc, plus analytic `kernel_matrix_mb` / `statevector_mb`
- [x] Loss / cost function and train / validation / test metrics on every run
- [ ] Optional: noisy simulator

**Milestone:** all CSVs in `results/`.

---

## Phase 3 — Analysis, report, presentation · Nov 23–Dec 2

- [ ] Figures listed in [figures/README.md](figures/README.md)
- [ ] Summary tables (mean ± std over seeds)
- [ ] Interpretation: if QSVM wins, is it worth the cost? If it loses, which factor explains it?
- [ ] Final report (cite [1]–[4])
- [ ] Presentation + dry run

# Classical vs. Quantum Kernels for Binary Classification and Their Practicality

Steve Cho, Kushal Agrawal, Shunzhe Yang, Prateek Tandon

We compare three support vector machines on the same binary classification tasks:

- **Linear SVM**
- **RBF SVM**
- **Quantum kernel SVM (QSVM)** — encode data into quantum states, estimate a kernel with a quantum circuit (QKE, Havlíček et al. [3]), train a classical SVM on it

Datasets: Breast Cancer Wisconsin [2] and MNIST digits 3 vs 5 [1], reduced with PCA to a few features so they fit a small number of qubits.

The question is not only whether QSVM can match or beat classical kernels, but whether any gain is worth its cost under limited qubits and shots — and if it loses, why (e.g. kernel concentration [4]).

## Start here

1. **[EXPERIMENTS.md](EXPERIMENTS.md)** — what we measure, how, what we sweep, defaults, and the results CSV format. The single source of truth.
2. **[todo.md](todo.md)** — phased task list.
3. The `README.md` in each folder — what the folder is for and its TODOs.

## How to run (quick start)

```bash
python scripts/check_data.py                          # data sanity checks + PCA variance table
python scripts/check_kernels.py                       # are the kernels correct? (TODOs are skipped)
python scripts/run_experiment.py --sweep baseline     # all models x datasets x 5 seeds -> results/baseline.csv
python scripts/run_experiment.py --sweep features     # also: train_size, params, shots
python scripts/summarize.py results/baseline.csv      # mean ± std over seeds
```

Who fills in what:

| Part | File | Status |
|------|------|--------|
| Data (Steve) | `src/data/` | done |
| Evaluation harness (shared) | `src/evaluation/`, `scripts/run_experiment.py` | done: tuning, timing, metrics, CSV rows, sweeps, resume |
| Classical (Jerry + Colin) | `src/classical/kernels.py` | **template: fill the 2 TODOs** |
| Quantum (Kushal) | `src/quantum/kernels.py` | **template: fill the TODOs** |
| Analysis (Everyone) | run the sweeps, `scripts/summarize.py`, figures | figures TODO |

A kernel is just `kernel(XA, XB=None, **params) -> matrix`. The harness does everything else, identically for all three models.

## Repository layout

```
EXPERIMENTS.md       measurement spec (source of truth)
todo.md              task list and timeline
requirements.txt     Python dependencies
configs/             shared defaults and sweep values (default.yaml)
data/                datasets (raw/, processed/) + split rules
src/
  data/              load, split, reduce, scale
  classical/         linear + RBF kernels, C tuning
  quantum/           quantum feature maps, kernels, cost counts, diagnostics
  evaluation/        metrics, timers, sweep runner
scripts/             command-line entry points
results/             experiment CSVs
figures/             plots for the report
notebooks/           exploration
```

The interfaces between the `src/` parts are in [src/README.md](src/README.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## References

1. Y. LeCun, C. Cortes, and C. J. C. Burges, "MNIST handwritten digit database," 2010.
2. W. Wolberg, O. Mangasarian, N. Street, and W. Street, "Breast Cancer Wisconsin (Diagnostic)," UCI Machine Learning Repository, 1993. https://doi.org/10.24432/C5DW2B
3. V. Havlíček et al., "Supervised learning with quantum-enhanced feature spaces," *Nature*, vol. 567, pp. 209–212, 2019. https://doi.org/10.1038/s41586-019-0980-2
4. S. Thanasilp, S. Wang, M. Cerezo, and Z. Holmes, "Exponential concentration in quantum kernel methods," *Nature Communications*, vol. 15, 2024. https://doi.org/10.1038/s41467-024-49287-w

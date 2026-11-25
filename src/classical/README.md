# src/classical/

Classical baselines: Linear SVM and RBF SVM.

## Responsibilities
- Build kernel matrices explicitly so kernel computation time can be measured:
  - Linear: `sklearn.metrics.pairwise.linear_kernel`
  - RBF: `sklearn.metrics.pairwise.rbf_kernel` with `gamma`
- Return `K_train` `(n_train, n_train)` and `K_test` `(n_test, n_train)` (see [src/README.md](../README.md)).
- Tune `C` by 5-fold CV on the training set using the precomputed kernel (grid in [EXPERIMENTS.md](../../EXPERIMENTS.md) Section 4).
- Support the `gamma` sweep for RBF.
- Compute kernel diagnostics (off-diagonal mean / std / min / max) the same way as for the quantum kernel, so they can be compared.

`"scale"` gamma means `1 / (n_features * X_train.var())`; compute it yourself since we pass precomputed kernels.

## Start here
Fill the two TODOs in [kernels.py](kernels.py), then run `python scripts/check_kernels.py classical`.
It passes when your precomputed kernels give exactly the same predictions as `SVC(kernel="linear")` and `SVC(kernel="rbf", gamma="scale")`.
Then `python scripts/run_experiment.py --sweep baseline --models linear rbf`.
C tuning, timing, metrics and diagnostics are already done by `src/evaluation/runner.py`.

## TODO
- [x] Linear kernel builder (train + test matrices)
- [x] RBF kernel builder with `gamma` (including `"scale"`)
- [x] CV tuning of `C` with precomputed kernels (shared, in `src/evaluation/runner.py`)
- [x] Sanity check: precomputed results match `SVC(kernel="linear"/"rbf")` on the same data
- [x] First baseline numbers at the defaults for both datasets (`results/baseline.csv`; also `features.csv`, `params.csv`)
- [ ] Classical `train_size` sweep (includes the full-data reference; MNIST full is ~13k points, slow)

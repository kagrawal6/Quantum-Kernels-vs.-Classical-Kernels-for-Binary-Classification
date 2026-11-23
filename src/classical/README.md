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

## TODO
- [ ] Linear kernel builder (train + test matrices)
- [ ] RBF kernel builder with `gamma` (including `"scale"`)
- [ ] CV tuning of `C` with precomputed kernels
- [ ] Sanity check: precomputed results match `SVC(kernel="linear"/"rbf")` on the same data
- [ ] First baseline numbers at the defaults for both datasets

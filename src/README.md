# src/

All reusable project code lives here, split into four parts so people can work in parallel.

| Folder | Responsibility |
|--------|----------------|
| `data/` | Load, subsample, split, reduce features, scale |
| `classical/` | Linear and RBF kernel matrices + SVM tuning |
| `quantum/` | Quantum feature maps, quantum kernel matrices, cost counts, diagnostics |
| `evaluation/` | Metrics, timers, sweep runner, writing results |

## Interfaces between parts

These contracts let each part be built and tested independently. Agree before changing them.

1. **Data → everyone**
   Given `(dataset, n_features, n_train, seed)`, return `X_train, X_test, y_train, y_test`.
   Features are already reduced and scaled to `[0, π]`; the scaler is fit on the training set only.
   PCA is fit on all non-test data (unsupervised, no labels) so features are identical across training sizes; SelectKBest (uses labels) is fit on the training set only.
   Kernel functions have the signature `kernel(XA, XB=None, **params)`: `XB=None` → `K_train`, otherwise `K(XA=X_test, XB=X_train)`.

2. **Kernels → evaluation**
   Every kernel (linear, RBF, quantum) provides two matrices:
   - `K_train`: shape `(n_train, n_train)`
   - `K_test`: shape `(n_test, n_train)`

   Kernel builders must not compute anything else inside their timed section.

3. **Model**
   All three models are `sklearn.svm.SVC(kernel="precomputed", C=...)`, fit on `K_train` and predicted on `K_test`.

4. **Evaluation → results**
   Each run writes exactly one CSV row with the columns in [EXPERIMENTS.md](../EXPERIMENTS.md) Section 7.

Timing boundaries and metric definitions: [EXPERIMENTS.md](../EXPERIMENTS.md) Section 2.

# src/data/

Turns raw datasets into the arrays every model uses.

**Output contract:** given `(dataset, n_features, n_train, seed)`, return `X_train, X_test, y_train, y_test` (see [src/README.md](../README.md)).

## Responsibilities
- Load Breast Cancer [2] and MNIST 3 vs 5 [1]; map labels so the positive class is `1` (see [data/README.md](../../data/README.md)).
- Make the fixed stratified test set (`n_test` = 100) and the stratified training subsample (`n_train`).
- Reduce features to `n_features` with PCA (default) or feature selection (`SelectKBest`).
- Scale features, fitting on the training set only.
  - Quantum feature maps expect angles, so scale to a range like `[0, π]`. Use the same scaled input for all three models so the comparison stays fair.
- Report how much variance PCA keeps for each `n_features` (useful for explaining the feature-count sweep).

## Usage

```python
from src.data import get_data
X_train, X_test, y_train, y_test = get_data("mnist_3v5", n_features=4, n_train=100, seed=0)
# options: n_test=100, reduction="pca" | "select_k", reduction_fit="train" | "pool"
```

- `loaders.py`: raw `(X, y)` per dataset, positive class = 1. MNIST downloads once to `data/raw/` (~3 min), then loads from `data/raw/mnist_3v5.npz`.
- `preprocess.py`: fixed test split (seed 1234, same for every run seed) → stratified train subsample (run seed) → PCA / SelectKBest → min-max to `[0, π]` (test clipped).
  - PCA standardizes first for Breast Cancer (feature scales differ ~1000×); MNIST is only centered.
  - Default `reduction_fit`: PCA is fit on the pool (all non-test data, no labels) so features don't get worse as `n_train` shrinks; SelectKBest is fit on the training subsample only (it uses labels; `select_k` + `"pool"` raises an error). `"train"` is still available for PCA.
  - Why PCA and not SelectKBest: `python scripts/compare_reduction.py` (validation split, test untouched). PCA matched or beat SelectKBest on both datasets, e.g. MNIST RBF F1 at 4 features 0.887 vs 0.761.
- `python scripts/check_data.py`: shape / range / stratification / reproducibility / no-overlap checks, and writes `results/pca_explained_variance.csv`.

PCA cumulative explained variance (non-test pool):

| dataset | k=2 | k=4 | k=6 | k=8 |
|---|---|---|---|---|
| breast_cancer | 0.633 | 0.793 | 0.890 | 0.927 |
| mnist_3v5 | 0.206 | 0.333 | 0.416 | 0.475 |

## TODO
- [x] Breast Cancer loader (positive class = malignant)
- [x] MNIST 3 vs 5 loader
- [x] Stratified fixed test set + training subsample by seed
- [x] PCA and SelectKBest reduction
- [x] Train-only scaling to `[0, π]`
- [x] PCA explained-variance table for `n_features` ∈ {2, 4, 6, 8}
- [x] `reduction_fit` default: PCA on the pool, SelectKBest on train (see `scripts/compare_reduction.py`)
- [ ] Optional: cache outputs to `data/processed/` (preprocessing takes <1 s per run, so probably unnecessary)

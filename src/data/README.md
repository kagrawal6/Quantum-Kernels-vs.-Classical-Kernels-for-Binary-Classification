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

## TODO
- [ ] Breast Cancer loader (positive class = malignant)
- [ ] MNIST 3 vs 5 loader
- [ ] Stratified fixed test set + training subsample by seed
- [ ] PCA and SelectKBest reduction
- [ ] Train-only scaling to `[0, π]`
- [ ] PCA explained-variance table for `n_features` ∈ {2, 4, 6, 8}
- [ ] Optional: cache outputs to `data/processed/`

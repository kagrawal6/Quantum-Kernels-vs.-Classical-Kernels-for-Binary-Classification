# data/

Datasets and the rules for preparing them. Data files are git-ignored; only these READMEs are committed.

| Folder | Contents |
|--------|----------|
| `raw/` | Original downloads, untouched |
| `processed/` | Optional cached arrays after subsampling / reduction / scaling |

## Datasets

| Dataset | Source | Positive class | Notes |
|---------|--------|----------------|-------|
| Breast Cancer Wisconsin (Diagnostic) [2] | `sklearn.datasets.load_breast_cancer` | malignant | 569 samples, 30 features, slightly imbalanced (212 malignant / 357 benign) |
| MNIST binary [1] | MNIST, digits 3 vs 5 | 5 | 784 features; 3 vs 5 is harder than 0 vs 1, so differences between models are more visible |

The positive class is what precision, recall, and F1 are computed for.

## Split rules
- Fixed, stratified test set of `n_test` = 100 per dataset and seed.
- Training set is a stratified subsample of size `n_train` from the remaining data.
- Training-size sweeps grow the training subset only; the test set stays the same.

See [EXPERIMENTS.md](../EXPERIMENTS.md) for defaults and sweep values.

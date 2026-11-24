"""Turn a raw dataset into the arrays every model uses.

Main entry point:  get_data(dataset, n_features, n_train, seed)
                   -> X_train, X_test, y_train, y_test

Pipeline (order matters):
  1. Load full dataset, positive class = 1.
  2. Split off a FIXED stratified test set (same for every seed and every
     sweep, so model differences aren't test-set noise).
  3. Draw a stratified training subsample of size n_train using `seed`.
     Everything not in the test set is the "pool" it is drawn from.
  4. Reduce to n_features with PCA (default) or SelectKBest.
  5. Min-max scale to [0, pi] (angles for the quantum feature map); clip
     test values that fall outside the training range.

Every transform in steps 4-5 is FIT on training data only and then
APPLIED to the test set, so no test information leaks into the model.
The same scaled arrays go to all three models (linear, RBF, quantum).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .loaders import load_dataset

# Fixed seed for the test split. Deliberately NOT the run seed, so the test
# set is identical across seeds 0-4 and across every sweep.
TEST_SPLIT_SEED = 1234

# Breast Cancer features differ in scale by ~1000x (e.g. area vs smoothness),
# so we standardize before PCA. MNIST pixels already share a 0-1 scale, and
# dividing by each pixel's std would blow up nearly-blank border pixels,
# so for MNIST we only center (subtract the mean).
_STANDARDIZE = {"breast_cancer": True, "mnist_3v5": False}


def split_test(X, y, n_test=100, test_seed=TEST_SPLIT_SEED):
    """Split into (pool, test). Stratified so both keep the class ratio."""
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=n_test, stratify=y, random_state=test_seed
    )
    return X_pool, X_test, y_pool, y_test


def subsample_train(X_pool, y_pool, n_train, seed):
    """Stratified subsample of n_train points from the pool."""
    if n_train is None or n_train >= len(y_pool):
        return X_pool, y_pool  # use the whole pool (classical "full data" reference)
    X_tr, _, y_tr, _ = train_test_split(
        X_pool, y_pool, train_size=n_train, stratify=y_pool, random_state=seed
    )
    return X_tr, y_tr


def make_reducer(dataset, n_features, reduction="pca"):
    """Build an (unfitted) sklearn pipeline that maps raw X -> n_features."""
    if reduction == "pca":
        return make_pipeline(
            StandardScaler(with_std=_STANDARDIZE[dataset]),
            PCA(n_components=n_features, svd_solver="full"),
        )
    if reduction == "select_k":
        # Drop constant columns first (MNIST has always-black border pixels,
        # whose F-score is undefined), then keep the n_features columns
        # with the highest ANOVA F-score against the training labels.
        return make_pipeline(
            VarianceThreshold(0.0),
            SelectKBest(f_classif, k=n_features),
        )
    raise ValueError(f"reduction must be 'pca' or 'select_k', got {reduction!r}")


def get_data(
    dataset,
    n_features=4,
    n_train=100,
    seed=0,
    *,
    n_test=100,
    reduction="pca",
    reduction_fit=None,
    test_seed=TEST_SPLIT_SEED,
):
    """Return X_train, X_test, y_train, y_test for one run.

    Args:
        dataset: "breast_cancer" or "mnist_3v5".
        n_features: features after reduction (= qubits with angle encoding).
        n_train: training subsample size (None = whole pool).
        seed: controls which training points are drawn.
        n_test: size of the fixed test set.
        reduction: "pca" or "select_k".
        reduction_fit: what the PCA / SelectKBest is fit on.
            None    - (default, team decision) "pool" for PCA, "train" for
                      SelectKBest.
            "pool"  - all non-test data. PCA uses no labels, so this has no
                      test leakage, and the reduced features are identical
                      at every n_train: the training-size sweep then
                      measures the SVM alone, not PCA quality.
            "train" - only the n_train subsample.
            SelectKBest uses labels, so it must be fit on "train" only;
            asking for select_k + "pool" raises an error.
            The [0, pi] scaler is always fit on the training subsample.
        test_seed: seed for the fixed test split (leave at default).

    Returns:
        X_train (n_train, n_features), X_test (n_test, n_features),
        y_train (n_train,), y_test (n_test,).  X values lie in [0, pi].
    """
    X, y = load_dataset(dataset)
    X = X.astype(np.float64)  # MNIST is cached as float32; avoid rounding past pi
    X_pool, X_test, y_pool, y_test = split_test(X, y, n_test, test_seed)
    X_train, y_train = subsample_train(X_pool, y_pool, n_train, seed)

    # Step 4: dimensionality reduction (fit on train or pool, never test).
    if reduction_fit is None:
        reduction_fit = "pool" if reduction == "pca" else "train"
    if reduction == "select_k" and reduction_fit == "pool":
        raise ValueError("SelectKBest uses labels; fit it on 'train' only "
                         "(pool labels would leak extra label information).")
    reducer = make_reducer(dataset, n_features, reduction)
    if reduction_fit == "train":
        reducer.fit(X_train, y_train)
    elif reduction_fit == "pool":
        reducer.fit(X_pool, y_pool)
    else:
        raise ValueError("reduction_fit must be 'train' or 'pool'")
    Z_train = reducer.transform(X_train)
    Z_test = reducer.transform(X_test)

    # Step 5: scale each feature to [0, pi] using the training min/max.
    # Test points can land slightly outside the training range; clip them
    # so every quantum rotation angle stays in the intended range.
    scaler = MinMaxScaler(feature_range=(0.0, np.pi)).fit(Z_train)
    X_train_out = scaler.transform(Z_train)
    X_test_out = np.clip(scaler.transform(Z_test), 0.0, np.pi)

    return X_train_out, X_test_out, y_train, y_test


def pca_explained_variance(dataset, max_components=8, n_test=100,
                           test_seed=TEST_SPLIT_SEED):
    """Cumulative fraction of variance kept by the first k PCA components.

    Fit on the full non-test pool (a property of the dataset, not of one
    run). Returns an array whose entry [k-1] is the variance kept by k
    components. Useful for explaining the feature-count sweep.
    """
    X, y = load_dataset(dataset)
    X_pool, _, _, _ = split_test(X, y, n_test, test_seed)
    pipe = make_reducer(dataset, max_components, "pca").fit(X_pool)
    return np.cumsum(pipe.named_steps["pca"].explained_variance_ratio_)

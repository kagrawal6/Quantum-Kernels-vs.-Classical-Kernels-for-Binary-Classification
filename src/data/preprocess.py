"""Feature reduction and train/test splitting.

Owner: Steve Cho
Use PCA or feature selection so data fits limited qubits.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def reduce_features(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 4,
    method: str = "pca",
) -> np.ndarray:
    """Reduce dimensionality to `n_features`.

    method: "pca" | "select_k"
    """
    if method == "pca":
        return PCA(n_components=n_features).fit_transform(X)
    if method == "select_k":
        return SelectKBest(score_func=f_classif, k=n_features).fit_transform(X, y)
    raise ValueError(f"Unknown reduction method: {method}")


def train_test_split_scaled(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    feature_range: Tuple[float, float] = (0.0, np.pi),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data and optionally scale features.

    MinMax to [0, pi] is often useful for quantum angle encoding / ZZ maps.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scale:
        # Fit scalers on train only, then transform test.
        std = StandardScaler()
        X_train = std.fit_transform(X_train)
        X_test = std.transform(X_test)
        # Map to a range friendly for quantum angle / ZZ feature maps.
        mm = MinMaxScaler(feature_range=feature_range)
        X_train = mm.fit_transform(X_train)
        X_test = mm.transform(X_test)

    return X_train, X_test, y_train, y_test

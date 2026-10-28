"""Quick classical baseline run.

Usage (from repo root):
  python scripts/run_classical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.classical.svm import train_linear_svm, train_rbf_svm
from src.data.load import load_dataset
from src.data.preprocess import reduce_features, train_test_split_scaled
from src.evaluation.metrics import evaluate_classifier


def main() -> None:
    X, y = load_dataset("breast_cancer")
    X = reduce_features(X, y, n_features=4, method="pca")
    X_train, X_test, y_train, y_test = train_test_split_scaled(X, y)

    linear = train_linear_svm(X_train, y_train)
    rbf = train_rbf_svm(X_train, y_train)

    print("Linear:", evaluate_classifier(y_test, linear.predict(X_test)))
    print("RBF:   ", evaluate_classifier(y_test, rbf.predict(X_test)))


if __name__ == "__main__":
    main()

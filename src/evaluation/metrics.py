"""Shared metrics: accuracy + F1 (proposal evaluation criteria).

Owner: Prateek Tandon
"""

from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_classifier(y_true, y_pred, average: str = "binary") -> Dict[str, float]:
    """Return accuracy and F1 for a binary classifier."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def pretty_report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=4, zero_division=0)

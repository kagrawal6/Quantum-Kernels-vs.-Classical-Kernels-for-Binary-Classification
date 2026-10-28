"""Experiment runners: vary features, train size, qubits, shots, etc."""


from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


def run_baseline_comparison(
    dataset: str = "breast_cancer",
    n_features: int = 4,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Run linear vs RBF vs (later) QSVM on one setting.

    TODO:
      - Load + preprocess data
      - Train classical models
      - Train QSVM when quantum kernel is ready
      - Return a results table (accuracy, f1, runtime)
    """
    raise NotImplementedError("Wire up loaders + classical/quantum trainers here.")


def sweep_parameters(param_grid: Dict[str, List]) -> pd.DataFrame:
    """Placeholder for sweeps over n_features, train size, shots, qubits."""
    raise NotImplementedError("Parameter sweep not implemented yet.")

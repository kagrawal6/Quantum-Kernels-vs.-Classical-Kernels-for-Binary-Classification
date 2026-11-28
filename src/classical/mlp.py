"""Small neural network (MLP) baseline.

The syllabus asks for a neural-network benchmark next to the quantum model.
It is deliberately small and CPU-only (no GPU), and it sees exactly the same
data as the SVMs: the same PCA features scaled to [0, pi], the same training
points per seed and the same fixed test set.

Unlike the SVMs it has no kernel: it learns directly from the features.
The one hyper-parameter tuned is the L2 penalty `alpha` (the MLP's
counterpart of the SVM's C), by the same 5-fold CV with F1 scoring.
"""

from sklearn.neural_network import MLPClassifier

HIDDEN_LAYERS = (32, 16)          # two small ReLU layers
ALPHA_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def build_mlp(seed):
    """Return (untrained MLP, hyper-parameter grid for CV)."""
    est = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYERS, activation="relu",
                        solver="adam", max_iter=2000, random_state=seed)
    return est, {"alpha": ALPHA_GRID}

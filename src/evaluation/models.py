"""The models, described in one place.

Every SVM model is "a kernel function + SVC(kernel='precomputed')". The runner
only needs to know, for each one:
  kernel  - function(XA, XB=None, **params) -> kernel matrix
            XB=None  -> training kernel  K(X_train, X_train), shape (n, n)
            XB given -> test kernel      K(X_test,  X_train), shape (m, n)
            So the TRAINING set is XA when XB is None, otherwise XB.
  params  - which config keys this kernel takes (everything else is ignored)
  cost    - optional function returning quantum cost columns
  seeded  - True if the kernel is random (shot noise) and needs the run seed

The neural-network baseline ("mlp") is the exception: it has no kernel and is
trained directly on the features, so instead of "kernel" it has
  estimator - function(seed) -> (untrained sklearn model, CV grid)
and the runner takes a separate, simpler path for it.
"""

from src.classical.kernels import linear_kernel, rbf_kernel
from src.classical.mlp import build_mlp
from src.quantum.kernels import quantum_cost, quantum_kernel

MODELS = {
    "linear": {
        "kernel": linear_kernel,
        "params": [],
        "cost": None,
        "seeded": False,
    },
    "rbf": {
        "kernel": rbf_kernel,
        "params": ["gamma"],
        "cost": None,
        "seeded": False,
    },
    "qsvm": {
        "kernel": quantum_kernel,
        "params": ["feature_map", "reps", "entanglement", "bandwidth", "shots"],
        "cost": quantum_cost,
        "seeded": True,
    },
    "mlp": {
        "estimator": build_mlp,
        "params": [],
        "cost": None,
        "seeded": True,
    },
}

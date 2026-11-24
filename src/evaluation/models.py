"""The three models, described in one place.

Every model is "a kernel function + SVC(kernel='precomputed')". The runner
only needs to know, for each model:
  kernel  - function(XA, XB=None, **params) -> kernel matrix
            XB=None  -> training kernel  K(X_train, X_train), shape (n, n)
            XB given -> test kernel      K(X_test,  X_train), shape (m, n)
            So the TRAINING set is XA when XB is None, otherwise XB.
  params  - which config keys this kernel takes (everything else is ignored)
  cost    - optional function returning quantum cost columns
  seeded  - True if the kernel is random (shot noise) and needs the run seed
"""

from src.classical.kernels import linear_kernel, rbf_kernel
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
}

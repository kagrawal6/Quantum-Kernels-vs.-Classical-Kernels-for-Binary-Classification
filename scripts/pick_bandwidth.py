"""Choose the QSVM bandwidth default on a validation split (test set untouched).

Same protocol as compare_reduction.py: 100 training points, 200 validation
points from the non-test pool, 4 PCA features, ZZ reps=2 linear, C tuned by
5-fold CV, 5 seeds. Result (Nov 24): bandwidth 0.05 was best on both datasets
(validation F1 0.925 Breast Cancer, 0.873 MNIST; 1.0 gave 0.529 / 0.505).
Needs src/quantum/kernels.py to be implemented.
Usage: python scripts/pick_bandwidth.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from src.data.loaders import load_dataset
from src.data.preprocess import split_test, make_reducer
from src.quantum.kernels import quantum_kernel
from src.evaluation.runner import tune_C
# validation split from the non-test pool; test set untouched
for ds in ["breast_cancer", "mnist_3v5"]:
    X, y = load_dataset(ds); X = X.astype(float)
    Xp, _, yp, _ = split_test(X, y, 100)
    red = make_reducer(ds, 4, "pca").fit(Xp)
    print(ds)
    for bw in [0.025, 0.05, 0.1, 0.25, 0.5, 1.0]:
        f = []
        for s in range(5):
            Xtr, Xr, ytr, yr = train_test_split(Xp, yp, train_size=100, stratify=yp, random_state=s)
            Xva, _, yva, _ = train_test_split(Xr, yr, train_size=200, stratify=yr, random_state=100+s)
            sc = MinMaxScaler((0, np.pi)).fit(red.transform(Xtr))
            Ztr = sc.transform(red.transform(Xtr)); Zva = np.clip(sc.transform(red.transform(Xva)), 0, np.pi)
            Ktr = quantum_kernel(Ztr, bandwidth=bw); Kva = quantum_kernel(Zva, Ztr, bandwidth=bw)
            C = tune_C(Ktr, ytr, [0.01, 0.1, 1, 10, 100], 5, s)
            f.append(f1_score(yva, SVC(kernel="precomputed", C=C).fit(Ktr, ytr).predict(Kva)))
        off = Ktr[~np.eye(len(Ktr), dtype=bool)]
        print(f"  bw={bw:<5}  val F1 {np.mean(f):.3f} ± {np.std(f):.3f}   offdiag mean {off.mean():.3f} std {off.std():.3f}")

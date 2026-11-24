"""Choose the dimensionality reduction: PCA (pool / train) vs SelectKBest.

Classical SVMs only, scored on a validation split drawn from the non-test
pool, so the fixed test set is never used to make this choice.
Result (Nov 24): PCA fit on the pool matched or beat the others on both
datasets, so all experiments use it. Usage: python scripts/compare_reduction.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from src.data.loaders import load_dataset
from src.data.preprocess import split_test, make_reducer

# Validation-only comparison: the fixed test set is never touched.
for ds in ["breast_cancer", "mnist_3v5"]:
    X, y = load_dataset(ds); X = X.astype(np.float64)
    Xp, _, yp, _ = split_test(X, y, 100)
    print(f"\n{ds}  (n_train=100, validation=200 from non-test pool, 5 seeds, C tuned by 5-fold CV)")
    print("  k   method          linear F1        RBF F1")
    for k in [2, 4, 6, 8]:
        for method, fit_on in [("pca/pool", "pool"), ("pca/train", "train"), ("selectk/train", "train")]:
            res = {"linear": [], "rbf": []}
            for s in range(5):
                Xtr, Xrest, ytr, yrest = train_test_split(Xp, yp, train_size=100, stratify=yp, random_state=s)
                Xva, _, yva, _ = train_test_split(Xrest, yrest, train_size=200, stratify=yrest, random_state=100 + s)
                red = make_reducer(ds, k, "pca" if method.startswith("pca") else "select_k")
                red.fit(Xp if fit_on == "pool" else Xtr, yp if fit_on == "pool" else ytr)
                sc = MinMaxScaler((0, np.pi)).fit(red.transform(Xtr))
                Ztr = sc.transform(red.transform(Xtr)); Zva = np.clip(sc.transform(red.transform(Xva)), 0, np.pi)
                for kern in res:
                    m = GridSearchCV(SVC(kernel=kern), {"C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Ztr, ytr)
                    res[kern].append(f1_score(yva, m.predict(Zva)))
            print(f"  {k}   {method:14s}  " + "   ".join(f"{np.mean(v):.3f}±{np.std(v):.3f}" for v in res.values()))

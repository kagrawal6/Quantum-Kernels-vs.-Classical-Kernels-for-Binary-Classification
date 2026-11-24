# data/raw/

Original dataset files exactly as downloaded. Never edit these by hand.

Breast Cancer loads directly from scikit-learn, so nothing needs to be stored here for it. MNIST files (or a cached download) go here.

## TODO
- [x] MNIST is downloaded automatically by `src/data/loaders.py` via `sklearn.datasets.fetch_openml("mnist_784")` into `openml/`; the 3-vs-5 subset (all 70k images pooled, 13,454 kept) is cached as `mnist_3v5.npz`.

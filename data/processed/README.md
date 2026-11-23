# data/processed/

Optional cache of prepared arrays so experiments don't redo preprocessing.

Suggested file name: `{dataset}_f{n_features}_n{n_train}_seed{seed}.npz` containing `X_train, X_test, y_train, y_test`.

Anything here must be reproducible from `raw/` + the code in `src/data/`. Safe to delete at any time.

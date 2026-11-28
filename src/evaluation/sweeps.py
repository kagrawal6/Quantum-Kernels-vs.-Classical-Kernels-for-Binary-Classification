"""Turn configs/default.yaml into the list of runs for each sweep.

Each job is a dict: sweep, dataset, model, n_features, n_train, params.
Seeds are looped over by the caller. A sweep changes ONE thing and keeps
everything else at the defaults (EXPERIMENTS.md ground rule 6).
"""

import yaml

SWEEPS = ("baseline", "features", "train_size", "params", "shots")
ALL_MODELS = ("linear", "rbf", "qsvm", "mlp")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def default_params(cfg, model):
    """Kernel parameters at the defaults for one model."""
    if model in ("linear", "mlp"):
        return {}
    if model == "rbf":
        return {"gamma": cfg["rbf"]["gamma"]}
    if model == "qsvm":
        q = cfg["quantum"]
        return {"feature_map": q["feature_map"], "reps": q["reps"],
                "entanglement": q["entanglement"], "bandwidth": q["bandwidth"],
                "shots": None}  # None = exact statevector
    raise ValueError(model)


def make_jobs(cfg, sweep, datasets=None, models=None):
    d, s = cfg["data"], cfg["sweeps"]
    datasets = datasets or d["datasets"]
    models = models or ALL_MODELS
    jobs = []

    def add(dataset, model, n_features=None, n_train=None, **override):
        params = {**default_params(cfg, model), **override}
        jobs.append({
            "sweep": sweep, "dataset": dataset, "model": model,
            "n_features": n_features or d["n_features"],
            "n_train": d["n_train"] if n_train is None else n_train,
            "params": params,
        })

    for ds in datasets:
        for m in models:
            if sweep == "baseline":
                add(ds, m)
            elif sweep == "features":
                # With angle encoding qubits = features, so this is also the qubit sweep.
                for nf in s["n_features"]:
                    add(ds, m, n_features=nf)
            elif sweep == "train_size":
                # Ascending order matters: the run script stops QSVM once one
                # run exceeds the time budget.
                for nt in sorted(s["n_train"]):
                    add(ds, m, n_train=nt)
                add(ds, m, n_train="full")  # all non-test data, every model
            elif sweep == "params":
                if m == "rbf":
                    for g in s["rbf_gamma"]:
                        add(ds, m, gamma=g)
                elif m == "qsvm":
                    for key, cfg_key in [("feature_map", "feature_map"),
                                         ("reps", "reps"),
                                         ("entanglement", "entanglement"),
                                         ("bandwidth", "bandwidth")]:
                        for v in s[cfg_key]:
                            add(ds, m, **{key: v})
            elif sweep == "shots":
                if m == "qsvm":
                    for sh in s["shots"]:
                        add(ds, m, shots=None if sh == "exact" else int(sh))
            else:
                raise ValueError(f"unknown sweep {sweep!r}; choose from {SWEEPS}")
    return jobs

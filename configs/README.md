# configs/

Shared settings so everyone runs the same experiments.

- `default.yaml` — agreed defaults and sweep values. It is the machine-readable copy of [EXPERIMENTS.md](../EXPERIMENTS.md) Sections 3–6; change both together.
- Scripts should read values from here instead of hardcoding them.
- For a one-off experiment, copy `default.yaml` to a new file (e.g. `quick_test.yaml`) rather than editing the default.

## TODO
- [ ] Agree on `time_budget_minutes` (caps the training-size sweep)
- [ ] Confirm the MNIST digit pair (default 3 vs 5)
- [ ] Add config files for any extra experiments (e.g. noisy simulator)

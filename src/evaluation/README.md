# src/evaluation/

Runs experiments the same way for all three models and records results.

Everything here follows [EXPERIMENTS.md](../../EXPERIMENTS.md) — metric definitions (Section 2), sweeps (Sections 3–4), defaults (Section 5), and the CSV columns (Section 7).

## Responsibilities
- **Metrics:** accuracy, precision, recall, F1 (positive class = 1).
- **Timers:** time exactly these four steps and nothing else:
  `kernel_train_time`, `kernel_test_time`, `fit_time`, `predict_time` (+ `total_time` = sum).
  Use `time.perf_counter()`; exclude preprocessing.
- **Memory (low priority):** peak memory during kernel computation (e.g. `tracemalloc`).
- **Run one setting:** data → kernel → tune `C` → fit → predict → metrics → one CSV row.
- **Sweeps:** loop one variable at a time over all models and 5 seeds, holding everything else at the defaults.
- **Summaries:** group by setting and model, report mean ± std over seeds.

## TODO
- [ ] Metrics helper (accuracy, precision, recall, F1)
- [ ] Timing helper for the four timed steps
- [ ] Single-run function that writes one row with all Section 7 columns
- [ ] Sweep runner: feature count, training size, model parameters (priority)
- [ ] Shots sweep for QSVM (secondary)
- [ ] Summary table (mean ± std over seeds)
- [ ] Record the machine specs used for timing

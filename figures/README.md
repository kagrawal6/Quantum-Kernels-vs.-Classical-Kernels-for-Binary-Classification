# figures/

Plots generated from `results/` CSVs (by `scripts/make_figures`). Git-ignored by default; commit final figures on purpose.

Every plot: one line or bar per model (Linear / RBF / QSVM), error bars = std over seeds, one plot per dataset.

## Required plots

**Priority**
- [ ] Baseline bar chart: accuracy and F1 for all models at the defaults
- [ ] Accuracy and F1 vs feature count
- [ ] Accuracy and F1 vs training set size
- [ ] Kernel computation time and SVM training time vs feature count (log scale)
- [ ] Kernel computation time and SVM training time vs training set size (log scale)
- [ ] Accuracy and F1 vs each model parameter (RBF `gamma`; QSVM map / reps / entanglement / bandwidth)
- [ ] Inference time comparison at the defaults

**Secondary**
- [ ] QSVM accuracy vs shots
- [ ] Kernel off-diagonal std vs feature count, reps, and bandwidth (concentration check [4])
- [ ] QSVM total shots / circuits vs training set size

Suggested file name: `{plot}_{dataset}.png`, e.g. `acc_vs_features_breast_cancer.png`.

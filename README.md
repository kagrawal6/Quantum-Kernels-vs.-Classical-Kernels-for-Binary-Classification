# Classical vs Quantum Kernels for Binary Classification and Their Practicality

**Team:** Steve Cho, Kushal Agrawal, Shunzhe Yang, Prateek Tandon

Compare **Linear SVM**, **RBF SVM**, and **Quantum Kernel SVM** on binary datasets (MNIST subset, Breast Cancer Wisconsin), under limited qubits / features.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Classical baselines (works now)
python scripts/run_classical.py
```

See **[todo.md](todo.md)** for tasks, owners, timeline, and references.

## Layout

```
configs/          default experiment settings
src/data/         load + PCA / feature selection
src/classical/    linear + RBF SVM
src/quantum/      feature maps, quantum kernel, QSVM
src/evaluation/   metrics + experiment sweeps
scripts/          run_classical / run_quantum / run_experiments
data/             raw + processed
results/ figures/ experiment outputs
notebooks/        optional EDA
```

## References

1. LeCun et al., MNIST (2010)  
2. Wolberg et al., Breast Cancer Wisconsin (Diagnostic), UCI (1993)  
3. Havlíček et al., *Nature* (2019) — quantum-enhanced feature spaces / QKE  
4. Thanasilp et al., *Nature Communications* (2024) — exponential concentration in quantum kernels  

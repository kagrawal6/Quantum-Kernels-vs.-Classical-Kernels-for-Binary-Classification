# Why the QSVM bandwidth is 0.05

The bandwidth λ multiplies the inputs (already scaled to [0, π]) before they are encoded into the circuit. It sets how quickly two points stop "looking alike" to the quantum kernel, the same job γ does for the RBF kernel.

Picked on a validation split (100 train / 200 val from the non-test pool, 5 seeds, ZZ map, reps 2, 4 qubits). The test set was never used. Reproduce with `python scripts/bandwidth_study.py` (~3 min).

| λ | Breast Cancer val F1 | MNIST val F1 | Kernel off-diagonal mean |
|---|---|---|---|
| 0.01 | 0.898 | 0.852 | 0.99 |
| **0.05** | **0.925** | **0.873** | 0.92 |
| 0.25 | 0.827 | 0.820 | 0.22 |
| 1.0 | 0.529 | 0.505 | 0.08 |
| *RBF, tuned γ* | *0.927* | *0.886* | |

**λ = 1 is too large.** The encoded states become nearly orthogonal (average overlap 0.08 ≈ 1/2⁴, what random 4-qubit states give), so every point looks unrelated to every other. This is the kernel concentration of Thanasilp et al. [4]. The SVM memorizes the training set (75–85% of points become support vectors) and fails on new data.

**Very small λ is too small.** Every overlap approaches 1, so all points look the same. The SVM can only compensate with a very large C, and our C grid stops at 100, so F1 drops below λ ≈ 0.02.

**0.05 is in between.** It sits in the middle of a flat region (0.02–0.07, all within ~0.01 F1), so the choice is not fragile, and the QSVM matches a tuned RBF SVM there. For the Z feature map the kernel is exactly ∏ cos²(λΔᵢ) ≈ RBF with γ = λ², which is why λ behaves like γ.

**Caveats**
- 0.05 stays within 0.02 F1 of the best λ for 2–8 qubits; with reps = 3, 0.025 is slightly better.
- With finite shots, λ = 0.1 holds up better: when all kernel values are near 1, shot noise hides the small differences between them.
- The hardware run uses 0.25 because it has only 8 training points and a fixed C = 1. At 0.05 that setup predicts "benign" for almost everything.

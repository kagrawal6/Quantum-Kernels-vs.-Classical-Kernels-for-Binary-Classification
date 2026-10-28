"""Full comparison / sweeps entrypoint (stub).

Usage (from repo root):
  python scripts/run_experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("TODO: run linear vs RBF vs QSVM sweeps; save CSV to results/")
    print("See todo.md.")


if __name__ == "__main__":
    main()

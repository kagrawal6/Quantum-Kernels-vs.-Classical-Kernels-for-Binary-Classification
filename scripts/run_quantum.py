"""Quantum kernel SVM entrypoint (stub until kernel is implemented).

Usage (from repo root):
  python scripts/run_quantum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    print("TODO: load data → feature map → quantum kernel → precomputed SVM")
    print("See src/quantum/ and todo.md (Shunzhe + team).")


if __name__ == "__main__":
    main()

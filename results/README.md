# results/

Experiment output CSVs. Files here are git-ignored by default; commit final CSVs on purpose (e.g. `git add -f`) when they go into the report.

## Rules
- One CSV per sweep (names listed in [scripts/README.md](../scripts/README.md)).
- One row per run (dataset × model × setting × seed).
- Columns exactly as in [EXPERIMENTS.md](../EXPERIMENTS.md) Section 7, in that order. Leave a cell empty if it does not apply.
- Append, don't overwrite; each row has a unique `run_id` and a `timestamp`.
- Save the specs of the machine used for timing in `results/machine.txt` (CPU, RAM, OS, Python and Qiskit versions).

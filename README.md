# AIRS

AIRS (Adaptation-Aware Inference and Resource Scheduling) performs normal
online inference first, estimates a label-free Batch Value, and then selects
an adaptation tier from the current value, resource availability, and budget.
The update affects future batches only. RRTTA is used as the adaptation backend.

## Complete experiment suite

From the workspace root, run all six experiment groups and generate CSV tables
plus SVG/PDF/PNG/TIFF figures:

```powershell
.\.venv\python.exe run_all_experiments.py
```

The command is resumable. A completed atomic configuration is skipped when its
`completed.json` exists. Use a new `--suite-id` when changing the protocol.
The current protocol writes to `airs_full_v7_s5_seed1` by default so results
from the earlier suites cannot be silently reused. Version 7 contains
23 atomic configurations: E4 excludes Static, and E6 compares only Credit Bank
and Fixed Quota.

The four tiers use 0/25/50/100% final update caps and 0/0/1/4 augmentations
respectively. The teacher and reliability router inspect the complete batch
before the tier cap is applied to the final gradient samples. AIRS does not
create or update prototypes and does not use a prototype loss.

Useful checks:

```powershell
.\.venv\python.exe run_all_experiments.py --smoke
.\.venv\python.exe run_all_experiments.py --smoke --dry-run
.\.venv\python.exe run_experiment.py E2 --smoke
.\.venv\python.exe run_airs.py --policy airs --budget 60 --resource-trace steady
```

Default inputs are `data/`, `checkpoints/robustbench/`, CIFAR-10-C severity 5,
10,000 examples per corruption, batch size 200, and seed 1. Results are written
under `outputs/airs/<suite-id>/`.

See `EXPERIMENT_DESIGN.md` for the exact experimental controls and metrics.

## Tests

```powershell
.\.venv\python.exe -m unittest discover -s tests -v
.\.venv\python.exe -m unittest discover -s RRTTA-main\tests -v
```

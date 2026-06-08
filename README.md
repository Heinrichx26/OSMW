# OSMW

Code and derived data for reproducing the observable-selected magic
calculations.

This repository contains reproducibility materials only. Manuscript text,
LaTeX source, submission files, manuscript PDFs, and generated figure files are
kept separately.
All paths below are relative to the repository root.

## Layout

- `src/`: simulation, validation, and plotting scripts.
- `data/`: generated CSV and JSON data tables used by the figures and checks.
- `results/`: validation files and intermediate result summaries. Generated
  figure files are written here locally and are excluded from version control.
- `raw_data_package/README.md`: description of the raw inputs and how they are
  regenerated.

## Setup

Create a Python environment and install the listed packages:

```bash
python -m pip install -r requirements.txt
```

The Munich Quantum Toolkit (MQT) Bench validation uses optional public benchmark
circuits through `mqt.bench` and `qiskit`. The other scripts run from generated
circuits and fixed random seeds.

## Quick Check

Run a small validation pass:

```bash
python src/reproduce_all.py --quick --skip-mqt
```

Run the MQT Bench quick validation as well when `mqt.bench` and `qiskit` are
available:

```bash
python src/reproduce_all.py --quick
```

## Reproduce Results

Regenerate the non-MQT data products and figures:

```bash
python src/reproduce_all.py --skip-mqt
```

Regenerate the full data products, including MQT Bench:

```bash
python src/reproduce_all.py
```

Regenerate figure files locally from the included data tables:

```bash
python src/build_mechanism_figure.py
python src/r41_observable_dark_gap.py
python src/plot_r41.py
python src/r41_proxy_counterexamples.py
python src/r41_stress_workloads.py --plot-only
python src/r41_2d_stress_workloads.py --plot-only
python src/r41_estimation_workload_benchmark.py --plot-only
```

The full reproduction command also regenerates the observable-blind validations,
the analytic same-support positive-weight separation, and the random
same-support coefficient validation. It also runs the one- and two-dimensional
local-window calculations after quick validations.

The figure-generation scripts write local outputs to `results/`.

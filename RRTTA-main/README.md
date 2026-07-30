# RRTTA

This repository contains the research implementation of RRTTA for online
test-time adaptation on CIFAR10-C and CIFAR100-C. The same method code and
command-line interface are used for both datasets.

RRTTA estimates sample reliability from teacher predictions, routes samples
into high-, medium-, and low-reliability groups, and adapts normalization
parameters with reliability-weighted symmetric cross entropy. A conservative
prototype branch uses only high-reliability teacher features to update class
prototypes and adds a small feature-alignment loss after warm-up.

## Repository layout

```text
rrtta/method.py                         RRTTA implementation
rrtta/runner.py                         shared configuration and evaluation
run_rrtta.py                            main CIFAR-C entry point
experiments/feature_visualization.py    before/after feature visualization
experiments/routing_analysis.py         route-ratio analysis
experiments/prototype_purity.py         prototype-update purity
experiments/routing_ablation.py         routing ablation
experiments/reliability_validation.py   reliability validation
experiments/batch_size_sensitivity.py   batch-size sensitivity
```

## Installation

Python 3.9 and a CUDA-capable PyTorch installation are recommended. Install a
PyTorch build that matches the local CUDA runtime, then install the remaining
dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The original experiments used PyTorch 1.10.0, torchvision 0.11.1, NumPy
1.22.4, scikit-learn 1.0.1, and RobustBench v0.1.

## Data and checkpoints

CIFAR10-C/CIFAR100-C data and model checkpoints are intentionally not stored
in this repository. Pass their locations explicitly:

```bash
python run_rrtta.py \
  --dataset cifar10 \
  --data-dir /path/to/cifar-c-data \
  --ckpt-dir /path/to/robustbench-checkpoints
```

The defaults are `./data`, `./checkpoints`, and `./outputs`. They can also be
set with `RRTTA_DATA_DIR`, `RRTTA_CKPT_DIR`, and `RRTTA_OUTPUT_DIR`.

Default source models are `Standard` for CIFAR10-C and
`Hendrycks2020AugMix_ResNeXt` for CIFAR100-C. Override either model with
`--arch` and `--teacher-arch`.

Portable shell entry points are also provided:

```bash
PYTHON=python bash scripts/run_cifar10.sh --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
PYTHON=python bash scripts/run_cifar100.sh --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
```

## Diagnostic experiments

Each experiment supports both datasets and the same RRTTA hyperparameters:

```bash
python -m experiments.feature_visualization --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
python -m experiments.routing_analysis --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
python -m experiments.prototype_purity --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
python -m experiments.routing_ablation --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
python -m experiments.reliability_validation --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
python -m experiments.batch_size_sensitivity --dataset cifar10 --data-dir /path/to/data --ckpt-dir /path/to/checkpoints
```

Use `--dataset cifar100` for CIFAR100-C. Run any command with `--help` for all
available settings.

## Default protocol

- corruption severity: 5
- examples per corruption: 10,000
- online test batch size: 200
- adaptation batch size: 16
- teacher momentum: 0.999
- high/low reliability thresholds: 0.40/0.10
- high/medium/low loss weights: 1.0/0.75/0.0
- prototype momentum: 0.95
- prototype-alignment weight: 0.02
- prototype warm-up: 5 batches

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

Paper and citation information will be added with the manuscript release.

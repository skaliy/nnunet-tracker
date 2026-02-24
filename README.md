# nnunet-tracker

[![PyPI](https://img.shields.io/pypi/v/nnunet-tracker)](https://pypi.org/project/nnunet-tracker/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://pypi.org/project/nnunet-tracker/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Lightweight MLflow-based experiment tracking for nnU-Net. Wraps any nnU-Net trainer with automatic metric logging, artifact tracking, and cross-validation summarization, without modifying your nnU-Net installation.

## Quick Start

```bash
pip install nnunet-tracker[nnunet]
```

Set your MLflow tracking location:

```bash
export MLFLOW_TRACKING_URI=./mlruns
```

Train with tracking:

```bash
nnunet-tracker train Dataset001_BrainTumour 3d_fullres all
```

Summarize cross-validation results:

```bash
nnunet-tracker summarize --experiment Dataset001_BrainTumour
```

Export publication-ready tables:

```bash
nnunet-tracker export --experiment Dataset001_BrainTumour --format latex -o results.tex
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `nnunet-tracker train` | Wrap `nnUNetv2_train` with automatic MLflow logging |
| `nnunet-tracker summarize` | Aggregate metrics across CV folds; exits 2 if incomplete |
| `nnunet-tracker list` | List all tracked experiments and fold completion status |
| `nnunet-tracker export` | Write CV results to CSV or LaTeX booktabs table |

## Python API

```python
from nnunet_tracker import (
    create_tracked_trainer,   # Factory: wrap any nnUNetTrainer subclass
    summarize_experiment,     # Query MLflow and return CVSummary
    log_cv_summary,           # Write aggregate metrics back to MLflow
    CVSummary,                # Dataclass: fold results + aggregation
    FoldResult,               # Dataclass: per-fold metrics
    export_csv,               # Export CVSummary to CSV string or file
    export_latex,             # Export CVSummary to LaTeX booktabs table
    __version__,
)
```

Wrap a trainer and run with automatic tracking:

```python
TrackedTrainer = create_tracked_trainer(nnUNetTrainer)
trainer = TrackedTrainer(plans=plans, configuration="3d_fullres", fold=0, ...)
trainer.run_training()  # metrics, params, and artifacts logged to MLflow
```

Summarize cross-validation results:

```python
summary = summarize_experiment("Dataset001_BrainTumour")
print(summary.is_complete)          # True if all 5 folds finished
print(summary.completed_folds)      # [0, 1, 2, 3, 4]
aggregates = summary.compute_aggregate_metrics()
print(aggregates["cv_mean_fg_dice"])  # e.g. 0.8747
```

Export results for publication:

```python
csv_text = export_csv(summary)
latex_text = export_latex(summary, caption="Results", label="tab:results")

# Or write directly to file
with open("results.tex", "w", encoding="utf-8") as f:
    export_latex(summary, output=f, caption="Brain Tumour CV", label="tab:bt")
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | `./mlruns` | MLflow server or local path |
| `MLFLOW_EXPERIMENT_NAME` | Auto from dataset | Experiment grouping |
| `NNUNET_TRACKER_ENABLED` | `1` | Set to `0` to disable tracking |
| `NNUNET_TRACKER_LOG_ARTIFACTS` | `1` | Set to `0` to skip artifact logging |

nnU-Net's own environment variables (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`) are read from the environment as usual; nnunet-tracker adds no requirements beyond standard nnU-Net setup.

## How It Works

nnunet-tracker uses a **dynamic subclass factory**: `create_tracked_trainer(BaseClass)` returns a new Python class that is a subclass of `BaseClass` (so `isinstance` checks pass) with nnU-Net hook methods overridden to log metrics and artifacts to MLflow. No monkey-patching of the global nnU-Net installation is involved.

Every MLflow call is wrapped with a `@failsafe` decorator that catches all exceptions and emits a generic warning -- training is never interrupted by tracking failures.

In distributed (DDP) training, only rank 0 logs metrics. Each fold is tagged with `nnunet_tracker.fold` and `nnunet_tracker.cv_group` (e.g. `Dataset001|3d_fullres|nnUNetPlans`) so fold runs can be grouped and queried precisely.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (Python 3.10, 3.11, 3.12)
pytest tests/ -v --tb=short --cov=nnunet_tracker

# Lint and format check
ruff check src/ tests/
ruff format --check src/ tests/
```

## Requirements

- Python >= 3.10
- MLflow >= 2.10.0, < 3.0.0
- nnU-Net v2 >= 2.2, < 2.7 (optional, for training)

## Documentation

### Guides

- [Getting Started](docs/getting-started.md) -- Installation, setup, and first training run
- [Configuration](docs/configuration.md) -- Environment variables and `.env` examples

### Reference

- [CLI Reference](docs/cli-reference.md) -- All commands, options, and exit codes
- [Python API](docs/python-api.md) -- Factory, summarization, export, and dataclasses

### Examples

Runnable scripts in [`examples/`](examples/) covering the factory pattern, CV summarization, and direct MLflow queries.

## License

Apache-2.0

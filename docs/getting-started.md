# Getting Started

## Prerequisites

- Python >= 3.10
- MLflow >= 2.10.0
- (Optional) nnU-Net v2 >= 2.2, < 2.7

## Installation

Install from PyPI:

```bash
pip install nnunet-tracker
```

Or install from Git:

```bash
pip install git+https://github.com/sathiesh/nnunet-tracker.git
```

### With nnU-Net

To install nnunet-tracker together with a compatible nnU-Net version:

```bash
pip install nnunet-tracker[nnunet]
```

This installs `nnunetv2>=2.2,<2.7` alongside the tracker.

### Development Install

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/sathiesh/nnunet-tracker.git
cd nnunet-tracker
pip install -e ".[dev]"
```

Dev dependencies include `pytest`, `pytest-cov`, and `ruff`.

## Verify Installation

```bash
nnunet-tracker --version
```

Expected output:

```
nnunet-tracker 0.1.0
```

## Quick Start

### 1. Set Environment Variables

Configure MLflow and nnU-Net paths before running training:

```bash
# MLflow configuration
export MLFLOW_TRACKING_URI="./mlruns"
export MLFLOW_EXPERIMENT_NAME="my-segmentation-experiment"

# nnU-Net paths (required by nnU-Net itself)
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### 2. Run Training with Tracking

```bash
nnunet-tracker train Dataset001 3d_fullres 0
```

This command:

1. Resolves the `nnUNetTrainer` class from your nnU-Net installation.
2. Wraps it with MLflow tracking hooks using the dynamic subclass factory.
3. Starts an MLflow run, logging parameters, metrics, and artifacts automatically.
4. Runs the standard nnU-Net training pipeline for fold 0 of `Dataset001` using the `3d_fullres` configuration.

To train all 5 folds sequentially:

```bash
nnunet-tracker train Dataset001 3d_fullres all
```

### 3. View Results

After training completes, view your experiment in the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Or summarize cross-validation results from the command line:

```bash
nnunet-tracker summarize -e my-segmentation-experiment
```

### 4. Using the Python API

You can also use nnunet-tracker programmatically:

```python
from nnunet_tracker import create_tracked_trainer
from nnunet_tracker.config import TrackerConfig

# The factory wraps any nnUNetTrainer subclass
TrackedTrainer = create_tracked_trainer(nnUNetTrainer)

# Use TrackedTrainer exactly like the original class
trainer = TrackedTrainer(
    plans=plans,
    configuration="3d_fullres",
    fold=0,
    dataset_json=dataset_json,
    device=device,
)
trainer.run_training()
```

See [python-api.md](python-api.md) for the full API reference and [configuration.md](configuration.md) for all environment variables.

---

## Next Steps

Now that you have nnunet-tracker installed and running, here are the recommended next steps:

- **Train all 5 folds** -- Run `nnunet-tracker train Dataset001 3d_fullres all` to complete a full cross-validation. See the [CLI Reference](cli-reference.md) for all training options.
- **Summarize results** -- Use `nnunet-tracker summarize -e <experiment>` to compute aggregate Dice scores across folds. See [CLI Reference: summarize](cli-reference.md#nnunet-tracker-summarize).
- **Export for publication** -- Generate LaTeX or CSV tables with `nnunet-tracker export`. See [CLI Reference: export](cli-reference.md#nnunet-tracker-export).
- **Configure tracking** -- Customize MLflow URI, artifact logging, and more. See [Configuration](configuration.md).
- **Use the Python API** -- For programmatic access, see the [Python API Reference](python-api.md) and the [examples/](../examples/) directory.

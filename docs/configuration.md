# Configuration

nnunet-tracker is configured entirely through environment variables. There are no configuration files to manage. All settings are read at runtime by `TrackerConfig.from_env()`.

## Tracker Environment Variables

### `MLFLOW_TRACKING_URI`

- **Default:** `./mlruns`
- **Description:** MLflow tracking server URI or local directory path. This is the standard MLflow environment variable.

Local file-based tracking (recommended for single-machine training):

```bash
export MLFLOW_TRACKING_URI="./mlruns"
export MLFLOW_TRACKING_URI="/data/experiments/mlruns"
```

Remote tracking server:

```bash
export MLFLOW_TRACKING_URI="https://mlflow.example.com"
```

**Security note:** Using an `http://` (non-TLS) tracking URI will emit a warning at run start. In hospital and clinical environments, use HTTPS or local file-based tracking to avoid transmitting experiment data over unencrypted connections.

### `MLFLOW_EXPERIMENT_NAME`

- **Default:** Auto-detected from the nnU-Net dataset name (e.g., `Dataset001_BrainTumour`)
- **Description:** MLflow experiment name used to group runs. This is the standard MLflow environment variable.

```bash
export MLFLOW_EXPERIMENT_NAME="my-segmentation-experiment"
```

When not set, nnunet-tracker reads the dataset name from the trainer's `plans_manager.dataset_name` attribute and uses that as the experiment name. This means each nnU-Net dataset gets its own MLflow experiment by default.

### `NNUNET_TRACKER_ENABLED`

- **Default:** `1` (enabled)
- **Description:** Master switch for tracking. Set to `0`, `false`, or `no` (case-insensitive) to disable all MLflow logging. The trainer will run as if unwrapped.

```bash
# Disable tracking
export NNUNET_TRACKER_ENABLED=0
```

When disabled, `create_tracked_trainer()` returns the original base class unmodified. If passed an already-tracked class, it unwraps back to the original base. This makes it safe to toggle tracking without changing any other code.

### `NNUNET_TRACKER_LOG_ARTIFACTS`

- **Default:** `1` (enabled)
- **Description:** Controls whether training artifacts are logged to MLflow at the end of a run. Set to `0`, `false`, or `no` to skip artifact logging.

```bash
# Disable artifact logging (still logs metrics and parameters)
export NNUNET_TRACKER_LOG_ARTIFACTS=0
```

When enabled, the following artifacts are logged at the end of training:

| Artifact | MLflow Path | Description |
|----------|-------------|-------------|
| `checkpoint_final.pth` | `checkpoints/` | Final model checkpoint |
| `checkpoint_best.pth` | `checkpoints/` | Best model checkpoint |
| `progress.png` | `plots/` | Training progress plot |
| `plans.json` | `config/` | Full nnU-Net plans JSON |
| `dataset_fingerprint.json` | `config/` | Dataset fingerprint file |

Disabling artifact logging is useful when storage is limited or when running many experimental iterations where only metrics matter.

## nnU-Net Environment Variables

These are standard nnU-Net environment variables required for training. nnunet-tracker does not modify them but requires them to be set for the `train` command.

### `nnUNet_raw`

- **Description:** Path to the raw nnU-Net dataset directory.

```bash
export nnUNet_raw="/data/nnUNet_raw"
```

### `nnUNet_preprocessed`

- **Description:** Path to the preprocessed nnU-Net dataset directory. The `train` command reads plans and dataset JSON files from here.

```bash
export nnUNet_preprocessed="/data/nnUNet_preprocessed"
```

### `nnUNet_results`

- **Description:** Path to the nnU-Net results directory. Used for artifact path containment validation (prevents path traversal when logging artifacts).

```bash
export nnUNet_results="/data/nnUNet_results"
```

## Boolean Parsing

All boolean environment variables (`NNUNET_TRACKER_ENABLED`, `NNUNET_TRACKER_LOG_ARTIFACTS`) accept the following values (case-insensitive):

| Truthy | Falsy |
|--------|-------|
| `1` | `0` |
| `true` | `false` |
| `yes` | `no` |

Any value not in the truthy set is treated as `false`.

## Example `.env` File

A typical configuration for a research environment:

```bash
# MLflow
MLFLOW_TRACKING_URI=./mlruns
MLFLOW_EXPERIMENT_NAME=cardiac-segmentation

# nnunet-tracker
NNUNET_TRACKER_ENABLED=1
NNUNET_TRACKER_LOG_ARTIFACTS=1

# nnU-Net paths
nnUNet_raw=/data/nnUNet_raw
nnUNet_preprocessed=/data/nnUNet_preprocessed
nnUNet_results=/data/nnUNet_results
```

A minimal configuration for quick local experimentation (artifact logging off, auto-detect experiment name):

```bash
MLFLOW_TRACKING_URI=./mlruns
NNUNET_TRACKER_LOG_ARTIFACTS=0
nnUNet_raw=/data/nnUNet_raw
nnUNet_preprocessed=/data/nnUNet_preprocessed
nnUNet_results=/data/nnUNet_results
```

## Programmatic Access

Configuration can also be read and passed explicitly in Python:

```python
from nnunet_tracker.config import TrackerConfig

# Read from environment
config = TrackerConfig.from_env()

# Or construct explicitly
config = TrackerConfig(
    tracking_uri="https://mlflow.example.com",
    experiment_name="my-experiment",
    enabled=True,
    log_artifacts=True,
)
```

See [python-api.md](python-api.md) for how to pass config to `create_tracked_trainer()`.

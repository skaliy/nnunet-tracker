# CLI Reference

nnunet-tracker provides a command-line interface with four subcommands: `train`, `summarize`, `list`, and `export`.

```
nnunet-tracker [--version] [--help] <command> [<args>]
```

## Global Options

| Option | Description |
|--------|-------------|
| `--version` | Print version and exit |
| `--help` | Print help and exit |

---

## `nnunet-tracker train`

Run nnU-Net training with automatic MLflow experiment tracking. Wraps `nnUNetv2_train` by resolving the trainer class, applying the tracking subclass factory, and executing the standard training pipeline.

### Usage

```bash
nnunet-tracker train [--tracker-disable] DATASET CONFIGURATION FOLD [-tr TRAINER] [-p PLANS] [-c]
```

### Positional Arguments

| Argument | Description |
|----------|-------------|
| `dataset_name_or_id` | nnU-Net dataset name or numeric ID (e.g., `Dataset001_BrainTumour` or `1`) |
| `configuration` | nnU-Net configuration (e.g., `2d`, `3d_fullres`, `3d_lowres`, `3d_cascade_fullres`) |
| `fold` | Fold number (0-4) or `all` to train all 5 folds sequentially |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `-tr` | `nnUNetTrainer` | Trainer class name. Must be a valid Python identifier. |
| `-p` | `nnUNetPlans` | Plans identifier. The command loads `<plans>.json` from the preprocessed directory. |
| `-c` | `false` | Continue training from the latest checkpoint. Looks for `checkpoint_final.pth` first, then `checkpoint_latest.pth`. |
| `-pretrained_weights` | `None` | Path to pretrained weights file for transfer learning. |
| `-num_gpus` | `1` | Number of GPUs (only 1 supported; use `nnUNetv2_train` directly for multi-GPU). |
| `--npz` | `false` | Save softmax predictions as `.npz` from validation. |
| `--val` | `false` | Validation-only mode (skip training, requires completed training). |
| `--val_best` | `false` | Use `checkpoint_best.pth` for validation instead of `checkpoint_final.pth`. |
| `--use_compressed` | `false` | Do not decompress training cases. More CPU/RAM intensive. |
| `--disable_checkpointing` | `false` | Disable checkpoint saving during training. |
| `-device` | `cuda` | Device: `cuda`, `cpu`, or `mps`. |
| `--tracker-disable` | `false` | Disable MLflow tracking for this run. The trainer runs as if unwrapped. |

### Behavior

- When `fold` is set to `all`, training runs sequentially for folds 0 through 4. Each fold gets its own MLflow run.
- Trainer and plans names are validated against `^[A-Za-z_][A-Za-z0-9_]*$`. Invalid names are rejected before training starts.
- If training crashes, the MLflow run is automatically marked as `FAILED` to prevent orphaned runs.
- Unrecognized extra arguments are printed as a warning and ignored.

### Examples

```bash
# Train fold 0 with default trainer and plans
nnunet-tracker train Dataset001_BrainTumour 3d_fullres 0

# Train all folds
nnunet-tracker train Dataset001_BrainTumour 3d_fullres all

# Use a custom trainer class
nnunet-tracker train Dataset001_BrainTumour 3d_fullres 0 -tr nnUNetTrainerNoMirroring

# Continue from checkpoint
nnunet-tracker train Dataset001_BrainTumour 3d_fullres 2 -c

# Train without tracking
nnunet-tracker train Dataset001_BrainTumour 3d_fullres 0 --tracker-disable
```

---

## `nnunet-tracker summarize`

Query MLflow for completed fold runs and compute aggregate cross-validation metrics (mean and standard deviation of Dice scores, validation loss, etc.).

### Usage

```bash
nnunet-tracker summarize --experiment NAME [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--experiment` | `-e` | (required) | MLflow experiment name to query |
| `--tracking-uri` | | `$MLFLOW_TRACKING_URI` or `./mlruns` | MLflow tracking URI |
| `--cv-group` | | Auto-detect latest | CV group identifier to filter fold runs |
| `--folds` | | `0,1,2,3,4` | Expected fold numbers, comma-separated |
| `--json` | | `false` | Output as JSON instead of a human-readable table |
| `--log-to-mlflow` | | `false` | Create a summary run in MLflow with aggregate metrics |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | All expected folds are complete |
| `1` | Error (experiment not found, invalid arguments, etc.) |
| `2` | Partial -- some expected folds are missing |

Exit code 2 is useful for CI pipelines that check whether all folds have finished before proceeding.

### Behavior

- Fold runs are deduplicated by fold number, keeping the most recent run per fold.
- When `--cv-group` is omitted, the command auto-detects the most recent cv_group tag in the experiment.
- The `--log-to-mlflow` flag creates (or updates) a summary run tagged with `nnunet_tracker.run_type=cv_summary`. This operation is idempotent.

### Examples

```bash
# Summarize with table output
nnunet-tracker summarize -e Dataset001_BrainTumour

# JSON output for scripting
nnunet-tracker summarize -e Dataset001_BrainTumour --json

# Summarize specific cv_group and folds
nnunet-tracker summarize -e Dataset001_BrainTumour --cv-group "Dataset001|3d_fullres|nnUNetPlans" --folds 0,1,2

# Log aggregate metrics back to MLflow
nnunet-tracker summarize -e Dataset001_BrainTumour --log-to-mlflow

# Use a remote tracking server
nnunet-tracker summarize -e Dataset001_BrainTumour --tracking-uri https://mlflow.example.com
```

---

## `nnunet-tracker list`

List all MLflow experiments that contain nnunet-tracker fold runs, with fold completion status per CV group.

### Usage

```bash
nnunet-tracker list [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tracking-uri` | `$MLFLOW_TRACKING_URI` or `./mlruns` | MLflow tracking URI |
| `--max-experiments` | `50` | Maximum number of experiments to query |
| `--json` | `false` | Output as JSON instead of a table |

### Output Columns (Table Mode)

| Column | Description |
|--------|-------------|
| Experiment | MLflow experiment name |
| CV Group | Cross-validation group identifier |
| Folds | Number of completed folds (e.g., `3`) |
| Runs | Total fold runs in the experiment |

### Examples

```bash
# List experiments with table output
nnunet-tracker list

# JSON output
nnunet-tracker list --json

# List from a remote server
nnunet-tracker list --tracking-uri https://mlflow.example.com
```

---

## `nnunet-tracker export`

Export cross-validation results as publication-ready CSV or LaTeX tables.

### Usage

```bash
nnunet-tracker export --experiment NAME --format FORMAT [OPTIONS]
```

### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--experiment` | `-e` | (required) | MLflow experiment name to export |
| `--format` | `-f` | (required) | Output format: `csv` or `latex` |
| `--output` | `-o` | stdout | Output file path. If omitted, writes to stdout. |
| `--tracking-uri` | | `$MLFLOW_TRACKING_URI` or `./mlruns` | MLflow tracking URI |
| `--cv-group` | | Auto-detect latest | CV group identifier |
| `--folds` | | `0,1,2,3,4` | Expected fold numbers, comma-separated |
| `--caption` | | Auto-generated from cv_group | Table caption (LaTeX only) |
| `--label` | | `tab:cv_results` | Table label for cross-referencing (LaTeX only) |

### CSV Output

The CSV includes columns for Fold, Mean FG Dice, Val Loss, EMA FG Dice, and per-class Dice scores. Aggregate mean and standard deviation rows are appended at the bottom.

### LaTeX Output

Generates a `booktabs`-style table wrapped in a `table` environment with `\toprule`, `\midrule`, and `\bottomrule`. The aggregate row combines mean and standard deviation as `mean +/- std`. LaTeX special characters in captions and data are automatically escaped.

### Examples

```bash
# Export to CSV file
nnunet-tracker export -e Dataset001_BrainTumour -f csv -o results.csv

# Export LaTeX table to stdout
nnunet-tracker export -e Dataset001_BrainTumour -f latex

# LaTeX with custom caption and label
nnunet-tracker export -e Dataset001_BrainTumour -f latex \
    --caption "5-fold cross-validation on BrainTumour" \
    --label "tab:brain_tumour_cv" \
    -o table.tex

# Export specific folds
nnunet-tracker export -e Dataset001_BrainTumour -f csv --folds 0,1,2
```

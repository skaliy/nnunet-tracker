# Python API Reference

All public symbols are importable from the top-level `nnunet_tracker` package.

```python
from nnunet_tracker import (
    create_tracked_trainer,
    summarize_experiment,
    log_cv_summary,
    export_csv,
    export_latex,
    CVSummary,
    FoldResult,
)
```

---

## `create_tracked_trainer`

```python
def create_tracked_trainer(
    base_class: type,
    config: TrackerConfig | None = None,
) -> type
```

Factory function that creates a dynamically subclassed trainer with MLflow tracking hooks. The returned class is a subclass of `base_class` and can be used as a drop-in replacement.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_class` | `type` | (required) | The nnU-Net trainer class to wrap. Must be a class with a `run_training` method. |
| `config` | `TrackerConfig \| None` | `None` | Tracker configuration. If `None`, reads from environment variables via `TrackerConfig.from_env()`. |

**Returns:** A new class that is a subclass of `base_class` with tracking hooks injected.

**Raises:**

- `TypeError` -- if `base_class` is not a class.
- `ValueError` -- if `base_class` does not have a `run_training` method.

**Behavior:**

- Double-wrapping is prevented. If `base_class` is already tracked (has the `_nnunet_tracker_wrapped` sentinel), the same class is returned unchanged.
- If `config.enabled` is `False`, returns the original `base_class` unmodified. If passed an already-tracked class with tracking disabled, it unwraps back to the original base.
- Tracking state (`_mlflow_run_id`) is stored per instance, not per class, to isolate state across sequential multi-fold training.
- If training crashes, the MLflow run is automatically marked as `FAILED`.

**Example:**

```python
from nnunet_tracker import create_tracked_trainer
from nnunet_tracker.config import TrackerConfig

# Using environment variables
TrackedTrainer = create_tracked_trainer(nnUNetTrainer)

# With explicit configuration
config = TrackerConfig(
    tracking_uri="https://mlflow.example.com",
    experiment_name="my-experiment",
    enabled=True,
    log_artifacts=True,
)
TrackedTrainer = create_tracked_trainer(nnUNetTrainer, config=config)

# The returned class works exactly like the original
trainer = TrackedTrainer(
    plans=plans,
    configuration="3d_fullres",
    fold=0,
    dataset_json=dataset_json,
    device=device,
)
trainer.run_training()
```

---

## `summarize_experiment`

```python
def summarize_experiment(
    experiment_name: str,
    tracking_uri: str = "./mlruns",
    cv_group: str | None = None,
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> CVSummary
```

Query MLflow for completed fold runs and build a cross-validation summary.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `experiment_name` | `str` | (required) | MLflow experiment name to query. |
| `tracking_uri` | `str` | `"./mlruns"` | MLflow tracking URI. |
| `cv_group` | `str \| None` | `None` | CV group identifier. If `None`, auto-detects the most recent group. |
| `expected_folds` | `tuple[int, ...]` | `(0, 1, 2, 3, 4)` | Expected fold numbers for completeness checking. |

**Returns:** A `CVSummary` instance with fold results and aggregate metrics.

**Raises:**

- `ValueError` -- if the experiment is not found or no fold runs are found.

**Behavior:**

- Fold runs are deduplicated by fold number. When multiple runs exist for the same fold, the most recent run is kept.
- When `cv_group` is `None`, the function auto-detects the latest cv_group tag. This supports backward compatibility with pre-v0.3.0 runs that lack cv_group tags.

**Example:**

```python
from nnunet_tracker import summarize_experiment

summary = summarize_experiment(
    experiment_name="Dataset001_BrainTumour",
    tracking_uri="./mlruns",
)

print(f"Completed folds: {summary.completed_folds}")
print(f"Missing folds: {summary.missing_folds}")
print(f"All complete: {summary.is_complete}")

aggregates = summary.compute_aggregate_metrics()
print(f"Mean Dice: {aggregates.get('cv_mean_fg_dice', 'N/A')}")
```

---

## `log_cv_summary`

```python
def log_cv_summary(
    summary: CVSummary,
    tracking_uri: str = "./mlruns",
) -> str | None
```

Create or update an MLflow summary run with aggregate cross-validation metrics.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `summary` | `CVSummary` | (required) | The cross-validation summary to log. |
| `tracking_uri` | `str` | `"./mlruns"` | MLflow tracking URI. |

**Returns:** The summary run ID as a string, or `None` on failure.

**Behavior:**

- Tags the summary run with `nnunet_tracker.run_type=cv_summary` and `nnunet_tracker.cv_group`.
- Idempotent: searches for an existing summary run with matching tags before creating a new one. Re-running with updated fold results updates the existing summary.
- Uses `MlflowClient` API exclusively to avoid mutating global MLflow state (tracking URI, active run, etc.).
- Returns `None` if `cv_group` is empty or contains unsafe characters.

**Example:**

```python
from nnunet_tracker import summarize_experiment, log_cv_summary

summary = summarize_experiment("Dataset001_BrainTumour")

run_id = log_cv_summary(summary, tracking_uri="./mlruns")
if run_id:
    print(f"Summary logged as run: {run_id}")
```

---

## `export_csv`

```python
def export_csv(
    summary: CVSummary,
    output: TextIO | None = None,
) -> str
```

Export cross-validation results as a CSV string.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `summary` | `CVSummary` | (required) | The cross-validation summary to export. |
| `output` | `TextIO \| None` | `None` | Optional file-like object to write to. If `None`, only returns the string. |

**Returns:** The CSV content as a string (always returned, also written to `output` if provided).

**Output format:** Columns are Fold, Mean FG Dice, Val Loss, EMA FG Dice, and one column per class for per-class Dice. Aggregate mean and standard deviation rows are appended.

**Example:**

```python
from nnunet_tracker import summarize_experiment, export_csv

summary = summarize_experiment("Dataset001_BrainTumour")

# Write to file
with open("results.csv", "w") as f:
    export_csv(summary, output=f)

# Or get as string
csv_str = export_csv(summary)
```

---

## `export_latex`

```python
def export_latex(
    summary: CVSummary,
    output: TextIO | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str
```

Export cross-validation results as a LaTeX booktabs table.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `summary` | `CVSummary` | (required) | The cross-validation summary to export. |
| `output` | `TextIO \| None` | `None` | Optional file-like object to write to. |
| `caption` | `str \| None` | `None` | Table caption. Defaults to `"Cross-validation results for {cv_group}"`. |
| `label` | `str \| None` | `None` | Table label for `\ref{}`. Defaults to `"tab:cv_results"`. |

**Returns:** The LaTeX content as a string (always returned, also written to `output` if provided).

**Output format:** A `table` environment with `\toprule`, `\midrule`, `\bottomrule` (requires the `booktabs` LaTeX package). The aggregate row shows mean +/- std for each metric. LaTeX special characters are automatically escaped.

**Example:**

```python
from nnunet_tracker import summarize_experiment, export_latex

summary = summarize_experiment("Dataset001_BrainTumour")

latex_str = export_latex(
    summary,
    caption="5-fold cross-validation on BrainTumour",
    label="tab:brain_cv",
)

with open("table.tex", "w") as f:
    f.write(latex_str)
```

---

## `CVSummary`

```python
@dataclass
class CVSummary:
    cv_group: str
    experiment_name: str
    fold_results: dict[int, FoldResult]
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4)
```

Cross-validation summary aggregating results across folds.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `cv_group` | `str` | CV group identifier (e.g., `"Dataset001\|3d_fullres\|nnUNetPlans"`) |
| `experiment_name` | `str` | MLflow experiment name |
| `fold_results` | `dict[int, FoldResult]` | Mapping from fold number to its result |
| `expected_folds` | `tuple[int, ...]` | Fold numbers expected for completeness (default: 0-4) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `completed_folds` | `list[int]` | Sorted list of fold numbers with FINISHED runs |
| `missing_folds` | `list[int]` | Sorted list of expected fold numbers without FINISHED runs |
| `is_complete` | `bool` | `True` if all expected folds have FINISHED runs |

### Methods

#### `compute_aggregate_metrics() -> dict[str, float]`

Compute mean and standard deviation of metrics across completed folds.

Returns a dictionary with keys following the pattern:

| Key Pattern | Description |
|-------------|-------------|
| `cv_mean_fg_dice` | Mean of mean foreground Dice across folds |
| `cv_std_fg_dice` | Standard deviation of mean foreground Dice |
| `cv_mean_val_loss` | Mean validation loss |
| `cv_std_val_loss` | Standard deviation of validation loss |
| `cv_mean_ema_fg_dice` | Mean EMA foreground Dice |
| `cv_std_ema_fg_dice` | Standard deviation of EMA foreground Dice |
| `cv_mean_dice_class_{i}` | Mean Dice for class `i` |
| `cv_std_dice_class_{i}` | Standard deviation of Dice for class `i` |

Standard deviation keys are only present when 2 or more folds have values (uses `statistics.stdev`).

---

## `FoldResult`

```python
@dataclass(frozen=True)
class FoldResult:
    fold: int
    run_id: str
    status: str
    mean_fg_dice: float | None = None
    dice_per_class: dict[int, float] = field(default_factory=dict)
    val_loss: float | None = None
    ema_fg_dice: float | None = None
```

Metrics extracted from a single fold's MLflow run. This is a frozen (immutable) dataclass.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `fold` | `int` | Fold number (e.g., 0-4) |
| `run_id` | `str` | MLflow run ID |
| `status` | `str` | MLflow run status (e.g., `"FINISHED"`, `"FAILED"`) |
| `mean_fg_dice` | `float \| None` | Mean foreground Dice coefficient |
| `dice_per_class` | `dict[int, float]` | Per-class Dice scores keyed by class index |
| `val_loss` | `float \| None` | Final validation loss |
| `ema_fg_dice` | `float \| None` | Exponential moving average foreground Dice |

---

## `TrackerConfig`

```python
from nnunet_tracker.config import TrackerConfig
```

```python
@dataclass(frozen=True)
class TrackerConfig:
    tracking_uri: str
    experiment_name: str | None
    enabled: bool
    log_artifacts: bool
```

Immutable configuration read from environment variables. See [configuration.md](configuration.md) for the full environment variable reference.

### Class Methods

#### `TrackerConfig.from_env() -> TrackerConfig`

Read configuration from environment variables and return a new `TrackerConfig` instance.

| Environment Variable | Attribute | Default |
|---------------------|-----------|---------|
| `MLFLOW_TRACKING_URI` | `tracking_uri` | `"./mlruns"` |
| `MLFLOW_EXPERIMENT_NAME` | `experiment_name` | `None` |
| `NNUNET_TRACKER_ENABLED` | `enabled` | `True` |
| `NNUNET_TRACKER_LOG_ARTIFACTS` | `log_artifacts` | `True` |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `tracking_uri` | `str` | MLflow tracking server URI or local path |
| `experiment_name` | `str \| None` | Experiment name. `None` means auto-detect from dataset. |
| `enabled` | `bool` | Whether tracking is enabled |
| `log_artifacts` | `bool` | Whether to log artifacts (checkpoints, plots, config files) |

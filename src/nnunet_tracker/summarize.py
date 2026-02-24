"""Cross-validation fold summarization for nnU-Net experiments."""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from nnunet_tracker.config import SAFE_TAG_VALUE_PATTERN

logger = logging.getLogger("nnunet_tracker")

# Upper bound for search_runs queries. Covers 200 re-runs of 5-fold CV.
# Full pagination is not needed for typical cross-validation workflows.
_MAX_FOLD_RESULTS = 1000


@dataclass(frozen=True)
class FoldResult:
    """Metrics extracted from a single fold's MLflow run.

    Attributes:
        fold: Fold number (e.g., 0-4).
        run_id: MLflow run ID.
        status: MLflow run status (e.g., ``"FINISHED"``).
        mean_fg_dice: Mean foreground Dice coefficient, or None if unavailable.
        dice_per_class: Per-class Dice scores keyed by class index.
        val_loss: Final validation loss, or None if unavailable.
        ema_fg_dice: Exponential moving average foreground Dice, or None.
    """

    fold: int
    run_id: str
    status: str
    mean_fg_dice: float | None = None
    dice_per_class: dict[int, float] = field(default_factory=dict)
    val_loss: float | None = None
    ema_fg_dice: float | None = None


@dataclass
class CVSummary:
    """Cross-validation summary across folds.

    Attributes:
        cv_group: CV group identifier (e.g., ``"Dataset001|3d_fullres|nnUNetPlans"``).
        experiment_name: MLflow experiment name.
        fold_results: Mapping from fold number to its FoldResult.
        expected_folds: Fold numbers expected for completeness checking.
    """

    cv_group: str
    experiment_name: str
    fold_results: dict[int, FoldResult] = field(default_factory=dict)
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4)

    @property
    def completed_folds(self) -> list[int]:
        """Fold numbers present in fold_results (sorted)."""
        return sorted(self.fold_results.keys())

    @property
    def missing_folds(self) -> list[int]:
        """Expected fold numbers not present in fold_results."""
        return sorted(set(self.expected_folds) - set(self.fold_results.keys()))

    @property
    def is_complete(self) -> bool:
        """True if all expected folds are present in fold_results."""
        return set(self.expected_folds) <= set(self.fold_results.keys())

    @property
    def all_class_indices(self) -> list[int]:
        """All per-class Dice indices across all folds, sorted."""
        indices: set[int] = set()
        for r in self.fold_results.values():
            indices.update(r.dice_per_class.keys())
        return sorted(indices)

    def compute_aggregate_metrics(self) -> dict[str, float]:
        """Compute mean/std of metrics across completed folds.

        Uses statistics.mean/stdev from stdlib.
        Std requires >= 2 folds; single-fold returns mean only.
        Returns keys like cv_mean_fg_dice, cv_std_fg_dice, etc.
        """
        results = list(self.fold_results.values())
        if not results:
            return {}

        metrics: dict[str, float] = {}

        fg_dice_vals = [r.mean_fg_dice for r in results if r.mean_fg_dice is not None]
        if fg_dice_vals:
            metrics["cv_mean_fg_dice"] = statistics.mean(fg_dice_vals)
            if len(fg_dice_vals) >= 2:
                metrics["cv_std_fg_dice"] = statistics.stdev(fg_dice_vals)

        val_loss_vals = [r.val_loss for r in results if r.val_loss is not None]
        if val_loss_vals:
            metrics["cv_mean_val_loss"] = statistics.mean(val_loss_vals)
            if len(val_loss_vals) >= 2:
                metrics["cv_std_val_loss"] = statistics.stdev(val_loss_vals)

        ema_vals = [r.ema_fg_dice for r in results if r.ema_fg_dice is not None]
        if ema_vals:
            metrics["cv_mean_ema_fg_dice"] = statistics.mean(ema_vals)
            if len(ema_vals) >= 2:
                metrics["cv_std_ema_fg_dice"] = statistics.stdev(ema_vals)

        for cls_idx in self.all_class_indices:
            cls_vals = [r.dice_per_class[cls_idx] for r in results if cls_idx in r.dice_per_class]
            if cls_vals:
                metrics[f"cv_mean_dice_class_{cls_idx}"] = statistics.mean(cls_vals)
                if len(cls_vals) >= 2:
                    metrics[f"cv_std_dice_class_{cls_idx}"] = statistics.stdev(cls_vals)

        return metrics


def summarize_experiment(
    experiment_name: str,
    tracking_uri: str = "./mlruns",
    cv_group: str | None = None,
    expected_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> CVSummary:
    """Query MLflow for fold runs and build a CVSummary.

    Args:
        experiment_name: MLflow experiment name to query.
        tracking_uri: MLflow tracking URI.
        cv_group: CV group identifier. If None, auto-detects the latest group.
        expected_folds: Expected fold numbers (default: 5-fold CV).

    Returns:
        CVSummary with fold results and aggregate metrics.

    Raises:
        ValueError: If experiment not found or no fold runs found.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name!r}")

    exp_id = experiment.experiment_id

    # Auto-detect cv_group if not specified
    if cv_group is None:
        cv_group = _auto_detect_cv_group(client, exp_id)

    runs = _query_fold_runs(client, exp_id, cv_group)

    if not runs:
        raise ValueError(
            f"No completed fold runs found in experiment {experiment_name!r}"
            + (f" for cv_group {cv_group!r}" if cv_group else "")
        )

    # Extract fold results, deduplicating by fold (keep most recent)
    fold_results: dict[int, FoldResult] = {}
    for run in runs:
        result = _extract_fold_result(run)
        if result is not None:
            # Keep the most recent run per fold (runs are ordered by start_time DESC)
            if result.fold not in fold_results:
                fold_results[result.fold] = result

    if not fold_results:
        raise ValueError(
            f"No fold results could be extracted from runs in experiment {experiment_name!r}"
            + (f" for cv_group {cv_group!r}" if cv_group else "")
        )

    return CVSummary(
        cv_group=cv_group or "",
        experiment_name=experiment_name,
        fold_results=fold_results,
        expected_folds=expected_folds,
    )


def _auto_detect_cv_group(client: Any, experiment_id: str) -> str | None:
    """Find the most recent cv_group tag value in the experiment."""
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.`nnunet_tracker.run_type` = 'fold'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs:
        cv_group: str | None = runs[0].data.tags.get("nnunet_tracker.cv_group")
        if cv_group is None:
            logger.debug("Most recent fold run has no cv_group tag (v0.2.0 run?)")
        return cv_group
    logger.debug("No fold runs found for cv_group auto-detection")
    return None


def _query_fold_runs(client: Any, experiment_id: str, cv_group: str | None) -> list:
    """Query fold runs by cv_group tag.

    Returns matching runs ordered by start_time DESC, or empty list if none found.
    """
    if cv_group is None:
        # No cv_group: query all fold-typed FINISHED runs
        runs = list(
            client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=(
                    "tags.`nnunet_tracker.run_type` = 'fold' and attributes.status = 'FINISHED'"
                ),
                order_by=["start_time DESC"],
                max_results=_MAX_FOLD_RESULTS,
            )
        )
        if len(runs) >= _MAX_FOLD_RESULTS:
            logger.warning(
                "Query returned %d runs (limit %d). Some fold runs may be missing. "
                "Consider filtering by --cv-group.",
                len(runs),
                _MAX_FOLD_RESULTS,
            )
        return runs

    if not SAFE_TAG_VALUE_PATTERN.fullmatch(cv_group):
        logger.warning("cv_group contains unsafe characters, skipping tag-based query")
        return []

    runs = list(
        client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.`nnunet_tracker.cv_group` = '{cv_group}' "
                "and tags.`nnunet_tracker.run_type` = 'fold' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=["start_time DESC"],
            max_results=_MAX_FOLD_RESULTS,
        )
    )
    if len(runs) >= _MAX_FOLD_RESULTS:
        logger.warning(
            "Query returned %d runs (limit %d). Some fold runs may be missing.",
            len(runs),
            _MAX_FOLD_RESULTS,
        )
    return runs


def _extract_fold_result(run: Any) -> FoldResult | None:
    """Extract metrics from an MLflow Run object into a FoldResult.

    Returns None if fold number cannot be determined.
    """
    tags = run.data.tags
    params = run.data.params
    metrics = run.data.metrics

    # Determine fold number: prefer tag, fall back to param
    fold_str = tags.get("nnunet_tracker.fold") or params.get("fold")
    if fold_str is None:
        return None
    try:
        fold = int(fold_str)
    except (ValueError, TypeError):
        return None

    dice_per_class: dict[int, float] = {}
    for key, value in metrics.items():
        if key.startswith("dice_class_"):
            try:
                cls_idx = int(key[len("dice_class_") :])
                val = float(value)
                if math.isfinite(val):
                    dice_per_class[cls_idx] = val
            except (ValueError, TypeError):
                continue

    def _safe_float(key: str) -> float | None:
        val = metrics.get(key)
        if val is None:
            return None
        result = float(val)
        return result if math.isfinite(result) else None

    return FoldResult(
        fold=fold,
        run_id=run.info.run_id,
        status=run.info.status,
        mean_fg_dice=_safe_float("mean_fg_dice"),
        dice_per_class=dice_per_class,
        val_loss=_safe_float("val_loss"),
        ema_fg_dice=_safe_float("ema_fg_dice"),
    )


def log_cv_summary(
    summary: CVSummary,
    tracking_uri: str = "./mlruns",
) -> str | None:
    """Create/update an MLflow summary run with aggregate metrics.

    Tags the run with nnunet_tracker.run_type = 'cv_summary'.
    Idempotent: searches for existing summary run before creating new one.
    Uses MlflowClient API exclusively to avoid mutating global MLflow state.
    Returns the summary run_id, or None on failure.
    """
    if not summary.cv_group:
        logger.warning("Cannot log CV summary: cv_group is empty")
        return None

    if not SAFE_TAG_VALUE_PATTERN.fullmatch(summary.cv_group):
        logger.warning("Cannot log CV summary: cv_group contains unsafe characters")
        return None

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)

        experiment = client.get_experiment_by_name(summary.experiment_name)
        if experiment is None:
            return None

        exp_id = experiment.experiment_id
        existing_runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=(
                f"tags.`nnunet_tracker.cv_group` = '{summary.cv_group}' "
                "and tags.`nnunet_tracker.run_type` = 'cv_summary'"
            ),
            max_results=1,
        )

        if existing_runs:
            run_id = existing_runs[0].info.run_id
        else:
            run = client.create_run(
                experiment_id=exp_id,
                run_name=f"{summary.cv_group}_summary",
            )
            run_id = run.info.run_id

        # Set tags (idempotent — tags are overwritable)
        client.set_tag(run_id, "nnunet_tracker.run_type", "cv_summary")
        client.set_tag(run_id, "nnunet_tracker.cv_group", summary.cv_group)

        aggregates = summary.compute_aggregate_metrics()
        if aggregates:
            import time

            from mlflow.entities import Metric

            ts = int(time.time() * 1000)
            batch = [Metric(key=k, value=v, timestamp=ts, step=0) for k, v in aggregates.items()]
            client.log_batch(run_id, metrics=batch)

        # Log summary info as tags (not params) since they change on re-runs
        client.set_tag(run_id, "nnunet_tracker.completed_folds", str(summary.completed_folds))
        client.set_tag(run_id, "nnunet_tracker.missing_folds", str(summary.missing_folds))
        client.set_tag(run_id, "nnunet_tracker.num_completed", str(len(summary.completed_folds)))
        client.set_tag(run_id, "nnunet_tracker.is_complete", str(summary.is_complete))

        client.set_terminated(run_id, status="FINISHED")

        return str(run_id)

    except Exception:
        logger.debug("Failed to log CV summary", exc_info=True)
        return None

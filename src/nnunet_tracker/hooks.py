"""MLflow logging hook implementations for nnU-Net trainers."""

from __future__ import annotations

import functools
import logging
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from nnunet_tracker.config import SAFE_TAG_VALUE_PATTERN, TrackerConfig
from nnunet_tracker.ddp import should_log

logger = logging.getLogger("nnunet_tracker")

# Foreground-mean metrics from summary.json -> MLflow metric names.
_FOREGROUND_MEAN_METRICS = {"Dice": "final_val_mean_fg_dice", "IoU": "final_val_mean_fg_iou"}
# Per-class metric prefixes from summary.json 'mean' section.
_PER_CLASS_METRIC_PREFIXES = {"Dice": "final_val_dice_class_", "IoU": "final_val_iou_class_"}


def _should_track(trainer: Any, config: TrackerConfig) -> bool:
    """Check if tracking should proceed for this trainer and config."""
    return config.enabled and should_log(trainer)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def failsafe(func: Callable[_P, _R]) -> Callable[_P, _R | None]:
    """Decorator that catches all exceptions in tracking calls.

    Logs errors as warnings but never raises, ensuring training
    is never interrupted by a tracking failure.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
        try:
            return func(*args, **kwargs)
        except MemoryError:
            raise
        except Exception:
            warnings.warn(
                f"nnunet-tracker: {func.__name__} failed. Enable debug logging for details.",
                UserWarning,
                stacklevel=2,
            )
            logger.debug("Tracking error in %s", func.__name__, exc_info=True)
            return None

    return wrapper


@failsafe
def log_run_start(trainer: Any, config: TrackerConfig) -> str | None:
    """Start an MLflow run and log initial parameters.

    Called from on_train_start() AFTER super().on_train_start().
    """
    if not _should_track(trainer, config):
        return None

    import mlflow

    mlflow.set_tracking_uri(config.tracking_uri)

    if not config.tracking_uri.startswith(("http://", "https://")):
        logger.debug(
            "MLflow tracking URI resolved to: %s",
            os.path.abspath(config.tracking_uri),
        )

    if config.tracking_uri.startswith("http://"):
        warnings.warn(
            "nnunet-tracker: MLflow tracking URI uses unencrypted HTTP. "
            "Use HTTPS or local file-based tracking for clinical environments.",
            UserWarning,
            stacklevel=2,
        )

    experiment_name = config.experiment_name
    if experiment_name is None:
        experiment_name = getattr(getattr(trainer, "plans_manager", None), "dataset_name", "nnunet")
    mlflow.set_experiment(experiment_name)

    dataset_name = getattr(getattr(trainer, "plans_manager", None), "dataset_name", "unknown")
    fold = getattr(trainer, "fold", "?")
    configuration = getattr(trainer, "configuration_name", None) or getattr(
        trainer, "configuration", "?"
    )
    if fold == "all":
        run_name = f"{dataset_name}_all_{configuration}"
    else:
        run_name = f"{dataset_name}_fold{fold}_{configuration}"

    # Guard against double-call or stale run from a previous fold
    existing_run = mlflow.active_run()
    if existing_run is not None:
        existing_id = existing_run.info.run_id
        trainer_run_id = getattr(trainer, "_mlflow_run_id", None)
        if trainer_run_id == existing_id:
            # Genuine double-call for same fold — skip
            logger.debug("MLflow run already active (%s), skipping start_run", existing_id)
            return str(existing_id)
        # Foreign/stale run detected — do NOT end it (could be user's outer run).
        # Log a warning and skip starting a new run to avoid destroying external state.
        logger.warning(
            "An MLflow run %s is already active but does not belong to this trainer. "
            "nnunet-tracker will not start a new run. End the existing run first, "
            "or use nested runs.",
            existing_id,
        )
        return None

    active_run = mlflow.start_run(run_name=run_name)
    run_id: str = active_run.info.run_id

    try:
        cv_tags = _build_cv_tags(trainer)
        if cv_tags:
            mlflow.set_tags(cv_tags)
    except Exception:
        logger.debug("Failed to set fold tags for run %s", run_id, exc_info=True)

    # Log params separately so a param failure doesn't lose the run_id
    try:
        params = _extract_params(trainer)
        if params:
            mlflow.log_params(params)
    except Exception:
        logger.debug("Failed to log params for run %s", run_id, exc_info=True)

    return run_id


def _extract_params(trainer: Any) -> dict[str, str | int | float]:
    """Extract training parameters from the trainer for MLflow logging."""
    params: dict[str, Any] = {}

    params["num_epochs"] = getattr(trainer, "num_epochs", None)
    params["current_epoch"] = getattr(trainer, "current_epoch", 0)
    params["fold"] = getattr(trainer, "fold", None)

    # Use stored original class name if available (set by factory),
    # otherwise walk to first base class
    original_name = getattr(type(trainer), "_original_trainer_class_name", None)
    if original_name is not None:
        params["trainer_class"] = original_name
    else:
        base_classes = type(trainer).__bases__
        if base_classes:
            params["trainer_class"] = base_classes[0].__name__

    cm = getattr(trainer, "configuration_manager", None)
    if cm is not None:
        params["batch_size"] = getattr(cm, "batch_size", None)
        patch_size = getattr(cm, "patch_size", None)
        if patch_size is not None:
            params["patch_size"] = str(patch_size)
        spacing = getattr(cm, "spacing", None)
        if spacing is not None:
            params["spacing"] = str(spacing)
        # nnU-Net v2 uses UNet_class_name on ConfigurationManager
        params["network_arch"] = getattr(cm, "UNet_class_name", None) or getattr(
            cm, "network_arch_class_name", None
        )

    pm = getattr(trainer, "plans_manager", None)
    if pm is not None:
        params["dataset_name"] = getattr(pm, "dataset_name", None)
        params["plans_name"] = getattr(pm, "plans_name", None)

    # Use trainer's initial_lr attribute (true initial value) rather than
    # optimizer's current LR which may reflect scheduler state on resume
    params["initial_lr"] = getattr(trainer, "initial_lr", None)
    params["weight_decay"] = getattr(trainer, "weight_decay", None)

    return {k: v for k, v in params.items() if v is not None}


def _build_cv_tags(trainer: Any) -> dict[str, str]:
    """Build cross-validation fold tags from trainer state.

    Returns dict of tag key-value pairs for MLflow run grouping.
    Tags are namespaced with ``nnunet_tracker.`` to avoid collision.
    Returns empty dict if fold is None.
    """
    fold = getattr(trainer, "fold", None)
    if fold is None:
        return {}

    dataset_name = getattr(getattr(trainer, "plans_manager", None), "dataset_name", None)
    plans_name = getattr(getattr(trainer, "plans_manager", None), "plans_name", None)
    configuration = getattr(trainer, "configuration_name", None) or getattr(
        trainer, "configuration", None
    )

    run_type = "final" if fold == "all" else "fold"
    tags: dict[str, str] = {
        "nnunet_tracker.fold": str(fold),
        "nnunet_tracker.run_type": run_type,
    }

    parts = [p for p in (dataset_name, configuration, plans_name) if p is not None]
    if parts:
        cv_group = "|".join(parts)
        if SAFE_TAG_VALUE_PATTERN.fullmatch(cv_group):
            tags["nnunet_tracker.cv_group"] = cv_group
        else:
            logger.debug("cv_group contains unsafe characters, skipping tag: %r", cv_group)

    return tags


@failsafe
def log_train_loss(trainer: Any, config: TrackerConfig) -> None:
    """Log training loss. Called from on_train_epoch_end()."""
    if not _should_track(trainer, config):
        return

    import mlflow

    log = getattr(getattr(trainer, "logger", None), "my_fantastic_logging", None)
    if log is None:
        return
    losses = log.get("train_losses", [])
    if not losses:
        return

    epoch = getattr(trainer, "current_epoch", 0)
    mlflow.log_metric("train_loss", float(losses[-1]), step=epoch)


@failsafe
def log_validation_metrics(trainer: Any, config: TrackerConfig) -> None:
    """Log validation metrics. Called from on_validation_epoch_end()."""
    if not _should_track(trainer, config):
        return

    import mlflow

    log = getattr(getattr(trainer, "logger", None), "my_fantastic_logging", None)
    if log is None:
        return

    epoch = getattr(trainer, "current_epoch", 0)
    metrics: dict[str, float] = {}

    val_losses = log.get("val_losses", [])
    if val_losses:
        metrics["val_loss"] = float(val_losses[-1])

    dice_per_class = log.get("dice_per_class_or_region", [])
    if dice_per_class:
        for i, dice_val in enumerate(dice_per_class[-1]):
            if dice_val is not None:
                metrics[f"dice_class_{i}"] = float(dice_val)

    if metrics:
        mlflow.log_metrics(metrics, step=epoch)


@failsafe
def log_epoch_end(trainer: Any, config: TrackerConfig, epoch: int | None = None) -> None:
    """Log epoch-end metrics. Called from on_epoch_end().

    Args:
        epoch: Explicit epoch number. Required because super().on_epoch_end()
            increments current_epoch before this hook runs.
    """
    if not _should_track(trainer, config):
        return

    import mlflow

    if epoch is None:
        epoch = getattr(trainer, "current_epoch", 0)

    log = getattr(getattr(trainer, "logger", None), "my_fantastic_logging", None)
    if log is None:
        return
    ema_list = log.get("ema_fg_dice", [])
    if not ema_list:
        return

    mlflow.log_metric("ema_fg_dice", float(ema_list[-1]), step=epoch)


def _is_safe_output_folder(output_folder: str) -> bool:
    """Check output_folder is within nnUNet_results to prevent path traversal."""
    nnunet_results = os.environ.get("nnUNet_results")
    if not nnunet_results:
        logger.warning(
            "nnUNet_results not set; cannot verify artifact path safety, skipping artifacts"
        )
        return False
    real_output = Path(output_folder).resolve()
    real_base = Path(nnunet_results).resolve()
    if not real_output.is_relative_to(real_base):
        logger.warning(
            "output_folder %r is outside nnUNet_results (%r); skipping artifact logging",
            output_folder,
            nnunet_results,
        )
        return False
    return True


def _sanitize_label_key(key: str) -> str:
    """Convert summary.json label key to MLflow-safe metric name component."""
    import re as _re

    return _re.sub(r"[^A-Za-z0-9_]", "", key.replace(", ", "_").replace(",", "_"))


@failsafe
def log_run_end(trainer: Any, config: TrackerConfig) -> None:
    """End the MLflow run and optionally log artifacts.

    Called from on_train_end() AFTER super().on_train_end().

    Note:
        When ``log_artifacts=True``, checkpoint files (``checkpoint_final.pth``,
        ``checkpoint_best.pth``) are uploaded synchronously. For 3D medical
        imaging models these can be several GB, resulting in significant upload
        time with remote tracking servers. Set ``NNUNET_TRACKER_LOG_ARTIFACTS=0``
        to skip artifact uploads.
    """
    if not _should_track(trainer, config):
        return

    import mlflow

    if config.log_artifacts:
        output_folder = getattr(trainer, "output_folder", None)
        if output_folder is not None and _is_safe_output_folder(output_folder):
            for filename, artifact_path in (
                ("checkpoint_final.pth", "checkpoints"),
                ("checkpoint_best.pth", "checkpoints"),
                ("progress.png", "plots"),
            ):
                filepath = os.path.join(output_folder, filename)
                if os.path.isfile(filepath):
                    mlflow.log_artifact(filepath, artifact_path=artifact_path)

    active = mlflow.active_run()
    trainer_run_id = getattr(trainer, "_mlflow_run_id", None)
    if active is not None and trainer_run_id is not None and active.info.run_id == trainer_run_id:
        mlflow.end_run()


@failsafe
def log_validation_summary(trainer: Any, config: TrackerConfig) -> None:
    """Log post-training validation metrics from summary.json to MLflow.

    Called from perform_actual_validation() AFTER super() completes.
    Uses MlflowClient to write to the (already finished) run.
    """
    if not _should_track(trainer, config):
        return

    run_id = getattr(trainer, "_mlflow_run_id", None)
    if run_id is None:
        return

    output_folder = getattr(trainer, "output_folder", None)
    if output_folder is None:
        return

    summary_path = os.path.join(output_folder, "validation", "summary.json")
    if not os.path.isfile(summary_path):
        return

    import json
    import math
    import time

    from mlflow.entities import Metric
    from mlflow.tracking import MlflowClient

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    metrics_dict: dict[str, float] = {}

    # Foreground-mean metrics (only Dice and IoU).
    fg_mean = summary.get("foreground_mean")
    if isinstance(fg_mean, dict):
        for json_key, mlflow_key in _FOREGROUND_MEAN_METRICS.items():
            val = fg_mean.get(json_key)
            if isinstance(val, (int, float)) and math.isfinite(val):
                metrics_dict[mlflow_key] = float(val)

    # Per-class metrics from 'mean' section.
    mean_section = summary.get("mean")
    if isinstance(mean_section, dict):
        for label_key, label_metrics in mean_section.items():
            if not isinstance(label_metrics, dict):
                continue
            safe_key = _sanitize_label_key(str(label_key))
            if not safe_key:
                continue
            for json_key, prefix in _PER_CLASS_METRIC_PREFIXES.items():
                val = label_metrics.get(json_key)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    metrics_dict[prefix + safe_key] = float(val)

    if not metrics_dict:
        return

    client = MlflowClient(tracking_uri=config.tracking_uri)
    ts = int(time.time() * 1000)
    batch = [Metric(key=k, value=v, timestamp=ts, step=0) for k, v in metrics_dict.items()]
    client.log_batch(run_id, metrics=batch)

    # Optionally log summary.json as artifact.
    if config.log_artifacts and _is_safe_output_folder(output_folder):
        client.log_artifact(run_id, summary_path, artifact_path="validation")


@failsafe
def log_fingerprint(trainer: Any, config: TrackerConfig) -> None:
    """Log dataset fingerprint hashes as MLflow params.

    Called from on_train_start() AFTER log_run_start().
    Each fingerprint component is independently computed so a failure
    in one does not prevent others from being logged.
    """
    if not _should_track(trainer, config):
        return

    import mlflow

    from nnunet_tracker.fingerprint import compute_fingerprint

    hashes = compute_fingerprint(trainer)
    if hashes:
        mlflow.log_params(hashes)


@failsafe
def log_plans_and_config(trainer: Any, config: TrackerConfig) -> None:
    """Log extended plans/config params and plans artifact.

    Called from on_train_start() AFTER log_run_start().
    Logs:
      - Extended preprocessing/topology/hyperparam params
      - Full plans JSON as artifact (if log_artifacts enabled)
      - dataset_fingerprint.json as artifact (if found and log_artifacts enabled)
    """
    if not _should_track(trainer, config):
        return

    import mlflow

    from nnunet_tracker.fingerprint import find_dataset_fingerprint_path
    from nnunet_tracker.plans_extraction import (
        extract_all_extended_params,
        get_plans_json_str,
    )

    params = extract_all_extended_params(trainer)
    if params:
        try:
            mlflow.log_params(params)
        except Exception:
            logger.debug("Failed to log extended params", exc_info=True)

    # Log full plans as artifact
    if config.log_artifacts:
        plans_json = get_plans_json_str(trainer)
        if plans_json is not None:
            _log_json_artifact(plans_json, "plans.json", "config")

        # Log dataset_fingerprint.json as artifact
        fp_path = find_dataset_fingerprint_path(trainer)
        if fp_path is not None:
            try:
                mlflow.log_artifact(fp_path, artifact_path="config")
            except Exception:
                logger.debug(
                    "Failed to log dataset_fingerprint.json artifact",
                    exc_info=True,
                )


def _log_json_artifact(json_str: str, filename: str, artifact_path: str) -> None:
    """Write a JSON string to a temp file and log it as an MLflow artifact.

    Uses a temporary directory context manager for guaranteed cleanup.
    """
    import tempfile

    import mlflow

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
        mlflow.log_artifact(filepath, artifact_path=artifact_path)


def end_run_as_failed(run_id: str | None = None) -> None:
    """Mark the current MLflow run as failed and end it.

    Called when training crashes to avoid orphaned runs.
    If *run_id* is provided, only ends the run if it matches the active run,
    preventing accidental termination of a user's outer MLflow run.
    """
    try:
        import mlflow

        active = mlflow.active_run()
        if active is None:
            return
        if run_id is not None and active.info.run_id != run_id:
            return
        mlflow.end_run(status="FAILED")
    except MemoryError:
        raise
    except Exception:
        logger.debug("Failed to end MLflow run as FAILED", exc_info=True)

"""Dynamic subclass factory for creating tracked nnU-Net trainers."""

from __future__ import annotations

from nnunet_tracker._compat import check_nnunet_available, check_nnunet_version
from nnunet_tracker.config import TrackerConfig
from nnunet_tracker.hooks import (
    end_run_as_failed,
    log_epoch_end,
    log_fingerprint,
    log_learning_rate,
    log_plans_and_config,
    log_run_end,
    log_run_start,
    log_train_loss,
    log_validation_metrics,
)

# Sentinel attribute to detect already-tracked classes
_TRACKED_MARKER = "_nnunet_tracker_wrapped"


def create_tracked_trainer(
    base_class: type,
    config: TrackerConfig | None = None,
) -> type:
    """Create a dynamically subclassed trainer with MLflow tracking hooks.

    Args:
        base_class: The nnU-Net trainer class to wrap (e.g., nnUNetTrainer).
            Must be a class (not an instance).
        config: Tracker configuration. If None, reads from environment variables.

    Returns:
        A new class that is a subclass of base_class with tracking hooks.

    Raises:
        TypeError: If base_class is not a class.
        ValueError: If base_class does not appear to be an nnU-Net trainer.
    """
    if not isinstance(base_class, type):
        raise TypeError(
            f"create_tracked_trainer expects a class, got {type(base_class).__name__}: "
            f"{base_class!r}"
        )

    if not hasattr(base_class, "run_training"):
        raise ValueError(
            f"{base_class.__name__} does not have a 'run_training' method. "
            "It may not be an nnU-Net trainer class."
        )

    if config is None:
        config = TrackerConfig.from_env()

    if not config.enabled:
        # If passed an already-tracked class while disabled, unwrap it
        if getattr(base_class, _TRACKED_MARKER, False):
            return getattr(base_class, "_nnunet_tracker_base_class", base_class.__bases__[0])
        return base_class

    # Guard against double-wrapping
    if getattr(base_class, _TRACKED_MARKER, False):
        return base_class

    if check_nnunet_available():
        check_nnunet_version()

    class TrackedTrainer(base_class):
        """Dynamically created trainer subclass with MLflow tracking."""

        _tracker_config: TrackerConfig = config
        _original_trainer_class_name: str = base_class.__name__

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._mlflow_run_id: str | None = None

        def run_training(self) -> None:
            """Wrap run_training with crash-safe MLflow run cleanup.

            Only catches Exception (not BaseException) so that
            KeyboardInterrupt/SystemExit propagate immediately without
            attempting network calls that could hang on SLURM SIGTERM.
            """
            try:
                super().run_training()
            except Exception:
                if self._tracker_config.enabled:
                    end_run_as_failed(run_id=self._mlflow_run_id)
                raise

        def on_train_start(self) -> None:
            super().on_train_start()
            run_id = log_run_start(self, self._tracker_config)
            if run_id is not None:
                self._mlflow_run_id = run_id
            log_fingerprint(self, self._tracker_config)
            log_plans_and_config(self, self._tracker_config)

        def on_train_epoch_start(self) -> None:
            super().on_train_epoch_start()
            log_learning_rate(self, self._tracker_config)

        def on_train_epoch_end(self, train_outputs: list) -> None:
            super().on_train_epoch_end(train_outputs)
            log_train_loss(self, self._tracker_config)

        def on_validation_epoch_end(self, val_outputs: list) -> None:
            super().on_validation_epoch_end(val_outputs)
            log_validation_metrics(self, self._tracker_config)

        def on_epoch_end(self) -> None:
            # Capture epoch BEFORE super() increments current_epoch
            epoch = self.current_epoch
            super().on_epoch_end()
            log_epoch_end(self, self._tracker_config, epoch=epoch)

        def on_train_end(self) -> None:
            super().on_train_end()
            log_run_end(self, self._tracker_config)

    TrackedTrainer.__name__ = f"Tracked{base_class.__name__}"
    TrackedTrainer.__qualname__ = f"create_tracked_trainer.<locals>.Tracked{base_class.__name__}"
    TrackedTrainer.__module__ = __name__
    setattr(TrackedTrainer, _TRACKED_MARKER, True)
    TrackedTrainer._nnunet_tracker_base_class = base_class

    return TrackedTrainer

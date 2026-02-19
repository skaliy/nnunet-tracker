"""
01_basic_tracking.py -- Programmatic usage of nnunet-tracker's factory pattern.

This script demonstrates how to use nnunet-tracker to wrap an nnU-Net trainer
class with automatic MLflow experiment tracking. The key concepts shown are:

1. Configuring tracking via TrackerConfig (environment variables or explicit).
2. Using create_tracked_trainer() to produce a tracked subclass.
3. Understanding what the factory returns and how it relates to the base class.

NOTE: This is a *conceptual* example. The tracked trainer class returned by
create_tracked_trainer() must be instantiated with real nnU-Net arguments
(dataset, plans, configuration, fold) which require a preprocessed dataset.
The factory call itself, however, works without data and is shown here in full.

Equivalent CLI usage:

    # The CLI wraps this factory pattern behind a single command:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_EXPERIMENT_NAME=Dataset001_BrainTumour
    nnunet-tracker train 1 3d_fullres 0 -tr nnUNetTrainer

"""

from __future__ import annotations

import os


def demonstrate_tracker_config() -> None:
    """Show how TrackerConfig reads from environment variables.

    TrackerConfig is a frozen (immutable) dataclass with four fields:

        tracking_uri      -- Where MLflow stores runs.
                             Env: MLFLOW_TRACKING_URI (default: ./mlruns)
        experiment_name   -- MLflow experiment name.
                             Env: MLFLOW_EXPERIMENT_NAME (default: None,
                             which causes nnunet-tracker to use the nnU-Net
                             dataset name automatically)
        enabled           -- Master switch for tracking.
                             Env: NNUNET_TRACKER_ENABLED (default: '1')
        log_artifacts     -- Whether to log checkpoints and plots.
                             Env: NNUNET_TRACKER_LOG_ARTIFACTS (default: '1')

    You can either let TrackerConfig.from_env() read the environment, or
    construct a TrackerConfig directly for programmatic control.
    """
    from nnunet_tracker.config import TrackerConfig

    # --- Option A: Read from environment variables ---
    # Set environment variables before calling from_env().
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Dataset001_BrainTumour"
    os.environ["NNUNET_TRACKER_ENABLED"] = "1"
    os.environ["NNUNET_TRACKER_LOG_ARTIFACTS"] = "1"

    config_from_env = TrackerConfig.from_env()
    print("Config from environment:")
    print(f"  tracking_uri:    {config_from_env.tracking_uri}")
    print(f"  experiment_name: {config_from_env.experiment_name}")
    print(f"  enabled:         {config_from_env.enabled}")
    print(f"  log_artifacts:   {config_from_env.log_artifacts}")
    print()

    # --- Option B: Construct directly (useful in notebooks or scripts) ---
    config_explicit = TrackerConfig(
        tracking_uri="./mlruns",
        experiment_name="my_experiment",
        enabled=True,
        log_artifacts=False,  # Skip artifact logging to save disk space
    )
    print("Explicit config:")
    print(f"  tracking_uri:    {config_explicit.tracking_uri}")
    print(f"  experiment_name: {config_explicit.experiment_name}")
    print(f"  enabled:         {config_explicit.enabled}")
    print(f"  log_artifacts:   {config_explicit.log_artifacts}")
    print()


def demonstrate_factory_pattern() -> None:
    """Show the dynamic subclass factory that is central to nnunet-tracker.

    create_tracked_trainer(base_class, config=...) takes any nnU-Net trainer
    class and returns a NEW class that:

      - Is a subclass of base_class (isinstance checks still pass).
      - Has the same __init__ signature and all methods of the base.
      - Overrides the nnU-Net hook methods (on_train_start, on_epoch_end, etc.)
        to log metrics, parameters, and artifacts to MLflow automatically.
      - Uses a @failsafe decorator on every MLflow call so that a tracking
        failure never interrupts training.

    The factory is idempotent: wrapping an already-tracked class returns
    the same class (double-wrapping is prevented by a sentinel attribute).
    Passing config with enabled=False returns the original base class unchanged.
    """
    from nnunet_tracker import create_tracked_trainer
    from nnunet_tracker.config import TrackerConfig

    # For this demonstration, we create a minimal stand-in class that has
    # the run_training() method the factory expects. In real usage you would
    # import the actual nnU-Net trainer class.
    class MockNnUNetTrainer:
        """Minimal stand-in for nnUNetTrainer to show the factory pattern."""

        def run_training(self):
            pass

        def on_train_start(self):
            pass

        def on_train_epoch_start(self):
            pass

        def on_train_epoch_end(self, train_outputs):
            pass

        def on_validation_epoch_end(self, val_outputs):
            pass

        def on_epoch_end(self):
            pass

        def on_train_end(self):
            pass

    # --- Create a tracked version of the trainer ---
    config = TrackerConfig(
        tracking_uri="./mlruns",
        experiment_name="demo",
        enabled=True,
        log_artifacts=True,
    )

    TrackedTrainer = create_tracked_trainer(MockNnUNetTrainer, config=config)

    # The returned class is a proper subclass of the base
    print(f"TrackedTrainer name:       {TrackedTrainer.__name__}")
    print(f"Is subclass of base:       {issubclass(TrackedTrainer, MockNnUNetTrainer)}")
    print(f"Has run_training:          {hasattr(TrackedTrainer, 'run_training')}")
    print(f"Has on_train_start:        {hasattr(TrackedTrainer, 'on_train_start')}")
    print()

    # --- Double-wrapping protection ---
    # Calling create_tracked_trainer again on an already-tracked class
    # returns the same class, not a doubly-wrapped one.
    DoubleWrapped = create_tracked_trainer(TrackedTrainer, config=config)
    print(f"Double-wrap same class:    {DoubleWrapped is TrackedTrainer}")
    print()

    # --- Disabling tracking ---
    # When enabled=False, the factory returns the original base class.
    disabled_config = TrackerConfig(
        tracking_uri="./mlruns",
        experiment_name="demo",
        enabled=False,
        log_artifacts=True,
    )
    Passthrough = create_tracked_trainer(MockNnUNetTrainer, config=disabled_config)
    print(f"Disabled returns base:     {Passthrough is MockNnUNetTrainer}")

    # Passing an already-tracked class with enabled=False unwraps it
    Unwrapped = create_tracked_trainer(TrackedTrainer, config=disabled_config)
    print(f"Disabled unwraps tracked:  {Unwrapped is MockNnUNetTrainer}")
    print()


def show_real_usage_pattern() -> None:
    """Show the pattern you would use with a real nnU-Net installation.

    This code block is commented out because it requires nnU-Net to be
    installed and a preprocessed dataset to be available. It is included
    here as a reference for real-world usage.
    """
    print("Real-world usage pattern (requires nnU-Net + preprocessed data):")
    print()
    print("    import os")
    print("    from nnunet_tracker import create_tracked_trainer")
    print("    from nnunet_tracker.config import TrackerConfig")
    print("    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer")
    print()
    print("    # Configure tracking")
    print('    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"')
    print('    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Dataset001_BrainTumour"')
    print()
    print("    # Create tracked trainer class")
    print("    TrackedTrainer = create_tracked_trainer(nnUNetTrainer)")
    print()
    print("    # Instantiate with standard nnU-Net arguments")
    print('    trainer = TrackedTrainer(plans=plans, configuration="3d_fullres", fold=0, ...)')
    print()
    print("    # Training proceeds exactly as normal -- all MLflow logging is automatic")
    print("    trainer.run_training()")
    print()
    print("    # What gets logged to MLflow automatically:")
    print("    #   - Run parameters: fold, num_epochs, batch_size, patch_size, LR, etc.")
    print("    #   - Dataset fingerprint hashes (SHA-256)")
    print("    #   - Extended plans/config params (preprocessing, network topology)")
    print("    #   - Per-epoch metrics: train_loss, val_loss, mean_fg_dice, per-class dice")
    print("    #   - Learning rate at each epoch")
    print("    #   - EMA foreground Dice")
    print("    #   - Cross-validation tags: nnunet_tracker.fold, nnunet_tracker.cv_group")
    print("    #   - Artifacts: checkpoint_final.pth, checkpoint_best.pth, progress.png")
    print("    #   - Full plans JSON as artifact")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("nnunet-tracker: Basic Tracking Example")
    print("=" * 70)
    print()

    print("--- 1. TrackerConfig ---")
    print()
    demonstrate_tracker_config()

    print("--- 2. Factory Pattern ---")
    print()
    demonstrate_factory_pattern()

    print("--- 3. Real-World Usage ---")
    print()
    show_real_usage_pattern()

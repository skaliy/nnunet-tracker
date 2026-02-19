"""Tests for nnunet_tracker.factory module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nnunet_tracker.factory import create_tracked_trainer


class TestCreateTrackedTrainer:
    """Tests for create_tracked_trainer factory function."""

    def test_returns_subclass(self, mock_trainer_class, tracker_config_enabled) -> None:
        tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        assert issubclass(tracked, mock_trainer_class)

    def test_class_name(self, mock_trainer_class, tracker_config_enabled) -> None:
        tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        assert tracked.__name__ == "TrackedMockTrainerBase"

    def test_raises_type_error_for_non_class(self, tracker_config_enabled) -> None:
        with pytest.raises(TypeError, match="expects a class"):
            create_tracked_trainer("not_a_class", config=tracker_config_enabled)

    def test_raises_type_error_for_instance(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        instance = mock_trainer_class()
        with pytest.raises(TypeError, match="expects a class"):
            create_tracked_trainer(instance, config=tracker_config_enabled)

    def test_raises_value_error_for_non_trainer(self, tracker_config_enabled) -> None:
        class NotATrainer:
            pass

        with pytest.raises(ValueError, match="run_training"):
            create_tracked_trainer(NotATrainer, config=tracker_config_enabled)

    def test_returns_base_class_when_disabled(
        self, mock_trainer_class, tracker_config_disabled
    ) -> None:
        result = create_tracked_trainer(mock_trainer_class, config=tracker_config_disabled)
        assert result is mock_trainer_class

    def test_unwraps_tracked_class_when_disabled(
        self, mock_trainer_class, tracker_config_enabled, tracker_config_disabled
    ) -> None:
        """When config is disabled and class is already tracked, unwrap to original base."""
        tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        assert getattr(tracked, "_nnunet_tracker_wrapped", False) is True
        unwrapped = create_tracked_trainer(tracked, config=tracker_config_disabled)
        assert unwrapped is mock_trainer_class

    def test_multi_level_inheritance(self, mock_trainer_class, tracker_config_enabled) -> None:
        class CustomTrainer(mock_trainer_class):
            custom_flag = True

        tracked = create_tracked_trainer(CustomTrainer, config=tracker_config_enabled)
        assert issubclass(tracked, CustomTrainer)
        assert issubclass(tracked, mock_trainer_class)
        instance = tracked()
        assert instance.custom_flag is True

    def test_super_called_before_logging(self, mock_trainer_class, tracker_config_enabled) -> None:
        """Verify super() is called before hook logging functions."""
        call_order = []

        class OrderTracker(mock_trainer_class):
            def on_train_start(self):
                call_order.append("super_on_train_start")

            def on_train_end(self):
                call_order.append("super_on_train_end")

        with (
            patch("nnunet_tracker.factory.log_run_start") as mock_start,
            patch("nnunet_tracker.factory.log_fingerprint") as mock_fp,
            patch("nnunet_tracker.factory.log_plans_and_config") as mock_plans,
            patch("nnunet_tracker.factory.log_run_end") as mock_end,
        ):
            mock_start.side_effect = lambda *a, **kw: call_order.append("log_run_start")
            mock_fp.side_effect = lambda *a, **kw: call_order.append("log_fingerprint")
            mock_plans.side_effect = lambda *a, **kw: call_order.append("log_plans_and_config")
            mock_end.side_effect = lambda *a, **kw: call_order.append("log_run_end")

            tracked = create_tracked_trainer(OrderTracker, config=tracker_config_enabled)
            instance = tracked()
            instance.on_train_start()
            instance.on_train_end()

        assert call_order == [
            "super_on_train_start",
            "log_run_start",
            "log_fingerprint",
            "log_plans_and_config",
            "super_on_train_end",
            "log_run_end",
        ]

    def test_full_training_loop(self, mock_trainer_class, tracker_config_enabled) -> None:
        """Wrap, instantiate, and run full training loop without crash."""
        with patch("nnunet_tracker.hooks._should_track", return_value=True):
            mock_mlflow = MagicMock()
            mock_mlflow.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))
            with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
                tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
                instance = tracked()
                instance.run_training()  # Should not raise

    def test_config_from_env_when_none(self, mock_trainer_class) -> None:
        """Config is read from env when not provided."""
        with patch.dict(
            "os.environ",
            {"NNUNET_TRACKER_ENABLED": "0"},
        ):
            result = create_tracked_trainer(mock_trainer_class, config=None)
            assert result is mock_trainer_class  # disabled, returns base

    def test_double_wrap_guard(self, mock_trainer_class, tracker_config_enabled) -> None:
        """M5 fix: wrapping an already-tracked class returns it unchanged."""
        tracked_once = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        tracked_twice = create_tracked_trainer(tracked_once, config=tracker_config_enabled)
        assert tracked_twice is tracked_once

    def test_mlflow_run_id_is_instance_attribute(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        """H1 fix: _mlflow_run_id is per-instance, not shared class state."""
        tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        instance_a = tracked()
        instance_b = tracked()
        instance_a._mlflow_run_id = "run-a"
        assert instance_b._mlflow_run_id is None

    def test_original_trainer_class_name_stored(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
        assert tracked._original_trainer_class_name == "MockTrainerBase"

    def test_epoch_end_uses_pre_increment_epoch(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        """C3 fix: on_epoch_end captures epoch before super() increments it."""
        with (
            patch("nnunet_tracker.factory.log_epoch_end") as mock_log,
            patch("nnunet_tracker.factory.log_run_start"),
            patch("nnunet_tracker.factory.log_fingerprint"),
            patch("nnunet_tracker.factory.log_plans_and_config"),
            patch("nnunet_tracker.factory.log_learning_rate"),
            patch("nnunet_tracker.factory.log_train_loss"),
            patch("nnunet_tracker.factory.log_validation_metrics"),
            patch("nnunet_tracker.factory.log_run_end"),
        ):
            tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
            instance = tracked()
            instance.current_epoch = 5
            instance.run_training()

            # on_epoch_end should pass epoch=5 (pre-increment), not 6
            _, kwargs = mock_log.call_args
            assert kwargs["epoch"] == 5

    def test_crash_safety_ends_run_as_failed(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        """C2 fix: run_training wraps with try/finally to end run on crash."""

        class CrashingTrainer(mock_trainer_class):
            def run_training(self):
                self.on_train_start()
                raise RuntimeError("OOM")

        with (
            patch("nnunet_tracker.factory.log_run_start", return_value="crash-run-id"),
            patch("nnunet_tracker.factory.end_run_as_failed") as mock_end_failed,
        ):
            tracked = create_tracked_trainer(CrashingTrainer, config=tracker_config_enabled)
            instance = tracked()
            with pytest.raises(RuntimeError, match="OOM"):
                instance.run_training()

            mock_end_failed.assert_called_once()

    def test_on_train_start_calls_fingerprint_and_plans(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        """Phase 2: on_train_start calls log_fingerprint and log_plans_and_config."""
        with (
            patch("nnunet_tracker.factory.log_run_start", return_value="test-run-id"),
            patch("nnunet_tracker.factory.log_fingerprint") as mock_fp,
            patch("nnunet_tracker.factory.log_plans_and_config") as mock_plans,
        ):
            tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
            instance = tracked()
            instance.on_train_start()
            mock_fp.assert_called_once_with(instance, tracker_config_enabled)
            mock_plans.assert_called_once_with(instance, tracker_config_enabled)

    def test_crash_safety_does_not_interfere_on_success(
        self, mock_trainer_class, tracker_config_enabled
    ) -> None:
        """end_run_as_failed not called on normal completion."""
        with (
            patch("nnunet_tracker.factory.end_run_as_failed") as mock_end_failed,
            patch("nnunet_tracker.factory.log_run_start", return_value="ok-run-id"),
            patch("nnunet_tracker.factory.log_fingerprint"),
            patch("nnunet_tracker.factory.log_plans_and_config"),
            patch("nnunet_tracker.factory.log_learning_rate"),
            patch("nnunet_tracker.factory.log_train_loss"),
            patch("nnunet_tracker.factory.log_validation_metrics"),
            patch("nnunet_tracker.factory.log_epoch_end"),
            patch("nnunet_tracker.factory.log_run_end"),
        ):
            tracked = create_tracked_trainer(mock_trainer_class, config=tracker_config_enabled)
            instance = tracked()
            instance.run_training()

            mock_end_failed.assert_not_called()

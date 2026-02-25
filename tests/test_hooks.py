"""Tests for nnunet_tracker.hooks module."""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock, patch

import pytest

from nnunet_tracker.config import TrackerConfig
from nnunet_tracker.hooks import (
    _build_cv_tags,
    _should_track,
    end_run_as_failed,
    failsafe,
    log_epoch_end,
    log_fingerprint,
    log_plans_and_config,
    log_run_end,
    log_run_start,
    log_train_loss,
    log_validation_metrics,
    log_validation_summary,
)
from tests.conftest import MockTrainerBase, mock_mlflow_modules


class TestFailsafe:
    """Tests for the failsafe decorator."""

    def test_successful_function_returns_normally(self) -> None:
        @failsafe
        def good_func():
            return 42

        assert good_func() == 42

    def test_exception_caught_returns_none(self) -> None:
        @failsafe
        def bad_func():
            raise RuntimeError("boom")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = bad_func()
            assert result is None
            assert len(w) == 1
            assert "bad_func failed" in str(w[0].message)

    def test_exception_details_not_in_warning(self) -> None:
        """Warning message should NOT leak exception details (C4 security fix)."""

        @failsafe
        def leaky_func():
            raise RuntimeError("/secret/path/to/patient/data")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            leaky_func()
            msg = str(w[0].message)
            assert "/secret/path" not in msg
            assert "Enable debug logging" in msg

    def test_keyboard_interrupt_not_caught(self) -> None:
        @failsafe
        def interrupt_func():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            interrupt_func()

    def test_system_exit_not_caught(self) -> None:
        @failsafe
        def exit_func():
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            exit_func()

    def test_memory_error_not_caught(self) -> None:
        @failsafe
        def oom_func():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            oom_func()

    def test_preserves_function_name(self) -> None:
        @failsafe
        def my_func():
            pass

        assert my_func.__name__ == "my_func"


class TestShouldTrack:
    """Tests for _should_track helper."""

    def test_returns_true_when_enabled_and_main_rank(
        self, mock_trainer, tracker_config_enabled
    ) -> None:
        assert _should_track(mock_trainer, tracker_config_enabled) is True

    def test_returns_false_when_disabled(self, mock_trainer, tracker_config_disabled) -> None:
        assert _should_track(mock_trainer, tracker_config_disabled) is False

    def test_returns_false_on_non_primary_rank(self, tracker_config_enabled) -> None:
        from tests.conftest import MockTrainerBase

        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        assert _should_track(trainer, tracker_config_enabled) is False


class TestLogRunStart:
    """Tests for log_run_start."""

    def test_starts_mlflow_run(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id == "test-run-id"
        mock_mlflow.set_tracking_uri.assert_called_once_with("./test_mlruns")
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()

    def test_uses_dataset_name_when_experiment_name_none(self, mock_trainer, mock_mlflow) -> None:
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name=None,
            enabled=True,
            log_artifacts=False,
        )
        log_run_start(mock_trainer, config)
        mock_mlflow.set_experiment.assert_called_once_with("Dataset001_BrainTumour")

    def test_skipped_when_disabled(self, mock_trainer, tracker_config_disabled) -> None:
        result = log_run_start(mock_trainer, tracker_config_disabled)
        assert result is None

    def test_skipped_on_non_primary_rank(self, tracker_config_enabled, mock_mlflow) -> None:
        from tests.conftest import MockTrainerBase

        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        result = log_run_start(trainer, tracker_config_enabled)
        assert result is None
        mock_mlflow.start_run.assert_not_called()

    def test_logs_params(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_run_start(mock_trainer, tracker_config_enabled)
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["fold"] == 0
        assert params["num_epochs"] == 1000
        assert params["batch_size"] == 2
        assert params["dataset_name"] == "Dataset001_BrainTumour"

    def test_run_id_returned_even_if_params_fail(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """C1 fix: run_id must be returned even when log_params raises."""
        mock_mlflow.log_params.side_effect = RuntimeError("param error")
        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id == "test-run-id"
        mock_mlflow.start_run.assert_called_once()

    def test_logs_initial_lr_from_trainer_attribute(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """M3 fix: initial_lr comes from trainer attribute, not optimizer."""
        mock_trainer.initial_lr = 0.005
        log_run_start(mock_trainer, tracker_config_enabled)
        params = mock_mlflow.log_params.call_args[0][0]
        assert params["initial_lr"] == 0.005

    def test_fold_all_run_name_format(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """fold='all' produces run name like 'Dataset_all_config' not 'Dataset_foldall_config'."""
        mock_trainer.fold = "all"
        log_run_start(mock_trainer, tracker_config_enabled)
        run_name = mock_mlflow.start_run.call_args[1]["run_name"]
        assert "_all_" in run_name
        assert "_foldall_" not in run_name

    def test_configuration_name_fallback(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """M6 fix: falls back to 'configuration' attr for older nnU-Net."""
        del mock_trainer.configuration_name
        mock_trainer.configuration = "2d"
        log_run_start(mock_trainer, tracker_config_enabled)
        run_name = mock_mlflow.start_run.call_args[1]["run_name"]
        assert "2d" in run_name


class TestExtractParams:
    """Tests for _extract_params edge cases."""

    def test_missing_configuration_manager(self, mock_trainer) -> None:
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.configuration_manager = None
        params = _extract_params(mock_trainer)
        assert "batch_size" not in params
        assert "patch_size" not in params

    def test_missing_plans_manager(self, mock_trainer) -> None:
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.plans_manager = None
        params = _extract_params(mock_trainer)
        assert "dataset_name" not in params

    def test_missing_optimizer(self, mock_trainer) -> None:
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.optimizer = None
        params = _extract_params(mock_trainer)
        # Should not crash, and initial_lr comes from trainer attribute
        assert params["initial_lr"] == 0.01

    def test_no_none_string_for_patch_size(self, mock_trainer) -> None:
        """M2 fix: patch_size=None should not produce string 'None'."""
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.configuration_manager.patch_size = None
        params = _extract_params(mock_trainer)
        assert params.get("patch_size") != "None"
        assert "patch_size" not in params

    def test_no_none_string_for_spacing(self, mock_trainer) -> None:
        """M2 fix: spacing=None should not produce string 'None'."""
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.configuration_manager.spacing = None
        params = _extract_params(mock_trainer)
        assert params.get("spacing") != "None"
        assert "spacing" not in params

    def test_uses_unet_class_name(self, mock_trainer) -> None:
        """H5 fix: uses UNet_class_name (nnU-Net v2 convention)."""
        from nnunet_tracker.hooks import _extract_params

        mock_trainer.configuration_manager.UNet_class_name = "ResidualEncoderUNet"
        params = _extract_params(mock_trainer)
        assert params["network_arch"] == "ResidualEncoderUNet"


class TestLogTrainLoss:
    """Tests for log_train_loss."""

    def test_logs_train_loss(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_train_loss(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metric.assert_called_once_with("train_loss", 0.3, step=0)

    def test_skipped_when_logger_none(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        mock_trainer.logger = None
        log_train_loss(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metric.assert_not_called()

    def test_skipped_when_train_losses_empty(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        mock_trainer.logger.my_fantastic_logging["train_losses"] = []
        log_train_loss(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metric.assert_not_called()


class TestLogValidationMetrics:
    """Tests for log_validation_metrics."""

    def test_logs_all_metrics(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_validation_metrics(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metrics.assert_called_once()
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "val_loss" in metrics
        assert "dice_class_0" in metrics
        assert "dice_class_1" in metrics

    def test_correct_values(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_validation_metrics(mock_trainer, tracker_config_enabled)
        metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert metrics["val_loss"] == 0.4
        assert metrics["dice_class_0"] == 0.8
        assert metrics["dice_class_1"] == 0.9


class TestLogEpochEnd:
    """Tests for log_epoch_end."""

    def test_logs_ema_fg_dice(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_epoch_end(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metric.assert_called_once_with("ema_fg_dice", 0.84, step=0)

    def test_uses_explicit_epoch(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        """C3 fix: explicit epoch param avoids off-by-one from super() increment."""
        mock_trainer.current_epoch = 5  # simulate post-increment
        log_epoch_end(mock_trainer, tracker_config_enabled, epoch=4)
        mock_mlflow.log_metric.assert_called_once_with("ema_fg_dice", 0.84, step=4)


class TestLogRunEnd:
    """Tests for log_run_end."""

    def test_calls_end_run(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        active = MagicMock()
        active.info.run_id = "owned-run-123"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run-123"
        log_run_end(mock_trainer, tracker_config_enabled)
        mock_mlflow.end_run.assert_called_once()

    def test_does_not_end_foreign_run(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """log_run_end must not end a run the trainer does not own."""
        foreign = MagicMock()
        foreign.info.run_id = "foreign-run-456"
        mock_mlflow.active_run.return_value = foreign
        mock_trainer._mlflow_run_id = "different-run-789"
        log_run_end(mock_trainer, tracker_config_enabled)
        mock_mlflow.end_run.assert_not_called()

    def test_does_not_end_run_when_no_run_id(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """log_run_end must not end a run when trainer has no _mlflow_run_id."""
        active = MagicMock()
        active.info.run_id = "some-run"
        mock_mlflow.active_run.return_value = active
        # trainer._mlflow_run_id defaults to None
        log_run_end(mock_trainer, tracker_config_enabled)
        mock_mlflow.end_run.assert_not_called()

    def test_skipped_when_disabled(
        self, mock_trainer, tracker_config_disabled, mock_mlflow
    ) -> None:
        log_run_end(mock_trainer, tracker_config_disabled)
        mock_mlflow.end_run.assert_not_called()

    def test_logs_artifacts_when_enabled(self, mock_trainer, mock_mlflow) -> None:
        active = MagicMock()
        active.info.run_id = "owned-run"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run"
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        # Set nnUNet_results so path containment check passes
        with (
            patch("nnunet_tracker.hooks.os.path.isfile", return_value=True),
            patch.dict("os.environ", {"nnUNet_results": "/nonexistent"}),
        ):
            log_run_end(mock_trainer, config)
        assert mock_mlflow.log_artifact.call_count == 3  # 2 checkpoints + progress.png
        mock_mlflow.end_run.assert_called_once()

    def test_end_run_when_output_folder_none(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """Artifact logging skipped when output_folder is None."""
        active = MagicMock()
        active.info.run_id = "owned-run"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run"
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        mock_trainer.output_folder = None
        log_run_end(mock_trainer, config)
        mock_mlflow.log_artifact.assert_not_called()
        mock_mlflow.end_run.assert_called_once()


class TestLogFingerprint:
    """Tests for log_fingerprint."""

    def test_logs_fingerprint_params(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """Fingerprint hashes logged as params."""
        log_fingerprint(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert "fingerprint_dataset_json" in params
        assert "fingerprint_plans" in params
        assert "fingerprint_composite" in params

    def test_skipped_when_disabled(
        self, mock_trainer, tracker_config_disabled, mock_mlflow
    ) -> None:
        log_fingerprint(mock_trainer, tracker_config_disabled)
        mock_mlflow.log_params.assert_not_called()

    def test_skipped_on_non_primary_rank(self, tracker_config_enabled, mock_mlflow) -> None:
        from tests.conftest import MockTrainerBase

        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        log_fingerprint(trainer, tracker_config_enabled)
        mock_mlflow.log_params.assert_not_called()

    def test_failsafe_on_error(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        """Fingerprint failure does not raise."""
        mock_mlflow.log_params.side_effect = RuntimeError("boom")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = log_fingerprint(mock_trainer, tracker_config_enabled)
            assert result is None
            assert len(w) == 1


class TestLogPlansAndConfig:
    """Tests for log_plans_and_config."""

    def test_logs_extended_params(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_plans_and_config(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert "preproc_normalization_schemes" in params
        assert "net_n_stages" in params
        assert "deep_supervision" in params

    def test_logs_plans_artifact_when_artifacts_enabled(self, mock_trainer, mock_mlflow) -> None:
        """Full plans JSON logged as artifact when log_artifacts=True."""
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        log_plans_and_config(mock_trainer, config)
        # Should have called log_artifact at least once (for plans.json)
        assert mock_mlflow.log_artifact.call_count >= 1
        # Verify artifact_path="config" was used
        calls = mock_mlflow.log_artifact.call_args_list
        artifact_paths = [
            c.kwargs.get("artifact_path") or c[1].get("artifact_path", None)
            if len(c) > 1
            else c.kwargs.get("artifact_path")
            for c in calls
        ]
        assert "config" in artifact_paths

    def test_no_artifact_when_artifacts_disabled(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """No artifacts logged when log_artifacts=False."""
        log_plans_and_config(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_artifact.assert_not_called()

    def test_skipped_when_disabled(
        self, mock_trainer, tracker_config_disabled, mock_mlflow
    ) -> None:
        log_plans_and_config(mock_trainer, tracker_config_disabled)
        mock_mlflow.log_params.assert_not_called()

    def test_skipped_on_non_primary_rank(self, tracker_config_enabled, mock_mlflow) -> None:
        from tests.conftest import MockTrainerBase

        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        log_plans_and_config(trainer, tracker_config_enabled)
        mock_mlflow.log_params.assert_not_called()

    def test_param_failure_does_not_prevent_completion(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """Inner param failure is caught, function completes without crashing."""
        mock_mlflow.log_params.side_effect = RuntimeError("boom")
        # Inner try/except catches this, @failsafe is not triggered
        result = log_plans_and_config(mock_trainer, tracker_config_enabled)
        assert result is None  # @failsafe returns None on success too

    def test_failsafe_on_unexpected_error(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """Unexpected failure in _should_track path triggers @failsafe."""
        with (
            patch("nnunet_tracker.hooks._should_track", side_effect=RuntimeError("boom")),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = log_plans_and_config(mock_trainer, tracker_config_enabled)
            assert result is None
            assert len(w) == 1

    def test_logs_dataset_fingerprint_artifact(self, mock_trainer, mock_mlflow, tmp_path) -> None:
        """dataset_fingerprint.json logged as artifact when found."""
        import os

        fp_dir = tmp_path / "Dataset001_BrainTumour"
        fp_dir.mkdir()
        fp_file = fp_dir / "dataset_fingerprint.json"
        fp_file.write_text('{"median_spacing": [1.0, 1.0, 1.0]}')

        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        with patch.dict(os.environ, {"nnUNet_preprocessed": str(tmp_path)}):
            log_plans_and_config(mock_trainer, config)

        # Should have called log_artifact for both plans.json and dataset_fingerprint.json
        artifact_calls = mock_mlflow.log_artifact.call_args_list
        assert len(artifact_calls) >= 2


class TestStaleRunCleanup:
    """Tests for stale/foreign run detection in log_run_start."""

    def test_genuine_double_call_returns_existing_id(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """When the active run matches the trainer's run_id, skip and return it."""
        existing_run = MagicMock()
        existing_run.info.run_id = "existing-run-123"
        mock_mlflow.active_run.return_value = existing_run
        mock_trainer._mlflow_run_id = "existing-run-123"

        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id == "existing-run-123"
        mock_mlflow.start_run.assert_not_called()
        mock_mlflow.end_run.assert_not_called()

    def test_foreign_run_returns_none_without_ending(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """When a foreign MLflow run is active, warn and return None (don't end it)."""
        foreign_run = MagicMock()
        foreign_run.info.run_id = "foreign-run-456"
        mock_mlflow.active_run.return_value = foreign_run
        mock_trainer._mlflow_run_id = "different-run-789"

        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id is None
        mock_mlflow.end_run.assert_not_called()
        mock_mlflow.start_run.assert_not_called()

    def test_stale_run_no_trainer_run_id(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """When trainer has no _mlflow_run_id, active run is treated as foreign."""
        foreign_run = MagicMock()
        foreign_run.info.run_id = "stale-run-000"
        mock_mlflow.active_run.return_value = foreign_run
        # trainer._mlflow_run_id defaults to None (no match)

        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id is None
        mock_mlflow.end_run.assert_not_called()


class TestEndRunAsFailed:
    """Tests for end_run_as_failed."""

    def test_ends_active_run(self, mock_mlflow) -> None:
        mock_mlflow.active_run.return_value = MagicMock()
        end_run_as_failed()
        mock_mlflow.end_run.assert_called_once_with(status="FAILED")

    def test_no_op_when_no_active_run(self, mock_mlflow) -> None:
        mock_mlflow.active_run.return_value = None
        end_run_as_failed()
        mock_mlflow.end_run.assert_not_called()

    def test_does_not_raise_on_error(self, mock_mlflow) -> None:
        mock_mlflow.active_run.side_effect = RuntimeError("boom")
        end_run_as_failed()  # Should not raise

    def test_memory_error_propagated(self, mock_mlflow) -> None:
        """MemoryError must not be swallowed (M6 consistency fix)."""
        mock_mlflow.active_run.side_effect = MemoryError("OOM")
        with pytest.raises(MemoryError):
            end_run_as_failed()


class TestBuildCVTags:
    """Tests for _build_cv_tags helper."""

    def test_returns_correct_tags(self, mock_trainer) -> None:
        tags = _build_cv_tags(mock_trainer)
        assert tags["nnunet_tracker.fold"] == "0"
        assert tags["nnunet_tracker.run_type"] == "fold"
        assert tags["nnunet_tracker.cv_group"] == "Dataset001_BrainTumour|3d_fullres|nnUNetPlans"

    @pytest.mark.parametrize("fold", [0, 1, 2, 3, 4])
    def test_fold_values(self, mock_trainer, fold) -> None:
        mock_trainer.fold = fold
        tags = _build_cv_tags(mock_trainer)
        assert tags["nnunet_tracker.fold"] == str(fold)

    def test_empty_when_fold_is_none(self, mock_trainer) -> None:
        mock_trainer.fold = None
        tags = _build_cv_tags(mock_trainer)
        assert tags == {}

    def test_missing_plans_manager(self, mock_trainer) -> None:
        mock_trainer.plans_manager = None
        tags = _build_cv_tags(mock_trainer)
        assert tags["nnunet_tracker.fold"] == "0"
        assert tags["nnunet_tracker.run_type"] == "fold"
        # cv_group built from available parts only (configuration)
        assert "nnunet_tracker.cv_group" in tags

    def test_configuration_name_fallback(self, mock_trainer) -> None:
        """Falls back to 'configuration' attr for older nnU-Net."""
        del mock_trainer.configuration_name
        mock_trainer.configuration = "2d"
        tags = _build_cv_tags(mock_trainer)
        assert "2d" in tags["nnunet_tracker.cv_group"]

    def test_fold_all_sets_run_type_final(self, mock_trainer) -> None:
        """fold='all' should produce run_type='final' instead of 'fold'."""
        mock_trainer.fold = "all"
        tags = _build_cv_tags(mock_trainer)
        assert tags["nnunet_tracker.fold"] == "all"
        assert tags["nnunet_tracker.run_type"] == "final"
        assert "nnunet_tracker.cv_group" in tags

    def test_no_cv_group_when_all_parts_none(self, mock_trainer) -> None:
        mock_trainer.plans_manager = None
        del mock_trainer.configuration_name
        mock_trainer.configuration = None
        tags = _build_cv_tags(mock_trainer)
        assert "nnunet_tracker.cv_group" not in tags
        assert tags["nnunet_tracker.fold"] == "0"


class TestCVTagsInLogRunStart:
    """Tests for fold tagging integration in log_run_start."""

    def test_tags_set_on_run_start(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_run_start(mock_trainer, tracker_config_enabled)
        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["nnunet_tracker.fold"] == "0"
        assert tags["nnunet_tracker.run_type"] == "fold"
        assert "nnunet_tracker.cv_group" in tags

    def test_tags_not_set_when_disabled(
        self, mock_trainer, tracker_config_disabled, mock_mlflow
    ) -> None:
        log_run_start(mock_trainer, tracker_config_disabled)
        mock_mlflow.set_tags.assert_not_called()

    def test_tags_not_set_on_non_primary_rank(self, tracker_config_enabled, mock_mlflow) -> None:
        from tests.conftest import MockTrainerBase

        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        log_run_start(trainer, tracker_config_enabled)
        mock_mlflow.set_tags.assert_not_called()

    def test_tag_failure_does_not_lose_run_id(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        """Tag failure is isolated; run_id still returned and params still logged."""
        mock_mlflow.set_tags.side_effect = RuntimeError("tag error")
        run_id = log_run_start(mock_trainer, tracker_config_enabled)
        assert run_id == "test-run-id"
        mock_mlflow.log_params.assert_called_once()


class TestProgressArtifact:
    """Tests for progress.png artifact logging in log_run_end."""

    def test_progress_png_logged_when_found(self, mock_trainer, mock_mlflow) -> None:
        active = MagicMock()
        active.info.run_id = "owned-run"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run"
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        with (
            patch("nnunet_tracker.hooks.os.path.isfile", return_value=True),
            patch.dict("os.environ", {"nnUNet_results": "/nonexistent"}),
        ):
            log_run_end(mock_trainer, config)

        artifact_calls = mock_mlflow.log_artifact.call_args_list
        # Check that "plots" artifact_path is used for progress.png
        paths_and_files = [(c[0][0], c.kwargs.get("artifact_path")) for c in artifact_calls]
        assert any("progress.png" in str(f) and ap == "plots" for f, ap in paths_and_files)

    def test_missing_progress_png_no_error(self, mock_trainer, mock_mlflow) -> None:
        active = MagicMock()
        active.info.run_id = "owned-run"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run"
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        with patch("nnunet_tracker.hooks.os.path.isfile", return_value=False):
            log_run_end(mock_trainer, config)
        mock_mlflow.log_artifact.assert_not_called()
        mock_mlflow.end_run.assert_called_once()

    def test_skipped_when_artifacts_disabled(
        self, mock_trainer, tracker_config_enabled, mock_mlflow
    ) -> None:
        # tracker_config_enabled has log_artifacts=False
        log_run_end(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_artifact.assert_not_called()

    def test_selective_file_logging(self, mock_trainer, mock_mlflow, tmp_path) -> None:
        """Only expected files are logged; other files in output_folder are ignored."""
        import os

        active = MagicMock()
        active.info.run_id = "owned-run"
        mock_mlflow.active_run.return_value = active
        mock_trainer._mlflow_run_id = "owned-run"

        # Create progress.png but not checkpoints
        mock_trainer.output_folder = str(tmp_path)
        (tmp_path / "progress.png").write_bytes(b"PNG")
        (tmp_path / "random_file.txt").write_text("should be ignored")

        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        with patch.dict("os.environ", {"nnUNet_results": str(tmp_path.parent)}):
            log_run_end(mock_trainer, config)
        # Only progress.png should be logged
        assert mock_mlflow.log_artifact.call_count == 1
        logged_path = mock_mlflow.log_artifact.call_args[0][0]
        assert logged_path == os.path.join(str(tmp_path), "progress.png")

    def test_artifact_failure_does_not_crash(self, mock_trainer, mock_mlflow) -> None:
        """Artifact upload failure caught by @failsafe."""
        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test",
            enabled=True,
            log_artifacts=True,
        )
        mock_mlflow.log_artifact.side_effect = RuntimeError("upload failed")
        with (
            patch("nnunet_tracker.hooks.os.path.isfile", return_value=True),
            patch.dict("os.environ", {"nnUNet_results": "/nonexistent"}),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            result = log_run_end(mock_trainer, config)
            assert result is None
            assert len(w) == 1


class TestValidationMetricsBatching:
    """Regression: validation metrics use batched log_metrics, not individual log_metric."""

    def test_log_metric_not_called(self, mock_trainer, tracker_config_enabled, mock_mlflow) -> None:
        log_validation_metrics(mock_trainer, tracker_config_enabled)
        mock_mlflow.log_metric.assert_not_called()
        mock_mlflow.log_metrics.assert_called_once()


class TestPathContainment:
    """Tests for _is_safe_output_folder path containment check."""

    def test_unsafe_when_no_nnunet_results_set(self, tmp_path) -> None:
        from nnunet_tracker.hooks import _is_safe_output_folder

        with patch.dict("os.environ", {}, clear=False):
            # Remove nnUNet_results if present
            import os

            os.environ.pop("nnUNet_results", None)
            # Fail closed: cannot verify path safety without nnUNet_results
            assert _is_safe_output_folder(str(tmp_path)) is False

    def test_safe_when_inside_results(self, tmp_path) -> None:
        from nnunet_tracker.hooks import _is_safe_output_folder

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_dir = results_dir / "Dataset001" / "fold_0"
        output_dir.mkdir(parents=True)

        with patch.dict("os.environ", {"nnUNet_results": str(results_dir)}):
            assert _is_safe_output_folder(str(output_dir)) is True

    def test_unsafe_when_outside_results(self, tmp_path) -> None:
        from nnunet_tracker.hooks import _is_safe_output_folder

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        outside_dir = tmp_path / "other"
        outside_dir.mkdir()

        with patch.dict("os.environ", {"nnUNet_results": str(results_dir)}):
            assert _is_safe_output_folder(str(outside_dir)) is False

    def test_unsafe_with_traversal(self, tmp_path) -> None:
        from nnunet_tracker.hooks import _is_safe_output_folder

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        traversal_path = str(results_dir / ".." / "secrets")

        with patch.dict("os.environ", {"nnUNet_results": str(results_dir)}):
            assert _is_safe_output_folder(traversal_path) is False


class TestLogValidationSummary:
    """Tests for log_validation_summary()."""

    def test_happy_path_logs_foreground_and_per_class(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {
            "foreground_mean": {"Dice": 0.85, "IoU": 0.75},
            "mean": {
                "1": {"Dice": 0.80, "IoU": 0.70},
                "2": {"Dice": 0.90, "IoU": 0.85},
            },
        }
        (val_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_called_once()
        call_args = mock_client.log_batch.call_args
        assert call_args[0][0] == "run-123"
        metrics = call_args[1]["metrics"]
        metric_dict = {m.key: m.value for m in metrics}
        assert metric_dict["final_val_mean_fg_dice"] == 0.85
        assert metric_dict["final_val_mean_fg_iou"] == 0.75
        assert metric_dict["final_val_dice_class_1"] == 0.80
        assert metric_dict["final_val_iou_class_1"] == 0.70
        assert metric_dict["final_val_dice_class_2"] == 0.90
        assert metric_dict["final_val_iou_class_2"] == 0.85

    def test_returns_early_when_disabled(
        self, mock_trainer, tracker_config_disabled, tmp_path
    ) -> None:
        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_disabled)

        mock_client.assert_not_called()

    def test_returns_early_when_no_run_id(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = None

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_not_called()

    def test_returns_early_when_no_output_folder(
        self, mock_trainer, tracker_config_enabled
    ) -> None:
        mock_trainer.output_folder = None
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_not_called()

    def test_returns_early_when_summary_missing(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"
        # Do NOT create validation/summary.json

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_not_called()

    def test_nan_values_filtered(self, mock_trainer, tracker_config_enabled, tmp_path) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {
            "foreground_mean": {"Dice": 0.85, "IoU": float("nan")},
            "mean": {
                "1": {"Dice": float("nan"), "IoU": 0.70},
                "2": {"Dice": "not_a_number", "IoU": 0.85},
            },
        }
        # json.dumps with allow_nan=True writes NaN as NaN (JavaScript literal)
        (val_dir / "summary.json").write_text(json.dumps(summary, allow_nan=True), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_called_once()
        metrics = mock_client.log_batch.call_args[1]["metrics"]
        metric_dict = {m.key: m.value for m in metrics}
        # NaN values should be filtered out
        assert "final_val_mean_fg_iou" not in metric_dict
        assert "final_val_dice_class_1" not in metric_dict
        # Non-numeric string should be filtered out
        assert "final_val_dice_class_2" not in metric_dict
        # Valid values should remain
        assert metric_dict["final_val_mean_fg_dice"] == 0.85
        assert metric_dict["final_val_iou_class_1"] == 0.70
        assert metric_dict["final_val_iou_class_2"] == 0.85

    def test_inf_values_filtered(self, mock_trainer, tracker_config_enabled, tmp_path) -> None:
        """Inf/-Inf values should be filtered out (isfinite consistency fix)."""
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {
            "foreground_mean": {"Dice": float("inf"), "IoU": 0.75},
            "mean": {
                "1": {"Dice": float("-inf"), "IoU": 0.70},
            },
        }
        (val_dir / "summary.json").write_text(json.dumps(summary, allow_nan=True), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_called_once()
        metrics = mock_client.log_batch.call_args[1]["metrics"]
        metric_dict = {m.key: m.value for m in metrics}
        # Inf values should be filtered out
        assert "final_val_mean_fg_dice" not in metric_dict
        assert "final_val_dice_class_1" not in metric_dict
        # Valid values should remain
        assert metric_dict["final_val_mean_fg_iou"] == 0.75
        assert metric_dict["final_val_iou_class_1"] == 0.70

    def test_empty_summary_no_log_batch(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        (val_dir / "summary.json").write_text("{}", encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_not_called()

    def test_artifact_logged_when_enabled(self, mock_trainer, tmp_path) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {"foreground_mean": {"Dice": 0.85, "IoU": 0.75}}
        (val_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        config = TrackerConfig(
            tracking_uri="./test_mlruns",
            experiment_name="test_experiment",
            enabled=True,
            log_artifacts=True,
        )

        mock_client = MagicMock()
        with (
            mock_mlflow_modules(mock_client),
            patch("nnunet_tracker.hooks._is_safe_output_folder", return_value=True),
        ):
            log_validation_summary(mock_trainer, config)

        mock_client.log_artifact.assert_called_once()
        call_args = mock_client.log_artifact.call_args
        assert call_args[0][0] == "run-123"
        assert call_args[0][1] == str(val_dir / "summary.json")
        assert call_args[1]["artifact_path"] == "validation"

    def test_artifact_not_logged_when_disabled(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {"foreground_mean": {"Dice": 0.85, "IoU": 0.75}}
        (val_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"
        # tracker_config_enabled has log_artifacts=False

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_artifact.assert_not_called()

    def test_region_label_key_sanitized(
        self, mock_trainer, tracker_config_enabled, tmp_path
    ) -> None:
        val_dir = tmp_path / "validation"
        val_dir.mkdir()
        summary = {
            "mean": {
                "(1, 2)": {"Dice": 0.88, "IoU": 0.78},
            },
        }
        (val_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

        mock_trainer.output_folder = str(tmp_path)
        mock_trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(mock_trainer, tracker_config_enabled)

        mock_client.log_batch.assert_called_once()
        metrics = mock_client.log_batch.call_args[1]["metrics"]
        metric_dict = {m.key: m.value for m in metrics}
        assert "final_val_dice_class_1_2" in metric_dict
        assert metric_dict["final_val_dice_class_1_2"] == 0.88
        assert "final_val_iou_class_1_2" in metric_dict
        assert metric_dict["final_val_iou_class_1_2"] == 0.78

    def test_ddp_non_primary_rank_skipped(self, tracker_config_enabled, tmp_path) -> None:
        trainer = MockTrainerBase()
        trainer.is_ddp = True
        trainer.local_rank = 1
        trainer.output_folder = str(tmp_path)
        trainer._mlflow_run_id = "run-123"

        mock_client = MagicMock()
        with mock_mlflow_modules(mock_client):
            log_validation_summary(trainer, tracker_config_enabled)

        mock_client.log_batch.assert_not_called()

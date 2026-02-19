"""Tests for nnunet_tracker.cli module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nnunet_tracker import __version__
from nnunet_tracker.cli import main
from nnunet_tracker.cli.train import (
    _VALID_PLANS_NAME,
    _VALID_TRAINER_NAME,
    _build_nnunet_arg_parser,
    _run_train,
)


class TestMainCli:
    """Tests for the main CLI entry point."""

    def test_no_command_exits_with_error(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_version_flag(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out

    def test_help_flag(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "nnunet-tracker" in captured.out
        assert "train" in captured.out


class TestNnunetArgParser:
    """Tests for the nnU-Net argument parser."""

    def test_parses_positional_args(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.dataset_name_or_id == "Dataset001"
        assert args.configuration == "3d_fullres"
        assert args.fold == "0"

    def test_default_trainer(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.tr == "nnUNetTrainer"

    def test_custom_trainer(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(
            ["Dataset001", "3d_fullres", "0", "-tr", "nnUNetTrainerNoMirroring"]
        )
        assert args.tr == "nnUNetTrainerNoMirroring"

    def test_custom_plans(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(
            ["Dataset001", "3d_fullres", "0", "-p", "nnUNetResEncUNetLPlans"]
        )
        assert args.p == "nnUNetResEncUNetLPlans"

    def test_continue_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "-c"])
        assert args.c is True

    def test_fold_all(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "all"])
        assert args.fold == "all"

    def test_unknown_args_passed_through(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, extra = parser.parse_known_args(
            ["Dataset001", "3d_fullres", "0", "--some-unknown", "value"]
        )
        assert "--some-unknown" in extra
        assert "value" in extra


class TestTrainerNameValidation:
    """Tests for trainer name validation regex."""

    def test_valid_trainer_names(self) -> None:
        assert _VALID_TRAINER_NAME.fullmatch("nnUNetTrainer") is not None
        assert _VALID_TRAINER_NAME.fullmatch("nnUNetTrainerNoMirroring") is not None
        assert _VALID_TRAINER_NAME.fullmatch("MyCustomTrainer_v2") is not None

    def test_invalid_trainer_names(self) -> None:
        assert _VALID_TRAINER_NAME.fullmatch("../../etc/passwd") is None
        assert _VALID_TRAINER_NAME.fullmatch("trainer;rm -rf /") is None
        assert _VALID_TRAINER_NAME.fullmatch("trainer\ninjection") is None
        assert _VALID_TRAINER_NAME.fullmatch("123StartWithNumber") is None
        assert _VALID_TRAINER_NAME.fullmatch("") is None


class TestPlansNameValidation:
    """Tests for plans identifier validation regex."""

    def test_valid_plans_names(self) -> None:
        assert _VALID_PLANS_NAME.fullmatch("nnUNetPlans") is not None
        assert _VALID_PLANS_NAME.fullmatch("nnUNetResEncUNetMPlans") is not None
        assert _VALID_PLANS_NAME.fullmatch("CustomPlans_v2") is not None

    def test_invalid_plans_names(self) -> None:
        assert _VALID_PLANS_NAME.fullmatch("../../etc/passwd") is None
        assert _VALID_PLANS_NAME.fullmatch("plans;rm -rf /") is None
        assert _VALID_PLANS_NAME.fullmatch("plans/../../traversal") is None
        assert _VALID_PLANS_NAME.fullmatch("") is None


class TestRunTrain:
    """Tests for _run_train with mocked nnU-Net."""

    def _make_args(self, tracker_disable: bool = False) -> MagicMock:
        """Build a mock args namespace for _run_train."""
        args = MagicMock()
        args.tracker_disable = tracker_disable
        return args

    def test_exits_when_nnunet_not_installed(self, capsys) -> None:
        """_run_train should exit 1 when nnU-Net is not importable."""
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=False):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0"])
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "nnU-Net v2 is not installed" in captured.err

    def test_exits_on_invalid_trainer_name(self, capsys) -> None:
        """_run_train should exit 1 when trainer name fails regex validation."""
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "-tr", "../../bad"])
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid trainer class name" in captured.err

    def test_exits_on_invalid_plans_name(self, capsys) -> None:
        """_run_train should exit 1 when plans name fails regex validation."""
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "-p", "plans;injection"])
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid plans identifier" in captured.err

    def test_exits_when_trainer_class_not_found(self, capsys) -> None:
        """_run_train should exit 1 when trainer class cannot be resolved."""
        args = self._make_args()
        with (
            patch("nnunet_tracker._compat.check_nnunet_available", return_value=True),
            patch("nnunet_tracker.cli.train._resolve_trainer_class", return_value=None),
        ):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0"])
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Trainer class not found" in captured.err

    def test_tracker_disable_flag_creates_disabled_config(self) -> None:
        """--tracker-disable creates a config with enabled=False."""
        args = self._make_args(tracker_disable=True)
        mock_tracked_class = MagicMock()
        mock_run_with = MagicMock()

        with (
            patch("nnunet_tracker._compat.check_nnunet_available", return_value=True),
            patch(
                "nnunet_tracker.cli.train._resolve_trainer_class",
                return_value=MagicMock,
            ),
            patch(
                "nnunet_tracker.factory.create_tracked_trainer",
                return_value=mock_tracked_class,
            ) as mock_create,
            patch(
                "nnunet_tracker.cli.train._run_with_tracked_trainer",
                mock_run_with,
            ),
        ):
            _run_train(args, ["Dataset001", "3d_fullres", "0"])

        # Verify that create_tracked_trainer was called with enabled=False config
        call_kwargs = mock_create.call_args
        config = call_kwargs[1]["config"]
        assert config.enabled is False

    def test_delegates_to_run_with_tracked_trainer(self) -> None:
        """_run_train calls _run_with_tracked_trainer with correct arguments."""
        args = self._make_args()
        mock_tracked_class = MagicMock()
        mock_run_with = MagicMock()

        with (
            patch("nnunet_tracker._compat.check_nnunet_available", return_value=True),
            patch(
                "nnunet_tracker.cli.train._resolve_trainer_class",
                return_value=MagicMock,
            ),
            patch(
                "nnunet_tracker.factory.create_tracked_trainer",
                return_value=mock_tracked_class,
            ),
            patch(
                "nnunet_tracker.cli.train._run_with_tracked_trainer",
                mock_run_with,
            ),
        ):
            _run_train(args, ["Dataset001", "3d_fullres", "all"])

        mock_run_with.assert_called_once()
        call_kwargs = mock_run_with.call_args[1]
        assert call_kwargs["tracked_class"] is mock_tracked_class
        assert call_kwargs["dataset_name_or_id"] == "Dataset001"
        assert call_kwargs["configuration"] == "3d_fullres"
        assert call_kwargs["fold"] == "all"

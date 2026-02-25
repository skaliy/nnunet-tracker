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
    _run_with_tracked_trainer,
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


class TestValidationAfterTraining:
    """Tests for validation call after training in _run_with_tracked_trainer."""

    def _make_preprocessed_dir(self, tmp_path):
        """Create a fake preprocessed directory with plans.json and dataset.json."""
        import json

        dataset_dir = tmp_path / "Dataset001_Test"
        dataset_dir.mkdir()
        plans_file = dataset_dir / "nnUNetPlans.json"
        plans_file.write_text(json.dumps({"dataset_name": "Dataset001_Test"}), encoding="utf-8")
        dataset_file = dataset_dir / "dataset.json"
        dataset_file.write_text(json.dumps({"labels": {"background": 0}}), encoding="utf-8")
        return tmp_path

    def _mock_nnunet_modules(self, tmp_path):
        """Return a patch.dict context manager that mocks torch and nnunetv2 modules."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        mock_paths = MagicMock()
        mock_paths.nnUNet_preprocessed = str(tmp_path)

        mock_conversion = MagicMock()
        mock_conversion.maybe_convert_to_dataset_name.return_value = "Dataset001_Test"

        return patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "nnunetv2": MagicMock(),
                "nnunetv2.paths": mock_paths,
                "nnunetv2.utilities": MagicMock(),
                "nnunetv2.utilities.dataset_name_id_conversion": mock_conversion,
            },
        )

    def _run(self, tmp_path, trainer_class, fold: str = "0", **kwargs):
        """Invoke _run_with_tracked_trainer with mocked nnU-Net imports."""
        self._make_preprocessed_dir(tmp_path)
        defaults = dict(
            tracked_class=trainer_class,
            dataset_name_or_id="1",
            configuration="3d_fullres",
            fold=fold,
            plans_identifier="nnUNetPlans",
            continue_training=False,
            pretrained_weights=None,
            export_validation_probabilities=False,
            only_run_validation=False,
            val_with_best=False,
            disable_checkpointing=False,
            device_name="cpu",
        )
        defaults.update(kwargs)
        with self._mock_nnunet_modules(tmp_path):
            _run_with_tracked_trainer(**defaults)

    def test_validation_called_after_training(self, tmp_path):
        """perform_actual_validation is called after run_training."""
        call_order = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                call_order.append("run_training")

            def load_checkpoint(self, path):
                call_order.append("load_ckpt")

            def perform_actual_validation(self, save_probabilities=False):
                call_order.append("validate")

        self._run(tmp_path, FakeTrainer)
        assert "run_training" in call_order
        assert "validate" in call_order
        assert call_order.index("run_training") < call_order.index("validate")

    def test_best_checkpoint_loaded_before_validation_with_val_best(self, tmp_path):
        """checkpoint_best.pth loaded before validation when --val_best is passed."""
        call_order = []

        # Create output dir and checkpoint file
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_best.pth").touch()

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                call_order.append("run_training")

            def load_checkpoint(self, path):
                call_order.append(("load_ckpt", path))

            def perform_actual_validation(self, save_probabilities=False):
                call_order.append("validate")

        self._run(tmp_path, FakeTrainer, val_with_best=True)

        expected_path = str(output_dir / "checkpoint_best.pth")
        assert ("load_ckpt", expected_path) in call_order
        load_idx = call_order.index(("load_ckpt", expected_path))
        validate_idx = call_order.index("validate")
        assert load_idx < validate_idx

    def test_validation_runs_without_best_checkpoint(self, tmp_path):
        """Validation still runs when no checkpoint_best.pth exists."""
        calls = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                calls.append("load_ckpt")

            def perform_actual_validation(self, save_probabilities=False):
                calls.append("validate")

        self._run(tmp_path, FakeTrainer)
        assert "validate" in calls
        assert "load_ckpt" not in calls

    def test_fold_all_passes_string_to_trainer(self, tmp_path):
        """fold='all' should pass the string 'all' to trainer, not expand to 5 folds."""
        init_calls = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                init_calls.append(kwargs.get("fold"))

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                pass

            def perform_actual_validation(self):
                pass

        self._run(tmp_path, FakeTrainer, fold="all")
        assert init_calls == ["all"]

    def test_validation_failure_does_not_crash(self, tmp_path):
        """Exception in perform_actual_validation is caught and does not propagate."""

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                raise RuntimeError("validation failed")

        # Should not raise -- the exception is caught internally
        self._run(tmp_path, FakeTrainer)


class TestFileReadErrors:
    """Tests for OSError handling when reading plans/dataset JSON files."""

    def _make_preprocessed_dir(self, tmp_path, plans_content=None, dataset_content=None):
        """Create a fake preprocessed directory."""
        import json

        dataset_dir = tmp_path / "Dataset001_Test"
        dataset_dir.mkdir(exist_ok=True)
        if plans_content is not None:
            (dataset_dir / "nnUNetPlans.json").write_text(
                json.dumps(plans_content), encoding="utf-8"
            )
        if dataset_content is not None:
            (dataset_dir / "dataset.json").write_text(json.dumps(dataset_content), encoding="utf-8")
        return dataset_dir

    def _mock_nnunet_modules(self, tmp_path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        mock_paths = MagicMock()
        mock_paths.nnUNet_preprocessed = str(tmp_path)
        mock_conversion = MagicMock()
        mock_conversion.maybe_convert_to_dataset_name.return_value = "Dataset001_Test"
        return patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "nnunetv2": MagicMock(),
                "nnunetv2.paths": mock_paths,
                "nnunetv2.utilities": MagicMock(),
                "nnunetv2.utilities.dataset_name_id_conversion": mock_conversion,
            },
        )

    def test_plans_file_permission_error(self, tmp_path, capsys) -> None:
        """OSError on plans file produces clean error message."""
        dataset_dir = self._make_preprocessed_dir(
            tmp_path, plans_content={"x": 1}, dataset_content={"labels": {}}
        )
        plans_file = dataset_dir / "nnUNetPlans.json"
        plans_file.chmod(0o000)
        try:
            with self._mock_nnunet_modules(tmp_path):
                with pytest.raises(SystemExit) as exc_info:
                    _run_with_tracked_trainer(
                        tracked_class=MagicMock,
                        dataset_name_or_id="1",
                        configuration="3d_fullres",
                        fold="0",
                        plans_identifier="nnUNetPlans",
                        continue_training=False,
                    )
                assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Cannot read plans file" in captured.err
        finally:
            plans_file.chmod(0o644)

    def test_dataset_json_permission_error(self, tmp_path, capsys) -> None:
        """OSError on dataset.json produces clean error message."""
        dataset_dir = self._make_preprocessed_dir(
            tmp_path, plans_content={"x": 1}, dataset_content={"labels": {}}
        )
        dataset_file = dataset_dir / "dataset.json"
        dataset_file.chmod(0o000)
        try:
            with self._mock_nnunet_modules(tmp_path):
                with pytest.raises(SystemExit) as exc_info:
                    _run_with_tracked_trainer(
                        tracked_class=MagicMock,
                        dataset_name_or_id="1",
                        configuration="3d_fullres",
                        fold="0",
                        plans_identifier="nnUNetPlans",
                        continue_training=False,
                    )
                assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Cannot read dataset JSON" in captured.err
        finally:
            dataset_file.chmod(0o644)


class TestNewNnunetArgs:
    """Tests for newly added nnUNetv2_train arguments in the parser."""

    def test_pretrained_weights_arg(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(
            ["Dataset001", "3d_fullres", "0", "-pretrained_weights", "/path/to/weights.pth"]
        )
        assert args.pretrained_weights == "/path/to/weights.pth"

    def test_pretrained_weights_default_none(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.pretrained_weights is None

    def test_num_gpus_default(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.num_gpus == 1

    def test_num_gpus_explicit(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "-num_gpus", "4"])
        assert args.num_gpus == 4

    def test_npz_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "--npz"])
        assert args.npz is True

    def test_val_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "--val"])
        assert args.val is True

    def test_val_best_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "--val_best"])
        assert args.val_best is True

    def test_disable_checkpointing_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(
            ["Dataset001", "3d_fullres", "0", "--disable_checkpointing"]
        )
        assert args.disable_checkpointing is True

    def test_device_arg(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "-device", "cpu"])
        assert args.device == "cpu"

    def test_device_default(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.device == "cuda"

    def test_use_compressed_flag(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "--use_compressed"])
        assert args.use_compressed is True

    def test_use_compressed_default_false(self) -> None:
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0"])
        assert args.use_compressed is False


class TestArgumentValidation:
    """Tests for mutual exclusion and argument validation in _run_train."""

    def _make_args(self, tracker_disable: bool = False) -> MagicMock:
        args = MagicMock()
        args.tracker_disable = tracker_disable
        return args

    def test_c_and_val_mutually_exclusive(self, capsys) -> None:
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "-c", "--val"])
            assert exc_info.value.code == 1
        assert "Cannot use -c and --val" in capsys.readouterr().err

    def test_pretrained_weights_and_c_mutually_exclusive(self, capsys, tmp_path) -> None:
        args = self._make_args()
        weights_file = tmp_path / "weights.pth"
        weights_file.touch()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(
                    args,
                    [
                        "Dataset001",
                        "3d_fullres",
                        "0",
                        "-c",
                        "-pretrained_weights",
                        str(weights_file),
                    ],
                )
            assert exc_info.value.code == 1
        assert "Cannot use -pretrained_weights and -c" in capsys.readouterr().err

    def test_val_best_and_disable_checkpointing_mutually_exclusive(self, capsys) -> None:
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(
                    args,
                    ["Dataset001", "3d_fullres", "0", "--val_best", "--disable_checkpointing"],
                )
            assert exc_info.value.code == 1
        assert "--val_best is not compatible" in capsys.readouterr().err

    def test_num_gpus_greater_than_1_rejected(self, capsys) -> None:
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "-num_gpus", "2"])
            assert exc_info.value.code == 1
        assert "not supported" in capsys.readouterr().err

    def test_invalid_device_rejected(self, capsys) -> None:
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "-device", "tpu"])
            assert exc_info.value.code == 1
        assert "Invalid device" in capsys.readouterr().err

    def test_pretrained_weights_file_not_found(self, capsys) -> None:
        args = self._make_args()
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(
                    args,
                    [
                        "Dataset001",
                        "3d_fullres",
                        "0",
                        "-pretrained_weights",
                        "/nonexistent/weights.pth",
                    ],
                )
            assert exc_info.value.code == 1
        assert "Pretrained weights file not found" in capsys.readouterr().err


class TestNewTrainingBehaviors:
    """Tests for new nnUNet-parity behaviors in _run_with_tracked_trainer."""

    def _make_preprocessed_dir(self, tmp_path):
        import json

        dataset_dir = tmp_path / "Dataset001_Test"
        dataset_dir.mkdir()
        plans_file = dataset_dir / "nnUNetPlans.json"
        plans_file.write_text(json.dumps({"dataset_name": "Dataset001_Test"}), encoding="utf-8")
        dataset_file = dataset_dir / "dataset.json"
        dataset_file.write_text(json.dumps({"labels": {"background": 0}}), encoding="utf-8")
        return tmp_path

    def _mock_nnunet_modules(self, tmp_path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        mock_paths = MagicMock()
        mock_paths.nnUNet_preprocessed = str(tmp_path)
        mock_conversion = MagicMock()
        mock_conversion.maybe_convert_to_dataset_name.return_value = "Dataset001_Test"
        return patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "nnunetv2": MagicMock(),
                "nnunetv2.paths": mock_paths,
                "nnunetv2.utilities": MagicMock(),
                "nnunetv2.utilities.dataset_name_id_conversion": mock_conversion,
            },
        )

    def _run(self, tmp_path, trainer_class, **kwargs):
        self._make_preprocessed_dir(tmp_path)
        defaults = dict(
            tracked_class=trainer_class,
            dataset_name_or_id="1",
            configuration="3d_fullres",
            fold="0",
            plans_identifier="nnUNetPlans",
            continue_training=False,
            pretrained_weights=None,
            export_validation_probabilities=False,
            only_run_validation=False,
            val_with_best=False,
            disable_checkpointing=False,
            device_name="cpu",
        )
        defaults.update(kwargs)
        with self._mock_nnunet_modules(tmp_path):
            _run_with_tracked_trainer(**defaults)

    def test_disable_checkpointing_set_on_trainer(self, tmp_path):
        """--disable_checkpointing sets trainer.disable_checkpointing = True."""
        instances = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")
            disable_checkpointing = False

            def __init__(self, **kwargs):
                instances.append(self)

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, disable_checkpointing=True)
        assert instances[0].disable_checkpointing is True

    def test_use_compressed_passes_unpack_false(self, tmp_path):
        """--use_compressed passes unpack_dataset=False to trainer constructor."""
        init_kwargs_list = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                init_kwargs_list.append(kwargs)

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, use_compressed_data=True)
        assert init_kwargs_list[0]["unpack_dataset"] is False

    def test_default_unpack_dataset_true(self, tmp_path):
        """By default, unpack_dataset=True is passed to trainer constructor."""
        init_kwargs_list = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                init_kwargs_list.append(kwargs)

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer)
        assert init_kwargs_list[0]["unpack_dataset"] is True

    def test_npz_passed_to_validation(self, tmp_path):
        """--npz passes save_probabilities=True to perform_actual_validation."""
        val_kwargs = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                val_kwargs.append(save_probabilities)

        self._run(tmp_path, FakeTrainer, export_validation_probabilities=True)
        assert val_kwargs == [True]

    def test_val_only_mode_skips_training(self, tmp_path):
        """--val skips run_training, only validates."""
        call_order = []
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_final.pth").touch()

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                call_order.append("train")

            def load_checkpoint(self, path):
                call_order.append("load_ckpt")

            def perform_actual_validation(self, save_probabilities=False):
                call_order.append("validate")

        self._run(tmp_path, FakeTrainer, only_run_validation=True)
        assert "train" not in call_order
        assert "validate" in call_order
        assert "load_ckpt" in call_order

    def test_val_only_exits_when_no_final_checkpoint(self, tmp_path, capsys):
        """--val exits with error if checkpoint_final.pth not found."""

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with pytest.raises(SystemExit) as exc_info:
            self._run(tmp_path, FakeTrainer, only_run_validation=True)
        assert exc_info.value.code == 1
        assert "training is not finished" in capsys.readouterr().err

    def test_val_best_loads_best_checkpoint(self, tmp_path):
        """--val_best loads checkpoint_best before validation."""
        loaded = []
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_best.pth").touch()

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                loaded.append(path)

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, val_with_best=True)
        assert str(output_dir / "checkpoint_best.pth") in loaded

    def test_without_val_best_does_not_load_best(self, tmp_path):
        """Without --val_best, checkpoint_best is NOT loaded before validation."""
        loaded = []

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                loaded.append(path)

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer)
        assert loaded == []

    def test_checkpoint_priority_final_first(self, tmp_path):
        """With -c, checkpoint_final.pth is checked first (nnUNet behavior)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_final.pth").touch()
        (output_dir / "checkpoint_latest.pth").touch()

        loaded = []

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                loaded.append(path)

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, continue_training=True)
        assert loaded[0] == str(output_dir / "checkpoint_final.pth")

    def test_checkpoint_fallback_to_latest(self, tmp_path):
        """With -c and only latest checkpoint, loads latest."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_latest.pth").touch()

        loaded = []

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                loaded.append(path)

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, continue_training=True)
        assert loaded[0] == str(output_dir / "checkpoint_latest.pth")

    def test_checkpoint_fallback_to_best(self, tmp_path):
        """With -c and only best checkpoint, loads best."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        (output_dir / "checkpoint_best.pth").touch()

        loaded = []

        class FakeTrainer:
            output_folder = str(output_dir)

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def load_checkpoint(self, path):
                loaded.append(path)

            def perform_actual_validation(self, save_probabilities=False):
                pass

        self._run(tmp_path, FakeTrainer, continue_training=True)
        assert loaded[0] == str(output_dir / "checkpoint_best.pth")


class TestDelegationOfNewArgs:
    """Tests that _run_train forwards new args to _run_with_tracked_trainer."""

    def _make_args(self, tracker_disable: bool = False) -> MagicMock:
        args = MagicMock()
        args.tracker_disable = tracker_disable
        return args

    def test_forwards_new_args(self) -> None:
        """_run_train forwards all new nnUNet args to _run_with_tracked_trainer."""
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
            _run_train(
                args,
                [
                    "Dataset001",
                    "3d_fullres",
                    "0",
                    "--npz",
                    "--val_best",
                    "-device",
                    "cpu",
                ],
            )

        call_kwargs = mock_run_with.call_args[1]
        assert call_kwargs["export_validation_probabilities"] is True
        assert call_kwargs["val_with_best"] is True
        assert call_kwargs["device_name"] == "cpu"
        assert call_kwargs["only_run_validation"] is False
        assert call_kwargs["disable_checkpointing"] is False
        assert call_kwargs["pretrained_weights"] is None
        assert call_kwargs["use_compressed_data"] is False

    def test_forwards_use_compressed(self) -> None:
        """_run_train forwards --use_compressed to _run_with_tracked_trainer."""
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
            _run_train(
                args,
                ["Dataset001", "3d_fullres", "0", "--use_compressed"],
            )

        call_kwargs = mock_run_with.call_args[1]
        assert call_kwargs["use_compressed_data"] is True


class TestContinueFlagDoubleDash:
    """Tests that --c (double-dash, nnUNet convention) works in nnunet-tracker."""

    def test_continue_flag_double_dash(self) -> None:
        """--c (double-dash) is recognized by the arg parser."""
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "--c"])
        assert args.c is True

    def test_continue_flag_single_dash(self) -> None:
        """-c (single-dash) still works."""
        parser = _build_nnunet_arg_parser()
        args, _ = parser.parse_known_args(["Dataset001", "3d_fullres", "0", "-c"])
        assert args.c is True

    def test_c_double_dash_and_val_mutually_exclusive(self, capsys) -> None:
        """--c (double-dash) + --val is rejected."""
        args = MagicMock()
        args.tracker_disable = False
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=True):
            with pytest.raises(SystemExit) as exc_info:
                _run_train(args, ["Dataset001", "3d_fullres", "0", "--c", "--val"])
            assert exc_info.value.code == 1
        assert "Cannot use -c and --val" in capsys.readouterr().err


class TestBlasThreadEnvVars:
    """Tests that BLAS thread environment variables are set."""

    def test_omp_thread_env_vars_set(self, monkeypatch) -> None:
        """_run_train sets OMP/MKL/OPENBLAS/TORCHINDUCTOR thread limits."""
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "TORCHINDUCTOR_COMPILE_THREADS",
        ):
            monkeypatch.delenv(var, raising=False)

        args = MagicMock()
        args.tracker_disable = False
        # Exit early when nnUNet is not available -- we only need to verify env vars
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=False):
            with pytest.raises(SystemExit):
                _run_train(args, ["Dataset001", "3d_fullres", "0"])

        import os

        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"

    def test_omp_thread_env_vars_respect_existing(self, monkeypatch) -> None:
        """_run_train does not override user-set thread variables."""
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        monkeypatch.setenv("MKL_NUM_THREADS", "8")

        args = MagicMock()
        args.tracker_disable = False
        with patch("nnunet_tracker._compat.check_nnunet_available", return_value=False):
            with pytest.raises(SystemExit):
                _run_train(args, ["Dataset001", "3d_fullres", "0"])

        import os

        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "8"


class TestTorchThreadAndCudnnSettings:
    """Tests for torch thread settings and cuDNN benchmark configuration."""

    def _make_preprocessed_dir(self, tmp_path):
        import json

        dataset_dir = tmp_path / "Dataset001_Test"
        dataset_dir.mkdir()
        plans_file = dataset_dir / "nnUNetPlans.json"
        plans_file.write_text(json.dumps({"dataset_name": "Dataset001_Test"}), encoding="utf-8")
        dataset_file = dataset_dir / "dataset.json"
        dataset_file.write_text(json.dumps({"labels": {"background": 0}}), encoding="utf-8")
        return tmp_path

    def _mock_nnunet_modules(self, tmp_path, cuda_available=False):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.device.return_value = MagicMock()

        mock_paths = MagicMock()
        mock_paths.nnUNet_preprocessed = str(tmp_path)

        mock_conversion = MagicMock()
        mock_conversion.maybe_convert_to_dataset_name.return_value = "Dataset001_Test"

        return mock_torch, patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "torch.backends": MagicMock(),
                "torch.backends.cudnn": MagicMock(),
                "nnunetv2": MagicMock(),
                "nnunetv2.paths": mock_paths,
                "nnunetv2.utilities": MagicMock(),
                "nnunetv2.utilities.dataset_name_id_conversion": mock_conversion,
            },
        )

    def test_cuda_sets_threads_to_1(self, tmp_path):
        """CUDA device sets torch threads to 1 (matching nnUNet behavior)."""
        self._make_preprocessed_dir(tmp_path)
        mock_torch, ctx = self._mock_nnunet_modules(tmp_path, cuda_available=False)

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with ctx:
            _run_with_tracked_trainer(
                tracked_class=FakeTrainer,
                dataset_name_or_id="1",
                configuration="3d_fullres",
                fold="0",
                plans_identifier="nnUNetPlans",
                continue_training=False,
                device_name="cuda",
            )

        mock_torch.set_num_threads.assert_called_with(1)
        mock_torch.set_num_interop_threads.assert_called_with(1)

    def test_cpu_sets_threads_to_cpu_count(self, tmp_path):
        """CPU device sets torch threads to multiprocessing.cpu_count()."""
        import multiprocessing

        self._make_preprocessed_dir(tmp_path)
        mock_torch, ctx = self._mock_nnunet_modules(tmp_path, cuda_available=False)

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with ctx:
            _run_with_tracked_trainer(
                tracked_class=FakeTrainer,
                dataset_name_or_id="1",
                configuration="3d_fullres",
                fold="0",
                plans_identifier="nnUNetPlans",
                continue_training=False,
                device_name="cpu",
            )

        mock_torch.set_num_threads.assert_called_with(multiprocessing.cpu_count())

    def test_mps_does_not_set_threads(self, tmp_path):
        """MPS device does not set thread counts (matches nnUNet behavior)."""
        self._make_preprocessed_dir(tmp_path)
        mock_torch, ctx = self._mock_nnunet_modules(tmp_path, cuda_available=False)

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with ctx:
            _run_with_tracked_trainer(
                tracked_class=FakeTrainer,
                dataset_name_or_id="1",
                configuration="3d_fullres",
                fold="0",
                plans_identifier="nnUNetPlans",
                continue_training=False,
                device_name="mps",
            )

        mock_torch.set_num_threads.assert_not_called()


class TestPlansInfoMessage:
    """Tests for the plans info message (ResEnc preset recommendation)."""

    def _make_preprocessed_dir(self, tmp_path, plans_name="nnUNetPlans"):
        import json

        dataset_dir = tmp_path / "Dataset001_Test"
        dataset_dir.mkdir(exist_ok=True)
        plans_file = dataset_dir / (plans_name + ".json")
        plans_file.write_text(json.dumps({"dataset_name": "Dataset001_Test"}), encoding="utf-8")
        dataset_file = dataset_dir / "dataset.json"
        if not dataset_file.exists():
            dataset_file.write_text(json.dumps({"labels": {"background": 0}}), encoding="utf-8")
        return tmp_path

    def _mock_nnunet_modules(self, tmp_path):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()

        mock_paths = MagicMock()
        mock_paths.nnUNet_preprocessed = str(tmp_path)

        mock_conversion = MagicMock()
        mock_conversion.maybe_convert_to_dataset_name.return_value = "Dataset001_Test"

        return patch.dict(
            "sys.modules",
            {
                "torch": mock_torch,
                "torch.cuda": mock_torch.cuda,
                "nnunetv2": MagicMock(),
                "nnunetv2.paths": mock_paths,
                "nnunetv2.utilities": MagicMock(),
                "nnunetv2.utilities.dataset_name_id_conversion": mock_conversion,
            },
        )

    def test_default_plans_info_message(self, tmp_path, capsys):
        """Info message shown when plans_identifier is 'nnUNetPlans'."""
        self._make_preprocessed_dir(tmp_path)

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with self._mock_nnunet_modules(tmp_path):
            _run_with_tracked_trainer(
                tracked_class=FakeTrainer,
                dataset_name_or_id="1",
                configuration="3d_fullres",
                fold="0",
                plans_identifier="nnUNetPlans",
                continue_training=False,
            )

        captured = capsys.readouterr()
        assert "old nnU-Net default plans" in captured.out

    def test_no_info_message_for_custom_plans(self, tmp_path, capsys):
        """No info message for non-default plans identifiers."""
        self._make_preprocessed_dir(tmp_path, plans_name="nnUNetResEncUNetLPlans")

        class FakeTrainer:
            output_folder = str(tmp_path / "output")

            def __init__(self, **kwargs):
                pass

            def run_training(self):
                pass

            def perform_actual_validation(self, save_probabilities=False):
                pass

        with self._mock_nnunet_modules(tmp_path):
            _run_with_tracked_trainer(
                tracked_class=FakeTrainer,
                dataset_name_or_id="1",
                configuration="3d_fullres",
                fold="0",
                plans_identifier="nnUNetResEncUNetLPlans",
                continue_training=False,
            )

        captured = capsys.readouterr()
        assert "old nnU-Net default plans" not in captured.out

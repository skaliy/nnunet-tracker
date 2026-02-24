"""Shared test fixtures for nnunet-tracker."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from nnunet_tracker.config import TrackerConfig


@dataclass
class MockMLflowRunInfo:
    """Mimics mlflow.entities.RunInfo."""

    run_id: str = "run-0"
    status: str = "FINISHED"
    experiment_id: str = "exp-1"
    start_time: int = 1000


@dataclass
class MockMLflowRunData:
    """Mimics mlflow.entities.RunData."""

    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)


@dataclass
class MockMLflowRun:
    """Mimics mlflow.entities.Run."""

    info: MockMLflowRunInfo = field(default_factory=MockMLflowRunInfo)
    data: MockMLflowRunData = field(default_factory=MockMLflowRunData)


@dataclass
class MockMLflowExperiment:
    """Mimics mlflow.entities.Experiment."""

    experiment_id: str = "exp-1"
    name: str = "test_experiment"


def mock_mlflow_modules(mock_client: MagicMock):
    """Create a mock mlflow + mlflow.tracking + mlflow.entities module pair for testing.

    The summarize/list modules use lazy imports:
        from mlflow.tracking import MlflowClient
        from mlflow.entities import Metric
    We mock sys.modules entries so these resolve correctly.
    """
    mock_mlflow = MagicMock()
    mock_tracking = MagicMock()
    mock_tracking.MlflowClient = MagicMock(return_value=mock_client)
    mock_mlflow.tracking = mock_tracking

    mock_entities = MagicMock()
    mock_entities.Metric = _MockMetric
    mock_mlflow.entities = mock_entities

    return patch.dict(
        "sys.modules",
        {
            "mlflow": mock_mlflow,
            "mlflow.tracking": mock_tracking,
            "mlflow.entities": mock_entities,
        },
    )


class _MockMetric:
    """Lightweight stand-in for mlflow.entities.Metric."""

    def __init__(self, key: str, value: float, timestamp: int, step: int):
        self.key = key
        self.value = value
        self.timestamp = timestamp
        self.step = step


class MockLogger:
    """Mimics nnUNetLogger with my_fantastic_logging dict."""

    def __init__(self):
        self.my_fantastic_logging = {
            "train_losses": [0.5, 0.4, 0.3],
            "val_losses": [0.6, 0.5, 0.4],
            "dice_per_class_or_region": [
                [0.7, 0.8],
                [0.75, 0.85],
                [0.8, 0.9],
            ],
            "mean_fg_dice": [0.75, 0.8, 0.85],
            "ema_fg_dice": [0.74, 0.79, 0.84],
            "epoch_start_timestamps": [1000.0, 1100.0, 1200.0],
            "epoch_end_timestamps": [1050.0, 1150.0, 1250.0],
        }


class MockPlansManager:
    """Mimics PlansManager."""

    def __init__(self):
        self.dataset_name = "Dataset001_BrainTumour"
        self.plans_name = "nnUNetPlans"
        self.plans = {
            "dataset_name": "Dataset001_BrainTumour",
            "plans_name": "nnUNetPlans",
            "original_median_spacing_after_transp": [1.0, 1.0, 1.0],
            "original_median_shape_after_transp": [128, 256, 256],
            "image_reader_writer": "SimpleITKIO",
            "transpose_forward": [0, 1, 2],
            "transpose_backward": [0, 1, 2],
            "configurations": {
                "3d_fullres": {"batch_size": 2, "patch_size": [128, 128, 128]},
            },
        }


class MockConfigurationManager:
    """Mimics ConfigurationManager."""

    def __init__(self):
        self.batch_size = 2
        self.patch_size = [128, 128, 128]
        self.spacing = [1.0, 1.0, 1.0]
        self.UNet_class_name = "PlainConvUNet"
        self.network_arch_class_name = (
            "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        )
        self.normalization_schemes = ["ZScoreNormalization"]
        self.use_mask_for_norm = [True]
        self.preprocessor_name = "DefaultPreprocessor"
        self.data_identifier = "nnUNetPlans_3d_fullres"
        self.batch_dice = True
        self.network_arch_init_kwargs = {
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[3, 3, 3]] * 6,
            "strides": [[1, 1, 1]] + [[2, 2, 2]] * 5,
            "n_conv_per_stage": [2, 2, 2, 2, 2, 2],
            "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        }
        self.configuration = {
            "resampling_fn_data": "resample_data_or_seg_to_shape",
            "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0},
            "resampling_fn_seg": "resample_data_or_seg_to_shape",
            "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0},
        }


class MockTrainerBase:
    """Mimics nnUNetTrainer with the interface that hooks expect.

    This is the 'base_class' that create_tracked_trainer wraps.
    """

    def __init__(self, plans=None, configuration=None, fold=0, dataset_json=None, device=None):
        self.plans_manager = MockPlansManager()
        self.configuration_manager = MockConfigurationManager()
        self.logger = MockLogger()
        self.optimizer = MagicMock()
        self.optimizer.param_groups = [{"lr": 0.01, "weight_decay": 3e-5}]
        self.current_epoch = 0
        self.num_epochs = 1000
        self.initial_lr = 0.01
        self.weight_decay = 3e-5
        self.fold = fold
        self.configuration_name = configuration or "3d_fullres"
        self.output_folder = "/nonexistent/nnunet_output"
        self.is_ddp = False
        self.local_rank = 0
        self.dataset_json = dataset_json or {
            "name": "BrainTumour",
            "numTraining": 100,
            "labels": {"background": 0, "tumor": 1, "edema": 2},
        }
        self.enable_deep_supervision = True
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33

    def run_training(self):
        self.on_train_start()
        self.on_epoch_start()
        self.on_train_epoch_start()
        self.on_train_epoch_end([{"loss": 0.3}])
        self.on_validation_epoch_start()
        self.on_validation_epoch_end([{"loss": 0.4}])
        self.on_epoch_end()
        self.on_train_end()

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_epoch_start(self):
        pass

    def on_train_epoch_start(self):
        pass

    def on_train_epoch_end(self, train_outputs):
        pass

    def on_validation_epoch_start(self):
        pass

    def on_validation_epoch_end(self, val_outputs):
        pass

    def on_epoch_end(self):
        # Mimics real nnU-Net which increments current_epoch in on_epoch_end
        self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities=False):
        pass

    def load_checkpoint(self, filename_or_checkpoint):
        pass


@pytest.fixture
def mock_trainer_class():
    """Return MockTrainerBase as a stand-in for nnUNetTrainer."""
    return MockTrainerBase


@pytest.fixture
def mock_trainer():
    """Return a MockTrainerBase instance."""
    return MockTrainerBase()


@pytest.fixture
def tracker_config_enabled():
    """Return a TrackerConfig with tracking enabled."""
    return TrackerConfig(
        tracking_uri="./test_mlruns",
        experiment_name="test_experiment",
        enabled=True,
        log_artifacts=False,
    )


@pytest.fixture
def tracker_config_disabled():
    """Return a TrackerConfig with tracking disabled."""
    return TrackerConfig(
        tracking_uri="./test_mlruns",
        experiment_name="test_experiment",
        enabled=False,
        log_artifacts=False,
    )


@pytest.fixture
def mock_mlflow():
    """Provide a mock mlflow module injected into sys.modules."""
    mock = MagicMock()
    mock.start_run.return_value = MagicMock(info=MagicMock(run_id="test-run-id"))
    mock.active_run.return_value = None  # No active run by default
    with patch.dict("sys.modules", {"mlflow": mock}):
        yield mock

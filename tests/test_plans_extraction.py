"""Tests for nnunet_tracker.plans_extraction module."""

from __future__ import annotations

import json

from nnunet_tracker.plans_extraction import (
    _str_truncate,
    extract_all_extended_params,
    extract_network_topology_params,
    extract_preprocessing_params,
    extract_trainer_hyperparams,
    get_plans_json_str,
)


class TestStrTruncate:
    """Tests for _str_truncate helper."""

    def test_short_string_unchanged(self) -> None:
        assert _str_truncate("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        long = "x" * 600
        result = _str_truncate(long)
        assert len(result) == 500
        assert result.endswith("...")
        assert result == "x" * 497 + "..."

    def test_converts_non_string(self) -> None:
        assert _str_truncate([1, 2, 3]) == "[1, 2, 3]"

    def test_exactly_500_chars_unchanged(self) -> None:
        """String of exactly _MAX_PARAM_LEN chars should not be truncated."""
        s = "x" * 500
        result = _str_truncate(s)
        assert len(result) == 500
        assert result == s


class TestExtractPreprocessingParams:
    """Tests for extract_preprocessing_params."""

    def test_extracts_normalization_schemes(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_normalization_schemes"] == "['ZScoreNormalization']"

    def test_extracts_use_mask_for_norm(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_use_mask_for_norm"] == "[True]"

    def test_extracts_preprocessor_name(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_preprocessor_name"] == "DefaultPreprocessor"

    def test_extracts_data_identifier(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_data_identifier"] == "nnUNetPlans_3d_fullres"

    def test_extracts_batch_dice(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_batch_dice"] is True

    def test_extracts_resampling_params(self, mock_trainer) -> None:
        params = extract_preprocessing_params(mock_trainer)
        assert params["preproc_resampling_fn_data"] == "resample_data_or_seg_to_shape"
        assert params["preproc_resampling_fn_seg"] == "resample_data_or_seg_to_shape"
        assert "preproc_resampling_fn_data_kwargs" in params
        assert "preproc_resampling_fn_seg_kwargs" in params

    def test_missing_configuration_manager(self, mock_trainer) -> None:
        """Returns empty dict when cm is None."""
        mock_trainer.configuration_manager = None
        params = extract_preprocessing_params(mock_trainer)
        assert params == {}

    def test_partial_attrs(self, mock_trainer) -> None:
        """Handles cm with some attributes missing."""
        del mock_trainer.configuration_manager.preprocessor_name
        params = extract_preprocessing_params(mock_trainer)
        assert "preproc_preprocessor_name" not in params
        # Other params still present
        assert "preproc_normalization_schemes" in params

    def test_configuration_not_dict(self, mock_trainer) -> None:
        """Handles cm.configuration that is not a dict."""
        mock_trainer.configuration_manager.configuration = "not_a_dict"
        params = extract_preprocessing_params(mock_trainer)
        # Resampling params should be absent, others still present
        assert "preproc_resampling_fn_data" not in params
        assert "preproc_normalization_schemes" in params

    def test_no_none_values(self, mock_trainer) -> None:
        """No None values in output dict."""
        params = extract_preprocessing_params(mock_trainer)
        assert all(v is not None for v in params.values())


class TestExtractNetworkTopologyParams:
    """Tests for extract_network_topology_params."""

    def test_extracts_class_name(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        expected = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
        assert params["net_class_name"] == expected

    def test_extracts_n_stages(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert params["net_n_stages"] == 6

    def test_extracts_features_per_stage(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert params["net_features_per_stage"] == "[32, 64, 128, 256, 320, 320]"

    def test_extracts_kernel_sizes(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert "net_kernel_sizes" in params

    def test_extracts_strides(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert "net_strides" in params

    def test_extracts_n_conv_per_stage(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert params["net_n_conv_per_stage"] == "[2, 2, 2, 2, 2, 2]"

    def test_extracts_conv_op(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert params["net_conv_op"] == "torch.nn.modules.conv.Conv3d"

    def test_extracts_norm_op(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert "InstanceNorm3d" in params["net_norm_op"]

    def test_extracts_nonlin(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert "LeakyReLU" in params["net_nonlin"]

    def test_missing_configuration_manager(self, mock_trainer) -> None:
        mock_trainer.configuration_manager = None
        params = extract_network_topology_params(mock_trainer)
        assert params == {}

    def test_missing_arch_kwargs(self, mock_trainer) -> None:
        """Returns only class name when arch_kwargs is missing."""
        mock_trainer.configuration_manager.network_arch_init_kwargs = None
        params = extract_network_topology_params(mock_trainer)
        assert "net_class_name" in params
        assert "net_n_stages" not in params

    def test_arch_kwargs_not_dict(self, mock_trainer) -> None:
        """Handles non-dict arch_kwargs gracefully."""
        mock_trainer.configuration_manager.network_arch_init_kwargs = "not_a_dict"
        params = extract_network_topology_params(mock_trainer)
        assert "net_class_name" in params
        assert "net_n_stages" not in params

    def test_no_none_values(self, mock_trainer) -> None:
        params = extract_network_topology_params(mock_trainer)
        assert all(v is not None for v in params.values())


class TestExtractTrainerHyperparams:
    """Tests for extract_trainer_hyperparams."""

    def test_deep_supervision(self, mock_trainer) -> None:
        params = extract_trainer_hyperparams(mock_trainer)
        assert params["deep_supervision"] is True

    def test_iterations_per_epoch(self, mock_trainer) -> None:
        params = extract_trainer_hyperparams(mock_trainer)
        assert params["num_iterations_per_epoch"] == 250
        assert params["num_val_iterations_per_epoch"] == 50

    def test_oversample_foreground(self, mock_trainer) -> None:
        params = extract_trainer_hyperparams(mock_trainer)
        assert params["oversample_foreground_pct"] == 0.33

    def test_mirror_axes(self, mock_trainer) -> None:
        """Mirror axes logged when present."""
        mock_trainer.allowed_mirroring_axes = (0, 1, 2)
        params = extract_trainer_hyperparams(mock_trainer)
        assert params["mirror_axes"] == "(0, 1, 2)"

    def test_mirror_axes_absent(self, mock_trainer) -> None:
        """No mirror_axes when attribute missing."""
        params = extract_trainer_hyperparams(mock_trainer)
        assert "mirror_axes" not in params

    def test_label_names(self, mock_trainer) -> None:
        """Label names extracted from dataset_json."""
        params = extract_trainer_hyperparams(mock_trainer)
        assert "label_names" in params
        assert "tumor" in params["label_names"]

    def test_label_names_missing(self, mock_trainer) -> None:
        """No label_names when dataset_json has no labels key."""
        mock_trainer.dataset_json = {"name": "test"}
        params = extract_trainer_hyperparams(mock_trainer)
        assert "label_names" not in params

    def test_dataset_json_not_dict(self, mock_trainer) -> None:
        """Handles non-dict dataset_json."""
        mock_trainer.dataset_json = None
        params = extract_trainer_hyperparams(mock_trainer)
        assert "label_names" not in params

    def test_missing_attrs(self, mock_trainer) -> None:
        """Handles trainer with missing hyperparams."""
        del mock_trainer.enable_deep_supervision
        del mock_trainer.num_iterations_per_epoch
        params = extract_trainer_hyperparams(mock_trainer)
        assert "deep_supervision" not in params
        assert "num_iterations_per_epoch" not in params

    def test_no_none_values(self, mock_trainer) -> None:
        params = extract_trainer_hyperparams(mock_trainer)
        assert all(v is not None for v in params.values())


class TestExtractAllExtendedParams:
    """Tests for extract_all_extended_params."""

    def test_merges_all_sources(self, mock_trainer) -> None:
        params = extract_all_extended_params(mock_trainer)
        # Should contain params from all three sources
        assert "preproc_normalization_schemes" in params
        assert "net_n_stages" in params
        assert "deep_supervision" in params

    def test_no_none_values(self, mock_trainer) -> None:
        params = extract_all_extended_params(mock_trainer)
        assert all(v is not None for v in params.values())

    def test_empty_when_no_data(self, mock_trainer) -> None:
        """Returns empty when trainer has no relevant attributes."""
        mock_trainer.configuration_manager = None
        mock_trainer.dataset_json = None
        del mock_trainer.enable_deep_supervision
        del mock_trainer.num_iterations_per_epoch
        del mock_trainer.num_val_iterations_per_epoch
        del mock_trainer.oversample_foreground_percent
        params = extract_all_extended_params(mock_trainer)
        assert params == {}


class TestGetPlansJsonStr:
    """Tests for get_plans_json_str."""

    def test_returns_json_string(self, mock_trainer) -> None:
        result = get_plans_json_str(mock_trainer)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["dataset_name"] == "Dataset001_BrainTumour"

    def test_sorted_keys(self, mock_trainer) -> None:
        result = get_plans_json_str(mock_trainer)
        parsed_keys = list(json.loads(result).keys())
        assert parsed_keys == sorted(parsed_keys)

    def test_none_when_no_plans(self, mock_trainer) -> None:
        mock_trainer.plans_manager = None
        assert get_plans_json_str(mock_trainer) is None

    def test_none_when_plans_not_dict(self, mock_trainer) -> None:
        mock_trainer.plans_manager.plans = "not_a_dict"
        assert get_plans_json_str(mock_trainer) is None

    def test_none_when_no_plans_manager(self, mock_trainer) -> None:
        mock_trainer.plans_manager = None
        assert get_plans_json_str(mock_trainer) is None

    def test_handles_non_serializable(self, mock_trainer) -> None:
        """Non-serializable values in plans handled via default=str."""
        from pathlib import Path

        mock_trainer.plans_manager.plans["custom"] = Path("/some/path")
        result = get_plans_json_str(mock_trainer)
        assert result is not None
        assert "/some/path" in result

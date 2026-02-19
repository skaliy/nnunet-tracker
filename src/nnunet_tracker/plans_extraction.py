"""nnU-Net plans and configuration parameter extraction."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("nnunet_tracker")

__all__ = ["extract_all_extended_params", "get_plans_json_str"]

# Maximum length for stringified param values (MLflow limit is 500 chars)
_MAX_PARAM_LEN = 500


def _str_truncate(value: Any) -> str:
    """Convert value to string, truncating to MLflow param limit."""
    s = str(value)
    return (s[: _MAX_PARAM_LEN - 3] + "...") if len(s) > _MAX_PARAM_LEN else s


def extract_preprocessing_params(trainer: Any) -> dict[str, Any]:
    """Extract preprocessing pipeline parameters from the trainer.

    Returns a flat dict suitable for mlflow.log_params().
    Keys are prefixed with 'preproc_' to namespace them.
    """
    params: dict[str, Any] = {}
    cm = getattr(trainer, "configuration_manager", None)
    if cm is None:
        return params

    norm = getattr(cm, "normalization_schemes", None)
    if norm is not None:
        params["preproc_normalization_schemes"] = _str_truncate(norm)

    mask_norm = getattr(cm, "use_mask_for_norm", None)
    if mask_norm is not None:
        params["preproc_use_mask_for_norm"] = _str_truncate(mask_norm)

    for attr_name, param_key in (
        ("preprocessor_name", "preproc_preprocessor_name"),
        ("data_identifier", "preproc_data_identifier"),
        ("batch_dice", "preproc_batch_dice"),
    ):
        val = getattr(cm, attr_name, None)
        if val is not None:
            params[param_key] = (
                val if isinstance(val, (str, int, float, bool)) else _str_truncate(val)
            )

    config_dict = getattr(cm, "configuration", None)
    if isinstance(config_dict, dict):
        for key in (
            "resampling_fn_data",
            "resampling_fn_seg",
            "resampling_fn_data_kwargs",
            "resampling_fn_seg_kwargs",
        ):
            val = config_dict.get(key)
            if val is not None:
                params[f"preproc_{key}"] = val if isinstance(val, str) else _str_truncate(val)

    return {k: v for k, v in params.items() if v is not None}


def extract_network_topology_params(trainer: Any) -> dict[str, Any]:
    """Extract network architecture/topology parameters.

    Returns a flat dict suitable for mlflow.log_params().
    Keys are prefixed with 'net_' to namespace them.
    """
    params: dict[str, Any] = {}
    cm = getattr(trainer, "configuration_manager", None)
    if cm is None:
        return params

    net_class = getattr(cm, "network_arch_class_name", None)
    if net_class is not None:
        params["net_class_name"] = _str_truncate(net_class)

    arch_kwargs = getattr(cm, "network_arch_init_kwargs", None)
    if isinstance(arch_kwargs, dict):
        # Log key topology values individually for easy MLflow filtering
        for key in ("n_stages", "conv_op", "norm_op", "nonlin"):
            val = arch_kwargs.get(key)
            if val is not None:
                params[f"net_{key}"] = (
                    val if isinstance(val, (int, float, bool)) else _str_truncate(val)
                )

        for key in (
            "features_per_stage",
            "kernel_sizes",
            "strides",
            "n_conv_per_stage",
            "n_conv_per_stage_decoder",
        ):
            val = arch_kwargs.get(key)
            if val is not None:
                params[f"net_{key}"] = _str_truncate(val)

    return {k: v for k, v in params.items() if v is not None}


def extract_trainer_hyperparams(trainer: Any) -> dict[str, Any]:
    """Extract trainer-level hyperparameters not already in core _extract_params.

    Returns a flat dict suitable for mlflow.log_params().
    """
    params: dict[str, Any] = {}

    for trainer_attr, param_key in (
        ("enable_deep_supervision", "deep_supervision"),
        ("num_iterations_per_epoch", "num_iterations_per_epoch"),
        ("num_val_iterations_per_epoch", "num_val_iterations_per_epoch"),
        ("oversample_foreground_percent", "oversample_foreground_pct"),
    ):
        val = getattr(trainer, trainer_attr, None)
        if val is not None:
            params[param_key] = val

    mirror_axes = getattr(trainer, "allowed_mirroring_axes", None)
    if mirror_axes is not None:
        params["mirror_axes"] = _str_truncate(mirror_axes)

    # Label names from dataset.json for Dice metric interpretability
    dataset_json = getattr(trainer, "dataset_json", None)
    if isinstance(dataset_json, dict):
        labels = dataset_json.get("labels")
        if labels is not None:
            params["label_names"] = _str_truncate(labels)

    return {k: v for k, v in params.items() if v is not None}


def extract_all_extended_params(trainer: Any) -> dict[str, Any]:
    """Extract all extended parameters (preprocessing, network topology, and trainer hyperparams).

    Convenience function that merges results from all extraction functions.
    """
    params: dict[str, Any] = {}
    params.update(extract_preprocessing_params(trainer))
    params.update(extract_network_topology_params(trainer))
    params.update(extract_trainer_hyperparams(trainer))
    return params


def get_plans_json_str(trainer: Any) -> str | None:
    """Get the full plans dict as a formatted JSON string for artifact logging.

    Returns None if plans are not available.
    """
    pm = getattr(trainer, "plans_manager", None)
    plans = getattr(pm, "plans", None) if pm is not None else None
    if plans is None or not isinstance(plans, dict):
        return None
    try:
        return json.dumps(plans, indent=2, sort_keys=True, ensure_ascii=True, default=str)
    except (TypeError, ValueError):
        logger.debug("Failed to serialize plans to JSON", exc_info=True)
        return None

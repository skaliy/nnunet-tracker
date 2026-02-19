"""Environment variable configuration for nnunet-tracker."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

logger = logging.getLogger("nnunet_tracker")

# Strict character allowlist for cv_group and tag values used in MLflow filter strings.
# Only alphanumeric, underscore, hyphen, dot, and pipe are allowed.
SAFE_TAG_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_\-\.\|]+$")


@dataclass(frozen=True)
class TrackerConfig:
    """Immutable configuration read from environment variables."""

    tracking_uri: str
    experiment_name: str | None
    enabled: bool
    log_artifacts: bool

    @classmethod
    def from_env(cls) -> TrackerConfig:
        """Read configuration from environment variables.

        Environment Variables:
            MLFLOW_TRACKING_URI: MLflow tracking server URI (default: ./mlruns)
            MLFLOW_EXPERIMENT_NAME: Experiment name (default: None, uses dataset name)
            NNUNET_TRACKER_ENABLED: Enable tracking, '1'/'true'/'yes' (default: '1')
            NNUNET_TRACKER_LOG_ARTIFACTS: Log artifacts, '1'/'true'/'yes' (default: '1')
        """
        return cls(
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"),
            experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
            enabled=_parse_bool(os.environ.get("NNUNET_TRACKER_ENABLED", "1")),
            log_artifacts=_parse_bool(os.environ.get("NNUNET_TRACKER_LOG_ARTIFACTS", "1")),
        )


_BOOL_TRUE = frozenset({"1", "true", "yes"})
_BOOL_FALSE = frozenset({"0", "false", "no"})


def _parse_bool(value: str) -> bool:
    """Parse a string to boolean. Accepts '1', 'true', 'yes' (case-insensitive)."""
    normalized = value.strip().lower()
    if normalized in _BOOL_TRUE:
        return True
    if normalized not in _BOOL_FALSE:
        logger.warning(
            "Unrecognized boolean value %r; treating as False. "
            "Use one of: 1/true/yes or 0/false/no.",
            value,
        )
    return False

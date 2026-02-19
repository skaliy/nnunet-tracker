"""Tests for nnunet_tracker.config module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from nnunet_tracker.config import TrackerConfig, _parse_bool


class TestParserBool:
    """Tests for _parse_bool helper."""

    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes", "YES"])
    def test_truthy_values(self, value: str) -> None:
        assert _parse_bool(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "False", "no", "No", "", "random"])
    def test_falsy_values(self, value: str) -> None:
        assert _parse_bool(value) is False

    def test_whitespace_stripped(self) -> None:
        assert _parse_bool("  true  ") is True
        assert _parse_bool("  0  ") is False


class TestTrackerConfig:
    """Tests for TrackerConfig dataclass."""

    def test_defaults_from_env_no_vars(self) -> None:
        """When no env vars are set, sensible defaults are used."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in (
                "MLFLOW_TRACKING_URI",
                "MLFLOW_EXPERIMENT_NAME",
                "NNUNET_TRACKER_ENABLED",
                "NNUNET_TRACKER_LOG_ARTIFACTS",
            )
        }
        with patch.dict(os.environ, env, clear=True):
            config = TrackerConfig.from_env()
            assert config.tracking_uri == "./mlruns"
            assert config.experiment_name is None
            assert config.enabled is True
            assert config.log_artifacts is True

    def test_custom_env_vars(self) -> None:
        """Environment variables override defaults."""
        env = {
            "MLFLOW_TRACKING_URI": "http://localhost:5000",
            "MLFLOW_EXPERIMENT_NAME": "my_experiment",
            "NNUNET_TRACKER_ENABLED": "0",
            "NNUNET_TRACKER_LOG_ARTIFACTS": "false",
        }
        with patch.dict(os.environ, env):
            config = TrackerConfig.from_env()
            assert config.tracking_uri == "http://localhost:5000"
            assert config.experiment_name == "my_experiment"
            assert config.enabled is False
            assert config.log_artifacts is False

    def test_experiment_name_none_when_unset(self) -> None:
        """MLFLOW_EXPERIMENT_NAME not set yields None."""
        env = {k: v for k, v in os.environ.items() if k != "MLFLOW_EXPERIMENT_NAME"}
        with patch.dict(os.environ, env, clear=True):
            config = TrackerConfig.from_env()
            assert config.experiment_name is None

    def test_frozen_immutability(self) -> None:
        """Config is frozen and cannot be modified."""
        config = TrackerConfig(
            tracking_uri="./mlruns",
            experiment_name=None,
            enabled=True,
            log_artifacts=True,
        )
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]

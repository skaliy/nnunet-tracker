"""Tests for nnunet_tracker._compat module."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock, patch

from nnunet_tracker._compat import (
    check_nnunet_available,
    check_nnunet_version,
    get_nnunet_version,
)


class TestCheckNnunetAvailable:
    """Tests for check_nnunet_available."""

    def test_returns_false_when_not_installed(self) -> None:
        """nnunetv2 is not installed in test environment."""
        assert check_nnunet_available() is False

    def test_returns_true_when_installed(self) -> None:
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"nnunetv2": mock_module}):
            assert check_nnunet_available() is True


class TestGetNnunetVersion:
    """Tests for get_nnunet_version."""

    def test_returns_none_when_not_installed(self) -> None:
        assert get_nnunet_version() is None

    def test_returns_version_string(self) -> None:
        with patch("importlib.metadata.version", return_value="2.5.1"):
            assert get_nnunet_version() == "2.5.1"


class TestCheckNnunetVersion:
    """Tests for check_nnunet_version."""

    def setup_method(self) -> None:
        """Clear lru_cache before each test."""
        check_nnunet_version.cache_clear()

    def test_warns_when_not_installed(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            check_nnunet_version()
            assert len(w) == 1
            assert "Could not determine" in str(w[0].message)

    def test_no_warning_when_version_in_range(self) -> None:
        with patch("nnunet_tracker._compat.get_nnunet_version", return_value="2.5.1"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_nnunet_version()
                assert len(w) == 0

    def test_warns_when_version_too_old(self) -> None:
        with patch("nnunet_tracker._compat.get_nnunet_version", return_value="2.0.0"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_nnunet_version()
                assert len(w) == 1
                assert "outside the tested range" in str(w[0].message)

    def test_warns_when_version_too_new(self) -> None:
        with patch("nnunet_tracker._compat.get_nnunet_version", return_value="2.7.0"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                check_nnunet_version()
                assert len(w) == 1
                assert "outside the tested range" in str(w[0].message)

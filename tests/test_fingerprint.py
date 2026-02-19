"""Tests for nnunet_tracker.fingerprint module."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

from nnunet_tracker.fingerprint import (
    _canonical_json_bytes,
    _sha256_hex,
    compute_fingerprint,
    find_dataset_fingerprint_path,
    hash_dict,
)


class TestCanonicalJson:
    """Tests for _canonical_json_bytes."""

    def test_sorted_keys(self) -> None:
        """Dicts with same keys in different order produce same bytes."""
        a = _canonical_json_bytes({"b": 2, "a": 1})
        b = _canonical_json_bytes({"a": 1, "b": 2})
        assert a == b

    def test_deterministic(self) -> None:
        """Same input always produces same output."""
        obj = {"key": "value", "nested": {"x": 1}}
        assert _canonical_json_bytes(obj) == _canonical_json_bytes(obj)

    def test_compact_format(self) -> None:
        """Output has no extra whitespace."""
        result = _canonical_json_bytes({"a": 1})
        assert result == b'{"a":1}'

    def test_non_serializable_uses_default_str(self) -> None:
        """Non-serializable values fall back to str() via default=str."""
        from pathlib import Path

        result = _canonical_json_bytes({"path": Path("/tmp/test")})
        assert b"/tmp/test" in result


class TestSha256Hex:
    """Tests for _sha256_hex."""

    def test_produces_hex_string(self) -> None:
        result = _sha256_hex(b"hello")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_deterministic(self) -> None:
        assert _sha256_hex(b"test") == _sha256_hex(b"test")

    def test_different_input_different_hash(self) -> None:
        assert _sha256_hex(b"a") != _sha256_hex(b"b")


class TestHashDict:
    """Tests for hash_dict."""

    def test_hashes_dict(self) -> None:
        result = hash_dict({"key": "value"})
        assert isinstance(result, str)
        assert len(result) == 64

    def test_nested_dict(self) -> None:
        result = hash_dict({"a": {"b": {"c": 1}}})
        assert isinstance(result, str)

    def test_different_order_same_hash(self) -> None:
        """Dicts with different insertion order produce same hash."""
        h1 = hash_dict({"z": 1, "a": 2})
        h2 = hash_dict({"a": 2, "z": 1})
        assert h1 == h2

    def test_different_content_different_hash(self) -> None:
        h1 = hash_dict({"a": 1})
        h2 = hash_dict({"a": 2})
        assert h1 != h2

    def test_empty_dict(self) -> None:
        result = hash_dict({})
        assert isinstance(result, str)
        assert len(result) == 64


class TestComputeFingerprint:
    """Tests for compute_fingerprint."""

    def test_all_components_present(self, mock_trainer, tmp_path) -> None:
        """Fingerprint includes all hashes when all data available."""
        # Set up dataset_fingerprint.json on disk
        fp_dir = tmp_path / "Dataset001_BrainTumour"
        fp_dir.mkdir()
        fp_file = fp_dir / "dataset_fingerprint.json"
        fp_file.write_text(json.dumps({"median_spacing": [1.0, 1.0, 1.0]}))

        with patch.dict(os.environ, {"nnUNet_preprocessed": str(tmp_path)}):
            result = compute_fingerprint(mock_trainer)

        assert "fingerprint_dataset_json" in result
        assert "fingerprint_plans" in result
        assert "fingerprint_dataset_fingerprint" in result
        assert "fingerprint_composite" in result
        assert len(result) == 4

    def test_without_dataset_fingerprint_file(self, mock_trainer) -> None:
        """Missing dataset_fingerprint.json still produces partial result."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("nnUNet_preprocessed", None)
            result = compute_fingerprint(mock_trainer)

        assert "fingerprint_dataset_json" in result
        assert "fingerprint_plans" in result
        assert "fingerprint_dataset_fingerprint" not in result
        assert "fingerprint_composite" in result

    def test_missing_dataset_json(self, mock_trainer) -> None:
        """Gracefully handles missing dataset_json."""
        mock_trainer.dataset_json = None
        result = compute_fingerprint(mock_trainer)
        assert "fingerprint_dataset_json" not in result
        assert "fingerprint_plans" in result

    def test_missing_plans_manager(self, mock_trainer) -> None:
        """Gracefully handles missing plans_manager."""
        mock_trainer.plans_manager = None
        result = compute_fingerprint(mock_trainer)
        assert "fingerprint_plans" not in result
        assert "fingerprint_dataset_json" in result

    def test_missing_everything(self, mock_trainer) -> None:
        """Returns empty dict when all data missing."""
        mock_trainer.dataset_json = None
        mock_trainer.plans_manager = None
        result = compute_fingerprint(mock_trainer)
        assert result == {}

    def test_composite_changes_when_plans_change(self, mock_trainer) -> None:
        """Composite hash changes when plans change."""
        result1 = compute_fingerprint(mock_trainer)
        mock_trainer.plans_manager.plans["new_key"] = "new_value"
        result2 = compute_fingerprint(mock_trainer)
        assert result1["fingerprint_composite"] != result2["fingerprint_composite"]
        assert result1["fingerprint_plans"] != result2["fingerprint_plans"]

    def test_dataset_json_not_dict(self, mock_trainer) -> None:
        """Non-dict dataset_json is handled gracefully."""
        mock_trainer.dataset_json = "not a dict"
        result = compute_fingerprint(mock_trainer)
        assert "fingerprint_dataset_json" not in result

    def test_plans_not_dict(self, mock_trainer) -> None:
        """Non-dict plans is handled gracefully."""
        mock_trainer.plans_manager.plans = "not a dict"
        result = compute_fingerprint(mock_trainer)
        assert "fingerprint_plans" not in result

    def test_malformed_fingerprint_file(self, mock_trainer, tmp_path) -> None:
        """Malformed dataset_fingerprint.json is skipped gracefully."""
        fp_dir = tmp_path / "Dataset001_BrainTumour"
        fp_dir.mkdir()
        fp_file = fp_dir / "dataset_fingerprint.json"
        fp_file.write_text("not valid json {{{")

        with patch.dict(os.environ, {"nnUNet_preprocessed": str(tmp_path)}):
            result = compute_fingerprint(mock_trainer)

        assert "fingerprint_dataset_fingerprint" not in result
        # Other components still present
        assert "fingerprint_dataset_json" in result
        assert "fingerprint_plans" in result

    def test_empty_dataset_json(self, mock_trainer) -> None:
        """Empty dict dataset_json produces valid hash."""
        mock_trainer.dataset_json = {}
        result = compute_fingerprint(mock_trainer)
        assert "fingerprint_dataset_json" in result
        assert len(result["fingerprint_dataset_json"]) == 64


class TestFindDatasetFingerprintPath:
    """Tests for find_dataset_fingerprint_path."""

    def test_found_when_exists(self, mock_trainer, tmp_path) -> None:
        """Returns path when file exists at expected location."""
        fp_dir = tmp_path / "Dataset001_BrainTumour"
        fp_dir.mkdir()
        fp_file = fp_dir / "dataset_fingerprint.json"
        fp_file.write_text("{}")

        with patch.dict(os.environ, {"nnUNet_preprocessed": str(tmp_path)}):
            result = find_dataset_fingerprint_path(mock_trainer)

        assert result == str(fp_file)

    def test_none_when_no_preprocessed_env(self, mock_trainer) -> None:
        """Returns None when nnUNet_preprocessed not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("nnUNet_preprocessed", None)
            result = find_dataset_fingerprint_path(mock_trainer)
        assert result is None

    def test_none_when_file_missing(self, mock_trainer, tmp_path) -> None:
        """Returns None when file does not exist."""
        with patch.dict(os.environ, {"nnUNet_preprocessed": str(tmp_path)}):
            result = find_dataset_fingerprint_path(mock_trainer)
        assert result is None

    def test_none_when_no_dataset_name(self, mock_trainer) -> None:
        """Returns None when plans_manager exists but has no dataset_name."""
        from unittest.mock import MagicMock

        mock_trainer.plans_manager = MagicMock(spec=[])  # no dataset_name attr
        with patch.dict(os.environ, {"nnUNet_preprocessed": "/tmp"}):
            result = find_dataset_fingerprint_path(mock_trainer)
        assert result is None

    def test_none_when_no_plans_manager(self, mock_trainer) -> None:
        """Returns None when trainer has no plans_manager."""
        mock_trainer.plans_manager = None
        result = find_dataset_fingerprint_path(mock_trainer)
        assert result is None

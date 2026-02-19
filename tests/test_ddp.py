"""Tests for nnunet_tracker.ddp module."""

from __future__ import annotations

import os
from unittest.mock import patch

from nnunet_tracker.ddp import is_main_process, should_log


class TestIsMainProcess:
    """Tests for is_main_process."""

    def test_default_single_gpu(self) -> None:
        """No DDP env vars set -> main process."""
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "SLURM_LOCALID")}
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True

    def test_local_rank_zero(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            assert is_main_process() is True

    def test_local_rank_nonzero(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "1"}):
            assert is_main_process() is False

    def test_local_rank_two(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            assert is_main_process() is False

    def test_slurm_rank_zero(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        env["SLURM_LOCALID"] = "0"
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True

    def test_slurm_rank_nonzero(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        env["SLURM_LOCALID"] = "2"
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is False

    def test_local_rank_takes_priority_over_slurm(self) -> None:
        with patch.dict(os.environ, {"LOCAL_RANK": "1", "SLURM_LOCALID": "0"}):
            assert is_main_process() is False

    def test_invalid_local_rank_defaults_to_main(self) -> None:
        """H2 fix: non-integer LOCAL_RANK should not crash."""
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "SLURM_LOCALID")}
        env["LOCAL_RANK"] = "abc"
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True

    def test_invalid_slurm_localid_defaults_to_main(self) -> None:
        """H2 fix: non-integer SLURM_LOCALID should not crash."""
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "SLURM_LOCALID")}
        env["SLURM_LOCALID"] = "not_a_number"
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True

    def test_empty_local_rank_defaults_to_main(self) -> None:
        """Empty string LOCAL_RANK should not crash."""
        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "SLURM_LOCALID")}
        env["LOCAL_RANK"] = ""
        with patch.dict(os.environ, env, clear=True):
            assert is_main_process() is True


class TestShouldLog:
    """Tests for should_log."""

    def test_non_ddp_trainer(self) -> None:
        """Non-DDP trainer always logs."""

        class MockTrainer:
            is_ddp = False
            local_rank = 0

        assert should_log(MockTrainer()) is True

    def test_ddp_rank_zero(self) -> None:
        class MockTrainer:
            is_ddp = True
            local_rank = 0

        assert should_log(MockTrainer()) is True

    def test_ddp_rank_nonzero(self) -> None:
        class MockTrainer:
            is_ddp = True
            local_rank = 1

        assert should_log(MockTrainer()) is False

    def test_no_ddp_attributes_defaults_to_env(self) -> None:
        """Object without DDP attributes falls back to env detection."""

        class PlainObject:
            pass

        env = {k: v for k, v in os.environ.items() if k not in ("LOCAL_RANK", "SLURM_LOCALID")}
        with patch.dict(os.environ, env, clear=True):
            assert should_log(PlainObject()) is True

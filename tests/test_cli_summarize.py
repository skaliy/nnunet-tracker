"""Tests for nnunet-tracker summarize CLI command."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from nnunet_tracker import __version__
from nnunet_tracker.cli import main
from nnunet_tracker.summarize import CVSummary, FoldResult


class TestSummarizeRegistration:
    """Tests for summarize subcommand registration."""

    def test_help_lists_summarize(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "summarize" in captured.out

    def test_summarize_help(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["summarize", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--experiment" in captured.out
        assert "--cv-group" in captured.out
        assert "--json" in captured.out

    def test_version_still_works(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert __version__ in captured.out


class TestSummarizeCommand:
    """Tests for summarize command execution."""

    def _make_summary(self, complete: bool = True) -> CVSummary:
        """Create a CVSummary for testing."""
        fold_count = 5 if complete else 3
        fold_results = {
            i: FoldResult(
                fold=i,
                run_id=f"run-fold-{i}",
                status="FINISHED",
                mean_fg_dice=0.82 + i * 0.01,
                val_loss=0.35 - i * 0.01,
                ema_fg_dice=0.81 + i * 0.01,
                dice_per_class={0: 0.78 + i * 0.01, 1: 0.88 + i * 0.01},
            )
            for i in range(fold_count)
        }
        return CVSummary(
            cv_group="Dataset001|3d_fullres|nnUNetPlans",
            experiment_name="test_experiment",
            fold_results=fold_results,
        )

    def test_experiment_required(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["summarize"])
        assert exc_info.value.code == 2  # argparse exits with 2 for missing required

    def test_five_fold_complete_table(self, capsys) -> None:
        summary = self._make_summary(complete=True)
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(["summarize", "--experiment", "test_experiment"])
        captured = capsys.readouterr()
        assert "Cross-Validation Summary" in captured.out
        assert "5/5 folds" in captured.out
        assert "Mean FG Dice" in captured.out

    def test_partial_folds_exit_code_2(self) -> None:
        summary = self._make_summary(complete=False)
        with (
            patch(
                "nnunet_tracker.summarize.summarize_experiment",
                return_value=summary,
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["summarize", "--experiment", "test_experiment"])
        assert exc_info.value.code == 2

    def test_partial_folds_shows_missing(self, capsys) -> None:
        summary = self._make_summary(complete=False)
        with (
            patch(
                "nnunet_tracker.summarize.summarize_experiment",
                return_value=summary,
            ),
            pytest.raises(SystemExit),
        ):
            main(["summarize", "--experiment", "test_experiment"])
        captured = capsys.readouterr()
        assert "3/5 folds" in captured.out
        assert "[3, 4]" in captured.out

    def test_experiment_not_found_exits_1(self, capsys) -> None:
        with (
            patch(
                "nnunet_tracker.summarize.summarize_experiment",
                side_effect=ValueError("Experiment not found: 'bad'"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["summarize", "--experiment", "bad"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Experiment not found" in captured.err

    def test_json_output(self, capsys) -> None:
        summary = self._make_summary(complete=True)
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(["summarize", "--experiment", "test_experiment", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["is_complete"] is True
        assert len(data["completed_folds"]) == 5
        assert "aggregates" in data
        assert "cv_mean_fg_dice" in data["aggregates"]

    def test_cv_group_passed_through(self) -> None:
        summary = self._make_summary(complete=True)
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            main(
                [
                    "summarize",
                    "--experiment",
                    "test_experiment",
                    "--cv-group",
                    "my_group",
                ]
            )
        _, kwargs = mock_summarize.call_args
        assert kwargs["cv_group"] == "my_group"

    def test_custom_folds(self) -> None:
        summary = CVSummary(
            cv_group="test",
            experiment_name="test",
            fold_results={
                0: FoldResult(fold=0, run_id="r0", status="FINISHED", mean_fg_dice=0.85),
            },
            expected_folds=(0, 1, 2),
        )
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            with pytest.raises(SystemExit) as exc_info:
                main(
                    [
                        "summarize",
                        "--experiment",
                        "test",
                        "--folds",
                        "0,1,2",
                    ]
                )
        _, kwargs = mock_summarize.call_args
        assert kwargs["expected_folds"] == (0, 1, 2)
        assert exc_info.value.code == 2  # partial (1/3)

    def test_invalid_folds_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["summarize", "--experiment", "test", "--folds", "a,b,c"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "comma-separated integers" in captured.err

    def test_tracking_uri_override(self) -> None:
        summary = self._make_summary(complete=True)
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            main(
                [
                    "summarize",
                    "--experiment",
                    "test",
                    "--tracking-uri",
                    "/custom/path",
                ]
            )
        _, kwargs = mock_summarize.call_args
        assert kwargs["tracking_uri"] == "/custom/path"

    def test_log_to_mlflow_flag(self) -> None:
        summary = self._make_summary(complete=True)
        with (
            patch(
                "nnunet_tracker.summarize.summarize_experiment",
                return_value=summary,
            ),
            patch(
                "nnunet_tracker.summarize.log_cv_summary",
                return_value="summary-run-id",
            ) as mock_log,
        ):
            main(
                [
                    "summarize",
                    "--experiment",
                    "test",
                    "--log-to-mlflow",
                ]
            )
        mock_log.assert_called_once()

    def test_invalid_cv_group_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "summarize",
                    "--experiment",
                    "test",
                    "--cv-group",
                    "test' or '1'='1",
                ]
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid characters" in captured.err

"""Tests for nnunet-tracker export CLI command."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from nnunet_tracker.cli import main
from nnunet_tracker.summarize import CVSummary, FoldResult


def _make_summary(complete: bool = True) -> CVSummary:
    """Create a CVSummary for testing."""
    fold_count = 5 if complete else 3
    fold_results = {
        i: FoldResult(
            fold=i,
            run_id=f"run-fold-{i}",
            status="FINISHED",
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


class TestExportRegistration:
    """Tests for export subcommand registration."""

    def test_help_lists_export_command(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "export" in captured.out

    def test_export_help(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--experiment" in captured.out
        assert "--format" in captured.out
        assert "--output" in captured.out
        assert "--caption" in captured.out
        assert "--label" in captured.out


class TestExportCommand:
    """Tests for export command execution."""

    def test_experiment_required(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--format", "csv"])
        assert exc_info.value.code == 2  # argparse missing required

    def test_format_required(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--experiment", "test"])
        assert exc_info.value.code == 2  # argparse missing required

    def test_invalid_format_rejected(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["export", "--experiment", "test", "--format", "html"])
        assert exc_info.value.code == 2  # argparse invalid choice

    def test_csv_stdout(self, capsys) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(["export", "--experiment", "test", "--format", "csv"])
        captured = capsys.readouterr()
        assert "Fold" in captured.out
        assert "Val Loss" in captured.out
        assert "0.3500" in captured.out

    def test_latex_stdout(self, capsys) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(["export", "--experiment", "test", "--format", "latex"])
        captured = capsys.readouterr()
        assert r"\toprule" in captured.out
        assert r"\bottomrule" in captured.out

    def test_output_file(self, tmp_path, capsys) -> None:
        summary = _make_summary()
        output_path = str(tmp_path / "results.csv")
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--output",
                    output_path,
                ]
            )
        captured = capsys.readouterr()
        assert "Exported to" in captured.err
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "Fold" in content

    def test_experiment_not_found_exits_1(self, capsys) -> None:
        with (
            patch(
                "nnunet_tracker.summarize.summarize_experiment",
                side_effect=ValueError("Experiment not found: 'bad'"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["export", "--experiment", "bad", "--format", "csv"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Experiment not found" in captured.err

    def test_tracking_uri_override(self) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--tracking-uri",
                    "/custom/path",
                ]
            )
        _, kwargs = mock_summarize.call_args
        assert kwargs["tracking_uri"] == "/custom/path"

    def test_cv_group_passed_through(self) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--cv-group",
                    "my_group",
                ]
            )
        _, kwargs = mock_summarize.call_args
        assert kwargs["cv_group"] == "my_group"

    def test_custom_folds(self) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ) as mock_summarize:
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--folds",
                    "0,1,2",
                ]
            )
        _, kwargs = mock_summarize.call_args
        assert kwargs["expected_folds"] == (0, 1, 2)

    def test_invalid_folds_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--folds",
                    "a,b,c",
                ]
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "comma-separated integers" in captured.err

    def test_invalid_cv_group_exits_1(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "csv",
                    "--cv-group",
                    "test' or '1'='1",
                ]
            )
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "invalid characters" in captured.err

    def test_caption_and_label_for_latex(self, capsys) -> None:
        summary = _make_summary()
        with patch(
            "nnunet_tracker.summarize.summarize_experiment",
            return_value=summary,
        ):
            main(
                [
                    "export",
                    "--experiment",
                    "test",
                    "--format",
                    "latex",
                    "--caption",
                    "My Results",
                    "--label",
                    "tab:my_results",
                ]
            )
        captured = capsys.readouterr()
        assert "My Results" in captured.out
        assert r"\label{tab:my_results}" in captured.out

    def test_output_write_oserror_exits_1(self, tmp_path, capsys) -> None:
        """OSError during file write produces clean error and exits 1."""
        # Use a non-writable directory to trigger OSError
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o555)
        output_path = str(readonly_dir / "subdir" / "results.csv")
        try:
            with pytest.raises(SystemExit) as exc_info:
                # Validate output dir existence check
                main(
                    [
                        "export",
                        "--experiment",
                        "test",
                        "--format",
                        "csv",
                        "--output",
                        output_path,
                    ]
                )
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "output directory does not exist" in captured.err
        finally:
            readonly_dir.chmod(0o755)

    def test_output_write_permission_error_exits_1(self, tmp_path, capsys) -> None:
        """Permission error during atomic write produces clean error and exits 1."""
        summary = _make_summary()
        # Create a non-writable directory where the output file should go
        output_dir = tmp_path / "noperm"
        output_dir.mkdir()
        output_path = str(output_dir / "results.csv")
        output_dir.chmod(0o555)
        try:
            with (
                patch(
                    "nnunet_tracker.summarize.summarize_experiment",
                    return_value=summary,
                ),
                pytest.raises(SystemExit) as exc_info,
            ):
                main(
                    [
                        "export",
                        "--experiment",
                        "test",
                        "--format",
                        "csv",
                        "--output",
                        output_path,
                    ]
                )
            assert exc_info.value.code == 1
            captured = capsys.readouterr()
            assert "Failed to write output file" in captured.err
        finally:
            output_dir.chmod(0o755)

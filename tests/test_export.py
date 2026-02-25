"""Tests for nnunet_tracker.export module."""

from __future__ import annotations

import csv
import io

from nnunet_tracker.export import (
    _build_rows,
    _escape_latex,
    export_csv,
    export_latex,
)
from nnunet_tracker.summarize import CVSummary, FoldResult


def _make_summary(fold_count: int = 5, with_classes: bool = True) -> CVSummary:
    """Create a CVSummary for testing."""
    fold_results = {}
    for i in range(fold_count):
        dice_per_class = {0: 0.78 + i * 0.01, 1: 0.88 + i * 0.01} if with_classes else {}
        fold_results[i] = FoldResult(
            fold=i,
            run_id=f"run-fold-{i}",
            status="FINISHED",
            val_loss=0.35 - i * 0.01,
            ema_fg_dice=0.81 + i * 0.01,
            dice_per_class=dice_per_class,
        )
    return CVSummary(
        cv_group="Dataset001|3d_fullres|nnUNetPlans",
        experiment_name="test_experiment",
        fold_results=fold_results,
    )


class TestBuildRows:
    """Tests for _build_rows helper."""

    def test_empty_summary(self) -> None:
        summary = CVSummary(cv_group="test", experiment_name="test")
        headers, rows = _build_rows(summary)
        assert headers == ["Fold", "Val Loss", "EMA FG Dice"]
        assert rows == []

    def test_single_fold(self) -> None:
        summary = _make_summary(fold_count=1)
        headers, rows = _build_rows(summary)
        assert len(rows) == 1
        assert rows[0][0] == "0"  # Fold number
        assert rows[0][1] == "0.3500"  # Val Loss

    def test_five_folds(self) -> None:
        summary = _make_summary(fold_count=5)
        headers, rows = _build_rows(summary)
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row[0] == str(i)

    def test_missing_metrics(self) -> None:
        summary = CVSummary(
            cv_group="test",
            experiment_name="test",
            fold_results={
                0: FoldResult(fold=0, run_id="r0", status="FINISHED"),
            },
        )
        headers, rows = _build_rows(summary)
        assert rows[0][1] == ""  # val_loss is None
        assert rows[0][2] == ""  # ema_fg_dice is None

    def test_per_class_auto_detection(self) -> None:
        summary = _make_summary(fold_count=2, with_classes=True)
        headers, rows = _build_rows(summary)
        assert "Dice Class 0" in headers
        assert "Dice Class 1" in headers

    def test_no_per_class_columns(self) -> None:
        summary = _make_summary(fold_count=2, with_classes=False)
        headers, rows = _build_rows(summary)
        assert len(headers) == 3  # No per-class columns

    def test_varying_class_indices(self) -> None:
        """Union of class indices across folds."""
        summary = CVSummary(
            cv_group="test",
            experiment_name="test",
            fold_results={
                0: FoldResult(
                    fold=0,
                    run_id="r0",
                    status="FINISHED",
                    dice_per_class={0: 0.80, 1: 0.90},
                ),
                1: FoldResult(
                    fold=1,
                    run_id="r1",
                    status="FINISHED",
                    dice_per_class={0: 0.82, 2: 0.75},
                ),
            },
        )
        headers, rows = _build_rows(summary)
        assert "Dice Class 0" in headers
        assert "Dice Class 1" in headers
        assert "Dice Class 2" in headers
        # Fold 0 has no class 2 -> empty string
        cls2_idx = headers.index("Dice Class 2")
        assert rows[0][cls2_idx] == ""
        # Fold 1 has no class 1 -> empty string
        cls1_idx = headers.index("Dice Class 1")
        assert rows[1][cls1_idx] == ""


class TestExportCSV:
    """Tests for export_csv function."""

    def test_csv_parseable(self) -> None:
        summary = _make_summary(fold_count=3)
        result = export_csv(summary)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) >= 4  # header + 3 folds + mean (+ maybe std)

    def test_csv_header_row(self) -> None:
        summary = _make_summary(fold_count=1)
        result = export_csv(summary)
        reader = csv.reader(io.StringIO(result))
        header = next(reader)
        assert header[0] == "Fold"
        assert "Val Loss" in header

    def test_csv_data_rows(self) -> None:
        summary = _make_summary(fold_count=3)
        result = export_csv(summary)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        # First data row (after header) is fold 0
        assert rows[1][0] == "0"

    def test_csv_mean_std_rows(self) -> None:
        summary = _make_summary(fold_count=5)
        result = export_csv(summary)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        # Last two rows should be Mean and Std
        assert rows[-2][0] == "Mean"
        assert rows[-1][0] == "Std"

    def test_csv_single_fold_no_std(self) -> None:
        summary = _make_summary(fold_count=1)
        result = export_csv(summary)
        reader = csv.reader(io.StringIO(result))
        rows = list(reader)
        # header + 1 fold + mean only (no std for single fold)
        row_labels = [r[0] for r in rows]
        assert "Mean" in row_labels
        assert "Std" not in row_labels

    def test_csv_writes_to_output(self) -> None:
        summary = _make_summary(fold_count=2)
        buf = io.StringIO()
        result = export_csv(summary, output=buf)
        assert buf.getvalue() == result
        assert len(result) > 0

    def test_csv_returns_string(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_csv(summary)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_csv_formatting_precision(self) -> None:
        summary = _make_summary(fold_count=1)
        result = export_csv(summary)
        assert "0.3500" in result  # 4 decimal places


class TestExportLatex:
    """Tests for export_latex function."""

    def test_latex_booktabs_structure(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary)
        assert r"\toprule" in result
        assert r"\midrule" in result
        assert r"\bottomrule" in result

    def test_latex_tabular_columns(self) -> None:
        summary = _make_summary(fold_count=2, with_classes=True)
        result = export_latex(summary)
        # 5 columns: Fold + Val Loss + EMA FG Dice + 2 classes
        assert r"\begin{tabular}{lcccc}" in result

    def test_latex_mean_pm_std_row(self) -> None:
        summary = _make_summary(fold_count=5)
        result = export_latex(summary)
        assert r"Mean $\pm$ Std" in result
        assert r"$\pm$" in result

    def test_latex_default_caption(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary)
        assert "Cross-validation results for" in result

    def test_latex_custom_caption(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary, caption="My Custom Table")
        assert "My Custom Table" in result

    def test_latex_custom_label(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary, label="tab:my_table")
        assert r"\label{tab:my_table}" in result

    def test_latex_default_label(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary)
        assert r"\label{tab:cv_results}" in result

    def test_latex_writes_to_output(self) -> None:
        summary = _make_summary(fold_count=2)
        buf = io.StringIO()
        result = export_latex(summary, output=buf)
        assert buf.getvalue() == result

    def test_latex_returns_string(self) -> None:
        summary = _make_summary(fold_count=2)
        result = export_latex(summary)
        assert isinstance(result, str)
        assert result.endswith("\n")

    def test_latex_special_chars_escaped(self) -> None:
        """Underscores in cv_group are escaped for LaTeX."""
        summary = CVSummary(
            cv_group="Dataset_001",
            experiment_name="test",
            fold_results={
                0: FoldResult(
                    fold=0,
                    run_id="r0",
                    status="FINISHED",
                ),
            },
        )
        result = export_latex(summary)
        assert r"Dataset\_001" in result

    def test_latex_single_fold_no_std(self) -> None:
        summary = _make_summary(fold_count=1)
        result = export_latex(summary)
        # Mean value present but no ± since only 1 fold
        assert "0.3500" in result
        # The aggregate row label still says Mean ± Std
        assert r"Mean $\pm$ Std" in result

    def test_latex_missing_metrics_show_dashes(self) -> None:
        summary = CVSummary(
            cv_group="test",
            experiment_name="test",
            fold_results={
                0: FoldResult(fold=0, run_id="r0", status="FINISHED"),
            },
        )
        result = export_latex(summary)
        assert "--" in result


class TestEscapeLatex:
    """Tests for _escape_latex helper."""

    def test_escapes_underscore(self) -> None:
        assert _escape_latex("hello_world") == r"hello\_world"

    def test_escapes_ampersand(self) -> None:
        assert _escape_latex("a & b") == r"a \& b"

    def test_escapes_percent(self) -> None:
        assert _escape_latex("100%") == r"100\%"

    def test_preserves_math_mode(self) -> None:
        assert _escape_latex("$x_1$") == "$x_1$"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_latex("hello world") == "hello world"

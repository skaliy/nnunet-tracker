"""Publication-ready export of cross-validation results (CSV, LaTeX)."""

from __future__ import annotations

import csv
import io
import logging
import re
from typing import TextIO

from nnunet_tracker.summarize import CVSummary

logger = logging.getLogger("nnunet_tracker")

__all__ = ["export_csv", "export_latex"]

_DICE_FMT = ".4f"
_LOSS_FMT = ".4f"

# Valid characters for LaTeX \label{} values
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9:_./-]+$")

# LaTeX special characters that must be escaped in text mode.
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def export_csv(
    summary: CVSummary,
    output: TextIO | None = None,
) -> str:
    """Export CV results as CSV.

    Args:
        summary: CVSummary to export.
        output: Optional file-like object to write to.
            If None, returns the CSV as a string only.

    Returns:
        CSV string (always returned, also written to output if provided).
    """
    headers, rows = _build_rows(summary)
    aggregates = summary.compute_aggregate_metrics()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)

    for row in rows:
        writer.writerow(row)

    if aggregates:
        mean_row, std_row = _build_aggregate_rows(headers, aggregates)
        writer.writerow(mean_row)
        if std_row is not None:
            writer.writerow(std_row)

    result = buf.getvalue()
    if output is not None:
        output.write(result)
    return result


def export_latex(
    summary: CVSummary,
    output: TextIO | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Export CV results as a LaTeX booktabs table.

    Args:
        summary: CVSummary to export.
        output: Optional file-like object to write to.
        caption: Optional table caption. Defaults to cv_group value.
        label: Optional table label for cross-referencing. Defaults to 'tab:cv_results'.

    Returns:
        LaTeX string (always returned, also written to output if provided).
    """
    headers, rows = _build_rows(summary)
    aggregates = summary.compute_aggregate_metrics()

    if caption is None:
        caption = f"Cross-validation results for {summary.cv_group}"
    if label is None:
        label = "tab:cv_results"

    # Column spec: first col left-aligned, rest centered
    col_spec = "l" + "c" * (len(headers) - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{_escape_latex(caption)}}}",
        f"\\label{{{_sanitize_label(label)}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    escaped_headers = [_escape_latex(h) for h in headers]
    lines.append(" & ".join(escaped_headers) + r" \\")
    lines.append(r"\midrule")

    for row in rows:
        escaped_row = [_escape_latex(cell) if cell else "--" for cell in row]
        lines.append(" & ".join(escaped_row) + r" \\")

    # Aggregate row (mean ± std combined)
    if aggregates:
        lines.append(r"\midrule")
        agg_row = _build_latex_aggregate_row(headers, aggregates)
        lines.append(" & ".join(agg_row) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    result = "\n".join(lines) + "\n"
    if output is not None:
        output.write(result)
    return result


def _build_rows(summary: CVSummary) -> tuple[list[str], list[list[str]]]:
    """Build header and data rows from a CVSummary.

    Returns:
        (headers, rows) tuple where headers is a list of column names
        and rows is a list of lists of formatted string values.
    """
    sorted_classes = summary.all_class_indices

    headers = ["Fold", "Val Loss", "EMA FG Dice"]
    for cls_idx in sorted_classes:
        headers.append(f"Dice Class {cls_idx}")

    rows: list[list[str]] = []
    for fold_num in sorted(summary.fold_results.keys()):
        r = summary.fold_results[fold_num]
        row = [
            str(fold_num),
            f"{r.val_loss:{_LOSS_FMT}}" if r.val_loss is not None else "",
            f"{r.ema_fg_dice:{_DICE_FMT}}" if r.ema_fg_dice is not None else "",
        ]
        for cls_idx in sorted_classes:
            val = r.dice_per_class.get(cls_idx)
            row.append(f"{val:{_DICE_FMT}}" if val is not None else "")
        rows.append(row)

    return headers, rows


def _build_aggregate_rows(
    headers: list[str],
    aggregates: dict[str, float],
) -> tuple[list[str], list[str] | None]:
    """Build mean and std rows for CSV export.

    Returns:
        (mean_row, std_row) where std_row is None if no std values exist.
    """
    mean_row = ["Mean"]
    std_row = ["Std"]
    has_std = False

    for header in headers[1:]:  # Skip "Fold" column
        keys = _header_to_metric_keys(header)
        if keys is None:
            mean_row.append("")
            std_row.append("")
            continue

        mean_key, std_key = keys
        mean_val = aggregates.get(mean_key)
        std_val = aggregates.get(std_key)
        fmt = _LOSS_FMT if header == "Val Loss" else _DICE_FMT

        mean_row.append(f"{mean_val:{fmt}}" if mean_val is not None else "")
        if std_val is not None:
            std_row.append(f"{std_val:{fmt}}")
            has_std = True
        else:
            std_row.append("")

    return mean_row, std_row if has_std else None


def _build_latex_aggregate_row(
    headers: list[str],
    aggregates: dict[str, float],
) -> list[str]:
    """Build combined mean ± std row for LaTeX export."""
    agg_row = [r"Mean $\pm$ Std"]

    for header in headers[1:]:  # Skip "Fold" column
        keys = _header_to_metric_keys(header)
        if keys is None:
            agg_row.append("--")
            continue

        mean_key, std_key = keys
        mean_val = aggregates.get(mean_key)
        std_val = aggregates.get(std_key)
        fmt = _LOSS_FMT if header == "Val Loss" else _DICE_FMT

        if mean_val is not None:
            mean_str = f"{mean_val:{fmt}}"
            if std_val is not None:
                agg_row.append(f"{mean_str} $\\pm$ {std_val:{fmt}}")
            else:
                agg_row.append(mean_str)
        else:
            agg_row.append("--")

    return agg_row


def _header_to_metric_keys(header: str) -> tuple[str, str] | None:
    """Map a column header to its aggregate metric keys.

    Returns:
        (mean_key, std_key) tuple for looking up in aggregates dict,
        or None if the header doesn't map to a known metric.
    """
    if header == "Val Loss":
        return "cv_mean_val_loss", "cv_std_val_loss"
    elif header == "EMA FG Dice":
        return "cv_mean_ema_fg_dice", "cv_std_ema_fg_dice"
    elif header.startswith("Dice Class "):
        cls_idx = header[len("Dice Class ") :]
        return f"cv_mean_dice_class_{cls_idx}", f"cv_std_dice_class_{cls_idx}"
    logger.debug("Unknown header %r, skipping aggregate mapping", header)
    return None


def _sanitize_label(label: str) -> str:
    """Sanitize a LaTeX label value.

    LaTeX labels accept alphanumeric, colon, underscore, hyphen, dot, and slash.
    Characters outside this set are stripped with a warning.
    """
    if _SAFE_LABEL.fullmatch(label):
        return label
    sanitized = re.sub(r"[^A-Za-z0-9:_./-]", "", label)
    logger.warning("LaTeX label %r contains invalid characters; sanitized to %r", label, sanitized)
    return sanitized or "tab:unnamed"


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in text.

    Does not escape characters inside math mode ($...$).

    Note:
        Escaped dollar signs (``\\$``) in input are not supported and will
        cause incorrect math-mode state tracking. This is acceptable because
        nnU-Net metric names and dataset identifiers never contain dollar signs.
    """
    result = []
    in_math = False
    for char in text:
        if char == "$":
            in_math = not in_math
            result.append(char)
        elif in_math:
            result.append(char)
        elif char in _LATEX_SPECIAL:
            result.append(_LATEX_SPECIAL[char])
        else:
            result.append(char)
    if in_math:
        logger.warning("Unclosed math mode ($) in LaTeX text: %r", text)
    return "".join(result)

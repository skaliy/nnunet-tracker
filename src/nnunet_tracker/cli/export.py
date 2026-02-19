"""nnunet-tracker export command -- publication-ready table generation."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from nnunet_tracker.cli._validation import parse_folds, validate_cv_group

logger = logging.getLogger("nnunet_tracker")

__all__ = ["register_export_command"]

_VALID_FORMATS = ("csv", "latex")


def register_export_command(subparsers: Any) -> None:
    """Register the 'export' subcommand."""
    export_parser = subparsers.add_parser(
        "export",
        help="Export cross-validation results as CSV or LaTeX tables",
        description=(
            "Generate publication-ready comparison tables from completed "
            "cross-validation experiments."
        ),
    )
    export_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="MLflow experiment name to export",
    )
    export_parser.add_argument(
        "--format",
        "-f",
        required=True,
        choices=_VALID_FORMATS,
        help="Output format: csv or latex",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    export_parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns)",
    )
    export_parser.add_argument(
        "--cv-group",
        default=None,
        help="CV group identifier (default: auto-detect latest)",
    )
    export_parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Expected fold numbers, comma-separated (default: 0,1,2,3,4)",
    )
    export_parser.add_argument(
        "--caption",
        default=None,
        help="Table caption (LaTeX only)",
    )
    export_parser.add_argument(
        "--label",
        default=None,
        help="Table label for cross-referencing (LaTeX only)",
    )
    export_parser.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace, remaining: list[str]) -> None:
    """Execute the export command."""
    if remaining:
        print(f"Error: unrecognized arguments: {remaining}", file=sys.stderr)
        sys.exit(1)

    from nnunet_tracker.config import TrackerConfig
    from nnunet_tracker.export import export_csv, export_latex
    from nnunet_tracker.summarize import summarize_experiment

    config = TrackerConfig.from_env()
    tracking_uri = args.tracking_uri or config.tracking_uri

    expected_folds = parse_folds(args.folds)
    validate_cv_group(args.cv_group)

    # Validate output path early (before expensive MLflow query)
    if args.output:
        import os

        output_dir = os.path.dirname(args.output) or "."
        if not os.path.isdir(output_dir):
            print(
                f"Error: output directory does not exist: {output_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

    try:
        summary = summarize_experiment(
            experiment_name=args.experiment,
            tracking_uri=tracking_uri,
            cv_group=args.cv_group,
            expected_folds=expected_folds,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        print(
            "Error: failed to query experiment. Enable debug logging for details.",
            file=sys.stderr,
        )
        logger.debug("Export query error", exc_info=True)
        sys.exit(1)

    if args.format == "csv":
        result = export_csv(summary)
    else:  # latex
        result = export_latex(
            summary,
            caption=args.caption,
            label=args.label,
        )

    if args.output:
        import os
        import tempfile

        abs_output = os.path.abspath(args.output)
        output_dir = os.path.dirname(abs_output)
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(result)
            os.replace(tmp_path, abs_output)
            tmp_path = None  # Replaced successfully, no cleanup needed
        except Exception:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
        print(f"Exported to {args.output}", file=sys.stderr)
    else:
        print(result, end="")

"""nnunet-tracker summarize command -- aggregate cross-validation metrics."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from nnunet_tracker.cli._validation import parse_folds, validate_cv_group

logger = logging.getLogger("nnunet_tracker")

__all__ = ["register_summarize_command"]


def register_summarize_command(subparsers: Any) -> None:
    """Register the 'summarize' subcommand."""
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize cross-validation results across folds",
        description=(
            "Query MLflow for completed fold runs and compute aggregate "
            "Dice scores (mean +/- std) across folds."
        ),
    )
    summarize_parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="MLflow experiment name to query",
    )
    summarize_parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns)",
    )
    summarize_parser.add_argument(
        "--cv-group",
        default=None,
        help="CV group identifier (default: auto-detect latest)",
    )
    summarize_parser.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="Expected fold numbers, comma-separated (default: 0,1,2,3,4)",
    )
    summarize_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of table",
    )
    summarize_parser.add_argument(
        "--log-to-mlflow",
        action="store_true",
        default=False,
        help="Create a summary run in MLflow with aggregate metrics",
    )
    summarize_parser.set_defaults(func=_run_summarize)


def _run_summarize(args: argparse.Namespace, remaining: list[str]) -> None:
    """Execute the summarize command."""
    if remaining:
        print(f"Error: unrecognized arguments: {remaining}", file=sys.stderr)
        sys.exit(1)

    from nnunet_tracker.config import TrackerConfig
    from nnunet_tracker.summarize import log_cv_summary, summarize_experiment

    config = TrackerConfig.from_env()
    tracking_uri = args.tracking_uri or config.tracking_uri

    expected_folds = parse_folds(args.folds)
    validate_cv_group(args.cv_group)

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
        logger.debug("Summarize query error", exc_info=True)
        sys.exit(1)

    aggregates = summary.compute_aggregate_metrics()

    if args.json:
        _print_json(summary, aggregates)
    else:
        _print_table(summary, aggregates)

    if args.log_to_mlflow:
        run_id = log_cv_summary(summary, tracking_uri=tracking_uri)
        if run_id:
            print(f"\nSummary logged to MLflow run: {run_id}", file=sys.stderr)

    # Exit code: 0 if complete, 2 if partial
    if not summary.is_complete:
        sys.exit(2)


def _print_table(summary: Any, aggregates: dict[str, float]) -> None:
    """Print human-readable summary table."""
    print("Cross-Validation Summary")
    print(f"  Experiment:  {summary.experiment_name}")
    print(f"  CV Group:    {summary.cv_group}")
    total = len(summary.expected_folds)
    completed = len(summary.completed_folds)
    print(f"  Completed:   {completed}/{total} folds")

    if summary.missing_folds:
        print(f"  Missing:     {summary.missing_folds}")

    print()
    print("Fold Results:")
    print(f"  {'Fold':<6}{'Status':<12}{'Val Loss':<12}{'Run ID'}")
    print(f"  {'----':<6}{'------':<12}{'--------':<12}{'------'}")

    for fold_num in sorted(summary.fold_results.keys()):
        r = summary.fold_results[fold_num]
        loss_str = f"{r.val_loss:.4f}" if r.val_loss is not None else "N/A"
        run_id_short = r.run_id[:8] if r.run_id else "N/A"
        print(f"  {fold_num:<6}{r.status:<12}{loss_str:<12}{run_id_short}")

    if aggregates:
        print()
        print("Aggregate Metrics:")
        mean_loss = aggregates.get("cv_mean_val_loss")
        std_loss = aggregates.get("cv_std_val_loss")
        if mean_loss is not None:
            std_str = f" +/- {std_loss:.4f}" if std_loss is not None else ""
            print(f"  Val Loss:      {mean_loss:.4f}{std_str}")

        cls_indices = sorted(
            {int(k.split("_")[-1]) for k in aggregates if k.startswith("cv_mean_dice_class_")}
        )
        if cls_indices:
            print("  Per-class Dice:")
            for cls_idx in cls_indices:
                mean_cls = aggregates.get(f"cv_mean_dice_class_{cls_idx}")
                std_cls = aggregates.get(f"cv_std_dice_class_{cls_idx}")
                if mean_cls is not None:
                    std_str = f" +/- {std_cls:.4f}" if std_cls is not None else ""
                    print(f"    Class {cls_idx}: {mean_cls:.4f}{std_str}")


def _print_json(summary: Any, aggregates: dict[str, float]) -> None:
    """Print JSON summary to stdout."""
    output = {
        "cv_group": summary.cv_group,
        "experiment_name": summary.experiment_name,
        "is_complete": summary.is_complete,
        "completed_folds": summary.completed_folds,
        "missing_folds": summary.missing_folds,
        "folds": {
            fold_num: {
                "fold": r.fold,
                "run_id": r.run_id,
                "status": r.status,
                "val_loss": r.val_loss,
                "ema_fg_dice": r.ema_fg_dice,
                "dice_per_class": {str(k): v for k, v in r.dice_per_class.items()},
            }
            for fold_num, r in sorted(summary.fold_results.items())
        },
        "aggregates": aggregates,
    }
    print(json.dumps(output, indent=2))

"""nnunet-tracker list command -- display tracked experiments."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("nnunet_tracker")

__all__ = ["register_list_command"]


def register_list_command(subparsers: Any) -> None:
    """Register the 'list' subcommand."""
    list_parser = subparsers.add_parser(
        "list",
        help="List nnunet-tracker experiments and their fold status",
        description=(
            "Query MLflow for experiments containing nnunet-tracker fold runs "
            "and display their completion status."
        ),
    )
    list_parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI or ./mlruns)",
    )
    list_parser.add_argument(
        "--max-experiments",
        type=int,
        default=50,
        help="Maximum number of experiments to query (default: 50, range: 1-1000)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON instead of table",
    )
    list_parser.set_defaults(func=_run_list)


def _run_list(args: argparse.Namespace, remaining: list[str]) -> None:
    """Execute the list command."""
    if remaining:
        print(f"Error: unrecognized arguments: {remaining}", file=sys.stderr)
        sys.exit(1)

    if args.max_experiments < 1 or args.max_experiments > 1000:
        print(
            "Error: --max-experiments must be between 1 and 1000.",
            file=sys.stderr,
        )
        sys.exit(1)

    from nnunet_tracker.config import TrackerConfig

    config = TrackerConfig.from_env()
    tracking_uri = args.tracking_uri or config.tracking_uri

    try:
        experiments = _query_experiments(tracking_uri, max_experiments=args.max_experiments)
    except Exception:
        print("Error: could not query experiments.", file=sys.stderr)
        logger.debug("Query error details", exc_info=True)
        sys.exit(1)

    if not experiments:
        print("No nnunet-tracker experiments found.", file=sys.stderr)
        sys.exit(0)

    if args.json:
        _print_json(experiments)
    else:
        _print_table(experiments)


def _query_experiments(tracking_uri: str, max_experiments: int = 50) -> list[dict[str, Any]]:
    """Query MLflow for experiments with nnunet-tracker fold runs.

    Args:
        tracking_uri: MLflow tracking server URI.
        max_experiments: Maximum number of experiments to query.

    Returns a list of dicts, each containing:
        - name: experiment name
        - experiment_id: MLflow experiment ID
        - cv_groups: list of dicts with cv_group, completed_folds, latest_start_time
        - total_runs: total fold runs in the experiment
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    experiments = client.search_experiments(max_results=max_experiments)

    result = []
    for exp in experiments:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`nnunet_tracker.run_type` = 'fold'",
            order_by=["start_time DESC"],
            max_results=1000,
        )

        if not runs:
            continue

        cv_groups: dict[str, dict[str, Any]] = {}
        for run in runs:
            cv_group = run.data.tags.get("nnunet_tracker.cv_group", "unknown")
            if cv_group not in cv_groups:
                cv_groups[cv_group] = {
                    "cv_group": cv_group,
                    "folds": set(),
                    "latest_start_time": run.info.start_time,
                }
            fold_str = run.data.tags.get("nnunet_tracker.fold")
            if fold_str is not None:
                try:
                    cv_groups[cv_group]["folds"].add(int(fold_str))
                except (ValueError, TypeError):
                    pass

        # Convert sets to sorted lists for serialization
        cv_group_list = []
        for group_info in cv_groups.values():
            cv_group_list.append(
                {
                    "cv_group": group_info["cv_group"],
                    "completed_folds": sorted(group_info["folds"]),
                    "num_completed": len(group_info["folds"]),
                    "latest_start_time": _format_timestamp(group_info["latest_start_time"]),
                }
            )

        result.append(
            {
                "name": exp.name,
                "experiment_id": exp.experiment_id,
                "cv_groups": cv_group_list,
                "total_runs": len(runs),
            }
        )

    return result


def _format_timestamp(epoch_ms: int | None) -> str | None:
    """Convert epoch milliseconds to ISO 8601 string."""
    if epoch_ms is None:
        return None
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


def _print_table(experiments: list[dict[str, Any]]) -> None:
    """Print aligned table to stdout."""
    # Compute column widths from data
    exp_w = len("Experiment")
    grp_w = len("CV Group")
    for exp in experiments:
        exp_w = max(exp_w, len(exp["name"]))
        for group in exp["cv_groups"]:
            grp_w = max(grp_w, len(group["cv_group"]))
    exp_w = min(exp_w + 2, 60)  # pad + cap
    grp_w = min(grp_w + 2, 60)
    folds_w = 12

    print(f"{'Experiment':<{exp_w}}{'CV Group':<{grp_w}}{'Folds':<{folds_w}}{'Runs'}")
    print(f"{'----------':<{exp_w}}{'--------':<{grp_w}}{'-----':<{folds_w}}{'----'}")

    for exp in experiments:
        for i, group in enumerate(exp["cv_groups"]):
            exp_name = exp["name"] if i == 0 else ""
            folds_str = str(group["num_completed"])
            runs_str = str(exp["total_runs"]) if i == 0 else ""
            print(
                f"{exp_name:<{exp_w}}{group['cv_group']:<{grp_w}}{folds_str:<{folds_w}}{runs_str}"
            )


def _print_json(experiments: list[dict[str, Any]]) -> None:
    """Print JSON to stdout."""
    print(json.dumps(experiments, indent=2))

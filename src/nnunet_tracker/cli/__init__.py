"""nnunet-tracker CLI."""

from __future__ import annotations

import argparse
import sys

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point for nnunet-tracker."""
    parser = argparse.ArgumentParser(
        prog="nnunet-tracker",
        description="Lightweight MLflow-based experiment tracking for nnU-Net",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand registration (lightweight imports — only argparse setup, no MLflow/nnU-Net)
    from nnunet_tracker.cli.export import register_export_command
    from nnunet_tracker.cli.list import register_list_command
    from nnunet_tracker.cli.summarize import register_summarize_command
    from nnunet_tracker.cli.train import register_train_command

    register_train_command(subparsers)
    register_summarize_command(subparsers)
    register_list_command(subparsers)
    register_export_command(subparsers)

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args, remaining)


def _get_version() -> str:
    from nnunet_tracker import __version__

    return __version__

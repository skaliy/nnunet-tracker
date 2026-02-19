"""nnunet-tracker train command -- wraps nnUNetv2_train with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from typing import Any

logger = logging.getLogger("nnunet_tracker")

__all__ = ["register_train_command"]

# Both trainer names and plans identifiers follow Python identifier rules.
_VALID_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_VALID_TRAINER_NAME = _VALID_IDENTIFIER
_VALID_PLANS_NAME = _VALID_IDENTIFIER


def register_train_command(subparsers: Any) -> None:
    """Register the 'train' subcommand."""
    train_parser = subparsers.add_parser(
        "train",
        help="Run nnU-Net training with MLflow tracking",
        description=(
            "Wraps nnUNetv2_train with automatic MLflow experiment tracking. "
            "All standard nnUNetv2_train arguments are forwarded."
        ),
        epilog=(
            "Positional arguments (forwarded to nnUNetv2_train):\n"
            "  dataset_name_or_id  Dataset name or ID (e.g., Dataset001_BrainTumour or 1)\n"
            "  configuration       nnU-Net configuration (e.g., 2d, 3d_fullres, 3d_lowres)\n"
            "  fold                Fold number (0-4) or 'all' for 5-fold CV\n"
            "\n"
            "Optional forwarded arguments:\n"
            "  -tr TRAINER         Trainer class name (default: nnUNetTrainer)\n"
            "  -p PLANS            Plans identifier (default: nnUNetPlans)\n"
            "  -c                  Continue training from latest checkpoint"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument(
        "--tracker-disable",
        action="store_true",
        default=False,
        help="Disable MLflow tracking for this run",
    )
    train_parser.set_defaults(func=_run_train)


def _run_train(args: argparse.Namespace, remaining: list[str]) -> None:
    """Execute the wrapped training pipeline."""
    from nnunet_tracker._compat import check_nnunet_available
    from nnunet_tracker.config import TrackerConfig

    if not check_nnunet_available():
        print(
            "Error: nnU-Net v2 is not installed. Install it with: pip install nnunetv2",
            file=sys.stderr,
        )
        sys.exit(1)

    nnunet_parser = _build_nnunet_arg_parser()
    nnunet_args, extra = nnunet_parser.parse_known_args(remaining)

    if extra:
        msg = f"unrecognized arguments will be ignored: {extra}"
        print(f"Warning: {msg}", file=sys.stderr)
        logger.warning(msg)

    if not _VALID_TRAINER_NAME.fullmatch(nnunet_args.tr):
        print(
            "Error: invalid trainer class name. "
            "Must contain only alphanumeric characters and underscores.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _VALID_PLANS_NAME.fullmatch(nnunet_args.p):
        print(
            "Error: invalid plans identifier. "
            "Must contain only alphanumeric characters and underscores.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build config explicitly rather than mutating os.environ
    config = TrackerConfig.from_env()
    if args.tracker_disable:
        config = TrackerConfig(
            tracking_uri=config.tracking_uri,
            experiment_name=config.experiment_name,
            enabled=False,
            log_artifacts=config.log_artifacts,
        )

    trainer_class = _resolve_trainer_class(nnunet_args.tr)
    if trainer_class is None:
        print("Error: Trainer class not found.", file=sys.stderr)
        sys.exit(1)

    from nnunet_tracker.factory import create_tracked_trainer

    tracked_class = create_tracked_trainer(trainer_class, config=config)

    _run_with_tracked_trainer(
        tracked_class=tracked_class,
        dataset_name_or_id=nnunet_args.dataset_name_or_id,
        configuration=nnunet_args.configuration,
        fold=nnunet_args.fold,
        plans_identifier=nnunet_args.p,
        continue_training=nnunet_args.c,
    )


def _resolve_trainer_class(trainer_name: str) -> type | None:
    """Resolve an nnU-Net trainer class by name."""
    try:
        import nnunetv2
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

        trainer_search_path = os.path.join(
            os.path.dirname(os.path.abspath(nnunetv2.__file__)),
            "training",
            "nnUNetTrainer",
        )
        return recursive_find_python_class(
            folder=trainer_search_path,
            class_name=trainer_name,
            current_module="nnunetv2.training.nnUNetTrainer",
        )
    except (ImportError, ModuleNotFoundError):
        logger.debug("Failed to import nnU-Net for trainer resolution", exc_info=True)
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error resolving trainer class %r: %s", trainer_name, type(e).__name__
        )
        logger.debug("Full traceback for trainer resolution", exc_info=True)
        return None


def _run_with_tracked_trainer(
    tracked_class: type,
    dataset_name_or_id: str,
    configuration: str,
    fold: str,
    plans_identifier: str,
    continue_training: bool,
) -> None:
    """Instantiate the tracked trainer and run training."""
    import torch
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessed_folder = os.path.join(nnUNet_preprocessed, dataset_name)
    plans_file = os.path.join(preprocessed_folder, plans_identifier + ".json")
    dataset_json_file = os.path.join(preprocessed_folder, "dataset.json")

    for filepath, label in [(plans_file, "Plans"), (dataset_json_file, "Dataset JSON")]:
        if not os.path.isfile(filepath):
            print(f"Error: {label} file not found: {filepath}", file=sys.stderr)
            sys.exit(1)

    try:
        with open(plans_file, encoding="utf-8") as f:
            plans = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Plans file is not valid JSON: {plans_file}\n  {e}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(dataset_json_file, encoding="utf-8") as f:
            dataset_json = json.load(f)
    except json.JSONDecodeError as e:
        print(
            f"Error: Dataset JSON is not valid JSON: {dataset_json_file}\n  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if fold == "all":
        folds = list(range(5))
    else:
        try:
            fold_int = int(fold)
        except ValueError:
            print(
                f"Error: fold must be a non-negative integer or 'all', got '{fold}'",
                file=sys.stderr,
            )
            sys.exit(1)
        if fold_int < 0:
            print(
                f"Error: fold must be non-negative, got {fold_int}",
                file=sys.stderr,
            )
            sys.exit(1)
        folds = [fold_int]

    for f_idx in folds:
        trainer = tracked_class(
            plans=plans,
            configuration=configuration,
            fold=f_idx,
            dataset_json=dataset_json,
            device=device,
        )

        if continue_training:
            checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_latest.pth")
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_final.pth")
            if os.path.isfile(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)
            else:
                logger.warning(
                    "-c specified but no checkpoint found in %s. Starting from scratch.",
                    trainer.output_folder,
                )

        trainer.run_training()


def _build_nnunet_arg_parser() -> argparse.ArgumentParser:
    """Build an argument parser that mirrors nnUNetv2_train's arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset_name_or_id", type=str)
    parser.add_argument("configuration", type=str)
    parser.add_argument("fold", type=str)
    parser.add_argument("-tr", type=str, default="nnUNetTrainer")
    parser.add_argument("-p", type=str, default="nnUNetPlans")
    parser.add_argument("-c", action="store_true", default=False)
    return parser

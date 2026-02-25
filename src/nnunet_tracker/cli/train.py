"""nnunet-tracker train command -- wraps nnUNetv2_train with MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
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
            "  dataset_name_or_id       Dataset name or ID (e.g., Dataset001_BrainTumour or 1)\n"
            "  configuration            nnU-Net configuration (e.g., 2d, 3d_fullres, 3d_lowres)\n"
            "  fold                     Fold number (0-4) or 'all' (all data, no val split)\n"
            "\n"
            "Optional forwarded arguments:\n"
            "  -tr TRAINER              Trainer class name (default: nnUNetTrainer)\n"
            "  -p PLANS                 Plans identifier (default: nnUNetPlans)\n"
            "  -pretrained_weights PATH Pretrained weights for transfer learning\n"
            "  -num_gpus N              Number of GPUs (only 1 supported)\n"
            "  --npz                    Save softmax predictions as .npz from validation\n"
            "  -c                       Continue training from latest checkpoint\n"
            "  --val                    Validation-only mode (skip training)\n"
            "  --val_best               Use checkpoint_best for validation\n"
            "  --use_compressed         Use compressed training data (no decompression)\n"
            "  --disable_checkpointing  Disable checkpoint saving during training\n"
            "  -device DEVICE           Device: cuda, cpu, or mps (default: cuda)"
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
    # Set BLAS thread limits before any torch/numpy imports (matches nnUNet).
    # Prevents thread oversubscription during GPU training.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

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

    # Mutual exclusion checks (matching nnUNet behavior)
    if nnunet_args.c and nnunet_args.val:
        print("Error: Cannot use -c and --val at the same time.", file=sys.stderr)
        sys.exit(1)

    if nnunet_args.pretrained_weights is not None and nnunet_args.c:
        print(
            "Error: Cannot use -pretrained_weights and -c at the same time. "
            "Pretrained weights can only be used at the beginning of training.",
            file=sys.stderr,
        )
        sys.exit(1)

    if nnunet_args.val_best and nnunet_args.disable_checkpointing:
        print(
            "Error: --val_best is not compatible with --disable_checkpointing.",
            file=sys.stderr,
        )
        sys.exit(1)

    if nnunet_args.num_gpus > 1:
        print(
            "Error: Multi-GPU DDP training (num_gpus > 1) is not supported by nnunet-tracker. "
            "Use nnUNetv2_train directly for multi-GPU training.",
            file=sys.stderr,
        )
        sys.exit(1)

    valid_devices = {"cuda", "cpu", "mps"}
    if nnunet_args.device not in valid_devices:
        print(
            f"Error: Invalid device '{nnunet_args.device}'. "
            f"Must be one of: {', '.join(sorted(valid_devices))}",
            file=sys.stderr,
        )
        sys.exit(1)

    if nnunet_args.pretrained_weights is not None:
        if not os.path.isfile(nnunet_args.pretrained_weights):
            print(
                f"Error: Pretrained weights file not found: {nnunet_args.pretrained_weights}",
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
        pretrained_weights=nnunet_args.pretrained_weights,
        use_compressed_data=nnunet_args.use_compressed,
        export_validation_probabilities=nnunet_args.npz,
        only_run_validation=nnunet_args.val,
        val_with_best=nnunet_args.val_best,
        disable_checkpointing=nnunet_args.disable_checkpointing,
        device_name=nnunet_args.device,
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
        cls: type | None = recursive_find_python_class(
            folder=trainer_search_path,
            class_name=trainer_name,
            current_module="nnunetv2.training.nnUNetTrainer",
        )
        return cls
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
    pretrained_weights: str | None = None,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    only_run_validation: bool = False,
    val_with_best: bool = False,
    disable_checkpointing: bool = False,
    device_name: str = "cuda",
) -> None:
    """Instantiate the tracked trainer and run training."""
    import torch
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    # Inform about default plans (matches nnUNet behavior)
    if plans_identifier == "nnUNetPlans":
        print(
            "\n############################\n"
            "INFO: You are using the old nnU-Net default plans. We have updated our "
            "recommendations. Please consider using those instead! "
            "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/"
            "resenc_presets.md"
            "\n############################\n"
        )

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    device = torch.device(device_name)

    # Match nnUNet's threading configuration for consistent performance
    if device_name == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    elif device_name == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    # Match nnUNet's cuDNN settings for optimal GPU performance
    if torch.cuda.is_available():
        from torch.backends import cudnn

        cudnn.deterministic = False
        cudnn.benchmark = True

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
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error: Cannot read plans file: {plans_file}\n  {e}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(dataset_json_file, encoding="utf-8") as f:
            dataset_json = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(
            f"Error: Cannot read dataset JSON: {dataset_json_file}\n  {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    if fold == "all":
        folds: list[str | int] = ["all"]
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
            unpack_dataset=not use_compressed_data,
            device=device,
        )

        if disable_checkpointing:
            trainer.disable_checkpointing = True

        # Checkpoint loading: match nnUNet priority (final -> latest -> best)
        if continue_training:
            checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_final.pth")
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_latest.pth")
            if not os.path.isfile(checkpoint_path):
                checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_best.pth")
            if os.path.isfile(checkpoint_path):
                trainer.load_checkpoint(checkpoint_path)
            else:
                logger.warning(
                    "-c specified but no checkpoint found in %s. Starting from scratch.",
                    trainer.output_folder,
                )
        elif only_run_validation:
            checkpoint_path = os.path.join(trainer.output_folder, "checkpoint_final.pth")
            if not os.path.isfile(checkpoint_path):
                print(
                    f"Error: Cannot run validation because training is not finished. "
                    f"Missing: {checkpoint_path}",
                    file=sys.stderr,
                )
                sys.exit(1)
            trainer.load_checkpoint(checkpoint_path)
        elif pretrained_weights is not None:
            try:
                from nnunetv2.run.load_pretrained_weights import (
                    load_pretrained_weights as _load_weights,
                )
            except ImportError:
                print(
                    "Error: Pretrained weights loading is not supported "
                    "with this version of nnU-Net.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not getattr(trainer, "was_initialized", True):
                trainer.initialize()
            _load_weights(trainer.network, pretrained_weights, verbose=True)

        if not only_run_validation:
            trainer.run_training()

        try:
            if val_with_best:
                best_ckpt = os.path.join(trainer.output_folder, "checkpoint_best.pth")
                if os.path.isfile(best_ckpt):
                    trainer.load_checkpoint(best_ckpt)
                else:
                    logger.warning("--val_best specified but checkpoint_best.pth not found.")
            trainer.perform_actual_validation(
                save_probabilities=export_validation_probabilities,
            )
        except Exception:
            logger.warning(
                "perform_actual_validation() failed for fold %s. Training completed successfully.",
                f_idx,
            )
            logger.debug("Validation error details", exc_info=True)


def _build_nnunet_arg_parser() -> argparse.ArgumentParser:
    """Build an argument parser that mirrors nnUNetv2_train's arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("dataset_name_or_id", type=str)
    parser.add_argument("configuration", type=str)
    parser.add_argument("fold", type=str)
    parser.add_argument("-tr", type=str, default="nnUNetTrainer")
    parser.add_argument("-p", type=str, default="nnUNetPlans")
    parser.add_argument("-pretrained_weights", type=str, default=None)
    parser.add_argument("-num_gpus", type=int, default=1)
    parser.add_argument("--npz", action="store_true", default=False)
    parser.add_argument("-c", "--c", action="store_true", default=False)
    parser.add_argument("--val", action="store_true", default=False)
    parser.add_argument("--val_best", action="store_true", default=False)
    parser.add_argument("--use_compressed", action="store_true", default=False)
    parser.add_argument("--disable_checkpointing", action="store_true", default=False)
    parser.add_argument("-device", type=str, default="cuda")
    return parser

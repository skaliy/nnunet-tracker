"""Shared validation helpers for CLI subcommands."""

from __future__ import annotations

import sys

from nnunet_tracker.config import SAFE_TAG_VALUE_PATTERN

__all__ = ["parse_folds", "validate_cv_group"]


def parse_folds(value: str) -> tuple[int, ...]:
    """Parse a comma-separated fold string into a tuple of non-negative integers.

    Exits with code 1 and prints an error message on invalid input.
    """
    try:
        folds = tuple(int(f.strip()) for f in value.split(","))
    except ValueError:
        print(
            f"Error: --folds must be comma-separated integers, got '{value}'",
            file=sys.stderr,
        )
        sys.exit(1)

    for fold in folds:
        if fold < 0:
            print(
                f"Error: fold values must be non-negative, got {fold}",
                file=sys.stderr,
            )
            sys.exit(1)

    return folds


def validate_cv_group(value: str | None) -> str | None:
    """Validate a --cv-group value against the safe tag pattern.

    Returns the value unchanged if valid (or None).
    Exits with code 1 and prints an error message on invalid input.
    """
    if value is not None and not SAFE_TAG_VALUE_PATTERN.fullmatch(value):
        print(
            "Error: --cv-group contains invalid characters. "
            "Only alphanumeric, underscore, hyphen, dot, and pipe are allowed.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value

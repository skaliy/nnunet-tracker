"""nnU-Net version compatibility checks."""

from __future__ import annotations

import functools
import logging
import re
import warnings

logger = logging.getLogger("nnunet_tracker")

NNUNET_MIN_VERSION = "2.2"
NNUNET_MAX_VERSION = "2.7"


def check_nnunet_available() -> bool:
    """Check if nnU-Net v2 is importable."""
    try:
        import nnunetv2  # noqa: F401

        return True
    except ImportError:
        return False


def get_nnunet_version() -> str | None:
    """Get the installed nnU-Net version string."""
    try:
        from importlib.metadata import version

        return version("nnunetv2")
    except Exception:
        return None


def _parse_version_tuple(ver_str: str) -> tuple[int, ...]:
    """Parse a version string like '2.5.1' into a comparable tuple (2, 5, 1).

    Strips non-numeric suffixes (e.g. '2.5.1rc1' -> (2, 5, 1)).
    """
    parts = []
    for segment in ver_str.split("."):
        match = re.match(r"(\d+)", segment)
        if match:
            parts.append(int(match.group(1)))
    return tuple(parts)


@functools.lru_cache(maxsize=1)
def check_nnunet_version() -> None:
    """Verify nnU-Net version is within the supported range.

    Emits a warning if outside range. Never raises.
    """
    ver = get_nnunet_version()
    if ver is None:
        warnings.warn(
            "nnunet-tracker: Could not determine nnU-Net version. "
            "Version compatibility checks skipped.",
            UserWarning,
            stacklevel=2,
        )
        return

    try:
        v = _parse_version_tuple(ver)
        v_min = _parse_version_tuple(NNUNET_MIN_VERSION)
        v_max = _parse_version_tuple(NNUNET_MAX_VERSION)
        if v < v_min or v >= v_max:
            warnings.warn(
                f"nnunet-tracker: nnU-Net {ver} is outside the tested range "
                f"[{NNUNET_MIN_VERSION}, {NNUNET_MAX_VERSION}). "
                "Some features may not work correctly.",
                UserWarning,
                stacklevel=2,
            )
    except (ValueError, TypeError):
        logger.debug("Failed to parse nnU-Net version %r, skipping version check", ver)

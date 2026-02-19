"""Dataset fingerprinting and hashing for reproducibility tracking."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("nnunet_tracker")

__all__ = ["compute_fingerprint", "find_dataset_fingerprint_path", "hash_dict"]


def _canonical_json_bytes(obj: Any) -> bytes:
    """Convert a JSON-serializable object to canonical UTF-8 bytes.

    Uses sorted keys and no extra whitespace to ensure deterministic
    output regardless of dict insertion order.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_dict(obj: Any) -> str:
    return _sha256_hex(_canonical_json_bytes(obj))


def compute_fingerprint(trainer: Any) -> dict[str, str]:
    """Compute dataset/config fingerprint hashes from trainer state.

    Returns a dict suitable for mlflow.log_params() with keys:
        - fingerprint_dataset_json: SHA-256 of dataset.json dict
        - fingerprint_plans: SHA-256 of plans dict
        - fingerprint_dataset_fingerprint: SHA-256 of dataset_fingerprint.json
        - fingerprint_composite: SHA-256 of all above concatenated

    Each component is independently guarded — failure of one does not
    prevent others from being computed.

    Note:
        The composite hash depends on which components are available. If a
        component fails (e.g., missing ``dataset_fingerprint.json``), the
        composite will differ from an environment where all components succeed.
        Compare individual component hashes for stable cross-environment checks.
    """
    hashes: dict[str, str] = {}
    composite_parts: list[str] = []

    dataset_json = getattr(trainer, "dataset_json", None)
    if dataset_json is not None and isinstance(dataset_json, dict):
        try:
            h = hash_dict(dataset_json)
            hashes["fingerprint_dataset_json"] = h
            composite_parts.append(h)
        except Exception:
            logger.debug("Failed to hash dataset_json", exc_info=True)

    pm = getattr(trainer, "plans_manager", None)
    plans = getattr(pm, "plans", None) if pm is not None else None
    if plans is not None and isinstance(plans, dict):
        try:
            h = hash_dict(plans)
            hashes["fingerprint_plans"] = h
            composite_parts.append(h)
        except Exception:
            logger.debug("Failed to hash plans", exc_info=True)

    fp_path = find_dataset_fingerprint_path(trainer)
    if fp_path is not None:
        try:
            with open(fp_path, encoding="utf-8") as f:
                fp_dict = json.load(f)
            h = hash_dict(fp_dict)
            hashes["fingerprint_dataset_fingerprint"] = h
            composite_parts.append(h)
        except Exception:
            logger.debug(
                "Could not read dataset_fingerprint.json at %s",
                fp_path,
                exc_info=True,
            )

    if composite_parts:
        combined = "|".join(composite_parts)
        hashes["fingerprint_composite"] = _sha256_hex(combined.encode("utf-8"))

    return hashes


def find_dataset_fingerprint_path(trainer: Any) -> str | None:
    """Locate the dataset_fingerprint.json file for this trainer's dataset.

    Uses the nnUNet_preprocessed environment variable combined with
    the dataset name from the trainer's plans manager.

    Returns the file path if it exists, or None.
    """
    dataset_name = getattr(getattr(trainer, "plans_manager", None), "dataset_name", None)
    if dataset_name is None:
        return None

    preprocessed_dir = os.environ.get("nnUNet_preprocessed")
    if preprocessed_dir is None:
        return None

    path = os.path.join(preprocessed_dir, dataset_name, "dataset_fingerprint.json")

    # Path containment check: ensure resolved path is within preprocessed_dir
    real_path = Path(path).resolve()
    real_base = Path(preprocessed_dir).resolve()
    if not real_path.is_relative_to(real_base):
        logger.warning(
            "dataset_fingerprint path %r is outside nnUNet_preprocessed (%r), skipping",
            path,
            preprocessed_dir,
        )
        return None

    resolved = str(real_path)
    if os.path.isfile(resolved):
        return resolved

    return None

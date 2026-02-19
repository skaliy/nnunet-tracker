"""DDP rank detection and process isolation."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("nnunet_tracker")


def _safe_parse_rank(var_name: str) -> int | None:
    """Safely parse an integer rank from an environment variable.

    Returns None if the variable is unset or not a valid integer.
    """
    value = os.environ.get(var_name)
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        logger.warning(
            "%s=%r is not a valid integer, ignoring for rank detection.",
            var_name,
            value,
        )
        return None


def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0).

    Detection strategy (ordered by priority):
    1. LOCAL_RANK env var (set by torchrun / torch.distributed.launch)
    2. SLURM_LOCALID (set by SLURM)
    3. Default to True (single-GPU training)

    Note:
        Uses *local* rank, not global rank. In multi-node DDP, rank 0 on
        each node will return True. Use a shared MLflow tracking server
        (not file-based ``./mlruns``) to avoid write conflicts in
        multi-node setups.
    """
    local_rank = _safe_parse_rank("LOCAL_RANK")
    if local_rank is not None:
        return local_rank == 0

    slurm_local_id = _safe_parse_rank("SLURM_LOCALID")
    if slurm_local_id is not None:
        return slurm_local_id == 0

    return True


def should_log(trainer_instance: object) -> bool:
    """Check if this trainer instance should log to MLflow.

    Uses the trainer's own DDP attributes first (most reliable at runtime),
    then falls back to environment variable detection.

    Note:
        In multi-node DDP, multiple processes may pass this check (one per
        node). See :func:`is_main_process` for details.
    """
    is_ddp = getattr(trainer_instance, "is_ddp", False)
    if is_ddp:
        local_rank = getattr(trainer_instance, "local_rank", None)
        if local_rank is None:
            return is_main_process()
        return local_rank == 0

    return is_main_process()

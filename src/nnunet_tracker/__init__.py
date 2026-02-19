"""nnunet-tracker: Lightweight MLflow-based experiment tracking for nnU-Net."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("nnunet-tracker")
except PackageNotFoundError:
    __version__ = "0.0.0"

from nnunet_tracker.config import TrackerConfig
from nnunet_tracker.export import export_csv, export_latex
from nnunet_tracker.factory import create_tracked_trainer
from nnunet_tracker.summarize import CVSummary, FoldResult, log_cv_summary, summarize_experiment

__all__ = [
    "create_tracked_trainer",
    "summarize_experiment",
    "log_cv_summary",
    "CVSummary",
    "FoldResult",
    "TrackerConfig",
    "export_csv",
    "export_latex",
    "__version__",
]

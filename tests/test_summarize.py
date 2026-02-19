"""Tests for nnunet_tracker.summarize module."""

from __future__ import annotations

import statistics
from unittest.mock import MagicMock

import pytest

from nnunet_tracker.summarize import (
    CVSummary,
    FoldResult,
    _extract_fold_result,
    _query_fold_runs,
    log_cv_summary,
    summarize_experiment,
)
from tests.conftest import (
    MockMLflowExperiment as MockExperiment,
)
from tests.conftest import (
    MockMLflowRun as MockRun,
)
from tests.conftest import (
    MockMLflowRunData as MockRunData,
)
from tests.conftest import (
    MockMLflowRunInfo as MockRunInfo,
)
from tests.conftest import (
    mock_mlflow_modules as _mock_mlflow_for_summarize,
)


def _make_fold_run(
    fold: int,
    mean_fg_dice: float = 0.85,
    val_loss: float = 0.35,
    ema_fg_dice: float = 0.84,
    dice_per_class: dict[int, float] | None = None,
    status: str = "FINISHED",
    run_id: str | None = None,
    with_tags: bool = True,
) -> MockRun:
    """Create a mock fold run with realistic metrics."""
    if dice_per_class is None:
        dice_per_class = {0: 0.80, 1: 0.90}

    tags: dict[str, str] = {}
    if with_tags:
        tags = {
            "nnunet_tracker.fold": str(fold),
            "nnunet_tracker.run_type": "fold",
            "nnunet_tracker.cv_group": "Dataset001|3d_fullres|nnUNetPlans",
        }

    metrics: dict[str, float] = {
        "mean_fg_dice": mean_fg_dice,
        "val_loss": val_loss,
        "ema_fg_dice": ema_fg_dice,
    }
    for cls_idx, val in dice_per_class.items():
        metrics[f"dice_class_{cls_idx}"] = val

    return MockRun(
        info=MockRunInfo(
            run_id=run_id or f"run-fold-{fold}",
            status=status,
            start_time=1000 + fold,
        ),
        data=MockRunData(
            metrics=metrics,
            params={"fold": str(fold), "dataset_name": "Dataset001"},
            tags=tags,
        ),
    )


class TestFoldResult:
    """Tests for FoldResult dataclass."""

    def test_basic_creation(self) -> None:
        result = FoldResult(fold=0, run_id="r1", status="FINISHED", mean_fg_dice=0.85)
        assert result.fold == 0
        assert result.run_id == "r1"
        assert result.mean_fg_dice == 0.85
        assert result.dice_per_class == {}

    def test_frozen(self) -> None:
        result = FoldResult(fold=0, run_id="r1", status="FINISHED")
        with pytest.raises(AttributeError):
            result.fold = 1  # type: ignore[misc]


class TestCVSummary:
    """Tests for CVSummary dataclass and aggregate computation."""

    def _make_summary(self, fold_results: dict[int, FoldResult] | None = None) -> CVSummary:
        if fold_results is None:
            fold_results = {}
        return CVSummary(
            cv_group="Dataset001|3d_fullres|nnUNetPlans",
            experiment_name="test",
            fold_results=fold_results,
        )

    def test_completed_folds(self) -> None:
        summary = self._make_summary(
            {
                0: FoldResult(fold=0, run_id="r0", status="FINISHED"),
                2: FoldResult(fold=2, run_id="r2", status="FINISHED"),
            }
        )
        assert summary.completed_folds == [0, 2]

    def test_missing_folds(self) -> None:
        summary = self._make_summary(
            {
                0: FoldResult(fold=0, run_id="r0", status="FINISHED"),
                2: FoldResult(fold=2, run_id="r2", status="FINISHED"),
            }
        )
        assert summary.missing_folds == [1, 3, 4]

    def test_is_complete_true(self) -> None:
        results = {i: FoldResult(fold=i, run_id=f"r{i}", status="FINISHED") for i in range(5)}
        summary = self._make_summary(results)
        assert summary.is_complete is True

    def test_is_complete_false(self) -> None:
        summary = self._make_summary(
            {
                0: FoldResult(fold=0, run_id="r0", status="FINISHED"),
            }
        )
        assert summary.is_complete is False

    def test_aggregate_empty(self) -> None:
        summary = self._make_summary()
        assert summary.compute_aggregate_metrics() == {}

    def test_aggregate_single_fold_no_std(self) -> None:
        """Single fold: mean computed, no std (requires >= 2 values)."""
        summary = self._make_summary(
            {
                0: FoldResult(
                    fold=0,
                    run_id="r0",
                    status="FINISHED",
                    mean_fg_dice=0.85,
                    val_loss=0.35,
                ),
            }
        )
        agg = summary.compute_aggregate_metrics()
        assert agg["cv_mean_fg_dice"] == 0.85
        assert "cv_std_fg_dice" not in agg
        assert agg["cv_mean_val_loss"] == 0.35
        assert "cv_std_val_loss" not in agg

    def test_aggregate_mean_std_correct(self) -> None:
        """Verify mean/std against statistics module."""
        dice_values = [0.82, 0.84, 0.86, 0.83, 0.85]
        results = {
            i: FoldResult(
                fold=i,
                run_id=f"r{i}",
                status="FINISHED",
                mean_fg_dice=dice_values[i],
            )
            for i in range(5)
        }
        summary = self._make_summary(results)
        agg = summary.compute_aggregate_metrics()

        assert agg["cv_mean_fg_dice"] == pytest.approx(statistics.mean(dice_values))
        assert agg["cv_std_fg_dice"] == pytest.approx(statistics.stdev(dice_values))

    def test_aggregate_per_class_dice(self) -> None:
        """Per-class Dice aggregated correctly."""
        results = {
            0: FoldResult(
                fold=0,
                run_id="r0",
                status="FINISHED",
                dice_per_class={0: 0.80, 1: 0.90},
            ),
            1: FoldResult(
                fold=1,
                run_id="r1",
                status="FINISHED",
                dice_per_class={0: 0.82, 1: 0.88},
            ),
        }
        summary = self._make_summary(results)
        agg = summary.compute_aggregate_metrics()

        assert agg["cv_mean_dice_class_0"] == pytest.approx(statistics.mean([0.80, 0.82]))
        assert agg["cv_std_dice_class_0"] == pytest.approx(statistics.stdev([0.80, 0.82]))
        assert agg["cv_mean_dice_class_1"] == pytest.approx(statistics.mean([0.90, 0.88]))

    def test_aggregate_missing_metrics_handled(self) -> None:
        """Folds with None metrics are excluded from aggregation."""
        results = {
            0: FoldResult(fold=0, run_id="r0", status="FINISHED", mean_fg_dice=0.85),
            1: FoldResult(fold=1, run_id="r1", status="FINISHED", mean_fg_dice=None),
        }
        summary = self._make_summary(results)
        agg = summary.compute_aggregate_metrics()
        # Only one value, so mean but no std
        assert agg["cv_mean_fg_dice"] == 0.85
        assert "cv_std_fg_dice" not in agg

    def test_aggregate_identical_metrics(self) -> None:
        """All folds have identical metrics: std = 0."""
        results = {
            i: FoldResult(
                fold=i,
                run_id=f"r{i}",
                status="FINISHED",
                mean_fg_dice=0.85,
            )
            for i in range(5)
        }
        summary = self._make_summary(results)
        agg = summary.compute_aggregate_metrics()
        assert agg["cv_std_fg_dice"] == pytest.approx(0.0)


class TestExtractFoldResult:
    """Tests for _extract_fold_result helper."""

    def test_extracts_from_tagged_run(self) -> None:
        run = _make_fold_run(fold=2, mean_fg_dice=0.87, val_loss=0.30)
        result = _extract_fold_result(run)
        assert result is not None
        assert result.fold == 2
        assert result.mean_fg_dice == 0.87
        assert result.val_loss == 0.30
        assert result.run_id == "run-fold-2"

    def test_extracts_per_class_dice(self) -> None:
        run = _make_fold_run(fold=0, dice_per_class={0: 0.75, 1: 0.90, 2: 0.82})
        result = _extract_fold_result(run)
        assert result is not None
        assert result.dice_per_class == {0: 0.75, 1: 0.90, 2: 0.82}

    def test_falls_back_to_param_fold(self) -> None:
        """v0.2.0 compat: fold from params when tag is missing."""
        run = _make_fold_run(fold=3, with_tags=False)
        result = _extract_fold_result(run)
        assert result is not None
        assert result.fold == 3

    def test_returns_none_when_no_fold(self) -> None:
        run = MockRun(
            data=MockRunData(metrics={"mean_fg_dice": 0.85}, params={}, tags={}),
        )
        result = _extract_fold_result(run)
        assert result is None

    def test_returns_none_when_fold_not_int(self) -> None:
        run = MockRun(
            data=MockRunData(
                metrics={"mean_fg_dice": 0.85},
                params={"fold": "abc"},
                tags={},
            ),
        )
        result = _extract_fold_result(run)
        assert result is None

    def test_handles_missing_metrics(self) -> None:
        run = MockRun(
            info=MockRunInfo(run_id="r0"),
            data=MockRunData(
                metrics={},
                params={"fold": "0"},
                tags={"nnunet_tracker.fold": "0"},
            ),
        )
        result = _extract_fold_result(run)
        assert result is not None
        assert result.mean_fg_dice is None
        assert result.val_loss is None
        assert result.dice_per_class == {}


class TestQueryFoldRuns:
    """Tests for _query_fold_runs helper."""

    def test_tag_based_query(self) -> None:
        client = MagicMock()
        runs = [_make_fold_run(0), _make_fold_run(1)]
        client.search_runs.return_value = runs

        result = _query_fold_runs(client, "exp-1", "Dataset001|3d_fullres|nnUNetPlans")
        assert len(result) == 2
        # Verify filter_string contains tag query
        filter_str = client.search_runs.call_args[1]["filter_string"]
        assert "nnunet_tracker.cv_group" in filter_str

    def test_cv_group_no_tagged_runs_returns_empty(self) -> None:
        """When cv_group specified but no tagged runs exist, returns empty list."""
        client = MagicMock()
        client.search_runs.return_value = []

        result = _query_fold_runs(client, "exp-1", "Dataset001|3d_fullres|nnUNetPlans")
        assert result == []
        # Only one search call (no fallback)
        assert client.search_runs.call_count == 1

    def test_no_cv_group_queries_all_fold_runs(self) -> None:
        """When cv_group is None, queries all fold-typed FINISHED runs."""
        client = MagicMock()
        runs = [_make_fold_run(0), _make_fold_run(1)]
        client.search_runs.return_value = runs

        result = _query_fold_runs(client, "exp-1", None)
        assert len(result) == 2
        filter_str = client.search_runs.call_args[1]["filter_string"]
        assert "nnunet_tracker.run_type" in filter_str
        assert "nnunet_tracker.cv_group" not in filter_str

    def test_no_runs_returns_empty(self) -> None:
        client = MagicMock()
        client.search_runs.return_value = []
        result = _query_fold_runs(client, "exp-1", None)
        assert result == []

    def test_unsafe_cv_group_returns_empty(self) -> None:
        client = MagicMock()
        result = _query_fold_runs(client, "exp-1", "test' or '1'='1")
        assert result == []
        client.search_runs.assert_not_called()


class TestSummarizeExperiment:
    """Tests for summarize_experiment top-level function."""

    def test_raises_on_missing_experiment(self) -> None:
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with _mock_mlflow_for_summarize(mock_client):
            with pytest.raises(ValueError, match="Experiment not found"):
                summarize_experiment("nonexistent")

    def test_raises_on_no_runs(self) -> None:
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        mock_client.search_runs.return_value = []

        with _mock_mlflow_for_summarize(mock_client):
            with pytest.raises(ValueError, match="No completed fold runs"):
                summarize_experiment("test_experiment")

    def test_five_fold_complete(self) -> None:
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        runs = [_make_fold_run(i, mean_fg_dice=0.82 + i * 0.01) for i in range(5)]
        # First call for auto-detect, second for tag query
        mock_client.search_runs.side_effect = [runs[:1], runs]

        with _mock_mlflow_for_summarize(mock_client):
            summary = summarize_experiment("test_experiment")

        assert summary.is_complete
        assert len(summary.completed_folds) == 5
        assert summary.missing_folds == []

    def test_partial_folds(self) -> None:
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        runs = [_make_fold_run(i) for i in range(3)]
        mock_client.search_runs.side_effect = [runs[:1], runs]

        with _mock_mlflow_for_summarize(mock_client):
            summary = summarize_experiment("test_experiment")

        assert not summary.is_complete
        assert summary.completed_folds == [0, 1, 2]
        assert summary.missing_folds == [3, 4]

    def test_deduplicates_fold_runs(self) -> None:
        """If a fold has multiple FINISHED runs, keep the most recent."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        # Two runs for fold 0 (first is more recent due to DESC order)
        runs = [
            _make_fold_run(0, mean_fg_dice=0.90, run_id="newer"),
            _make_fold_run(0, mean_fg_dice=0.80, run_id="older"),
        ]
        mock_client.search_runs.side_effect = [runs[:1], runs]

        with _mock_mlflow_for_summarize(mock_client):
            summary = summarize_experiment("test_experiment")

        assert summary.fold_results[0].run_id == "newer"
        assert summary.fold_results[0].mean_fg_dice == 0.90

    def test_raises_on_no_extractable_folds(self) -> None:
        """Runs returned but none have a fold number -> ValueError."""
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        # Runs with no fold tag or param
        no_fold_run = MockRun(
            info=MockRunInfo(run_id="r0", status="FINISHED"),
            data=MockRunData(metrics={}, params={}, tags={}),
        )
        mock_client.search_runs.side_effect = [[no_fold_run], [no_fold_run]]

        with _mock_mlflow_for_summarize(mock_client):
            with pytest.raises(ValueError, match="No fold results could be extracted"):
                summarize_experiment("test_experiment")


class TestLogCvSummary:
    """Tests for log_cv_summary function."""

    def _make_complete_summary(self) -> CVSummary:
        results = {
            i: FoldResult(
                fold=i,
                run_id=f"r{i}",
                status="FINISHED",
                mean_fg_dice=0.82 + i * 0.01,
                val_loss=0.35 - i * 0.01,
            )
            for i in range(5)
        }
        return CVSummary(
            cv_group="Dataset001|3d_fullres|nnUNetPlans",
            experiment_name="test",
            fold_results=results,
        )

    def test_returns_none_on_empty_cv_group(self) -> None:
        summary = CVSummary(cv_group="", experiment_name="test")
        result = log_cv_summary(summary)
        assert result is None

    def test_returns_none_on_unsafe_cv_group(self) -> None:
        summary = CVSummary(cv_group="test' or '1'='1", experiment_name="test")
        result = log_cv_summary(summary)
        assert result is None

    def test_creates_new_run_via_client(self) -> None:
        summary = self._make_complete_summary()
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        mock_client.search_runs.return_value = []
        mock_client.create_run.return_value = MagicMock(info=MagicMock(run_id="new-summary-id"))

        with _mock_mlflow_for_summarize(mock_client):
            run_id = log_cv_summary(summary)

        assert run_id == "new-summary-id"
        mock_client.create_run.assert_called_once()
        # Verify tags set via client API
        tag_calls = {c[0][1]: c[0][2] for c in mock_client.set_tag.call_args_list}
        assert tag_calls["nnunet_tracker.run_type"] == "cv_summary"
        assert tag_calls["nnunet_tracker.is_complete"] == "True"

    def test_reopens_existing_run(self) -> None:
        summary = self._make_complete_summary()
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        existing_run = MagicMock(info=MagicMock(run_id="existing-id"))
        mock_client.search_runs.return_value = [existing_run]

        with _mock_mlflow_for_summarize(mock_client):
            run_id = log_cv_summary(summary)

        assert run_id == "existing-id"
        mock_client.create_run.assert_not_called()

    def test_logs_aggregate_metrics(self) -> None:
        summary = self._make_complete_summary()
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = MockExperiment()
        mock_client.search_runs.return_value = []
        mock_client.create_run.return_value = MagicMock(info=MagicMock(run_id="summary-id"))

        with _mock_mlflow_for_summarize(mock_client):
            log_cv_summary(summary)

        mock_client.log_batch.assert_called_once()
        _, kwargs = mock_client.log_batch.call_args
        metric_keys = {m.key for m in kwargs["metrics"]}
        assert "cv_mean_fg_dice" in metric_keys

    def test_returns_none_on_failure(self) -> None:
        summary = self._make_complete_summary()
        mock_client = MagicMock()
        mock_client.get_experiment_by_name.side_effect = RuntimeError("boom")

        with _mock_mlflow_for_summarize(mock_client):
            result = log_cv_summary(summary)

        assert result is None

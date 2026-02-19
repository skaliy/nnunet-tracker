"""Tests for nnunet-tracker list CLI command."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from nnunet_tracker.cli.list import _query_experiments
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
    mock_mlflow_modules as _mock_mlflow_for_list,
)


def _make_fold_run(fold: int, cv_group: str = "Dataset001|3d_fullres|nnUNetPlans") -> MockRun:
    return MockRun(
        info=MockRunInfo(run_id=f"run-{fold}", start_time=1000 + fold),
        data=MockRunData(
            tags={
                "nnunet_tracker.fold": str(fold),
                "nnunet_tracker.run_type": "fold",
                "nnunet_tracker.cv_group": cv_group,
            },
        ),
    )


class TestListRegistration:
    """Tests for list subcommand registration."""

    def test_help_lists_list_command(self, capsys) -> None:
        from nnunet_tracker.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "list" in captured.out

    def test_list_help(self, capsys) -> None:
        from nnunet_tracker.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["list", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--tracking-uri" in captured.out
        assert "--json" in captured.out


class TestQueryExperiments:
    """Tests for _query_experiments helper."""

    def test_no_experiments_with_fold_runs(self) -> None:
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = [MockExperiment()]
        mock_client.search_runs.return_value = []

        with _mock_mlflow_for_list(mock_client):
            result = _query_experiments("./mlruns")
        assert result == []

    def test_single_experiment(self) -> None:
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = [MockExperiment()]
        runs = [_make_fold_run(i) for i in range(5)]
        mock_client.search_runs.return_value = runs

        with _mock_mlflow_for_list(mock_client):
            result = _query_experiments("./mlruns")

        assert len(result) == 1
        assert result[0]["name"] == "test_experiment"
        assert result[0]["total_runs"] == 5
        assert result[0]["cv_groups"][0]["num_completed"] == 5

    def test_multiple_cv_groups(self) -> None:
        mock_client = MagicMock()
        mock_client.search_experiments.return_value = [MockExperiment()]
        runs = [
            _make_fold_run(0, cv_group="group_a"),
            _make_fold_run(1, cv_group="group_a"),
            _make_fold_run(0, cv_group="group_b"),
        ]
        mock_client.search_runs.return_value = runs

        with _mock_mlflow_for_list(mock_client):
            result = _query_experiments("./mlruns")

        assert len(result) == 1
        assert len(result[0]["cv_groups"]) == 2
        cv_group_names = {g["cv_group"] for g in result[0]["cv_groups"]}
        assert cv_group_names == {"group_a", "group_b"}


class TestListCommand:
    """Tests for list command execution."""

    def test_no_experiments_prints_message(self, capsys) -> None:
        from nnunet_tracker.cli import main

        with (
            patch(
                "nnunet_tracker.cli.list._query_experiments",
                return_value=[],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["list"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "No nnunet-tracker experiments found" in captured.err

    def test_table_output(self, capsys) -> None:
        from nnunet_tracker.cli import main

        experiments = [
            {
                "name": "test_exp",
                "experiment_id": "exp-1",
                "cv_groups": [
                    {
                        "cv_group": "Dataset001|3d_fullres|nnUNetPlans",
                        "completed_folds": [0, 1, 2, 3, 4],
                        "num_completed": 5,
                        "latest_start_time": 1000,
                    }
                ],
                "total_runs": 5,
            }
        ]
        with patch(
            "nnunet_tracker.cli.list._query_experiments",
            return_value=experiments,
        ):
            main(["list"])
        captured = capsys.readouterr()
        assert "test_exp" in captured.out
        assert "5" in captured.out

    def test_json_output(self, capsys) -> None:
        from nnunet_tracker.cli import main

        experiments = [
            {
                "name": "test_exp",
                "experiment_id": "exp-1",
                "cv_groups": [
                    {
                        "cv_group": "group_a",
                        "completed_folds": [0, 1, 2],
                        "num_completed": 3,
                        "latest_start_time": 1000,
                    }
                ],
                "total_runs": 3,
            }
        ]
        with patch(
            "nnunet_tracker.cli.list._query_experiments",
            return_value=experiments,
        ):
            main(["list", "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 1
        assert data[0]["name"] == "test_exp"

    def test_tracking_uri_override(self) -> None:
        from nnunet_tracker.cli import main

        with (
            patch(
                "nnunet_tracker.cli.list._query_experiments",
                return_value=[],
            ) as mock_query,
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["list", "--tracking-uri", "/custom/path"])
        assert exc_info.value.code == 0
        mock_query.assert_called_once_with("/custom/path", max_experiments=50)

    def test_mlflow_error_exits_1(self, capsys) -> None:
        from nnunet_tracker.cli import main

        with (
            patch(
                "nnunet_tracker.cli.list._query_experiments",
                side_effect=RuntimeError("Connection failed"),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            main(["list"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "could not query experiments" in captured.err

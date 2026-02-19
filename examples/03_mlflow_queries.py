"""
03_mlflow_queries.py -- Direct MLflow API access for nnunet-tracker data.

This script demonstrates how to query nnunet-tracker's MLflow data directly
using the MLflow Python API. This is useful when you need to:

- Build custom dashboards or reports beyond what the CLI provides.
- Compare metrics across multiple experiments programmatically.
- Extract raw metric histories (per-epoch values, not just final).
- Integrate with other analysis pipelines or tools.

nnunet-tracker tags its MLflow runs with a consistent set of tags:

    nnunet_tracker.fold       -- Fold number (0, 1, 2, 3, 4).
    nnunet_tracker.run_type   -- Either 'fold' or 'cv_summary'.
    nnunet_tracker.cv_group   -- Pipe-separated identifier:
                                 "dataset_name|configuration|plans_name"
                                 e.g. "Dataset001|3d_fullres|nnUNetPlans".

These tags allow you to filter and group runs precisely using MLflow's
search API. This script shows the common query patterns.

NOTE: All examples require an MLflow tracking store with existing runs.
Each function includes error handling for the case where no data exists.

"""

from __future__ import annotations

from typing import Any


def demonstrate_finding_experiments(client: Any) -> str | None:
    """Show how to discover experiments in the MLflow store.

    MlflowClient.search_experiments() returns all experiments. You can
    filter by name or other attributes. Each experiment has an
    experiment_id that you need for querying runs.
    """
    experiments = client.search_experiments()

    if not experiments:
        print("  No experiments found in the tracking store.")
        return None

    print(f"  Found {len(experiments)} experiment(s):")
    for exp in experiments:
        print(f"    - {exp.name} (id={exp.experiment_id})")
    print()

    # Get a specific experiment by name
    target_name = "Dataset001_BrainTumour"
    experiment = client.get_experiment_by_name(target_name)
    if experiment is not None:
        print(f"  Found target experiment: {experiment.name}")
        print(f"  Experiment ID: {experiment.experiment_id}")
        return experiment.experiment_id
    else:
        print(f"  Target experiment '{target_name}' not found.")
        print(f"  Using first available experiment for demonstration.")
        return experiments[0].experiment_id


def demonstrate_querying_fold_runs(client: Any, experiment_id: str) -> list:
    """Query fold runs using nnunet-tracker tags.

    nnunet-tracker tags every training run with:
        nnunet_tracker.run_type = 'fold'
        nnunet_tracker.fold = '<fold_number>'
        nnunet_tracker.cv_group = '<dataset>|<config>|<plans>'

    You can use these tags in MLflow filter strings to find exactly the
    runs you need. Tag names containing dots must be backtick-quoted
    in filter strings.
    """
    # --- Query all fold runs (any cv_group) ---
    print("  Querying all fold runs...")
    all_fold_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.`nnunet_tracker.run_type` = 'fold'",
        order_by=["start_time DESC"],
        max_results=100,
    )
    print(f"  Found {len(all_fold_runs)} fold run(s).")
    print()

    if not all_fold_runs:
        return []

    # --- Query fold runs for a specific cv_group ---
    # First, find what cv_groups exist
    cv_groups = set()
    for run in all_fold_runs:
        group = run.data.tags.get("nnunet_tracker.cv_group")
        if group:
            cv_groups.add(group)

    if cv_groups:
        print(f"  CV groups found: {sorted(cv_groups)}")
        target_group = sorted(cv_groups)[0]
        print(f"  Querying runs for cv_group: {target_group}")
        print()

        group_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                f"tags.`nnunet_tracker.cv_group` = '{target_group}' "
                "and tags.`nnunet_tracker.run_type` = 'fold' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=["start_time DESC"],
            max_results=20,
        )
        print(f"  Found {len(group_runs)} finished fold run(s) for this group.")
    else:
        # Pre-v0.3.0 runs may not have cv_group tags.
        print("  No cv_group tags found (possibly pre-v0.3.0 runs).")
        print("  Falling back to querying all finished fold runs.")
        group_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=(
                "tags.`nnunet_tracker.run_type` = 'fold' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=["start_time DESC"],
            max_results=20,
        )

    print()
    return group_runs


def demonstrate_extracting_metrics(runs: list) -> None:
    """Extract metrics from fold runs.

    Each run's data object has:
        run.data.metrics  -- Dict of final metric values (latest logged value).
        run.data.params   -- Dict of parameter values.
        run.data.tags     -- Dict of tag values.
        run.info          -- Run metadata (run_id, status, start_time, etc.).

    Key metric names logged by nnunet-tracker:
        mean_fg_dice      -- Mean foreground Dice score (final epoch).
        val_loss          -- Validation loss (final epoch).
        train_loss        -- Training loss (final epoch).
        ema_fg_dice       -- Exponential moving average of foreground Dice.
        learning_rate     -- Learning rate (final epoch).
        dice_class_0      -- Per-class Dice for class 0.
        dice_class_1      -- Per-class Dice for class 1.
        ... (one metric per foreground class)
    """
    if not runs:
        print("  No runs to extract metrics from.")
        return

    print("  Extracting metrics from fold runs:")
    print()

    for run in runs:
        tags = run.data.tags
        metrics = run.data.metrics
        params = run.data.params

        fold = tags.get("nnunet_tracker.fold", params.get("fold", "?"))
        run_id = run.info.run_id[:8]
        status = run.info.status

        # Final metric values
        mean_dice = metrics.get("mean_fg_dice")
        val_loss = metrics.get("val_loss")
        ema_dice = metrics.get("ema_fg_dice")

        dice_str = f"{mean_dice:.4f}" if mean_dice is not None else "N/A"
        loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        ema_str = f"{ema_dice:.4f}" if ema_dice is not None else "N/A"

        print(f"  Fold {fold} (run {run_id}, {status}):")
        print(f"    mean_fg_dice = {dice_str}")
        print(f"    val_loss     = {loss_str}")
        print(f"    ema_fg_dice  = {ema_str}")

        # Per-class Dice scores
        class_dice = {
            k: float(v) for k, v in metrics.items()
            if k.startswith("dice_class_")
        }
        if class_dice:
            for cls_key in sorted(class_dice, key=lambda k: int(k.split("_")[-1])):
                print(f"    {cls_key} = {class_dice[cls_key]:.4f}")

        # Training parameters
        trainer_class = params.get("trainer_class", "unknown")
        num_epochs = params.get("num_epochs", "?")
        print(f"    trainer: {trainer_class}, epochs: {num_epochs}")
        print()


def demonstrate_metric_history(client: Any, run_id: str) -> None:
    """Retrieve the full per-epoch metric history for a run.

    MlflowClient.get_metric_history() returns a list of Metric objects
    with (key, value, step, timestamp) for every logged step. This gives
    you the complete training curve, not just the final value.
    """
    print(f"  Fetching metric history for run {run_id[:8]}...")
    print()

    history = client.get_metric_history(run_id, "mean_fg_dice")

    if not history:
        print("  No mean_fg_dice history found for this run.")
        return

    print(f"  mean_fg_dice history ({len(history)} points):")
    # Show first 5 and last 5 points
    display_points = history[:5]
    if len(history) > 10:
        print("    (showing first 5 and last 5 epochs)")
        display_points = history[:5]
        for point in display_points:
            print(f"    epoch {point.step:>4d}: {point.value:.4f}")
        print("    ...")
        for point in history[-5:]:
            print(f"    epoch {point.step:>4d}: {point.value:.4f}")
    else:
        for point in history:
            print(f"    epoch {point.step:>4d}: {point.value:.4f}")
    print()


def demonstrate_querying_summary_runs(client: Any, experiment_id: str) -> None:
    """Query cv_summary runs created by log_cv_summary().

    Summary runs are tagged with nnunet_tracker.run_type = 'cv_summary'
    and contain aggregate metrics (cv_mean_fg_dice, cv_std_fg_dice, etc.)
    plus metadata tags about fold completeness.
    """
    print("  Querying summary runs...")

    summary_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.`nnunet_tracker.run_type` = 'cv_summary'",
        order_by=["start_time DESC"],
        max_results=10,
    )

    if not summary_runs:
        print("  No summary runs found.")
        print("  (Summary runs are created by log_cv_summary() or the CLI --log-to-mlflow flag.)")
        return

    print(f"  Found {len(summary_runs)} summary run(s):")
    print()

    for run in summary_runs:
        tags = run.data.tags
        metrics = run.data.metrics

        cv_group = tags.get("nnunet_tracker.cv_group", "unknown")
        is_complete = tags.get("nnunet_tracker.is_complete", "?")
        num_completed = tags.get("nnunet_tracker.num_completed", "?")
        completed_folds = tags.get("nnunet_tracker.completed_folds", "?")
        missing_folds = tags.get("nnunet_tracker.missing_folds", "?")

        print(f"  Summary for: {cv_group}")
        print(f"    Run ID:         {run.info.run_id[:8]}...")
        print(f"    Is complete:    {is_complete}")
        print(f"    Completed:      {num_completed} folds: {completed_folds}")
        print(f"    Missing:        {missing_folds}")

        mean_dice = metrics.get("cv_mean_fg_dice")
        std_dice = metrics.get("cv_std_fg_dice")
        if mean_dice is not None:
            std_str = f" +/- {std_dice:.4f}" if std_dice is not None else ""
            print(f"    Mean FG Dice:   {mean_dice:.4f}{std_str}")

        print()


def demonstrate_comparing_experiments(client: Any) -> None:
    """Compare aggregate metrics across multiple experiments.

    This pattern is useful when you want to compare different datasets,
    configurations, or training strategies side by side.
    """
    experiments = client.search_experiments()
    if len(experiments) < 2:
        print("  Need at least 2 experiments for comparison.")
        print("  Showing the comparison pattern instead:")
        print()
        print("    experiments_to_compare = [")
        print('        "Dataset001_BrainTumour",')
        print('        "Dataset002_Heart",')
        print('        "Dataset003_Liver",')
        print("    ]")
        print()
        print("    results = {}")
        print("    for exp_name in experiments_to_compare:")
        print("        exp = client.get_experiment_by_name(exp_name)")
        print("        if exp is None:")
        print("            continue")
        print("        summary_runs = client.search_runs(")
        print("            experiment_ids=[exp.experiment_id],")
        print('            filter_string="tags.`nnunet_tracker.run_type` = \'cv_summary\'",')
        print('            order_by=["start_time DESC"],')
        print("            max_results=1,")
        print("        )")
        print("        if summary_runs:")
        print('            dice = summary_runs[0].data.metrics.get("cv_mean_fg_dice")')
        print("            results[exp_name] = dice")
        print()
        print("    # Sort by Dice score descending")
        print("    for name, dice in sorted(results.items(), key=lambda x: x[1] or 0, reverse=True):")
        print('        print(f"  {name}: {dice:.4f}")')
        return

    print("  Comparing experiments:")
    print()

    comparison = {}
    for exp in experiments:
        # Skip the Default experiment
        if exp.name == "Default":
            continue

        summary_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`nnunet_tracker.run_type` = 'cv_summary'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if summary_runs:
            dice = summary_runs[0].data.metrics.get("cv_mean_fg_dice")
            comparison[exp.name] = dice
        else:
            # Fall back to averaging fold runs directly
            fold_runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=(
                    "tags.`nnunet_tracker.run_type` = 'fold' "
                    "and attributes.status = 'FINISHED'"
                ),
                max_results=10,
            )
            if fold_runs:
                dice_values = [
                    r.data.metrics.get("mean_fg_dice")
                    for r in fold_runs
                    if r.data.metrics.get("mean_fg_dice") is not None
                ]
                if dice_values:
                    import statistics
                    comparison[exp.name] = statistics.mean(dice_values)

    if not comparison:
        print("  No experiments with Dice metrics found.")
        return

    # Sort by Dice descending
    print(f"  {'Experiment':<40} {'Mean FG Dice':>12}")
    print(f"  {'-' * 40} {'-' * 12}")
    for name, dice in sorted(comparison.items(), key=lambda x: x[1] or 0, reverse=True):
        dice_str = f"{dice:.4f}" if dice is not None else "N/A"
        print(f"  {name:<40} {dice_str:>12}")
    print()


if __name__ == "__main__":
    from mlflow.tracking import MlflowClient

    print("=" * 70)
    print("nnunet-tracker: Direct MLflow Queries Example")
    print("=" * 70)
    print()

    # Initialize the MLflow client.
    # This does not create or modify any global state -- all operations
    # go through the client instance.
    tracking_uri = "./mlruns"
    client = MlflowClient(tracking_uri=tracking_uri)
    print(f"Connected to MLflow at: {tracking_uri}")
    print()

    # --- 1. Find experiments ---
    print("--- 1. Finding Experiments ---")
    print()
    experiment_id = demonstrate_finding_experiments(client)
    print()

    if experiment_id is None:
        print("No experiments found. Run some training first, then re-run this script.")
        print()
        print("To create sample data, use the nnunet-tracker CLI:")
        print("  nnunet-tracker train <dataset_id> <config> <fold>")
        print()
        print("Exiting.")
        exit(0)

    # --- 2. Query fold runs ---
    print("--- 2. Querying Fold Runs ---")
    print()
    fold_runs = demonstrate_querying_fold_runs(client, experiment_id)

    # --- 3. Extract metrics ---
    print("--- 3. Extracting Metrics ---")
    print()
    demonstrate_extracting_metrics(fold_runs)

    # --- 4. Metric history ---
    print("--- 4. Metric History ---")
    print()
    if fold_runs:
        demonstrate_metric_history(client, fold_runs[0].info.run_id)
    else:
        print("  No runs available for metric history demonstration.")
    print()

    # --- 5. Summary runs ---
    print("--- 5. Summary Runs ---")
    print()
    demonstrate_querying_summary_runs(client, experiment_id)

    # --- 6. Cross-experiment comparison ---
    print("--- 6. Cross-Experiment Comparison ---")
    print()
    demonstrate_comparing_experiments(client)

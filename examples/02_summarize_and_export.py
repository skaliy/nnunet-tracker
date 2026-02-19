"""
02_summarize_and_export.py -- Cross-validation summarization and publication export.

This script demonstrates how to use nnunet-tracker's summarization and export
APIs to aggregate metrics across cross-validation folds, inspect the results
programmatically, and produce publication-ready CSV and LaTeX output.

The key APIs shown are:

1. summarize_experiment()  -- Query MLflow for fold runs, build a CVSummary.
2. CVSummary properties    -- completed_folds, missing_folds, is_complete.
3. compute_aggregate_metrics() -- Mean/std across folds.
4. export_csv()            -- CSV string or file output.
5. export_latex()          -- LaTeX booktabs table string or file output.
6. log_cv_summary()        -- Write aggregate metrics back to MLflow.

Equivalent CLI usage:

    # Summarize and print table
    nnunet-tracker summarize --experiment Dataset001_BrainTumour

    # JSON output for scripting
    nnunet-tracker summarize --experiment Dataset001_BrainTumour --json

    # Also log the summary back to MLflow
    nnunet-tracker summarize --experiment Dataset001_BrainTumour --log-to-mlflow

    # Specify tracking URI and CV group
    nnunet-tracker summarize \\
        --experiment Dataset001_BrainTumour \\
        --tracking-uri http://localhost:5000 \\
        --cv-group "Dataset001|3d_fullres|nnUNetPlans"

NOTE: This script requires completed MLflow runs to query. If you run it
against an empty or nonexistent MLflow store, summarize_experiment() will
raise a ValueError. The script handles this gracefully for demonstration.

"""

from __future__ import annotations

import sys


def demonstrate_summarize_experiment() -> None:
    """Show how to query MLflow and build a CVSummary.

    summarize_experiment() connects to the MLflow tracking store, finds
    completed fold runs for the given experiment, deduplicates by fold
    (keeping the most recent run per fold), and returns a CVSummary object.

    Parameters:
        experiment_name  -- MLflow experiment name (required).
        tracking_uri     -- MLflow tracking URI (default: "./mlruns").
        cv_group         -- CV group identifier to filter by (default: None,
                            which auto-detects the most recent group).
        expected_folds   -- Tuple of expected fold numbers
                            (default: (0, 1, 2, 3, 4) for 5-fold CV).
    """
    from nnunet_tracker import summarize_experiment

    print("Querying MLflow for cross-validation results...")
    print()

    try:
        summary = summarize_experiment(
            experiment_name="Dataset001_BrainTumour",
            tracking_uri="./mlruns",
            # cv_group=None means auto-detect the latest cv_group
            # expected_folds defaults to (0, 1, 2, 3, 4)
        )
    except ValueError as e:
        # This is expected if no MLflow runs exist yet.
        print(f"  Could not summarize: {e}")
        print("  (This is expected if no training runs have been logged yet.)")
        print()
        print("  Falling back to a synthetic CVSummary for demonstration...")
        print()
        summary = _build_synthetic_summary()

    return summary


def demonstrate_cvsummary_properties(summary) -> None:
    """Inspect the CVSummary object returned by summarize_experiment().

    CVSummary provides several properties for checking completion status:

        completed_folds  -- Sorted list of fold numbers with FINISHED runs.
        missing_folds    -- Expected folds that do not have FINISHED runs.
        is_complete      -- True if all expected folds are present.
        cv_group         -- The dataset|config|plans identifier.
        experiment_name  -- The MLflow experiment name.
        fold_results     -- Dict mapping fold number to FoldResult objects.
    """
    print(f"  Experiment:      {summary.experiment_name}")
    print(f"  CV Group:        {summary.cv_group}")
    print(f"  Expected folds:  {list(summary.expected_folds)}")
    print(f"  Completed folds: {summary.completed_folds}")
    print(f"  Missing folds:   {summary.missing_folds}")
    print(f"  Is complete:     {summary.is_complete}")
    print()

    # Iterate over individual fold results
    print("  Per-fold results:")
    for fold_num in sorted(summary.fold_results):
        result = summary.fold_results[fold_num]
        dice_str = f"{result.mean_fg_dice:.4f}" if result.mean_fg_dice is not None else "N/A"
        loss_str = f"{result.val_loss:.4f}" if result.val_loss is not None else "N/A"
        print(f"    Fold {result.fold}: dice={dice_str}, val_loss={loss_str}, "
              f"run_id={result.run_id[:8]}...")
    print()


def demonstrate_aggregate_metrics(summary) -> None:
    """Compute and display aggregate metrics across folds.

    compute_aggregate_metrics() returns a dict with keys like:
        cv_mean_fg_dice, cv_std_fg_dice,
        cv_mean_val_loss, cv_std_val_loss,
        cv_mean_ema_fg_dice, cv_std_ema_fg_dice,
        cv_mean_dice_class_0, cv_std_dice_class_0, ...

    Standard deviation requires >= 2 folds; with a single fold only
    the mean is returned.
    """
    aggregates = summary.compute_aggregate_metrics()

    if not aggregates:
        print("  No aggregate metrics available.")
        return

    print("  Aggregate metrics:")
    for key, value in sorted(aggregates.items()):
        print(f"    {key}: {value:.4f}")
    print()

    # Common access pattern: get mean +/- std for reporting
    mean_dice = aggregates.get("cv_mean_fg_dice")
    std_dice = aggregates.get("cv_std_fg_dice")
    if mean_dice is not None:
        std_str = f" +/- {std_dice:.4f}" if std_dice is not None else ""
        print(f"  Summary: Mean FG Dice = {mean_dice:.4f}{std_str}")
    print()


def demonstrate_csv_export(summary) -> None:
    """Export cross-validation results as CSV.

    export_csv() returns the CSV as a string and optionally writes it to
    a file-like object. The CSV includes per-fold rows plus aggregate
    mean and std rows at the bottom.
    """
    from nnunet_tracker import export_csv

    # --- Get CSV as a string ---
    csv_string = export_csv(summary)
    print("  CSV output:")
    for line in csv_string.strip().split("\n"):
        print(f"    {line}")
    print()

    # --- Write CSV directly to a file ---
    # Uncomment the following to save to disk:
    #
    # with open("results.csv", "w", encoding="utf-8") as f:
    #     export_csv(summary, output=f)
    # print("  Saved to results.csv")

    print("  To save to file:")
    print('    with open("results.csv", "w", encoding="utf-8") as f:')
    print("        export_csv(summary, output=f)")
    print()


def demonstrate_latex_export(summary) -> None:
    """Export cross-validation results as a LaTeX booktabs table.

    export_latex() produces a complete table environment with:
      - \\toprule, \\midrule, \\bottomrule (requires booktabs package)
      - Per-fold rows with formatted Dice scores and losses
      - An aggregate row showing mean +/- std
      - Optional caption and label for cross-referencing

    Parameters:
        summary  -- CVSummary object.
        output   -- Optional file-like object to write to.
        caption  -- Table caption (default: auto-generated from cv_group).
        label    -- Table label for \\ref{} (default: "tab:cv_results").
    """
    from nnunet_tracker import export_latex

    # --- Basic LaTeX export ---
    latex_string = export_latex(summary)
    print("  LaTeX output (default caption/label):")
    for line in latex_string.strip().split("\n"):
        print(f"    {line}")
    print()

    # --- Custom caption and label ---
    latex_custom = export_latex(
        summary,
        caption="Cross-validation results for brain tumour segmentation",
        label="tab:brain_tumour_cv",
    )
    print("  LaTeX output (custom caption/label):")
    for line in latex_custom.strip().split("\n"):
        print(f"    {line}")
    print()

    # --- Write LaTeX to file ---
    # Uncomment to save:
    #
    # with open("results.tex", "w", encoding="utf-8") as f:
    #     export_latex(summary, output=f, caption="My Results", label="tab:results")

    print("  To save to file:")
    print('    with open("results.tex", "w", encoding="utf-8") as f:')
    print('        export_latex(summary, output=f, caption="My Results", label="tab:results")')
    print()


def demonstrate_log_cv_summary(summary) -> None:
    """Log aggregate metrics back to MLflow as a summary run.

    log_cv_summary() creates (or updates) an MLflow run tagged with
    nnunet_tracker.run_type = 'cv_summary'. It is idempotent: if a
    summary run already exists for the same cv_group, it updates that
    run rather than creating a duplicate.

    The summary run contains:
      - Aggregate metrics (cv_mean_fg_dice, cv_std_fg_dice, etc.)
      - Tags: completed_folds, missing_folds, is_complete, num_completed

    This uses the MlflowClient API internally, so it does not mutate
    any global MLflow state (no active run side effects).
    """
    from nnunet_tracker import log_cv_summary

    print("  log_cv_summary() creates an MLflow run with aggregate metrics.")
    print("  Usage:")
    print()
    print("    run_id = log_cv_summary(summary, tracking_uri='./mlruns')")
    print("    if run_id:")
    print("        print(f'Summary logged as run {run_id}')")
    print("    else:")
    print("        print('Failed to log summary (check MLflow connection)')")
    print()

    # Uncomment to actually log (requires a running MLflow store with the
    # experiment already created):
    #
    # run_id = log_cv_summary(summary, tracking_uri="./mlruns")
    # if run_id:
    #     print(f"  Summary run created: {run_id}")


def _build_synthetic_summary():
    """Build a synthetic CVSummary for demonstration when no MLflow runs exist.

    This creates realistic-looking fold results so the export functions
    can be demonstrated without requiring an actual MLflow tracking store.
    """
    from nnunet_tracker import CVSummary, FoldResult

    fold_results = {
        0: FoldResult(
            fold=0, run_id="a1b2c3d4e5f6a1b2", status="FINISHED",
            mean_fg_dice=0.8712, dice_per_class={0: 0.9103, 1: 0.8321},
            val_loss=0.3521, ema_fg_dice=0.8698,
        ),
        1: FoldResult(
            fold=1, run_id="b2c3d4e5f6a1b2c3", status="FINISHED",
            mean_fg_dice=0.8845, dice_per_class={0: 0.9211, 1: 0.8479},
            val_loss=0.3412, ema_fg_dice=0.8830,
        ),
        2: FoldResult(
            fold=2, run_id="c3d4e5f6a1b2c3d4", status="FINISHED",
            mean_fg_dice=0.8653, dice_per_class={0: 0.9045, 1: 0.8261},
            val_loss=0.3598, ema_fg_dice=0.8641,
        ),
        3: FoldResult(
            fold=3, run_id="d4e5f6a1b2c3d4e5", status="FINISHED",
            mean_fg_dice=0.8791, dice_per_class={0: 0.9156, 1: 0.8426},
            val_loss=0.3467, ema_fg_dice=0.8779,
        ),
        4: FoldResult(
            fold=4, run_id="e5f6a1b2c3d4e5f6", status="FINISHED",
            mean_fg_dice=0.8734, dice_per_class={0: 0.9089, 1: 0.8379},
            val_loss=0.3544, ema_fg_dice=0.8720,
        ),
    }

    return CVSummary(
        cv_group="Dataset001|3d_fullres|nnUNetPlans",
        experiment_name="Dataset001_BrainTumour",
        fold_results=fold_results,
        expected_folds=(0, 1, 2, 3, 4),
    )


if __name__ == "__main__":
    print("=" * 70)
    print("nnunet-tracker: Summarization and Export Example")
    print("=" * 70)
    print()

    print("--- 1. Summarize Experiment ---")
    print()
    summary = demonstrate_summarize_experiment()

    print("--- 2. CVSummary Properties ---")
    print()
    demonstrate_cvsummary_properties(summary)

    print("--- 3. Aggregate Metrics ---")
    print()
    demonstrate_aggregate_metrics(summary)

    print("--- 4. CSV Export ---")
    print()
    demonstrate_csv_export(summary)

    print("--- 5. LaTeX Export ---")
    print()
    demonstrate_latex_export(summary)

    print("--- 6. Log Summary to MLflow ---")
    print()
    demonstrate_log_cv_summary(summary)

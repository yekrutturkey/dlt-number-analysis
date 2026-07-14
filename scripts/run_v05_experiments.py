"""Run selected v0.5 baselines on a strict temporal phase and refresh reports."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_prize_rule_schedule, load_verified_history
from dlt_number_analysis.experiments import (
    ablation_experiment_specs,
    baseline_experiment_specs,
    compare_to_constraint_matched_baseline,
    run_experiment_batch,
    write_ablation_results_csv,
    write_experiment_summary_report,
    write_holdout_results_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _legacy_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draws",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "draws.csv",
    )
    parser.add_argument(
        "--experiment-ids",
        nargs="+",
        default=["B0", "B7", "B8"],
    )
    parser.add_argument(
        "--phase",
        choices=("development", "calibration", "final_holdout"),
        default="development",
    )
    parser.add_argument("--seed", type=int, default=20260000)
    parser.add_argument("--minimum-history", type=int, default=100)
    parser.add_argument("--profile", choices=("fast", "standard", "final"), default="fast")
    parser.add_argument(
        "--reuse-observations",
        action="store_true",
        help="refresh reports from the existing observations CSV without rerunning strategies",
    )
    parser.add_argument(
        "--observations-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "backtests" / "v05_observations.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "experiment_summary.md",
    )
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "ablation_results.csv",
    )
    parser.add_argument(
        "--holdout-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "holdout_results.md",
    )
    parser.add_argument(
        "--holdout-lock",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "backtests" / "holdout_lock.json",
    )
    return parser


def _pending_ablation_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for spec in ablation_experiment_specs():
        parameters = spec.scorer_spec.parameters
        records.append(
            {
                "experiment_id": spec.experiment_id,
                "experiment_version": spec.experiment_version,
                "window": parameters["window"],
                "decay": parameters["decay"],
                "structure_score_weight": spec.structure_score_weight,
                "target_front_pool_size": (spec.portfolio_constraints.target_front_pool_size),
                "core_number_count": spec.portfolio_constraints.min_core_front_numbers,
                "execution_status": "not_executed",
                "result_note": "configuration registered; no historical result claimed",
            }
        )
    return pd.DataFrame.from_records(records)


def _legacy_main(argv: list[str] | None = None) -> int:
    args = _legacy_build_parser().parse_args(argv)
    specifications = baseline_experiment_specs(seeds=(args.seed,), phase=args.phase)
    selected = tuple(spec for spec in specifications if spec.experiment_id in args.experiment_ids)
    missing = set(args.experiment_ids).difference(spec.experiment_id for spec in selected)
    if missing:
        raise ValueError(f"unknown experiment ids: {sorted(missing)}")
    draws = load_verified_history(args.draws)
    schedule = load_prize_rule_schedule(PROJECT_ROOT / "config" / "prize_tiers.json")
    options: dict[str, object] = {
        "profile": args.profile,
        "minimum_history": args.minimum_history,
        "prize_tables": schedule.tables,
    }
    if args.phase == "final_holdout":
        options["holdout_lock_path"] = args.holdout_lock
    result = None if args.reuse_observations else run_experiment_batch(draws, selected, **options)
    observations = (
        pd.read_csv(
            args.observations_output,
            dtype={"target_issue": "string", "data_cutoff_issue": "string"},
        )
        if result is None
        else result.observations
    )
    if result is None:
        issues = draws["issue"].astype(str).tolist()
        previous_issue = {issues[index]: issues[index - 1] for index in range(1, len(issues))}
        observations["data_cutoff_issue"] = observations["target_issue"].map(previous_issue)
        if observations["data_cutoff_issue"].isna().any():
            raise ValueError("reused observations contain a target outside verified history")
        specs_by_id = {spec.experiment_id: spec for spec in specifications}
        observations["portfolio_strategy"] = observations["experiment_id"].map(
            {name: spec.portfolio_strategy for name, spec in specs_by_id.items()}
        )
        observations["candidate_generation_method"] = observations["experiment_id"].map(
            {name: spec.candidate_generation_method for name, spec in specs_by_id.items()}
        )
    observations["risk_disclaimer"] = DISCLAIMER
    args.observations_output.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(
        args.observations_output,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    comparisons = compare_to_constraint_matched_baseline(observations)
    write_experiment_summary_report(observations, comparisons, args.summary_output)
    write_ablation_results_csv(_pending_ablation_rows(), args.ablation_output)
    write_holdout_results_report(observations, args.holdout_output)
    if result is None:
        print(f"reused observations: {len(observations)}")
    else:
        for execution in result.executions:
            print(
                f"{execution.experiment_id}: periods={execution.period_count}; "
                f"runtime={execution.runtime_seconds:.3f}s"
            )
    print(DISCLAIMER)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Delegate to v0.5.1 partitioned, resumable, ProcessPool execution."""
    from dlt_number_analysis.experiments.command import main as run_partitioned

    return run_partitioned(argv)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface for validation, backtesting, generation, and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DrawRecord,
    generate_data_quality_report,
    load_draws_csv,
    load_issue_prizes,
    load_prize_rule_schedule,
)
from dlt_number_analysis.evaluation import apply_issue_prize_record, evaluate_prediction
from dlt_number_analysis.models import load_prediction_record
from dlt_number_analysis.pipeline import (
    NextPredictionArtifact,
    PipelineConfig,
    PipelineSeeds,
    PredictionPipeline,
)
from dlt_number_analysis.scoring import ScorerSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAWS = PROJECT_ROOT / "data" / "raw" / "draws.csv"
DEFAULT_PRIZE_RULES = PROJECT_ROOT / "config" / "prize_tiers.json"


def _write_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump_json"):
        text = payload.model_dump_json(indent=2)
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")
    return path


def _git_commit_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _next_issue(cutoff_issue: str) -> str:
    return str(int(cutoff_issue) + 1).zfill(len(cutoff_issue))


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        scorer_spec=ScorerSpec(
            name=args.scorer,
            parameters=(
                {"window": args.window, "decay": args.decay}
                if args.scorer == "recency_weighted_frequency_score"
                else {}
            ),
        ),
        optimizer_search_trials=args.search_trials,
    )


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scorer",
        choices=(
            "uniform_score",
            "cumulative_frequency_score",
            "recency_weighted_frequency_score",
        ),
        default="recency_weighted_frequency_score",
    )
    parser.add_argument("--window", type=int, choices=(10, 30, 100), default=30)
    parser.add_argument("--decay", type=float, default=0.93)
    parser.add_argument("--search-trials", type=int, default=25_000)


def _validate_data(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    report = generate_data_quality_report(draws)
    target = _write_json(report, args.output)
    print(f"data quality report: {target}")
    print(f"valid={report.is_valid}; blocks_backtest={report.blocks_backtest}")
    return 0 if report.is_valid else 2


def _run_backtest(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    schedule = load_prize_rule_schedule(args.prize_rules)
    issue_prizes = load_issue_prizes(args.issue_prizes) if args.issue_prizes else {}
    pipelines = (
        {"optimized_portfolio_strategy": _pipeline_config(args)} if args.include_optimized else None
    )
    report = run_rolling_backtest(
        draws,
        prize_tables=schedule.tables,
        pipeline_configs=pipelines,
        min_history=args.min_history,
        base_random_seed=args.random_seed,
        random_baseline_seed_count=args.baseline_seeds,
        bootstrap_resamples=args.bootstrap_resamples,
        issue_prize_records=issue_prizes,
    )
    target = _write_json(report, args.output)
    print(f"backtest report: {target}")
    print(DISCLAIMER)
    return 0


def _generate_next(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    cutoff_issue = str(draws.iloc[-1]["issue"])
    target_issue = args.target_issue or _next_issue(cutoff_issue)
    generated_at = datetime.now(UTC)
    config = _pipeline_config(args)
    seeds = PipelineSeeds.from_base_seed(args.random_seed)
    result = PredictionPipeline(config).run(
        draws,
        target_issue=target_issue,
        generated_at=generated_at,
        random_seeds=seeds,
    )
    artifact = NextPredictionArtifact(
        target_issue=target_issue,
        data_cutoff_issue=cutoff_issue,
        generated_at=generated_at,
        git_commit_sha=_git_commit_sha(),
        pipeline_config=config,
        random_seeds=seeds,
        candidate_pool_summary=result.candidate_pool_summary,
        prediction=result.prediction,
    )
    target = args.output or PROJECT_ROOT / "outputs" / "predictions" / f"{target_issue}.json"
    _write_json(artifact, target)
    print(f"next prediction artifact: {target}")
    print(DISCLAIMER)
    return 0


def _evaluate_latest(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    prediction = load_prediction_record(args.prediction)
    matching = draws.loc[draws["issue"].astype(str) == prediction.target_issue]
    if matching.empty:
        raise ValueError(f"draw history does not contain target issue {prediction.target_issue}")
    row = matching.iloc[-1]
    actual_draw = DrawRecord.model_validate({column: row[column] for column in CSV_COLUMNS})
    schedule = load_prize_rule_schedule(args.prize_rules)
    table = schedule.table_for_issue(prediction.target_issue)
    if table is None:
        raise ValueError("no prize rule applies to the prediction target issue")
    prize_context = str(prediction.parameters.get("prize_context", table.default_context))
    review = evaluate_prediction(
        prediction,
        actual_draw,
        table,
        prize_context=prize_context,
    )
    if args.issue_prizes:
        record = load_issue_prizes(args.issue_prizes).get(prediction.target_issue)
        if record is not None:
            review = apply_issue_prize_record(review, table, record)
    target = _write_json(review, args.output)
    print(f"latest evaluation: {target}")
    print(DISCLAIMER)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dlt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-data")
    validate.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    validate.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "data_quality.json",
    )
    validate.set_defaults(handler=_validate_data)

    backtest = subparsers.add_parser("run-backtest")
    backtest.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    backtest.add_argument("--prize-rules", type=Path, default=DEFAULT_PRIZE_RULES)
    backtest.add_argument("--issue-prizes", type=Path)
    backtest.add_argument("--min-history", type=int, default=30)
    backtest.add_argument("--random-seed", type=int, default=20260000)
    backtest.add_argument("--baseline-seeds", type=int, default=1000)
    backtest.add_argument("--bootstrap-resamples", type=int, default=1000)
    backtest.add_argument("--include-optimized", action="store_true")
    backtest.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "backtests" / "latest.json",
    )
    _add_pipeline_arguments(backtest)
    backtest.set_defaults(handler=_run_backtest)

    generate = subparsers.add_parser("generate-next")
    generate.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    generate.add_argument("--target-issue")
    generate.add_argument("--random-seed", type=int, default=20260714)
    generate.add_argument("--output", type=Path)
    _add_pipeline_arguments(generate)
    generate.set_defaults(handler=_generate_next)

    evaluate = subparsers.add_parser("evaluate-latest")
    evaluate.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    evaluate.add_argument("--prize-rules", type=Path, default=DEFAULT_PRIZE_RULES)
    evaluate.add_argument("--issue-prizes", type=Path)
    evaluate.add_argument(
        "--prediction",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "predictions" / "26078_manual_chat.json",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "latest_evaluation.json",
    )
    evaluate.set_defaults(handler=_evaluate_latest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())

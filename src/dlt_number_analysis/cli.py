"""Command-line interface for validation, backtesting, generation, and evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.backtesting import run_rolling_backtest
from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DrawRecord,
    generate_data_quality_report,
    load_draws_csv,
    load_issue_prizes,
    load_prize_rule_schedule,
    load_verified_history,
    verified_history_prefix_check,
)
from dlt_number_analysis.evaluation import apply_issue_prize_record, evaluate_prediction
from dlt_number_analysis.models import PredictionRecord, load_prediction_record
from dlt_number_analysis.pipeline import (
    NextPredictionArtifact,
    PipelineConfig,
    PipelineSeeds,
    PredictionPipeline,
    build_history_audit,
    collect_git_audit,
    load_next_prediction_artifact,
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


def _pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    scorer_parameters: dict[str, int | float] = {}
    if args.scorer == "recency_weighted_frequency_score":
        scorer_parameters = {"window": args.window, "decay": args.decay}
    elif args.scorer == "hot_cold_blend_score":
        scorer_parameters = {
            "hot_window": args.hot_window,
            "cold_window": args.cold_window,
            "hot_weight": args.hot_weight,
            "decay": args.decay,
        }
    values: dict[str, Any] = {
        "profile": args.profile,
        "scorer_spec": ScorerSpec(name=args.scorer, parameters=scorer_parameters),
        "minimum_history_size": args.minimum_history_size,
    }
    optional_overrides = {
        "candidate_count": args.candidate_count,
        "optimizer_search_trials": args.search_trials,
        "parallel_workers": args.parallel_workers,
    }
    values.update({name: value for name, value in optional_overrides.items() if value is not None})
    return PipelineConfig(**values)


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scorer",
        choices=(
            "uniform_score",
            "cumulative_frequency_score",
            "recency_weighted_frequency_score",
            "hot_cold_blend_score",
        ),
        default="recency_weighted_frequency_score",
    )
    parser.add_argument("--window", type=int, choices=(10, 30, 100), default=30)
    parser.add_argument("--hot-window", type=int, choices=(10, 30, 100), default=10)
    parser.add_argument("--cold-window", type=int, choices=(10, 30, 100), default=100)
    parser.add_argument("--hot-weight", type=float, default=0.65)
    parser.add_argument("--decay", type=float, default=0.93)
    parser.add_argument("--profile", choices=("fast", "standard", "final"), default="standard")
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--search-trials", type=int)
    parser.add_argument("--parallel-workers", type=int)
    parser.add_argument("--minimum-history-size", type=int, default=100)
    parser.add_argument("--allow-short-history", action="store_true")


def _validate_data(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    report = generate_data_quality_report(draws)
    target = _write_json(report, args.output)
    print(f"data quality report: {target}")
    print(f"valid={report.is_valid}; blocks_backtest={report.blocks_backtest}")
    return 0 if report.is_valid else 2


def _run_backtest(args: argparse.Namespace) -> int:
    draws = (
        load_draws_csv(args.draws)
        if args.allow_unverified_history
        else load_verified_history(args.draws)
    )
    schedule = load_prize_rule_schedule(args.prize_rules)
    issue_prizes = load_issue_prizes(args.issue_prizes) if args.issue_prizes else {}
    pipeline_config = _pipeline_config(args) if args.include_optimized else None
    pipelines = (
        {"optimized_portfolio_strategy": pipeline_config} if pipeline_config is not None else None
    )
    minimum_history = args.min_history
    if pipeline_config is not None and not args.allow_short_history:
        minimum_history = max(minimum_history, pipeline_config.minimum_history_size)
    report = run_rolling_backtest(
        draws,
        prize_tables=schedule.tables,
        pipeline_configs=pipelines,
        min_history=minimum_history,
        base_random_seed=args.random_seed,
        random_baseline_seed_count=args.baseline_seeds,
        bootstrap_resamples=args.bootstrap_resamples,
        issue_prize_records=issue_prizes,
        allow_short_history=args.allow_short_history,
    )
    target = _write_json(report, args.output)
    print(f"backtest report: {target}")
    print(DISCLAIMER)
    return 0


def _generate_next(args: argparse.Namespace) -> int:
    history_verified = not args.allow_unverified_history
    draws = load_verified_history(args.draws) if history_verified else load_draws_csv(args.draws)
    cutoff_issue = str(draws.iloc[-1]["issue"])
    target_issue = str(args.target_issue)
    git_audit = collect_git_audit(PROJECT_ROOT)
    if git_audit.dirty and not args.allow_dirty:
        raise ValueError(
            "formal generate-next refuses a dirty Git working tree; "
            "pass --allow-dirty only for an explicit test or research override"
        )
    history_audit = build_history_audit(args.draws, draws)
    generated_at = datetime.now(UTC)
    config = _pipeline_config(args)
    seeds = PipelineSeeds.from_base_seed(args.random_seed)
    result = PredictionPipeline(config).run(
        draws,
        target_issue=target_issue,
        generated_at=generated_at,
        random_seeds=seeds,
        allow_short_history=args.allow_short_history,
    )
    prediction = result.prediction.model_copy(
        update={
            "parameters": {
                **result.prediction.parameters,
                "history_verified": history_verified,
                "unverified_history_override": args.allow_unverified_history,
            }
        }
    )
    artifact = NextPredictionArtifact(
        target_issue=target_issue,
        data_cutoff_issue=cutoff_issue,
        generated_at=generated_at,
        git_commit_sha=git_audit.commit_sha,
        git_dirty=git_audit.dirty,
        git_diff_hash=git_audit.diff_hash,
        raw_file_sha256=history_audit.raw_file_sha256,
        canonical_history_sha256=history_audit.canonical_history_sha256,
        history_record_count=history_audit.record_count,
        history_start_issue=history_audit.start_issue,
        history_cutoff_issue=history_audit.cutoff_issue,
        history_verified=history_verified,
        unverified_history_override=args.allow_unverified_history,
        short_history_override=result.short_history_override,
        pipeline_config=config,
        random_seeds=seeds,
        candidate_pool_summary=result.candidate_pool_summary,
        prediction=prediction,
    )
    target = args.output or PROJECT_ROOT / "outputs" / "predictions" / f"{target_issue}.json"
    _write_json(artifact, target)
    print(f"next prediction artifact: {target}")
    print(DISCLAIMER)
    return 0


def _load_prediction_input(path: Path, draws: pd.DataFrame) -> PredictionRecord:
    """Accept either a bare prediction log or a complete next-prediction artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "prediction" in payload:
        artifact = load_next_prediction_artifact(path)
        findings = verified_history_prefix_check(
            draws,
            cutoff_issue=artifact.data_cutoff_issue,
            expected_canonical_sha256=artifact.canonical_history_sha256,
            expected_record_count=artifact.history_record_count,
        )
        if findings:
            raise ValueError("history was modified before the artifact data cutoff")
        return artifact.prediction
    return load_prediction_record(path)


def _evaluate_latest(args: argparse.Namespace) -> int:
    draws = load_draws_csv(args.draws)
    prediction = _load_prediction_input(args.prediction, draws)
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
        "--allow-unverified-history",
        action="store_true",
        help="research-only override; formal backtests require a verified history manifest",
    )
    backtest.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "backtests" / "latest.json",
    )
    _add_pipeline_arguments(backtest)
    backtest.set_defaults(handler=_run_backtest)

    generate = subparsers.add_parser("generate-next")
    generate.add_argument("--draws", type=Path, default=DEFAULT_DRAWS)
    generate.add_argument("--target-issue", required=True)
    generate.add_argument("--random-seed", type=int, default=20260714)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--allow-dirty", action="store_true")
    generate.add_argument(
        "--allow-unverified-history",
        action="store_true",
        help="research-only override; formal generation requires a verified history manifest",
    )
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

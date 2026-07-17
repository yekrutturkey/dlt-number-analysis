"""Strict expanding-window backtests with a cached random five-ticket baseline."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from functools import lru_cache
from random import Random
from statistics import fmean, median, stdev
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import JsonValue

from dlt_number_analysis.backtesting.models import (
    BacktestPeriodResult,
    BacktestReport,
    BacktestStrategySummary,
    ConfidenceInterval,
    RandomBaselineSummary,
    RandomMetricSummary,
    RawObservationResult,
)
from dlt_number_analysis.data import (
    CSV_COLUMNS,
    DataQualityReport,
    DrawRecord,
    assert_backtest_ready,
    generate_data_quality_report,
    validate_draw_dataframe,
)
from dlt_number_analysis.evaluation import (
    IssuePrizeRecord,
    PrizeRuleSchedule,
    PrizeTable,
    apply_issue_prize_record,
    evaluate_prediction,
)
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.pipeline import PipelineConfig, optimized_portfolio_strategy
from dlt_number_analysis.scoring import NumberScorer, cumulative_frequency_score
from dlt_number_analysis.strategies import StrategyFunction, random_baseline

RANDOM_METRICS: tuple[str, ...] = (
    "best_front_hits",
    "best_back_hits",
    "best_total_hits",
    "at_least_three_front",
    "at_least_2_plus_1",
    "front_pool_coverage",
    "back_pool_coverage",
    "unique_hit_concentration",
)


class MissingPrizeAmountError(ValueError):
    """Deprecated: missing historical amounts now preserve hits and disable ROI."""


@dataclass(frozen=True, slots=True)
class _HitMetrics:
    best_front_hits: int
    best_back_hits: int
    best_total_hits: int
    at_least_three_front: bool
    at_least_2_plus_1: bool
    ticket_hit_share: float
    unique_hit_concentration: float
    front_pool_coverage: int
    back_pool_coverage: int


def _draw_from_row(row: pd.Series) -> DrawRecord:
    return DrawRecord.model_validate({column: row[column] for column in CSV_COLUMNS})


def _historical_generation_time(cutoff_draw: DrawRecord) -> datetime:
    return datetime.combine(
        cutoff_draw.draw_date,
        time(23, 59),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )


def _calculate_hit_metrics(
    tickets: Sequence[tuple[set[int], set[int]]],
    actual_front: set[int],
    actual_back: set[int],
) -> _HitMetrics:
    hit_pairs = [
        (len(front.intersection(actual_front)), len(back.intersection(actual_back)))
        for front, back in tickets
    ]
    totals = [front + back for front, back in hit_pairs]
    all_ticket_hits = sum(totals)
    front_pool = set().union(*(front for front, _ in tickets))
    back_pool = set().union(*(back for _, back in tickets))
    front_pool_coverage = len(front_pool.intersection(actual_front))
    back_pool_coverage = len(back_pool.intersection(actual_back))
    unique_pool_hits = front_pool_coverage + back_pool_coverage
    best_total = max(totals)
    return _HitMetrics(
        best_front_hits=max(front for front, _ in hit_pairs),
        best_back_hits=max(back for _, back in hit_pairs),
        best_total_hits=best_total,
        at_least_three_front=any(front >= 3 for front, _ in hit_pairs),
        at_least_2_plus_1=any(front >= 2 and back >= 1 for front, back in hit_pairs),
        ticket_hit_share=(0.0 if all_ticket_hits == 0 else best_total / all_ticket_hits),
        unique_hit_concentration=(0.0 if unique_pool_hits == 0 else best_total / unique_pool_hits),
        front_pool_coverage=front_pool_coverage,
        back_pool_coverage=back_pool_coverage,
    )


def _hit_metrics(prediction: PredictionRecord, actual_draw: DrawRecord) -> _HitMetrics:
    return _calculate_hit_metrics(
        [(set(ticket.front_numbers), set(ticket.back_numbers)) for ticket in prediction.tickets],
        set(actual_draw.front_numbers),
        set(actual_draw.back_numbers),
    )


def calculate_raw_hit_metrics(
    prediction: PredictionRecord,
    actual_draw: DrawRecord,
) -> dict[str, int | bool | float]:
    """Return hit-only metrics without Monte Carlo or historical bootstrap work."""
    hits = _hit_metrics(prediction, actual_draw)
    return {
        "best_front_hits": hits.best_front_hits,
        "best_back_hits": hits.best_back_hits,
        "best_total_hits": hits.best_total_hits,
        "at_least_three_front": hits.at_least_three_front,
        "at_least_2_plus_1": hits.at_least_2_plus_1,
        "ticket_hit_share": hits.ticket_hit_share,
        "unique_hit_concentration": hits.unique_hit_concentration,
        "front_pool_coverage": hits.front_pool_coverage,
        "back_pool_coverage": hits.back_pool_coverage,
    }


def evaluate_prediction_raw_observation(
    prediction: PredictionRecord,
    actual_draw: DrawRecord,
    prize_table: PrizeTable | None = None,
    *,
    prize_context_by_issue: Mapping[str, str] | None = None,
    prize_tables: Sequence[PrizeTable] | None = None,
    issue_prize_records: Mapping[str, IssuePrizeRecord] | None = None,
    default_ticket_cost: Decimal = Decimal("2"),
) -> RawObservationResult:
    """Evaluate one saved prediction directly, with no resampling side effects."""
    if prediction.target_issue != actual_draw.issue:
        raise ValueError("prediction target issue differs from actual draw")
    if int(prediction.data_cutoff_issue) >= int(actual_draw.issue):
        raise ValueError("prediction cutoff must precede the actual draw")
    if default_ticket_cost <= 0:
        raise ValueError("default_ticket_cost must be positive")
    hits = _hit_metrics(prediction, actual_draw)
    (
        any_prize,
        total_cost,
        total_prize,
        roi,
        prize_rule_version,
        prize_data_available,
    ) = _evaluate_monetary_result(
        prediction,
        actual_draw,
        _build_schedule(prize_table, prize_tables),
        prize_contexts=dict(prize_context_by_issue or {}),
        issue_prize_records=dict(issue_prize_records or {}),
        default_ticket_cost=default_ticket_cost,
    )
    return RawObservationResult(
        target_issue=actual_draw.issue,
        data_cutoff_issue=prediction.data_cutoff_issue,
        strategy_name=prediction.strategy_name,
        random_seed=prediction.random_seed,
        best_front_hits=hits.best_front_hits,
        best_back_hits=hits.best_back_hits,
        best_total_hits=hits.best_total_hits,
        at_least_three_front=hits.at_least_three_front,
        at_least_2_plus_1=hits.at_least_2_plus_1,
        ticket_hit_share=hits.ticket_hit_share,
        unique_hit_concentration=hits.unique_hit_concentration,
        front_pool_coverage=hits.front_pool_coverage,
        back_pool_coverage=hits.back_pool_coverage,
        total_cost=total_cost,
        total_prize=total_prize,
        roi=roi,
        any_prize=any_prize,
        prize_rule_version=prize_rule_version,
        prize_data_available=prize_data_available,
    )


def _random_hit_metrics(random_seed: int) -> _HitMetrics:
    rng = Random(random_seed)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    tickets: list[tuple[set[int], set[int]]] = []
    while len(tickets) < 5:
        front = tuple(sorted(rng.sample(range(1, 36), 5)))
        back = tuple(sorted(rng.sample(range(1, 13), 2)))
        if (front, back) in seen:
            continue
        seen.add((front, back))
        tickets.append((set(front), set(back)))
    # Lottery-number symmetry makes one canonical actual draw valid for every issue.
    return _calculate_hit_metrics(tickets, set(range(1, 6)), set(range(1, 3)))


@lru_cache(maxsize=64)
def _cached_random_distribution(
    seed_start: int,
    seed_count: int,
) -> tuple[_HitMetrics, ...]:
    return tuple(_random_hit_metrics(seed_start + offset) for offset in range(seed_count))


def clear_random_baseline_cache() -> None:
    """Clear the in-process Monte Carlo cache (primarily for deterministic tests)."""
    _cached_random_distribution.cache_clear()


def random_baseline_cache_info() -> object:
    """Expose cache statistics without exposing mutable cached samples."""
    return _cached_random_distribution.cache_info()


def _metric_values(metrics: _HitMetrics) -> dict[str, float]:
    return {
        "best_front_hits": float(metrics.best_front_hits),
        "best_back_hits": float(metrics.best_back_hits),
        "best_total_hits": float(metrics.best_total_hits),
        "at_least_three_front": float(metrics.at_least_three_front),
        "at_least_2_plus_1": float(metrics.at_least_2_plus_1),
        "front_pool_coverage": float(metrics.front_pool_coverage),
        "back_pool_coverage": float(metrics.back_pool_coverage),
        "unique_hit_concentration": metrics.unique_hit_concentration,
        "ticket_hit_share": metrics.ticket_hit_share,
    }


def _monte_carlo_summaries(
    distribution: Sequence[_HitMetrics],
) -> dict[str, RandomMetricSummary]:
    summaries: dict[str, RandomMetricSummary] = {}
    for metric in RANDOM_METRICS:
        values = [_metric_values(sample)[metric] for sample in distribution]
        mean = fmean(values)
        standard_error = 0.0 if len(values) < 2 else stdev(values) / len(values) ** 0.5
        summaries[metric] = RandomMetricSummary(
            mean=mean,
            standard_error=standard_error,
            monte_carlo_error_interval=ConfidenceInterval(
                method="monte_carlo_normal_error_interval",
                lower=mean - 1.96 * standard_error,
                upper=mean + 1.96 * standard_error,
            ),
        )
    return summaries


def _percentile_in_distribution(value: float, distribution: Sequence[float]) -> float:
    lower = sum(sample < value for sample in distribution)
    equal = sum(sample == value for sample in distribution)
    return 100.0 * (lower + 0.5 * equal) / len(distribution)


def _build_schedule(
    prize_table: PrizeTable | None,
    prize_tables: Sequence[PrizeTable] | None,
) -> PrizeRuleSchedule | None:
    tables = list(prize_tables or ())
    if prize_table is not None and all(table.version != prize_table.version for table in tables):
        tables.append(prize_table)
    return None if not tables else PrizeRuleSchedule(tables=tuple(tables))


def _evaluate_monetary_result(
    prediction: PredictionRecord,
    target_draw: DrawRecord,
    schedule: PrizeRuleSchedule | None,
    *,
    prize_contexts: Mapping[str, str],
    issue_prize_records: Mapping[str, IssuePrizeRecord],
    default_ticket_cost: Decimal,
) -> tuple[bool | None, Decimal, Decimal | None, Decimal | None, str | None, bool]:
    table = None if schedule is None else schedule.table_for_issue(target_draw.issue)
    if table is None:
        return None, default_ticket_cost * len(prediction.tickets), None, None, None, False
    record = issue_prize_records.get(target_draw.issue)
    context_is_known = (
        record is not None or target_draw.issue in prize_contexts or len(table.contexts) == 1
    )
    context = (
        record.prize_context
        if record is not None
        else prize_contexts.get(target_draw.issue, table.default_context)
    )
    review = evaluate_prediction(prediction, target_draw, table, prize_context=context)
    if record is not None:
        review = apply_issue_prize_record(review, table, record)
    if not context_is_known and review.any_prize:
        return review.any_prize, review.total_cost, None, None, table.version, False
    available = review.total_prize is not None and review.roi is not None
    return (
        review.any_prize,
        review.total_cost,
        review.total_prize,
        review.roi,
        table.version,
        available,
    )


def _longest_no_prize_streak(results: Sequence[BacktestPeriodResult]) -> int:
    longest = 0
    current = 0
    for result in results:
        if result.any_prize:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _historical_bootstrap_intervals(
    results: Sequence[BacktestPeriodResult],
    *,
    random_seed: int,
    resamples: int,
) -> dict[str, ConfidenceInterval]:
    metric_rows = [
        {
            "best_front_hits": float(result.best_front_hits),
            "best_back_hits": float(result.best_back_hits),
            "best_total_hits": float(result.best_total_hits),
            "at_least_three_front": float(result.at_least_three_front),
            "at_least_2_plus_1": float(result.at_least_2_plus_1),
            "front_pool_coverage": float(result.front_pool_coverage),
            "back_pool_coverage": float(result.back_pool_coverage),
            "unique_hit_concentration": result.unique_hit_concentration,
            "ticket_hit_share": result.ticket_hit_share,
        }
        for result in results
    ]
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(results), size=(resamples, len(results)))
    intervals: dict[str, ConfidenceInterval] = {}
    for metric in metric_rows[0]:
        values = np.asarray([row[metric] for row in metric_rows], dtype=float)
        means = values[indices].mean(axis=1)
        lower, upper = np.quantile(means, (0.025, 0.975), method="linear")
        intervals[metric] = ConfidenceInterval(
            method="cross_historical_period_bootstrap",
            lower=float(lower),
            upper=float(upper),
        )
    return intervals


def _strategy_summaries(
    results: Sequence[BacktestPeriodResult],
    *,
    random_seed: int,
    bootstrap_resamples: int,
) -> tuple[BacktestStrategySummary, ...]:
    grouped: dict[str, list[BacktestPeriodResult]] = defaultdict(list)
    for result in results:
        grouped[result.strategy_name].append(result)
    summaries: list[BacktestStrategySummary] = []
    for offset, (strategy_name, strategy_results) in enumerate(grouped.items()):
        count = len(strategy_results)
        monetary_complete = all(
            result.prize_data_available and result.total_prize is not None
            for result in strategy_results
        )
        prizes = [
            result.total_prize for result in strategy_results if result.total_prize is not None
        ]
        prize_flags_known = all(result.any_prize is not None for result in strategy_results)
        summaries.append(
            BacktestStrategySummary(
                strategy_name=strategy_name,
                period_count=count,
                at_least_three_front_rate=(
                    sum(result.at_least_three_front for result in strategy_results) / count
                ),
                at_least_2_plus_1_rate=(
                    sum(result.at_least_2_plus_1 for result in strategy_results) / count
                ),
                ticket_hit_share=fmean(result.ticket_hit_share for result in strategy_results),
                unique_hit_concentration=fmean(
                    result.unique_hit_concentration for result in strategy_results
                ),
                average_prize=(sum(prizes, Decimal("0")) / count if monetary_complete else None),
                median_prize=median(prizes) if monetary_complete else None,
                longest_no_prize_streak=(
                    _longest_no_prize_streak(strategy_results) if prize_flags_known else None
                ),
                monetary_metrics_complete=monetary_complete,
                bootstrap_performance_intervals=_historical_bootstrap_intervals(
                    strategy_results,
                    random_seed=random_seed + offset,
                    resamples=bootstrap_resamples,
                ),
            )
        )
    return tuple(summaries)


def run_rolling_backtest(
    draws: pd.DataFrame,
    prize_table: PrizeTable | None = None,
    *,
    strategies: Mapping[str, StrategyFunction] | None = None,
    pipeline_configs: Mapping[str, PipelineConfig] | None = None,
    min_history: int = 1,
    base_random_seed: int = 20260000,
    strategy_parameters: Mapping[str, Mapping[str, JsonValue]] | None = None,
    prize_context_by_issue: Mapping[str, str] | None = None,
    prize_tables: Sequence[PrizeTable] | None = None,
    issue_prize_records: Mapping[str, IssuePrizeRecord] | None = None,
    number_scorer: NumberScorer = cumulative_frequency_score,
    number_scorer_name: str | None = None,
    random_baseline_seed_count: int = 1000,
    bootstrap_resamples: int = 1000,
    default_ticket_cost: Decimal = Decimal("2"),
    data_quality_report: DataQualityReport | None = None,
    allow_short_history: bool = False,
    allow_reduced_resampling: bool = False,
) -> BacktestReport:
    """Run strict expanding-window predictions; never expose the target draw to a strategy."""
    if min_history < 1:
        raise ValueError("min_history must be at least 1")
    if random_baseline_seed_count < 1:
        raise ValueError("random baseline must use at least one seed")
    if random_baseline_seed_count < 1000 and not allow_reduced_resampling:
        raise ValueError("random baseline must use at least 1000 seeds")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least one")
    if bootstrap_resamples < 100 and not allow_reduced_resampling:
        raise ValueError("bootstrap_resamples must be at least 100")
    if default_ticket_cost <= 0:
        raise ValueError("default_ticket_cost must be positive")
    quality = data_quality_report or generate_data_quality_report(draws)
    assert_backtest_ready(quality)
    validated = validate_draw_dataframe(draws)
    if min_history >= len(validated):
        return BacktestReport(
            results=(),
            strategy_summaries=(),
            random_baseline_summaries=(),
            includes_random_baseline=True,
            methodology="expanding_window_strictly_before_target_issue",
            comparative_claim="当前框架不声称任何策略优于完全随机 baseline。",
        )

    active_strategies = dict(strategies or {})
    active_strategies["random_baseline"] = random_baseline
    active_pipelines = dict(pipeline_configs or {})
    if active_pipelines and set(active_pipelines) != {"optimized_portfolio_strategy"}:
        raise ValueError("pipeline strategy must be named optimized_portfolio_strategy")
    all_parameters = dict(strategy_parameters or {})
    prize_contexts = dict(prize_context_by_issue or {})
    actual_prizes = dict(issue_prize_records or {})
    schedule = _build_schedule(prize_table, prize_tables)
    resolved_scorer_name = number_scorer_name or str(
        getattr(number_scorer, "__name__", number_scorer.__class__.__name__)
    )
    results: list[BacktestPeriodResult] = []
    baseline_summaries: list[RandomBaselineSummary] = []
    baseline_seed_start = base_random_seed + 1_000_000

    for target_index in range(min_history, len(validated)):
        history = validated.iloc[:target_index].copy()
        target_draw = _draw_from_row(validated.iloc[target_index])
        cutoff_draw = _draw_from_row(history.iloc[-1])
        if int(cutoff_draw.issue) >= int(target_draw.issue):
            raise ValueError("rolling split cutoff must precede target issue")
        generated_at = _historical_generation_time(cutoff_draw)

        cache_before = _cached_random_distribution.cache_info().hits
        baseline_distribution = _cached_random_distribution(
            baseline_seed_start,
            random_baseline_seed_count,
        )
        cache_reused = _cached_random_distribution.cache_info().hits > cache_before
        baseline_summaries.append(
            RandomBaselineSummary(
                target_issue=target_draw.issue,
                cache_key=f"random-five-v1:{baseline_seed_start}:{random_baseline_seed_count}",
                cache_reused=cache_reused,
                seed_start=baseline_seed_start,
                seed_count=random_baseline_seed_count,
                metric_summaries=_monte_carlo_summaries(baseline_distribution),
            )
        )
        distributions_by_metric = {
            metric: [_metric_values(sample)[metric] for sample in baseline_distribution]
            for metric in RANDOM_METRICS
        }

        front_scores = dict(number_scorer(history, area="front"))
        back_scores = dict(number_scorer(history, area="back"))
        predictions: list[tuple[str, int, PredictionRecord]] = []
        for strategy_offset, (strategy_name, strategy) in enumerate(active_strategies.items()):
            random_seed = base_random_seed + target_index * 100 + strategy_offset
            parameters = dict(all_parameters.get(strategy_name, {}))
            parameters.setdefault("number_scorer", resolved_scorer_name)
            prediction = strategy(
                target_issue=target_draw.issue,
                data_cutoff_issue=cutoff_draw.issue,
                generated_at=generated_at,
                random_seed=random_seed,
                parameters=parameters,
                front_scores=front_scores,
                back_scores=back_scores,
            )
            predictions.append((strategy_name, random_seed, prediction))
        for pipeline_offset, (strategy_name, config) in enumerate(active_pipelines.items()):
            random_seed = base_random_seed + target_index * 100 + 50 + pipeline_offset
            prediction = optimized_portfolio_strategy(
                history,
                target_issue=target_draw.issue,
                generated_at=generated_at,
                random_seed=random_seed,
                pipeline_config=config,
                allow_short_history=allow_short_history,
            )
            predictions.append((strategy_name, random_seed, prediction))

        for strategy_name, random_seed, prediction in predictions:
            if prediction.strategy_name != strategy_name:
                raise ValueError(
                    "strategy mapping name differs from PredictionRecord.strategy_name"
                )
            if prediction.data_cutoff_issue != cutoff_draw.issue:
                raise ValueError("strategy returned an incorrect data cutoff issue")
            hits = _hit_metrics(prediction, target_draw)
            (
                any_prize,
                total_cost,
                total_prize,
                roi,
                prize_rule_version,
                prize_data_available,
            ) = _evaluate_monetary_result(
                prediction,
                target_draw,
                schedule,
                prize_contexts=prize_contexts,
                issue_prize_records=actual_prizes,
                default_ticket_cost=default_ticket_cost,
            )
            metric_values = _metric_values(hits)
            results.append(
                BacktestPeriodResult(
                    target_issue=target_draw.issue,
                    data_cutoff_issue=cutoff_draw.issue,
                    strategy_name=strategy_name,
                    random_seed=random_seed,
                    best_front_hits=hits.best_front_hits,
                    best_back_hits=hits.best_back_hits,
                    best_total_hits=hits.best_total_hits,
                    at_least_three_front=hits.at_least_three_front,
                    at_least_2_plus_1=hits.at_least_2_plus_1,
                    ticket_hit_share=hits.ticket_hit_share,
                    unique_hit_concentration=hits.unique_hit_concentration,
                    any_prize=any_prize,
                    front_pool_coverage=hits.front_pool_coverage,
                    back_pool_coverage=hits.back_pool_coverage,
                    total_cost=total_cost,
                    total_prize=total_prize,
                    roi=roi,
                    prize_rule_version=prize_rule_version,
                    prize_data_available=prize_data_available,
                    random_baseline_seed_count=random_baseline_seed_count,
                    random_metric_percentiles={
                        metric: _percentile_in_distribution(
                            metric_values[metric],
                            distributions_by_metric[metric],
                        )
                        for metric in RANDOM_METRICS
                    },
                )
            )

    return BacktestReport(
        results=tuple(results),
        strategy_summaries=_strategy_summaries(
            results,
            random_seed=base_random_seed + 2_000_000,
            bootstrap_resamples=bootstrap_resamples,
        ),
        random_baseline_summaries=tuple(baseline_summaries),
        includes_random_baseline=True,
        methodology="expanding_window_strictly_before_target_issue",
        comparative_claim="当前框架不声称任何策略优于完全随机 baseline。",
    )

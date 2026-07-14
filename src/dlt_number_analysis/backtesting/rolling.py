"""只用目标期之前数据生成历史预测的向前滚动回测。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time
from decimal import Decimal
from random import Random
from statistics import median
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import JsonValue

from dlt_number_analysis.backtesting.models import (
    BacktestPeriodResult,
    BacktestReport,
    BacktestStrategySummary,
    RandomBaselineSummary,
)
from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.evaluation import (
    IssuePrizeRecord,
    PrizeRuleSchedule,
    PrizeTable,
    apply_issue_prize_record,
    evaluate_prediction,
)
from dlt_number_analysis.models import PredictionRecord
from dlt_number_analysis.scoring import NumberScorer, cumulative_frequency_score
from dlt_number_analysis.strategies import StrategyFunction, random_baseline


class MissingPrizeAmountError(ValueError):
    """已弃用：v0.3 对缺失奖金保留命中指标并把 ROI 标为不可用。"""


@dataclass(frozen=True, slots=True)
class _HitMetrics:
    """与奖级规则无关的五注命中指标。"""

    best_front_hits: int
    best_back_hits: int
    best_total_hits: int
    at_least_three_front: bool
    at_least_2_plus_1: bool
    hit_concentration_ratio: float
    front_pool_coverage: int
    back_pool_coverage: int


def _draw_from_row(row: pd.Series) -> DrawRecord:
    """把已经校验的数据行恢复为 DrawRecord。"""
    return DrawRecord.model_validate({column: row[column] for column in CSV_COLUMNS})


def _historical_generation_time(cutoff_draw: DrawRecord) -> datetime:
    """在最后一期历史开奖日结束时建立带时区的历史预测时间戳。"""
    return datetime.combine(
        cutoff_draw.draw_date,
        time(23, 59),
        tzinfo=ZoneInfo("Asia/Shanghai"),
    )


def _hit_metrics(prediction: PredictionRecord, actual_draw: DrawRecord) -> _HitMetrics:
    """不依赖任何奖级规则计算五注命中与号码池覆盖。"""
    actual_front = set(actual_draw.front_numbers)
    actual_back = set(actual_draw.back_numbers)
    hit_pairs = [
        (
            len(set(ticket.front_numbers).intersection(actual_front)),
            len(set(ticket.back_numbers).intersection(actual_back)),
        )
        for ticket in prediction.tickets
    ]
    total_hits = [front_hits + back_hits for front_hits, back_hits in hit_pairs]
    all_hits = sum(total_hits)
    front_pool = {number for ticket in prediction.tickets for number in ticket.front_numbers}
    back_pool = {number for ticket in prediction.tickets for number in ticket.back_numbers}
    return _HitMetrics(
        best_front_hits=max(front_hits for front_hits, _ in hit_pairs),
        best_back_hits=max(back_hits for _, back_hits in hit_pairs),
        best_total_hits=max(total_hits),
        at_least_three_front=any(front_hits >= 3 for front_hits, _ in hit_pairs),
        at_least_2_plus_1=any(
            front_hits >= 2 and back_hits >= 1 for front_hits, back_hits in hit_pairs
        ),
        hit_concentration_ratio=0.0 if all_hits == 0 else max(total_hits) / all_hits,
        front_pool_coverage=len(front_pool.intersection(actual_front)),
        back_pool_coverage=len(back_pool.intersection(actual_back)),
    )


def _random_best_total_hits(actual_draw: DrawRecord, random_seed: int) -> int:
    """用一个种子直接模拟五注均匀随机 baseline 的最佳总命中数。"""
    rng = Random(random_seed)
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    totals: list[int] = []
    actual_front = set(actual_draw.front_numbers)
    actual_back = set(actual_draw.back_numbers)
    while len(totals) < 5:
        front = tuple(sorted(rng.sample(range(1, 36), 5)))
        back = tuple(sorted(rng.sample(range(1, 13), 2)))
        combination = (front, back)
        if combination in seen:
            continue
        seen.add(combination)
        totals.append(
            len(set(front).intersection(actual_front)) + len(set(back).intersection(actual_back))
        )
    return max(totals)


def _bootstrap_intervals(
    values: Sequence[int],
    *,
    random_seed: int,
    resamples: int,
) -> tuple[float, float, dict[int, tuple[float, float]]]:
    """Bootstrap 随机均值及各策略命中值的随机百分位 95% 区间。"""
    sample = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    bootstrap_means: list[float] = []
    bootstrap_percentiles: dict[int, list[float]] = {value: [] for value in range(8)}
    remaining = resamples
    while remaining:
        batch_size = min(remaining, 256)
        indices = rng.integers(0, len(sample), size=(batch_size, len(sample)))
        resampled = sample[indices]
        bootstrap_means.extend(resampled.mean(axis=1).tolist())
        for value in bootstrap_percentiles:
            percentiles = (
                100.0
                * ((resampled < value).sum(axis=1) + 0.5 * (resampled == value).sum(axis=1))
                / len(sample)
            )
            bootstrap_percentiles[value].extend(percentiles.tolist())
        remaining -= batch_size
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975), method="linear")
    percentile_intervals = {}
    for value, percentiles in bootstrap_percentiles.items():
        percentile_lower, percentile_upper = np.quantile(
            percentiles,
            (0.025, 0.975),
            method="linear",
        )
        percentile_intervals[value] = (float(percentile_lower), float(percentile_upper))
    return float(lower), float(upper), percentile_intervals


def _percentile_in_random_distribution(value: int, distribution: Sequence[int]) -> float:
    """用中秩计算策略值在随机分布中的百分位。"""
    lower = sum(sample < value for sample in distribution)
    equal = sum(sample == value for sample in distribution)
    return 100.0 * (lower + 0.5 * equal) / len(distribution)


def _build_schedule(
    prize_table: PrizeTable | None,
    prize_tables: Sequence[PrizeTable] | None,
) -> PrizeRuleSchedule | None:
    """合并兼容参数并构建历史规则时间表。"""
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
    """按该期生效规则计算金额；没有适用规则时只返回成本。"""
    table = None if schedule is None else schedule.table_for_issue(target_draw.issue)
    if table is None:
        total_cost = default_ticket_cost * len(prediction.tickets)
        return None, total_cost, None, None, None, False

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
        return (
            review.any_prize,
            review.total_cost,
            None,
            None,
            table.version,
            False,
        )
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
    """计算连续无奖期数；调用方保证 any_prize 均已知。"""
    longest = 0
    current = 0
    for result in results:
        if result.any_prize:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _strategy_summaries(
    results: Sequence[BacktestPeriodResult],
) -> tuple[BacktestStrategySummary, ...]:
    """按策略汇总新增命中率、集中度和奖金统计。"""
    grouped: dict[str, list[BacktestPeriodResult]] = defaultdict(list)
    for result in results:
        grouped[result.strategy_name].append(result)

    summaries: list[BacktestStrategySummary] = []
    for strategy_name, strategy_results in grouped.items():
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
                hit_concentration_ratio=(
                    sum(result.hit_concentration_ratio for result in strategy_results) / count
                ),
                average_prize=(sum(prizes, Decimal("0")) / count if monetary_complete else None),
                median_prize=median(prizes) if monetary_complete else None,
                longest_no_prize_streak=(
                    _longest_no_prize_streak(strategy_results) if prize_flags_known else None
                ),
                monetary_metrics_complete=monetary_complete,
            )
        )
    return tuple(summaries)


def run_rolling_backtest(
    draws: pd.DataFrame,
    prize_table: PrizeTable | None = None,
    *,
    strategies: Mapping[str, StrategyFunction] | None = None,
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
) -> BacktestReport:
    """逐期扩展历史窗口，比较策略和至少 1000 个种子的随机分布。

    对索引为 ``N`` 的目标期开奖，只把 ``draws.iloc[:N]`` 交给号码评分器和策略。
    实际目标期开奖仅在预测生成完成后用于命中和奖金评估，不存在随机时间切分。
    ROI 只在找到该期已生效规则且奖金金额完整时计算；否则保留命中指标并标为不可用。
    """
    if min_history < 1:
        raise ValueError("min_history 必须至少为 1")
    if random_baseline_seed_count < 1000:
        raise ValueError("每个历史时点的随机 baseline 必须至少运行 1000 个种子")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples 必须至少为 100")
    if default_ticket_cost <= 0:
        raise ValueError("default_ticket_cost 必须大于 0")

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
    all_parameters = dict(strategy_parameters or {})
    prize_contexts = dict(prize_context_by_issue or {})
    actual_prizes = dict(issue_prize_records or {})
    schedule = _build_schedule(prize_table, prize_tables)
    resolved_scorer_name = number_scorer_name or str(
        getattr(number_scorer, "__name__", number_scorer.__class__.__name__)
    )
    results: list[BacktestPeriodResult] = []
    baseline_summaries: list[RandomBaselineSummary] = []

    for target_index in range(min_history, len(validated)):
        history = validated.iloc[:target_index].copy()
        target_draw = _draw_from_row(validated.iloc[target_index])
        cutoff_draw = _draw_from_row(history.iloc[-1])
        if int(cutoff_draw.issue) >= int(target_draw.issue):
            raise ValueError("滚动切分错误：数据截止期号没有早于目标期号")

        front_scores = dict(number_scorer(history, area="front"))
        back_scores = dict(number_scorer(history, area="back"))
        generated_at = _historical_generation_time(cutoff_draw)
        baseline_seed_start = base_random_seed + target_index * 10_000
        baseline_distribution = tuple(
            _random_best_total_hits(target_draw, baseline_seed_start + offset)
            for offset in range(random_baseline_seed_count)
        )
        ci_lower, ci_upper, percentile_intervals = _bootstrap_intervals(
            baseline_distribution,
            random_seed=baseline_seed_start + random_baseline_seed_count,
            resamples=bootstrap_resamples,
        )
        baseline_summaries.append(
            RandomBaselineSummary(
                target_issue=target_draw.issue,
                seed_start=baseline_seed_start,
                seed_count=random_baseline_seed_count,
                mean=sum(baseline_distribution) / len(baseline_distribution),
                bootstrap_ci_lower=ci_lower,
                bootstrap_ci_upper=ci_upper,
            )
        )

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
            if prediction.strategy_name != strategy_name:
                raise ValueError("策略映射名称与预测记录 strategy_name 不一致")
            if prediction.data_cutoff_issue != cutoff_draw.issue:
                raise ValueError("策略返回了错误的数据截止期号")

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
                    hit_concentration_ratio=hits.hit_concentration_ratio,
                    any_prize=any_prize,
                    front_pool_coverage=hits.front_pool_coverage,
                    back_pool_coverage=hits.back_pool_coverage,
                    total_cost=total_cost,
                    total_prize=total_prize,
                    roi=roi,
                    prize_rule_version=prize_rule_version,
                    prize_data_available=prize_data_available,
                    random_baseline_seed_count=random_baseline_seed_count,
                    best_total_hits_percentile=_percentile_in_random_distribution(
                        hits.best_total_hits,
                        baseline_distribution,
                    ),
                    best_total_hits_percentile_ci_lower=percentile_intervals[hits.best_total_hits][
                        0
                    ],
                    best_total_hits_percentile_ci_upper=percentile_intervals[hits.best_total_hits][
                        1
                    ],
                    random_best_total_hits_mean_ci_lower=ci_lower,
                    random_best_total_hits_mean_ci_upper=ci_upper,
                )
            )

    return BacktestReport(
        results=tuple(results),
        strategy_summaries=_strategy_summaries(results),
        random_baseline_summaries=tuple(baseline_summaries),
        includes_random_baseline=True,
        methodology="expanding_window_strictly_before_target_issue",
        comparative_claim="当前框架不声称任何策略优于完全随机 baseline。",
    )

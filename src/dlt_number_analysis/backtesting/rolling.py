"""只用目标期之前数据生成历史预测的向前滚动回测。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import JsonValue

from dlt_number_analysis.backtesting.models import BacktestPeriodResult, BacktestReport
from dlt_number_analysis.data import BACK_COLUMNS, CSV_COLUMNS, FRONT_COLUMNS, DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.evaluation import PrizeTable, evaluate_prediction
from dlt_number_analysis.strategies import StrategyFunction, random_baseline


class MissingPrizeAmountError(ValueError):
    """命中浮动奖但没有提供该历史期实际奖金。"""


def _frequency_scores(
    history: pd.DataFrame,
    columns: Sequence[str],
    maximum: int,
) -> dict[int, float]:
    """仅从传入历史窗口计算简单出现频率分数。"""
    counts = dict.fromkeys(range(1, maximum + 1), 0)
    for values in history.loc[:, columns].itertuples(index=False, name=None):
        for value in values:
            counts[int(value)] += 1
    denominator = len(history) * len(columns)
    return {number: count / denominator for number, count in counts.items()}


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


def run_rolling_backtest(
    draws: pd.DataFrame,
    prize_table: PrizeTable,
    *,
    strategies: Mapping[str, StrategyFunction] | None = None,
    min_history: int = 1,
    base_random_seed: int = 20260000,
    strategy_parameters: Mapping[str, Mapping[str, JsonValue]] | None = None,
    prize_context_by_issue: Mapping[str, str] | None = None,
) -> BacktestReport:
    """逐期扩展历史窗口并回测策略，始终附加完全随机五注 baseline。

    对索引为 ``N`` 的目标期开奖，只把 ``draws.iloc[:N]`` 交给特征统计。
    实际目标期开奖仅在预测生成完成后用于评估，函数中不存在随机时间切分。
    """
    if min_history < 1:
        raise ValueError("min_history 必须至少为 1")
    validated = validate_draw_dataframe(draws)
    if min_history >= len(validated):
        return BacktestReport(
            results=(),
            includes_random_baseline=True,
            methodology="expanding_window_strictly_before_target_issue",
            comparative_claim="当前骨架不声称任何策略优于完全随机 baseline。",
        )

    active_strategies = dict(strategies or {})
    active_strategies["random_baseline"] = random_baseline
    all_parameters = dict(strategy_parameters or {})
    prize_contexts = dict(prize_context_by_issue or {})
    results: list[BacktestPeriodResult] = []

    for target_index in range(min_history, len(validated)):
        history = validated.iloc[:target_index].copy()
        target_draw = _draw_from_row(validated.iloc[target_index])
        cutoff_draw = _draw_from_row(history.iloc[-1])
        if int(cutoff_draw.issue) >= int(target_draw.issue):
            raise ValueError("滚动切分错误：数据截止期号没有早于目标期号")

        front_scores = _frequency_scores(history, FRONT_COLUMNS, 35)
        back_scores = _frequency_scores(history, BACK_COLUMNS, 12)
        generated_at = _historical_generation_time(cutoff_draw)
        context = prize_contexts.get(target_draw.issue, prize_table.default_context)

        for strategy_offset, (strategy_name, strategy) in enumerate(active_strategies.items()):
            random_seed = base_random_seed + target_index * 100 + strategy_offset
            prediction = strategy(
                target_issue=target_draw.issue,
                data_cutoff_issue=cutoff_draw.issue,
                generated_at=generated_at,
                random_seed=random_seed,
                parameters=all_parameters.get(strategy_name),
                front_scores=front_scores,
                back_scores=back_scores,
            )
            if prediction.strategy_name != strategy_name:
                raise ValueError("策略映射名称与预测记录 strategy_name 不一致")
            if prediction.data_cutoff_issue != cutoff_draw.issue:
                raise ValueError("策略返回了错误的数据截止期号")

            review = evaluate_prediction(
                prediction,
                target_draw,
                prize_table,
                prize_context=context,
            )
            if review.total_prize is None or review.roi is None:
                raise MissingPrizeAmountError(
                    f"{target_draw.issue} 命中浮动奖，必须补充该期实际单注奖金后才能计算 ROI"
                )
            results.append(
                BacktestPeriodResult(
                    target_issue=target_draw.issue,
                    data_cutoff_issue=cutoff_draw.issue,
                    strategy_name=strategy_name,
                    random_seed=random_seed,
                    best_front_hits=review.best_front_hits,
                    best_back_hits=review.best_back_hits,
                    any_prize=review.any_prize,
                    front_pool_coverage=review.front_pool_coverage,
                    back_pool_coverage=review.back_pool_coverage,
                    total_cost=review.total_cost,
                    total_prize=review.total_prize,
                    roi=review.roi,
                )
            )

    return BacktestReport(
        results=tuple(results),
        includes_random_baseline=True,
        methodology="expanding_window_strictly_before_target_issue",
        comparative_claim="当前骨架不声称任何策略优于完全随机 baseline。",
    )

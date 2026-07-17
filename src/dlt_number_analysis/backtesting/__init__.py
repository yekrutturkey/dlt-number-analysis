"""严格向前滚动的时间序列回测框架。"""

from dlt_number_analysis.backtesting.models import (
    BacktestPeriodResult,
    BacktestReport,
    BacktestStrategySummary,
    ConfidenceInterval,
    RandomBaselineSummary,
    RandomMetricSummary,
    RawObservationResult,
)
from dlt_number_analysis.backtesting.rolling import (
    MissingPrizeAmountError,
    calculate_raw_hit_metrics,
    clear_random_baseline_cache,
    evaluate_prediction_raw_observation,
    random_baseline_cache_info,
    run_rolling_backtest,
)

__all__ = [
    "BacktestPeriodResult",
    "BacktestReport",
    "BacktestStrategySummary",
    "ConfidenceInterval",
    "MissingPrizeAmountError",
    "RandomBaselineSummary",
    "RandomMetricSummary",
    "RawObservationResult",
    "calculate_raw_hit_metrics",
    "clear_random_baseline_cache",
    "evaluate_prediction_raw_observation",
    "random_baseline_cache_info",
    "run_rolling_backtest",
]

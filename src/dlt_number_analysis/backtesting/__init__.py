"""严格向前滚动的时间序列回测框架。"""

from dlt_number_analysis.backtesting.models import (
    BacktestPeriodResult,
    BacktestReport,
    BacktestStrategySummary,
    ConfidenceInterval,
    RandomBaselineSummary,
    RandomMetricSummary,
)
from dlt_number_analysis.backtesting.rolling import (
    MissingPrizeAmountError,
    clear_random_baseline_cache,
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
    "clear_random_baseline_cache",
    "random_baseline_cache_info",
    "run_rolling_backtest",
]

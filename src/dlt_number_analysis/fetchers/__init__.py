"""外部数据抓取层边界。

抓取实现只能返回待校验的原始记录，不得依赖分析或回测模块；当前阶段不提供网络抓取器。
"""

from dlt_number_analysis.fetchers.history import (
    FIVE_HUNDRED_ENDPOINT,
    SPORTTERY_ENDPOINT,
    FiveHundredHistoryFetcher,
    SportteryHistoryFetcher,
)

__all__ = [
    "FIVE_HUNDRED_ENDPOINT",
    "SPORTTERY_ENDPOINT",
    "FiveHundredHistoryFetcher",
    "SportteryHistoryFetcher",
]

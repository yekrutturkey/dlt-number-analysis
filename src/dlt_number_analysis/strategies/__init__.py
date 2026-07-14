"""可替换的五注组合策略接口。"""

from dlt_number_analysis.strategies.combination import (
    StrategyFunction,
    core_rotation,
    hybrid_portfolio,
    max_coverage,
    random_baseline,
)

__all__ = [
    "StrategyFunction",
    "core_rotation",
    "hybrid_portfolio",
    "max_coverage",
    "random_baseline",
]

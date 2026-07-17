"""Paired strategy statistics that respect shared historical target periods."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

METRIC_SOURCE_COLUMNS: dict[str, str] = {
    "best_front_hits": "best_front_hits",
    "best_total_hits": "best_total_hits",
    "at_least_three_front_rate": "at_least_three_front",
    "at_least_2_plus_1_rate": "at_least_2_plus_1",
    "unique_hit_concentration": "unique_hit_concentration",
    "any_prize_rate": "any_prize",
    "roi": "roi",
}


class PairedBootstrapResult(BaseModel):
    """Bootstrap interval over paired target/seed differences."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = "paired_bootstrap_percentile"
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    sample_size: int = Field(ge=1)
    resamples: int = Field(ge=100)
    observed_mean_difference: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("bootstrap lower bound must not exceed upper bound")
        return self


class PermutationTestResult(BaseModel):
    """Two-sided paired sign-flip permutation test result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = "paired_sign_flip_permutation"
    sample_size: int = Field(ge=1)
    permutations: int = Field(ge=100)
    observed_mean_difference: float
    p_value: float = Field(ge=0, le=1)


def _metric_source(metric: str) -> str:
    try:
        return METRIC_SOURCE_COLUMNS[metric]
    except KeyError as error:
        raise ValueError(f"unsupported comparison metric: {metric}") from error


def paired_metric_differences(
    strategy_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    *,
    metrics: Sequence[str] = tuple(METRIC_SOURCE_COLUMNS),
    pair_keys: Sequence[str] = ("target_issue", "seed"),
) -> pd.DataFrame:
    """Align identical target/seed observations and return strategy-minus-baseline values."""
    keys = list(pair_keys)
    sources = {_metric_source(metric) for metric in metrics}
    required = set(keys).union(sources)
    for label, frame in (("strategy", strategy_results), ("baseline", baseline_results)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} results are missing columns: {sorted(missing)}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{label} results contain duplicate paired observations")

    merged = strategy_results.loc[:, [*keys, *sorted(sources)]].merge(
        baseline_results.loc[:, [*keys, *sorted(sources)]],
        on=keys,
        how="inner",
        suffixes=("_strategy", "_baseline"),
        validate="one_to_one",
    )
    records: list[dict[str, object]] = []
    for metric in metrics:
        source = _metric_source(metric)
        for row in merged.to_dict(orient="records"):
            strategy_value = row[f"{source}_strategy"]
            baseline_value = row[f"{source}_baseline"]
            if pd.isna(strategy_value) or pd.isna(baseline_value):
                continue
            records.append(
                {
                    **{key: row[key] for key in keys},
                    "metric": metric,
                    "strategy_value": float(strategy_value),
                    "baseline_value": float(baseline_value),
                    "difference": float(strategy_value) - float(baseline_value),
                }
            )
    return pd.DataFrame(
        records,
        columns=[
            *keys,
            "metric",
            "strategy_value",
            "baseline_value",
            "difference",
        ],
    )


def _finite_differences(differences: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("paired comparison requires at least one finite difference")
    return values


def paired_bootstrap_95_interval(
    differences: Sequence[float] | np.ndarray,
    *,
    random_seed: int = 20260000,
    resamples: int = 10_000,
) -> PairedBootstrapResult:
    """Resample paired differences, not the two strategies independently."""
    if resamples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    values = _finite_differences(differences)
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    bootstrap_means = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, (0.025, 0.975))
    return PairedBootstrapResult(
        sample_size=int(values.size),
        resamples=resamples,
        observed_mean_difference=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
    )


def paired_permutation_test(
    differences: Sequence[float] | np.ndarray,
    *,
    random_seed: int = 20260001,
    permutations: int = 10_000,
) -> PermutationTestResult:
    """Test a zero paired mean by randomly flipping within-pair difference signs."""
    if permutations < 100:
        raise ValueError("paired permutation test requires at least 100 permutations")
    values = _finite_differences(differences)
    observed = float(values.mean())
    rng = np.random.default_rng(random_seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(permutations, values.size))
    permuted = (signs * values).mean(axis=1)
    extreme = int(np.count_nonzero(np.abs(permuted) >= abs(observed)))
    return PermutationTestResult(
        sample_size=int(values.size),
        permutations=permutations,
        observed_mean_difference=observed,
        p_value=(extreme + 1) / (permutations + 1),
    )


def _validated_p_values(p_values: Sequence[float]) -> list[float]:
    values = [float(value) for value in p_values]
    if any(not isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("p-values must be finite values in [0, 1]")
    return values


def adjust_p_values_holm(p_values: Sequence[float]) -> tuple[float, ...]:
    """Apply Holm's family-wise error correction with monotone adjusted values."""
    values = _validated_p_values(p_values)
    count = len(values)
    if count == 0:
        return ()
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return tuple(adjusted)


def adjust_p_values_benjamini_hochberg(
    p_values: Sequence[float],
) -> tuple[float, ...]:
    """Apply Benjamini-Hochberg false-discovery-rate correction."""
    values = _validated_p_values(p_values)
    count = len(values)
    if count == 0:
        return ()
    order = sorted(range(count), key=values.__getitem__)
    adjusted = [0.0] * count
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, values[index] * count / rank)
        adjusted[index] = min(running, 1.0)
    return tuple(adjusted)


def _aggregate_performance(results: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    required = {
        *group_columns,
        "target_issue",
        *METRIC_SOURCE_COLUMNS.values(),
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"performance results are missing columns: {sorted(missing)}")
    grouped = results.assign(
        at_least_three_front=results["at_least_three_front"].astype(float),
        at_least_2_plus_1=results["at_least_2_plus_1"].astype(float),
        any_prize=pd.to_numeric(results["any_prize"], errors="coerce"),
        roi=pd.to_numeric(results["roi"], errors="coerce"),
    ).groupby(group_columns, dropna=False, sort=True)
    output = grouped.agg(
        period_count=("target_issue", "size"),
        best_front_hits=("best_front_hits", "mean"),
        best_total_hits=("best_total_hits", "mean"),
        at_least_three_front_rate=("at_least_three_front", "mean"),
        at_least_2_plus_1_rate=("at_least_2_plus_1", "mean"),
        unique_hit_concentration=("unique_hit_concentration", "mean"),
        any_prize_rate=("any_prize", "mean"),
        roi=("roi", "mean"),
        roi_period_count=("roi", "count"),
    )
    return output.reset_index()


def performance_by_year(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strategy metrics by actual draw year."""
    if "draw_date" not in results:
        raise ValueError("performance-by-year requires draw_date")
    dated = results.copy()
    dated["year"] = pd.to_datetime(dated["draw_date"], errors="raise").dt.year
    context = [
        column
        for column in (
            "phase",
            "cohort_id",
            "cohort_definition_sha256",
            "run_context_sha256",
            "history_sha256",
            "evaluation_mode",
            "profile",
        )
        if column in dated
    ]
    return _aggregate_performance(dated, [*context, "experiment_id", "year"])


def performance_by_seed(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate strategy metrics independently for every experiment seed."""
    context = [
        column
        for column in (
            "phase",
            "cohort_id",
            "cohort_definition_sha256",
            "run_context_sha256",
            "history_sha256",
            "evaluation_mode",
            "profile",
        )
        if column in results
    ]
    return _aggregate_performance(results, [*context, "experiment_id", "seed"])


def parameter_sensitivity_report(
    results: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    metric_columns: Sequence[str] = (
        "best_front_hits",
        "best_total_hits",
        "unique_hit_concentration",
    ),
) -> pd.DataFrame:
    """Return long-form mean response for each varied parameter value."""
    required = set(parameter_columns).union(metric_columns)
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"sensitivity results are missing columns: {sorted(missing)}")
    records: list[dict[str, object]] = []
    for parameter in parameter_columns:
        grouped = results.groupby(parameter, dropna=False, sort=True)
        for parameter_value, group in grouped:
            for metric in metric_columns:
                values = pd.to_numeric(group[metric], errors="coerce").dropna()
                records.append(
                    {
                        "parameter": parameter,
                        "parameter_value": parameter_value,
                        "metric": metric,
                        "mean": float(values.mean()) if not values.empty else float("nan"),
                        "observation_count": len(values),
                    }
                )
    return pd.DataFrame.from_records(records)

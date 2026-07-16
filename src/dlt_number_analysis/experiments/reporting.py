"""Markdown and CSV reports that separate fit, calibration, and final holdout."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import DataQualityReport, VerifiedHistoryManifest
from dlt_number_analysis.experiments.benchmark import (
    RuntimeBenchmarkRecord,
    choose_default_parallel_workers,
)
from dlt_number_analysis.experiments.scheduler import ProcessSchedulerReport
from dlt_number_analysis.experiments.statistics import (
    METRIC_SOURCE_COLUMNS,
    adjust_p_values_benjamini_hochberg,
    adjust_p_values_holm,
    paired_bootstrap_95_interval,
    paired_metric_differences,
    paired_permutation_test,
    performance_by_seed,
    performance_by_year,
)


class ExperimentComparison(BaseModel):
    """One paired, baseline-relative metric conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str
    cohort_id: str
    target_start_issue: str
    target_end_issue: str
    common_target_count: int = Field(ge=1)
    experiment_id: str
    baseline_id: str
    metric: str
    paired_observation_count: int = Field(ge=1)
    mean_paired_difference: float
    bootstrap_95_lower: float
    bootstrap_95_upper: float
    permutation_p_value: float = Field(ge=0, le=1)
    holm_adjusted_p_value: float = Field(default=1.0, ge=0, le=1)
    benjamini_hochberg_adjusted_p_value: float = Field(default=1.0, ge=0, le=1)
    holm_significant_advantage: bool = False
    benjamini_hochberg_significant_advantage: bool = False
    statistically_significant_advantage: bool
    inference_method: str = (
        "paired bootstrap plus paired sign-flip permutation test with Holm and "
        "Benjamini-Hochberg corrections"
    )


@dataclass(frozen=True, slots=True)
class RawObservationStatistics:
    """Statistics materialized only from stored raw observation partitions."""

    experiment_summary: pd.DataFrame
    performance_by_year: pd.DataFrame
    performance_by_seed: pd.DataFrame
    baseline_comparisons: tuple[ExperimentComparison, ...]


def summarize_raw_observation_partitions(
    observations: pd.DataFrame,
    *,
    bootstrap_resamples: int = 2000,
    permutations: int = 5000,
    minimum_paired_observations: int = 30,
) -> RawObservationStatistics:
    """Run statistics from raw rows only, without any prediction-generation dependency."""
    if observations.empty:
        return RawObservationStatistics(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), ())
    if "evaluation_mode" not in observations:
        raise ValueError("raw statistics require an evaluation_mode audit column")
    modes = set(observations["evaluation_mode"].astype(str))
    if modes != {"raw_observation"}:
        raise ValueError("raw statistics accept only raw_observation partitions")
    return RawObservationStatistics(
        experiment_summary=summarize_observations(observations),
        performance_by_year=performance_by_year(observations),
        performance_by_seed=performance_by_seed(observations),
        baseline_comparisons=compare_to_constraint_matched_baseline(
            observations,
            bootstrap_resamples=bootstrap_resamples,
            permutations=permutations,
            minimum_paired_observations=minimum_paired_observations,
        ),
    )


def _issue_tuple(values: pd.Series) -> tuple[str, ...]:
    return tuple(sorted(set(values.astype(str)), key=int))


def assign_observation_cohorts(observations: pd.DataFrame) -> pd.DataFrame:
    """Assign exact target cohorts without modifying stored historical partitions."""
    if observations.empty:
        return observations.copy()
    required = {"phase", "experiment_id", "target_issue", "seed"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"experiment observations are missing cohort fields: {sorted(missing)}")
    annotated = observations.copy()
    annotated["target_issue"] = annotated["target_issue"].astype(str)
    if "experiment_version" not in annotated:
        annotated["experiment_version"] = "unknown"
    if "cohort_id" not in annotated:
        annotated["cohort_id"] = ""
    annotated["cohort_id"] = annotated["cohort_id"].fillna("").astype(str)
    unassigned = annotated["cohort_id"].eq("")
    issues = pd.to_numeric(annotated["target_issue"], errors="raise")
    paired_smoke = (
        unassigned
        & annotated["phase"].eq("development")
        & annotated["experiment_id"].isin(("B1", "B2", "B3", "B4", "B5", "B6"))
        & issues.between(8009, 8108)
    )
    annotated.loc[paired_smoke, "cohort_id"] = "v051_paired_smoke_100"
    single_smoke = (
        annotated["cohort_id"].eq("")
        & annotated["phase"].eq("development")
        & annotated["experiment_id"].eq("B1")
        & issues.eq(8008)
    )
    annotated.loc[single_smoke, "cohort_id"] = "single_correctness_smoke"

    remaining = annotated.loc[annotated["cohort_id"].eq("")]
    if remaining.empty:
        return annotated
    group_columns = ["phase", "experiment_id", "experiment_version", "seed"]
    target_sets = {
        key: _issue_tuple(group["target_issue"])
        for key, group in remaining.groupby(group_columns, sort=True, dropna=False)
    }
    signatures_by_phase: dict[str, set[tuple[str, ...]]] = {}
    for key, targets in target_sets.items():
        signatures_by_phase.setdefault(str(key[0]), set()).add(targets)
    cohort_by_signature: dict[tuple[str, tuple[str, ...]], str] = {}
    for phase, signatures in signatures_by_phase.items():
        largest = max((len(signature) for signature in signatures), default=0)
        largest_signatures = [signature for signature in signatures if len(signature) == largest]
        for signature in signatures:
            start, end = signature[0], signature[-1]
            if len(largest_signatures) == 1 and signature == largest_signatures[0]:
                cohort = {
                    "development": "development_full",
                    "calibration": "calibration",
                    "final_holdout": "final_holdout",
                }.get(phase, f"{phase}_full")
            else:
                cohort = f"{phase}_partial_{start}_{end}_{len(signature)}"
            cohort_by_signature[(phase, signature)] = cohort
    for key, targets in target_sets.items():
        phase, experiment_id, experiment_version, seed = key
        mask = (
            annotated["cohort_id"].eq("")
            & annotated["phase"].eq(phase)
            & annotated["experiment_id"].eq(experiment_id)
            & annotated["experiment_version"].eq(experiment_version)
            & annotated["seed"].eq(seed)
        )
        annotated.loc[mask, "cohort_id"] = cohort_by_signature[(str(phase), targets)]
    return annotated


def _cohort_common_target_counts(observations: pd.DataFrame) -> dict[str, int]:
    common: dict[str, int] = {}
    for cohort_id, cohort in observations.groupby("cohort_id", sort=True):
        target_sets = [
            set(group["target_issue"].astype(str))
            for _, group in cohort.groupby("experiment_id", sort=True)
        ]
        common[str(cohort_id)] = len(set.intersection(*target_sets)) if target_sets else 0
    return common


def summarize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics only inside exact target cohorts and experiment versions."""
    if observations.empty:
        return pd.DataFrame()
    numeric = assign_observation_cohorts(observations)
    common_counts = _cohort_common_target_counts(numeric)
    for column in (
        "best_front_hits",
        "best_total_hits",
        "at_least_three_front",
        "at_least_2_plus_1",
        "unique_hit_concentration",
        "any_prize",
        "total_prize",
        "roi",
    ):
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return (
        numeric.groupby(
            ["phase", "cohort_id", "experiment_id", "experiment_version"],
            sort=True,
            dropna=False,
        )
        .agg(
            observation_count=("target_issue", "size"),
            target_start_issue=("target_issue", lambda values: min(values, key=int)),
            target_end_issue=("target_issue", lambda values: max(values, key=int)),
            seed_count=("seed", "nunique"),
            best_front_hits=("best_front_hits", "mean"),
            best_total_hits=("best_total_hits", "mean"),
            at_least_three_front_rate=("at_least_three_front", "mean"),
            at_least_2_plus_1_rate=("at_least_2_plus_1", "mean"),
            unique_hit_concentration=("unique_hit_concentration", "mean"),
            any_prize_rate=("any_prize", "mean"),
            average_prize=("total_prize", "mean"),
            median_prize=("total_prize", "median"),
            roi=("roi", "mean"),
            roi_period_count=("roi", "count"),
        )
        .reset_index()
        .assign(common_target_count=lambda frame: frame["cohort_id"].map(common_counts).astype(int))
    )


def compare_to_constraint_matched_baseline(
    observations: pd.DataFrame,
    *,
    baseline_id: str = "B1",
    bootstrap_resamples: int = 2000,
    permutations: int = 5000,
    random_seed: int = 20260714,
    minimum_paired_observations: int = 30,
) -> tuple[ExperimentComparison, ...]:
    """Compare on paired target/seed rows; never infer from independent CI overlap."""
    if minimum_paired_observations < 2:
        raise ValueError("minimum_paired_observations must be at least 2")
    comparisons: list[ExperimentComparison] = []
    if observations.empty:
        return ()
    annotated = assign_observation_cohorts(observations)
    for (phase, cohort_id), phase_frame in annotated.groupby(["phase", "cohort_id"], sort=True):
        baseline = phase_frame.loc[phase_frame["experiment_id"] == baseline_id]
        if baseline.empty:
            continue
        for experiment_id, strategy in phase_frame.groupby("experiment_id", sort=True):
            if experiment_id == baseline_id:
                continue
            differences = paired_metric_differences(strategy, baseline)
            for metric in METRIC_SOURCE_COLUMNS:
                values = differences.loc[
                    differences["metric"] == metric,
                    "difference",
                ].to_numpy(dtype=float)
                if values.size < minimum_paired_observations:
                    continue
                interval = paired_bootstrap_95_interval(
                    values,
                    random_seed=random_seed,
                    resamples=bootstrap_resamples,
                )
                test = paired_permutation_test(
                    values,
                    random_seed=random_seed + 1,
                    permutations=permutations,
                )
                comparisons.append(
                    ExperimentComparison(
                        phase=str(phase),
                        cohort_id=str(cohort_id),
                        target_start_issue=str(
                            min(differences["target_issue"].astype(str), key=int)
                        ),
                        target_end_issue=str(max(differences["target_issue"].astype(str), key=int)),
                        common_target_count=int(differences["target_issue"].nunique()),
                        experiment_id=str(experiment_id),
                        baseline_id=baseline_id,
                        metric=metric,
                        paired_observation_count=len(values),
                        mean_paired_difference=interval.observed_mean_difference,
                        bootstrap_95_lower=interval.lower,
                        bootstrap_95_upper=interval.upper,
                        permutation_p_value=test.p_value,
                        statistically_significant_advantage=False,
                    )
                )
    corrected: list[ExperimentComparison] = []
    families = sorted({(comparison.phase, comparison.cohort_id) for comparison in comparisons})
    for phase, cohort_id in families:
        family = [
            comparison
            for comparison in comparisons
            if comparison.phase == phase and comparison.cohort_id == cohort_id
        ]
        p_values = [comparison.permutation_p_value for comparison in family]
        holm = adjust_p_values_holm(p_values)
        benjamini = adjust_p_values_benjamini_hochberg(p_values)
        for comparison, holm_p, benjamini_p in zip(family, holm, benjamini, strict=True):
            positive = comparison.bootstrap_95_lower > 0
            holm_advantage = positive and holm_p < 0.05
            benjamini_advantage = positive and benjamini_p < 0.05
            corrected.append(
                comparison.model_copy(
                    update={
                        "holm_adjusted_p_value": holm_p,
                        "benjamini_hochberg_adjusted_p_value": benjamini_p,
                        "holm_significant_advantage": holm_advantage,
                        "benjamini_hochberg_significant_advantage": benjamini_advantage,
                        "statistically_significant_advantage": holm_advantage,
                    }
                )
            )
    return tuple(corrected)


def _write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
    return target


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without adding a runtime dependency."""
    columns = list(frame.columns)

    def value_text(value: object) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(value_text(row[column]) for column in columns) + " |"
        for row in frame.to_dict(orient="records")
    )
    return "\n".join(lines)


def write_data_quality_report(
    report: DataQualityReport,
    manifest: VerifiedHistoryManifest | None,
    path: str | Path,
) -> Path:
    """Write coverage, source, conflict, and formal-use status."""
    verified = manifest is not None and report.is_valid and report.conflict_count == 0
    lines = [
        "# 大乐透历史数据质量报告",
        "",
        f"- 记录数：{report.record_count}",
        (
            f"- 覆盖期次：{report.data_start_issue or '不可用'} "
            f"至 {report.data_cutoff_issue or '不可用'}"
        ),
        f"- 来源：{', '.join(report.source_names)}",
        f"- 来源冲突数：{report.conflict_count}",
        f"- 可用于正式回测：{'是' if verified and not report.blocks_backtest else '否'}",
    ]
    if manifest is not None:
        lines.extend(
            [
                f"- canonical_history_sha256：`{manifest.canonical_history_sha256}`",
                f"- 原始快照哈希数：{len(manifest.snapshot_content_hashes)}",
            ]
        )
    if report.findings:
        lines.extend(["", "## Findings", ""])
        lines.extend(
            f"- [{finding.severity}] {finding.code}: {finding.message}"
            for finding in report.findings
        )
    lines.extend(["", f"> {DISCLAIMER}"])
    return _write_text(path, "\n".join(lines))


def write_experiment_summary_report(
    observations: pd.DataFrame,
    comparisons: Sequence[ExperimentComparison],
    path: str | Path,
    *,
    inference_context: Literal["formal", "smoke"] = "formal",
) -> Path:
    """Write cohort-separated results and never rank incompatible target ranges."""
    summary = summarize_observations(observations)
    lines = [
        "# 严格策略实验汇总",
        "",
        (
            "本报告按共同目标期 cohort 分组；不同目标范围、种子或实验版本的平均值"
            "不会放入同一横向排名。随机波动不会被解释为预测能力。"
        ),
    ]
    cohort_titles = {
        "development_full": "完整 development 结果",
        "v051_paired_smoke_100": "v0.5.1 100期 paired smoke 结果",
        "single_correctness_smoke": "单期 correctness smoke 结果",
        "calibration": "校准（calibration）结果",
        "final_holdout": "最终留出（final holdout）结果",
    }
    available = set(summary["cohort_id"].astype(str)) if not summary.empty else set()
    ordered_cohorts = list(cohort_titles)
    ordered_cohorts.extend(sorted(available.difference(ordered_cohorts)))
    for cohort_id in ordered_cohorts:
        title = cohort_titles.get(cohort_id, f"其他独立 cohort：{cohort_id}")
        lines.extend(["", f"## {title}", ""])
        cohort_summary = summary.loc[
            summary.get("cohort_id", pd.Series(dtype=str)).astype(str) == cohort_id
        ]
        if cohort_summary.empty:
            lines.append("尚未执行或没有可用观测。")
        else:
            lines.append(_markdown_table(cohort_summary))
    lines.extend(["", "## 与 B1 约束匹配随机基线的配对比较", ""])
    if comparisons:
        for cohort_id in sorted({comparison.cohort_id for comparison in comparisons}):
            cohort_comparisons = [
                comparison for comparison in comparisons if comparison.cohort_id == cohort_id
            ]
            lines.extend(
                [
                    f"### {cohort_titles.get(cohort_id, cohort_id)}",
                    "",
                    (
                        f"共同配对期数：{cohort_comparisons[0].common_target_count}；"
                        "配对键为 cohort_id、target_issue 和 seed。"
                    ),
                    "",
                    _markdown_table(
                        pd.DataFrame(
                            [item.model_dump(mode="python") for item in cohort_comparisons]
                        )
                    ),
                ]
            )
            if inference_context == "smoke" or "smoke" in cohort_id:
                lines.extend(
                    [
                        "",
                        "该 cohort 仅用于正确性冒烟，不用于参数选择或策略优势声明。",
                    ]
                )
            else:
                significant = [
                    comparison
                    for comparison in cohort_comparisons
                    if comparison.statistically_significant_advantage
                ]
                lines.extend(
                    [
                        "",
                        (
                            "结论仅依据配对 bootstrap 与配对置换检验；"
                            "正式显著性结论采用更保守的 Holm 校正。"
                        ),
                        f"该 cohort 显著优势条目数：{len(significant)}。",
                    ]
                )
    else:
        lines.append("没有同一 cohort、同一期次、同一种子的足量 B1 配对结果。")
    lines.extend(["", f"> {DISCLAIMER}"])
    return _write_text(path, "\n".join(lines))


def write_ablation_results_csv(results: pd.DataFrame, path: str | Path) -> Path:
    """Write ablation rows with the mandatory risk statement on every record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    output = results.copy()
    output["risk_disclaimer"] = DISCLAIMER
    output.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    return target


def write_holdout_results_report(
    observations: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Write only final-holdout observations, explicitly noting when still locked/unrun."""
    holdout = (
        observations.loc[observations["phase"] == "final_holdout"]
        if not observations.empty and "phase" in observations
        else pd.DataFrame()
    )
    lines = ["# 最终留出结果", ""]
    if holdout.empty:
        lines.append("最终留出集尚未正式运行；没有解锁或生成可供调参查看的结果。")
    else:
        lines.append(_markdown_table(summarize_observations(holdout)))
        lines.append("\n该结果由一次性实验版本锁生成，不得用于回头修改同版本参数。")
    lines.extend(["", f"> {DISCLAIMER}"])
    return _write_text(path, "\n".join(lines))


def write_runtime_benchmark_report(
    records: Sequence[RuntimeBenchmarkRecord],
    path: str | Path,
) -> Path:
    """Write profile/worker timings and the conservative worker recommendation."""
    frame = pd.DataFrame([record.model_dump(mode="python") for record in records])
    lines = [
        "# Pipeline 性能基准",
        "",
        "peak_memory_mb 使用 tracemalloc，仅表示 Python 分配峰值，不包含解释器外全部内存。",
        "",
    ]
    if frame.empty:
        lines.append("尚未运行性能基准。")
    else:
        lines.append(_markdown_table(frame))
        scalar = [
            record
            for record in records
            if record.profile == "fast"
            and record.parallel_workers == 1
            and record.candidate_scoring_method == "scalar"
        ]
        numpy_batch = [
            record
            for record in records
            if record.profile == "fast"
            and record.parallel_workers == 1
            and record.candidate_scoring_method == "numpy_batch_features"
        ]
        if scalar and numpy_batch and numpy_batch[0].candidate_scoring_seconds > 0:
            speedup = scalar[0].candidate_scoring_seconds / numpy_batch[0].candidate_scoring_seconds
            lines.extend(
                [
                    "",
                    (
                        "NumPy 批量特征路径评估：fast/1-worker 候选评分加速比 "
                        f"{speedup:.3f}x（标量 / NumPy）；最终票据与种子配置相同。"
                    ),
                ]
            )
        lines.extend(
            [
                "",
                f"建议 final 默认 parallel_workers={choose_default_parallel_workers(records)}。",
                "只有 4 线程在每次配对测量中均达到稳定加速时，才会建议改为 4。",
            ]
        )
    lines.extend(["", f"> {DISCLAIMER}"])
    return _write_text(path, "\n".join(lines))


def write_experiment_runtime_report(
    reports: Sequence[ProcessSchedulerReport],
    path: str | Path,
) -> Path:
    """Aggregate shared-bank cache, process, CPU, wall time, and failure metrics."""
    rows: list[dict[str, object]] = []
    for report in reports:
        for completed in report.completed_tasks:
            executions = completed.result.get("executions", [])
            if not isinstance(executions, list):
                continue
            for execution in executions:
                if not isinstance(execution, dict):
                    continue
                rows.append(
                    {
                        "task_id": completed.task_id,
                        "experiment_id": execution.get("experiment_id"),
                        "phase": execution.get("phase"),
                        "seed": execution.get("seed"),
                        "period_count": execution.get("period_count"),
                        "candidate_cache_hit_rate": execution.get("candidate_cache_hit_rate"),
                        "feasible_bank_generation_seconds": execution.get(
                            "feasible_bank_generation_seconds"
                        ),
                        "portfolio_bank_reuse_count": execution.get("portfolio_bank_reuse_count"),
                        "experiment_throughput_per_hour": execution.get(
                            "experiment_throughput_per_hour"
                        ),
                        "estimated_remaining_runtime_seconds": execution.get(
                            "estimated_remaining_runtime_seconds"
                        ),
                        "process_count": report.process_count,
                        "hostname": report.hostname,
                        "started_at": report.started_at,
                        "ended_at": report.ended_at,
                        "total_cpu_time_seconds": execution.get("total_cpu_time_seconds"),
                        "wall_clock_seconds": execution.get("wall_clock_seconds"),
                        "candidate_array_conversion_seconds": execution.get(
                            "candidate_array_conversion_seconds"
                        ),
                        "portfolio_index_bank_seconds": execution.get(
                            "portfolio_index_bank_seconds"
                        ),
                        "vectorized_scoring_seconds": execution.get("vectorized_scoring_seconds"),
                        "final_object_construction_seconds": execution.get(
                            "final_object_construction_seconds"
                        ),
                        "failed_task_count": len(report.failed_tasks),
                    }
                )
    lines = [
        "# v0.5.1 实验计算性能",
        "",
        "该报告来自追加保存的 ProcessPool 运行元数据；未执行的任务不会生成虚构指标。",
        "",
    ]
    if rows:
        lines.append(_markdown_table(pd.DataFrame.from_records(rows)))
    else:
        lines.append("尚无已完成的 v0.5.1 ProcessPool 实验任务。")
    lines.extend(["", f"> {DISCLAIMER}"])
    return _write_text(path, "\n".join(lines))

"""Markdown and CSV reports that separate fit, calibration, and final holdout."""

from __future__ import annotations

from collections.abc import Sequence
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
)


class ExperimentComparison(BaseModel):
    """One paired, baseline-relative metric conclusion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str
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


def summarize_observations(observations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate key metrics independently inside each declared data phase."""
    if observations.empty:
        return pd.DataFrame()
    numeric = observations.copy()
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
        numeric.groupby(["phase", "experiment_id"], sort=True, dropna=False)
        .agg(
            observation_count=("target_issue", "size"),
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
    for phase, phase_frame in observations.groupby("phase", sort=True):
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
    for phase in sorted({comparison.phase for comparison in comparisons}):
        family = [comparison for comparison in comparisons if comparison.phase == phase]
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
    """Write phase-separated results and paired-inference conclusions."""
    summary = summarize_observations(observations)
    lines = [
        "# 严格策略实验汇总",
        "",
        "本报告将历史拟合、校准和最终留出结果分开；随机波动不会被解释为预测能力。",
    ]
    phase_titles = {
        "development": "历史拟合（development）",
        "calibration": "校准（calibration）",
        "final_holdout": "最终留出（final holdout）",
    }
    for phase, title in phase_titles.items():
        lines.extend(["", f"## {title}", ""])
        phase_summary = summary.loc[summary.get("phase", pd.Series(dtype=str)) == phase]
        if phase_summary.empty:
            lines.append("尚未执行或没有可用观测。")
        else:
            lines.append(_markdown_table(phase_summary))
    lines.extend(["", "## 与 B1 约束匹配随机基线的配对比较", ""])
    if comparisons:
        comparison_frame = pd.DataFrame(
            [comparison.model_dump(mode="python") for comparison in comparisons]
        )
        lines.append(_markdown_table(comparison_frame))
        if inference_context == "smoke":
            lines.extend(
                [
                    "",
                    (
                        "本表是配对计算冒烟检查；不使用这 100 期选择参数，"
                        "也不根据本表声明任何策略优势。"
                    ),
                ]
            )
        else:
            significant = [
                comparison
                for comparison in comparisons
                if comparison.statistically_significant_advantage
            ]
            lines.extend(
                [
                    "",
                    (
                        "结论仅依据配对 bootstrap 与配对置换检验；"
                        "没有使用两个独立置信区间是否重叠来判断优势；"
                        "正式显著性结论采用更保守的 Holm 校正。"
                    ),
                    f"显著优势条目数：{len(significant)}。",
                ]
            )
    else:
        lines.append("缺少至少 30 个同一期次、同一种子的 B1 配对结果，暂不能进行统计比较。")
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

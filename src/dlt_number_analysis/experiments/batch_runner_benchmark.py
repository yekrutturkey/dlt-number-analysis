"""Actively bounded v0.5.4 benchmark of the production shared B1-B6 Runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, median, pstdev
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import (
    canonical_history_sha256,
    load_prize_rule_schedule,
    load_verified_history,
)
from dlt_number_analysis.experiments.benchmark_checkpoints import (
    IsolatedStageOutcome,
    atomic_write_bytes,
    atomic_write_json,
    load_checkpoint,
    run_isolated_stage,
)
from dlt_number_analysis.experiments.runner import TargetExecutionTiming, run_experiment_batch
from dlt_number_analysis.experiments.specs import baseline_experiment_specs

V054_BENCHMARK_VERSION = "v0.5.4-production-runner-v1"
V054_CHECKPOINT_SCHEMA = "v0.5.4"
V054_SEED = 20260000
V054_TARGET_ISSUES = tuple(f"{issue:05d}" for issue in range(8009, 8019))
V054_EXPERIMENT_IDS = tuple(f"B{index}" for index in range(1, 7))
V054_TIMEOUT_SECONDS = 300.0
DEVELOPMENT_TARGET_COUNT = 1637
V054_PRIMARY_KEY = (
    "benchmark_version",
    "experiment_id",
    "experiment_version",
    "phase",
    "seed",
    "target_issue",
)

TIMING_COLUMNS = (
    "history_slice_seconds",
    "candidate_generation_seconds",
    "candidate_array_conversion_seconds",
    "bank_generation_seconds",
    "portfolio_index_bank_seconds",
    "b1_sampling_seconds",
    "score_view_seconds",
    "vectorized_scoring_seconds",
    "final_object_construction_seconds",
    "raw_evaluation_seconds",
    "observation_serialization_seconds",
    "checkpoint_write_seconds",
    "target_total_seconds",
)


class V054BenchmarkConfig(BaseModel):
    """Closed benchmark configuration that rejects accidental scope expansion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issues: tuple[str, ...] = V054_TARGET_ISSUES
    experiment_ids: tuple[str, ...] = V054_EXPERIMENT_IDS
    seed: int = V054_SEED
    profile: Literal["fast"] = "fast"
    portfolio_scoring_method: Literal["numpy_vectorized"] = "numpy_vectorized"
    evaluation_mode: Literal["raw_observation"] = "raw_observation"
    process_count: Literal[1] = 1
    minimum_history: int = 100
    minimum_bank_size: int = 500
    maximum_bank_search_trials: int = 80_000

    @model_validator(mode="after")
    def enforce_scope(self) -> V054BenchmarkConfig:
        if self.target_issues != V054_TARGET_ISSUES:
            raise ValueError("v0.5.4 benchmark permits only targets 08009 through 08018")
        if self.experiment_ids != V054_EXPERIMENT_IDS:
            raise ValueError("v0.5.4 benchmark permits only B1-B6")
        if self.seed != V054_SEED:
            raise ValueError("v0.5.4 benchmark permits only seed 20260000")
        return self

    @property
    def signature(self) -> str:
        """Hash the complete closed benchmark configuration."""
        payload = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class V054BenchmarkPaths:
    """All v0.5.4 artifacts, physically isolated from formal experiment partitions."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints = self.output_dir / "checkpoints"
        self.controller = self.checkpoints / "controller.json"
        self.observations = self.output_dir / "benchmark_observations.parquet"
        self.execution = self.output_dir / "benchmark_execution.json"
        self.timings = self.output_dir / "per_target_timings.csv"
        self.summary = self.output_dir / "benchmark_summary.json"
        self.summary_markdown = self.output_dir / "benchmark_summary.md"

    def target_parquet(self, target_issue: str) -> Path:
        return self.checkpoints / f"target_{target_issue}.parquet"

    def target_manifest(self, target_issue: str) -> Path:
        return self.checkpoints / f"target_{target_issue}.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _history_prefix(draws: pd.DataFrame, target_issue: str) -> pd.DataFrame:
    matches = draws.index[draws["issue"].astype(str) == target_issue].tolist()
    if len(matches) != 1:
        raise ValueError(f"benchmark target must exist exactly once: {target_issue}")
    target_index = int(matches[0])
    if target_index < 1:
        raise ValueError("benchmark target has no pre-target history")
    return draws.iloc[:target_index].copy()


def _checkpoint_expected_identity(
    draws: pd.DataFrame,
    target_issue: str,
    config: V054BenchmarkConfig,
) -> dict[str, object]:
    return {
        "schema_version": V054_CHECKPOINT_SCHEMA,
        "benchmark_version": V054_BENCHMARK_VERSION,
        "target_issue": target_issue,
        "seed": config.seed,
        "evaluation_mode": config.evaluation_mode,
        "benchmark_config_sha256": config.signature,
        "history_prefix_sha256": canonical_history_sha256(_history_prefix(draws, target_issue)),
    }


def load_v054_target_checkpoint(
    draws: pd.DataFrame,
    paths: V054BenchmarkPaths,
    target_issue: str,
    config: V054BenchmarkConfig,
) -> tuple[pd.DataFrame, TargetExecutionTiming]:
    """Validate all hashes and primary keys before reusing a target checkpoint."""
    manifest_path = paths.target_manifest(target_issue)
    parquet_path = paths.target_parquet(target_issue)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = _checkpoint_expected_identity(draws, target_issue, config)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"v0.5.4 checkpoint identity mismatch for {key}: {target_issue}")
    if not parquet_path.exists() or _sha256_file(parquet_path) != manifest.get("parquet_sha256"):
        raise ValueError(f"v0.5.4 checkpoint Parquet hash mismatch: {target_issue}")
    frame = pd.read_parquet(parquet_path)
    if len(frame) != len(V054_EXPERIMENT_IDS):
        raise ValueError(f"v0.5.4 checkpoint must contain six observations: {target_issue}")
    missing = set(V054_PRIMARY_KEY).difference(frame.columns)
    if missing:
        raise ValueError(f"v0.5.4 checkpoint primary key fields missing: {sorted(missing)}")
    if frame.duplicated(list(V054_PRIMARY_KEY)).any():
        raise ValueError(f"v0.5.4 checkpoint contains duplicate primary keys: {target_issue}")
    if set(frame["experiment_id"].astype(str)) != set(V054_EXPERIMENT_IDS):
        raise ValueError(f"v0.5.4 checkpoint experiment IDs differ: {target_issue}")
    if set(frame["experiment_config_sha256"].astype(str)) != set(
        manifest.get("experiment_config_sha256", {}).values()
    ):
        raise ValueError(f"v0.5.4 experiment config hash mismatch: {target_issue}")
    if frame["candidate_numbers_hash"].nunique() != 1:
        raise ValueError(f"v0.5.4 candidate hash differs within target: {target_issue}")
    if frame["bank_hash"].nunique() != 1:
        raise ValueError(f"v0.5.4 bank hash differs within target: {target_issue}")
    if str(frame.iloc[0]["candidate_numbers_hash"]) != manifest.get("candidate_numbers_hash"):
        raise ValueError(f"v0.5.4 candidate hash differs from manifest: {target_issue}")
    if str(frame.iloc[0]["bank_hash"]) != manifest.get("bank_hash"):
        raise ValueError(f"v0.5.4 bank hash differs from manifest: {target_issue}")
    timing = TargetExecutionTiming.model_validate(manifest["timing"])
    return frame, timing


def write_v054_target_checkpoint(
    draws: pd.DataFrame,
    paths: V054BenchmarkPaths,
    target_issue: str,
    config: V054BenchmarkConfig,
    observations: pd.DataFrame,
    timing: TargetExecutionTiming,
) -> TargetExecutionTiming:
    checkpoint_started = perf_counter()
    frame = observations.assign(benchmark_version=V054_BENCHMARK_VERSION)
    if frame.duplicated(list(V054_PRIMARY_KEY)).any():
        raise ValueError(f"duplicate v0.5.4 benchmark primary key: {target_issue}")
    parquet_path = paths.target_parquet(target_issue)
    _atomic_write_parquet(frame, parquet_path)
    initial_manifest = {
        **_checkpoint_expected_identity(draws, target_issue, config),
        "parquet_sha256": _sha256_file(parquet_path),
        "candidate_numbers_hash": str(frame.iloc[0]["candidate_numbers_hash"]),
        "bank_hash": str(frame.iloc[0]["bank_hash"]),
        "experiment_config_sha256": dict(
            zip(
                frame["experiment_id"].astype(str),
                frame["experiment_config_sha256"].astype(str),
                strict=True,
            )
        ),
        "primary_key": list(V054_PRIMARY_KEY),
        "timing": timing.model_dump(mode="json"),
    }
    atomic_write_json(paths.target_manifest(target_issue), initial_manifest)
    checkpoint_seconds = perf_counter() - checkpoint_started
    updated = timing.model_copy(
        update={
            "checkpoint_write_seconds": checkpoint_seconds,
            "target_total_seconds": timing.target_total_seconds + checkpoint_seconds,
        }
    )
    atomic_write_json(
        paths.target_manifest(target_issue),
        {**initial_manifest, "timing": updated.model_dump(mode="json")},
    )
    return updated


def v054_batch_runner_worker(
    *,
    history_path: str,
    prize_rule_path: str,
    output_dir: str,
    config_payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Run each allowed target through production ``run_experiment_batch`` once."""
    started_at = datetime.now(UTC)
    draws = load_verified_history(history_path)
    config = V054BenchmarkConfig.model_validate(config_payload)
    paths = V054BenchmarkPaths(output_dir)
    paths.checkpoints.mkdir(parents=True, exist_ok=True)
    tables = load_prize_rule_schedule(prize_rule_path).tables
    specifications = tuple(
        spec
        for spec in baseline_experiment_specs(seeds=(config.seed,), phase="development")
        if spec.experiment_id in config.experiment_ids
    )
    completed: list[str] = []
    skipped: list[str] = []
    for target_issue in config.target_issues:
        manifest_path = paths.target_manifest(target_issue)
        if manifest_path.exists():
            load_v054_target_checkpoint(draws, paths, target_issue, config)
            skipped.append(target_issue)
            continue
        result = run_experiment_batch(
            draws,
            specifications,
            profile=config.profile,
            minimum_history=config.minimum_history,
            random_baseline_seed_count=1000,
            bootstrap_resamples=1000,
            prize_tables=tables,
            target_issues=(target_issue,),
            minimum_bank_size=config.minimum_bank_size,
            maximum_bank_search_trials=config.maximum_bank_search_trials,
            portfolio_scoring_method=config.portfolio_scoring_method,
            evaluation_mode=config.evaluation_mode,
        )
        if len(result.observations) != len(V054_EXPERIMENT_IDS):
            raise ValueError(f"production Runner did not return six rows: {target_issue}")
        if len(result.target_timings) != 1:
            raise ValueError(f"production Runner did not return one target timing: {target_issue}")
        write_v054_target_checkpoint(
            draws,
            paths,
            target_issue,
            config,
            result.observations,
            result.target_timings[0],
        )
        completed.append(target_issue)
    return {
        "benchmark_version": V054_BENCHMARK_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "process_count": config.process_count,
        "completed_target_issues": completed,
        "skipped_target_issues": skipped,
        "risk_disclaimer": DISCLAIMER,
    }


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "median": float(median(values)),
        "mean": float(fmean(values)),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "maximum": float(array.max()),
        "standard_deviation": float(pstdev(values)),
    }


def _autodl_decision(mean_target_seconds: float) -> dict[str, str | bool]:
    if mean_target_seconds <= 4:
        return {
            "band": "at_or_below_4_seconds",
            "local_run": "本地串行可行，预计约数小时。",
            "autodl": "暂不需要AutoDL。",
            "enter_full_development": True,
        }
    if mean_target_seconds <= 8:
        return {
            "band": "above_4_to_8_seconds",
            "local_run": "本地夜间串行运行可行。",
            "autodl": "AutoDL可选但不是必要。",
            "enter_full_development": True,
        }
    if mean_target_seconds <= 15:
        return {
            "band": "above_8_to_15_seconds",
            "local_run": "先测试多进程，不立即运行完整development。",
            "autodl": "待多进程基准后再判断。",
            "enter_full_development": False,
        }
    return {
        "band": "above_15_seconds",
        "local_run": "继续profiling，不运行完整development。",
        "autodl": "可在进一步profiling后评估。",
        "enter_full_development": False,
    }


def aggregate_v054_benchmark(
    history_path: str | Path,
    output_dir: str | Path,
    *,
    config: V054BenchmarkConfig | None = None,
) -> dict[str, object]:
    """Build reports solely from validated raw target checkpoints; generate no predictions."""
    active = config or V054BenchmarkConfig()
    draws = load_verified_history(history_path)
    paths = V054BenchmarkPaths(output_dir)
    frames: list[pd.DataFrame] = []
    timings: list[TargetExecutionTiming] = []
    for target_issue in active.target_issues:
        if not paths.target_manifest(target_issue).exists():
            continue
        frame, timing = load_v054_target_checkpoint(draws, paths, target_issue, active)
        frames.append(frame)
        timings.append(timing)
    observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not observations.empty and observations.duplicated(list(V054_PRIMARY_KEY)).any():
        raise ValueError("aggregated v0.5.4 observations contain duplicate primary keys")
    _atomic_write_parquet(observations, paths.observations)
    timing_frame = pd.DataFrame([item.model_dump(mode="python") for item in timings])
    atomic_write_bytes(
        paths.timings, timing_frame.to_csv(index=False, lineterminator="\n").encode()
    )
    target_values = [item.target_total_seconds for item in timings]
    target_distribution = _distribution(target_values) if target_values else None
    stage_distributions = {
        column: _distribution([float(getattr(item, column)) for item in timings])
        for column in TIMING_COLUMNS
        if timings
    }
    measured_count = len(timings)
    extrapolation: dict[str, object]
    if measured_count >= 5 and target_distribution is not None:
        mean_seconds = target_distribution["mean"]
        p90_seconds = target_distribution["p90"]
        extrapolation = {
            "kind": "estimated",
            "development_target_count": DEVELOPMENT_TARGET_COUNT,
            "estimated_development_serial_seconds": mean_seconds * DEVELOPMENT_TARGET_COUNT,
            "conservative_development_serial_seconds": (
                p90_seconds * DEVELOPMENT_TARGET_COUNT * 1.25
            ),
        }
        decision = _autodl_decision(mean_seconds)
    else:
        extrapolation = {
            "kind": "unavailable",
            "development_target_count": DEVELOPMENT_TARGET_COUNT,
            "reason": "at least five measured targets are required",
        }
        decision = {
            "band": "unavailable",
            "local_run": "实测期次不足，无法判断。",
            "autodl": "实测期次不足，无法判断。",
            "enter_full_development": False,
        }
    summary: dict[str, object] = {
        "benchmark_version": V054_BENCHMARK_VERSION,
        "benchmark_config_sha256": active.signature,
        "evaluation_mode": active.evaluation_mode,
        "measurement_kind": "measured",
        "requested_target_count": len(active.target_issues),
        "measured_target_count": measured_count,
        "measured_total_seconds": float(sum(target_values)),
        "completed_target_issues": [item.target_issue for item in timings],
        "target_total_seconds_distribution": target_distribution,
        "stage_distributions": stage_distributions,
        "extrapolation": extrapolation,
        "engineering_decision": decision,
        "unrun_experiments": [
            "object_reference",
            "20/100 target benchmarks",
            "full development",
            "360 ablations",
            "calibration",
            "final holdout",
            "multiple seeds",
            "multi-process benchmark",
        ],
        "risk_disclaimer": DISCLAIMER,
    }
    atomic_write_json(paths.summary, summary)
    markdown = render_v054_benchmark_markdown(summary, timing_frame)
    atomic_write_bytes(paths.summary_markdown, markdown.encode("utf-8"))
    return summary


def render_v054_benchmark_markdown(
    summary: Mapping[str, object],
    timings: pd.DataFrame,
) -> str:
    """Render the required measured/estimated boundary and timing audit."""
    lines = [
        "# v0.5.4 正式批量 Runner 10 期性能验证",
        "",
        "## 运行范围",
        "",
        "仅运行 08009–08018、B1–B6、seed=20260000、fast、numpy_vectorized、"
        "raw_observation、单进程。未运行正式历史实验。",
        "",
        f"完成目标期：{summary['measured_target_count']}/10。",
        f"5 分钟主动停止条件是否触发：{bool(summary.get('timed_out', False))}。",
        "",
        "raw_observation 仅计算已生成 PredictionRecord 的直接命中和可用历史奖金；"
        "它完全跳过随机 Monte Carlo、百分位、Bootstrap 和策略汇总。"
        "full_resampling 才适用于最终正式统计报告。",
        "",
        "## 每期完整耗时",
        "",
    ]
    if timings.empty:
        lines.append("尚无完成目标期。")
    else:
        lines.extend(["| target_issue | seconds |", "|---|---:|"])
        for row in timings.to_dict(orient="records"):
            lines.append(f"| {row['target_issue']} | {float(row['target_total_seconds']):.6f} |")
    distribution = summary.get("target_total_seconds_distribution")
    lines.extend(["", "## 汇总与外推", ""])
    if isinstance(distribution, dict):
        lines.append(
            "实测 target_total_seconds："
            f"最小 {distribution['minimum']:.6f}s；中位 {distribution['median']:.6f}s；"
            f"平均 {distribution['mean']:.6f}s；P90 {distribution['p90']:.6f}s；"
            f"最大 {distribution['maximum']:.6f}s；标准差 "
            f"{distribution['standard_deviation']:.6f}s。"
        )
    extrapolation = summary["extrapolation"]
    if isinstance(extrapolation, dict) and extrapolation.get("kind") == "estimated":
        lines.append(
            "完整 development 串行中心外推（估算，不是实测）："
            f"{float(extrapolation['estimated_development_serial_seconds']):.2f}s；"
            "保守外推（估算）："
            f"{float(extrapolation['conservative_development_serial_seconds']):.2f}s。"
        )
    else:
        lines.append("完成少于 5 期，完整 development 外推不可用。")
    measured_wall = summary.get("controller_elapsed_seconds")
    if measured_wall is not None:
        lines.append(f"包含子进程启动与汇总控制的实测墙钟时间：{float(measured_wall):.3f}s。")
    stage_distributions = summary.get("stage_distributions")
    if isinstance(stage_distributions, dict) and stage_distributions:
        lines.extend(
            [
                "",
                "## 各阶段耗时汇总",
                "",
                "| stage | mean_seconds | p90_seconds | maximum_seconds |",
                "|---|---:|---:|---:|",
            ]
        )
        for stage, values in stage_distributions.items():
            if not isinstance(values, dict):
                continue
            lines.append(
                f"| {stage} | {float(values['mean']):.6f} | "
                f"{float(values['p90']):.6f} | {float(values['maximum']):.6f} |"
            )
        ranked = sorted(
            (
                (stage, float(values["mean"]))
                for stage, values in stage_distributions.items()
                if stage != "target_total_seconds" and isinstance(values, dict)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            top = "、".join(f"{stage} ({value:.3f}s)" for stage, value in ranked[:3])
            lines.append(f"当前主要瓶颈（按每期平均实测耗时）：{top}。")
    decision = summary["engineering_decision"]
    if isinstance(decision, dict):
        lines.extend(
            [
                "",
                "## 工程判断",
                "",
                f"本地运行：{decision['local_run']}",
                f"AutoDL：{decision['autodl']}",
                f"是否进入完整 development：{decision['enter_full_development']}。",
            ]
        )
    lines.extend(
        [
            "",
            "潜在瓶颈：完整历史窗口增长可能增加候选结构评分成本；银行接受率变化可能触发"
            "search_trials 扩展。本轮未据 10 期结果修改策略参数。",
            "",
            "未运行：object_reference、20/100 期、完整 development、360 组消融、"
            "calibration、final holdout、多 seed、多进程基准。",
            "",
            f"> 风险声明：{DISCLAIMER}",
            "",
        ]
    )
    return "\n".join(lines)


def run_v054_batch_runner_benchmark(
    history_path: str | Path,
    prize_rule_path: str | Path,
    output_dir: str | Path,
    *,
    timeout_seconds: float = V054_TIMEOUT_SECONDS,
) -> IsolatedStageOutcome:
    """Run the one permitted benchmark under an active five-minute hard deadline."""
    if timeout_seconds <= 0 or timeout_seconds > V054_TIMEOUT_SECONDS:
        raise ValueError("v0.5.4 benchmark timeout must be in (0, 300] seconds")
    config = V054BenchmarkConfig()
    draws = load_verified_history(history_path)
    paths = V054BenchmarkPaths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    for target_issue in config.target_issues:
        if paths.target_manifest(target_issue).exists():
            load_v054_target_checkpoint(draws, paths, target_issue, config)
    history_sha = canonical_history_sha256(draws)
    outcome = run_isolated_stage(
        stage="v054_production_runner_10_targets",
        worker_path=(
            "dlt_number_analysis.experiments.batch_runner_benchmark:v054_batch_runner_worker"
        ),
        worker_kwargs={
            "history_path": str(Path(history_path).resolve()),
            "prize_rule_path": str(Path(prize_rule_path).resolve()),
            "output_dir": str(paths.output_dir.resolve()),
            "config_payload": config.model_dump(mode="json"),
        },
        checkpoint_path=paths.controller,
        checkpoint_identity={
            "benchmark_version": V054_BENCHMARK_VERSION,
            "benchmark_config_sha256": config.signature,
            "history_sha256": history_sha,
        },
        timeout_seconds=timeout_seconds,
    )
    controller = load_checkpoint(
        paths.controller,
        expected_identity={
            "benchmark_version": V054_BENCHMARK_VERSION,
            "benchmark_config_sha256": config.signature,
            "history_sha256": history_sha,
        },
    )
    aggregate_v054_benchmark(history_path, output_dir, config=config)
    execution = {
        "benchmark_version": V054_BENCHMARK_VERSION,
        "benchmark_config_sha256": config.signature,
        "history_sha256": history_sha,
        "status": outcome.status,
        "timed_out": outcome.status == "timeout",
        "active_timeout_seconds": timeout_seconds,
        "controller_elapsed_seconds": outcome.elapsed_seconds,
        "process_exit_status": outcome.process_exit_status,
        "controller": controller,
        "risk_disclaimer": DISCLAIMER,
    }
    atomic_write_json(paths.execution, execution)
    return outcome


def validate_v054_resume(
    history_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Read and aggregate checkpoints only; never start candidates or Portfolio banks."""
    return aggregate_v054_benchmark(history_path, output_dir, config=V054BenchmarkConfig())


def write_v054_report(
    history_path: str | Path,
    output_dir: str | Path,
    report_path: str | Path,
) -> Path:
    """Write reports from existing summaries only; perform no resume or generation work."""
    del history_path
    paths = V054BenchmarkPaths(output_dir)
    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
    if paths.execution.exists():
        execution = json.loads(paths.execution.read_text(encoding="utf-8"))
        summary.update(
            {
                "benchmark_status": execution.get("status"),
                "timed_out": execution.get("timed_out"),
                "controller_elapsed_seconds": execution.get("controller_elapsed_seconds"),
            }
        )
        atomic_write_json(paths.summary, summary)
    timing_frame = (
        pd.read_csv(paths.timings, dtype={"target_issue": "string"})
        if paths.timings.exists()
        else pd.DataFrame()
    )
    content = render_v054_benchmark_markdown(summary, timing_frame)
    atomic_write_bytes(paths.summary_markdown, content.encode("utf-8"))
    report = Path(report_path)
    atomic_write_bytes(report, content.encode("utf-8"))
    return report

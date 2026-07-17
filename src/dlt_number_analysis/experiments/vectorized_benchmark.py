"""Legacy v0.5.2 benchmark plus v0.5.3 staged-benchmark compatibility exports."""

from __future__ import annotations

import ctypes
import gc
import importlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, time
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.data import load_verified_history
from dlt_number_analysis.experiments.shared_computation import (
    SharedComputationCache,
    deterministic_subseed,
)
from dlt_number_analysis.experiments.specs import baseline_experiment_specs
from dlt_number_analysis.experiments.staged_benchmark import (
    PreparedBenchmarkPayload as PreparedBenchmarkPayload,
)
from dlt_number_analysis.experiments.staged_benchmark import (
    V053BenchmarkPaths as V053BenchmarkPaths,
)
from dlt_number_analysis.experiments.staged_benchmark import (
    run_v053_staged_benchmark as run_v053_staged_benchmark,
)
from dlt_number_analysis.experiments.staged_benchmark import (
    write_v053_benchmark_report as write_v053_benchmark_report,
)
from dlt_number_analysis.experiments.staged_benchmark import (
    write_v053_benchmark_summary as write_v053_benchmark_summary,
)
from dlt_number_analysis.pipeline import PROFILE_DEFAULTS, PipelineConfig
from dlt_number_analysis.portfolio import (
    build_candidate_score_view,
    candidate_pool_to_array_bundle,
    feasible_portfolio_bank_to_index_bank,
    portfolio_constraints_signature,
    rescore_candidate_pool,
    sample_constraint_matched_portfolio,
    score_portfolio_bank,
    score_portfolio_bank_vectorized,
)
from dlt_number_analysis.portfolio.models import PortfolioSelection


class VectorizedTargetBenchmark(BaseModel):
    """Stage timings and semantic equality for one historical target issue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_issue: str
    data_cutoff_issue: str
    candidate_count: int = Field(ge=1)
    bank_size: int = Field(ge=1)
    candidate_generation_seconds: float = Field(ge=0)
    feasible_bank_generation_seconds: float = Field(ge=0)
    candidate_array_conversion_seconds: float = Field(ge=0)
    index_bank_build_seconds: float = Field(ge=0)
    object_reference_total_seconds: float = Field(ge=0)
    numpy_score_view_seconds: float = Field(ge=0)
    numpy_vectorized_scoring_seconds: float = Field(ge=0)
    final_object_construction_seconds: float = Field(ge=0)
    numpy_vectorized_total_seconds: float = Field(gt=0)
    object_reference_peak_memory_mb: float = Field(ge=0)
    numpy_vectorized_peak_memory_mb: float = Field(ge=0)
    speedup: float = Field(gt=0)
    b1_entry_unchanged: bool
    all_final_tickets_identical: bool
    risk_disclaimer: str = DISCLAIMER


class VectorizedBenchmarkRun(BaseModel):
    """One bounded worker-count run over no more than three targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workers: int = Field(ge=1, le=4)
    target_issues: tuple[str, ...] = Field(min_length=1, max_length=3)
    started_at: datetime
    ended_at: datetime
    wall_clock_seconds: float = Field(gt=0)
    average_seconds_per_target: float = Field(gt=0)
    targets: tuple[VectorizedTargetBenchmark, ...]
    risk_disclaimer: str = DISCLAIMER


def _generated_at(draw_date: object) -> datetime:
    value = pd.Timestamp(draw_date).date()
    return datetime.combine(value, time(23, 59), tzinfo=ZoneInfo("Asia/Shanghai"))


def _ticket_identities(
    selection: PortfolioSelection,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    return tuple((ticket.front_numbers, ticket.back_numbers) for ticket in selection.tickets)


def _peak_process_memory_mb() -> float:
    """Return OS process peak working-set/RSS without instrumenting timed code."""
    if os.name == "nt":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = (
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            )

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
        return counters.peak_working_set_size / (1024 * 1024)
    resource = importlib.import_module("resource")
    peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak_rss / 1024


def benchmark_vectorized_target(
    draws: pd.DataFrame,
    *,
    target_issue: str,
    seed: int = 20260000,
) -> VectorizedTargetBenchmark:
    """Benchmark only scoring stages on one strict pre-target history prefix."""
    issue_values = draws["issue"].astype(str).tolist()
    try:
        target_index = issue_values.index(target_issue)
    except ValueError as error:
        raise ValueError(f"unknown benchmark target issue: {target_issue}") from error
    if target_index < 100:
        raise ValueError("benchmark target requires at least 100 prior draws")
    history = draws.iloc[:target_index].copy()
    generated_at = _generated_at(history.iloc[-1]["draw_date"])
    specs = baseline_experiment_specs(seeds=(seed,), phase="development")[1:7]
    constraints = specs[0].portfolio_constraints
    if any(spec.portfolio_constraints != constraints for spec in specs):
        raise ValueError("B1-B6 benchmark requires identical Portfolio constraints")
    defaults = PROFILE_DEFAULTS["fast"]
    cache = SharedComputationCache()
    candidate_started = perf_counter()
    pool = cache.get_candidate_pool(
        history,
        target_issue=target_issue,
        generated_at=generated_at,
        candidate_seed=deterministic_subseed(seed, target_issue, "candidates"),
        candidate_count=defaults["candidate_count"],
    )
    candidate_seconds = perf_counter() - candidate_started
    signature = portfolio_constraints_signature(constraints)
    bank = cache.get_portfolio_bank(
        pool,
        bank_seed=deterministic_subseed(seed, target_issue, "bank", signature),
        constraints=constraints,
        search_trials=defaults["optimizer_search_trials"],
        maximum_bank_size=min(2_000, defaults["optimizer_search_trials"]),
        minimum_bank_size=500,
        maximum_search_trials=80_000,
    )
    array_started = perf_counter()
    arrays = candidate_pool_to_array_bundle(pool)
    array_seconds = perf_counter() - array_started
    index_started = perf_counter()
    index_bank = feasible_portfolio_bank_to_index_bank(arrays, bank)
    index_seconds = perf_counter() - index_started
    objective = PipelineConfig(profile="fast")
    b1_seed = deterministic_subseed(seed, target_issue, "B1", "selection")
    b1_first = sample_constraint_matched_portfolio(pool, bank, random_seed=b1_seed)
    b1_second = sample_constraint_matched_portfolio(pool, bank, random_seed=b1_seed)

    vector_identities: dict[str, tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] = {}
    vector_hashes: dict[str, str] = {}
    vector_scores: dict[str, dict[str, float]] = {}
    view_seconds = 0.0
    vector_score_seconds = 0.0
    final_object_seconds = 0.0
    vector_started = perf_counter()
    for spec in specs[1:]:
        view_started = perf_counter()
        score_view = build_candidate_score_view(
            history,
            arrays,
            scorer_spec=spec.scorer_spec,
            number_score_weight=spec.number_score_weight,
            structure_score_weight=spec.structure_score_weight,
        )
        view_seconds += perf_counter() - view_started
        vectorized = score_portfolio_bank_vectorized(
            score_view,
            index_bank,
            random_seed=deterministic_subseed(seed, target_issue, spec.experiment_id, "selection"),
            single_ticket_weight=objective.single_ticket_weight,
            diversity_weight=objective.diversity_weight,
            core_weight=objective.core_weight,
            structure_weight=objective.structure_weight,
            repeat_penalty_weight=objective.repeat_penalty_weight,
        )
        vector_score_seconds += vectorized.vectorized_scoring_seconds
        final_object_seconds += vectorized.final_object_construction_seconds
        vector_identities[spec.experiment_id] = _ticket_identities(vectorized.selection)
        vector_hashes[spec.experiment_id] = vectorized.selected_entry_hash
        vector_scores[spec.experiment_id] = vectorized.selection.scores.model_dump()
    vector_path_seconds = perf_counter() - vector_started
    vector_peak = _peak_process_memory_mb()

    all_identical = True
    object_started = perf_counter()
    for spec in specs[1:]:
        materialized = rescore_candidate_pool(
            history,
            pool,
            scorer_spec=spec.scorer_spec,
            number_score_weight=spec.number_score_weight,
            structure_score_weight=spec.structure_score_weight,
        )
        reference = score_portfolio_bank(
            materialized,
            bank,
            random_seed=deterministic_subseed(seed, target_issue, spec.experiment_id, "selection"),
            single_ticket_weight=objective.single_ticket_weight,
            diversity_weight=objective.diversity_weight,
            core_weight=objective.core_weight,
            structure_weight=objective.structure_weight,
            repeat_penalty_weight=objective.repeat_penalty_weight,
        )
        score_match = all(
            abs(float(value) - float(vector_scores[spec.experiment_id][key])) <= 1e-12
            for key, value in reference.selection.scores.model_dump().items()
        )
        all_identical = all_identical and (
            str(reference.selection.optimizer_parameters["selected_bank_entry_hash"])
            == vector_hashes[spec.experiment_id]
            and _ticket_identities(reference.selection) == vector_identities[spec.experiment_id]
            and score_match
        )
        del materialized, reference
        gc.collect()
    object_seconds = perf_counter() - object_started
    object_peak = _peak_process_memory_mb()
    vector_total = array_seconds + index_seconds + vector_path_seconds
    return VectorizedTargetBenchmark(
        target_issue=target_issue,
        data_cutoff_issue=str(history.iloc[-1]["issue"]),
        candidate_count=len(pool.candidates),
        bank_size=bank.bank_size,
        candidate_generation_seconds=candidate_seconds,
        feasible_bank_generation_seconds=bank.generation_seconds,
        candidate_array_conversion_seconds=array_seconds,
        index_bank_build_seconds=index_seconds,
        object_reference_total_seconds=object_seconds,
        numpy_score_view_seconds=view_seconds,
        numpy_vectorized_scoring_seconds=vector_score_seconds,
        final_object_construction_seconds=final_object_seconds,
        numpy_vectorized_total_seconds=vector_total,
        object_reference_peak_memory_mb=object_peak,
        numpy_vectorized_peak_memory_mb=vector_peak,
        speedup=object_seconds / vector_total,
        b1_entry_unchanged=(
            b1_first.optimizer_parameters["selected_bank_entry_hash"]
            == b1_second.optimizer_parameters["selected_bank_entry_hash"]
        ),
        all_final_tickets_identical=all_identical,
    )


def _benchmark_target_from_path(
    history_path: str,
    target_issue: str,
    seed: int,
) -> VectorizedTargetBenchmark:
    draws = load_verified_history(history_path)
    return benchmark_vectorized_target(draws, target_issue=target_issue, seed=seed)


def run_vectorized_benchmark(
    history_path: str | Path,
    *,
    target_issues: tuple[str, ...] = ("08009", "08010", "08011"),
    seed: int = 20260000,
    workers: int = 1,
    checkpoint_dir: str | Path | None = None,
) -> VectorizedBenchmarkRun:
    """Run no more than three target benchmarks with a bounded process count."""
    if not 1 <= len(target_issues) <= 3:
        raise ValueError("vectorized benchmark permits only one to three targets")
    if workers not in (1, 2):
        raise ValueError("bounded vectorized benchmark permits only one or two workers")
    path = str(Path(history_path).resolve())
    checkpoint_root = None if checkpoint_dir is None else Path(checkpoint_dir)

    def checkpoint(target: VectorizedTargetBenchmark) -> None:
        if checkpoint_root is None:
            return
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root / (f"workers_{workers}_target_{target.target_issue}.json")
        checkpoint_path.write_text(
            target.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    started_at = datetime.now(tz=ZoneInfo("Asia/Shanghai"))
    started = perf_counter()
    if workers == 1:
        completed: list[VectorizedTargetBenchmark] = []
        for issue in target_issues:
            target = _benchmark_target_from_path(path, issue, seed)
            checkpoint(target)
            completed.append(target)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_benchmark_target_from_path, path, issue, seed): issue
                for issue in target_issues
            }
            completed = []
            for future in as_completed(futures):
                target = future.result()
                checkpoint(target)
                completed.append(target)
    by_issue = {target.target_issue: target for target in completed}
    targets = tuple(by_issue[issue] for issue in target_issues)
    wall_seconds = perf_counter() - started
    return VectorizedBenchmarkRun(
        workers=workers,
        target_issues=target_issues,
        started_at=started_at,
        ended_at=datetime.now(tz=ZoneInfo("Asia/Shanghai")),
        wall_clock_seconds=wall_seconds,
        average_seconds_per_target=wall_seconds / len(targets),
        targets=targets,
    )


def write_vectorized_benchmark_json(
    result: VectorizedBenchmarkRun,
    path: str | Path,
) -> Path:
    """Persist one worker-count result for later bounded aggregation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    return target


def load_vectorized_benchmark_json(path: str | Path) -> VectorizedBenchmarkRun:
    """Load one immutable benchmark result."""
    return VectorizedBenchmarkRun.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_vectorized_benchmark_report(
    runs: tuple[VectorizedBenchmarkRun, ...],
    path: str | Path,
    *,
    incomplete_note: str | None = None,
) -> Path:
    """Write the small benchmark without turning runtime into a strategy claim."""
    rows: list[dict[str, object]] = []
    for run in runs:
        for target in run.targets:
            rows.append(
                {
                    "workers": run.workers,
                    **target.model_dump(exclude={"risk_disclaimer"}),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    lines = [
        "# v0.5.2 Portfolio 向量化小型基准",
        "",
        "仅使用目标期 08009、08010、08011；不写入实验 Parquet，不运行100期或消融。",
        "峰值内存由操作系统进程 peak working set/RSS 读取；对象路径在向量路径后测量。",
        "",
    ]
    if frame.empty:
        lines.append("尚无完成的基准结果。")
    else:
        columns = list(frame.columns)
        lines.extend(
            [
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join("---" for _ in columns) + " |",
            ]
        )
        for row in frame.to_dict(orient="records"):
            lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
        lines.extend(["", "## Worker 汇总", ""])
        for run in sorted(runs, key=lambda item: item.workers):
            speedups = [target.speedup for target in run.targets]
            lines.append(
                f"- workers={run.workers}: wall={run.wall_clock_seconds:.3f}s，"
                f"每期平均={run.average_seconds_per_target:.3f}s，"
                f"对象/向量平均加速比={sum(speedups) / len(speedups):.3f}x。"
            )
        minimum_speedup = min(target.speedup for run in runs for target in run.targets)
        lines.extend(
            [
                "",
                (
                    "所有目标期的最终 entry、5注号码和评分均一致。"
                    if all(
                        target.all_final_tickets_identical for run in runs for target in run.targets
                    )
                    else "存在对象路径与向量路径不一致，结果不可接受。"
                ),
                (
                    f"最低加速比为 {minimum_speedup:.3f}x；达到2倍目标。"
                    if minimum_speedup >= 2
                    else f"最低加速比为 {minimum_speedup:.3f}x；未达到2倍目标。"
                ),
            ]
        )
    if incomplete_note:
        lines.extend(["", "## 停止条件", "", incomplete_note])
    lines.extend(["", f"> {DISCLAIMER}"])
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return target

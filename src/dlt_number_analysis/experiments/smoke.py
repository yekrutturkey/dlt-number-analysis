"""Auditable reports for the bounded v0.5.1 B1-B6 paired smoke experiment."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.experiments.reporting import (
    ExperimentComparison,
    compare_to_constraint_matched_baseline,
)
from dlt_number_analysis.experiments.specs import ExperimentSpec, baseline_experiment_specs
from dlt_number_analysis.experiments.splits import split_history
from dlt_number_analysis.experiments.storage import (
    DuplicateExperimentResultError,
    ExperimentResultStore,
)

SMOKE_EXPERIMENT_IDS: tuple[str, ...] = ("B1", "B2", "B3", "B4", "B5", "B6")


class DistributionSummary(BaseModel):
    """Four-number distribution summary used for bank diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: float
    median: float
    mean: float
    maximum: float


class PairedSmokeAudit(BaseModel):
    """Validated cross-strategy invariants for one bounded smoke experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str = "development"
    profile: str = "fast"
    seed: int
    target_count: int = Field(ge=1)
    start_target_issue: str = Field(pattern=r"^\d+$")
    end_target_issue: str = Field(pattern=r"^\d+$")
    experiment_ids: tuple[str, ...] = SMOKE_EXPERIMENT_IDS
    candidate_hash_shared: bool
    bank_hash_shared_within_constraints: bool
    bank_generation_count_per_target: int = Field(ge=1)
    portfolio_bank_reuse_count_per_target: int = Field(ge=0)
    minimum_bank_size_required: int = Field(ge=1)
    expanded_target_count: int = Field(ge=0)
    bank_size: DistributionSummary
    acceptance_rate: DistributionSummary
    resume_status_verified: bool
    duplicate_primary_key_rejected: bool
    completed_task_skip_verified: bool
    risk_disclaimer: str = DISCLAIMER


def select_consecutive_development_targets(
    draws: pd.DataFrame,
    *,
    count: int,
    start_issue: str | None = None,
    minimum_history: int = 100,
) -> tuple[str, ...]:
    """Select consecutive verified-history rows wholly inside development."""
    if count < 1:
        raise ValueError("count must be positive")
    specification = baseline_experiment_specs(seeds=(20260000,), phase="development")[1]
    phase = split_history(draws, specification.data_split).phase_frame("development")
    phase_issues = set(phase["issue"].astype(str).tolist())
    eligible = tuple(
        issue
        for index, issue in enumerate(draws["issue"].astype(str).tolist())
        if index >= minimum_history and issue in phase_issues
    )
    if start_issue is None:
        start_index = 0
    else:
        try:
            start_index = eligible.index(str(start_issue))
        except ValueError as error:
            raise ValueError("start_issue is not an eligible development target") from error
    selected = eligible[start_index : start_index + count]
    if len(selected) != count:
        raise ValueError("development does not contain enough consecutive targets")
    positions = {issue: index for index, issue in enumerate(draws["issue"].astype(str))}
    selected_positions = [positions[issue] for issue in selected]
    if any(right != left + 1 for left, right in pairwise(selected_positions)):
        raise ValueError("selected development targets are not consecutive history rows")
    return selected


def _summary(values: pd.Series) -> DistributionSummary:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    return DistributionSummary(
        minimum=float(numeric.min()),
        median=float(numeric.median()),
        mean=float(numeric.mean()),
        maximum=float(numeric.max()),
    )


def _smoke_rows(
    observations: pd.DataFrame,
    *,
    target_issues: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    required = {
        "phase",
        "experiment_id",
        "seed",
        "target_issue",
        "candidate_numbers_hash",
        "constraints_signature",
        "bank_hash",
        "bank_size",
        "bank_acceptance_rate",
        "bank_search_expansion_count",
        "target_bank_generation_count",
        "target_portfolio_bank_reuse_count",
    }
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"smoke observations are missing columns: {sorted(missing)}")
    targets = {str(issue) for issue in target_issues}
    selected = observations.loc[
        (observations["phase"] == "development")
        & (observations["seed"] == seed)
        & observations["experiment_id"].isin(SMOKE_EXPERIMENT_IDS)
        & observations["target_issue"].astype(str).isin(targets)
    ].copy()
    selected["target_issue"] = selected["target_issue"].astype(str)
    return selected


def validate_paired_smoke_observations(
    observations: pd.DataFrame,
    *,
    target_issues: Sequence[str],
    seed: int,
    minimum_bank_size: int = 500,
    resume_status_verified: bool,
    duplicate_primary_key_rejected: bool,
    completed_task_skip_verified: bool,
) -> PairedSmokeAudit:
    """Reject any smoke output that does not prove B1-B6 sharing invariants."""
    targets = tuple(str(issue) for issue in target_issues)
    selected = _smoke_rows(observations, target_issues=targets, seed=seed)
    expected_rows = len(targets) * len(SMOKE_EXPERIMENT_IDS)
    if len(selected) != expected_rows:
        raise ValueError(f"smoke experiment expected {expected_rows} rows, found {len(selected)}")
    if selected.duplicated(["target_issue", "seed", "experiment_id"]).any():
        raise ValueError("smoke observations contain duplicate target/seed/experiment rows")
    expected_ids = set(SMOKE_EXPERIMENT_IDS)
    for target_issue, group in selected.groupby("target_issue", sort=False):
        if set(group["experiment_id"]) != expected_ids:
            raise ValueError(f"target {target_issue} does not contain exactly B1-B6")
        if group["candidate_numbers_hash"].nunique() != 1:
            raise ValueError(f"target {target_issue} does not share one candidate hash")
        for _, constrained in group.groupby("constraints_signature", sort=False):
            if constrained["bank_hash"].nunique() != 1:
                raise ValueError(f"target {target_issue} has divergent bank hashes")
        if set(group["target_bank_generation_count"].astype(int)) != {1}:
            raise ValueError(f"target {target_issue} did not generate exactly one bank")
        if set(group["target_portfolio_bank_reuse_count"].astype(int)) != {5}:
            raise ValueError(f"target {target_issue} did not reuse the bank exactly five times")
        if int(group["bank_size"].min()) < minimum_bank_size:
            raise ValueError(f"target {target_issue} bank is below the required threshold")

    banks = selected.drop_duplicates(["target_issue", "seed", "constraints_signature", "bank_hash"])
    if len(banks) != len(targets):
        raise ValueError("the smoke experiment did not persist exactly one bank per target")
    return PairedSmokeAudit(
        seed=seed,
        target_count=len(targets),
        start_target_issue=targets[0],
        end_target_issue=targets[-1],
        candidate_hash_shared=True,
        bank_hash_shared_within_constraints=True,
        bank_generation_count_per_target=1,
        portfolio_bank_reuse_count_per_target=5,
        minimum_bank_size_required=minimum_bank_size,
        expanded_target_count=int(
            (pd.to_numeric(banks["bank_search_expansion_count"], errors="raise") > 0).sum()
        ),
        bank_size=_summary(banks["bank_size"]),
        acceptance_rate=_summary(banks["bank_acceptance_rate"]),
        resume_status_verified=resume_status_verified,
        duplicate_primary_key_rejected=duplicate_primary_key_rejected,
        completed_task_skip_verified=completed_task_skip_verified,
    )


def smoke_comparisons(
    observations: pd.DataFrame,
    *,
    target_issues: Sequence[str],
    seed: int,
    bootstrap_resamples: int = 2_000,
    permutations: int = 5_000,
) -> tuple[ExperimentComparison, ...]:
    """Compute paired B2-B6 minus B1 statistics on only the smoke target set."""
    selected = _smoke_rows(observations, target_issues=target_issues, seed=seed)
    return compare_to_constraint_matched_baseline(
        selected,
        bootstrap_resamples=bootstrap_resamples,
        permutations=permutations,
        minimum_paired_observations=len(tuple(target_issues)),
    )


def comparisons_frame(comparisons: Sequence[ExperimentComparison]) -> pd.DataFrame:
    """Return smoke statistics without turning p-values into an advantage claim."""
    columns = (
        "phase",
        "experiment_id",
        "baseline_id",
        "metric",
        "paired_observation_count",
        "mean_paired_difference",
        "bootstrap_95_lower",
        "bootstrap_95_upper",
        "permutation_p_value",
        "holm_adjusted_p_value",
        "benjamini_hochberg_adjusted_p_value",
    )
    records = [comparison.model_dump(include=set(columns)) for comparison in comparisons]
    return pd.DataFrame.from_records(records, columns=columns)


def verify_smoke_storage(
    store: ExperimentResultStore,
    specifications: Sequence[ExperimentSpec],
    *,
    seed: int,
    target_issues: Sequence[str],
) -> tuple[bool, bool]:
    """Verify resume completion and reject an actual duplicate without mutating storage."""
    expected = tuple(str(issue) for issue in target_issues)
    resume_complete = all(
        store.status(spec, seed=seed, expected_target_issues=expected).is_complete
        for spec in specifications
    )
    observations = store.load_all(phase="development")
    duplicate = _smoke_rows(observations, target_issues=expected, seed=seed).iloc[[0]]
    try:
        store.append(duplicate)
    except DuplicateExperimentResultError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    return resume_complete, duplicate_rejected


def write_paired_smoke_report(
    audit: PairedSmokeAudit,
    comparisons: Sequence[ExperimentComparison],
    path: str | Path,
) -> Path:
    """Write bounded correctness evidence without selecting parameters or claiming advantage."""
    comparison_table = comparisons_frame(comparisons)
    lines = [
        "# v0.5.1 B1-B6 100期配对冒烟实验",
        "",
        f"- 阶段：{audit.phase}",
        f"- profile：{audit.profile}",
        f"- seed：{audit.seed}",
        f"- 连续目标期：{audit.start_target_issue} 至 {audit.end_target_issue}",
        f"- 目标期数量：{audit.target_count}",
        f"- 每期银行生成次数：{audit.bank_generation_count_per_target}",
        f"- 每期银行复用次数：{audit.portfolio_bank_reuse_count_per_target}",
        f"- 自动扩大搜索的目标期数：{audit.expanded_target_count}",
        f"- 断点续跑状态验证：{'通过' if audit.resume_status_verified else '失败'}",
        f"- 重复主键拒绝验证：{'通过' if audit.duplicate_primary_key_rejected else '失败'}",
        f"- 已完成任务跳过验证：{'通过' if audit.completed_task_skip_verified else '失败'}",
        "",
        "## 可行银行分布",
        "",
        "| 指标 | 最小值 | 中位数 | 均值 | 最大值 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| bank_size | {audit.bank_size.minimum:.0f} | {audit.bank_size.median:.1f} | "
            f"{audit.bank_size.mean:.2f} | {audit.bank_size.maximum:.0f} |"
        ),
        (
            f"| acceptance_rate | {audit.acceptance_rate.minimum:.6f} | "
            f"{audit.acceptance_rate.median:.6f} | {audit.acceptance_rate.mean:.6f} | "
            f"{audit.acceptance_rate.maximum:.6f} |"
        ),
        "",
        "## B2-B6 相对 B1 的配对统计",
        "",
    ]
    if comparison_table.empty:
        lines.append("没有可用的完整配对指标。")
    else:
        lines.extend(
            [
                "| " + " | ".join(comparison_table.columns) + " |",
                "| " + " | ".join("---" for _ in comparison_table.columns) + " |",
            ]
        )
        for row in comparison_table.to_dict(orient="records"):
            lines.append("| " + " | ".join(str(value) for value in row.values()) + " |")
    lines.extend(
        [
            "",
            "本实验只验证配对计算和工程闭环，不使用这100期结果选择参数，也不声明策略优势。",
            "",
            f"> {DISCLAIMER}",
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")
    return target


def write_smoke_comparisons_csv(
    comparisons: Sequence[ExperimentComparison],
    path: str | Path,
) -> Path:
    """Persist the paired statistics as a machine-readable no-claim artifact."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = comparisons_frame(comparisons)
    frame["inference_context"] = "smoke_only_no_parameter_selection_or_advantage_claim"
    frame["risk_disclaimer"] = DISCLAIMER
    frame.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    return target

"""带种子的合法候选票生成、动态评分和明细持久化。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from pathlib import Path
from random import Random

import pandas as pd
from pydantic import JsonValue

from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.portfolio.models import CandidatePool, CandidateTicketScore
from dlt_number_analysis.scoring import (
    NumberScorer,
    compute_ticket_structure,
    cumulative_frequency_score,
    fit_structure_profile,
    score_ticket_structure,
)

MINIMUM_CANDIDATE_COUNT = 10_000


def _normalize_scores(scores: Mapping[int, float], maximum: int) -> dict[int, float]:
    """校验完整号码空间并缩放到 0–1。"""
    expected = set(range(1, maximum + 1))
    if set(scores) != expected:
        raise ValueError(f"号码评分必须完整覆盖 1 到 {maximum}")
    if any(not isfinite(float(value)) for value in scores.values()):
        raise ValueError("号码评分必须是有限数值")
    minimum = min(float(value) for value in scores.values())
    maximum_score = max(float(value) for value in scores.values())
    if minimum == maximum_score:
        return dict.fromkeys(expected, 0.5)
    return {
        number: (float(scores[number]) - minimum) / (maximum_score - minimum) for number in expected
    }


def _draw_from_last_row(history: pd.DataFrame) -> DrawRecord:
    """把历史窗口最后一行恢复为 DrawRecord。"""
    row = history.iloc[-1]
    return DrawRecord.model_validate({column: row[column] for column in CSV_COLUMNS})


def _resolve_scorer_name(number_scorer: NumberScorer, scorer_name: str | None) -> str:
    """为普通函数、partial 或可调用对象生成稳定名称。"""
    if scorer_name:
        return scorer_name
    return str(getattr(number_scorer, "__name__", number_scorer.__class__.__name__))


def generate_candidate_pool(
    history: pd.DataFrame,
    *,
    target_issue: str,
    generated_at: datetime,
    random_seed: int,
    number_scorer: NumberScorer = cumulative_frequency_score,
    scorer_name: str | None = None,
    scorer_parameters: Mapping[str, JsonValue] | None = None,
    candidate_count: int = MINIMUM_CANDIDATE_COUNT,
    front_number_weight: float = 5 / 7,
    back_number_weight: float = 2 / 7,
    number_score_weight: float = 0.5,
    structure_score_weight: float = 0.5,
) -> CandidatePool:
    """生成至少一万注不重复合法候选，并保存号码、结构和综合评分。"""
    if candidate_count < MINIMUM_CANDIDATE_COUNT:
        raise ValueError(f"candidate_count 必须至少为 {MINIMUM_CANDIDATE_COUNT}")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at 必须包含时区")
    if any(
        not isfinite(weight) or weight < 0
        for weight in (
            front_number_weight,
            back_number_weight,
            number_score_weight,
            structure_score_weight,
        )
    ):
        raise ValueError("评分权重必须是非负有限数值")
    if front_number_weight + back_number_weight <= 0:
        raise ValueError("前后区号码评分权重之和必须大于 0")
    if number_score_weight + structure_score_weight <= 0:
        raise ValueError("号码与结构评分权重之和必须大于 0")

    validated = validate_draw_dataframe(history)
    cutoff_issue = str(validated.iloc[-1]["issue"])
    if int(cutoff_issue) >= int(target_issue):
        raise ValueError("候选票池只能使用目标期开奖前的数据")
    previous_draw = _draw_from_last_row(validated)
    profile = fit_structure_profile(validated)
    raw_front_scores = dict(number_scorer(validated, area="front"))
    raw_back_scores = dict(number_scorer(validated, area="back"))
    front_scores = _normalize_scores(raw_front_scores, 35)
    back_scores = _normalize_scores(raw_back_scores, 12)
    front_weight_total = front_number_weight + back_number_weight
    combined_weight_total = number_score_weight + structure_score_weight

    rng = Random(random_seed)
    combinations_seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    candidates: list[CandidateTicketScore] = []
    while len(candidates) < candidate_count:
        front = tuple(sorted(rng.sample(range(1, 36), 5)))
        back = tuple(sorted(rng.sample(range(1, 13), 2)))
        combination = (front, back)
        if combination in combinations_seen:
            continue
        combinations_seen.add(combination)

        features = compute_ticket_structure(front, back, previous_draw=previous_draw)
        structure = score_ticket_structure(features, profile)
        front_number_score = sum(front_scores[number] for number in front) / len(front)
        back_number_score = sum(back_scores[number] for number in back) / len(back)
        number_score = (
            front_number_weight * front_number_score + back_number_weight * back_number_score
        ) / front_weight_total
        combined_score = (
            number_score_weight * number_score + structure_score_weight * structure.overall_score
        ) / combined_weight_total
        candidate_index = len(candidates) + 1
        candidates.append(
            CandidateTicketScore(
                candidate_id=f"{target_issue}-candidate-{candidate_index:05d}",
                front_numbers=front,
                back_numbers=back,
                features=features,
                front_number_score=front_number_score,
                back_number_score=back_number_score,
                number_score=number_score,
                structure_score=structure.overall_score,
                combined_ticket_score=combined_score,
                structure_component_scores=structure.component_scores,
                sum_interval=profile.front_sum_interval(features.front_sum),
                zone_structure=features.zone_signature,
            )
        )

    resolved_name = _resolve_scorer_name(number_scorer, scorer_name)
    return CandidatePool(
        target_issue=target_issue,
        data_cutoff_issue=cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        scorer_name=resolved_name,
        scorer_parameters=dict(scorer_parameters or {}),
        generation_parameters={
            "candidate_count": candidate_count,
            "sampling_method": "seeded_uniform_without_replacement_within_ticket",
            "unique_across_candidate_pool": True,
            "front_number_weight": front_number_weight,
            "back_number_weight": back_number_weight,
            "number_score_weight": number_score_weight,
            "structure_score_weight": structure_score_weight,
            "structure_scoring_method": profile.scoring_method,
            "structure_profile_start_issue": profile.data_start_issue,
            "structure_profile_cutoff_issue": profile.data_cutoff_issue,
            "front_sum_quantile_edges": list(profile.front_sum_quantile_edges),
            "raw_front_scores": {
                str(number): float(score) for number, score in sorted(raw_front_scores.items())
            },
            "raw_back_scores": {
                str(number): float(score) for number, score in sorted(raw_back_scores.items())
            },
        },
        candidates=tuple(candidates),
    )


def write_candidate_score_details(pool: CandidatePool, path: str | Path) -> Path:
    """以 JSONL 保存候选池元数据和每注评分明细。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = pool.model_dump(mode="json", exclude={"candidates"})
    lines = [json.dumps({"record_type": "candidate_pool_metadata", **metadata}, ensure_ascii=False)]
    lines.extend(
        json.dumps(
            {
                "record_type": "candidate_score",
                "risk_disclaimer": pool.risk_disclaimer,
                **candidate.model_dump(mode="json"),
            },
            ensure_ascii=False,
        )
        for candidate in pool.candidates
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def load_candidate_score_details(path: str | Path) -> CandidatePool:
    """读取并完整校验候选评分 JSONL。"""
    source = Path(path)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not records or records[0].pop("record_type", None) != "candidate_pool_metadata":
        raise ValueError("候选评分文件缺少元数据记录")
    candidates = []
    for record in records[1:]:
        if record.pop("record_type", None) != "candidate_score":
            raise ValueError("候选评分文件包含未知记录类型")
        if record.pop("risk_disclaimer", None) != records[0]["risk_disclaimer"]:
            raise ValueError("候选评分记录缺少固定风险声明")
        candidates.append(record)
    return CandidatePool.model_validate({**records[0], "candidates": candidates})

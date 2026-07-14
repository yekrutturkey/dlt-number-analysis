"""基于简单可替换号码分数的五注组合策略。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import isfinite
from random import Random
from typing import Protocol

from pydantic import JsonValue

from dlt_number_analysis.models import PredictionRecord, TicketRecord

TICKET_COUNT = 5
FRONT_COUNT = 5
BACK_COUNT = 2
DEFAULT_MODEL_VERSION = "simple-score-v0.2"


class StrategyFunction(Protocol):
    """滚动回测可调用的统一组合策略签名。"""

    def __call__(
        self,
        *,
        target_issue: str,
        data_cutoff_issue: str,
        generated_at: datetime,
        random_seed: int,
        parameters: Mapping[str, JsonValue] | None = None,
        front_scores: Mapping[int, float] | None = None,
        back_scores: Mapping[int, float] | None = None,
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> PredictionRecord: ...


def _rank_numbers(
    maximum: int,
    scores: Mapping[int, float] | None,
    rng: Random,
) -> list[int]:
    """按简单外部分数和带种子的随机平分排序号码。"""
    normalized_scores = dict(scores or {})
    if any(number < 1 or number > maximum for number in normalized_scores):
        raise ValueError(f"号码评分键必须在 1 到 {maximum} 之间")
    if any(not isfinite(float(score)) for score in normalized_scores.values()):
        raise ValueError("号码评分必须是有限数值")
    return sorted(
        range(1, maximum + 1),
        key=lambda number: (-float(normalized_scores.get(number, 0.0)), rng.random()),
    )


def _normalize_core_numbers(
    value: JsonValue | None, ranked_front: Sequence[int]
) -> tuple[int, ...]:
    """读取五个显式核心号码，未提供时使用简单评分前五名。"""
    if value is None:
        return tuple(ranked_front[:5])
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError("core_front_numbers 必须包含 5 个号码")
    if any(isinstance(number, bool) or not isinstance(number, int) for number in value):
        raise ValueError("core_front_numbers 必须是整数列表")
    numbers = tuple(int(number) for number in value)
    if len(set(numbers)) != 5 or any(number < 1 or number > 35 for number in numbers):
        raise ValueError("core_front_numbers 必须是 1 到 35 内的 5 个不同号码")
    return numbers


def _stored_parameters(
    strategy_parameters: Mapping[str, JsonValue] | None,
    front_scores: Mapping[int, float] | None,
    back_scores: Mapping[int, float] | None,
    **fixed: JsonValue,
) -> dict[str, JsonValue]:
    """构建可 JSON 序列化的完整参数快照。"""
    return {
        "ticket_count": TICKET_COUNT,
        "scoring_method": "provided_score_then_seeded_random_tiebreak",
        "strategy_parameters": dict(strategy_parameters or {}),
        "front_scores": {
            str(number): float(score) for number, score in sorted((front_scores or {}).items())
        },
        "back_scores": {
            str(number): float(score) for number, score in sorted((back_scores or {}).items())
        },
        **fixed,
    }


def _build_prediction(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    strategy_name: str,
    model_version: str,
    ticket_numbers: Sequence[tuple[Sequence[int], Sequence[int], str]],
    stored_parameters: dict[str, JsonValue],
) -> PredictionRecord:
    """从五组号码构建统一预测日志，并由模型禁止完全重复票据。"""
    if len(ticket_numbers) != TICKET_COUNT:
        raise ValueError("每种策略必须生成 5 注")
    tickets = tuple(
        TicketRecord(
            target_issue=target_issue,
            ticket_id=f"{target_issue}-{strategy_name}-{index:02d}",
            front_numbers=tuple(sorted(front)),
            back_numbers=tuple(sorted(back)),
            ticket_role=role,
        )
        for index, (front, back, role) in enumerate(ticket_numbers, start=1)
    )
    return PredictionRecord(
        target_issue=target_issue,
        tickets=tickets,
        strategy_name=strategy_name,
        model_version=model_version,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        parameters=stored_parameters,
        prediction_origin="generated",
    )


def max_coverage(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    parameters: Mapping[str, JsonValue] | None = None,
    front_scores: Mapping[int, float] | None = None,
    back_scores: Mapping[int, float] | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> PredictionRecord:
    """生成号码跨票不重复的五注最大覆盖组合。"""
    rng = Random(random_seed)
    ranked_front = _rank_numbers(35, front_scores, rng)[: TICKET_COUNT * FRONT_COUNT]
    ranked_back = _rank_numbers(12, back_scores, rng)[: TICKET_COUNT * BACK_COUNT]
    tickets = [
        (
            ranked_front[index * FRONT_COUNT : (index + 1) * FRONT_COUNT],
            ranked_back[index * BACK_COUNT : (index + 1) * BACK_COUNT],
            "max_coverage",
        )
        for index in range(TICKET_COUNT)
    ]
    return _build_prediction(
        target_issue=target_issue,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        strategy_name="max_coverage",
        model_version=model_version,
        ticket_numbers=tickets,
        stored_parameters=_stored_parameters(
            parameters,
            front_scores,
            back_scores,
            cross_ticket_front_reuse=False,
            cross_ticket_back_reuse=False,
        ),
    )


def core_rotation(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    parameters: Mapping[str, JsonValue] | None = None,
    front_scores: Mapping[int, float] | None = None,
    back_scores: Mapping[int, float] | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> PredictionRecord:
    """让五个核心前区号码分别出现 2 或 3 次并轮转生成五注。"""
    strategy_parameters = dict(parameters or {})
    repetitions_value = strategy_parameters.get("core_repetitions", 2)
    if isinstance(repetitions_value, bool) or repetitions_value not in (2, 3):
        raise ValueError("core_repetitions 只能是 2 或 3")
    repetitions = int(repetitions_value)

    rng = Random(random_seed)
    ranked_front = _rank_numbers(35, front_scores, rng)
    ranked_back = _rank_numbers(12, back_scores, rng)[: TICKET_COUNT * BACK_COUNT]
    core_numbers = _normalize_core_numbers(
        strategy_parameters.get("core_front_numbers"), ranked_front
    )
    ticket_fronts: list[list[int]] = [[] for _ in range(TICKET_COUNT)]
    for core_index, number in enumerate(core_numbers):
        for offset in range(repetitions):
            ticket_fronts[(core_index + offset) % TICKET_COUNT].append(number)

    non_core = [number for number in ranked_front if number not in core_numbers]
    non_core_cursor = 0
    for ticket_front in ticket_fronts:
        fill_count = FRONT_COUNT - len(ticket_front)
        ticket_front.extend(non_core[non_core_cursor : non_core_cursor + fill_count])
        non_core_cursor += fill_count

    tickets = [
        (
            ticket_fronts[index],
            ranked_back[index * BACK_COUNT : (index + 1) * BACK_COUNT],
            "core_rotation",
        )
        for index in range(TICKET_COUNT)
    ]
    return _build_prediction(
        target_issue=target_issue,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        strategy_name="core_rotation",
        model_version=model_version,
        ticket_numbers=tickets,
        stored_parameters=_stored_parameters(
            strategy_parameters,
            front_scores,
            back_scores,
            core_front_numbers=list(core_numbers),
            core_repetitions=repetitions,
        ),
    )


def hybrid_portfolio(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    parameters: Mapping[str, JsonValue] | None = None,
    front_scores: Mapping[int, float] | None = None,
    back_scores: Mapping[int, float] | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
) -> PredictionRecord:
    """生成三注核心轮转和两注低重叠探索组合。"""
    strategy_parameters = dict(parameters or {})
    rng = Random(random_seed)
    ranked_front = _rank_numbers(35, front_scores, rng)
    ranked_back = _rank_numbers(12, back_scores, rng)[: TICKET_COUNT * BACK_COUNT]
    configured_core = _normalize_core_numbers(
        strategy_parameters.get("core_front_numbers"), ranked_front
    )
    core_numbers = configured_core[:3]
    non_core = [number for number in ranked_front if number not in core_numbers]

    ticket_fronts: list[list[int]] = []
    cursor = 0
    core_pairs = (
        (core_numbers[0], core_numbers[1]),
        (core_numbers[1], core_numbers[2]),
        (core_numbers[2], core_numbers[0]),
    )
    for pair in core_pairs:
        ticket_fronts.append([*pair, *non_core[cursor : cursor + 3]])
        cursor += 3
    for _ in range(2):
        ticket_fronts.append(non_core[cursor : cursor + FRONT_COUNT])
        cursor += FRONT_COUNT

    roles = ("core_rotation", "core_rotation", "core_rotation", "exploration", "exploration")
    tickets = [
        (
            ticket_fronts[index],
            ranked_back[index * BACK_COUNT : (index + 1) * BACK_COUNT],
            roles[index],
        )
        for index in range(TICKET_COUNT)
    ]
    return _build_prediction(
        target_issue=target_issue,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        strategy_name="hybrid_portfolio",
        model_version=model_version,
        ticket_numbers=tickets,
        stored_parameters=_stored_parameters(
            strategy_parameters,
            front_scores,
            back_scores,
            core_front_numbers=list(core_numbers),
            core_ticket_count=3,
            exploration_ticket_count=2,
        ),
    )


def random_baseline(
    *,
    target_issue: str,
    data_cutoff_issue: str,
    generated_at: datetime,
    random_seed: int,
    parameters: Mapping[str, JsonValue] | None = None,
    front_scores: Mapping[int, float] | None = None,
    back_scores: Mapping[int, float] | None = None,
    model_version: str = "uniform-random-v0.2",
) -> PredictionRecord:
    """完全均匀随机生成五注 baseline，不使用号码分数。"""
    del front_scores, back_scores
    rng = Random(random_seed)
    tickets: list[tuple[Sequence[int], Sequence[int], str]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    while len(tickets) < TICKET_COUNT:
        front = tuple(sorted(rng.sample(range(1, 36), FRONT_COUNT)))
        back = tuple(sorted(rng.sample(range(1, 13), BACK_COUNT)))
        combination = (front, back)
        if combination in seen:
            continue
        seen.add(combination)
        tickets.append((front, back, "random_baseline"))

    return _build_prediction(
        target_issue=target_issue,
        data_cutoff_issue=data_cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        strategy_name="random_baseline",
        model_version=model_version,
        ticket_numbers=tickets,
        stored_parameters=_stored_parameters(
            parameters,
            None,
            None,
            distribution="uniform_without_replacement_within_ticket",
        ),
    )

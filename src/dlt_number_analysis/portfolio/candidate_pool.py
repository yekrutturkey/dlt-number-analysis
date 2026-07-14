"""Seeded legal candidate generation with bound scorer configuration and audit details."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from math import isfinite
from pathlib import Path
from random import Random

import pandas as pd

from dlt_number_analysis.data import CSV_COLUMNS, DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe
from dlt_number_analysis.portfolio.models import CandidatePool, CandidateTicketScore
from dlt_number_analysis.scoring import (
    HistoricalStructureProfile,
    ScorerSpec,
    build_number_scorer,
    compute_ticket_structure,
    fit_structure_profile,
    score_ticket_structure,
)

MINIMUM_CANDIDATE_COUNT = 10_000


def _normalize_scores(scores: Mapping[int, float], maximum: int) -> dict[int, float]:
    expected = set(range(1, maximum + 1))
    if set(scores) != expected:
        raise ValueError(f"number scores must cover every value from 1 to {maximum}")
    if any(not isfinite(float(value)) for value in scores.values()):
        raise ValueError("number scores must be finite")
    minimum = min(float(value) for value in scores.values())
    maximum_score = max(float(value) for value in scores.values())
    if minimum == maximum_score:
        return dict.fromkeys(expected, 0.5)
    return {
        number: (float(scores[number]) - minimum) / (maximum_score - minimum) for number in expected
    }


def _draw_from_last_row(history: pd.DataFrame) -> DrawRecord:
    row = history.iloc[-1]
    return DrawRecord.model_validate({column: row[column] for column in CSV_COLUMNS})


def generate_candidate_pool(
    history: pd.DataFrame,
    *,
    target_issue: str,
    generated_at: datetime,
    random_seed: int,
    scorer_spec: ScorerSpec | None = None,
    structure_profile: HistoricalStructureProfile | None = None,
    structure_feature_weights: Mapping[str, float] | None = None,
    laplace_alpha: float = 1.0,
    candidate_count: int = MINIMUM_CANDIDATE_COUNT,
    front_number_weight: float = 5 / 7,
    back_number_weight: float = 2 / 7,
    number_score_weight: float = 0.5,
    structure_score_weight: float = 0.5,
) -> CandidatePool:
    """Generate at least 10,000 unique legal tickets from pre-target history."""
    if candidate_count < MINIMUM_CANDIDATE_COUNT:
        raise ValueError(f"candidate_count must be at least {MINIMUM_CANDIDATE_COUNT}")
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must include a timezone")
    weights = (
        front_number_weight,
        back_number_weight,
        number_score_weight,
        structure_score_weight,
    )
    if any(not isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError("candidate scoring weights must be finite and non-negative")
    if front_number_weight + back_number_weight <= 0:
        raise ValueError("front/back number score weights must sum above zero")
    if number_score_weight + structure_score_weight <= 0:
        raise ValueError("number/structure score weights must sum above zero")

    validated = validate_draw_dataframe(history)
    cutoff_issue = str(validated.iloc[-1]["issue"])
    if int(cutoff_issue) >= int(target_issue):
        raise ValueError("candidate generation may only use draws before the target issue")
    previous_draw = _draw_from_last_row(validated)
    profile = structure_profile or fit_structure_profile(
        validated,
        feature_weights=structure_feature_weights,
        laplace_alpha=laplace_alpha,
    )
    if profile.data_cutoff_issue != cutoff_issue:
        raise ValueError("structure profile cutoff does not match candidate history cutoff")

    resolved_spec = scorer_spec or ScorerSpec(name="cumulative_frequency_score")
    scorer = build_number_scorer(resolved_spec)
    raw_front_scores = dict(scorer(validated, area="front"))
    raw_back_scores = dict(scorer(validated, area="back"))
    front_scores = _normalize_scores(raw_front_scores, 35)
    back_scores = _normalize_scores(raw_back_scores, 12)
    area_weight_total = front_number_weight + back_number_weight
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
        front_score = sum(front_scores[number] for number in front) / len(front)
        back_score = sum(back_scores[number] for number in back) / len(back)
        number_score = (
            front_number_weight * front_score + back_number_weight * back_score
        ) / area_weight_total
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
                front_number_score=front_score,
                back_number_score=back_score,
                number_score=number_score,
                structure_score=structure.overall_score,
                combined_ticket_score=combined_score,
                structure_component_scores=structure.component_scores,
                structure_component_details=structure.component_details,
                sum_interval=profile.front_sum_interval(features.front_sum),
                zone_structure=features.zone_signature,
            )
        )

    return CandidatePool(
        target_issue=target_issue,
        data_cutoff_issue=cutoff_issue,
        generated_at=generated_at,
        random_seed=random_seed,
        scorer_spec=resolved_spec,
        generation_parameters={
            "candidate_count": candidate_count,
            "sampling_method": "seeded_uniform_without_replacement_within_ticket",
            "unique_across_candidate_pool": True,
            "front_number_weight": front_number_weight,
            "back_number_weight": back_number_weight,
            "number_score_weight": number_score_weight,
            "structure_score_weight": structure_score_weight,
            "structure_profile": profile.model_dump(
                mode="json", exclude={"continuous_distributions", "discrete_frequencies"}
            ),
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
    """Persist pool metadata and every candidate score as UTF-8 JSONL."""
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
    """Load and validate a candidate-score JSONL artifact."""
    source = Path(path)
    records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not records or records[0].pop("record_type", None) != "candidate_pool_metadata":
        raise ValueError("candidate score file is missing its metadata record")
    candidates = []
    for record in records[1:]:
        if record.pop("record_type", None) != "candidate_score":
            raise ValueError("candidate score file contains an unknown record type")
        if record.pop("risk_disclaimer", None) != records[0]["risk_disclaimer"]:
            raise ValueError("candidate score record is missing the fixed disclaimer")
        candidates.append(record)
    return CandidatePool.model_validate({**records[0], "candidates": candidates})

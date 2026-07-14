"""History-only empirical structure profiles and auditable ticket scores."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from math import isclose
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from dlt_number_analysis import DISCLAIMER
from dlt_number_analysis.analysis import (
    compute_back_features,
    compute_front_features,
    engineer_features,
)
from dlt_number_analysis.data import DrawRecord
from dlt_number_analysis.data.data_validator import validate_draw_dataframe

StructureMethod = Literal[
    "empirical_quantile_centrality",
    "laplace_smoothed_empirical_frequency",
]

DEFAULT_STRUCTURE_WEIGHTS: dict[str, float] = {
    "front_sum": 0.10,
    "front_span": 0.08,
    "front_odd_count": 0.08,
    "front_large_count": 0.08,
    "front_zone_signature": 0.12,
    "front_consecutive_pair_count": 0.07,
    "front_same_tail_pair_count": 0.07,
    "front_repeat_from_previous_count": 0.08,
    "back_sum": 0.10,
    "back_odd_count": 0.06,
    "back_large_count": 0.06,
    "back_consecutive_pair_count": 0.05,
    "back_repeat_from_previous_count": 0.05,
}

_CONTINUOUS_COLUMNS: dict[str, str] = {
    "front_sum": "front_sum",
    "front_span": "front_span",
    "back_sum": "back_sum",
}
_DISCRETE_COLUMNS: dict[str, str] = {
    "front_odd_count": "front_odd_count",
    "front_large_count": "front_large_count",
    "front_zone_signature": "zone_ratio",
    "front_consecutive_pair_count": "consecutive_pair_count",
    "front_same_tail_pair_count": "same_tail_pair_count",
    "front_repeat_from_previous_count": "repeat_from_previous_count",
    "back_odd_count": "back_odd_count",
    "back_large_count": "back_large_count",
    "back_consecutive_pair_count": "back_consecutive_pair_count",
    "back_repeat_from_previous_count": "back_repeat_from_previous_count",
}


class TicketStructureFeatures(BaseModel):
    """Raw front/back structure features retained for candidate-level audit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    front_sum: int
    front_span: int
    front_odd_count: int
    front_even_count: int
    front_large_count: int
    front_small_count: int
    front_zone_1_count: int
    front_zone_2_count: int
    front_zone_3_count: int
    front_consecutive_pair_count: int
    front_same_tail_pair_count: int
    front_repeat_from_previous_count: int
    back_sum: int
    back_odd_count: int
    back_even_count: int
    back_large_count: int
    back_small_count: int
    back_consecutive_pair_count: int
    back_repeat_from_previous_count: int

    @property
    def zone_signature(self) -> str:
        """Return the complete three-zone signature, for example ``2:2:1``."""
        return f"{self.front_zone_1_count}:{self.front_zone_2_count}:{self.front_zone_3_count}"


class StructureFeatureDefinition(BaseModel):
    """Persisted method, parameters, and normalized weight for one concept feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    method: StructureMethod
    parameters: dict[str, JsonValue]
    weight: float = Field(ge=0, le=1)


class HistoricalStructureProfile(BaseModel):
    """Empirical profile fitted exclusively from draws before the target issue."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data_start_issue: str = Field(pattern=r"^\d+$")
    data_cutoff_issue: str = Field(pattern=r"^\d+$")
    history_size: int = Field(ge=1)
    continuous_distributions: dict[str, tuple[float, ...]]
    discrete_frequencies: dict[str, dict[str, int]]
    feature_definitions: tuple[StructureFeatureDefinition, ...]
    front_sum_quantile_edges: tuple[float, float]
    scoring_method: str = "weighted_empirical_v0.4"
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("structure profile is missing the fixed risk disclaimer")
        return value

    @model_validator(mode="after")
    def validate_feature_contract(self) -> HistoricalStructureProfile:
        names = [definition.name for definition in self.feature_definitions]
        expected = set(_CONTINUOUS_COLUMNS) | set(_DISCRETE_COLUMNS)
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("structure feature definitions must cover each concept exactly once")
        if not isclose(
            sum(definition.weight for definition in self.feature_definitions),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("structure feature weights must sum to 1")
        return self

    @property
    def distributions(self) -> dict[str, tuple[float, ...]]:
        """Compatibility accessor for v0.3 continuous distribution consumers."""
        return self.continuous_distributions

    def front_sum_interval(self, value: int) -> str:
        """Classify a sum by historical terciles fitted before the target issue."""
        lower, upper = self.front_sum_quantile_edges
        if value <= lower:
            return "historical_lower_third"
        if value <= upper:
            return "historical_middle_third"
        return "historical_upper_third"


class StructureComponentScore(BaseModel):
    """Auditable score contribution for one non-duplicated concept feature."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: JsonValue
    score: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0, le=1)
    method: StructureMethod
    parameters: dict[str, JsonValue]
    weight: float = Field(ge=0, le=1)


class StructureScore(BaseModel):
    """Weighted structure score with method, parameters, and weight per item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    overall_score: float = Field(ge=0, le=1)
    component_scores: dict[str, float]
    component_details: dict[str, StructureComponentScore]
    scoring_method: str = "weighted_empirical_v0.4"
    risk_disclaimer: str = DISCLAIMER

    @field_validator("risk_disclaimer")
    @classmethod
    def validate_disclaimer(cls, value: str) -> str:
        if value != DISCLAIMER:
            raise ValueError("structure score is missing the fixed risk disclaimer")
        return value


def _validate_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    resolved = dict(DEFAULT_STRUCTURE_WEIGHTS if weights is None else weights)
    expected = set(_CONTINUOUS_COLUMNS) | set(_DISCRETE_COLUMNS)
    if set(resolved) != expected:
        raise ValueError("structure weights must cover every concept feature exactly once")
    if any(not np.isfinite(weight) or weight < 0 for weight in resolved.values()):
        raise ValueError("structure weights must be finite and non-negative")
    if not isclose(sum(resolved.values()), 1.0, abs_tol=1e-9):
        raise ValueError("structure weights must sum to 1")
    return resolved


def fit_structure_profile(
    history: pd.DataFrame,
    *,
    feature_weights: Mapping[str, float] | None = None,
    laplace_alpha: float = 1.0,
) -> HistoricalStructureProfile:
    """Fit continuous and discrete empirical distributions without future data."""
    if not np.isfinite(laplace_alpha) or laplace_alpha <= 0:
        raise ValueError("laplace_alpha must be finite and greater than zero")
    weights = _validate_weights(feature_weights)
    validated = validate_draw_dataframe(history)
    featured = engineer_features(validated)
    continuous = {
        name: tuple(sorted(featured[column].dropna().astype(float).tolist()))
        for name, column in _CONTINUOUS_COLUMNS.items()
    }
    discrete = {
        name: dict(
            sorted(Counter(str(value) for value in featured[column].dropna().tolist()).items())
        )
        for name, column in _DISCRETE_COLUMNS.items()
    }
    definitions = tuple(
        StructureFeatureDefinition(
            name=name,
            method=(
                "empirical_quantile_centrality"
                if name in _CONTINUOUS_COLUMNS
                else "laplace_smoothed_empirical_frequency"
            ),
            parameters=(
                {"quantile_method": "empirical_midrank", "center_quantile": 0.5}
                if name in _CONTINUOUS_COLUMNS
                else {"alpha": laplace_alpha, "normalization": "relative_to_profile_mode"}
            ),
            weight=weights[name],
        )
        for name in weights
    )
    front_sums = np.asarray(continuous["front_sum"], dtype=float)
    lower, upper = np.quantile(front_sums, (1 / 3, 2 / 3), method="linear")
    return HistoricalStructureProfile(
        data_start_issue=str(validated.iloc[0]["issue"]),
        data_cutoff_issue=str(validated.iloc[-1]["issue"]),
        history_size=len(validated),
        continuous_distributions=continuous,
        discrete_frequencies=discrete,
        feature_definitions=definitions,
        front_sum_quantile_edges=(float(lower), float(upper)),
    )


def compute_ticket_structure(
    front_numbers: Sequence[int],
    back_numbers: Sequence[int],
    *,
    previous_draw: DrawRecord | None,
) -> TicketStructureFeatures:
    """Compute raw structure, comparing repeats only with the latest known draw."""
    front_features = compute_front_features(front_numbers)
    back_features = compute_back_features(back_numbers)
    front = tuple(sorted(int(number) for number in front_numbers))
    back = tuple(sorted(int(number) for number in back_numbers))
    front_values = asdict(front_features)
    back_values = asdict(back_features)
    return TicketStructureFeatures(
        front_sum=front_values["front_sum"],
        front_span=front_values["front_span"],
        front_odd_count=front_values["front_odd_count"],
        front_even_count=front_values["front_even_count"],
        front_large_count=front_values["front_large_count"],
        front_small_count=front_values["front_small_count"],
        front_zone_1_count=front_values["front_zone_1_count"],
        front_zone_2_count=front_values["front_zone_2_count"],
        front_zone_3_count=front_values["front_zone_3_count"],
        front_consecutive_pair_count=front_values["consecutive_pair_count"],
        front_same_tail_pair_count=front_values["same_tail_pair_count"],
        front_repeat_from_previous_count=(
            0
            if previous_draw is None
            else len(set(front).intersection(previous_draw.front_numbers))
        ),
        back_sum=back_values["back_sum"],
        back_odd_count=back_values["back_odd_count"],
        back_even_count=back_values["back_even_count"],
        back_large_count=back_values["back_large_count"],
        back_small_count=back_values["back_small_count"],
        back_consecutive_pair_count=back_values["back_consecutive_pair_count"],
        back_repeat_from_previous_count=(
            0 if previous_draw is None else len(set(back).intersection(previous_draw.back_numbers))
        ),
    )


def _empirical_quantile_centrality(value: float, distribution: tuple[float, ...]) -> float:
    if not distribution:
        raise ValueError("empirical distribution cannot be empty")
    if value < distribution[0] or value > distribution[-1]:
        return 0.0
    lower = bisect_left(distribution, value)
    upper = bisect_right(distribution, value)
    midrank = (lower + (upper - lower) / 2) / len(distribution)
    return max(0.0, 1.0 - 2.0 * abs(midrank - 0.5))


def _discrete_feature_values(features: TicketStructureFeatures) -> dict[str, int | str]:
    return {
        "front_odd_count": features.front_odd_count,
        "front_large_count": features.front_large_count,
        "front_zone_signature": features.zone_signature,
        "front_consecutive_pair_count": features.front_consecutive_pair_count,
        "front_same_tail_pair_count": features.front_same_tail_pair_count,
        "front_repeat_from_previous_count": features.front_repeat_from_previous_count,
        "back_odd_count": features.back_odd_count,
        "back_large_count": features.back_large_count,
        "back_consecutive_pair_count": features.back_consecutive_pair_count,
        "back_repeat_from_previous_count": features.back_repeat_from_previous_count,
    }


def score_ticket_structure(
    features: TicketStructureFeatures,
    profile: HistoricalStructureProfile,
) -> StructureScore:
    """Score each distinct concept using its persisted empirical method and weight."""
    continuous_values = {
        "front_sum": features.front_sum,
        "front_span": features.front_span,
        "back_sum": features.back_sum,
    }
    discrete_values = _discrete_feature_values(features)
    details: dict[str, StructureComponentScore] = {}
    for definition in profile.feature_definitions:
        if definition.method == "empirical_quantile_centrality":
            value: int | str = continuous_values[definition.name]
            score = _empirical_quantile_centrality(
                float(value),
                profile.continuous_distributions[definition.name],
            )
            parameters = dict(definition.parameters)
            parameters["history_observations"] = len(
                profile.continuous_distributions[definition.name]
            )
        else:
            value = discrete_values[definition.name]
            key = str(value)
            frequencies = profile.discrete_frequencies[definition.name]
            alpha = float(definition.parameters["alpha"])
            maximum_count = max(frequencies.values(), default=0)
            score = (frequencies.get(key, 0) + alpha) / (maximum_count + alpha)
            parameters = dict(definition.parameters)
            parameters.update(
                {
                    "history_observations": sum(frequencies.values()),
                    "observed_category_count": len(frequencies),
                    "candidate_observed_count": frequencies.get(key, 0),
                }
            )
        details[definition.name] = StructureComponentScore(
            value=value,
            score=score,
            weighted_score=score * definition.weight,
            method=definition.method,
            parameters=parameters,
            weight=definition.weight,
        )
    return StructureScore(
        overall_score=sum(component.weighted_score for component in details.values()),
        component_scores={name: component.score for name, component in details.items()},
        component_details=details,
    )

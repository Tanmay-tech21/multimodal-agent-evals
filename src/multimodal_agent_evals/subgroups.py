"""Stratified agreement reports with deterministic item-level bootstrapping."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .agreement import AgreementReport, Rating, RatingStudy, evaluate_agreement
from .schema import TraceValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


@dataclass(frozen=True)
class SubgroupManifest:
    dimensions: tuple[str, ...]
    assignments: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if not self.dimensions:
            raise TraceValidationError("subgroup dimensions must be non-empty")
        for dimension in self.dimensions:
            _identifier("dimension", dimension)
        if len(set(self.dimensions)) != len(self.dimensions):
            raise TraceValidationError("subgroup dimensions must be unique")
        if not self.assignments:
            raise TraceValidationError("subgroup assignments must be non-empty")
        for item_id, values in self.assignments.items():
            _identifier("item_id", item_id)
            if set(values) != set(self.dimensions):
                raise TraceValidationError(
                    "every subgroup item must define exactly the declared dimensions"
                )
            for dimension, group in values.items():
                _identifier(dimension, group)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SubgroupManifest":
        if not isinstance(value, Mapping) or set(value) != {"dimensions", "items"}:
            raise TraceValidationError("invalid subgroup manifest fields")
        dimensions = value["dimensions"]
        items = value["items"]
        if not isinstance(dimensions, list) or not isinstance(items, list):
            raise TraceValidationError("subgroup dimensions and items must be lists")
        assignments: dict[str, dict[str, str]] = {}
        for item in items:
            if not isinstance(item, Mapping) or "item_id" not in item:
                raise TraceValidationError("each subgroup item requires an item_id")
            item_id = item["item_id"]
            if item_id in assignments:
                raise TraceValidationError("subgroup item identifiers must be unique")
            assignments[item_id] = {
                key: value for key, value in item.items() if key != "item_id"
            }
        return cls(tuple(dimensions), assignments)

    def validate_against(self, study: RatingStudy) -> None:
        if set(self.assignments) != set(study.items):
            raise TraceValidationError(
                "subgroup assignments must cover exactly the rating-study items"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimensions": list(self.dimensions),
            "items": [
                {"item_id": item_id, **dict(values)}
                for item_id, values in self.assignments.items()
            ],
        }


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float | None
    upper: float | None
    confidence_level: float
    valid_resamples: int
    total_resamples: int

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence_level": self.confidence_level,
            "valid_resamples": self.valid_resamples,
            "total_resamples": self.total_resamples,
        }


@dataclass(frozen=True)
class SubgroupResult:
    group: str
    item_count: int
    agreement: AgreementReport
    alpha_interval: BootstrapInterval
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "item_count": self.item_count,
            "agreement": self.agreement.to_dict(),
            "alpha_interval": self.alpha_interval.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SubgroupReport:
    study_id: str
    dimension: str
    seed: int
    results: tuple[SubgroupResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "dimension": self.dimension,
            "bootstrap_seed": self.seed,
            "results": [result.to_dict() for result in self.results],
        }


def _subset(study: RatingStudy, items: tuple[str, ...], suffix: str) -> RatingStudy:
    selected = set(items)
    return RatingStudy(
        study_id=f"{study.study_id}:{suffix}",
        labels=study.labels,
        items=items,
        raters=study.raters,
        ratings=tuple(rating for rating in study.ratings if rating.item_id in selected),
    )


def _resample(study: RatingStudy, rng: random.Random) -> RatingStudy:
    by_item: dict[str, list[Rating]] = {item: [] for item in study.items}
    for rating in study.ratings:
        by_item[rating.item_id].append(rating)
    sampled = rng.choices(study.items, k=len(study.items))
    items: list[str] = []
    ratings: list[Rating] = []
    for index, source_id in enumerate(sampled):
        item_id = f"resample-{index}"
        items.append(item_id)
        ratings.extend(
            Rating(item_id, rating.rater_id, rating.label)
            for rating in by_item[source_id]
        )
    return RatingStudy(
        study_id=f"{study.study_id}:bootstrap",
        labels=study.labels,
        items=tuple(items),
        raters=study.raters,
        ratings=tuple(ratings),
    )


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_alpha(
    study: RatingStudy,
    *,
    resamples: int,
    confidence_level: float,
    rng: random.Random,
) -> BootstrapInterval:
    values = [
        value
        for _ in range(resamples)
        if (value := evaluate_agreement(_resample(study, rng)).krippendorff_alpha)
        is not None
    ]
    if not values:
        return BootstrapInterval(None, None, confidence_level, 0, resamples)
    tail = (1.0 - confidence_level) / 2.0
    return BootstrapInterval(
        _quantile(values, tail),
        _quantile(values, 1.0 - tail),
        confidence_level,
        len(values),
        resamples,
    )


def evaluate_subgroups(
    study: RatingStudy,
    manifest: SubgroupManifest,
    dimension: str,
    *,
    resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
    minimum_items: int = 10,
) -> SubgroupReport:
    """Estimate subgroup agreement and item-bootstrap uncertainty intervals."""
    manifest.validate_against(study)
    if dimension not in manifest.dimensions:
        raise TraceValidationError("requested subgroup dimension is not declared")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise TraceValidationError("resamples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise TraceValidationError("confidence_level must be between zero and one")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TraceValidationError("bootstrap seed must be an integer")
    if not isinstance(minimum_items, int) or minimum_items < 2:
        raise TraceValidationError("minimum_items must be an integer of at least two")

    groups: dict[str, list[str]] = {}
    for item_id in study.items:
        group = manifest.assignments[item_id][dimension]
        groups.setdefault(group, []).append(item_id)

    rng = random.Random(seed)
    results: list[SubgroupResult] = []
    for group, item_ids in sorted(groups.items()):
        subgroup = _subset(study, tuple(item_ids), f"{dimension}:{group}")
        agreement = evaluate_agreement(subgroup)
        interval = _bootstrap_alpha(
            subgroup,
            resamples=resamples,
            confidence_level=confidence_level,
            rng=rng,
        )
        warnings: list[str] = []
        if len(item_ids) < minimum_items:
            warnings.append("limited_items")
        if interval.valid_resamples < 0.8 * interval.total_resamples:
            warnings.append("many_undefined_resamples")
        results.append(
            SubgroupResult(
                group=group,
                item_count=len(item_ids),
                agreement=agreement,
                alpha_interval=interval,
                warnings=tuple(warnings),
            )
        )
    return SubgroupReport(study.study_id, dimension, seed, tuple(results))

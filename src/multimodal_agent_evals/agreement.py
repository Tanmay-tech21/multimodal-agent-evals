"""Agreement diagnostics for human reviewers and model-based evaluators."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

from .metrics import MetricValue
from .schema import TraceValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class Rating:
    item_id: str
    rater_id: str
    label: str

    def __post_init__(self) -> None:
        _identifier("item_id", self.item_id)
        _identifier("rater_id", self.rater_id)
        _identifier("label", self.label)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Rating":
        data = _mapping("rating", value)
        if set(data) != {"item_id", "rater_id", "label"}:
            raise TraceValidationError("invalid rating fields")
        return cls(data["item_id"], data["rater_id"], data["label"])

    def to_dict(self) -> dict[str, str]:
        return {
            "item_id": self.item_id,
            "rater_id": self.rater_id,
            "label": self.label,
        }


@dataclass(frozen=True)
class RatingStudy:
    study_id: str
    labels: tuple[str, ...]
    items: tuple[str, ...]
    raters: tuple[str, ...]
    ratings: tuple[Rating, ...]

    def __post_init__(self) -> None:
        _identifier("study_id", self.study_id)
        for name, values in (
            ("label", self.labels),
            ("item_id", self.items),
            ("rater_id", self.raters),
        ):
            if not values:
                raise TraceValidationError(f"{name}s must be non-empty")
            for value in values:
                _identifier(name, value)
            if len(set(values)) != len(values):
                raise TraceValidationError(f"{name}s must be unique")
        if len(self.raters) < 2:
            raise TraceValidationError("an agreement study requires at least two raters")
        if not self.ratings:
            raise TraceValidationError("ratings must be non-empty")

        declared_items = set(self.items)
        declared_raters = set(self.raters)
        declared_labels = set(self.labels)
        keys: list[tuple[str, str]] = []
        for rating in self.ratings:
            if rating.item_id not in declared_items:
                raise TraceValidationError("rating references an undeclared item")
            if rating.rater_id not in declared_raters:
                raise TraceValidationError("rating references an undeclared rater")
            if rating.label not in declared_labels:
                raise TraceValidationError("rating uses an undeclared label")
            keys.append((rating.item_id, rating.rater_id))
        if len(set(keys)) != len(keys):
            raise TraceValidationError("each rater may label an item at most once")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RatingStudy":
        data = _mapping("rating study", value)
        expected = {"study_id", "labels", "items", "raters", "ratings"}
        if set(data) != expected:
            raise TraceValidationError("invalid rating study fields")
        if any(not isinstance(data[name], list) for name in expected - {"study_id"}):
            raise TraceValidationError("labels, items, raters and ratings must be lists")
        return cls(
            study_id=data["study_id"],
            labels=tuple(data["labels"]),
            items=tuple(data["items"]),
            raters=tuple(data["raters"]),
            ratings=tuple(Rating.from_dict(item) for item in data["ratings"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "labels": list(self.labels),
            "items": list(self.items),
            "raters": list(self.raters),
            "ratings": [rating.to_dict() for rating in self.ratings],
        }


@dataclass(frozen=True)
class PairwiseAgreement:
    first_rater: str
    second_rater: str
    overlap: int
    observed_agreement: MetricValue
    cohen_kappa: float | None
    confusion: Mapping[str, Mapping[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raters": [self.first_rater, self.second_rater],
            "overlap": self.overlap,
            "observed_agreement": self.observed_agreement.to_dict(),
            "cohen_kappa": self.cohen_kappa,
            "confusion": {
                actual: dict(predicted) for actual, predicted in self.confusion.items()
            },
        }


@dataclass(frozen=True)
class AgreementReport:
    study_id: str
    completeness: MetricValue
    rater_coverage: Mapping[str, MetricValue]
    label_counts: Mapping[str, int]
    unanimous_items: MetricValue
    pairwise: tuple[PairwiseAgreement, ...]
    krippendorff_alpha: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "completeness": self.completeness.to_dict(),
            "rater_coverage": {
                rater: metric.to_dict() for rater, metric in self.rater_coverage.items()
            },
            "label_counts": dict(self.label_counts),
            "unanimous_items": self.unanimous_items.to_dict(),
            "pairwise": [item.to_dict() for item in self.pairwise],
            "krippendorff_alpha": self.krippendorff_alpha,
        }


def _pairwise_report(
    first: str,
    second: str,
    labels: tuple[str, ...],
    by_rater: Mapping[str, Mapping[str, str]],
) -> PairwiseAgreement:
    shared = sorted(set(by_rater[first]) & set(by_rater[second]))
    pairs = [(by_rater[first][item], by_rater[second][item]) for item in shared]
    agreements = sum(left == right for left, right in pairs)
    observed = MetricValue(agreements, len(pairs))
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)

    kappa: float | None = None
    if pairs:
        expected = sum(
            left_counts[label] * right_counts[label] for label in labels
        ) / len(pairs) ** 2
        if expected < 1.0:
            kappa = (observed.value - expected) / (1.0 - expected)  # type: ignore[operator]

    confusion = {
        left: {right: sum(pair == (left, right) for pair in pairs) for right in labels}
        for left in labels
    }
    return PairwiseAgreement(
        first,
        second,
        len(pairs),
        observed,
        kappa,
        confusion,
    )


def _nominal_alpha(
    labels: tuple[str, ...], by_item: Mapping[str, tuple[str, ...]]
) -> float | None:
    coincidence = {left: Counter({right: 0.0 for right in labels}) for left in labels}
    for item_labels in by_item.values():
        if len(item_labels) < 2:
            continue
        counts = Counter(item_labels)
        scale = len(item_labels) - 1
        for left in labels:
            for right in labels:
                pairs = counts[left] * (counts[right] - (left == right))
                coincidence[left][right] += pairs / scale

    marginals = {label: sum(coincidence[label].values()) for label in labels}
    total = sum(marginals.values())
    if total <= 1:
        return None
    observed_disagreement = sum(
        coincidence[left][right]
        for left in labels
        for right in labels
        if left != right
    ) / total
    expected_disagreement = sum(
        count * (total - count) for count in marginals.values()
    ) / (total * (total - 1))
    if expected_disagreement == 0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def evaluate_agreement(study: RatingStudy) -> AgreementReport:
    """Measure label coverage and nominal agreement without imputing missing ratings."""
    by_rater: dict[str, dict[str, str]] = {rater: {} for rater in study.raters}
    by_item_lists: dict[str, list[str]] = {item: [] for item in study.items}
    for rating in study.ratings:
        by_rater[rating.rater_id][rating.item_id] = rating.label
        by_item_lists[rating.item_id].append(rating.label)
    by_item = {item: tuple(labels) for item, labels in by_item_lists.items()}

    comparable = [labels for labels in by_item.values() if len(labels) >= 2]
    unanimous = sum(len(set(labels)) == 1 for labels in comparable)
    pairwise = tuple(
        _pairwise_report(first, second, study.labels, by_rater)
        for first, second in combinations(study.raters, 2)
    )
    label_counts = Counter(rating.label for rating in study.ratings)
    return AgreementReport(
        study_id=study.study_id,
        completeness=MetricValue(
            len(study.ratings), len(study.items) * len(study.raters)
        ),
        rater_coverage={
            rater: MetricValue(len(by_rater[rater]), len(study.items))
            for rater in study.raters
        },
        label_counts={label: label_counts[label] for label in study.labels},
        unanimous_items=MetricValue(unanimous, len(comparable)),
        pairwise=pairwise,
        krippendorff_alpha=_nominal_alpha(study.labels, by_item),
    )

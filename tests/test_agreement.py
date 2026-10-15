import pytest

from multimodal_agent_evals import (
    RatingStudy,
    TraceValidationError,
    evaluate_agreement,
)


def study_dict(**overrides) -> dict:
    value = {
        "study_id": "study-1",
        "labels": ["yes", "no"],
        "items": ["item-1", "item-2", "item-3"],
        "raters": ["reviewer-a", "reviewer-b"],
        "ratings": [
            {"item_id": "item-1", "rater_id": "reviewer-a", "label": "yes"},
            {"item_id": "item-2", "rater_id": "reviewer-a", "label": "no"},
            {"item_id": "item-3", "rater_id": "reviewer-a", "label": "yes"},
            {"item_id": "item-1", "rater_id": "reviewer-b", "label": "yes"},
            {"item_id": "item-2", "rater_id": "reviewer-b", "label": "no"},
            {"item_id": "item-3", "rater_id": "reviewer-b", "label": "yes"},
        ],
    }
    value.update(overrides)
    return value


def test_study_round_trips() -> None:
    raw = study_dict()
    assert RatingStudy.from_dict(raw).to_dict() == raw


def test_perfect_varied_ratings_have_unit_agreement() -> None:
    report = evaluate_agreement(RatingStudy.from_dict(study_dict()))
    assert report.completeness.value == 1.0
    assert report.unanimous_items.value == 1.0
    assert report.pairwise[0].observed_agreement.value == 1.0
    assert report.pairwise[0].cohen_kappa == pytest.approx(1.0)
    assert report.krippendorff_alpha == pytest.approx(1.0)


def test_missing_ratings_are_reported_not_imputed() -> None:
    raw = study_dict()
    raw["ratings"].pop()
    report = evaluate_agreement(RatingStudy.from_dict(raw))
    assert report.completeness.to_dict() == {
        "numerator": 5,
        "denominator": 6,
        "value": pytest.approx(5 / 6),
    }
    assert report.rater_coverage["reviewer-b"].numerator == 2
    assert report.pairwise[0].overlap == 2


def test_confusion_matrix_preserves_label_direction() -> None:
    raw = study_dict()
    raw["ratings"][-1]["label"] = "no"
    pair = evaluate_agreement(RatingStudy.from_dict(raw)).pairwise[0]
    assert pair.confusion["yes"]["no"] == 1
    assert pair.observed_agreement.to_dict()["numerator"] == 2


def test_disjoint_raters_have_undefined_pairwise_metrics() -> None:
    raw = study_dict(
        ratings=[
            {"item_id": "item-1", "rater_id": "reviewer-a", "label": "yes"},
            {"item_id": "item-2", "rater_id": "reviewer-b", "label": "no"},
        ]
    )
    pair = evaluate_agreement(RatingStudy.from_dict(raw)).pairwise[0]
    assert pair.overlap == 0
    assert pair.observed_agreement.value is None
    assert pair.cohen_kappa is None


def test_constant_marginals_make_chance_correction_undefined() -> None:
    raw = study_dict(labels=["yes"])
    for rating in raw["ratings"]:
        rating["label"] = "yes"
    report = evaluate_agreement(RatingStudy.from_dict(raw))
    assert report.pairwise[0].observed_agreement.value == 1.0
    assert report.pairwise[0].cohen_kappa is None
    assert report.krippendorff_alpha is None


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("items", ["item-1", "item-1"], "item_ids must be unique"),
        ("raters", ["reviewer-a"], "at least two raters"),
        ("ratings", [], "ratings must be non-empty"),
    ],
)
def test_study_rejects_invalid_collections(field, value, message) -> None:
    with pytest.raises(TraceValidationError, match=message):
        RatingStudy.from_dict(study_dict(**{field: value}))


@pytest.mark.parametrize(
    "change,message",
    [
        ({"item_id": "missing"}, "undeclared item"),
        ({"rater_id": "missing"}, "undeclared rater"),
        ({"label": "maybe"}, "undeclared label"),
    ],
)
def test_study_rejects_undeclared_rating_values(change, message) -> None:
    raw = study_dict()
    raw["ratings"][0].update(change)
    with pytest.raises(TraceValidationError, match=message):
        RatingStudy.from_dict(raw)


def test_study_rejects_duplicate_item_rater_pair() -> None:
    raw = study_dict()
    raw["ratings"].append(dict(raw["ratings"][0]))
    with pytest.raises(TraceValidationError, match="at most once"):
        RatingStudy.from_dict(raw)


def test_three_raters_produce_every_pair() -> None:
    raw = study_dict(raters=["reviewer-a", "reviewer-b", "judge-model"])
    raw["ratings"].append(
        {"item_id": "item-1", "rater_id": "judge-model", "label": "yes"}
    )
    report = evaluate_agreement(RatingStudy.from_dict(raw))
    assert len(report.pairwise) == 3
    assert report.completeness.denominator == 9

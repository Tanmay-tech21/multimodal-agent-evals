import pytest

from multimodal_agent_evals import (
    RatingStudy,
    SubgroupManifest,
    TraceValidationError,
    evaluate_subgroups,
)


def fixture():
    study = RatingStudy.from_dict(
        {
            "study_id": "study-1",
            "labels": ["yes", "no"],
            "items": ["item-1", "item-2", "item-3", "item-4"],
            "raters": ["reviewer-a", "reviewer-b"],
            "ratings": [
                {"item_id": item, "rater_id": rater, "label": label}
                for item, labels in {
                    "item-1": ("yes", "yes"),
                    "item-2": ("no", "no"),
                    "item-3": ("yes", "no"),
                    "item-4": ("no", "yes"),
                }.items()
                for rater, label in zip(("reviewer-a", "reviewer-b"), labels)
            ],
        }
    )
    manifest = SubgroupManifest.from_dict(
        {
            "dimensions": ["modality"],
            "items": [
                {"item_id": "item-1", "modality": "image"},
                {"item_id": "item-2", "modality": "image"},
                {"item_id": "item-3", "modality": "text"},
                {"item_id": "item-4", "modality": "text"},
            ],
        }
    )
    return study, manifest


def test_manifest_round_trips() -> None:
    _, manifest = fixture()
    assert SubgroupManifest.from_dict(manifest.to_dict()) == manifest


def test_subgroups_preserve_item_counts_and_point_estimates() -> None:
    study, manifest = fixture()
    report = evaluate_subgroups(
        study, manifest, "modality", resamples=50, seed=7, minimum_items=2
    )
    assert [result.group for result in report.results] == ["image", "text"]
    assert [result.item_count for result in report.results] == [2, 2]
    assert report.results[0].agreement.krippendorff_alpha == 1.0
    assert report.results[1].agreement.krippendorff_alpha < 0.0


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    study, manifest = fixture()
    first = evaluate_subgroups(study, manifest, "modality", resamples=100, seed=9)
    second = evaluate_subgroups(study, manifest, "modality", resamples=100, seed=9)
    assert first.to_dict() == second.to_dict()


def test_sparse_and_degenerate_resamples_are_flagged() -> None:
    study, manifest = fixture()
    result = evaluate_subgroups(
        study, manifest, "modality", resamples=100, seed=2, minimum_items=10
    ).results[0]
    assert "limited_items" in result.warnings
    assert result.alpha_interval.valid_resamples < 100
    assert "many_undefined_resamples" in result.warnings


def test_manifest_must_cover_exactly_the_study_items() -> None:
    study, manifest = fixture()
    incomplete = SubgroupManifest(
        manifest.dimensions,
        {key: value for key, value in manifest.assignments.items() if key != "item-4"},
    )
    with pytest.raises(TraceValidationError, match="cover exactly"):
        evaluate_subgroups(study, incomplete, "modality")


def test_undeclared_dimension_is_rejected() -> None:
    study, manifest = fixture()
    with pytest.raises(TraceValidationError, match="not declared"):
        evaluate_subgroups(study, manifest, "task_family")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"resamples": 0}, "positive integer"),
        ({"confidence_level": 1.0}, "between zero and one"),
        ({"seed": True}, "seed must be an integer"),
        ({"minimum_items": 1}, "at least two"),
    ],
)
def test_invalid_bootstrap_configuration_is_rejected(kwargs, message) -> None:
    study, manifest = fixture()
    with pytest.raises(TraceValidationError, match=message):
        evaluate_subgroups(study, manifest, "modality", **kwargs)


def test_manifest_rejects_duplicate_items() -> None:
    raw = {
        "dimensions": ["modality"],
        "items": [
            {"item_id": "item-1", "modality": "image"},
            {"item_id": "item-1", "modality": "text"},
        ],
    }
    with pytest.raises(TraceValidationError, match="identifiers must be unique"):
        SubgroupManifest.from_dict(raw)


def test_manifest_rejects_missing_declared_dimension() -> None:
    with pytest.raises(TraceValidationError, match="exactly the declared dimensions"):
        SubgroupManifest.from_dict(
            {"dimensions": ["modality"], "items": [{"item_id": "item-1"}]}
        )

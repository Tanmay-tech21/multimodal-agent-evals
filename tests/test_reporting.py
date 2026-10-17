import json

import pytest

from multimodal_agent_evals import (
    EvidenceClaim,
    EvidenceSection,
    EvaluationSnapshot,
    TraceValidationError,
)


DIGEST = "a" * 64


def snapshot(**overrides) -> EvaluationSnapshot:
    values = {
        "project": "project",
        "schema_version": "1.0.0",
        "configuration": {"seed": 7},
        "sections": (
            EvidenceSection(
                "metrics",
                "Synthetic metric fixture.",
                {"examples/input.json": DIGEST},
                {"accuracy": {"numerator": 3, "denominator": 4, "value": 0.75}},
                ("Synthetic data only.",),
            ),
        ),
        "claims": (
            EvidenceClaim(
                "accuracy_claim",
                "Three of four cases passed.",
                "/sections/metrics/result/accuracy/numerator",
                3,
            ),
        ),
    }
    values.update(overrides)
    return EvaluationSnapshot(**values)


def test_snapshot_round_trips_and_preserves_stable_section_keys() -> None:
    item = snapshot()
    assert EvaluationSnapshot.from_dict(item.to_dict()).to_dict() == item.to_dict()
    assert list(item.to_dict()["sections"]) == ["metrics"]


def test_markdown_is_deterministic_and_contains_claim_binding() -> None:
    item = snapshot()
    assert item.render_markdown() == item.render_markdown()
    assert "/sections/metrics/result/accuracy/numerator" in item.render_markdown()
    assert "Observed value: `3`" in item.render_markdown()


def test_claim_must_match_observed_evidence() -> None:
    claim = EvidenceClaim(
        "wrong", "Incorrect claim.", "/sections/metrics/result/accuracy/numerator", 4
    )
    with pytest.raises(TraceValidationError, match="does not match"):
        snapshot(claims=(claim,))


def test_unresolved_claim_path_is_rejected() -> None:
    claim = EvidenceClaim("missing", "Missing.", "/sections/metrics/result/missing", 1)
    with pytest.raises(TraceValidationError, match="does not resolve"):
        snapshot(claims=(claim,))


def test_duplicate_section_and_claim_identifiers_are_rejected() -> None:
    item = snapshot()
    with pytest.raises(TraceValidationError, match="section identifiers"):
        snapshot(sections=item.sections * 2)
    with pytest.raises(TraceValidationError, match="claim identifiers"):
        snapshot(claims=item.claims * 2)


def test_source_fingerprint_must_be_sha256() -> None:
    with pytest.raises(TraceValidationError, match="SHA-256"):
        EvidenceSection(
            "bad", "Bad digest.", {"input": "abc"}, {"value": 1}, ()
        )


def test_non_finite_result_is_rejected() -> None:
    with pytest.raises(TraceValidationError, match="finite JSON"):
        EvidenceSection(
            "bad", "Bad value.", {"input": DIGEST}, {"value": float("nan")}, ()
        )


def test_serialized_snapshot_is_strict_json() -> None:
    rendered = json.dumps(snapshot().to_dict(), allow_nan=False, sort_keys=True)
    assert json.loads(rendered)["claims"][0]["expected_value"] == 3

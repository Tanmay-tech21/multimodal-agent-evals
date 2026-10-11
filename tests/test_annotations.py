import pytest

from multimodal_agent_evals import (
    AdjudicationDecision,
    AgentTrace,
    AnnotationSet,
    FailureAnnotation,
    FailureCode,
    Severity,
    TraceValidationError,
    summarise_annotations,
)

from test_schema import valid_trace


def annotation_dict(**overrides) -> dict:
    value = {
        "annotation_id": "annotation-1",
        "trace_id": "trace-1",
        "annotator_id": "reviewer-a",
        "failure_code": "verification.contradiction_ignored",
        "event_ids": ["event-2", "event-3"],
        "severity": "major",
        "rationale": "The answer contradicts the tool result.",
    }
    value.update(overrides)
    return value


def annotation_set(annotator_id="reviewer-a", **annotation_overrides) -> AnnotationSet:
    raw = annotation_dict(annotator_id=annotator_id, **annotation_overrides)
    return AnnotationSet.from_dict(
        {
            "trace_id": "trace-1",
            "annotator_id": annotator_id,
            "annotations": [raw],
        }
    )


def decision_dict(**overrides) -> dict:
    value = {
        "decision_id": "decision-1",
        "trace_id": "trace-1",
        "adjudicator_id": "lead-reviewer",
        "source_annotation_ids": ["annotation-1", "annotation-2"],
        "failure_code": "verification.contradiction_ignored",
        "event_ids": ["event-2", "event-3"],
        "severity": "major",
        "rationale": "Verification is the earliest actionable failure.",
    }
    value.update(overrides)
    return value


def test_failure_annotation_round_trips_and_exposes_stage() -> None:
    raw = annotation_dict()
    annotation = FailureAnnotation.from_dict(raw)
    assert annotation.to_dict() == raw
    assert annotation.failure_code.stage.value == "verification"


def test_annotation_set_validates_ordered_trace_references() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    item = annotation_set()
    item.validate_against(trace)
    reversed_item = annotation_set(event_ids=["event-3", "event-2"])
    with pytest.raises(TraceValidationError, match="trace order"):
        reversed_item.validate_against(trace)


def test_annotation_set_rejects_unknown_event() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    item = annotation_set(event_ids=["missing-event"])
    with pytest.raises(TraceValidationError, match="unknown events"):
        item.validate_against(trace)


def test_annotation_must_match_its_set_identity() -> None:
    annotation = FailureAnnotation.from_dict(annotation_dict(annotator_id="other"))
    with pytest.raises(TraceValidationError, match="annotator_id"):
        AnnotationSet("trace-1", "reviewer-a", (annotation,))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"event_ids": []}, "non-empty list"),
        ({"rationale": "  "}, "rationale"),
        ({"failure_code": "imaginary.failure"}, "invalid annotation"),
    ],
)
def test_annotation_rejects_invalid_evidence(overrides, message) -> None:
    with pytest.raises(TraceValidationError, match=message):
        FailureAnnotation.from_dict(annotation_dict(**overrides))


def test_adjudication_validates_sources_from_distinct_annotators() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    first = annotation_set()
    second = annotation_set(
        annotator_id="reviewer-b",
        annotation_id="annotation-2",
        failure_code="completion.incorrect_answer",
        event_ids=["event-3"],
    )
    decision = AdjudicationDecision.from_dict(decision_dict())
    decision.validate_against(trace, (first, second))


def test_adjudication_rejects_unknown_source() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    first = annotation_set()
    decision = AdjudicationDecision.from_dict(decision_dict())
    with pytest.raises(TraceValidationError, match="unknown annotations"):
        decision.validate_against(trace, (first,))


def test_no_failure_adjudication_has_no_events_or_severity() -> None:
    raw = decision_dict(failure_code=None, event_ids=[], severity=None)
    decision = AdjudicationDecision.from_dict(raw)
    assert decision.to_dict() == raw
    with pytest.raises(TraceValidationError, match="cannot include"):
        AdjudicationDecision.from_dict(
            decision_dict(failure_code=None, event_ids=["event-3"], severity=None)
        )


def test_summary_preserves_disagreement_before_adjudication() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    first = annotation_set()
    second = annotation_set(
        annotator_id="reviewer-b",
        annotation_id="annotation-2",
        failure_code="completion.incorrect_answer",
        event_ids=["event-3"],
    )
    decision = AdjudicationDecision.from_dict(decision_dict())
    summary = summarise_annotations(trace, (first, second), (decision,))
    assert summary["annotators"] == 2
    assert summary["annotations"] == 2
    assert summary["unanimous_codes"] == []
    assert summary["disputed_codes"] == [
        "completion.incorrect_answer",
        "verification.contradiction_ignored",
    ]
    assert summary["adjudications"] == 1


def test_summary_rejects_duplicate_annotator_sets() -> None:
    trace = AgentTrace.from_dict(valid_trace())
    with pytest.raises(TraceValidationError, match="annotator identifiers"):
        summarise_annotations(trace, (annotation_set(), annotation_set()))


def test_failure_decision_requires_severity() -> None:
    with pytest.raises(TraceValidationError, match="require events and severity"):
        AdjudicationDecision.from_dict(decision_dict(severity=None))


def test_enum_construction_is_explicit() -> None:
    annotation = FailureAnnotation(
        "annotation-direct",
        "trace-1",
        "reviewer-a",
        FailureCode.COMPLETION_INCORRECT,
        ("event-3",),
        Severity.MAJOR,
        "The answer is incorrect.",
    )
    assert annotation.failure_code.stage.value == "completion"

"""Failure taxonomy, trace-linked annotations and adjudication records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from .schema import AgentTrace, TraceValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class FailureStage(str, Enum):
    PERCEPTION = "perception"
    GROUNDING = "grounding"
    PLANNING = "planning"
    TOOL_SELECTION = "tool_selection"
    TOOL_ARGUMENTS = "tool_arguments"
    TOOL_EXECUTION = "tool_execution"
    VERIFICATION = "verification"
    COMPLETION = "completion"


class FailureCode(str, Enum):
    PERCEPTION_OMISSION = "perception.omission"
    PERCEPTION_MISREAD = "perception.misread"
    GROUNDING_UNSUPPORTED_CLAIM = "grounding.unsupported_claim"
    GROUNDING_WRONG_EVIDENCE = "grounding.wrong_evidence"
    PLANNING_MISSING_STEP = "planning.missing_step"
    PLANNING_INVALID_ORDER = "planning.invalid_order"
    TOOL_SELECTION_WRONG_TOOL = "tool_selection.wrong_tool"
    TOOL_SELECTION_UNNECESSARY_TOOL = "tool_selection.unnecessary_tool"
    TOOL_ARGUMENTS_INVALID = "tool_arguments.invalid"
    TOOL_ARGUMENTS_WRONG_REFERENCE = "tool_arguments.wrong_reference"
    TOOL_EXECUTION_UNHANDLED_ERROR = "tool_execution.unhandled_error"
    TOOL_EXECUTION_RESULT_MISUSED = "tool_execution.result_misused"
    VERIFICATION_MISSING = "verification.missing"
    VERIFICATION_CONTRADICTION_IGNORED = "verification.contradiction_ignored"
    COMPLETION_INCOMPLETE = "completion.incomplete"
    COMPLETION_INCORRECT = "completion.incorrect_answer"

    @property
    def stage(self) -> FailureStage:
        return FailureStage(self.value.split(".", 1)[0])


class Severity(str, Enum):
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{name} must be an object")
    return dict(value)


def _event_ids(value: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TraceValidationError(f"event_ids must be {qualifier}")
    result = tuple(_identifier("event_id", item) for item in value)
    if len(set(result)) != len(result):
        raise TraceValidationError("event_ids must be unique")
    return result


@dataclass(frozen=True)
class FailureAnnotation:
    annotation_id: str
    trace_id: str
    annotator_id: str
    failure_code: FailureCode
    event_ids: tuple[str, ...]
    severity: Severity
    rationale: str

    def __post_init__(self) -> None:
        _identifier("annotation_id", self.annotation_id)
        _identifier("trace_id", self.trace_id)
        _identifier("annotator_id", self.annotator_id)
        if not self.event_ids:
            raise TraceValidationError("an annotation must reference at least one event")
        for event_id in self.event_ids:
            _identifier("event_id", event_id)
        if len(set(self.event_ids)) != len(self.event_ids):
            raise TraceValidationError("annotation event_ids must be unique")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise TraceValidationError("annotation rationale must be non-empty text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FailureAnnotation":
        data = _mapping("annotation", value)
        expected = {
            "annotation_id",
            "trace_id",
            "annotator_id",
            "failure_code",
            "event_ids",
            "severity",
            "rationale",
        }
        if set(data) != expected:
            raise TraceValidationError("invalid annotation fields")
        try:
            return cls(
                annotation_id=data["annotation_id"],
                trace_id=data["trace_id"],
                annotator_id=data["annotator_id"],
                failure_code=FailureCode(data["failure_code"]),
                event_ids=_event_ids(data["event_ids"]),
                severity=Severity(data["severity"]),
                rationale=data["rationale"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TraceValidationError):
                raise
            raise TraceValidationError("invalid annotation") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_id": self.annotation_id,
            "trace_id": self.trace_id,
            "annotator_id": self.annotator_id,
            "failure_code": self.failure_code.value,
            "event_ids": list(self.event_ids),
            "severity": self.severity.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class AnnotationSet:
    trace_id: str
    annotator_id: str
    annotations: tuple[FailureAnnotation, ...]

    def __post_init__(self) -> None:
        _identifier("trace_id", self.trace_id)
        _identifier("annotator_id", self.annotator_id)
        ids = [annotation.annotation_id for annotation in self.annotations]
        if len(set(ids)) != len(ids):
            raise TraceValidationError("annotation identifiers must be unique within a set")
        signatures = []
        for annotation in self.annotations:
            if annotation.trace_id != self.trace_id:
                raise TraceValidationError("annotation trace_id must match its set")
            if annotation.annotator_id != self.annotator_id:
                raise TraceValidationError("annotation annotator_id must match its set")
            signatures.append((annotation.failure_code, annotation.event_ids))
        if len(set(signatures)) != len(signatures):
            raise TraceValidationError("duplicate failure labels are not permitted")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnnotationSet":
        data = _mapping("annotation set", value)
        if set(data) != {"trace_id", "annotator_id", "annotations"}:
            raise TraceValidationError("invalid annotation set fields")
        if not isinstance(data["annotations"], list):
            raise TraceValidationError("annotations must be a list")
        return cls(
            trace_id=data["trace_id"],
            annotator_id=data["annotator_id"],
            annotations=tuple(
                FailureAnnotation.from_dict(item) for item in data["annotations"]
            ),
        )

    def validate_against(self, trace: AgentTrace) -> None:
        if self.trace_id != trace.trace_id:
            raise TraceValidationError("annotation set does not match the trace")
        positions = {event.event_id: event.index for event in trace.events}
        for annotation in self.annotations:
            missing = set(annotation.event_ids) - positions.keys()
            if missing:
                raise TraceValidationError(
                    f"annotation references unknown events: {sorted(missing)}"
                )
            indices = [positions[event_id] for event_id in annotation.event_ids]
            if indices != sorted(indices):
                raise TraceValidationError("annotation event_ids must follow trace order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "annotator_id": self.annotator_id,
            "annotations": [annotation.to_dict() for annotation in self.annotations],
        }


@dataclass(frozen=True)
class AdjudicationDecision:
    decision_id: str
    trace_id: str
    adjudicator_id: str
    source_annotation_ids: tuple[str, ...]
    failure_code: FailureCode | None
    event_ids: tuple[str, ...]
    severity: Severity | None
    rationale: str

    def __post_init__(self) -> None:
        _identifier("decision_id", self.decision_id)
        _identifier("trace_id", self.trace_id)
        _identifier("adjudicator_id", self.adjudicator_id)
        if len(self.source_annotation_ids) < 2:
            raise TraceValidationError("adjudication requires at least two source annotations")
        for annotation_id in self.source_annotation_ids:
            _identifier("source annotation id", annotation_id)
        if len(set(self.source_annotation_ids)) != len(self.source_annotation_ids):
            raise TraceValidationError("source annotation identifiers must be unique")
        if self.failure_code is None:
            if self.event_ids or self.severity is not None:
                raise TraceValidationError("no-failure decisions cannot include events or severity")
        elif not self.event_ids or self.severity is None:
            raise TraceValidationError("failure decisions require events and severity")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise TraceValidationError("adjudication rationale must be non-empty text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AdjudicationDecision":
        data = _mapping("adjudication", value)
        expected = {
            "decision_id",
            "trace_id",
            "adjudicator_id",
            "source_annotation_ids",
            "failure_code",
            "event_ids",
            "severity",
            "rationale",
        }
        if set(data) != expected:
            raise TraceValidationError("invalid adjudication fields")
        sources = data["source_annotation_ids"]
        if not isinstance(sources, list):
            raise TraceValidationError("source_annotation_ids must be a list")
        try:
            return cls(
                decision_id=data["decision_id"],
                trace_id=data["trace_id"],
                adjudicator_id=data["adjudicator_id"],
                source_annotation_ids=tuple(sources),
                failure_code=(
                    None if data["failure_code"] is None else FailureCode(data["failure_code"])
                ),
                event_ids=_event_ids(data["event_ids"], allow_empty=True),
                severity=None if data["severity"] is None else Severity(data["severity"]),
                rationale=data["rationale"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TraceValidationError):
                raise
            raise TraceValidationError("invalid adjudication") from exc

    def validate_against(
        self, trace: AgentTrace, annotation_sets: Iterable[AnnotationSet]
    ) -> None:
        if self.trace_id != trace.trace_id:
            raise TraceValidationError("adjudication does not match the trace")
        annotations = {
            annotation.annotation_id: annotation
            for annotation_set in annotation_sets
            for annotation in annotation_set.annotations
        }
        missing = set(self.source_annotation_ids) - annotations.keys()
        if missing:
            raise TraceValidationError(
                f"adjudication references unknown annotations: {sorted(missing)}"
            )
        source_annotators = {
            annotations[annotation_id].annotator_id
            for annotation_id in self.source_annotation_ids
        }
        if len(source_annotators) < 2:
            raise TraceValidationError("adjudication sources must represent two annotators")
        positions = {event.event_id: event.index for event in trace.events}
        unknown_events = set(self.event_ids) - positions.keys()
        if unknown_events:
            raise TraceValidationError(
                f"adjudication references unknown events: {sorted(unknown_events)}"
            )
        indices = [positions[event_id] for event_id in self.event_ids]
        if indices != sorted(indices):
            raise TraceValidationError("adjudication event_ids must follow trace order")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "trace_id": self.trace_id,
            "adjudicator_id": self.adjudicator_id,
            "source_annotation_ids": list(self.source_annotation_ids),
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "event_ids": list(self.event_ids),
            "severity": None if self.severity is None else self.severity.value,
            "rationale": self.rationale,
        }


def summarise_annotations(
    trace: AgentTrace,
    annotation_sets: Iterable[AnnotationSet],
    decisions: Iterable[AdjudicationDecision] = (),
) -> dict[str, Any]:
    """Validate and summarise annotation evidence without scoring agent quality."""
    sets = tuple(annotation_sets)
    adjudications = tuple(decisions)
    annotator_ids = [annotation_set.annotator_id for annotation_set in sets]
    if len(set(annotator_ids)) != len(annotator_ids):
        raise TraceValidationError("annotator identifiers must be unique per trace")
    for annotation_set in sets:
        annotation_set.validate_against(trace)
    for decision in adjudications:
        decision.validate_against(trace, sets)

    code_sets = [
        {annotation.failure_code for annotation in annotation_set.annotations}
        for annotation_set in sets
    ]
    union = set().union(*code_sets) if code_sets else set()
    intersection = set.intersection(*code_sets) if code_sets else set()
    stage_counts = {stage.value: 0 for stage in FailureStage}
    code_counts = {code.value: 0 for code in FailureCode}
    for annotation_set in sets:
        for annotation in annotation_set.annotations:
            stage_counts[annotation.failure_code.stage.value] += 1
            code_counts[annotation.failure_code.value] += 1

    return {
        "trace_id": trace.trace_id,
        "annotators": len(sets),
        "annotations": sum(len(item.annotations) for item in sets),
        "stage_counts": {key: value for key, value in stage_counts.items() if value},
        "code_counts": {key: value for key, value in code_counts.items() if value},
        "unanimous_codes": sorted(code.value for code in intersection),
        "disputed_codes": sorted(code.value for code in union - intersection),
        "adjudications": len(adjudications),
    }

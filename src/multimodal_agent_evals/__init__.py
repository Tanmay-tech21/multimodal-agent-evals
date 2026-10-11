"""Trace-first evaluation primitives for multimodal tool-using agents."""

from .annotations import (
    AdjudicationDecision,
    AnnotationSet,
    FailureAnnotation,
    FailureCode,
    FailureStage,
    Severity,
    summarise_annotations,
)
from .schema import (
    AgentTrace,
    EventType,
    InputArtifact,
    Modality,
    TraceEvent,
    TraceValidationError,
    load_jsonl,
)

__all__ = [
    "AdjudicationDecision",
    "AgentTrace",
    "AnnotationSet",
    "EventType",
    "FailureAnnotation",
    "FailureCode",
    "FailureStage",
    "InputArtifact",
    "Modality",
    "Severity",
    "TraceEvent",
    "TraceValidationError",
    "load_jsonl",
    "summarise_annotations",
]

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
from .metrics import (
    ExpectedToolCall,
    MetricValue,
    ToolPlan,
    ToolUseReport,
    evaluate_tool_use,
)
from .grounding import (
    EvidenceReference,
    EvidenceSourceType,
    GroundedClaim,
    GroundingRecord,
    GroundingReport,
    GroundingStatus,
    LocatorType,
    summarise_grounding,
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
    "EvidenceReference",
    "EvidenceSourceType",
    "ExpectedToolCall",
    "FailureAnnotation",
    "FailureCode",
    "FailureStage",
    "InputArtifact",
    "GroundedClaim",
    "GroundingRecord",
    "GroundingReport",
    "GroundingStatus",
    "LocatorType",
    "Modality",
    "MetricValue",
    "Severity",
    "TraceEvent",
    "TraceValidationError",
    "ToolPlan",
    "ToolUseReport",
    "evaluate_tool_use",
    "load_jsonl",
    "summarise_annotations",
    "summarise_grounding",
]

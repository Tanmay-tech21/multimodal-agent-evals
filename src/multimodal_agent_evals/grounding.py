"""Claim-level evidence attribution for multimodal agent traces.

The validators in this module establish structural provenance: they check that a
claim points to a real artifact or tool result and that its locator is valid.
They do not infer whether an annotator's semantic judgement is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence

from .metrics import MetricValue
from .schema import AgentTrace, EventType, Modality, TraceValidationError


class EvidenceSourceType(str, Enum):
    ARTIFACT = "artifact"
    TOOL_RESULT = "tool_result"


class LocatorType(str, Enum):
    WHOLE = "whole"
    BOUNDING_BOX = "bounding_box"
    TIME_RANGE = "time_range"
    JSON_POINTER = "json_pointer"


class GroundingStatus(str, Enum):
    SUPPORTED = "supported"
    INFERRED = "inferred"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


def _require_keys(data: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = required - set(data)
    extra = set(data) - required
    if missing or extra:
        raise TraceValidationError(
            f"{context} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TraceValidationError(f"{name} must be a real number")
    return float(value)


def _validate_locator(locator: Mapping[str, Any]) -> None:
    try:
        locator_type = LocatorType(locator.get("type"))
    except (TypeError, ValueError) as exc:
        raise TraceValidationError("locator.type is invalid") from exc

    if locator_type is LocatorType.WHOLE:
        _require_keys(locator, {"type"}, "whole locator")
        return

    if locator_type is LocatorType.BOUNDING_BOX:
        _require_keys(
            locator, {"type", "x", "y", "width", "height"}, "bounding-box locator"
        )
        x = _real(locator["x"], "locator.x")
        y = _real(locator["y"], "locator.y")
        width = _real(locator["width"], "locator.width")
        height = _real(locator["height"], "locator.height")
        if not (0 <= x < 1 and 0 <= y < 1 and width > 0 and height > 0):
            raise TraceValidationError("bounding box must have a positive normalized extent")
        if x + width > 1 or y + height > 1:
            raise TraceValidationError("bounding box must remain inside normalized image bounds")
        return

    if locator_type is LocatorType.TIME_RANGE:
        _require_keys(locator, {"type", "start_seconds", "end_seconds"}, "time locator")
        start = _real(locator["start_seconds"], "locator.start_seconds")
        end = _real(locator["end_seconds"], "locator.end_seconds")
        if start < 0 or end <= start:
            raise TraceValidationError("time range must satisfy 0 <= start_seconds < end_seconds")
        return

    _require_keys(locator, {"type", "path"}, "JSON-pointer locator")
    path = locator["path"]
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise TraceValidationError("JSON pointer must be empty or begin with '/'")
    _pointer_tokens(path)


def _pointer_tokens(path: str) -> list[str]:
    if path == "":
        return []
    tokens: list[str] = []
    for raw in path.split("/")[1:]:
        decoded: list[str] = []
        index = 0
        while index < len(raw):
            if raw[index] != "~":
                decoded.append(raw[index])
                index += 1
                continue
            if index + 1 >= len(raw) or raw[index + 1] not in {"0", "1"}:
                raise TraceValidationError(f"invalid JSON pointer escape in {path!r}")
            decoded.append("~" if raw[index + 1] == "0" else "/")
            index += 2
        tokens.append("".join(decoded))
    return tokens


def resolve_json_pointer(document: Any, path: str) -> Any:
    """Resolve an RFC 6901 JSON pointer or raise a validation error."""
    current = document
    for token in _pointer_tokens(path):
        if isinstance(current, Mapping):
            if token not in current:
                raise TraceValidationError(f"JSON pointer {path!r} does not resolve")
            current = current[token]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise TraceValidationError(f"JSON pointer {path!r} has an invalid array index")
            position = int(token)
            if position >= len(current):
                raise TraceValidationError(f"JSON pointer {path!r} does not resolve")
            current = current[position]
        else:
            raise TraceValidationError(f"JSON pointer {path!r} traverses a scalar")
    return current


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id: str
    source_type: EvidenceSourceType
    source_id: str
    locator: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.source_id:
            raise TraceValidationError("evidence_id and source_id must be non-empty")
        if not isinstance(self.source_type, EvidenceSourceType):
            raise TraceValidationError("source_type must be an EvidenceSourceType")
        if not isinstance(self.locator, Mapping):
            raise TraceValidationError("locator must be an object")
        _validate_locator(self.locator)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceReference":
        _require_keys(data, {"evidence_id", "source_type", "source_id", "locator"}, "evidence")
        try:
            source_type = EvidenceSourceType(data["source_type"])
        except ValueError as exc:
            raise TraceValidationError(f"invalid evidence source type: {data['source_type']!r}") from exc
        return cls(
            evidence_id=str(data["evidence_id"]),
            source_type=source_type,
            source_id=str(data["source_id"]),
            locator=data["locator"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_type": self.source_type.value,
            "source_id": self.source_id,
            "locator": dict(self.locator),
        }

    def validate_against(self, trace: AgentTrace) -> None:
        locator_type = LocatorType(self.locator["type"])
        if self.source_type is EvidenceSourceType.ARTIFACT:
            artifact = next((item for item in trace.artifacts if item.artifact_id == self.source_id), None)
            if artifact is None:
                raise TraceValidationError(f"unknown artifact evidence source: {self.source_id}")
            if locator_type is LocatorType.BOUNDING_BOX and artifact.modality is not Modality.IMAGE:
                raise TraceValidationError("bounding-box evidence requires an image artifact")
            if locator_type is LocatorType.TIME_RANGE and artifact.modality not in {Modality.AUDIO, Modality.VIDEO}:
                raise TraceValidationError("time-range evidence requires an audio or video artifact")
            if locator_type is LocatorType.JSON_POINTER:
                raise TraceValidationError("JSON-pointer evidence requires a tool result")
            return

        event = next((item for item in trace.events if item.event_id == self.source_id), None)
        if event is None or event.event_type is not EventType.TOOL_RESULT:
            raise TraceValidationError(f"tool-result evidence source is invalid: {self.source_id}")
        if locator_type not in {LocatorType.WHOLE, LocatorType.JSON_POINTER}:
            raise TraceValidationError("tool-result evidence requires a whole or JSON-pointer locator")
        if locator_type is LocatorType.JSON_POINTER:
            if event.payload.get("status") != "ok":
                raise TraceValidationError("JSON-pointer evidence requires a successful tool result")
            resolve_json_pointer(event.payload.get("output"), str(self.locator["path"]))


@dataclass(frozen=True)
class GroundedClaim:
    claim_id: str
    trace_id: str
    event_id: str
    text: str
    status: GroundingStatus
    evidence_ids: tuple[str, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not all((self.claim_id, self.trace_id, self.event_id, self.text.strip(), self.rationale.strip())):
            raise TraceValidationError("claim identifiers, text, and rationale must be non-empty")
        if not isinstance(self.status, GroundingStatus):
            raise TraceValidationError("status must be a GroundingStatus")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise TraceValidationError(f"claim {self.claim_id} repeats an evidence_id")
        if self.status is GroundingStatus.UNSUPPORTED and self.evidence_ids:
            raise TraceValidationError("unsupported claims must not cite evidence")
        if self.status is not GroundingStatus.UNSUPPORTED and not self.evidence_ids:
            raise TraceValidationError(f"{self.status.value} claims must cite evidence")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundedClaim":
        required = {"claim_id", "trace_id", "event_id", "text", "status", "evidence_ids", "rationale"}
        _require_keys(data, required, "claim")
        if not isinstance(data["evidence_ids"], list) or not all(
            isinstance(item, str) for item in data["evidence_ids"]
        ):
            raise TraceValidationError("claim.evidence_ids must be an array of strings")
        try:
            status = GroundingStatus(data["status"])
        except ValueError as exc:
            raise TraceValidationError(f"invalid grounding status: {data['status']!r}") from exc
        return cls(
            claim_id=str(data["claim_id"]),
            trace_id=str(data["trace_id"]),
            event_id=str(data["event_id"]),
            text=str(data["text"]),
            status=status,
            evidence_ids=tuple(data["evidence_ids"]),
            rationale=str(data["rationale"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "text": self.text,
            "status": self.status.value,
            "evidence_ids": list(self.evidence_ids),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class GroundingRecord:
    trace_id: str
    evidence: tuple[EvidenceReference, ...]
    claims: tuple[GroundedClaim, ...]

    def __post_init__(self) -> None:
        if not self.trace_id or not self.claims:
            raise TraceValidationError("grounding record requires a trace_id and at least one claim")
        evidence_ids = [item.evidence_id for item in self.evidence]
        claim_ids = [item.claim_id for item in self.claims]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise TraceValidationError("evidence_id values must be unique")
        if len(set(claim_ids)) != len(claim_ids):
            raise TraceValidationError("claim_id values must be unique")
        known = set(evidence_ids)
        for claim in self.claims:
            if claim.trace_id != self.trace_id:
                raise TraceValidationError(f"claim {claim.claim_id} has a mismatched trace_id")
            unknown = set(claim.evidence_ids) - known
            if unknown:
                raise TraceValidationError(f"claim {claim.claim_id} cites unknown evidence: {sorted(unknown)}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundingRecord":
        _require_keys(data, {"trace_id", "evidence", "claims"}, "grounding record")
        if not isinstance(data["evidence"], list) or not isinstance(data["claims"], list):
            raise TraceValidationError("evidence and claims must be arrays")
        return cls(
            trace_id=str(data["trace_id"]),
            evidence=tuple(EvidenceReference.from_dict(item) for item in data["evidence"]),
            claims=tuple(GroundedClaim.from_dict(item) for item in data["claims"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
        }

    def validate_against(self, trace: AgentTrace) -> None:
        if trace.trace_id != self.trace_id:
            raise TraceValidationError("grounding record and trace_id do not match")
        for item in self.evidence:
            item.validate_against(trace)
        events = {event.event_id: event for event in trace.events}
        for claim in self.claims:
            event = events.get(claim.event_id)
            if event is None or event.event_type not in {EventType.ASSISTANT_MESSAGE, EventType.FINAL_ANSWER}:
                raise TraceValidationError(f"claim {claim.claim_id} must target an assistant response")
            if claim.text not in str(event.payload.get("text", "")):
                raise TraceValidationError(f"claim {claim.claim_id} text is absent from its event")


@dataclass(frozen=True)
class GroundingReport:
    total_claims: int
    evidence_references: int
    evidence_coverage: MetricValue
    supported_claim_rate: MetricValue
    inferred_claim_rate: MetricValue
    unsupported_claim_rate: MetricValue
    contradicted_claim_rate: MetricValue
    artifact_evidence_rate: MetricValue
    tool_result_evidence_rate: MetricValue

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "evidence_references": self.evidence_references,
            "evidence_coverage": self.evidence_coverage.to_dict(),
            "supported_claim_rate": self.supported_claim_rate.to_dict(),
            "inferred_claim_rate": self.inferred_claim_rate.to_dict(),
            "unsupported_claim_rate": self.unsupported_claim_rate.to_dict(),
            "contradicted_claim_rate": self.contradicted_claim_rate.to_dict(),
            "artifact_evidence_rate": self.artifact_evidence_rate.to_dict(),
            "tool_result_evidence_rate": self.tool_result_evidence_rate.to_dict(),
        }


def summarise_grounding(trace: AgentTrace, record: GroundingRecord) -> GroundingReport:
    """Validate one annotation record and summarize curator-assigned statuses."""

    record.validate_against(trace)
    total = len(record.claims)
    evidence_total = len(record.evidence)

    def claim_rate(status: GroundingStatus) -> MetricValue:
        return MetricValue(sum(claim.status is status for claim in record.claims), total)

    return GroundingReport(
        total_claims=total,
        evidence_references=evidence_total,
        evidence_coverage=MetricValue(sum(bool(claim.evidence_ids) for claim in record.claims), total),
        supported_claim_rate=claim_rate(GroundingStatus.SUPPORTED),
        inferred_claim_rate=claim_rate(GroundingStatus.INFERRED),
        unsupported_claim_rate=claim_rate(GroundingStatus.UNSUPPORTED),
        contradicted_claim_rate=claim_rate(GroundingStatus.CONTRADICTED),
        artifact_evidence_rate=MetricValue(
            sum(item.source_type is EvidenceSourceType.ARTIFACT for item in record.evidence),
            evidence_total,
        ),
        tool_result_evidence_rate=MetricValue(
            sum(item.source_type is EvidenceSourceType.TOOL_RESULT for item in record.evidence),
            evidence_total,
        ),
    )

"""Validated JSON-compatible traces for multimodal tool-using agents."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TraceValidationError(ValueError):
    """Raised when a trace cannot support trustworthy evaluation."""


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class EventType(str, Enum):
    USER_INPUT = "user_input"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class InputArtifact:
    artifact_id: str
    modality: Modality
    uri: str | None = None
    inline_text: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        _identifier("artifact_id", self.artifact_id)
        if (self.uri is None) == (self.inline_text is None):
            raise TraceValidationError(
                "an artifact must provide exactly one of uri or inline_text"
            )
        if self.uri is not None and not self.uri.strip():
            raise TraceValidationError("artifact uri must not be empty")
        if self.inline_text is not None:
            if self.modality is not Modality.TEXT:
                raise TraceValidationError("inline_text is valid only for text artifacts")
            if not self.inline_text.strip():
                raise TraceValidationError("inline_text must not be empty")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise TraceValidationError("sha256 must contain 64 lowercase hex characters")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InputArtifact":
        data = _mapping("artifact", value)
        allowed = {"artifact_id", "modality", "uri", "inline_text", "sha256"}
        unknown = set(data) - allowed
        if unknown:
            raise TraceValidationError(f"unknown artifact fields: {sorted(unknown)}")
        try:
            return cls(
                artifact_id=data["artifact_id"],
                modality=Modality(data["modality"]),
                uri=data.get("uri"),
                inline_text=data.get("inline_text"),
                sha256=data.get("sha256"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, TraceValidationError):
                raise
            raise TraceValidationError("invalid artifact") from exc

    def to_dict(self) -> dict[str, Any]:
        result = {"artifact_id": self.artifact_id, "modality": self.modality.value}
        for name in ("uri", "inline_text", "sha256"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True)
class TraceEvent:
    event_id: str
    index: int
    event_type: EventType
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier("event_id", self.event_id)
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise TraceValidationError("event index must be a non-negative integer")
        object.__setattr__(self, "payload", _mapping("payload", self.payload))
        self._validate_payload()

    def _validate_payload(self) -> None:
        payload = self.payload
        if self.event_type is EventType.USER_INPUT:
            if set(payload) != {"artifact_ids"}:
                raise TraceValidationError("user_input requires only artifact_ids")
            artifact_ids = payload["artifact_ids"]
            if not isinstance(artifact_ids, list) or not artifact_ids:
                raise TraceValidationError("artifact_ids must be a non-empty list")
            for artifact_id in artifact_ids:
                _identifier("artifact_id", artifact_id)
        elif self.event_type in {EventType.ASSISTANT_MESSAGE, EventType.FINAL_ANSWER}:
            if set(payload) != {"text"} or not isinstance(payload["text"], str) or not payload["text"].strip():
                raise TraceValidationError(f"{self.event_type.value} requires non-empty text")
        elif self.event_type is EventType.TOOL_CALL:
            if set(payload) != {"call_id", "tool_name", "arguments"}:
                raise TraceValidationError("tool_call requires call_id, tool_name and arguments")
            _identifier("call_id", payload["call_id"])
            _identifier("tool_name", payload["tool_name"])
            _mapping("arguments", payload["arguments"])
        elif self.event_type is EventType.TOOL_RESULT:
            required = {"call_id", "tool_name", "status"}
            if not required.issubset(payload) or set(payload) - (required | {"output", "error"}):
                raise TraceValidationError("invalid tool_result fields")
            _identifier("call_id", payload["call_id"])
            _identifier("tool_name", payload["tool_name"])
            if payload["status"] not in {"ok", "error"}:
                raise TraceValidationError("tool result status must be ok or error")
            if payload["status"] == "ok":
                if "output" not in payload or "error" in payload:
                    raise TraceValidationError("successful tool results require only output")
            elif "error" not in payload or "output" in payload:
                raise TraceValidationError("failed tool results require only error")
            if "error" in payload and (
                not isinstance(payload["error"], str) or not payload["error"].strip()
            ):
                raise TraceValidationError("tool error must be non-empty text")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TraceEvent":
        data = _mapping("event", value)
        if set(data) != {"event_id", "index", "event_type", "payload"}:
            raise TraceValidationError("event fields must be event_id, index, event_type and payload")
        try:
            return cls(
                event_id=data["event_id"],
                index=data["index"],
                event_type=EventType(data["event_type"]),
                payload=data["payload"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, TraceValidationError):
                raise
            raise TraceValidationError("invalid event") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "index": self.index,
            "event_type": self.event_type.value,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class AgentTrace:
    trace_id: str
    task_id: str
    artifacts: tuple[InputArtifact, ...]
    events: tuple[TraceEvent, ...]

    def __post_init__(self) -> None:
        _identifier("trace_id", self.trace_id)
        _identifier("task_id", self.task_id)
        if not self.artifacts:
            raise TraceValidationError("a trace must declare at least one artifact")
        if not self.events:
            raise TraceValidationError("a trace must contain events")
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        event_ids = [event.event_id for event in self.events]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise TraceValidationError("artifact identifiers must be unique")
        if len(set(event_ids)) != len(event_ids):
            raise TraceValidationError("event identifiers must be unique")
        if [event.index for event in self.events] != list(range(len(self.events))):
            raise TraceValidationError("event indices must be contiguous and ordered")
        if self.events[0].event_type is not EventType.USER_INPUT:
            raise TraceValidationError("the first event must be user_input")
        final_indices = [
            index
            for index, event in enumerate(self.events)
            if event.event_type is EventType.FINAL_ANSWER
        ]
        if final_indices != [len(self.events) - 1]:
            raise TraceValidationError("a complete trace requires one final answer at the end")

        known_artifacts = set(artifact_ids)
        calls: dict[str, str] = {}
        resolved: set[str] = set()
        for event in self.events:
            payload = event.payload
            if event.event_type is EventType.USER_INPUT:
                missing = set(payload["artifact_ids"]) - known_artifacts
                if missing:
                    raise TraceValidationError(f"unknown artifact references: {sorted(missing)}")
            elif event.event_type is EventType.TOOL_CALL:
                call_id = payload["call_id"]
                if call_id in calls:
                    raise TraceValidationError("tool call identifiers must be unique")
                calls[call_id] = payload["tool_name"]
            elif event.event_type is EventType.TOOL_RESULT:
                call_id = payload["call_id"]
                if call_id not in calls:
                    raise TraceValidationError("tool results must reference an earlier call")
                if call_id in resolved:
                    raise TraceValidationError("a tool call may have only one result")
                if calls[call_id] != payload["tool_name"]:
                    raise TraceValidationError("tool result name must match its call")
                resolved.add(call_id)
        unresolved = set(calls) - resolved
        if unresolved:
            raise TraceValidationError(f"unresolved tool calls: {sorted(unresolved)}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentTrace":
        data = _mapping("trace", value)
        if set(data) != {"trace_id", "task_id", "artifacts", "events"}:
            raise TraceValidationError("trace fields must be trace_id, task_id, artifacts and events")
        try:
            artifacts = tuple(InputArtifact.from_dict(item) for item in data["artifacts"])
            events = tuple(TraceEvent.from_dict(item) for item in data["events"])
            return cls(data["trace_id"], data["task_id"], artifacts, events)
        except (KeyError, TypeError) as exc:
            raise TraceValidationError("invalid trace") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "events": [event.to_dict() for event in self.events],
        }


def load_jsonl(path: str | Path) -> tuple[AgentTrace, ...]:
    """Load one validated trace per non-empty JSONL line."""
    traces = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            traces.append(AgentTrace.from_dict(json.loads(line)))
        except (json.JSONDecodeError, TraceValidationError) as exc:
            raise TraceValidationError(f"line {line_number}: {exc}") from exc
    if not traces:
        raise TraceValidationError("JSONL file contains no traces")
    trace_ids = [trace.trace_id for trace in traces]
    if len(set(trace_ids)) != len(trace_ids):
        raise TraceValidationError("trace identifiers must be unique within a file")
    return tuple(traces)

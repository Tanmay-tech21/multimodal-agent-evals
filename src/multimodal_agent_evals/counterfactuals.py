"""Deterministic counterfactual comparisons for recorded agent replays."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from .grounding import resolve_json_pointer
from .metrics import MetricValue
from .schema import AgentTrace, EventType, TraceValidationError


class InterventionTarget(str, Enum):
    ARTIFACT = "artifact"
    TOOL_RESULT = "tool_result"


class ExpectedEffect(str, Enum):
    CHANGE = "change"
    INVARIANT = "invariant"


def _require_fields(data: Mapping[str, Any], fields: set[str], context: str) -> None:
    if set(data) != fields:
        raise TraceValidationError(f"invalid {context} fields")


def _pointer_tokens(path: str) -> list[str]:
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise TraceValidationError("intervention path must be a JSON pointer")
    tokens: list[str] = []
    for raw in path.split("/")[1:] if path else []:
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


def _replace_pointer(document: Any, path: str, value: Any) -> Any:
    result = deepcopy(document)
    tokens = _pointer_tokens(path)
    if not tokens:
        return deepcopy(value)
    current = result
    for token in tokens[:-1]:
        if isinstance(current, Mapping):
            if token not in current:
                raise TraceValidationError(f"intervention path {path!r} does not resolve")
            current = current[token]
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if not token.isdigit() or int(token) >= len(current):
                raise TraceValidationError(f"intervention path {path!r} does not resolve")
            current = current[int(token)]
        else:
            raise TraceValidationError(f"intervention path {path!r} traverses a scalar")
    final = tokens[-1]
    if isinstance(current, dict):
        if final not in current:
            raise TraceValidationError(f"intervention path {path!r} does not resolve")
        current[final] = deepcopy(value)
    elif isinstance(current, list):
        if not final.isdigit() or int(final) >= len(current):
            raise TraceValidationError(f"intervention path {path!r} does not resolve")
        current[int(final)] = deepcopy(value)
    else:
        raise TraceValidationError(f"intervention path {path!r} traverses a scalar")
    return result


def _final_answer(trace: AgentTrace) -> str:
    return str(trace.events[-1].payload["text"])


@dataclass(frozen=True)
class CounterfactualCase:
    case_id: str
    baseline_trace_id: str
    variant_trace_id: str
    target_type: InterventionTarget
    target_id: str
    path: str
    before: Any
    after: Any
    expected_effect: ExpectedEffect
    rationale: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.case_id,
                self.baseline_trace_id,
                self.variant_trace_id,
                self.target_id,
                self.path,
                self.rationale,
            )
        ):
            raise TraceValidationError("counterfactual identifiers, path, and rationale are required")
        if not isinstance(self.target_type, InterventionTarget):
            raise TraceValidationError("target_type must be an InterventionTarget")
        if not isinstance(self.expected_effect, ExpectedEffect):
            raise TraceValidationError("expected_effect must be an ExpectedEffect")
        if self.before == self.after:
            raise TraceValidationError("counterfactual before and after values must differ")
        _pointer_tokens(self.path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CounterfactualCase":
        fields = {
            "case_id",
            "baseline_trace_id",
            "variant_trace_id",
            "target_type",
            "target_id",
            "path",
            "before",
            "after",
            "expected_effect",
            "rationale",
        }
        _require_fields(data, fields, "counterfactual case")
        try:
            return cls(
                case_id=str(data["case_id"]),
                baseline_trace_id=str(data["baseline_trace_id"]),
                variant_trace_id=str(data["variant_trace_id"]),
                target_type=InterventionTarget(data["target_type"]),
                target_id=str(data["target_id"]),
                path=str(data["path"]),
                before=data["before"],
                after=data["after"],
                expected_effect=ExpectedEffect(data["expected_effect"]),
                rationale=str(data["rationale"]),
            )
        except ValueError as exc:
            if isinstance(exc, TraceValidationError):
                raise
            raise TraceValidationError("invalid counterfactual enum value") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "baseline_trace_id": self.baseline_trace_id,
            "variant_trace_id": self.variant_trace_id,
            "target_type": self.target_type.value,
            "target_id": self.target_id,
            "path": self.path,
            "before": self.before,
            "after": self.after,
            "expected_effect": self.expected_effect.value,
            "rationale": self.rationale,
        }

    def validate_pair(self, baseline: AgentTrace, variant: AgentTrace) -> None:
        if baseline.trace_id != self.baseline_trace_id or variant.trace_id != self.variant_trace_id:
            raise TraceValidationError(f"case {self.case_id} trace identifiers do not match")
        if baseline.task_id != variant.task_id:
            raise TraceValidationError(f"case {self.case_id} task identifiers do not match")
        if self.target_type is InterventionTarget.ARTIFACT:
            self._validate_artifact_intervention(baseline, variant)
        else:
            self._validate_tool_intervention(baseline, variant)

    def _validate_artifact_intervention(self, baseline: AgentTrace, variant: AgentTrace) -> None:
        baseline_artifacts = {item.artifact_id: item.to_dict() for item in baseline.artifacts}
        variant_artifacts = {item.artifact_id: item.to_dict() for item in variant.artifacts}
        if set(baseline_artifacts) != set(variant_artifacts) or self.target_id not in baseline_artifacts:
            raise TraceValidationError(f"case {self.case_id} artifact target is invalid")
        if resolve_json_pointer(baseline_artifacts[self.target_id], self.path) != self.before:
            raise TraceValidationError(f"case {self.case_id} baseline value does not match manifest")
        expected = _replace_pointer(baseline_artifacts[self.target_id], self.path, self.after)
        expected_artifacts = dict(baseline_artifacts)
        expected_artifacts[self.target_id] = expected
        if variant_artifacts != expected_artifacts:
            raise TraceValidationError(f"case {self.case_id} changes more than its artifact target")
        if baseline.events[0] != variant.events[0]:
            raise TraceValidationError(f"case {self.case_id} changes the user-input event")

    def _validate_tool_intervention(self, baseline: AgentTrace, variant: AgentTrace) -> None:
        if baseline.artifacts != variant.artifacts:
            raise TraceValidationError(f"case {self.case_id} changes artifacts during tool replay")
        baseline_events = {item.event_id: item for item in baseline.events}
        variant_events = {item.event_id: item for item in variant.events}
        target = baseline_events.get(self.target_id)
        replay_target = variant_events.get(self.target_id)
        if (
            target is None
            or replay_target is None
            or target.event_type is not EventType.TOOL_RESULT
            or replay_target.event_type is not EventType.TOOL_RESULT
            or target.index != replay_target.index
        ):
            raise TraceValidationError(f"case {self.case_id} tool-result target is invalid")
        output_path = f"/output{self.path}" if self.path else "/output"
        if resolve_json_pointer(target.payload, output_path) != self.before:
            raise TraceValidationError(f"case {self.case_id} baseline value does not match manifest")
        expected_payload = _replace_pointer(target.payload, output_path, self.after)
        if dict(replay_target.payload) != expected_payload:
            raise TraceValidationError(f"case {self.case_id} changes more than its tool-result target")
        prefix = target.index
        if baseline.events[:prefix] != variant.events[:prefix]:
            raise TraceValidationError(f"case {self.case_id} changes the pre-intervention trajectory")


@dataclass(frozen=True)
class CounterfactualOutcome:
    case_id: str
    expected_effect: ExpectedEffect
    answer_changed: bool
    expectation_met: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "expected_effect": self.expected_effect.value,
            "answer_changed": self.answer_changed,
            "expectation_met": self.expectation_met,
        }


@dataclass(frozen=True)
class CounterfactualReport:
    total_cases: int
    answer_change_rate: MetricValue
    expectation_pass_rate: MetricValue
    sensitivity_rate: MetricValue
    invariance_rate: MetricValue
    outcomes: tuple[CounterfactualOutcome, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "answer_change_rate": self.answer_change_rate.to_dict(),
            "expectation_pass_rate": self.expectation_pass_rate.to_dict(),
            "sensitivity_rate": self.sensitivity_rate.to_dict(),
            "invariance_rate": self.invariance_rate.to_dict(),
            "outcomes": [item.to_dict() for item in self.outcomes],
        }


def evaluate_counterfactuals(
    traces: Sequence[AgentTrace], cases: Sequence[CounterfactualCase]
) -> CounterfactualReport:
    """Validate paired replays and evaluate declared answer-change expectations."""

    if not cases:
        raise TraceValidationError("counterfactual evaluation requires at least one case")
    trace_map = {trace.trace_id: trace for trace in traces}
    if len(trace_map) != len(traces):
        raise TraceValidationError("counterfactual trace identifiers must be unique")
    if len({case.case_id for case in cases}) != len(cases):
        raise TraceValidationError("counterfactual case identifiers must be unique")

    outcomes: list[CounterfactualOutcome] = []
    for case in cases:
        try:
            baseline = trace_map[case.baseline_trace_id]
            variant = trace_map[case.variant_trace_id]
        except KeyError as exc:
            raise TraceValidationError(f"case {case.case_id} references an unknown trace") from exc
        case.validate_pair(baseline, variant)
        changed = _final_answer(baseline) != _final_answer(variant)
        expected_changed = case.expected_effect is ExpectedEffect.CHANGE
        outcomes.append(
            CounterfactualOutcome(
                case_id=case.case_id,
                expected_effect=case.expected_effect,
                answer_changed=changed,
                expectation_met=changed is expected_changed,
            )
        )

    change_cases = [item for item in outcomes if item.expected_effect is ExpectedEffect.CHANGE]
    invariant_cases = [item for item in outcomes if item.expected_effect is ExpectedEffect.INVARIANT]
    return CounterfactualReport(
        total_cases=len(outcomes),
        answer_change_rate=MetricValue(sum(item.answer_changed for item in outcomes), len(outcomes)),
        expectation_pass_rate=MetricValue(sum(item.expectation_met for item in outcomes), len(outcomes)),
        sensitivity_rate=MetricValue(sum(item.answer_changed for item in change_cases), len(change_cases)),
        invariance_rate=MetricValue(
            sum(not item.answer_changed for item in invariant_cases), len(invariant_cases)
        ),
        outcomes=tuple(outcomes),
    )

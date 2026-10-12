"""Denominator-aware metrics for tool-use trajectories."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from .schema import AgentTrace, EventType, TraceValidationError


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceValidationError(f"{name} must be a stable identifier")
    return value


def _mapping(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TraceValidationError(f"{name} must be an object")
    return dict(value)


@dataclass(frozen=True)
class ExpectedToolCall:
    step_id: str
    tool_name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        _identifier("step_id", self.step_id)
        _identifier("tool_name", self.tool_name)
        object.__setattr__(self, "arguments", _mapping("arguments", self.arguments))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExpectedToolCall":
        data = _mapping("expected tool call", value)
        if set(data) != {"step_id", "tool_name", "arguments"}:
            raise TraceValidationError("invalid expected tool call fields")
        try:
            return cls(data["step_id"], data["tool_name"], data["arguments"])
        except (KeyError, TypeError) as exc:
            raise TraceValidationError("invalid expected tool call") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class ToolPlan:
    task_id: str
    steps: tuple[ExpectedToolCall, ...]

    def __post_init__(self) -> None:
        _identifier("task_id", self.task_id)
        if not self.steps:
            raise TraceValidationError("a tool plan must contain at least one step")
        step_ids = [step.step_id for step in self.steps]
        tool_names = [step.tool_name for step in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise TraceValidationError("tool-plan step identifiers must be unique")
        if len(set(tool_names)) != len(tool_names):
            raise TraceValidationError(
                "tool names must be unique so calls can be aligned unambiguously"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolPlan":
        data = _mapping("tool plan", value)
        if set(data) != {"task_id", "steps"}:
            raise TraceValidationError("invalid tool plan fields")
        if not isinstance(data["steps"], list):
            raise TraceValidationError("tool-plan steps must be a list")
        return cls(
            task_id=data["task_id"],
            steps=tuple(ExpectedToolCall.from_dict(step) for step in data["steps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class MetricValue:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        values = (self.numerator, self.denominator)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TraceValidationError("metric counts must be integers")
        if self.numerator < 0 or self.denominator < 0:
            raise TraceValidationError("metric counts must be non-negative")
        if self.numerator > self.denominator:
            raise TraceValidationError("metric numerator cannot exceed denominator")

    @property
    def value(self) -> float | None:
        return None if self.denominator == 0 else self.numerator / self.denominator

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
        }


@dataclass(frozen=True)
class ToolUseReport:
    trace_id: str
    task_id: str
    expected_calls: int
    actual_calls: int
    selection_precision: MetricValue
    selection_recall: MetricValue
    argument_accuracy: MetricValue
    order_consistency: MetricValue
    execution_success_rate: MetricValue
    plan_completion_rate: MetricValue
    exact_plan_success: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "task_id": self.task_id,
            "expected_calls": self.expected_calls,
            "actual_calls": self.actual_calls,
            "metrics": {
                "selection_precision": self.selection_precision.to_dict(),
                "selection_recall": self.selection_recall.to_dict(),
                "argument_accuracy": self.argument_accuracy.to_dict(),
                "order_consistency": self.order_consistency.to_dict(),
                "execution_success_rate": self.execution_success_rate.to_dict(),
                "plan_completion_rate": self.plan_completion_rate.to_dict(),
            },
            "exact_plan_success": self.exact_plan_success,
        }


def _concordant_pairs(expected_positions: list[int]) -> MetricValue:
    denominator = len(expected_positions) * (len(expected_positions) - 1) // 2
    numerator = sum(
        expected_positions[left] < expected_positions[right]
        for left in range(len(expected_positions))
        for right in range(left + 1, len(expected_positions))
    )
    return MetricValue(numerator, denominator)


def evaluate_tool_use(trace: AgentTrace, plan: ToolPlan) -> ToolUseReport:
    """Compare a validated trace with one declared, task-specific tool plan."""
    if trace.task_id != plan.task_id:
        raise TraceValidationError("tool plan task_id does not match the trace")

    calls = [event for event in trace.events if event.event_type is EventType.TOOL_CALL]
    results = {
        event.payload["call_id"]: event.payload["status"]
        for event in trace.events
        if event.event_type is EventType.TOOL_RESULT
    }
    expected_counts = Counter(step.tool_name for step in plan.steps)
    actual_counts = Counter(event.payload["tool_name"] for event in calls)
    selection_matches = sum((expected_counts & actual_counts).values())

    first_calls: dict[str, Any] = {}
    for event in calls:
        first_calls.setdefault(event.payload["tool_name"], event)

    argument_matches = 0
    completed_steps = 0
    for step in plan.steps:
        event = first_calls.get(step.tool_name)
        if event is None:
            continue
        arguments_match = dict(event.payload["arguments"]) == dict(step.arguments)
        argument_matches += arguments_match
        completed_steps += arguments_match and results[event.payload["call_id"]] == "ok"

    expected_index = {step.tool_name: index for index, step in enumerate(plan.steps)}
    seen: set[str] = set()
    observed_expected_positions = []
    for event in calls:
        tool_name = event.payload["tool_name"]
        if tool_name in expected_index and tool_name not in seen:
            seen.add(tool_name)
            observed_expected_positions.append(expected_index[tool_name])

    successful_calls = sum(
        results[event.payload["call_id"]] == "ok" for event in calls
    )
    exact_plan_success = len(calls) == len(plan.steps) and all(
        event.payload["tool_name"] == step.tool_name
        and dict(event.payload["arguments"]) == dict(step.arguments)
        and results[event.payload["call_id"]] == "ok"
        for event, step in zip(calls, plan.steps)
    )

    return ToolUseReport(
        trace_id=trace.trace_id,
        task_id=trace.task_id,
        expected_calls=len(plan.steps),
        actual_calls=len(calls),
        selection_precision=MetricValue(selection_matches, len(calls)),
        selection_recall=MetricValue(selection_matches, len(plan.steps)),
        argument_accuracy=MetricValue(argument_matches, selection_matches),
        order_consistency=_concordant_pairs(observed_expected_positions),
        execution_success_rate=MetricValue(successful_calls, len(calls)),
        plan_completion_rate=MetricValue(completed_steps, len(plan.steps)),
        exact_plan_success=exact_plan_success,
    )

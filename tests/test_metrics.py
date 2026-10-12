from copy import deepcopy

import pytest

from multimodal_agent_evals import (
    AgentTrace,
    MetricValue,
    ToolPlan,
    TraceValidationError,
    evaluate_tool_use,
)

from test_schema import valid_trace


def one_step_plan() -> ToolPlan:
    return ToolPlan.from_dict(
        {
            "task_id": "inspect-image",
            "steps": [
                {
                    "step_id": "inspect",
                    "tool_name": "inspect_image",
                    "arguments": {"artifact_id": "image-1"},
                }
            ],
        }
    )


def multi_step_trace() -> dict:
    raw = deepcopy(valid_trace())
    raw["task_id"] = "multi-step"
    raw["events"] = [
        raw["events"][0],
        {
            "event_id": "event-1",
            "index": 1,
            "event_type": "tool_call",
            "payload": {
                "call_id": "call-b",
                "tool_name": "tool_b",
                "arguments": {"value": 2},
            },
        },
        {
            "event_id": "event-2",
            "index": 2,
            "event_type": "tool_result",
            "payload": {
                "call_id": "call-b",
                "tool_name": "tool_b",
                "status": "ok",
                "output": {},
            },
        },
        {
            "event_id": "event-3",
            "index": 3,
            "event_type": "tool_call",
            "payload": {
                "call_id": "call-a",
                "tool_name": "tool_a",
                "arguments": {"value": "wrong"},
            },
        },
        {
            "event_id": "event-4",
            "index": 4,
            "event_type": "tool_result",
            "payload": {
                "call_id": "call-a",
                "tool_name": "tool_a",
                "status": "ok",
                "output": {},
            },
        },
        {
            "event_id": "event-5",
            "index": 5,
            "event_type": "tool_call",
            "payload": {
                "call_id": "call-extra",
                "tool_name": "extra_tool",
                "arguments": {},
            },
        },
        {
            "event_id": "event-6",
            "index": 6,
            "event_type": "tool_result",
            "payload": {
                "call_id": "call-extra",
                "tool_name": "extra_tool",
                "status": "ok",
                "output": {},
            },
        },
        {
            "event_id": "event-7",
            "index": 7,
            "event_type": "final_answer",
            "payload": {"text": "Done."},
        },
    ]
    return raw


def multi_step_plan() -> ToolPlan:
    return ToolPlan.from_dict(
        {
            "task_id": "multi-step",
            "steps": [
                {"step_id": "a", "tool_name": "tool_a", "arguments": {"value": 1}},
                {"step_id": "b", "tool_name": "tool_b", "arguments": {"value": 2}},
            ],
        }
    )


def test_perfect_single_step_plan_is_exact() -> None:
    report = evaluate_tool_use(AgentTrace.from_dict(valid_trace()), one_step_plan())
    assert report.selection_precision.value == 1.0
    assert report.selection_recall.value == 1.0
    assert report.argument_accuracy.value == 1.0
    assert report.order_consistency.value is None
    assert report.execution_success_rate.value == 1.0
    assert report.plan_completion_rate.value == 1.0
    assert report.exact_plan_success


def test_metrics_keep_selection_arguments_order_and_completion_separate() -> None:
    report = evaluate_tool_use(
        AgentTrace.from_dict(multi_step_trace()), multi_step_plan()
    )
    assert report.selection_precision == MetricValue(2, 3)
    assert report.selection_recall == MetricValue(2, 2)
    assert report.argument_accuracy == MetricValue(1, 2)
    assert report.order_consistency == MetricValue(0, 1)
    assert report.execution_success_rate == MetricValue(3, 3)
    assert report.plan_completion_rate == MetricValue(1, 2)
    assert not report.exact_plan_success


def test_no_calls_are_ineligible_for_precision_and_execution_success() -> None:
    raw = valid_trace()
    raw["events"] = [raw["events"][0], raw["events"][-1]]
    raw["events"][1]["index"] = 1
    report = evaluate_tool_use(AgentTrace.from_dict(raw), one_step_plan())
    assert report.selection_precision.value is None
    assert report.execution_success_rate.value is None
    assert report.selection_recall.value == 0.0
    assert report.plan_completion_rate.value == 0.0


def test_failed_result_reduces_success_and_completion_not_selection() -> None:
    raw = valid_trace()
    result = raw["events"][2]["payload"]
    result.pop("output")
    result.update(status="error", error="tool unavailable")
    report = evaluate_tool_use(AgentTrace.from_dict(raw), one_step_plan())
    assert report.selection_recall.value == 1.0
    assert report.argument_accuracy.value == 1.0
    assert report.execution_success_rate.value == 0.0
    assert report.plan_completion_rate.value == 0.0
    assert not report.exact_plan_success


def test_plan_must_match_trace_task() -> None:
    raw = valid_trace()
    raw["task_id"] = "different-task"
    with pytest.raises(TraceValidationError, match="task_id"):
        evaluate_tool_use(AgentTrace.from_dict(raw), one_step_plan())


def test_plan_rejects_duplicate_tool_names() -> None:
    raw = multi_step_plan().to_dict()
    raw["steps"][1]["tool_name"] = "tool_a"
    with pytest.raises(TraceValidationError, match="unique"):
        ToolPlan.from_dict(raw)


def test_plan_rejects_duplicate_step_ids() -> None:
    raw = multi_step_plan().to_dict()
    raw["steps"][1]["step_id"] = "a"
    with pytest.raises(TraceValidationError, match="step identifiers"):
        ToolPlan.from_dict(raw)


def test_plan_rejects_empty_steps() -> None:
    with pytest.raises(TraceValidationError, match="at least one"):
        ToolPlan.from_dict({"task_id": "task", "steps": []})


@pytest.mark.parametrize(
    "numerator, denominator, message",
    [(-1, 1, "non-negative"), (2, 1, "exceed"), (True, 1, "integers")],
)
def test_metric_value_rejects_invalid_counts(numerator, denominator, message) -> None:
    with pytest.raises(TraceValidationError, match=message):
        MetricValue(numerator, denominator)


def test_report_serialises_explicit_denominators() -> None:
    report = evaluate_tool_use(AgentTrace.from_dict(valid_trace()), one_step_plan())
    encoded = report.to_dict()
    assert encoded["metrics"]["selection_precision"] == {
        "numerator": 1,
        "denominator": 1,
        "value": 1.0,
    }
    assert encoded["metrics"]["order_consistency"]["value"] is None

import json

import pytest

from multimodal_agent_evals import AgentTrace, TraceValidationError, load_jsonl


def valid_trace() -> dict:
    return {
        "trace_id": "trace-1",
        "task_id": "inspect-image",
        "artifacts": [
            {"artifact_id": "image-1", "modality": "image", "uri": "image.png"}
        ],
        "events": [
            {
                "event_id": "event-0",
                "index": 0,
                "event_type": "user_input",
                "payload": {"artifact_ids": ["image-1"]},
            },
            {
                "event_id": "event-1",
                "index": 1,
                "event_type": "tool_call",
                "payload": {
                    "call_id": "call-1",
                    "tool_name": "inspect_image",
                    "arguments": {"artifact_id": "image-1"},
                },
            },
            {
                "event_id": "event-2",
                "index": 2,
                "event_type": "tool_result",
                "payload": {
                    "call_id": "call-1",
                    "tool_name": "inspect_image",
                    "status": "ok",
                    "output": {"label": "coin"},
                },
            },
            {
                "event_id": "event-3",
                "index": 3,
                "event_type": "final_answer",
                "payload": {"text": "The image contains a coin."},
            },
        ],
    }


def test_valid_trace_round_trips_without_losing_structure() -> None:
    raw = valid_trace()
    trace = AgentTrace.from_dict(raw)
    assert trace.to_dict() == raw


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda trace: trace["events"][1].update(index=4), "contiguous"),
        (
            lambda trace: trace["events"][0]["payload"].update(
                artifact_ids=["missing"]
            ),
            "unknown artifact",
        ),
        (
            lambda trace: trace["events"][2]["payload"].update(
                call_id="missing-call"
            ),
            "earlier call",
        ),
        (
            lambda trace: trace["events"][2]["payload"].update(
                tool_name="different_tool"
            ),
            "name must match",
        ),
        (
            lambda trace: (
                trace["events"].pop(2),
                trace["events"][2].update(index=2),
            ),
            "unresolved",
        ),
        (lambda trace: trace["events"].append(trace["events"][-1]), "unique"),
    ],
)
def test_trace_rejects_cross_event_inconsistency(mutation, message) -> None:
    raw = valid_trace()
    mutation(raw)
    with pytest.raises(TraceValidationError, match=message):
        AgentTrace.from_dict(raw)


def test_error_result_requires_error_text_and_no_output() -> None:
    raw = valid_trace()
    result = raw["events"][2]["payload"]
    result.update(status="error", error="OCR timed out")
    result.pop("output")
    assert AgentTrace.from_dict(raw).events[2].payload["status"] == "error"


def test_inline_text_is_restricted_to_text_modality() -> None:
    raw = valid_trace()
    raw["artifacts"][0] = {
        "artifact_id": "image-1",
        "modality": "image",
        "inline_text": "not pixels",
    }
    with pytest.raises(TraceValidationError, match="only for text"):
        AgentTrace.from_dict(raw)


def test_jsonl_loader_reports_line_number(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(valid_trace()) + "\nnot-json\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="line 2"):
        load_jsonl(path)


def test_jsonl_loader_rejects_duplicate_trace_ids(tmp_path) -> None:
    encoded = json.dumps(valid_trace())
    path = tmp_path / "traces.jsonl"
    path.write_text(encoded + "\n" + encoded + "\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="trace identifiers"):
        load_jsonl(path)

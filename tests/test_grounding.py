import copy
import json
from pathlib import Path

import pytest

from multimodal_agent_evals import (
    GroundingRecord,
    TraceValidationError,
    load_jsonl,
    summarise_grounding,
)


ROOT = Path(__file__).parents[1]


def fixture_data():
    trace = load_jsonl(ROOT / "examples" / "grounding_trace.jsonl")[0]
    data = json.loads((ROOT / "examples" / "grounding_annotations.json").read_text())
    return trace, data


def test_grounding_fixture_round_trips_and_reports_rates():
    trace, data = fixture_data()
    record = GroundingRecord.from_dict(data)

    assert record.to_dict() == data
    report = summarise_grounding(trace, record)
    assert report.total_claims == 2
    assert report.evidence_coverage.value == 1.0
    assert report.supported_claim_rate.value == 0.5
    assert report.contradicted_claim_rate.value == 0.5
    assert report.artifact_evidence_rate.value == pytest.approx(1 / 3)
    assert report.tool_result_evidence_rate.value == pytest.approx(2 / 3)


def test_bounding_box_must_remain_inside_image():
    trace, data = fixture_data()
    data["evidence"][2]["locator"]["width"] = 0.7
    with pytest.raises(TraceValidationError, match="inside normalized image bounds"):
        GroundingRecord.from_dict(data)


def test_bounding_box_requires_image_artifact():
    trace, data = fixture_data()
    data["evidence"][2]["source_id"] = "prompt"
    with pytest.raises(TraceValidationError, match="requires an image"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_time_range_requires_audio_or_video():
    trace, data = fixture_data()
    data["evidence"][2]["locator"] = {
        "type": "time_range",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
    }
    with pytest.raises(TraceValidationError, match="audio or video"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_unknown_artifact_is_rejected():
    trace, data = fixture_data()
    data["evidence"][2]["source_id"] = "missing"
    with pytest.raises(TraceValidationError, match="unknown artifact"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_json_pointer_must_resolve():
    trace, data = fixture_data()
    data["evidence"][0]["locator"]["path"] = "/missing"
    with pytest.raises(TraceValidationError, match="does not resolve"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_json_pointer_rejects_invalid_escape():
    _, data = fixture_data()
    data["evidence"][0]["locator"]["path"] = "/bad~2token"
    with pytest.raises(TraceValidationError, match="invalid JSON pointer escape"):
        GroundingRecord.from_dict(data)


def test_tool_evidence_must_reference_tool_result():
    trace, data = fixture_data()
    data["evidence"][0]["source_id"] = "event-1"
    with pytest.raises(TraceValidationError, match="tool-result evidence source is invalid"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_claim_must_reference_assistant_response():
    trace, data = fixture_data()
    data["claims"][0]["event_id"] = "event-2"
    with pytest.raises(TraceValidationError, match="assistant response"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_claim_text_must_be_present_in_event():
    trace, data = fixture_data()
    data["claims"][0]["text"] = "A claim absent from the response."
    with pytest.raises(TraceValidationError, match="text is absent"):
        GroundingRecord.from_dict(data).validate_against(trace)


def test_supported_claim_requires_evidence():
    _, data = fixture_data()
    data["claims"][0]["evidence_ids"] = []
    with pytest.raises(TraceValidationError, match="supported claims must cite evidence"):
        GroundingRecord.from_dict(data)


def test_unsupported_claim_must_not_cite_evidence():
    _, data = fixture_data()
    data["claims"][0]["status"] = "unsupported"
    with pytest.raises(TraceValidationError, match="must not cite evidence"):
        GroundingRecord.from_dict(data)


def test_claim_cannot_cite_unknown_evidence():
    _, data = fixture_data()
    data["claims"][0]["evidence_ids"] = ["missing"]
    with pytest.raises(TraceValidationError, match="unknown evidence"):
        GroundingRecord.from_dict(data)


def test_evidence_identifiers_must_be_unique():
    _, data = fixture_data()
    duplicate = copy.deepcopy(data["evidence"][0])
    data["evidence"].append(duplicate)
    with pytest.raises(TraceValidationError, match="must be unique"):
        GroundingRecord.from_dict(data)


def test_grounding_record_requires_claims():
    _, data = fixture_data()
    data["claims"] = []
    with pytest.raises(TraceValidationError, match="at least one claim"):
        GroundingRecord.from_dict(data)

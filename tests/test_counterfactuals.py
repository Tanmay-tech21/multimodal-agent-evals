from copy import deepcopy
import json
from pathlib import Path

import pytest

from multimodal_agent_evals import (
    AgentTrace,
    CounterfactualCase,
    TraceValidationError,
    evaluate_counterfactuals,
    load_jsonl,
)


ROOT = Path(__file__).parents[1]


def fixture_data():
    traces = load_jsonl(ROOT / "examples" / "counterfactual_traces.jsonl")
    document = json.loads((ROOT / "examples" / "counterfactual_manifest.json").read_text())
    return traces, document["cases"]


def test_counterfactual_fixture_reports_sensitivity_and_invariance():
    traces, raw_cases = fixture_data()
    cases = tuple(CounterfactualCase.from_dict(item) for item in raw_cases)
    report = evaluate_counterfactuals(traces, cases)

    assert [case.to_dict() for case in cases] == raw_cases
    assert report.total_cases == 2
    assert report.answer_change_rate.value == 0.5
    assert report.expectation_pass_rate.value == 1.0
    assert report.sensitivity_rate.value == 1.0
    assert report.invariance_rate.value == 1.0
    assert all(item.expectation_met for item in report.outcomes)


def test_failed_expectation_remains_visible_in_report():
    traces, raw_cases = fixture_data()
    raw_cases[0]["expected_effect"] = "invariant"
    report = evaluate_counterfactuals(
        traces, tuple(CounterfactualCase.from_dict(item) for item in raw_cases)
    )
    assert report.expectation_pass_rate.value == 0.5
    assert not report.outcomes[0].expectation_met


def test_manifest_before_value_must_match_baseline():
    traces, raw_cases = fixture_data()
    raw_cases[0]["before"] = "99.99"
    with pytest.raises(TraceValidationError, match="baseline value"):
        evaluate_counterfactuals(traces, (CounterfactualCase.from_dict(raw_cases[0]),))


def test_variant_may_only_change_declared_tool_result_field():
    traces, raw_cases = fixture_data()
    variant = traces[1].to_dict()
    variant["events"][2]["payload"]["output"]["currency"] = "EUR"
    altered = (traces[0], AgentTrace.from_dict(variant))
    with pytest.raises(TraceValidationError, match="more than its tool-result target"):
        evaluate_counterfactuals(altered, (CounterfactualCase.from_dict(raw_cases[0]),))


def test_pre_intervention_trajectory_must_be_identical():
    traces, raw_cases = fixture_data()
    variant = traces[1].to_dict()
    variant["events"][1]["payload"]["arguments"]["artifact_id"] = "prompt"
    altered = (traces[0], AgentTrace.from_dict(variant))
    with pytest.raises(TraceValidationError, match="pre-intervention trajectory"):
        evaluate_counterfactuals(altered, (CounterfactualCase.from_dict(raw_cases[0]),))


def test_tool_replay_cannot_change_artifacts():
    traces, raw_cases = fixture_data()
    variant = traces[1].to_dict()
    variant["artifacts"][1]["uri"] = "fixture://different.png"
    altered = (traces[0], AgentTrace.from_dict(variant))
    with pytest.raises(TraceValidationError, match="changes artifacts"):
        evaluate_counterfactuals(altered, (CounterfactualCase.from_dict(raw_cases[0]),))


def test_case_rejects_identical_before_and_after_values():
    _, raw_cases = fixture_data()
    raw_cases[0]["after"] = raw_cases[0]["before"]
    with pytest.raises(TraceValidationError, match="must differ"):
        CounterfactualCase.from_dict(raw_cases[0])


def test_unknown_trace_is_rejected():
    traces, raw_cases = fixture_data()
    raw_cases[0]["variant_trace_id"] = "missing"
    with pytest.raises(TraceValidationError, match="unknown trace"):
        evaluate_counterfactuals(traces, (CounterfactualCase.from_dict(raw_cases[0]),))


def test_duplicate_case_identifiers_are_rejected():
    traces, raw_cases = fixture_data()
    raw_cases[1]["case_id"] = raw_cases[0]["case_id"]
    cases = tuple(CounterfactualCase.from_dict(item) for item in raw_cases)
    with pytest.raises(TraceValidationError, match="case identifiers"):
        evaluate_counterfactuals(traces, cases)


def test_empty_suite_is_rejected():
    traces, _ = fixture_data()
    with pytest.raises(TraceValidationError, match="at least one case"):
        evaluate_counterfactuals(traces, ())


def test_artifact_intervention_can_change_only_declared_field():
    traces, _ = fixture_data()
    baseline = traces[0]
    variant_data = deepcopy(baseline.to_dict())
    variant_data["trace_id"] = "artifact-variant"
    variant_data["artifacts"][1]["uri"] = "fixture://masked-receipt.png"
    variant_data["events"][-1]["payload"]["text"] = "The receipt total is unreadable."
    variant = AgentTrace.from_dict(variant_data)
    case = CounterfactualCase.from_dict(
        {
            "case_id": "mask-receipt",
            "baseline_trace_id": baseline.trace_id,
            "variant_trace_id": variant.trace_id,
            "target_type": "artifact",
            "target_id": "receipt-image",
            "path": "/uri",
            "before": "fixture://receipt.png",
            "after": "fixture://masked-receipt.png",
            "expected_effect": "change",
            "rationale": "Masking the receipt should change the total response.",
        }
    )
    report = evaluate_counterfactuals((baseline, variant), (case,))
    assert report.expectation_pass_rate.value == 1.0
    assert report.invariance_rate.value is None

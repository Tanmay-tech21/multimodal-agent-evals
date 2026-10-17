"""Build matching Markdown and JSON evidence from the synthetic evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

from multimodal_agent_evals import (
    CounterfactualCase,
    EvidenceClaim,
    EvidenceSection,
    EvaluationSnapshot,
    EventType,
    GroundingRecord,
    RatingStudy,
    SubgroupManifest,
    ToolPlan,
    evaluate_agreement,
    evaluate_counterfactuals,
    evaluate_subgroups,
    evaluate_tool_use,
    load_jsonl,
    sha256_file,
    summarise_grounding,
)


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
ARTIFACTS = ROOT / "artifacts"


def _read_json(name: str):
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))


def _fingerprints(*names: str) -> dict[str, str]:
    return {f"examples/{name}": sha256_file(EXAMPLES / name) for name in names}


def build_snapshot() -> EvaluationSnapshot:
    trace = load_jsonl(EXAMPLES / "receipt_trace.jsonl")[0]
    trace_result = {
        "valid": True,
        "trace_count": 1,
        "artifact_count": len(trace.artifacts),
        "event_count": len(trace.events),
        "tool_call_count": sum(
            event.event_type is EventType.TOOL_CALL for event in trace.events
        ),
    }

    tool_trace = load_jsonl(EXAMPLES / "tool_use_trace.jsonl")[0]
    tool_result = evaluate_tool_use(
        tool_trace, ToolPlan.from_dict(_read_json("tool_use_plan.json"))
    ).to_dict()

    grounding_trace = load_jsonl(EXAMPLES / "grounding_trace.jsonl")[0]
    grounding_result = summarise_grounding(
        grounding_trace,
        GroundingRecord.from_dict(_read_json("grounding_annotations.json")),
    ).to_dict()

    counterfactual_document = _read_json("counterfactual_manifest.json")
    counterfactual_result = evaluate_counterfactuals(
        load_jsonl(EXAMPLES / "counterfactual_traces.jsonl"),
        tuple(CounterfactualCase.from_dict(item) for item in counterfactual_document["cases"]),
    ).to_dict()

    ratings = RatingStudy.from_dict(_read_json("evaluator_ratings.json"))
    agreement_result = evaluate_agreement(ratings).to_dict()
    subgroup_result = evaluate_subgroups(
        ratings,
        SubgroupManifest.from_dict(_read_json("evaluator_subgroups.json")),
        "modality",
        resamples=1000,
        confidence_level=0.95,
        seed=2026,
        minimum_items=10,
    ).to_dict()

    sections = (
        EvidenceSection(
            "trace_validation",
            "Cross-event validation of one multimodal tool-use trace.",
            _fingerprints("receipt_trace.jsonl"),
            trace_result,
            ("Synthetic fixture; no live agent was evaluated.",),
        ),
        EvidenceSection(
            "tool_use",
            "Denominator-aware comparison against a declared tool plan.",
            _fingerprints("tool_use_trace.jsonl", "tool_use_plan.json"),
            tool_result,
            ("Exact argument equality is stricter than semantic equivalence.",),
        ),
        EvidenceSection(
            "grounding",
            "Claim-level structural provenance and curator-assigned evidence status.",
            _fingerprints("grounding_trace.jsonl", "grounding_annotations.json"),
            grounding_result,
            ("Structural validation does not establish semantic truth.",),
        ),
        EvidenceSection(
            "counterfactual",
            "Isolated evidence interventions with declared response expectations.",
            _fingerprints("counterfactual_traces.jsonl", "counterfactual_manifest.json"),
            counterfactual_result,
            ("Hand-constructed replays do not measure live-model causal grounding.",),
        ),
        EvidenceSection(
            "agreement",
            "Pairwise and multi-rater reliability for synthetic evaluator labels.",
            _fingerprints("evaluator_ratings.json"),
            agreement_result,
            ("Eight items are insufficient for a production reliability estimate.",),
        ),
        EvidenceSection(
            "subgroups",
            "Modality-stratified agreement with item-bootstrap uncertainty.",
            _fingerprints("evaluator_ratings.json", "evaluator_subgroups.json"),
            subgroup_result,
            ("Four items per modality produce highly uncertain intervals.",),
        ),
    )
    claims = (
        EvidenceClaim(
            "tool_execution_not_plan_success",
            "Successful tool execution did not imply exact plan success.",
            "/sections/tool_use/result/exact_plan_success",
            False,
        ),
        EvidenceClaim(
            "citations_can_contradict",
            "Complete citation coverage coexisted with a contradicted claim.",
            "/sections/grounding/result/contradicted_claim_rate/numerator",
            1,
        ),
        EvidenceClaim(
            "counterfactual_expectations_met",
            "Both declared counterfactual expectations were satisfied.",
            "/sections/counterfactual/result/expectation_pass_rate/value",
            1.0,
        ),
        EvidenceClaim(
            "subgroups_are_limited",
            "The image subgroup is explicitly flagged as sample-limited.",
            "/sections/subgroups/result/results/0/warnings/0",
            "limited_items",
        ),
    )
    return EvaluationSnapshot(
        "multimodal-agent-evals",
        "1.0.0",
        {
            "bootstrap_resamples": 1000,
            "bootstrap_seed": 2026,
            "confidence_level": 0.95,
            "subgroup_dimension": "modality",
            "minimum_subgroup_items": 10,
        },
        sections,
        claims,
    )


def main() -> None:
    snapshot = build_snapshot()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "evaluation_snapshot.json").write_text(
        json.dumps(snapshot.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "EVALUATION_REPORT.md").write_text(
        snapshot.render_markdown(), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

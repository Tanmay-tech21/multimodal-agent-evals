# Multimodal Agent Evals

A nine-day evaluation laboratory for multimodal, tool-using agents. The project
begins with auditable traces before adding failure taxonomies, tool-use metrics,
grounding checks, perturbation tests, evaluator agreement, subgroup analysis and
reproducible reporting.

## Day 1: make agent behaviour inspectable

An agent evaluation should preserve more than its final answer. This first layer
defines a strict trace containing:

- declared input artifacts and their modalities;
- ordered user, assistant and tool events;
- stable tool-call identifiers and structured arguments;
- matching tool results with explicit success or error status;
- artifact references used during the trajectory; and
- exactly one final answer at the end of a complete trace.

Validation checks relationships across the entire trajectory. Event indices must
be contiguous, identifiers must be unique, artifact references must resolve,
tool results must match earlier calls, and every call must terminate exactly once.
These invariants prevent an evaluator from scoring a malformed log as though it
were valid agent behaviour.

The schema stores artifact references rather than media bytes. A SHA-256 digest
may be supplied when the referenced asset must be pinned, but the validator does
not fetch files or claim that an omitted digest has been verified.

## Reproduce the validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/validate_traces.py examples/receipt_trace.jsonl
python -m pytest -q
```

The example is synthetic and validates only the trace machinery. It is not a
benchmark result, model-quality claim, or record of a deployed agent.

## Trace contract

Each JSONL line contains one complete trace. Event payloads depend on event type:

| Event | Required payload |
|---|---|
| `user_input` | Non-empty `artifact_ids` |
| `assistant_message` | Non-empty `text` |
| `tool_call` | `call_id`, `tool_name`, object-valued `arguments` |
| `tool_result` | Matching `call_id`, `tool_name`, `status`, plus `output` or `error` |
| `final_answer` | Non-empty `text` |

## Planned progression

1. Multimodal trace schema and cross-event validation (complete)
2. Failure taxonomy with trace-level annotations
3. Tool selection, argument, ordering and completion metrics
4. Multimodal grounding and evidence-attribution checks
5. Deterministic replay and counterfactual perturbations
6. Evaluator agreement and judge calibration
7. Subgroup analysis and uncertainty intervals
8. Reproducible evaluation report and machine-readable evidence
9. Cross-project portfolio integration and final audit

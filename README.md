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

## Day 2: locate and adjudicate failures

Failure annotations now identify *where* a trajectory failed and *which stage*
best describes the failure. The taxonomy contains stable codes beneath eight
stages: perception, grounding, planning, tool selection, tool arguments, tool
execution, verification and completion.

Each annotation records an annotator, severity, rationale and an ordered set of
event identifiers. Independent annotation sets remain separate. An adjudication
record cites the source annotations and preserves the final decision without
erasing the original disagreement.

```bash
python scripts/summarise_failures.py \
  examples/failed_receipt_trace.jsonl \
  examples/failed_receipt_annotations.json
```

The failed-receipt fixture is synthetic. Its two reviewers intentionally assign
different labels to demonstrate disagreement and adjudication; their labels are
not measurements of human agreement or model quality.

## Day 3: measure tool use without collapsing failure modes

A declared `ToolPlan` now provides the expected tool names, order and exact
arguments for one task. Evaluation reports selection precision and recall,
argument accuracy among name-matched calls, pairwise order consistency,
execution success and plan completion as separate quantities.

Every metric preserves its numerator and denominator. A zero denominator is
reported as `null`, not silently converted into zero. Tool names in a plan must
currently be unique so repeated tools cannot be aligned ambiguously.

```bash
python scripts/evaluate_tool_use.py \
  examples/tool_use_trace.jsonl \
  examples/tool_use_plan.json
```

The synthetic fixture intentionally calls the two expected tools in reverse
order, supplies one incorrect argument object and adds an unnecessary tool. It
tests metric separation; it is not a benchmark of an agent or model.

## Day 4: ground claims in multimodal evidence

Final-answer claims can now cite immutable evidence references rather than a
free-form explanation alone. References target an input artifact or a successful
tool result and use a modality-aware locator:

- normalized bounding boxes for image regions;
- time ranges for audio and video;
- RFC 6901-style JSON pointers for structured tool output; or
- the whole source when a finer locator would be misleading.

The evaluator validates source existence, locator compatibility, JSON-pointer
resolution, claim-to-event linkage and evidence-ID integrity. Curators assign one
of four explicit statuses: `supported`, `inferred`, `unsupported` or
`contradicted`. These labels remain human judgements; structural validation does
not claim to prove semantic truth.

```bash
python scripts/evaluate_grounding.py \
  examples/grounding_trace.jsonl \
  examples/grounding_annotations.json
```

The synthetic fixture deliberately contains one supported merchant claim and one
contradicted total claim. Its output validates metric accounting and provenance,
not OCR quality or model performance.

## Day 5: test causal sensitivity with counterfactual replay

A counterfactual manifest now declares one isolated intervention between a
baseline trace and a recorded replay. An intervention targets either an artifact
field or an exact JSON-pointer location inside a tool result. Validation rejects
undeclared changes to artifacts, pre-intervention events or the targeted result.

Each case declares whether the final answer should change or remain invariant.
The report keeps overall answer-change rate separate from:

- sensitivity when task-relevant evidence changes;
- invariance when irrelevant evidence changes; and
- the proportion of declared expectations that the replay satisfies.

```bash
python scripts/evaluate_counterfactuals.py \
  examples/counterfactual_traces.jsonl \
  examples/counterfactual_manifest.json
```

The synthetic suite changes a receipt total in one replay and only the merchant
name in another. It validates isolated-intervention accounting; it does not show
that a live model is causally grounded.

## Day 6: calibrate the evaluators

An evaluation is only as credible as its labels. `RatingStudy` declares the
complete item, rater and label universes so omitted judgements remain visible
rather than being silently treated as agreement or disagreement. The report
includes:

- per-rater coverage and overall study completeness;
- raw pairwise agreement and directional confusion matrices;
- Cohen's kappa for each rater pair;
- nominal Krippendorff's alpha across all raters with missing ratings; and
- the rate of unanimous decisions among items rated at least twice.

Chance-corrected agreement is returned as `null` when the relevant marginals are
degenerate. This is preferable to reporting a reassuring but undefined number.
The confusion matrices retain rater direction so systematic judge-model label
substitutions can be inspected before adjudication.

```bash
python scripts/evaluate_agreement.py examples/evaluator_ratings.json
```

The example is a synthetic calibration exercise containing two reviewers and
one model-based judge. Its agreement values validate the implementation; they
are not measurements of a deployed evaluator or evidence that one judge is
better than another.

## Day 7: expose subgroup uncertainty

Aggregate reliability can conceal evaluator failures concentrated in one
modality or task family. A strict subgroup manifest now assigns every study item
to each declared dimension. The evaluator filters ratings without imputing
missing judgements, reports an agreement study for each group, and calculates a
percentile interval for nominal Krippendorff's alpha by resampling complete items.

Item-level resampling preserves the dependence among reviewers rating the same
case. Fixed seeds make the analysis reproducible. Reports retain the number of
valid bootstrap replicates and flag groups with fewer than the configured
minimum number of items or many undefined replicates.

```bash
python scripts/evaluate_subgroups.py \
  examples/evaluator_ratings.json \
  examples/evaluator_subgroups.json \
  modality --resamples 1000 --seed 2026
```

The example contains only four items per modality and is intentionally marked
`limited_items`. Its intervals demonstrate uncertainty propagation; they do not
support claims about modality-level judge performance.

## Day 8: bind every report claim to evidence

The full synthetic suite now executes once and produces matching Markdown and
JSON artifacts. Each evidence section preserves its complete metric result,
limitations and SHA-256 fingerprints for every input fixture. Bootstrap settings
are stored at snapshot level.

Narrative conclusions are represented as `EvidenceClaim` records. Each claim
contains a JSON Pointer and the expected value at that path. Snapshot validation
fails if a metric moves, disappears or no longer supports the written statement,
preventing a stale report from surviving a changed experiment.

```bash
python scripts/build_evaluation_report.py
python -m pytest -q
```

Generated artifacts are committed under `artifacts/`. They summarise synthetic
pipeline tests only and contain explicit limitations against model-performance,
human-reliability or deployment claims.

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
2. Failure taxonomy with trace-level annotations (complete)
3. Tool selection, argument, ordering and completion metrics (complete)
4. Multimodal grounding and evidence-attribution checks (complete)
5. Deterministic replay and counterfactual perturbations (complete)
6. Evaluator agreement and judge calibration (complete)
7. Subgroup analysis and uncertainty intervals (complete)
8. Reproducible evaluation report and machine-readable evidence (complete)
9. Cross-project portfolio integration and final audit

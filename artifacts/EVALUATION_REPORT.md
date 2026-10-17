# Multimodal Agent Evaluation Report

This report and its JSON snapshot were generated from the same validated
in-memory evidence. All included examples are synthetic pipeline fixtures.
They are not measurements of a deployed model or human review programme.

## Configuration

```json
{
  "bootstrap_resamples": 1000,
  "bootstrap_seed": 2026,
  "confidence_level": 0.95,
  "minimum_subgroup_items": 10,
  "subgroup_dimension": "modality"
}
```

## Evidence sections

| Section | Inputs | Limitations |
|---|---:|---|
| `trace_validation` | 1 | Synthetic fixture; no live agent was evaluated. |
| `tool_use` | 2 | Exact argument equality is stricter than semantic equivalence. |
| `grounding` | 2 | Structural validation does not establish semantic truth. |
| `counterfactual` | 2 | Hand-constructed replays do not measure live-model causal grounding. |
| `agreement` | 1 | Eight items are insufficient for a production reliability estimate. |
| `subgroups` | 2 | Four items per modality produce highly uncertain intervals. |

## Evidence-linked claims

### Successful tool execution did not imply exact plan success.

- Evidence: `/sections/tool_use/result/exact_plan_success`
- Observed value: `false`

### Complete citation coverage coexisted with a contradicted claim.

- Evidence: `/sections/grounding/result/contradicted_claim_rate/numerator`
- Observed value: `1`

### Both declared counterfactual expectations were satisfied.

- Evidence: `/sections/counterfactual/result/expectation_pass_rate/value`
- Observed value: `1.0`

### The image subgroup is explicitly flagged as sample-limited.

- Evidence: `/sections/subgroups/result/results/0/warnings/0`
- Observed value: `"limited_items"`

## Machine-readable evidence

See `evaluation_snapshot.json` for source fingerprints, complete metrics,
denominators, bootstrap configuration, warnings and claim bindings.

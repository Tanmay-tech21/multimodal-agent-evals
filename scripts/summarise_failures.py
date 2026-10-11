"""Validate trace annotations and print a descriptive failure summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from multimodal_agent_evals import (
    AdjudicationDecision,
    AnnotationSet,
    TraceValidationError,
    load_jsonl,
    summarise_annotations,
)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"invalid annotation JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise TraceValidationError("annotation file must contain one object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("annotations", type=Path)
    args = parser.parse_args()

    traces = {trace.trace_id: trace for trace in load_jsonl(args.traces)}
    document = _load_object(args.annotations)
    if set(document) != {"trace_id", "annotation_sets", "adjudications"}:
        raise TraceValidationError("invalid annotation document fields")
    trace_id = document["trace_id"]
    if trace_id not in traces:
        raise TraceValidationError("annotation document references an unknown trace")
    if not isinstance(document["annotation_sets"], list):
        raise TraceValidationError("annotation_sets must be a list")
    if not isinstance(document["adjudications"], list):
        raise TraceValidationError("adjudications must be a list")

    annotation_sets = tuple(
        AnnotationSet.from_dict(value) for value in document["annotation_sets"]
    )
    decisions = tuple(
        AdjudicationDecision.from_dict(value) for value in document["adjudications"]
    )
    summary = summarise_annotations(traces[trace_id], annotation_sets, decisions)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

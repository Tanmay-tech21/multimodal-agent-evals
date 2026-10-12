"""Evaluate a trace against a declared tool plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_agent_evals import ToolPlan, TraceValidationError, evaluate_tool_use, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    if len(traces) != 1:
        raise TraceValidationError("tool-use evaluation expects exactly one trace")
    try:
        plan_document = json.loads(args.plan.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"invalid tool-plan JSON: {exc}") from exc
    plan = ToolPlan.from_dict(plan_document)
    report = evaluate_tool_use(traces[0], plan)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

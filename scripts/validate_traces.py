"""Validate multimodal agent traces without executing their referenced tools."""

from __future__ import annotations

import argparse
import json

from multimodal_agent_evals import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="JSONL file containing complete agent traces")
    args = parser.parse_args()
    traces = load_jsonl(args.path)
    print(
        json.dumps(
            {
                "status": "valid",
                "traces": len(traces),
                "artifacts": sum(len(trace.artifacts) for trace in traces),
                "events": sum(len(trace.events) for trace in traces),
                "tool_calls": sum(
                    event.event_type.value == "tool_call"
                    for trace in traces
                    for event in trace.events
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

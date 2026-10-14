"""Validate deterministic counterfactual replays and report sensitivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_agent_evals import CounterfactualCase, evaluate_counterfactuals, load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    traces = load_jsonl(args.traces)
    document = json.loads(args.manifest.read_text(encoding="utf-8"))
    if set(document) != {"cases"} or not isinstance(document["cases"], list):
        raise ValueError("manifest must contain exactly one cases array")
    cases = tuple(CounterfactualCase.from_dict(item) for item in document["cases"])
    print(json.dumps(evaluate_counterfactuals(traces, cases).to_dict(), indent=2))


if __name__ == "__main__":
    main()

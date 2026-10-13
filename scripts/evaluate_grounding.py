"""Validate and summarize claim-level grounding annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_agent_evals import GroundingRecord, load_jsonl, summarise_grounding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="JSONL file containing exactly one trace")
    parser.add_argument("annotations", type=Path, help="JSON grounding annotation file")
    args = parser.parse_args()

    traces = load_jsonl(args.trace)
    if len(traces) != 1:
        raise ValueError("grounding evaluation requires exactly one trace")
    record = GroundingRecord.from_dict(json.loads(args.annotations.read_text()))
    print(json.dumps(summarise_grounding(traces[0], record).to_dict(), indent=2))


if __name__ == "__main__":
    main()

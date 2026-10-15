"""Evaluate agreement for the example human-and-model rating study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_agent_evals import RatingStudy, evaluate_agreement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study",
        nargs="?",
        type=Path,
        default=Path("examples/evaluator_ratings.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    study = RatingStudy.from_dict(json.loads(args.study.read_text()))
    rendered = json.dumps(evaluate_agreement(study).to_dict(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

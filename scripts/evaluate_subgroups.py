"""Report evaluator agreement by subgroup with bootstrap uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multimodal_agent_evals import RatingStudy, SubgroupManifest, evaluate_subgroups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dimension")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-items", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    study = RatingStudy.from_dict(json.loads(args.study.read_text()))
    manifest = SubgroupManifest.from_dict(json.loads(args.manifest.read_text()))
    report = evaluate_subgroups(
        study,
        manifest,
        args.dimension,
        resamples=args.resamples,
        confidence_level=args.confidence,
        seed=args.seed,
        minimum_items=args.minimum_items,
    )
    rendered = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()

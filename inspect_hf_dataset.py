#!/usr/bin/env python3
"""Inspect a Hugging Face dataset for evaluation planning."""
from __future__ import annotations

import argparse
import json

from eval_agent_core import inspect_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subset", default="")
    parser.add_argument("--split", default="auto")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--question-column", default="")
    parser.add_argument("--answer-column", default="")
    parser.add_argument("--choices-column", default="")
    parser.add_argument("--image-column", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_dataset(
        dataset=args.dataset,
        subset=args.subset or None,
        split=args.split,
        sample_size=args.sample_size,
        question_column=args.question_column,
        answer_column=args.answer_column,
        choices_column=args.choices_column,
        image_column=args.image_column,
    )
    print(json.dumps(inspection.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

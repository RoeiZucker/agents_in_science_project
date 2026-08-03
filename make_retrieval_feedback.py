#!/usr/bin/env python3
"""Create retrieval-agent feedback from refinement error analysis.

Examples:
  python make_retrieval_feedback.py \
    --error-analysis runtime/eval_results/_agent_runs/google_boolq_validation/error_analysis.json

  python make_retrieval_feedback.py \
    --error-analysis runtime/eval_results/_agent_runs/google_boolq_validation/error_analysis.json \
    --next-round 2 \
    --output-json runtime/eval_results/_agent_runs/google_boolq_validation/retrieval_feedback.json \
    --output-prompt runtime/eval_results/_agent_runs/google_boolq_validation/retrieval_feedback_prompt.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from refinement_agent_core import feedback_prompt, make_retrieval_feedback, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--error-analysis", required=True, type=Path)
    parser.add_argument("--next-round", type=int, default=2)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-prompt", type=Path, default=None)
    return parser.parse_args()


def default_json_path(error_analysis: Path) -> Path:
    return error_analysis.parent / "retrieval_feedback.json"


def default_prompt_path(error_analysis: Path) -> Path:
    return error_analysis.parent / "retrieval_feedback_prompt.txt"


def main() -> None:
    args = parse_args()
    analysis = read_json(args.error_analysis)
    feedback = make_retrieval_feedback(analysis, args.next_round)
    output_json = args.output_json or default_json_path(args.error_analysis)
    output_prompt = args.output_prompt or default_prompt_path(args.error_analysis)
    write_json(output_json, feedback)
    output_prompt.write_text(feedback_prompt(feedback), encoding="utf-8")
    print(output_json)
    print(output_prompt)


if __name__ == "__main__":
    main()

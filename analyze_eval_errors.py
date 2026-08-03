#!/usr/bin/env python3
"""Analyze an evaluation-agent run and summarize model failure modes.

Examples:
  python analyze_eval_errors.py \
    --run-dir runtime/eval_results/_agent_runs/google_boolq_validation

  python analyze_eval_errors.py \
    --run-dir runtime/eval_results/_agent_runs/google_boolq_validation \
    --candidate-models mock_candidates/boolq_round1.json \
    --output runtime/eval_results/_agent_runs/google_boolq_validation/error_analysis.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from refinement_agent_core import analyze_run, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--candidate-models", type=Path, default=None)
    parser.add_argument("--max-failed-examples", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def default_output(run_dir: Path) -> Path:
    return run_dir / "error_analysis.json"


def main() -> None:
    args = parse_args()
    output = args.output or default_output(args.run_dir)
    analysis = analyze_run(args.run_dir, args.candidate_models, args.max_failed_examples)
    write_json(output, analysis)
    print(output)


if __name__ == "__main__":
    main()

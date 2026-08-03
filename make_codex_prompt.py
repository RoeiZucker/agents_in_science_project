#!/usr/bin/env python3
"""Generate a Codex prompt for operating the evaluation agent."""
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--project-root", default="$PWD/runtime")
    parser.add_argument("--hf-home", default="$PWD/runtime/.hf_cache")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
    models = "\n".join(f"- {model}" for model in args.models)
    full_instruction = "Run full evaluation only if smoke passes." if args.stage == "full" else "Stop after the requested stage."
    return f"""You are in the hf-eval-agent repo.

Use the local evaluation-agent scripts to evaluate this dataset/model request.

Dataset:
- {args.dataset}

Models:
{models}

Split:
- {args.split}

Runtime paths:
- PROJECT_ROOT={args.project_root}
- HF_HOME={args.hf_home}

Rules:
- Do not ask follow-up questions.
- Make reasonable assumptions and keep moving.
- Do not modify code unless a real script bug blocks the run.
- Always inspect generated JSON/CSV/audit/log files before deciding the next step.
- If a model is gated, missing, or too large to run, record the exact failure and continue to the next model when possible.
- Use --continue-on-error for multi-model runs.

Workflow:
1. Export PROJECT_ROOT and HF_HOME, and create PROJECT_ROOT.
2. Run plan with run_eval_agent.py.
3. Inspect plans.json and results.csv.
4. Run smoke with --smoke-limit {args.smoke_limit}.
5. Inspect results.csv, audits/*.json, and logs/*.stderr.txt if present.
6. {full_instruction}
7. Summarize final scores and provide paths to results.csv, summary.json, and predictions.jsonl.

Requested final stage:
- {args.stage}
"""


def main() -> None:
    args = parse_args()
    prompt = build_prompt(args)
    if args.output_file:
        args.output_file.write_text(prompt, encoding="utf-8")
    else:
        print(prompt)


if __name__ == "__main__":
    main()

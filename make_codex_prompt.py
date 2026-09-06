#!/usr/bin/env python3
"""Generate a metadata-only Codex context-scout prompt.

Examples:
  # Print a prompt for direct use with ``codex exec``.
  python make_codex_prompt.py --dataset google/boolq --models google/flan-t5-small

  # Save the prompt and ask Codex to write context to a chosen path.
  python make_codex_prompt.py --dataset google/boolq --models google/flan-t5-small \
    --context-output runtime/boolq_context.json --output-file codex_prompt.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--project-root", default="$PWD/runtime")
    parser.add_argument("--hf-home", default="$PWD/runtime/.hf_cache")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--context-output", type=Path, default=Path("codex_context.json"))
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--smoke-limit", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full", help=argparse.SUPPRESS)
    return parser.parse_args()


def build_prompt(args: argparse.Namespace) -> str:
    models = "\n".join(f"- {model}" for model in args.models)
    return f"""You are a metadata-only context scout for hf-eval-agent.

Dataset: {args.dataset}
Requested split: {args.split}
Models:
{models}

Your only task is to inspect dataset/model cards and lightweight metadata, then write extra context for the deterministic evaluator.

Hard restrictions:
- Never run run_eval_agent.py or evaluate_hf_pair.py.
- Never load model weights, create a model pipeline, train, or run inference.
- Never run smoke or full evaluation.
- Never edit repository code or delete caches.

Allowed actions:
- Read local repository files and existing metadata.
- Inspect Hugging Face cards, builder schema/features, model config, and at most a tiny dataset sample when metadata is insufficient.
- If Python is needed, use {args.python}.

Runtime:
- PROJECT_ROOT={args.project_root}
- HF_HOME={args.hf_home}

Write exactly one JSON object to {args.context_output}:
{{
  "dataset": "{args.dataset}",
  "models": ["model ids"],
  "dataset_context": {{
    "question_column": "existing column or dotted path, or empty",
    "answer_column": "existing column or dotted/list path, or empty",
    "choices_column": "existing choices column, or empty",
    "image_column": "existing image column, or empty",
    "task": "generation|multiple_choice|image_classification or empty",
    "label_map": {{}},
    "prompt_template": "optional template using dataset columns",
    "notes": "brief evidence for the context"
  }},
  "model_context": {{"notes": "metadata-only notes; no inference claims"}},
  "confidence": "high|medium|low"
}}

Leave fields empty rather than guessing. After writing the JSON file, stop immediately.
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

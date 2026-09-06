#!/usr/bin/env python3
"""Keep untried tasks and failures fixed by the text-only GGUF adapter.

Example:
    python prepare_actionable_pending_array.py \
      --source config/intersection90_all_pending_conditions.csv \
      --results-root /path/to/cancelled/array \
      --selected-output config/intersection90_pending_after_cancel_conditions.csv \
      --tasks-output config/intersection90_pending_after_cancel_tasks.csv
"""

import argparse
from collections import defaultdict
from pathlib import Path

from prepare_retry_or_untried_array import build_manifests, pair_key, read_csv, write_csv


SUCCESS = {"ok", "success", "succeeded"}
FIXED_FAILURE_TEXT = "no direct adapter exists for this model format"
UNSUPPORTED_GGUF_MODELS = {
    "bartowski/google_medgemma-4b-it-GGUF",
    "ggml-org/Qwen2.5-Omni-7B-GGUF",
    "lmstudio-community/medgemma-4b-it-GGUF",
    "mradermacher/GuardReasoner-VL-Eco-7B-GGUF",
    "mradermacher/Lingshu-7B-GGUF",
    "unsloth/Qwen2.5-Omni-7B-GGUF",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    return parser.parse_args()


def load_task_results(root):
    results = defaultdict(list)
    for path in root.rglob("batch_results.csv"):
        for row in read_csv(path):
            results[pair_key(row)].append(row)
    return results


def result_status(row):
    return (row.get("eval_status") or row.get("status") or "").lower()


def fixed_failure(row):
    notes = (row.get("eval_notes") or "").lower()
    model = (row.get("model_name") or row.get("model") or "").lower()
    return "gguf" in model and result_status(row) not in SUCCESS and FIXED_FAILURE_TEXT in notes


def actionable_pairs(source_rows, results):
    pairs = {pair_key(row) for row in source_rows}
    return {
        pair
        for pair in pairs
        if pair[1] not in UNSUPPORTED_GGUF_MODELS
        and (pair not in results or any(fixed_failure(row) for row in results[pair]))
    }


def main():
    args = parse_args()
    source_rows = read_csv(args.source)
    results = load_task_results(args.results_root)
    allowed = actionable_pairs(source_rows, results)
    rows = [row for row in source_rows if pair_key(row) in allowed]
    selected, tasks = build_manifests(rows, excluded=set())
    write_csv(args.selected_output, selected)
    write_csv(args.tasks_output, tasks)
    untried = sum(pair not in results for pair in allowed)
    source_pairs = {pair_key(row) for row in source_rows}
    successful = sum(
        any(result_status(row) in SUCCESS for row in results.get(pair, []))
        for pair in source_pairs
    )
    unresolved = len(source_pairs) - len(allowed) - successful
    print(f"Untried or interrupted pairs: {untried}", flush=True)
    print(f"Pairs retryable after GGUF fix: {len(allowed) - untried}", flush=True)
    print(f"Previously successful pairs omitted: {successful}", flush=True)
    print(f"Excluded unresolved or unsupported failures: {unresolved}", flush=True)
    print(f"Selected condition rows: {len(selected)}", flush=True)
    print(f"Unique actionable tasks: {len(tasks)}", flush=True)


if __name__ == "__main__":
    main()

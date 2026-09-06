#!/usr/bin/env python3
"""Build manifests for pairs without a successful full evaluation.

Example:
    python prepare_retry_or_untried_array.py \
      --source config/evaluation_conditions_intersection90.csv \
      --results-root /path/to/first/full/run \
      --results-root /path/to/second/full/run \
      --exclude-pair 'dataset-id::model-id' \
      --selected-output config/all_pending_conditions.csv \
      --tasks-output config/all_pending_tasks.csv
"""

import argparse
import csv
from pathlib import Path


SUCCESS = {"ok", "success", "succeeded"}
RESULT_NAMES = {
    "array_results.csv",
    "batch_results.csv",
    "retry_failed_results_combined.csv",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, action="append", default=[])
    parser.add_argument("--exclude-pair", action="append", default=[])
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_key(row):
    dataset = row.get("query_dataset") or row.get("dataset") or ""
    model = row.get("model_name") or row.get("model") or ""
    return dataset, model


def result_status(row):
    return (row.get("eval_status") or row.get("status") or row.get("outcome") or "").lower()


def result_files(root):
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("*.csv") if path.name in RESULT_NAMES)


def successful_pairs(roots):
    successful = set()
    for root in roots:
        for path in result_files(root):
            successful.update(
                pair_key(row) for row in read_csv(path) if result_status(row) in SUCCESS
            )
    return successful


def parse_excluded_pair(value):
    parts = value.split("::", 1)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected DATASET::MODEL, got {value!r}")
    return tuple(parts)


def build_manifests(rows, excluded):
    task_indexes = {}
    tasks = []
    selected = []
    for source_row, row in enumerate(rows, start=1):
        key = pair_key(row)
        if key in excluded:
            continue
        if key not in task_indexes:
            task_indexes[key] = len(tasks)
            tasks.append({**row, "task_index": task_indexes[key]})
        selected.append(
            {**row, "task_index": task_indexes[key], "source_row": source_row}
        )
    return selected, tasks


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    successes = successful_pairs(args.results_root)
    active = {parse_excluded_pair(value) for value in args.exclude_pair}
    selected, tasks = build_manifests(read_csv(args.source), successes | active)
    write_csv(args.selected_output, selected)
    write_csv(args.tasks_output, tasks)
    print(f"Successful pairs excluded: {len(successes)}", flush=True)
    print(f"Active/manual pairs excluded: {len(active)}", flush=True)
    print(f"Selected condition rows: {len(selected)}", flush=True)
    print(f"Unique retry or untried tasks: {len(tasks)}", flush=True)


if __name__ == "__main__":
    main()

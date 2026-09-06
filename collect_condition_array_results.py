#!/usr/bin/env python3
"""Collect Slurm array outputs and restore all original condition rows.

Example:
    python collect_condition_array_results.py \
      --selected-csv config/intersection90_first50_conditions.csv \
      --array-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_intersection90_array50/123456
"""

import argparse
import csv
import json
from pathlib import Path


EVAL_COLUMNS = (
    "eval_split",
    "eval_protocol",
    "eval_metric",
    "eval_score",
    "eval_total",
    "eval_labeled_total",
    "eval_status",
    "eval_output_dir",
    "eval_notes",
)
SUCCESS_STATUSES = {"ok", "success"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--array-root", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_task_result(array_root, task_index):
    path = array_root / f"task_{task_index:03d}" / "batch_results.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    return rows[0] if rows else None


def merge_results(selected_rows, array_root):
    cache = {}
    merged = []
    for condition in selected_rows:
        task_index = int(condition["task_index"])
        if task_index not in cache:
            cache[task_index] = read_task_result(array_root, task_index)
        result = cache[task_index]
        evaluation = {column: "" for column in EVAL_COLUMNS}
        if result:
            evaluation.update({column: result.get(column, "") for column in EVAL_COLUMNS})
        else:
            evaluation["eval_status"] = "missing_task_result"
        merged.append({**condition, **evaluation})
    return merged


def collect_failures(array_root, task_indexes):
    failures = []
    for task_index in task_indexes:
        path = array_root / f"task_{task_index:03d}" / "batch_failures.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            for failure in payload:
                failures.append({"task_index": task_index, **failure})
    return failures


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def task_statuses(rows):
    statuses = {}
    for row in rows:
        statuses.setdefault(int(row["task_index"]), row["eval_status"])
    return statuses


def count_statuses(statuses):
    counts = {}
    for status in statuses.values():
        counts[status] = counts.get(status, 0) + 1
    return counts


def make_summary(rows, task_indexes, output_path):
    statuses = task_statuses(rows)
    completed = {
        task_index
        for task_index, status in statuses.items()
        if status != "missing_task_result"
    }
    successful = {
        task_index
        for task_index, status in statuses.items()
        if status in SUCCESS_STATUSES
    }
    return {
        "selected_condition_rows": len(rows),
        "unique_tasks": len(task_indexes),
        "tasks_with_results": len(completed),
        "successful_tasks": len(successful),
        "failed_or_missing_tasks": sorted(task_indexes - successful),
        "missing_tasks": sorted(task_indexes - completed),
        "task_status_counts": count_statuses(statuses),
        "successful_condition_rows": sum(
            row["eval_status"] in SUCCESS_STATUSES for row in rows
        ),
        "results_csv": str(output_path),
    }


def main():
    args = parse_args()
    selected = read_csv(args.selected_csv)
    merged = merge_results(selected, args.array_root)
    task_indexes = {int(row["task_index"]) for row in selected}
    results_path = args.array_root / "array_results.csv"
    write_csv(results_path, merged)
    write_json(
        args.array_root / "array_failures.json",
        collect_failures(args.array_root, task_indexes),
    )
    summary = make_summary(merged, task_indexes, results_path)
    write_json(args.array_root / "array_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

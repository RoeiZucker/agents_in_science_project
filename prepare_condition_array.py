#!/usr/bin/env python3
"""Prepare condition and task manifests for a Slurm evaluation array.

Prefix-row example:
    python prepare_condition_array.py \
      --source config/evaluation_conditions_intersection90.csv \
      --limit 50 \
      --selected-output config/intersection90_first50_conditions.csv \
      --tasks-output config/intersection90_first50_tasks.csv

Balanced unique-task example:
    python prepare_condition_array.py \
      --source config/evaluation_conditions_intersection90.csv \
      --task-limit 100 --selection-strategy dataset-round-robin \
      --selected-output config/intersection90_balanced100_conditions.csv \
      --tasks-output config/intersection90_balanced100_tasks.csv

Prior-task retry example:
    python prepare_condition_array.py \
      --source config/intersection90_balanced100_tasks.csv \
      --source-task-indexes 2,9,33,48,49,53,54,72,80,81 \
      --selected-output config/intersection90_fixed10_conditions.csv \
      --tasks-output config/intersection90_fixed10_tasks.csv
"""

import argparse
import csv
from pathlib import Path


PAIR_COLUMNS = ("query_dataset", "model_name")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--source-task-indexes",
        default="",
        help="Optional comma-separated task indexes to select from a prior task manifest.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument(
        "--selection-strategy",
        choices=("prefix", "dataset-round-robin"),
        default="prefix",
    )
    parser.add_argument("--selected-output", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))



def parse_task_indexes(value):
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def filter_task_rows(rows, task_indexes):
    if not task_indexes:
        return rows
    by_index = {int(row["task_index"]): row for row in rows}
    missing = [index for index in task_indexes if index not in by_index]
    if missing:
        raise ValueError(f"Source task indexes are missing: {missing}")
    return [by_index[index] for index in task_indexes]


def build_manifests(rows, limit):
    selected = []
    tasks = []
    task_indexes = {}
    for source_row, row in enumerate(rows[:limit], start=1):
        key = tuple(row[column] for column in PAIR_COLUMNS)
        if key not in task_indexes:
            task_indexes[key] = len(tasks)
            tasks.append({**row, "task_index": task_indexes[key]})
        selected.append(
            {**row, "source_row": source_row, "task_index": task_indexes[key]}
        )
    return selected, tasks

def pair_key(row):
    return tuple(row[column] for column in PAIR_COLUMNS)


def unique_pair_rows(rows):
    unique = {}
    for row in rows:
        unique.setdefault(pair_key(row), row)
    return list(unique.values())


def round_robin_rows(rows, limit):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["query_dataset"], []).append(row)
    selected = []
    depth = 0
    while len(selected) < limit:
        added = False
        for candidates in by_dataset.values():
            if depth < len(candidates):
                selected.append(candidates[depth])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            return selected
        depth += 1
    return selected


def select_task_rows(rows, task_limit, strategy):
    unique = unique_pair_rows(rows)
    if strategy == "prefix":
        return unique[:task_limit]
    return round_robin_rows(unique, task_limit)


def build_task_manifests(rows, task_limit, strategy):
    task_rows = select_task_rows(rows, task_limit, strategy)
    task_indexes = {pair_key(row): index for index, row in enumerate(task_rows)}
    tasks = [{**row, "task_index": index} for index, row in enumerate(task_rows)]
    selected = [
        {**row, "source_row": source_row, "task_index": task_indexes[pair_key(row)]}
        for source_row, row in enumerate(rows, start=1)
        if pair_key(row) in task_indexes
    ]
    return selected, tasks


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    rows = read_rows(args.source)
    rows = filter_task_rows(
        rows, parse_task_indexes(args.source_task_indexes)
    )
    if args.task_limit:
        selected, tasks = build_task_manifests(
            rows, args.task_limit, args.selection_strategy
        )
    else:
        selected, tasks = build_manifests(rows, args.limit)
    write_rows(args.selected_output, selected)
    write_rows(args.tasks_output, tasks)
    print(f"Selected condition rows: {len(selected)}", flush=True)
    print(f"Unique evaluation tasks: {len(tasks)}", flush=True)
    print(f"Duplicate evaluations avoided: {len(selected) - len(tasks)}", flush=True)


if __name__ == "__main__":
    main()

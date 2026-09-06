#!/usr/bin/env python3
"""Build full-evaluation tasks by joining ranked pairs to their saved plans.

Example:
  python prepare_oren_ranked_full.py     --ranked-pairs oren62_partial_ranked_pairs_31511291.csv     --plans-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren62_selection_final_20260905/selection     --tasks-output config/oren62_ranked97_full_tasks.csv     --audit-output config/oren62_ranked97_full_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

TASK_FIELDS = (
    "task_index",
    "source_task_index",
    "dataset",
    "model",
    "plan_file",
    "final_rank",
    "selection_score",
    "selection_metric",
    "selection_evaluation_protocol",
    "evaluated_split",
    "selection_labeled_total",
    "round",
    "retrieval_score",
    "source",
    "candidate_id",
)
AUDIT_FIELDS = (
    "source_task_index",
    "dataset",
    "model",
    "round",
    "decision",
    "reason",
    "plan_file",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-pairs", type=Path, required=True)
    parser.add_argument("--plans-root", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plan_round(path: Path) -> int:
    for parent in path.parents:
        match = re.fullmatch(r"round_(\d+)", parent.name)
        if match:
            return int(match.group(1))
    raise ValueError(f"No round directory found above {path}")


def plan_key(plan: dict, round_number: int) -> tuple[str, str, int]:
    dataset = plan["dataset"]["dataset"]
    model = plan["model"]["model"]
    return dataset, model, round_number


def build_plan_index(plans_root: Path) -> dict[tuple[str, str, int], list[Path]]:
    index: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in sorted(plans_root.rglob("plans.json")):
        plans = json.loads(path.read_text(encoding="utf-8"))
        for plan in plans:
            index[plan_key(plan, plan_round(path))].append(path)
    return dict(index)


def row_key(row: dict[str, str]) -> tuple[str, str, int]:
    return row["dataset"], row["model"], int(row["round"])


def task_row(
    task_index: int, source_index: int, row: dict[str, str], plan_file: Path
) -> dict[str, object]:
    return {
        "task_index": task_index,
        "source_task_index": source_index,
        "dataset": row["dataset"],
        "model": row["model"],
        "plan_file": str(plan_file),
        "final_rank": row["final_rank"],
        "selection_score": row["score"],
        "selection_metric": row["metric"],
        "selection_evaluation_protocol": row["evaluation_protocol"],
        "evaluated_split": row["evaluated_split"],
        "selection_labeled_total": row["labeled_total"],
        "round": row["round"],
        "retrieval_score": row["retrieval_score"],
        "source": row["source"],
        "candidate_id": row["candidate_id"],
    }


def audit_row(
    source_index: int, row: dict[str, str], matches: list[Path]
) -> dict[str, object]:
    unique = len(matches) == 1
    return {
        "source_task_index": source_index,
        "dataset": row["dataset"],
        "model": row["model"],
        "round": row["round"],
        "decision": "run" if unique else "skip",
        "reason": "" if unique else f"Expected one saved plan; found {len(matches)}",
        "plan_file": str(matches[0]) if unique else "",
    }


def build_tasks(
    rows: list[dict[str, str]],
    plan_index: dict[tuple[str, str, int], list[Path]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks: list[dict[str, object]] = []
    audit: list[dict[str, object]] = []
    for source_index, row in enumerate(rows):
        matches = plan_index.get(row_key(row), [])
        audit.append(audit_row(source_index, row, matches))
        if len(matches) == 1:
            tasks.append(task_row(len(tasks), source_index, row, matches[0]))
    return tasks, audit


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.ranked_pairs)
    plan_index = build_plan_index(args.plans_root)
    tasks, audit = build_tasks(rows, plan_index)
    write_rows(args.tasks_output, TASK_FIELDS, tasks)
    write_rows(args.audit_output, AUDIT_FIELDS, audit)
    skipped = len(rows) - len(tasks)
    print(f"Full-evaluation tasks: {len(tasks)}")
    print(f"Unmapped/ambiguous: {skipped}")
    if skipped:
        raise SystemExit("Every ranked pair must map to exactly one saved plan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

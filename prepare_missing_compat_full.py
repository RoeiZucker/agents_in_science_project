#!/usr/bin/env python3
"""Build a full-evaluation manifest from successful compatibility smoke tasks.

Example:
  python prepare_missing_compat_full.py \
    --smoke-run /path/to/missing_compat_smoke_20260901_181014 \
    --tasks-output config/missing_compat_full42_tasks.csv \
    --audit-output config/missing_compat_full42_audit.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_FIELDS = ("task_index", "source_task_index", "dataset", "model", "plan_file")
AUDIT_FIELDS = (
    "smoke_task_index",
    "source_task_index",
    "dataset",
    "model",
    "decision",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def smoke_outcome(task_dir: Path) -> tuple[dict, str]:
    task_path = task_dir / "task_summary.json"
    if not task_path.exists():
        return {}, "task_summary.json is missing"
    task = read_json(task_path)
    if task.get("returncode") != 0:
        return task, f"returncode={task.get('returncode')}"
    summary_path = task_dir / "eval_result" / "summary.json"
    if not summary_path.exists():
        return task, "summary.json is missing"
    if read_json(summary_path).get("status") == "unsupported":
        return task, "status=unsupported"
    return task, "passed smoke"


def task_directories(smoke_run: Path) -> list[Path]:
    return sorted(smoke_run.glob("task_*"))


def full_task(index: int, task: dict) -> dict[str, str]:
    return {
        "task_index": str(index),
        "source_task_index": str(task["source_task_index"]),
        "dataset": task["dataset"],
        "model": task["model"],
        "plan_file": task["plan_file"],
    }


def audit_row(task_dir: Path, task: dict, reason: str) -> dict[str, str]:
    return {
        "smoke_task_index": task_dir.name.removeprefix("task_"),
        "source_task_index": str(task.get("source_task_index", "")),
        "dataset": task.get("dataset", ""),
        "model": task.get("model", ""),
        "decision": "run" if reason == "passed smoke" else "skip",
        "reason": reason,
    }


def select_tasks(smoke_run: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected = []
    audit = []
    for task_dir in task_directories(smoke_run):
        task, reason = smoke_outcome(task_dir)
        if reason == "passed smoke":
            selected.append(full_task(len(selected), task))
        audit.append(audit_row(task_dir, task, reason))
    return selected, audit


def write_rows(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected, audit = select_tasks(args.smoke_run)
    write_rows(args.tasks_output, selected, TASK_FIELDS)
    write_rows(args.audit_output, audit, AUDIT_FIELDS)
    print(f"Full-evaluation tasks: {len(selected)}")
    print(f"Skipped smoke tasks: {len(audit) - len(selected)}")


if __name__ == "__main__":
    main()

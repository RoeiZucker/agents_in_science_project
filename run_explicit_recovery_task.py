#!/usr/bin/env python3
"""Run one explicitly configured recovery task.

Examples:
  python run_explicit_recovery_task.py --tasks-csv config/top1_final_recovery3_tasks.csv --task-index 0 --project-root runtime --output-root runtime/recovery --stage plan
  python run_explicit_recovery_task.py --tasks-csv config/top1_final_recovery3_tasks.csv --task-index 2 --project-root runtime --output-root runtime/recovery --stage full --full-limit 5000 --trust-remote-code
  python run_explicit_recovery_task.py --tasks-csv config/max_coverage_recovery_tasks.csv --task-index 0 --project-root runtime --output-root runtime/max_coverage --stage full --full-limit 5000
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", required=True)
    parser.add_argument("--task-index", required=True, type=int)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument("--full-limit", type=int, default=5000)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_task(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if index < 0 or index >= len(rows):
        raise IndexError(f"Task index {index} is outside 0..{len(rows) - 1}")
    return rows[index]


def task_output(root: Path, index: int) -> Path:
    return root / f"task_{index:03d}"


def append_task_option(
    command: list[str], task: dict[str, str], column: str, flag: str
) -> None:
    value = task.get(column, "").strip()
    if value:
        command.extend((flag, value))

def build_command(args: argparse.Namespace, task: dict[str, str]) -> list[str]:
    command = [
        args.python,
        str(Path(__file__).with_name("run_eval_agent.py")),
        "--dataset",
        task["dataset"],
        "--models",
        task["model"],
        "--split",
        task.get("split") or "auto",
        "--project-root",
        args.project_root,
        "--output-root",
        str(task_output(Path(args.output_root), args.task_index)),
        "--stage",
        args.stage,
        "--full-limit",
        str(args.full_limit),
    ]
    for column, flag in (
        ("context_file", "--context-file"),
        ("model_source", "--model-source"),
        ("dataset_source", "--dataset-source"),
        ("dataset_kwargs", "--dataset-kwargs"),
        ("subset", "--subset"),
        ("label_threshold", "--label-threshold"),
        ("answer_regex", "--answer-regex"),
    ):
        append_task_option(command, task, column, flag)
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def main() -> int:
    args = parse_args()
    task = read_task(Path(args.tasks_csv), args.task_index)
    command = build_command(args, task)
    print(f"Recovery task {args.task_index}: {task['dataset']} -> {task['model']}", flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

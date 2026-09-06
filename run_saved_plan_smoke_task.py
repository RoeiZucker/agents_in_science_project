#!/usr/bin/env python3
"""Run a three-example smoke test from a previously saved evaluation plan.

Examples:
  python run_saved_plan_smoke_task.py \
    --tasks-csv config/missing_compat_smoke_tasks.csv \
    --task-index 0 \
    --output-root /tmp/missing_compat_smoke \
    --python ../.venv-artifact-linker/bin/python \
    --limit 3
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def read_task(path: Path, index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["task_index"]) == index:
                return row
    raise IndexError(f"Task index {index} is not present in {path}")


def read_plan(path: Path, model: str) -> dict:
    plans = json.loads(path.read_text(encoding="utf-8"))
    matches = [plan for plan in plans if plan.get("model", {}).get("model") == model]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved plan for {model!r}, found {len(matches)}")
    return matches[0]


def replace_option(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend((flag, value))


def smoke_command(plan: dict, python: str, output_dir: Path, limit: int) -> list[str]:
    command = list(plan["smoke_command"])
    command[0] = python
    command[1] = str(Path(__file__).with_name("evaluate_hf_pair.py"))
    replace_option(command, "--limit", str(limit))
    replace_option(command, "--output-dir", str(output_dir))
    return command


def run_task(args: argparse.Namespace, task: dict[str, str]) -> int:
    task_dir = args.output_root / f"task_{args.task_index:03d}"
    output_dir = task_dir / "eval_result"
    logs = task_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plan = read_plan(Path(task["plan_file"]), task["model"])
    command = smoke_command(plan, args.python, output_dir, args.limit)
    (logs / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Smoke task {args.task_index}: {task['dataset']} :: {task['model']}", flush=True)
    with (logs / "stdout.txt").open("w") as stdout, (logs / "stderr.txt").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    summary = {
        **task,
        "returncode": result.returncode,
        "output_dir": str(output_dir),
        "summary_exists": (output_dir / "summary.json").exists(),
    }
    (task_dir / "task_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if summary["summary_exists"] else result.returncode


def main() -> int:
    args = parse_args()
    return run_task(args, read_task(args.tasks_csv, args.task_index))


if __name__ == "__main__":
    raise SystemExit(main())

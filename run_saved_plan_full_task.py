#!/usr/bin/env python3
"""Run a full evaluation from a previously saved and smoke-tested plan.

Example:
  python run_saved_plan_full_task.py \
    --tasks-csv config/missing_compat_full42_tasks.csv \
    --task-index 0 \
    --output-root /tmp/missing_compat_full5000 \
    --python ../.venv-artifact-linker/bin/python \
    --limit 5000 \
    --seed 1042
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from run_saved_plan_smoke_task import read_plan, read_task, replace_option, smoke_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def read_summary(output_dir: Path) -> dict:
    path = output_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def result_status(returncode: int, summary: dict) -> str:
    if summary.get("status") == "unsupported":
        return "unsupported"
    return "ok" if returncode == 0 and summary else "failed"


def result_row(
    task: dict[str, str], output_dir: Path, returncode: int, summary: dict
) -> dict[str, object]:
    return {
        "dataset": task["dataset"],
        "model": task["model"],
        "eval_status": result_status(returncode, summary),
        "eval_protocol": summary.get("evaluation_protocol", ""),
        "eval_metric": summary.get("metric", ""),
        "eval_score": summary.get("score", ""),
        "eval_total": summary.get("total", ""),
        "eval_labeled_total": summary.get("labeled_total", ""),
        "eval_output_dir": str(output_dir),
        "failure_reason": (
            summary.get("unsupported_reason", "")
            if summary
            else "summary.json is missing"
        ),
    }


def write_csv(path: Path, row: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run_task(args: argparse.Namespace, task: dict[str, str]) -> int:
    task_dir = args.output_root / f"task_{args.task_index:03d}"
    output_dir = task_dir / "eval_result"
    logs = task_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    plan = read_plan(Path(task["plan_file"]), task["model"])
    command = smoke_command(plan, args.python, output_dir, args.limit)
    seed = getattr(args, "seed", None)
    if seed is not None:
        replace_option(command, "--seed", str(seed))
    write_json(logs / "command.json", command)
    print(
        f"Full saved-plan task {args.task_index}: "
        f"{task['dataset']} :: {task['model']}",
        flush=True,
    )
    with (logs / "stdout.txt").open("w") as stdout, (
        logs / "stderr.txt"
    ).open("w") as stderr:
        returncode = subprocess.run(
            command, stdout=stdout, stderr=stderr, check=False
        ).returncode
    summary = read_summary(output_dir)
    write_csv(
        task_dir / "batch_results.csv",
        result_row(task, output_dir, returncode, summary),
    )
    write_json(
        task_dir / "task_summary.json",
        {
            **task,
            "returncode": returncode,
            "output_dir": str(output_dir),
            "summary_exists": bool(summary),
        },
    )
    return 0 if summary else returncode


def main() -> int:
    args = parse_args()
    return run_task(args, read_task(args.tasks_csv, args.task_index))


if __name__ == "__main__":
    raise SystemExit(main())

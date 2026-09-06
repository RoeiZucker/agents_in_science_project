#!/usr/bin/env python3
"""Run one dataset-model pair from a prepared Slurm task manifest.

Example:
    python run_condition_array_task.py \
      --tasks-csv config/intersection90_first50_tasks.csv \
      --task-index 0 \
      --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
      --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/test_array \
      --python ../.venv-artifact-linker/bin/python \
      --trust-remote-code \
      --full-limit 1000

    Encoder/classifier direct scoring is automatic; --allow-label-scores remains optional.
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--runner", choices=("codex", "script"), default="codex")
    parser.add_argument("--full-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-label-scores", action="store_true")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-timeout", type=int, default=120)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def read_task(path, task_index):
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["task_index"]) == task_index:
                return row
    raise ValueError(f"Task index {task_index} is not present in {path}")


def write_condition(path, task):
    row = {key: value for key, value in task.items() if key != "task_index"}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def build_command(args, condition_path, task_dir):
    command = [
        args.python,
        "run_evaluation_conditions.py",
        "--conditions-csv",
        str(condition_path),
        "--project-root",
        str(args.project_root),
        "--output-root",
        str(task_dir),
        "--stage",
        args.stage,
        "--smoke-limit",
        str(args.smoke_limit),
        "--full-limit",
        str(getattr(args, "full_limit", 1000)),
        "--seed",
        str(getattr(args, "seed", 42)),
        "--runner",
        args.runner,
        "--codex-bin",
        args.codex_bin,
        "--codex-timeout",
        str(args.codex_timeout),
        "--codex-bypass-sandbox",
        "--skip-refinement",
        "--fresh-run-dir",
        "--cleanup-cache",
        "none",
        "--python",
        args.python,
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if getattr(args, "allow_label_scores", False):
        command.append("--allow-label-scores")
    return command


def write_summary(path, task, returncode):
    payload = {
        "task_index": int(task["task_index"]),
        "dataset": task["query_dataset"],
        "model": task["model_name"],
        "returncode": returncode,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    task = read_task(args.tasks_csv, args.task_index)
    task_dir = args.output_root / f"task_{args.task_index:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    condition_path = task_dir / "input_condition.csv"
    write_condition(condition_path, task)
    command = build_command(args, condition_path, task_dir)
    print(
        f"[task {args.task_index}] {task['query_dataset']} :: {task['model_name']}",
        flush=True,
    )
    print("Command: " + " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    write_summary(task_dir / "task_summary.json", task, result.returncode)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

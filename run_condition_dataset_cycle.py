#!/usr/bin/env python3
"""Run conditions evaluation as download -> evaluate -> delete per dataset.

Examples:
  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --stage smoke \
    --runner codex \
    --trust-remote-code

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --stage smoke \
    --runner codex \
    --trust-remote-code

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --skip-model-download \
    --stage smoke

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --condition A_merged \
    --stage full \
    --runner script \
    --trust-remote-code
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"condition", "query_dataset", "rank", "model_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--download-split", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument("--runner", choices=("script", "codex"), default="script")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-approval", default="never")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--fresh-run-dir", action="store_true")
    parser.add_argument("--keep-dataset-cache", action="store_true")
    parser.add_argument("--skip-model-download", action="store_true", help="Only predownload datasets, not model snapshots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    datasets = selected_datasets(args)
    results = run_cycle(datasets, args, script_dir)
    write_cycle_report(args, results)
    print(json.dumps(summary(args, results), indent=2))
    if has_failure(results):
        sys.exit(1)


def selected_datasets(args: argparse.Namespace) -> list[str]:
    rows = selected_rows(args)
    names = unique([row["query_dataset"] for row in rows])
    if args.limit_datasets:
        names = names[: args.limit_datasets]
    return names


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_conditions(args.conditions_csv)
    if args.condition:
        rows = [row for row in rows if row["condition"] in set(args.condition)]
    if args.dataset:
        rows = [row for row in rows if row["query_dataset"] in set(args.dataset)]
    if args.limit_pairs:
        rows = rows[: args.limit_pairs]
    return rows


def read_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("conditions CSV is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"conditions CSV is missing columns: {sorted(missing)}")


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def run_cycle(datasets: list[str], args: argparse.Namespace, script_dir: Path) -> list[dict[str, Any]]:
    results = []
    for dataset in datasets:
        result = run_dataset_cycle(dataset, args, script_dir)
        results.append(result)
        if result["status"] == "failed" and not args.continue_on_error:
            break
    return results


def run_dataset_cycle(dataset: str, args: argparse.Namespace, script_dir: Path) -> dict[str, Any]:
    steps = []
    steps.append(run_step("download", download_command(dataset, args, script_dir), script_dir))
    if steps[-1]["returncode"] == 0:
        steps.append(evaluate_step(evaluate_command(dataset, args, script_dir), script_dir))
    if not args.keep_dataset_cache:
        steps.append(run_step("delete", delete_command(dataset, args, script_dir), script_dir))
    return {"dataset": dataset, "status": status(steps), "steps": steps}


def run_step(name: str, command: list[str], cwd: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": tail(result.stdout),
        "stderr_tail": tail(result.stderr),
    }


def evaluate_step(command: list[str], cwd: Path) -> dict[str, Any]:
    step = run_step("evaluate", command, cwd)
    failures = reported_failures(step["stdout_tail"])
    if failures:
        step["returncode"] = 1
        step["reported_failures"] = failures
    return step


def reported_failures(text: str) -> int:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return 0
    return int(value.get("failures", 0))


def download_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [args.python, str(script_dir / "download_condition_datasets.py"), "--conditions-csv", str(args.conditions_csv), "--project-root", str(args.project_root), "--dataset", dataset]
    command.extend(download_split_args(args))
    if args.output_root:
        command.extend(["--output-root", str(args.output_root / "_dataset_downloads")])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if not args.skip_model_download:
        command.append("--include-models")
    if args.limit_pairs:
        command.extend(["--limit-pairs", str(args.limit_pairs)])
    for condition in args.condition:
        command.extend(["--condition", condition])
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def download_split_args(args: argparse.Namespace) -> list[str]:
    return [item for split in args.download_split for item in ["--split", split]]


def evaluate_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [
        args.python,
        str(script_dir / "run_evaluation_conditions.py"),
        "--conditions-csv",
        str(args.conditions_csv),
        "--project-root",
        str(args.project_root),
        "--split",
        args.split,
        "--stage",
        args.stage,
        "--runner",
        args.runner,
        "--codex-bin",
        args.codex_bin,
        "--codex-approval",
        args.codex_approval,
        "--smoke-limit",
        str(args.smoke_limit),
        "--sample-size",
        str(args.sample_size),
        "--dataset",
        dataset,
        "--python",
        args.python,
        "--cleanup-cache",
        "none",
    ]
    command.extend(optional_eval_args(args))
    return command


def optional_eval_args(args: argparse.Namespace) -> list[str]:
    values = []
    if args.output_root:
        values.extend(["--output-root", str(args.output_root)])
    if args.timeout:
        values.extend(["--timeout", str(args.timeout)])
    if args.limit_pairs:
        values.extend(["--limit-pairs", str(args.limit_pairs)])
    if args.trust_remote_code:
        values.append("--trust-remote-code")
    if args.continue_on_error:
        values.append("--continue-on-error")
    if args.skip_refinement:
        values.append("--skip-refinement")
    if args.fresh_run_dir:
        values.append("--fresh-run-dir")
    for condition in args.condition:
        values.extend(["--condition", condition])
    return values


def delete_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [args.python, str(script_dir / "delete_condition_datasets.py"), "--project-root", str(args.project_root), "--dataset", dataset]
    if args.output_root:
        command.extend(["--output-root", str(args.output_root / "_dataset_downloads")])
    return command


def status(steps: list[dict[str, Any]]) -> str:
    return "ok" if all(step["returncode"] == 0 for step in steps) else "failed"


def tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def has_failure(results: list[dict[str, Any]]) -> bool:
    return any(result["status"] == "failed" for result in results)


def write_cycle_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    output_root = args.output_root or args.project_root / "eval_results" / "_condition_runs"
    path = output_root / "dataset_cycle_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    output_root = args.output_root or args.project_root / "eval_results" / "_condition_runs"
    return {
        "datasets": len(results),
        "failed_datasets": sum(1 for result in results if result["status"] == "failed"),
        "report": str(output_root / "dataset_cycle_report.json"),
    }


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a smoke manifest from live missing results, excluding OOM and recovered rows.

Examples:
  python prepare_missing_compat_smoke.py \
    --source-tasks config/salvage_recovery_generic_tasks.csv \
    --prior-run /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_salvage_recovery_full5000/salvage_recovery_20260831_190812/generic \
    --tasks-output config/missing_compat_smoke_tasks.csv \
    --audit-output config/missing_compat_smoke_audit.csv \
    --fallback-plan /path/to/recovered/plans.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TASK_FIELDS = ("task_index", "source_task_index", "dataset", "model", "plan_file")
AUDIT_FIELDS = ("source_task_index", "dataset", "model", "decision", "reason")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tasks", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--tasks-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--fallback-plan", type=Path, action="append", default=[])
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_status(task_dir: Path) -> str:
    result = task_dir / "batch_results.csv"
    if not result.exists():
        return "no_batch_result"
    rows = read_rows(result)
    return rows[0].get("eval_status", "") if rows else "empty_batch_result"


def has_smoke_summary(task_dir: Path) -> bool:
    return next(task_dir.glob("eval_results/_smoke/**/summary.json"), None) is not None


def has_cuda_oom(task_dir: Path) -> bool:
    for path in task_dir.glob("**/*smoke.stderr.txt"):
        if "CUDA out of memory" in path.read_text(errors="replace"):
            return True
    return False


def plan_key(plan: dict) -> tuple[str, str]:
    return plan.get("dataset", {}).get("dataset", ""), plan.get("model", {}).get("model", "")


def fallback_plan_index(paths: list[Path]) -> dict[tuple[str, str], Path]:
    index = {}
    for path in paths:
        for plan in json.loads(path.read_text(encoding="utf-8")):
            index[plan_key(plan)] = path
    return index


def find_plan(
    task_dir: Path, dataset: str, model: str, fallback_plans: dict[tuple[str, str], Path]
) -> Path | None:
    plans = sorted(task_dir.glob("**/plans.json"))
    if len(plans) == 1:
        return plans[0]
    return fallback_plans.get((dataset, model))


def source_index(row: dict[str, str]) -> int:
    return int(row["task_index"])


def audit_row(index: int, row: dict[str, str], decision: str, reason: str) -> dict[str, str]:
    return {
        "source_task_index": str(index),
        "dataset": row["query_dataset"],
        "model": row["model_name"],
        "decision": decision,
        "reason": reason,
    }


def select_tasks(
    source_rows: list[dict[str, str]],
    prior_run: Path,
    fallback_plans: dict[tuple[str, str], Path] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected = []
    audit = []
    fallback_plans = fallback_plans or {}
    for row in source_rows:
        index = source_index(row)
        task_dir = prior_run / f"task_{index:03d}"
        status = task_status(task_dir)
        if status != "missing":
            audit.append(audit_row(index, row, "skip", f"status={status}"))
            continue
        if has_smoke_summary(task_dir):
            audit.append(audit_row(index, row, "skip", "smoke summary already exists"))
            continue
        if has_cuda_oom(task_dir):
            audit.append(audit_row(index, row, "skip", "CUDA OOM excluded"))
            continue
        plan = find_plan(
            task_dir, row["query_dataset"], row["model_name"], fallback_plans
        )
        if plan is None:
            audit.append(audit_row(index, row, "skip", "expected exactly one saved plans.json"))
            continue
        selected.append({
            "task_index": str(len(selected)),
            "source_task_index": str(index),
            "dataset": row["query_dataset"],
            "model": row["model_name"],
            "plan_file": str(plan),
        })
        audit.append(audit_row(index, row, "run", "missing non-OOM smoke result"))
    return selected, audit


def main() -> None:
    args = parse_args()
    selected, audit = select_tasks(
        read_rows(args.source_tasks), args.prior_run, fallback_plan_index(args.fallback_plan)
    )
    write_rows(args.tasks_output, selected, TASK_FIELDS)
    write_rows(args.audit_output, audit, AUDIT_FIELDS)
    excluded_oom = sum(row["reason"] == "CUDA OOM excluded" for row in audit)
    recovered = sum(row["reason"] == "smoke summary already exists" for row in audit)
    print(f"Smoke tasks: {len(selected)}")
    print(f"Excluded CUDA OOM: {excluded_oom}")
    print(f"Already smoke-recovered: {recovered}")


if __name__ == "__main__":
    main()

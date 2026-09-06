#!/usr/bin/env python3
"""Collect ranked-pair full evaluation results into one CSV and summary.

Example:
  python collect_oren_ranked_full.py     --tasks-csv config/oren62_ranked97_full_tasks.csv     --run-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_oren62_ranked97_full5000/RUN_ID     --csv-output /tmp/ranked_pairs_full_results.csv     --summary-output /tmp/run_summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-csv", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_batch(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def collected_row(task: dict[str, str], batch: dict[str, str]) -> dict[str, str]:
    if not batch:
        result = {
            "full_status": "missing",
            "full_score": "",
            "full_metric": "",
            "full_evaluation_protocol": "",
            "full_total": "",
            "full_labeled_total": "",
            "full_output_dir": "",
            "failure_reason": "batch_results.csv is missing",
        }
    else:
        result = {
            "full_status": batch.get("eval_status", ""),
            "full_score": batch.get("eval_score", ""),
            "full_metric": batch.get("eval_metric", ""),
            "full_evaluation_protocol": batch.get("eval_protocol", ""),
            "full_total": batch.get("eval_total", ""),
            "full_labeled_total": batch.get("eval_labeled_total", ""),
            "full_output_dir": batch.get("eval_output_dir", ""),
            "failure_reason": batch.get("failure_reason", ""),
        }
    return {**task, **result}


def collect(tasks: list[dict[str, str]], run_root: Path) -> list[dict[str, str]]:
    rows = []
    for task in tasks:
        index = int(task["task_index"])
        batch = read_batch(run_root / f"task_{index:03d}" / "batch_results.csv")
        rows.append(collected_row(task, batch))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["full_status"] for row in rows)
    payload = {"total": len(rows), "status_counts": dict(sorted(counts.items()))}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = collect(read_rows(args.tasks_csv), args.run_root)
    write_csv(args.csv_output, rows)
    write_summary(args.summary_output, rows)
    print(f"Collected rows: {len(rows)}")
    print(f"Results CSV: {args.csv_output}")
    print(f"Summary: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

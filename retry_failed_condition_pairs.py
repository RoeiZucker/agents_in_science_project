#!/usr/bin/env python3
"""Retry failed dataset/model pairs from a condition-run report.

Examples:
  # Smoke-retry every failed pair from a previous full run, reusing Codex context files.
  python retry_failed_condition_pairs.py \
    --report-csv /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real/condition_run_report.csv \
    --source-run-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real \
    --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
    --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_retry_failed_smoke \
    --hf-home /sci/labs/michall/roeizucker/hf_eval_runtime/.hf_cache \
    --stage smoke \
    --dataset ethz/food101 \
    --dataset-timeout 900 \
    --trust-remote-code

  # Full-retry failed pairs after smoke looks stable.
  python retry_failed_condition_pairs.py \
    --report-csv /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real/condition_run_report.csv \
    --source-run-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real \
    --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
    --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_retry_failed_full \
    --hf-home /sci/labs/michall/roeizucker/hf_eval_runtime/.hf_cache \
    --stage full \
    --full-limit 1000 \
    --trust-remote-code

  Encoder/classifier direct scoring is automatic; --allow-label-scores remains optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_agent_core import safe_name
from result_contract import result_outcome

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_EVAL_AGENT = SCRIPT_DIR / "run_eval_agent.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-csv", required=True, type=Path)
    parser.add_argument("--source-run-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="smoke")
    parser.add_argument("--full-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-label-scores", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--dataset-timeout", type=int, default=0, help="Optional timeout in seconds for each dataset group retry.")
    parser.add_argument("--dataset", action="append", default=[], help="Retry only these dataset ids from the report.")
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = failed_models_by_dataset(args.report_csv)
    if args.dataset:
        grouped = {key: value for key, value in grouped.items() if key in set(args.dataset)}
    if args.limit_datasets:
        grouped = dict(list(grouped.items())[: args.limit_datasets])
    results = retry_groups(args, grouped)
    write_json(args.output_root / "retry_failed_summary.json", results)
    write_retry_csv(args.output_root / "retry_failed_results.csv", results)
    counts = summary(results)
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    if counts["failed_pairs"] or counts["failed_processes"]:
        raise SystemExit(1)


def failed_models_by_dataset(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not is_failed(row):
                continue
            dataset = row.get("query_dataset") or row.get("dataset")
            model = row.get("model_name") or row.get("model")
            if dataset and model and model not in grouped[dataset]:
                grouped[dataset].append(model)
    return dict(grouped)


def is_failed(row: dict[str, str]) -> bool:
    status = row.get("status") or row.get("eval_status")
    if status:
        score = row.get("score") or row.get("eval_score")
        labeled = row.get("eval_labeled_total") or row.get("labeled_total")
        return result_outcome(status, score, labeled) == "failure"
    return row.get("outcome", "").lower() == "failure"


def retry_groups(args: argparse.Namespace, grouped: dict[str, list[str]]) -> list[dict[str, object]]:
    results = []
    total = len(grouped)
    for index, (dataset, models) in enumerate(grouped.items(), start=1):
        print(f"[retry] dataset {index}/{total}: {dataset} ({len(models)} models)", flush=True)
        result = run_dataset_retry(args, dataset, models)
        results.append(result)
        if retry_failed(result) and args.stop_on_error:
            break
    return results


def run_dataset_retry(args: argparse.Namespace, dataset: str, models: list[str]) -> dict[str, object]:
    command = build_command(args, dataset, models)
    try:
        completed = subprocess.run(command, text=True, timeout=args.dataset_timeout or None)
        pairs = read_retry_results(args, dataset, models)
        return {"dataset": dataset, "models": models, "returncode": completed.returncode, "command": command, "pair_results": pairs}
    except subprocess.TimeoutExpired:
        print(f"[retry] dataset timed out after {args.dataset_timeout}s: {dataset}", flush=True)
        pairs = [missing_pair(dataset, model, "timeout") for model in models]
        return {"dataset": dataset, "models": models, "returncode": 124, "command": command, "timed_out": True, "pair_results": pairs}


def read_retry_results(args: argparse.Namespace, dataset: str, models: list[str]) -> list[dict[str, Any]]:
    path = args.output_root / f"{safe_name(dataset)}_{args.split}" / "results.csv"
    rows = read_csv(path)
    by_model = {row.get("model"): row for row in rows}
    return [pair_result(dataset, model, by_model.get(model)) for model in models]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pair_result(dataset: str, model: str, row: dict[str, str] | None) -> dict[str, Any]:
    if row is None:
        return missing_pair(dataset, model, "results.csv has no row for this pair")
    outcome = result_outcome(row.get("status"), row.get("score"), row.get("labeled_total"))
    return {
        "dataset": dataset,
        "model": model,
        "outcome": outcome,
        "status": row.get("status"),
        "evaluation_protocol": row.get("evaluation_protocol"),
        "metric": row.get("metric"),
        "score": row.get("score"),
        "total": row.get("total"),
        "labeled_total": row.get("labeled_total"),
        "notes": row.get("notes"),
        "output_dir": row.get("output_dir"),
    }


def missing_pair(dataset: str, model: str, reason: str) -> dict[str, Any]:
    return {"dataset": dataset, "model": model, "outcome": "failure", "status": "missing", "notes": reason}


def retry_failed(result: dict[str, object]) -> bool:
    return bool(result.get("returncode")) or any(
        pair.get("outcome") == "failure" for pair in result.get("pair_results", [])
    )


def build_command(args: argparse.Namespace, dataset: str, models: list[str]) -> list[str]:
    command = [
        args.python,
        str(RUN_EVAL_AGENT),
        "--dataset",
        dataset,
        "--models",
        *models,
        "--split",
        args.split,
        "--stage",
        args.stage,
        "--smoke-limit",
        str(args.smoke_limit),
        "--full-limit",
        str(args.full_limit),
        "--seed",
        str(args.seed),
        "--sample-size",
        str(args.sample_size),
        "--project-root",
        str(args.project_root),
        "--output-root",
        str(args.output_root),
        "--python",
        args.python,
        "--continue-on-error",
        "--context-file",
        str(context_path(args.source_run_root, dataset, args.split)),
    ]
    if args.hf_home:
        command.extend(["--hf-home", str(args.hf_home)])
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if getattr(args, "allow_label_scores", False):
        command.append("--allow-label-scores")
    return command


def context_path(root: Path, dataset: str, split: str) -> Path:
    return root / f"{safe_name(dataset)}_{split}" / "codex_context.json"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def all_pair_results(results: list[dict[str, object]]) -> list[dict[str, Any]]:
    return [pair for result in results for pair in result.get("pair_results", [])]


def write_retry_csv(path: Path, results: list[dict[str, object]]) -> None:
    rows = all_pair_results(results)
    fields = ["dataset", "model", "outcome", "status", "evaluation_protocol", "metric", "score", "total", "labeled_total", "notes", "output_dir"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary(results: list[dict[str, object]]) -> dict[str, int]:
    pairs = all_pair_results(results)
    return {
        "datasets": len(results),
        "pairs": len(pairs),
        "successful_pairs": sum(1 for pair in pairs if pair.get("outcome") == "success"),
        "failed_pairs": sum(1 for pair in pairs if pair.get("outcome") == "failure"),
        "failed_processes": sum(1 for item in results if item.get("returncode") != 0 and not item.get("pair_results")),
    }


if __name__ == "__main__":
    main()

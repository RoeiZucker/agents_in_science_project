#!/usr/bin/env python3
"""Create a one-row-per-dataset CSV from collected array results.

Example:
    python create_dataset_level_array_report.py \
      --array-results /path/to/run/array_results.csv \
      --output /path/to/run/array_results_by_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


SUCCESS_STATUSES = {"ok", "success"}
MODEL_FIELDS = (
    "model",
    "conditions",
    "source_methods",
    "recommendation_scores",
    "recommendation_reasoning",
    "status",
    "protocol",
    "metric",
    "score",
    "eval_total",
    "eval_labeled_total",
    "codex_prompt",
    "codex_context",
    "model_context",
    "eval_notes",
    "failure_reason",
    "eval_output_dir",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--failures-json", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def flatten(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s*\r?\n+\s*", "; ", text)
    return re.sub(r"[ \t]+", " ", text).strip(" ;")


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def condition_name(row: dict[str, str]) -> str:
    condition = row.get("condition", "")
    rank = row.get("rank", "")
    return f"{condition}#{rank}" if rank else condition


def group_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("query_dataset", ""), row.get("model_name", ""))].append(row)
    return [merge_pair(items) for items in grouped.values()]


def merge_pair(rows: list[dict[str, str]]) -> dict[str, str]:
    merged = max(rows, key=result_completeness).copy()
    merged["conditions"] = "; ".join(condition_name(row) for row in rows)
    merged["source_methods"] = "; ".join(unique([row.get("source_method", "") for row in rows]))
    merged["recommendation_scores"] = "; ".join(
        f"{condition_name(row)}={row.get('recommendation_score', '')}" for row in rows
    )
    merged["recommendation_reasoning"] = "; ".join(
        f"{condition_name(row)}: {flatten(row.get('reasoning', ''))}" for row in rows
    )
    return merged


def result_completeness(row: dict[str, str]) -> int:
    fields = ("eval_status", "eval_score", "eval_notes", "eval_output_dir")
    return sum(bool(row.get(field)) for field in fields)


def load_context(array_root: Path, task_index: str) -> dict[str, Any]:
    if not task_index:
        return {}
    task_dir = array_root / f"task_{int(task_index):03d}"
    paths = sorted(task_dir.glob("*/codex_context.json"))
    return read_json(paths[0]) if paths else {}


def index_failures(path: Path) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for failure in read_json(path):
        if "task_index" in failure:
            indexed[int(failure["task_index"])].append(failure)
    return indexed


def audit_messages(failure: dict[str, Any]) -> list[str]:
    audit = failure.get("audit")
    if not isinstance(audit, dict):
        return []
    messages = list(audit.get("issues") or [])
    messages.extend(audit.get("warnings") or [])
    return [flatten(message) for message in messages if message]


def failure_reason(row: dict[str, str], failures: list[dict[str, Any]]) -> str:
    if row.get("eval_status", "") in SUCCESS_STATUSES:
        return ""
    messages = []
    for failure in failures:
        messages.extend(audit_messages(failure))
        for field in ("reason", "error", "message"):
            if failure.get(field):
                messages.append(flatten(failure[field]))
    if messages:
        return "; ".join(unique(messages))
    if row.get("eval_notes"):
        return flatten(row["eval_notes"])
    return "No task result was collected."


def context_fields(context: dict[str, Any]) -> tuple[str, str, str]:
    dataset_context = context.get("dataset_context") or {}
    model_context = context.get("model_context") or {}
    return (
        flatten(dataset_context.get("prompt_template", "")),
        flatten(dataset_context.get("notes", "")),
        flatten(model_context.get("notes", "")),
    )


def model_values(
    pair: dict[str, str], array_root: Path, failures: dict[int, list[dict[str, Any]]]
) -> dict[str, str]:
    task_index = int(pair["task_index"]) if pair.get("task_index") else -1
    prompt, context_notes, model_notes = context_fields(load_context(array_root, pair.get("task_index", "")))
    return {
        "model": pair.get("model_name", ""),
        "conditions": pair.get("conditions", ""),
        "source_methods": pair.get("source_methods", ""),
        "recommendation_scores": pair.get("recommendation_scores", ""),
        "recommendation_reasoning": pair.get("recommendation_reasoning", ""),
        "status": pair.get("eval_status", ""),
        "protocol": pair.get("eval_protocol", ""),
        "metric": pair.get("eval_metric", ""),
        "score": pair.get("eval_score", ""),
        "eval_total": pair.get("eval_total", ""),
        "eval_labeled_total": pair.get("eval_labeled_total", ""),
        "codex_prompt": prompt,
        "codex_context": context_notes,
        "model_context": model_notes,
        "eval_notes": flatten(pair.get("eval_notes", "")),
        "failure_reason": failure_reason(pair, failures.get(task_index, [])),
        "eval_output_dir": pair.get("eval_output_dir", ""),
    }


def status_counts(pairs: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        counts[pair.get("eval_status", "missing") or "missing"] += 1
    return counts


def dataset_row(
    dataset: str,
    pairs: list[dict[str, str]],
    array_root: Path,
    failures: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    pairs.sort(key=lambda pair: int(pair.get("task_index") or 10**9))
    counts = status_counts(pairs)
    row: dict[str, Any] = {
        "dataset": dataset,
        "model_count": len(pairs),
        "success_count": sum(counts.get(status, 0) for status in SUCCESS_STATUSES),
        "failed_count": counts.get("failed", 0),
        "unsupported_count": counts.get("unsupported", 0),
        "missing_count": sum(count for status, count in counts.items() if status.startswith("missing")),
    }
    for number, pair in enumerate(pairs, 1):
        for field, value in model_values(pair, array_root, failures).items():
            row[f"model_{number}_{field}"] = value
    return row


def build_rows(
    pairs: list[dict[str, str]],
    array_root: Path,
    failures: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.get("query_dataset", "")].append(pair)
    return [dataset_row(dataset, grouped[dataset], array_root, failures) for dataset in sorted(grouped)]


def fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    summary = ["dataset", "model_count", "success_count", "failed_count", "unsupported_count", "missing_count"]
    max_models = max(int(row["model_count"]) for row in rows)
    model_columns = [f"model_{number}_{field}" for number in range(1, max_models + 1) for field in MODEL_FIELDS]
    return summary + model_columns


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    array_root = args.array_results.parent
    failures_path = args.failures_json or array_root / "array_failures.json"
    pairs = group_pairs(read_csv(args.array_results))
    rows = build_rows(pairs, array_root, index_failures(failures_path))
    write_report(args.output, rows)
    print(json.dumps({"output": str(args.output), "datasets": len(rows), "dataset_model_pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()

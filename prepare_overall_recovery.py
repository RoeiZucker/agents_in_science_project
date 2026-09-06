#!/usr/bin/env python3
"""Prepare broad recovery manifests from the consolidated pair audit.

Example:
  python prepare_overall_recovery.py \
    --pairs-csv /path/to/all_results_pairs.csv \
    --conditions-csv ../evaluation_conditions.csv \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --explicit-template-csv config/max_coverage_recovery_tasks.csv \
    --audit-output config/overall_recovery_audit.csv \
    --generic-output config/overall_recovery_generic_tasks.csv \
    --explicit-output config/overall_recovery_explicit_tasks.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


ACTIONABLE = {
    "dataset_context",
    "missing_output",
    "never_attempted",
    "planned_only",
    "runtime_failure",
    "stale_adapter",
}
BLOCKED_DATASETS = {"bigbio/n2c2_2018_track2"}
AUDIT_FIELDS = (
    "dataset",
    "model",
    "status",
    "category",
    "route",
    "conditions",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--conditions-csv", type=Path, action="append", required=True)
    parser.add_argument("--explicit-template-csv", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--generic-output", type=Path, required=True)
    parser.add_argument("--explicit-output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    dataset = row.get("query_dataset") or row.get("dataset") or ""
    model = row.get("model_name") or row.get("model") or ""
    return dataset, model


def condition_index(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    indexed = {}
    for path in paths:
        for row in read_csv(path):
            indexed.setdefault(pair_key(row), row)
    return indexed


def explicit_templates(path: Path) -> dict[str, dict[str, str]]:
    return {row["dataset"]: row for row in read_csv(path)}


def selected_source_row(
    pair: dict[str, str], cache: dict[Path, list[dict[str, str]]]
) -> dict[str, str]:
    source = pair.get("source_results_csv", "")
    if not source:
        return {}
    path = Path(source)
    if path not in cache:
        try:
            cache[path] = read_csv(path)
        except OSError:
            cache[path] = []
    key = pair_key(pair)
    return next((row for row in cache[path] if pair_key(row) == key), {})


def failure_reason(pair: dict[str, str], source: dict[str, str]) -> str:
    fields = ("failure_reason", "error", "eval_notes", "notes")
    return next(
        (source.get(field, "").strip() for field in fields if source.get(field)), ""
    )


def failure_category(status: str, reason: str) -> str:
    if status == "untried":
        return "never_attempted"
    if status == "planned":
        return "planned_only"
    if status == "missing":
        return "missing_output"
    if status == "failed":
        return "runtime_failure"
    if status == "unsupported" and "no direct adapter exists" in reason:
        return "stale_adapter"
    if status == "unsupported" and "no detected gold labels" in reason:
        return "dataset_context"
    return "incompatible"


def recovery_route(
    dataset: str,
    model: str,
    category: str,
    templates: dict[str, dict[str, str]],
) -> str:
    if dataset in BLOCKED_DATASETS:
        return "blocked_remote_code"
    if category not in ACTIONABLE:
        return "skip_incompatible"
    if dataset in templates:
        if model == templates[dataset]["model"]:
            return "active_coverage_run"
        return "explicit_source_override"
    return "generic_retry"


def audit_rows(
    pairs: list[dict[str, str]], templates: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    rows = []
    cache: dict[Path, list[dict[str, str]]] = {}
    for pair in pairs:
        if pair.get("outcome") == "success":
            continue
        source = selected_source_row(pair, cache)
        reason = failure_reason(pair, source)
        status = (pair.get("eval_status") or "untried").lower()
        category = failure_category(status, reason)
        dataset, model = pair_key(pair)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "status": status,
                "category": category,
                "route": recovery_route(dataset, model, category, templates),
                "conditions": pair.get("conditions", ""),
                "reason": reason,
            }
        )
    return rows


def generic_tasks(
    audit: list[dict[str, str]], conditions: dict[tuple[str, str], dict[str, str]]
) -> list[dict[str, str]]:
    tasks = []
    for row in audit:
        if row["route"] != "generic_retry":
            continue
        condition = conditions.get((row["dataset"], row["model"]))
        if condition:
            tasks.append({**condition, "task_index": str(len(tasks))})
    return tasks


def explicit_tasks(
    audit: list[dict[str, str]], templates: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    tasks = []
    for row in audit:
        if row["route"] != "explicit_source_override":
            continue
        template = dict(templates[row["dataset"]])
        template["model"] = row["model"]
        template["task_index"] = str(len(tasks))
        tasks.append(template)
    return tasks


def write_csv(path: Path, rows: list[dict[str, str]], fields=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(audit, generic, explicit) -> None:
    print(f"Unresolved pairs: {len(audit)}")
    print(f"Generic retries: {len(generic)}")
    print(f"Explicit source overrides: {len(explicit)}")
    print("Routes: " + repr(dict(Counter(row["route"] for row in audit))))
    print("Categories: " + repr(dict(Counter(row["category"] for row in audit))))


def main() -> None:
    args = parse_args()
    templates = explicit_templates(args.explicit_template_csv)
    audit = audit_rows(read_csv(args.pairs_csv), templates)
    conditions = condition_index(args.conditions_csv)
    generic = generic_tasks(audit, conditions)
    explicit = explicit_tasks(audit, templates)
    write_csv(args.audit_output, audit, AUDIT_FIELDS)
    write_csv(args.generic_output, generic)
    write_csv(args.explicit_output, explicit)
    print_summary(audit, generic, explicit)


if __name__ == "__main__":
    main()

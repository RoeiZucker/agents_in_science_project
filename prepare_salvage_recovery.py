#!/usr/bin/env python3
"""Prepare the high-priority recovery campaign from current consolidated pairs.

Example:
  python prepare_salvage_recovery.py \
    --pairs-csv /tmp/current_all_results_pairs.csv \
    --conditions-csv ../evaluation_conditions.csv \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --explicit-template-csv config/max_coverage_recovery_tasks.csv \
    --audit-output config/salvage_recovery_audit.csv \
    --generic-output config/salvage_recovery_generic_tasks.csv \
    --explicit-output config/salvage_recovery_explicit_tasks.csv \
    --blocked-output config/salvage_recovery_blocked_tasks.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from prepare_overall_recovery import (
    AUDIT_FIELDS,
    audit_rows,
    condition_index,
    explicit_templates,
    read_csv,
    write_csv,
)


SALVAGE_CATEGORIES = {
    "dataset_context",
    "missing_output",
    "never_attempted",
    "runtime_failure",
}
BLOCKED_ROUTE = "blocked_remote_code"
EXPLICIT_ROUTES = {"active_coverage_run", "explicit_source_override"}
SALVAGE_AUDIT_FIELDS = (*AUDIT_FIELDS, "selected", "salvage_lane")
BLOCKED_FIELDS = (
    "dataset",
    "model",
    "status",
    "category",
    "conditions",
    "reason",
    "task_index",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--conditions-csv", type=Path, action="append", required=True)
    parser.add_argument("--explicit-template-csv", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--generic-output", type=Path, required=True)
    parser.add_argument("--explicit-output", type=Path, required=True)
    parser.add_argument("--blocked-output", type=Path, required=True)
    return parser.parse_args()


def salvage_lane(row: dict[str, str]) -> str:
    if row["category"] not in SALVAGE_CATEGORIES:
        return "deferred"
    if row["route"] == BLOCKED_ROUTE:
        return "blocked"
    if row["route"] in EXPLICIT_ROUTES:
        return "explicit"
    if row["route"] == "generic_retry":
        return "generic"
    return "deferred"


def annotate_audit(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    annotated = []
    for row in rows:
        lane = salvage_lane(row)
        annotated.append(
            {
                **row,
                "selected": "yes" if lane != "deferred" else "no",
                "salvage_lane": lane,
            }
        )
    return annotated


def generic_tasks(
    audit: list[dict[str, str]],
    conditions: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, str]]:
    tasks = []
    for row in audit:
        if row["salvage_lane"] != "generic":
            continue
        condition = conditions.get((row["dataset"], row["model"]))
        if condition is None:
            continue
        tasks.append({**condition, "task_index": str(len(tasks))})
    return tasks


def explicit_tasks(
    audit: list[dict[str, str]], templates: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    tasks = []
    for row in audit:
        if row["salvage_lane"] != "explicit":
            continue
        template = dict(templates[row["dataset"]])
        template["model"] = row["model"]
        template["task_index"] = str(len(tasks))
        tasks.append(template)
    return tasks


def blocked_tasks(audit: list[dict[str, str]]) -> list[dict[str, str]]:
    tasks = []
    for row in audit:
        if row["salvage_lane"] != "blocked":
            continue
        tasks.append(
            {
                key: row.get(key, "")
                for key in BLOCKED_FIELDS
                if key != "task_index"
            }
            | {"task_index": str(len(tasks))}
        )
    return tasks


def output_fields(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def print_summary(
    audit: list[dict[str, str]],
    generic: list[dict[str, str]],
    explicit: list[dict[str, str]],
    blocked: list[dict[str, str]],
) -> None:
    selected = [row for row in audit if row["selected"] == "yes"]
    print(f"Unresolved pairs: {len(audit)}")
    print(f"Selected salvage pairs: {len(selected)}")
    print(f"Generic tasks: {len(generic)}")
    print(f"Explicit tasks: {len(explicit)}")
    print(f"Blocked tasks: {len(blocked)}")
    print("Categories: " + repr(dict(Counter(row["category"] for row in selected))))


def main() -> None:
    args = parse_args()
    templates = explicit_templates(args.explicit_template_csv)
    base_audit = audit_rows(read_csv(args.pairs_csv), templates)
    audit = annotate_audit(base_audit)
    conditions = condition_index(args.conditions_csv)
    generic = generic_tasks(audit, conditions)
    explicit = explicit_tasks(audit, templates)
    blocked = blocked_tasks(audit)
    write_csv(args.audit_output, audit, SALVAGE_AUDIT_FIELDS)
    write_csv(args.generic_output, generic)
    write_csv(
        args.explicit_output,
        explicit,
        output_fields(args.explicit_template_csv) + ["task_index"],
    )
    write_csv(args.blocked_output, blocked, BLOCKED_FIELDS)
    print_summary(audit, generic, explicit, blocked)


if __name__ == "__main__":
    main()

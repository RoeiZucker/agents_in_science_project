#!/usr/bin/env python3
"""Consolidate historical full evaluations and compare our methods with Claude.

Example:
  python build_all_results_method_comparison.py \
    --conditions-csv ../evaluation_conditions.csv \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --results-root /path/to/eval_results \
    --pairs-output /path/to/all_results_pairs.csv \
    --csv-output /path/to/method_vs_claude_all_results.csv \
    --markdown-output /path/to/method_vs_claude_all_results.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from compare_methods_to_claude import build_comparisons, write_csv, write_markdown


SUCCESS_STATUSES = {"ok", "success", "succeeded", "warn"}
RESULT_FILENAMES = {"batch_results.csv", "results.csv"}
SUMMARY_METRICS = (
    "accuracy", "macro_f1", "f1", "qa_f1", "rouge_l", "numeric_match", "set_f1"
)
PAIR_FIELDS = [
    "dataset",
    "model",
    "score",
    "outcome",
    "conditions",
    "eval_status",
    "eval_protocol",
    "eval_metric",
    "eval_total",
    "eval_labeled_total",
    "eval_output_dir",
    "source_results_csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path, action="append", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    dataset = row.get("query_dataset") or row.get("dataset") or ""
    model = row.get("model_name") or row.get("model") or ""
    return dataset, model


def condition_text(row: dict[str, str]) -> str:
    condition = row.get("condition", "")
    rank = row.get("rank", "")
    return f"{condition}#{rank}" if rank else condition


def load_conditions(paths: list[Path]) -> dict[tuple[str, str], list[str]]:
    conditions: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in paths:
        for row in read_csv(path):
            value = condition_text(row)
            if value and value not in conditions[pair_key(row)]:
                conditions[pair_key(row)].append(value)
    return dict(conditions)


def is_smoke(row: dict[str, str], source: Path) -> bool:
    output = row.get("eval_output_dir") or row.get("output_dir") or ""
    return "/_smoke/" in output.lower() or "smoke" in str(source).lower()


def numeric_value(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def is_measurement(row: dict[str, str], source: Path) -> bool:
    status = (row.get("eval_status") or row.get("status") or "").lower()
    score = numeric_value(row.get("eval_score") or row.get("score"))
    labeled = numeric_value(row.get("eval_labeled_total") or row.get("labeled_total"))
    return (
        not is_smoke(row, source)
        and status in SUCCESS_STATUSES
        and score is not None
        and (labeled is None or labeled > 0)
    )


def discover_results(root: Path) -> dict[tuple[str, str], list[tuple[dict[str, str], Path]]]:
    results: dict[tuple[str, str], list[tuple[dict[str, str], Path]]] = defaultdict(list)
    for source in root.rglob("*.csv"):
        if source.name not in RESULT_FILENAMES:
            continue
        for row in read_csv(source):
            key = pair_key(row)
            if all(key):
                results[key].append((row, source))
    return results


def result_priority(candidate: tuple[dict[str, str], Path]) -> tuple[int, float]:
    row, source = candidate
    return int(is_measurement(row, source)), source.stat().st_mtime


def select_result(
    candidates: list[tuple[dict[str, str], Path]],
) -> tuple[dict[str, str], Path] | None:
    full = [candidate for candidate in candidates if not is_smoke(*candidate)]
    return max(full, key=result_priority, default=None)


def summary_metric(row: dict[str, str]) -> str:
    output = row.get("eval_output_dir") or row.get("output_dir") or ""
    if not output:
        return ""
    try:
        summary = json.loads(
            (Path(output) / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return ""
    declared = summary.get("metric") or summary.get("evaluation_method")
    if isinstance(declared, str) and declared:
        return declared
    score = numeric_value(row.get("eval_score") or row.get("score"))
    matches = [
        name
        for name in SUMMARY_METRICS
        if score is not None
        and numeric_value(summary.get(name)) is not None
        and math.isclose(float(summary[name]), score, rel_tol=1e-9, abs_tol=1e-12)
    ]
    return matches[0] if len(matches) == 1 else ""


def normalized_pair(
    key: tuple[str, str],
    conditions: list[str],
    selected: tuple[dict[str, str], Path] | None,
) -> dict[str, str]:
    dataset, model = key
    row, source = selected if selected else ({}, Path())
    measured = bool(selected and is_measurement(row, source))
    return {
        "dataset": dataset,
        "model": model,
        "score": (row.get("eval_score") or row.get("score") or "") if measured else "",
        "outcome": "success" if measured else "failure",
        "conditions": ", ".join(conditions),
        "eval_status": row.get("eval_status") or row.get("status") or "untried",
        "eval_protocol": row.get("eval_protocol") or row.get("evaluation_protocol", ""),
        "eval_metric": row.get("eval_metric") or summary_metric(row),
        "eval_total": row.get("eval_total") or row.get("total") or "",
        "eval_labeled_total": row.get("eval_labeled_total") or row.get("labeled_total") or "",
        "eval_output_dir": row.get("eval_output_dir") or row.get("output_dir") or "",
        "source_results_csv": str(source) if selected else "",
    }


def build_pairs(
    conditions: dict[tuple[str, str], list[str]],
    results: dict[tuple[str, str], list[tuple[dict[str, str], Path]]],
) -> list[dict[str, str]]:
    return [
        normalized_pair(key, memberships, select_result(results.get(key, [])))
        for key, memberships in conditions.items()
    ]


def write_pairs(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    conditions = load_conditions(args.conditions_csv)
    results = discover_results(args.results_root)
    pairs = build_pairs(conditions, results)
    comparisons = build_comparisons(pairs)
    write_pairs(args.pairs_output, pairs)
    write_csv(args.csv_output, comparisons)
    write_markdown(args.markdown_output, comparisons)
    measured = sum(row["outcome"] == "success" for row in pairs)
    print(f"Wrote {len(comparisons)} comparisons from {measured}/{len(pairs)} measured pairs.")


if __name__ == "__main__":
    main()

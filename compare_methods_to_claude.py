#!/usr/bin/env python3
"""Compare method1/method2 evaluation results with the Claude baseline.

Example:
  python compare_methods_to_claude.py \
    --input condition_run_report.csv \
    --csv-output method_vs_claude.csv \
    --markdown-output method_vs_claude.md

  python compare_methods_to_claude.py \
    --input /path/to/array_results.csv \
    --csv-output /path/to/method_vs_claude.csv \
    --markdown-output /path/to/method_vs_claude.md
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


CONDITION_PATTERN = re.compile(r"([A-Za-z0-9_]+)#(\d+)")
OUR_CONDITIONS = {"B_method1_only", "C_method2_only"}
BASELINE_CONDITIONS = {"D_claude_baseline"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--csv-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return normalize_rows(list(csv.DictReader(handle)))


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows or "query_dataset" not in rows[0]:
        return rows
    return normalize_array_rows(rows)


def normalize_array_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("query_dataset", ""), row.get("model_name", ""))].append(row)
    return [normalize_array_pair(pair_rows) for pair_rows in grouped.values()]


def normalize_array_pair(rows: list[dict[str, str]]) -> dict[str, str]:
    result = max(rows, key=array_result_priority)
    return {
        "dataset": result.get("query_dataset", ""),
        "model": result.get("model_name", ""),
        "score": result.get("eval_score", ""),
        "eval_metric": result.get("eval_metric", ""),
        "outcome": (
            "success"
            if result.get("eval_status", "") in {"ok", "success"}
            else "failure"
        ),
        "conditions": ", ".join(array_condition(row) for row in rows),
    }


def array_result_priority(row: dict[str, str]) -> tuple[int, int]:
    success = int(row.get("eval_status", "") in {"ok", "success"})
    measured = int(bool(row.get("eval_score", "")))
    return success, measured


def array_condition(row: dict[str, str]) -> str:
    condition = row.get("condition", "")
    rank = row.get("rank", "")
    return f"{condition}#{rank}" if rank else condition


def conditions(row: dict[str, str]) -> list[tuple[str, int]]:
    return [(name, int(rank)) for name, rank in CONDITION_PATTERN.findall(row.get("conditions", ""))]


def belongs_to(row: dict[str, str], names: set[str], max_rank: int) -> bool:
    return any(name in names and rank <= max_rank for name, rank in conditions(row))


def numeric_score(row: dict[str, str]) -> float | None:
    if row.get("outcome") != "success":
        return None
    try:
        score = float(row.get("score", ""))
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def method_label(row: dict[str, str], max_rank: int) -> str:
    names = {
        name
        for name, rank in conditions(row)
        if name in OUR_CONDITIONS and rank <= max_rank
    }
    labels = []
    if "B_method1_only" in names:
        labels.append("method1")
    if "C_method2_only" in names:
        labels.append("method2")
    return "+".join(labels)


def candidate_pool(
    rows: list[dict[str, str]], names: set[str], max_rank: int
) -> list[dict[str, str]]:
    return [row for row in rows if belongs_to(row, names, max_rank)]


def best_candidate(rows: list[dict[str, str]]) -> dict[str, str] | None:
    measured = [row for row in rows if numeric_score(row) is not None]
    return max(measured, key=measured_score, default=None)


def measured_score(row: dict[str, str]) -> float:
    score = numeric_score(row)
    if score is None:
        raise ValueError("Expected a measured row")
    return score


def coverage(rows: list[dict[str, str]]) -> str:
    measured = sum(numeric_score(row) is not None for row in rows)
    return f"{measured}/{len(rows)}"


def winner(ours: dict[str, str] | None, baseline: dict[str, str] | None) -> str:
    if ours is None or baseline is None:
        return "incomplete"
    if not metrics_match(ours, baseline):
        return "incomplete"
    ours_score = numeric_score(ours)
    baseline_score = numeric_score(baseline)
    if math.isclose(ours_score or 0.0, baseline_score or 0.0, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    return "ours" if (ours_score or 0.0) > (baseline_score or 0.0) else "claude"


def metrics_match(ours: dict[str, str], baseline: dict[str, str]) -> bool:
    ours_metric = ours.get("eval_metric", "")
    baseline_metric = baseline.get("eval_metric", "")
    return not ours_metric or not baseline_metric or ours_metric == baseline_metric


def comparison(dataset: str, rows: list[dict[str, str]], max_rank: int) -> dict[str, str]:
    ours_pool = candidate_pool(rows, OUR_CONDITIONS, max_rank)
    baseline_pool = candidate_pool(rows, BASELINE_CONDITIONS, max_rank)
    ours = best_candidate(ours_pool)
    baseline = best_candidate(baseline_pool)
    return {
        "dataset": dataset,
        "comparison": "top1" if max_rank == 1 else "best_of_top5",
        "our_winner": method_label(ours, max_rank) if ours else "",
        "our_model": ours.get("model", "") if ours else "",
        "our_score": score_text(ours),
        "our_coverage": coverage(ours_pool),
        "claude_model": baseline.get("model", "") if baseline else "",
        "claude_score": score_text(baseline),
        "claude_coverage": coverage(baseline_pool),
        "winner": winner(ours, baseline),
    }


def score_text(row: dict[str, str] | None) -> str:
    score = numeric_score(row) if row else None
    return f"{score:.9g}" if score is not None else ""


def build_comparisons(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    datasets = list(dict.fromkeys(row.get("dataset", "") for row in rows))
    output = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row.get("dataset") == dataset]
        output.append(comparison(dataset, dataset_rows, 1))
        output.append(comparison(dataset, dataset_rows, 5))
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, str]], title: str) -> list[str]:
    lines = [f"## {title}", "", "| Dataset | Ours: method, model, score | Claude: model, score | Coverage (ours / Claude) | Winner |", "|---|---|---|---|---|"]
    for row in rows:
        ours = f"{row['our_winner']}; {row['our_model']}; {row['our_score']}" if row["our_model"] else "unavailable"
        claude = f"{row['claude_model']}; {row['claude_score']}" if row["claude_model"] else "unavailable"
        lines.append(
            f"| {row['dataset']} | {ours} | {claude} | "
            f"{row['our_coverage']} / {row['claude_coverage']} | {row['winner']} |"
        )
    return lines


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    top1 = [row for row in rows if row["comparison"] == "top1"]
    top5 = [row for row in rows if row["comparison"] == "best_of_top5"]
    lines = [
        "# Method vs Claude Baseline",
        "",
        "Higher evaluation scores are better. Missing evaluations are excluded; a comparison is incomplete when either side has no measured score or the selected scores use different known metrics.",
        "For our side, top1 means the better measured result between method1 rank 1 and method2 rank 1. Best-of-top5 means the best measured result from the union of both methods' top-five lists.",
        "",
    ]
    lines.extend(markdown_table(top1, "Top 1 Comparison"))
    lines.extend([""])
    lines.extend(markdown_table(top5, "Best of Top 5 Comparison"))
    lines.extend([
        "",
        "## Overall Outcomes",
        "",
        outcome_summary("Top 1", top1),
        "",
        outcome_summary("Best of top 5", top5),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def outcome_summary(label: str, rows: list[dict[str, str]]) -> str:
    counts = Counter(row["winner"] for row in rows)
    return (
        f"{label}: ours={counts['ours']}, claude={counts['claude']}, "
        f"ties={counts['tie']}, incomplete={counts['incomplete']}."
    )


def main() -> None:
    args = parse_args()
    comparisons = build_comparisons(read_rows(args.input))
    write_csv(args.csv_output, comparisons)
    write_markdown(args.markdown_output, comparisons)
    print(f"Wrote {len(comparisons)} comparisons to {args.csv_output} and {args.markdown_output}")


if __name__ == "__main__":
    main()

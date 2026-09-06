#!/usr/bin/env python3
"""Rescore saved classification predictions without rerunning inference.

Loop-results example:
  python rescore_loop_classification.py \
    --input oren15_all75_full5000_final_results.csv \
    --dataset fancyzhx/ag_news \
    --metric accuracy \
    --output oren15_all75_full5000_final_results_ag_news_accuracy.csv

Historical-pairs example:
  python rescore_loop_classification.py \
    --input all_results_pairs.csv \
    --dataset google-research-datasets/poem_sentiment \
    --model answerdotai/ModernBERT-Large-Instruct \
    --metric macro_f1 \
    --output all_results_pairs_poem_sentiment_macro_f1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROVENANCE_FIELDS = ("rescore_source_metric", "rescore_method")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model")
    parser.add_argument("--metric", choices=("accuracy", "macro_f1"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_predictions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prediction_is_correct(record: dict) -> bool:
    correct = record.get("correct")
    if isinstance(correct, bool):
        return correct
    return record.get("prediction") == record.get("target_answer")


def accuracy_from_predictions(path: Path) -> tuple[float, int]:
    records = read_predictions(path)
    if not records:
        raise ValueError(f"No predictions found in {path}")
    correct = sum(prediction_is_correct(record) for record in records)
    return correct / len(records), len(records)


def label_f1(records: list[dict], label: str) -> float:
    true_positive = sum(
        row.get("target_answer") == label and row.get("prediction") == label
        for row in records
    )
    false_positive = sum(
        row.get("target_answer") != label and row.get("prediction") == label
        for row in records
    )
    false_negative = sum(
        row.get("target_answer") == label and row.get("prediction") != label
        for row in records
    )
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0


def observed_labels(records: list[dict]) -> set[str]:
    return {
        value
        for row in records
        for value in (row.get("target_answer"), row.get("prediction"))
        if value not in (None, "")
    }


def macro_f1_from_predictions(path: Path) -> tuple[float, int]:
    records = read_predictions(path)
    if not records:
        raise ValueError(f"No predictions found in {path}")
    labels = observed_labels(records)
    if not labels:
        raise ValueError(f"No labels found in {path}")
    score = sum(label_f1(records, label) for label in labels) / len(labels)
    return score, len(records)


def score_predictions(path: Path, metric: str) -> tuple[float, int]:
    if metric == "accuracy":
        return accuracy_from_predictions(path)
    return macro_f1_from_predictions(path)


def matches_target(row: dict[str, str], dataset: str, model: str | None) -> bool:
    return row.get("dataset") == dataset and (not model or row.get("model") == model)


def metric_field(row: dict[str, str]) -> str:
    return "eval_metric" if "eval_metric" in row else "metric"


def output_directory(row: dict[str, str]) -> Path:
    return Path(row.get("output_dir") or row.get("eval_output_dir") or "")


def method_description(metric: str) -> str:
    if metric == "accuracy":
        return "mean(predictions.correct)"
    return "macro_f1(saved predictions)"


def total_field(row: dict[str, str]) -> str:
    return "eval_total" if "eval_total" in row else "evaluated_total"


def labeled_field(row: dict[str, str]) -> str:
    return "eval_labeled_total" if "eval_labeled_total" in row else "labeled_total"


def rescore_row(
    row: dict[str, str],
    dataset: str,
    metric: str = "accuracy",
    model: str | None = None,
) -> dict[str, str]:
    if not matches_target(row, dataset, model):
        return {**row, **dict.fromkeys(PROVENANCE_FIELDS, "")}
    predictions = output_directory(row) / "predictions.jsonl"
    score, total = score_predictions(predictions, metric)
    expected = int(row.get("evaluated_total") or row.get("eval_total") or total)
    if total != expected:
        raise ValueError(f"Expected {expected} predictions in {predictions}; found {total}")
    metric_name = metric_field(row)
    result = {
        **row,
        metric_name: metric,
        "score": str(score),
        "rescore_source_metric": row.get(metric_name, ""),
        "rescore_method": method_description(metric),
    }
    result[total_field(row)] = str(total)
    result[labeled_field(row)] = str(total)
    return result


def rescore_rows(
    rows: list[dict[str, str]],
    dataset: str,
    metric: str = "accuracy",
    model: str | None = None,
) -> list[dict[str, str]]:
    return [rescore_row(row, dataset, metric, model) for row in rows]


def output_fields(fields: list[str], rows: list[dict[str, str]]) -> list[str]:
    extras = dict.fromkeys(key for row in rows for key in row if key not in fields)
    return [*fields, *extras]


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    fields = output_fields(fields, rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    fields, rows = read_rows(args.input)
    rescored = rescore_rows(rows, args.dataset, args.metric, args.model)
    write_rows(args.output, fields, rescored)
    changed = sum(matches_target(row, args.dataset, args.model) for row in rows)
    print(f"Rescored {changed} rows for {args.dataset} with {args.metric}.")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

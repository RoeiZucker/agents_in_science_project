#!/usr/bin/env python3
"""Compare paired likelihood and tagged-generation evaluation outputs.

Example:
  python compare_prediction_protocols.py \
    --likelihood-dir /path/to/likelihood \
    --tagged-dir /path/to/tagged \
    --output-dir /path/to/comparison
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--likelihood-dir", type=Path, required=True)
    parser.add_argument("--tagged-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    total = int(summary.get("total") or 0)
    correct = int(summary.get("correct") or 0)
    return {
        "method": name,
        "protocol": summary.get("evaluation_protocol"),
        "metric": summary.get("metric"),
        "score": summary.get("score"),
        "accuracy": correct / total if total else None,
        "correct": correct,
        "total": total,
        "tag_parse_failures": summary.get("tag_parse_failures"),
    }


def index_rows(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    indexed = {}
    for row in rows:
        source_index = int(row["source_index"])
        if source_index in indexed:
            raise ValueError(f"Duplicate source index: {source_index}")
        indexed[source_index] = row
    return indexed


def pair_predictions(
    likelihood: list[dict[str, Any]], tagged: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    left = index_rows(likelihood)
    right = index_rows(tagged)
    if left.keys() != right.keys():
        raise ValueError("The two runs did not evaluate the same source indices")
    return [paired_row(source_index, left[source_index], right[source_index]) for source_index in sorted(left)]


def paired_row(
    source_index: int, likelihood: dict[str, Any], tagged: dict[str, Any]
) -> dict[str, Any]:
    if likelihood.get("target_answer") != tagged.get("target_answer"):
        raise ValueError(f"Target mismatch at source index {source_index}")
    likelihood_correct = bool(likelihood.get("correct"))
    tagged_correct = bool(tagged.get("correct"))
    return {
        "source_index": source_index,
        "question": likelihood.get("question"),
        "target_answer": likelihood.get("target_answer"),
        "likelihood_prediction": likelihood.get("prediction"),
        "likelihood_correct": likelihood_correct,
        "tagged_prediction": tagged.get("prediction"),
        "tagged_correct": tagged_correct,
        "predictions_agree": likelihood.get("prediction") == tagged.get("prediction"),
        "winner": example_winner(likelihood_correct, tagged_correct),
    }


def example_winner(likelihood_correct: bool, tagged_correct: bool) -> str:
    if likelihood_correct == tagged_correct:
        return "both_correct" if likelihood_correct else "both_wrong"
    return "likelihood_only" if likelihood_correct else "tagged_only"


def comparison_summary(
    method_rows: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> dict[str, Any]:
    likelihood, tagged = method_rows
    counts = {
        outcome: sum(row["winner"] == outcome for row in paired)
        for outcome in ("both_correct", "likelihood_only", "tagged_only", "both_wrong")
    }
    agreements = sum(row["predictions_agree"] for row in paired)
    return {
        "examples": len(paired),
        "score_delta_likelihood_minus_tagged": numeric_delta(
            likelihood["score"], tagged["score"]
        ),
        "accuracy_delta_likelihood_minus_tagged": numeric_delta(
            likelihood["accuracy"], tagged["accuracy"]
        ),
        "predictions_agree": agreements,
        "prediction_agreement_rate": agreements / len(paired) if paired else None,
        **counts,
    }


def numeric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    likelihood_summary = read_json(args.likelihood_dir / "summary.json")
    tagged_summary = read_json(args.tagged_dir / "summary.json")
    method_rows = [
        method_row("label_likelihood", likelihood_summary),
        method_row("tagged_generation", tagged_summary),
    ]
    paired = pair_predictions(
        read_jsonl(args.likelihood_dir / "predictions.jsonl"),
        read_jsonl(args.tagged_dir / "predictions.jsonl"),
    )
    summary = comparison_summary(method_rows, paired)
    write_csv(args.output_dir / "method_comparison.csv", method_rows)
    write_csv(args.output_dir / "paired_predictions.csv", paired)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

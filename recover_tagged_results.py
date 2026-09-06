#!/usr/bin/env python3
"""Recover constrained tagged results from existing raw generations.

The original evaluation directories are never modified. Only responses that map
unambiguously to declared labels are recovered.
"""

# Preview use case:
#   python recover_tagged_results.py --pairs-csv /tmp/current_all_results_pairs.csv \
#     --output-root /tmp/tagged_recovery_preview --dry-run
#
# Materialize audited recovered copies:
#   python recover_tagged_results.py --pairs-csv /tmp/current_all_results_pairs.csv \
#     --output-root /path/to/eval_results/_tagged_parser_recovery/RUN_ID

from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Any

from tagged_answer_parser import extract_tagged_answer


TAGGED_PROTOCOLS = {
    "tagged_label_generation_accuracy",
    "tagged_multiple_choice_accuracy",
    "tagged_set_generation",
}
METRIC_FIELDS = (
    "accuracy",
    "exact_match",
    "macro_f1",
    "qa_f1",
    "numeric_match",
    "rouge_l",
    "set_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-recoveries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def has_target(value: Any) -> bool:
    if isinstance(value, list):
        return any(has_target(item) for item in value)
    return value is not None and str(value).strip() != ""


def parse_json_value(value: str) -> Any:
    text = value.strip()
    if not text.startswith(("[", "{")):
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def canonical_set(value: Any) -> set[str]:
    if isinstance(value, str):
        parsed = parse_json_value(value)
        if parsed is not value:
            return canonical_set(parsed)
        return {
            normalize_answer(item)
            for item in re.split(r"[,;\n]+", value)
            if normalize_answer(item)
        }
    if isinstance(value, dict):
        return {normalize_answer(json.dumps(value, sort_keys=True))}
    if isinstance(value, (list, tuple, set)):
        items = set()
        for item in value:
            items.update(canonical_set(item))
        return items
    normalized = normalize_answer(value)
    return {normalized} if normalized else set()


def set_f1(prediction: Any, target: Any) -> float:
    predicted = canonical_set(prediction)
    expected = canonical_set(target)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = len(predicted & expected)
    return 2 * overlap / (len(predicted) + len(expected))


def exact_score(prediction: Any, target: Any) -> float:
    targets = target if isinstance(target, list) else [target]
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(item) for item in targets))


def row_score(metric: str, prediction: Any, target: Any) -> float:
    if metric == "set_f1":
        return set_f1(prediction, target)
    return exact_score(prediction, target)


def macro_f1(predictions: list[str], targets: list[str]) -> float:
    labels = sorted(set(predictions) | set(targets))
    scores = []
    for label in labels:
        true_positive = sum(
            prediction == label and target == label
            for prediction, target in zip(predictions, targets)
        )
        false_positive = sum(
            prediction == label and target != label
            for prediction, target in zip(predictions, targets)
        )
        false_negative = sum(
            prediction != label and target == label
            for prediction, target in zip(predictions, targets)
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score_predictions(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    scores = []
    predictions = []
    targets = []
    for row in rows:
        target = row.get("target_answer")
        if not has_target(target):
            row["example_score"] = None
            row["correct"] = None
            continue
        score = row_score(metric, row.get("prediction", ""), target)
        row["example_score"] = score
        row["correct"] = score == 1.0
        scores.append(score)
        predictions.append(normalize_answer(row.get("prediction", "")))
        primary = target[0] if isinstance(target, list) else target
        targets.append(normalize_answer(primary))
    aggregate = (
        macro_f1(predictions, targets)
        if metric == "macro_f1"
        else sum(scores) / len(scores)
    )
    return {
        "score": aggregate,
        "total": len(rows),
        "labeled_total": len(scores),
        "correct": sum(score == 1.0 for score in scores),
    }


def recover_prediction(
    row: dict[str, Any], labels: list[str], expect_set: bool
) -> bool:
    if row.get("answer_parse_status") == "ok":
        return False
    answer, status = extract_tagged_answer(
        row.get("raw_generation", ""), labels=labels, expect_set=expect_set
    )
    if status not in {"untagged_json", "untagged_labels", "untagged_label"}:
        return False
    row["prediction"] = answer
    row["answer_parse_status"] = status
    row["posthoc_recovered"] = True
    return True


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "unknown"


def destination_path(root: Path, summary: dict[str, Any]) -> Path:
    model = safe_name(str(summary.get("model", "")))
    dataset = safe_name(str(summary.get("dataset", "")))
    split = safe_name(str(summary.get("split", "")))
    return root / "eval_results" / model / f"{dataset}_{split}"


def update_summary(
    summary: dict[str, Any],
    stats: dict[str, Any],
    predictions_path: Path,
    source: Path,
    recovered: int,
) -> None:
    metric = str(summary["metric"])
    original_score = summary.get("score")
    original_failures = summary.get("tag_parse_failures")
    summary.update(stats)
    for field in METRIC_FIELDS:
        summary[field] = stats["score"] if field == metric else None
    summary["predictions_path"] = str(predictions_path)
    summary["posthoc_recovery"] = {
        "parser": "conservative_declared_labels_v1",
        "source_output_dir": str(source),
        "recovered_predictions": recovered,
        "original_score": original_score,
        "original_tag_parse_failures": original_failures,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def recover_pair(
    pair: dict[str, str], root: Path, minimum: int, dry_run: bool
) -> dict[str, Any] | None:
    source = Path(pair["eval_output_dir"])
    summary_path = source / "summary.json"
    predictions_path = source / "predictions.jsonl"
    if not summary_path.exists() or not predictions_path.exists():
        return None
    summary = read_json(summary_path)
    protocol = str(summary.get("evaluation_protocol", ""))
    labels = [str(label) for label in summary.get("label_values") or []]
    if protocol not in TAGGED_PROTOCOLS or not labels:
        return None
    rows = read_jsonl(predictions_path)
    expect_set = protocol == "tagged_set_generation"
    recovered = sum(recover_prediction(row, labels, expect_set) for row in rows)
    if recovered < minimum:
        return None
    stats = score_predictions(rows, str(summary["metric"]))
    destination = destination_path(root, summary)
    output_predictions = destination / "predictions.jsonl"
    if not dry_run:
        destination.mkdir(parents=True, exist_ok=False)
        update_summary(
            summary, stats, output_predictions, source, recovered
        )
        write_jsonl(output_predictions, rows)
        (destination / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return recovery_row(pair, summary, stats, source, destination, recovered)


def recovery_row(
    pair: dict[str, str],
    summary: dict[str, Any],
    stats: dict[str, Any],
    source: Path,
    destination: Path,
    recovered: int,
) -> dict[str, Any]:
    original_score = summary.get("score")
    if summary.get("posthoc_recovery"):
        original_score = summary["posthoc_recovery"]["original_score"]
    return {
        "condition": "posthoc_parser_recovery",
        "query_dataset": pair["dataset"],
        "rank": "",
        "model_name": pair["model"],
        "recommendation_score": "",
        "source_method": "posthoc_parser_recovery",
        "reasoning": "Conservative recovery from stored raw generation and declared labels.",
        "eval_split": summary.get("split", ""),
        "eval_protocol": summary.get("evaluation_protocol", ""),
        "eval_metric": summary.get("metric", ""),
        "eval_score": stats["score"],
        "eval_total": stats["total"],
        "eval_labeled_total": stats["labeled_total"],
        "eval_status": "ok",
        "eval_output_dir": str(destination),
        "eval_notes": (
            f"Recovered {recovered} stored untagged predictions without new inference; "
            f"source={source}; original_score={original_score}."
        ),
        "recovered_predictions": recovered,
        "original_score": original_score,
        "source_eval_output_dir": str(source),
    }


def eligible_pair(pair: dict[str, str]) -> bool:
    return pair.get("outcome") == "success" and bool(pair.get("eval_output_dir"))


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and not args.dry_run:
        raise FileExistsError(f"Output root already exists: {args.output_root}")
    recovered = []
    for pair in read_csv(args.pairs_csv):
        if not eligible_pair(pair):
            continue
        row = recover_pair(
            pair,
            args.output_root,
            args.minimum_recoveries,
            args.dry_run,
        )
        if row:
            recovered.append(row)
    recovered.sort(
        key=lambda row: int(row["recovered_predictions"]), reverse=True
    )
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_root / "batch_results.csv", recovered)
        write_csv(args.output_root / "recovery_audit.csv", recovered)
    print(f"Recovered pairs: {len(recovered)}")
    print(
        "Recovered predictions: "
        f"{sum(int(row['recovered_predictions']) for row in recovered)}"
    )
    for row in recovered[:20]:
        print(
            f"{row['recovered_predictions']}: "
            f"{row['query_dataset']} :: {row['model_name']} "
            f"{row['original_score']} -> {row['eval_score']}"
        )


if __name__ == "__main__":
    main()

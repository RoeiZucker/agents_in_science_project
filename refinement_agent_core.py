#!/usr/bin/env python3
"""Error-analysis and retrieval-feedback helpers for evaluation runs."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from result_contract import measured_success, result_outcome


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_read_json(path: Path) -> Any:
    return read_json(path) if path.exists() else None


def load_candidate_models(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    data = read_json(path)
    candidates = data if isinstance(data, list) else data.get("candidates", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("model"):
            grouped[candidate["model"]].append(candidate)
    return dict(grouped)


def analyze_run(
    run_dir: Path,
    candidate_models_path: Path | None = None,
    max_failed_examples: int = 20,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    results = read_results_csv(run_dir / "results.csv")
    dataset_inspection = maybe_read_json(run_dir / "dataset_inspection.json")
    plans = maybe_read_json(run_dir / "plans.json")
    candidates = load_candidate_models(candidate_models_path)
    models = analyze_models(results, candidates, max_failed_examples)
    return {
        "run_dir": str(run_dir),
        "dataset": first_value(results, "dataset"),
        "split": first_value(results, "split"),
        "task": infer_task(dataset_inspection, models),
        "dataset_inspection": dataset_inspection,
        "plans_path": str(run_dir / "plans.json") if plans is not None else None,
        "candidate_models_path": str(candidate_models_path) if candidate_models_path else None,
        "models": models,
        "comparisons": compare_models(models),
        "failure_modes": collect_failure_modes(models),
        "recommendations": recommend_next_steps(dataset_inspection, models),
    }


def analyze_models(
    results: list[dict[str, str]],
    candidates: dict[str, list[dict[str, Any]]],
    max_failed_examples: int,
) -> list[dict[str, Any]]:
    analyses = []
    for row in results:
        output_dir = Path(row.get("output_dir", ""))
        summary = maybe_read_json(output_dir / "summary.json")
        predictions = read_predictions(output_dir)
        analysis = analyze_model(row, summary, predictions, max_failed_examples)
        if row.get("model") in candidates:
            analysis["retrieval_candidates"] = candidates[row["model"]]
        analyses.append(analysis)
    return analyses


def read_predictions(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "predictions.jsonl"
    return read_jsonl(path) if path.exists() else []


def analyze_model(
    result_row: dict[str, str],
    summary: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    max_failed_examples: int,
) -> dict[str, Any]:
    protocol = result_row.get("evaluation_protocol") or (summary or {}).get("evaluation_protocol")
    target_values = known_targets(predictions) if fixed_label_protocol(protocol, summary) else set()
    return {
        "model": result_row.get("model"),
        "score": parse_float(result_row.get("score")),
        "status": result_row.get("status"),
        "outcome": result_outcome(result_row.get("status"), result_row.get("score"), result_row.get("labeled_total")),
        "evaluation_protocol": protocol,
        "metric": result_row.get("metric") or (summary or {}).get("metric"),
        "total": parse_int(result_row.get("total")),
        "labeled_total": parse_int(result_row.get("labeled_total")),
        "output_dir": result_row.get("output_dir"),
        "notes": result_row.get("notes", ""),
        "summary": summary,
        "prediction_distribution": count_field(predictions, "prediction"),
        "target_distribution": count_targets(predictions),
        "blank_predictions": count_blank_predictions(predictions),
        "invalid_predictions": count_invalid_predictions(predictions, target_values),
        "per_target_accuracy": per_target_accuracy(predictions),
        "common_wrong_predictions": common_wrong_predictions(predictions),
        "failed_examples": failed_examples(predictions, max_failed_examples),
        "failure_modes": model_failure_modes(result_row, summary, predictions, target_values),
    }


def fixed_label_protocol(protocol: Any, summary: dict[str, Any] | None) -> bool:
    fixed = {
        "multiple_choice_accuracy",
        "label_generation_accuracy",
        "label_logprob_accuracy",
        "tagged_label_generation_accuracy",
        "tagged_multiple_choice_accuracy",
        "classifier_label_accuracy",
        "zero_shot_nli_accuracy",
    }
    return str(protocol) in fixed or bool((summary or {}).get("label_values"))


def known_targets(predictions: list[dict[str, Any]]) -> set[str]:
    return {normalize_value(target_value(row)) for row in predictions if target_value(row) is not None}


def target_value(row: dict[str, Any]) -> Any:
    return row.get("target_answer", row.get("answer"))


def normalize_value(value: Any) -> str:
    return str(value).strip().lower()


def count_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(Counter(stringify(row.get(field)) for row in rows))


def count_targets(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(stringify(target_value(row)) for row in rows))


def stringify(value: Any) -> str:
    return "<blank>" if value is None or str(value).strip() == "" else str(value)


def count_blank_predictions(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if stringify(row.get("prediction")) == "<blank>")


def count_invalid_predictions(rows: list[dict[str, Any]], targets: set[str]) -> int:
    if not targets:
        return 0
    return sum(1 for row in rows if normalize_value(row.get("prediction")) not in targets)


def per_target_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for row in rows:
        target = stringify(target_value(row))
        if row.get("correct") is None:
            continue
        totals[target] += 1
        correct[target] += int(bool(row.get("correct")))
    return {
        target: {
            "correct": correct[target],
            "total": total,
            "accuracy": correct[target] / total if total else None,
        }
        for target, total in sorted(totals.items())
    }


def common_wrong_predictions(rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    pairs = Counter()
    for row in rows:
        if row.get("correct") is False:
            pairs[(stringify(target_value(row)), stringify(row.get("prediction")))] += 1
    return [
        {"target": target, "prediction": prediction, "count": count}
        for (target, prediction), count in pairs.most_common(limit)
    ]


def failed_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        if row.get("correct") is not False:
            continue
        examples.append(compact_example(row))
        if len(examples) >= limit:
            break
    return examples


def compact_example(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": row.get("index"),
        "id": row.get("id"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "target_answer": row.get("target_answer"),
        "prediction": row.get("prediction"),
    }


def model_failure_modes(
    result_row: dict[str, str],
    summary: dict[str, Any] | None,
    predictions: list[dict[str, Any]],
    target_values: set[str],
) -> list[dict[str, Any]]:
    modes = []
    score = parse_float(result_row.get("score"))
    labeled_total = parse_int(result_row.get("labeled_total"))
    if not measured_success(result_row.get("status"), result_row.get("score"), result_row.get("labeled_total")):
        modes.append(mode("evaluation_failed", "high", "No valid labeled measurement was produced."))
    if summary is None:
        modes.append(mode("missing_summary", "high", "summary.json was not found."))
    if not predictions:
        modes.append(mode("missing_predictions", "high", "predictions.jsonl was not found or empty."))
    if labeled_total == 0:
        modes.append(mode("unlabeled_split", "high", "No labeled examples were scored."))
    modes.extend(output_quality_modes(predictions, target_values))
    tag_failures = parse_int((summary or {}).get("tag_parse_failures"))
    if tag_failures:
        total = max(1, parse_int((summary or {}).get("total")))
        modes.append(mode("tag_parse_failures", severity_for_fraction(tag_failures / total), f"{tag_failures} tagged answers could not be parsed."))
    modes.extend(per_label_modes(predictions, score))
    return modes


def output_quality_modes(rows: list[dict[str, Any]], targets: set[str]) -> list[dict[str, Any]]:
    if not rows:
        return []
    modes = []
    blank_count = count_blank_predictions(rows)
    invalid_count = count_invalid_predictions(rows, targets)
    if blank_count:
        modes.append(mode("blank_outputs", severity_for_fraction(blank_count / len(rows)), f"{blank_count} blank predictions."))
    if invalid_count:
        modes.append(mode("invalid_outputs", severity_for_fraction(invalid_count / len(rows)), f"{invalid_count} predictions outside known targets."))
    top_prediction, top_count = Counter(stringify(row.get("prediction")) for row in rows).most_common(1)[0]
    if top_count / len(rows) >= 0.8:
        modes.append(mode("prediction_collapse", "high", f"Prediction {top_prediction!r} appears in {top_count}/{len(rows)} rows."))
    return modes


def per_label_modes(rows: list[dict[str, Any]], score: float | None) -> list[dict[str, Any]]:
    modes = []
    for target, stats in per_target_accuracy(rows).items():
        accuracy = stats["accuracy"]
        if accuracy is None or stats["total"] < 3:
            continue
        baseline = score if score is not None else 0.5
        if accuracy < 0.5 and accuracy + 0.2 < baseline:
            modes.append(mode("weak_target_label", "medium", f"Low accuracy for target {target!r}: {accuracy:.3f}."))
    return modes


def severity_for_fraction(value: float) -> str:
    return "high" if value >= 0.25 else "medium"


def mode(name: str, severity: str, detail: str) -> dict[str, str]:
    return {"name": name, "severity": severity, "detail": detail}


def compare_models(models: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [model for model in models if model.get("outcome") == "success"]
    best = max(scored, key=lambda item: item["score"], default=None)
    worst = min(scored, key=lambda item: item["score"], default=None)
    return {
        "best_model": model_score_entry(best),
        "worst_model": model_score_entry(worst),
        "score_gap": score_gap(best, worst),
        "all_models_failed_examples": all_models_failed_examples(models),
        "retrieval_vs_measured": retrieval_vs_measured(models),
    }


def model_score_entry(model: dict[str, Any] | None) -> dict[str, Any] | None:
    if model is None:
        return None
    return {"model": model.get("model"), "score": model.get("score")}


def score_gap(best: dict[str, Any] | None, worst: dict[str, Any] | None) -> float | None:
    if best is None or worst is None:
        return None
    return best["score"] - worst["score"]


def all_models_failed_examples(models: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    failure_sets = []
    example_by_key = {}
    for model in models:
        failed = model.get("failed_examples", [])
        keys = set()
        for example in failed:
            key = example_key(example)
            keys.add(key)
            example_by_key[key] = example
        failure_sets.append(keys)
    if not failure_sets:
        return []
    common = set.intersection(*failure_sets)
    return [example_by_key[key] for key in sorted(common)[:limit]]


def example_key(example: dict[str, Any]) -> str:
    value = example.get("id", example.get("index"))
    return str(value)


def retrieval_vs_measured(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        for candidate in model.get("retrieval_candidates", []):
            rows.append({
                "candidate_id": candidate.get("candidate_id"),
                "condition": candidate.get("condition"),
                "model": model.get("model"),
                "retrieval_rank": candidate.get("rank"),
                "retrieval_score": candidate.get("score"),
                "measured_score": model.get("score"),
            })
    return sorted(rows, key=lambda row: none_last(row.get("retrieval_rank")))


def none_last(value: Any) -> tuple[bool, Any]:
    return (value is None, value)


def collect_failure_modes(models: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for model in models:
        for item in model.get("failure_modes", []):
            counts[item["name"]] += 1
    return dict(counts)


def recommend_next_steps(dataset_inspection: dict[str, Any] | None, models: list[dict[str, Any]]) -> list[str]:
    task = infer_task(dataset_inspection, models)
    recommendations = []
    if any_mode(models, "prediction_collapse"):
        recommendations.append("Prefer instruction-tuned models or scoring-based evaluation for constrained-answer tasks.")
    if any_mode(models, "invalid_outputs"):
        recommendations.append("Retrieve models whose training or model card explicitly mentions the target task format.")
    if any_mode(models, "weak_target_label"):
        recommendations.append("Ask retrieval for candidates with stronger coverage of the weakest labels/classes.")
    if task == "image_qa":
        recommendations.append("Retrieve vision-language models rather than text-only language models.")
    if task in {"classification", "boolean_qa"}:
        recommendations.append("Include sequence-to-sequence instruction models and task-specific classifiers in the next retrieval round.")
    return recommendations or ["Use the best measured model as the baseline and retrieve nearby stronger candidates."]


def any_mode(models: list[dict[str, Any]], name: str) -> bool:
    return any(item["name"] == name for model in models for item in model.get("failure_modes", []))


def make_retrieval_feedback(analysis: dict[str, Any], next_round: int = 2) -> dict[str, Any]:
    task = analysis.get("task", "unknown")
    best = analysis.get("comparisons", {}).get("best_model") or {}
    models = analysis.get("models", [])
    requested_split = analysis.get("requested_split") or analysis.get("split")
    evaluated_split = analysis.get("evaluated_split") or analysis.get("split")
    return {
        "dataset": analysis.get("dataset"),
        "split": requested_split,
        "requested_split": requested_split,
        "evaluated_split": evaluated_split,
        "next_round": next_round,
        "best_model": best.get("model"),
        "best_score": best.get("score"),
        "observations": build_observations(analysis),
        "retrieval_constraints": build_retrieval_constraints(task, models),
        "candidate_request": build_candidate_request(analysis, next_round),
    }


def build_observations(analysis: dict[str, Any]) -> list[str]:
    observations = []
    best = analysis.get("comparisons", {}).get("best_model")
    if best:
        observations.append(f"Best measured model was {best['model']} with score {best['score']:.4f}.")
    for name, count in analysis.get("failure_modes", {}).items():
        observations.append(f"Observed failure mode {name} in {count} model(s).")
    recommendations = analysis.get("recommendations", [])
    observations.extend(recommendations)
    return observations


def build_retrieval_constraints(task: str, models: list[dict[str, Any]]) -> dict[str, Any]:
    constraints = task_constraints(task)
    constraints["candidate_exclusions"] = weak_or_failed_models(models)
    constraints["baseline_models"] = [model["model"] for model in models if model.get("outcome") == "success"]
    return constraints


def task_constraints(task: str) -> dict[str, list[str]]:
    if task == "multiple_choice":
        return {
            "prefer_tasks": ["multiple-choice", "commonsense-reasoning", "question-answering"],
            "prefer_keywords": ["multiple choice", "commonsense", "instruction tuned", "answer selection"],
            "avoid_keywords": ["base-only", "unlabeled", "embedding-only"],
        }
    if task == "boolean_qa":
        return {
            "prefer_tasks": ["question-answering", "reading-comprehension", "natural-language-inference"],
            "prefer_keywords": ["boolq", "yes no", "true false", "flan", "instruction tuned"],
            "avoid_keywords": ["image-only", "embedding-only"],
        }
    if task == "classification":
        return {
            "prefer_tasks": ["text-classification", "sentiment-analysis", "sequence-classification"],
            "prefer_keywords": ["classification", "sentiment", "sst2", "label mapping"],
            "avoid_keywords": ["image-only", "chat-only without classification examples"],
        }
    if task == "image_qa":
        return {
            "prefer_tasks": ["visual-question-answering", "image-text-to-text", "vision-language"],
            "prefer_keywords": ["VQA", "multimodal", "vision language", "image question answering"],
            "avoid_keywords": ["text-only", "embedding-only"],
        }
    return {
        "prefer_tasks": ["instruction-following", "question-answering"],
        "prefer_keywords": ["instruction tuned", "evaluation-ready"],
        "avoid_keywords": ["gated without access", "missing transformers support"],
    }


def weak_or_failed_models(models: list[dict[str, Any]]) -> list[str]:
    return [model["model"] for model in models if model.get("model") and model.get("outcome") != "success"]


def build_candidate_request(analysis: dict[str, Any], next_round: int) -> dict[str, Any]:
    return {
        "round": next_round,
        "dataset": analysis.get("dataset"),
        "split": analysis.get("split"),
        "task": analysis.get("task"),
        "minimum_candidates": 5,
        "expected_format": "candidate_models.json",
        "fields": ["candidate_id", "condition", "model", "rank", "score", "source", "intended_use", "reason"],
    }


def feedback_prompt(feedback: dict[str, Any]) -> str:
    lines = [
        "Use this refinement feedback to retrieve the next model candidates.",
        "",
        f"Dataset: {feedback.get('dataset')}",
        f"Split: {feedback.get('split')}",
        f"Next round: {feedback.get('next_round')}",
        f"Best baseline: {feedback.get('best_model')} ({feedback.get('best_score')})",
        "",
        "Observations:",
    ]
    lines.extend(f"- {item}" for item in feedback.get("observations", []))
    lines.extend(["", "Retrieval constraints:"])
    constraints = feedback.get("retrieval_constraints", {})
    for key, values in constraints.items():
        rendered = ", ".join(str(value) for value in values)
        lines.append(f"- {key}: {rendered}")
    lines.extend([
        "",
        "Return candidate_models.json with a top-level candidates list.",
        "Each candidate must include candidate_id, condition, model, rank, score, source, intended_use, and reason.",
    ])
    return "\n".join(lines) + "\n"


def infer_task(dataset_inspection: dict[str, Any] | None, models: list[dict[str, Any]]) -> str:
    if dataset_inspection and dataset_inspection.get("task"):
        task = dataset_inspection["task"]
        if task == "generation" and looks_boolean_dataset(dataset_inspection):
            return "boolean_qa"
        if task == "generation" and looks_finite_label_text_dataset(dataset_inspection):
            return "classification"
        return task
    summaries = [model.get("summary") for model in models if model.get("summary")]
    for summary in summaries:
        if summary.get("task"):
            return summary["task"]
    return "unknown"


def looks_finite_label_text_dataset(dataset_inspection: dict[str, Any]) -> bool:
    label_map = dataset_inspection.get("label_map") or {}
    columns = set(dataset_inspection.get("columns") or [])
    image_column = dataset_inspection.get("image_column")
    choices_column = dataset_inspection.get("choices_column")
    has_image = bool(image_column and image_column in columns)
    has_choices = bool(choices_column and choices_column in columns)
    return isinstance(label_map, dict) and 1 < len(label_map) <= 100 and not has_image and not has_choices


def looks_boolean_dataset(dataset_inspection: dict[str, Any]) -> bool:
    columns = set(dataset_inspection.get("columns") or [])
    label_map = dataset_inspection.get("label_map") or {}
    values = {str(value).lower() for value in label_map.values()}
    return {"passage", "question"}.issubset(columns) and values.issubset({"false", "true"})


def first_value(rows: list[dict[str, str]], key: str) -> str | None:
    for row in rows:
        if row.get(key):
            return row[key]
    return None


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def parse_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)

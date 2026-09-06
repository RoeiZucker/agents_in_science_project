#!/usr/bin/env python3
"""Validate protocol-pinned evaluation contexts against dataset schemas.

Examples:
  python validate_evaluation_contexts.py \
    --manifest config/oren_local_cached_18_final.json \
    --output runtime/oren_context_preflight.json

  python validate_evaluation_contexts.py \
    --manifest config/oren_local_cached_18_final.json \
    --dataset-subset-file config/oren_cached_18_datasets.txt \
    --inspect-datasets --sample-size 20 \
    --output runtime/oren_context_preflight.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_agent_core import (
    apply_known_dataset_contract,
    inspect_dataset,
    method_matches_task,
    selected_evaluation_method,
)
from evaluation_context_spec import read_context_template
from run_eval_agent import apply_context
from run_selection_loop import load_manifest, resolve_path, selected_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-subset-file", type=Path)
    parser.add_argument("--inspect-datasets", action="store_true")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = selected_specs(
        load_manifest(args.manifest), args.dataset_subset_file
    )
    rows = [validate_spec(spec, args) for spec in specs]
    report = {
        "manifest": str(args.manifest.resolve()),
        "inspected_datasets": args.inspect_datasets,
        "datasets": len(rows),
        "ready": sum(row["status"] == "ready" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["failed"] else 0


def validate_spec(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset = spec["dataset"]
    try:
        template = load_spec_template(spec, args.manifest)
        if not args.inspect_datasets:
            return template_row(spec, template)
        inspection = inspect_spec_dataset(spec, template, args)
        return inspection_row(spec, template, inspection)
    except Exception as exc:
        return {
            "dataset": dataset,
            "status": "failed",
            "errors": [f"{type(exc).__name__}: {exc}"],
        }


def load_spec_template(
    spec: dict[str, Any], manifest: Path
) -> dict[str, Any]:
    configured = spec.get("evaluation_context")
    if not configured:
        raise ValueError("evaluation_context is missing")
    return read_context_template(resolve_path(manifest, configured), spec["dataset"])


def template_row(
    spec: dict[str, Any], template: dict[str, Any]
) -> dict[str, Any]:
    context = template["dataset_context"]
    return {
        "dataset": spec["dataset"],
        "status": "ready",
        "split": spec.get("split", "auto"),
        "task": context["task"],
        "metric": context.get("evaluation_method", "auto"),
        "confidence": template["confidence"],
        "errors": [],
    }


def inspect_spec_dataset(
    spec: dict[str, Any],
    template: dict[str, Any],
    args: argparse.Namespace,
) -> Any:
    values = template["dataset_context"]
    inspection = inspect_dataset(
        dataset=spec["dataset"],
        dataset_source=str(values.get("dataset_source", "")),
        dataset_kwargs=dict(values.get("dataset_kwargs", {})),
        subset=str(values.get("subset", "")) or None,
        split=str(values.get("split", "")) or spec.get("split", "auto"),
        sample_size=args.sample_size,
        question_column=values["question_column"],
        context_column=str(values.get("context_column", "")),
        answer_column=values["answer_column"],
        choices_column=values["choices_column"],
        choices_columns=list(values.get("choices_columns", [])),
        image_column=values["image_column"],
        evaluation_method=str(values.get("evaluation_method", "auto")),
        trust_remote_code=args.trust_remote_code,
    )
    inspection = apply_context(inspection, template)
    return apply_known_dataset_contract(inspection)


def inspection_row(
    spec: dict[str, Any], template: dict[str, Any], inspection: Any
) -> dict[str, Any]:
    errors = readiness_errors(spec, template, inspection)
    return {
        "dataset": spec["dataset"],
        "status": "failed" if errors else "ready",
        "dataset_source": inspection.dataset_source,
        "split": inspection.selected_split,
        "total": inspection.total,
        "task": inspection.task,
        "metric": selected_evaluation_method(inspection),
        "question_column": inspection.question_column,
        "context_column": inspection.context_column,
        "answer_column": inspection.answer_column,
        "choices_columns": inspection.choices_columns,
        "label_count": len(inspection.label_map),
        "labeled_sample": inspection.labeled_total_sample,
        "confidence": template["confidence"],
        "errors": errors,
    }


def readiness_errors(
    spec: dict[str, Any], template: dict[str, Any], inspection: Any
) -> list[str]:
    expected = template["dataset_context"]
    errors = []
    if inspection.selected_split != spec.get("split", "auto"):
        errors.append(
            f"selected split {inspection.selected_split!r} does not match manifest"
        )
    if not inspection.has_gold or inspection.labeled_total_sample < 1:
        errors.append("no labeled examples were detected")
    if inspection.task != expected["task"]:
        errors.append(f"task resolved to {inspection.task!r}")
    metric = selected_evaluation_method(inspection)
    if not method_matches_task(inspection.task, metric):
        errors.append(f"metric {metric!r} is incompatible with the task")
    prompt = expected.get("prompt_template", "")
    if prompt and inspection.prompt_template != prompt:
        errors.append("prompt template was not applied")
    if inspection.task == "multiple_choice" and not inspection.choices_columns:
        errors.append("multiple-choice fields were not applied")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())

"""Load and materialize versioned per-dataset evaluation context templates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_evaluation_conditions import (
    validate_context,
    validate_dataset_context,
    validate_model_context,
)


TEMPLATE_FIELDS = {
    "context_version",
    "dataset",
    "dataset_context",
    "model_context",
    "confidence",
}


def read_context_template(path: Path, dataset: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    error = validate_context_template(value, dataset)
    if error:
        raise ValueError(f"Invalid evaluation context template {path}: {error}")
    return value


def validate_context_template(value: Any, dataset: str) -> str:
    if not isinstance(value, dict):
        return "template must be an object."
    missing = TEMPLATE_FIELDS - set(value)
    if missing:
        return f"template is missing fields: {sorted(missing)}."
    unknown = set(value) - TEMPLATE_FIELDS
    if unknown:
        return f"template has unknown fields: {sorted(unknown)}."
    if value["context_version"] != 1:
        return "context_version must be 1."
    if value["dataset"] != dataset:
        return f"dataset mismatch: expected {dataset!r}, got {value['dataset']!r}."
    if value["confidence"] not in {"high", "medium", "low"}:
        return "confidence must be one of: high, medium, low."
    return validate_dataset_context(value["dataset_context"]) or validate_model_context(
        value["model_context"]
    )


def materialize_context(
    template: dict[str, Any], models: list[str]
) -> dict[str, Any]:
    context = {
        "dataset": template["dataset"],
        "models": models,
        "dataset_context": template["dataset_context"],
        "model_context": template["model_context"],
        "confidence": template["confidence"],
    }
    error = validate_context(context, template["dataset"], models)
    if error:
        raise ValueError(f"Could not materialize evaluation context: {error}")
    return context

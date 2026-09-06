#!/usr/bin/env python3
"""Prepare rerun manifests for unsupported pairs fixed by current adapters.

Example:
  python prepare_unsupported_recovery.py \
    --results-root /path/to/overall_recovery_run \
    --generic-source config/overall_recovery_generic_tasks.csv \
    --explicit-source config/overall_recovery_explicit_tasks.csv \
    --generic-output config/unsupported_recovery_generic_tasks.csv \
    --explicit-output config/unsupported_recovery_explicit_tasks.csv \
    --audit-output config/unsupported_recovery_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval_agent_core import has_entailment_label, infer_model_type_from_metadata


QA_CONTEXT_DATASETS = {
    "bigbio/mediqa_qa",
    "deepset/covid_qa_deepset",
    "theatticusproject/cuad",
    "virattt/financial-qa-10K",
}
VISION_DATASETS = {"lmms-lab/DocVQA"}
KNOWN_CONTRACT_DATASETS = {"tuetschek/multi_woz_v22"}
AUDIT_FIELDS = (
    "side",
    "source_task_index",
    "dataset",
    "model",
    "status",
    "old_model_type",
    "inferred_model_type",
    "selected",
    "recovery_reason",
    "original_reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--generic-source", type=Path, required=True)
    parser.add_argument("--explicit-source", type=Path, required=True)
    parser.add_argument("--generic-output", type=Path, required=True)
    parser.add_argument("--explicit-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def task_index(path: Path) -> int:
    part = next(value for value in path.parts if value.startswith("task_"))
    return int(part.removeprefix("task_"))


def result_identity(row: dict[str, str]) -> tuple[str, str]:
    dataset = row.get("query_dataset") or row.get("dataset") or ""
    model = row.get("model_name") or row.get("model") or ""
    return dataset, model


def result_status(row: dict[str, str]) -> str:
    return row.get("eval_status") or row.get("status") or ""


def result_reason(row: dict[str, str]) -> str:
    return row.get("eval_notes") or row.get("notes") or ""


def result_files(root: Path, side: str) -> list[Path]:
    name = "batch_results.csv" if side == "generic" else "results.csv"
    return sorted((root / side).glob(f"task_*/**/{name}"))


def unsupported_results(root: Path, side: str) -> list[tuple[Path, dict[str, str]]]:
    results = []
    for path in result_files(root, side):
        for row in read_csv(path):
            if result_status(row) == "unsupported":
                results.append((path, row))
    return results


def load_matching_plan(path: Path, model: str) -> dict[str, Any] | None:
    task_root = next(parent for parent in path.parents if parent.name.startswith("task_"))
    for plan_path in task_root.glob("**/plans.json"):
        for plan in json.loads(plan_path.read_text(encoding="utf-8")):
            if plan.get("model", {}).get("model") == model:
                return plan
    return None


def inferred_model_type(model: dict[str, Any]) -> str:
    return infer_model_type_from_metadata(
        model.get("model", ""),
        model.get("config_model_type"),
        model.get("architectures") or [],
        model.get("pipeline_tag"),
        model.get("library_name"),
    )


def metadata_is_blocked(model: dict[str, Any]) -> bool:
    notes = "\n".join(model.get("notes") or []).lower()
    markers = ("gated repo", "401 client", "403 client", "awaiting a review")
    return any(marker in notes for marker in markers)


def is_causal_architecture(model: dict[str, Any]) -> bool:
    return any("causallm" in value.lower() for value in model.get("architectures") or [])


def recovery_decision(dataset: str, plan: dict[str, Any] | None) -> tuple[bool, str, str]:
    if plan is None:
        return False, "missing_plan", ""
    model = plan["model"]
    new_type = inferred_model_type(model)
    if metadata_is_blocked(model):
        return False, "gated_or_inaccessible_model", new_type
    dataset_info = plan.get("dataset", {})
    incompatible_contract_models = {
        "paligemma",
        "token_classifier",
        "unsupported_image",
        "unsupported_model",
        "vlm_chat",
        "vlm_processor",
        "zero_shot_image",
    }
    if (
        dataset in KNOWN_CONTRACT_DATASETS
        and new_type not in incompatible_contract_models
    ):
        return True, "known_dataset_contract", new_type
    if not dataset_info.get("has_gold"):
        return False, "no_validated_gold_contract", new_type
    if new_type == "extractive_qa" and dataset in QA_CONTEXT_DATASETS:
        return True, "extractive_qa_adapter", new_type
    if (
        new_type in {"vlm_chat", "vlm_processor", "paligemma"}
        and dataset in VISION_DATASETS
    ):
        return True, "standard_vlm_adapter", new_type
    if new_type == "sequence_classifier":
        task = dataset_info.get("task")
        labels = model.get("classifier_labels") or []
        if task == "multilabel_classification" and has_entailment_label(labels):
            return True, "zero_shot_nli_multilabel_adapter", new_type
    if new_type == "causal_lm" and is_causal_architecture(model):
        return True, "causal_architecture_routing", new_type
    return False, "still_incompatible", new_type


def audit_record(side: str, path: Path, row: dict[str, str]) -> dict[str, str]:
    dataset, model_name = result_identity(row)
    plan = load_matching_plan(path, model_name)
    selected, reason, new_type = recovery_decision(dataset, plan)
    old_type = plan.get("model", {}).get("model_type", "") if plan else ""
    return {
        "side": side,
        "source_task_index": str(task_index(path)),
        "dataset": dataset,
        "model": model_name,
        "status": result_status(row),
        "old_model_type": old_type,
        "inferred_model_type": new_type,
        "selected": "yes" if selected else "no",
        "recovery_reason": reason,
        "original_reason": result_reason(row),
    }


def build_audit(root: Path) -> list[dict[str, str]]:
    rows = []
    for side in ("generic", "explicit"):
        rows.extend(audit_record(side, path, row) for path, row in unsupported_results(root, side))
    return sorted(rows, key=lambda row: (row["side"], int(row["source_task_index"])))


def source_index(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["task_index"]): row for row in read_csv(path)}


def selected_tasks(audit: list[dict[str, str]], side: str, source: Path) -> list[dict[str, str]]:
    indexed = source_index(source)
    tasks = []
    for row in audit:
        if row["side"] != side or row["selected"] != "yes":
            continue
        source_row = dict(indexed[int(row["source_task_index"])])
        source_row["task_index"] = str(len(tasks))
        tasks.append(source_row)
    return tasks


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_fields(source: Path) -> list[str]:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def main() -> None:
    args = parse_args()
    audit = build_audit(args.results_root)
    generic = selected_tasks(audit, "generic", args.generic_source)
    explicit = selected_tasks(audit, "explicit", args.explicit_source)
    write_csv(args.audit_output, audit, AUDIT_FIELDS)
    write_csv(args.generic_output, generic, output_fields(args.generic_source))
    write_csv(args.explicit_output, explicit, output_fields(args.explicit_source))
    print(f"Unsupported pairs audited: {len(audit)}")
    print(f"Selected generic retries: {len(generic)}")
    print(f"Selected explicit retries: {len(explicit)}")


if __name__ == "__main__":
    main()

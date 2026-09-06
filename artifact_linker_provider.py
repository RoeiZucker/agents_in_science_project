#!/usr/bin/env python3
"""Adapt real Artifact Linker ranking output to candidate handoff version 1.

Examples:
  # Adapt a scored Artifact Linker export.
  python artifact_linker_provider.py --input provider_input.json --output candidate_handoff.json

  # Validate and print the handoff instead of writing it.
  python artifact_linker_provider.py --input provider_input.json

Input is a versioned JSON object written by ``run_selection_loop.py``. Its
``provider_config`` must name ``artifact_output`` and may name ``score_field``,
``node_metadata``, ``dataset_id``, ``method``, ``version``, and
``max_candidates``. The adapter never treats a retrieval value as a measured
evaluation score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROVIDER_INPUT_VERSION = 1
DEFAULT_SCORE_FIELDS = ("combined_score", "link_probability", "score", "retrieval_score")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    handoff = adapt_file(args.input)
    rendered = json.dumps(handoff, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


def adapt_file(input_path: Path) -> dict[str, Any]:
    raw = input_path.read_bytes()
    request = json.loads(raw)
    validate_request(request)
    config = request["provider_config"]
    artifact_path = resolve_artifact_path(input_path, config["artifact_output"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    records = artifact_records(payload, config, input_path)
    candidates, rejected = convert_records(records, request, config)
    identity = provider_identity(config)
    return {
        "handoff_version": 1,
        "mock": False,
        "owner": "artifact_linker_retrieval",
        "retrieval_agent": identity,
        "provider_version": str(config.get("version", "unknown")),
        "provider_input_checksum": hashlib.sha256(raw).hexdigest(),
        "artifact_output": str(artifact_path.resolve()),
        "artifact_output_checksum": file_checksum(artifact_path),
        "dataset": request["dataset"],
        "split": request["requested_split"],
        "round": request["round"],
        "candidates": candidates,
        "retrieval_rejections": rejected,
        "no_candidates_reason": "no_valid_candidates" if not candidates else "",
    }


def validate_request(request: Any) -> None:
    if not isinstance(request, dict):
        raise ValueError("Provider input must be a JSON object.")
    required = {"provider_input_version", "dataset", "requested_split", "round", "provider_config"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"Provider input is missing fields: {sorted(missing)}")
    if request["provider_input_version"] != PROVIDER_INPUT_VERSION:
        raise ValueError("Unsupported provider input version.")
    config = request["provider_config"]
    if not isinstance(config, dict) or not config.get("artifact_output"):
        raise ValueError("provider_config.artifact_output is required.")


def resolve_artifact_path(input_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else input_path.parent / path


def artifact_records(payload: Any, config: dict[str, Any], input_path: Path) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Artifact Linker output must be a JSON object.")
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Artifact Linker output has no candidate rows. Aggregate-only "
            "test_metrics output cannot be converted to model candidates."
        )
    scored = [row for row in rows if isinstance(row, dict) and row.get("model_name")]
    if scored:
        validate_artifact_dataset(payload, config)
        return [row for row in rows if isinstance(row, dict)]
    ranking = select_ranking_row(rows, config)
    metadata = load_node_metadata(config, input_path)
    return expand_ranked_ids(ranking, metadata)


def validate_artifact_dataset(payload: dict[str, Any], config: dict[str, Any]) -> None:
    expected = config.get("artifact_dataset")
    actual = payload.get("dataset_name") or payload.get("parsed_dataset")
    if expected and actual and expected != actual:
        raise ValueError(f"Artifact dataset mismatch: expected {expected!r}, got {actual!r}.")


def select_ranking_row(rows: list[Any], config: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in rows if isinstance(row, dict) and row.get("ranked_model_ids")]
    dataset_id = config.get("dataset_id")
    if dataset_id is not None:
        valid = [row for row in valid if str(row.get("dataset_id")) == str(dataset_id)]
    if len(valid) != 1:
        raise ValueError(
            "Raw ranked_model_ids output requires exactly one matching row; "
            "set provider_config.dataset_id when the output contains multiple datasets."
        )
    return valid[0]


def load_node_metadata(config: dict[str, Any], input_path: Path) -> dict[str, Any]:
    value = config.get("node_metadata")
    if not value:
        raise ValueError("Raw ranked_model_ids output requires provider_config.node_metadata.")
    path = resolve_artifact_path(input_path, value)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Artifact Linker node metadata must be an object.")
    return metadata


def expand_ranked_ids(row: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for rank, model_id in enumerate(row["ranked_model_ids"], start=1):
        node = metadata.get(str(model_id), metadata.get(model_id, {}))
        records.append({
            "model_id": model_id,
            "model_name": node.get("name") if isinstance(node, dict) else None,
            "rank": rank,
            "score": 1.0 / rank,
            "score_derivation": "reciprocal_rank_from_artifact_linker_order",
        })
    return records


def convert_records(
    records: list[dict[str, Any]],
    request: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    excluded = excluded_models(request)
    seen: set[str] = set()
    candidates = []
    rejected = []
    limit = positive_int(config.get("max_candidates"), 5)
    ordered = sorted(enumerate(records), key=lambda item: record_order(item[1], item[0]))
    for source_index, record in ordered:
        candidate, reason = convert_record(record, source_index, request, config)
        model = candidate.get("model", "") if candidate else record_model(record)
        normalized = model.casefold()
        if not reason and normalized in excluded:
            reason = "already_evaluated_or_excluded"
        if not reason and normalized in seen:
            reason = "duplicate_model"
        if reason:
            rejected.append(rejection(source_index, model, reason))
            continue
        seen.add(normalized)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break
    return candidates, rejected


def excluded_models(request: dict[str, Any]) -> set[str]:
    values = list(request.get("seen_models") or [])
    feedback = request.get("feedback") or {}
    constraints = feedback.get("retrieval_constraints") or {}
    values.extend(constraints.get("candidate_exclusions") or [])
    return {str(value).strip().casefold() for value in values if str(value).strip()}


def convert_record(
    record: dict[str, Any],
    source_index: int,
    request: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    model = record_model(record)
    if not model:
        return None, "missing_model"
    if is_gated(record):
        return None, "gated_model"
    rank = record_rank(record, source_index)
    score = record_score(record, config)
    if rank is None or rank < 1:
        return None, "invalid_rank"
    if score is None or not math.isfinite(score):
        return None, "non_finite_or_missing_score"
    identity = provider_identity(config)
    reason = retrieval_reason(record, identity, score, rank)
    return {
        "candidate_id": stable_candidate_id(request["dataset"], model, identity),
        "condition": f"artifact_linker_round_{request['round']}",
        "model": model,
        "rank": rank,
        "score": score,
        "source": "artifact_linker",
        "intended_use": "direct_inference",
        "reason": reason,
    }, ""


def record_model(record: dict[str, Any]) -> str:
    value = record.get("model_name") or record.get("model") or record.get("model_id_string")
    return str(value).strip() if value is not None else ""


def record_rank(record: dict[str, Any], source_index: int) -> int | None:
    value = record.get("rank", source_index + 1)
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def record_score(record: dict[str, Any], config: dict[str, Any]) -> float | None:
    fields = [config.get("score_field")] if config.get("score_field") else list(DEFAULT_SCORE_FIELDS)
    for field in fields:
        if field and record.get(field) is not None:
            try:
                return float(record[field])
            except (TypeError, ValueError):
                return None
    return None


def record_order(record: dict[str, Any], source_index: int) -> tuple[int, int]:
    rank = record_rank(record, source_index)
    return (rank is None, rank or source_index + 1)


def is_gated(record: dict[str, Any]) -> bool:
    status = str(record.get("access_status", "")).casefold()
    return record.get("gated") is True or status in {"gated", "private", "denied"}


def retrieval_reason(record: dict[str, Any], identity: str, score: float, rank: int) -> str:
    detail = record.get("reason") or record.get("score_derivation")
    suffix = f"; {detail}" if detail else ""
    return f"{identity} ranked this model #{rank} with retrieval signal {score:.8g}{suffix}."


def stable_candidate_id(dataset: str, model: str, identity: str) -> str:
    payload = f"{dataset}\0{model}\0{identity}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    prefix = re.sub(r"[^a-z0-9]+", "-", model.casefold()).strip("-")[-24:]
    return f"al-{prefix}-{digest}"


def provider_identity(config: dict[str, Any]) -> str:
    method = str(config.get("method", "artifact-linker")).strip()
    version = str(config.get("version", "unknown")).strip()
    return f"{method}@{version}"


def rejection(index: int, model: str, reason: str) -> dict[str, Any]:
    return {"source_index": index, "model": model, "reason": reason}


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()

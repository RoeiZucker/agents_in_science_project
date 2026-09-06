#!/usr/bin/env python3
"""Recommend models with Oren's ArtifactBench retrieval and the Codex CLI.

Examples:
  # Produce five candidates for an existing ArtifactBench dataset.
  python oren_local_provider.py --input provider_input.json --output candidate_handoff.json

  # Use this provider through run_selection_loop.py; the manifest supplies its
  # paths and limits, while the loop supplies feedback and seen models.
  python run_selection_loop.py --manifest config/oren_local_existing_smoke.json \
    --stage plan --runner script --max-rounds 1 --target-ranked-models 5

  # Use Oren's committed cold-start descriptions and embeddings.
  python run_selection_loop.py --manifest config/oren_iterative_18_sample200.json \
    --stage smoke --runner script --max-rounds 3 --target-ranked-models 5 \
    --provider-retries 2 --smoke-limit 200

Once the measured minimum exists, the provider runs a structured discussion
between retrieval, an independent evaluation critic, and an orchestrator.
Only unanimous agreement that improvement is unlikely requests an early stop.

This provider supports datasets already present in the local ArtifactBench
graph or Oren's committed cold-start cache. It reuses their stored description
and Voyage embedding, runs Method 1 and Method 2 locally, and uses the
authenticated Codex CLI for the final structured ranking. It never calls the
OpenAI or Voyage APIs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
INTEGRATION_ROOT = SCRIPT_DIR
DEFAULT_ARTIFACT_ROOT = INTEGRATION_ROOT / "external" / "artifact-linker"
PROVIDER_INPUT_VERSION = 1
SOURCE_METHODS = {"method1", "method2", "both"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = args.input.read_bytes()
    request = json.loads(raw)
    validate_request(request)
    handoff = recommend(request, args.input, hashlib.sha256(raw).hexdigest())
    write_json(args.output, handoff)


def validate_request(request: Any) -> None:
    if not isinstance(request, dict):
        raise ValueError("Provider input must be a JSON object.")
    required = {"provider_input_version", "dataset", "requested_split", "round", "provider_config"}
    missing = required - set(request)
    if missing:
        raise ValueError(f"Provider input is missing fields: {sorted(missing)}")
    if request["provider_input_version"] != PROVIDER_INPUT_VERSION:
        raise ValueError("Unsupported provider input version.")


def recommend(request: dict[str, Any], input_path: Path, checksum: str) -> dict[str, Any]:
    config = request["provider_config"]
    artifact_root = configured_path(config, "artifact_linker_root", DEFAULT_ARTIFACT_ROOT)
    validate_artifact_root(artifact_root)
    cache_dir = evidence_cache_dir(input_path, request, config)
    evidence = ensure_evidence(request["dataset"], artifact_root, cache_dir, config)
    allowed = evidence_models(cache_dir)
    unseen = exclude_seen(allowed, request)
    limit = min(positive_int(config.get("max_candidates"), 5), len(unseen))
    identity = provider_identity(config)
    if limit == 0:
        return empty_handoff(request, checksum, identity, cache_dir)
    review_required = continuation_review_required(request)
    prompt = ranking_prompt(request, cache_dir, unseen, limit, config)
    response = run_codex(prompt, input_path.parent, limit, config, review_required)
    candidates, rejected = convert_recommendations(response, request, unseen, limit)
    handoff = {
        "handoff_version": 1,
        "mock": False,
        "owner": "oren_automodeladvisor_retrieval",
        "retrieval_agent": identity,
        "provider_version": str(config.get("version", "unknown")),
        "provider_input_checksum": checksum,
        "dataset": request["dataset"],
        "split": request["requested_split"],
        "round": request["round"],
        "candidates": candidates,
        "retrieval_rejections": rejected,
        "evidence_cache": str(cache_dir),
        "evidence": evidence,
        "no_candidates_reason": "no_valid_candidates" if not candidates else "",
    }
    if review_required and candidates:
        handoff["continuation_review"] = deliberate_continuation(
            request, response, candidates, input_path.parent, config
        )
    return handoff


def continuation_review_required(request: dict[str, Any]) -> bool:
    review = request.get("continuation_review") or {}
    return review.get("required") is True


def configured_path(config: dict[str, Any], field: str, default: Path) -> Path:
    value = config.get(field)
    if not value:
        return default.resolve()
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def validate_artifact_root(root: Path) -> None:
    required = [
        root / "retrieval_agent" / "merge_contexts.py",
        root / "data" / "hf_graph" / "full" / "node_metadata.json",
        root / "data" / "artifact_graph_splits_v3_0314_transductive" / "train_split" / "node_metadata.json",
        root / "data" / "joint_sweep_gatv2_trans" / "trans_joint_gatv2_model_emb.pth",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Oren Artifact Linker checkout is incomplete: {missing}")


def evidence_cache_dir(
    input_path: Path, request: dict[str, Any], config: dict[str, Any]
) -> Path:
    output_root = input_path.resolve().parents[2]
    version = safe_name(str(config.get("version", "unknown")))[:16]
    return output_root / "_oren_retrieval_evidence" / safe_name(request["dataset"]) / version


def ensure_evidence(
    dataset: str, artifact_root: Path, cache_dir: Path, config: dict[str, Any]
) -> dict[str, Any]:
    outputs = evidence_outputs(cache_dir)
    if all(path.is_file() for path in outputs.values()):
        return evidence_provenance(outputs, reused=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir = artifact_root / "data" / "hf_graph" / "full"
    assets = dataset_assets(dataset, data_dir, artifact_root, config)
    write_json(cache_dir / "dataset_source.json", asset_provenance(assets))
    write_local_similarity(dataset, data_dir, cache_dir, assets)
    run_context_builders(dataset, artifact_root, data_dir, cache_dir, config, assets)
    return evidence_provenance(outputs, reused=False)


def evidence_outputs(cache_dir: Path) -> dict[str, Path]:
    return {
        "dataset_source": cache_dir / "dataset_source.json",
        "similarities": cache_dir / "similarities.json",
        "method1_context": cache_dir / "graph_context.json",
        "method2_predictions": cache_dir / "method2_predictions.json",
        "method2_context": cache_dir / "gnn_context.json",
        "merged_prompt": cache_dir / "merged_prompt.md",
    }


def dataset_assets(
    dataset: str, data_dir: Path, artifact_root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    metadata = read_json(data_dir / "node_metadata.json")
    matches = exact_dataset_ids(metadata, dataset)
    if len(matches) > 1:
        raise ValueError(f"ArtifactBench has multiple exact nodes for {dataset!r}.")
    if matches:
        return graph_dataset_assets(dataset, matches[0], metadata, data_dir)
    cache_root = configured_path(
        config,
        "precomputed_cache_root",
        artifact_root / "data" / "advisor_runs_initial_run_max14b",
    )
    return cached_dataset_assets(dataset, cache_root)


def exact_dataset_ids(metadata: dict[str, Any], dataset: str) -> list[int]:
    return [
        int(node_id)
        for node_id, value in metadata.items()
        if value.get("type") == "dataset" and value.get("name") == dataset
    ]


def graph_dataset_assets(
    dataset: str, dataset_id: int, metadata: dict[str, Any], data_dir: Path
) -> dict[str, Any]:
    summary = str(metadata[str(dataset_id)].get("info", "")).strip()
    if not summary:
        raise ValueError(f"ArtifactBench dataset has no stored description: {dataset}")
    embedding_path = data_dir / "node_embeddings_voyage.npy"
    embeddings = np.load(embedding_path).astype(np.float32)
    return {
        "mode": "artifactbench_graph",
        "dataset": dataset,
        "dataset_id": dataset_id,
        "summary": summary,
        "embedding": unit_vector(embeddings[dataset_id]),
        "summary_path": "",
        "embedding_path": str(embedding_path),
    }


def cached_dataset_assets(dataset: str, cache_root: Path) -> dict[str, Any]:
    dataset_dir = cache_root / safe_name(dataset)
    summary_path = dataset_dir / "summary.txt"
    embedding_path = dataset_dir / "summary_voyage.npy"
    if not summary_path.is_file() or not embedding_path.is_file():
        raise ValueError(
            f"Dataset {dataset!r} is absent from ArtifactBench and has no complete "
            f"committed cold-start cache under {dataset_dir}."
        )
    summary = summary_path.read_text(encoding="utf-8").strip()
    embedding = np.load(embedding_path).astype(np.float32).reshape(-1)
    if not summary:
        raise ValueError(f"Committed dataset description is empty: {summary_path}")
    return {
        "mode": "committed_cold_start_cache",
        "dataset": dataset,
        "dataset_id": None,
        "summary": summary,
        "embedding": unit_vector(embedding),
        "summary_path": str(summary_path),
        "embedding_path": str(embedding_path),
    }


def asset_provenance(assets: dict[str, Any]) -> dict[str, Any]:
    paths = [assets.get("summary_path"), assets.get("embedding_path")]
    files = {
        Path(path).name: {"path": str(path), "sha256": file_checksum(Path(path))}
        for path in paths
        if path and Path(path).is_file()
    }
    return {
        "mode": assets["mode"],
        "dataset": assets["dataset"],
        "dataset_id": assets["dataset_id"],
        "files": files,
    }


def write_local_similarity(
    dataset: str, data_dir: Path, cache_dir: Path, assets: dict[str, Any]
) -> None:
    metadata = read_json(data_dir / "node_metadata.json")
    embeddings = np.load(data_dir / "node_embeddings_voyage.npy").astype(np.float32)
    query = assets["embedding"]
    rows = similarity_rows(metadata, embeddings, query)
    (cache_dir / "summary.txt").write_text(assets["summary"] + "\n", encoding="utf-8")
    np.save(cache_dir / "summary_voyage.npy", query)
    write_json(cache_dir / "similarities.json", rows)
    write_similarity_csv(cache_dir / "method1.csv", dataset, rows)


def exact_dataset_id(metadata: dict[str, Any], dataset: str) -> int:
    matches = exact_dataset_ids(metadata, dataset)
    if len(matches) != 1:
        raise ValueError(
            f"Local-only retrieval requires exactly one existing ArtifactBench node for "
            f"{dataset!r}; found {len(matches)}."
        )
    return matches[0]


def unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("ArtifactBench query embedding is not a finite nonzero vector.")
    return vector / norm


def similarity_rows(
    metadata: dict[str, Any], embeddings: np.ndarray, query: np.ndarray
) -> list[dict[str, Any]]:
    dataset_ids = sorted(
        int(node_id) for node_id, value in metadata.items() if value.get("type") == "dataset"
    )
    matrix = embeddings[dataset_ids]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.maximum(norms, 1e-8)
    scores = normalized @ query
    order = np.argsort(-scores)
    return [
        {
            "node_id": dataset_ids[index],
            "name": metadata[str(dataset_ids[index])]["name"],
            "cosine_similarity": float(scores[index]),
        }
        for index in order
    ]


def write_similarity_csv(path: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "method", "rank", "query_dataset", "dataset_node_id",
            "dataset_name", "cosine_similarity",
        ])
        for rank, row in enumerate(rows, 1):
            writer.writerow([
                "method1_cosine_retrieval", rank, dataset, row["node_id"],
                row["name"], row["cosine_similarity"],
            ])


def run_context_builders(
    dataset: str, artifact_root: Path, data_dir: Path,
    cache_dir: Path, config: dict[str, Any], assets: dict[str, Any],
) -> None:
    retrieval = artifact_root / "retrieval_agent"
    max_params = positive_float(config.get("max_params_b"), 14.0)
    run_step("method1_context", [
        sys.executable, retrieval / "method1_cosine_retrieval" / "build_graph_context.py",
        "--similarities", cache_dir / "similarities.json",
        "--query-name", dataset,
        "--query-summary-file", cache_dir / "summary.txt",
        "--data-dir", data_dir,
        "--top-k", positive_int(config.get("method1_top_k"), 10),
        "--max-models", positive_int(config.get("models_per_neighbor"), 15),
        "--max-params-b", max_params,
        "--out-text", cache_dir / "graph_context.md",
        "--out-json", cache_dir / "graph_context.json",
    ], cache_dir, None)
    run_method2(dataset, artifact_root, cache_dir, config, assets)
    run_step("method2_context", [
        sys.executable, retrieval / "method2_gnn_inference" / "build_gnn_context.py",
        "--predictions", cache_dir / "method2_predictions.json",
        "--data-dir", data_dir,
        "--top-k", positive_int(config.get("method2_context_top_k"), 15),
        "--max-evidence", positive_int(config.get("max_evidence"), 5),
        "--max-params-b", max_params,
        "--out-text", cache_dir / "gnn_context.md",
        "--out-json", cache_dir / "gnn_context.json",
    ], cache_dir, None)
    constraint = f"Only recommend direct-inference models with at most {max_params:g}B parameters."
    run_step("merge", [
        sys.executable, retrieval / "merge_contexts.py",
        "--method1-md", cache_dir / "graph_context.md",
        "--method1-json", cache_dir / "graph_context.json",
        "--method2-md", cache_dir / "gnn_context.md",
        "--method2-json", cache_dir / "gnn_context.json",
        "--num-recommendations", positive_int(config.get("max_candidates"), 5),
        "--constraint", constraint,
        "--out", cache_dir / "merged_prompt.md",
    ], cache_dir, None)


def run_method2(
    dataset: str, artifact_root: Path, cache_dir: Path,
    config: dict[str, Any], assets: dict[str, Any],
) -> None:
    script = artifact_root / "retrieval_agent" / "method2_gnn_inference" / "rank_hf_dataset_combined.py"
    data = artifact_root / "data"
    split_dir = data / "artifact_graph_splits_v3_0314_transductive"
    if assets["mode"] == "committed_cold_start_cache":
        split_dir = prepare_augmented_split(split_dir, cache_dir, assets)
    environment = {**os.environ, "AGENTS_PROJECT_ROOT": str(INTEGRATION_ROOT)}
    run_step("method2_rank", [
        sys.executable, script, dataset,
        "--top-k", positive_int(config.get("method2_rank_pool"), 500),
        "--split-dir", split_dir,
        "--model-path", data / "joint_sweep_gatv2_trans" / "trans_joint_gatv2_model_emb.pth",
        "--output", cache_dir / "method2_predictions.json",
        "--csv", cache_dir / "method2.csv",
    ], cache_dir, environment)


def prepare_augmented_split(
    source: Path, cache_dir: Path, assets: dict[str, Any]
) -> Path:
    source_metadata = read_json(source / "train_split" / "node_metadata.json")
    matches = exact_dataset_ids(source_metadata, assets["dataset"])
    if len(matches) > 1:
        raise ValueError(f"GNN split has multiple exact nodes for {assets['dataset']!r}.")
    if matches:
        return source
    target = cache_dir / "augmented_split"
    if augmented_split_matches(target, assets["dataset"]):
        return target
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    link_split_files(source, target)
    append_cached_dataset(source, target, assets)
    return target


def augmented_split_matches(target: Path, dataset: str) -> bool:
    metadata_path = target / "train_split" / "node_metadata.json"
    voyage_path = target / "node_embeddings_voyage.npy"
    if not metadata_path.is_file() or not voyage_path.is_file():
        return False
    metadata = read_json(metadata_path)
    return len(exact_dataset_ids(metadata, dataset)) == 1


def link_split_files(source: Path, target: Path) -> None:
    for item in source.resolve().iterdir():
        if item.name == "node_embeddings_voyage.npy":
            continue
        if item.is_dir():
            link_split_directory(item, target / item.name)
        else:
            (target / item.name).symlink_to(item)


def link_split_directory(source: Path, target: Path) -> None:
    target.mkdir()
    for item in source.iterdir():
        if item.name != "node_metadata.json":
            (target / item.name).symlink_to(item)


def append_cached_dataset(
    source: Path, target: Path, assets: dict[str, Any]
) -> None:
    source = source.resolve()
    voyage = np.load(source / "node_embeddings_voyage.npy").astype(np.float32)
    embedding = assets["embedding"].reshape(1, -1)
    if voyage.shape[1] != embedding.shape[1]:
        raise ValueError(
            f"Cold-start embedding dimension {embedding.shape[1]} does not match "
            f"the GNN split dimension {voyage.shape[1]}."
        )
    new_id = voyage.shape[0]
    entry = {
        "type": "dataset",
        "name": assets["dataset"],
        "downloads": 0,
        "info": assets["summary"],
        "source": "oren_committed_cold_start_cache",
    }
    for split_name in ("train_split", "test_split"):
        metadata = read_json(source / split_name / "node_metadata.json")
        if max(map(int, metadata)) + 1 != new_id:
            raise ValueError("GNN split metadata and embeddings are not index-aligned.")
        metadata[str(new_id)] = entry
        write_json(target / split_name / "node_metadata.json", metadata)
    np.save(target / "node_embeddings_voyage.npy", np.vstack([voyage, embedding]))


def run_step(name: str, command: list[Any], workdir: Path, environment: dict[str, str] | None) -> None:
    rendered = [str(item) for item in command]
    completed = subprocess.run(
        rendered, cwd=workdir, env=environment, text=True, capture_output=True
    )
    (workdir / f"{name}_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
    (workdir / f"{name}_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "")[-2000:]
        raise RuntimeError(f"Oren retrieval step {name} failed ({completed.returncode}): {detail}")


def evidence_provenance(outputs: dict[str, Path], reused: bool) -> dict[str, Any]:
    return {
        "reused": reused,
        "files": {
            name: {"path": str(path), "sha256": file_checksum(path)}
            for name, path in outputs.items()
        },
    }


def evidence_models(cache_dir: Path) -> set[str]:
    method1 = read_json(cache_dir / "graph_context.json")
    method2 = read_json(cache_dir / "gnn_context.json")
    names = {
        str(model["name"])
        for neighbor in method1.get("neighbor_datasets", [])
        for model in neighbor.get("models", [])
    }
    names.update(
        str(model["name"]) for model in method1.get("models_on_multiple_neighbors", [])
    )
    names.update(str(model["name"]) for model in method2.get("recommendations", []))
    return {name for name in names if name}


def exclude_seen(models: set[str], request: dict[str, Any]) -> list[str]:
    excluded = {str(model).casefold() for model in request.get("seen_models") or []}
    feedback = request.get("feedback") or {}
    constraints = feedback.get("retrieval_constraints") or {}
    excluded.update(str(model).casefold() for model in constraints.get("candidate_exclusions") or [])
    return sorted(model for model in models if model.casefold() not in excluded)


def ranking_prompt(
    request: dict[str, Any], cache_dir: Path, allowed: list[str],
    limit: int, config: dict[str, Any],
) -> str:
    evidence = (cache_dir / "merged_prompt.md").read_text(encoding="utf-8")
    feedback = json.dumps(request.get("feedback"), indent=2, ensure_ascii=False)
    retry = json.dumps(request.get("retrieval_retry"), indent=2, ensure_ascii=False)
    dataset_context = json.dumps(request.get("dataset_context"), indent=2, ensure_ascii=False)
    allowed_text = "\n".join(f"- {model}" for model in allowed)
    continuation = continuation_instruction(request)
    return "\n".join([
        evidence,
        "",
        "# Runtime selection instructions",
        f"Return exactly {limit} NEW candidate models for dataset {request['dataset']}.",
        "Use the JSON output schema supplied by the caller, not the CSV format above.",
        "Do not call tools or inspect external information; decide only from this prompt.",
        "Real measurements and failure feedback outweigh GNN predictions.",
        "Choose only exact model IDs from the allowed list below.",
        "Never repeat a model named in the retry rejections or evaluation history.",
        "Prefer models evaluable with the same metric and protocol family as prior comparable results.",
        "Prefer checkpoints suitable for direct inference and the dataset task.",
        "Avoid guard/reward/embedding/base-only models when an instruction checkpoint is available.",
        "Standard Transformers checkpoints and text-only GGUF are supported.",
        "For an image dataset, require a standard vision-language checkpoint; vision GGUF is unsupported.",
        f"Maximum parameter count: {positive_float(config.get('max_params_b'), 14.0):g}B.",
        continuation,
        "",
        "## Dataset evaluation context",
        dataset_context,
        "",
        "## Previous measured feedback",
        feedback,
        "",
        "## Retry context",
        retry,
        "",
        "## Allowed unseen model IDs",
        allowed_text,
    ])


def continuation_instruction(request: dict[str, Any]) -> str:
    if not continuation_review_required(request):
        return "The minimum ranked-model pool is not complete; candidate retrieval must continue."
    return (
        "The minimum ranked-model pool already exists. In improvement_assessment, "
        "argue whether at least one proposed candidate has a realistic chance to enter "
        "or improve the current top-ranked pool after evaluation. Be candid: measured "
        "results outweigh retrieval scores, and uncertainty should be stated explicitly."
    )


def run_codex(
    prompt: str, workdir: Path, limit: int, config: dict[str, Any],
    review_required: bool = False,
) -> dict[str, Any]:
    return run_structured_codex(
        prompt, workdir, recommendation_schema(limit, review_required),
        "retrieval", config,
    )


def run_structured_codex(
    prompt: str, workdir: Path, schema: dict[str, Any], role: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    schema_name, response_name = codex_output_names(role)
    schema_path = workdir / schema_name
    response_path = workdir / response_name
    write_json(schema_path, schema)
    response_path.unlink(missing_ok=True)
    codex = str(config.get("codex_bin", "/usr/local/bin/codex"))
    command = [codex, "-C", str(workdir)]
    if config.get("codex_bypass_sandbox"):
        command.append("--dangerously-bypass-approvals-and-sandbox")
    command.extend([
        "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
        "--output-schema", str(schema_path),
        "--output-last-message", str(response_path), "-",
    ])
    timeout = positive_int(config.get("codex_timeout"), 300)
    completed = subprocess.run(
        command, input=prompt, text=True, capture_output=True,
        cwd=workdir, timeout=timeout,
    )
    log_prefix = "codex" if role == "retrieval" else f"{role}_codex"
    (workdir / f"{log_prefix}_stdout.log").write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (workdir / f"{log_prefix}_stderr.log").write_text(
        completed.stderr or "", encoding="utf-8"
    )
    (workdir / f"{log_prefix}_prompt.md").write_text(prompt, encoding="utf-8")
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "")[-2000:]
        raise RuntimeError(f"Codex ranking failed ({completed.returncode}): {detail}")
    if not response_path.is_file():
        raise RuntimeError("Codex did not write its final structured response.")
    return json.loads(response_path.read_text(encoding="utf-8"))


def codex_output_names(role: str) -> tuple[str, str]:
    if role == "retrieval":
        return "recommendation_schema.json", "codex_recommendations.json"
    return f"{role}_schema.json", f"{role}_response.json"


def recommendation_schema(
    limit: int, review_required: bool = False
) -> dict[str, Any]:
    item = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "rank": {"type": "integer", "minimum": 1, "maximum": limit},
            "model_name": {"type": "string", "minLength": 1},
            "recommendation_score": {"type": "number", "minimum": 0, "maximum": 1},
            "source_method": {"type": "string", "enum": sorted(SOURCE_METHODS)},
            "reasoning": {"type": "string", "minLength": 1},
        },
        "required": [
            "rank", "model_name", "recommendation_score", "source_method", "reasoning"
        ],
    }
    properties: dict[str, Any] = {
        "recommendations": {
            "type": "array", "minItems": limit, "maxItems": limit, "items": item
        }
    }
    required = ["recommendations"]
    if review_required:
        properties["improvement_assessment"] = improvement_assessment_schema()
        required.append("improvement_assessment")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def improvement_assessment_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "improvement_probable": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string", "minLength": 1},
        },
        "required": ["improvement_probable", "confidence", "reasoning"],
    }


def orchestrator_decision_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decision": {"type": "string", "enum": ["continue", "stop"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string", "minLength": 1},
        },
        "required": ["decision", "confidence", "reasoning"],
    }


def deliberate_continuation(
    request: dict[str, Any], response: dict[str, Any],
    candidates: list[dict[str, Any]], workdir: Path, config: dict[str, Any],
) -> dict[str, Any]:
    retrieval = response["improvement_assessment"]
    critic = run_structured_codex(
        critic_prompt(request, candidates, retrieval), workdir,
        {"$schema": "https://json-schema.org/draft/2020-12/schema", **improvement_assessment_schema()},
        "evaluation_critic", config,
    )
    orchestrator = run_structured_codex(
        orchestrator_prompt(request, candidates, retrieval, critic), workdir,
        orchestrator_decision_schema(), "orchestrator", config,
    )
    return consensus_review(retrieval, critic, orchestrator)


def critic_prompt(
    request: dict[str, Any], candidates: list[dict[str, Any]],
    retrieval: dict[str, Any],
) -> str:
    evidence = {
        "continuation_review": request.get("continuation_review"),
        "measured_feedback": request.get("feedback"),
        "proposed_candidates": candidates,
        "retrieval_assessment": retrieval,
    }
    return "\n".join([
        "You are the independent evaluation critic in a model-selection discussion.",
        "Assess whether any proposed candidate is realistically likely to enter or improve",
        "the measured top-model pool. Challenge unsupported retrieval claims and account",
        "for small-sample uncertainty. Do not call tools. Return only the supplied schema.",
        json.dumps(evidence, indent=2, ensure_ascii=False),
    ])


def orchestrator_prompt(
    request: dict[str, Any], candidates: list[dict[str, Any]],
    retrieval: dict[str, Any], critic: dict[str, Any],
) -> str:
    discussion = {
        "continuation_review": request.get("continuation_review"),
        "proposed_candidates": candidates,
        "retrieval_agent": retrieval,
        "evaluation_critic": critic,
    }
    return "\n".join([
        "You are the orchestrator resolving a model-selection continuation discussion.",
        "Decide whether evaluating the proposed candidates is likely to improve or enter",
        "the current top pool. Prefer continue under disagreement or meaningful uncertainty.",
        "Choose stop only when further improvement is genuinely improbable. Do not call tools.",
        "Return only the supplied schema.",
        json.dumps(discussion, indent=2, ensure_ascii=False),
    ])


def consensus_review(
    retrieval: dict[str, Any], critic: dict[str, Any],
    orchestrator: dict[str, Any],
) -> dict[str, Any]:
    consensus = bool(
        retrieval.get("improvement_probable") is False
        and critic.get("improvement_probable") is False
        and orchestrator.get("decision") == "stop"
    )
    return {
        "required": True,
        "decision": "stop" if consensus else "continue",
        "consensus": consensus,
        "retrieval": retrieval,
        "critic": critic,
        "orchestrator": orchestrator,
    }


def convert_recommendations(
    response: dict[str, Any], request: dict[str, Any], allowed: list[str], limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    exact = {model.casefold(): model for model in allowed}
    rows = sorted(response.get("recommendations") or [], key=lambda row: row.get("rank", 10**9))
    candidates = []
    rejected = []
    seen = set()
    for row in rows:
        raw_model = str(row.get("model_name", "")).strip()
        model = exact.get(raw_model.casefold())
        reason = invalid_recommendation(row, model, seen)
        if reason:
            rejected.append({"model": raw_model, "reason": reason})
            continue
        seen.add(model.casefold())
        rank = len(candidates) + 1
        candidates.append(candidate(request, row, model, rank))
        if len(candidates) == limit:
            break
    return candidates, rejected


def invalid_recommendation(
    row: dict[str, Any], model: str | None, seen: set[str]
) -> str:
    if model is None:
        return "model_not_in_evidence_or_excluded"
    if model.casefold() in seen:
        return "duplicate_model"
    try:
        score = float(row.get("recommendation_score"))
    except (TypeError, ValueError):
        return "invalid_score"
    if not math.isfinite(score) or not 0 <= score <= 1:
        return "invalid_score"
    if row.get("source_method") not in SOURCE_METHODS:
        return "invalid_source_method"
    if not str(row.get("reasoning", "")).strip():
        return "missing_reasoning"
    return ""


def candidate(
    request: dict[str, Any], row: dict[str, Any], model: str, rank: int
) -> dict[str, Any]:
    identity = f"{request['dataset']}\0{model}\0{request['round']}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return {
        "candidate_id": f"oren-{safe_name(model)[-28:]}-{digest}",
        "condition": f"oren_automodeladvisor_round_{request['round']}",
        "model": model,
        "rank": rank,
        "score": float(row["recommendation_score"]),
        "source": f"oren_automodeladvisor_{row['source_method']}_codex",
        "intended_use": "direct_inference",
        "reason": str(row["reasoning"]).strip(),
    }


def empty_handoff(
    request: dict[str, Any], checksum: str, identity: str, cache_dir: Path
) -> dict[str, Any]:
    return {
        "handoff_version": 1,
        "mock": False,
        "owner": "oren_automodeladvisor_retrieval",
        "retrieval_agent": identity,
        "provider_version": str(request["provider_config"].get("version", "unknown")),
        "provider_input_checksum": checksum,
        "dataset": request["dataset"],
        "split": request["requested_split"],
        "round": request["round"],
        "candidates": [],
        "retrieval_rejections": [],
        "evidence_cache": str(cache_dir),
        "no_candidates_reason": "no_unseen_evidence_models",
    }


def provider_identity(config: dict[str, Any]) -> str:
    version = str(config.get("version", "unknown"))
    return f"oren-automodeladvisor-local-codex@{version}"


def positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def positive_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

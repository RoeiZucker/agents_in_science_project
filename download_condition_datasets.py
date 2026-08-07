#!/usr/bin/env python3
"""Download/cache Hugging Face datasets from a conditions CSV before evaluation.

Examples:
  python download_condition_datasets.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --trust-remote-code

  python download_condition_datasets.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --split validation \
    --trust-remote-code

  python download_condition_datasets.py \
    --dataset stanfordnlp/imdb \
    --project-root runtime \
    --split train \
    --split test

  python download_condition_datasets.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --include-models \
    --trust-remote-code
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import snapshot_download


COMMON_SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--subset", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--include-models", action="store_true", help="Also cache selected model snapshots from the conditions CSV.")
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root / "eval_results" / "_dataset_downloads").resolve()
    set_cache_env(project_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = selected_datasets(args)
    dataset_results = download_all(datasets, args)
    model_results = download_models(selected_models(args), project_root)
    results = {"datasets": dataset_results, "models": model_results}
    write_json(output_root / "dataset_downloads.json", results)
    print(json.dumps(summary(results, output_root), indent=2))
    if failed(results):
        sys.exit(1)


def set_cache_env(project_root: Path) -> None:
    os.environ["HF_HOME"] = str(project_root / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = str(project_root / ".hf_datasets_cache")
    os.environ["TRANSFORMERS_CACHE"] = str(project_root / ".hf_cache" / "hub")


def selected_datasets(args: argparse.Namespace) -> list[str]:
    names = dataset_names(args)
    if args.dataset:
        allowed = set(args.dataset)
        names = [name for name in names if name in allowed]
    if args.limit_datasets:
        names = names[: args.limit_datasets]
    if not names:
        raise ValueError("No datasets selected. Pass --dataset or --conditions-csv.")
    return names


def dataset_names(args: argparse.Namespace) -> list[str]:
    names = list(args.dataset)
    if args.conditions_csv:
        names.extend(row["query_dataset"] for row in selected_rows(args))
    return unique(names)


def selected_models(args: argparse.Namespace) -> list[str]:
    if not args.include_models or not args.conditions_csv:
        return []
    return unique([row["model_name"] for row in selected_rows(args)])


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = rows_from_csv(args.conditions_csv)
    if args.condition:
        rows = [row for row in rows if row.get("condition") in set(args.condition)]
    if args.dataset:
        rows = [row for row in rows if row.get("query_dataset") in set(args.dataset)]
    if args.limit_pairs:
        rows = rows[: args.limit_pairs]
    return rows


def rows_from_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "query_dataset" not in rows[0]:
        raise ValueError("conditions CSV must include query_dataset")
    if "model_name" not in rows[0]:
        raise ValueError("conditions CSV must include model_name when downloading models")
    return rows


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def download_all(datasets: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    results = []
    for dataset in datasets:
        result = download_dataset(dataset, args)
        results.append(result)
        if result["status"] == "failed" and not args.continue_on_error:
            break
    return results


def download_dataset(dataset: str, args: argparse.Namespace) -> dict[str, Any]:
    result = {"dataset": dataset, "subset": args.subset or "", "splits": [], "failures": [], "status": "ok"}
    for split in target_splits(dataset, args):
        try:
            ds = load_dataset(dataset, args.subset or None, split=split, trust_remote_code=args.trust_remote_code)
            result["splits"].append({"split": split, "rows": len(ds), "status": "ok"})
        except Exception as exc:
            result["failures"].append(failure(split, exc))
            result["status"] = "failed"
            if not args.continue_on_error:
                break
    return result


def target_splits(dataset: str, args: argparse.Namespace) -> list[str]:
    if args.split:
        return args.split
    try:
        splits = get_dataset_split_names(dataset, args.subset or None, trust_remote_code=args.trust_remote_code)
    except Exception:
        return list(COMMON_SPLITS)
    return splits or list(COMMON_SPLITS)


def failure(split: str, exc: Exception) -> dict[str, str]:
    return {"split": split, "type": type(exc).__name__, "message": str(exc)}


def download_models(models: list[str], project_root: Path) -> list[dict[str, Any]]:
    return [download_model(model, project_root) for model in models]


def download_model(model: str, project_root: Path) -> dict[str, Any]:
    try:
        cache_dir = model_cache_dir(project_root)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(repo_id=model, cache_dir=str(cache_dir))
        return {"model": model, "snapshot": path, "status": "ok"}
    except Exception as exc:
        return {"model": model, "status": "failed", "type": type(exc).__name__, "message": str(exc)}


def model_cache_dir(project_root: Path) -> Path:
    return project_root / ".hf_cache" / "hub"


def failed(results: dict[str, list[dict[str, Any]]]) -> bool:
    dataset_failed = any(result["status"] == "failed" for result in results["datasets"])
    model_failed = any(result["status"] == "failed" for result in results["models"])
    return dataset_failed or model_failed


def summary(results: dict[str, list[dict[str, Any]]], output_root: Path) -> dict[str, Any]:
    datasets = results["datasets"]
    models = results["models"]
    return {
        "output_root": str(output_root),
        "datasets": len(datasets),
        "downloaded_splits": sum(len(result["splits"]) for result in datasets),
        "failed_datasets": sum(1 for result in datasets if result["status"] == "failed"),
        "models": len(models),
        "downloaded_models": sum(1 for result in models if result["status"] == "ok"),
        "failed_models": sum(1 for result in models if result["status"] == "failed"),
        "report": str(output_root / "dataset_downloads.json"),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

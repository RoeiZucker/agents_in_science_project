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
    parser.add_argument("--limit-datasets", type=int, default=0)
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
    results = download_all(datasets, args)
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
        names.extend(names_from_csv(args.conditions_csv, set(args.condition)))
    return unique(names)


def names_from_csv(path: Path, conditions: set[str]) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "query_dataset" not in rows[0]:
        raise ValueError("conditions CSV must include query_dataset")
    if conditions:
        rows = [row for row in rows if row.get("condition") in conditions]
    return [row["query_dataset"] for row in rows]


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


def failed(results: list[dict[str, Any]]) -> bool:
    return any(result["status"] == "failed" for result in results)


def summary(results: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "datasets": len(results),
        "downloaded_splits": sum(len(result["splits"]) for result in results),
        "failed_datasets": sum(1 for result in results if result["status"] == "failed"),
        "report": str(output_root / "dataset_downloads.json"),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

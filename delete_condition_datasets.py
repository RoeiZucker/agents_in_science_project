#!/usr/bin/env python3
"""Delete cached Hugging Face datasets used by conditions/evaluation runs.

Examples:
  python delete_condition_datasets.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime

  python delete_condition_datasets.py \
    --dataset ImperialCollegeLondon/health_fact \
    --project-root runtime

  python delete_condition_datasets.py \
    --project-root runtime \
    --all-datasets-cache
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from eval_agent_core import safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--all-datasets-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root / "eval_results" / "_dataset_downloads").resolve()
    targets = deletion_targets(project_root, args)
    deleted = delete_targets(targets, args.dry_run)
    report = {"project_root": str(project_root), "dry_run": args.dry_run, "targets": deleted}
    write_json(output_root / "dataset_cache_deletions.json", report)
    print(json.dumps(report, indent=2))


def deletion_targets(project_root: Path, args: argparse.Namespace) -> list[Path]:
    if args.all_datasets_cache:
        return existing([datasets_cache(project_root), dataset_modules(project_root)])
    names = selected_datasets(args)
    return existing(matching_cache_paths(project_root, names))


def selected_datasets(args: argparse.Namespace) -> list[str]:
    names = dataset_names(args)
    if args.dataset:
        allowed = set(args.dataset)
        names = [name for name in names if name in allowed]
    if args.limit_datasets:
        names = names[: args.limit_datasets]
    if not names:
        raise ValueError("No datasets selected. Pass --dataset, --conditions-csv, or --all-datasets-cache.")
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


def matching_cache_paths(project_root: Path, datasets: list[str]) -> list[Path]:
    roots = existing([datasets_cache(project_root), dataset_modules(project_root)])
    names = {name for dataset in datasets for name in cache_names(dataset)}
    return [path for root in roots for path in root.iterdir() if path.name in names]


def cache_names(dataset: str) -> set[str]:
    owner_repo = dataset.replace("/", "___")
    return {dataset, owner_repo, safe_name(dataset), dataset.replace("/", "--"), dataset.split("/")[-1]}


def datasets_cache(project_root: Path) -> Path:
    return project_root / ".hf_datasets_cache"


def dataset_modules(project_root: Path) -> Path:
    return project_root / ".hf_cache" / "modules" / "datasets_modules" / "datasets"


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def delete_targets(targets: list[Path], dry_run: bool) -> list[dict[str, Any]]:
    deleted = []
    for path in targets:
        deleted.append({"path": str(path), "type": path_type(path), "deleted": not dry_run})
        if not dry_run:
            delete_path(path)
    return deleted


def path_type(path: Path) -> str:
    return "directory" if path.is_dir() else "file"


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

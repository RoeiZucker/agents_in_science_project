#!/usr/bin/env python3
"""Freeze measured preflight rankings into a static comparison manifest.

Examples:
  python freeze_ranked_candidates.py \
    --ranked-models runtime/preflight/ranked_models.csv \
    --source-manifest config/oren_local_cached_18_final.json \
    --output-dir runtime/frozen_comparison --target 5

  python freeze_ranked_candidates.py \
    --ranked-models runtime/preflight/ranked_models.csv \
    --ranked-models runtime/corrected_retest/ranked_models.csv \
    --source-manifest config/oren_local_cached_18_final.json \
    --output-dir runtime/frozen_corrected --target 5

When more than one ranking file is supplied, later files replace earlier rows
for the datasets they contain. This lets a focused corrected retest supersede
the original preflight without changing unaffected datasets. A source dataset
may omit evaluation_context; the frozen manifest then leaves context discovery
to the selected final-evaluation runner.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_selection_loop import load_manifest, resolve_path, safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-models", required=True, type=Path, action="append")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target < 1:
        raise ValueError("--target must be at least 1")
    rows = read_ranked_sources(args.ranked_models)
    specs = load_manifest(args.source_manifest)
    grouped = group_rows(rows)
    missing = [
        spec["dataset"]
        for spec in specs
        if len(grouped.get(spec["dataset"], [])) < args.target
    ]
    if missing and not args.allow_incomplete:
        raise ValueError(
            "Fewer than the target measured models for: " + ", ".join(missing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = ranked_sources(args.ranked_models)
    checksum = combined_source_checksum(sources)
    frozen_specs = freeze_specs(specs, grouped, args, checksum, sources)
    manifest_path = args.output_dir / "frozen_manifest.json"
    write_json(
        manifest_path,
        {
            "manifest_version": 1,
            "purpose": "Static candidates frozen after protocol-pinned measured preflight.",
            "ranked_models_sources": sources,
            "combined_ranked_models_checksum": checksum,
            "target_ranked_models": args.target,
            "datasets": frozen_specs,
        },
    )
    report = {
        "manifest": str(manifest_path),
        "datasets": len(frozen_specs),
        "missing_or_incomplete": missing,
        "target": args.target,
        "ranked_models_sources": sources,
    }
    write_json(args.output_dir / "freeze_report.json", report)
    print(json.dumps(report, indent=2))
    return 0


def read_ranked(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_ranked_sources(paths: list[Path]) -> list[dict[str, str]]:
    by_dataset: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        current: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in read_ranked(path):
            current[row["dataset"]].append(row)
        by_dataset.update(current)
    return [row for values in by_dataset.values() for row in values]


def ranked_sources(paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"path": str(path.resolve()), "checksum": file_checksum(path)}
        for path in paths
    ]


def combined_source_checksum(sources: list[dict[str, str]]) -> str:
    content = json.dumps(sources, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def group_rows(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["final_rank"]))
    return grouped


def freeze_specs(
    specs: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, str]]],
    args: argparse.Namespace,
    checksum: str,
    sources: list[dict[str, str]],
) -> list[dict[str, Any]]:
    frozen = []
    for spec in specs:
        rows = grouped.get(spec["dataset"], [])[: args.target]
        if not rows:
            continue
        handoff_path = write_handoff(spec, rows, args, checksum, sources)
        frozen_spec = {
            "dataset": spec["dataset"],
            "split": spec.get("split", "auto"),
            "rounds": [{"round": 1, "candidates": str(handoff_path)}],
        }
        if spec.get("evaluation_context"):
            context_path = resolve_path(
                args.source_manifest, spec["evaluation_context"]
            ).resolve()
            frozen_spec["evaluation_context"] = str(context_path)
        frozen.append(frozen_spec)
    return frozen


def write_handoff(
    spec: dict[str, Any],
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    checksum: str,
    sources: list[dict[str, str]],
) -> Path:
    path = args.output_dir / "candidate_handoffs" / f"{safe_name(spec['dataset'])}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    handoff = {
        "handoff_version": 1,
        "owner": "frozen_measured_preflight",
        "mock": False,
        "retrieval_agent": f"frozen-preflight@{checksum[:12]}",
        "dataset": spec["dataset"],
        "split": spec.get("split", "auto"),
        "round": 1,
        "frozen_from": {
            "ranked_models_sources": sources,
            "combined_checksum": checksum,
        },
        "candidates": [frozen_candidate(row, rank) for rank, row in enumerate(rows, 1)],
    }
    write_json(path, handoff)
    return path.resolve()


def frozen_candidate(row: dict[str, str], rank: int) -> dict[str, Any]:
    score = float(row.get("retrieval_score") or 0.0)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite retrieval score for {row.get('model')}")
    return {
        "candidate_id": row["candidate_id"],
        "condition": "frozen_measured_preflight",
        "model": row["model"],
        "rank": rank,
        "score": score,
        "source": row.get("source") or "preflight_selection",
        "intended_use": row.get("intended_use") or "direct_inference",
        "reason": row.get("reason") or "Selected by measured preflight.",
        "preflight_score": row.get("score", ""),
        "preflight_metric": row.get("metric", ""),
        "preflight_protocol": row.get("evaluation_protocol", ""),
    }


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

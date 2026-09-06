"""Smoke-to-full gating for one selection-loop round.

This module stays separate from provider retrieval and winner selection. It
imports the orchestrator lazily so the public CLI and its existing helper API
remain backward compatible.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def run_smoke_then_full(
    spec: dict[str, Any],
    handoff: dict[str, Any],
    args: Any,
    _root: Path,
    round_dir: Path,
) -> dict[str, Any]:
    import run_selection_loop as loop

    dataset = spec["dataset"]
    split = spec.get("split", "auto")
    conditions = round_dir / "conditions.csv"
    loop.write_conditions(conditions, handoff)
    context_path = loop.prepare_evaluation_context(
        spec, handoff, args, round_dir, "smoke_evaluation_context.json"
    )
    smoke = run_smoke(
        args, dataset, split, conditions, round_dir, context_path
    )
    smoke_rows = read_smoke_rows(loop, dataset, split, handoff, smoke["results"])
    passing_ids = successful_candidate_ids(smoke_rows)
    if not passing_ids:
        return failed_smoke_record(
            loop, dataset, split, handoff, round_dir, smoke, smoke_rows
        )
    return run_full_for_passing(
        loop, dataset, split, handoff, args, round_dir,
        smoke, smoke_rows, passing_ids, spec,
    )


def run_smoke(
    args: Any,
    dataset: str,
    split: str,
    conditions: Path,
    round_dir: Path,
    context_path: Path | None,
) -> dict[str, Any]:
    import run_selection_loop as loop

    output = round_dir / "smoke_evaluation"
    completed = subprocess.run(
        loop.round_command(
            args,
            dataset,
            split,
            conditions,
            output,
            stage="smoke",
            context_path=context_path,
        ),
        text=True,
    )
    return {
        "output": output,
        "results": dataset_results_path(loop, output, dataset, split),
        "returncode": completed.returncode,
    }


def read_smoke_rows(
    loop: Any,
    dataset: str,
    split: str,
    handoff: dict[str, Any],
    results_path: Path,
) -> list[dict[str, Any]]:
    return loop.join_candidate_results(
        dataset, handoff["round"], handoff, loop.read_csv(results_path)
    )


def successful_candidate_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        row["candidate_id"] for row in rows if row["outcome"] == "success"
    }


def failed_smoke_record(
    loop: Any,
    dataset: str,
    split: str,
    handoff: dict[str, Any],
    round_dir: Path,
    smoke: dict[str, Any],
    smoke_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    record = loop.build_round_record(
        dataset,
        split,
        handoff["round"],
        handoff,
        round_dir,
        smoke["returncode"] or 1,
    )
    record["candidate_results"] = [smoke_failure_row(row) for row in smoke_rows]
    use_smoke_outputs(loop, record, smoke)
    add_smoke_record(loop, record, smoke, 0)
    record["round_failures"].append({
        "stage": "smoke_gate",
        "error": "No candidate produced a valid measured smoke result.",
    })
    return record


def use_smoke_outputs(
    loop: Any, record: dict[str, Any], smoke: dict[str, Any]
) -> None:
    run_dir = (
        smoke["output"]
        / f"{loop.safe_name(record['dataset'])}_{record['requested_split']}"
    )
    rows = record["candidate_results"]
    record.update({
        "results_csv": str(smoke["results"]),
        "feedback_path": str(run_dir / "retrieval_feedback.json"),
        "dataset_inspection": str(run_dir / "dataset_inspection.json"),
        "plans": str(run_dir / "plans.json"),
        "context_paths": [
            str(run_dir / "codex_context.json"),
            str(run_dir / "manual_context.json"),
        ],
        "evaluated_split": loop.first_nonempty(rows, "evaluated_split"),
        "metric": loop.first_nonempty(rows, "metric"),
        "protocol": loop.first_nonempty(rows, "evaluation_protocol"),
        "round_failures": (
            loop.read_json_if_exists(smoke["output"] / "batch_failures.json") or []
        ),
    })


def run_full_for_passing(
    loop: Any,
    dataset: str,
    split: str,
    handoff: dict[str, Any],
    args: Any,
    round_dir: Path,
    smoke: dict[str, Any],
    smoke_rows: list[dict[str, Any]],
    passing_ids: set[str],
    spec: dict[str, Any],
) -> dict[str, Any]:
    passing = passing_handoff(handoff, passing_ids)
    full_conditions = round_dir / "full_conditions.csv"
    loop.write_conditions(full_conditions, passing)
    context_path = loop.prepare_evaluation_context(
        spec, passing, args, round_dir, "full_evaluation_context.json"
    )
    output = round_dir / "evaluation"
    completed = subprocess.run(
        loop.round_command(
            args,
            dataset,
            split,
            full_conditions,
            output,
            stage="full",
            context_path=context_path,
        ),
        text=True,
    )
    record = loop.build_round_record(
        dataset, split, handoff["round"], handoff, round_dir,
        completed.returncode,
    )
    record["full_conditions_csv"] = str(full_conditions)
    record["candidate_results"] = merge_gate_results(
        record["candidate_results"], smoke_rows, passing_ids
    )
    add_smoke_record(loop, record, smoke, len(passing_ids))
    return record


def passing_handoff(
    handoff: dict[str, Any], passing_ids: set[str]
) -> dict[str, Any]:
    return {
        **handoff,
        "candidates": [
            candidate for candidate in handoff["candidates"]
            if candidate["candidate_id"] in passing_ids
        ],
    }


def merge_gate_results(
    full_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    passing_ids: set[str],
) -> list[dict[str, Any]]:
    smoke_by_id = {row["candidate_id"]: row for row in smoke_rows}
    return [
        row if row["candidate_id"] in passing_ids
        else smoke_failure_row(smoke_by_id[row["candidate_id"]])
        for row in full_rows
    ]


def smoke_failure_row(row: dict[str, Any]) -> dict[str, Any]:
    note = f"Full evaluation skipped after smoke failure. {row.get('notes', '')}"
    return {
        **row,
        "outcome": "failure",
        "status": "smoke_failed",
        "score": "",
        "comparable": False,
        "non_comparable_reason": "no_valid_measured_smoke_result",
        "notes": note.strip(),
    }


def add_smoke_record(
    loop: Any,
    record: dict[str, Any],
    smoke: dict[str, Any],
    success_count: int,
) -> None:
    record["smoke_results_csv"] = str(smoke["results"])
    record["smoke_returncode"] = smoke["returncode"]
    record["smoke_measured_success_count"] = success_count
    record["smoke_failures"] = (
        loop.read_json_if_exists(smoke["output"] / "batch_failures.json") or []
    )


def dataset_results_path(
    loop: Any, root: Path, dataset: str, split: str
) -> Path:
    return root / f"{loop.safe_name(dataset)}_{split}" / "results.csv"

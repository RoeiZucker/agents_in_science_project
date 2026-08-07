#!/usr/bin/env python3
"""Plan and run HF model/dataset evaluations with smoke tests and auditing.

Examples:
  python run_eval_agent.py \
    --dataset xai-org/RealworldQA \
    --models google/paligemma-3b-ft-nlvr2-448 \
    --split test \
    --stage smoke \
    --project-root runtime

  python run_eval_agent.py \
    --dataset ImperialCollegeLondon/health_fact \
    --models yaxili96/FactCG-DeBERTa-v3-Large lytang/MiniCheck-Flan-T5-Large \
    --split auto \
    --stage plan \
    --trust-remote-code \
    --project-root runtime

  python run_eval_agent.py \
    --dataset tau/commonsense_qa \
    --models HuggingFaceTB/SmolLM3-3B \
    --split validation \
    --stage full \
    --continue-on-error \
    --project-root runtime
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from eval_agent_core import (
    DEFAULT_PROJECT_ROOT,
    audit_eval_dir,
    build_plan,
    command_to_shell,
    csv_row_from_audit,
    inspect_dataset,
    inspect_model,
    resolve_python,
    run_command,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--subset", default="")
    parser.add_argument("--split", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face dataset loading scripts to run.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--python", default="")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="smoke")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--question-column", default="")
    parser.add_argument("--answer-column", default="")
    parser.add_argument("--choices-column", default="")
    parser.add_argument("--image-column", default="")
    parser.add_argument("--timeout", type=int, default=0, help="Optional per-command timeout in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    output_root = args.output_root or project_root / "eval_results" / "_agent_runs"
    run_dir = output_root / f"{args.dataset.replace('/', '_')}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hf_home = args.hf_home or project_root / ".hf_cache"
    python_exe = resolve_python(args.python)

    trace = make_trace(args)
    add_trace_step(trace, "inspect_dataset", "started", {"dataset": args.dataset, "split": args.split})
    try:
        dataset = inspect_dataset(
            dataset=args.dataset,
            subset=args.subset or None,
            split=args.split,
            sample_size=args.sample_size,
            question_column=args.question_column,
            answer_column=args.answer_column,
            choices_column=args.choices_column,
            image_column=args.image_column,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        add_trace_step(trace, "inspect_dataset", "failed", {"error": f"{type(exc).__name__}: {exc}"})
        record_dataset_inspection_failure(args, run_dir, exc, trace)
        return
    add_trace_step(trace, "inspect_dataset", "completed", dataset.to_dict())

    plans = []
    csv_rows = []
    failures = []

    for model_name in args.models:
        add_trace_step(trace, "inspect_model", "started", {"model": model_name})
        model = inspect_model(model_name)
        add_trace_step(trace, "inspect_model", "completed", model.to_dict())
        plan = build_plan(
            dataset=dataset,
            model=model,
            project_root=project_root,
            python_exe=python_exe,
            smoke_limit=args.smoke_limit,
            output_root=output_root / "eval_results",
        )
        plans.append(plan)
        add_trace_step(trace, "build_plan", "completed", plan.to_dict())

        if args.stage == "plan":
            audit = {"status": "planned", "summary": {}, "warnings": [], "issues": []}
            csv_rows.append(csv_row_from_audit(plan, audit, plan.output_dir))
            add_trace_step(trace, "evaluate", "planned", {"model": model_name, "output_dir": str(plan.output_dir)})
            continue

        smoke = run_and_record(plan.smoke_command, plan.smoke_output_dir, run_dir, model_name, "smoke", hf_home, args.timeout)
        smoke_audit = audit_eval_dir(plan.smoke_output_dir)
        write_json(run_dir / "audits" / f"{plan.model.model.replace('/', '_')}_smoke_audit.json", smoke_audit)
        record_eval_trace(trace, model_name, "smoke", smoke.returncode, smoke_audit)
        if smoke.returncode != 0 or smoke_audit.get("status") == "bad":
            failures.append({"model": model_name, "stage": "smoke", "returncode": smoke.returncode, "audit": smoke_audit})
            csv_rows.append(csv_row_from_audit(plan, smoke_audit, plan.smoke_output_dir))
            if not args.continue_on_error:
                break
            continue

        if args.stage == "smoke":
            csv_rows.append(csv_row_from_audit(plan, smoke_audit, plan.smoke_output_dir))
            continue

        full = run_and_record(plan.full_command, plan.output_dir, run_dir, model_name, "full", hf_home, args.timeout)
        full_audit = audit_eval_dir(plan.output_dir)
        write_json(run_dir / "audits" / f"{plan.model.model.replace('/', '_')}_full_audit.json", full_audit)
        record_eval_trace(trace, model_name, "full", full.returncode, full_audit)
        csv_rows.append(csv_row_from_audit(plan, full_audit, plan.output_dir))
        if full.returncode != 0 or full_audit.get("status") == "bad":
            failures.append({"model": model_name, "stage": "full", "returncode": full.returncode, "audit": full_audit})
            if not args.continue_on_error:
                break

    trace["outputs"] = output_paths(run_dir)
    trace["failures"] = failures
    write_json(run_dir / "dataset_inspection.json", dataset.to_dict())
    write_json(run_dir / "plans.json", [plan.to_dict() for plan in plans])
    write_json(run_dir / "failures.json", failures)
    write_json(run_dir / "agent_trace.json", trace)
    write_csv(run_dir / "results.csv", csv_rows)

    print(json.dumps({
        "run_dir": str(run_dir),
        "stage": args.stage,
        "plans": len(plans),
        "failures": failures,
        "results_csv": str(run_dir / "results.csv"),
        "plans_json": str(run_dir / "plans.json"),
    }, indent=2, ensure_ascii=False))



def make_trace(args: argparse.Namespace) -> dict:
    return {
        "dataset": args.dataset,
        "requested_split": args.split,
        "requested_stage": args.stage,
        "needed_to_do": required_actions(args),
        "steps": [],
        "outputs": {},
        "failures": [],
    }


def required_actions(args: argparse.Namespace) -> list[str]:
    actions = [
        "Inspect dataset schema, split, input columns, answer column, label semantics, and task type.",
        "Inspect each model config and metadata to choose the evaluator adapter.",
        "Build reproducible evaluation commands and write plans.json.",
    ]
    if args.stage == "plan":
        actions.append("Stop after planning without running model inference.")
    elif args.stage == "smoke":
        actions.append("Run smoke evaluation, audit predictions, and write results.csv.")
    else:
        actions.append("Run smoke first, then full evaluation, audit predictions, and write results.csv.")
    return actions


def add_trace_step(trace: dict, name: str, status: str, details: dict) -> None:
    trace["steps"].append({"name": name, "status": status, "details": details})


def record_eval_trace(trace: dict, model: str, stage: str, returncode: int, audit: dict) -> None:
    add_trace_step(
        trace,
        "evaluate",
        audit.get("status", "unknown"),
        {"model": model, "stage": stage, "returncode": returncode, "audit": audit},
    )


def output_paths(run_dir: Path) -> dict[str, str]:
    return {
        "dataset_inspection": str(run_dir / "dataset_inspection.json"),
        "plans": str(run_dir / "plans.json"),
        "results": str(run_dir / "results.csv"),
        "failures": str(run_dir / "failures.json"),
        "trace": str(run_dir / "agent_trace.json"),
        "logs": str(run_dir / "logs"),
        "audits": str(run_dir / "audits"),
    }

def run_and_record(command: list[str], output_dir: Path, run_dir: Path, model_name: str, stage: str, hf_home: Path, timeout: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_name.replace('/', '_')}_{stage}"
    command_file = log_dir / f"{stem}.command.txt"
    stdout_file = log_dir / f"{stem}.stdout.txt"
    stderr_file = log_dir / f"{stem}.stderr.txt"
    command_file.write_text(command_to_shell(command, hf_home) + "\n", encoding="utf-8")
    result = run_command(command, hf_home=hf_home, timeout=timeout or None)
    stdout_file.write_text(result.stdout, encoding="utf-8")
    stderr_file.write_text(result.stderr, encoding="utf-8")
    return result


def record_dataset_inspection_failure(args: argparse.Namespace, run_dir: Path, exc: Exception, trace: dict | None = None) -> None:
    """Preserve actionable run artifacts when evaluation cannot reach model loading."""
    error = f"{type(exc).__name__}: {exc}"
    trace = traceback.format_exc()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "dataset_inspection.stderr.txt").write_text(trace, encoding="utf-8")

    inspection = {
        "dataset": args.dataset,
        "subset": args.subset or None,
        "requested_split": args.split,
        "status": "failed",
        "error": error,
    }
    failures = [
        {
            "model": model_name,
            "stage": "dataset_inspection",
            "returncode": None,
            "error": error,
        }
        for model_name in args.models
    ]
    rows = [
        {
            "dataset": args.dataset,
            "split": args.split,
            "model": model_name,
            "score": None,
            "total": None,
            "labeled_total": None,
            "status": "failed",
            "output_dir": "",
            "notes": f"Dataset inspection failed before model evaluation: {error}",
        }
        for model_name in args.models
    ]
    if trace is not None:
        trace["outputs"] = output_paths(run_dir)
        trace["failures"] = failures
    write_json(run_dir / "dataset_inspection.json", inspection)
    write_json(run_dir / "audits" / "dataset_inspection_audit.json", {
        "status": "bad",
        "issues": [error],
        "warnings": [],
    })
    write_json(run_dir / "plans.json", [])
    write_json(run_dir / "failures.json", failures)
    if trace is not None:
        write_json(run_dir / "agent_trace.json", trace)
    write_csv(run_dir / "results.csv", rows)
    print(json.dumps({
        "run_dir": str(run_dir),
        "stage": args.stage,
        "plans": 0,
        "failures": failures,
        "results_csv": str(run_dir / "results.csv"),
        "plans_json": str(run_dir / "plans.json"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

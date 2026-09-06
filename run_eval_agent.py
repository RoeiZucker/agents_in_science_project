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
    --full-limit 1000 \
    --continue-on-error \
    --project-root runtime

  python run_eval_agent.py \
    --dataset ImperialCollegeLondon/health_fact \
    --models yaxili96/FactCG-DeBERTa-v3-Large \
    --split auto \
    --stage smoke \
    --context-file runtime/eval_results/_condition_runs/ImperialCollegeLondon_health_fact_auto/codex_context.json \
    --trust-remote-code \
    --project-root runtime

  python run_eval_agent.py \
    --dataset deepmind/narrativeqa \
    --models meta-llama/Llama-3.2-3B-Instruct \
    --model-source second-state/Llama-3.2-3B-Instruct-GGUF \
    --split validation \
    --stage smoke \
    --project-root runtime

  python run_eval_agent.py \
    --dataset theatticusproject/cuad \
    --dataset-source chenghao/cuad_qa \
    --models Qwen/Qwen2.5-7B-Instruct \
    --context-column context \
    --context-file config/recovery_contexts/cuad.json \
    --stage full \
    --project-root runtime

  python run_eval_agent.py \
    --dataset HUPD/hupd --subset sample \
    --dataset-kwargs '{"uniform_split":true}' \
    --models Qwen/Qwen2.5-7B-Instruct \
    --stage smoke --project-root runtime


  Direct classifier/encoder scoring is selected automatically when generation is unavailable.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from string import Formatter

from eval_agent_core import (
    DEFAULT_PROJECT_ROOT,
    apply_known_dataset_contract,
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
    write_unsupported_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-source", default="", help="Equivalent data-only dataset source; the canonical dataset id remains in results.")
    parser.add_argument("--dataset-kwargs", default="", help="Optional JSON object passed to the Hugging Face dataset loader.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument(
        "--model-source",
        default="",
        help="Equivalent checkpoint source for a single canonical --models entry.",
    )
    parser.add_argument("--subset", default="")
    parser.add_argument("--split", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face dataset loading scripts to run.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--python", default="")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--full-limit", type=int, default=1000, help="Maximum randomly selected examples per full evaluation; 0 evaluates the entire split.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible full-evaluation sampling.")
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="smoke")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--allow-label-scores",
        action="store_true",
        help=(
            "Optional: prefer answer-choice likelihood for generative multiple-choice models; "
            "non-generative direct scoring is automatic."
        ),
    )
    parser.add_argument("--question-column", default="")
    parser.add_argument("--context-column", default="")
    parser.add_argument("--answer-column", default="")
    parser.add_argument("--choices-column", default="")
    parser.add_argument("--image-column", default="")
    parser.add_argument("--label-threshold", type=float, default=None, help="Optional threshold converting a numeric target into labels 0 and 1.")
    parser.add_argument("--answer-regex", default="", help="Optional regex extracting the gold answer from its dataset field.")
    parser.add_argument("--timeout", type=int, default=0, help="Optional per-command timeout in seconds.")
    parser.add_argument("--context-file", type=Path, default=None, help="Optional Codex-written JSON context with validated evaluation hints.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_model_source_args(args.models, args.model_source)
    project_root = args.project_root
    output_root = args.output_root or project_root / "eval_results" / "_agent_runs"
    run_dir = output_root / f"{args.dataset.replace('/', '_')}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)
    hf_home = args.hf_home or project_root / ".hf_cache"
    python_exe = resolve_python(args.python)

    context = load_context(args.context_file)
    values = dataset_context(context)
    inspection_source = args.dataset_source or str(values.get("dataset_source", "")).strip()
    inspection_kwargs = dataset_loader_kwargs(args.dataset_kwargs, values)
    inspection_subset, inspection_split = inspection_location(args, values)

    trace = make_trace(args)
    add_trace_step(
        trace,
        "inspect_dataset",
        "started",
        {"dataset": args.dataset, "dataset_source": inspection_source, "subset": inspection_subset, "split": inspection_split},
    )
    try:
        dataset = inspect_dataset(
            dataset=args.dataset,
            dataset_source=inspection_source,
            dataset_kwargs=inspection_kwargs,
            subset=inspection_subset or None,
            split=inspection_split,
            sample_size=args.sample_size,
            question_column=args.question_column or str(values.get("question_column", "")).strip(),
            context_column=args.context_column or str(values.get("context_column", "")).strip(),
            answer_column=args.answer_column or str(values.get("answer_column", "")).strip(),
            choices_column=args.choices_column or str(values.get("choices_column", "")).strip(),
            choices_columns=values.get("choices_columns", []),
            image_column=args.image_column or str(values.get("image_column", "")).strip(),
            evaluation_method=str(values.get("evaluation_method", "auto")).strip() or "auto",
            trust_remote_code=args.trust_remote_code,
        )
    except Exception as exc:
        add_trace_step(trace, "inspect_dataset", "failed", {"error": f"{type(exc).__name__}: {exc}"})
        record_dataset_inspection_failure(args, run_dir, exc, trace)
        raise SystemExit(1)
    add_trace_step(trace, "inspect_dataset", "completed", dataset.to_dict())
    dataset = apply_context(dataset, context)
    dataset = apply_known_dataset_contract(dataset)
    if args.label_threshold is not None:
        dataset.label_threshold = args.label_threshold
    if args.answer_regex:
        dataset.answer_regex = args.answer_regex
    if context:
        add_trace_step(trace, "apply_context", "completed", {"context_file": str(args.context_file), "dataset": dataset.to_dict()})

    plans = []
    csv_rows = []
    failures = []

    for model_name in args.models:
        add_trace_step(trace, "inspect_model", "started", {"model": model_name})
        model = inspect_model(args.model_source or model_name)
        model = preserve_canonical_model(model, model_name, args.model_source)
        add_trace_step(trace, "inspect_model", "completed", model.to_dict())
        plan = build_plan(
            dataset=dataset,
            model=model,
            project_root=project_root,
            python_exe=python_exe,
            smoke_limit=args.smoke_limit,
            output_root=output_root / "eval_results",
            full_limit=args.full_limit,
            seed=args.seed,
            allow_label_scores=args.allow_label_scores,
            model_source=args.model_source,
        )
        plans.append(plan)
        add_trace_step(trace, "build_plan", "completed", plan.to_dict())

        if args.stage == "plan":
            audit = {"status": "planned", "summary": {}, "warnings": [], "issues": []}
            csv_rows.append(csv_row_from_audit(plan, audit, plan.output_dir))
            add_trace_step(trace, "evaluate", "planned", {"model": model_name, "output_dir": str(plan.output_dir)})
            continue

        if plan.protocol == "unsupported":
            audit = write_unsupported_result(plan, plan.output_dir)
            write_json(run_dir / "audits" / f"{plan.model.model.replace('/', '_')}_protocol_audit.json", audit)
            failures.append({"model": model_name, "stage": "protocol", "returncode": None, "audit": audit})
            csv_rows.append(csv_row_from_audit(plan, audit, plan.output_dir))
            record_eval_trace(trace, model_name, "protocol", 0, audit)
            if not args.continue_on_error:
                break
            continue

        smoke = run_and_record(plan.smoke_command, plan.smoke_output_dir, run_dir, model_name, "smoke", hf_home, args.timeout)
        smoke_audit = audit_eval_dir(plan.smoke_output_dir)
        write_json(run_dir / "audits" / f"{plan.model.model.replace('/', '_')}_smoke_audit.json", smoke_audit)
        record_eval_trace(trace, model_name, "smoke", smoke.returncode, smoke_audit)
        if smoke.returncode != 0 or smoke_audit.get("status") != "ok":
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
        if full.returncode != 0 or full_audit.get("status") != "ok":
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
    if failures:
        raise SystemExit(1)


def validate_model_source_args(models: list[str], model_source: str) -> None:
    if model_source and len(models) != 1:
        raise ValueError("--model-source requires exactly one canonical --models entry.")


def preserve_canonical_model(model: Any, canonical: str, source: str) -> Any:
    if not source:
        return model
    model.model = canonical
    model.notes.append(f"Using equivalent checkpoint source: {source}")
    return model



def dataset_context(context: dict) -> dict:
    values = context.get("dataset_context", {}) if isinstance(context, dict) else {}
    return values if isinstance(values, dict) else {}


def inspection_location(args: argparse.Namespace, values: dict) -> tuple[str, str]:
    subset = args.subset or str(values.get("subset", "")).strip()
    split = args.split
    if split == "auto":
        split = str(values.get("split", "")).strip() or "auto"
    return subset, split


def dataset_loader_kwargs(raw: str, values: dict) -> dict:
    if raw:
        return parse_json_object(raw)
    value = values.get("dataset_kwargs", {})
    if not isinstance(value, dict):
        raise ValueError("dataset_context.dataset_kwargs must be a JSON object.")
    return dict(value)


def parse_json_object(raw: str) -> dict:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--dataset-kwargs must decode to a JSON object.")
    return value



def load_context(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_context(dataset: Any, context: dict) -> Any:
    values = context.get("dataset_context", context) if isinstance(context, dict) else {}
    if not isinstance(values, dict):
        return dataset
    apply_column_context(dataset, values)
    apply_task_context(dataset, values)
    apply_label_context(dataset, values)
    note = values.get("notes") or values.get("reasoning")
    if note:
        dataset.notes.append(f"Codex context: {note}")
    return dataset


def apply_column_context(dataset: Any, values: dict) -> None:
    for key in ("question_column", "context_column", "answer_column", "choices_column", "image_column"):
        if key not in values:
            continue
        value = str(values.get(key, "")).strip()
        if not value:
            setattr(dataset, key, "")
        elif context_path_in_columns(value, dataset.columns):
            setattr(dataset, key, value)
    choices_columns = values.get("choices_columns")
    if isinstance(choices_columns, list):
        valid = [value for value in choices_columns if context_path_in_columns(value, dataset.columns)]
        if len(valid) == len(choices_columns):
            dataset.choices_columns = valid


def context_path_in_columns(path: str, columns: list[str]) -> bool:
    first = path.split(".", 1)[0].removesuffix("[]")
    return first in columns


def context_feature_key(path: str) -> str:
    return path.split(".", 1)[0].removesuffix("[]")


def apply_label_context(dataset: Any, values: dict) -> None:
    label_map = values.get("label_map")
    if isinstance(label_map, dict) and label_map:
        normalized = {str(key): str(value) for key, value in label_map.items()}
        if context_label_map_compatible(dataset, normalized):
            dataset.label_map = normalized
        else:
            dataset.notes.append(
                "Ignored Codex label_map because it is incompatible with observed dataset labels."
            )
    threshold = values.get("label_threshold")
    if threshold is not None:
        dataset.label_threshold = float(threshold)
    answer_regex = str(values.get("answer_regex", "")).strip()
    if answer_regex:
        dataset.answer_regex = answer_regex
    prompt_template = str(values.get("prompt_template", "")).strip()
    if not prompt_template:
        return
    if prompt_template_in_columns(prompt_template, dataset.columns):
        dataset.prompt_template = prompt_template
        return
    dataset.notes.append(
        "Ignored Codex prompt_template because it references unavailable dataset columns."
    )


def context_label_map_compatible(dataset: Any, label_map: dict[str, str]) -> bool:
    if not answer_column_accepts_label_map(dataset):
        return False
    if not dataset.label_map:
        return True
    known_keys = set(dataset.label_map)
    known_values = set(dataset.label_map.values())
    proposed_keys = set(label_map)
    proposed_values = set(label_map.values())
    return known_keys <= proposed_keys or known_values <= proposed_values


def prompt_template_in_columns(template: str, columns: list[str]) -> bool:
    try:
        fields = [field for _, field, _, _ in Formatter().parse(template) if field]
    except ValueError:
        return False
    return all(context_path_in_columns(field, columns) for field in fields)


def answer_column_accepts_label_map(dataset: Any) -> bool:
    if dataset.task in {
        "classification", "multilabel_classification", "multiple_choice",
        "image_classification",
    }:
        return True
    answer_feature = str(dataset.features.get(context_feature_key(dataset.answer_column), ""))
    return "ClassLabel" in answer_feature or (
        "Sequence" not in answer_feature and "Value(dtype='string'" not in answer_feature
    )


def apply_task_context(dataset: Any, values: dict) -> None:
    task = str(values.get("task", "")).strip()
    if task == "multiple_choice":
        has_choices = context_path_in_columns(dataset.choices_column, dataset.columns)
        has_separate_choices = bool(dataset.choices_columns)
        if not has_choices and not has_separate_choices:
            dataset.notes.append("Ignored Codex task=multiple_choice because no choice columns exist.")
            return
    if task == "image_classification" and not context_path_in_columns(dataset.image_column, dataset.columns):
        dataset.notes.append("Ignored Codex task=image_classification because no image column exists.")
        return
    if task:
        dataset.task = task
    method = str(values.get("evaluation_method", "")).strip()
    if method:
        dataset.evaluation_method = method


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
    traceback_text = traceback.format_exc()
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "dataset_inspection.stderr.txt").write_text(traceback_text, encoding="utf-8")

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
        "status": "failed",
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

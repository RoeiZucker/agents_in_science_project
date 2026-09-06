#!/usr/bin/env python3
"""Run evaluation-agent jobs from a partner-provided conditions CSV.

Examples:
  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --stage full \
    --full-limit 1000

  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --condition A_merged \
    --limit-datasets 1 \
    --stage smoke \
    --fresh-run-dir

  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --stage smoke \
    --runner codex \
    --limit-datasets 2 \
    --limit-pairs 2 \
    --codex-bypass-sandbox \
    --trust-remote-code \
    --cleanup-cache none \
    --skip-refinement

  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset fancyzhx/ag_news \
    --runner codex \
    --codex-timeout 120 \
    --codex-bypass-sandbox

  # Use manually authored context instead of launching Codex. Select one dataset;
  # the JSON must match that dataset and its ordered model list exactly.
  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --runner manual \
    --context-file config/health_fact_context.json \
    --stage smoke \
    --cleanup-cache none

  # Advanced manual contexts may also pin a data-only mirror, context column,
  # loader kwargs, numeric-label threshold, and answer extraction regex.

  Direct encoder/classifier scoring is selected automatically when generation is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_agent_core import method_matches_task, safe_name
from result_contract import measured_success


REQUIRED_COLUMNS = {"condition", "query_dataset", "rank", "model_name"}
CONTEXT_FIELDS = {"dataset", "models", "dataset_context", "model_context", "confidence"}
DATASET_CONTEXT_FIELDS = {
    "question_column", "answer_column", "choices_column", "image_column",
    "task", "label_map", "prompt_template", "notes",
}
DATASET_CONTEXT_STRING_FIELDS = DATASET_CONTEXT_FIELDS - {"label_map"}
DATASET_CONTEXT_OPTIONAL_FIELDS = {
    "subset", "split", "choices_columns", "evaluation_method",
    "context_column", "dataset_source", "dataset_kwargs",
    "label_threshold", "answer_regex",
}
MODEL_CONTEXT_FIELDS = {"notes"}
CONTEXT_TASKS = {
    "", "generation", "classification", "multilabel_classification",
    "multiple_choice", "qa", "numeric_qa", "summarization", "token_classification",
    "relation_extraction", "image_classification",
}
CONTEXT_EVALUATION_METHODS = {
    "", "auto", "accuracy", "macro_f1", "exact_match",
    "qa_f1", "numeric_match", "rouge_l", "set_f1",
}
CONTEXT_CONFIDENCE = {"high", "medium", "low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face dataset loading scripts to run.")
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument(
        "--runner", choices=("script", "codex", "manual"), default="script"
    )
    parser.add_argument(
        "--context-file", type=Path,
        help="Validated context JSON; required with --runner manual.",
    )
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-approval", default="never")
    parser.add_argument("--codex-timeout", type=int, default=120, help="Seconds to wait for Codex before falling back or failing.")
    parser.add_argument("--codex-context-attempts", type=int, default=2, help="Metadata-scout attempts when context JSON is missing or invalid.")
    parser.add_argument("--codex-bypass-sandbox", action="store_true", help="Run nested Codex without its command sandbox for systems where bwrap/user namespaces fail.")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--full-limit", type=int, default=1000, help="Maximum randomly selected examples per full evaluation; 0 evaluates every example.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-label-scores",
        action="store_true",
        help=(
            "Optional: prefer answer-choice likelihood for generative multiple-choice models; "
            "non-generative direct scoring is automatic."
        ),
    )
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--limit-datasets", type=int, default=0)
    parser.add_argument("--limit-pairs", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--fresh-run-dir", action="store_true", help="Delete each selected dataset run directory before starting it.")
    parser.add_argument(
        "--cleanup-cache",
        choices=("none", "after-dataset", "end"),
        default="after-dataset",
        help="Delete downloaded HF model/dataset caches while preserving evaluation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = args.project_root.resolve()
    output_root = (args.output_root or project_root / "eval_results" / "_condition_runs").resolve()
    hf_home = project_root / ".hf_cache"
    datasets_cache = project_root / ".hf_datasets_cache"
    project_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = select_rows(read_conditions(args.conditions_csv), args)
    grouped = group_by_dataset(rows)
    validate_runner_context(args, grouped)
    progress(f"[batch] selected {len(rows)} pairs across {len(grouped)} datasets; runner={args.runner}; stage={args.stage}; output={output_root}")
    batch_rows = []
    failures = []

    for index, (dataset, dataset_rows) in enumerate(grouped.items(), start=1):
        progress(f"[batch] dataset {index}/{len(grouped)} start: {dataset} pairs={len(dataset_rows)}")
        result = run_dataset_group(dataset, dataset_rows, args, script_dir, project_root, output_root, hf_home, datasets_cache)
        progress(f"[batch] dataset {index}/{len(grouped)} done: {dataset} failures={len(result['failures'])}")
        batch_rows.extend(join_results(dataset_rows, result["results_csv"]))
        failures.extend(result["failures"])
        if args.cleanup_cache == "after-dataset":
            cleanup_cache_dirs(hf_home, datasets_cache)
        if result["failures"] and not args.continue_on_error:
            break

    write_csv(output_root / "batch_results.csv", batch_rows)
    write_json(output_root / "batch_failures.json", failures)
    write_json(output_root / "batch_contexts.json", collect_batch_contexts(output_root, grouped, args.split))
    if args.cleanup_cache == "end":
        cleanup_cache_dirs(hf_home, datasets_cache)
    print(json.dumps({
        "conditions_csv": str(args.conditions_csv),
        "output_root": str(output_root),
        "datasets": len(grouped),
        "input_rows": len(rows),
        "result_rows": len(batch_rows),
        "failures": len(failures),
        "batch_results_csv": str(output_root / "batch_results.csv"),
    }, indent=2))
    if failures:
        sys.exit(1)


def read_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_columns(rows)
    return rows


def validate_columns(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("conditions CSV is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"conditions CSV is missing columns: {sorted(missing)}")


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    selected = rows
    if args.condition:
        selected = [row for row in selected if row["condition"] in set(args.condition)]
    if args.dataset:
        selected = [row for row in selected if row["query_dataset"] in set(args.dataset)]
    if args.limit_pairs:
        selected = selected[: args.limit_pairs]
    if args.limit_datasets:
        keep = list(dict.fromkeys(row["query_dataset"] for row in selected))[: args.limit_datasets]
        selected = [row for row in selected if row["query_dataset"] in set(keep)]
    return selected


def group_by_dataset(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["query_dataset"]].append(row)
    return dict(groups)


def collect_batch_contexts(output_root: Path, grouped: dict[str, list[dict[str, str]]], split: str) -> list[dict[str, Any]]:
    contexts = []
    for dataset in grouped:
        run_dir = output_root / f"{safe_name(dataset)}_{split}"
        contexts.append(collect_dataset_context(dataset, run_dir))
    return contexts


def collect_dataset_context(dataset: str, run_dir: Path) -> dict[str, Any]:
    codex_context = read_json_if_exists(run_dir / "codex_context.json")
    manual_context = read_json_if_exists(run_dir / "manual_context.json")
    return {
        "dataset": dataset,
        "run_dir": str(run_dir),
        "context_source": context_source(codex_context, manual_context),
        "codex_context": codex_context,
        "manual_context": manual_context,
        "evaluation_plans": summarize_plans(read_json_if_exists(run_dir / "plans.json")),
    }


def context_source(codex_context: Any, manual_context: Any) -> str:
    if manual_context is not None:
        return "manual"
    return "codex" if codex_context is not None else "none"


def summarize_plans(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    summaries = []
    for plan in value:
        if not isinstance(plan, dict):
            continue
        dataset = plan.get("dataset", {}) if isinstance(plan.get("dataset"), dict) else {}
        model = plan.get("model", {}) if isinstance(plan.get("model"), dict) else {}
        summaries.append({
            "model": model.get("model"),
            "task": plan.get("task") or dataset.get("task"),
            "evaluation_protocol": plan.get("evaluation_protocol"),
            "metric": plan.get("metric"),
            "model_type": model.get("model_type"),
            "split": plan.get("split"),
            "question_column": dataset.get("question_column"),
            "answer_column": dataset.get("answer_column"),
            "choices_column": dataset.get("choices_column"),
            "image_column": dataset.get("image_column"),
            "label_map": dataset.get("label_map") or {},
            "prompt_template": dataset.get("prompt_template") or "",
            "notes": (dataset.get("notes") or []) + (plan.get("notes") or []),
        })
    return summaries


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON: {exc}"}


def run_dataset_group(
    dataset: str,
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    hf_home: Path,
    datasets_cache: Path,
) -> dict[str, Any]:
    run_dir = output_root / f"{safe_name(dataset)}_{args.split}"
    prepare_run_dir(run_dir, args.fresh_run_dir)
    candidates_path = write_candidates(run_dir, dataset, rows)
    if args.runner == "codex":
        failures = run_codex_context_then_script(dataset, rows, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)
    elif args.runner == "manual":
        context_path, failures = prepare_manual_context(
            dataset, rows, run_dir, args.context_file
        )
        if not failures:
            failures = run_script_dataset_group(
                dataset, run_dir, candidates_path, args, script_dir, project_root,
                output_root, hf_home, datasets_cache, context_path,
            )
    else:
        failures = run_script_dataset_group(dataset, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache, None)
    failures.extend(expected_result_failures(rows, run_dir / "results.csv"))
    return {"results_csv": run_dir / "results.csv", "failures": dedupe_failures(failures)}



def run_codex_context_then_script(
    dataset: str,
    rows: list[dict[str, str]],
    run_dir: Path,
    candidates_path: Path,
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    hf_home: Path,
    datasets_cache: Path,
) -> list[dict[str, Any]]:
    progress(f"[batch] codex context start: {dataset}")
    context_path = run_dir / "codex_context.json"
    codex_failures = run_codex_dataset_group(dataset, rows, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)
    progress(f"[batch] codex context done: {dataset} failures={len(codex_failures)}")
    if codex_failures:
        return codex_failures
    return run_script_dataset_group(
        dataset, run_dir, candidates_path, args, script_dir, project_root,
        output_root, hf_home, datasets_cache, context_path,
    )


def validate_runner_context(
    args: argparse.Namespace,
    grouped: dict[str, list[dict[str, str]]],
) -> None:
    context_file = getattr(args, "context_file", None)
    if args.runner != "manual":
        if context_file:
            raise ValueError("--context-file requires --runner manual.")
        return
    if not context_file:
        raise ValueError("--runner manual requires --context-file.")
    if len(grouped) != 1:
        raise ValueError("Manual context mode requires exactly one selected dataset.")
    if not context_file.is_file():
        raise FileNotFoundError(f"Manual context file does not exist: {context_file}")


def prepare_manual_context(
    dataset: str,
    rows: list[dict[str, str]],
    run_dir: Path,
    source: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    destination = run_dir / "manual_context.json"
    progress(f"[batch] manual context start: {dataset}")
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    write_json(run_dir / "manual_context_source.json", {"source": str(source.resolve())})
    failures = context_failures(
        dataset, unique_models(rows), destination, stage="manual_context"
    )
    progress(f"[batch] manual context done: {dataset} failures={len(failures)}")
    return destination, failures


def prepare_run_dir(run_dir: Path, fresh: bool) -> None:
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

def unique_models(rows: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(row["model_name"] for row in rows))


def write_candidates(run_dir: Path, dataset: str, rows: list[dict[str, str]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for index, row in enumerate(rows, start=1):
        candidates.append({
            "candidate_id": row.get("candidate_id") or candidate_id(row, index),
            "model": row["model_name"],
            "rank": parse_int(row.get("rank")),
            "score": parse_float(row.get("recommendation_score")),
            "condition": row.get("condition"),
            "source": row.get("source_method"),
            "intended_use": row.get("intended_use") or "direct_inference",
            "reason": row.get("reasoning"),
        })
    path = run_dir / "candidate_models_from_conditions.json"
    write_json(path, {
        "handoff_version": 1,
        "mock": parse_bool(rows[0].get("handoff_mock")) if rows else False,
        "owner": (rows[0].get("handoff_owner") if rows else None) or "partner_retrieval",
        "dataset": dataset,
        "retrieval_agent": (rows[0].get("retrieval_agent") if rows else None) or "partner_conditions_csv",
        "candidates": candidates,
    })
    return path


def candidate_id(row: dict[str, str], index: int) -> str:
    condition = row.get("condition") or "condition"
    rank = row.get("rank") or str(index)
    return f"{condition}:{rank}:{index}"


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def run_script_dataset_group(
    dataset: str,
    run_dir: Path,
    candidates_path: Path,
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    hf_home: Path,
    datasets_cache: Path,
    context_path: Path | None,
) -> list[dict[str, Any]]:
    command = eval_command(args, script_dir, project_root, output_root, dataset, unique_models_from_candidates(candidates_path), hf_home, context_path)
    progress(f"[batch] script runner start: {dataset}")
    result = run_command(command, script_dir, hf_home, datasets_cache, args.timeout)
    progress(f"[batch] script runner done: {dataset} returncode={result.returncode}")
    write_command_logs(run_dir, "eval", command, result)
    failures = command_failures(dataset, "eval", result)
    failures.extend(internal_eval_failures(run_dir))
    failures.extend(missing_or_bad_result_failures(run_dir / "results.csv"))
    if not args.skip_refinement and (run_dir / "results.csv").exists():
        failures.extend(run_refinement(run_dir, candidates_path, args, script_dir, hf_home, datasets_cache))
    return failures


def unique_models_from_candidates(candidates_path: Path) -> list[str]:
    value = json.loads(candidates_path.read_text(encoding="utf-8"))
    return list(dict.fromkeys(item["model"] for item in value["candidates"]))


def run_codex_dataset_group(
    dataset: str,
    rows: list[dict[str, str]],
    run_dir: Path,
    candidates_path: Path,
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    hf_home: Path,
    datasets_cache: Path,
) -> list[dict[str, Any]]:
    models = unique_models(rows)
    context_path = run_dir / "codex_context.json"
    with tempfile.TemporaryDirectory(prefix="hf-eval-context-") as temp_dir:
        staged_path = Path(temp_dir) / "codex_context.json"
        prompt = build_codex_prompt(
            dataset, rows, run_dir, candidates_path, args, project_root,
            output_root, hf_home, datasets_cache, staged_path,
        )
        (run_dir / "codex_prompt.txt").write_text(prompt, encoding="utf-8")
        attempts = max(1, getattr(args, "codex_context_attempts", 2))
        failures: list[dict[str, Any]] = []
        last_result = None

        for attempt in range(1, attempts + 1):
            staged_path.unlink(missing_ok=True)
            current_prompt = (
                prompt if attempt == 1 else retry_codex_prompt(prompt, failures)
            )
            context_workspace = Path(temp_dir)
            command = codex_command(args, context_workspace, current_prompt)
            last_result = run_command(
                command, context_workspace, hf_home, datasets_cache,
                args.codex_timeout,
            )
            log_name = "codex" if attempt == 1 else f"codex_retry_{attempt}"
            write_command_logs(
                run_dir, log_name, command_without_prompt(command), last_result
            )
            failures = context_failures(dataset, models, staged_path)
            if failures and recover_context_from_stdout(
                staged_path, last_result.stdout, dataset, models
            ):
                failures = []
            if not failures:
                context_path.write_bytes(staged_path.read_bytes())
                if last_result.returncode != 0:
                    write_json(
                        run_dir / "codex_context_warning.json",
                        {
                            "returncode": last_result.returncode,
                            "stderr_tail": tail(last_result.stderr),
                        },
                    )
                return []
            if attempt < attempts:
                progress(
                    f"[batch] codex context retry {attempt + 1}/{attempts}: "
                    f"{failures[0]['error']}"
                )

    failures.extend(command_failures(dataset, "codex_context", last_result))
    return dedupe_failures(failures)


def retry_codex_prompt(
    prompt: str, failures: list[dict[str, Any]]
) -> str:
    error = failures[0].get("error", "context JSON was missing or invalid")
    return (
        f"{prompt}\nPrevious context attempt failed validation: {error}\n"
        "Overwrite CONTEXT_OUTPUT with one valid JSON object, then stop.\n"
    )


def recover_context_from_stdout(
    path: Path, stdout: str, dataset: str, models: list[str]
) -> bool:
    value = valid_context_from_text(stdout, dataset, models)
    if value is None:
        return False
    write_json(path, value)
    return True


def valid_context_from_text(
    text: str, dataset: str, models: list[str]
) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for start, character in enumerate(text or ""):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not validate_context(value, dataset, models):
            return value
    return None


def codex_command(args: argparse.Namespace, script_dir: Path, prompt: str) -> list[str]:
    command = [args.codex_bin, "-C", str(script_dir)]
    if args.codex_bypass_sandbox:
        command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        command.extend(["-a", args.codex_approval])
    command.extend(["-s", "workspace-write"])
    command.extend(["exec", "--skip-git-repo-check", prompt])
    return command


def command_without_prompt(command: list[str]) -> list[str]:
    if len(command) < 2:
        return command
    return [*command[:-1], "<prompt omitted; see codex_prompt.txt>"]


def build_codex_prompt(
    dataset: str,
    rows: list[dict[str, str]],
    run_dir: Path,
    candidates_path: Path,
    args: argparse.Namespace,
    project_root: Path,
    output_root: Path,
    hf_home: Path,
    datasets_cache: Path,
    context_output: Path | None = None,
) -> str:
    model_bullets = "\n".join(f"- {model}" for model in unique_models(rows))
    context_path = context_output or run_dir / "codex_context.json"
    return "\n".join([
        "You are a context scout for hf-eval-agent, not an evaluator.",
        "",
        "Your only job is to inspect Hugging Face dataset/model metadata and write extra context that helps the local evaluator choose columns, labels, task type, and prompt wording.",
        "",
        "Hard restrictions:",
        "- Do not run run_eval_agent.py.",
        "- Do not run evaluate_hf_pair.py.",
        "- Do not load model weights, instantiate models, create pipelines, train, fine-tune, or run inference.",
        "- Do not run smoke or full evaluation commands.",
        "- Do not delete downloads, caches, or evaluation outputs.",
        "- Do not edit repository code.",
        "",
        "Allowed actions:",
        "- Read local files in this repo.",
        "- Read candidate metadata.",
        "- Inspect Hugging Face dataset cards, model cards, dataset schemas, feature metadata, and at most a tiny row sample only if metadata is insufficient.",
        "- Prefer lightweight metadata APIs such as datasets.load_dataset_builder, builder.info, and huggingface_hub model_info/dataset_info.",
        "- Avoid full dataset materialization. If schema/card/model metadata is enough, write the context immediately without sample loading.",
        "- If Python is needed, use the PYTHON path listed below, not plain python or python3.",
        "",
        f"Dataset: {dataset}",
        "Models:",
        model_bullets,
        "",
        "Paths:",
        f"- RUN_DIR={run_dir}",
        f"- CONTEXT_OUTPUT={context_path}",
        f"- CANDIDATES={candidates_path}",
        f"- PROJECT_ROOT={project_root}",
        f"- HF_HOME={hf_home}",
        f"- HF_DATASETS_CACHE={datasets_cache}",
        f"- PYTHON={args.python}",
        "",
        "Write exactly one JSON file at CONTEXT_OUTPUT with this schema:",
        "{",
        "  \"dataset\": \"dataset id\",",
        "  \"models\": [\"model id\"],",
        "  \"dataset_context\": {",
        "    \"subset\": \"optional dataset config; use empty for default\",",
        "    \"split\": \"optional labeled split; use empty for automatic selection\",",
        "    \"question_column\": \"optional existing column or dotted path such as question.text\",",
        "    \"answer_column\": \"optional existing column or dotted/list path such as answers.text\",",
        "    \"choices_column\": \"optional existing column or dotted path; leave empty when choices_columns is used\",",
        "    \"choices_columns\": [\"optional\", \"separate\", \"choice columns\"],",
        "    \"image_column\": \"optional existing column or dotted path\",",
        "    \"task\": \"generation|classification|multilabel_classification|multiple_choice|qa|numeric_qa|summarization|token_classification|relation_extraction|image_classification or empty\",",
        "    \"evaluation_method\": \"auto|accuracy|macro_f1|exact_match|qa_f1|numeric_match|rouge_l|set_f1\",",
        "    \"label_map\": {\"raw label\": \"human label\"},",
        "    \"prompt_template\": \"optional template using dataset column names\",",
        "    \"notes\": \"brief reason for the context\"",
        "  },",
        "  \"model_context\": {",
        "    \"notes\": \"brief metadata-only notes; no inference results\"",
        "  },",
        "  \"confidence\": \"high|medium|low\"",
        "}",
        "",
        "Choose the evaluation method from the registered methods based on the dataset card, schema, and standard benchmark practice. You have freedom to choose among them, but do not invent a metric.",
        "Use accuracy or macro_f1 for finite single-label classification, set_f1 for multilabel/entity sets, qa_f1 for short-answer QA, numeric_match for numeric QA, rouge_l for summarization, and exact_match only when normalized exact equality is meaningful.",
        "Use task=multiple_choice only when the dataset has an explicit choices/options/candidates column or separate answer columns listed in choices_columns. An answers field is often reference answers for QA, not choices.",
        "For free-form QA, use task=qa, put the question/context in prompt_template, put the reference answer path/list in answer_column, and leave choice fields empty.",
        "Set label_map for finite classification labels, including categorical strings. Leave label_map empty for summarization, QA, unrestricted generation, or auxiliary labels that are not the answer.",
        "Prompt templates may use dotted placeholders, for example {document.summary.text}, {question.text}, or {answers.text}.",
        "After writing CONTEXT_OUTPUT, do not inspect it, do not run more commands, and finish immediately.",
        "If writing the file is impossible, return the exact same JSON object as your entire final response so the outer script can recover it.",
        "Leave a field empty rather than guessing. The outer script will validate columns and run evaluation after you exit.",
    ]) + "\n"


def context_failures(
    dataset: str,
    models: list[str],
    path: Path,
    stage: str = "codex_context",
) -> list[dict[str, Any]]:
    if not path.exists():
        return context_failure(dataset, f"Missing {path.name}", stage)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return context_failure(dataset, f"Invalid JSON: {exc}", stage)
    error = validate_context(value, dataset, models)
    return context_failure(dataset, error, stage) if error else []


def context_failure(
    dataset: str, error: str, stage: str = "codex_context"
) -> list[dict[str, Any]]:
    return [{"dataset": dataset, "stage": stage, "error": error}]


def validate_context(value: Any, dataset: str, models: list[str]) -> str:
    if not isinstance(value, dict):
        return "Context JSON must be an object."
    error = exact_fields_error(value, CONTEXT_FIELDS, "Context JSON")
    if error:
        return error
    if not isinstance(value["dataset"], str):
        return "dataset must be a string."
    if value["dataset"] != dataset:
        return f"dataset mismatch: expected {dataset!r}, got {value['dataset']!r}."
    if not isinstance(value["models"], list) or not all(
        isinstance(model, str) for model in value["models"]
    ):
        return "models must be a list of strings."
    if value["models"] != models:
        return f"models mismatch: expected {models!r}, got {value['models']!r}."
    if not isinstance(value["confidence"], str):
        return "confidence must be a string."
    if value["confidence"] not in CONTEXT_CONFIDENCE:
        return "confidence must be one of: high, medium, low."
    return validate_context_objects(value)


def validate_context_objects(value: dict[str, Any]) -> str:
    error = validate_dataset_context(value["dataset_context"])
    if error:
        return error
    return validate_model_context(value["model_context"])


def validate_dataset_context(value: Any) -> str:
    if not isinstance(value, dict):
        return "dataset_context must be an object."
    error = exact_fields_error(
        value, DATASET_CONTEXT_FIELDS, "dataset_context", DATASET_CONTEXT_OPTIONAL_FIELDS
    )
    if error:
        return error
    for field in sorted(DATASET_CONTEXT_STRING_FIELDS):
        if not isinstance(value[field], str):
            return f"dataset_context.{field} must be a string."
    if value["task"] not in CONTEXT_TASKS:
        return f"dataset_context.task must be one of: {sorted(CONTEXT_TASKS)}."
    for field in ("subset", "split", "evaluation_method"):
        if field in value and not isinstance(value[field], str):
            return f"dataset_context.{field} must be a string."
    for field in ("context_column", "dataset_source", "answer_regex"):
        if field in value and not isinstance(value[field], str):
            return f"dataset_context.{field} must be a string."
    if "dataset_kwargs" in value and not isinstance(value["dataset_kwargs"], dict):
        return "dataset_context.dataset_kwargs must be an object."
    threshold = value.get("label_threshold")
    if threshold is not None and (
        not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
    ):
        return "dataset_context.label_threshold must be a number."
    method = value.get("evaluation_method", "auto")
    if method not in CONTEXT_EVALUATION_METHODS:
        return f"dataset_context.evaluation_method must be one of: {sorted(CONTEXT_EVALUATION_METHODS)}."
    task = value["task"]
    if task and method not in {"", "auto"} and not method_matches_task(task, method):
        return (
            f"dataset_context.evaluation_method {method!r} "
            f"is incompatible with task {task!r}."
        )
    choices_columns = value.get("choices_columns", [])
    if not isinstance(choices_columns, list) or not all(isinstance(item, str) for item in choices_columns):
        return "dataset_context.choices_columns must be a list of strings."
    label_map = value["label_map"]
    if not isinstance(label_map, dict):
        return "dataset_context.label_map must be an object."
    if not all(
        isinstance(key, str) and isinstance(label, str)
        for key, label in label_map.items()
    ):
        return "dataset_context.label_map keys and values must be strings."
    return ""


def validate_model_context(value: Any) -> str:
    if not isinstance(value, dict):
        return "model_context must be an object."
    error = exact_fields_error(value, MODEL_CONTEXT_FIELDS, "model_context")
    if error:
        return error
    if not isinstance(value["notes"], str):
        return "model_context.notes must be a string."
    return ""


def exact_fields_error(
    value: dict[str, Any], expected: set[str], label: str, optional: set[str] | None = None
) -> str:
    missing = expected - set(value)
    if missing:
        return f"{label} is missing fields: {sorted(missing)}."
    unknown = set(value) - expected - (optional or set())
    if unknown:
        return f"{label} has unknown fields: {sorted(unknown)}."
    return ""



def eval_command(
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    dataset: str,
    models: list[str],
    hf_home: Path,
    context_path: Path | None = None,
) -> list[str]:
    command = [
        args.python,
        str(script_dir / "run_eval_agent.py"),
        "--dataset",
        dataset,
        "--models",
        *models,
        "--split",
        args.split,
        "--stage",
        args.stage,
        "--smoke-limit",
        str(args.smoke_limit),
        "--sample-size",
        str(args.sample_size),
        "--full-limit",
        str(getattr(args, "full_limit", 1000)),
        "--seed",
        str(getattr(args, "seed", 42)),
        "--project-root",
        str(project_root),
        "--output-root",
        str(output_root),
        "--hf-home",
        str(hf_home),
        "--python",
        args.python,
    ]
    if args.timeout:
        command.extend(["--timeout", str(args.timeout)])
    if getattr(args, "allow_label_scores", False):
        command.append("--allow-label-scores")
    if context_path and context_path.exists():
        command.extend(["--context-file", str(context_path)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.continue_on_error:
        command.append("--continue-on-error")
    return command


def run_refinement(
    run_dir: Path,
    candidates_path: Path,
    args: argparse.Namespace,
    script_dir: Path,
    hf_home: Path,
    datasets_cache: Path,
) -> list[dict[str, Any]]:
    commands = [
        [args.python, str(script_dir / "analyze_eval_errors.py"), "--run-dir", str(run_dir), "--candidate-models", str(candidates_path)],
        [args.python, str(script_dir / "make_retrieval_feedback.py"), "--error-analysis", str(run_dir / "error_analysis.json"), "--next-round", "2"],
    ]
    failures = []
    for name, command in zip(["error_analysis", "retrieval_feedback"], commands):
        result = run_command(command, script_dir, hf_home, datasets_cache, args.timeout)
        write_command_logs(run_dir, name, command, result)
        failures.extend(command_failures(str(run_dir), name, result))
        if result.returncode != 0:
            break
    return failures


def run_command(
    command: list[str],
    cwd: Path,
    hf_home: Path,
    datasets_cache: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_home)
    env["HF_DATASETS_CACHE"] = str(datasets_cache)
    env["TRANSFORMERS_CACHE"] = str(hf_home / "hub")
    progress(f"[batch] command start: {command[0]} {' '.join(command[1:3])}")
    result = run_streaming_command(command, cwd, env, timeout)
    progress(f"[batch] command done: returncode={result.returncode}")
    return result


def run_streaming_command(command: list[str], cwd: Path, env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        start_new_session=True,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = start_stream_threads(process, stdout_lines, stderr_lines)
    try:
        returncode = process.wait(timeout=timeout or None)
    except subprocess.TimeoutExpired:
        returncode = 124
        kill_process_group(process)
        stderr_lines.append(f"Timed out after {timeout} seconds: {shell_join(command)}\n")
        print(stderr_lines[-1], end="", flush=True)
    join_stream_threads(threads)
    return subprocess.CompletedProcess(command, returncode, "".join(stdout_lines), "".join(stderr_lines))


def start_stream_threads(
    process: subprocess.Popen[str],
    stdout_lines: list[str],
    stderr_lines: list[str],
) -> list[threading.Thread]:
    threads = [
        threading.Thread(target=collect_stream, args=(process.stdout, stdout_lines), daemon=True),
        threading.Thread(target=collect_stream, args=(process.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()
    return threads


def kill_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


def join_stream_threads(threads: list[threading.Thread]) -> None:
    for thread in threads:
        thread.join(timeout=5)


def collect_stream(stream: Any, lines: list[str]) -> None:
    if stream is None:
        return
    for line in stream:
        lines.append(line)
        print(line, end="", flush=True)


def progress(message: str) -> None:
    print(message, flush=True)



def write_command_logs(run_dir: Path, stem: str, command: list[str], result: subprocess.CompletedProcess[str]) -> None:
    log_dir = run_dir / "batch_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{stem}.command.txt").write_text(shell_join(command) + "\n", encoding="utf-8")
    (log_dir / f"{stem}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{stem}.stderr.txt").write_text(result.stderr, encoding="utf-8")


def command_failures(dataset: str, stage: str, result: subprocess.CompletedProcess[str]) -> list[dict[str, Any]]:
    if result.returncode == 0:
        return []
    return [{"dataset": dataset, "stage": stage, "returncode": result.returncode}]


def internal_eval_failures(run_dir: Path) -> list[dict[str, Any]]:
    failures_path = run_dir / "failures.json"
    if not failures_path.exists():
        return []
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    return [{"dataset": str(run_dir), "stage": "internal_eval", **failure} for failure in failures]


def missing_or_bad_result_failures(results_csv: Path) -> list[dict[str, Any]]:
    if not results_csv.exists():
        return [{"stage": "result_audit", "status": "missing", "notes": f"{results_csv} is missing."}]
    rows = read_eval_rows(results_csv)
    if not rows:
        return [{"stage": "result_audit", "status": "empty", "notes": f"{results_csv} has no rows."}]
    return bad_result_failures(rows)


def bad_result_failures(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        if row.get("status") == "planned" or measured_success(
            row.get("status"), row.get("score"), row.get("labeled_total")
        ):
            continue
        failures.append({
            "dataset": row.get("dataset"),
            "model": row.get("model"),
            "stage": "result_audit",
            "status": row.get("status"),
            "notes": row.get("notes"),
        })
    return failures


def expected_result_failures(condition_rows: list[dict[str, str]], results_csv: Path) -> list[dict[str, Any]]:
    if not results_csv.exists():
        return [{"stage": "result_audit", "status": "missing", "notes": f"{results_csv} is missing."}]
    eval_rows = index_eval_results(results_csv)
    failures = []
    for row in condition_rows:
        if (row["query_dataset"], row["model_name"]) in eval_rows:
            continue
        failures.append({
            "dataset": row["query_dataset"],
            "model": row["model_name"],
            "stage": "result_audit",
            "status": "missing",
            "notes": "Expected dataset/model pair is absent from results.csv.",
        })
    return failures


def dedupe_failures(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for failure in failures:
        key = json.dumps(failure, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(failure)
    return unique


def join_results(condition_rows: list[dict[str, str]], results_csv: Path) -> list[dict[str, Any]]:
    eval_rows = index_eval_results(results_csv)
    joined = []
    for row in condition_rows:
        eval_row = eval_rows.get((row["query_dataset"], row["model_name"]), {})
        joined.append({
            **row,
            "eval_split": eval_row.get("split", ""),
            "eval_protocol": eval_row.get("evaluation_protocol", ""),
            "eval_metric": eval_row.get("metric", ""),
            "eval_score": eval_row.get("score", ""),
            "eval_total": eval_row.get("total", ""),
            "eval_labeled_total": eval_row.get("labeled_total", ""),
            "eval_status": eval_row.get("status", "missing"),
            "eval_output_dir": eval_row.get("output_dir", ""),
            "eval_notes": eval_row.get("notes", ""),
        })
    return joined


def index_eval_results(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["dataset"], row["model"]): row for row in read_eval_rows(path)}


def read_eval_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cleanup_cache_dirs(hf_home: Path, datasets_cache: Path) -> None:
    for path in [hf_home / "hub", hf_home / "transformers", datasets_cache]:
        if path.exists():
            shutil.rmtree(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else batch_fields()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def batch_fields() -> list[str]:
    return [
        "condition",
        "query_dataset",
        "rank",
        "model_name",
        "recommendation_score",
        "source_method",
        "reasoning",
        "eval_split",
        "eval_protocol",
        "eval_metric",
        "eval_score",
        "eval_total",
        "eval_labeled_total",
        "eval_status",
        "eval_output_dir",
        "eval_notes",
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def tail(text: str, limit: int = 2000) -> str:
    return (text or "")[-limit:]


def shell_join(command: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def parse_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


if __name__ == "__main__":
    main()

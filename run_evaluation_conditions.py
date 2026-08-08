#!/usr/bin/env python3
"""Run evaluation-agent jobs from a partner-provided conditions CSV.

Examples:
  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --stage full

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
    --stage full \
    --runner codex \
    --trust-remote-code \
    --cleanup-cache after-dataset \
    --skip-refinement

  python run_evaluation_conditions.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset fancyzhx/ag_news \
    --runner codex \
    --codex-timeout 120 \
    --no-codex-fallback-script
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from eval_agent_core import safe_name


REQUIRED_COLUMNS = {"condition", "query_dataset", "rank", "model_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face dataset loading scripts to run.")
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="full")
    parser.add_argument("--runner", choices=("script", "codex"), default="script")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-approval", default="never")
    parser.add_argument("--codex-timeout", type=int, default=120, help="Seconds to wait for Codex before falling back or failing.")
    parser.add_argument("--codex-fallback-script", action="store_true", default=True, help="Run the script runner if Codex does not produce usable results.")
    parser.add_argument("--no-codex-fallback-script", action="store_false", dest="codex_fallback_script")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=20)
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
    batch_rows = []
    failures = []

    for dataset, dataset_rows in grouped.items():
        result = run_dataset_group(dataset, dataset_rows, args, script_dir, project_root, output_root, hf_home, datasets_cache)
        batch_rows.extend(join_results(dataset_rows, result["results_csv"]))
        failures.extend(result["failures"])
        if args.cleanup_cache == "after-dataset":
            cleanup_cache_dirs(hf_home, datasets_cache)
        if result["failures"] and not args.continue_on_error:
            break

    write_csv(output_root / "batch_results.csv", batch_rows)
    write_json(output_root / "batch_failures.json", failures)
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
        failures = run_codex_with_fallback(dataset, rows, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)
    else:
        failures = run_script_dataset_group(dataset, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)
    failures.extend(expected_result_failures(rows, run_dir / "results.csv"))
    return {"results_csv": run_dir / "results.csv", "failures": dedupe_failures(failures)}



def run_codex_with_fallback(
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
    codex_failures = run_codex_dataset_group(dataset, rows, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)
    if not should_fallback_to_script(args, rows, run_dir / "results.csv", codex_failures):
        return codex_failures
    write_json(run_dir / "codex_fallback.json", {"codex_failures": codex_failures})
    return run_script_dataset_group(dataset, run_dir, candidates_path, args, script_dir, project_root, output_root, hf_home, datasets_cache)


def should_fallback_to_script(args: argparse.Namespace, rows: list[dict[str, str]], results_csv: Path, failures: list[dict[str, Any]]) -> bool:
    if not args.codex_fallback_script:
        return False
    return bool(failures or expected_result_failures(rows, results_csv))


def prepare_run_dir(run_dir: Path, fresh: bool) -> None:
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

def unique_models(rows: list[dict[str, str]]) -> list[str]:
    return list(dict.fromkeys(row["model_name"] for row in rows))


def write_candidates(run_dir: Path, dataset: str, rows: list[dict[str, str]]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for row in rows:
        candidates.append({
            "model": row["model_name"],
            "rank": parse_int(row.get("rank")),
            "score": parse_float(row.get("recommendation_score")),
            "condition": row.get("condition"),
            "source": row.get("source_method"),
            "reason": row.get("reasoning"),
        })
    path = run_dir / "candidate_models_from_conditions.json"
    write_json(path, {"dataset": dataset, "retrieval_agent": "partner_conditions_csv", "candidates": candidates})
    return path


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
) -> list[dict[str, Any]]:
    command = eval_command(args, script_dir, project_root, output_root, dataset, unique_models_from_candidates(candidates_path), hf_home)
    result = run_command(command, script_dir, hf_home, datasets_cache, args.timeout)
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
    prompt = build_codex_prompt(dataset, rows, run_dir, candidates_path, args, project_root, output_root, hf_home, datasets_cache)
    prompt_path = run_dir / "codex_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    command = codex_command(args, script_dir, prompt)
    result = run_command(command, script_dir, hf_home, datasets_cache, args.codex_timeout)
    write_command_logs(run_dir, "codex", command_without_prompt(command), result)
    failures = command_failures(dataset, "codex", result)
    failures.extend(internal_eval_failures(run_dir))
    failures.extend(missing_or_bad_result_failures(run_dir / "results.csv"))
    return failures


def codex_command(args: argparse.Namespace, script_dir: Path, prompt: str) -> list[str]:
    return [args.codex_bin, "-C", str(script_dir), "-a", args.codex_approval, "exec", prompt]


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
) -> str:
    model_ids = unique_models(rows)
    model_bullets = "\n".join(f"- {model}" for model in model_ids)
    model_args = " ".join(model_ids)
    trust_flag = " --trust-remote-code" if args.trust_remote_code else ""
    refinement = "Skip refinement." if args.skip_refinement else "Run analyze_eval_errors.py and make_retrieval_feedback.py after evaluation."
    cleanup = cleanup_instruction(args.cleanup_cache)
    lines = [
        "You are in the hf-eval-agent repo.",
        "",
        "Evaluate the following partner-recommended dataset/model group using the local scripts, and adapt if the generic evaluator fails.",
        "",
        "Dataset:",
        f"- {dataset}",
        "",
        "Models:",
        model_bullets,
        "",
        "Partner candidate metadata:",
        f"- {candidates_path}",
        "",
        "Runtime paths:",
        f"- PROJECT_ROOT={project_root}",
        f"- OUTPUT_ROOT={output_root}",
        f"- RUN_DIR={run_dir}",
        f"- HF_HOME={hf_home}",
        f"- HF_DATASETS_CACHE={datasets_cache}",
        "",
        "Required command shape to start from:",
        f"python run_eval_agent.py --dataset {dataset} --models {model_args} --split {args.split} --stage {args.stage} --smoke-limit {args.smoke_limit} --sample-size {args.sample_size} --project-root {project_root} --output-root {output_root} --hf-home {hf_home} --python {args.python} --continue-on-error{trust_flag}",
        "",
        "Rules:",
        "- Do not ask follow-up questions.",
        "- Prefer the existing repo scripts over writing new code: run run_eval_agent.py first, then analyze_eval_errors.py and make_retrieval_feedback.py when refinement is enabled.",
        "- Only edit evaluator code when a generic local script cannot handle a real compatibility issue.",
        "- Do not use tokens exploring unrelated alternatives after the required local script path is clear.",
        "- Use the candidate metadata file when running refinement.",
        "- Always inspect results.csv, failures.json, audits, and stderr logs before deciding whether evaluation succeeded.",
        "- If a model is gated, missing, incompatible, unsupported, or too large, record the exact failure and continue.",
        "- If a model family needs a small generic evaluator fix, implement it, test it, and rerun the affected smoke/full command.",
        "- Do not delete evaluation outputs.",
        f"- {refinement}",
        f"- {cleanup}",
        "",
        "Final requirements:",
        "- Ensure RUN_DIR/results.csv exists.",
        "- Ensure RUN_DIR/failures.json exists if run_eval_agent.py ran.",
        "- If refinement is enabled, ensure RUN_DIR/error_analysis.json and RUN_DIR/retrieval_feedback.json exist when possible.",
        "- Summarize final scores and unresolved failures.",
    ]
    return "\n".join(lines) + "\n"


def cleanup_instruction(mode: str) -> str:
    if mode == "none":
        return "Keep downloaded Hugging Face caches."
    if mode == "end":
        return "The outer batch runner will clean caches after the full batch; do not clean them inside Codex."
    return "You may clean only downloaded Hugging Face caches after this dataset group, never eval_results."


def eval_command(
    args: argparse.Namespace,
    script_dir: Path,
    project_root: Path,
    output_root: Path,
    dataset: str,
    models: list[str],
    hf_home: Path,
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
    try:
        return subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout or None, check=False)
    except subprocess.TimeoutExpired as error:
        return timeout_result(command, error)


def timeout_result(command: list[str], error: subprocess.TimeoutExpired) -> subprocess.CompletedProcess[str]:
    stdout = text_or_empty(error.stdout)
    stderr = text_or_empty(error.stderr)
    stderr = f"{stderr}\nTimed out after {error.timeout} seconds: {shell_join(command)}\n".lstrip()
    return subprocess.CompletedProcess(command, 124, stdout, stderr)


def text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


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
        if row.get("status") in {"ok", "warn", "planned"}:
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

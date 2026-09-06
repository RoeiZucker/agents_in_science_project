#!/usr/bin/env python3
"""Run conditions evaluation as download -> evaluate -> delete per dataset.

Examples:
  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --stage smoke \
    --runner codex \
    --limit-datasets 2 \
    --limit-pairs-per-dataset 2 \
    --codex-bypass-sandbox \
    --keep-dataset-cache \
    --keep-model-cache \
    --trust-remote-code

  # Full pipeline restricted by a durable dataset allowlist.
  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --full-limit 1000 \
    --stage full \
    --dataset-subset-file config/full_pipeline_datasets.txt \
    --runner script


  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --stage smoke \
    --runner codex \
    --trust-remote-code

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --skip-model-download \
    --stage smoke

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --full-limit 1000 \
    --project-root runtime \
    --condition A_merged \
    --stage full \
    --dataset-subset-file config/full_pipeline_datasets.txt \
    --runner script \
    --trust-remote-code

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --dataset-subset-file config/full_pipeline_datasets.txt \
    --fail-version-incompatible-datasets

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset TimSchopf/medical_abstracts \
    --limit-pairs 2 \
    --stage smoke \
    --runner codex \
    --trust-remote-code

  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 3 \
    --stage smoke \
    --runner codex \
    --codex-timeout 120 \
    --trust-remote-code \
    --allow-partial-download

  # Evaluate one dataset with manually authored context and no Codex scout.
  python run_condition_dataset_cycle.py \
    --conditions-csv ../evaluation_conditions.csv \
    --project-root runtime \
    --dataset ImperialCollegeLondon/health_fact \
    --limit-pairs 2 \
    --stage smoke \
    --runner manual \
    --context-file config/health_fact_context.json \
    --keep-dataset-cache \
    --keep-model-cache

  Direct encoder/classifier scoring is selected automatically when generation is unavailable.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from eval_agent_core import safe_name


REQUIRED_COLUMNS = {"condition", "query_dataset", "rank", "model_name"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split", default="auto")
    parser.add_argument("--download-split", action="append", default=[])
    parser.add_argument("--trust-remote-code", action="store_true")
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
    parser.add_argument("--limit-pairs-per-dataset", type=int, default=0, help="Keep this many condition/model rows per selected dataset.")
    parser.add_argument("--dataset-subset-file", type=Path, help="Keep only dataset ids listed one per line for this pipeline run.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_false", dest="continue_on_error")
    parser.add_argument("--skip-refinement", action="store_true")
    parser.add_argument("--fresh-run-dir", action="store_true")
    parser.add_argument("--keep-dataset-cache", action="store_true")
    parser.add_argument("--keep-model-cache", action="store_true")
    parser.add_argument("--skip-model-download", action="store_true", help="Only predownload datasets, not model snapshots.")
    parser.add_argument("--allow-partial-download", action="store_true", default=True, help="Evaluate when datasets downloaded but some model snapshots failed.")
    parser.add_argument("--fail-partial-download", action="store_false", dest="allow_partial_download")
    parser.add_argument("--skip-version-incompatible-datasets", action="store_true", default=True)
    parser.add_argument("--fail-version-incompatible-datasets", action="store_false", dest="skip_version_incompatible_datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_full_subset(args)
    script_dir = Path(__file__).resolve().parent
    datasets = selected_datasets(args)
    validate_runner_context(args, datasets)
    progress(f"[cycle] selected {len(datasets)} datasets; runner={args.runner}; stage={args.stage}; output={output_root(args)}")
    results = run_cycle(datasets, args, script_dir)
    write_cycle_report(args, results)
    write_cycle_batch_results(args)
    print(json.dumps(summary(args, results), indent=2))
    if has_failure(results):
        sys.exit(1)


def validate_runner_context(args: argparse.Namespace, datasets: list[str]) -> None:
    context_file = getattr(args, "context_file", None)
    if args.runner != "manual":
        if context_file:
            raise ValueError("--context-file requires --runner manual.")
        return
    if not context_file:
        raise ValueError("--runner manual requires --context-file.")
    if len(datasets) != 1:
        raise ValueError("Manual context mode requires exactly one selected dataset.")
    if not context_file.is_file():
        raise FileNotFoundError(f"Manual context file does not exist: {context_file}")


def require_full_subset(args: argparse.Namespace) -> None:
    if args.stage == "full" and not args.dataset_subset_file:
        raise ValueError("Full condition-cycle runs require --dataset-subset-file.")


def selected_datasets(args: argparse.Namespace) -> list[str]:
    rows = selected_rows(args)
    names = unique([row["query_dataset"] for row in rows])
    if args.limit_datasets:
        names = names[: args.limit_datasets]
    return names


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_conditions(args.conditions_csv)
    if args.condition:
        rows = [row for row in rows if row["condition"] in set(args.condition)]
    if args.dataset:
        rows = [row for row in rows if row["query_dataset"] in set(args.dataset)]
    if getattr(args, "dataset_subset_file", None):
        allowed = set(read_dataset_subset(args.dataset_subset_file))
        rows = [row for row in rows if row["query_dataset"] in allowed]
    if args.limit_pairs_per_dataset:
        rows = limit_rows_per_dataset(rows, args.limit_pairs_per_dataset)
    if args.limit_pairs:
        rows = rows[: args.limit_pairs]
    return rows



def read_dataset_subset(path: Path) -> list[str]:
    datasets = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        value = raw_line.split("#", 1)[0].strip()
        if value and value not in datasets:
            datasets.append(value)
    if not datasets:
        raise ValueError(f"Dataset subset file is empty: {path}")
    return datasets

def limit_rows_per_dataset(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    counts: dict[str, int] = {}
    selected = []
    for row in rows:
        dataset = row["query_dataset"]
        count = counts.get(dataset, 0)
        if count >= limit:
            continue
        selected.append(row)
        counts[dataset] = count + 1
    return selected


def read_conditions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("conditions CSV is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"conditions CSV is missing columns: {sorted(missing)}")


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def run_cycle(datasets: list[str], args: argparse.Namespace, script_dir: Path) -> list[dict[str, Any]]:
    results = []
    total = len(datasets)
    for index, dataset in enumerate(datasets, start=1):
        progress(f"[cycle] dataset {index}/{total} start: {dataset}")
        result = run_dataset_cycle(dataset, args, script_dir)
        progress(f"[cycle] dataset {index}/{total} done: {dataset} status={result['status']}")
        results.append(result)
        if result["status"] == "failed" and not args.continue_on_error:
            break
    return results


def run_dataset_cycle(dataset: str, args: argparse.Namespace, script_dir: Path) -> dict[str, Any]:
    steps = []
    steps.append(run_step("download", download_command(dataset, args, script_dir), script_dir))
    if download_skipped(steps[-1]):
        steps[-1]["skipped_datasets"] = reported_skips(steps[-1]["stdout_tail"])
    elif should_evaluate_after_download(steps[-1], args):
        if steps[-1]["returncode"] != 0:
            steps[-1]["partial_download_allowed"] = True
        steps.append(evaluate_step(evaluate_command(dataset, args, script_dir), script_dir))
    if not args.keep_dataset_cache:
        steps.append(run_step("delete", delete_command(dataset, args, script_dir), script_dir))
    return {"dataset": dataset, "status": status(steps), "steps": steps}


def run_step(name: str, command: list[str], cwd: Path) -> dict[str, Any]:
    progress(f"[cycle] step start: {name}")
    result = run_streaming_command(command, cwd)
    progress(f"[cycle] step done: {name} returncode={result.returncode}")
    return {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": tail(result.stdout),
        "stderr_tail": tail(result.stderr),
    }


def run_streaming_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        threading.Thread(target=collect_stream, args=(process.stdout, stdout_lines), daemon=True),
        threading.Thread(target=collect_stream, args=(process.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()
    returncode = process.wait()
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(command, returncode, "".join(stdout_lines), "".join(stderr_lines))


def collect_stream(stream: Any, lines: list[str]) -> None:
    if stream is None:
        return
    for line in stream:
        lines.append(line)
        print(line, end="", flush=True)


def progress(message: str) -> None:
    print(message, flush=True)


def evaluate_step(command: list[str], cwd: Path) -> dict[str, Any]:
    step = run_step("evaluate", command, cwd)
    failures = reported_failures(step["stdout_tail"])
    if failures:
        step["returncode"] = 1
        step["reported_failures"] = failures
    return step


def reported_failures(text: str) -> int:
    value = summary_json(text)
    if not value:
        return 0
    return int(value.get("failures", 0))


def should_evaluate_after_download(step: dict[str, Any], args: argparse.Namespace) -> bool:
    if step["returncode"] == 0:
        return True
    if not args.allow_partial_download:
        return False
    value = summary_json(step["stdout_tail"])
    if not value:
        return False
    return dataset_download_succeeded(value) and has_usable_download(args, value)


def dataset_download_succeeded(value: dict[str, Any]) -> bool:
    return int(value.get("failed_datasets", 0)) == 0 and int(value.get("skipped_datasets", 0)) == 0


def has_usable_download(args: argparse.Namespace, value: dict[str, Any]) -> bool:
    if args.skip_model_download:
        return True
    return int(value.get("downloaded_models", 0)) > 0


def download_skipped(step: dict[str, Any]) -> bool:
    return step["returncode"] == 0 and reported_skips(step["stdout_tail"]) > 0


def reported_skips(text: str) -> int:
    value = summary_json(text)
    if not value:
        return 0
    return int(value.get("skipped_datasets", 0))


def summary_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    values = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    return values[-1] if values else {}


def download_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [args.python, str(script_dir / "download_condition_datasets.py"), "--conditions-csv", str(args.conditions_csv), "--project-root", str(args.project_root), "--dataset", dataset]
    command.extend(download_split_args(args))
    if args.output_root:
        command.extend(["--output-root", str(args.output_root / "_dataset_downloads")])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if not args.skip_model_download:
        command.append("--include-models")
    pair_limit = dataset_pair_limit(args)
    if pair_limit:
        command.extend(["--limit-pairs", str(pair_limit)])
    for condition in args.condition:
        command.extend(["--condition", condition])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if not args.skip_version_incompatible_datasets:
        command.append("--fail-version-incompatible-datasets")
    return command


def dataset_pair_limit(args: argparse.Namespace) -> int:
    return args.limit_pairs_per_dataset or args.limit_pairs


def download_split_args(args: argparse.Namespace) -> list[str]:
    return [item for split in args.download_split for item in ["--split", split]]


def evaluate_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [
        args.python,
        str(script_dir / "run_evaluation_conditions.py"),
        "--conditions-csv",
        str(args.conditions_csv),
        "--project-root",
        str(args.project_root),
        "--split",
        args.split,
        "--stage",
        args.stage,
        "--runner",
        args.runner,
        "--codex-bin",
        args.codex_bin,
        "--codex-approval",
        args.codex_approval,
        "--codex-timeout",
        str(args.codex_timeout),
        "--codex-context-attempts",
        str(getattr(args, "codex_context_attempts", 2)),
        "--smoke-limit",
        str(args.smoke_limit),
        "--sample-size",
        str(args.sample_size),
        "--full-limit",
        str(getattr(args, "full_limit", 1000)),
        "--seed",
        str(getattr(args, "seed", 42)),
        "--dataset",
        dataset,
        "--python",
        args.python,
        "--cleanup-cache",
        "none",
    ]
    command.extend(optional_eval_args(args))
    return command


def optional_eval_args(args: argparse.Namespace) -> list[str]:
    values = []
    if args.output_root:
        values.extend(["--output-root", str(args.output_root)])
    if args.context_file:
        values.extend(["--context-file", str(args.context_file)])
    if args.codex_bypass_sandbox:
        values.append("--codex-bypass-sandbox")
    if getattr(args, "allow_label_scores", False):
        values.append("--allow-label-scores")
    if args.timeout:
        values.extend(["--timeout", str(args.timeout)])
    pair_limit = dataset_pair_limit(args)
    if pair_limit:
        values.extend(["--limit-pairs", str(pair_limit)])
    if args.trust_remote_code:
        values.append("--trust-remote-code")
    if args.continue_on_error:
        values.append("--continue-on-error")
    if args.skip_refinement:
        values.append("--skip-refinement")
    if args.fresh_run_dir:
        values.append("--fresh-run-dir")
    for condition in args.condition:
        values.extend(["--condition", condition])
    return values


def delete_command(dataset: str, args: argparse.Namespace, script_dir: Path) -> list[str]:
    command = [args.python, str(script_dir / "delete_condition_datasets.py"), "--project-root", str(args.project_root), "--dataset", dataset]
    if not args.keep_model_cache:
        command.append("--model-cache")
    if args.output_root:
        command.extend(["--output-root", str(args.output_root / "_dataset_downloads")])
    return command


def status(steps: list[dict[str, Any]]) -> str:
    if any(step.get("skipped_datasets", 0) for step in steps):
        return "skipped"
    return "ok" if all(step_succeeded(step) for step in steps) else "failed"


def step_succeeded(step: dict[str, Any]) -> bool:
    return step["returncode"] == 0 or bool(step.get("partial_download_allowed"))


def tail(text: str, lines: int = 30) -> str:
    return "\n".join(text.splitlines()[-lines:])


def has_failure(results: list[dict[str, Any]]) -> bool:
    return any(result["status"] == "failed" for result in results)


def write_cycle_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    path = output_root(args) / "dataset_cycle_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")



def write_cycle_batch_results(args: argparse.Namespace) -> None:
    rows = joined_cycle_rows(args)
    write_csv(output_root(args) / "batch_results.csv", rows)


def joined_cycle_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    eval_rows = all_eval_rows(args)
    datasets = set(selected_datasets(args))
    rows = []
    for row in selected_rows(args):
        if row["query_dataset"] not in datasets:
            continue
        eval_row = eval_rows.get((row["query_dataset"], row["model_name"]), {})
        rows.append(join_condition_row(row, eval_row))
    return rows


def all_eval_rows(args: argparse.Namespace) -> dict[tuple[str, str], dict[str, str]]:
    rows = {}
    for dataset in selected_datasets(args):
        for row in read_result_rows(result_csv_path(args, dataset)):
            rows[(row.get("dataset", ""), row.get("model", ""))] = row
    return rows


def result_csv_path(args: argparse.Namespace, dataset: str) -> Path:
    return output_root(args) / f"{safe_name(dataset)}_{args.split}" / "results.csv"


def output_root(args: argparse.Namespace) -> Path:
    return args.output_root or args.project_root / "eval_results" / "_condition_runs"


def join_condition_row(row: dict[str, str], eval_row: dict[str, str]) -> dict[str, Any]:
    return {
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
    }


def read_result_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def summary(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    root = output_root(args)
    return {
        "datasets": len(results),
        "skipped_datasets": sum(1 for result in results if result["status"] == "skipped"),
        "failed_datasets": sum(1 for result in results if result["status"] == "failed"),
        "report": str(root / "dataset_cycle_report.json"),
        "batch_results_csv": str(root / "batch_results.csv"),
    }


if __name__ == "__main__":
    main()

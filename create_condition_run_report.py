#!/usr/bin/env python3
"""Create a readable report from a condition-dataset-cycle output directory.

Examples:
  python create_condition_run_report.py \
    --run-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real \
    --output /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real/condition_run_report.md

  python create_condition_run_report.py \
    --run-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real \
    --output /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real/condition_run_report.md \
    --csv-output /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_full_codex_real/condition_run_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from result_contract import measured_success, result_outcome

MAX_TEXT = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path, help="Directory containing batch_results.csv and *_auto run dirs.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown report path. Defaults to RUN_ROOT/condition_run_report.md.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional CSV report path. Defaults to RUN_ROOT/condition_run_report.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.run_root
    output = args.output or run_root / "condition_run_report.md"
    rows = dedupe_pair_rows(read_batch_rows(run_root))
    contexts = load_dataset_contexts(run_root)
    failures = load_failure_reasons(run_root)
    report = build_report(run_root, rows, contexts, failures)
    output.write_text(report, encoding="utf-8")
    csv_output = args.csv_output or run_root / "condition_run_report.csv"
    write_csv_report(csv_output, rows, contexts, failures)
    print(json.dumps({"report": str(output), "csv_report": str(csv_output), "dataset_model_pairs": len(rows)}, indent=2))


def read_batch_rows(run_root: Path) -> list[dict[str, str]]:
    path = run_root / "batch_results.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def dedupe_pair_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("query_dataset", ""), row.get("model_name", ""))].append(row)
    return [merge_pair_rows(items) for items in grouped.values()]


def merge_pair_rows(rows: list[dict[str, str]]) -> dict[str, str]:
    chosen = choose_representative_row(rows).copy()
    chosen["conditions"] = ", ".join(compact_condition(row) for row in rows)
    chosen["source_methods"] = ", ".join(dedupe([row.get("source_method", "") for row in rows]))
    return chosen


def choose_representative_row(rows: list[dict[str, str]]) -> dict[str, str]:
    priority = {"success": 0, "planned": 1, "failure": 2}
    return min(rows, key=lambda row: priority[row_outcome(row)])


def compact_condition(row: dict[str, str]) -> str:
    condition = row.get("condition", "")
    rank = row.get("rank", "")
    return f"{condition}#{rank}" if rank else condition


def load_dataset_contexts(run_root: Path) -> dict[str, dict[str, Any]]:
    contexts = {}
    for path in sorted(run_root.glob("*_auto/codex_context.json")):
        dataset = dataset_from_run_dir(path.parent)
        contexts[dataset] = read_json(path)
    return contexts


def dataset_from_run_dir(run_dir: Path) -> str:
    candidates = read_candidates(run_dir / "candidate_models_from_conditions.json")
    if candidates.get("dataset"):
        return str(candidates["dataset"])
    return run_dir.name.removesuffix("_auto").replace("_", "/", 1)


def read_candidates(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def load_failure_reasons(run_root: Path) -> dict[tuple[str, str], list[str]]:
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    add_batch_failures(reasons, run_root)
    for run_dir in sorted(run_root.glob("*_auto")):
        dataset = dataset_from_run_dir(run_dir)
        add_run_failures(reasons, dataset, run_dir)
        add_audit_failures(reasons, dataset, run_dir)
        add_log_tails(reasons, dataset, run_dir)
    return {key: dedupe(values) for key, values in reasons.items()}


def add_batch_failures(reasons: dict[tuple[str, str], list[str]], run_root: Path) -> None:
    for failure in read_json_list(run_root / "batch_failures.json"):
        model = str(failure.get("model") or "")
        dataset = clean_dataset_name(str(failure.get("dataset") or ""))
        if model:
            reasons[(dataset, model)].append(format_failure(failure))


def add_run_failures(reasons: dict[tuple[str, str], list[str]], dataset: str, run_dir: Path) -> None:
    for failure in read_json_list(run_dir / "failures.json"):
        model = str(failure.get("model") or "")
        if model:
            reasons[(dataset, model)].append(format_failure(failure))


def add_audit_failures(reasons: dict[tuple[str, str], list[str]], dataset: str, run_dir: Path) -> None:
    for path in sorted((run_dir / "audits").glob("*_audit.json")):
        audit = read_json(path)
        summary = audit.get("summary") or {}
        if measured_success(audit.get("status"), summary.get("accuracy"), summary.get("labeled_total")):
            continue
        model = model_from_audit_name(path.name)
        text = audit_reason(audit)
        if model and text:
            reasons[(dataset, model)].append(text)


def add_log_tails(reasons: dict[tuple[str, str], list[str]], dataset: str, run_dir: Path) -> None:
    logs = run_dir / "logs"
    if not logs.exists():
        return
    for path in sorted(logs.glob("*.stderr.txt")):
        if path.stat().st_size == 0:
            continue
        model = model_from_log_name(path.name, run_dir)
        if not model:
            continue
        tail = tail_text(path)
        if tail:
            reasons[(dataset, model)].append("stderr tail: " + tail)


def model_from_audit_name(name: str) -> str:
    stem = name.removesuffix("_smoke_audit.json").removesuffix("_full_audit.json")
    return unsafe_model_name(stem)


def model_from_log_name(name: str, run_dir: Path) -> str:
    stem = name.removesuffix("_smoke.stderr.txt").removesuffix("_full.stderr.txt")
    candidates = candidate_models(run_dir)
    by_safe = {safe_name(model): model for model in candidates}
    return by_safe.get(stem, unsafe_model_name(stem))


def candidate_models(run_dir: Path) -> list[str]:
    value = read_candidates(run_dir / "candidate_models_from_conditions.json")
    return [str(item.get("model")) for item in value.get("candidates", []) if item.get("model")]


def write_csv_report(
    path: Path,
    rows: list[dict[str, str]],
    contexts: dict[str, dict[str, Any]],
    failures: dict[tuple[str, str], list[str]],
) -> None:
    fields = csv_fields()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_report_row(row, contexts, failures))


def csv_fields() -> list[str]:
    return [
        "outcome",
        "dataset",
        "model",
        "score",
        "status",
        "evaluation_protocol",
        "metric",
        "eval_total",
        "eval_labeled_total",
        "conditions",
        "source_methods",
        "codex_prompt_template",
        "codex_context_notes",
        "eval_warnings",
        "failure_reason",
        "eval_output_dir",
    ]


def csv_report_row(
    row: dict[str, str],
    contexts: dict[str, dict[str, Any]],
    failures: dict[tuple[str, str], list[str]],
) -> dict[str, str]:
    dataset = row.get("query_dataset", "")
    model = row.get("model_name", "")
    status = row.get("eval_status", "")
    outcome = row_outcome(row)
    context = contexts.get(dataset, {})
    dataset_context = context.get("dataset_context", {}) if isinstance(context, dict) else {}
    failure_reason = "" if outcome != "failure" else choose_reason(row, failures.get((dataset, model), []))
    return {
        "outcome": outcome,
        "dataset": dataset,
        "model": model,
        "score": row.get("eval_score", ""),
        "status": status,
        "evaluation_protocol": row.get("eval_protocol", ""),
        "metric": row.get("eval_metric", ""),
        "eval_total": row.get("eval_total", ""),
        "eval_labeled_total": row.get("eval_labeled_total", ""),
        "conditions": row.get("conditions", ""),
        "source_methods": row.get("source_methods", ""),
        "codex_prompt_template": csv_one_line(str(dataset_context.get("prompt_template") or "")),
        "codex_context_notes": csv_one_line(str(dataset_context.get("notes") or "")),
        "eval_warnings": row.get("eval_notes", "") if outcome == "success" else "",
        "failure_reason": failure_reason,
        "eval_output_dir": row.get("eval_output_dir", ""),
    }


def build_report(
    run_root: Path,
    rows: list[dict[str, str]],
    contexts: dict[str, dict[str, Any]],
    failures: dict[tuple[str, str], list[str]],
) -> str:
    lines = ["# Condition Run Report", ""]
    lines.extend(summary_lines(run_root, rows))
    lines.extend(success_lines(rows, contexts))
    lines.extend(planned_lines(rows))
    lines.extend(failure_lines(rows, contexts, failures))
    return "\n".join(lines).rstrip() + "\n"


def summary_lines(run_root: Path, rows: list[dict[str, str]]) -> list[str]:
    counts = Counter(row_outcome(row) for row in rows)
    datasets = list(dict.fromkeys(row.get("query_dataset", "") for row in rows))
    return [
        "## Summary",
        "",
        f"- Run root: `{run_root}`",
        f"- Dataset-model pairs: {len(rows)}",
        f"- Datasets: {len(datasets)}",
        f"- Measured successes: {counts['success']}",
        f"- Planned only: {counts['planned']}",
        f"- Failed, unsupported, or unmeasured: {counts['failure']}",
        f"- Outcome counts: {dict(sorted(counts.items()))}",
        "",
    ]


def success_lines(rows: list[dict[str, str]], contexts: dict[str, dict[str, Any]]) -> list[str]:
    lines = ["## Successful Dataset-Model Pairs", ""]
    success_rows = [row for row in rows if row_outcome(row) == "success"]
    for dataset, group in grouped_by_dataset(success_rows).items():
        lines.extend([f"### {dataset}", ""])
        for row in group:
            lines.extend(success_item(row, contexts.get(dataset, {})))
    if not success_rows:
        lines.append("No successful rows found.")
    lines.append("")
    return lines


def success_item(row: dict[str, str], context: dict[str, Any]) -> list[str]:
    dataset_context = context.get("dataset_context", {}) if isinstance(context, dict) else {}
    prompt = str(dataset_context.get("prompt_template") or "")
    notes = str(dataset_context.get("notes") or "")
    lines = [
        f"- `{row.get('model_name', '')}`",
        f"  - Score: `{row.get('eval_score', '')}`",
        f"  - Status: `{row.get('eval_status', '')}`",
        f"  - Protocol/metric: `{row.get('eval_protocol', '')}` / `{row.get('eval_metric', '')}`",
        f"  - Eval total/labeled: `{row.get('eval_total', '')}` / `{row.get('eval_labeled_total', '')}`",
        f"  - Conditions: {row.get('conditions', '')}",
        f"  - Codex prompt template: {format_inline_or_block(prompt)}",
    ]
    if notes:
        lines.append(f"  - Codex context notes: {shorten(notes)}")
    if row.get("eval_notes"):
        lines.append(f"  - Eval warnings: {shorten(row['eval_notes'])}")
    return lines


def planned_lines(rows: list[dict[str, str]]) -> list[str]:
    planned = [row for row in rows if row_outcome(row) == "planned"]
    if not planned:
        return []
    lines = ["## Planned But Not Evaluated", ""]
    for row in planned:
        lines.append(f"- `{row.get('query_dataset', '')}` / `{row.get('model_name', '')}`")
    lines.append("")
    return lines


def failure_lines(
    rows: list[dict[str, str]],
    contexts: dict[str, dict[str, Any]],
    failures: dict[tuple[str, str], list[str]],
) -> list[str]:
    lines = ["## Failed Or Missing Dataset-Model Pairs", ""]
    failed_rows = [row for row in rows if row_outcome(row) == "failure"]
    for dataset, group in grouped_by_dataset(failed_rows).items():
        lines.extend([f"### {dataset}", ""])
        for row in group:
            lines.extend(failure_item(row, contexts.get(dataset, {}), failures))
    if not failed_rows:
        lines.append("No failed or missing rows found.")
    lines.append("")
    return lines


def failure_item(
    row: dict[str, str],
    context: dict[str, Any],
    failures: dict[tuple[str, str], list[str]],
) -> list[str]:
    dataset = row.get("query_dataset", "")
    model = row.get("model_name", "")
    reason = choose_reason(row, failures.get((dataset, model), []))
    prompt = context_prompt(context)
    lines = [
        f"- `{model}`",
        f"  - Status: `{row.get('eval_status', '')}`",
        f"  - Protocol/metric: `{row.get('eval_protocol', '')}` / `{row.get('eval_metric', '')}`",
        f"  - Conditions: {row.get('conditions', '')}",
        f"  - Reason: {shorten(reason)}",
    ]
    if prompt:
        lines.append(f"  - Codex prompt template: {format_inline_or_block(prompt)}")
    return lines


def context_prompt(context: dict[str, Any]) -> str:
    if not isinstance(context, dict):
        return ""
    dataset_context = context.get("dataset_context", {})
    if not isinstance(dataset_context, dict):
        return ""
    return str(dataset_context.get("prompt_template") or "")


def choose_reason(row: dict[str, str], reasons: list[str]) -> str:
    parts = []
    if row.get("eval_notes"):
        parts.append(row["eval_notes"])
    parts.extend(reasons)
    if parts:
        return " | ".join(dedupe(parts))
    return "No detailed failure reason found; the row was marked missing or failed without captured stderr."


def row_outcome(row: dict[str, str]) -> str:
    return result_outcome(
        row.get("eval_status"),
        row.get("eval_score"),
        row.get("eval_labeled_total"),
    )


def grouped_by_dataset(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("query_dataset", "")].append(row)
    return dict(groups)


def audit_reason(audit: dict[str, Any]) -> str:
    issues = audit.get("issues") or []
    warnings = audit.get("warnings") or []
    parts = [str(item) for item in issues + warnings if item]
    return " | ".join(parts)


def format_failure(failure: dict[str, Any]) -> str:
    parts = []
    if failure.get("stage"):
        parts.append(f"stage={failure['stage']}")
    if failure.get("returncode") is not None:
        parts.append(f"returncode={failure['returncode']}")
    audit = failure.get("audit") if isinstance(failure.get("audit"), dict) else {}
    audit_text = audit_reason(audit)
    if audit_text:
        parts.append(audit_text)
    if failure.get("notes"):
        parts.append(str(failure["notes"]))
    if failure.get("error"):
        parts.append(str(failure["error"]))
    return "; ".join(parts) or json.dumps(failure, ensure_ascii=False)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def read_json_list(path: Path) -> list[dict[str, Any]]:
    value = read_json(path)
    return value if isinstance(value, list) else []


def tail_text(path: Path, limit: int = 1200) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > limit:
        text = text[-limit:]
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return simplify_error_text(text)


def simplify_error_text(text: str) -> str:
    patterns = [
        r"(HTTPStatusError: Client error '[^']+'[^\n]+)",
        r"(OSError: [^\n]+)",
        r"(ValueError: [^\n]+)",
        r"(RuntimeError: [^\n]+)",
        r"(TypeError: [^\n]+)",
        r"(KeyError: [^\n]+)",
        r"(ImportError: [^\n]+)",
        r"(ModuleNotFoundError: [^\n]+)",
    ]
    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, text))
    return matches[-1] if matches else shorten(text, 500)


def clean_dataset_name(value: str) -> str:
    if value.endswith("_auto"):
        return dataset_from_run_dir(Path(value))
    return value


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))


def unsafe_model_name(value: str) -> str:
    return value.replace("_", "/", 1)


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def csv_one_line(value: str) -> str:
    return "; ".join(part.strip() for part in str(value).splitlines() if part.strip())


def format_inline_or_block(value: str) -> str:
    if not value:
        return "`<none>`"
    if "\n" not in value and len(value) <= 100:
        return f"`{value}`"
    return "\n\n```text\n" + value + "\n```"


def shorten(value: str, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    main()

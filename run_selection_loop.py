#!/usr/bin/env python3
"""Run static or feedback-driven candidate rounds and select measured winners.

After the requested ranked-model minimum is reached, a provider may attach a
retrieval/critic/orchestrator discussion. Unanimous no-improvement consensus
stops the loop; disagreement continues until the hard round limit.

Examples:
  python run_selection_loop.py --manifest config/mock_selection_loop.json \
    --dataset-subset-file config/full_pipeline_datasets.txt --stage plan --runner script

  python run_selection_loop.py --manifest config/artifact_linker_selection_loop.json \
    --stage smoke --runner script --max-rounds 2 --smoke-limit 3

  python run_selection_loop.py --manifest config/artifact_linker_selection_loop.json \
    --stage smoke --runner script --max-rounds 2 --resume

  python run_selection_loop.py --manifest config/oren_local_existing_smoke.json \
    --stage smoke --runner script --max-rounds 3 --target-ranked-models 5 \
    --provider-retries 2 --smoke-limit 200

  python run_selection_loop.py --manifest config/oren_local_cached_18_final.json \
    --dataset-subset-file config/oren_cached_18_datasets.txt --stage full \
    --runner script --max-rounds 5 --target-ranked-models 5 --full-limit 1000

  python run_selection_loop.py --manifest runtime/frozen/frozen_manifest.json \
    --dataset-subset-file config/oren_full_62_shard_1.txt --stage full \
    --runner codex --max-rounds 1 --target-ranked-models 5 --full-limit 100 \
    --skip-refinement
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from result_contract import result_outcome
from selection_loop_full_gate import run_smoke_then_full
from evaluation_context_spec import materialize_context, read_context_template

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_RUNNER = SCRIPT_DIR / "run_evaluation_conditions.py"
DEFAULT_PROVIDER = SCRIPT_DIR / "artifact_linker_provider.py"
HANDOFF_VERSION = 1
SUPPORTED_INTENDED_USES = {"direct_inference"}
HANDOFF_FIELDS = {
    "handoff_version", "owner", "mock", "retrieval_agent",
    "dataset", "split", "round", "candidates",
}
CANDIDATE_FIELDS = {
    "candidate_id", "condition", "model", "rank", "score",
    "source", "intended_use", "reason",
}
CANDIDATE_TEXT_FIELDS = {
    "candidate_id", "condition", "model", "source", "intended_use", "reason",
}
ACCURACY_PROTOCOLS = {
    "multiple_choice_accuracy", "label_generation_accuracy",
    "label_logprob_accuracy", "tagged_label_generation_accuracy",
    "classifier_label_accuracy", "zero_shot_nli_accuracy",
}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-subset-file", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path("runtime"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stage", choices=("plan", "smoke", "full"), default="smoke")
    parser.add_argument("--runner", choices=("script", "codex"), default="script")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-timeout", type=int, default=120)
    parser.add_argument("--codex-bypass-sandbox", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--allow-label-scores", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--full-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--target-ranked-models", type=int, default=0)
    parser.add_argument("--provider-retries", type=int, default=2)
    parser.add_argument("--minimum-measured-improvement", type=float, default=0.0)
    parser.add_argument(
        "--evaluation-failure-policy", choices=("continue", "stop"), default="continue"
    )
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Run candidate evaluation without the post-evaluation refinement agent.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_args(args)
    require_full_subset(args)
    specs = selected_specs(load_manifest(args.manifest), args.dataset_subset_file)
    validate_target_capacity(specs, args)
    root = args.output_root or args.project_root / "eval_results" / "_selection_loop"
    records = run_specs(specs, args, root)
    rows = flatten_results(records)
    selected = select_winners(rows)
    ranked = rank_models(rows, args.target_ranked_models or 1)
    summary = make_summary(
        records, rows, selected, args.target_ranked_models, root / "ranked_models.csv"
    )
    write_csv(root / "selection_loop_results.csv", rows, result_fields())
    write_csv(root / "selected_models.csv", selected, result_fields())
    write_csv(root / "ranked_models.csv", ranked, ranked_result_fields())
    write_json(root / "selection_loop_summary.json", summary)
    print(json.dumps(summary, indent=2))
    failed = any(row["outcome"] == "failure" for row in selected)
    if args.stage != "plan" and (failed or summary["datasets_below_target"]):
        raise SystemExit(1)


def normalize_args(args: argparse.Namespace) -> None:
    if args.stop_on_error:
        args.evaluation_failure_policy = "stop"
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1.")
    if getattr(args, "target_ranked_models", 0) < 0:
        raise ValueError("--target-ranked-models cannot be negative.")
    if getattr(args, "provider_retries", 0) < 0:
        raise ValueError("--provider-retries cannot be negative.")
    if args.minimum_measured_improvement < 0:
        raise ValueError("--minimum-measured-improvement cannot be negative.")


def require_full_subset(args: argparse.Namespace) -> None:
    if args.stage == "full" and not args.dataset_subset_file:
        raise ValueError("Full selection-loop runs require --dataset-subset-file.")


def validate_target_capacity(
    specs: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    target = getattr(args, "target_ranked_models", 0)
    if target <= 0:
        return
    for spec in specs:
        provider = spec.get("provider") or {}
        if not provider:
            continue
        per_round = int(provider.get("max_candidates", 5))
        capacity = args.max_rounds * per_round
        if capacity < target:
            raise ValueError(
                f"Dataset {spec['dataset']!r} can retrieve at most {capacity} models "
                f"({args.max_rounds} rounds x {per_round} candidates), below target {target}."
            )


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    specs = value.get("datasets", [])
    if not specs:
        raise ValueError("Selection-loop manifest has no datasets.")
    defaults = value.get("provider_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("provider_defaults must be an object.")
    return [merge_provider_defaults(spec, defaults) for spec in specs]


def merge_provider_defaults(
    spec: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    if "provider" not in spec or not defaults:
        return spec
    provider = spec["provider"]
    if not isinstance(provider, dict):
        raise ValueError("Dataset provider must be an object.")
    return {**spec, "provider": {**defaults, **provider}}


def selected_specs(specs: list[dict[str, Any]], subset_path: Path | None) -> list[dict[str, Any]]:
    selected = specs
    if subset_path:
        allowed = set(read_subset(subset_path))
        selected = [spec for spec in specs if spec.get("dataset") in allowed]
    if not selected:
        raise ValueError("No manifest datasets match the selected subset.")
    if any(not spec.get("rounds") and not spec.get("provider") for spec in selected):
        raise ValueError("Every selected dataset must define a candidate round or provider.")
    return selected


def read_subset(path: Path) -> list[str]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.split("#", 1)[0].strip()
        if value and value not in values:
            values.append(value)
    if not values:
        raise ValueError(f"Dataset subset file is empty: {path}")
    return values


def run_specs(specs: list[dict[str, Any]], args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        records.extend(run_dataset(spec, args, root))
    return records


def run_dataset(spec: dict[str, Any], args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    dataset = spec["dataset"]
    split = spec.get("split", "auto")
    static = {int(item["round"]): item for item in spec.get("rounds", [])}
    provider = spec.get("provider")
    records: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    canonical: dict[str, str] | None = None
    previous_best: float | None = None
    stop_reason = "max_rounds"

    for round_number in range(1, args.max_rounds + 1):
        round_dir = root / safe_name(dataset) / f"round_{round_number}"
        try:
            handoff, source = obtain_handoff(
                spec, static.get(round_number), provider, records, seen_models,
                args, round_dir, round_number,
            )
        except ProviderError as exc:
            records.append(stop_record(
                dataset, split, round_number, round_dir, "provider_failure",
                str(exc), include_candidate_result=not records,
            ))
            stop_reason = "provider_failure"
            break
        if handoff is None:
            stop_reason = source
            records.append(stop_record(
                dataset, split, round_number, round_dir, source, source,
                include_candidate_result=not records,
            ))
            break
        if should_stop_from_deliberation(
            handoff, records, getattr(args, "target_ranked_models", 0)
        ):
            stop_reason = "agent_consensus_no_likely_improvement"
            record_stopping_deliberation(records[-1], handoff, round_dir)
            break
        handoff, duplicate_count = deduplicate_handoff(handoff, seen_models)
        if not handoff["candidates"]:
            stop_reason = "no_new_candidates" if duplicate_count else "no_valid_candidates"
            records.append(stop_record(
                dataset, split, round_number, round_dir, stop_reason, stop_reason,
                include_candidate_result=not records,
            ))
            break
        write_json(round_dir / "candidate_handoff.json", handoff)
        validate_handoff(handoff, round_dir / "candidate_handoff.json", dataset, split, round_number)
        seen_models.update(candidate["model"].casefold() for candidate in handoff["candidates"])

        record = run_or_resume_round(spec, handoff, source, args, root, round_dir)
        saved_canonical = record.get("canonical_evaluation") or {}
        if canonical is None and saved_canonical:
            canonical = saved_canonical
        canonical = annotate_comparability(record["candidate_results"], canonical)
        record["canonical_evaluation"] = canonical or {}
        attach_continuation_review(record, handoff, round_dir)
        finalize_round_feedback(record)
        records.append(record)
        write_round_summary(record)
        if record["returncode"] and args.evaluation_failure_policy == "stop":
            stop_reason = "evaluation_failure"
            break
        current_best = best_comparable_score(record["candidate_results"])
        if should_stop_for_improvement(previous_best, current_best, args.minimum_measured_improvement):
            stop_reason = "minimum_measured_improvement"
            break
        previous_best = max_optional(previous_best, current_best)
        if round_number == args.max_rounds:
            stop_reason = "max_rounds"
            break
        if not provider and round_number + 1 not in static:
            stop_reason = "static_manifest_complete"
            break

    records[-1]["stop_reason"] = stop_reason
    write_round_summary(records[-1])
    return records


def obtain_handoff(
    spec: dict[str, Any], static_round: dict[str, Any] | None,
    provider: dict[str, Any] | None, records: list[dict[str, Any]],
    seen_models: set[str], args: argparse.Namespace, round_dir: Path,
    round_number: int,
) -> tuple[dict[str, Any] | None, str]:
    dataset = spec["dataset"]
    split = spec.get("split", "auto")
    round_dir.mkdir(parents=True, exist_ok=True)
    output = round_dir / "candidate_handoff.json"
    if static_round:
        source_path = resolve_path(args.manifest, static_round["candidates"])
        handoff = read_handoff(source_path, dataset, split, round_number)
        write_json(output, handoff)
        return handoff, "static_manifest"
    if not provider:
        return None, "static_manifest_complete"
    if records and not feedback_exists(records[-1]):
        return None, "evaluation_failure"
    request = make_provider_input(spec, provider, records, seen_models, args, round_number)
    reused = reusable_handoff_on_resume(
        request, spec, seen_models, args, round_dir, round_number
    )
    if reused is not None:
        return reused, "provider"
    return retrieve_with_retries(
        request, spec, provider, seen_models, args, round_dir, round_number
    )


def reusable_handoff_on_resume(
    request: dict[str, Any], spec: dict[str, Any], seen_models: set[str],
    args: argparse.Namespace, round_dir: Path, round_number: int,
) -> dict[str, Any] | None:
    if not args.resume:
        return None
    handoff = reusable_provider_handoff(request, spec, round_dir, round_number)
    if handoff is None:
        return None
    deduplicated, _ = deduplicate_handoff(handoff, seen_models)
    return deduplicated if deduplicated["candidates"] else None


def retrieve_with_retries(
    request: dict[str, Any], spec: dict[str, Any], provider: dict[str, Any],
    seen_models: set[str], args: argparse.Namespace, round_dir: Path,
    round_number: int,
) -> tuple[dict[str, Any] | None, str]:
    rejected: list[dict[str, str]] = []
    accepted: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    duplicate_count = 0
    last_error: ProviderError | None = None
    attempts = getattr(args, "provider_retries", 2) + 1
    requested = int(request["provider_config"].get("max_candidates", 5))
    for attempt in range(1, attempts + 1):
        attempt_request = provider_attempt_request(
            request, attempt, rejected, accepted
        )
        try:
            handoff = call_provider_once(
                provider, args, round_dir, attempt_request, spec, round_number
            )
            last_error = None
        except ProviderError as exc:
            last_error = exc
            rejected.append(provider_error_rejection(exc))
            archive_provider_attempt(round_dir, attempt)
            continue
        archive_provider_attempt(round_dir, attempt)
        if handoff is None:
            rejected.extend(read_provider_rejections(round_dir))
            continue
        template = handoff
        within_round_seen = {
            str(candidate["model"]).casefold() for candidate in accepted
        }
        handoff, duplicates = deduplicate_handoff(
            handoff, seen_models | within_round_seen
        )
        duplicate_count += duplicates
        accepted.extend(handoff["candidates"])
        rejected.extend(handoff.get("retrieval_rejections") or [])
        if len(accepted) >= requested:
            combined = combined_handoff(
                template, accepted[:requested], rejected, request, round_dir, attempt
            )
            write_json(round_dir / "candidate_handoff.json", combined)
            return combined, "provider"
        rejected.append({
            "model": "",
            "reason": "insufficient_unique_candidates",
            "detail": f"Collected {len(accepted)} of {requested} requested candidates.",
        })
    if accepted and template is not None:
        combined = combined_handoff(
            template, accepted, rejected, request, round_dir, attempts
        )
        write_json(round_dir / "candidate_handoff.json", combined)
        return combined, "provider"
    if last_error is not None:
        raise last_error
    reason = "no_new_candidates" if duplicate_count else "no_valid_candidates"
    return None, reason


def provider_attempt_request(
    request: dict[str, Any], attempt: int, rejected: list[dict[str, str]],
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    if attempt == 1:
        return request
    accepted_models = [str(candidate["model"]).casefold() for candidate in accepted]
    retry = {
        "attempt": attempt,
        "base_seen_models": request.get("seen_models") or [],
        "previous_rejections": rejected,
        "instruction": "Return different unseen candidates and obey the allowed-model list.",
    }
    return {
        **request,
        "seen_models": unique_strings([
            *(request.get("seen_models") or []), *accepted_models,
        ]),
        "retrieval_retry": retry,
    }


def combined_handoff(
    template: dict[str, Any], candidates: list[dict[str, Any]],
    rejected: list[dict[str, str]], request: dict[str, Any], round_dir: Path,
    attempt_count: int,
) -> dict[str, Any]:
    input_path = round_dir / "provider_input.json"
    write_json(input_path, request)
    ranked = [{**candidate, "rank": rank} for rank, candidate in enumerate(candidates, 1)]
    return {
        **template,
        "provider_input_checksum": file_checksum(input_path),
        "provider_attempt_count": attempt_count,
        "candidates": ranked,
        "retrieval_rejections": rejected,
        "no_candidates_reason": "",
    }


def call_provider_once(
    provider: dict[str, Any], args: argparse.Namespace, round_dir: Path,
    request: dict[str, Any], spec: dict[str, Any], round_number: int,
) -> dict[str, Any] | None:
    input_path = round_dir / "provider_input.json"
    output_path = round_dir / "candidate_handoff.json"
    write_json(input_path, request)
    output_path.unlink(missing_ok=True)
    run_provider(provider, args, round_dir)
    if not output_path.exists():
        raise ProviderError("Provider did not write candidate_handoff.json.")
    try:
        value = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError(f"Provider wrote invalid JSON: {exc}") from exc
    candidates = value.get("candidates") if isinstance(value, dict) else None
    if not candidates:
        return None
    try:
        validate_handoff(
            value, output_path, spec["dataset"], spec.get("split", "auto"),
            round_number,
        )
    except ValueError as exc:
        raise ProviderError(f"Provider wrote an invalid handoff: {exc}") from exc
    if value.get("provider_input_checksum") != file_checksum(input_path):
        raise ProviderError(
            "Provider response input checksum does not match provider_input.json."
        )
    return value


def archive_provider_attempt(round_dir: Path, attempt: int) -> None:
    for name in (
        "provider_input.json", "candidate_handoff.json",
        "provider_stdout.log", "provider_stderr.log",
    ):
        source = round_dir / name
        if source.exists():
            target = round_dir / f"provider_attempt_{attempt}_{name}"
            target.write_bytes(source.read_bytes())


def provider_error_rejection(error: ProviderError) -> dict[str, str]:
    return {"model": "", "reason": "provider_error", "detail": str(error)}


def read_provider_rejections(round_dir: Path) -> list[dict[str, str]]:
    path = round_dir / "candidate_handoff.json"
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rejections = value.get("retrieval_rejections") if isinstance(value, dict) else []
    return rejections if isinstance(rejections, list) else []


def reusable_provider_handoff(
    request: dict[str, Any], spec: dict[str, Any], round_dir: Path, round_number: int
) -> dict[str, Any] | None:
    input_path = round_dir / "provider_input.json"
    output_path = round_dir / "candidate_handoff.json"
    if not input_path.is_file() or not output_path.is_file():
        return None
    try:
        previous_request = json.loads(input_path.read_text(encoding="utf-8"))
        handoff = json.loads(output_path.read_text(encoding="utf-8"))
        validate_handoff(
            handoff, output_path, spec["dataset"], spec.get("split", "auto"), round_number
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if comparable_provider_request(previous_request) != comparable_provider_request(request):
        return None
    if handoff.get("provider_input_checksum") != file_checksum(input_path):
        return None
    return handoff


def comparable_provider_request(request: dict[str, Any]) -> dict[str, Any]:
    comparable = {
        key: value for key, value in request.items() if key != "retrieval_retry"
    }
    retry = request.get("retrieval_retry") or {}
    if "base_seen_models" in retry:
        comparable["seen_models"] = retry["base_seen_models"]
    return comparable


def make_provider_input(
    spec: dict[str, Any], provider: dict[str, Any], records: list[dict[str, Any]],
    seen_models: set[str], args: argparse.Namespace, round_number: int,
) -> dict[str, Any]:
    feedback = accumulated_feedback(records, round_number) if records else None
    config = resolve_provider_config(provider, args.manifest)
    align_candidate_request(feedback, config)
    context = (
        read_dataset_context(records[-1])
        if records
        else initial_dataset_context(spec, args.manifest)
    )
    return {
        "provider_input_version": 1,
        "dataset": spec["dataset"],
        "requested_split": spec.get("split", "auto"),
        "round": round_number,
        "retrieval_task": retrieval_task(context, feedback),
        "dataset_context": context,
        "feedback": feedback,
        "seen_models": sorted(seen_models),
        "continuation_review": continuation_review_request(
            records, getattr(args, "target_ranked_models", 0), round_number
        ),
        "provider_config": config,
    }


def continuation_review_request(
    records: list[dict[str, Any]], target: int, next_round: int
) -> dict[str, Any]:
    rows = [
        dict(row)
        for record in records
        for row in record.get("candidate_results", [])
    ]
    ranked = rank_models(rows, target) if target > 0 else []
    return {
        "required": should_stop_for_target(records, target),
        "minimum_ranked_models": target,
        "ranked_model_count": comparable_success_count(records),
        "proposed_round": next_round,
        "current_top_models": [ranked_model_summary(row) for row in ranked],
    }


def ranked_model_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": row.get("final_rank"),
        "model": row.get("model"),
        "score": row.get("score"),
        "metric": row.get("metric"),
        "evaluation_protocol": row.get("evaluation_protocol"),
        "round": row.get("round"),
        "labeled_total": row.get("labeled_total"),
    }


def align_candidate_request(
    feedback: dict[str, Any] | None, config: dict[str, Any]
) -> None:
    if not feedback:
        return
    request = feedback.setdefault("candidate_request", {})
    count = int(config.get("max_candidates", 5))
    request["minimum_candidates"] = count
    request["requested_candidates"] = count


def accumulated_feedback(
    records: list[dict[str, Any]], next_round: int
) -> dict[str, Any] | None:
    documents = feedback_documents(records)
    if not documents:
        return None
    latest = documents[-1]
    history = evaluation_history(records)
    best = best_history_result(history)
    feedback = {**latest}
    feedback.update({
        "next_round": next_round,
        "best_model": best.get("model") if best else None,
        "best_score": best.get("score") if best else None,
        "observations": accumulated_observations(documents, history, best),
        "retrieval_constraints": accumulated_constraints(documents, history),
        "evaluation_history": history,
    })
    request = dict(feedback.get("candidate_request") or {})
    request["round"] = next_round
    feedback["candidate_request"] = request
    return feedback


def feedback_documents(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [value for value in (read_feedback(record) for record in records) if value]


def evaluation_history(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "round", "model", "outcome", "status", "score", "metric",
        "evaluation_protocol", "comparable", "non_comparable_reason",
    )
    return [
        {field: row.get(field) for field in fields}
        for record in records for row in record.get("candidate_results", [])
        if row.get("model")
    ]


def best_history_result(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    comparable = [
        item for item in history
        if item.get("outcome") == "success" and item.get("comparable") is not False
    ]
    return max(comparable, key=lambda item: float(item["score"]), default=None)


def accumulated_observations(
    documents: list[dict[str, Any]], history: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> list[str]:
    values = []
    if best:
        values.append(
            f"Best comparable model across all completed rounds is {best['model']} "
            f"with score {float(best['score']):.4f}."
        )
    values.extend(history_observation(item) for item in history)
    for document in documents:
        values.extend(
            item for item in document.get("observations", [])
            if not str(item).startswith("Best measured model was ")
        )
    return unique_strings(values)


def history_observation(item: dict[str, Any]) -> str:
    result = f"Round {item.get('round')} model {item.get('model')}"
    if item.get("outcome") == "success":
        return f"{result} produced comparable score {item.get('score')}."
    reason = item.get("non_comparable_reason") or item.get("status") or item.get("outcome")
    return f"{result} was not rankable: {reason}."


def accumulated_constraints(
    documents: list[dict[str, Any]], history: list[dict[str, Any]]
) -> dict[str, Any]:
    latest = dict((documents[-1].get("retrieval_constraints") or {}))
    exclusions = []
    for document in documents:
        constraints = document.get("retrieval_constraints") or {}
        exclusions.extend(constraints.get("candidate_exclusions") or [])
    exclusions.extend(
        item["model"] for item in history
        if item.get("model") and item.get("outcome") != "success"
    )
    baselines = [
        item["model"] for item in history
        if item.get("model") and item.get("outcome") == "success"
        and item.get("comparable") is not False
    ]
    latest["candidate_exclusions"] = unique_strings(exclusions)
    latest["baseline_models"] = unique_strings(baselines)
    return latest


def unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def initial_dataset_context(spec: dict[str, Any], manifest: Path) -> Any:
    configured = spec.get("evaluation_context")
    if not configured:
        return spec.get("initial_context")
    template = read_context_template(
        resolve_path(manifest, configured), spec["dataset"]
    )
    return template["dataset_context"]


def resolve_provider_config(provider: dict[str, Any], manifest: Path) -> dict[str, Any]:
    config = {key: value for key, value in provider.items() if key != "command"}
    for field in (
        "artifact_output", "node_metadata", "artifact_linker_root", "precomputed_cache_root",
    ):
        if config.get(field):
            config[field] = str(resolve_path(manifest, config[field]).resolve())
    return config


def run_provider(provider: dict[str, Any], args: argparse.Namespace, round_dir: Path) -> None:
    command = provider_command(provider, args)
    command.extend(["--input", str(round_dir / "provider_input.json"), "--output", str(round_dir / "candidate_handoff.json")])
    try:
        result = subprocess.run(command, text=True, capture_output=True)
    except OSError as exc:
        raise ProviderError(f"Provider launch failed: {exc}") from exc
    (round_dir / "provider_stdout.log").write_text(result.stdout or "", encoding="utf-8")
    (round_dir / "provider_stderr.log").write_text(result.stderr or "", encoding="utf-8")
    if result.returncode:
        raise ProviderError(f"Provider exited with {result.returncode}: {tail(result.stderr)}")


def provider_command(provider: dict[str, Any], args: argparse.Namespace) -> list[str]:
    configured = provider.get("command")
    if configured is None:
        return [args.python, str(DEFAULT_PROVIDER)]
    if isinstance(configured, list) and all(isinstance(item, str) for item in configured):
        return list(configured)
    if not isinstance(configured, str) or not configured.strip():
        raise ProviderError("Provider command must be a string or list of strings.")
    parts = shlex.split(configured)
    if len(parts) == 1 and parts[0].endswith(".py"):
        return [args.python, str(resolve_path(args.manifest, parts[0]))]
    return parts


def feedback_exists(record: dict[str, Any]) -> bool:
    path = Path(record.get("feedback_path", ""))
    return bool(path.name) and path.exists()


def read_feedback(record: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(record.get("feedback_path", ""))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_dataset_context(record: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(record.get("dataset_inspection", ""))
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def retrieval_task(context: Any, feedback: Any) -> str:
    if isinstance(feedback, dict):
        task = (feedback.get("candidate_request") or {}).get("task")
        if task:
            return str(task)
    if not isinstance(context, dict):
        return "unknown"
    task = str(context.get("task", "unknown"))
    labels = context.get("label_map") or {}
    if task == "generation" and isinstance(labels, dict) and 1 < len(labels) <= 100:
        return "classification"
    return task


def deduplicate_handoff(handoff: dict[str, Any], seen_models: set[str]) -> tuple[dict[str, Any], int]:
    retained = []
    local = set()
    duplicates = 0
    rejections = list(handoff.get("retrieval_rejections") or [])
    for candidate in handoff.get("candidates", []):
        model = str(candidate.get("model", "")).casefold()
        if model in seen_models:
            duplicates += 1
            rejections.append({
                "model": candidate.get("model", ""),
                "reason": "already_seen_model",
            })
            continue
        if model in local:
            duplicates += 1
            rejections.append({
                "model": candidate.get("model", ""),
                "reason": "duplicate_model_in_handoff",
            })
            continue
        local.add(model)
        retained.append(candidate)
    return {
        **handoff,
        "candidates": retained,
        "retrieval_rejections": rejections,
    }, duplicates


def run_or_resume_round(
    spec: dict[str, Any], handoff: dict[str, Any], source: str,
    args: argparse.Namespace, root: Path, round_dir: Path,
) -> dict[str, Any]:
    provenance = make_round_provenance(spec, handoff, source, args, round_dir)
    provenance_path = round_dir / "round_provenance.json"
    if args.resume and reusable_round(round_dir, provenance):
        record = load_round_record(spec, handoff, round_dir)
        record["reused"] = True
        record["reuse_decision"] = "reused_provenance_equivalent"
        return record
    write_json(provenance_path, provenance)
    record = run_round(spec, handoff, args, root, round_dir)
    record["reuse_decision"] = "rerun_provenance_mismatch_or_missing"
    provenance["output_provenance"] = output_provenance(record)
    write_json(provenance_path, provenance)
    return record


def make_round_provenance(
    spec: dict[str, Any], handoff: dict[str, Any], source: str,
    args: argparse.Namespace, round_dir: Path,
) -> dict[str, Any]:
    inputs = {
        "dataset": spec["dataset"], "requested_split": spec.get("split", "auto"),
        "round": handoff["round"], "stage": args.stage, "runner": args.runner,
        "smoke_limit": args.smoke_limit, "sample_size": args.sample_size,
        "full_limit": args.full_limit, "seed": args.seed,
        "target_ranked_models": getattr(args, "target_ranked_models", 0),
        "provider_retries": getattr(args, "provider_retries", 2),
        "trust_remote_code": args.trust_remote_code,
        "allow_label_scores": args.allow_label_scores, "model_source": handoff,
        "handoff_source": source,
        "provider_input_checksum": handoff.get("provider_input_checksum", ""),
        "evaluation_runner_checksum": file_checksum(BATCH_RUNNER),
        "orchestrator_checksum": file_checksum(Path(__file__)),
        "full_gate_checksum": file_checksum(SCRIPT_DIR / "selection_loop_full_gate.py"),
        "refinement_checksum": file_checksum(SCRIPT_DIR / "refinement_agent_core.py"),
        "evaluation_context": evaluation_context_provenance(spec, args.manifest),
    }
    return {"provenance_version": 1, "input_checksum": json_checksum(inputs), "inputs": inputs, "round_dir": str(round_dir)}


def reusable_round(round_dir: Path, expected: dict[str, Any]) -> bool:
    provenance_path = round_dir / "round_provenance.json"
    summary_path = round_dir / "round_summary.json"
    if not provenance_path.exists() or not summary_path.exists():
        return False
    current = json.loads(provenance_path.read_text(encoding="utf-8"))
    if current.get("input_checksum") != expected.get("input_checksum"):
        return False
    output = current.get("output_provenance") or {}
    for field in ("results_csv", "plans", "smoke_results_csv"):
        item = output.get(field) or {}
        if field == "smoke_results_csv" and not item:
            continue
        path = Path(item.get("path", ""))
        if not path.is_file() or file_checksum(path) != item.get("checksum"):
            return False
    context = output.get("context") or {}
    context_path = Path(context.get("path", ""))
    return not context_path.name or (context_path.is_file() and file_checksum(context_path) == context.get("checksum"))


def output_provenance(record: dict[str, Any]) -> dict[str, Any]:
    output = {
        "results_csv": path_provenance(Path(record["results_csv"])),
        "context": first_existing_provenance(record.get("context_paths", [])),
        "plans": path_provenance(Path(record["plans"])),
        "metric": record.get("metric", ""), "protocol": record.get("protocol", ""),
        "evaluated_split": record.get("evaluated_split", ""),
    }
    if record.get("smoke_results_csv"):
        output["smoke_results_csv"] = path_provenance(Path(record["smoke_results_csv"]))
    return output


def first_existing_provenance(paths: list[str]) -> dict[str, str]:
    for value in paths:
        path = Path(value)
        if path.exists():
            return path_provenance(path)
    return {"path": "", "checksum": ""}


def path_provenance(path: Path) -> dict[str, str]:
    return {"path": str(path), "checksum": file_checksum(path) if path.is_file() else ""}


def run_round(
    spec: dict[str, Any], handoff: dict[str, Any], args: argparse.Namespace,
    root: Path, round_dir: Path,
) -> dict[str, Any]:
    dataset = spec["dataset"]
    if args.stage == "full":
        return run_smoke_then_full(spec, handoff, args, root, round_dir)
    split = spec.get("split", "auto")
    conditions = round_dir / "conditions.csv"
    write_conditions(conditions, handoff)
    context_path = prepare_evaluation_context(spec, handoff, args, round_dir)
    command = round_command(
        args,
        dataset,
        split,
        conditions,
        round_dir / "evaluation",
        context_path=context_path,
    )
    print(f"[selection-loop] {dataset} round {handoff['round']}: {len(handoff['candidates'])} candidates", flush=True)
    completed = subprocess.run(command, text=True)
    return build_round_record(dataset, split, handoff["round"], handoff, round_dir, completed.returncode)


def build_round_record(
    dataset: str, split: str, round_number: int, handoff: dict[str, Any],
    round_dir: Path, returncode: int,
) -> dict[str, Any]:
    run_dir = round_dir / "evaluation" / f"{safe_name(dataset)}_{split}"
    results_path = run_dir / "results.csv"
    results = join_candidate_results(dataset, round_number, handoff, read_csv(results_path))
    return {
        "dataset": dataset, "round": round_number, "requested_split": split,
        "evaluated_split": first_nonempty(results, "evaluated_split"),
        "metric": first_nonempty(results, "metric"),
        "protocol": first_nonempty(results, "evaluation_protocol"),
        "provider_version": handoff.get("provider_version", handoff.get("retrieval_agent", "")),
        "provider_input_checksum": handoff.get("provider_input_checksum", ""),
        "candidate_handoff": str(round_dir / "candidate_handoff.json"),
        "conditions_csv": str(round_dir / "conditions.csv"), "results_csv": str(results_path),
        "feedback_path": str(run_dir / "retrieval_feedback.json"),
        "dataset_inspection": str(run_dir / "dataset_inspection.json"),
        "plans": str(run_dir / "plans.json"),
        "context_paths": [str(run_dir / "codex_context.json"), str(run_dir / "manual_context.json")],
        "returncode": returncode, "candidate_count": len(handoff["candidates"]),
        "candidate_results": results,
        "round_failures": read_json_if_exists(round_dir / "evaluation" / "batch_failures.json") or [],
        "reused": False, "reuse_decision": "", "stop_reason": "",
    }


def load_round_record(spec: dict[str, Any], handoff: dict[str, Any], round_dir: Path) -> dict[str, Any]:
    summary = json.loads((round_dir / "round_summary.json").read_text(encoding="utf-8"))
    record = build_round_record(spec["dataset"], spec.get("split", "auto"), handoff["round"], handoff, round_dir, int(summary.get("returncode", 0)))
    record.update(summary)
    return record


def finalize_round_feedback(record: dict[str, Any]) -> None:
    path = Path(record["feedback_path"])
    if not path.exists():
        return
    feedback = json.loads(path.read_text(encoding="utf-8"))
    feedback.update({
        "split": record["requested_split"], "requested_split": record["requested_split"],
        "evaluated_split": record["evaluated_split"], "next_round": record["round"] + 1,
    })
    request = feedback.get("candidate_request")
    if isinstance(request, dict):
        request.update({
            "split": record["requested_split"], "requested_split": record["requested_split"],
            "evaluated_split": record["evaluated_split"], "round": record["round"] + 1,
        })
    align_feedback_with_comparable_results(feedback, record["candidate_results"])
    write_json(path, feedback)


def align_feedback_with_comparable_results(
    feedback: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    comparable = [
        row for row in rows
        if row.get("outcome") == "success" and row.get("comparable") is not False
    ]
    best = max(comparable, key=lambda row: float(row["score"]), default=None)
    feedback["best_model"] = best.get("model") if best else None
    feedback["best_score"] = float(best["score"]) if best else None
    constraints = feedback.setdefault("retrieval_constraints", {})
    constraints["baseline_models"] = unique_strings(
        [row.get("model") for row in comparable]
    )
    add_failed_candidate_exclusions(feedback, rows)
    feedback["evaluated_models"] = evaluation_history([{"candidate_results": rows}])
    feedback["observations"] = comparable_observations(
        feedback.get("observations") or [], rows, best
    )


def comparable_observations(
    observations: list[Any], rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> list[str]:
    values = [
        item for item in observations
        if not str(item).startswith("Best measured model was ")
    ]
    if best:
        values.insert(
            0,
            f"Best comparable model in this round was {best['model']} "
            f"with score {float(best['score']):.4f}.",
        )
    for row in rows:
        if row.get("outcome") == "non_comparable":
            values.append(
                f"Exclude {row.get('model')}: {row.get('non_comparable_reason')}."
            )
    return unique_strings(values)


def add_failed_candidate_exclusions(
    feedback: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    constraints = feedback.setdefault("retrieval_constraints", {})
    if not isinstance(constraints, dict):
        return
    exclusions = constraints.setdefault("candidate_exclusions", [])
    if not isinstance(exclusions, list):
        return
    for row in rows:
        model = row.get("model")
        if row.get("outcome") != "success" and model and model not in exclusions:
            exclusions.append(model)


def resolve_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent.parent / path


def read_handoff(path: Path, dataset: str, split: str, round_number: int) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_handoff(value, path, dataset, split, round_number)
    return value


def validate_handoff(value: Any, path: Path, dataset: str, split: str, round_number: int) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Candidate handoff must be an object: {path}")
    missing = HANDOFF_FIELDS - set(value)
    if missing:
        raise ValueError(f"Candidate handoff is missing fields {sorted(missing)}: {path}")
    if value["handoff_version"] != HANDOFF_VERSION:
        raise ValueError(f"Unsupported candidate handoff version: {path}")
    if not isinstance(value["mock"], bool):
        raise ValueError(f"Candidate handoff mock must be boolean: {path}")
    require_text(value, "owner", "Candidate handoff", path)
    require_text(value, "retrieval_agent", "Candidate handoff", path)
    for field, expected in {"dataset": dataset, "split": split, "round": round_number}.items():
        if value[field] != expected:
            raise ValueError(f"Candidate handoff {field} mismatch: expected {expected!r}, got {value[field]!r}: {path}")
    candidates = value["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"Candidate handoff has no candidates: {path}")
    validate_candidates(candidates, path)
    if "continuation_review" in value:
        validate_continuation_review(value["continuation_review"], path)
    if "provider_input_checksum" in value:
        require_checksum(value["provider_input_checksum"], "provider_input_checksum", path)


def validate_continuation_review(value: Any, path: Path) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"Continuation review must be an object: {path}")
    required = {"required", "decision", "consensus", "retrieval", "critic", "orchestrator"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"Continuation review is missing fields {sorted(missing)}: {path}")
    if not isinstance(value["required"], bool) or not isinstance(value["consensus"], bool):
        raise ValueError(f"Continuation review flags must be boolean: {path}")
    if value["decision"] not in {"continue", "stop"}:
        raise ValueError(f"Continuation review decision must be continue or stop: {path}")
    for role in ("retrieval", "critic", "orchestrator"):
        if not isinstance(value[role], dict):
            raise ValueError(f"Continuation review {role} result must be an object: {path}")
    validate_improvement_assessment(value["retrieval"], "retrieval", path)
    validate_improvement_assessment(value["critic"], "critic", path)
    orchestrator = value["orchestrator"]
    if orchestrator.get("decision") not in {"continue", "stop"}:
        raise ValueError(f"Continuation review orchestrator decision is invalid: {path}")
    validate_review_confidence(orchestrator, "orchestrator", path)


def validate_improvement_assessment(
    value: dict[str, Any], role: str, path: Path
) -> None:
    if not isinstance(value.get("improvement_probable"), bool):
        raise ValueError(
            f"Continuation review {role} improvement_probable must be boolean: {path}"
        )
    validate_review_confidence(value, role, path)


def validate_review_confidence(
    value: dict[str, Any], role: str, path: Path
) -> None:
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError(
            f"Continuation review {role} confidence must be between 0 and 1: {path}"
        )
    if not isinstance(value.get("reasoning"), str) or not value["reasoning"].strip():
        raise ValueError(f"Continuation review {role} reasoning is required: {path}")


def validate_candidates(candidates: list[Any], path: Path) -> None:
    candidate_ids = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be an object: {path}")
        missing = CANDIDATE_FIELDS - set(candidate)
        if missing:
            raise ValueError(f"Candidate {index} is missing fields {sorted(missing)}: {path}")
        for field in CANDIDATE_TEXT_FIELDS:
            require_text(candidate, field, f"Candidate {index}", path)
        validate_intended_use(candidate, index, path)
        if candidate["candidate_id"] in candidate_ids:
            raise ValueError(f"Duplicate candidate_id {candidate['candidate_id']!r}: {path}")
        candidate_ids.add(candidate["candidate_id"])
        if isinstance(candidate["rank"], bool) or not isinstance(candidate["rank"], int):
            raise ValueError(f"Candidate {index} rank must be an integer: {path}")
        score = candidate["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Candidate {index} score must be numeric: {path}")
        if not math.isfinite(score):
            raise ValueError(f"Candidate {index} score must be finite: {path}")


def validate_intended_use(candidate: dict[str, Any], index: int, path: Path) -> None:
    intended_use = candidate["intended_use"]
    if intended_use not in SUPPORTED_INTENDED_USES:
        supported = ", ".join(sorted(SUPPORTED_INTENDED_USES))
        raise ValueError(f"Candidate {index} intended_use {intended_use!r} is unsupported; expected one of: {supported}: {path}")


def require_text(value: dict[str, Any], field: str, label: str, path: Path) -> None:
    if not isinstance(value[field], str) or not value[field].strip():
        raise ValueError(f"{label} {field} must be non-empty: {path}")


def require_checksum(value: Any, field: str, path: Path) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Candidate handoff {field} must be a SHA-256 hex digest: {path}")


def write_conditions(path: Path, handoff: dict[str, Any]) -> None:
    fields = [
        "candidate_id", "condition", "query_dataset", "rank", "model_name",
        "recommendation_score", "source_method", "reasoning", "intended_use",
        "handoff_owner", "handoff_mock", "retrieval_agent",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in handoff["candidates"]:
            writer.writerow(condition_row(handoff, candidate))


def condition_row(handoff: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"), "condition": candidate.get("condition"),
        "query_dataset": handoff["dataset"], "rank": candidate.get("rank"),
        "model_name": candidate.get("model"), "recommendation_score": candidate.get("score"),
        "source_method": candidate.get("source"), "reasoning": candidate.get("reason"),
        "intended_use": candidate.get("intended_use"), "handoff_owner": handoff.get("owner"),
        "handoff_mock": handoff.get("mock", False), "retrieval_agent": handoff.get("retrieval_agent"),
    }


def round_command(
    args: argparse.Namespace,
    dataset: str,
    split: str,
    conditions: Path,
    output_root: Path,
    stage: str | None = None,
    context_path: Path | None = None,
) -> list[str]:
    runner = "manual" if context_path else args.runner
    command = [
        args.python, str(BATCH_RUNNER), "--conditions-csv", str(conditions),
        "--project-root", str(args.project_root), "--output-root", str(output_root),
        "--dataset", dataset, "--split", split, "--stage", stage or args.stage,
        "--runner", runner, "--python", args.python, "--codex-bin", args.codex_bin,
        "--codex-timeout", str(args.codex_timeout), "--cleanup-cache", "none",
        "--fresh-run-dir", "--continue-on-error", "--smoke-limit", str(args.smoke_limit),
        "--sample-size", str(args.sample_size), "--full-limit", str(args.full_limit), "--seed", str(args.seed),
    ]
    if context_path:
        command.extend(["--context-file", str(context_path)])
    if args.codex_bypass_sandbox:
        command.append("--codex-bypass-sandbox")
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    if args.allow_label_scores:
        command.append("--allow-label-scores")
    if getattr(args, "skip_refinement", False):
        command.append("--skip-refinement")
    return command


def prepare_evaluation_context(
    spec: dict[str, Any],
    handoff: dict[str, Any],
    args: argparse.Namespace,
    round_dir: Path,
    filename: str = "evaluation_context.json",
) -> Path | None:
    configured = spec.get("evaluation_context")
    if not configured:
        return None
    source = resolve_path(args.manifest, configured)
    template = read_context_template(source, spec["dataset"])
    models = [candidate["model"] for candidate in handoff["candidates"]]
    destination = round_dir / filename
    write_json(destination, materialize_context(template, models))
    return destination


def evaluation_context_provenance(
    spec: dict[str, Any], manifest: Path
) -> dict[str, str]:
    configured = spec.get("evaluation_context")
    if not configured:
        return {"path": "", "checksum": ""}
    path = resolve_path(manifest, configured).resolve()
    return {"path": str(path), "checksum": file_checksum(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def join_candidate_results(dataset: str, round_number: int, handoff: dict[str, Any], results: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_model = {row.get("model"): row for row in results}
    return [candidate_result(dataset, round_number, candidate, by_model.get(candidate["model"]), handoff) for candidate in handoff["candidates"]]


def candidate_result(
    dataset: str, round_number: int, candidate: dict[str, Any],
    result: dict[str, str] | None, handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    handoff = handoff or {}
    result = result or {}
    status = result.get("status", "missing")
    score = result.get("score", "")
    labeled = result.get("labeled_total", "")
    return {
        "dataset": dataset, "round": round_number, "candidate_id": candidate.get("candidate_id"),
        "condition": candidate.get("condition"), "model": candidate.get("model"),
        "retrieval_rank": candidate.get("rank"), "retrieval_score": candidate.get("score"),
        "source": candidate.get("source"), "intended_use": candidate.get("intended_use"),
        "reason": candidate.get("reason"), "handoff_owner": handoff.get("owner"),
        "handoff_mock": handoff.get("mock"), "retrieval_agent": handoff.get("retrieval_agent"),
        "provider_version": handoff.get("provider_version", ""),
        "provider_input_checksum": handoff.get("provider_input_checksum", ""),
        "requested_split": handoff.get("split", ""), "evaluated_split": result.get("split", ""),
        "outcome": result_outcome(status, score, labeled), "status": status,
        "evaluation_protocol": result.get("evaluation_protocol", ""), "metric": result.get("metric", ""),
        "score": score, "labeled_total": labeled, "comparable": "", "non_comparable_reason": "",
        "notes": result.get("notes", "results.csv has no row for this model"),
        "output_dir": result.get("output_dir", ""), "reused": False,
    }


def annotate_comparability(rows: list[dict[str, Any]], canonical: dict[str, str] | None) -> dict[str, str] | None:
    for row in rows:
        if row["outcome"] not in {"success", "non_comparable"}:
            row["comparable"] = False
            row["non_comparable_reason"] = "not_a_valid_measured_result"
            continue
        key = comparison_key(row)
        if canonical is None:
            canonical = key
        if key != canonical:
            row["outcome"] = "non_comparable"
            row["comparable"] = False
            row["non_comparable_reason"] = f"expected metric/protocol family {canonical}, got {key}"
        else:
            row["outcome"] = "success"
            row["comparable"] = True
            row["non_comparable_reason"] = ""
    return canonical


def comparison_key(row: dict[str, Any]) -> dict[str, str]:
    protocol = str(row.get("evaluation_protocol", ""))
    family = "label_accuracy" if protocol in ACCURACY_PROTOCOLS else protocol
    return {"metric": str(row.get("metric", "")), "protocol_family": family}


def flatten_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        for row in record["candidate_results"]:
            row["reused"] = record.get("reused", False)
            rows.append(row)
    return rows


def select_winners(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return [select_winner(dataset, candidates) for dataset, candidates in grouped.items()]


def rank_models(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    ranked = []
    for dataset in sorted(grouped):
        candidates = rankable_rows(grouped[dataset])
        for final_rank, row in enumerate(candidates[:limit], 1):
            ranked.append({"final_rank": final_rank, **row})
    return ranked


def rankable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [
        row for row in rows
        if row["outcome"] == "success" and row.get("comparable") is not False
    ]
    if successes:
        return sorted(successes, key=measured_rank_key)
    planned = [row for row in rows if row["outcome"] == "planned"]
    return sorted(planned, key=rank_key)


def measured_rank_key(row: dict[str, Any]) -> tuple[float, int, int, str]:
    retrieval = rank_key(row)[1]
    return (
        -float(row["score"]),
        retrieval,
        int(row.get("round") or 0),
        str(row.get("model", "")),
    )


def select_winner(dataset: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["outcome"] == "success" and row.get("comparable") is not False]
    if successes:
        return max(successes, key=lambda row: float(row["score"]))
    planned = [row for row in rows if row["outcome"] == "planned"]
    if planned:
        return min(planned, key=rank_key)
    failures = [row for row in rows if row["outcome"] in {"failure", "non_comparable"}]
    if failures:
        return min(failures, key=rank_key)
    return candidate_result(dataset, 0, {"model": ""}, None)


def rank_key(row: dict[str, Any]) -> tuple[int, int]:
    try:
        return (0, int(row.get("retrieval_rank")))
    except (TypeError, ValueError):
        return (1, 0)


def best_comparable_score(rows: list[dict[str, Any]]) -> float | None:
    values = [float(row["score"]) for row in rows if row["outcome"] == "success" and row.get("comparable")]
    return max(values) if values else None


def should_stop_for_improvement(previous: float | None, current: float | None, minimum: float) -> bool:
    return minimum > 0 and previous is not None and (current is None or current - previous < minimum)


def should_stop_for_target(records: list[dict[str, Any]], target: int) -> bool:
    if target <= 0:
        return False
    return comparable_success_count(records) >= target


def comparable_success_count(records: list[dict[str, Any]]) -> int:
    return sum(
        row["outcome"] == "success" and row.get("comparable") is not False
        for record in records
        for row in record.get("candidate_results", [])
    )


def should_stop_from_deliberation(
    handoff: dict[str, Any], records: list[dict[str, Any]], target: int
) -> bool:
    review = handoff.get("continuation_review") or {}
    return bool(
        should_stop_for_target(records, target)
        and review.get("required") is True
        and review.get("consensus") is True
        and review.get("decision") == "stop"
    )


def record_stopping_deliberation(
    record: dict[str, Any], handoff: dict[str, Any], round_dir: Path
) -> None:
    review = handoff["continuation_review"]
    path = round_dir / "continuation_review.json"
    write_json(path, review)
    record["stopping_continuation_review"] = review
    record["stopping_continuation_review_path"] = str(path)


def attach_continuation_review(
    record: dict[str, Any], handoff: dict[str, Any], round_dir: Path
) -> None:
    review = handoff.get("continuation_review") or {}
    record["continuation_review"] = review
    if not review:
        return
    path = round_dir / "continuation_review.json"
    write_json(path, review)
    record["continuation_review_path"] = str(path)


def max_optional(left: float | None, right: float | None) -> float | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def stop_record(
    dataset: str,
    split: str,
    round_number: int,
    round_dir: Path,
    reason: str,
    note: str,
    include_candidate_result: bool = True,
) -> dict[str, Any]:
    row = candidate_result(dataset, round_number, {"model": "", "rank": ""}, None, {"split": split})
    row["notes"] = note
    return {
        "dataset": dataset, "round": round_number, "requested_split": split,
        "evaluated_split": "", "metric": "", "protocol": "", "provider_version": "",
        "provider_input_checksum": "", "candidate_handoff": str(round_dir / "candidate_handoff.json"),
        "conditions_csv": str(round_dir / "conditions.csv"), "results_csv": "", "feedback_path": "",
        "dataset_inspection": "", "plans": "", "context_paths": [], "returncode": 1,
        "candidate_count": 0,
        "candidate_results": [row] if include_candidate_result else [],
        "round_failures": [{"stage": reason, "error": note}], "reused": False,
        "reuse_decision": "not_reusable", "stop_reason": reason,
    }


def result_fields() -> list[str]:
    return [
        "dataset", "round", "candidate_id", "condition", "model", "retrieval_rank",
        "retrieval_score", "source", "intended_use", "reason", "handoff_owner",
        "handoff_mock", "retrieval_agent", "provider_version", "provider_input_checksum",
        "requested_split", "evaluated_split", "outcome", "status", "evaluation_protocol",
        "metric", "score", "labeled_total", "comparable", "non_comparable_reason",
        "notes", "output_dir", "reused",
    ]


def ranked_result_fields() -> list[str]:
    return ["final_rank", *result_fields()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_round_summary(record: dict[str, Any]) -> None:
    conditions = record.get("conditions_csv")
    if not conditions:
        return
    summary = {key: value for key, value in record.items() if key != "candidate_results"}
    summary["measured_success_count"] = sum(row["outcome"] == "success" for row in record.get("candidate_results", []))
    summary["candidate_results"] = record.get("candidate_results", [])
    write_json(Path(conditions).parent / "round_summary.json", summary)


def make_summary(
    records: list[dict[str, Any]], rows: list[dict[str, Any]],
    selected: list[dict[str, Any]], target: int = 0,
    ranked_path: Path | None = None,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("dataset"):
            grouped[record["dataset"]].append(record)
    datasets = []
    for dataset, items in grouped.items():
        datasets.append({
            "dataset": dataset, "requested_split": items[0].get("requested_split", ""),
            "evaluated_split": next((item.get("evaluated_split") for item in reversed(items) if item.get("evaluated_split")), ""),
            "canonical_evaluation": next((item.get("canonical_evaluation") for item in reversed(items) if item.get("canonical_evaluation")), {}),
            "rounds": len(items), "stop_reason": items[-1].get("stop_reason", ""),
            "ranked_model_count": len(rank_models(
                [row for row in rows if row.get("dataset") == dataset], target or 1
            )),
            "round_summaries": [round_summary_entry(item) for item in items],
        })
    selected_path = ""
    if records and records[0].get("conditions_csv"):
        selected_path = str(Path(records[0]["conditions_csv"]).parents[2] / "selected_models.csv")
    below_target = datasets_below_target(datasets, target)
    return {
        "rounds": len(records), "candidate_attempts": len(rows),
        "measured_successes": sum(row["outcome"] == "success" for row in rows),
        "non_comparable_results": sum(row["outcome"] == "non_comparable" for row in rows),
        "selected_datasets": len(selected),
        "datasets_planned": sum(row["outcome"] == "planned" for row in selected),
        "datasets_without_winner": sum(row["outcome"] == "failure" for row in selected),
        "selected_models_csv": selected_path,
        "target_ranked_models": target,
        "datasets_below_target": below_target,
        "ranked_models_csv": str(ranked_path) if ranked_path else "",
        "stop_reasons": {item["dataset"]: item["stop_reason"] for item in datasets},
        "datasets": datasets,
    }


def datasets_below_target(
    datasets: list[dict[str, Any]], target: int
) -> list[str]:
    if target <= 0:
        return []
    return [
        item["dataset"] for item in datasets
        if item.get("ranked_model_count", 0) < target
    ]


def round_summary_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": record.get("round"), "provider_version": record.get("provider_version", ""),
        "provider_input_checksum": record.get("provider_input_checksum", ""),
        "requested_split": record.get("requested_split", ""), "evaluated_split": record.get("evaluated_split", ""),
        "metric": record.get("metric", ""), "protocol": record.get("protocol", ""),
        "candidate_count": record.get("candidate_count", 0),
        "measured_success_count": sum(row["outcome"] == "success" for row in record.get("candidate_results", [])),
        "reuse_decision": record.get("reuse_decision", ""), "returncode": record.get("returncode", 0),
        "smoke_results_csv": record.get("smoke_results_csv", ""),
        "smoke_returncode": record.get("smoke_returncode", ""),
        "smoke_measured_success_count": record.get("smoke_measured_success_count", ""),
        "continuation_review": record.get("continuation_review", {}),
        "continuation_review_path": record.get("continuation_review_path", ""),
        "stopping_continuation_review": record.get(
            "stopping_continuation_review", {}
        ),
        "stopping_continuation_review_path": record.get(
            "stopping_continuation_review_path", ""
        ),
        "failures": record.get("round_failures", []),
    }


def first_nonempty(rows: list[dict[str, Any]], field: str) -> str:
    return next((str(row.get(field)) for row in rows if row.get(field)), "")


def read_json_if_exists(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def json_checksum(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tail(value: str, limit: int = 1000) -> str:
    return (value or "")[-limit:]


class ProviderError(RuntimeError):
    """Raised when the configured retrieval provider cannot produce a handoff."""


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared planning, inspection, and auditing helpers for HF evaluation runs."""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import get_dataset_split_names, load_dataset
from huggingface_hub import model_info


DEFAULT_PROJECT_ROOT = Path("/sci/labs/michall/roeizucker/agents_project")
DEFAULT_PYTHON = Path(__file__).resolve().parent / ".venv-artifact-linker" / "bin" / "python"
EVALUATOR = Path(__file__).resolve().parent / "evaluate_hf_pair.py"


QUESTION_CANDIDATES = ("question", "prompt", "query", "instruction", "text")
ANSWER_CANDIDATES = ("answer", "answerKey", "label", "labels", "target", "gold", "correct_answer")
CHOICES_CANDIDATES = ("choices", "options", "answers", "candidates")
IMAGE_CANDIDATES = ("image", "img", "picture", "pixel_values")


@dataclass
class DatasetInspection:
    dataset: str
    subset: str | None
    splits: list[str]
    selected_split: str
    total: int
    columns: list[str]
    features: dict[str, str]
    question_column: str
    answer_column: str
    choices_column: str
    image_column: str
    task: str
    labeled_total_sample: int
    inspected_sample: int
    has_gold: bool
    label_map: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ModelInspection:
    model: str
    model_type: str
    config_model_type: str | None
    architectures: list[str]
    pipeline_tag: str | None
    library_name: str | None
    needs_eager_attention: bool
    needs_sanitized_config: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EvalPlan:
    dataset: DatasetInspection
    model: ModelInspection
    split: str
    task: str
    output_dir: Path
    smoke_output_dir: Path
    args: dict[str, Any]
    smoke_command: list[str]
    full_command: list[str]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset.to_dict(),
            "model": self.model.to_dict(),
            "split": self.split,
            "task": self.task,
            "output_dir": str(self.output_dir),
            "smoke_output_dir": str(self.smoke_output_dir),
            "args": self.args,
            "smoke_command": self.smoke_command,
            "full_command": self.full_command,
            "notes": self.notes,
        }


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip("/"))


def choose_first(columns: list[str], candidates: tuple[str, ...], default: str) -> str:
    for name in candidates:
        if name in columns:
            return name
    return default



def sanitize_model_dir(source_dir: Path, args: Any) -> Path:
    config_path = source_dir / "config.json"
    preprocessor_path = source_dir / "preprocessor_config.json"
    sanitized_config, config_changed = load_sanitized_json(config_path, sanitize_config_value)
    sanitized_preprocessor, preprocessor_changed = load_sanitized_json(preprocessor_path, sanitize_preprocessor_config)
    if not config_changed and not preprocessor_changed:
        return source_dir

    target_dir = Path(args.sanitized_model_dir) if getattr(args, "sanitized_model_dir", "") else default_sanitized_model_dir(args)
    target_dir.mkdir(parents=True, exist_ok=True)
    link_snapshot_files(source_dir, target_dir, {"config.json": config_changed, "preprocessor_config.json": preprocessor_changed})
    if config_changed:
        write_json_file(target_dir / "config.json", sanitized_config)
    if preprocessor_changed:
        write_json_file(target_dir / "preprocessor_config.json", sanitized_preprocessor)
    return target_dir


def load_sanitized_json(path: Path, sanitizer: Any) -> tuple[Any, bool]:
    if not path.exists():
        return None, False
    value = json.loads(path.read_text(encoding="utf-8"))
    return sanitizer(value)


def write_json_file(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sanitize_config_value(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        changed = False
        sanitized = {}
        for key, item in value.items():
            if key == "use_cache" and item is None:
                sanitized[key] = True
                changed = True
            else:
                sanitized_item, item_changed = sanitize_config_value(item)
                sanitized[key] = sanitized_item
                changed = changed or item_changed
        return sanitized, changed
    if isinstance(value, list):
        changed = False
        sanitized_items = []
        for item in value:
            sanitized_item, item_changed = sanitize_config_value(item)
            sanitized_items.append(sanitized_item)
            changed = changed or item_changed
        return sanitized_items, changed
    return value, False


def sanitize_preprocessor_config(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, dict):
        return value, False
    sanitized = dict(value)
    if sanitized.get("image_processor_type") == "Qwen2_5_VLImageProcessor":
        sanitized["image_processor_type"] = "Qwen2VLImageProcessor"
        return sanitized, True
    return sanitized, False


def default_sanitized_model_dir(args: Any) -> Path:
    safe = safe_name(getattr(args, "model", "model"))
    return Path(args.output_dir) / "_sanitized_models" / safe


def link_snapshot_files(source_dir: Path, target_dir: Path, skip_changed: dict[str, bool]) -> None:
    for source in source_dir.iterdir():
        target = target_dir / source.name
        if skip_changed.get(source.name, False):
            if target.exists() or target.is_symlink():
                target.unlink()
            continue
        if target.exists() or target.is_symlink():
            continue
        os.symlink(source.resolve(), target)

def inspect_dataset(
    dataset: str,
    subset: str | None = None,
    split: str = "auto",
    sample_size: int = 20,
    question_column: str = "",
    answer_column: str = "",
    choices_column: str = "",
    image_column: str = "",
) -> DatasetInspection:
    notes: list[str] = []
    try:
        splits = get_dataset_split_names(dataset, subset or None)
    except Exception as exc:
        splits = []
        notes.append(f"Could not list splits before loading: {type(exc).__name__}: {exc}")

    selected_split = select_split(dataset, subset, split, splits, sample_size, answer_column)
    ds = load_dataset(dataset, subset or None, split=selected_split)
    columns = list(ds.column_names)
    features = {name: type(feature).__name__ for name, feature in ds.features.items()}

    q_col = question_column or choose_first(columns, QUESTION_CANDIDATES, "question")
    a_col = answer_column or choose_answer_column(ds, columns, sample_size)
    c_col = choices_column or choose_first(columns, CHOICES_CANDIDATES, "choices")
    i_col = image_column or choose_first(columns, IMAGE_CANDIDATES, "image")

    task = infer_dataset_task(columns, c_col, i_col)
    inspected, labeled = count_labeled_sample(ds, a_col, sample_size)
    if labeled == 0:
        notes.append(f"No gold answers found in the first {inspected} rows of split {selected_split!r}.")
    if split == "auto" and selected_split != "test":
        notes.append(f"Auto-selected split {selected_split!r}.")
    if c_col in columns and task == "multiple_choice":
        notes.append("Detected multiple-choice task from choices/options column.")
    if i_col in columns:
        notes.append("Detected image column; vision-language capable models are required.")
    label_map = infer_label_map(ds, columns, a_col, sample_size)
    prompt_template = infer_prompt_template(columns, q_col, a_col, label_map)
    if label_map:
        notes.append(f"Using label map for raw labels: {label_map}.")
    if prompt_template:
        notes.append("Using inferred prompt template for generation/classification.")

    return DatasetInspection(
        dataset=dataset,
        subset=subset or None,
        splits=splits,
        selected_split=selected_split,
        total=len(ds),
        columns=columns,
        features=features,
        question_column=q_col,
        answer_column=a_col,
        choices_column=c_col,
        image_column=i_col,
        task=task,
        labeled_total_sample=labeled,
        inspected_sample=inspected,
        has_gold=labeled > 0,
        label_map=label_map,
        prompt_template=prompt_template,
        notes=notes,
    )



def infer_label_map(ds: Any, columns: list[str], answer_column: str, sample_size: int) -> dict[str, str]:
    if answer_column == "label" and "label_text" in columns:
        mapping: dict[str, str] = {}
        for idx in range(min(sample_size, len(ds))):
            row = ds[idx]
            if row.get("label") is not None and row.get("label_text") is not None:
                mapping[str(row["label"])] = str(row["label_text"])
        return mapping
    if answer_column in columns:
        for idx in range(min(sample_size, len(ds))):
            value = ds[idx].get(answer_column)
            if isinstance(value, bool):
                return {"False": "no", "True": "yes"}
    return {}


def infer_prompt_template(columns: list[str], question_column: str, answer_column: str, label_map: dict[str, str]) -> str:
    if label_map == {"False": "no", "True": "yes"} and "passage" in columns and question_column in columns:
        return "Passage: {passage}\nQuestion: {question}\nAnswer yes or no:"
    if answer_column == "label" and "label_text" in columns:
        values = ", ".join(sorted(set(label_map.values()))) or "the correct label"
        text_column = "text" if "text" in columns else question_column
        return f"Text: {{{text_column}}}\nAnswer with one of: {values}.\nLabel:"
    return ""

def select_split(
    dataset: str,
    subset: str | None,
    requested: str,
    splits: list[str],
    sample_size: int,
    answer_column: str,
) -> str:
    if requested != "auto":
        return requested
    candidates = [name for name in ("validation", "val", "dev", "test", "train") if name in splits]
    candidates.extend(name for name in splits if name not in candidates)
    if not candidates:
        return "test"
    for candidate in candidates:
        try:
            ds = load_dataset(dataset, subset or None, split=candidate)
        except Exception:
            continue
        a_col = answer_column or choose_answer_column(ds, list(ds.column_names), sample_size)
        _, labeled = count_labeled_sample(ds, a_col, sample_size)
        if labeled:
            return candidate
    return candidates[0]


def choose_answer_column(ds: Any, columns: list[str], sample_size: int) -> str:
    best = ""
    best_count = -1
    for candidate in ANSWER_CANDIDATES:
        if candidate not in columns:
            continue
        _, labeled = count_labeled_sample(ds, candidate, sample_size)
        if labeled > best_count:
            best = candidate
            best_count = labeled
    return best or choose_first(columns, ANSWER_CANDIDATES, "answer")


def infer_dataset_task(columns: list[str], choices_column: str, image_column: str) -> str:
    if choices_column in columns:
        return "multiple_choice"
    if image_column in columns:
        return "generation"
    return "generation"


def count_labeled_sample(ds: Any, answer_column: str, sample_size: int) -> tuple[int, int]:
    if answer_column not in ds.column_names:
        return min(sample_size, len(ds)), 0
    inspected = min(sample_size, len(ds))
    labeled = 0
    for idx in range(inspected):
        value = ds[idx].get(answer_column)
        if value is not None and str(value).strip() != "":
            labeled += 1
    return inspected, labeled


def inspect_model(model: str) -> ModelInspection:
    notes: list[str] = []
    pipeline_tag = None
    library_name = None
    try:
        info = model_info(model)
        pipeline_tag = info.pipeline_tag
        library_name = info.library_name
    except Exception as exc:
        notes.append(f"Could not read Hub model metadata: {type(exc).__name__}: {exc}")

    config = load_config_dict_light(model, notes)
    needs_sanitized = has_null_use_cache(config)
    if needs_sanitized:
        notes.append("Model config has use_cache=null; evaluator will create a sanitized local snapshot.")

    architectures = read_architectures(config)
    config_model_type = read_model_type(config)
    model_type = infer_model_type_from_metadata(model, config_model_type, architectures)
    needs_eager = model_type in {"vlm_chat", "vlm_processor"} or "internvl" in model.lower()
    if needs_eager:
        notes.append("Using eager attention for compatibility.")

    return ModelInspection(
        model=model,
        model_type=model_type,
        config_model_type=config_model_type,
        architectures=architectures,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        needs_eager_attention=needs_eager,
        needs_sanitized_config=needs_sanitized,
        notes=notes,
    )


def has_null_use_cache(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "use_cache" and item is None:
                return True
            if has_null_use_cache(item):
                return True
    if isinstance(value, list):
        return any(has_null_use_cache(item) for item in value)
    return False

def load_config_dict_light(model: str, notes: list[str]) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model, "config.json")
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        notes.append(f"Could not inspect raw config.json: {type(exc).__name__}: {exc}")
        return None


def read_architectures(config: Any) -> list[str]:
    if config is None:
        return []
    if isinstance(config, dict):
        return [str(value) for value in config.get("architectures", []) or []]
    return [str(value) for value in getattr(config, "architectures", []) or []]


def read_model_type(config: Any) -> str | None:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("model_type")
    return getattr(config, "model_type", None)


def infer_model_type_from_metadata(model: str, config_model_type: str | None, architectures: list[str]) -> str:
    lower_model = model.lower()
    lower_arch = [arch.lower() for arch in architectures]
    model_type = (config_model_type or "").lower()
    if "paligemma" in lower_model or model_type == "paligemma":
        return "paligemma"
    if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return "vlm_processor"
    if any("internvl" in arch and "chat" in arch for arch in lower_arch):
        return "vlm_chat"
    if any("conditionalgeneration" in arch for arch in lower_arch) and ("vl" in model_type or "vision" in model_type):
        return "vlm_processor"
    if any("seq2seq" in arch or "t5" in arch for arch in lower_arch) or model_type in {"t5", "mt5", "bart"}:
        return "seq2seq_lm"
    return "causal_lm"


def build_plan(
    dataset: DatasetInspection,
    model: ModelInspection,
    project_root: Path,
    python_exe: Path,
    smoke_limit: int,
    output_root: Path | None = None,
) -> EvalPlan:
    out_root = output_root or project_root / "eval_results"
    dataset_name = safe_name(dataset.dataset)
    model_name = safe_name(model.model)
    split = dataset.selected_split
    output_dir = out_root / model_name / f"{dataset_name}_{split}"
    smoke_output_dir = out_root / "_smoke" / model_name / f"{dataset_name}_{split}"

    args: dict[str, Any] = {
        "dataset": dataset.dataset,
        "model": model.model,
        "split": split,
        "task": dataset.task,
        "model_type": model.model_type,
        "question_column": dataset.question_column,
        "answer_column": dataset.answer_column,
        "choices_column": dataset.choices_column,
        "image_column": dataset.image_column,
        "attn_implementation": "eager" if model.needs_eager_attention else "sdpa",
    }
    if dataset.prompt_template:
        args["prompt_template"] = dataset.prompt_template
    if dataset.label_map:
        args["label_map"] = json.dumps(dataset.label_map)
        args["max_new_tokens"] = 4
    if dataset.subset:
        args["subset"] = dataset.subset
    if dataset.task == "multiple_choice":
        args["scoring"] = "answer_text" if model.model_type == "causal_lm" else "label"
        if args["scoring"] == "answer_text":
            args["normalize_by_length"] = True

    notes = []
    if dataset.task == "multiple_choice" and model.model_type == "causal_lm":
        notes.append("Using answer_text scoring with length normalization to reduce label-token bias.")
    if not dataset.has_gold:
        notes.append("Selected split appears unlabeled; summary accuracy will be null.")
    if dataset.image_column not in dataset.columns and model.model_type in {"paligemma", "vlm_chat", "vlm_processor"}:
        notes.append("Vision-language model on dataset without an image column.")
    if dataset.image_column in dataset.columns and model.model_type not in {"paligemma", "vlm_chat", "vlm_processor"}:
        notes.append("Dataset has images but model appears text-only; evaluator may ignore images or be inappropriate.")

    smoke_command = build_eval_command(python_exe, smoke_output_dir, args, limit=smoke_limit)
    full_command = build_eval_command(python_exe, output_dir, args, limit=0)
    return EvalPlan(dataset, model, split, dataset.task, output_dir, smoke_output_dir, args, smoke_command, full_command, notes)


def build_eval_command(python_exe: Path, output_dir: Path, args: dict[str, Any], limit: int) -> list[str]:
    command = [str(python_exe), str(EVALUATOR)]
    for key, value in args.items():
        if value is None or value == "":
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        command.extend([flag, str(value)])
    if limit:
        command.extend(["--limit", str(limit)])
    command.extend(["--output-dir", str(output_dir)])
    return command


def command_to_shell(command: list[str], hf_home: Path | None = None) -> str:
    import shlex

    prefix = ""
    if hf_home is not None:
        prefix = f"HF_HOME={shlex.quote(str(hf_home))} "
    return prefix + " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], hf_home: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    return subprocess.run(command, env=env, text=True, capture_output=True, timeout=timeout, check=False)


def audit_eval_dir(output_dir: Path, max_rows: int = 200) -> dict[str, Any]:
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "predictions.jsonl"
    audit: dict[str, Any] = {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "status": "missing",
        "issues": [],
        "warnings": [],
    }
    if not summary_path.exists():
        audit["issues"].append("summary.json is missing.")
        return audit
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit["summary"] = summary
    audit["status"] = "ok"
    if summary.get("accuracy") is None:
        audit["warnings"].append("Accuracy is null, usually because no gold labels were available.")
    if summary.get("labeled_total") == 0:
        audit["warnings"].append("labeled_total is 0.")
    if not predictions_path.exists():
        audit["issues"].append("predictions.jsonl is missing.")
        audit["status"] = "bad"
        return audit

    rows = []
    with predictions_path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_rows:
                break
            rows.append(json.loads(line))
    predictions = [str(row.get("prediction", "")) for row in rows]
    blanks = sum(1 for value in predictions if not value.strip())
    counts = Counter(predictions)
    audit["sample_rows"] = len(rows)
    audit["blank_predictions"] = blanks
    audit["prediction_distribution_sample"] = dict(counts.most_common(20))
    if rows and blanks == len(rows):
        audit["issues"].append("All sampled predictions are blank.")
        audit["status"] = "bad"
    if rows:
        top_value, top_count = counts.most_common(1)[0]
        dominance = top_count / len(rows)
        audit["top_prediction"] = top_value
        audit["top_prediction_fraction"] = dominance
        if len(rows) >= 10 and dominance >= 0.95:
            audit["warnings"].append(f"Degenerate prediction distribution: {top_value!r} appears in {dominance:.1%} of sampled rows.")
    if audit["issues"]:
        audit["status"] = "bad"
    elif audit["warnings"]:
        audit["status"] = "warn"
    return audit


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "model",
        "score",
        "total",
        "labeled_total",
        "status",
        "output_dir",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def csv_row_from_audit(plan: EvalPlan, audit: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    summary = audit.get("summary") or {}
    notes = list(plan.notes)
    notes.extend(audit.get("warnings", []))
    notes.extend(audit.get("issues", []))
    score = summary.get("accuracy")
    if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
        score = None
    return {
        "dataset": plan.dataset.dataset,
        "split": plan.split,
        "model": plan.model.model,
        "score": score,
        "total": summary.get("total"),
        "labeled_total": summary.get("labeled_total"),
        "status": audit.get("status"),
        "output_dir": str(output_dir),
        "notes": " | ".join(notes),
    }


def resolve_python(path: str | None) -> Path:
    if path:
        return Path(path)
    if DEFAULT_PYTHON.exists():
        return DEFAULT_PYTHON
    return Path(sys.executable)

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
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from huggingface_hub import model_info

from dialogue_dataset_adapter import adapt_dialogue_dataset, is_dialogue_dataset
from gguf_backend import is_gguf_model_name


DEFAULT_PROJECT_ROOT = Path("/sci/labs/michall/roeizucker/agents_project")
DEFAULT_PYTHON = Path(__file__).resolve().parent / ".venv-artifact-linker" / "bin" / "python"
EVALUATOR = Path(__file__).resolve().parent / "evaluate_hf_pair.py"


QUESTION_CANDIDATES = (
    "question",
    "prompt",
    "query",
    "instruction",
    "text",
    "claim",
    "medical_abstract",
    "Consumer complaint narrative",
    "Complaint Text",
)
ANSWER_CANDIDATES = (
    "answer",
    "answers",
    "answerKey",
    "label",
    "labels",
    "condition_label",
    "target",
    "gold",
    "correct_answer",
    "Product",
)
CHOICES_CANDIDATES = ("choices", "options", "candidates")
IMAGE_CANDIDATES = ("image", "img", "picture", "pixel_values")
CONTEXT_CANDIDATES = ("context", "document", "passage")
DATASET_SOURCE_FALLBACKS = {
    "cfpb/consumer-finance-complaints": {
        "path": "csv",
        "data_files": {"train": "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"},
    }
}
KNOWN_CATEGORICAL_DATASETS = {"cfpb/consumer-finance-complaints"}


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
    context_column: str = ""
    label_map: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""
    choices_columns: list[str] = field(default_factory=list)
    evaluation_method: str = "auto"
    dataset_source: str = ""
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)
    label_threshold: float | None = None
    answer_regex: str = ""
    trust_remote_code: bool = False
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
    classifier_labels: list[str] = field(default_factory=list)
    classifier_problem_type: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EvalPlan:
    dataset: DatasetInspection
    model: ModelInspection
    split: str
    task: str
    protocol: str
    metric: str | None
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
            "evaluation_protocol": self.protocol,
            "metric": self.metric,
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


def choose_first(columns: list[str], candidates: tuple[str, ...], default: str = "") -> str:
    for name in candidates:
        if name in columns:
            return name
    if default and default in columns:
        return default
    return default


def requested_dataset_column(requested: str, columns: list[str]) -> str:
    first = requested.split(".", 1)[0].removesuffix("[]")
    return requested if requested and first in columns else ""


def load_dataset_compatible(dataset: str, subset: str | None = None, **kwargs: Any) -> Any:
    try:
        return load_dataset(dataset, subset, **kwargs)
    except Exception as first_error:
        if isinstance(first_error, TypeError) and "trust_remote_code" in str(first_error):
            kwargs.pop("trust_remote_code", None)
            try:
                return load_dataset(dataset, subset, **kwargs)
            except Exception as retry_error:
                first_error = retry_error
        fallback = DATASET_SOURCE_FALLBACKS.get(dataset.lower())
        if not fallback:
            raise first_error
        fallback_kwargs = dict(kwargs)
        fallback_kwargs.pop("trust_remote_code", None)
        try:
            return load_dataset(**fallback, **fallback_kwargs)
        except Exception as fallback_error:
            raise fallback_error from first_error


def known_string_label_map(
    dataset: str, ds: Any, answer_column: str
) -> dict[str, str]:
    if (
        dataset.lower() not in KNOWN_CATEGORICAL_DATASETS
        or answer_column not in ds.column_names
    ):
        return {}
    values = sorted(
        str(value)
        for value in ds.unique(answer_column)
        if value is not None and str(value).strip()
    )
    return {value: value for value in values}


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
    dataset_source: str = "",
    dataset_kwargs: dict[str, Any] | None = None,
    subset: str | None = None,
    split: str = "auto",
    sample_size: int = 20,
    question_column: str = "",
    context_column: str = "",
    answer_column: str = "",
    choices_column: str = "",
    choices_columns: list[str] | None = None,
    image_column: str = "",
    evaluation_method: str = "auto",
    trust_remote_code: bool = False,
) -> DatasetInspection:
    notes: list[str] = []
    source = dataset_source or dataset
    load_kwargs = dict(dataset_kwargs or {})
    subset = choose_dataset_subset(source, subset, trust_remote_code, notes)
    try:
        splits = get_dataset_split_names(
            source,
            subset or None,
            trust_remote_code=trust_remote_code,
            **load_kwargs,
        )
    except Exception as exc:
        splits = []
        notes.append(f"Could not list splits before loading: {type(exc).__name__}: {exc}")

    selected_split = select_split(
        source,
        subset,
        split,
        splits,
        sample_size,
        answer_column,
        trust_remote_code,
        load_kwargs,
    )
    ds = load_dataset_compatible(
        source,
        subset or None,
        split=selected_split,
        trust_remote_code=trust_remote_code,
        **load_kwargs,
    )
    ds = adapt_dialogue_dataset(dataset, ds)
    if is_dialogue_dataset(dataset):
        notes.append("Flattened dialogue records into labeled next-response examples.")
    columns = list(ds.column_names)
    features = {name: str(feature) for name, feature in ds.features.items()}

    q_col = requested_dataset_column(question_column, columns)
    q_col = q_col or choose_first(columns, QUESTION_CANDIDATES, columns[0] if columns else "question")
    x_col = requested_dataset_column(context_column, columns)
    x_col = x_col or choose_first(columns, CONTEXT_CANDIDATES, "")
    a_col = requested_dataset_column(answer_column, columns)
    a_col = a_col or choose_answer_column(ds, columns, sample_size)
    c_col = requested_dataset_column(choices_column, columns)
    c_col = c_col or choose_first(columns, CHOICES_CANDIDATES, "choices")
    i_col = requested_dataset_column(image_column, columns)
    i_col = i_col or choose_first(columns, IMAGE_CANDIDATES, "image")
    for role, requested, selected in (("question", question_column, q_col), ("answer", answer_column, a_col)):
        if requested and requested != selected:
            notes.append(f"Ignored unavailable {role} column {requested!r}; selected {selected!r}.")

    task = infer_dataset_task(columns, c_col, i_col)
    choices_columns = list(choices_columns or [])
    inspected, labeled = count_labeled_sample(ds, a_col, sample_size)
    if labeled == 0:
        notes.append(f"No gold answers found in the first {inspected} rows of split {selected_split!r}.")
    if split != "auto" and selected_split != split:
        notes.append(
            f"Requested split {split!r} is unavailable or unlabeled; selected labeled split "
            f"{selected_split!r} instead."
        )
    elif split == "auto" and selected_split != "test":
        notes.append(f"Auto-selected split {selected_split!r}.")
    if c_col in columns and task == "multiple_choice":
        notes.append("Detected multiple-choice task from choices/options column.")
    if i_col in columns:
        notes.append("Detected image column; vision-language capable models are required.")
    label_map = infer_label_map(ds, columns, a_col, sample_size)
    label_map = label_map or known_string_label_map(dataset, ds, a_col)
    if label_map and dataset.lower() in KNOWN_CATEGORICAL_DATASETS:
        task = "classification"
    prompt_template = infer_prompt_template(columns, q_col, a_col, label_map)
    if label_map:
        notes.append(f"Using label map for raw labels: {label_map}.")
        notes.append("Resolved label names from dataset metadata; verify whether scoring should preserve all labels or collapse them for this task.")
    elif needs_label_semantics_lookup(ds, a_col, sample_size):
        notes.append(
            "Answer column contains opaque numeric labels. Inspect dataset features, "
            "dataset card metadata, and sample rows to recover label meanings before "
            "choosing label_map, prompt_template, or any binary label collapse."
        )
    if prompt_template:
        notes.append("Using inferred prompt template for generation/classification.")

    return DatasetInspection(
        dataset=dataset,
        dataset_source=source,
        dataset_kwargs=load_kwargs,
        subset=subset or None,
        splits=splits,
        selected_split=selected_split,
        total=len(ds),
        columns=columns,
        features=features,
        question_column=q_col,
        context_column=x_col,
        answer_column=a_col,
        choices_column=c_col,
        image_column=i_col,
        task=task,
        labeled_total_sample=labeled,
        inspected_sample=inspected,
        has_gold=labeled > 0,
        label_map=label_map,
        prompt_template=prompt_template,
        choices_columns=choices_columns,
        evaluation_method=evaluation_method,
        trust_remote_code=trust_remote_code,
        notes=notes,
    )



def apply_known_dataset_contract(dataset: DatasetInspection) -> DatasetInspection:
    if (
        dataset.dataset.lower() != "tuetschek/multi_woz_v22"
    ):
        return dataset
    label_values = (
        "NONE",
        "find_attraction",
        "find_bus",
        "find_hospital",
        "find_hotel",
        "find_police",
        "find_restaurant",
        "find_taxi",
        "find_train",
        "book_hotel",
        "book_restaurant",
        "book_train",
    )
    dataset.task = "multilabel_classification"
    dataset.question_column = "turns.utterance"
    dataset.answer_column = "turns.frames.state.active_intent"
    dataset.label_map = {value: value for value in label_values}
    dataset.evaluation_method = "set_f1"
    dataset.prompt_template = (
        "Identify every active user intent expressed in the dialogue. "
        "Ignore assistant actions and conversational wording.\n"
        "Dialogue:\n{turns.utterance}"
    )
    dataset.has_gold = True
    dataset.labeled_total_sample = max(dataset.labeled_total_sample, 1)
    dataset.notes.append(
        "Applied the validated MultiWOZ active-intent set evaluation contract."
    )
    return dataset


def choose_dataset_subset(
    dataset: str,
    subset: str | None,
    trust_remote_code: bool,
    notes: list[str],
) -> str | None:
    if subset:
        return subset
    try:
        configs = get_dataset_config_names(dataset, trust_remote_code=trust_remote_code)
    except Exception:
        return None
    if "default" in configs:
        notes.append("Using declared default dataset config.")
        return "default"
    useful = [name for name in configs if name and name != "default"]
    if not useful:
        return None
    selected = preferred_dataset_config(dataset, useful)
    notes.append(f"Auto-selected dataset config {selected!r} from available configs: {useful}.")
    return selected


def preferred_dataset_config(dataset: str, configs: list[str]) -> str:
    if dataset.lower() == "lmms-lab/docvqa" and "DocVQA" in configs:
        return "DocVQA"
    for candidate in ("default", "main", "DocVQA"):
        if candidate in configs:
            return candidate
    return configs[0]


def infer_label_map(ds: Any, columns: list[str], answer_column: str, sample_size: int) -> dict[str, str]:
    feature_map = infer_feature_label_map(ds, answer_column)
    if feature_map:
        return feature_map
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


def infer_feature_label_map(ds: Any, answer_column: str) -> dict[str, str]:
    feature_name = answer_column.split(".", 1)[0].removesuffix("[]")
    feature = ds.features.get(feature_name) if feature_name in ds.features else None
    names = getattr(feature, "names", None)
    if not names:
        names = getattr(getattr(feature, "feature", None), "names", None)
    if not names:
        return {}
    return {str(idx): str(name) for idx, name in enumerate(names)}


def needs_label_semantics_lookup(ds: Any, answer_column: str, sample_size: int) -> bool:
    if answer_column not in ds.column_names:
        return False
    for idx in range(min(sample_size, len(ds))):
        value = ds[idx].get(answer_column)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int):
            return True
        if str(value).strip().isdigit():
            return True
    return False


def infer_prompt_template(columns: list[str], question_column: str, answer_column: str, label_map: dict[str, str]) -> str:
    if label_map == {"False": "no", "True": "yes"} and "passage" in columns and question_column in columns:
        return "Passage: {passage}\nQuestion: {question}\nAnswer yes or no:"
    if label_map and question_column in columns:
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
    trust_remote_code: bool = False,
    dataset_kwargs: dict[str, Any] | None = None,
) -> str:
    if requested != "auto" and not splits:
        return requested
    candidates = [name for name in ("validation", "val", "dev", "test", "train") if name in splits]
    candidates.extend(name for name in splits if name not in candidates)
    if not candidates:
        return "test"
    if requested != "auto" and requested in candidates:
        candidates.remove(requested)
        candidates.insert(0, requested)
    for candidate in candidates:
        try:
            ds = load_dataset_compatible(
                dataset,
                subset or None,
                split=candidate,
                trust_remote_code=trust_remote_code,
                **dict(dataset_kwargs or {}),
            )
            ds = adapt_dialogue_dataset(dataset, ds)
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
    first = answer_column.split(".", 1)[0].removesuffix("[]")
    if first not in ds.column_names:
        return min(sample_size, len(ds)), 0
    inspected = min(sample_size, len(ds))
    labeled = sum(has_dataset_value(dataset_value(ds[idx], answer_column)) for idx in range(inspected))
    return inspected, labeled


def has_dataset_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(has_dataset_value(item) for item in value)
    return value is not None and str(value).strip() != ""


def dataset_value(row: Any, path: str) -> Any:
    value = nested_dataset_value(row, path.split("."))
    return None if value is _MISSING_DATASET_VALUE else value


_MISSING_DATASET_VALUE = object()


def nested_dataset_value(value: Any, parts: list[str]) -> Any:
    if not parts:
        return value
    if isinstance(value, list):
        return collect_nested_dataset_values(value, parts)
    if not isinstance(value, dict):
        return _MISSING_DATASET_VALUE
    key = parts[0].removesuffix("[]")
    if key not in value:
        return _MISSING_DATASET_VALUE
    return nested_dataset_value(value[key], parts[1:])


def collect_nested_dataset_values(values: list[Any], parts: list[str]) -> list[Any]:
    collected: list[Any] = []
    for item in values:
        found = nested_dataset_value(item, parts)
        if found is _MISSING_DATASET_VALUE:
            continue
        if isinstance(found, list):
            collected.extend(found)
        else:
            collected.append(found)
    return collected


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
    metadata_blocked = has_auth_or_metadata_blocker(notes)
    needs_sanitized = has_null_use_cache(config)
    if needs_sanitized:
        notes.append("Model config has use_cache=null; evaluator will create a sanitized local snapshot.")

    architectures = read_architectures(config)
    config_model_type = read_model_type(config)
    model_type = infer_model_type_from_metadata(model, config_model_type, architectures, pipeline_tag, library_name)
    if metadata_blocked:
        model_type = "unsupported_model"
    needs_eager = model_type in {"vlm_chat", "vlm_processor"} or "internvl" in model.lower()
    if needs_eager:
        notes.append("Using eager attention for compatibility.")

    return ModelInspection(
        model=model,
        model_type=model_type,
        config_model_type=config_model_type,
        architectures=architectures,
        classifier_labels=read_classifier_labels(config),
        classifier_problem_type=read_classifier_problem_type(config),
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        needs_eager_attention=needs_eager,
        needs_sanitized_config=needs_sanitized,
        notes=notes,
    )


def has_auth_or_metadata_blocker(notes: list[str]) -> bool:
    text = "\n".join(notes).lower()
    return "gated repo" in text or "401" in text or "access to model" in text


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


def read_classifier_labels(config: Any) -> list[str]:
    if not isinstance(config, dict):
        return []
    labels = config.get("id2label") or {}
    if not isinstance(labels, dict):
        return []
    return [str(value) for _, value in sorted(labels.items(), key=label_sort_key)]


def read_classifier_problem_type(config: Any) -> str | None:
    if not isinstance(config, dict):
        return None
    value = config.get("problem_type")
    return str(value) if value else None


def label_sort_key(item: tuple[Any, Any]) -> tuple[bool, int | str]:
    try:
        return (False, int(item[0]))
    except (TypeError, ValueError):
        return (True, str(item[0]))


def infer_model_type_from_metadata(
    model: str,
    config_model_type: str | None,
    architectures: list[str],
    pipeline_tag: str | None = None,
    library_name: str | None = None,
) -> str:
    lower_model = model.lower()
    lower_arch = [arch.lower() for arch in architectures]
    model_type = (config_model_type or "").lower()
    if any("sequenceclassification" in arch for arch in lower_arch):
        return "sequence_classifier"
    if pipeline_tag == "sentence-similarity" or library_name == "sentence-transformers":
        return "sentence_encoder"
    if is_gguf_model_name(lower_model):
        return "gguf"
    if pipeline_tag == "token-classification" or any("tokenclassification" in arch for arch in lower_arch):
        return "token_classifier"
    if pipeline_tag == "fill-mask" or any("maskedlm" in arch for arch in lower_arch):
        return "masked_lm"
    if any("questionanswering" in arch for arch in lower_arch):
        return "extractive_qa"
    if is_zero_shot_image_metadata(model_type, lower_arch, pipeline_tag):
        return "zero_shot_image"
    if "paligemma" in lower_model or model_type == "paligemma":
        return "paligemma"
    if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return "vlm_processor"
    if any("internvl" in arch and "chat" in arch for arch in lower_arch):
        return "vlm_chat"
    if is_standard_vlm_processor(model_type, lower_arch):
        return "vlm_processor"
    if is_custom_unsupported_vlm(lower_model, model_type):
        return "unsupported_model"
    if any("causallm" in arch for arch in lower_arch):
        return "causal_lm"
    if is_unsupported_model_name(lower_model, model_type, pipeline_tag):
        return "unsupported_model"
    if is_image_encoder_metadata(lower_model, model_type, lower_arch, pipeline_tag, library_name):
        return "unsupported_image"
    if is_headless_text_encoder(model_type, lower_arch):
        return "sentence_encoder"
    if "sequence-classification" in lower_model or "zeroshot" in lower_model:
        return "sequence_classifier"
    if any("seq2seq" in arch or "t5" in arch for arch in lower_arch) or model_type in {"t5", "mt5", "bart"}:
        return "seq2seq_lm"
    if any(name in lower_model for name in ("flan-t5", "minicheck-flan", "mt5", "t5-")):
        return "seq2seq_lm"
    if any(name in lower_model for name in ("bert", "roberta", "deberta")):
        return "sequence_classifier"
    return "causal_lm"


def is_zero_shot_image_metadata(
    model_type: str,
    lower_arch: list[str],
    pipeline_tag: str | None,
) -> bool:
    if pipeline_tag == "zero-shot-image-classification":
        return True
    if model_type in {"clip", "siglip", "siglip2"}:
        return True
    return any(
        name.endswith("clipmodel") or name.endswith("siglipmodel")
        for name in lower_arch
    )


def is_headless_text_encoder(model_type: str, lower_arch: list[str]) -> bool:
    encoder_types = {
        "albert", "bert", "deberta", "deberta-v2", "distilbert",
        "electra", "roberta", "xlm-roberta",
    }
    if model_type not in encoder_types:
        return False
    return any(arch.endswith("model") and "for" not in arch for arch in lower_arch)


def is_standard_vlm_processor(model_type: str, lower_arch: list[str]) -> bool:
    standard_types = {"llava", "llava_next", "llava_onevision"}
    if model_type in standard_types:
        return any("conditionalgeneration" in arch for arch in lower_arch)
    return any("conditionalgeneration" in arch for arch in lower_arch) and (
        "vl" in model_type or "vision" in model_type
    )


def is_custom_unsupported_vlm(lower_model: str, model_type: str) -> bool:
    unsupported_names = (
        "kimi-vl",
        "deepseek-vl",
        "ristretto",
        "llava",
        "onevision",
    )
    if any(name in lower_model for name in unsupported_names):
        return True
    return model_type in {"deepseek_vl_v2", "llava_qwen"}


def is_unsupported_model_name(lower_model: str, model_type: str, pipeline_tag: str | None) -> bool:
    if "span-marker" in lower_model or model_type == "span-marker":
        return True
    if "mamba" in lower_model or model_type == "mamba":
        return True
    if "awq" in lower_model or "gptq" in lower_model or "bnb" in lower_model:
        return True
    return False


def is_image_encoder_metadata(
    lower_model: str,
    model_type: str,
    lower_arch: list[str],
    pipeline_tag: str | None,
    library_name: str | None,
) -> bool:
    if pipeline_tag in {"image-classification", "zero-shot-image-classification"}:
        return True
    if library_name in {"timm", "torchgeo"}:
        return True
    if model_type in {"clip", "siglip", "vit", "swin", "convnext", "dinov2", "beit", "eva"}:
        return True
    if any("clip" in arch or "vision" in arch for arch in lower_arch):
        return True
    image_names = ("clip", "siglip", "eva02", "vit_", "vit-", "swin", "convnext", "dinov2", "dfn2b", "llava", "onevision", "vision", "granite-vision")
    return any(name in lower_model for name in image_names)


def build_plan(
    dataset: DatasetInspection,
    model: ModelInspection,
    project_root: Path,
    python_exe: Path,
    smoke_limit: int,
    output_root: Path | None = None,
    full_limit: int = 1000,
    seed: int = 42,
    allow_label_scores: bool = False,
    model_source: str = "",
) -> EvalPlan:
    out_root = output_root or project_root / "eval_results"
    dataset_name = safe_name(dataset.dataset)
    model_name = safe_name(model.model)
    split = dataset.selected_split
    output_dir = out_root / model_name / f"{dataset_name}_{split}"
    smoke_output_dir = out_root / "_smoke" / model_name / f"{dataset_name}_{split}"

    protocol, metric, protocol_note = choose_evaluation_protocol(
        dataset, model, allow_label_scores=allow_label_scores
    )
    args: dict[str, Any] = {
        "dataset": dataset.dataset,
        "model": model.model,
        "split": split,
        "task": dataset.task,
        "model_type": model.model_type,
        "evaluation_protocol": protocol,
        "question_column": dataset.question_column,
        "context_column": dataset.context_column,
        "answer_column": dataset.answer_column,
        "choices_column": dataset.choices_column,
        "choices_columns": json.dumps(dataset.choices_columns) if dataset.choices_columns else "",
        "image_column": dataset.image_column,
        "evaluation_method": metric or "auto",
        "attn_implementation": "eager" if model.needs_eager_attention else "sdpa",
        "seed": seed,
    }
    if model_source:
        args["model_source"] = model_source
    if dataset.dataset_source and dataset.dataset_source != dataset.dataset:
        args["dataset_source"] = dataset.dataset_source
    if dataset.dataset_kwargs:
        args["dataset_kwargs"] = json.dumps(dataset.dataset_kwargs)
    if dataset.label_threshold is not None:
        args["label_threshold"] = dataset.label_threshold
    if dataset.answer_regex:
        args["answer_regex"] = dataset.answer_regex
    if allow_label_scores:
        args["allow_label_scores"] = True
    if should_disable_thinking(model, protocol):
        args["disable_thinking"] = True
    if dataset.trust_remote_code:
        args["trust_remote_code"] = True
    if dataset.prompt_template:
        args["prompt_template"] = dataset.prompt_template
    if dataset.label_map:
        args["label_map"] = json.dumps(dataset.label_map)
        if protocol == "label_logprob_accuracy":
            args["normalize_by_length"] = True
        elif protocol == "label_generation_accuracy":
            args["max_new_tokens"] = 4
    if protocol == "tagged_label_generation_accuracy":
        args["max_new_tokens"] = 32
    elif protocol == "tagged_multiple_choice_accuracy":
        args["max_new_tokens"] = 16
    elif protocol == "tagged_set_generation":
        args["max_new_tokens"] = min(256, max(64, 8 * len(dataset.label_map)))
    elif protocol == "tagged_answer_exact_match":
        args["max_new_tokens"] = tagged_answer_token_budget(dataset.task)
    if dataset.subset:
        args["subset"] = dataset.subset
    if dataset.task == "multiple_choice":
        direct_scoring = protocol == "multiple_choice_accuracy"
        args["scoring"] = (
            "answer_text" if direct_scoring and model.model_type == "causal_lm" else "label"
        )
        if args["scoring"] == "answer_text":
            args["normalize_by_length"] = True

    notes = [protocol_note]
    if allow_label_scores:
        notes.append("Direct label-score evaluation was explicitly allowed by the caller.")
    if args.get("disable_thinking"):
        notes.append(
            "Disabled model thinking through the tokenizer chat template for constrained tagged output."
        )
    if dataset.image_column not in dataset.columns and model.model_type in {"paligemma", "vlm_chat", "vlm_processor", "zero_shot_image", "unsupported_image"}:
        args["model_type"] = "unsupported_model"
        notes.append("Vision/image model on dataset without an image column; it will be reported as unsupported for this pairing.")
    if (
        dataset.task == "multiple_choice"
        and protocol == "multiple_choice_accuracy"
        and model.model_type == "causal_lm"
    ):
        notes.append("Using answer_text scoring with length normalization to reduce label-token bias.")
    if not dataset.has_gold:
        notes.append("Selected split appears unlabeled; no measurement will be reported.")
    if model.model_type == "unsupported_model":
        notes.append("Checkpoint/task format is not directly supported by the generic evaluator; it will be reported as unsupported.")
    if dataset.image_column in dataset.columns and model.model_type == "unsupported_image":
        notes.append("Image-classification checkpoint is not directly supported by the generic evaluator; it will be reported as unsupported.")
    elif dataset.image_column in dataset.columns and model.model_type not in {"paligemma", "vlm_chat", "vlm_processor", "zero_shot_image", "unsupported_model"}:
        notes.append("Dataset has images but model appears text-only; evaluator may ignore images or be inappropriate.")

    smoke_command = build_eval_command(python_exe, smoke_output_dir, args, limit=smoke_limit)
    full_command = build_eval_command(python_exe, output_dir, args, limit=full_limit)
    if full_limit:
        notes.append(
            f"Full evaluation uses at most {full_limit} randomly selected examples "
            f"with seed {seed}."
        )
    return EvalPlan(
        dataset,
        model,
        split,
        dataset.task,
        protocol,
        metric,
        output_dir,
        smoke_output_dir,
        args,
        smoke_command,
        full_command,
        notes,
    )


def tagged_answer_token_budget(task: str) -> int:
    if task == "numeric_qa":
        return 512
    if task == "summarization":
        return 256
    return 128


def should_disable_thinking(model: ModelInspection, protocol: str) -> bool:
    if not protocol.startswith("tagged_"):
        return False
    model_family = str(model.config_model_type or "").lower()
    return model_family == "qwen3" or "qwen3" in model.model.lower()


def choose_evaluation_protocol(
    dataset: DatasetInspection,
    model: ModelInspection,
    allow_label_scores: bool = False,
) -> tuple[str, str | None, str]:
    if not dataset.has_gold:
        return "unsupported", None, "Unsupported evaluation: the selected split has no detected gold labels."
    if model.model_type in {"unsupported_image", "unsupported_model"}:
        return "unsupported", None, "Unsupported evaluation: no direct adapter exists for this model format."
    vision_models = {"paligemma", "vlm_chat", "vlm_processor", "zero_shot_image"}
    has_image = dataset.image_column in dataset.columns
    if model.model_type in vision_models and not has_image:
        return "unsupported", None, "Unsupported evaluation: a vision-language model was paired with a dataset without images."
    if has_image and model.model_type not in vision_models:
        return "unsupported", None, "Unsupported evaluation: an image dataset was paired with a model that cannot consume images."

    metric = selected_evaluation_method(dataset)
    if not method_matches_task(dataset.task, metric):
        return (
            "unsupported",
            None,
            f"Unsupported evaluation: method {metric!r} is incompatible with task {dataset.task!r}.",
        )

    if dataset.task == "token_classification":
        if model.model_type == "token_classifier":
            return (
                "token_entity_set_f1",
                "set_f1",
                "Protocol: reconstruct typed entity spans and score example-level entity-set F1.",
            )
        return (
            "unsupported",
            None,
            "Unsupported evaluation: token classification requires a token-classification model.",
        )

    if model.model_type == "extractive_qa":
        if dataset.task != "qa":
            return "unsupported", None, "Unsupported evaluation: extractive QA models require a QA dataset."
        context_root = dataset.context_column.split(".", 1)[0].removesuffix("[]")
        if not dataset.context_column or context_root not in dataset.columns:
            return (
                "unsupported",
                None,
                "Unsupported evaluation: extractive QA requires a question and context column.",
            )
        return "extractive_qa", "qa_f1", "Protocol: extractive span QA scored with qa_f1."

    if model.model_type == "zero_shot_image":
        if not dataset.label_map:
            return "unsupported", None, "Unsupported evaluation: zero-shot image classification requires a finite label map."
        return "zero_shot_image_accuracy", metric, f"Protocol: zero-shot image/text similarity scored with {metric}."

    if dataset.task == "multiple_choice":
        if model.model_type == "masked_lm":
            return "masked_choice_likelihood", "accuracy", "Protocol: masked-language-model pseudo-likelihood over answer choices."
        if model.model_type in {"sentence_encoder", "token_classifier"}:
            return "unsupported", None, f"Unsupported evaluation: {model.model_type} cannot score arbitrary answer choices safely."
        if model.model_type == "sequence_classifier":
            return "unsupported", None, "Unsupported evaluation: sequence classifiers cannot score arbitrary answer choices safely."
        if allow_label_scores:
            return "multiple_choice_accuracy", "accuracy", "Protocol: answer-choice likelihood scoring with accuracy."
        return (
            "tagged_multiple_choice_accuracy",
            "accuracy",
            "Protocol: generate one choice label inside <answer>...</answer> and score accuracy.",
        )

    if dataset.task == "multilabel_classification":
        if model.model_type == "sequence_classifier":
            if (
                model.classifier_problem_type == "multi_label_classification"
                and labels_semantically_align(dataset.label_map, model.classifier_labels)
            ):
                return "multilabel_classifier_set_f1", "set_f1", "Protocol: direct multilabel classifier outputs with example-level set F1."
            if has_entailment_label(model.classifier_labels):
                return "zero_shot_nli_multilabel", "set_f1", "Protocol: independent zero-shot NLI label decisions scored with set F1."
            return "unsupported", None, "Unsupported evaluation: the classifier is not declared multilabel or its labels do not match the dataset labels."
        return "tagged_set_generation", "set_f1", "Protocol: generate a JSON label set inside <answer>...</answer> and score set F1."

    if not dataset.label_map:
        if model.model_type in {"sentence_encoder", "masked_lm"}:
            return (
                "unsupported",
                None,
                f"Unsupported evaluation: {model.model_type} requires a finite label map.",
            )
        if model.model_type == "token_classifier":
            if dataset.task == "token_classification":
                return "token_entity_set_f1", "set_f1", "Protocol: reconstruct typed entity spans and score example-level entity-set F1."
            return "unsupported", None, "Unsupported evaluation: token classifiers require a token-classification dataset."
        if model.model_type == "sequence_classifier":
            return (
                "unsupported",
                None,
                "Unsupported evaluation: a sequence classifier cannot generate free-text answers.",
            )
        return (
            "tagged_answer_exact_match",
            metric,
            f"Protocol: tagged free-text generation scored with {metric}.",
        )

    if model.model_type == "sentence_encoder":
        return "embedding_label_similarity", metric, f"Protocol: normalized sentence/label embedding similarity scored with {metric}."
    if model.model_type == "masked_lm":
        return "masked_label_likelihood", metric, f"Protocol: masked-language-model pseudo-likelihood over labels scored with {metric}."
    if model.model_type == "token_classifier":
        if dataset.task == "token_classification":
            return "token_entity_set_f1", "set_f1", "Protocol: reconstruct typed entity spans and score example-level entity-set F1."
        return "unsupported", None, "Unsupported evaluation: token classifiers require a token-classification dataset."
    if model.model_type == "sequence_classifier":
        if has_entailment_label(model.classifier_labels):
            return "zero_shot_nli_accuracy", metric, f"Protocol: zero-shot NLI scored with {metric}."
        if labels_semantically_align(dataset.label_map, model.classifier_labels):
            return "classifier_label_accuracy", metric, f"Protocol: direct classifier labels scored with {metric}."
        return (
            "unsupported",
            None,
            "Unsupported evaluation: classifier output labels do not match the dataset labels and cannot be remapped by index.",
        )
    return (
        "tagged_label_generation_accuracy",
        metric,
        f"Protocol: generate one dataset label inside <answer>...</answer> and score with {metric}.",
    )


def selected_evaluation_method(dataset: DatasetInspection) -> str:
    if dataset.evaluation_method not in {"", "auto"}:
        return dataset.evaluation_method
    defaults = {
        "classification": "accuracy",
        "image_classification": "accuracy",
        "multiple_choice": "accuracy",
        "multilabel_classification": "set_f1",
        "qa": "qa_f1",
        "numeric_qa": "numeric_match",
        "summarization": "rouge_l",
        "token_classification": "set_f1",
        "relation_extraction": "set_f1",
        "generation": "accuracy" if dataset.label_map else "exact_match",
    }
    return defaults.get(dataset.task, "exact_match")


def method_matches_task(task: str, method: str) -> bool:
    allowed = {
        "classification": {"accuracy", "macro_f1"},
        "image_classification": {"accuracy", "macro_f1"},
        "multiple_choice": {"accuracy"},
        "multilabel_classification": {"set_f1"},
        "qa": {"exact_match", "qa_f1"},
        "numeric_qa": {"exact_match", "numeric_match"},
        "summarization": {"rouge_l"},
        "token_classification": {"set_f1"},
        "relation_extraction": {"set_f1"},
        "generation": {"accuracy", "macro_f1", "exact_match", "qa_f1", "numeric_match", "rouge_l", "set_f1"},
    }
    return method in allowed.get(task, set())


def has_entailment_label(labels: list[str]) -> bool:
    return any("entail" in normalize_label(label) and "not entail" not in normalize_label(label) for label in labels)


def labels_semantically_align(label_map: dict[str, str], classifier_labels: list[str]) -> bool:
    dataset_labels = {normalize_label(value) for key, value in label_map.items() if usable_label(key, value)}
    model_labels = {normalize_label(value) for value in classifier_labels if not generic_label(value)}
    return bool(dataset_labels) and dataset_labels.issubset(model_labels)


def usable_label(key: Any, value: Any) -> bool:
    text = str(value).strip()
    return str(key).strip() != "-1" and bool(text) and not text.lower().startswith("missing")


def generic_label(value: Any) -> bool:
    return bool(re.fullmatch(r"label[_ -]?\d+|\d+", str(value).strip(), flags=re.IGNORECASE))


def normalize_label(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


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
    return run_streaming_command(command, env, timeout)


def run_streaming_command(
    command: list[str], env: dict[str, str], timeout: int | None
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        command, env=env, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=1,
    )
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    threads = [
        threading.Thread(target=collect_command_stream, args=(process.stdout, stdout_lines), daemon=True),
        threading.Thread(target=collect_command_stream, args=(process.stderr, stderr_lines), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        returncode = 124
        message = f"Timed out after {timeout} seconds.\n"
        stderr_lines.append(message)
        print(message, end="", flush=True)
    for thread in threads:
        thread.join()
    return subprocess.CompletedProcess(
        command, returncode, "".join(stdout_lines), "".join(stderr_lines)
    )


def collect_command_stream(stream: Any, lines: list[str]) -> None:
    if stream is None:
        return
    for line in stream:
        lines.append(line)
        print(line, end="", flush=True)


def warn_for_out_of_vocab_predictions(audit: dict[str, Any], summary: dict[str, Any], predictions: list[str]) -> None:
    label_values = set(summary.get("label_values") or [])
    if not label_values or not predictions:
        return
    outside = sorted(set(value for value in predictions if value and value not in label_values))
    if outside:
        audit["warnings"].append(f"Predictions outside label map values: {outside[:10]}")


def write_unsupported_result(plan: EvalPlan, output_dir: Path) -> dict[str, Any]:
    """Record an unsupported pair without loading model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reason = next((note for note in plan.notes if note.startswith("Unsupported evaluation:")), plan.notes[0])
    summary = {
        "model": plan.model.model,
        "dataset": plan.dataset.dataset,
        "split": plan.split,
        "task": plan.task,
        "evaluation_protocol": plan.protocol,
        "metric": plan.metric,
        "total": 0,
        "labeled_total": 0,
        "correct": 0,
        "score": None,
        "accuracy": None,
        "unsupported_reason": reason,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    return audit_eval_dir(output_dir)


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
    if summary.get("unsupported_reason"):
        audit["issues"].append(str(summary["unsupported_reason"]))
        audit["status"] = "unsupported"
        return audit
    score = summary.get("score") if "score" in summary else summary.get("accuracy")
    if score is None or not summary.get("labeled_total"):
        audit["issues"].append("No labeled score was produced.")
        audit["status"] = "unmeasured"
        return audit
    tag_parse_failures = int(summary.get("tag_parse_failures") or 0)
    if tag_parse_failures:
        audit["warnings"].append(
            f"Tagged-answer parsing failed for {tag_parse_failures} examples."
        )
    if not predictions_path.exists():
        audit["issues"].append("predictions.jsonl is missing.")
        audit["status"] = "failed"
        return audit

    counts: Counter[str] = Counter()
    blanks = 0
    total_rows = 0
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prediction = str(row.get("prediction", ""))
            counts[prediction] += 1
            blanks += int(not prediction.strip())
            total_rows += 1
    warn_for_out_of_vocab_predictions(audit, summary, list(counts))
    audit["prediction_rows"] = total_rows
    audit["sample_rows"] = min(total_rows, max_rows)
    audit["blank_predictions"] = blanks
    audit["prediction_distribution"] = dict(counts.most_common(20))
    if not total_rows:
        audit["issues"].append("predictions.jsonl is empty despite a labeled score.")
        audit["status"] = "failed"
    elif blanks == total_rows:
        audit["issues"].append("All predictions are blank.")
        audit["status"] = "failed"
    if total_rows:
        top_value, top_count = counts.most_common(1)[0]
        dominance = top_count / total_rows
        audit["top_prediction"] = top_value
        audit["top_prediction_fraction"] = dominance
        if total_rows >= 10 and dominance >= 0.95:
            audit["warnings"].append(f"Degenerate prediction distribution: {top_value!r} appears in {dominance:.1%} of predictions.")
    if audit["issues"]:
        audit["status"] = "failed"
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
        "evaluation_protocol",
        "metric",
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
    score = summary.get("score") if "score" in summary else summary.get("accuracy")
    if isinstance(score, float) and (math.isnan(score) or math.isinf(score)):
        score = None
    return {
        "dataset": plan.dataset.dataset,
        "split": plan.split,
        "model": plan.model.model,
        "evaluation_protocol": summary.get("evaluation_protocol", plan.protocol),
        "metric": summary.get("metric", plan.metric),
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

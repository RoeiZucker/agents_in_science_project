#!/usr/bin/env python3
"""Evaluate a Hugging Face model/dataset pair with lightweight task adapters.

Examples:
  python evaluate_hf_pair.py \
    --dataset xai-org/RealworldQA \
    --model google/paligemma-3b-ft-nlvr2-448 \
    --split test \
    --output-dir eval_results/paligemma_realworldqa_test

  python evaluate_hf_pair.py \
    --dataset tau/commonsense_qa \
    --model HuggingFaceTB/SmolLM3-3B \
    --split validation \
    --task multiple_choice \
    --output-dir eval_results/smollm3_commonsenseqa_validation_generic

  python evaluate_hf_pair.py \
    --dataset TimSchopf/medical_abstracts \
    --model MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33 \
    --split test \
    --task generation \
    --question-column medical_abstract \
    --answer-column condition_label \
    --label-map '{"1":"neoplasms","2":"digestive system diseases","3":"nervous system diseases","4":"cardiovascular diseases","5":"general pathological conditions"}' \
    --output-dir eval_results/medical_abstracts_zeroshot_smoke \
    --limit 3

  # Short-answer QA with tagged output and token-overlap F1.
  python evaluate_hf_pair.py \
    --dataset rajpurkar/squad \
    --model Qwen/Qwen3-8B \
    --split validation \
    --task qa \
    --question-column question \
    --answer-column answers.text \
    --evaluation-method qa_f1 \
    --disable-thinking \
    --output-dir eval_results/qwen3_squad_tagged \
    --limit 3

  # Extractive QA selects an answer span from a supplied context field.
  python evaluate_hf_pair.py \
    --dataset rajpurkar/squad \
    --model deepset/roberta-base-squad2 \
    --split validation --task qa \
    --model-type extractive_qa --evaluation-protocol extractive_qa \
    --question-column question --context-column context \
    --answer-column answers.text --evaluation-method qa_f1 \
    --output-dir eval_results/roberta_squad_extractive \
    --limit 3

  # Numeric QA treats values such as 42 and 42.0 as equivalent.
  python evaluate_hf_pair.py \
    --dataset MU-NLPC/Calc-asdiv_a \
    --model Qwen/Qwen3-8B \
    --split train \
    --task numeric_qa \
    --question-column question \
    --answer-column result_float \
    --evaluation-method numeric_match \
    --output-dir eval_results/qwen3_asdiv_numeric \
    --limit 3

  # Summarization and multilabel generation use deterministic task metrics.
  python evaluate_hf_pair.py \
    --dataset ccdv/pubmed-summarization \
    --model Qwen/Qwen3-8B \
    --task summarization \
    --question-column article \
    --answer-column abstract \
    --evaluation-method rouge_l \
    --output-dir eval_results/qwen3_pubmed_summary \
    --limit 3

  python evaluate_hf_pair.py \
    --dataset google-research-datasets/go_emotions \
    --subset simplified \
    --model Qwen/Qwen3-8B \
    --task multilabel_classification \
    --question-column text \
    --answer-column labels \
    --label-map-file config/go_emotions_label_map.json \
    --evaluation-method set_f1 \
    --output-dir eval_results/qwen3_go_emotions \
    --limit 3
  # Zero-shot classification with a sentence-embedding checkpoint.
  python evaluate_hf_pair.py --dataset SetFit/bbc-news \
    --model sentence-transformers/sentence-t5-base --split test \
    --task classification --model-type sentence_encoder \
    --evaluation-protocol embedding_label_similarity --question-column text \
    --answer-column label --label-map '{"0":"business","1":"entertainment","2":"politics","3":"sport","4":"tech"}' \
    --output-dir eval_results/sentence_t5_bbc

  # Finite-label classification with a masked-language model.
  python evaluate_hf_pair.py --dataset ade-benchmark-corpus/ade_corpus_v2 \
    --subset Ade_corpus_v2_classification \
    --model PlanTL-GOB-ES/roberta-base-biomedical-clinical-es \
    --task classification --model-type masked_lm \
    --evaluation-protocol masked_label_likelihood --question-column text \
    --answer-column label --label-map '{"0":"not related","1":"related"}' \
    --output-dir eval_results/masked_lm_ade

  # Multiple choice with masked-language-model pseudo-likelihood.
  python evaluate_hf_pair.py --dataset allenai/quartz \
    --model albert/albert-xxlarge-v1 --split validation \
    --task multiple_choice --model-type masked_lm \
    --evaluation-protocol masked_choice_likelihood --question-column question \
    --answer-column answerKey --choices-column choices \
    --prompt-template 'Context: {para}\nQuestion: {question}\nAnswer: {label}' \
    --output-dir eval_results/albert_quartz

  # BigBio entity-set evaluation with a token classifier.
  python evaluate_hf_pair.py --dataset bigbio/bc5cdr \
    --model OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M \
    --task token_classification --model-type token_classifier \
    --evaluation-protocol token_entity_set_f1 --question-column passages \
    --answer-column entities --evaluation-method set_f1 \
    --output-dir eval_results/openmed_bc5cdr

  # Text-only GGUF generation through llama.cpp.
  python evaluate_hf_pair.py --dataset ImperialCollegeLondon/health_fact \
    --model bartowski/Qwen2.5-7B-Instruct-GGUF \
    --model-type gguf --gguf-file Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --task classification --split validation \
    --question-column claim --answer-column label \
    --label-map '{"0":"false","1":"mixture","2":"true","3":"unproven"}' \
    --evaluation-protocol tagged_label_generation_accuracy \
    --evaluation-method accuracy \
    --output-dir eval_results/qwen_gguf_health_fact --limit 3

  # Zero-shot image classification with a CLIP/SigLIP checkpoint.
  python evaluate_hf_pair.py --dataset blanchon/EuroSAT_RGB \
    --model google/siglip-so400m-patch14-384 --split test \
    --task image_classification --model-type zero_shot_image \
    --evaluation-protocol zero_shot_image_accuracy \
    --image-column image --answer-column label \
    --label-map '{"0":"annual crop","1":"forest"}' \
    --evaluation-method accuracy \
    --output-dir eval_results/siglip_eurosat

  # Dialogue datasets are flattened to next-turn examples before sampling.
  python evaluate_hf_pair.py --dataset google-research-datasets/taskmaster1 \
    --subset one_person_dialogs --split test \
    --model Qwen/Qwen3-8B --task generation \
    --question-column _dialogue_prompt \
    --answer-column _dialogue_response --evaluation-method rouge_l \
    --output-dir eval_results/qwen3_taskmaster --limit 100

"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import math
import os
import random
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PaliGemmaForConditionalGeneration,
    VoxtralForConditionalGeneration,
)
from transformers.modeling_utils import PreTrainedModel

from dialogue_dataset_adapter import adapt_dialogue_dataset
from eval_agent_core import load_dataset_compatible
from gguf_backend import GgufBackend, is_gguf_model_name, resolve_gguf_path
from tagged_answer_parser import extract_tagged_answer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UNSUPPORTED_MODEL_TYPES = {"unsupported_image", "unsupported_model"}
TAGGED_PROTOCOLS = {
    "tagged_answer_exact_match",
    "tagged_label_generation_accuracy",
    "tagged_multiple_choice_accuracy",
    "tagged_set_generation",
}

FORCE_SANITIZE_MODELS = {
    "internlm/internlm2-7b",
    "MagicXin/Med3DVLM-Qwen-2.5-7B",
    "lyogavin/Anima-7B-100K",
    "MAGAer13/mplug-owl2-llama2-7b",
    "X-iZhang/libra-v1.0-7b",
    "zongzhuofan/llama3-mova-8b",
}

MODEL_CONFIG_OVERRIDES = {
    "lyogavin/Anima-7B-100K": "llama",
    "MAGAer13/mplug-owl2-llama2-7b": "llama",
    "X-iZhang/libra-v1.0-7b": "llama",
    "zongzhuofan/llama3-mova-8b": "llama",
}

MODEL_SOURCE_ALIASES = {
    "ais ingapore/SEA-LION-v1-7B-IT": "aisingapore/SEA-LION-v1-7B-IT",
    "ais singapore/SEA-LION-v1-7B-IT": "aisingapore/SEA-LION-v1-7B-IT",
    "mlc-ai/Hermes-2-Pro-Mistral-7B-q4f16_1-MLC": "NousResearch/Hermes-2-Pro-Mistral-7B",
    "mlc-ai/DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "onnx-community/Llama-3.2-1B-Instruct-q4f16": "meta-llama/Llama-3.2-1B-Instruct",
    "protectai/MoritzLaurer-roberta-base-zeroshot-v2.0-c-onnx": "MoritzLaurer/roberta-base-zeroshot-v2.0-c",
    "mlx-community/gemma-3-4b-pt-4bit": "google/gemma-3-4b-pt",
    "mlx-community/medgemma-4b-it-4bit": "google/medgemma-4b-it",
    "mlx-community/gemma-3n-E4B-it-lm-bf16": "google/gemma-3n-E4B-it",
    "mlx-community/gemma-3n-E4B-bf16": "google/gemma-3n-E4B",
    "lmstudio-community/granite-vision-3.2-2b-GGUF": "ibm-granite/granite-vision-3.2-2b",
    "mradermacher/Lingshu-7B-GGUF": "lingshu-medical-mllm/Lingshu-7B",
}

TOKENIZER_SOURCE_ALIASES = {
    "THUDM/glm-4-9b-chat-1m-hf": "THUDM/glm-4-9b-chat-1m",
}


def patch_transformers_compat() -> None:
    """Handle remote model code written against slightly different Transformers APIs."""
    patch_tied_weights_compat()
    patch_torch_device_compat()
    patch_remote_code_imports()
    patch_dynamic_cache_compat()


def patch_tied_weights_compat() -> None:
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def get_keys(self: PreTrainedModel) -> dict[str, None]:
        keys = getattr(self, "_all_tied_weights_keys", None)
        if keys is None:
            keys = getattr(self, "_tied_weights_keys", None) or []
        if hasattr(keys, "keys"):
            return keys
        return {key: None for key in keys}

    def set_keys(self: PreTrainedModel, keys: Any) -> None:
        self._all_tied_weights_keys = keys

    PreTrainedModel.all_tied_weights_keys = property(get_keys, set_keys)


def patch_torch_device_compat() -> None:
    from transformers.utils import generic

    if not hasattr(generic, "_is_torch_device"):
        generic._is_torch_device = lambda value: isinstance(value, torch.device)


def patch_remote_code_imports() -> None:
    import transformers.pytorch_utils as pytorch_utils
    import transformers.utils as utils
    from transformers.utils import import_utils

    if not hasattr(utils, "LossKwargs"):
        from typing import TypedDict

        class LossKwargs(TypedDict, total=False):
            num_items_in_batch: torch.Tensor

        utils.LossKwargs = LossKwargs
    if not hasattr(utils, "is_flash_attn_available"):
        utils.is_flash_attn_available = utils.is_flash_attn_2_available
    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True
    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices


def find_pruneable_heads_and_indices(
    heads: list[int], n_heads: int, head_size: int, already_pruned_heads: set[int]
) -> tuple[set[int], torch.Tensor]:
    remaining = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in remaining:
        shifted = head - sum(pruned < head for pruned in already_pruned_heads)
        mask[shifted] = 0
    index = torch.arange(mask.numel())[mask.view(-1).eq(1)].long()
    return remaining, index


def patch_dynamic_cache_compat() -> None:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = lambda self: getattr(self, "max_cache_len", None)


patch_transformers_compat()


@dataclass
class EvalContext:
    args: argparse.Namespace
    model_type: str
    task: str
    device: torch.device
    dtype: torch.dtype
    model: Any
    io: Any

    label_embeddings: torch.Tensor | None = None

@dataclass
class Prediction:
    value: str
    extra: dict[str, Any]


@dataclass
class Scoreboard:
    total: int = 0
    labeled_total: int = 0
    correct: int = 0
    tag_parse_failures: int = 0
    example_scores: list[float] = None
    predictions: list[str] = None
    targets: list[str] = None

    def __post_init__(self) -> None:
        self.example_scores = [] if self.example_scores is None else self.example_scores
        self.predictions = [] if self.predictions is None else self.predictions
        self.targets = [] if self.targets is None else self.targets

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labeled_total if self.labeled_total else None

    def add(self, score: float | bool | None, prediction: str = "", target: str = "") -> None:
        self.total += 1
        if score is None:
            return
        value = float(score)
        self.labeled_total += 1
        self.correct += int(value == 1.0)
        self.example_scores.append(value)
        self.predictions.append(prediction)
        self.targets.append(target)

    def score(self, metric: str | None) -> float | None:
        if not self.example_scores:
            return None
        if metric == "macro_f1":
            return macro_f1_score(self.predictions, self.targets)
        return sum(self.example_scores) / len(self.example_scores)

    def add_parse_status(self, status: str | None) -> None:
        self.tag_parse_failures += int(bool(status and status != "ok"))


def progress(message: str) -> None:
    print(f"[eval] {message}", flush=True)


def report_example_progress(completed: int, total: int) -> None:
    interval = max(1, total // 20)
    if completed <= 3 or completed == total or completed % interval == 0:
        progress(f"examples complete: {completed}/{total}")


def model_source(args: argparse.Namespace) -> str:
    return getattr(args, "model_source", args.model)


def tokenizer_source(args: argparse.Namespace) -> str:
    return TOKENIZER_SOURCE_ALIASES.get(args.model, model_source(args))


def dataset_source(args: argparse.Namespace) -> str:
    return getattr(args, "dataset_source", "") or args.dataset


def prepare_model_source(args: argparse.Namespace) -> None:
    requested = getattr(args, "model_source", "") or args.model
    source = MODEL_SOURCE_ALIASES.get(requested, requested)
    args.model_source = source
    if source != requested and args.model_type == "gguf":
        args.model_type = "auto"
    if args.model_type in UNSUPPORTED_MODEL_TYPES:
        return
    if args.model_type == "gguf" or (
        args.model_type == "auto" and is_gguf_model_name(source)
    ):
        args.model_source = str(resolve_gguf_path(source, args.gguf_file))
        return
    if not args.sanitize_model_config:
        return
    if args.model in FORCE_SANITIZE_MODELS:
        snapshot_dir = Path(snapshot_download(source))
        args.model_source = str(sanitize_model_dir(snapshot_dir, args))
        return
    if Path(source).exists():
        args.model_source = str(sanitize_model_dir(Path(source), args))
        return
    try:
        AutoConfig.from_pretrained(source, trust_remote_code=True)
        return
    except Exception as exc:
        if not is_sanitizable_config_error(exc):
            raise
    snapshot_dir = Path(snapshot_download(source))
    args.model_source = str(sanitize_model_dir(snapshot_dir, args))


def is_sanitizable_config_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    if "use_cache" in text and ("NoneType" in text or "got None" in text):
        return True
    if "Should have a `model_type` key" in text:
        return True
    return isinstance(exc, SyntaxError)


def sanitize_model_dir(source_dir: Path, args: argparse.Namespace) -> Path:
    config_path = source_dir / "config.json"
    preprocessor_path = source_dir / "preprocessor_config.json"

    sanitized_config, config_changed = load_sanitized_json(config_path, sanitize_config_value)
    sanitized_config, override_changed = apply_model_config_override(
        sanitized_config, args.model
    )
    config_changed = config_changed or override_changed
    sanitized_preprocessor, preprocessor_changed = load_sanitized_json(
        preprocessor_path,
        sanitize_preprocessor_config,
    )
    python_changes = sanitized_python_sources(source_dir)
    if not config_changed and not preprocessor_changed and not python_changes:
        return source_dir

    target_dir = Path(args.sanitized_model_dir) if args.sanitized_model_dir else default_sanitized_model_dir(args)
    target_dir.mkdir(parents=True, exist_ok=True)
    changed_files = {
        "config.json": config_changed,
        "preprocessor_config.json": preprocessor_changed,
        **{name: True for name in python_changes},
    }
    link_snapshot_files(source_dir, target_dir, skip_changed=changed_files)
    if config_changed:
        write_json_file(target_dir / "config.json", sanitized_config)
    if preprocessor_changed:
        write_json_file(target_dir / "preprocessor_config.json", sanitized_preprocessor)
    for name, content in python_changes.items():
        (target_dir / name).write_text(content, encoding="utf-8")
    return target_dir


def apply_model_config_override(value: Any, model: str) -> tuple[Any, bool]:
    model_type = MODEL_CONFIG_OVERRIDES.get(model)
    if not model_type or not isinstance(value, dict):
        return value, False
    updated = dict(value)
    updated["model_type"] = model_type
    updated["architectures"] = ["LlamaForCausalLM"]
    updated.pop("auto_map", None)
    return updated, updated != value


def sanitized_python_sources(source_dir: Path) -> dict[str, str]:
    changes = {}
    broken = "k.split(keyword + \".\")"
    fixed = "k.split(keyword + " + chr(39) + "." + chr(39) + ")"
    for path in source_dir.glob("*.py"):
        content = path.read_text(encoding="utf-8")
        updated = content.replace(broken, fixed)
        if updated != content:
            changes[path.name] = updated
    return changes


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
        value = dict(value)
        architectures = value.get("architectures") or []
        if "model_type" not in value and any("Bert" in name for name in architectures):
            value["model_type"] = "bert"
            changed = True
        if "rope_type" in value and "type" not in value:
            value["type"] = value["rope_type"]
            changed = True
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

def default_sanitized_model_dir(args: argparse.Namespace) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model.strip("/"))
    return Path(args.output_dir) / "_sanitized_models" / safe_name


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



def normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def has_gold_answer(answer: Any) -> bool:
    if isinstance(answer, list):
        return any(has_gold_answer(item) for item in answer)
    return answer is not None and str(answer).strip() != ""


def load_label_map(args: argparse.Namespace) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if args.label_map_file:
        mapping.update(json.loads(Path(args.label_map_file).read_text(encoding="utf-8")))
    if args.label_map:
        mapping.update(json.loads(args.label_map))
    return {str(key): str(value) for key, value in mapping.items()}


def load_dataset_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    raw = getattr(args, "dataset_kwargs", "")
    if not raw:
        return {}
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("--dataset-kwargs must decode to a JSON object.")
    return value


def example_value(example: dict[str, Any], path: str, default: Any = None) -> Any:
    if not path:
        return default
    if path in example:
        return example[path]
    value = read_path(example, path.split("."))
    return default if value is MISSING else value


MISSING = object()


def read_path(value: Any, parts: list[str]) -> Any:
    if not parts:
        return value
    if isinstance(value, list):
        return collect_list_path(value, parts)
    if not isinstance(value, dict):
        return MISSING
    key = clean_path_part(parts[0])
    if key not in value:
        return MISSING
    return read_path(value[key], parts[1:])


def clean_path_part(part: str) -> str:
    return part[:-2] if part.endswith("[]") else part


def collect_list_path(values: list[Any], parts: list[str]) -> list[Any]:
    collected = []
    for item in values:
        found = read_path(item, parts)
        if found is MISSING:
            continue
        if isinstance(found, list):
            collected.extend(found)
        else:
            collected.append(found)
    return collected


def render_prompt_template(template: str, example: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        value = example_value(example, name, "")
        return format_template_value(example, name, value)

    return re.sub(r"\{([A-Za-z0-9_.\[\]-]+)\}", replace, template)


def format_template_value(example: dict[str, Any], name: str, value: Any) -> str:
    if name == "turns.utterance":
        speakers = example_value(example, "turns.speaker", [])
        return format_dialogue_turns(value, speakers)
    return format_prompt_value(value)


def format_dialogue_turns(utterances: Any, speakers: Any) -> str:
    if not isinstance(utterances, list):
        return format_prompt_value(utterances)
    if not isinstance(speakers, list) or len(speakers) != len(utterances):
        return "\n".join(f"Turn {index}: {text}" for index, text in enumerate(utterances, 1))
    roles = {0: "User", 1: "Assistant"}
    return "\n".join(
        f"{roles.get(speaker, f'Turn {index}')}: {text}"
        for index, (speaker, text) in enumerate(zip(speakers, utterances), 1)
    )


def target_answer(ctx: EvalContext, answer: Any) -> Any:
    if not has_gold_answer(answer):
        return answer
    mapping = getattr(ctx.args, "_label_map", {})
    extracted = regex_target_answer(ctx.args, answer)
    raw_label = threshold_target_label(ctx.args, extracted)
    return mapping.get(raw_label, extracted)


def evaluation_answer(ctx: EvalContext, example: dict[str, Any]) -> Any:
    answer = example_value(example, ctx.args.answer_column)
    if ctx.args.evaluation_protocol != "token_entity_set_f1":
        return answer
    tokens = example_value(example, ctx.args.question_column, [])
    return tagged_token_entities(ctx, tokens, answer)


def tagged_token_entities(
    ctx: EvalContext, tokens: Any, raw_labels: Any
) -> list[dict[str, str]]:
    if not isinstance(tokens, list) or not isinstance(raw_labels, list):
        return []
    mapping = getattr(ctx.args, "_label_map", {})
    labels = [mapping.get(str(value), str(value)) for value in raw_labels]
    entities: list[dict[str, str]] = []
    current_type = ""
    current_tokens: list[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_tokens:
            entities.append({"type": current_type, "text": " ".join(current_tokens)})
        current_type = ""
        current_tokens = []

    for token, label in zip(tokens, labels):
        prefix, entity_type = split_bio_label(label)
        if prefix == "O":
            flush()
        elif prefix == "B" or not current_tokens or current_type != entity_type:
            flush()
            current_type = entity_type
            current_tokens = [str(token)]
        else:
            current_tokens.append(str(token))
    flush()
    return entities


def regex_target_answer(args: argparse.Namespace, answer: Any) -> Any:
    pattern = getattr(args, "answer_regex", "")
    if not pattern:
        return answer
    match = re.search(pattern, str(answer), flags=re.DOTALL)
    if not match:
        return answer
    group = match.group(1) if match.lastindex else match.group(0)
    return group.strip()

def threshold_target_label(args: argparse.Namespace, answer: Any) -> str:
    threshold = getattr(args, "label_threshold", None)
    if threshold is None:
        return str(answer)
    return "1" if float(answer) >= threshold else "0"




def target_answers(ctx: EvalContext, answer: Any) -> list[Any]:
    if isinstance(answer, list):
        return [target_answer(ctx, item) for item in answer if has_gold_answer(item)]
    return [target_answer(ctx, answer)]


def is_correct_prediction(ctx: EvalContext, prediction: str, answer: Any) -> bool | None:
    score = score_prediction(ctx, prediction, answer)
    return None if score is None else score == 1.0


def score_prediction(ctx: EvalContext, prediction: str, answer: Any) -> float | None:
    method = resolved_evaluation_method(ctx.args)
    answers = target_answers(ctx, answer)
    if method == "set_f1" and not answers and isinstance(
        answer, (dict, list, tuple, set)
    ):
        return set_f1_score(prediction, answer)
    if not answers:
        return None
    if method == "set_f1":
        return set_f1_score(prediction, answers)
    if method == "qa_f1":
        return max(token_f1_score(prediction, reference) for reference in answers)
    if method == "numeric_match":
        return max(numeric_match_score(prediction, reference) for reference in answers)
    if method == "rouge_l":
        return max(rouge_l_score(prediction, reference) for reference in answers)
    return float(
        any(normalize_answer(prediction) == normalize_answer(reference) for reference in answers)
    )


def resolved_evaluation_method(args: argparse.Namespace) -> str:
    method = getattr(args, "evaluation_method", "auto")
    if method not in {"", "auto"}:
        return method
    protocol = getattr(args, "evaluation_protocol", "")
    if protocol == "tagged_answer_exact_match":
        return "exact_match"
    if protocol in {"tagged_set_generation", "multilabel_classifier_set_f1", "token_entity_set_f1"}:
        return "set_f1"
    return "accuracy"


def token_f1_score(prediction: Any, reference: Any) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def numeric_match_score(prediction: Any, reference: Any) -> float:
    try:
        return float(parse_number(prediction) == parse_number(reference))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return float(normalize_answer(prediction) == normalize_answer(reference))


def parse_number(value: Any) -> Fraction:
    text = str(value).strip().replace(",", "")
    expressions = numeric_expressions(text)
    if not expressions:
        raise ValueError(f"No numeric value found in {value!r}")
    return parse_numeric_expression(expressions[-1])


def numeric_expressions(text: str) -> list[str]:
    number = r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)"
    pattern = rf"(?<!\w)({number}(?:\s*[:/]\s*{number})?%?)(?!\w)"
    return [match.group(1) for match in re.finditer(pattern, text)]


def parse_numeric_expression(value: str) -> Fraction:
    text = value.replace(" ", "")
    if text.endswith("%"):
        return Fraction(Decimal(text[:-1])) / 100
    separator = ":" if ":" in text else "/" if "/" in text else ""
    if separator:
        numerator, denominator = text.split(separator, 1)
        return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
    return Fraction(Decimal(text))


def rouge_l_score(prediction: Any, reference: Any) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = lcs_length(predicted, expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1]


def set_f1_score(prediction: Any, references: list[Any]) -> float:
    predicted = canonical_set(prediction)
    expected = canonical_set(references)
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = len(predicted & expected)
    return 2 * overlap / (len(predicted) + len(expected))



def canonical_entity_items(value: dict[str, Any]) -> set[str]:
    text = value.get("text", value.get("mention"))
    if text is None:
        return set()
    mentions = text if isinstance(text, list) else [text]
    entity_type = normalize_entity_type(value.get("type", "entity"))
    return {
        f"{entity_type}::{normalize_answer(mention)}"
        for mention in mentions
        if normalize_answer(mention)
    }



def canonical_set(value: Any) -> set[str]:
    if isinstance(value, str):
        parsed = parse_json_value(value)
        if parsed is not value:
            return canonical_set(parsed)
        pieces = re.split(r"[,;\n]+", value)
        return {normalize_answer(piece) for piece in pieces if normalize_answer(piece)}
    if isinstance(value, dict):
        entities = canonical_entity_items(value)
        return entities or {normalize_answer(json.dumps(value, sort_keys=True, ensure_ascii=False))}
    if isinstance(value, (list, tuple, set)):
        items: set[str] = set()
        for item in value:
            items.update(canonical_set(item))
        return items
    text = normalize_answer(value)
    return {text} if text else set()


def parse_json_value(value: str) -> Any:
    text = value.strip()
    if not text.startswith(("[", "{")):
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def macro_f1_score(predictions: list[str], targets: list[str]) -> float:
    labels = sorted(set(predictions) | set(targets))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        true_positive = sum(p == label and t == label for p, t in zip(predictions, targets))
        false_positive = sum(p == label and t != label for p, t in zip(predictions, targets))
        false_negative = sum(p != label and t == label for p, t in zip(predictions, targets))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return sum(scores) / len(scores)


def primary_target(ctx: EvalContext, answer: Any) -> str:
    answers = target_answers(ctx, answer)
    return normalize_answer(answers[0]) if answers else ""


def infer_model_type(model_name: str, explicit: str) -> str:
    if explicit == "gguf" or (explicit == "auto" and is_gguf_model_name(model_name)):
        return "gguf"
    if explicit == "paligemma" or (explicit == "auto" and "paligemma" in model_name.lower()):
        return "paligemma"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if is_voxtral_config(config):
        return "voxtral"
    if is_vlm_chat_config(config):
        return "vlm_chat"
    if is_vlm_processor_config(config):
        return "vlm_processor"
    if explicit != "auto":
        return explicit
    if is_zero_shot_image_config(config):
        return "zero_shot_image"
    architectures = [str(name).lower() for name in getattr(config, "architectures", []) or []]
    if any("questionanswering" in name for name in architectures):
        return "extractive_qa"
    if any("tokenclassification" in name for name in architectures):
        return "token_classifier"
    if any("maskedlm" in name for name in architectures):
        return "masked_lm"
    if "sentence-t5" in model_name.lower() or any(name == "t5encodermodel" for name in architectures):
        return "sentence_encoder"
    if any("sequenceclassification" in name for name in architectures):
        return "sequence_classifier"
    return "seq2seq_lm" if getattr(config, "is_encoder_decoder", False) else "causal_lm"


def is_voxtral_config(config: Any) -> bool:
    return str(getattr(config, "model_type", "")).lower() == "voxtral"


def is_zero_shot_image_config(config: Any) -> bool:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [str(name).lower() for name in getattr(config, "architectures", []) or []]
    if model_type in {"clip", "siglip", "siglip2"}:
        return True
    return any(name.endswith("clipmodel") or name.endswith("siglipmodel") for name in architectures)



def is_vlm_chat_config(config: Any) -> bool:
    architectures = [str(name).lower() for name in getattr(config, "architectures", []) or []]
    has_vision_config = hasattr(config, "vision_config")
    has_text_config = hasattr(config, "llm_config") or hasattr(config, "text_config")
    return has_vision_config and has_text_config and any("chat" in name for name in architectures)


def is_vlm_processor_config(config: Any) -> bool:
    model_type = str(getattr(config, "model_type", "")).lower()
    architectures = [str(name).lower() for name in getattr(config, "architectures", []) or []]
    has_vision_config = hasattr(config, "vision_config")
    if model_type in {"qwen2_vl", "qwen2_5_vl", "qwen3_vl"}:
        return True
    return has_vision_config and any("conditionalgeneration" in name for name in architectures)


def infer_task(example: dict[str, Any], args: argparse.Namespace) -> str:
    if args.task != "auto":
        return args.task
    if example_value(example, args.image_column) is not None:
        return "image_classification"
    has_choices = example_value(example, args.choices_column) is not None
    return "multiple_choice" if has_choices else "generation"


def choose_device(device_name: str) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def finite_tokenizer_limit(tokenizer: Any) -> int | None:
    limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or limit <= 0 or limit >= 1_000_000:
        return None
    return limit


def config_token_limit(model: torch.nn.Module) -> int | None:
    config = getattr(model, "config", None)
    for name in ("max_position_embeddings", "n_positions", "seq_length"):
        limit = getattr(config, name, None)
        if isinstance(limit, int) and 0 < limit < 1_000_000:
            return limit
    return None


def max_input_tokens(ctx: EvalContext, reserve: int = 0) -> int:
    limits = [finite_tokenizer_limit(ctx.io), config_token_limit(ctx.model)]
    usable = [limit for limit in limits if limit]
    limit = min(usable) if usable else 512
    if ctx.args.max_input_tokens:
        limit = min(limit, ctx.args.max_input_tokens)
    return max(8, limit - max(0, reserve))


def load_eval_dataset(args: argparse.Namespace):
    dataset = load_dataset_compatible(
        dataset_source(args),
        args.subset or None,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
        **getattr(args, "_dataset_kwargs", {}),
    )
    dataset = adapt_dialogue_dataset(args.dataset, dataset)
    args._source_total = len(dataset)
    dataset, indices = random_subset(dataset, args.limit, args.seed)
    args._selected_indices = indices
    return dataset


def random_subset(dataset: Any, limit: int, seed: int) -> tuple[Any, list[int] | None]:
    if limit <= 0 or len(dataset) <= limit:
        return dataset, None
    indices = random.Random(seed).sample(range(len(dataset)), limit)
    return dataset.select(indices), indices


def build_context(args: argparse.Namespace, first_example: dict[str, Any]) -> EvalContext:
    model_type = infer_model_type(model_source(args), args.model_type)
    task = infer_task(first_example, args)
    device = choose_device(args.device)
    dtype = choose_dtype(device)
    started = time.monotonic()
    progress(
        f"model initialization start: {args.model} type={model_type} device={device} dtype={dtype}"
    )
    model, io = load_model_and_io(args, model_type, device, dtype)
    progress(f"model initialization complete in {time.monotonic() - started:.1f}s: {args.model}")
    ctx = EvalContext(args, model_type, task, device, dtype, model, io)
    if args.evaluation_protocol == "auto":
        args.evaluation_protocol = infer_evaluation_protocol(ctx)
    return ctx

def infer_evaluation_protocol(ctx: EvalContext) -> str:
    if ctx.model_type == "extractive_qa":
        return "extractive_qa" if ctx.task == "qa" else "unsupported"
    if ctx.task == "multiple_choice":
        if ctx.model_type == "masked_lm":
            return "masked_choice_likelihood"
        if ctx.model_type in {"sentence_encoder", "sequence_classifier", "token_classifier"}:
            return "unsupported"
        if getattr(ctx.args, "allow_label_scores", False):
            return "multiple_choice_accuracy"
        return "tagged_multiple_choice_accuracy"
    if ctx.task == "multilabel_classification":
        if ctx.model_type == "sequence_classifier":
            return "multilabel_classifier_set_f1"
        return "tagged_set_generation"
    if ctx.model_type == "zero_shot_image":
        return "zero_shot_image_accuracy"
    if ctx.model_type == "sentence_encoder":
        return "embedding_label_similarity" if getattr(ctx.args, "_label_map", {}) else "unsupported"
    if ctx.model_type == "masked_lm":
        return "masked_label_likelihood" if getattr(ctx.args, "_label_map", {}) else "unsupported"
    if ctx.model_type == "token_classifier":
        return "token_entity_set_f1" if ctx.task == "token_classification" else "unsupported"
    if ctx.model_type == "sequence_classifier":
        return "unsupported"
    if getattr(ctx.args, "_label_map", {}):
        return "tagged_label_generation_accuracy"
    return "tagged_answer_exact_match"


def load_model_and_io(args: argparse.Namespace, model_type: str, device: torch.device, dtype: torch.dtype):
    if model_type == "gguf":
        return load_gguf_backend(args)
    if model_type == "sentence_encoder":
        return load_sentence_encoder(args, device, dtype)
    if model_type == "masked_lm":
        return load_encoder_adapter(args, device, dtype, AutoModelForMaskedLM)
    if model_type == "token_classifier":
        return load_encoder_adapter(args, device, dtype, AutoModelForTokenClassification, require_fast=True)
    if model_type == "extractive_qa":
        return load_encoder_adapter(args, device, dtype, AutoModelForQuestionAnswering, require_fast=True)
    if model_type == "zero_shot_image":
        return load_zero_shot_image(args, device, dtype)
    if model_type == "paligemma":
        return load_paligemma(args, device, dtype)
    if model_type == "vlm_chat":
        return load_vlm_chat(args, device, dtype)
    if model_type == "vlm_processor":
        return load_vlm_processor(args, device, dtype)
    if model_type == "voxtral":
        return load_voxtral(args, device, dtype)
    if model_type == "sequence_classifier":
        return load_sequence_classifier(args, device, dtype)
    if model_type in UNSUPPORTED_MODEL_TYPES:
        return torch.nn.Identity().eval(), None
    return load_text_model(args, model_type, device, dtype)


def load_gguf_backend(args: argparse.Namespace) -> tuple[GgufBackend, GgufBackend]:
    backend = GgufBackend.load(
        Path(model_source(args)),
        context_size=args.gguf_context_size,
        gpu_layers=args.gguf_gpu_layers,
        seed=args.seed,
        chat_format=args.gguf_chat_format,
    )
    return backend, backend




def load_zero_shot_image(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
):
    processor = AutoProcessor.from_pretrained(model_source(args), trust_remote_code=True)
    model_dtype = torch.float32 if device.type == "cpu" else dtype
    model = AutoModel.from_pretrained(
        model_source(args), torch_dtype=model_dtype, trust_remote_code=True
    ).to(device)
    return model.eval(), processor

def load_paligemma(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_source(args), trust_remote_code=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_source(args),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    return model.eval(), processor


def load_voxtral(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
) -> tuple[Any, Any]:
    tokenizer = load_tokenizer(args)
    model_dtype = torch.float32 if device.type == "cpu" else dtype
    model = VoxtralForConditionalGeneration.from_pretrained(
        model_source(args), torch_dtype=model_dtype
    ).to(device)
    sync_generation_special_tokens(model, tokenizer)
    return model.eval(), tokenizer


def load_tokenizer(args: argparse.Namespace, prefer_slow: bool = False):
    source = tokenizer_source(args)
    if prefer_slow:
        return AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)
    try:
        return AutoTokenizer.from_pretrained(source, trust_remote_code=True)
    except Exception as exc:
        if not should_retry_slow_tokenizer(exc):
            raise
    return AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=False)


def should_retry_slow_tokenizer(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    markers = ("SentencePiece", "tiktoken", "protobuf", "add_prefix_space", "prepend_scheme")
    return any(marker in text for marker in markers)


def valid_token_id(value: Any, vocab_size: int) -> bool:
    return isinstance(value, int) and 0 <= value < vocab_size


def tokenizer_vocab_size(tokenizer: Any) -> int | None:
    try:
        return len(tokenizer)
    except (AttributeError, TypeError):
        value = getattr(tokenizer, "vocab_size", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def sync_generation_special_tokens(model: Any, tokenizer: Any) -> None:
    size = tokenizer_vocab_size(tokenizer)
    if size is None:
        return
    names = ("pad_token_id", "eos_token_id", "bos_token_id")
    token_ids = {name: getattr(tokenizer, name, None) for name in names}
    if not valid_token_id(token_ids["pad_token_id"], size):
        fallback = next(
            (
                token_ids[name]
                for name in ("eos_token_id", "bos_token_id")
                if valid_token_id(token_ids[name], size)
            ),
            None,
        )
        if fallback is not None:
            tokenizer.pad_token_id = fallback
            token_ids["pad_token_id"] = fallback

    config = getattr(model, "config", None)
    current = [getattr(config, name, None) for name in names]
    if all(valid_token_id(value, size) for value in current):
        return
    for target in (config, getattr(model, "generation_config", None)):
        if target is None:
            continue
        for name in names:
            token_id = token_ids[name]
            if valid_token_id(token_id, size):
                setattr(target, name, token_id)


def load_text_model(args: argparse.Namespace, model_type: str, device: torch.device, dtype: torch.dtype):
    progress(f"tokenizer load start: {args.model}")
    tokenizer = load_tokenizer(args, prefer_slow=model_type == "seq2seq_lm")
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    progress(f"tokenizer load complete: {args.model}")

    model_class = AutoModelForSeq2SeqLM if model_type == "seq2seq_lm" else AutoModelForCausalLM
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if model_type == "causal_lm":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cpu":
        model_kwargs["torch_dtype"] = torch.float32

    progress(f"weight load start: {args.model}")
    model = load_pretrained_text_model(model_class, model_source(args), model_kwargs)
    sync_model_token_embeddings(model, tokenizer)
    sync_generation_special_tokens(model, tokenizer)
    progress(f"weight load complete; moving model to {device}: {args.model}")
    model = model.to(device)
    progress(f"device transfer complete: {args.model}")
    return model.eval(), tokenizer


def load_pretrained_text_model(model_class: Any, source: str, kwargs: dict[str, Any]) -> Any:
    try:
        return model_class.from_pretrained(source, **kwargs)
    except ValueError as exc:
        text = str(exc)
        if "does not support an attention implementation" in text:
            eager_kwargs = {**kwargs, "attn_implementation": "eager"}
            return model_class.from_pretrained(source, **eager_kwargs)
        if "for this kind of AutoModel: AutoModelForCausalLM" in text:
            return AutoModel.from_pretrained(source, **kwargs)
        raise


def sync_model_token_embeddings(model: Any, tokenizer: Any) -> None:
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tokenizer_size = tokenizer_vocab_size(tokenizer)
    if embeddings is None or tokenizer_size is None:
        return
    if tokenizer_size > embeddings.num_embeddings and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(tokenizer_size)


def load_sentence_encoder(
    args: argparse.Namespace, device: torch.device, dtype: torch.dtype
):
    model, tokenizer = load_encoder_adapter(args, device, dtype, AutoModel)
    if getattr(model.config, "is_encoder_decoder", False) and hasattr(model, "get_encoder"):
        model = model.get_encoder().to(device).eval()
    return model, tokenizer


def load_encoder_adapter(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    model_class: Any,
    require_fast: bool = False,
):
    tokenizer = load_tokenizer(args)
    if require_fast and not getattr(tokenizer, "is_fast", False):
        raise TypeError(f"Model {args.model!r} requires a fast tokenizer for offset-based token evaluation.")
    model_dtype = torch.float32 if device.type == "cpu" else dtype
    model = model_class.from_pretrained(
        model_source(args), torch_dtype=model_dtype, trust_remote_code=True
    )
    sync_model_token_embeddings(model, tokenizer)
    return model.to(device).eval(), tokenizer


def load_sequence_classifier(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    tokenizer = load_tokenizer(args)
    model_kwargs = {"torch_dtype": torch.float32 if device.type == "cpu" else dtype, "trust_remote_code": True}
    model = AutoModelForSequenceClassification.from_pretrained(model_source(args), **model_kwargs)
    sync_model_token_embeddings(model, tokenizer)
    return model.to(device).eval(), tokenizer


def load_vlm_chat(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    tokenizer = load_tokenizer(args)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float32 if device.type == "cpu" else dtype,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    model = AutoModel.from_pretrained(model_source(args), **model_kwargs).to(device)
    if not hasattr(model, "chat"):
        raise TypeError(f"Model {args.model!r} was detected as vlm_chat but does not expose a chat() method.")
    return model.eval(), tokenizer




def load_vlm_processor(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_source(args), trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": torch.float32 if device.type == "cpu" else dtype,
        "trust_remote_code": True,
        "attn_implementation": args.attn_implementation,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_source(args), **model_kwargs)
    except ValueError as exc:
        if "does not support an attention implementation" not in str(exc):
            raise
        model_kwargs["attn_implementation"] = "eager"
        model = AutoModelForImageTextToText.from_pretrained(model_source(args), **model_kwargs)
    return model.to(device).eval(), processor

def make_image(ctx: EvalContext, example: dict[str, Any]) -> Image.Image:
    image = example_value(example, ctx.args.image_column)
    if image is None:
        image = Image.new("RGB", (ctx.args.image_size, ctx.args.image_size), color="white")
    if not isinstance(image, Image.Image):
        raise TypeError(f"Column {ctx.args.image_column!r} is not a PIL image; got {type(image).__name__}")
    return image.convert("RGB")


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC).convert("RGB")
    pixels = torch.as_tensor(bytearray(image.tobytes()), dtype=torch.uint8).view(image_size, image_size, 3)
    pixels = pixels.permute(2, 0, 1).to(torch.float32) / 255.0
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (pixels - mean) / std


def make_pixel_values(ctx: EvalContext, example: dict[str, Any]) -> torch.Tensor:
    image = make_image(ctx, example)
    pixels = image_to_tensor(image, ctx.args.image_size).unsqueeze(0)
    return pixels.to(device=ctx.device, dtype=ctx.dtype)


def make_processor_image(ctx: EvalContext, example: dict[str, Any]) -> Image.Image:
    image = make_image(ctx, example)
    max_side = max(image.size)
    if max_side <= ctx.args.image_size:
        return image
    resized = image.copy()
    resized.thumbnail((ctx.args.image_size, ctx.args.image_size), Image.Resampling.BICUBIC)
    return resized


def get_choices(ctx: EvalContext, example: dict[str, Any]) -> tuple[list[str], list[str]]:
    separate = getattr(ctx.args, "_choices_columns", [])
    if separate:
        texts = [format_prompt_value(example_value(example, column)) for column in separate]
        labels = [chr(ord("A") + idx) for idx in range(len(texts))]
        return labels, texts
    choices = parse_choices_value(example_value(example, ctx.args.choices_column))
    if isinstance(choices, dict):
        labels = [str(label) for label in choices.get("label", [])]
        texts = [str(text) for text in choices.get("text", [])]
        return labels, texts
    if isinstance(choices, list):
        labels = [chr(ord("A") + idx) for idx in range(len(choices))]
        texts = [str(choice) for choice in choices]
        return labels, texts
    raise TypeError(f"Unsupported choices format in column {ctx.args.choices_column!r}: {type(choices).__name__}")


def parse_choices_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, (dict, list)):
            return parsed
    return value


def build_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    if ctx.task == "multiple_choice":
        prompt = build_multiple_choice_prompt(ctx, example)
    else:
        prompt = build_generation_prompt(ctx, example)
    return add_image_prefix(ctx, prompt)


def build_generation_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    question = format_prompt_value(example_value(example, ctx.args.question_column))
    if ctx.args.prompt_template:
        return render_prompt_template(ctx.args.prompt_template, example)
    return question


def format_prompt_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(format_prompt_value(item) for item in value if item is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_multiple_choice_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    labels, texts = get_choices(ctx, example)
    choice_lines = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    question = format_prompt_value(example_value(example, ctx.args.question_column))
    if ctx.args.prompt_template:
        question = render_prompt_template(ctx.args.prompt_template, example)
    return (
        "Choose the best answer.\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choice_lines}\n"
        "Answer:"
    )


def add_image_prefix(ctx: EvalContext, prompt: str) -> str:
    if ctx.model_type != "paligemma":
        return prompt
    return prompt if prompt.startswith("<image>") else f"<image>answer en {prompt}"


def option_continuations(ctx: EvalContext, example: dict[str, Any]) -> tuple[list[str], list[str]]:
    labels, texts = get_choices(ctx, example)
    if ctx.args.scoring == "answer_text":
        return labels, [" " + text for text in texts]
    return labels, [" " + label for label in labels]



def mean_pool_embeddings(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch.nn.functional.normalize(pooled, p=2, dim=1)


def encode_texts(ctx: EvalContext, texts: list[str]) -> torch.Tensor:
    batch = ctx.io(
        texts,
        padding=True,
        truncation=True,
        max_length=max_input_tokens(ctx),
        return_tensors="pt",
    )
    batch = move_batch(ctx, dict(batch))
    with torch.no_grad():
        hidden = ctx.model(**batch).last_hidden_state
    return mean_pool_embeddings(hidden, batch["attention_mask"])


def predict_embedding_label(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels = candidate_label_values(ctx)
    if not labels:
        raise ValueError("embedding_label_similarity requires at least one candidate label")
    if ctx.label_embeddings is None:
        ctx.label_embeddings = encode_texts(ctx, labels)
    embedding = encode_texts(ctx, [build_generation_prompt(ctx, example)])
    scores = torch.matmul(ctx.label_embeddings, embedding[0])
    score_map = {label: float(score) for label, score in zip(labels, scores)}
    return Prediction(max(score_map, key=score_map.get), {"embedding_scores": score_map})


def masked_prompt_template(ctx: EvalContext, example: dict[str, Any]) -> str:
    if ctx.args.prompt_template:
        sentinel = "__EVALUATION_LABEL__"
        template = ctx.args.prompt_template.replace("{label}", sentinel)
        rendered = render_prompt_template(template, example).replace(sentinel, "{label}")
        return rendered if "{label}" in rendered else f"{rendered.rstrip()}\n\nAnswer: {{label}}"
    return f"{build_prompt(ctx, example).rstrip()}\n\nAnswer: {{label}}"


def candidate_token_positions(
    offsets: list[list[int]] | list[tuple[int, int]], start: int, end: int
) -> list[int]:
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]


def masked_candidate_score(ctx: EvalContext, template: str, candidate: str) -> tuple[float, int]:
    if template.count("{label}") != 1:
        raise ValueError("Masked-LM prompt template must contain exactly one {label} placeholder")
    if ctx.io.mask_token_id is None:
        raise ValueError("Masked-LM tokenizer does not define a mask token")
    prefix, suffix = template.split("{label}")
    text = f"{prefix}{candidate}{suffix}"
    encoding = ctx.io(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens(ctx),
    )
    offsets = encoding.pop("offset_mapping")[0].tolist()
    positions = candidate_token_positions(offsets, len(prefix), len(prefix) + len(candidate))
    if not positions:
        text = candidate
        encoding = ctx.io(
            text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens(ctx),
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        positions = candidate_token_positions(offsets, 0, len(candidate))
    if not positions:
        raise ValueError(f"Candidate {candidate!r} produced no scoreable tokens")
    batch = move_batch(ctx, dict(encoding))
    original_ids = batch["input_ids"][0]
    token_scores = []
    for position in positions:
        masked_ids = batch["input_ids"].clone()
        masked_ids[0, position] = ctx.io.mask_token_id
        model_inputs = {**batch, "input_ids": masked_ids}
        with torch.no_grad():
            logits = ctx.model(**model_inputs).logits[0, position]
        token_scores.append(float(torch.log_softmax(logits, dim=-1)[original_ids[position]]))
    return sum(token_scores) / len(token_scores), len(token_scores)


def predict_masked_candidates(
    ctx: EvalContext,
    example: dict[str, Any],
    labels: list[str],
    candidates: list[str],
) -> Prediction:
    template = masked_prompt_template(ctx, example)
    scored = [masked_candidate_score(ctx, template, candidate) for candidate in candidates]
    scores = {label: score for label, (score, _) in zip(labels, scored)}
    lengths = {label: length for label, (_, length) in zip(labels, scored)}
    return Prediction(
        max(scores, key=scores.get),
        {"masked_candidate_scores": scores, "candidate_token_lengths": lengths},
    )


def predict_masked_label(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels = candidate_label_values(ctx)
    if not labels:
        raise ValueError("masked_label_likelihood requires at least one candidate label")
    return predict_masked_candidates(ctx, example, labels, labels)


def predict_masked_choice(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels, choices = get_choices(ctx, example)
    return predict_masked_candidates(ctx, example, labels, choices)


def input_text_segments(ctx: EvalContext, example: dict[str, Any]) -> list[str]:
    value = example_value(example, ctx.args.question_column)
    segments = collect_text_segments(value)
    return segments or [build_generation_prompt(ctx, example)]


def collect_text_segments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        if "text" in value:
            return collect_text_segments(value["text"])
        segments = []
        for child in value.values():
            segments.extend(collect_text_segments(child))
        return segments
    if isinstance(value, list):
        segments = []
        for child in value:
            segments.extend(collect_text_segments(child))
        return segments
    return []


def normalize_entity_type(value: Any) -> str:
    normalized = normalize_label_text(str(value))
    aliases = {"chem": "chemical", "chemical": "chemical", "disease": "disease"}
    return aliases.get(normalized, normalized or "entity")


def split_bio_label(label: str) -> tuple[str, str]:
    if label.upper() == "O":
        return "O", ""
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
        if prefix.upper() in {"B", "I"}:
            return prefix.upper(), normalize_entity_type(entity_type)
    return "B", normalize_entity_type(label)


def decode_bio_entities(
    text: str, labels: list[str], offsets: list[list[int]] | list[tuple[int, int]]
) -> list[dict[str, str]]:
    entities = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            entities.append({
                "type": current["type"],
                "text": text[current["start"]:current["end"]],
            })
        current = None

    for label, (start, end) in zip(labels, offsets):
        if end <= start:
            continue
        prefix, entity_type = split_bio_label(label)
        if prefix == "O":
            flush()
            continue
        if prefix == "B" or current is None or current["type"] != entity_type:
            flush()
            current = {"type": entity_type, "start": start, "end": end}
        else:
            current["end"] = max(current["end"], end)
    flush()
    return entities


def token_label(ctx: EvalContext, label_id: int) -> str:
    labels = getattr(ctx.model.config, "id2label", {}) or {}
    return str(labels.get(label_id, labels.get(str(label_id), label_id)))


def predict_token_entities(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    found: dict[str, dict[str, str]] = {}
    for text in input_text_segments(ctx, example):
        encoding = ctx.io(
            text,
            padding="longest",
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens(ctx),
            stride=ctx.args.token_stride,
        )
        offsets = encoding.pop("offset_mapping", None)
        encoding.pop("overflow_to_sample_mapping", None)
        if offsets is None:
            offsets = approximate_token_offsets(ctx.io, encoding["input_ids"], text)
        batch = move_batch(ctx, dict(encoding))
        with torch.no_grad():
            predictions = torch.argmax(ctx.model(**batch).logits, dim=-1).cpu()
        for row, row_offsets in zip(predictions, offsets):
            labels = [token_label(ctx, int(label_id)) for label_id in row]
            for entity in decode_bio_entities(text, labels, row_offsets.tolist()):
                key = f"{entity['type']}::{normalize_answer(entity['text'])}"
                found[key] = entity
    entities = list(found.values())
    return Prediction(
        json.dumps(entities, ensure_ascii=False),
        {"entities": entities, "entity_count": len(entities)},
    )



def approximate_token_offsets(tokenizer: Any, input_ids: torch.Tensor, text: str) -> torch.Tensor:
    rows = []
    folded = text.casefold()
    for input_row in input_ids:
        cursor = 0
        offsets = []
        for token_id in input_row.tolist():
            piece = tokenizer.decode(
                [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            ).strip()
            if not piece:
                offsets.append([0, 0])
                continue
            start = folded.find(piece.casefold(), cursor)
            if start < 0:
                offsets.append([0, 0])
                continue
            end = start + len(piece)
            offsets.append([start, end])
            cursor = end
        rows.append(offsets)
    return torch.tensor(rows, dtype=torch.long)


def extractive_window_span(
    context: str,
    sequence_ids: list[int | None],
    offsets: list[list[int]] | list[tuple[int, int]],
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_tokens: int,
) -> tuple[str, float, dict[str, int]]:
    valid = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1 and offsets[index][1] > offsets[index][0]
    ]
    if not valid:
        return "", float("-inf"), {}
    starts = sorted(valid, key=lambda index: float(start_logits[index]), reverse=True)[:20]
    ends = sorted(valid, key=lambda index: float(end_logits[index]), reverse=True)[:20]
    best_text = ""
    best_score = float("-inf")
    best_span: dict[str, int] = {}
    for start in starts:
        for end in ends:
            if end < start or end - start + 1 > max_answer_tokens:
                continue
            score = float(start_logits[start] + end_logits[end])
            if score <= best_score:
                continue
            left, right = int(offsets[start][0]), int(offsets[end][1])
            best_text = context[left:right]
            best_score = score
            best_span = {"start_token": start, "end_token": end}
    return best_text, best_score, best_span


def predict_extractive_qa(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    question = format_prompt_value(example_value(example, ctx.args.question_column, ""))
    context = format_prompt_value(example_value(example, ctx.args.context_column, ""))
    if not question or not context:
        raise ValueError("Extractive QA requires non-empty question and context values.")
    encoding = ctx.io(
        question,
        context,
        truncation="only_second",
        max_length=max_input_tokens(ctx),
        stride=ctx.args.token_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    sequence_ids = [
        encoding.sequence_ids(index)
        for index in range(encoding["input_ids"].shape[0])
    ]
    offsets = encoding.pop("offset_mapping")
    encoding.pop("overflow_to_sample_mapping", None)
    batch = move_batch(ctx, dict(encoding))
    with torch.no_grad():
        outputs = ctx.model(**batch)
    best_text = ""
    best_score = float("-inf")
    best_null = float("-inf")
    best_details: dict[str, Any] = {}
    for index, row_offsets in enumerate(offsets):
        text, score, span = extractive_window_span(
            context,
            sequence_ids[index],
            row_offsets.tolist(),
            outputs.start_logits[index],
            outputs.end_logits[index],
            ctx.args.max_answer_tokens,
        )
        null_score = float(outputs.start_logits[index, 0] + outputs.end_logits[index, 0])
        best_null = max(best_null, null_score)
        if score > best_score:
            best_text = text
            best_score = score
            best_details = {"window": index, **span}
    details = {"span_score": best_score, "null_score": best_null}
    details.update(best_details)
    if best_null >= best_score:
        best_text = ""
        details["predicted_no_answer"] = True
    return Prediction(best_text, details)


def predict_example(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    if ctx.model_type == "extractive_qa":
        return predict_extractive_qa(ctx, example)
    if ctx.model_type == "zero_shot_image":
        return predict_zero_shot_image(ctx, example)
    if ctx.model_type == "sentence_encoder":
        return predict_embedding_label(ctx, example)
    if ctx.model_type == "masked_lm":
        return predict_masked_choice(ctx, example) if ctx.task == "multiple_choice" else predict_masked_label(ctx, example)
    if ctx.model_type == "token_classifier":
        return predict_token_entities(ctx, example)
    if ctx.model_type == "sequence_classifier":
        return predict_sequence_classifier(ctx, example)
    if ctx.args.evaluation_protocol in TAGGED_PROTOCOLS:
        return predict_tagged_answer(ctx, example)
    if ctx.task == "multiple_choice":
        return predict_multiple_choice(ctx, example)
    if ctx.args.evaluation_protocol == "label_logprob_accuracy":
        return predict_label_likelihood(ctx, example)
    return predict_generation(ctx, example)


def predict_zero_shot_image(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels = candidate_label_values(ctx)
    if not labels:
        raise ValueError("Zero-shot image classification requires at least one candidate label.")
    prompts = [image_label_prompt(ctx, label) for label in labels]
    inputs = ctx.io(
        text=prompts,
        images=make_image(ctx, example),
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = ctx.model(**move_batch(ctx, dict(inputs)))
    logits = getattr(outputs, "logits_per_image", None)
    if logits is None:
        raise TypeError(f"Model {ctx.args.model!r} does not expose logits_per_image.")
    scores = logits[0].detach().to(torch.float32).cpu().tolist()
    label_id = int(torch.argmax(logits[0]).item())
    return Prediction(labels[label_id], {"label_id": label_id, "label_scores": dict(zip(labels, scores))})


def image_label_prompt(ctx: EvalContext, label: str) -> str:
    dataset = ctx.args.dataset.lower()
    if "eurosat" in dataset:
        return f"a satellite image of {label}"
    return f"a photo of {label}"



def predict_sequence_classifier(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    if ctx.args.evaluation_protocol == "multilabel_classifier_set_f1":
        return predict_multilabel_sequence_classifier(ctx, example)
    if ctx.args.evaluation_protocol == "zero_shot_nli_accuracy":
        return predict_zero_shot_sequence_classifier(ctx, example)
    if ctx.args.evaluation_protocol == "zero_shot_nli_multilabel":
        return predict_zero_shot_multilabel_classifier(ctx, example)
    if ctx.args.evaluation_protocol != "classifier_label_accuracy":
        raise ValueError(f"Unsupported sequence-classifier protocol: {ctx.args.evaluation_protocol}")
    prompt = build_generation_prompt(ctx, example)
    inputs = ctx.io(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens(ctx)).to(ctx.device)
    with torch.no_grad():
        logits = ctx.model(**inputs).logits[0]
    label_id = int(torch.argmax(logits).item())
    label = classifier_label(ctx, label_id)
    return Prediction(label, {"label_id": label_id, "logits": logits.detach().cpu().tolist()})


def predict_multilabel_sequence_classifier(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    prompt = build_generation_prompt(ctx, example)
    inputs = ctx.io(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens(ctx)).to(ctx.device)
    with torch.no_grad():
        logits = ctx.model(**inputs).logits[0]
    selected = [idx for idx, score in enumerate(torch.sigmoid(logits)) if float(score) >= 0.5]
    if not selected:
        selected = [int(torch.argmax(logits).item())]
    candidates = candidate_label_values(ctx)
    labels = []
    for label_id in selected:
        raw = classifier_config_labels(ctx).get(label_id, str(label_id))
        labels.append(closest_candidate_label(raw, candidates) or raw)
    return Prediction(json.dumps(labels, ensure_ascii=False), {"label_ids": selected, "logits": logits.detach().cpu().tolist()})


def should_use_zero_shot_sequence_classifier(ctx: EvalContext) -> bool:
    return bool(candidate_label_values(ctx) and entailment_label_id(ctx) is not None)


def predict_zero_shot_multilabel_classifier(
    ctx: EvalContext, example: dict[str, Any]
) -> Prediction:
    text = format_prompt_value(example_value(example, ctx.args.question_column, ""))
    labels = candidate_label_values(ctx)
    scores = {label: zero_shot_entailment_score(ctx, text, label) for label in labels}
    selected = [label for label, score in scores.items() if score >= 0.5]
    if not selected and scores:
        selected = [max(scores, key=scores.get)]
    return Prediction(
        json.dumps(selected, ensure_ascii=False), {"zero_shot_scores": scores}
    )


def predict_zero_shot_sequence_classifier(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    text = format_prompt_value(example_value(example, ctx.args.question_column, ""))
    labels = candidate_label_values(ctx)
    scores = {label: zero_shot_entailment_score(ctx, text, label) for label in labels}
    prediction = max(scores, key=scores.get)
    return Prediction(prediction, {"zero_shot_scores": scores})


def zero_shot_entailment_score(ctx: EvalContext, text: str, label: str) -> float:
    hypothesis = zero_shot_hypothesis(ctx, label)
    inputs = ctx.io(text, hypothesis, return_tensors="pt", truncation=True, max_length=max_input_tokens(ctx)).to(ctx.device)
    with torch.no_grad():
        logits = ctx.model(**inputs).logits[0]
    label_id = entailment_label_id(ctx)
    if label_id is None or label_id >= logits.numel():
        return float(torch.max(logits).item())
    return float(torch.softmax(logits, dim=-1)[label_id].item())


def zero_shot_hypothesis(ctx: EvalContext, label: str) -> str:
    if "medical_abstracts" in ctx.args.dataset.lower():
        return f"This medical abstract is about {label}."
    if "health_fact" in ctx.args.dataset.lower():
        return f"The public health claim is {label}."
    return f"This text is about {label}."


def candidate_label_values(ctx: EvalContext) -> list[str]:
    values = []
    for raw, label in getattr(ctx.args, "_label_map", {}).items():
        text = str(label).strip()
        if not text or text.lower().startswith("missing") or str(raw).strip() == "-1":
            continue
        if text not in values:
            values.append(text)
    return values


def entailment_label_id(ctx: EvalContext) -> int | None:
    for label_id, label in classifier_config_labels(ctx).items():
        text = str(label).lower()
        if "entail" in text and not text.startswith("not_") and "not entail" not in text:
            return label_id
    return None


def classifier_config_labels(ctx: EvalContext) -> dict[int, str]:
    labels = getattr(ctx.model.config, "id2label", {}) or {}
    converted = {}
    for key, value in labels.items():
        try:
            converted[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return converted


def classifier_label(ctx: EvalContext, label_id: int) -> str:
    raw_label = str(getattr(ctx.model.config, "id2label", {}).get(label_id, label_id))
    return closest_candidate_label(raw_label, label_values(ctx.args)) or raw_label


def closest_candidate_label(raw_label: str, candidates: list[str]) -> str:
    normalized_raw = normalize_label_text(raw_label)
    for candidate in candidates:
        if normalize_label_text(candidate) == normalized_raw:
            return candidate
    if normalized_raw in {"science", "tech", "technology", "scitech", "sci tech"}:
        return find_candidate(candidates, {"scitech", "sci tech", "science technology"})
    if normalized_raw in {"world news", "worldpost", "the worldpost", "international"}:
        return find_candidate(candidates, {"world"})
    return ""


def find_candidate(candidates: list[str], aliases: set[str]) -> str:
    for candidate in candidates:
        if normalize_label_text(candidate) in aliases:
            return candidate
    return ""


def normalize_label_text(value: str) -> str:
    text = str(value).lower().replace("&", " and ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def should_map_classifier_label(ctx: EvalContext) -> bool:
    label_map = getattr(ctx.args, "_label_map", {})
    if not label_map:
        return False
    num_labels = int(getattr(ctx.model.config, "num_labels", 0) or 0)
    usable_labels = [key for key in label_map if str(key).strip() != "-1"]
    return not num_labels or len(usable_labels) <= num_labels


def predict_multiple_choice(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels, continuations = option_continuations(ctx, example)
    scores, lengths = score_options(ctx, example, continuations)
    score_by_label = dict(zip(labels, scores))
    prediction = max(score_by_label, key=score_by_label.get)
    return Prediction(
        prediction,
        {
            "scores": score_by_label,
            "continuation_token_lengths": dict(zip(labels, lengths)),
        },
    )


def predict_label_likelihood(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    labels = candidate_label_values(ctx)
    if not labels:
        raise ValueError("label_logprob_accuracy requires at least one candidate label")
    continuations = [f" {label}" for label in labels]
    scores, lengths = score_label_options(ctx, example, continuations)
    score_by_label = dict(zip(labels, scores))
    prediction = max(score_by_label, key=score_by_label.get)
    return Prediction(
        prediction,
        {
            "label_scores": score_by_label,
            "continuation_token_lengths": dict(zip(labels, lengths)),
            "score_normalization": (
                "mean_token_logprob" if ctx.args.normalize_by_length else "sum_token_logprob"
            ),
        },
    )


def score_label_options(
    ctx: EvalContext,
    example: dict[str, Any],
    continuations: list[str],
) -> tuple[list[float], list[int]]:
    prompt = f"{build_prompt(ctx, example).rstrip()}\n\nAnswer:"
    if ctx.model_type == "seq2seq_lm":
        return seq2seq_logprobs(ctx, prompt, continuations)
    prompt = render_text_generation_prompt(ctx, prompt)
    return causal_logprobs(ctx, prompt, continuations)


def score_options(ctx: EvalContext, example: dict[str, Any], continuations: list[str]) -> tuple[list[float], list[int]]:
    prompt = build_prompt(ctx, example)
    if ctx.model_type == "paligemma":
        return paligemma_logprobs(ctx, example, prompt, continuations)
    if ctx.model_type == "seq2seq_lm":
        return seq2seq_logprobs(ctx, prompt, continuations)
    return causal_logprobs(ctx, prompt, continuations)


def predict_tagged_answer(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    prompt = build_tagged_answer_prompt(ctx, example)
    raw_generation = generate_response(ctx, example, prompt)
    labels = tagged_candidate_values(ctx, example)
    expect_set = resolved_evaluation_method(ctx.args) == "set_f1"
    answer, status = extract_tagged_answer(
        raw_generation, labels=labels, expect_set=expect_set
    )
    return Prediction(
        answer,
        {
            "raw_generation": raw_generation,
            "answer_parse_status": status,
        },
    )


def build_tagged_answer_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    prompt = build_prompt(ctx, example).rstrip()
    labels = tagged_candidate_values(ctx, example)
    constraint = tagged_label_constraint(ctx, labels)
    return (
        "Task and response contract:\n"
        f"{constraint}"
        "Do not explain your answer. Return exactly one XML element named answer:\n"
        "<answer>...</answer>\n"
        "Replace ... with the final answer.\n\n"
        f"Input:\n{prompt}\n\n"
        "Output only: <answer>...</answer>"
    )


def tagged_candidate_values(
    ctx: EvalContext, example: dict[str, Any]
) -> list[str]:
    if ctx.task == "multiple_choice":
        labels, _ = get_choices(ctx, example)
        return labels
    return candidate_label_values(ctx)


def tagged_label_constraint(ctx: EvalContext, labels: list[str]) -> str:
    if not labels:
        return "Answer the question concisely.\n"
    rendered = ", ".join(json.dumps(label, ensure_ascii=False) for label in labels)
    if resolved_evaluation_method(ctx.args) == "set_f1":
        return (
            f"Choose every applicable answer from: {rendered}. "
            "Return them as a JSON array inside the answer element.\n"
        )
    return f"Choose exactly one of these answers: {rendered}.\n"


def predict_generation(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    value = generate_response(ctx, example, build_prompt(ctx, example))
    return Prediction(value, {})


def generate_response(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    if ctx.model_type == "gguf":
        return generate_with_gguf(ctx, prompt)
    if ctx.model_type == "paligemma":
        return generate_with_paligemma(ctx, example, prompt)
    if ctx.model_type == "vlm_chat":
        return generate_with_vlm_chat(ctx, example, prompt)
    if ctx.model_type == "vlm_processor":
        return generate_with_vlm_processor(ctx, example, prompt)
    return generate_with_text_model(ctx, prompt)


def generate_with_gguf(ctx: EvalContext, prompt: str) -> str:
    return ctx.model.generate(prompt, ctx.args.max_new_tokens)


def generate_with_paligemma(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    image = make_image(ctx, example)
    inputs = ctx.io(text=prompt, images=image, return_tensors="pt")
    inputs = move_batch(ctx, inputs)
    generated = run_generation(ctx, inputs)
    return ctx.io.tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_with_text_model(ctx: EvalContext, prompt: str) -> str:
    prompt = render_text_generation_prompt(ctx, prompt)
    inputs = ctx.io(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens(ctx, reserve=ctx.args.max_new_tokens),
    ).to(ctx.device)
    generated = run_generation(ctx, inputs)
    return ctx.io.decode(generated, skip_special_tokens=True).strip()


def render_text_generation_prompt(ctx: EvalContext, prompt: str) -> str:
    chat_template = getattr(ctx.io, "chat_template", None)
    if not chat_template:
        if getattr(ctx.args, "disable_thinking", False):
            raise ValueError("--disable-thinking requires a tokenizer chat template")
        return prompt

    messages = [{"role": "user", "content": prompt}]
    options = {"tokenize": False, "add_generation_prompt": True}
    if getattr(ctx.args, "disable_thinking", False):
        options["enable_thinking"] = False
    return ctx.io.apply_chat_template(messages, **options)


def generate_with_vlm_chat(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    pixel_values = make_pixel_values(ctx, example)
    generation_config = {"max_new_tokens": ctx.args.max_new_tokens, "do_sample": False}
    response = ctx.model.chat(ctx.io, pixel_values, prompt, generation_config)
    return str(response).strip()




def generate_with_vlm_processor(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    image = make_processor_image(ctx, example)
    message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    if hasattr(ctx.io, "apply_chat_template"):
        text = ctx.io.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = ctx.io(text=[text], images=[image], return_tensors="pt")
    inputs = move_batch(ctx, inputs)
    generated = run_generation(ctx, inputs)
    tokenizer = getattr(ctx.io, "tokenizer", ctx.io)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

def run_generation(ctx: EvalContext, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    input_len = int(inputs["input_ids"].shape[1])
    generation_args = {
        **inputs,
        "max_new_tokens": ctx.args.max_new_tokens,
        "do_sample": False,
    }
    with torch.no_grad():
        try:
            output = ctx.model.generate(**generation_args)
        except TypeError as exc:
            if "tuple" not in str(exc):
                raise
            output = ctx.model.generate(**generation_args, use_cache=False)
    if ctx.model_type == "seq2seq_lm":
        return output[0]
    return output[0, input_len:]


def move_batch(ctx: EvalContext, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key == "pixel_values":
            moved[key] = value.to(device=ctx.device, dtype=ctx.dtype)
        else:
            moved[key] = value.to(device=ctx.device)
    return moved


def causal_logprobs(ctx: EvalContext, prompt: str, continuations: list[str]) -> tuple[list[float], list[int]]:
    tokenizer = ctx.io
    prefix_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    continuation_ids = tokenize_continuations(tokenizer, continuations)
    input_ids, attention_mask = build_padded_rows(ctx, prefix_ids, continuation_ids, tokenizer)
    with torch.no_grad():
        logits = ctx.model(input_ids=input_ids, attention_mask=attention_mask).logits
    return score_continuations(ctx, logits, len(prefix_ids), continuation_ids)


def paligemma_logprobs(
    ctx: EvalContext,
    example: dict[str, Any],
    prompt: str,
    continuations: list[str],
) -> tuple[list[float], list[int]]:
    tokenizer = ctx.io.tokenizer
    prompt_inputs = ctx.io(text=prompt, images=make_image(ctx, example), return_tensors="pt")
    prefix_ids = prompt_inputs["input_ids"][0].tolist()
    continuation_ids = tokenize_continuations(tokenizer, continuations)
    input_ids, attention_mask = build_padded_rows(ctx, prefix_ids, continuation_ids, tokenizer)
    pixel_values = expand_pixel_values(ctx, prompt_inputs, len(continuations))
    with torch.no_grad():
        logits = ctx.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values).logits
    return score_continuations(ctx, logits, len(prefix_ids), continuation_ids)


def seq2seq_logprobs(ctx: EvalContext, prompt: str, continuations: list[str]) -> tuple[list[float], list[int]]:
    encoded = ctx.io(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens(ctx)).to(ctx.device)
    scores, lengths = [], []
    for text in continuations:
        labels = ctx.io(text.strip(), add_special_tokens=True, return_tensors="pt").input_ids.to(ctx.device)
        with torch.no_grad():
            logits = ctx.model(**encoded, labels=labels).logits
        score = sum_token_logprobs(ctx, logits[0], labels[0])
        scores.append(normalize_score(ctx, score, len(labels[0])))
        lengths.append(int(len(labels[0])))
    return scores, lengths


def tokenize_continuations(tokenizer: Any, continuations: list[str]) -> list[list[int]]:
    return [tokenizer(text, add_special_tokens=False).input_ids for text in continuations]


def build_padded_rows(
    ctx: EvalContext,
    prefix_ids: list[int],
    continuation_ids: list[list[int]],
    tokenizer: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(prefix_ids) + len(ids) for ids in continuation_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows, masks = [], []
    for ids in continuation_ids:
        row, mask = pad_row(prefix_ids, ids, pad_id, max_len)
        rows.append(row)
        masks.append(mask)
    return (
        torch.tensor(rows, dtype=torch.long, device=ctx.device),
        torch.tensor(masks, dtype=torch.long, device=ctx.device),
    )


def pad_row(prefix_ids: list[int], continuation: list[int], pad_id: int, max_len: int) -> tuple[list[int], list[int]]:
    row = prefix_ids + continuation
    padding = [pad_id] * (max_len - len(row))
    return row + padding, [1] * len(row) + [0] * len(padding)


def expand_pixel_values(ctx: EvalContext, prompt_inputs: dict[str, torch.Tensor], count: int) -> torch.Tensor:
    pixel_values = prompt_inputs["pixel_values"].to(device=ctx.device, dtype=ctx.dtype)
    return pixel_values.expand(count, -1, -1, -1).contiguous()


def score_continuations(
    ctx: EvalContext,
    logits: torch.Tensor,
    prompt_len: int,
    continuation_ids: list[list[int]],
) -> tuple[list[float], list[int]]:
    scores, lengths = [], []
    for row_idx, ids in enumerate(continuation_ids):
        score = score_one_continuation(ctx, logits[row_idx], prompt_len, ids)
        scores.append(score)
        lengths.append(len(ids))
    return scores, lengths


def score_one_continuation(ctx: EvalContext, row_logits: torch.Tensor, prompt_len: int, ids: list[int]) -> float:
    if not ids:
        return -math.inf
    logits = row_logits[prompt_len - 1 : prompt_len + len(ids) - 1]
    token_ids = torch.tensor(ids, dtype=torch.long, device=ctx.device)
    score = sum_token_logprobs(ctx, logits, token_ids)
    return normalize_score(ctx, score, len(ids))


def sum_token_logprobs(ctx: EvalContext, logits: torch.Tensor, token_ids: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits[: len(token_ids)], dim=-1)
    return log_probs.gather(1, token_ids.unsqueeze(1)).sum().item()


def normalize_score(ctx: EvalContext, score: float, length: int) -> float:
    if ctx.args.normalize_by_length and length:
        return score / length
    return score


def prediction_row(
    ctx: EvalContext,
    index: int,
    example: dict[str, Any],
    prediction: Prediction,
    example_score: float | None,
) -> dict[str, Any]:
    raw_answer = example_value(example, ctx.args.answer_column)
    answer = evaluation_answer(ctx, example)
    mapped_answer = target_answer(ctx, answer)
    return {
        "index": index,
        "source_index": (
            ctx.args._selected_indices[index]
            if getattr(ctx.args, "_selected_indices", None) is not None
            else index
        ),
        "id": example_value(example, ctx.args.id_column, index),
        "question": example_value(example, ctx.args.question_column),
        "answer": raw_answer if has_gold_answer(raw_answer) else None,
        "target_answer": mapped_answer if has_gold_answer(mapped_answer) else None,
        "prediction": prediction.value,
        "example_score": example_score,
        "correct": None if example_score is None else example_score == 1.0,
        **prediction.extra,
    }


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluation_metric(args: argparse.Namespace) -> str | None:
    if args.evaluation_protocol == "unsupported":
        return None
    return resolved_evaluation_method(args)


def build_summary(ctx: EvalContext, scoreboard: Scoreboard, predictions_path: Path) -> dict[str, Any]:
    args = ctx.args
    metric = evaluation_metric(args)
    score = scoreboard.score(metric)
    summary = {
        "model": args.model,
        "model_source": model_source(args) if model_source(args) != args.model else None,
        "model_type": ctx.model_type,
        "dataset": args.dataset,
        "dataset_source": dataset_source(args) if dataset_source(args) != args.dataset else None,
        "subset": args.subset or None,
        "split": args.split,
        "task": ctx.task,
        "evaluation_protocol": args.evaluation_protocol,
        "metric": metric,
        "score": score,
        "total": scoreboard.total,
        "labeled_total": scoreboard.labeled_total,
        "correct": scoreboard.correct,
        "accuracy": score if metric == "accuracy" else None,
        "exact_match": score if metric == "exact_match" else None,
        "macro_f1": score if metric == "macro_f1" else None,
        "qa_f1": score if metric == "qa_f1" else None,
        "numeric_match": score if metric == "numeric_match" else None,
        "rouge_l": score if metric == "rouge_l" else None,
        "set_f1": score if metric == "set_f1" else None,
        "tag_parse_failures": (
            scoreboard.tag_parse_failures
            if args.evaluation_protocol in TAGGED_PROTOCOLS
            else None
        ),
        "thinking_disabled": bool(getattr(args, "disable_thinking", False)),
        "scoring": args.scoring if ctx.task == "multiple_choice" else None,
        "normalize_by_length": (
            args.normalize_by_length
            if ctx.task == "multiple_choice" or args.evaluation_protocol == "label_logprob_accuracy"
            else None
        ),
        "device": str(ctx.device),
        "predictions_path": str(predictions_path),
        "label_values": label_values(args),
        "sampling": {
            "source_total": getattr(args, "_source_total", scoreboard.total),
            "limit": args.limit,
            "seed": args.seed,
            "sampled": getattr(args, "_selected_indices", None) is not None,
        },
    }
    return summary


def label_values(args: argparse.Namespace) -> list[str]:
    values = []
    for raw, label in getattr(args, "_label_map", {}).items():
        text = str(label).strip()
        if not text or str(raw).strip() == "-1" or text.lower().startswith("missing"):
            continue
        if text not in values:
            values.append(text)
    return values


def validate_label_score_permission(args: argparse.Namespace) -> None:
    """Retained for callers from older releases; label scores no longer need opt-in."""


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validate_label_score_permission(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args._label_map = load_label_map(args)
    args._dataset_kwargs = load_dataset_kwargs(args)
    args._choices_columns = json.loads(args.choices_columns) if args.choices_columns else []
    progress(f"model source check start: {args.model}")
    prepare_model_source(args)

    progress(f"model source check complete: {model_source(args)}")

    progress(f"dataset load start: {args.dataset} split={args.split}")
    dataset = load_eval_dataset(args)
    progress(f"dataset load complete: {len(dataset)} examples")
    ctx = build_context(args, dataset[0])
    validate_label_score_permission(args)
    scoreboard = Scoreboard()
    predictions_path = output_dir / args.predictions_file

    if ctx.model_type in UNSUPPORTED_MODEL_TYPES or args.evaluation_protocol == "unsupported":
        return write_unsupported_summary(ctx, scoreboard, predictions_path, output_dir)

    progress(f"inference start: {len(dataset)} examples protocol={args.evaluation_protocol}")
    with predictions_path.open("w", encoding="utf-8") as out:
        for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {args.split}")):
            prediction = predict_example(ctx, example)
            answer = evaluation_answer(ctx, example)
            example_score = score_prediction(ctx, prediction.value, answer)
            scoreboard.add(
                example_score, normalize_answer(prediction.value), primary_target(ctx, answer)
            )
            scoreboard.add_parse_status(prediction.extra.get("answer_parse_status"))
            write_jsonl_row(
                out, prediction_row(ctx, idx, example, prediction, example_score)
            )
            report_example_progress(idx + 1, len(dataset))

    summary = build_summary(ctx, scoreboard, predictions_path)
    (output_dir / args.summary_file).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    progress(f"evaluation complete: {summary.get('metric')}={summary.get('score')} output={output_dir}")
    return summary


def write_unsupported_summary(
    ctx: EvalContext,
    scoreboard: Scoreboard,
    predictions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    predictions_path.write_text("", encoding="utf-8")
    summary = build_summary(ctx, scoreboard, predictions_path)
    summary["status"] = "unsupported"
    summary["unsupported_reason"] = unsupported_reason(ctx)
    (output_dir / ctx.args.summary_file).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def unsupported_reason(ctx: EvalContext) -> str:
    if ctx.model_type == "unsupported_model":
        return (
            "This checkpoint format or task adapter is not supported by the generic evaluator. "
            "It likely needs a dedicated loader or dependency before direct evaluation."
        )
    return (
        "This image-classification checkpoint is not loadable by the generic Transformers/VLM "
        "adapters. It likely needs a model-family-specific timm/torchgeo linear-probe or "
        "fine-tuning adapter before direct evaluation."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. xai-org/RealworldQA.")
    parser.add_argument("--dataset-source", default="", help="Equivalent data-only source; canonical --dataset remains in results.")
    parser.add_argument("--dataset-kwargs", default="", help="Optional JSON object passed to the dataset loader.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument(
        "--model-source",
        default="",
        help="Equivalent checkpoint or local weight source; the canonical --model id remains in results.",
    )
    parser.add_argument("--subset", default="", help="Optional dataset config/subset name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow Hugging Face dataset loading scripts to run.")
    parser.add_argument(
        "--task",
        choices=(
            "auto", "generation", "classification", "multilabel_classification",
            "multiple_choice", "qa", "numeric_qa", "summarization", "token_classification",
            "relation_extraction", "image_classification",
        ),
        default="auto",
    )
    parser.add_argument("--model-type", choices=("auto", "paligemma", "vlm_chat", "vlm_processor", "voxtral", "zero_shot_image", "gguf", "causal_lm", "seq2seq_lm", "sequence_classifier", "sentence_encoder", "masked_lm", "token_classifier", "extractive_qa", "unsupported_image", "unsupported_model"), default="auto")
    parser.add_argument(
        "--evaluation-protocol",
        choices=(
            "auto", "multiple_choice_accuracy", "label_generation_accuracy", "label_logprob_accuracy",
            "tagged_label_generation_accuracy", "tagged_answer_exact_match",
            "tagged_multiple_choice_accuracy", "tagged_set_generation",
            "multilabel_classifier_set_f1",
            "zero_shot_nli_multilabel", "extractive_qa",
            "zero_shot_nli_accuracy", "classifier_label_accuracy", "unsupported",
            "embedding_label_similarity", "masked_label_likelihood",
            "masked_choice_likelihood", "token_entity_set_f1", "zero_shot_image_accuracy",
        ),
        default="auto",
        help="Explicit scoring contract; unsupported protocols produce no measured score.",
    )
    parser.add_argument(
        "--evaluation-method",
        choices=("auto", "accuracy", "macro_f1", "exact_match", "qa_f1", "numeric_match", "rouge_l", "set_f1"),
        default="auto",
    )
    parser.add_argument(
        "--allow-label-scores",
        action="store_true",
        help=(
            "Compatibility flag accepted by the low-level evaluator; protocol selection happens "
            "before this command is launched."
        ),
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Use a tokenizer chat template with enable_thinking=False for constrained generation.",
    )
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--context-column", default="context")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--choices-column", default="choices")
    parser.add_argument("--choices-columns", default="", help="Optional JSON list of separate choice columns.")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--prompt-template", default="", help="Optional Python format string using dataset columns, e.g. {question} or {passage}.")
    parser.add_argument("--label-map", default="", help="Optional JSON object mapping raw labels to target answer text.")
    parser.add_argument("--label-threshold", type=float, default=None, help="Optional threshold converting numeric targets to label-map keys 0 and 1.")
    parser.add_argument("--answer-regex", default="", help="Optional regex extracting the gold answer; the first capture group is used when present.")
    parser.add_argument("--label-map-file", default="", help="Optional path to a JSON label map file.")
    parser.add_argument("--scoring", choices=("label", "answer_text"), default="label")
    parser.add_argument("--token-stride", type=int, default=64, help="Overlap between token-classification windows.")
    parser.add_argument("--max-answer-tokens", type=int, default=64, help="Maximum extracted QA answer span length.")
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=0, help="Optional text-token cap. By default the model/tokenizer context limit is used.")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gguf-file", default="", help="Optional GGUF filename within a Hugging Face repository.")
    parser.add_argument("--gguf-chat-format", default="", help="Optional llama.cpp chat format override.")
    parser.add_argument("--gguf-context-size", type=int, default=4096)
    parser.add_argument("--gguf-gpu-layers", type=int, default=-1, help="GGUF layers to offload; -1 requests all layers.")
    parser.add_argument("--sanitized-model-dir", default="", help="Optional local directory for sanitized model snapshots.")
    parser.add_argument("--no-sanitize-model-config", dest="sanitize_model_config", action="store_false")
    parser.set_defaults(sanitize_model_config=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0, help="Maximum examples to evaluate; selected randomly when the split is larger.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible random example selection.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-file", default="summary.json")
    parser.add_argument("--predictions-file", default="predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    summary = evaluate(parse_args())
    print(json.dumps(summary, indent=2))
    if summary.get("unsupported_reason"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

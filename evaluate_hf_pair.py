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
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    PaliGemmaForConditionalGeneration,
)
from transformers.modeling_utils import PreTrainedModel


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def patch_transformers_compat() -> None:
    """Handle remote model code written against slightly different Transformers APIs."""
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        return

    def get_all_tied_weights_keys(self: PreTrainedModel) -> dict[str, None]:
        keys = getattr(self, "_all_tied_weights_keys", None)
        if keys is None:
            keys = getattr(self, "_tied_weights_keys", None) or []
        if hasattr(keys, "keys"):
            return keys
        return {key: None for key in keys}

    def set_all_tied_weights_keys(self: PreTrainedModel, keys: Any) -> None:
        self._all_tied_weights_keys = keys

    PreTrainedModel.all_tied_weights_keys = property(get_all_tied_weights_keys, set_all_tied_weights_keys)


patch_transformers_compat()


@dataclass
class EvalContext:
    args: argparse.Namespace
    model_type: str
    task: str
    device: torch.device
    dtype: torch.dtype
    model: torch.nn.Module
    io: Any


@dataclass
class Prediction:
    value: str
    extra: dict[str, Any]


@dataclass
class Scoreboard:
    total: int = 0
    labeled_total: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.labeled_total if self.labeled_total else None

    def add(self, is_correct: bool | None) -> None:
        self.total += 1
        if is_correct is None:
            return
        self.labeled_total += 1
        self.correct += int(is_correct)


def model_source(args: argparse.Namespace) -> str:
    return getattr(args, "model_source", args.model)


def prepare_model_source(args: argparse.Namespace) -> None:
    args.model_source = args.model
    if not args.sanitize_model_config:
        return
    if Path(args.model).exists():
        args.model_source = str(sanitize_model_dir(Path(args.model), args))
        return
    try:
        AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        return
    except Exception as exc:
        if not is_sanitizable_config_error(exc):
            raise
    snapshot_dir = Path(snapshot_download(args.model))
    args.model_source = str(sanitize_model_dir(snapshot_dir, args))


def is_sanitizable_config_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return "use_cache" in text and ("NoneType" in text or "got None" in text)


def sanitize_model_dir(source_dir: Path, args: argparse.Namespace) -> Path:
    config_path = source_dir / "config.json"
    preprocessor_path = source_dir / "preprocessor_config.json"

    sanitized_config, config_changed = load_sanitized_json(config_path, sanitize_config_value)
    sanitized_preprocessor, preprocessor_changed = load_sanitized_json(
        preprocessor_path,
        sanitize_preprocessor_config,
    )
    if not config_changed and not preprocessor_changed:
        return source_dir

    target_dir = Path(args.sanitized_model_dir) if args.sanitized_model_dir else default_sanitized_model_dir(args)
    target_dir.mkdir(parents=True, exist_ok=True)
    link_snapshot_files(source_dir, target_dir, skip_changed={
        "config.json": config_changed,
        "preprocessor_config.json": preprocessor_changed,
    })
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
    return answer is not None and str(answer).strip() != ""


def load_label_map(args: argparse.Namespace) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if args.label_map_file:
        mapping.update(json.loads(Path(args.label_map_file).read_text(encoding="utf-8")))
    if args.label_map:
        mapping.update(json.loads(args.label_map))
    return {str(key): str(value) for key, value in mapping.items()}


def target_answer(ctx: EvalContext, answer: Any) -> Any:
    if not has_gold_answer(answer):
        return answer
    mapping = getattr(ctx.args, "_label_map", {})
    return mapping.get(str(answer), answer)


def is_correct_prediction(ctx: EvalContext, prediction: str, answer: Any) -> bool | None:
    answer = target_answer(ctx, answer)
    if not has_gold_answer(answer):
        return None
    return normalize_answer(prediction) == normalize_answer(answer)


def infer_model_type(model_name: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    if "paligemma" in model_name.lower():
        return "paligemma"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if is_vlm_chat_config(config):
        return "vlm_chat"
    if is_vlm_processor_config(config):
        return "vlm_processor"
    return "seq2seq_lm" if getattr(config, "is_encoder_decoder", False) else "causal_lm"


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
    has_choices = args.choices_column in example and example[args.choices_column] is not None
    return "multiple_choice" if has_choices else "generation"


def choose_device(device_name: str) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cpu" else torch.bfloat16


def load_eval_dataset(args: argparse.Namespace):
    dataset = load_dataset(args.dataset, args.subset or None, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    return dataset


def build_context(args: argparse.Namespace, first_example: dict[str, Any]) -> EvalContext:
    model_type = infer_model_type(model_source(args), args.model_type)
    task = infer_task(first_example, args)
    device = choose_device(args.device)
    dtype = choose_dtype(device)
    model, io = load_model_and_io(args, model_type, device, dtype)
    return EvalContext(args, model_type, task, device, dtype, model, io)


def load_model_and_io(args: argparse.Namespace, model_type: str, device: torch.device, dtype: torch.dtype):
    if model_type == "paligemma":
        return load_paligemma(args, device, dtype)
    if model_type == "vlm_chat":
        return load_vlm_chat(args, device, dtype)
    if model_type == "vlm_processor":
        return load_vlm_processor(args, device, dtype)
    return load_text_model(args, model_type, device, dtype)


def load_paligemma(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    processor = AutoProcessor.from_pretrained(model_source(args), trust_remote_code=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_source(args),
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    return model.eval(), processor


def load_text_model(args: argparse.Namespace, model_type: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_source(args), trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_class = AutoModelForSeq2SeqLM if model_type == "seq2seq_lm" else AutoModelForCausalLM
    model_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if model_type == "causal_lm":
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cpu":
        model_kwargs["torch_dtype"] = torch.float32

    model = model_class.from_pretrained(model_source(args), **model_kwargs).to(device)
    return model.eval(), tokenizer


def load_vlm_chat(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_source(args), trust_remote_code=True)
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
    model = AutoModelForImageTextToText.from_pretrained(model_source(args), **model_kwargs).to(device)
    return model.eval(), processor

def make_image(ctx: EvalContext, example: dict[str, Any]) -> Image.Image:
    image = example.get(ctx.args.image_column)
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


def get_choices(ctx: EvalContext, example: dict[str, Any]) -> tuple[list[str], list[str]]:
    choices = example[ctx.args.choices_column]
    if isinstance(choices, dict):
        labels = [str(label) for label in choices.get("label", [])]
        texts = [str(text) for text in choices.get("text", [])]
        return labels, texts
    if isinstance(choices, list):
        labels = [chr(ord("A") + idx) for idx in range(len(choices))]
        texts = [str(choice) for choice in choices]
        return labels, texts
    raise TypeError(f"Unsupported choices format in column {ctx.args.choices_column!r}: {type(choices).__name__}")


def build_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    if ctx.task == "multiple_choice":
        prompt = build_multiple_choice_prompt(ctx, example)
    else:
        prompt = build_generation_prompt(ctx, example)
    return add_image_prefix(ctx, prompt)


def build_generation_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    question = str(example[ctx.args.question_column])
    if ctx.args.prompt_template:
        values = {key: "" if value is None else str(value) for key, value in example.items()}
        values.setdefault("question", question)
        return ctx.args.prompt_template.format_map(values)
    return question


def build_multiple_choice_prompt(ctx: EvalContext, example: dict[str, Any]) -> str:
    labels, texts = get_choices(ctx, example)
    choice_lines = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return (
        "Choose the best answer.\n\n"
        f"Question: {example[ctx.args.question_column]}\n"
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


def predict_example(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    if ctx.task == "multiple_choice":
        return predict_multiple_choice(ctx, example)
    return predict_generation(ctx, example)


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


def score_options(ctx: EvalContext, example: dict[str, Any], continuations: list[str]) -> tuple[list[float], list[int]]:
    prompt = build_prompt(ctx, example)
    if ctx.model_type == "paligemma":
        return paligemma_logprobs(ctx, example, prompt, continuations)
    if ctx.model_type == "seq2seq_lm":
        return seq2seq_logprobs(ctx, prompt, continuations)
    return causal_logprobs(ctx, prompt, continuations)


def predict_generation(ctx: EvalContext, example: dict[str, Any]) -> Prediction:
    prompt = build_prompt(ctx, example)
    if ctx.model_type == "paligemma":
        value = generate_with_paligemma(ctx, example, prompt)
    elif ctx.model_type == "vlm_chat":
        value = generate_with_vlm_chat(ctx, example, prompt)
    elif ctx.model_type == "vlm_processor":
        value = generate_with_vlm_processor(ctx, example, prompt)
    else:
        value = generate_with_text_model(ctx, prompt)
    return Prediction(value, {})


def generate_with_paligemma(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    image = make_image(ctx, example)
    inputs = ctx.io(text=prompt, images=image, return_tensors="pt")
    inputs = move_batch(ctx, inputs)
    generated = run_generation(ctx, inputs)
    return ctx.io.tokenizer.decode(generated, skip_special_tokens=True).strip()


def generate_with_text_model(ctx: EvalContext, prompt: str) -> str:
    inputs = ctx.io(prompt, return_tensors="pt").to(ctx.device)
    generated = run_generation(ctx, inputs)
    return ctx.io.decode(generated, skip_special_tokens=True).strip()


def generate_with_vlm_chat(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    pixel_values = make_pixel_values(ctx, example)
    generation_config = {"max_new_tokens": ctx.args.max_new_tokens, "do_sample": False}
    response = ctx.model.chat(ctx.io, pixel_values, prompt, generation_config)
    return str(response).strip()




def generate_with_vlm_processor(ctx: EvalContext, example: dict[str, Any], prompt: str) -> str:
    image = make_image(ctx, example)
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
    with torch.no_grad():
        output = ctx.model.generate(**inputs, max_new_tokens=ctx.args.max_new_tokens, do_sample=False)
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
    encoded = ctx.io(prompt, return_tensors="pt").to(ctx.device)
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
    is_correct: bool | None,
) -> dict[str, Any]:
    answer = example.get(ctx.args.answer_column)
    mapped_answer = target_answer(ctx, answer)
    return {
        "index": index,
        "id": example.get(ctx.args.id_column, index),
        "question": example.get(ctx.args.question_column),
        "answer": answer if has_gold_answer(answer) else None,
        "target_answer": mapped_answer if has_gold_answer(mapped_answer) else None,
        "prediction": prediction.value,
        "correct": is_correct,
        **prediction.extra,
    }


def write_jsonl_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_summary(ctx: EvalContext, scoreboard: Scoreboard, predictions_path: Path) -> dict[str, Any]:
    args = ctx.args
    return {
        "model": args.model,
        "model_source": model_source(args) if model_source(args) != args.model else None,
        "model_type": ctx.model_type,
        "dataset": args.dataset,
        "subset": args.subset or None,
        "split": args.split,
        "task": ctx.task,
        "total": scoreboard.total,
        "labeled_total": scoreboard.labeled_total,
        "correct": scoreboard.correct,
        "accuracy": scoreboard.accuracy,
        "scoring": args.scoring if ctx.task == "multiple_choice" else None,
        "normalize_by_length": args.normalize_by_length if ctx.task == "multiple_choice" else None,
        "device": str(ctx.device),
        "predictions_path": str(predictions_path),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args._label_map = load_label_map(args)
    prepare_model_source(args)

    dataset = load_eval_dataset(args)
    ctx = build_context(args, dataset[0])
    scoreboard = Scoreboard()
    predictions_path = output_dir / args.predictions_file

    with predictions_path.open("w", encoding="utf-8") as out:
        for idx, example in enumerate(tqdm(dataset, desc=f"Evaluating {args.split}")):
            prediction = predict_example(ctx, example)
            is_correct = is_correct_prediction(ctx, prediction.value, example.get(args.answer_column))
            scoreboard.add(is_correct)
            write_jsonl_row(out, prediction_row(ctx, idx, example, prediction, is_correct))

    summary = build_summary(ctx, scoreboard, predictions_path)
    (output_dir / args.summary_file).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. xai-org/RealworldQA.")
    parser.add_argument("--model", required=True, help="Hugging Face model id.")
    parser.add_argument("--subset", default="", help="Optional dataset config/subset name.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--task", choices=("auto", "generation", "multiple_choice"), default="auto")
    parser.add_argument("--model-type", choices=("auto", "paligemma", "vlm_chat", "vlm_processor", "causal_lm", "seq2seq_lm"), default="auto")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--choices-column", default="choices")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--prompt-template", default="", help="Optional Python format string using dataset columns, e.g. {question} or {passage}.")
    parser.add_argument("--label-map", default="", help="Optional JSON object mapping raw labels to target answer text.")
    parser.add_argument("--label-map-file", default="", help="Optional path to a JSON label map file.")
    parser.add_argument("--scoring", choices=("label", "answer_text"), default="label")
    parser.add_argument("--normalize-by-length", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--sanitized-model-dir", default="", help="Optional local directory for sanitized model snapshots.")
    parser.add_argument("--no-sanitize-model-config", dest="sanitize_model_config", action="store_false")
    parser.set_defaults(sanitize_model_config=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-file", default="summary.json")
    parser.add_argument("--predictions-file", default="predictions.jsonl")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2))


if __name__ == "__main__":
    main()

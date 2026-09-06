"""Optional llama.cpp backend for text-generation GGUF checkpoints."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download, list_repo_files


GGUF_SUFFIX = ".gguf"
PROJECTOR_MARKERS = ("mmproj", "projector")
QUANTIZATION_PREFERENCE = (
    "q4_k_m",
    "q5_k_m",
    "q4_k_s",
    "q5_0",
    "q4_0",
    "q3_k_m",
    "q6_k",
    "q8_0",
    "f16",
    "bf16",
)
SHARD_PATTERN = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
CHAT_TEMPLATE_OVERHEAD_TOKENS = 64


def is_gguf_model_name(model: str) -> bool:
    name = model.lower()
    return name.endswith(GGUF_SUFFIX) or "gguf" in name


def resolve_gguf_path(model: str, requested_file: str = "") -> Path:
    requested = Path(requested_file).expanduser() if requested_file else None
    if requested and requested.is_file():
        return requested.resolve()

    source = Path(model).expanduser()
    if source.is_file():
        validate_gguf_filename(source.name)
        return source.resolve()
    if source.is_dir():
        return resolve_local_gguf(source, requested_file)
    return resolve_hub_gguf(model, requested_file)


def resolve_local_gguf(directory: Path, requested_file: str = "") -> Path:
    files = [path.name for path in directory.iterdir() if path.is_file()]
    filename = select_gguf_filename(files, requested_file)
    ensure_shards_exist(directory, filename, files)
    return (directory / filename).resolve()


def resolve_hub_gguf(repo_id: str, requested_file: str = "") -> Path:
    files = list_repo_files(repo_id)
    filename = select_gguf_filename(files, requested_file)
    downloaded = {
        shard: Path(hf_hub_download(repo_id, shard))
        for shard in matching_shards(filename, files)
    }
    return downloaded[filename]


def select_gguf_filename(files: list[str], requested_file: str = "") -> str:
    projectors = [name for name in files if is_projector(name)]
    if projectors:
        raise ValueError(
            "Multimodal GGUF repositories with a projector are not supported by the "
            "text-only GGUF adapter."
        )

    candidates = [name for name in files if is_model_gguf(name)]
    if requested_file:
        return resolve_requested_filename(candidates, requested_file)
    if not candidates:
        raise ValueError("No text-generation .gguf model file was found.")
    first_shards = [name for name in candidates if is_first_or_unsharded(name)]
    if not first_shards:
        raise ValueError("No first GGUF shard was found in the repository.")
    return min(first_shards, key=gguf_preference_key)


def resolve_requested_filename(candidates: list[str], requested_file: str) -> str:
    requested = requested_file.strip().lstrip("/")
    exact = [name for name in candidates if name == requested]
    if exact:
        return exact[0]
    basename = Path(requested).name
    matches = [name for name in candidates if Path(name).name == basename]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Requested GGUF file {requested_file!r} was not found uniquely.")


def is_model_gguf(filename: str) -> bool:
    return filename.lower().endswith(GGUF_SUFFIX) and not is_projector(filename)


def is_projector(filename: str) -> bool:
    name = Path(filename).name.lower()
    return name.endswith(GGUF_SUFFIX) and any(
        marker in name for marker in PROJECTOR_MARKERS
    )


def is_first_or_unsharded(filename: str) -> bool:
    match = SHARD_PATTERN.match(filename)
    return not match or match.group(2) == "00001"


def matching_shards(filename: str, files: list[str]) -> list[str]:
    match = SHARD_PATTERN.match(filename)
    if not match:
        return [filename]
    prefix = match.group(1)
    total = int(match.group(3))
    expected = [
        f"{prefix}-{index:05d}-of-{total:05d}.gguf"
        for index in range(1, total + 1)
    ]
    missing = [name for name in expected if name not in files]
    if missing:
        raise ValueError(f"Missing GGUF shards: {', '.join(missing)}")
    return expected


def ensure_shards_exist(directory: Path, filename: str, files: list[str]) -> None:
    expected = matching_shards(filename, files)
    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Missing local GGUF shards: {', '.join(missing)}")


def gguf_preference_key(filename: str) -> tuple[int, str]:
    name = filename.lower()
    rank = next(
        (
            index
            for index, quant in enumerate(QUANTIZATION_PREFERENCE)
            if quant in name
        ),
        len(QUANTIZATION_PREFERENCE),
    )
    return rank, name


def validate_gguf_filename(filename: str) -> None:
    if not is_model_gguf(filename):
        raise ValueError(
            f"Expected a text-generation .gguf file, received {filename!r}."
        )


class GgufBackend:
    def __init__(self, model: Any):
        self.model = model

    @classmethod
    def load(
        cls,
        model_path: Path,
        context_size: int,
        gpu_layers: int,
        seed: int,
        chat_format: str = "",
    ) -> "GgufBackend":
        llama_class = import_llama_class()
        kwargs: dict[str, Any] = {
            "model_path": str(model_path),
            "n_ctx": context_size,
            "n_gpu_layers": gpu_layers,
            "seed": seed,
            "verbose": False,
        }
        if chat_format:
            kwargs["chat_format"] = chat_format
        return cls(llama_class(**kwargs))

    def generate(self, prompt: str, max_tokens: int) -> str:
        prompt = fit_prompt_to_context(self.model, prompt, max_tokens)
        try:
            response = self.model.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            return chat_response_text(response)
        except (TypeError, ValueError) as exc:
            if not is_chat_template_error(exc):
                raise
        response = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            echo=False,
        )
        return completion_response_text(response)


def fit_prompt_to_context(model: Any, prompt: str, max_tokens: int) -> str:
    try:
        context_size = int(model.n_ctx())
    except (TypeError, ValueError):
        return prompt
    budget = max(1, context_size - max_tokens - CHAT_TEMPLATE_OVERHEAD_TOKENS)
    tokens = model.tokenize(prompt.encode("utf-8"), add_bos=False)
    if len(tokens) <= budget:
        return prompt
    head_size = max(1, budget // 4)
    shortened = tokens[:head_size] + tokens[-(budget - head_size):]
    return model.detokenize(shortened).decode("utf-8", errors="replace")


def import_llama_class() -> Any:
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "GGUF evaluation requires llama-cpp-python. Install the optional 'gguf' "
            "dependencies with CUDA support before running this model."
        ) from exc
    return Llama


def is_chat_template_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "chat format" in text or "chat template" in text


def chat_response_text(response: Any) -> str:
    try:
        return str(response["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            "llama.cpp returned an invalid chat-completion response."
        ) from exc


def completion_response_text(response: Any) -> str:
    try:
        return str(response["choices"][0]["text"] or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("llama.cpp returned an invalid completion response.") from exc

#!/usr/bin/env python3
"""Create the description/embedding pair used by Oren's local provider.

Examples:
  python prepare_oren_cached_dataset.py \
    --summary-file config/oren_cold_start_summaries/fancyzhx_ag_news.txt \
    --cache-dir external/artifact-linker/data/advisor_runs_initial_run_max14b/fancyzhx_ag_news
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--model", default="voyage-3")
    return parser.parse_args()


def read_summary(path: Path) -> str:
    summary = path.read_text(encoding="utf-8").strip()
    if not summary:
        raise ValueError(f"Dataset summary is empty: {path}")
    return summary


def valid_embedding(path: Path) -> bool:
    if not path.is_file():
        return False
    vector = np.load(path).astype(np.float32).reshape(-1)
    return vector.shape == (1024,) and np.isfinite(vector).all() and float(np.linalg.norm(vector)) > 0


def cache_is_current(cache_dir: Path, summary: str) -> bool:
    summary_path = cache_dir / "summary.txt"
    embedding_path = cache_dir / "summary_voyage.npy"
    return (
        summary_path.is_file()
        and summary_path.read_text(encoding="utf-8").strip() == summary
        and valid_embedding(embedding_path)
    )


def embed_summary(summary: str, model: str) -> np.ndarray:
    import voyageai

    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise EnvironmentError("VOYAGE_API_KEY is required to create the missing embedding.")
    response = voyageai.Client(api_key=api_key).embed(
        [summary[:8000]], model=model, input_type="document"
    )
    vector = np.asarray(response.embeddings[0], dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (1024,) or not np.isfinite(vector).all() or norm == 0:
        raise ValueError(f"Voyage returned an invalid embedding with shape {vector.shape}.")
    return vector / norm


def write_cache(cache_dir: Path, summary: str, embedding: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    np.save(cache_dir / "summary_voyage.npy", embedding)


def main() -> None:
    args = parse_args()
    summary = read_summary(args.summary_file)
    if cache_is_current(args.cache_dir, summary):
        print(f"Retrieval cache already valid: {args.cache_dir}")
        return
    write_cache(args.cache_dir, summary, embed_summary(summary, args.model))
    print(f"Created retrieval cache: {args.cache_dir}")


if __name__ == "__main__":
    main()

"""Conservative parsing for constrained tagged model responses."""
from __future__ import annotations

import json
import re


ANSWER_TAG_PATTERN = re.compile(
    r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL
)


def extract_tagged_answer(
    raw_generation: str,
    labels: list[str] | None = None,
    expect_set: bool = False,
) -> tuple[str, str]:
    raw = str(raw_generation).strip()
    matches = ANSWER_TAG_PATTERN.findall(raw)
    if not matches:
        return extract_untagged_answer(raw, labels or [], expect_set)
    answer = matches[-1].strip()
    if not answer:
        return "", "empty_answer"
    return answer, "ok"


def extract_untagged_answer(
    raw: str, labels: list[str], expect_set: bool
) -> tuple[str, str]:
    if not raw:
        return "", "empty_generation"
    if expect_set:
        array = embedded_label_array(raw, labels)
        if array is not None:
            return json.dumps(array, ensure_ascii=False), "untagged_json"
        mentioned = mentioned_candidate_labels(raw, labels)
        if mentioned:
            return json.dumps(mentioned, ensure_ascii=False), "untagged_labels"
    elif labels:
        mentioned = mentioned_candidate_labels(raw, labels)
        if len(mentioned) == 1:
            return mentioned[0], "untagged_label"
    return raw, "untagged_fallback"


def embedded_label_array(raw: str, labels: list[str]) -> list[str] | None:
    canonical = {label.casefold(): label for label in labels}
    for candidate in re.findall(r"\[[^\[\]]*\]", raw, flags=re.DOTALL):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) and item.casefold() in canonical for item in parsed
        ):
            continue
        return [canonical[item.casefold()] for item in parsed]
    return None


def mentioned_candidate_labels(raw: str, labels: list[str]) -> list[str]:
    return [label for label in labels if candidate_label_is_mentioned(raw, label)]


def candidate_label_is_mentioned(raw: str, label: str) -> bool:
    stripped = raw.strip().strip("`'\" .,:;")
    if stripped.casefold() == label.casefold():
        return True
    escaped = re.escape(label)
    if len(label) == 1:
        pattern = rf"\b(?:answer|choice|option)\s*(?:is|:)\s*{escaped}\b"
        return re.search(pattern, raw, flags=re.IGNORECASE) is not None
    pattern = rf"(?<!\w){escaped}(?!\w)"
    return re.search(pattern, raw, flags=re.IGNORECASE) is not None

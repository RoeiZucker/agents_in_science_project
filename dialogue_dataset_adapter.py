"""Flatten supported dialogue corpora into next-response examples."""
from __future__ import annotations

from typing import Any


DIALOGUE_DATASETS = {
    "ParlAI/blended_skill_talk",
    "google-research-datasets/taskmaster1",
    "facebook/empathetic_dialogues",
}


def adapt_dialogue_dataset(dataset_name: str, dataset: Any) -> Any:
    adapters = {
        "ParlAI/blended_skill_talk": flatten_blended_skill_talk,
        "google-research-datasets/taskmaster1": flatten_taskmaster,
        "facebook/empathetic_dialogues": flatten_empathetic_dialogues,
    }
    adapter = adapters.get(dataset_name)
    return adapter(dataset) if adapter else dataset


def is_dialogue_dataset(dataset_name: str) -> bool:
    return dataset_name in DIALOGUE_DATASETS


def dialogue_dataset(rows: list[dict[str, str]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(rows)


def dialogue_prompt(history: list[str], details: str = "") -> str:
    sections = [details.strip(), "Conversation:\n" + "\n".join(history)]
    return "\n\n".join(section for section in sections if section)


def flatten_taskmaster(dataset: Any) -> Any:
    rows: list[dict[str, str]] = []
    for conversation in dataset:
        history: list[str] = []
        details = f"Task instruction: {conversation.get('instruction_id', '')}"
        add_taskmaster_turns(rows, history, conversation, details)
    return dialogue_dataset(rows)


def add_taskmaster_turns(
    rows: list[dict[str, str]], history: list[str],
    conversation: dict[str, Any], details: str,
) -> None:
    for utterance in conversation.get("utterances") or []:
        text = str(utterance.get("text") or "").strip()
        speaker = str(utterance.get("speaker") or "speaker").strip()
        append_response_example(rows, history, text, details)
        if text:
            history.append(f"{speaker}: {text}")


def flatten_blended_skill_talk(dataset: Any) -> Any:
    rows: list[dict[str, str]] = []
    for conversation in dataset:
        history = nonempty_strings(conversation.get("previous_utterance") or [])
        details = blended_details(conversation)
        for response in conversation.get("free_messages") or []:
            text = str(response).strip()
            append_response_example(rows, history, text, details)
            if text:
                history.append(text)
    return dialogue_dataset(rows)


def blended_details(conversation: dict[str, Any]) -> str:
    return "\n".join([
        "Personas: " + " | ".join(conversation.get("personas") or []),
        "Context: " + str(conversation.get("context") or ""),
        "Additional context: " + str(conversation.get("additional_context") or ""),
    ])


def flatten_empathetic_dialogues(dataset: Any) -> Any:
    rows: list[dict[str, str]] = []
    histories: dict[str, list[str]] = {}
    for turn in dataset:
        conversation_id = str(turn.get("conv_id") or "")
        history = histories.setdefault(conversation_id, [])
        response = str(turn.get("utterance") or "").strip()
        details = empathetic_details(turn)
        append_response_example(rows, history, response, details)
        if response:
            history.append(response)
    return dialogue_dataset(rows)


def empathetic_details(turn: dict[str, Any]) -> str:
    return "\n".join([
        "Emotion: " + str(turn.get("context") or ""),
        "Situation: " + str(turn.get("prompt") or ""),
    ])


def append_response_example(
    rows: list[dict[str, str]], history: list[str], response: str, details: str,
) -> None:
    if history and response:
        rows.append({
            "_dialogue_prompt": dialogue_prompt(history, details),
            "_dialogue_response": response,
        })


def nonempty_strings(values: list[Any]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]

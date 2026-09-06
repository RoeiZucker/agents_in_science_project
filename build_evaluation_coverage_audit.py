#!/usr/bin/env python3
"""Build the reviewed 40-pair evaluator coverage audit.

Example:
  python build_evaluation_coverage_audit.py \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --output-csv config/evaluation_coverage_40.csv \
    --output-md EVALUATION_COVERAGE_AUDIT.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


AUDITS = [
    (
        "Amod/mental_health_counseling_conversations", "generation", "rouge_l",
        "Use Context as input and Response as reference; include the full conversation context.",
        [
            ("nvidia/Llama3-ChatQA-2-8B", "context_required", "Generic causal generation is supported; ROUGE-L is only a lexical proxy for counseling responses."),
            ("NousResearch/Llama-2-7b-chat-hf", "context_required", "Generic causal generation is supported; ROUGE-L is only a lexical proxy for counseling responses."),
        ],
    ),
    (
        "CFPB/consumer-finance-complaints", "classification", "macro_f1",
        "Use complaint narrative as input and Product as the ClassLabel target.",
        [
            ("MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33", "generic_supported", "Use zero-shot NLI because classifier labels are entailment labels."),
            ("ChanceFocus/finma-7b-full", "context_required", "Generate one product label; prompt must list the complete product label set."),
        ],
    ),
    (
        "FinGPT/fingpt-sentiment-train", "classification", "macro_f1",
        "Use input as text, output as target, and instruction as prompt context.",
        [
            ("IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment", "context_required", "Direct classification requires semantic alignment between checkpoint labels and FinGPT labels."),
            ("NousResearch/Hermes-2-Pro-Mistral-7B", "generic_supported", "Tagged finite-label generation is supported."),
        ],
    ),
    (
        "HUPD/hupd", "generation", "rouge_l",
        "Select one HUPD config and target before loading; title/abstract generation and classification are different tasks.",
        [
            ("unsloth/DeepSeek-R1-Distill-Qwen-14B", "context_required", "Supported only after Codex selects a coherent config, columns, and task."),
            ("sdadas/mmlw-roberta-base", "model_task_mismatch", "A fixed-label classifier cannot perform title or abstract generation; use only if a compatible classification config is chosen."),
        ],
    ),
    (
        "MU-NLPC/Calc-asdiv_a", "numeric_qa", "numeric_match",
        "Use question as input and result_float or result as target; request only the final numeric answer.",
        [
            ("llm-agents/tora-code-7b-v1.0", "generic_supported", "Tagged generation plus numeric equivalence is supported."),
            ("microsoft/rho-math-1b-interpreter-v0.1", "generic_supported", "Tagged generation plus numeric equivalence is supported."),
        ],
    ),
    (
        "RobZamp/sick", "classification", "accuracy",
        "Prompt with sentence_A and sentence_B; map the label ClassLabel names exactly.",
        [
            ("tasksource/ModernBERT-large-nli", "generic_supported", "Direct NLI labels should align with the dataset labels."),
            ("cross-encoder/nli-deberta-v3-base", "generic_supported", "Direct NLI labels should align with the dataset labels."),
        ],
    ),
    (
        "SemEvalWorkshop/hyperpartisan_news_detection", "classification", "macro_f1",
        "Select byarticle or bypublisher explicitly; use article text and the hyperpartisan boolean target.",
        [
            ("cardiffnlp/twitter-roberta-base-hate-latest", "model_task_mismatch", "Hate-speech output labels do not represent hyperpartisan labels."),
            ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "generic_supported", "Use zero-shot NLI with explicit hyperpartisan label wording."),
        ],
    ),
    (
        "SetFit/bbc-news", "classification", "macro_f1",
        "Use article text as input and the news category label as target.",
        [
            ("dima806/news-category-classifier-distilbert", "context_required", "Direct scoring is valid only if the checkpoint category names cover BBC labels."),
            ("sentence-transformers/sentence-t5-base", "generic_supported", "Use zero-shot normalized text-to-label embedding similarity; this is not a trained classification head."),
        ],
    ),
    (
        "TheFinAI/fiqa-sentiment-classification", "classification", "macro_f1",
        "Use sentence plus aspect as input; derive the categorical target from the dataset label/score convention.",
        [
            ("gauneg/roberta-base-absa-ate-sentiment", "context_required", "Verify aspect-sentiment label semantics before direct classifier scoring."),
            ("CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment", "context_required", "Verify language and label alignment before direct classifier scoring."),
        ],
    ),
    (
        "ade-benchmark-corpus/ade_corpus_v2", "classification", "macro_f1",
        "Select the classification config explicitly; relation and entity configs require different protocols.",
        [
            ("PlanTL-GOB-ES/roberta-base-biomedical-clinical-es", "generic_supported", "Use masked-label pseudo-likelihood; Spanish pretraining remains a transfer-quality risk for English ADE text."),
            ("tomaarsen/span-marker-bert-base-uncased-acronyms", "model_task_mismatch", "Acronym span extraction does not answer ADE sentence classification or ADE relations."),
        ],
    ),
    (
        "allenai/cosmos_qa", "multiple_choice", "accuracy",
        "Use context and question in the prompt; use answer0 through answer3 as separate choices and label as target.",
        [
            ("soniox/Soniox-7B-v1.0", "generic_supported", "Separate-choice-column support removes the former dataset-specific gap."),
            ("context-labs/Meta-Llama-3.1-8B-Instruct-FP16", "generic_supported", "Separate-choice-column support removes the former dataset-specific gap."),
        ],
    ),
    (
        "allenai/quartz", "multiple_choice", "accuracy",
        "Use para plus question; use structured choices and answerKey.",
        [
            ("albert/albert-xxlarge-v1", "generic_supported", "Use masked-choice pseudo-likelihood over the Quartz answer texts."),
            ("Qwen/Qwen2-7B", "generic_supported", "Generic causal answer-choice likelihood is supported."),
        ],
    ),
    (
        "bigbio/bc5cdr", "token_classification", "set_f1",
        "Select a BigBio schema and compare normalized entity type/text or spans as sets.",
        [
            ("OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M", "generic_supported", "BIO span reconstruction and BigBio conversion are supported; this CHEM-only ontology cannot predict BC5CDR diseases."),
            ("OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M", "generic_supported", "BIO span reconstruction and BigBio conversion are supported; this CHEM-only ontology cannot predict BC5CDR diseases."),
        ],
    ),
    (
        "bigbio/bioasq_task_b", "qa", "qa_f1",
        "Select a labeled BioASQ config; build the prompt from body/context and read exact answer lists through dotted paths.",
        [
            ("johnsnowlabs/JSL-MedMNX-7B-v2.0", "context_required", "Generic tagged QA works once nested columns and answer type are specified."),
            ("axiong/PMC_LLaMA_13B", "context_required", "Generic tagged QA works once nested columns and answer type are specified."),
        ],
    ),
    (
        "bigbio/chemprot", "relation_extraction", "set_f1",
        "Select a BigBio KB schema; score normalized head-relation-tail tuples as a set.",
        [
            ("OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M", "model_task_mismatch", "An NER checkpoint detects entities but does not classify chemical-protein relations."),
            ("OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M", "model_task_mismatch", "An NER checkpoint detects entities but does not classify chemical-protein relations."),
        ],
    ),
    (
        "ccdv/WCEP-10", "summarization", "rouge_l",
        "Use document/article text as input and summary as reference; select the labeled config and split.",
        [
            ("THUDM/glm-10b", "context_required", "Generic generation is supported if remote model code loads under the cluster environment."),
            ("google/byt5-xxl", "generic_supported", "Generic seq2seq summarization with ROUGE-L is supported."),
        ],
    ),
    (
        "ccdv/pubmed-summarization", "summarization", "rouge_l",
        "Select document or section config deliberately; use article as input and abstract as reference.",
        [
            ("THUDM/glm-10b", "context_required", "Generic generation is supported if remote model code loads under the cluster environment."),
            ("nvidia/Llama3-ChatQA-2-8B", "generic_supported", "Generic tagged summarization with ROUGE-L is supported."),
        ],
    ),
    (
        "coastalcph/multi_eurlex", "multilabel_classification", "set_f1",
        "Select one language/config; map the Sequence(ClassLabel) target and request a JSON label array.",
        [
            ("openGPT-X/Teuken-7B-instruct-research-v0.4", "generic_supported", "Tagged label-set generation and set F1 are supported."),
            ("ReliableAI/UCCIX-Llama2-13B-Instruct", "generic_supported", "Tagged label-set generation and set F1 are supported."),
        ],
    ),
    (
        "google-research-datasets/go_emotions", "multilabel_classification", "set_f1",
        "Prefer the simplified config; use text and the Sequence(ClassLabel) labels field.",
        [
            ("cirimus/modernbert-base-emotions", "model_task_mismatch", "The config declares single-label classification with 7 labels, not GoEmotions multilabel prediction."),
            ("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest", "model_task_mismatch", "The config is multilabel but its 11-label ontology does not cover the 28 GoEmotions labels."),
        ],
    ),
    (
        "theatticusproject/cuad", "qa", "qa_f1",
        "Use contract context plus question; read all reference answer texts from the nested answers field.",
        [
            ("sdadas/mmlw-roberta-base", "model_task_mismatch", "A fixed-label sentence model cannot extract free-text contract spans."),
            ("yandex/YandexGPT-5-Lite-8B-instruct", "context_required", "Generic tagged QA works once nested CUAD columns are selected."),
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def condition_pairs(path: Path) -> set[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["query_dataset"], row["model_name"]) for row in csv.DictReader(handle)}


def audit_rows(available: set[tuple[str, str]]) -> list[dict[str, str]]:
    rows = []
    for dataset, task, method, context, models in AUDITS:
        for model, support, change in models:
            if (dataset, model) not in available:
                raise ValueError(f"Reviewed pair is absent from conditions CSV: {dataset} / {model}")
            rows.append({
                "dataset": dataset,
                "model": model,
                "task_family": task,
                "recommended_method": method,
                "dataset_context_needed": context,
                "support_level": support,
                "needed_change_or_constraint": change,
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["support_level"] for row in rows)
    lines = [
        "# Evaluation Coverage Audit",
        "",
        f"Reviewed {len(rows)} dataset-model pairs across {len({row['dataset'] for row in rows})} datasets.",
        "This is a design audit, not an inference result. Support claims still require smoke tests on the target cluster.",
        "",
        "## Support Summary",
        "",
        "| Support level | Pairs |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    lines.extend([
        "",
        "## Pair Review",
        "",
        "| Dataset | Model | Task | Method | Support | Remaining work |",
        "|---|---|---|---|---|---|",
    ])
    for row in rows:
        values = [
            row["dataset"], row["model"], row["task_family"],
            row["recommended_method"], row["support_level"],
            row["needed_change_or_constraint"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = audit_rows(condition_pairs(args.conditions_csv))
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {len(rows)} pairs across {len(AUDITS)} datasets.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the deterministic additional 50-dataset evaluator coverage audit.

Example:
  python build_additional_coverage_audit.py \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --existing-csv config/evaluation_coverage_40.csv \
    --output-csv config/evaluation_coverage_additional50.csv \
    --output-md EVALUATION_COVERAGE_ADDITIONAL50.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_DATA = """
Anthropic/llm_global_opinions|distribution_prediction|jensen_shannon|dedicated_adapter|Parse options and per-country selection distributions; choose a country or aggregate policy.|Add safe parsing for serialized lists/defaultdicts and a probability-distribution scorer; ordinary label accuracy is not valid.
JanosAudran/financial-reports-sec|classification|macro_f1|context_required|Use sentence as input; choose one horizon such as labels.1d as the binary target.|No evaluator change is required for one horizon, but Codex must choose the horizon and explain the return-direction labels.
McAuley-Lab/Amazon-Reviews-2023|classification|macro_f1|dataset_access_blocker|Choose one product-category config; use review text and rating or category as the target.|Add a config-aware parquet/raw-data fallback if the legacy repository cannot load; do not scan every Amazon category.
ParlAI/blended_skill_talk|dialogue_generation|rouge_l|dedicated_adapter|Construct one example per dialogue turn from persona, prior utterances, and free or guided messages.|Add a dialogue-turn flattener and define which response stream is gold; whole-list generation is not meaningful.
allenai/peer_read|classification|macro_f1|dataset_access_blocker|Choose a review/acceptance task and explicit PeerRead config before loading.|Repair legacy-script loading or use a maintained mirror, then recover the selected target's label semantics.
ashraq/financial-news-articles|summarization|rouge_l|context_required|Use article text as input and title as a headline-generation reference.|No classification gold exists; use a generative model because the highest-ranked fixed classifiers are task-mismatched.
bigbio/mediqa_qa|qa|qa_f1|context_required|Use question plus context and score against the answer list; handle empty choices as free-text QA.|Existing tagged QA supports this after selecting the BigBio QA config; avoid GGUF/unsupported quantized candidates.
bigbio/n2c2_2018_track2|relation_extraction|set_f1|dedicated_adapter|Use BigBio passages, entities, and medication-event relations.|Add normalized head-relation-tail extraction/scoring; NER checkpoints alone cannot predict medication relations.
bigbio/ncbi_disease|token_classification|set_f1|context_required|Select a BigBio schema and evaluate disease entity type/text sets.|The current BigBio entity adapter is sufficient, but choose a disease NER checkpoint rather than CHEM-only PharmaDetect models.
biglam/gutenberg-poetry-corpus|language_modeling|perplexity|reusable_evaluator_change|The dataset contains individual poetry lines and no supervised target pair.|Add causal/seq2seq perplexity or a documented continuation construction; ROUGE against the input would be invalid.
bitext/Bitext-customer-support-llm-chatbot-training-dataset|classification|macro_f1|generic_supported|Use instruction as input and intent or category as the finite target.|Current finite-label generation or zero-shot NLI works; Codex must choose intent versus coarse category.
ccdv/govreport-summarization|summarization|rouge_l|generic_supported|Use report as input and summary as reference with the document config.|Current tagged/seq2seq summarization works; long-input truncation must remain visible in the audit.
climatebert/climate_commitments_actions|classification|macro_f1|generic_supported|Use text and the dataset ClassLabel target.|Current zero-shot NLI and label-map handling are sufficient.
climatebert/environmental_claims|classification|macro_f1|generic_supported|Use text and the dataset ClassLabel target.|Current zero-shot NLI and label-map handling are sufficient.
coastalcph/lex_glue|multiple_choice|accuracy|context_required|Choose exactly one LexGLUE config; case_hold uses context, endings, and label while other configs change task family.|No new adapter is needed for case_hold; Codex must not mix configs and must reject incompatible model heads.
community-datasets/yahoo_answers_topics|classification|macro_f1|generic_supported|Combine question_title, question_content, and optionally best_answer; predict topic.|Current prompt templates and finite labels support this task.
cornell-movie-dialog/cornell_movie_dialog|dialogue_generation|rouge_l|dedicated_adapter|Reconstruct ordered conversation turns and produce next-utterance examples.|Add a legacy loader fallback and reusable turn flattener; evaluating serialized conversations directly is invalid.
cornell-movie-review-data/rotten_tomatoes|classification|accuracy|generic_supported|Use text and binary sentiment label.|Current classification protocols support this directly.
deepmind/aqua_rat|multiple_choice|accuracy|generic_supported|Use question, options, and correct; rationale is optional input context and not the target.|Current structured-choice scoring supports this directly.
deepmind/math_dataset|numeric_qa|numeric_match|dataset_access_blocker|Choose a single generated math module/config and use question/answer fields.|Repair legacy generator loading and verify symbolic answers; numeric_match alone may need exact-match fallback per config.
embedding-data/Amazon-QA|qa|qa_f1|context_required|Use query as question and pos as multiple acceptable answer texts.|Current best-of-reference QA F1 works, but the dataset has train-only retrieval pairs and is not a clean held-out benchmark.
facebook/empathetic_dialogues|dialogue_generation|rouge_l|dedicated_adapter|Create one next-response example per conversation turn with emotion/context metadata.|Add dialogue flattening and a maintained loader fallback; candidate chat models must be normal Transformers checkpoints.
fancyzhx/amazon_polarity|classification|accuracy|generic_supported|Combine title and content; predict binary label.|Current zero-shot NLI or tagged finite-label generation works.
fancyzhx/dbpedia_14|classification|accuracy|generic_supported|Combine title and content; predict one of 14 ontology labels.|Current finite-label classification works with ClassLabel names.
fancyzhx/yelp_polarity|classification|accuracy|generic_supported|Use review text and binary sentiment label.|Current classification protocols support this directly.
google-research-datasets/paws|classification|accuracy|generic_supported|Prompt with sentence1 and sentence2; predict paraphrase label.|Current pair-text prompt plus finite-label protocols work; sentence encoders remain a proxy rather than a trained head.
google-research-datasets/poem_sentiment|classification|macro_f1|generic_supported|Use verse_text and the four-way sentiment ClassLabel.|Current finite-label classification works after preserving the dataset label names.
google-research-datasets/taskmaster1|dialogue_generation|rouge_l|dedicated_adapter|Flatten conversation turns and retain API/slot context when predicting the next utterance.|Add legacy loading plus a reusable task-oriented-dialogue flattener.
google/civil_comments|multilabel_classification|set_f1|reusable_evaluator_change|Choose toxicity only or threshold the seven continuous toxicity attributes into a label set.|Add declared derived targets/thresholds and multi-column gold construction; current one-column label extraction is insufficient.
griffin/ChemSum|summarization|rouge_l|generic_supported|Use sections as source and abstract as reference; title is optional context.|Current summarization works, with explicit truncation reporting for very long articles.
gtfintechlab/finer-ord|token_classification|set_f1|reusable_evaluator_change|Group token rows by doc_idx and sent_idx before evaluating gold_label sequences.|Add row-grouping into token/BIO examples; the current evaluator assumes one complete text example per row.
james-burton/wine_reviews|classification|macro_f1|context_required|Use description as input and variety as classification target; points is a separate regression task.|Current classification works for variety, but point prediction requires a future regression metric.
joelniklaus/legal_case_document_summarization|summarization|rouge_l|generic_supported|Use judgement as input and summary as reference.|Current summarization works; record context truncation because judgments are long.
knkarthick/samsum|summarization|rouge_l|generic_supported|Use dialogue as input and summary as reference.|Current summarization works directly.
launch/gov_report_qs|qa|qa_f1|dedicated_adapter|Explode aligned question_summary_pairs arrays and combine them with flattened document_sections.|Add a nested aligned-array flattener that emits one QA example per question-summary pair.
microsoft/orca-math-word-problems-200k|numeric_qa|numeric_match|context_required|Use question and answer; instruct the model to place only the final numeric value inside answer tags.|Current numeric scoring works when a final number exists; route nonnumeric answers to exact/QA matching.
midas/inspec|token_classification|set_f1|reusable_evaluator_change|Join document tokens and convert doc_bio_tags into gold keyphrase spans.|Add generic pretokenized-sequence input and BIO-gold conversion.
midas/semeval2017|token_classification|set_f1|reusable_evaluator_change|Join document tokens and convert doc_bio_tags into gold keyphrase spans.|Reuse the same pretokenized-sequence/BIO-gold adapter as Inspec.
nlpaueb/finer-139|token_classification|set_f1|reusable_evaluator_change|Join tokens and map ner_tags ClassLabel values into typed gold spans.|Reuse a generic token-list/BIO adapter; verify checkpoint ontology against 139 financial labels.
nyu-mll/multi_nli|classification|accuracy|generic_supported|Prompt with premise and hypothesis and preserve contradiction/neutral/entailment semantics.|Current direct NLI classifier protocol supports this cleanly.
odegiber/hate_speech18|classification|macro_f1|dataset_access_blocker|Use tweet text and hate/offensive label after confirming the legacy schema.|Add a maintained raw-data/mirror fallback if the repository script cannot load.
raquiba/Sarcasm_News_Headline|classification|accuracy|generic_supported|Use headline and is_sarcastic as the binary target.|Current finite-label classification works; the headline-cause model may have mismatched output semantics.
sentence-transformers/eli5|qa|qa_f1|context_required|Use question and answer as open-ended QA pairs.|Current tagged QA can run, but lexical F1 is a weak quality proxy and ALBERT masked checkpoints are task-mismatched.
sentence-transformers/yahoo-answers|qa|qa_f1|context_required|Choose one pair config and use its question/title fields and answer reference.|Current tagged QA works; document the train-only retrieval-oriented nature of the dataset.
sileod/movie_recommendation|multiple_choice|accuracy|generic_supported|Use question, answer_0 through answer_3, and label.|Current separate-choice-column scoring works; reject ShotVL/video or GGUF candidates for this text-only task.
tals/vitaminc|classification|accuracy|generic_supported|Prompt with evidence and claim; predict support/refute/neutral label.|Current NLI protocol supports this directly.
ucberkeley-dlab/measuring-hate-speech|regression|correlation|dedicated_adapter|Aggregate repeated annotator rows by comment_id and choose hate_speech_score or a declared thresholded label.|Add grouped-example preprocessing plus regression metrics such as MAE/Spearman; row-level classification would be misleading.
ucsbnlp/liar|classification|macro_f1|dataset_access_blocker|Use statement text and six-way veracity label.|The Hub dataset endpoint is unavailable; use a maintained LIAR mirror or explicit raw-file fallback before evaluation.
virattt/financial-qa-10K|qa|qa_f1|context_required|Use context plus question and score against answer; detect numeric answers when appropriate.|Current tagged QA works; choose a generative model because generic ALBERT/embedding checkpoints do not extract answers.
zeroshot/twitter-financial-news-sentiment|classification|macro_f1|generic_supported|Use text and three-way financial sentiment label.|Current zero-shot NLI works; ABSA or ONNX candidates require label/format compatibility checks.
""".strip()


FIELDS = (
    "dataset",
    "task_family",
    "recommended_method",
    "support_level",
    "dataset_context_needed",
    "needed_change_or_constraint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path, required=True)
    parser.add_argument("--existing-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def audit_rows() -> list[dict[str, str]]:
    return [dict(zip(FIELDS, line.split("|", 5))) for line in AUDIT_DATA.splitlines()]


def models_by_dataset(path: Path) -> dict[str, list[str]]:
    models: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["model_name"]
            if name not in models[row["query_dataset"]]:
                models[row["query_dataset"]].append(name)
    return models


def existing_datasets(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["dataset"] for row in csv.DictReader(handle)}


def validate(rows: list[dict[str, str]], models: dict[str, list[str]], existing: set[str]) -> None:
    datasets = [row["dataset"] for row in rows]
    if len(rows) != 50 or len(set(datasets)) != 50:
        raise ValueError("The additional audit must contain exactly 50 unique datasets.")
    overlap = set(datasets) & existing
    if overlap:
        raise ValueError(f"Additional audit overlaps the existing audit: {sorted(overlap)}")
    missing = set(datasets) - set(models)
    if missing:
        raise ValueError(f"Datasets are absent from the conditions CSV: {sorted(missing)}")


def add_models(rows: list[dict[str, str]], models: dict[str, list[str]]) -> list[dict[str, str]]:
    return [
        {**row, "representative_models": "; ".join(models[row["dataset"]][:2])}
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    counts = Counter(row["support_level"] for row in rows)
    lines = [
        "# Additional 50-Dataset Evaluation Coverage Audit",
        "",
        "This deterministic sample is disjoint from `config/evaluation_coverage_40.csv`.",
        "Schemas and sample structures were checked through Hugging Face Hub and datasets-server metadata; no model inference was run.",
        "",
        "## Support Summary",
        "",
        "| Support level | Datasets |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
    lines.extend([
        "",
        "## Dataset Review",
        "",
        "| Dataset | Task | Method | Support | What is needed |",
        "|---|---|---|---|---|",
    ])
    for row in rows:
        values = [
            row["dataset"], row["task_family"], row["recommended_method"],
            row["support_level"], row["needed_change_or_constraint"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = audit_rows()
    models = models_by_dataset(args.conditions_csv)
    validate(rows, models, existing_datasets(args.existing_csv))
    rows = add_models(rows, models)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {len(rows)} additional datasets.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the deterministic evaluator coverage audit for the final 20 datasets.

Example:
  python build_remaining_coverage_audit.py \
    --conditions-csv config/evaluation_conditions_intersection90.csv \
    --coverage-csv config/evaluation_coverage_40.csv \
    --coverage-csv config/evaluation_coverage_additional50.csv \
    --output-csv config/evaluation_coverage_remaining20.csv \
    --output-md EVALUATION_COVERAGE_REMAINING20.md
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


AUDIT_DATA = """
allenai/mslr2022|available|cochrane|validation|summarization|rouge_l|generic_supported|Auto-routing sees unrestricted generation and would use exact match.|Join the title and abstract lists as the source; use target as the systematic-review conclusion.|Codex context can select summarization and ROUGE-L. The two leading candidates are ordinary causal LMs and can use tagged generation.
bigbio/ddi_corpus|available|ddi_corpus_bigbio_kb|test|relation_extraction|set_f1|dedicated_adapter|No recognized answer column exists, so automatic inspection reports no gold.|Flatten passages and entities; score typed drug-drug head-relation-tail tuples from relations.|Add normalized relation-tuple prompting, parsing, and scoring. The leading PharmaDetect checkpoints are NER token classifiers and cannot predict DDI relation types.
cardiffnlp/tweet_eval|available|emotion|validation|classification|macro_f1|context_required|The default config would be emoji; generic label generation is measurable but benchmark/model semantics may not match.|Choose one config explicitly and preserve that config's ClassLabel names.|No evaluator code is needed. Codex must align the config with the model head; the top emotion and hate classifiers cannot both be evaluated on the same TweetEval config.
chengxuphd/liar2|available|default|validation|classification|macro_f1|context_required|The integer label lacks ClassLabel metadata, so automatic planning cannot recover label meanings.|Use statement plus optional speaker/context metadata and restore the six LIAR veracity labels from the card.|No evaluator code is needed after supplying label_map, prompt context, classification, and macro-F1. Fixed classifier heads still require semantic label compatibility.
clinc/clinc_oos|available|imbalanced|validation|classification|macro_f1|context_required|The target is named intent, which automatic answer-column detection misses.|Choose imbalanced, plus, or small; use text and the 151-way intent ClassLabel target.|No evaluator code is needed after Codex selects the config and intent column. The first candidate is GGUF and unsupported; the second is a usable zero-shot classifier.
deepset/covid_qa_deepset|available|covid_qa_deepset|train|qa|qa_f1|generic_supported|Automatic planning finds question and answers but would score free text with exact match.|Prompt with context and question; score against every answers.text reference.|Current tagged QA and best-reference token F1 are sufficient. This repository is train-only, so results are not a clean held-out benchmark; the first two candidates are unsupported/gated formats.
dreamerdeo/finqa|available|default|validation|numeric_qa|numeric_match|context_required|Automatic planning treats answer as unrestricted text and omits the table and evidence.|Serialize pre_text, table, post_text, question, and optional gold_evidence; request only the final value.|Current numeric matching is sufficient for parseable values, with exact normalized fallback. Codex must construct the financial context; one of the first two models is a supported causal LM.
fever/fever|legacy_script_unloadable|v1.0|validation|classification|accuracy|dataset_access_blocker|The current datasets library rejects the repository's legacy fever.py loader before planning.|Use claim and three-way FEVER label; evidence text requires a separate Wikipedia join if included.|Create a maintained parquet/raw JSON fallback or pin an isolated legacy datasets environment. No Hugging Face sign-in is required.
gamino/wiki_medical_terms|available|default|train|generation|exact_match|context_required|Neither page_title nor page_text is recognized as an answer by default.|Use page_text as the definition and page_title as the medical term to generate.|Current tagged free-text generation works after column reversal and a manual prompt. The repository is train-only and the NER candidate is task-mismatched.
hover-nlp/hover|legacy_script_unloadable|default|validation|classification|accuracy|dataset_access_blocker|The current datasets library rejects hover.py before planning.|Use claim and binary support label; supporting_facts are Wikipedia pointers, not evidence text.|Add a maintained raw JSON fallback and optionally join the HoVer Wikipedia corpus. No sign-in is required; claim-only scoring should be labeled as closed-book fact verification.
launch/gov_report|available|plain_text|validation|summarization|rouge_l|generic_supported|The summary column is not in automatic answer candidates, so planning reports no gold.|Use document as source and summary as reference; select plain_text explicitly.|Current tagged summarization and ROUGE-L are sufficient. Record truncation because reports are long; both leading candidates are supported causal LMs.
li2017dailydialog/daily_dialog|legacy_script_unloadable|default|validation|classification|macro_f1|dataset_access_blocker|The current datasets library rejects daily_dialog.py before task planning.|Flatten dialogue turns and align each utterance with emotion or dialog-act labels; choose exactly one target.|Add a maintained raw-data fallback and a reusable aligned-turn flattener. The leading classifiers can only be used when their output ontology matches the chosen target.
medalpaca/medical_meadow_medqa|available|default|train|multiple_choice|accuracy|dedicated_adapter|Options and the answer are embedded in input/output strings, so automatic structured-choice detection fails.|Parse the option dictionary from input, use instruction as guidance, and normalize output such as E: Nitrofurantoin.|Add a small MedQA parser that exposes choices and the answer key. Tagged exact generation can run as a weaker fallback but is not equivalent multiple-choice evaluation. The dataset is train-only; one leading model is GGUF.
midas/duc2001|available|extraction|test|token_classification|set_f1|reusable_evaluator_change|document and doc_bio_tags are token lists outside the current automatic column vocabulary.|Join document tokens and convert doc_bio_tags into gold keyphrase spans.|Reuse the planned generic token-list/BIO adapter also needed by Inspec and SemEval2017. The two leading checkpoints are unsupported or have a mismatched token-label ontology.
pkavumba/balanced-copa|available|default|test|multiple_choice|accuracy|context_required|choice1 and choice2 are separate columns, so automatic planning incorrectly treats the task as generation.|Prompt with premise and whether the question asks for cause or effect; expose choice1 and choice2 through choices_columns.|Current separate-choice scoring is sufficient after Codex context. The first model is GGUF; the second is a supported causal LM.
starmpcc/Asclepius-Synthetic-Clinical-Notes|available|default|train|qa|qa_f1|context_required|Question and answer are found, but note and task are omitted and exact match is selected.|Prompt with task, clinical note, and question; score answer with QA F1.|Current tagged QA works after context construction. The repository is synthetic and train-only, so report that limitation; one leading model is GGUF.
tdiggelm/climate_fever|available|default|test|classification|macro_f1|context_required|The target claim_label is not an automatic answer candidate and evidences are nested.|Prompt with claim plus evidence strings; preserve SUPPORTS, REFUTES, NOT_ENOUGH_INFO, and DISPUTED.|Current finite-label/NLI protocols work after Codex supplies the nested-evidence prompt and label column. Verify classifier label semantics before direct-head scoring.
tuetschek/multi_woz_v22|available|default|validation|dialogue_generation|rouge_l|dedicated_adapter|The nested turns object has no single question/answer row for the generic evaluator.|Emit one example per system turn using all preceding utterances and optionally state/slot context.|Add a reusable task-oriented dialogue-turn flattener. The leading causal LM is usable after flattening; the other representative candidate is GGUF.
ucirvine/reuters21578|legacy_script_unloadable|ModApte|test|multilabel_classification|set_f1|dataset_access_blocker|The current datasets library rejects reuters21578.py before planning.|Use title plus text and predict the topics set under one declared Reuters split convention.|Add a raw tar/SGML or maintained parquet fallback, then use tagged JSON-set generation. No sign-in is required; the leading fixed classifier needs ontology verification.
wics/strategy-qa|available|strategyQA|test|classification|accuracy|context_required|The boolean answer is detected, but the default prompt omits facts and decomposition.|Prompt with facts and question; map boolean values to yes/no and retain decomposition only as optional context.|Current finite-label generation supports this after Codex context. One leading model is a supported causal checkpoint and one is GGUF.
""".strip()


FIELDS = (
    "dataset",
    "live_status",
    "selected_config",
    "selected_split",
    "task_family",
    "recommended_method",
    "support_level",
    "current_auto_route",
    "dataset_context_needed",
    "needed_change_or_constraint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions-csv", type=Path, required=True)
    parser.add_argument("--coverage-csv", type=Path, action="append", default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def audit_rows() -> list[dict[str, str]]:
    return [dict(zip(FIELDS, line.split("|", 9))) for line in AUDIT_DATA.splitlines()]


def models_by_dataset(path: Path) -> dict[str, list[str]]:
    models: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["model_name"]
            if name not in models[row["query_dataset"]]:
                models[row["query_dataset"]].append(name)
    return models


def covered_datasets(paths: list[Path]) -> set[str]:
    covered: set[str] = set()
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            covered.update(row["dataset"] for row in csv.DictReader(handle))
    return covered


def validate(rows: list[dict[str, str]], models: dict[str, list[str]], covered: set[str]) -> None:
    datasets = [row["dataset"] for row in rows]
    if len(rows) != 20 or len(set(datasets)) != 20:
        raise ValueError("The remaining audit must contain exactly 20 unique datasets.")
    overlap = set(datasets) & covered
    if overlap:
        raise ValueError(f"Remaining audit overlaps prior coverage: {sorted(overlap)}")
    condition_datasets = set(models)
    missing = set(datasets) - condition_datasets
    unexpected = condition_datasets - covered - set(datasets)
    if missing or unexpected:
        raise ValueError(f"Coverage mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")


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
    support = Counter(row["support_level"] for row in rows)
    access = Counter(row["live_status"] for row in rows)
    lines = [
        "# Remaining 20-Dataset Evaluation Coverage Audit",
        "",
        "This completes a disjoint audit of all 90 datasets in the condition manifest.",
        "Live configs, splits, schemas, and sample rows were checked through official Hugging Face metadata; no model weights or inference were run.",
        "",
        "## Summary",
        "",
        f"- Live datasets inspectable now: {access['available']}",
        f"- Legacy-script loader blockers: {access['legacy_script_unloadable']}",
        "- Datasets requiring Hugging Face access approval: 0",
        "",
        "| Support level | Datasets |",
        "|---|---:|",
    ]
    lines.extend(f"| {name} | {count} |" for name, count in sorted(support.items()))
    lines.extend([
        "",
        "## Dataset Review",
        "",
        "| Dataset | Config/split | Task | Metric | Support | Required work |",
        "|---|---|---|---|---|---|",
    ])
    for row in rows:
        values = [
            row["dataset"],
            f"{row['selected_config']} / {row['selected_split']}",
            row["task_family"],
            row["recommended_method"],
            row["support_level"],
            row["needed_change_or_constraint"],
        ]
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    lines.extend([
        "",
        "## Cross-Cutting Findings",
        "",
        "1. The production condition-run scout can choose all registered tasks and metrics, but `make_codex_prompt.py` still documents only generation, multiple choice, and image classification.",
        "2. Automatic column detection is intentionally shallow. Codex context is essential for targets such as intent, summary, output, claim_label, doc_bio_tags, and topics.",
        "3. Four public datasets are blocked by legacy Python loading scripts, not authentication.",
        "4. Dialogue flattening, relation tuples, and token-list/BIO conversion are the main reusable evaluator additions exposed by this sample.",
        "5. Candidate compatibility remains pair-specific: GGUF, gated checkpoints, fixed label heads, and task-mismatched NER models must be rejected before inference.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = audit_rows()
    models = models_by_dataset(args.conditions_csv)
    validate(rows, models, covered_datasets(args.coverage_csv))
    rows = add_models(rows, models)
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {len(rows)} remaining datasets.")


if __name__ == "__main__":
    main()

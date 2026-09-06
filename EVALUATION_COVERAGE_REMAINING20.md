# Remaining 20-Dataset Evaluation Coverage Audit

This completes a disjoint audit of all 90 datasets in the condition manifest.
Live configs, splits, schemas, and sample rows were checked through official Hugging Face metadata; no model weights or inference were run.

## Summary

- Live datasets inspectable now: 16
- Legacy-script loader blockers: 4
- Datasets requiring Hugging Face access approval: 0

| Support level | Datasets |
|---|---:|
| context_required | 9 |
| dataset_access_blocker | 4 |
| dedicated_adapter | 3 |
| generic_supported | 3 |
| reusable_evaluator_change | 1 |

## Dataset Review

| Dataset | Config/split | Task | Metric | Support | Required work |
|---|---|---|---|---|---|
| allenai/mslr2022 | cochrane / validation | summarization | rouge_l | generic_supported | Codex context can select summarization and ROUGE-L. The two leading candidates are ordinary causal LMs and can use tagged generation. |
| bigbio/ddi_corpus | ddi_corpus_bigbio_kb / test | relation_extraction | set_f1 | dedicated_adapter | Add normalized relation-tuple prompting, parsing, and scoring. The leading PharmaDetect checkpoints are NER token classifiers and cannot predict DDI relation types. |
| cardiffnlp/tweet_eval | emotion / validation | classification | macro_f1 | context_required | No evaluator code is needed. Codex must align the config with the model head; the top emotion and hate classifiers cannot both be evaluated on the same TweetEval config. |
| chengxuphd/liar2 | default / validation | classification | macro_f1 | context_required | No evaluator code is needed after supplying label_map, prompt context, classification, and macro-F1. Fixed classifier heads still require semantic label compatibility. |
| clinc/clinc_oos | imbalanced / validation | classification | macro_f1 | context_required | No evaluator code is needed after Codex selects the config and intent column. The first candidate is GGUF and unsupported; the second is a usable zero-shot classifier. |
| deepset/covid_qa_deepset | covid_qa_deepset / train | qa | qa_f1 | generic_supported | Current tagged QA and best-reference token F1 are sufficient. This repository is train-only, so results are not a clean held-out benchmark; the first two candidates are unsupported/gated formats. |
| dreamerdeo/finqa | default / validation | numeric_qa | numeric_match | context_required | Current numeric matching is sufficient for parseable values, with exact normalized fallback. Codex must construct the financial context; one of the first two models is a supported causal LM. |
| fever/fever | v1.0 / validation | classification | accuracy | dataset_access_blocker | Create a maintained parquet/raw JSON fallback or pin an isolated legacy datasets environment. No Hugging Face sign-in is required. |
| gamino/wiki_medical_terms | default / train | generation | exact_match | context_required | Current tagged free-text generation works after column reversal and a manual prompt. The repository is train-only and the NER candidate is task-mismatched. |
| hover-nlp/hover | default / validation | classification | accuracy | dataset_access_blocker | Add a maintained raw JSON fallback and optionally join the HoVer Wikipedia corpus. No sign-in is required; claim-only scoring should be labeled as closed-book fact verification. |
| launch/gov_report | plain_text / validation | summarization | rouge_l | generic_supported | Current tagged summarization and ROUGE-L are sufficient. Record truncation because reports are long; both leading candidates are supported causal LMs. |
| li2017dailydialog/daily_dialog | default / validation | classification | macro_f1 | dataset_access_blocker | Add a maintained raw-data fallback and a reusable aligned-turn flattener. The leading classifiers can only be used when their output ontology matches the chosen target. |
| medalpaca/medical_meadow_medqa | default / train | multiple_choice | accuracy | dedicated_adapter | Add a small MedQA parser that exposes choices and the answer key. Tagged exact generation can run as a weaker fallback but is not equivalent multiple-choice evaluation. The dataset is train-only; one leading model is GGUF. |
| midas/duc2001 | extraction / test | token_classification | set_f1 | reusable_evaluator_change | Reuse the planned generic token-list/BIO adapter also needed by Inspec and SemEval2017. The two leading checkpoints are unsupported or have a mismatched token-label ontology. |
| pkavumba/balanced-copa | default / test | multiple_choice | accuracy | context_required | Current separate-choice scoring is sufficient after Codex context. The first model is GGUF; the second is a supported causal LM. |
| starmpcc/Asclepius-Synthetic-Clinical-Notes | default / train | qa | qa_f1 | context_required | Current tagged QA works after context construction. The repository is synthetic and train-only, so report that limitation; one leading model is GGUF. |
| tdiggelm/climate_fever | default / test | classification | macro_f1 | context_required | Current finite-label/NLI protocols work after Codex supplies the nested-evidence prompt and label column. Verify classifier label semantics before direct-head scoring. |
| tuetschek/multi_woz_v22 | default / validation | dialogue_generation | rouge_l | dedicated_adapter | Add a reusable task-oriented dialogue-turn flattener. The leading causal LM is usable after flattening; the other representative candidate is GGUF. |
| ucirvine/reuters21578 | ModApte / test | multilabel_classification | set_f1 | dataset_access_blocker | Add a raw tar/SGML or maintained parquet fallback, then use tagged JSON-set generation. No sign-in is required; the leading fixed classifier needs ontology verification. |
| wics/strategy-qa | strategyQA / test | classification | accuracy | context_required | Current finite-label generation supports this after Codex context. One leading model is a supported causal checkpoint and one is GGUF. |

## Cross-Cutting Findings

1. The production condition-run scout can choose all registered tasks and metrics, but `make_codex_prompt.py` still documents only generation, multiple choice, and image classification.
2. Automatic column detection is intentionally shallow. Codex context is essential for targets such as intent, summary, output, claim_label, doc_bio_tags, and topics.
3. Four public datasets are blocked by legacy Python loading scripts, not authentication.
4. Dialogue flattening, relation tuples, and token-list/BIO conversion are the main reusable evaluator additions exposed by this sample.
5. Candidate compatibility remains pair-specific: GGUF, gated checkpoints, fixed label heads, and task-mismatched NER models must be rejected before inference.

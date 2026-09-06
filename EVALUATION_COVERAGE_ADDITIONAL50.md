# Additional 50-Dataset Evaluation Coverage Audit

This deterministic sample is disjoint from `config/evaluation_coverage_40.csv`.
Schemas and sample structures were checked through Hugging Face Hub and datasets-server metadata; no model inference was run.

## Support Summary

| Support level | Datasets |
|---|---:|
| context_required | 11 |
| dataset_access_blocker | 5 |
| dedicated_adapter | 8 |
| generic_supported | 20 |
| reusable_evaluator_change | 6 |

## Dataset Review

| Dataset | Task | Method | Support | What is needed |
|---|---|---|---|---|
| Anthropic/llm_global_opinions | distribution_prediction | jensen_shannon | dedicated_adapter | Add safe parsing for serialized lists/defaultdicts and a probability-distribution scorer; ordinary label accuracy is not valid. |
| JanosAudran/financial-reports-sec | classification | macro_f1 | context_required | No evaluator change is required for one horizon, but Codex must choose the horizon and explain the return-direction labels. |
| McAuley-Lab/Amazon-Reviews-2023 | classification | macro_f1 | dataset_access_blocker | Add a config-aware parquet/raw-data fallback if the legacy repository cannot load; do not scan every Amazon category. |
| ParlAI/blended_skill_talk | dialogue_generation | rouge_l | dedicated_adapter | Add a dialogue-turn flattener and define which response stream is gold; whole-list generation is not meaningful. |
| allenai/peer_read | classification | macro_f1 | dataset_access_blocker | Repair legacy-script loading or use a maintained mirror, then recover the selected target's label semantics. |
| ashraq/financial-news-articles | summarization | rouge_l | context_required | No classification gold exists; use a generative model because the highest-ranked fixed classifiers are task-mismatched. |
| bigbio/mediqa_qa | qa | qa_f1 | context_required | Existing tagged QA supports this after selecting the BigBio QA config; avoid GGUF/unsupported quantized candidates. |
| bigbio/n2c2_2018_track2 | relation_extraction | set_f1 | dedicated_adapter | Add normalized head-relation-tail extraction/scoring; NER checkpoints alone cannot predict medication relations. |
| bigbio/ncbi_disease | token_classification | set_f1 | context_required | The current BigBio entity adapter is sufficient, but choose a disease NER checkpoint rather than CHEM-only PharmaDetect models. |
| biglam/gutenberg-poetry-corpus | language_modeling | perplexity | reusable_evaluator_change | Add causal/seq2seq perplexity or a documented continuation construction; ROUGE against the input would be invalid. |
| bitext/Bitext-customer-support-llm-chatbot-training-dataset | classification | macro_f1 | generic_supported | Current finite-label generation or zero-shot NLI works; Codex must choose intent versus coarse category. |
| ccdv/govreport-summarization | summarization | rouge_l | generic_supported | Current tagged/seq2seq summarization works; long-input truncation must remain visible in the audit. |
| climatebert/climate_commitments_actions | classification | macro_f1 | generic_supported | Current zero-shot NLI and label-map handling are sufficient. |
| climatebert/environmental_claims | classification | macro_f1 | generic_supported | Current zero-shot NLI and label-map handling are sufficient. |
| coastalcph/lex_glue | multiple_choice | accuracy | context_required | No new adapter is needed for case_hold; Codex must not mix configs and must reject incompatible model heads. |
| community-datasets/yahoo_answers_topics | classification | macro_f1 | generic_supported | Current prompt templates and finite labels support this task. |
| cornell-movie-dialog/cornell_movie_dialog | dialogue_generation | rouge_l | dedicated_adapter | Add a legacy loader fallback and reusable turn flattener; evaluating serialized conversations directly is invalid. |
| cornell-movie-review-data/rotten_tomatoes | classification | accuracy | generic_supported | Current classification protocols support this directly. |
| deepmind/aqua_rat | multiple_choice | accuracy | generic_supported | Current structured-choice scoring supports this directly. |
| deepmind/math_dataset | numeric_qa | numeric_match | dataset_access_blocker | Repair legacy generator loading and verify symbolic answers; numeric_match alone may need exact-match fallback per config. |
| embedding-data/Amazon-QA | qa | qa_f1 | context_required | Current best-of-reference QA F1 works, but the dataset has train-only retrieval pairs and is not a clean held-out benchmark. |
| facebook/empathetic_dialogues | dialogue_generation | rouge_l | dedicated_adapter | Add dialogue flattening and a maintained loader fallback; candidate chat models must be normal Transformers checkpoints. |
| fancyzhx/amazon_polarity | classification | accuracy | generic_supported | Current zero-shot NLI or tagged finite-label generation works. |
| fancyzhx/dbpedia_14 | classification | accuracy | generic_supported | Current finite-label classification works with ClassLabel names. |
| fancyzhx/yelp_polarity | classification | accuracy | generic_supported | Current classification protocols support this directly. |
| google-research-datasets/paws | classification | accuracy | generic_supported | Current pair-text prompt plus finite-label protocols work; sentence encoders remain a proxy rather than a trained head. |
| google-research-datasets/poem_sentiment | classification | macro_f1 | generic_supported | Current finite-label classification works after preserving the dataset label names. |
| google-research-datasets/taskmaster1 | dialogue_generation | rouge_l | dedicated_adapter | Add legacy loading plus a reusable task-oriented-dialogue flattener. |
| google/civil_comments | multilabel_classification | set_f1 | reusable_evaluator_change | Add declared derived targets/thresholds and multi-column gold construction; current one-column label extraction is insufficient. |
| griffin/ChemSum | summarization | rouge_l | generic_supported | Current summarization works, with explicit truncation reporting for very long articles. |
| gtfintechlab/finer-ord | token_classification | set_f1 | reusable_evaluator_change | Add row-grouping into token/BIO examples; the current evaluator assumes one complete text example per row. |
| james-burton/wine_reviews | classification | macro_f1 | context_required | Current classification works for variety, but point prediction requires a future regression metric. |
| joelniklaus/legal_case_document_summarization | summarization | rouge_l | generic_supported | Current summarization works; record context truncation because judgments are long. |
| knkarthick/samsum | summarization | rouge_l | generic_supported | Current summarization works directly. |
| launch/gov_report_qs | qa | qa_f1 | dedicated_adapter | Add a nested aligned-array flattener that emits one QA example per question-summary pair. |
| microsoft/orca-math-word-problems-200k | numeric_qa | numeric_match | context_required | Current numeric scoring works when a final number exists; route nonnumeric answers to exact/QA matching. |
| midas/inspec | token_classification | set_f1 | reusable_evaluator_change | Add generic pretokenized-sequence input and BIO-gold conversion. |
| midas/semeval2017 | token_classification | set_f1 | reusable_evaluator_change | Reuse the same pretokenized-sequence/BIO-gold adapter as Inspec. |
| nlpaueb/finer-139 | token_classification | set_f1 | reusable_evaluator_change | Reuse a generic token-list/BIO adapter; verify checkpoint ontology against 139 financial labels. |
| nyu-mll/multi_nli | classification | accuracy | generic_supported | Current direct NLI classifier protocol supports this cleanly. |
| odegiber/hate_speech18 | classification | macro_f1 | dataset_access_blocker | Add a maintained raw-data/mirror fallback if the repository script cannot load. |
| raquiba/Sarcasm_News_Headline | classification | accuracy | generic_supported | Current finite-label classification works; the headline-cause model may have mismatched output semantics. |
| sentence-transformers/eli5 | qa | qa_f1 | context_required | Current tagged QA can run, but lexical F1 is a weak quality proxy and ALBERT masked checkpoints are task-mismatched. |
| sentence-transformers/yahoo-answers | qa | qa_f1 | context_required | Current tagged QA works; document the train-only retrieval-oriented nature of the dataset. |
| sileod/movie_recommendation | multiple_choice | accuracy | generic_supported | Current separate-choice-column scoring works; reject ShotVL/video or GGUF candidates for this text-only task. |
| tals/vitaminc | classification | accuracy | generic_supported | Current NLI protocol supports this directly. |
| ucberkeley-dlab/measuring-hate-speech | regression | correlation | dedicated_adapter | Add grouped-example preprocessing plus regression metrics such as MAE/Spearman; row-level classification would be misleading. |
| ucsbnlp/liar | classification | macro_f1 | dataset_access_blocker | The Hub dataset endpoint is unavailable; use a maintained LIAR mirror or explicit raw-file fallback before evaluation. |
| virattt/financial-qa-10K | qa | qa_f1 | context_required | Current tagged QA works; choose a generative model because generic ALBERT/embedding checkpoints do not extract answers. |
| zeroshot/twitter-financial-news-sentiment | classification | macro_f1 | generic_supported | Current zero-shot NLI works; ABSA or ONNX candidates require label/format compatibility checks. |

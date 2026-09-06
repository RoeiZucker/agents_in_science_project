# Evaluation Coverage Audit

Reviewed 40 dataset-model pairs across 20 datasets.
This is a design audit, not an inference result. Support claims still require smoke tests on the target cluster.

## Support Summary

| Support level | Pairs |
|---|---:|
| context_required | 13 |
| generic_supported | 19 |
| model_task_mismatch | 8 |

## Pair Review

| Dataset | Model | Task | Method | Support | Remaining work |
|---|---|---|---|---|---|
| Amod/mental_health_counseling_conversations | nvidia/Llama3-ChatQA-2-8B | generation | rouge_l | context_required | Generic causal generation is supported; ROUGE-L is only a lexical proxy for counseling responses. |
| Amod/mental_health_counseling_conversations | NousResearch/Llama-2-7b-chat-hf | generation | rouge_l | context_required | Generic causal generation is supported; ROUGE-L is only a lexical proxy for counseling responses. |
| CFPB/consumer-finance-complaints | MoritzLaurer/deberta-v3-large-zeroshot-v1.1-all-33 | classification | macro_f1 | generic_supported | Use zero-shot NLI because classifier labels are entailment labels. |
| CFPB/consumer-finance-complaints | ChanceFocus/finma-7b-full | classification | macro_f1 | context_required | Generate one product label; prompt must list the complete product label set. |
| FinGPT/fingpt-sentiment-train | IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment | classification | macro_f1 | context_required | Direct classification requires semantic alignment between checkpoint labels and FinGPT labels. |
| FinGPT/fingpt-sentiment-train | NousResearch/Hermes-2-Pro-Mistral-7B | classification | macro_f1 | generic_supported | Tagged finite-label generation is supported. |
| HUPD/hupd | unsloth/DeepSeek-R1-Distill-Qwen-14B | generation | rouge_l | context_required | Supported only after Codex selects a coherent config, columns, and task. |
| HUPD/hupd | sdadas/mmlw-roberta-base | generation | rouge_l | model_task_mismatch | A fixed-label classifier cannot perform title or abstract generation; use only if a compatible classification config is chosen. |
| MU-NLPC/Calc-asdiv_a | llm-agents/tora-code-7b-v1.0 | numeric_qa | numeric_match | generic_supported | Tagged generation plus numeric equivalence is supported. |
| MU-NLPC/Calc-asdiv_a | microsoft/rho-math-1b-interpreter-v0.1 | numeric_qa | numeric_match | generic_supported | Tagged generation plus numeric equivalence is supported. |
| RobZamp/sick | tasksource/ModernBERT-large-nli | classification | accuracy | generic_supported | Direct NLI labels should align with the dataset labels. |
| RobZamp/sick | cross-encoder/nli-deberta-v3-base | classification | accuracy | generic_supported | Direct NLI labels should align with the dataset labels. |
| SemEvalWorkshop/hyperpartisan_news_detection | cardiffnlp/twitter-roberta-base-hate-latest | classification | macro_f1 | model_task_mismatch | Hate-speech output labels do not represent hyperpartisan labels. |
| SemEvalWorkshop/hyperpartisan_news_detection | MoritzLaurer/deberta-v3-large-zeroshot-v2.0 | classification | macro_f1 | generic_supported | Use zero-shot NLI with explicit hyperpartisan label wording. |
| SetFit/bbc-news | dima806/news-category-classifier-distilbert | classification | macro_f1 | context_required | Direct scoring is valid only if the checkpoint category names cover BBC labels. |
| SetFit/bbc-news | sentence-transformers/sentence-t5-base | classification | macro_f1 | generic_supported | Use zero-shot normalized text-to-label embedding similarity; this is not a trained classification head. |
| TheFinAI/fiqa-sentiment-classification | gauneg/roberta-base-absa-ate-sentiment | classification | macro_f1 | context_required | Verify aspect-sentiment label semantics before direct classifier scoring. |
| TheFinAI/fiqa-sentiment-classification | CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment | classification | macro_f1 | context_required | Verify language and label alignment before direct classifier scoring. |
| ade-benchmark-corpus/ade_corpus_v2 | PlanTL-GOB-ES/roberta-base-biomedical-clinical-es | classification | macro_f1 | generic_supported | Use masked-label pseudo-likelihood; Spanish pretraining remains a transfer-quality risk for English ADE text. |
| ade-benchmark-corpus/ade_corpus_v2 | tomaarsen/span-marker-bert-base-uncased-acronyms | classification | macro_f1 | model_task_mismatch | Acronym span extraction does not answer ADE sentence classification or ADE relations. |
| allenai/cosmos_qa | soniox/Soniox-7B-v1.0 | multiple_choice | accuracy | generic_supported | Separate-choice-column support removes the former dataset-specific gap. |
| allenai/cosmos_qa | context-labs/Meta-Llama-3.1-8B-Instruct-FP16 | multiple_choice | accuracy | generic_supported | Separate-choice-column support removes the former dataset-specific gap. |
| allenai/quartz | albert/albert-xxlarge-v1 | multiple_choice | accuracy | generic_supported | Use masked-choice pseudo-likelihood over the Quartz answer texts. |
| allenai/quartz | Qwen/Qwen2-7B | multiple_choice | accuracy | generic_supported | Generic causal answer-choice likelihood is supported. |
| bigbio/bc5cdr | OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M | token_classification | set_f1 | generic_supported | BIO span reconstruction and BigBio conversion are supported; this CHEM-only ontology cannot predict BC5CDR diseases. |
| bigbio/bc5cdr | OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M | token_classification | set_f1 | generic_supported | BIO span reconstruction and BigBio conversion are supported; this CHEM-only ontology cannot predict BC5CDR diseases. |
| bigbio/bioasq_task_b | johnsnowlabs/JSL-MedMNX-7B-v2.0 | qa | qa_f1 | context_required | Generic tagged QA works once nested columns and answer type are specified. |
| bigbio/bioasq_task_b | axiong/PMC_LLaMA_13B | qa | qa_f1 | context_required | Generic tagged QA works once nested columns and answer type are specified. |
| bigbio/chemprot | OpenMed/OpenMed-NER-PharmaDetect-BigMed-278M | relation_extraction | set_f1 | model_task_mismatch | An NER checkpoint detects entities but does not classify chemical-protein relations. |
| bigbio/chemprot | OpenMed/OpenMed-NER-PharmaDetect-BioMed-109M | relation_extraction | set_f1 | model_task_mismatch | An NER checkpoint detects entities but does not classify chemical-protein relations. |
| ccdv/WCEP-10 | THUDM/glm-10b | summarization | rouge_l | context_required | Generic generation is supported if remote model code loads under the cluster environment. |
| ccdv/WCEP-10 | google/byt5-xxl | summarization | rouge_l | generic_supported | Generic seq2seq summarization with ROUGE-L is supported. |
| ccdv/pubmed-summarization | THUDM/glm-10b | summarization | rouge_l | context_required | Generic generation is supported if remote model code loads under the cluster environment. |
| ccdv/pubmed-summarization | nvidia/Llama3-ChatQA-2-8B | summarization | rouge_l | generic_supported | Generic tagged summarization with ROUGE-L is supported. |
| coastalcph/multi_eurlex | openGPT-X/Teuken-7B-instruct-research-v0.4 | multilabel_classification | set_f1 | generic_supported | Tagged label-set generation and set F1 are supported. |
| coastalcph/multi_eurlex | ReliableAI/UCCIX-Llama2-13B-Instruct | multilabel_classification | set_f1 | generic_supported | Tagged label-set generation and set F1 are supported. |
| google-research-datasets/go_emotions | cirimus/modernbert-base-emotions | multilabel_classification | set_f1 | model_task_mismatch | The config declares single-label classification with 7 labels, not GoEmotions multilabel prediction. |
| google-research-datasets/go_emotions | cardiffnlp/twitter-roberta-base-emotion-multilabel-latest | multilabel_classification | set_f1 | model_task_mismatch | The config is multilabel but its 11-label ontology does not cover the 28 GoEmotions labels. |
| theatticusproject/cuad | sdadas/mmlw-roberta-base | qa | qa_f1 | model_task_mismatch | A fixed-label sentence model cannot extract free-text contract spans. |
| theatticusproject/cuad | yandex/YandexGPT-5-Lite-8B-instruct | qa | qa_f1 | context_required | Generic tagged QA works once nested CUAD columns are selected. |

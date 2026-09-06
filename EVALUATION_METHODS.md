# Evaluation Method Contract

The Codex runner is a metadata scout. It may inspect dataset/model cards and schemas, then choose the dataset config, labeled split, task family, columns, prompt template, label map, and one registered evaluation method. It must not load model weights, run inference, invent evaluator code, or report model performance. The outer Python process validates the scout JSON before any model evaluation starts.

## Registered Methods

| Method | Intended tasks | Score |
|---|---|---|
| `accuracy` | single-label classification, image classification, multiple choice | fraction exactly correct |
| `macro_f1` | imbalanced single-label classification | unweighted mean F1 across observed labels |
| `exact_match` | constrained generation where normalized equality is meaningful | fraction exactly equal |
| `qa_f1` | short-answer and extractive QA | mean token-overlap F1, best of multiple references |
| `numeric_match` | numeric QA | exact numeric equivalence, including decimals and percentages |
| `rouge_l` | summarization and lexical generation proxies | mean ROUGE-L F1 |
| `set_f1` | multilabel classification, entity sets, relation sets | mean example-level set F1 |

`auto` chooses the task default. An explicit method is accepted only if it is registered and compatible with the selected task. This keeps method selection flexible while preserving deterministic, comparable scoring.

## Codex Context

The scout writes `codex_context.json`. Relevant `dataset_context` fields are:

- `subset` and `split`: chosen before dataset inspection, so multi-config datasets can be loaded correctly.
- `question_column`, `answer_column`, `choices_column`, `choices_columns`, and `image_column`: existing columns or dotted paths.
- `task`: `generation`, `classification`, `multilabel_classification`, `multiple_choice`, `qa`, `numeric_qa`, `summarization`, `token_classification`, `relation_extraction`, or `image_classification`.
- `evaluation_method`: `auto` or one registered method above.
- `label_map` and `prompt_template`: finite label semantics and model input context.

All generative answers are requested inside `<answer>...</answer>`. Classification produces one label, multilabel tasks produce a JSON array, and free-text tasks preserve the text inside the tags for the selected scorer. This tagged generation path is the default.

Direct classifier logits, encoder similarities, masked-token likelihoods, token-classifier scores, multiple-choice continuation likelihoods, and other label-score protocols are disabled by default. Pass `--allow-label-scores` explicitly at the chosen CLI entrypoint to permit them; the flag is propagated through condition cycles, Slurm array tasks, retries, and the selection loop. Models that require these protocols are otherwise reported as unsupported rather than silently evaluated by a different contract. Tagged Qwen3 plans also pass `--disable-thinking`, which renders the tokenizer chat template with `enable_thinking=False` so constrained answers are not displaced by reasoning tokens.

## Adapter Protocols

The evaluator also supports three encoder-only families:

- `embedding_label_similarity` mean-pools and normalizes sentence-encoder outputs, then chooses the closest finite label description. This is a zero-shot proxy, not a trained classifier or prototype fit.
- `masked_label_likelihood` and `masked_choice_likelihood` rank labels or answer texts by mean pseudo-log-likelihood, masking one candidate token at a time.
- `token_entity_set_f1` reconstructs BIO spans across overflow windows, normalizes entity type/text pairs, and scores example-level set F1.

These adapters make the model format executable; they do not repair semantic mismatch. In particular, the OpenMed PharmaDetect checkpoints expose only `CHEM` labels, while BC5CDR also contains disease entities. Their BC5CDR score therefore measures a chemical-only model against the full requested entity set and will count diseases as false negatives.

## Remaining Boundaries

Specialized relation extraction and standalone image encoders still need dedicated loaders. A dedicated adapter cannot make an NER checkpoint predict relations, make a fixed-label sentiment model extract a contract answer span, or make a checkpoint trained in another language understand the target data.


See `EVALUATION_COVERAGE_AUDIT.md` and `config/evaluation_coverage_40.csv` for the reviewed 40-pair sample and the distinction between context needs, adapter gaps, and model-task mismatches.

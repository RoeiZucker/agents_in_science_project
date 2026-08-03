# Evaluation Agent

This project now has a small evaluation agent around `evaluate_hf_pair.py`.
It inspects a Hugging Face dataset and one or more models, chooses evaluator
arguments, runs smoke tests, audits outputs, and writes CSV/JSON summaries.

## Files

- `run_eval_agent.py` - main orchestrator.
- `eval_agent_core.py` - shared planning, inspection, command, and audit helpers.
- `inspect_hf_dataset.py` - prints dataset schema/split/task inspection JSON.
- `inspect_hf_model.py` - prints model config/capability inspection JSON.
- `audit_eval_results.py` - audits an existing `summary.json`/`predictions.jsonl`.
- `tests/test_eval_agent_core.py` - offline unit tests for planning/audit helpers.

## Typical Usage

Plan only, with no model loading:

```bash
HF_HOME=/sci/labs/michall/roeizucker/agents_project/.hf_cache \
./.venv-artifact-linker/bin/python run_eval_agent.py \
  --dataset tau/commonsense_qa \
  --models TinyLlama/TinyLlama-1.1B-Chat-v0.1 google/flan-t5-xl \
  --split auto \
  --stage plan \
  --project-root /sci/labs/michall/roeizucker/agents_project
```

Run smoke tests only:

```bash
HF_HOME=/sci/labs/michall/roeizucker/agents_project/.hf_cache \
./.venv-artifact-linker/bin/python run_eval_agent.py \
  --dataset tau/commonsense_qa \
  --models TinyLlama/TinyLlama-1.1B-Chat-v0.1 \
  --split auto \
  --stage smoke \
  --smoke-limit 3 \
  --project-root /sci/labs/michall/roeizucker/agents_project
```

Run smoke test and then full evaluation:

```bash
HF_HOME=/sci/labs/michall/roeizucker/agents_project/.hf_cache \
./.venv-artifact-linker/bin/python run_eval_agent.py \
  --dataset xai-org/RealworldQA \
  --models OpenGVLab/InternVL3_5-2B ZTE-AIM/3B-Curr-ReFT \
  --split test \
  --stage full \
  --smoke-limit 3 \
  --project-root /sci/labs/michall/roeizucker/agents_project \
  --continue-on-error
```

## Outputs

Each run writes a run directory under:

```text
<output-root>/<dataset>_<split>/
```

The default `output-root` is:

```text
/sci/labs/michall/roeizucker/agents_project/eval_results/_agent_runs
```

Important files:

- `dataset_inspection.json` - detected split/schema/task/columns.
- `plans.json` - exact smoke/full evaluator commands for each model.
- `results.csv` - compact table: dataset, split, model, score, status, notes.
- `failures.json` - failed model stages, if any.
- `logs/*.command.txt` - exact command that was run.
- `logs/*.stdout.txt` and `logs/*.stderr.txt` - child process logs.
- `audits/*_audit.json` - audit results for smoke/full outputs.

Model outputs are written under:

```text
<output-root>/eval_results/<model>/<dataset>_<split>/
```

Smoke outputs are written under:

```text
<output-root>/eval_results/_smoke/<model>/<dataset>_<split>/
```

## Built-In Decisions

The agent currently:

- Auto-selects a labeled split when `--split auto` is used.
- Detects multiple-choice datasets from `choices`/`options` columns.
- Detects common answer columns like `answer` and `answerKey`.
- Detects image columns and vision-language model families.
- Uses `answer_text` scoring with length normalization for causal LMs on
  multiple-choice tasks to reduce label-token bias.
- Uses eager attention for known VLM compatibility paths.
- Detects Qwen2.5-VL-style `use_cache: null` configs and lets the evaluator
  create sanitized local snapshots.
- Audits missing labels, blank predictions, and degenerate prediction
  distributions.

## Tests

Offline unit tests:

```bash
./.venv-artifact-linker/bin/python -m unittest tests.test_eval_agent_core -v
```

Syntax checks:

```bash
python3 -m py_compile \
  evaluate_hf_pair.py \
  eval_agent_core.py \
  inspect_hf_dataset.py \
  inspect_hf_model.py \
  audit_eval_results.py \
  run_eval_agent.py \
  tests/test_eval_agent_core.py
```


## Running Through Codex Non-Interactively

Codex can receive the task prompt directly from the terminal. To generate a
no-follow-up prompt:

```bash
python make_codex_prompt.py \
  --dataset google/boolq \
  --models google/flan-t5-small google/flan-t5-base \
  --split validation \
  --stage full \
  --output-file codex_prompt.txt
```

Then run Codex non-interactively:

```bash
codex exec \
  --cd "$PWD" \
  --ask-for-approval never \
  "$(cat codex_prompt.txt)"
```

You can also pass the generated prompt directly:

```bash
codex exec --cd "$PWD" --ask-for-approval never "$(
  python make_codex_prompt.py \
    --dataset google/boolq \
    --models google/flan-t5-small google/flan-t5-base \
    --split validation \
    --stage full
)"
```

The prompt explicitly instructs Codex not to ask follow-up questions and to use
the local `plan -> smoke -> full` workflow.

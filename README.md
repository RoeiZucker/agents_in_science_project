# Evaluation Agent

This project now has a small evaluation agent around `evaluate_hf_pair.py`.
It inspects a Hugging Face dataset and one or more models, chooses evaluator
arguments, runs smoke tests, audits outputs, and writes CSV/JSON summaries.

## Files

- `run_eval_agent.py` - main orchestrator.
- `run_evaluation_conditions.py` - batch runner for partner-provided dataset/model CSVs.
- `download_condition_datasets.py` - pre-downloads condition datasets into the same Hugging Face cache used by the batch runner.
- `delete_condition_datasets.py` - deletes warmed dataset caches after evaluation.
- `run_condition_dataset_cycle.py` - loops through selected datasets as download -> evaluate -> delete.
- `eval_agent_core.py` - shared planning, inspection, command, and audit helpers.
- `inspect_hf_dataset.py` - prints dataset schema/split/task inspection JSON.
- `inspect_hf_model.py` - prints model config/capability inspection JSON.
- `audit_eval_results.py` - audits an existing `summary.json`/`predictions.jsonl`.
- `analyze_eval_errors.py` - refinement agent step that summarizes failures.
- `make_retrieval_feedback.py` - writes the next retrieval-agent feedback request.
- `refinement_agent_core.py` - shared refinement/error-analysis helpers.
- `mock_candidates/*.json` - mocked retrieval-agent candidate outputs.
- `tests/test_eval_agent_core.py` - offline unit tests for planning/audit helpers.
- `tests/test_refinement_agent_core.py` - offline unit tests for refinement helpers.

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
- `error_analysis.json` - per-model failure modes and cross-model comparisons.
- `retrieval_feedback.json` - structured feedback for the next retrieval round.
- `retrieval_feedback_prompt.txt` - prompt-style handoff to the retrieval agent.

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
./.venv-artifact-linker/bin/python -m unittest tests.test_refinement_agent_core -v
./.venv-artifact-linker/bin/python -m unittest tests.test_run_evaluation_conditions -v
```

Syntax checks:

```bash
python3 -m py_compile \
  evaluate_hf_pair.py \
  eval_agent_core.py \
  inspect_hf_dataset.py \
  inspect_hf_model.py \
  audit_eval_results.py \
  analyze_eval_errors.py \
  make_retrieval_feedback.py \
  refinement_agent_core.py \
  run_eval_agent.py \
  run_evaluation_conditions.py \
  tests/test_eval_agent_core.py \
  tests/test_refinement_agent_core.py \
  tests/test_run_evaluation_conditions.py
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
codex -C "$PWD" -a never exec "$(cat codex_prompt.txt)"
```

You can also pass the generated prompt directly:

```bash
codex -C "$PWD" -a never exec "$(
  python make_codex_prompt.py \
    --dataset google/boolq \
    --models google/flan-t5-small google/flan-t5-base \
    --split validation \
    --stage full
)"
```

The prompt explicitly instructs Codex not to ask follow-up questions and to use
the local `plan -> smoke -> full` workflow.


## Batch Conditions CSV

To evaluate every pair from a partner-provided CSV, use
`run_evaluation_conditions.py`. It groups rows by `query_dataset`, evaluates the
unique models for each dataset, writes joined results, runs refinement by
default, and can delete downloaded Hugging Face caches after each dataset.

Pre-download datasets outside nested Codex when dataset loading needs external
hosts such as Google Drive:

```bash
python download_condition_datasets.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root runtime \
  --trust-remote-code
```

Run the Codex-backed evaluation against the same `runtime/.hf_cache` and
`runtime/.hf_datasets_cache` directories:

```bash
python run_evaluation_conditions.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root runtime \
  --stage full \
  --runner codex \
  --trust-remote-code \
  --fresh-run-dir \
  --cleanup-cache after-dataset
```

Delete warmed dataset caches later:

```bash
python delete_condition_datasets.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root runtime
```

For the lowest network/cache risk, run the full flow one dataset at a time:

```bash
python run_condition_dataset_cycle.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root runtime \
  --stage smoke \
  --runner codex \
  --trust-remote-code \
  --fresh-run-dir
```

That cycle runs `download_condition_datasets.py --dataset ...`, then
`run_evaluation_conditions.py --dataset ... --cleanup-cache none`, then
`delete_condition_datasets.py --dataset ...` before moving to the next dataset.

Useful smoke-test command before a long run:

```bash
python run_evaluation_conditions.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root runtime \
  --stage smoke \
  --limit-datasets 1 \
  --runner codex \
  --trust-remote-code \
  --fresh-run-dir \
  --cleanup-cache after-dataset
```

Some Hugging Face datasets, including `ImperialCollegeLondon/health_fact`, require `--trust-remote-code` because they use a dataset loading script. Only use this for dataset repositories you trust.

When `--runner codex` is used, the batch runner creates one Codex prompt per
dataset group, saves it as `<dataset>_<split>/codex_prompt.txt`, and invokes:

```bash
codex -C <repo> -a never exec <prompt>
```

Codex then runs the evaluation scripts, inspects logs/results, can make small
evaluator fixes if needed, and produces the same output files. Use `--runner
script` to bypass Codex and run the deterministic Python-only path.

Use `--fresh-run-dir` for reruns or small subsets, so old per-dataset outputs do not get counted as current failures.

Batch outputs are written under:

```text
<project-root>/eval_results/_condition_runs/
```

Important batch files:

- `batch_results.csv` - original CSV rows joined with measured eval score/status.
- `batch_failures.json` - failed dataset-level commands.
- `<dataset>_<split>/candidate_models_from_conditions.json` - retrieval metadata from the CSV.
- `<dataset>_<split>/results.csv` - evaluation-agent scores for unique models.
- `<dataset>_<split>/error_analysis.json` - refinement analysis, unless `--skip-refinement` is used.
- `<dataset>_<split>/retrieval_feedback.json` - next retrieval request, unless `--skip-refinement` is used.

Cleanup only removes downloaded cache directories:

```text
<project-root>/.hf_cache/hub
<project-root>/.hf_cache/transformers
<project-root>/.hf_datasets_cache
```

It does not delete `eval_results`. Use `--cleanup-cache none` to keep models and
datasets cached between groups, or `--cleanup-cache end` to delete them only once
after the whole batch finishes.

## Refinement Agent

After an evaluation run finishes, the refinement agent analyzes errors and
prepares feedback for the next retrieval round. The retrieval agent can be
mocked with files under `mock_candidates/` until the real retrieval component is
connected.

Analyze an existing run:

```bash
python analyze_eval_errors.py \
  --run-dir runtime/eval_results/_agent_runs/google_boolq_validation \
  --candidate-models mock_candidates/boolq_round1.json
```

Create retrieval feedback from that analysis:

```bash
python make_retrieval_feedback.py \
  --error-analysis runtime/eval_results/_agent_runs/google_boolq_validation/error_analysis.json \
  --next-round 2
```

The expected retrieval-agent handoff format is:

```json
{
  "candidates": [
    {
      "model": "google/flan-t5-base",
      "rank": 1,
      "score": 0.87,
      "source": "retrieval_agent",
      "reason": "Why this model should fit the dataset/task."
    }
  ]
}
```

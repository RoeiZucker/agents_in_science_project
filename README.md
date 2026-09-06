# Hugging Face Evaluation And Selection Loop

This repository evaluates partner-recommended Hugging Face models and uses the measured outputs to choose the best valid model for each dataset. The runtime loop is candidate handoff -> metadata context -> deterministic evaluation -> error/refinement feedback -> optional later candidate round -> measured winner. Full pipeline runs are deliberately restricted to a dataset allowlist; report and analysis scripts can still inspect any existing run.

## Ownership

This repository owns:

- dataset/model inspection and evaluation protocols;
- smoke/full execution, auditing, failure semantics, and reports;
- refinement feedback and the multi-round loop controller;
- selection of the highest valid measured score;
- runtime context-scout and development-review agent descriptions.

The partner owns candidate retrieval/model recommendation from new context. Oren's implementation is included as the pinned `external/artifact-linker` submodule; this repository owns the provider integration. Files under `mock_candidates/` are static contract examples with `mock: true`.

## Architecture

1. `run_selection_loop.py` reads candidate handoffs listed in a manifest.
2. With `--runner codex`, Codex only inspects cards/metadata and writes `codex_context.json`. It may choose a validated task and evaluation method, but it never runs a model.
3. `run_evaluation_conditions.py` and `run_eval_agent.py` run the deterministic evaluator. Generative models use `<answer>...</answer>` by default; direct label scores require the explicit `--allow-label-scores` opt-in.
4. `analyze_eval_errors.py` and `make_retrieval_feedback.py` produce feedback for a later partner retrieval round.
5. The loop selects a winner only from rows with `status=ok`, a finite score, and at least one labeled example.

For direct conditions-CSV evaluation, `--runner manual --context-file ...` supplies
the validated context explicitly and does not launch Codex.

Unsupported, missing, unlabeled, warning-only, or non-finite rows are failures, not scores.

## Important Files

- `run_selection_loop.py`: larger multi-round loop and measured winner selection.
- `config/mock_selection_loop.json`: two-dataset mock loop manifest.
- `config/full_pipeline_datasets.txt`: full-run dataset allowlist.
- `run_condition_dataset_cycle.py`: legacy conditions-CSV download/evaluate/delete cycle.
- `run_evaluation_conditions.py`: evaluates one conditions CSV in dataset groups.
- `run_eval_agent.py`: plans, smoke-tests, fully evaluates, and audits model pairs.
- `evaluate_hf_pair.py`: deterministic model/dataset evaluator.
- `EVALUATION_METHODS.md`: Context Scout freedom, registered scoring methods, and adapter boundaries.
- `EVALUATION_COVERAGE_AUDIT.md`: readable review of 40 pairs across 20 datasets.
- `config/evaluation_coverage_40.csv`: machine-readable form of the coverage review.
- `build_evaluation_coverage_audit.py`: reproducibly rebuilds both audit files.
- `result_contract.py`: shared success/failure rules.
- `create_condition_run_report.py`: Markdown and CSV report generator.
- `retry_failed_condition_pairs.py`: pair-level retry runner and CSV/JSON output.
- `agents/`: development-review roles and runtime-agent boundaries.

## Setup

Clone with the pinned Oren retrieval repository:

```bash
git clone --recurse-submodules https://github.com/RoeiZucker/agents_in_science_project.git
```

For an existing clone, initialize it once with:

```bash
git submodule update --init --recursive
```

The submodule tracks Oren's source plus committed descriptions and embeddings. The large ArtifactBench graph and trained GNN checkpoint remain external runtime assets; follow `external/artifact-linker/retrieval_agent/method2_gnn_inference/README_INSTRUCTIONS.txt` when setting up a new machine.

From the repository after activating the project environment:

```bash
cd /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent
python -m unittest discover -s tests -v
```

The standard adapters need only `requirements.txt`. Text-only GGUF checkpoints use
the optional llama.cpp backend:

```bash
pip install -r requirements-gguf.txt
```

For a CUDA-enabled build on the cluster:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade --force-reinstall -r requirements-gguf.txt
```

GGUF repositories are detected automatically. The evaluator prefers Q4_K_M and
downloads every shard when that quantization is split. Use `--gguf-file` to select
a different file. Repositories containing `mmproj` or projector GGUF files are
rejected because this adapter is text-only.

```bash
python evaluate_hf_pair.py \
  --dataset ImperialCollegeLondon/health_fact \
  --model bartowski/Qwen2.5-7B-Instruct-GGUF \
  --model-type gguf \
  --gguf-file Qwen2.5-7B-Instruct-Q4_K_M.gguf \
  --task classification --split validation \
  --question-column claim --answer-column label \
  --label-map '{"0":"false","1":"mixture","2":"true","3":"unproven"}' \
  --evaluation-protocol tagged_label_generation_accuracy \
  --evaluation-method accuracy \
  --output-dir eval_results/qwen_gguf_health_fact \
  --limit 3
```

## Safe Wiring Check

This validates the two-dataset mock loop without loading model weights:

```bash
python run_selection_loop.py   --manifest config/mock_selection_loop.json   --dataset-subset-file config/full_pipeline_datasets.txt   --stage plan   --runner script   --project-root /sci/labs/michall/roeizucker/hf_eval_runtime
```

Plan mode validates the partner handoff and writes a `planned` selected row for
each dataset. It does not claim a measured winner or count the dataset as failed.

## Full Subset Run

A full selection-loop run requires `--dataset-subset-file`; the command fails without it. Codex is the metadata-only context scout, then Python performs evaluation.

```bash
python run_selection_loop.py   --manifest config/mock_selection_loop.json   --dataset-subset-file config/full_pipeline_datasets.txt   --stage full   --runner codex   --codex-bypass-sandbox   --trust-remote-code   --project-root /sci/labs/michall/roeizucker/hf_eval_runtime   --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_selection_loop_full
```

The loop keeps Hugging Face caches. That avoids repeat downloads while the pipeline is stabilizing.

## Full Conditions-CSV Cycle

Use this older entrypoint when the partner provides `evaluation_conditions.csv`.
Full runs require `--dataset-subset-file`; plan and smoke runs may omit it.

```bash
python run_condition_dataset_cycle.py   --conditions-csv ../evaluation_conditions.csv   --dataset-subset-file config/full_pipeline_datasets.txt   --project-root /sci/labs/michall/roeizucker/hf_eval_runtime   --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_condition_runs_subset   --stage full   --runner codex   --codex-bypass-sandbox   --trust-remote-code   --keep-dataset-cache   --keep-model-cache   --fresh-run-dir
```

Remove the two `--keep-*-cache` flags only when deletion is intentionally reintroduced.

## Manual Evaluation Context

Use manual mode when you want to provide the dataset columns, label meanings, task,
and `prompt_template` yourself instead of asking the Codex Context Scout. The JSON
must follow the exact schema in `agents/pipeline/context-scout.md`; its `dataset`
and ordered `models` list must exactly match the selected rows. One context file is
limited to one selected dataset.

```bash
python run_condition_dataset_cycle.py \
  --conditions-csv ../evaluation_conditions.csv \
  --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_manual_context_run \
  --dataset ImperialCollegeLondon/health_fact \
  --limit-pairs 2 \
  --stage smoke \
  --runner manual \
  --context-file /path/to/health_fact_context.json \
  --trust-remote-code \
  --keep-dataset-cache \
  --keep-model-cache \
  --fresh-run-dir
```

The run preserves the validated input as `manual_context.json`, records its source
path in `manual_context_source.json`, and reports `context_source=manual` in
`batch_contexts.json`. A manual `prompt_template` only affects evaluator protocols
that render that template; it does not alter a model-specific protocol such as the
current zero-shot NLI scoring path.

## Outputs

`run_selection_loop.py` writes:

- `selection_loop_results.csv`: every candidate attempt with complete handoff provenance, protocol, score, and outcome;
- `selected_models.csv`: one measured winner, planned candidate, or explicit failure row per selected dataset, retaining complete handoff provenance;
- `selection_loop_summary.json`: aggregate counts;
- `<dataset>/round_<n>/conditions.csv`: partner handoff adapted to the evaluator contract;
- `<dataset>/round_<n>/evaluation/<dataset>_<split>/`: context, plans, results, audits, logs, error analysis, and retrieval feedback.

Candidate handoffs are rejected unless all documented fields are present, candidate IDs are unique and non-empty, model IDs are non-empty, and dataset/split/round agree with the manifest. The evaluator currently supports only `intended_use=direct_inference`; fine-tuning and other use modes fail validation before evaluation.

`run_evaluation_conditions.py` writes `batch_results.csv`, `batch_failures.json`, and per-dataset run directories. `run_eval_agent.py` writes `results.csv`, `plans.json`, `failures.json`, `agent_trace.json`, audits, summaries, predictions, and command logs.

## Reports And Analysis

Analysis is not restricted by the full-run dataset allowlist.

```bash
python create_condition_run_report.py   --run-root /path/to/condition_run   --output /path/to/condition_run/condition_run_report.md   --csv-output /path/to/condition_run/condition_run_report.csv
```

Retry failed pairs and automatically write `retry_failed_summary.json` plus `retry_failed_results.csv`:

```bash
python retry_failed_condition_pairs.py   --report-csv /path/to/condition_run/condition_run_report.csv   --source-run-root /path/to/condition_run   --project-root /sci/labs/michall/roeizucker/hf_eval_runtime   --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_retry   --stage smoke
```

## Agent Files

`agents/developer.md`, `agents/partner-reviewer.md`, and `agents/professor-reviewer.md` define the strict development review sequence. `agents/pipeline/` documents the runtime context-scout and partner retrieval boundary. These Markdown files are prompts/descriptions only; this repository does not automatically execute the development reviewers.

See `agents/README.md` for the approval gates and `DEVELOPMENT.md` for verification commands.

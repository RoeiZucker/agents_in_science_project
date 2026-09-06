# Pipeline Handoff Map

> **Current provider topology (2026-08-29):** Steps 1 and 8 can now invoke
> `artifact_linker_provider.py` automatically after feedback exists.
> Static manifest rounds are still supported. Generated rounds preserve
> provider identity/checksums, requested and evaluated splits, deduplicate
> models across rounds, and stop deterministically. See
> `ARTIFACT_LINKER_PROVIDER.md` for the authoritative implemented interface.

This document maps every pipeline step to its responsible agent or deterministic
script, its input files, its output artifacts, and the next consumer. The complete
commands and JSON examples are in `PARTNER_INTEGRATION_GUIDE.md`.

## Runtime Pipeline

The runtime contains one implemented agent role, one external partner-agent
boundary, and deterministic Python stages.

### Step 0: Select datasets and declare rounds

Responsible files:

- `config/full_pipeline_datasets.txt` selects the dataset subset.
- `config/mock_selection_loop.json` declares each dataset, requested split, round,
  and candidate-handoff path.
- `run_selection_loop.py` reads both files and schedules matching datasets.

Input:

- allowed dataset IDs;
- paths to candidate handoffs for already declared rounds.

Output:

- selected dataset/round specifications in memory.

Handoff: each declared round proceeds to Step 1.

### Step 1: Retrieve or recommend candidate models

Responsible agent boundary:

- `agents/pipeline/partner-retrieval-interface.md` describes the partner-owned role.
- `mock_candidates/ag_news_round1.json` and
  `mock_candidates/imdb_round1.json` are static stand-ins for its output.

Initial-round input:

- new dataset ID and context available to the partner system;
- partner retrieval data, graph, model registry, and ranking implementation.

Later-round input:

- `retrieval_feedback.json` from Step 7;
- `results.csv` from Step 6;
- optionally `error_analysis.json` and `codex_context.json`.

Output:

- one complete candidate-handoff JSON file matching the contract in
  `PARTNER_INTEGRATION_GUIDE.md`.

Handoff: `run_selection_loop.py` validates it in Step 2.

Current limitation: only static mock output is present. The pipeline does not invoke
a real or mock partner agent after evaluation.

### Step 2: Validate and adapt the candidate handoff

Responsible code:

- `run_selection_loop.py`: `read_handoff()`, `validate_handoff()`, and
  `write_conditions()`.

Input:

- candidate handoff from Step 1;
- expected dataset, split, and round from the manifest.

Output:

```text
<output-root>/<dataset>/round_<N>/conditions.csv
```

The conditions adapter preserves candidate identity, rank, retrieval score, reason,
source, intended use, owner, mock status, and retrieval-agent identity.

Handoff: `run_evaluation_conditions.py` consumes `conditions.csv` in Step 3.

### Step 3: Add metadata context

Responsible agent and orchestration:

- `agents/pipeline/context-scout.md` defines the runtime Context Scout.
- `run_evaluation_conditions.py` constructs the bounded prompt and launches Codex
  when `--runner codex` is selected.
- `run_evaluation_conditions.py` validates and preserves a user-supplied context
  file when `--runner manual --context-file ...` is selected.
- `make_codex_prompt.py` provides the same metadata-only prompt pattern for manual
  use.

Input:

- dataset ID;
- ordered candidate model IDs;
- candidate handoff from Step 1;
- dataset/model cards, schemas, metadata, and permitted local files.

Output:

```text
<dataset-run-dir>/codex_prompt.txt
<dataset-run-dir>/codex_context.json
<dataset-run-dir>/codex_stdout.log
<dataset-run-dir>/codex_stderr.log
```

Manual mode instead writes a preserved copy at:

```text
<dataset-run-dir>/manual_context.json
```

`run_evaluation_conditions.py` validates the exact context schema and verifies the
dataset and ordered model list in either mode. Invalid context stops the run before
evaluation. A manual context file is explicit run configuration, not an agent
decision or an automatic prompt-refinement step.

Handoff: the validated context file goes to `run_eval_agent.py` in Step 4.

### Step 4: Inspect and plan evaluation

Responsible deterministic code:

- `run_eval_agent.py` orchestrates inspection and planning.
- `eval_agent_core.py` inspects dataset/model metadata and selects a supported
  protocol.
- `result_contract.py` defines the shared measured-success rule.

Input:

- `conditions.csv` from Step 2;
- `codex_context.json` from Step 3;
- selected stage: `plan`, `smoke`, or `full`.

When `run_selection_loop.py` is invoked with `--stage full`, it first invokes
this same inspection, smoke evaluation, and audit path in
`round_N/smoke_evaluation/`. Only valid measured smoke successes proceed via
`round_N/full_conditions.csv`; smoke failures remain explicit and are excluded
from full evaluation and winner selection.

Output:

```text
<dataset-run-dir>/dataset_inspection.json
<dataset-run-dir>/model_inspections.json
<dataset-run-dir>/plans.json
<dataset-run-dir>/agent_trace.json
```

Unsupported protocols are recorded without pretending to evaluate them.

Handoff: every supported pair proceeds to `evaluate_hf_pair.py` in Step 5.

### Step 5: Run deterministic model evaluation

Responsible code:

- `evaluate_hf_pair.py` loads one model and dataset split, predicts, and scores.
- `run_eval_agent.py` launches it once per supported model/dataset pair.

Input:

- protocol and columns from `plans.json`;
- label map and prompt context from Steps 3 and 4;
- selected dataset split and stage limit.

Per-pair output:

```text
<pair-output-dir>/predictions.jsonl
<pair-output-dir>/summary.json
```

Smoke evaluates three examples. Full evaluates up to the configured `--full-limit`.

Handoff: predictions and summaries go to Step 6.

### Step 6: Audit and aggregate results

Responsible deterministic code:

- `run_eval_agent.py` audits predictions and writes result rows.
- `result_contract.py` decides whether a row is a measured success.
- `run_evaluation_conditions.py` rejoins results to candidate provenance.

Input:

- pair predictions and summaries from Step 5;
- plans and candidate metadata from earlier steps.

Output:

```text
<dataset-run-dir>/audits/<model>_<stage>_audit.json
<dataset-run-dir>/results.csv
<dataset-run-dir>/failures.json
<round-evaluation-dir>/batch_results.csv
<round-evaluation-dir>/batch_failures.json
```

Handoff:

- Step 7 receives results and audits for refinement.
- Step 9 receives candidate results for winner selection.

### Step 7: Convert evaluation into retrieval feedback

Responsible deterministic code:

- `analyze_eval_errors.py` and `refinement_agent_core.py` analyze outcomes.
- `make_retrieval_feedback.py` writes machine-readable feedback and a partner prompt.

Input:

- `results.csv`, plans, inspections, audits, and candidate provenance from Step 6.

Output:

```text
<dataset-run-dir>/error_analysis.json
<dataset-run-dir>/retrieval_feedback.json
<dataset-run-dir>/retrieval_feedback_prompt.txt
```

Handoff: these artifacts go to the partner-owned boundary in Step 8.

### Step 8: Produce the next candidate round

Responsible agent boundary:

- the partner implementation represented by
  `agents/pipeline/partner-retrieval-interface.md`.

Input:

- feedback artifacts from Step 7;
- partner retrieval/model-selection system.

Expected output:

- a new complete candidate handoff whose `round` is the feedback `next_round`.

Handoff: return to Step 2 to validate and evaluate the new round.

Current limitation: this callback is not automated. For manual integration, the
partner writes the file and it is added to the manifest. For automatic integration,
add the candidate-provider adapter described in `PARTNER_INTEGRATION_GUIDE.md`.

### Step 9: Select the final measured winner

Responsible deterministic code:

- `run_selection_loop.py`: `join_candidate_results()`, `select_winners()`, and
  `select_winner()`.

Input:

- candidate results from all rounds declared or generated for the run.

Output:

```text
<output-root>/selection_loop_results.csv
<output-root>/selected_models.csv
<output-root>/selection_loop_summary.json
```

The winner is the candidate with the highest valid measured score for each dataset.
Retrieval scores are retained as provenance but do not determine the winner.

Handoff: `selected_models.csv` goes to the downstream user or application.

## Runtime Flow At A Glance

```text
dataset subset + manifest
  -> partner candidate handoff
  -> validated conditions.csv
  -> validated context (Context Scout or manual file)
  -> deterministic plans
  -> pair predictions and summaries
  -> audits and results.csv
  -> error_analysis.json
  -> retrieval_feedback.json
  -> partner next-round handoff [currently manual/missing]
  -> repeat evaluation
  -> selected_models.csv
```

## Development Review Agents

These agents review repository changes. They are separate from runtime evaluation
and model selection.

### Development Step 1: Developer

- Description: `agents/developer.md`
- Reads: code, tests, mocks, documentation, and assigned findings.
- Produces: `reviews/developer-handoff.md`.
- Handoff: Partner Reviewer.

### Development Step 2: Partner Reviewer

- Description: `agents/partner-reviewer.md`
- Requires: current `reviews/developer-handoff.md`.
- Reads: implementation, tests, mocks, documentation, and Developer handoff.
- Produces: `reviews/partner-review.md` with `Decision: APPROVE` or
  `Decision: REJECT`.
- Handoff: Developer on rejection; Professor Reviewer on approval.

### Development Step 3: Professor Reviewer

- Description: `agents/professor-reviewer.md`
- Requires: `reviews/partner-review.md` with `Decision: APPROVE`.
- Reads: repository files and both previous review artifacts.
- Produces: `reviews/professor-review.md` with `Decision: APPROVE` or
  `Decision: REJECT`.
- Handoff: Developer on rejection; a fresh Partner approval is then required.

`agents/README.md` documents this strict review order.

## Important Integration Warnings

1. `retrieval_feedback_prompt.txt` currently requests only a top-level candidates
   list, but Step 2 requires the complete candidate-handoff object.
2. Feedback records the evaluated split such as `test`, while the current manifest
   and next handoff validate against the requested split `auto`.
3. Text classification is currently represented as evaluator task `generation`, so
   retrieval feedback can recommend the wrong model task family.
4. Step 8 is not called by `run_selection_loop.py`; all rounds must currently be
   declared before the process starts.
5. The verified metadata scouts used about 80,000 Codex tokens for two simple
   datasets, so context caching or a cheaper scout configuration is recommended.

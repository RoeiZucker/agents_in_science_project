# Partner Integration Guide

> **Implemented iterative provider path (2026-08-29):** The automatic
> Artifact Linker callback, stopping rules, split propagation, comparability,
> and provenance-safe resume are now implemented in `run_selection_loop.py`.
> The static workflow below remains supported, but sections describing the
> provider callback as missing are historical. See
> `ARTIFACT_LINKER_PROVIDER.md` for the current JSON interface, schemas,
> example manifest, tests, and bounded commands.

This guide describes the pipeline as it currently works, the boundary owned by the
partner, and the changes needed for a genuinely feedback-driven multi-round run.

## 1. Purpose And Ownership

The project evaluates candidate Hugging Face models on a selected dataset and picks
the best valid measured result. It does not implement candidate retrieval.

Project-owned code:

- inspects dataset and model metadata;
- chooses a supported evaluation protocol;
- runs smoke or full evaluation;
- audits predictions and records honest failures;
- creates retrieval feedback;
- compares valid measured results and selects a winner.

Partner-owned code:

- reads the new dataset context and evaluation feedback;
- retrieves or recommends the next candidate models;
- writes a candidate handoff matching the JSON contract below.

The files in `mock_candidates/` contain static examples of the partner handoff. They
do not retrieve models and must not be presented as the partner implementation.

## 2. What Is Working Now

The verified smoke run completed this flow for two datasets and two models per
dataset:

1. Read a static mock candidate handoff.
2. Ask the Codex context scout for metadata-only evaluation context.
3. Validate the context artifact.
4. Evaluate every candidate on three labeled examples.
5. Audit predictions and generate refinement feedback.
6. Select the highest valid measured score per dataset.

It produced four measured successes and two selected winners. It did not ask a
partner component for new candidates after evaluation. The current manifest defines
only one static round per dataset.

Verified output root:

```text
/sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_selection_loop_smoke_check_20260822
```

## 3. Environment Setup

Run from the machine containing the repository and shared runtime:

```bash
source /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/.venv-artifact-linker/bin/activate
cd /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent
```

The pipeline uses these shared locations:

```text
Project/cache root: /sci/labs/michall/roeizucker/hf_eval_runtime
Hugging Face cache: /sci/labs/michall/roeizucker/hf_eval_runtime/.hf_cache
Dataset cache:      /sci/labs/michall/roeizucker/hf_eval_runtime/.hf_datasets_cache
```

Set `HF_TOKEN` when available to avoid unauthenticated Hub rate limits. Existing
cached datasets and models are reused. `run_selection_loop.py` sets cache cleanup to
`none`, so artifacts are not deleted after a round.

On this cluster, nested Codex requires `--codex-bypass-sandbox` because unprivileged
`bwrap` namespaces are unavailable. Do not use that flag on a machine where the
normal Codex sandbox works.

## 4. Input Files

### Dataset subset

`config/full_pipeline_datasets.txt` is the allowlist for full pipeline runs. It has
one dataset ID per line. Full runs fail when this argument is absent.

Analysis and report scripts do not use this allowlist and can inspect any completed
run.

### Loop manifest

`config/mock_selection_loop.json` declares datasets and candidate rounds:

```json
{
  "manifest_version": 1,
  "datasets": [
    {
      "dataset": "fancyzhx/ag_news",
      "split": "auto",
      "rounds": [
        {
          "round": 1,
          "candidates": "mock_candidates/ag_news_round1.json"
        }
      ]
    }
  ]
}
```

Candidate paths are resolved relative to the repository when the manifest is under
`config/`. Every declared round is evaluated in the order shown.

### Partner candidate handoff

The partner component must write the complete object below. For real retrieval,
`mock` must be `false` and `retrieval_agent` should identify the real implementation.

```json
{
  "handoff_version": 1,
  "mock": false,
  "owner": "partner_retrieval",
  "retrieval_agent": "partner-system-name-and-version",
  "dataset": "fancyzhx/ag_news",
  "split": "auto",
  "round": 2,
  "candidates": [
    {
      "candidate_id": "ag-news-r2-1",
      "condition": "partner_round_2",
      "model": "organization/model-name",
      "rank": 1,
      "score": 0.91,
      "source": "partner_retrieval",
      "intended_use": "direct_inference",
      "reason": "Why this model fits the dataset and feedback."
    }
  ]
}
```

Validation rules:

- `dataset`, `split`, and `round` must exactly match the manifest entry;
- every required field must be present and non-empty where textual;
- candidate IDs must be unique within the handoff;
- `rank` must be an integer;
- `score` must be a finite number;
- currently, `intended_use` must be exactly `direct_inference`;
- at least one candidate is required.

The partner score is retrieval provenance, not an evaluation score. Winners are
selected using measured evaluation scores only.

## 5. Commands

### Offline tests

```bash
python -m unittest discover -s tests -v
```

### Plan check

```bash
python run_selection_loop.py \
  --manifest config/mock_selection_loop.json \
  --dataset-subset-file config/full_pipeline_datasets.txt \
  --stage plan \
  --runner script \
  --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_selection_loop_plan
```

Plan mode does not load model weights or run inference. Dataset inspection can still
download or materialize dataset data needed to determine schema, labels, and splits.

### Bounded smoke run with context scout

```bash
python run_selection_loop.py \
  --manifest config/mock_selection_loop.json \
  --dataset-subset-file config/full_pipeline_datasets.txt \
  --stage smoke \
  --runner codex \
  --codex-timeout 180 \
  --codex-bypass-sandbox \
  --trust-remote-code \
  --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_selection_loop_smoke
```

Smoke mode evaluates three examples per model-dataset pair.

### Full allowlisted run

```bash
python run_selection_loop.py \
  --manifest config/mock_selection_loop.json \
  --dataset-subset-file config/full_pipeline_datasets.txt \
  --stage full \
  --runner codex \
  --codex-timeout 180 \
  --codex-bypass-sandbox \
  --trust-remote-code \
  --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_selection_loop_full
```

Full mode is smoke-gated per round. The orchestrator evaluates every proposed
candidate in `round_N/smoke_evaluation/`, audits those outputs, and writes only
measured smoke successes to `round_N/full_conditions.csv`. It then runs the
bounded full evaluation under `round_N/evaluation/`. Failed smoke candidates
remain visible as `smoke_failed`, are fed back as candidate exclusions, and
cannot become winners.

Use a new output root for a new experiment. `--trust-remote-code` permits dataset
loading scripts from the Hub and should be used only when that trust decision is
acceptable.

## 6. Round Outputs

For dataset `owner/name`, round `N`, the important paths are:

```text
<output-root>/owner_name/round_N/conditions.csv
<output-root>/owner_name/round_N/smoke_evaluation/owner_name_<requested-split>/results.csv
<output-root>/owner_name/round_N/full_conditions.csv
<output-root>/owner_name/round_N/evaluation/owner_name_<requested-split>/results.csv
<output-root>/owner_name/round_N/evaluation/owner_name_<requested-split>/codex_context.json
<output-root>/owner_name/round_N/evaluation/owner_name_<requested-split>/error_analysis.json
<output-root>/owner_name/round_N/evaluation/owner_name_<requested-split>/retrieval_feedback.json
<output-root>/owner_name/round_N/evaluation/owner_name_<requested-split>/retrieval_feedback_prompt.txt
```

Top-level outputs:

```text
<output-root>/selection_loop_results.csv
<output-root>/selected_models.csv
<output-root>/selection_loop_summary.json
```

`selection_loop_results.csv` contains every candidate attempt and its complete
partner provenance. `selected_models.csv` contains one winner or explicit
planned/failure row per dataset.

A row is a measured success only when:

- `status` is `ok`;
- `score` is finite;
- `labeled_total` is greater than zero.

Warnings remain visible in `notes` and audit JSON. Planned, unsupported, unlabeled,
missing, failed, or non-finite rows cannot win.

## 7. Integrating A Second Round Today Without Code Changes

This is the simplest integration path for the meeting.

1. Run round 1 in smoke mode.
2. Give the partner these files for each dataset:
   - `retrieval_feedback.json`;
   - `retrieval_feedback_prompt.txt`;
   - `results.csv`;
   - optionally `error_analysis.json` and `codex_context.json`.
3. The partner retrieves new candidates and writes a full round-2 handoff using the
   schema in section 4.
4. Keep `split` equal to the manifest value, currently `auto`. The generated
   feedback currently reports the resolved evaluation split, such as `test`; copying
   that value into the handoff would fail manifest validation.
5. Add the new handoff to the manifest:

```json
"rounds": [
  {
    "round": 1,
    "candidates": "mock_candidates/ag_news_round1.json"
  },
  {
    "round": 2,
    "candidates": "partner_candidates/ag_news_round2.json"
  }
]
```

6. Rerun `run_selection_loop.py` with the updated manifest and a new output root.

The loop will evaluate both declared rounds and select the best measured result
across all candidates from both rounds. It will rerun round 1; cache reuse keeps this
less expensive, but there is no resume/skip-completed feature yet.

## 8. Changes Required For A True Automatic Feedback Loop

The following changes are required before claiming that evaluation automatically
selects the next candidate models.

### Required: connect a candidate provider

`run_selection_loop.py` currently reads only candidate files already listed in the
manifest. Add an explicit partner-provider adapter, for example:

```text
python partner_retrieval.py \
  --feedback <round-dir>/retrieval_feedback.json \
  --output <round-dir>/candidate_handoff.json
```

Recommended orchestration:

1. Start with one initial handoff.
2. Evaluate the handoff.
3. Read the generated `retrieval_feedback.json`.
4. Invoke the partner provider for `next_round`.
5. Validate its output with the existing `read_handoff()` contract.
6. Evaluate the new candidates.
7. Stop at a configured `--max-rounds`, when no valid candidates are returned, or
   when a documented improvement threshold is not met.
8. Select the best compatible measured result across all completed rounds.

Keep the provider as a separate executable or module owned by the partner. Do not
put retrieval logic inside the evaluator.

### Required: align feedback with the handoff schema

`feedback_prompt()` currently asks only for a top-level `candidates` list. The
validator also requires `handoff_version`, `mock`, `owner`, `retrieval_agent`,
`dataset`, `split`, and `round`. Update the generated prompt or give the provider the
schema in section 4.

### Required: align requested and resolved split values

The manifest and handoff currently use the requested split (`auto`), while
`retrieval_feedback.json` uses the resolved split (`test` for the verified datasets).
Choose one canonical contract. The smallest change is to include both fields:

```json
{
  "requested_split": "auto",
  "evaluated_split": "test"
}
```

Use `requested_split` when validating the next handoff and `evaluated_split` when
describing the measurement.

### Required: improve retrieval task semantics

Text classification datasets currently use the evaluator's general `generation`
task with a label-generation protocol. Consequently, AG News feedback recommended
generic instruction-following and question-answering models instead of explicit
text classifiers. Either:

- add a `classification` task to the context/feedback contract; or
- infer retrieval task `classification` when the dataset has a finite label map and
  no choices or image column.

This changes retrieval guidance, not the already working accuracy calculation.

### Recommended: control context-scout cost

The verified smoke run used roughly 80,000 Codex tokens across two straightforward
metadata lookups. Add a cheaper model/reasoning setting or cache context by the
dataset plus ordered model list. Reuse a validated `codex_context.json` when the
request provenance is unchanged.

### Recommended: add resume and compatibility rules

- Skip completed candidate pairs when their result and context provenance match.
- Do not compare scores with incompatible metrics or evaluation protocols unless an
  explicit normalization rule exists.
- Record the partner provider version and input-feedback checksum in each handoff.
- Keep a maximum round count and a deterministic stop reason in the summary.

## 9. Meeting Checklist

Agree on these points with the partner before combining code:

- exact location and command for the partner candidate provider;
- complete input feedback schema;
- complete output handoff schema;
- requested versus evaluated split semantics;
- whether the partner returns only new models or may retain the best baseline;
- deduplication rules across rounds;
- maximum rounds and stopping rule;
- how retrieval scores are defined and whether they are comparable across rounds;
- allowed model size, licensing, gating, modality, and `trust_remote_code` policy;
- whether Codex context is reused or regenerated;
- ownership of integration tests and one shared two-round fixture.

The first shared acceptance test should use two datasets, two candidates in round 1,
and a deterministic mock provider that reads round-1 feedback and writes two round-2
candidates. The test passes only when round 2 is generated after evaluation, both
rounds are measured, provenance survives, and the final winner is chosen from all
valid measured rows.

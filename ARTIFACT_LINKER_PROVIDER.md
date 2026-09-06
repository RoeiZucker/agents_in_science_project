# Artifact Linker Provider Adapter

`artifact_linker_provider.py` is the versioned ownership boundary between
Artifact Linker Stage 1 and `hf-eval-agent`. It consumes retrieval/ranking
artifacts and emits candidate-handoff version 1. Retrieval signals remain
provenance only; the selection loop chooses winners exclusively from compatible,
finite, labeled evaluator measurements.

## Source schemas inspected

The checked-out Artifact Linker source at commit
`ad29148b64c96cd62a85e6937231b30d651e042d` has two relevant raw forms:

- scored exports under `data/custom_link_predictions/` and
  `data/custom_combined_predictions/` contain top-level dataset provenance and a
  `results` list. Each row contains `model_name`, `rank`, and
  `link_probability` or `combined_score`;
- reusable link-ranking runners create rows with `dataset_id`,
  `positive_models`, and ordered `ranked_model_ids`. These rows contain no
  per-model score, so the adapter labels `1 / rank` as a derived ordering signal
  and uses `node_metadata.json` to resolve model names.

The public joint `scripts/rank_link_gnn.py` writes `test_metrics` only.
`save_link_rankings()` also replaces result rows with aggregate metrics when
there are more than 100 queries. Aggregate-only files cannot identify candidate
models and are rejected explicitly.

## JSON input

The orchestrator writes `<round>/provider_input.json`:

```json
{
  "provider_input_version": 1,
  "dataset": "owner/dataset",
  "requested_split": "auto",
  "round": 2,
  "retrieval_task": "classification",
  "dataset_context": {},
  "feedback": {},
  "seen_models": ["organization/already-evaluated"],
  "continuation_review": {
    "required": true,
    "minimum_ranked_models": 5,
    "ranked_model_count": 5,
    "proposed_round": 3,
    "current_top_models": []
  },
  "provider_config": {
    "artifact_output": "/absolute/path/to/rankings.json",
    "artifact_dataset": "owner/dataset",
    "method": "gnn-gatv2-combined",
    "version": "ad29148b64c9",
    "score_field": "combined_score",
    "max_candidates": 5
  }
}
```

For raw `ranked_model_ids`, also provide `dataset_id` and `node_metadata`.
Relative artifact paths are resolved by the orchestrator before invocation.

Run the adapter independently:

```bash
python artifact_linker_provider.py \
  --input provider_input.json \
  --output candidate_handoff.json
```

The output includes `provider_version`, SHA-256 checksums of the exact provider
input and Artifact Linker artifact, stable model-based candidate IDs, and
`retrieval_rejections`. Malformed, duplicate, gated/private, invalid-rank, and
non-finite-score rows are excluded with reasons. A valid empty output carries
`no_candidates_reason`; the orchestrator serializes the deterministic stop.

## Iterative manifest

A dataset may seed round 1 from a static `rounds` entry and configure a provider
for subsequent rounds, or omit `rounds` and invoke the provider for round 1.

```json
{
  "manifest_version": 1,
  "datasets": [{
    "dataset": "owner/dataset",
    "split": "auto",
    "rounds": [{"round": 1, "candidates": "mock_candidates/seed.json"}],
    "provider": {
      "artifact_output": "external/artifact-linker/data/rankings.json",
      "method": "gnn-gatv2",
      "version": "ad29148b64c9",
      "max_candidates": 2
    }
  }]
}
```

`provider.command` may be a command list or a single Python-script path. If it is
omitted, the orchestrator invokes `artifact_linker_provider.py`. Commands receive
`--input` and `--output` without shell interpolation.

## Stops, comparison, and resume

The loop records one of `max_rounds`, `agent_consensus_no_likely_improvement`,
`static_manifest_complete`,
`no_valid_candidates`, `no_new_candidates`, `minimum_measured_improvement`,
`provider_failure`, or `evaluation_failure` for each dataset. Models are
deduplicated case-insensitively across rounds.

`target_ranked_models` is a minimum output count. After that minimum is met, a
provider can attach a `continuation_review` with retrieval, critic, and
orchestrator judgments. The orchestrator honors an early stop only when the
review reports unanimous no-improvement consensus; missing or disputed reviews
continue to the hard round limit.

The first valid measurement establishes the dataset's canonical metric and
protocol family. Fixed-label accuracy protocols are compatible with each other;
different metrics or other protocol families are marked `non_comparable` and
cannot win.

`--resume` requires the same dataset, requested split, stage, runner, sample
limits, seed, trust options, complete candidate/model source, provider-input
checksum, evaluator, orchestrator, full-gate, and refinement code checksums. It
also verifies checksums of results, plans, any generated context, and (for full
runs) smoke results before reuse. Changed or missing provenance
causes a fresh round evaluation.

## Commands

Offline acceptance tests:

```bash
PYTHONPATH=. pytest -q \
  tests/test_artifact_linker_provider.py \
  tests/test_iterative_selection_loop.py \
  tests/test_selection_loop_controls.py \
  tests/test_selection_loop.py \
  tests/test_refinement_agent_core.py
```

Bounded real-output plan (metadata inspection only, two candidates maximum):

```bash
python run_selection_loop.py \
  --manifest config/artifact_linker_plan_example.json \
  --stage plan --runner script --max-rounds 1 \
  --sample-size 3 --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_artifact_linker_plan
```

Smallest safe live inference follows only after inspecting that plan:

```bash
python run_selection_loop.py \
  --manifest config/artifact_linker_smoke_example.json \
  --stage smoke --runner codex --max-rounds 1 --smoke-limit 3 --sample-size 3 \
  --codex-timeout 180 --codex-bypass-sandbox \
  --project-root /sci/labs/michall/roeizucker/hf_eval_runtime \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/_artifact_linker_smoke
```

A `--stage full` run requires an explicit dataset subset. Each round first runs
the same measured smoke and audit under `round_N/smoke_evaluation/`. Only
candidates with `status=ok`, a finite score, and `labeled_total > 0` are written
to `full_conditions.csv` and admitted to the bounded full evaluation. Smoke
failures remain explicit failure rows, are added to next-round candidate
exclusions, and cannot win. `--full-limit` bounds each admitted evaluation; it
does not bypass the smoke gate.

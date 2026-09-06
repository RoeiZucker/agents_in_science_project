# Oren Cached 18-Dataset Evaluation Runs

All 18 datasets in `config/oren_cached_18_datasets.txt` have a committed,
non-empty description and 1024-dimensional Voyage embedding under Oren's
`data/advisor_runs_initial_run_max14b` directory. They are cold-start datasets:
most are intentionally absent from the base ArtifactBench graph. The local
provider appends each stored embedding to an isolated per-dataset GNN split;
it does not call OpenAI or Voyage.

## Feedback-driven sampled loop

The current iterative manifest retrieves three unseen candidates per round,
evaluates every candidate on the same deterministic sample, and supplies the
accumulated comparable results to the next retrieval. Five comparable measured
models is the minimum output pool, not an automatic stop. Once that minimum
exists, the next retrieval proposes candidates and argues their improvement
potential, an independent evaluation critic challenges the proposal, and an
orchestrator resolves the discussion. The loop stops early only when all three
agree that improvement is unlikely; disagreement continues to `--max-rounds`.
Duplicate or invalid provider responses are retried twice within the same round.

Run the two-dataset, 20-example smoke test from any directory:

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 06:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_iterative_18_sample200.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_iterative_smoke_2_datasets.txt \
  --stage smoke --runner script \
  --max-rounds 3 --target-ranked-models 5 --provider-retries 2 \
  --smoke-limit 20 --sample-size 20 --seed 42 --codex-timeout 300 --resume \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_iterative_smoke2x20_fixed_20260905
```

After that succeeds, run all 18 datasets on fixed samples of at most 200:

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 2-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_iterative_18_sample200.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_cached_18_datasets.txt \
  --stage smoke --runner script \
  --max-rounds 3 --target-ranked-models 5 --provider-retries 2 \
  --smoke-limit 200 --sample-size 20 --seed 42 --codex-timeout 300 --resume \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_iterative_18_sample200_20260905
```

`--target-ranked-models` is the required minimum ranked output count. The CLI
rejects a manifest whose theoretical round capacity is below that minimum. Failed or
protocol-incompatible evaluations do not count toward the five. If any dataset
still falls short after all rounds and retries, the summary lists it under
`datasets_below_target` and the command exits nonzero.

Post-minimum discussion artifacts are stored in the proposed round directory:
`codex_recommendations.json`, `evaluation_critic_response.json`,
`orchestrator_response.json`, and `continuation_review.json`. The final stop
reason is `agent_consensus_no_likely_improvement` for an agreed early stop or
`max_rounds` when the hard cap is reached.

## Legacy cached smoke workflow

The older cached-ranking smoke command is retained for provenance. New
feedback-driven runs should use the commands above.

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 1-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_cached_18_smoke.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_cached_18_datasets.txt \
  --stage smoke --runner script --max-rounds 5 --target-ranked-models 5 \
  --smoke-limit 3 --codex-timeout 300 \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_cached_18_smoke5_20260903
```

This is a smoke run: model ranking is based on only three examples per
dataset. Append `--resume` to the loop arguments when resubmitting the same
output directory after a timeout or cancellation.

The final per-dataset top-five rows are combined in:

`/sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_cached_18_smoke5_20260903/ranked_models.csv`

Slurm stdout and stderr are created as
`/sci/labs/michall/roeizucker/hf_eval_runtime/selection_loop_<job-id>.out` and
`.err`.

## Protocol-pinned reliable comparison

Do not expand the three-example run above directly to a full evaluation. It
used automatic field inference and produced several incorrect generic
generation protocols. The final workflow pins a versioned context for every
dataset under `config/oren_final_contexts/`.

Validate all context files and load their configured dataset splits without
running model inference:

```bash
HF_HOME=/sci/labs/michall/roeizucker/hf_eval_runtime/.hf_cache \
HF_DATASETS_CACHE=/sci/labs/michall/roeizucker/hf_eval_runtime/.hf_datasets_cache \
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/.venv-artifact-linker/bin/python \
  /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/validate_evaluation_contexts.py \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_cached_18_final.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_cached_18_datasets.txt \
  --inspect-datasets --sample-size 20 \
  --output /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_context_preflight_20260904.json
```

Run the measured 20-example candidate-selection preflight. Artifact Linker and
Codex still retrieve/rerank candidates on every round; the explicit contexts
replace only automatic evaluation-protocol inference.

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 2-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_cached_18_final.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_cached_18_datasets.txt \
  --stage smoke --runner script --max-rounds 5 --target-ranked-models 5 \
  --smoke-limit 20 --sample-size 20 --seed 42 --codex-timeout 300 \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_protocol_preflight20_20260904
```

After the preflight finishes, freeze exactly five measured, comparable models
per dataset. This command fails without writing a final manifest if any dataset
has fewer than five.

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/.venv-artifact-linker/bin/python \
  /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/freeze_ranked_candidates.py \
  --ranked-models /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_protocol_preflight20_20260904/ranked_models.csv \
  --source-manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_cached_18_final.json \
  --output-dir /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_frozen_candidates_20260904 \
  --target 5
```

Only after that strict freeze succeeds, evaluate the same candidates on a
fixed, deterministic sample of up to 1,000 labeled examples per dataset:

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 4-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_frozen_candidates_20260904/frozen_manifest.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_cached_18_datasets.txt \
  --stage full --runner script --max-rounds 1 --target-ranked-models 5 \
  --smoke-limit 20 --sample-size 20 --full-limit 1000 --seed 42 \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_frozen_full1000_20260904
```

Both measured stages preserve context and candidate checksums in round
provenance. The final command has no retrieval provider, so it cannot silently
replace a preflight candidate during the full comparison.

# Oren Retrieval Loop for Existing ArtifactBench Datasets

`oren_local_provider.py` uses the retrieval code from Oren's Artifact Linker
fork at commit `20b1e1d57f008cf1296b9dfbdb693029e0742892`. It accepts either an
exact dataset in the local ArtifactBench graph or an exact dataset with a
description and embedding in Oren's committed cold-start cache.

For each dataset it performs this sequence:

1. Load the stored dataset description and Voyage embedding. No description or
   embedding API is called.
2. Run Method 1 cosine retrieval and build measured-neighbor context.
3. Run Method 2 GNN inference with the local checkpoint and build GNN context.
4. Merge both evidence sources and use the authenticated Codex CLI to return a
   schema-validated list of unseen model IDs.
5. Evaluate candidates and call retrieval again with accumulated measurements
   and failed/seen models excluded.
6. Once five comparable models exist, have retrieval argue for the next
   candidates, an evaluation critic challenge that argument, and an orchestrator
   decide whether improvement is still probable. Only unanimous no-improvement
   consensus stops early; disagreement continues to the hard round cap.
7. Write the best five measured models per dataset to `ranked_models.csv`.

Neither `OPENAI_API_KEY` nor `VOYAGE_API_KEY` is needed. The Codex CLI must be
logged in on the submission host, and its `CODEX_HOME` must be visible on the
compute node.

## Full-path Slurm command

```bash
/sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/submit_selection_loop.sh \
  -A tomhope --time 1-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_existing_smoke.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_existing_smoke_datasets.txt \
  --stage smoke --runner script --max-rounds 5 --target-ranked-models 5 \
  --smoke-limit 3 --codex-timeout 300 \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/oren_commonsenseqa_5
```

The Slurm streams are written to
`/sci/labs/michall/roeizucker/hf_eval_runtime/selection_loop_<job-id>.out` and
`.err`. Loop artifacts go under the explicit `--output-root`; the most useful
top-level files are `ranked_models.csv`, `selected_models.csv`,
`selection_loop_results.csv`, and `selection_loop_summary.json`.
Post-minimum discussion prompts and structured responses are retained inside
the proposed round directory, together with `continuation_review.json`.

Use the identical command with `--resume` appended after a timeout or
cancellation. To evaluate more datasets, add exact graph or committed-cache
dataset names to the manifest and subset file. A dataset absent from both
sources is rejected before retrieval rather than summarized or embedded
remotely.

## Local plan-only check

```bash
/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/agents_project/.venv-artifact-linker/bin/python \
  /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/run_selection_loop.py \
  --manifest /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_local_existing_smoke.json \
  --dataset-subset-file /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent/config/oren_existing_smoke_datasets.txt \
  --project-root /tmp/oren-local-runtime \
  --output-root /tmp/oren-local-loop-smoke \
  --stage plan --runner script --max-rounds 1 --target-ranked-models 5
```

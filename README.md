# Hugging Face Model Evaluation and Selection Loop

This project selects strong Hugging Face models for a dataset through an iterative retrieval-and-evaluation loop. Oren's AutoModelAdvisor retrieves candidates from ArtifactBench evidence; this project evaluates them on real examples, returns measured feedback to retrieval, and produces a ranked candidate list for each dataset (five models in the experiments documented here).

The system separates agent judgment from model measurement: agents interpret metadata, discuss whether another round is worthwhile, and propose candidates, while deterministic Python code loads datasets and models, computes metrics, validates results, and writes the final ranking.

## How the loop works

```text
dataset + evaluation context
        -> Oren retrieval (cosine evidence + GNN evidence)
        -> candidate models
        -> deterministic sample evaluation
        -> scores, failures, and error feedback
        -> another retrieval round or early stop
        -> ranked_models.csv
```

A round is one complete retrieve, evaluate, and feedback cycle:

1. The retrieval provider proposes up to `max_candidates` previously unseen models.
2. The Context Scout validates the dataset schema and selects a supported evaluation protocol.
3. Each candidate is evaluated on the configured random sample.
4. The loop records scores, parsing failures, errors, and provenance, then supplies that context to the next retrieval call.
5. Once the target number of comparable models exists, the retrieval agent, evaluation critic, and orchestrator discuss whether another round is likely to improve the ranking. Unanimous no-improvement consensus stops early; otherwise the loop continues to `max_rounds`.

Provider retries handle a failed or malformed retrieval call; they are not additional rounds. Candidates are deduplicated across rounds, and only successful, comparable results with a finite score and at least one labeled example can enter the final ranking.

## Components

| Component | Responsibility |
| --- | --- |
| `run_selection_loop.py` | Multi-round control flow, feedback, deduplication, stopping, and final ranking |
| `oren_local_provider.py` | Adapter to Oren's retrieval pipeline and the retrieval/critic/orchestrator discussion |
| `external/artifact-linker/` | Pinned Oren Artifact Linker submodule containing retrieval code, descriptions, and embeddings |
| `run_evaluation_conditions.py` | Candidate-batch evaluation and result collection |
| `run_eval_agent.py` | Dataset/model inspection, planning, evaluation, and auditing |
| `evaluate_hf_pair.py` | Deterministic model-dataset execution and metric calculation |
| `refinement_agent_core.py` | Structured evaluation-error feedback |
| `result_contract.py` | Shared success, failure, and comparability rules |
| `config/` | Dataset manifests, evaluation contexts, subsets, and reproducible experiment inputs |
| `agents/pipeline/` | Runtime agent contracts and boundaries |
| `tests/` | Unit and integration coverage |

## Setup

Clone the project together with the pinned Oren repository:

```bash
git clone --recurse-submodules https://github.com/RoeiZucker/agents_in_science_project.git
cd agents_in_science_project
```

For an existing clone:

```bash
git submodule update --init --recursive
```

Create an environment and install the evaluator:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Text-only GGUF evaluation is optional:

```bash
pip install -r requirements-gguf.txt
```

The setup above installs the evaluator. The submodule includes Oren's committed dataset descriptions and Voyage embeddings; its retrieval/GNN dependencies, the large ArtifactBench graph, and the trained checkpoint are set up separately as described in `external/artifact-linker/retrieval_agent/method2_gnn_inference/README_INSTRUCTIONS.txt`.

The integrated cached retrieval path requires:

- a logged-in Codex CLI available to the process;
- Hugging Face authentication for gated models;
- GPU access for candidate evaluation and GNN inference.

It does not require `OPENAI_API_KEY`. `VOYAGE_API_KEY` is needed only when creating an embedding for a dataset that is not already present in the committed cache.

## Reviewer wiring check

This plan-only command validates the manifest, candidate handoff, filtering, and output wiring without loading model weights:

```bash
python run_selection_loop.py \
  --manifest config/mock_selection_loop.json \
  --dataset-subset-file config/full_pipeline_datasets.txt \
  --stage plan \
  --runner script \
  --project-root /tmp/hf-eval-agent-review \
  --output-root /tmp/hf-eval-agent-review/results
```

Run the test suite with:

```bash
python -m unittest discover -s tests -v
```

## Representative selection run

The following settings reproduce the structure used for the 15-dataset selection experiment: at most three total rounds, five candidates per retrieval call, five final ranked models, three provider retries, and at most 100 evaluated examples per candidate.

```bash
python run_selection_loop.py \
  --manifest config/oren_iterative_15_sample100.json \
  --dataset-subset-file config/oren_iterative_smoke_multi_nli.txt \
  --stage smoke \
  --runner codex \
  --max-rounds 3 \
  --target-ranked-models 5 \
  --provider-retries 3 \
  --smoke-limit 100 \
  --sample-size 20 \
  --seed 42 \
  --codex-timeout 600 \
  --project-root runtime \
  --output-root runtime/eval_results/multi_nli_selection \
  --trust-remote-code
```

The later large evaluation is intentionally separate: it freezes the selected five models and evaluates each model-dataset pair without further retrieval or refinement.

## Slurm

`submit_selection_loop.sh` accepts scheduler options before `--` and forwards all remaining options to `run_selection_loop.py`:

```bash
./submit_selection_loop.sh \
  -A tomhope --time 1-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest config/oren_iterative_15_sample100.json \
  --dataset-subset-file config/oren_iterative_smoke_multi_nli.txt \
  --stage smoke --runner codex \
  --max-rounds 3 --target-ranked-models 5 \
  --provider-retries 3 --smoke-limit 100 \
  --sample-size 20 --seed 42 --codex-timeout 600 \
  --output-root /path/to/results
```

Use `--dry-run` before `--` to inspect the submission and `--resume` among the loop options to continue an interrupted run. See [SLURM_SELECTION_LOOP.md](SLURM_SELECTION_LOOP.md) for scheduler, cache, log, and interactive-allocation details.

## Outputs

Each run writes aggregate files at its output root:

- `ranked_models.csv`: the final ranked candidates for every dataset;
- `selection_loop_results.csv`: every evaluated candidate and its provenance;
- `selected_models.csv`: the best valid measured model per dataset;
- `selection_loop_summary.json`: dataset counts, stopping reasons, and run metadata.

Each `<dataset>/round_<n>/` directory retains the candidate handoff, evaluation context, conditions, measured results, audits, feedback, and any continuation discussion. This makes every ranking traceable back to the retrieval evidence and measured examples that produced it.

## Further documentation

- [OREN_LOCAL_RETRIEVAL.md](OREN_LOCAL_RETRIEVAL.md): retrieval methods, feedback rounds, and cached-dataset behavior.
- [EVALUATION_METHODS.md](EVALUATION_METHODS.md): supported tasks, protocols, metrics, and adapter boundaries.
- [ARTIFACT_LINKER_PROVIDER.md](ARTIFACT_LINKER_PROVIDER.md): provider and handoff contracts.
- [SLURM_SELECTION_LOOP.md](SLURM_SELECTION_LOOP.md): queued and interactive cluster execution.
- [DEVELOPMENT.md](DEVELOPMENT.md): development workflow and verification.

# Hugging Face Model Evaluation and Selection Loop

This project ranks Hugging Face models for a dataset through an iterative retrieval-and-evaluation loop. The integrated Artifact Linker retrieval component combines similarity and graph evidence to propose candidates. The evaluation pipeline measures those candidates on real examples and returns the results as context for later retrieval rounds.

Agent judgment is separated from model measurement: agents interpret metadata, review evaluation feedback, and decide whether further retrieval is worthwhile, while deterministic Python code loads datasets and models, computes metrics, validates results, and writes the ranking.

## How the loop works

```text
dataset + evaluation context
        -> Artifact Linker retrieval
        -> candidate models
        -> deterministic sample evaluation
        -> scores, failures, and error feedback
        -> another retrieval round or early stop
        -> ranked_models.csv
```

A round is one complete retrieval, evaluation, and feedback cycle:

1. The retrieval provider proposes up to `max_candidates` previously unseen models.
2. The Context Scout validates the dataset schema and chooses a supported evaluation protocol.
3. Each candidate is evaluated on the configured random sample.
4. Scores, parsing failures, errors, and provenance are supplied to the next retrieval call.
5. Once enough comparable models exist, the retrieval agent, evaluation critic, and orchestrator decide whether another round is likely to improve the ranking. Unanimous no-improvement consensus stops the loop early; otherwise it continues up to `--max-rounds`.

Provider retries repeat a failed or malformed retrieval call; they are not additional rounds. Candidates are deduplicated across rounds. Only successful, comparable results with a finite score and at least one labeled example can enter the final ranking.

## Main components

| Component | Responsibility |
| --- | --- |
| `run_selection_loop.py` | Multi-round control flow, feedback, deduplication, stopping, and final ranking |
| Retrieval provider | Connects the loop to the Artifact Linker retrieval pipeline and coordinates the continuation discussion |
| `external/artifact-linker/` | Pinned retrieval submodule containing model evidence, dataset descriptions, and cached embeddings |
| `run_evaluation_conditions.py` | Candidate-batch evaluation and result collection |
| `run_eval_agent.py` | Dataset/model inspection, planning, evaluation, and auditing |
| `evaluate_hf_pair.py` | Deterministic model-dataset execution and metric calculation |
| `refinement_agent_core.py` | Structured evaluation-error feedback |
| `result_contract.py` | Shared success, failure, and comparability rules |
| `config/` | Reusable manifests and evaluation configuration |
| `agents/pipeline/` | Runtime agent contracts and boundaries |

## Setup

Clone the repository and its retrieval submodule:

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
pip install -e .
```

The cached retrieval path requires a logged-in Codex CLI, Hugging Face authentication for gated models, and GPU access for model evaluation and GNN inference. The Artifact Linker graph and trained GNN checkpoint must also be available to the retrieval component.

`OPENAI_API_KEY` is not required. `VOYAGE_API_KEY` is needed only to embed a dataset that is absent from the committed embedding cache.

## Running the loop

The controller is command-line based. Start with:

```bash
python run_selection_loop.py --help
```

A typical bounded run uses:

```bash
python run_selection_loop.py \
  --manifest /path/to/manifest.json \
  --stage smoke \
  --runner codex \
  --max-rounds 3 \
  --target-ranked-models 5 \
  --provider-retries 3 \
  --smoke-limit 100 \
  --sample-size 20 \
  --seed 42 \
  --project-root /path/to/runtime \
  --output-root /path/to/results
```

The important controls are:

- `--max-rounds`: maximum number of retrieval/evaluation/feedback cycles;
- `--target-ranked-models`: requested number of valid ranked models per dataset;
- `--provider-retries`: retries for a failed retrieval response within a round;
- `--smoke-limit`: maximum evaluated examples per candidate during selection;
- `--sample-size`: examples inspected while constructing evaluation context;
- `--seed`: reproducible sampling seed;
- `--resume`: continue an interrupted output directory.

Full evaluation is a separate phase: freeze the selected candidates, then evaluate each model-dataset pair on the desired larger sample without further retrieval or refinement.

## Outputs

Each run writes aggregate files under its configured output root:

- `ranked_models.csv`: final ranked candidates for every dataset;
- `selection_loop_results.csv`: every evaluated candidate and its provenance;
- `selected_models.csv`: the best valid measured model per dataset;
- `selection_loop_summary.json`: dataset counts, stopping reasons, and run metadata.

Each dataset's round directory retains the candidate handoff, evaluation context, measured results, audit, feedback, and continuation decision. Generated run outputs and local cluster files are ignored by Git.

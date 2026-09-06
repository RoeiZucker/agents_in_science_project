# Command-Line Slurm Selection Loop

Use `submit_selection_loop.sh` to run the multi-round selection loop on a Slurm
GPU node. Scheduler options go before `--`; every option after `--` is forwarded
unchanged to `run_selection_loop.py`.

## Queued smoke run

```bash
cd /sci/nosnap/michall/roeizucker/jupyter_notebooks/Tom_Hope_Project/agents_project/hf-eval-agent
./submit_selection_loop.sh --time 1-00:00:00 --mem 128G \
  --gres gg:g4:1 --constraint 'firefoot|cyril' -- \
  --manifest config/artifact_linker_two_dataset_test.json \
  --dataset-subset-file config/artifact_linker_two_datasets.txt \
  --stage smoke --runner codex --max-rounds 5 \
  --smoke-limit 3 --codex-timeout 300 --codex-bypass-sandbox \
  --trust-remote-code
```

Run `./submit_selection_loop.sh --help` for scheduler options and
`bash run_selection_loop_slurm.sbatch --help` for loop options. Add `--dry-run`
before the separator to inspect the exact `sbatch` command without submitting.

## Results and resume

By default, each job writes results to
`$HF_EVAL_RUNTIME/eval_results/_selection_loop_slurm/$SLURM_JOB_ID`, with
`HF_EVAL_RUNTIME` defaulting to `/sci/labs/michall/roeizucker/hf_eval_runtime`.
Supply an explicit `--output-root` for a resumable location:

```bash
./submit_selection_loop.sh -- \
  --manifest config/artifact_linker_two_dataset_test.json \
  --dataset-subset-file config/artifact_linker_two_datasets.txt \
  --stage smoke --runner codex --max-rounds 5 \
  --output-root /sci/labs/michall/roeizucker/hf_eval_runtime/eval_results/my_loop
```

To resume after timeout or cancellation, submit the same command with `--resume`.
Slurm stdout and stderr default to `selection_loop_<job-id>.out` and `.err` in
the runtime root.

## Interactive allocation

Run the payload directly inside an existing allocation:

```bash
salloc -A tomhope --gres=gg:g4:1 --constraint='firefoot|cyril' \
  --mem=128G -c 8 --time=12:00:00
srun ./run_selection_loop_slurm.sbatch \
  --manifest config/artifact_linker_two_dataset_test.json \
  --dataset-subset-file config/realworldqa_only.txt \
  --stage smoke --runner codex --max-rounds 5 --smoke-limit 3
```

The payload reuses the shared Hugging Face caches and project Python environment.
Override their locations with `HF_EVAL_RUNTIME` and `HF_EVAL_PYTHON`. Submission
uses `submit_with_hf_auth.sh`, which forwards the existing Hugging Face login
without printing the token.

# Development

Recommended local loop:

```bash
python -m py_compile \
  evaluate_hf_pair.py \
  eval_agent_core.py \
  inspect_hf_dataset.py \
  inspect_hf_model.py \
  audit_eval_results.py \
  run_eval_agent.py \
  tests/test_eval_agent_core.py

python -m unittest tests.test_eval_agent_core -v
```

Use `run_eval_agent.py --stage plan` before `--stage smoke`, and use
`--stage full` only after smoke tests pass.


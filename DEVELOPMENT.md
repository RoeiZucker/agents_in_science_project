# Development

Run the offline suite and syntax checks before every reviewer handoff:

```bash
python -m unittest discover -s tests -v
python -m py_compile   result_contract.py eval_agent_core.py evaluate_hf_pair.py gguf_backend.py run_eval_agent.py   run_evaluation_conditions.py run_condition_dataset_cycle.py run_selection_loop.py   refinement_agent_core.py create_condition_run_report.py retry_failed_condition_pairs.py   make_codex_prompt.py
```

Do not run model inference or pipeline agents while changing agent descriptions. Use `run_selection_loop.py --stage plan --runner script` only when a metadata-level wiring check is needed.

Review gate: Developer handoff -> Partner Reviewer approval -> Professor Reviewer approval. Any rejection returns to Developer, and Professor rejection requires fresh Partner approval.

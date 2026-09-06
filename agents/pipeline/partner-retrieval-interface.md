# Partner Retrieval Interface

This role belongs to the project partner and is not implemented here. It consumes dataset context and optional evaluation feedback, then writes a candidate handoff.

Each handoff must include `handoff_version`, `owner`, `mock`, `retrieval_agent`, `dataset`, `split`, `round`, and a `candidates` list. Every candidate includes `candidate_id`, `condition`, `model`, `rank`, `score`, `source`, `intended_use`, and `reason`.

The current evaluator measures direct inference only, so every candidate must use `"intended_use": "direct_inference"`. Fine-tuning and other use modes are rejected before evaluation until separate protocols and result joins exist for them.

Mocks under `mock_candidates/` exercise this contract only. They contain no retrieval or model-selection algorithm and must remain labeled with `mock: true`.

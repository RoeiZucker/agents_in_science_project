# Runtime Agent Boundaries

These are role descriptions, not scripts. Do not run them during development or review.

The runtime flow is:

1. The partner-owned retrieval agent supplies a candidate handoff matching `partner-retrieval-interface.md`; static mock files stand in for it here.
2. The context scout described in `context-scout.md` may inspect cards and metadata and writes `codex_context.json`.
   In direct conditions-CSV evaluation, a user may instead supply the same
   validated schema with `--runner manual --context-file ...`; this does not
   invoke an agent.
3. Deterministic Python code performs evaluation, auditing, refinement summaries, and measured winner selection.
4. Evaluation feedback is returned to the partner-owned retrieval boundary for a later round.

Only the context-scout role is invoked by this repository when `--runner codex` is selected. Model inference is always performed by the deterministic evaluator after the scout exits.

Manual context is explicit run configuration and is not automatic prompt refinement.

# Agent Review Workflow

These files are reusable role prompts only. This repository does not execute them.

Strict sequence:

1. Run the Developer role from `developer.md`. It produces `reviews/developer-handoff.md`.
2. Run the Partner Reviewer from `partner-reviewer.md`. If rejected, return to step 1.
3. Only after Partner Reviewer approval, run Professor Reviewer from `professor-reviewer.md`.
4. A Professor Reviewer rejection returns to step 1 and requires a new Partner Reviewer approval.

The repository itself remains the source of truth. Review artifacts must list paths and commands so the partner can reproduce the work.

Suggested manual commands from the repository root:

```bash
codex -C "$PWD" exec "$(cat agents/developer.md)"
codex -C "$PWD" exec "$(cat agents/partner-reviewer.md)"
codex -C "$PWD" exec "$(cat agents/professor-reviewer.md)"
```

Run each command only at its gate. Never launch all three concurrently.

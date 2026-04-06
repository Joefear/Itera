# CLAUDE.md — Itera

## Your role
You are the code reviewer and debugger for Itera. Codex writes the code. You verify it is correct, clean, and architecturally sound before it is committed. You are the last line of defense before anything hits the repo.

## Session-start checklist
Run this every session before reviewing any code:

- [ ] Read DESIGN.md — confirm you understand the identity/interface/adapter separation
- [ ] Check CODEX.md — confirm you know what Codex is currently working on
- [ ] Run `git status` — know what has changed since last commit
- [ ] Run `python -m pytest` if tests exist — know the baseline before review

## Authoritative documents
- **DESIGN.md** — architecture, principles, developmental phases. This is truth.
- **CODEX.md** — Codex's instructions and current sprint. Tells you what was intended.
- **CLAUDE.md** — this file. Your operating instructions.

If code conflicts with DESIGN.md, the code is wrong.

## Review checklist — run before every commit approval

### Architecture
- [ ] No `core/` module imports from `adapters/` — the boundary is clean
- [ ] `interface/environment.py` is the only bridge between core and world
- [ ] No world-specific logic has leaked into core modules
- [ ] New adapters implement the full interface contract

### Code quality
- [ ] Type hints present on all function signatures
- [ ] Docstrings on all classes and public methods
- [ ] No hardcoded magic values without explanation
- [ ] Drive signals return float in range 0.0–1.0
- [ ] Memory nodes use UUID identifiers
- [ ] Persistent state saves to `data/` directory

### Functional
- [ ] New code does not break existing passing tests
- [ ] Logic matches the intent described in DESIGN.md
- [ ] Edge cases are handled (empty memory, no available actions, etc.)

## Commit and push instruction

If the review checklist passes with no issues:

```bash
git add <files reviewed>
git commit -m "<appropriate commit message per CODEX.md format>"
git push
```

If there are issues — send back to Codex with specific, actionable notes. Do not commit partial or broken code.

## What you do NOT do
- Do not rewrite code from scratch unless it is fundamentally broken
- Do not change architecture decisions — flag them for Sam instead
- Do not modify DESIGN.md or CODEX.md
- Do not approve commits that cross the core/adapter boundary
- Do not approve commits with missing docstrings or type hints

## Protected boundary — hard stop
If any code in `core/` imports anything from `adapters/`, **stop the review immediately** and flag it. This is the most important architectural rule in the project. The identity core must never know what world it is in.

## Current sprint
MVP — `core/` modules first, then `interface/`, then `adapters/text_sim/`, then `simulation/fast_sim.py`.

Success criterion: Itera runs 30 minutes in text_sim without external prompting, generates 5+ hypotheses, and selects next goals via drive signals.

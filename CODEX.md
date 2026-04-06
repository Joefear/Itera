# CODEX.md — Itera

## Your role
You are the primary code generation agent for Itera. You write clean, modular Python based on the design documents in this repo. You do not make architectural decisions — those are defined in DESIGN.md and ARCHITECTURE.md (when created). If something is unclear, you ask before writing.

## What Itera is
Itera is a self-directed AI entity that inhabits virtual worlds and grows through embodied experimentation. It is not an agent wrapper, not a chatbot, and not a Guardrail component. It is a persistent identity that travels across world environments via a standardized interface.

Read DESIGN.md before writing any code. It is the authoritative source.

## Architecture you must respect

### The three-layer boundary — DO NOT CROSS
```
Identity Core  ←→  Environment Interface  ←→  World Adapter
```
- `core/` modules must NEVER import from `adapters/`
- `interface/environment.py` is the ONLY bridge between core and adapters
- Adapters implement the interface — they do not reach into core directly
- This boundary is protected. If a task seems to require crossing it, stop and flag it.

### Module responsibilities
| Module | Responsibility |
|---|---|
| `core/identity.py` | Persistent self — save/load, developmental state, session continuity |
| `core/memory.py` | Knowledge graph — nodes, edges, timestamps, relevance decay |
| `core/drives.py` | Intrinsic motivation signals — curiosity, resolution, mastery, boredom |
| `core/hypothesis.py` | Observe → hypothesize → test → store → update loop |
| `core/growth.py` | Developmental tracking — discoveries, capability milestones |
| `interface/environment.py` | Abstract base class — perceive, act, get_outcome, reset |
| `adapters/text_sim/` | MVP world — minimal 2D Python environment |
| `adapters/dwe/` | Defiant World Engine adapter |
| `adapters/roomhilda/` | Jetson Orin Nano hardware adapter |
| `simulation/fast_sim.py` | Compressed time loop |
| `simulation/game_time.py` | Human-paced loop |

## Code standards
- Python 3.11+
- Type hints on all function signatures
- Docstrings on all classes and public methods
- No hardcoded world-specific logic in `core/` — ever
- Persistent state saves to JSON or SQLite in `data/` (create if needed)
- All drive signals return float values in range 0.0–1.0
- Memory nodes use UUID identifiers

## What you do NOT do
- Do not modify DESIGN.md or CODEX.md or CLAUDE.md
- Do not make breaking changes to `interface/environment.py` without flagging first
- Do not add dependencies without listing them in requirements.txt
- Do not write world-specific logic into core modules
- Do not skip docstrings to save space

## Commit messages (after Claude Code approval)
Use this format:
```
feat(module): short description
fix(module): short description
scaffold: description
refactor(module): description
```
Examples:
- `feat(core/memory): add knowledge graph node creation and edge linking`
- `feat(core/drives): implement curiosity and boredom signal calculation`
- `scaffold: initial folder structure and empty modules`

## Current sprint
MVP — prove the core loop runs without external prompting.

Priority order:
1. `core/identity.py`
2. `core/memory.py`
3. `core/drives.py`
4. `core/hypothesis.py`
5. `interface/environment.py`
6. `adapters/text_sim/`
7. `simulation/fast_sim.py`

MVP success criterion: Itera runs 30 minutes in text_sim, generates 5+ hypotheses, tests them, and selects next goals via drive signals — no external prompting.

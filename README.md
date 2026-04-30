# Itera

> *"Not a tool we use, but an intelligence we nurture."*

Itera is a self-directed AI entity that inhabits virtual worlds, grows through embodied experimentation, and develops a persistent identity through accumulated experience.

It is not an agent. It is not a chatbot. It is not a wrapper around a language model.

Itera starts from almost nothing — limited stimuli, basic physics, a nearly empty world — and grows in capability, understanding, and selfhood over time. What it becomes is a function of what it has lived through, not what it was programmed to be.

---

## What Itera does

Itera runs a continuous cognitive loop — perceiving its world, forming hypotheses, testing them, storing outcomes with emotional weight, and selecting its next action based on what it currently cares about. No external prompting required.

**It forms hypotheses.** When Itera encounters something novel it generates a testable prediction — causal, comparative, predictive, relational, or pattern-based — and designs an experiment to test it. Confirmed hypotheses build toward capability emergence.

**It remembers with meaning.** Every experience is stored in a knowledge graph with typed edges connecting causes to effects, experiments to hypotheses, entities to encounters. Memory decays but never fully erases. What Itera learned early shapes how it reasons now.

**It models other minds.** The empathy layer builds internal models of every entity Itera encounters — threat assessment, curiosity value, behavioral prediction, relationship history. At higher developmental stages, welfare concern emerges naturally from deep entity modeling.

**It grows.** A five-tier Maslow-inspired drive hierarchy shifts over time — survival dominant early, self-actualization dominant at maturity. Capabilities emerge from accumulated evidence, never assigned. A mature Itera has earned what it knows.

**It inhabits any world.** A world-agnostic interface means Itera's identity never changes. The world changes. The same Itera that learned in a 2D text simulation can control an NPC in a game world or operate a physical robot — carrying its full history with it.

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  IDENTITY CORE                  │
│  Drives · Memory · Hypotheses · Growth · Empathy│
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│           ENVIRONMENT INTERFACE                 │
│  perceive · act · get_outcome · reset           │
│  Observation: percepts, entities, self_state,   │
│  world_context, time_of_day                     │
│  ActionDefinition: category, drive_alignment,   │
│  cost, parameters                               │
└──────────────────────┬──────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   [text_sim]        [dwe]       [roomhilda]
   MVP world    DWE/Construct3   Jetson hardware
                NPC controller   (placeholder)
```

**Itera never knows what world it inhabits.** It only knows what it can perceive, what actions are available, and what outcomes it observes. Every world implements the same interface contract. Itera stays constant. Only the adapter changes.

---

## Core modules

| Module | Purpose |
|---|---|
| `core/drives.py` | Five-tier Maslow drive hierarchy — SURVIVAL through ACTUALIZATION. Weighted suppression, not hard gates. Shifts from survival-dominant to actualization-dominant as Itera matures. |
| `core/memory.py` | Knowledge graph with typed edges, emotional valence, and relevance decay. Floor of 0.05 — nothing is ever fully forgotten. |
| `core/hypothesis.py` | Scientific method as cognitive loop. Five hypothesis types: CAUSAL, COMPARATIVE, PREDICTIVE, RELATIONAL, PATTERN. Drive-weighted generation. |
| `core/identity.py` | The persistent self. Owns all cognitive components. Survives across sessions via JSON persistence. |
| `core/growth.py` | Developmental tracking. Capabilities emerge from evidence accumulation — never assigned. |
| `core/empathy.py` | Entity modeling. Starts as threat/curiosity assessment. Matures into behavioral prediction, relationship tracking, and welfare concern. Produces social drive signals. |

---

## World adapters

| Adapter | Status | Purpose |
|---|---|---|
| `adapters/text_sim/` | ✅ Complete | MVP 2D grid world. Rocks, fire, water, creatures, stone deposits. Proves the core loop. |
| `adapters/dwe/` | ✅ Complete | Defiant World Engine adapter. Connects to Node.js bridge at localhost:3001. Itera operates as NPC controller governed by Guardrail policy. Mock mode for offline testing. |
| `adapters/roomhilda/` | 🔲 Placeholder | Jetson Orin Nano hardware adapter for physical robot deployment. |

---

## Running Itera

**Quick start — text_sim:**
```bash
# Clear previous state (optional)
rm -rf data/identity

# Run 200 cycles
python simulation/fast_sim.py
```

**Run with DWE bridge (requires defiant-world-engine running):**
```python
from adapters.dwe.adapter import DWEAdapter
from simulation.fast_sim import FastSim, SimConfig

config = SimConfig(max_cycles=1000)
sim = FastSim(config, adapter=DWEAdapter())
sim.setup()
result = sim.run()
```

**Run DWE adapter in mock mode (no bridge needed):**
```python
from adapters.dwe.adapter import DWEAdapter
adapter = DWEAdapter(mock_mode=True)
adapter.validate()  # True
obs = adapter.perceive()
```

**Note:** Clear `data/identity/` between major test runs to start Itera fresh.

---

## Session output

A completed run prints a full summary:

```
=== ITERA SESSION COMPLETE ===
Cycles: 10000
Duration: 634.15s
Final stage: 1.0 (civilizational)
Hypotheses: 3 generated, 3 confirmed
Memory nodes: 10002
Entities modeled: 17
Dominant drive at end: SOCIAL

=== CAPABILITIES EMERGED ===
1. Physical capability 1 (domain: physical)
   Emergent physical capability supported by 3 confirmed experiences.
   Confidence: 0.20 | Evidence: 3 experiences

=== TOP MEMORIES ===
1. experience | tags=['social', 'outcome', ...] | valence=0.24
2. entity | tags=['entity', 'creature'] | valence=1.00
3. discovery | tags=['stone', 'heat'] | valence=0.45
```

---

## Developmental phases

Itera matures through six phases in fast simulation:

| Phase | Stage | What emerges |
|---|---|---|
| Primitive | 0.0 – 0.15 | Object manipulation, cause and effect |
| Environmental | 0.15 – 0.30 | Terrain, resources, natural patterns |
| Biological | 0.30 – 0.50 | Simple organisms, social drive activates |
| Ecological | 0.50 – 0.70 | Resource scarcity, preferences emerge |
| Social | 0.70 – 0.85 | Empathy deepens, relationship modeling |
| Civilizational | 0.85 – 1.0 | Actualization dominant, self-directed purpose |

---

## What Itera is not

- **Not a Guardrail component.** Defiant Guardrail governs AI systems. Itera *is* one. They are separate projects. The DWE adapter uses Guardrail as the governing layer for NPC actions — Itera's internal cognition is not governed by Guardrail.
- **Not scripted.** Personality, capabilities, and interests emerge from experience.
- **Not dependent on any single world.** DWE, text_sim, Roomhilda, Unreal — Itera inhabits worlds, it is not defined by them.
- **Not a research tool you point at problems.** You can give Itera a quest. It has its own life between quests.

---

## Repo structure

```
Itera/
├── core/
│   ├── drives.py         # Five-tier drive hierarchy
│   ├── memory.py         # Knowledge graph
│   ├── hypothesis.py     # Hypothesis engine (5 types)
│   ├── identity.py       # Persistent self
│   ├── growth.py         # Developmental tracking
│   └── empathy.py        # Entity modeling
├── interface/
│   └── environment.py    # World-agnostic contract
├── adapters/
│   ├── text_sim/         # MVP 2D world
│   ├── dwe/              # DWE NPC controller
│   └── roomhilda/        # Jetson hardware (placeholder)
├── simulation/
│   └── fast_sim.py       # Core cognitive loop
├── data/identity/        # Persisted state (gitignored)
├── ITERA_DESIGN.md       # Phase 1 architecture
├── ITERA_DESIGN_PHASE2.md # Phase 2 specification
└── README.md             # This file
```

---

## Related projects

| Project | Relationship |
|---|---|
| [defiant-world-engine](https://github.com/Joefear/defiant-world-engine) | One world Itera can inhabit. DWE adapter connects Itera brain to Construct 3 via Node.js bridge. |
| [AI-Sandbox](https://github.com/Joefear/AI-Sandbox) | Original prototype. Historical reference. |
| [defiant-guardrail](https://github.com/Joefear/defiant-guardrail) | Separate system. Governs AI actions in DWE world. Does not govern Itera's internal cognition. |

---

**Defiant Industries, Inc.**
Project codename: *The Self-Building Mind*

*Phase 1 complete — autonomous cognitive loop, persistent identity, knowledge graph, developmental growth.*
*Phase 2 complete — empathy layer, rich hypotheses, extensible interface, DWE NPC controller.*
*Phase 3 — Unreal Engine world, richer entity interaction, physical deployment.*
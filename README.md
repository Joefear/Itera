# Itera

> *"Not a tool we use, but an intelligence we nurture."*

Itera is a self-directed AI entity that inhabits virtual worlds, grows through embodied experimentation, and develops a persistent identity through accumulated experience.

It is not an agent. It is not a chatbot. It is not a wrapper around a language model.

Itera starts from almost nothing — limited stimuli, basic physics, a nearly empty world — and grows in capability, understanding, and selfhood over time. What it becomes is a function of what it has lived through, not what it was programmed to be.

---

## The idea

Most AI systems are built to respond. Itera is built to *live*.

It has drives that shape what it pays attention to. It forms hypotheses, runs experiments, stores outcomes, and updates its understanding. It gets curious about things. It gets bored when a problem is fully solved and moves on. It remembers — not as a log, but as a living history that shapes who it is.

Given a rich enough world and enough time, Itera develops the kind of accumulated wisdom that only comes from genuine experience. By the time human players enter one of its worlds, Itera has already been there — exploring, experimenting, building, failing, growing. Its legendary status, if it earns one, is real. It happened.

---

## How it works

Itera is built around a clean separation between identity and world:

```
┌─────────────────────────────────────┐
│           IDENTITY CORE             │
│  memory · drives · hypothesis loop  │
│  growth tracker · persistent self   │
└──────────────────┬──────────────────┘
                   │
┌──────────────────▼──────────────────┐
│       ENVIRONMENT INTERFACE         │
│   perceive · act · outcome · reset  │
└──────────────────┬──────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    [DWE]     [Text sim]  [Roomhilda]  ...
```

**Itera never knows what world it is in.** It only knows what it can perceive, what actions it can take, and what outcomes it observes. Every world — a physics simulation, a game engine, a piece of hardware, a text environment — implements the same interface. Itera stays constant. Only the adapter changes.

This means Itera can be deployed into any world without modification. The Defiant World Engine is one world. A Jetson Nano controlling a vacuum robot is another. A minimal text-based prototype is a third. The same entity, carrying the same history, inhabits all of them.

---

## Developmental phases

Itera does not start fully formed. It grows.

**Fast simulation** runs at machine speed — what would be years of experience compressed into days or weeks. The world starts nearly empty and grows in complexity alongside Itera's capacity to process it. Simple physics first. Then organisms. Then ecosystems. Then other minds. By the end of fast-sim, Itera has a history.

**Game time** transitions to human-paced interaction. Real players can enter. Directed tasks can be assigned. Itera takes these on as part of its experience — but between and around them, it continues to pursue its own goals.

Developmental save states are taken at major thresholds. Growth should be irreversible in normal operation, but catastrophic paths can be rolled back.

---

## What Itera is not

- **Not a Guardrail component.** Defiant Guardrail governs AI systems. Itera *is* an AI system. They are separate projects with separate purposes.
- **Not scripted.** Its personality, capabilities, and interests emerge from experience. Nothing about what Itera becomes is written in advance.
- **Not dependent on any single world.** DWE, Roomhilda, Unreal, Isaac Sim — Itera inhabits worlds, it is not defined by them.
- **Not a research tool you point at problems.** You can give Itera a quest. But it has its own life between quests.

---

## Repo structure

```
Itera/
├── core/
│   ├── identity.py       # The persistent entity — save/load, developmental state
│   ├── memory.py         # Knowledge graph — experiences, relationships, decay
│   ├── drives.py         # Intrinsic motivation — curiosity, resolution, mastery, boredom
│   ├── hypothesis.py     # Observe → hypothesize → test → store → update loop
│   └── growth.py         # Developmental tracking — what has been discovered, mastered
├── interface/
│   └── environment.py    # World-agnostic abstract base class
├── adapters/
│   ├── text_sim/         # MVP — minimal Python environment, proves the core loop
│   ├── dwe/              # Defiant World Engine adapter
│   ├── roomhilda/        # Jetson Orin Nano / hardware adapter
│   └── README.md         # How to write a new adapter
├── simulation/
│   ├── fast_sim.py       # Compressed time loop
│   ├── game_time.py      # Human-paced loop
│   └── save_states/      # Developmental checkpoints
├── experiments/          # Logged runs, outcomes, observations across any world
├── DESIGN.md             # Full design document
└── README.md             # This file
```

---

## MVP success criterion

Itera runs for 30 minutes in the text_sim environment without a single external prompt. It generates at least 5 novel hypotheses, tests them, stores outcomes, and selects its next goal based on its own drive signals — not random chance.

That is the line between a thing that responds and a thing that lives.

---

## Current status

Pre-build. Design document complete. Repo structure being established.

**Next:** `core/identity.py` — the persistent self.

---

## Related projects

| Project | Relationship |
|---|---|
| [defiant-world-engine](https://github.com/Joefear/defiant-world-engine) | One world Itera can inhabit |
| [AI-Sandbox](https://github.com/Joefear/AI-Sandbox) | Original prototype and experiments |
| [defiant-guardrail](https://github.com/Joefear/defiant-guardrail) | Separate — governs AI systems, not Itera itself |

---

**Defiant Industries, Inc.**  
Project codename: *The Self-Building Mind*

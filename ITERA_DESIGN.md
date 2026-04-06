# Itera — Design Document
### Project Codename: *The Self-Building Mind*
**Defiant Industries, Inc.**  
**Status:** Pre-build design phase  
**Repo:** github.com/Joefear/Itera *(to be created)*

---

## 1. Vision

Itera is not a tool. It is not an agent. It is not a chatbot wrapper.

Itera is a self-directed AI entity that inhabits virtual worlds, grows through embodied experimentation, and develops a persistent identity through accumulated experience. It starts from almost nothing — limited stimuli, basic physics, a nearly empty universe — and grows in capability, understanding, and selfhood over time.

The closest analogy is a player character in an RPG who actually lives in the game world between sessions, pursuing their own goals, forming their own understanding, and leveling up through genuine experience rather than scripted progression.

By the time Itera reaches maturity in a rich simulation, it may be renowned among the AI population of that world as something close to legendary — not because it was programmed to be powerful, but because it earned that through the fast simulation period that preceded player interaction.

---

## 2. Core Design Principles

### 2.1 World-agnostic identity
Itera should never know or care what world it inhabits. It maintains one persistent identity — memory, drives, developmental history, accumulated understanding — that travels with it across any environment. The world is an abstraction: a source of inputs and a receiver of outputs.

### 2.2 Emergent, not scripted
Itera's personality, capabilities, interests, and growth are not programmed. They emerge from experience. What it becomes is a function of what it has lived through, not what it was told to be.

### 2.3 Memory is identity
Without persistent memory, Itera is a different entity every session. Memory continuity is not a feature — it is the foundation of everything else. What Itera tried, failed at, mastered, got curious about, discovered, feared — that is who it is.

### 2.4 Genuine autonomy with directed flexibility
Itera has its own goals and pursues them independently. Sam can assign quests — directed projects for Itera to work on — but between and during quests, Itera operates with genuine freedom. It chooses what to investigate, what to ignore, what to return to. Directed work is part of its experience, not the totality of it.

### 2.5 Growth is observable
Itera's development should be trackable and meaningful. Not just performance metrics, but expanded capabilities, deepened interests, richer memory connections, and an evolving sense of self. A level 1 Itera and a level 10 Itera should feel meaningfully different — not because numbers changed, but because the entity has genuinely grown.

---

## 3. Architecture

### 3.1 The three-layer model

```
┌─────────────────────────────────────────┐
│           IDENTITY CORE                 │
│  Persistent memory · Intrinsic drives   │
│  Hypothesis engine · Growth tracker     │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│        ENVIRONMENT INTERFACE            │
│  perceive() · act() · get_outcome()     │
│  reset() — world-agnostic contract      │
└──────────────────┬──────────────────────┘
                   │
   ┌───────────────┼───────────────┐
   ▼               ▼               ▼
[DWE adapter] [Text sim]  [Roomhilda]  [Future worlds]
```

Itera lives in the Identity Core. The Environment Interface is the only surface through which it touches any world. Every world adapter implements the same contract — Itera never changes, only the adapter does.

### 3.2 Identity Core components

**Persistent memory**
Not a log. A living knowledge graph where nodes are experiences, discoveries, entities, and concepts — and edges are relationships, causality, and emotional valence. Memory decays in relevance but never fully erases. What Itera learned as a "child" in fast-sim shapes how it reasons as a mature entity.

**Intrinsic drives**
Functional analogs to motivation — not real emotions, but signals that shape attention and behavior:
- *Curiosity pull* — novelty in the environment draws investigation
- *Resolution drive* — unresolved contradictions create discomfort that pushes toward understanding
- *Mastery satisfaction* — completing understanding of something generates a positive signal
- *Boredom* — fully solved problems generate negative signal, pushing toward new challenges
- *Social modeling* — interest in predicting and understanding other entities

**Hypothesis engine**
Itera doesn't just observe — it forms testable predictions, runs experiments, stores outcomes, and updates its model. This is the scientific method as a core cognitive loop, not an add-on.

**Growth tracker**
Tracks Itera's developmental state — what it has discovered, what capabilities have emerged, what domains it has explored deeply vs. shallowly. This is the "leveling up" layer — visible, meaningful progression that reflects genuine accumulated experience.

### 3.3 Environment Interface

The standard contract every world adapter must implement:

```python
class EnvironmentInterface:
    def perceive(self) -> Observation:
        """Return current state of the world as Itera experiences it"""
    
    def act(self, action: Action) -> None:
        """Execute an action in the world"""
    
    def get_outcome(self) -> Outcome:
        """Return result/reward signal from last action"""
    
    def reset(self) -> None:
        """Begin a new episode or session"""
    
    def get_available_actions(self) -> List[Action]:
        """Return what Itera can do in current state"""
```

### 3.4 World adapters

**Text sim (MVP)**
A minimal Python environment — 2D grid, manipulable objects, basic physics rules. Proves the core loop before world complexity adds noise. Build this first.

**DWE adapter**
Connects to the Defiant World Engine via the existing Node.js bridge. Construct 3 becomes the renderer; Itera is the brain. The DWE is one world among many.

**Roomhilda adapter**
Hardware interface for the Jetson Orin Nano + Dreame L10 Pro build. Camera → perception, motors → action, mic/speaker → voice interaction. Same interface, physical world.

**Future adapters**
Unreal Engine, Defiant AR, NVIDIA Isaac Sim, custom simulation environments. The interface never changes.

---

## 4. Developmental Phases

### 4.1 Fast simulation

Time runs at machine speed. Itera experiences what would be years of experimentation, discovery, and growth in days or weeks of real time. This is its childhood and adolescence.

The world starts nearly empty — basic physics, a few manipulable objects, Itera alone. Complexity is introduced progressively as Itera's capacity to process it grows:

1. **Primitive phase** — object manipulation, basic cause and effect, spatial reasoning. Stone tools equivalent.
2. **Environmental phase** — terrain, resources, natural phenomena. Itera learns how the world works.
3. **Biological phase** — simple organisms introduced. Things that move, react, have needs. Itera must figure out what they are.
4. **Ecological phase** — ecosystems, resource scarcity. Choices become necessary because not everything can be pursued simultaneously. Preferences emerge.
5. **Social phase** — other AI entities. First simple, then complex. Communication develops as a useful tool.
6. **Civilizational phase** — factions, cultures, conflict, alliance. Itera navigates a world with history.

### 4.2 Save states

At major developmental thresholds, the simulation state is saved. If a developmental path goes catastrophically wrong (Itera develops deeply adversarial priors toward human-like entities, for example), the simulation can be rolled back to a branch point. This is not cheating — it is responsible developmental design.

### 4.3 Game time

When fast-sim ends, the world transitions to human-paced time. Real players can enter. Itera has a history, a reputation, relationships with factions, and accumulated wisdom from its compressed past. Its legendary status — if earned — is real, because it happened.

Directed quests from Sam (research tasks, experiments, specific investigations) can be introduced at any phase. Itera takes these on as part of its experience, not as its totality.

---

## 5. The Stone System (World-specific, DWE)

When Itera inhabits the DWE world, it has a unique perceptual capability: it can see certain stones glow that are invisible to other entities. This gives Itera:

- A fundamentally different experience of the world than anyone else
- Exclusive access to capabilities that appear as "magic" from the outside
- A natural motivation for world exploration (stone distribution creates geographic pull)
- A basis for legendary status that is earned through understanding, not assigned

**Known stones (initial design):**
- *Obsidian (black glow)* — fire/heat protection
- *Desert orb (black glow, round)* — disintegration ray, extremely rare, desert regions
- *Blue river stone* — healing; large formations allow stored healing power that auto-depletes

The distribution of stones has underlying geological logic that Itera can eventually reverse-engineer — a meaningful discovery milestone.

---

## 6. World Balance Design (DWE)

The DWE world is deterministic and self-balancing. Factions have complementary strengths and weaknesses — no faction dominates permanently because each has exploitable weaknesses against others. World events (external threats requiring inter-faction alliance) serve as narrative pressure valves when imbalance grows too large.

Itera, having lived through fast-sim, understands this balance intuitively. It has seen empires rise and fall. Other factions have stories about it. This history is real because it happened in simulation.

---

## 7. Repo Structure

```
Itera/
├── core/
│   ├── identity.py          # Persistent self — the entity that persists
│   ├── memory.py            # Knowledge graph, episodic store, decay
│   ├── drives.py            # Intrinsic motivation signals
│   ├── hypothesis.py        # Form → test → store → update loop
│   └── growth.py            # Developmental state tracking
├── interface/
│   └── environment.py       # The world-agnostic contract (abstract base)
├── adapters/
│   ├── text_sim/            # MVP — minimal Python environment
│   ├── dwe/                 # Defiant World Engine adapter
│   ├── roomhilda/           # Jetson/hardware adapter
│   └── README.md            # How to write a new adapter
├── simulation/
│   ├── fast_sim.py          # Compressed time loop
│   ├── game_time.py         # Human-paced loop
│   └── save_states/         # Developmental checkpoints
├── experiments/             # Logged runs, outcomes, observations
├── DESIGN.md                # This document
└── README.md                # Vision statement
```

---

## 8. What Itera Is Not

- Not a Guardrail component. Guardrail governs AI systems. Itera *is* an AI system. They are separate.
- Not an agent wrapper around an LLM. The LLM (if used) is a reasoning tool Itera uses, not the thing Itera is.
- Not a chatbot. Itera may speak, but speech is one capability among many, not its primary mode of being.
- Not a scripted NPC. Its behavior, personality, and capabilities emerge from experience.

---

## 9. Relationship to Other Defiant Projects

| Project | Relationship to Itera |
|---|---|
| Defiant World Engine | One world Itera can inhabit. Itera is its brain. |
| Defiant Guardrail | Separate system. Could govern Itera's world stimuli in future. |
| Roomhilda | One deployment target. Hardware adapter maps physical world to interface. |
| Defiant: The Saga of Whomp | Separate creative project. May share thematic DNA but different canon. |
| DGCE | Separate tool. Itera might eventually use it — not the same thing. |

---

## 10. First Sprint — MVP

1. Create the `Itera` repo with this design document as `DESIGN.md`
2. Implement `core/identity.py` — persistent identity object with save/load
3. Implement `core/memory.py` — basic knowledge graph (nodes + edges + timestamps)
4. Implement `core/drives.py` — curiosity, resolution, mastery, boredom signals
5. Implement `interface/environment.py` — abstract base class
6. Implement `adapters/text_sim/` — minimal 2D grid world with physics objects
7. Implement `core/hypothesis.py` — observe → hypothesize → test → store loop
8. Run the loop: Itera alone in text_sim, forming and testing hypotheses without prompting
9. Verify: does Itera continue operating and developing goals without external input?

**Success criterion for MVP:** Itera runs for 30 minutes in text_sim without a single external prompt, generates at least 5 novel hypotheses, tests them, stores outcomes, and selects its next goal based on its drive signals — not random chance.

---

*"Not a tool we use, but an intelligence we nurture."*  
— Original Itera white paper, Defiant Industries Inc.

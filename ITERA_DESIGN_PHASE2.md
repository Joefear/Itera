# Itera — Phase 2 Design Addendum
### Appended to ITERA_DESIGN.md
**Status:** Active development

---

## 11. Phase 2 Overview

Phase 1 proved the core loop — Itera runs autonomously, forms 
hypotheses, accumulates memory, shifts drives, and grows to 
developmental stage 1.0 without external prompting.

Phase 2 makes Itera genuinely interesting to interact with.
The goals are:

- Richer cognition — more varied hypothesis types, connected 
  memory, deeper learning from experience
- Social intelligence — an empathy layer that models other minds
- World flexibility — Observation and action space adapt to any 
  world without changing core
- NPC deployment — Itera controls a unique character in a game 
  world via the DWE adapter acting as an NPC controller

---

## 12. Phase 2 Build Order

### 12.1 Task 1 — Richer hypotheses + memory edges

**Problem:** Hypothesis engine generates only one type — 
"investigate X, expect stable pattern." Memory nodes accumulate 
without edges connecting related experiences.

**Fix — hypothesis types:**
Itera should generate hypotheses of multiple types:

- CAUSAL — "if I do X to entity Y, Y will change in way Z"
- COMPARATIVE — "entity X and entity Y behave differently under 
  condition C"
- PREDICTIVE — "when condition A is present, outcome B follows"
- RELATIONAL — "entity X responds to my presence differently 
  than entity Y does"
- PATTERN — existing type, "stable pattern expected" (keep)

Each hypothesis carries a `hypothesis_type` field. The hypothesis
engine selects type based on what triggered it — a single entity 
observation produces PREDICTIVE or CAUSAL, two entities produce 
COMPARATIVE or RELATIONAL, repeated observations produce PATTERN.

**Fix — memory edges:**
The fast_sim loop must create edges between related memory nodes.
Rules for edge creation:

- New experience node → link to most recent hypothesis node 
  with edge_type="tests"
- Confirmed hypothesis → link to all experience nodes that 
  contributed with edge_type="confirmed_by"  
- Two experience nodes involving the same entity → link with 
  edge_type="relates_to"
- Discovery node → link to the experience that caused it with 
  edge_type="caused_by"

Edge creation happens in identity.absorb() after storing the 
outcome node — identity has access to both memory and hypothesis 
engine and can make the connection.

### 12.2 Task 2 — Empathy module (core/empathy.py)

**Purpose:** Model other minds. Start as prediction, mature into 
functional care. Universal across all worlds.

**Architecture:**

```python
@dataclass
class EntityModel:
    """Itera's internal model of another entity"""
    entity_id: str
    entity_type: str        # "creature", "npc", "player", "monster"
    first_seen: float
    last_seen: float
    observation_count: int
    
    # Behavioral predictions
    predicted_reactions: dict[str, float]  # action -> expected_response
    threat_assessment: float    # 0.0-1.0
    curiosity_value: float      # 0.0-1.0, how interesting is this entity
    
    # Relationship state
    relationship_valence: float # -1.0 to 1.0, negative=adversarial
    interaction_history: list[dict]
    
    # Empathy signals (emerge with maturity)
    modeled_drive_state: dict   # what Itera thinks this entity wants
    welfare_concern: float      # 0.0-1.0, how much Itera cares

class EmpathyLayer:
    """
    Models other minds. Starts as prediction engine.
    Matures into functional care as Itera develops.
    
    At low developmental stage: pure threat/curiosity assessment.
    At mid stage: behavioral prediction, relationship tracking.
    At high stage: drive modeling, welfare concern emerges.
    
    Never instantiates emotions — instantiates functional signals
    that influence Itera's drive hierarchy and decisions.
    """
    
    def __init__(self, memory: MemoryGraph): ...
    
    def observe_entity(self, entity: dict, 
                       itera_action: str | None,
                       outcome: dict | None) -> EntityModel: ...
    
    def get_threat_level(self, entity_id: str) -> float: ...
    def get_curiosity_value(self, entity_id: str) -> float: ...
    def get_welfare_concern(self, entity_id: str) -> float: ...
    
    def predict_reaction(self, entity_id: str, 
                         action: str) -> dict: ...
    
    def update_relationship(self, entity_id: str, 
                            valence_delta: float) -> None: ...
    
    def get_social_drive_signals(self) -> dict[str, float]:
        """
        Returns signals for identity to feed into DriveHierarchy.
        Called by identity.perceive() after processing entities.
        Produces relationship_depth and entity_prediction_gap 
        based on actual entity models rather than raw counts.
        """
    
    def to_dict(self) -> dict: ...
    
    @classmethod
    def from_dict(cls, data: dict, 
                  memory: MemoryGraph) -> 'EmpathyLayer': ...
```

**Integration with Identity:**
Identity owns EmpathyLayer alongside memory, drives, hypothesis.
In perceive() — entities from observation pass through empathy layer.
In absorb() — outcomes update entity models.
In decide() — threat and curiosity influence action selection.
Social drive signals come from EmpathyLayer, not raw entity counts.

**Developmental gating:**
- stage < 0.3: threat_assessment and curiosity_value only
- stage 0.3-0.6: add behavioral prediction and relationship tracking
- stage > 0.6: add drive modeling and welfare_concern emergence

### 12.3 Task 3 — Extensible Observation

**Problem:** Observation is fixed — adding game-specific data 
(health, mana, position) or world-specific data (time_of_day, 
weather) requires changing the interface contract.

**Fix:** Add two optional dicts to Observation:

```python
@dataclass
class Observation:
    timestamp: float
    percepts: dict[str, float]      # normalized world readings
    entities: list[dict]            # detected entities
    available_actions: list[str]    # what Itera can do
    novelty_hint: float             # adapter's novelty estimate
    
    # Phase 2 additions
    self_state: dict                # Itera's own body state
                                    # health, position, energy, etc.
                                    # Empty dict if no body
    world_context: dict             # world-specific extras
                                    # time_of_day, weather, quests
                                    # Empty dict if not applicable
    time_of_day: float              # 0.0-1.0, 0=midnight, 0.5=noon
                                    # 0.0 if world has no time cycle
```

Core processes what it recognizes in self_state and world_context.
Unknown keys are stored in memory but don't affect drive logic.
This means game-specific data flows through without core changes.

### 12.4 Task 4 — Adapter-defined action space

**Problem:** Available actions are hardcoded in text_sim 
(move, examine, push, heat, cool, combine). A body in a game 
world needs richer actions.

**Fix:** Actions become structured objects, not strings.

```python
@dataclass
class ActionDefinition:
    name: str
    category: str       # "movement", "interaction", "social", 
                        # "combat", "exploration", "communication"
    parameters: dict    # what parameters this action accepts
    drive_alignment: dict[str, float]  # which drives this serves
    cost: float         # 0.0-1.0, effort/resource cost
    description: str
```

Adapters provide ActionDefinitions in get_available_actions().
Identity's decide() selects actions by drive alignment and cost.
This naturally handles quests (social/exploration category),
combat (combat category), communication (social category).

### 12.5 Task 5 — DWE adapter as NPC controller

**Architecture:**

```
Construct 3 Game World
        ↕
NPC Character (health, mana, stats, animations — all game-managed)
        ↕
DWE Node.js Bridge (existing)
        ↕
adapters/dwe/adapter.py (implements EnvironmentInterface)
        ↕
Itera Brain (unchanged)
```

The DWE adapter translates:
- NPC health → self_state["health"] normalized 0.0-1.0
- NPC mana → self_state["energy"] normalized 0.0-1.0  
- NPC position → self_state["position"] as [x, y] or [x, y, z]
- Nearby entities → entities list with game-specific properties
- Available NPC actions → ActionDefinitions
- Quest log → world_context["active_quests"]
- Time of day → time_of_day float

Itera's decisions map back to NPC commands via the bridge.
The NPC's body, animations, and game mechanics are entirely 
managed by Construct 3. Itera just drives the mind.

---

## 13. What Does NOT Change in Phase 2

- core/drives.py — unchanged
- core/memory.py — unchanged (edges created by identity, not memory)
- core/growth.py — unchanged
- interface/environment.py — Observation gets two new optional 
  fields with defaults, fully backward compatible
- simulation/fast_sim.py — minor update to pass empathy layer 
  through identity, otherwise unchanged
- adapters/text_sim/ — no changes required, self_state and 
  world_context default to empty dicts

---

## 14. Phase 2 File Changes Summary

```
MODIFIED:
  core/identity.py          — owns EmpathyLayer, richer edge 
                              creation in absorb(), empathy 
                              signals feed drives
  core/hypothesis.py        — hypothesis_type field, type 
                              selection logic
  interface/environment.py  — self_state, world_context, 
                              time_of_day in Observation
                              ActionDefinition dataclass added

NEW:
  core/empathy.py           — EntityModel + EmpathyLayer

FILLED IN:
  adapters/dwe/adapter.py   — full NPC controller implementation
```

---

## 15. Phase 2 Success Criteria

1. Itera generates at least 3 distinct hypothesis types in a 
   200-cycle run
2. Memory graph has edges after a run — experiences link to 
   hypotheses, discoveries link to causes
3. Empathy layer builds entity models — after 50 encounters 
   with a creature, Itera has a meaningful threat/curiosity/
   relationship model of it
4. A game world adapter can pass health and quest data through 
   self_state and world_context without any core changes
5. DWE adapter connects to Construct 3 bridge and Itera 
   successfully controls an NPC for a full session

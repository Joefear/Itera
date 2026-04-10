"""Itera's persistent self.

Identity is the durable cognitive core that persists across sessions and
worlds. It owns memory, drives, and the hypothesis engine, but never talks to
the world directly. Simulation loops read from and write to this module using
plain dictionaries at the boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

try:
    from core.drives import DriveHierarchy
    from core.memory import MemoryGraph
    from core.hypothesis import HypothesisEngine
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    from drives import DriveHierarchy
    from memory import MemoryGraph
    from hypothesis import HypothesisEngine

IDENTITY_VERSION = "0.1.0"
DEFAULT_IDENTITY_NAME = "Itera"
DEFAULT_DATA_DIR = "data/identity"
IDENTITY_META_FILENAME = "identity_meta.json"
DRIVES_FILENAME = "drives.json"
MEMORY_FILENAME = "memory.json"
HYPOTHESIS_FILENAME = "hypothesis.json"
DEVELOPMENTAL_FILENAME = "developmental.json"

PRIMITIVE_STAGE_MAX = 0.15
ENVIRONMENTAL_STAGE_MAX = 0.30
BIOLOGICAL_STAGE_MAX = 0.50
ECOLOGICAL_STAGE_MAX = 0.70
SOCIAL_STAGE_MAX = 0.85
CIVILIZATIONAL_STAGE_MAX = 1.0

STAGE_ADVANCE_PER_CONFIRMATION = 0.001
STAGE_ADVANCE_PER_CYCLE = 0.0001
MEMORY_NOVELTY_THRESHOLD = 0.28
DOMINANT_DRIVE_HISTORY_LIMIT = 12
REFLECTION_MEMORY_LIMIT = 3
MATCH_TOLERANCE = 0.05
MIN_STAGE = 0.0
MAX_STAGE = 1.0
MIN_VALENCE = -1.0
MAX_VALENCE = 1.0
POSITIVE_VALENCE_DISCOVERY_THRESHOLD = 0.35
POSITIVE_VALENCE_MASTERY_THRESHOLD = 0.3
DRIVE_COMPETENCE_BASE = 0.25
DRIVE_DISCOVERY_BASE = 0.35
DRIVE_STABILITY_POSITIVE_BASE = 0.4
DRIVE_STABILITY_POSITIVE_SCALE = 0.6
DRIVE_THREAT_BASE = 0.3
DRIVE_SCARCITY_BASE = 0.2
DRIVE_UNKNOWN_BASE = 0.25
DRIVE_STABILITY_NEGATIVE_BASE = 0.5
DRIVE_STABILITY_NEGATIVE_SCALE = 0.5
DISCOVERY_MEMORY_VALENCE = 0.25
SOCIAL_RELATIONSHIP_DEPTH_PER_CREATURE = 0.47
SOCIAL_ENTITY_PREDICTION_GAP_SIGNAL = 0.72
MASTERY_COMPETENCE_PER_CONFIRMATION = 0.15
MASTERY_DISCOVERY_PULL_SIGNAL = 0.4


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


def _normalize_discovery_name(name: str) -> str:
    """Normalize discovery names for stable duplicate detection."""

    return str(name).strip()


def _values_match(expected: Any, observed: Any) -> bool:
    """Return whether an observed value matches an expected prediction."""

    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        return abs(float(expected) - float(observed)) <= MATCH_TOLERANCE
    return expected == observed


@dataclass
class DevelopmentalState:
    """Itera's current position in its growth arc."""

    stage: float
    phase_name: str
    total_cycles: int
    total_hypotheses: int
    total_confirmed: int
    discoveries: list[str]
    dominant_drive_history: list[str]

    def __post_init__(self) -> None:
        """Normalize counters, stage, and stored history."""

        self.stage = _clamp(self.stage, MIN_STAGE, MAX_STAGE)
        self.total_cycles = max(0, int(self.total_cycles))
        self.total_hypotheses = max(0, int(self.total_hypotheses))
        self.total_confirmed = max(0, int(self.total_confirmed))
        self.discoveries = [_normalize_discovery_name(name) for name in self.discoveries if _normalize_discovery_name(name)]
        self.dominant_drive_history = [str(name) for name in self.dominant_drive_history[-DOMINANT_DRIVE_HISTORY_LIMIT:]]
        self.phase_name = self.phase_from_stage(self.stage)

    def advance(self, confirmed_delta: int, cycle_delta: int) -> None:
        """Update counts and recalculate stage and phase name."""

        confirmed_steps = max(0, int(confirmed_delta))
        cycle_steps = max(0, int(cycle_delta))
        self.total_confirmed += confirmed_steps
        self.total_cycles += cycle_steps
        self.stage = _clamp(
            self.stage
            + (confirmed_steps * STAGE_ADVANCE_PER_CONFIRMATION)
            + (cycle_steps * STAGE_ADVANCE_PER_CYCLE),
            MIN_STAGE,
            MAX_STAGE,
        )
        self.phase_name = self.phase_from_stage(self.stage)

    def phase_from_stage(self, stage: float) -> str:
        """Map a developmental stage float to its named phase."""

        normalized_stage = _clamp(stage, MIN_STAGE, MAX_STAGE)
        if normalized_stage < PRIMITIVE_STAGE_MAX:
            return "primitive"
        if normalized_stage < ENVIRONMENTAL_STAGE_MAX:
            return "environmental"
        if normalized_stage < BIOLOGICAL_STAGE_MAX:
            return "biological"
        if normalized_stage < ECOLOGICAL_STAGE_MAX:
            return "ecological"
        if normalized_stage < SOCIAL_STAGE_MAX:
            return "social"
        return "civilizational"

    def record_dominant_drive(self, drive_name: str) -> None:
        """Append a dominant drive observation to rolling history."""

        self.dominant_drive_history.append(str(drive_name))
        if len(self.dominant_drive_history) > DOMINANT_DRIVE_HISTORY_LIMIT:
            self.dominant_drive_history = self.dominant_drive_history[-DOMINANT_DRIVE_HISTORY_LIMIT:]


@dataclass
class _OutcomeEvaluation:
    """Minimal result object for hypothesis evaluation without cross-importing types."""

    hypothesis_id: str
    timestamp: float
    actual_outcome: dict[str, Any]
    matched_prediction: bool
    confidence_delta: float
    valence: float


class Identity:
    """Itera's persistent self."""

    def __init__(self, name: str = DEFAULT_IDENTITY_NAME, data_dir: str = DEFAULT_DATA_DIR) -> None:
        """Initialize an identity shell with owned cognitive subsystems."""

        self.name = str(name)
        self.data_dir = Path(data_dir)
        self.drives = DriveHierarchy(developmental_stage=MIN_STAGE)
        self.memory = MemoryGraph(storage_path=self.data_dir / MEMORY_FILENAME)
        self.hypothesis_engine = HypothesisEngine(drives=self.drives)
        self.developmental = DevelopmentalState(
            stage=MIN_STAGE,
            phase_name="primitive",
            total_cycles=0,
            total_hypotheses=0,
            total_confirmed=0,
            discoveries=[],
            dominant_drive_history=[],
        )
        self.created_at = time.time()
        self.last_wake_at: float | None = None
        self.last_sleep_at: float | None = None
        self._awake = False
        self._last_decision: dict[str, Any] | None = None
        self._known_confirmed_ids: set[str] = set()
        self._last_social_updates: dict[str, float] = {}

    def wake(self) -> dict[str, Any]:
        """Load persisted state if present and return a session summary."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._component_path(IDENTITY_META_FILENAME)
        developmental_path = self._component_path(DEVELOPMENTAL_FILENAME)
        drives_path = self._component_path(DRIVES_FILENAME)
        memory_path = self._component_path(MEMORY_FILENAME)
        hypothesis_path = self._component_path(HYPOTHESIS_FILENAME)

        if all(path.exists() for path in (meta_path, developmental_path, drives_path, memory_path, hypothesis_path)):
            meta = self._read_json(meta_path)
            self.name = str(meta.get("name", self.name))
            self.created_at = float(meta.get("created_at", self.created_at))
            self.last_sleep_at = meta.get("last_sleep_at")
            if self.last_sleep_at is not None:
                self.last_sleep_at = float(self.last_sleep_at)

            self.developmental = DevelopmentalState(**self._read_json(developmental_path))
            self.drives = DriveHierarchy.from_dict(self._read_json(drives_path))
            self.memory = MemoryGraph.from_dict(self._read_json(memory_path), storage_path=memory_path)
            self.hypothesis_engine = HypothesisEngine.from_dict(self._read_json(hypothesis_path), drives=self.drives)
        else:
            self.drives = DriveHierarchy(developmental_stage=MIN_STAGE)
            self.memory = MemoryGraph(storage_path=memory_path)
            self.hypothesis_engine = HypothesisEngine(drives=self.drives)
            self.developmental = DevelopmentalState(
                stage=MIN_STAGE,
                phase_name="primitive",
                total_cycles=0,
                total_hypotheses=0,
                total_confirmed=0,
                discoveries=[],
                dominant_drive_history=[],
            )

        self.drives.developmental_stage = self.developmental.stage
        self.drives = DriveHierarchy.from_dict(self.drives.to_dict())
        self.hypothesis_engine.drives = self.drives
        self._known_confirmed_ids = {hypothesis.id for hypothesis in self.hypothesis_engine.get_confirmed()}
        self.last_wake_at = time.time()
        self._awake = True
        return {
            "name": self.name,
            "stage": round(self.developmental.stage, 4),
            "phase_name": self.developmental.phase_name,
            "dominant_drive": self.current_drive(),
            "memory_nodes": self.memory.summary()["node_count"],
            "active_hypotheses": len(self.hypothesis_engine.get_active()),
            "total_cycles": self.developmental.total_cycles,
        }

    def sleep(self) -> None:
        """Persist the full identity state to disk."""

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.last_sleep_at = time.time()
        self._write_json(
            self._component_path(IDENTITY_META_FILENAME),
            {
                "name": self.name,
                "version": IDENTITY_VERSION,
                "created_at": self.created_at,
                "last_wake_at": self.last_wake_at,
                "last_sleep_at": self.last_sleep_at,
            },
        )
        self._write_json(self._component_path(DRIVES_FILENAME), self.drives.to_dict())
        self._write_json(self._component_path(MEMORY_FILENAME), self.memory.to_dict())
        self._write_json(self._component_path(HYPOTHESIS_FILENAME), self.hypothesis_engine.to_dict())
        self._write_json(self._component_path(DEVELOPMENTAL_FILENAME), asdict(self.developmental))

    def perceive(self, observation: dict[str, Any]) -> None:
        """Process a world observation through drives, hypotheses, and memory."""

        social_updates = self._social_drive_updates_from_observation(observation)
        self._last_social_updates = dict(social_updates)
        observation_for_hypothesis = dict(observation)
        if "novelty_hint" in observation:
            observation_for_hypothesis["novelty_hint"] = observation["novelty_hint"]
        stored_observation = self.hypothesis_engine.observe(observation_for_hypothesis)
        if social_updates:
            self.drives.update(social_updates)
        generated = self.hypothesis_engine.generate_hypothesis(stored_observation)
        dominant_drive = self.current_drive()
        self.developmental.record_dominant_drive(dominant_drive)

        if generated is not None:
            self.developmental.total_hypotheses += 1

        if stored_observation.novelty_score < MEMORY_NOVELTY_THRESHOLD:
            return

        tags = [dominant_drive.lower(), "observation", stored_observation.drive_context.get("dominant_drive", dominant_drive).lower()]
        tags.extend(str(key).lower() for key in stored_observation.data.keys())
        if generated is not None:
            tags.append(generated.id)
            tags.append("hypothesis")

        self.memory.create_node(
            node_type="experience",
            content={
                "observation": dict(stored_observation.data),
                "novelty_score": stored_observation.novelty_score,
                "drive_context": dict(stored_observation.drive_context),
                "generated_hypothesis_id": None if generated is None else generated.id,
            },
            valence=0.0,
            tags=tags,
        )

    def decide(self) -> dict[str, Any]:
        """Produce the next plain-dict action for the simulation loop."""

        dominant_drive = self.current_drive()
        next_hypothesis = self.hypothesis_engine.select_next()
        if next_hypothesis is not None:
            design = self.hypothesis_engine.design_test(next_hypothesis)
            decision = {
                "action": str(design["suggested_action"]["type"]),
                "parameters": {
                    "focus": list(design["suggested_action"]["focus"]),
                    "hypothesis_id": design["hypothesis_id"],
                    "expected_outcome": dict(design["expected_outcome"]),
                },
                "drive_source": str(design["drive_source"]),
                "rationale": str(design["intent"]),
            }
        else:
            decision = {
                "action": "observe_environment",
                "parameters": {"focus": dominant_drive.lower()},
                "drive_source": dominant_drive,
                "rationale": f"No pending hypothesis; continue exploration under {dominant_drive}.",
            }

        self._last_decision = decision
        return decision

    def absorb(self, outcome: dict[str, Any], valence: float) -> None:
        """Process an action outcome into memory, hypotheses, drives, and growth."""

        outcome_payload = dict(outcome)
        clamped_valence = _clamp(valence, MIN_VALENCE, MAX_VALENCE)
        dominant_before = self.current_drive()

        tags = [dominant_before.lower(), "outcome"]
        if "hypothesis_id" in outcome_payload:
            tags.append(str(outcome_payload["hypothesis_id"]))
        memory_node = self.memory.absorb_outcome(outcome_payload, clamped_valence, tags=tags)

        confirmed_before = {hypothesis.id for hypothesis in self.hypothesis_engine.get_confirmed()}
        hypothesis = self._resolve_outcome_hypothesis(outcome_payload)
        if hypothesis is not None:
            evaluation = _OutcomeEvaluation(
                hypothesis_id=hypothesis.id,
                timestamp=float(outcome_payload.get("timestamp", time.time())),
                actual_outcome=self._outcome_state(outcome_payload),
                matched_prediction=self._outcome_matches_hypothesis(hypothesis.predicted_outcome, outcome_payload),
                confidence_delta=float(outcome_payload.get("confidence_delta", 0.0)),
                valence=clamped_valence,
            )
            self.hypothesis_engine.evaluate(hypothesis, evaluation)

        confirmed_after = {hypothesis.id for hypothesis in self.hypothesis_engine.get_confirmed()}
        confirmed_delta = len(confirmed_after - confirmed_before)
        drive_updates = self._drive_updates_from_valence(clamped_valence)
        if confirmed_delta > 0 or clamped_valence >= POSITIVE_VALENCE_MASTERY_THRESHOLD:
            mastery_updates = self._mastery_drive_updates(self.developmental.total_confirmed + confirmed_delta)
            for signal_name, signal_value in mastery_updates.items():
                drive_updates[signal_name] = max(drive_updates.get(signal_name, 0.0), signal_value)
        for signal_name, signal_value in self._last_social_updates.items():
            drive_updates[signal_name] = max(drive_updates.get(signal_name, 0.0), signal_value)
        self.drives.update(drive_updates)
        dominant_after = self.current_drive()
        self.developmental.record_dominant_drive(dominant_after)

        self.developmental.advance(confirmed_delta=confirmed_delta, cycle_delta=1)
        self.drives.advance_developmental_stage(self.developmental.stage - self.drives.developmental_stage)
        self._known_confirmed_ids = confirmed_after

        if clamped_valence >= POSITIVE_VALENCE_DISCOVERY_THRESHOLD and "discovery" in memory_node.tags:
            self.add_discovery(str(outcome_payload.get("discovery", "unnamed discovery")), json.dumps(outcome_payload, sort_keys=True))

    def reflect(self) -> str:
        """Generate a brief internal self-reflection summary."""

        dominant_drive, dominant_weight = self.drives.get_dominant_drive()
        memories = self.memory.retrieve(query="", tags=[dominant_drive.lower()], limit=REFLECTION_MEMORY_LIMIT)
        if not memories:
            memories = self.memory.retrieve(limit=REFLECTION_MEMORY_LIMIT)
        memory_ids = ", ".join(node.id for node in memories) if memories else "none"
        active_hypotheses = len(self.hypothesis_engine.get_active())
        confirmed = len(self.hypothesis_engine.get_confirmed())
        return (
            f"{self.name} is in the {self.developmental.phase_name} phase at stage {self.developmental.stage:.3f}. "
            f"Dominant drive: {dominant_drive} ({dominant_weight:.3f}). "
            f"Recent salient memories: {memory_ids}. "
            f"Active hypotheses: {active_hypotheses}; confirmed: {confirmed}. "
            f"Discoveries carried forward: {len(self.developmental.discoveries)}."
        )

    def add_discovery(self, name: str, description: str) -> None:
        """Record a named discovery in developmental state and memory."""

        normalized_name = _normalize_discovery_name(name)
        if not normalized_name:
            return
        if normalized_name not in self.developmental.discoveries:
            self.developmental.discoveries.append(normalized_name)
        self.memory.create_node(
            node_type="discovery",
            content={"name": normalized_name, "description": str(description)},
            valence=DISCOVERY_MEMORY_VALENCE,
            tags=["discovery", normalized_name.lower()],
        )

    def current_drive(self) -> str:
        """Return the name of the currently dominant drive tier."""

        drive_name, _ = self.drives.get_dominant_drive()
        return drive_name

    def developmental_stage(self) -> float:
        """Return the current developmental stage."""

        return self.developmental.stage

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full identity state."""

        return {
            "meta": {
                "name": self.name,
                "version": IDENTITY_VERSION,
                "data_dir": str(self.data_dir),
                "created_at": self.created_at,
                "last_wake_at": self.last_wake_at,
                "last_sleep_at": self.last_sleep_at,
            },
            "drives": self.drives.to_dict(),
            "memory": self.memory.to_dict(),
            "hypothesis": self.hypothesis_engine.to_dict(),
            "developmental": asdict(self.developmental),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Identity":
        """Deserialize an identity from saved state."""

        meta = dict(data.get("meta", {}))
        identity = cls(
            name=str(meta.get("name", DEFAULT_IDENTITY_NAME)),
            data_dir=str(meta.get("data_dir", DEFAULT_DATA_DIR)),
        )
        identity.created_at = float(meta.get("created_at", identity.created_at))
        identity.last_wake_at = meta.get("last_wake_at")
        if identity.last_wake_at is not None:
            identity.last_wake_at = float(identity.last_wake_at)
        identity.last_sleep_at = meta.get("last_sleep_at")
        if identity.last_sleep_at is not None:
            identity.last_sleep_at = float(identity.last_sleep_at)
        identity.developmental = DevelopmentalState(**dict(data.get("developmental", {})))
        identity.drives = DriveHierarchy.from_dict(dict(data.get("drives", {})))
        identity.memory = MemoryGraph.from_dict(dict(data.get("memory", {})), storage_path=identity.data_dir / MEMORY_FILENAME)
        identity.hypothesis_engine = HypothesisEngine.from_dict(dict(data.get("hypothesis", {})), drives=identity.drives)
        identity._known_confirmed_ids = {hypothesis.id for hypothesis in identity.hypothesis_engine.get_confirmed()}
        return identity

    def summary(self) -> str:
        """Return a human-readable snapshot of the current identity state."""

        memory_summary = self.memory.summary()
        return "\n".join(
            [
                f"Identity: {self.name}",
                f"Stage: {self.developmental.stage:.3f} ({self.developmental.phase_name})",
                f"Dominant drive: {self.current_drive()}",
                f"Cycles: {self.developmental.total_cycles}",
                f"Hypotheses: total={self.developmental.total_hypotheses} confirmed={self.developmental.total_confirmed}",
                f"Memory graph: nodes={memory_summary['node_count']} edges={memory_summary['edge_count']}",
                f"Discoveries: {len(self.developmental.discoveries)}",
            ]
        )

    def _component_path(self, filename: str) -> Path:
        """Return the absolute path for a persisted component file."""

        return self.data_dir / filename

    def _read_json(self, path: Path) -> dict[str, Any]:
        """Read a JSON file into a dictionary."""

        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write a dictionary payload to a JSON file."""

        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _resolve_outcome_hypothesis(self, outcome: dict[str, Any]) -> Any | None:
        """Find the hypothesis associated with an outcome if one is identifiable."""

        hypothesis_id = outcome.get("hypothesis_id")
        if hypothesis_id is not None:
            return self.hypothesis_engine.hypotheses.get(str(hypothesis_id))

        testing = [hypothesis for hypothesis in self.hypothesis_engine.hypotheses.values() if hypothesis.status == "testing"]
        if len(testing) == 1:
            return testing[0]
        return None

    def _outcome_state(self, outcome: dict[str, Any]) -> dict[str, Any]:
        """Extract the world-state-like payload from an outcome dictionary."""

        if isinstance(outcome.get("resulting_state"), dict):
            return dict(outcome["resulting_state"])
        if isinstance(outcome.get("actual_outcome"), dict):
            return dict(outcome["actual_outcome"])
        return dict(outcome)

    def _outcome_matches_hypothesis(self, predicted_outcome: dict[str, Any], outcome: dict[str, Any]) -> bool:
        """Return whether an outcome appears to match a hypothesis prediction."""

        actual_state = self._outcome_state(outcome)
        if not predicted_outcome:
            return bool(outcome.get("success", True))

        matches = 0
        comparisons = 0
        for key, expected_value in predicted_outcome.items():
            if key not in actual_state:
                continue
            comparisons += 1
            if _values_match(expected_value, actual_state[key]):
                matches += 1

        if comparisons == 0:
            return bool(outcome.get("success", True))
        return matches == comparisons

    def _drive_updates_from_valence(self, valence: float) -> dict[str, float]:
        """Translate raw outcome valence into drive-signal adjustments."""

        magnitude = abs(_clamp(valence, MIN_VALENCE, MAX_VALENCE))
        if valence >= 0.0:
            return {
                "competence_growth": min(1.0, DRIVE_COMPETENCE_BASE + magnitude),
                "discovery_pull": min(1.0, DRIVE_DISCOVERY_BASE + magnitude),
                "environmental_stability": min(1.0, DRIVE_STABILITY_POSITIVE_BASE + (magnitude * DRIVE_STABILITY_POSITIVE_SCALE)),
            }

        return {
            "threat_level": min(1.0, DRIVE_THREAT_BASE + magnitude),
            "resource_scarcity": min(1.0, DRIVE_SCARCITY_BASE + magnitude),
            "unknown_territory": min(1.0, DRIVE_UNKNOWN_BASE + magnitude),
            "environmental_stability": max(0.0, DRIVE_STABILITY_NEGATIVE_BASE - (magnitude * DRIVE_STABILITY_NEGATIVE_SCALE)),
        }

    def _social_drive_updates_from_observation(self, observation: dict[str, Any]) -> dict[str, float]:
        """Extract explicit social-drive updates from visible creature entities."""

        creatures = self._creature_entities(observation)
        creature_count = len(creatures)
        if creature_count <= 0:
            return {}
        average_reactivity = sum(
            float(creature.get("properties", {}).get("reactivity", SOCIAL_ENTITY_PREDICTION_GAP_SIGNAL))
            for creature in creatures
        ) / float(creature_count)
        return {
            "relationship_depth": min(1.0, creature_count * SOCIAL_RELATIONSHIP_DEPTH_PER_CREATURE),
            "entity_prediction_gap": max(SOCIAL_ENTITY_PREDICTION_GAP_SIGNAL, average_reactivity),
        }

    def _creature_entities(self, observation: dict[str, Any]) -> list[dict[str, Any]]:
        """Return creature entities from an observation payload."""

        entities = observation.get("entities", [])
        if not isinstance(entities, list):
            return []
        return [
            entity
            for entity in entities
            if isinstance(entity, dict) and "creature" in entity.get("tags", [])
        ]

    def _creature_observation_features(self, observation: dict[str, Any]) -> dict[str, float]:
        """Derive dynamic creature features that keep observation novelty alive."""

        creatures = self._creature_entities(observation)
        creature_count = len(creatures)
        if creature_count <= 0:
            return {}

        properties = [dict(creature.get("properties", {})) for creature in creatures]
        positions = [
            creature.get("position", (0, 0))
            for creature in creatures
            if isinstance(creature.get("position"), tuple) and len(creature.get("position")) == 2
        ]
        mean_x = 0.0
        mean_y = 0.0
        if positions:
            mean_x = sum(float(position[0]) for position in positions) / float(len(positions))
            mean_y = sum(float(position[1]) for position in positions) / float(len(positions))

        return {
            "visible_creature_count": float(creature_count),
            "creature_mean_x": mean_x,
            "creature_mean_y": mean_y,
            "creature_reactivity_average": sum(
                float(item.get("reactivity", SOCIAL_ENTITY_PREDICTION_GAP_SIGNAL))
                for item in properties
            ) / float(len(properties)),
            "creature_temperature_average": sum(
                float(item.get("temperature", 0.0))
                for item in properties
            ) / float(len(properties)),
            "creature_brightness_average": sum(
                float(item.get("brightness", 0.0))
                for item in properties
            ) / float(len(properties)),
        }

    def _mastery_drive_updates(self, confirmed_count: int) -> dict[str, float]:
        """Return mastery-signal updates from accumulated confirmed hypotheses."""

        return {
            "competence_growth": min(1.0, max(0, int(confirmed_count)) * MASTERY_COMPETENCE_PER_CONFIRMATION),
            "discovery_pull": MASTERY_DISCOVERY_PULL_SIGNAL,
        }


if __name__ == "__main__":
    demo_data_dir = Path(DEFAULT_DATA_DIR) / f"demo_{int(time.time())}"
    identity = Identity(name=DEFAULT_IDENTITY_NAME, data_dir=str(demo_data_dir))
    wake_summary = identity.wake()
    print("Wake summary:", wake_summary)

    observations = [
        {
            "unknown_territory": 0.82,
            "environmental_stability": 0.34,
            "distant_signal": "flickering",
            "distance_to_signal": 7,
        },
        {
            "unknown_territory": 0.94,
            "environmental_stability": 0.18,
            "distant_signal": "erratic",
            "distance_to_signal": 3,
            "unmapped_structure": True,
        },
        {
            "entity_prediction_gap": 0.62,
            "relationship_depth": 0.28,
            "competence_growth": 0.55,
            "discovery_pull": 0.73,
            "anomalous_pattern": "thermal pulse",
        },
    ]
    for observation in observations:
        identity.perceive(observation)

    decision = identity.decide()
    print("Decision:", decision)

    mock_outcome = {
        "action": decision["action"],
        "hypothesis_id": decision["parameters"].get("hypothesis_id"),
        "resulting_state": {
            "distance_to_signal": 2,
            "distant_signal": "erratic",
            "unknown_territory": 0.9,
        },
        "success": True,
    }
    identity.absorb(mock_outcome, valence=0.35)
    stage_before_sleep = identity.developmental_stage()
    print("Reflection:", identity.reflect())
    print("Summary before sleep:")
    print(identity.summary())
    identity.sleep()

    second_identity = Identity(name=DEFAULT_IDENTITY_NAME, data_dir=str(demo_data_dir))
    second_wake_summary = second_identity.wake()
    stage_after_wake = second_identity.developmental_stage()
    print("Second wake summary:", second_wake_summary)
    print(f"Stage before sleep: {stage_before_sleep:.4f}")
    print(f"Stage after second wake: {stage_after_wake:.4f}")

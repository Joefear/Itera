"""Itera's entity modeling and empathy layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Any

if __package__ in {None, ""}:  # pragma: no cover - convenience for direct execution
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.memory import MemoryGraph

MIN_SIGNAL_VALUE = 0.0
MAX_SIGNAL_VALUE = 1.0
MIN_RELATIONSHIP_VALENCE = -1.0
MAX_RELATIONSHIP_VALENCE = 1.0
MAX_INTERACTION_HISTORY = 10
UNKNOWN_THREAT_DEFAULT = 0.5
UNKNOWN_CURIOSITY_DEFAULT = 0.7
UNKNOWN_ENTITY_CURIOSITY = 0.5
WELFARE_STAGE_THRESHOLD = 0.6
PREDICTION_STAGE_THRESHOLD = 0.3
RELATIONSHIP_STAGE_THRESHOLD = 0.3
CURIOSITY_DECAY_RATE = 0.05
THREAT_LEARNING_RATE = 0.1
CURIOSITY_PREDICTION_GAP_SCALE = 2.0
DEFAULT_ENTITY_TYPE = "entity"
ENTITY_MEMORY_NODE_TYPE = "entity"
ENTITY_MEMORY_TAG = "entity"
DEFAULT_PREDICTED_RESPONSE = "uncertain"
DEFAULT_PREDICTIVE_CONFIDENCE = 0.0
DEFAULT_WELFARE_CONCERN = 0.0
DEFAULT_RELATIONSHIP_VALENCE = 0.0
DEFAULT_OBSERVATION_COUNT = 0
PROPERTY_REACTIVITY_KEY = "reactivity"
PROPERTY_AGGRESSION_KEY = "aggression"
PROPERTY_THREAT_KEY = "threat"
PROPERTY_MASS_KEY = "mass"
PROPERTY_RESOURCE_DENSITY_KEY = "resource_density"
REACTIVITY_THREAT_WEIGHT = 0.5
AGGRESSION_THREAT_WEIGHT = 0.3
MASS_THREAT_WEIGHT = 0.2
POSITIVE_VALENCE_THREAT_REDUCTION = 0.05
NEGATIVE_VALENCE_THREAT_INCREASE = 0.25
UNEXPECTED_CURIOSITY_BONUS = 0.15
UNEXPECTED_THREAT_BONUS = 0.05
RELATIONSHIP_OBSERVATION_NORMALIZER = 12.0
WELFARE_OBSERVATION_NORMALIZER = 15.0
INTERACTION_SUCCESS_DEFAULT = False
PREDICTION_POSITIVE_RESPONSE_THRESHOLD = 0.6
PREDICTION_NEGATIVE_RESPONSE_THRESHOLD = 0.4
PREDICTION_CONFIDENCE_CENTER = 0.5
PREDICTION_CONFIDENCE_SCALE = 2.0
RELATIONSHIP_FAMILIARITY_WEIGHT = 0.45
RELATIONSHIP_VALENCE_WEIGHT = 0.55
LOW_STAGE_RELATIONSHIP_SCALE = 0.25
HIGH_STAGE_WELFARE_SCALE = 0.9
INTERACTION_RELATIONSHIP_MEMORY_BLEND = 0.5
ENTITY_NODE_RELEVANCE = 1.0
KNOWN_ENTITY_DEFAULT_LIMIT = 3
MODELED_DRIVE_AVOIDANCE_WEIGHT = 0.6
MODELED_DRIVE_ENGAGEMENT_WEIGHT = 0.7
MODELED_DRIVE_CURIOSITY_WEIGHT = 0.8
MODELED_DRIVE_STABILITY_WEIGHT = 0.5
ENTITY_TYPE_PRIORITY: tuple[str, ...] = ("monster", "player", "npc", "creature")
ENTITY_TYPE_THREAT_BIAS: dict[str, float] = {
    "monster": 0.2,
    "player": 0.08,
    "npc": 0.02,
    "creature": 0.05,
}
ENTITY_SNAPSHOT_KEYS: tuple[str, ...] = ("name", "position", "interactable", "tags", "properties")
SOCIAL_SIGNAL_RELATIONSHIP_DEPTH = "relationship_depth"
SOCIAL_SIGNAL_ENTITY_PREDICTION_GAP = "entity_prediction_gap"
SOCIAL_SIGNAL_NAMES: tuple[str, str] = (
    SOCIAL_SIGNAL_RELATIONSHIP_DEPTH,
    SOCIAL_SIGNAL_ENTITY_PREDICTION_GAP,
)
INTERACTION_HISTORY_ACTION_KEY = "action"
INTERACTION_HISTORY_SUCCESS_KEY = "success"
INTERACTION_HISTORY_TIMESTAMP_KEY = "timestamp"
INTERACTION_HISTORY_VALENCE_KEY = "valence"
INTERACTION_HISTORY_UNEXPECTED_KEY = "unexpected"
DEMO_PREDICTION_STAGE = 0.35
DEMO_WELFARE_STAGE = 0.7
DEMO_RIVER_REACTIVITY = 0.85
DEMO_RIVER_MASS = 0.45
DEMO_KEEPER_REACTIVITY = 0.35
DEMO_KEEPER_RESOURCE_DENSITY = 0.4
DEMO_HUNTER_REACTIVITY = 0.9
DEMO_HUNTER_AGGRESSION = 0.85
DEMO_HUNTER_MASS = 0.8
DEMO_NEGATIVE_VALENCE_STRONG = -0.4
DEMO_NEGATIVE_VALENCE_MILD = -0.2
DEMO_POSITIVE_VALENCE = 0.3
DEMO_RELATIONSHIP_POSITIVE_SMALL = 0.3
DEMO_RELATIONSHIP_POSITIVE_LARGE = 0.5
DEMO_RELATIONSHIP_NEGATIVE = -0.6


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float when possible, otherwise return a default."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _normalize_identifier(value: Any, default: str) -> str:
    """Return a stable lowercase identifier."""

    cleaned = str(value).strip().lower()
    if cleaned:
        return cleaned
    return str(default).strip().lower()


@dataclass
class EntityModel:
    """Itera's internal model of another entity."""

    entity_id: str
    entity_type: str
    first_seen: float
    last_seen: float
    observation_count: int
    predicted_reactions: dict[str, float]
    threat_assessment: float
    curiosity_value: float
    relationship_valence: float
    interaction_history: list[dict[str, Any]]
    modeled_drive_state: dict[str, float]
    welfare_concern: float

    def __post_init__(self) -> None:
        """Normalize all modeled values into stable bounds."""

        self.entity_id = _normalize_identifier(self.entity_id, DEFAULT_ENTITY_TYPE)
        self.entity_type = _normalize_identifier(self.entity_type, DEFAULT_ENTITY_TYPE)
        self.first_seen = float(self.first_seen)
        self.last_seen = float(self.last_seen)
        self.observation_count = max(0, int(self.observation_count))
        self.predicted_reactions = {
            str(action): _clamp(confidence, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
            for action, confidence in dict(self.predicted_reactions).items()
        }
        self.threat_assessment = _clamp(self.threat_assessment, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
        self.curiosity_value = _clamp(self.curiosity_value, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
        self.relationship_valence = _clamp(
            self.relationship_valence,
            MIN_RELATIONSHIP_VALENCE,
            MAX_RELATIONSHIP_VALENCE,
        )
        self.interaction_history = list(self.interaction_history[-MAX_INTERACTION_HISTORY:])
        self.modeled_drive_state = {
            str(name): _clamp(value, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
            for name, value in dict(self.modeled_drive_state).items()
        }
        self.welfare_concern = _clamp(self.welfare_concern, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)


class EmpathyLayer:
    """Model other entities through threat, curiosity, prediction, and care.

    The layer starts with threat and curiosity estimates, adds behavioral
    prediction and relationship tracking after stage 0.3, and unlocks modeled
    drive state plus welfare concern after stage 0.6.
    """

    def __init__(self, memory: MemoryGraph, developmental_stage: float = 0.0) -> None:
        """Initialize the empathy layer with shared memory and stage gating."""

        self.memory = memory
        self.developmental_stage = _clamp(developmental_stage, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
        self.entities: dict[str, EntityModel] = {}
        self._entity_memory_node_ids: dict[str, str] = {}
        self._index_entity_memory_nodes()

    def observe_entity(
        self,
        entity: dict[str, Any],
        itera_action: str | None = None,
        outcome: dict[str, Any] | None = None,
        developmental_stage: float = 0.0,
    ) -> EntityModel:
        """Create or update an entity model from a fresh observation."""

        self.update_developmental_stage(developmental_stage)
        normalized_entity = dict(entity)
        entity_id = self._entity_id_from_entity(normalized_entity)
        entity_type = self._entity_type_from_entity(normalized_entity)
        timestamp = time.time()

        model = self.entities.get(entity_id)
        if model is None:
            model = EntityModel(
                entity_id=entity_id,
                entity_type=entity_type,
                first_seen=timestamp,
                last_seen=timestamp,
                observation_count=DEFAULT_OBSERVATION_COUNT,
                predicted_reactions={},
                threat_assessment=UNKNOWN_THREAT_DEFAULT,
                curiosity_value=UNKNOWN_CURIOSITY_DEFAULT,
                relationship_valence=DEFAULT_RELATIONSHIP_VALENCE,
                interaction_history=[],
                modeled_drive_state={},
                welfare_concern=DEFAULT_WELFARE_CONCERN,
            )
            self.entities[entity_id] = model

        model.entity_type = entity_type
        model.last_seen = timestamp
        model.observation_count += 1

        unexpected_interaction = False
        if itera_action is not None and outcome is not None:
            unexpected_interaction = self._record_interaction(model, str(itera_action), outcome)

        model.threat_assessment = self._updated_threat(model, normalized_entity, outcome, unexpected_interaction)
        model.curiosity_value = self._updated_curiosity(model, unexpected_interaction)

        if self.developmental_stage >= PREDICTION_STAGE_THRESHOLD:
            self._rebuild_predicted_reactions(model)
        else:
            model.predicted_reactions = {}

        if self.developmental_stage > WELFARE_STAGE_THRESHOLD:
            model.modeled_drive_state = self._modeled_drive_state(normalized_entity, model)
            model.welfare_concern = self._calculate_welfare_concern(model)
        else:
            model.modeled_drive_state = {}
            model.welfare_concern = DEFAULT_WELFARE_CONCERN

        self._upsert_entity_memory_node(model, normalized_entity)
        return model

    def get_threat_level(self, entity_id: str) -> float:
        """Return threat_assessment for entity. 0.0 if unknown."""

        model = self.entities.get(_normalize_identifier(entity_id, ""))
        if model is None:
            return MIN_SIGNAL_VALUE
        return model.threat_assessment

    def get_curiosity_value(self, entity_id: str) -> float:
        """Return curiosity_value for entity. 0.5 if unknown."""

        model = self.entities.get(_normalize_identifier(entity_id, ""))
        if model is None:
            return UNKNOWN_ENTITY_CURIOSITY
        return model.curiosity_value

    def get_welfare_concern(self, entity_id: str) -> float:
        """Return welfare_concern. 0.0 if unknown or below the stage threshold."""

        if self.developmental_stage <= WELFARE_STAGE_THRESHOLD:
            return DEFAULT_WELFARE_CONCERN
        model = self.entities.get(_normalize_identifier(entity_id, ""))
        if model is None:
            return DEFAULT_WELFARE_CONCERN
        return model.welfare_concern

    def predict_reaction(self, entity_id: str, action: str) -> dict[str, Any]:
        """Predict how an entity will react to an action."""

        if self.developmental_stage < PREDICTION_STAGE_THRESHOLD:
            return {}

        model = self.entities.get(_normalize_identifier(entity_id, ""))
        if model is None:
            return {}

        action_key = str(action).strip().lower()
        if action_key not in model.predicted_reactions:
            return {
                "predicted_response": DEFAULT_PREDICTED_RESPONSE,
                "confidence": DEFAULT_PREDICTIVE_CONFIDENCE,
            }

        success_rate = model.predicted_reactions[action_key]
        confidence = abs(success_rate - PREDICTION_CONFIDENCE_CENTER) * PREDICTION_CONFIDENCE_SCALE
        predicted_response = DEFAULT_PREDICTED_RESPONSE
        if success_rate >= PREDICTION_POSITIVE_RESPONSE_THRESHOLD:
            predicted_response = "likely_positive"
        elif success_rate <= PREDICTION_NEGATIVE_RESPONSE_THRESHOLD:
            predicted_response = "likely_negative"

        return {
            "predicted_response": predicted_response,
            "confidence": _clamp(confidence, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE),
        }

    def update_relationship(self, entity_id: str, valence_delta: float) -> None:
        """Update relationship_valence for an entity and refresh its memory node."""

        normalized_entity_id = _normalize_identifier(entity_id, "")
        if not normalized_entity_id:
            return

        model = self.entities.get(normalized_entity_id)
        if model is None:
            timestamp = time.time()
            model = EntityModel(
                entity_id=normalized_entity_id,
                entity_type=DEFAULT_ENTITY_TYPE,
                first_seen=timestamp,
                last_seen=timestamp,
                observation_count=DEFAULT_OBSERVATION_COUNT,
                predicted_reactions={},
                threat_assessment=UNKNOWN_THREAT_DEFAULT,
                curiosity_value=UNKNOWN_CURIOSITY_DEFAULT,
                relationship_valence=DEFAULT_RELATIONSHIP_VALENCE,
                interaction_history=[],
                modeled_drive_state={},
                welfare_concern=DEFAULT_WELFARE_CONCERN,
            )
            self.entities[normalized_entity_id] = model

        model.relationship_valence = _clamp(
            model.relationship_valence + float(valence_delta),
            MIN_RELATIONSHIP_VALENCE,
            MAX_RELATIONSHIP_VALENCE,
        )
        if self.developmental_stage > WELFARE_STAGE_THRESHOLD:
            model.welfare_concern = self._calculate_welfare_concern(model)
            model.modeled_drive_state = self._modeled_drive_state({}, model)
        self._upsert_entity_memory_node(model, {})

    def get_social_drive_signals(self) -> dict[str, float]:
        """Return social drive signals derived from known entity models."""

        if not self.entities:
            return {
                SOCIAL_SIGNAL_RELATIONSHIP_DEPTH: MIN_SIGNAL_VALUE,
                SOCIAL_SIGNAL_ENTITY_PREDICTION_GAP: MIN_SIGNAL_VALUE,
            }

        models = list(self.entities.values())
        familiarity = sum(
            min(model.observation_count / RELATIONSHIP_OBSERVATION_NORMALIZER, MAX_SIGNAL_VALUE)
            for model in models
        ) / float(len(models))
        positive_relationship = sum(max(MIN_SIGNAL_VALUE, model.relationship_valence) for model in models) / float(len(models))

        if self.developmental_stage < RELATIONSHIP_STAGE_THRESHOLD:
            relationship_depth = _clamp(
                familiarity * LOW_STAGE_RELATIONSHIP_SCALE,
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            )
        else:
            relationship_depth = _clamp(
                (familiarity * RELATIONSHIP_FAMILIARITY_WEIGHT)
                + (positive_relationship * RELATIONSHIP_VALENCE_WEIGHT),
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            )

        if self.developmental_stage < PREDICTION_STAGE_THRESHOLD:
            prediction_gap = MAX_SIGNAL_VALUE
        else:
            prediction_confidences = [self._entity_prediction_confidence(model) for model in models]
            average_confidence = sum(prediction_confidences) / float(len(prediction_confidences))
            prediction_gap = _clamp(MAX_SIGNAL_VALUE - average_confidence, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

        return {
            SOCIAL_SIGNAL_RELATIONSHIP_DEPTH: relationship_depth,
            SOCIAL_SIGNAL_ENTITY_PREDICTION_GAP: prediction_gap,
        }

    def get_most_threatening(self, limit: int = KNOWN_ENTITY_DEFAULT_LIMIT) -> list[EntityModel]:
        """Return entities with highest threat_assessment."""

        ordered = sorted(
            self.entities.values(),
            key=lambda model: (model.threat_assessment, model.observation_count, model.last_seen),
            reverse=True,
        )
        return ordered[: max(0, int(limit))]

    def get_most_curious(self, limit: int = KNOWN_ENTITY_DEFAULT_LIMIT) -> list[EntityModel]:
        """Return entities with highest curiosity_value."""

        ordered = sorted(
            self.entities.values(),
            key=lambda model: (model.curiosity_value, model.observation_count, model.last_seen),
            reverse=True,
        )
        return ordered[: max(0, int(limit))]

    def update_developmental_stage(self, stage: float) -> None:
        """Update the absolute developmental stage for gating logic."""

        self.developmental_stage = _clamp(stage, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

    def known_entity_count(self) -> int:
        """Return the number of currently modeled entities."""

        return len(self.entities)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the empathy layer for persistence."""

        return {
            "developmental_stage": self.developmental_stage,
            "entities": {
                entity_id: asdict(model)
                for entity_id, model in self.entities.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], memory: MemoryGraph) -> "EmpathyLayer":
        """Deserialize a saved empathy layer using shared memory."""

        layer = cls(memory=memory, developmental_stage=float(data.get("developmental_stage", MIN_SIGNAL_VALUE)))
        for entity_id, payload in dict(data.get("entities", {})).items():
            layer.entities[str(entity_id).strip().lower()] = EntityModel(
                entity_id=str(payload.get("entity_id", entity_id)),
                entity_type=str(payload.get("entity_type", DEFAULT_ENTITY_TYPE)),
                first_seen=float(payload.get("first_seen", time.time())),
                last_seen=float(payload.get("last_seen", time.time())),
                observation_count=int(payload.get("observation_count", DEFAULT_OBSERVATION_COUNT)),
                predicted_reactions=dict(payload.get("predicted_reactions", {})),
                threat_assessment=float(payload.get("threat_assessment", UNKNOWN_THREAT_DEFAULT)),
                curiosity_value=float(payload.get("curiosity_value", UNKNOWN_CURIOSITY_DEFAULT)),
                relationship_valence=float(payload.get("relationship_valence", DEFAULT_RELATIONSHIP_VALENCE)),
                interaction_history=list(payload.get("interaction_history", [])),
                modeled_drive_state=dict(payload.get("modeled_drive_state", {})),
                welfare_concern=float(payload.get("welfare_concern", DEFAULT_WELFARE_CONCERN)),
            )

        layer._index_entity_memory_nodes()
        for model in layer.entities.values():
            layer._upsert_entity_memory_node(model, {})
        return layer

    def summary(self) -> str:
        """Return a concise human-readable summary of empathy state."""

        if not self.entities:
            return "Entities: 0 | avg_threat=0.00 | avg_curiosity=0.00 | relationship_depth=0.00 | top_threat=none | top_curiosity=none"

        models = list(self.entities.values())
        social_signals = self.get_social_drive_signals()
        average_threat = sum(model.threat_assessment for model in models) / float(len(models))
        average_curiosity = sum(model.curiosity_value for model in models) / float(len(models))
        top_threat = self.get_most_threatening(limit=1)
        top_curiosity = self.get_most_curious(limit=1)
        top_threat_label = "none" if not top_threat else f"{top_threat[0].entity_id}:{top_threat[0].threat_assessment:.2f}"
        top_curiosity_label = "none" if not top_curiosity else f"{top_curiosity[0].entity_id}:{top_curiosity[0].curiosity_value:.2f}"

        return (
            f"Entities: {len(models)} | avg_threat={average_threat:.2f} | "
            f"avg_curiosity={average_curiosity:.2f} | "
            f"relationship_depth={social_signals[SOCIAL_SIGNAL_RELATIONSHIP_DEPTH]:.2f} | "
            f"top_threat={top_threat_label} | "
            f"top_curiosity={top_curiosity_label}"
        )

    def _entity_id_from_entity(self, entity: dict[str, Any]) -> str:
        """Return a stable identifier for an observed entity."""

        if "id" in entity:
            return _normalize_identifier(entity.get("id"), DEFAULT_ENTITY_TYPE)
        if "name" in entity:
            return _normalize_identifier(entity.get("name"), DEFAULT_ENTITY_TYPE)
        return _normalize_identifier(DEFAULT_ENTITY_TYPE, DEFAULT_ENTITY_TYPE)

    def _entity_type_from_entity(self, entity: dict[str, Any]) -> str:
        """Infer a canonical entity type from tags or explicit fields."""

        explicit_type = str(entity.get("entity_type", "")).strip().lower()
        if explicit_type:
            return explicit_type

        tags = entity.get("tags", [])
        if isinstance(tags, list):
            normalized_tags = {str(tag).strip().lower() for tag in tags}
            for candidate in ENTITY_TYPE_PRIORITY:
                if candidate in normalized_tags:
                    return candidate
            if normalized_tags:
                return sorted(normalized_tags)[0]
        return DEFAULT_ENTITY_TYPE

    def _index_entity_memory_nodes(self) -> None:
        """Index existing entity memory nodes for update-in-place behavior."""

        self._entity_memory_node_ids = {}
        for node in self.memory.nodes.values():
            if node.node_type != ENTITY_MEMORY_NODE_TYPE:
                continue
            entity_id = _normalize_identifier(node.content.get("entity_id", ""), "")
            if not entity_id:
                continue
            self._entity_memory_node_ids[entity_id] = node.id

    def _record_interaction(self, model: EntityModel, action: str, outcome: dict[str, Any]) -> bool:
        """Record a single interaction and return whether it was surprising."""

        action_key = str(action).strip().lower()
        if not action_key:
            return False

        actual_success = self._outcome_success(outcome)
        prior_success_rate = model.predicted_reactions.get(action_key)
        unexpected = False
        if prior_success_rate is not None:
            actual_value = MAX_SIGNAL_VALUE if actual_success else MIN_SIGNAL_VALUE
            unexpected = abs(prior_success_rate - actual_value) >= PREDICTION_POSITIVE_RESPONSE_THRESHOLD

        model.interaction_history.append(
            {
                INTERACTION_HISTORY_TIMESTAMP_KEY: float(outcome.get("timestamp", time.time())),
                INTERACTION_HISTORY_ACTION_KEY: action_key,
                INTERACTION_HISTORY_SUCCESS_KEY: actual_success,
                INTERACTION_HISTORY_VALENCE_KEY: self._outcome_valence(outcome),
                INTERACTION_HISTORY_UNEXPECTED_KEY: unexpected,
            }
        )
        if len(model.interaction_history) > MAX_INTERACTION_HISTORY:
            model.interaction_history = model.interaction_history[-MAX_INTERACTION_HISTORY:]
        return unexpected

    def _updated_threat(
        self,
        model: EntityModel,
        entity: dict[str, Any],
        outcome: dict[str, Any] | None,
        unexpected_interaction: bool,
    ) -> float:
        """Return an updated threat estimate from properties and outcomes."""

        property_threat = self._entity_property_threat(entity, model.entity_type)
        threat = model.threat_assessment + ((property_threat - model.threat_assessment) * THREAT_LEARNING_RATE)

        if outcome is not None:
            outcome_valence = self._outcome_valence(outcome)
            if outcome_valence < MIN_SIGNAL_VALUE:
                threat += abs(outcome_valence) * NEGATIVE_VALENCE_THREAT_INCREASE
            elif outcome_valence > MIN_SIGNAL_VALUE:
                threat -= outcome_valence * POSITIVE_VALENCE_THREAT_REDUCTION

        if unexpected_interaction:
            threat += UNEXPECTED_THREAT_BONUS

        return _clamp(threat, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

    def _updated_curiosity(self, model: EntityModel, unexpected_interaction: bool) -> float:
        """Return an updated curiosity estimate from familiarity and surprise."""

        familiarity_decay = min(
            max(MIN_SIGNAL_VALUE, float(model.observation_count - 1)) * CURIOSITY_DECAY_RATE,
            UNKNOWN_CURIOSITY_DEFAULT,
        )
        curiosity = UNKNOWN_CURIOSITY_DEFAULT - familiarity_decay
        if unexpected_interaction:
            curiosity += UNEXPECTED_CURIOSITY_BONUS
        if model.predicted_reactions:
            average_prediction_confidence = self._entity_prediction_confidence(model)
            curiosity += (MAX_SIGNAL_VALUE - average_prediction_confidence) * (
                CURIOSITY_DECAY_RATE * CURIOSITY_PREDICTION_GAP_SCALE
            )
        return _clamp(curiosity, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

    def _entity_property_threat(self, entity: dict[str, Any], entity_type: str) -> float:
        """Estimate raw threat from observable entity properties."""

        properties = entity.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        reactivity = _coerce_float(properties.get(PROPERTY_REACTIVITY_KEY), UNKNOWN_THREAT_DEFAULT)
        aggression = _coerce_float(
            properties.get(PROPERTY_AGGRESSION_KEY, properties.get(PROPERTY_THREAT_KEY)),
            reactivity,
        )
        mass = _coerce_float(properties.get(PROPERTY_MASS_KEY), UNKNOWN_THREAT_DEFAULT)
        type_bias = ENTITY_TYPE_THREAT_BIAS.get(entity_type, MIN_SIGNAL_VALUE)

        threat = (
            (reactivity * REACTIVITY_THREAT_WEIGHT)
            + (aggression * AGGRESSION_THREAT_WEIGHT)
            + (mass * MASS_THREAT_WEIGHT)
            + type_bias
        )
        return _clamp(threat, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

    def _rebuild_predicted_reactions(self, model: EntityModel) -> None:
        """Recompute action success rates from recent interaction history."""

        action_scores: dict[str, list[float]] = {}
        for interaction in model.interaction_history:
            action = str(interaction.get(INTERACTION_HISTORY_ACTION_KEY, "")).strip().lower()
            if not action:
                continue
            action_scores.setdefault(action, []).append(
                MAX_SIGNAL_VALUE if bool(interaction.get(INTERACTION_HISTORY_SUCCESS_KEY, INTERACTION_SUCCESS_DEFAULT)) else MIN_SIGNAL_VALUE
            )

        model.predicted_reactions = {
            action: _clamp(sum(scores) / float(len(scores)), MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
            for action, scores in action_scores.items()
            if scores
        }

    def _entity_prediction_confidence(self, model: EntityModel) -> float:
        """Return how confidently this entity's reactions are understood."""

        if not model.predicted_reactions:
            return MIN_SIGNAL_VALUE

        confidences = [
            abs(success_rate - PREDICTION_CONFIDENCE_CENTER) * PREDICTION_CONFIDENCE_SCALE
            for success_rate in model.predicted_reactions.values()
        ]
        return _clamp(sum(confidences) / float(len(confidences)), MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)

    def _modeled_drive_state(self, entity: dict[str, Any], model: EntityModel) -> dict[str, float]:
        """Infer a simple functional drive model for another entity."""

        properties = entity.get("properties", {})
        if not isinstance(properties, dict):
            properties = {}

        resource_interest = _coerce_float(properties.get(PROPERTY_RESOURCE_DENSITY_KEY), MIN_SIGNAL_VALUE)
        positive_relationship = max(MIN_SIGNAL_VALUE, model.relationship_valence)
        negative_relationship = max(MIN_SIGNAL_VALUE, -model.relationship_valence)

        return {
            "avoidance": _clamp(
                (model.threat_assessment * MODELED_DRIVE_AVOIDANCE_WEIGHT) + negative_relationship,
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            ),
            "engagement": _clamp(
                (positive_relationship * MODELED_DRIVE_ENGAGEMENT_WEIGHT)
                + (model.curiosity_value * (MAX_SIGNAL_VALUE - MODELED_DRIVE_ENGAGEMENT_WEIGHT)),
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            ),
            "curiosity": _clamp(
                model.curiosity_value * MODELED_DRIVE_CURIOSITY_WEIGHT,
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            ),
            "stability": _clamp(
                ((MAX_SIGNAL_VALUE - model.threat_assessment) * MODELED_DRIVE_STABILITY_WEIGHT)
                + (resource_interest * (MAX_SIGNAL_VALUE - MODELED_DRIVE_STABILITY_WEIGHT)),
                MIN_SIGNAL_VALUE,
                MAX_SIGNAL_VALUE,
            ),
        }

    def _calculate_welfare_concern(self, model: EntityModel) -> float:
        """Return welfare concern from positive familiarity and relationship."""

        if self.developmental_stage <= WELFARE_STAGE_THRESHOLD:
            return DEFAULT_WELFARE_CONCERN

        positive_relationship = max(MIN_SIGNAL_VALUE, model.relationship_valence)
        familiarity = min(model.observation_count / WELFARE_OBSERVATION_NORMALIZER, MAX_SIGNAL_VALUE)
        stage_factor = _clamp(
            (self.developmental_stage - WELFARE_STAGE_THRESHOLD) / (MAX_SIGNAL_VALUE - WELFARE_STAGE_THRESHOLD),
            MIN_SIGNAL_VALUE,
            MAX_SIGNAL_VALUE,
        )
        return _clamp(
            positive_relationship * familiarity * stage_factor * HIGH_STAGE_WELFARE_SCALE,
            MIN_SIGNAL_VALUE,
            MAX_SIGNAL_VALUE,
        )

    def _entity_snapshot(self, entity: dict[str, Any]) -> dict[str, Any]:
        """Return a stable subset of raw entity observation data."""

        snapshot: dict[str, Any] = {}
        for key in ENTITY_SNAPSHOT_KEYS:
            if key in entity:
                value = entity[key]
                if isinstance(value, dict):
                    snapshot[key] = dict(value)
                elif isinstance(value, list):
                    snapshot[key] = list(value)
                else:
                    snapshot[key] = value
        return snapshot

    def _upsert_entity_memory_node(self, model: EntityModel, entity: dict[str, Any]) -> None:
        """Create or update a single memory node for an entity model."""

        node_id = self._entity_memory_node_ids.get(model.entity_id)
        if node_id is None or node_id not in self.memory.nodes:
            node = self.memory.create_node(
                node_type=ENTITY_MEMORY_NODE_TYPE,
                content=self._entity_memory_content(model, entity),
                valence=model.relationship_valence,
                relevance=ENTITY_NODE_RELEVANCE,
                tags=[ENTITY_MEMORY_TAG, model.entity_id, model.entity_type],
            )
            self._entity_memory_node_ids[model.entity_id] = node.id
            return

        node = self.memory.nodes[node_id]
        node.content = self._entity_memory_content(model, entity)
        node.tags = [ENTITY_MEMORY_TAG, model.entity_id, model.entity_type]
        node.last_accessed = time.time()
        node.relevance = ENTITY_NODE_RELEVANCE
        node.valence = _clamp(
            ((node.valence * (MAX_SIGNAL_VALUE - INTERACTION_RELATIONSHIP_MEMORY_BLEND))
            + (model.relationship_valence * INTERACTION_RELATIONSHIP_MEMORY_BLEND)),
            MIN_RELATIONSHIP_VALENCE,
            MAX_RELATIONSHIP_VALENCE,
        )

    def _entity_memory_content(self, model: EntityModel, entity: dict[str, Any]) -> dict[str, Any]:
        """Build a memory-node payload for an entity model."""

        return {
            "entity_id": model.entity_id,
            "entity_type": model.entity_type,
            "first_seen": model.first_seen,
            "last_seen": model.last_seen,
            "observation_count": model.observation_count,
            "predicted_reactions": dict(model.predicted_reactions),
            "threat_assessment": model.threat_assessment,
            "curiosity_value": model.curiosity_value,
            "relationship_valence": model.relationship_valence,
            "modeled_drive_state": dict(model.modeled_drive_state),
            "welfare_concern": model.welfare_concern,
            "snapshot": self._entity_snapshot(entity),
        }

    def _outcome_success(self, outcome: dict[str, Any]) -> bool:
        """Return a coarse success signal from an outcome payload."""

        if "success" in outcome:
            return bool(outcome.get("success"))
        return self._outcome_valence(outcome) >= MIN_SIGNAL_VALUE

    def _outcome_valence(self, outcome: dict[str, Any]) -> float:
        """Extract a best-effort valence estimate from an outcome payload."""

        return _clamp(
            _coerce_float(outcome.get("valence", outcome.get("outcome_valence", MIN_SIGNAL_VALUE)), MIN_SIGNAL_VALUE),
            MIN_RELATIONSHIP_VALENCE,
            MAX_RELATIONSHIP_VALENCE,
        )


if __name__ == "__main__":
    demo_memory = MemoryGraph()
    empathy = EmpathyLayer(demo_memory, developmental_stage=DEMO_PREDICTION_STAGE)

    demo_entities = [
        {
            "id": "river_creature",
            "name": "River Creature",
            "tags": ["creature", "mobile", "entity"],
            "properties": {"reactivity": DEMO_RIVER_REACTIVITY, "mass": DEMO_RIVER_MASS},
        },
        {
            "id": "stone_keeper",
            "name": "Stone Keeper",
            "tags": ["npc", "entity"],
            "properties": {
                "reactivity": DEMO_KEEPER_REACTIVITY,
                "resource_density": DEMO_KEEPER_RESOURCE_DENSITY,
            },
        },
        {
            "id": "cavern_hunter",
            "name": "Cavern Hunter",
            "tags": ["monster", "entity"],
            "properties": {
                "reactivity": DEMO_HUNTER_REACTIVITY,
                "aggression": DEMO_HUNTER_AGGRESSION,
                "mass": DEMO_HUNTER_MASS,
            },
        },
    ]

    for entity in demo_entities:
        empathy.observe_entity(entity, developmental_stage=DEMO_PREDICTION_STAGE)

    empathy.observe_entity(
        demo_entities[0],
        itera_action="approach",
        outcome={"success": False, "valence": DEMO_NEGATIVE_VALENCE_STRONG, "timestamp": time.time()},
        developmental_stage=DEMO_PREDICTION_STAGE,
    )
    empathy.observe_entity(
        demo_entities[1],
        itera_action="approach",
        outcome={"success": True, "valence": DEMO_POSITIVE_VALENCE, "timestamp": time.time()},
        developmental_stage=DEMO_PREDICTION_STAGE,
    )
    empathy.observe_entity(
        demo_entities[2],
        itera_action="observe",
        outcome={"success": False, "valence": DEMO_NEGATIVE_VALENCE_MILD, "timestamp": time.time()},
        developmental_stage=DEMO_PREDICTION_STAGE,
    )

    empathy.update_relationship("river_creature", DEMO_RELATIONSHIP_POSITIVE_SMALL)
    empathy.update_relationship("stone_keeper", DEMO_RELATIONSHIP_POSITIVE_LARGE)
    empathy.update_relationship("cavern_hunter", DEMO_RELATIONSHIP_NEGATIVE)

    social_signals = empathy.get_social_drive_signals()
    print("Social drive signals:", social_signals)
    print("Predicted reaction:", empathy.predict_reaction("river_creature", "approach"))

    empathy.update_developmental_stage(DEMO_WELFARE_STAGE)
    for entity in demo_entities:
        empathy.observe_entity(entity, developmental_stage=DEMO_WELFARE_STAGE)

    river_welfare = empathy.get_welfare_concern("river_creature")
    print("River welfare concern:", round(river_welfare, 4))
    print("Welfare concern positive:", river_welfare > 0.0)
    print("Summary:")
    print(empathy.summary())

    restored = EmpathyLayer.from_dict(empathy.to_dict(), memory=demo_memory)
    print("Round trip confirmed:", restored.known_entity_count() == empathy.known_entity_count())

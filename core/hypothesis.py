"""Itera's hypothesis engine.

This module implements Itera's observe -> hypothesize -> test -> update loop.
The engine stays world-agnostic and uses drive salience to decide what is worth
investigating next.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import time
from typing import Any
from uuid import uuid4

try:
    from core.drives import DriveHierarchy
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    from drives import DriveHierarchy


class HypothesisType(str, Enum):
    """Supported hypothesis categories."""

    PATTERN = "pattern"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    RELATIONAL = "relational"


MIN_SIGNAL_VALUE = 0.0
MAX_SIGNAL_VALUE = 1.0
MIN_VALENCE = -1.0
MAX_VALENCE = 1.0
DEFAULT_OBSERVATION_WINDOW = 20
DEFAULT_PENDING_HYPOTHESIS_LIMIT = 128
NOVELTY_GENERATION_THRESHOLD = 0.18
DRIVE_GENERATION_THRESHOLD = 0.1
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
INITIAL_CONFIDENCE = 0.5
CONFIRM_CONFIDENCE_THRESHOLD = 0.8
REFUTE_CONFIDENCE_THRESHOLD = 0.2
MATCH_CONFIDENCE_GAIN = 0.18
MISMATCH_CONFIDENCE_LOSS = -0.22
TESTING_STATUS_PENALTY = 0.85
STALE_ABANDONED_VALENCE = -0.15
NOVELTY_KEY_WEIGHT = 0.4
NOVELTY_VALUE_WEIGHT = 0.6
PRIORITY_DRIVE_WEIGHT = 0.45
PRIORITY_NOVELTY_WEIGHT = 0.35
PRIORITY_CONFIDENCE_WEIGHT = 0.2
TRIM_DRIVE_WEIGHT = 0.5
TRIM_NOVELTY_WEIGHT = 0.3
TRIM_CONFIDENCE_WEIGHT = 0.2
CONFIDENCE_MIDPOINT = (MIN_CONFIDENCE + MAX_CONFIDENCE) / 2.0
HYPOTHESIS_PATTERN_NOVELTY_THRESHOLD = 0.3
HYPOTHESIS_SOCIAL_RELATION_NOVELTY_THRESHOLD = 0.5
SINGLE_ENTITY_COUNT = 1
MULTI_ENTITY_THRESHOLD = 2
DEVELOPMENTAL_REINQUIRY_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
REINQUIRY_HYPOTHESES_PER_THRESHOLD = 3
REINQUIRY_STAGE_KEY = "last_reinquiry_stage"
CURIOSITY_FLOOR_INTERVAL = 50
CURIOSITY_FLOOR_NOVELTY_OVERRIDE = 0.15
STAGE_COMPARATIVE_REINQUIRY_MIN = 0.5
STAGE_RELATIONAL_REINQUIRY_MIN = 0.7

DRIVE_TEST_INTENTS: dict[str, str] = {
    "SURVIVAL": "probe immediate risk or resource conditions",
    "SECURITY": "reduce uncertainty in the environment",
    "SOCIAL": "improve prediction of another entity",
    "MASTERY": "refine an explanatory or skill model",
    "ACTUALIZATION": "test a purpose-linked or identity-linked pattern",
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


def _coerce_numeric(value: Any) -> float | None:
    """Convert a value to float when possible."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_delta(delta: float) -> float:
    """Map a non-negative difference into the 0.0-1.0 range."""

    if delta <= 0.0:
        return 0.0
    return delta / (1.0 + delta)


@dataclass
class Observation:
    """A single perceived state from the world."""

    timestamp: float
    data: dict[str, Any]
    drive_context: dict[str, Any]
    novelty_score: float


@dataclass
class Hypothesis:
    """A testable prediction Itera has formed."""

    id: str
    created_at: float
    observation_ids: list[str]
    statement: str
    predicted_outcome: dict[str, Any]
    drive_source: str
    confidence: float
    test_count: int
    status: str
    valence: float
    hypothesis_type: HypothesisType = HypothesisType.PATTERN


@dataclass
class TestResult:
    """Outcome of testing a hypothesis."""

    hypothesis_id: str
    timestamp: float
    actual_outcome: dict[str, Any]
    matched_prediction: bool
    confidence_delta: float
    valence: float


class HypothesisEngine:
    """Itera's scientific mind."""

    def __init__(self, drives: DriveHierarchy) -> None:
        """Initialize the engine with a shared drive hierarchy."""

        self.drives = drives
        self.observations: dict[str, Observation] = {}
        self._observation_order: list[str] = []
        self._observation_identity_map: dict[int, str] = {}
        self.hypotheses: dict[str, Hypothesis] = {}
        self.test_results: list[TestResult] = []
        self.last_reinquiry_stage = MIN_SIGNAL_VALUE
        self.cycles_since_last_generation = 0

    def _drive_snapshot(self) -> dict[str, Any]:
        """Capture the current drive hierarchy as observation context."""

        dominant_drive, dominant_weight = self.drives.get_dominant_drive()
        return {
            "developmental_stage": self.drives.developmental_stage,
            "dominant_drive": dominant_drive,
            "dominant_weight": dominant_weight,
            "suppression_map": self.drives.get_suppression_map(),
            "tiers": {
                tier_id: tier.summary()
                for tier_id, tier in self.drives.tiers.items()
            },
        }

    def _recent_observations(self) -> list[Observation]:
        """Return the recent observation window for novelty scoring."""

        recent_ids = self._observation_order[-DEFAULT_OBSERVATION_WINDOW:]
        return [self.observations[observation_id] for observation_id in recent_ids]

    def _novelty_against_recent(self, raw_observation: dict[str, Any]) -> float:
        """Estimate how different an observation is from recent history."""

        recent = self._recent_observations()
        if not recent:
            return 1.0

        deltas: list[float] = []
        current_keys = set(raw_observation)
        for prior in recent:
            prior_keys = set(prior.data)
            key_union = current_keys | prior_keys
            if not key_union:
                deltas.append(0.0)
                continue

            key_difference = len(current_keys ^ prior_keys) / len(key_union)
            value_deltas: list[float] = []
            for key in key_union:
                current_value = raw_observation.get(key)
                prior_value = prior.data.get(key)
                current_numeric = _coerce_numeric(current_value)
                prior_numeric = _coerce_numeric(prior_value)
                if current_numeric is not None and prior_numeric is not None:
                    value_deltas.append(abs(current_numeric - prior_numeric))
                elif current_value != prior_value:
                    value_deltas.append(1.0)
                else:
                    value_deltas.append(0.0)

            average_value_delta = sum(value_deltas) / len(value_deltas)
            deltas.append(
                _clamp(
                    (key_difference * NOVELTY_KEY_WEIGHT)
                    + (_normalize_delta(average_value_delta) * NOVELTY_VALUE_WEIGHT),
                    0.0,
                    1.0,
                )
            )

        return sum(deltas) / len(deltas)

    def _observation_id(self, observation: Observation) -> str | None:
        """Return the engine-managed identifier for an observation."""

        return self._observation_identity_map.get(id(observation))

    def _observation_novelty(self, observation_ids: list[str]) -> float:
        """Return average novelty for a set of stored observations."""

        if not observation_ids:
            return 0.0
        scores = [
            self.observations[observation_id].novelty_score
            for observation_id in observation_ids
            if observation_id in self.observations
        ]
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _drive_weight(self, drive_name: str) -> float:
        """Return the effective weight for a named drive tier."""

        for tier in self.drives.tiers.values():
            if tier.name == drive_name:
                return tier.effective_weight()
        return 0.0

    def _top_driving_tier(self) -> tuple[str, float]:
        """Return the strongest current drive tier and its weight."""

        tier = max(self.drives.tiers.values(), key=lambda item: item.effective_weight())
        return tier.name, tier.effective_weight()

    def _focus_keys_for_observation(self, observation: Observation, drive_name: str) -> list[str]:
        """Select the observation features most worth explaining."""

        recent = self._recent_observations()
        if not recent:
            return list(observation.data.keys())[:3]

        similarity_scores: list[tuple[float, str]] = []
        for key, value in observation.data.items():
            current_numeric = _coerce_numeric(value)
            deltas: list[float] = []
            for prior in recent:
                prior_value = prior.data.get(key)
                prior_numeric = _coerce_numeric(prior_value)
                if current_numeric is not None and prior_numeric is not None:
                    deltas.append(abs(current_numeric - prior_numeric))
                elif value != prior_value:
                    deltas.append(1.0)
            if deltas:
                similarity_scores.append((_normalize_delta(sum(deltas) / len(deltas)), key))
            else:
                similarity_scores.append((1.0, key))

        similarity_scores.sort(reverse=True)
        focus_keys = [key for _, key in similarity_scores[:3]]

        if not focus_keys and drive_name in observation.data:
            focus_keys.append(drive_name)

        return focus_keys

    def _observation_entities(self, observation: Observation) -> list[dict[str, Any]]:
        """Return normalized entity dictionaries from an observation."""

        entities = observation.data.get("entities", [])
        if not isinstance(entities, list):
            return []
        return [entity for entity in entities if isinstance(entity, dict)]

    def _describe_observation_entities(self, observation: Observation) -> str:
        """Return a readable entity description for hypothesis statements."""

        names = [
            str(entity.get("name") or entity.get("id") or f"entity_{index + 1}")
            for index, entity in enumerate(self._observation_entities(observation))
        ]
        if not names:
            return "the current conditions"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return f"{', '.join(names[:-1])}, and {names[-1]}"

    def _select_hypothesis_type(self, observation: Observation, drive_source: str) -> HypothesisType:
        """Choose the hypothesis category that best fits an observation."""

        entity_count = len(self._observation_entities(observation))
        if observation.novelty_score < HYPOTHESIS_PATTERN_NOVELTY_THRESHOLD:
            return HypothesisType.PATTERN
        if entity_count == 0:
            return HypothesisType.PATTERN
        if entity_count == SINGLE_ENTITY_COUNT and drive_source == "SOCIAL":
            return HypothesisType.PREDICTIVE
        if entity_count == SINGLE_ENTITY_COUNT and self.test_results:
            return HypothesisType.CAUSAL
        if entity_count >= MULTI_ENTITY_THRESHOLD:
            if drive_source == "SOCIAL":
                # RELATIONAL requires repeated entity encounters to lower novelty below the
                # threshold — a fresh single run may never reach it. On persisted identity
                # with prior relationship history, novelty settles and RELATIONAL fires.
                if observation.novelty_score >= HYPOTHESIS_SOCIAL_RELATION_NOVELTY_THRESHOLD:
                    return HypothesisType.PREDICTIVE
                return HypothesisType.RELATIONAL
            return HypothesisType.COMPARATIVE
        return HypothesisType.PREDICTIVE

    def _build_hypothesis_statement(
        self,
        hypothesis_type: HypothesisType,
        observation: Observation,
        focus_keys: list[str],
        drive_source: str,
    ) -> str:
        """Return a readable statement matched to a selected hypothesis type."""

        focus_fragment = ", ".join(focus_keys) if focus_keys else "the current signals"
        entity_fragment = self._describe_observation_entities(observation)

        if hypothesis_type == HypothesisType.CAUSAL:
            return (
                f"If Itera engages with {entity_fragment}, changes in {focus_fragment} should follow "
                f"under the current {drive_source} concern."
            )
        if hypothesis_type == HypothesisType.COMPARATIVE:
            return (
                f"Comparing {entity_fragment}, Itera expects different behavior in {focus_fragment} "
                f"under the current {drive_source} concern."
            )
        if hypothesis_type == HypothesisType.RELATIONAL:
            return (
                f"Itera expects the relationship among {entity_fragment} to shape {focus_fragment} "
                f"under the current {drive_source} concern."
            )
        if hypothesis_type == HypothesisType.PREDICTIVE:
            return (
                f"When {entity_fragment} is present, future observations should show {focus_fragment} "
                f"in a predictable way for the current {drive_source} concern."
            )
        return (
            f"Itera expects {focus_fragment} to repeat in a stable pattern under the current "
            f"{drive_source} concern."
        )

    def observe(self, raw_observation: dict[str, Any]) -> Observation:
        """Process a raw world observation, score novelty, and store it."""

        self.drives.update(raw_observation)
        novelty_score = self._novelty_against_recent(raw_observation)
        observation = Observation(
            timestamp=time.time(),
            data=dict(raw_observation),
            drive_context=self._drive_snapshot(),
            novelty_score=_clamp(novelty_score, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE),
        )

        observation_id = str(uuid4())
        self.observations[observation_id] = observation
        self._observation_order.append(observation_id)
        self._observation_identity_map[id(observation)] = observation_id
        if len(self._observation_order) > DEFAULT_PENDING_HYPOTHESIS_LIMIT * 2:
            self._observation_order = self._observation_order[-(DEFAULT_PENDING_HYPOTHESIS_LIMIT * 2):]

        return observation

    def generate_hypothesis(
        self,
        observation: Observation,
        novelty_override: float | None = None,
    ) -> Hypothesis | None:
        """Generate a drive-weighted hypothesis from an observation when warranted."""

        drive_name, drive_weight = self._top_driving_tier()
        effective_novelty = observation.novelty_score if novelty_override is None else _clamp(
            novelty_override,
            MIN_SIGNAL_VALUE,
            MAX_SIGNAL_VALUE,
        )
        if novelty_override is None and effective_novelty < NOVELTY_GENERATION_THRESHOLD:
            return None
        if drive_weight < DRIVE_GENERATION_THRESHOLD:
            return None

        observation_id = self._observation_id(observation)
        if observation_id is None:
            return None

        focus_keys = self._focus_keys_for_observation(observation, drive_name)
        if not focus_keys:
            return None

        predicted_outcome = {key: observation.data[key] for key in focus_keys if key in observation.data}
        hypothesis_type = self._select_hypothesis_type(observation, drive_name)
        statement = self._build_hypothesis_statement(hypothesis_type, observation, focus_keys, drive_name)

        hypothesis = Hypothesis(
            id=str(uuid4()),
            created_at=time.time(),
            observation_ids=[observation_id],
            statement=statement,
            predicted_outcome=predicted_outcome,
            drive_source=drive_name,
            confidence=INITIAL_CONFIDENCE,
            test_count=0,
            status="pending",
            valence=0.0,
            hypothesis_type=hypothesis_type,
        )
        self.hypotheses[hypothesis.id] = hypothesis
        self.cycles_since_last_generation = 0
        self._trim_pending_hypotheses()
        return hypothesis

    def should_generate_from_curiosity_floor(self, cycles_since_last_generation: int) -> bool:
        """
        Returns True if enough cycles have passed without
        generating a new hypothesis.
        Triggered when cycles_since_last_generation >=
        CURIOSITY_FLOOR_INTERVAL and confirmed > 0.
        """

        self.cycles_since_last_generation = max(0, int(cycles_since_last_generation))
        return (
            self.cycles_since_last_generation >= CURIOSITY_FLOOR_INTERVAL
            and len(self.get_confirmed()) > 0
        )

    def check_developmental_reinquiry(
        self,
        current_stage: float,
        observation: Observation,
    ) -> list[Hypothesis]:
        """
        Check if Itera's developmental stage has crossed a
        threshold that warrants re-examining known phenomena.

        At each threshold in DEVELOPMENTAL_REINQUIRY_THRESHOLDS,
        generate REINQUIRY_HYPOTHESES_PER_THRESHOLD new hypotheses
        by taking confirmed hypotheses and forming deeper questions.

        Only fires once per threshold, tracked by last_reinquiry_stage.
        """

        normalized_stage = _clamp(current_stage, MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE)
        crossed_thresholds = [
            threshold
            for threshold in DEVELOPMENTAL_REINQUIRY_THRESHOLDS
            if self.last_reinquiry_stage < threshold <= normalized_stage
        ]
        if not crossed_thresholds:
            return []

        confirmed = sorted(
            self.get_confirmed(),
            key=lambda hypothesis: (hypothesis.confidence, hypothesis.created_at),
            reverse=True,
        )
        if not confirmed:
            return []

        generated: list[Hypothesis] = []
        for threshold in crossed_thresholds:
            for source_hypothesis in confirmed[:REINQUIRY_HYPOTHESES_PER_THRESHOLD]:
                generated.append(
                    self._generate_reinquiry_hypothesis(
                        source_hypothesis=source_hypothesis,
                        current_stage=threshold,
                        observation=observation,
                    )
                )
            self.last_reinquiry_stage = threshold

        if generated:
            self.cycles_since_last_generation = 0
        return generated

    def _generate_reinquiry_hypothesis(
        self,
        source_hypothesis: Hypothesis,
        current_stage: float,
        observation: Observation,
    ) -> Hypothesis:
        """
        Generate a deeper hypothesis from a confirmed one.

        New hypotheses start pending with fresh confidence and no tests,
        regardless of the source hypothesis confidence.
        """

        observation_id = self._observation_id(observation)
        observation_ids = list(dict.fromkeys(
            ([observation_id] if observation_id is not None else [])
            + list(source_hypothesis.observation_ids)
        ))
        predicted_outcome = dict(source_hypothesis.predicted_outcome)
        if not predicted_outcome:
            focus_keys = self._focus_keys_for_observation(observation, source_hypothesis.drive_source)
            predicted_outcome = {
                key: observation.data[key]
                for key in focus_keys
                if key in observation.data
            }

        if current_stage < STAGE_COMPARATIVE_REINQUIRY_MIN:
            hypothesis_type = HypothesisType.CAUSAL
            statement = (
                "Re-inquiry: what causes this confirmed pattern to hold? "
                f"Source: {source_hypothesis.statement}"
            )
        elif current_stage < STAGE_RELATIONAL_REINQUIRY_MIN:
            hypothesis_type = HypothesisType.COMPARATIVE
            statement = (
                "Re-inquiry: how does this confirmed pattern differ across entities or conditions? "
                f"Source: {source_hypothesis.statement}"
            )
        else:
            entity_count = len(self._observation_entities(observation))
            hypothesis_type = HypothesisType.RELATIONAL if entity_count >= MULTI_ENTITY_THRESHOLD else HypothesisType.PREDICTIVE
            statement = (
                "Re-inquiry: how does this confirmed pattern relate to other patterns and predict future outcomes? "
                f"Source: {source_hypothesis.statement}"
            )

        return Hypothesis(
            id=str(uuid4()),
            created_at=time.time(),
            observation_ids=observation_ids,
            statement=statement,
            predicted_outcome=predicted_outcome,
            drive_source=source_hypothesis.drive_source,
            confidence=INITIAL_CONFIDENCE,
            test_count=0,
            status="pending",
            valence=0.0,
            hypothesis_type=hypothesis_type,
        )

    def _trim_pending_hypotheses(self) -> None:
        """Bound pending-hypothesis growth by discarding the weakest stale items."""

        active_pending = [hypothesis for hypothesis in self.hypotheses.values() if hypothesis.status == "pending"]
        if len(active_pending) <= DEFAULT_PENDING_HYPOTHESIS_LIMIT:
            return

        ranked = sorted(
            active_pending,
            key=lambda hypothesis: (
                self._drive_weight(hypothesis.drive_source) * TRIM_DRIVE_WEIGHT
                + self._observation_novelty(hypothesis.observation_ids) * TRIM_NOVELTY_WEIGHT
                + hypothesis.confidence * TRIM_CONFIDENCE_WEIGHT
            ),
        )
        for hypothesis in ranked[: len(active_pending) - DEFAULT_PENDING_HYPOTHESIS_LIMIT]:
            hypothesis.status = "abandoned"
            hypothesis.valence = max(hypothesis.valence, STALE_ABANDONED_VALENCE)

    def design_test(self, hypothesis: Hypothesis) -> dict[str, Any]:
        """Return a world-agnostic test description for a hypothesis."""

        if hypothesis.id in self.hypotheses and self.hypotheses[hypothesis.id].status == "pending":
            self.hypotheses[hypothesis.id].status = "testing"

        target_keys = list(hypothesis.predicted_outcome.keys())
        return {
            "hypothesis_id": hypothesis.id,
            "drive_source": hypothesis.drive_source,
            "intent": DRIVE_TEST_INTENTS.get(hypothesis.drive_source, "gather clarifying evidence"),
            "observation_ids": list(hypothesis.observation_ids),
            "target_keys": target_keys,
            "expected_outcome": dict(hypothesis.predicted_outcome),
            "suggested_action": {
                "type": "gather_evidence",
                "focus": target_keys,
                "drive_alignment": {hypothesis.drive_source: 1.0},
            },
        }

    def evaluate(self, hypothesis: Hypothesis, result: TestResult) -> None:
        """Update hypothesis confidence and valence based on test result."""

        stored = self.hypotheses.get(hypothesis.id)
        if stored is None:
            return

        if stored.id != result.hypothesis_id:
            raise ValueError("TestResult hypothesis_id does not match the supplied hypothesis.")

        confidence_delta = result.confidence_delta
        if confidence_delta == 0.0:
            confidence_delta = MATCH_CONFIDENCE_GAIN if result.matched_prediction else MISMATCH_CONFIDENCE_LOSS

        stored.test_count += 1
        stored.confidence = _clamp(stored.confidence + confidence_delta, MIN_CONFIDENCE, MAX_CONFIDENCE)
        stored.valence = _clamp((stored.valence + result.valence) / 2.0 if stored.test_count > 1 else result.valence, MIN_VALENCE, MAX_VALENCE)

        if stored.confidence >= CONFIRM_CONFIDENCE_THRESHOLD:
            stored.status = "confirmed"
        elif stored.confidence <= REFUTE_CONFIDENCE_THRESHOLD:
            stored.status = "refuted"
        else:
            stored.status = "pending"

        normalized_result = TestResult(
            hypothesis_id=result.hypothesis_id,
            timestamp=result.timestamp,
            actual_outcome=dict(result.actual_outcome),
            matched_prediction=result.matched_prediction,
            confidence_delta=confidence_delta,
            valence=_clamp(result.valence, MIN_VALENCE, MAX_VALENCE),
        )
        self.test_results.append(normalized_result)

    def _hypothesis_priority(self, hypothesis: Hypothesis) -> float:
        """Score a hypothesis for selection."""

        if hypothesis.status not in {"pending", "testing"}:
            return -math.inf

        drive_score = self._drive_weight(hypothesis.drive_source)
        novelty_score = self._observation_novelty(hypothesis.observation_ids)
        status_multiplier = 1.0 if hypothesis.status == "pending" else TESTING_STATUS_PENALTY
        confidence_focus = 1.0 - abs(CONFIDENCE_MIDPOINT - hypothesis.confidence)

        return status_multiplier * (
            (drive_score * PRIORITY_DRIVE_WEIGHT)
            + (novelty_score * PRIORITY_NOVELTY_WEIGHT)
            + (confidence_focus * PRIORITY_CONFIDENCE_WEIGHT)
        )

    def select_next(self) -> Hypothesis | None:
        """Select the next hypothesis to test."""

        active = self.get_active()
        if not active:
            return None
        return max(active, key=self._hypothesis_priority)

    def abandon_stale(self, age_threshold: float) -> list[str]:
        """Mark old untested hypotheses as abandoned and return their IDs."""

        now = time.time()
        abandoned: list[str] = []
        for hypothesis in self.hypotheses.values():
            if hypothesis.status != "pending":
                continue
            if now - hypothesis.created_at >= age_threshold:
                hypothesis.status = "abandoned"
                hypothesis.valence = min(hypothesis.valence, STALE_ABANDONED_VALENCE)
                abandoned.append(hypothesis.id)
        return abandoned

    def get_active(self) -> list[Hypothesis]:
        """Return all pending or testing hypotheses."""

        return [
            hypothesis
            for hypothesis in self.hypotheses.values()
            if hypothesis.status in {"pending", "testing"}
        ]

    def get_confirmed(self) -> list[Hypothesis]:
        """Return all confirmed hypotheses."""

        return [
            hypothesis
            for hypothesis in self.hypotheses.values()
            if hypothesis.status == "confirmed"
        ]

    def get_refuted(self) -> list[Hypothesis]:
        """Return all refuted hypotheses."""

        return [
            hypothesis
            for hypothesis in self.hypotheses.values()
            if hypothesis.status == "refuted"
        ]

    def get_hypotheses_by_type(self, hypothesis_type: HypothesisType) -> list[Hypothesis]:
        """Return all stored hypotheses of a given type."""

        return [
            hypothesis
            for hypothesis in self.hypotheses.values()
            if hypothesis.hypothesis_type == hypothesis_type
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full hypothesis-engine state."""

        return {
            "observations": {
                observation_id: asdict(observation)
                for observation_id, observation in self.observations.items()
            },
            "observation_order": list(self._observation_order),
            "hypotheses": {
                hypothesis_id: asdict(hypothesis)
                for hypothesis_id, hypothesis in self.hypotheses.items()
            },
            "test_results": [asdict(result) for result in self.test_results],
            REINQUIRY_STAGE_KEY: self.last_reinquiry_stage,
            "cycles_since_last_generation": self.cycles_since_last_generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], drives: DriveHierarchy) -> "HypothesisEngine":
        """Deserialize an engine state using an existing drive hierarchy."""

        engine = cls(drives=drives)
        for observation_id, observation_data in data.get("observations", {}).items():
            observation = Observation(
                timestamp=float(observation_data["timestamp"]),
                data=dict(observation_data["data"]),
                drive_context=dict(observation_data["drive_context"]),
                novelty_score=_clamp(float(observation_data["novelty_score"]), MIN_SIGNAL_VALUE, MAX_SIGNAL_VALUE),
            )
            engine.observations[observation_id] = observation
            engine._observation_identity_map[id(observation)] = observation_id

        engine._observation_order = [
            observation_id
            for observation_id in data.get("observation_order", [])
            if observation_id in engine.observations
        ]

        for hypothesis_id, hypothesis_data in data.get("hypotheses", {}).items():
            raw_hypothesis_type = hypothesis_data.get("hypothesis_type", HypothesisType.PATTERN.value)
            engine.hypotheses[hypothesis_id] = Hypothesis(
                id=str(hypothesis_data["id"]),
                created_at=float(hypothesis_data["created_at"]),
                observation_ids=[str(item) for item in hypothesis_data.get("observation_ids", [])],
                statement=str(hypothesis_data["statement"]),
                predicted_outcome=dict(hypothesis_data.get("predicted_outcome", {})),
                drive_source=str(hypothesis_data["drive_source"]),
                confidence=_clamp(float(hypothesis_data["confidence"]), MIN_CONFIDENCE, MAX_CONFIDENCE),
                test_count=int(hypothesis_data["test_count"]),
                status=str(hypothesis_data["status"]),
                valence=_clamp(float(hypothesis_data["valence"]), MIN_VALENCE, MAX_VALENCE),
                hypothesis_type=(
                    raw_hypothesis_type
                    if isinstance(raw_hypothesis_type, HypothesisType)
                    else HypothesisType(str(raw_hypothesis_type))
                ),
            )

        engine.test_results = [
            TestResult(
                hypothesis_id=str(result_data["hypothesis_id"]),
                timestamp=float(result_data["timestamp"]),
                actual_outcome=dict(result_data.get("actual_outcome", {})),
                matched_prediction=bool(result_data["matched_prediction"]),
                confidence_delta=float(result_data["confidence_delta"]),
                valence=_clamp(float(result_data["valence"]), MIN_VALENCE, MAX_VALENCE),
            )
            for result_data in data.get("test_results", [])
        ]
        engine.last_reinquiry_stage = _clamp(
            float(data.get(REINQUIRY_STAGE_KEY, MIN_SIGNAL_VALUE)),
            MIN_SIGNAL_VALUE,
            MAX_SIGNAL_VALUE,
        )
        engine.cycles_since_last_generation = max(
            0,
            int(data.get("cycles_since_last_generation", 0)),
        )
        return engine

    def summary(self) -> str:
        """Return a concise summary of the current hypothesis state."""

        active = len(self.get_active())
        confirmed = len(self.get_confirmed())
        refuted = len(self.get_refuted())
        type_counts = ", ".join(
            f"{hypothesis_type.value}={len(self.get_hypotheses_by_type(hypothesis_type))}"
            for hypothesis_type in HypothesisType
        )
        next_hypothesis = self.select_next()
        next_line = "none"
        if next_hypothesis is not None:
            next_line = (
                f"{next_hypothesis.id} [{next_hypothesis.drive_source}/{next_hypothesis.hypothesis_type.value}] "
                f"confidence={next_hypothesis.confidence:.2f}"
            )

        return "\n".join(
            [
                f"Observations: {len(self.observations)}",
                f"Active hypotheses: {active}",
                f"Confirmed hypotheses: {confirmed}",
                f"Refuted hypotheses: {refuted}",
                f"Hypothesis types: {type_counts}",
                f"Stored test results: {len(self.test_results)}",
                f"Next hypothesis: {next_line}",
            ]
        )


if __name__ == "__main__":
    drives = DriveHierarchy(developmental_stage=0.65)
    engine = HypothesisEngine(drives=drives)

    observation = engine.observe(
        {
            "unknown_territory": 0.82,
            "environmental_stability": 0.34,
            "distant_signal": "flickering",
            "distance_to_signal": 7,
        }
    )
    hypothesis = engine.generate_hypothesis(observation)

    print(engine.summary())
    if hypothesis is not None:
        print()
        print("Generated hypothesis:")
        print(hypothesis.statement)
        print()
        print("Designed test:")
        print(engine.design_test(hypothesis))

        result = TestResult(
            hypothesis_id=hypothesis.id,
            timestamp=time.time(),
            actual_outcome={"distant_signal": "flickering", "distance_to_signal": 6},
            matched_prediction=True,
            confidence_delta=0.0,
            valence=0.4,
        )
        engine.evaluate(hypothesis, result)
        print()
        print("After evaluation:")
        print(engine.summary())

    second_observation = engine.observe(
        {
            "unknown_territory": 0.97,
            "environmental_stability": 0.12,
            "distant_signal": "erratic",
            "distance_to_signal": 2,
            "unmapped_structure": True,
            "acoustic_anomaly": "metallic_resonance",
        }
    )
    second_hypothesis = engine.generate_hypothesis(second_observation)
    if second_hypothesis is not None:
        print()
        print("Generated pending hypothesis:")
        print(second_hypothesis.statement)

    if hypothesis is not None:
        follow_up_result = TestResult(
            hypothesis_id=hypothesis.id,
            timestamp=time.time(),
            actual_outcome={
                "distant_signal": "flickering",
                "unknown_territory": 0.8,
                "environmental_stability": 0.31,
            },
            matched_prediction=True,
            confidence_delta=0.14,
            valence=0.45,
        )
        engine.evaluate(hypothesis, follow_up_result)

    print()
    print("Final summary:")
    print(engine.summary())

"""Itera's internal motivation system.

This module implements a five-tier, Maslow-inspired drive hierarchy.
Each drive tier remains continuously active, while stress in lower tiers
suppresses higher tiers multiplicatively instead of shutting them off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MIN_SIGNAL_VALUE = 0.0
MAX_SIGNAL_VALUE = 1.0
DEFAULT_HISTORY_LIMIT = 12
SUPPRESSION_THRESHOLD = 0.7
SUPPRESSION_STRENGTH = 0.85
MIN_SUPPRESSION_FACTOR = 0.15
BASELINE_RETURN_RATE = 0.2
STAGE_MIN = 0.0
STAGE_MAX = 1.0

TIER_NAMES: dict[int, str] = {
    1: "SURVIVAL",
    2: "SECURITY",
    3: "SOCIAL",
    4: "MASTERY",
    5: "ACTUALIZATION",
}

TIER_SIGNAL_NAMES: dict[int, tuple[str, str]] = {
    1: ("threat_level", "resource_scarcity"),
    2: ("environmental_stability", "unknown_territory"),
    3: ("relationship_depth", "entity_prediction_gap"),
    4: ("competence_growth", "discovery_pull"),
    5: ("purpose_clarity", "identity_coherence"),
}

TIER_DEVELOPMENTAL_WEIGHT_RANGES: dict[int, tuple[float, float]] = {
    1: (1.0, 0.45),
    2: (0.9, 0.55),
    3: (0.65, 0.8),
    4: (0.45, 1.0),
    # Tier 5 intentionally caps at 1.15 - mature Itera at peak actualization may exceed baseline range. Raw signals remain 0.0-1.0. Tier weights do not.
    5: (0.25, 1.15),
}

SIGNAL_BASELINE_RANGES: dict[str, tuple[float, float]] = {
    "threat_level": (0.35, 0.12),
    "resource_scarcity": (0.3, 0.1),
    "environmental_stability": (0.45, 0.3),
    "unknown_territory": (0.4, 0.2),
    "relationship_depth": (0.2, 0.65),
    "entity_prediction_gap": (0.25, 0.45),
    "competence_growth": (0.18, 0.72),
    "discovery_pull": (0.22, 0.82),
    "purpose_clarity": (0.12, 0.78),
    "identity_coherence": (0.15, 0.8),
}

ACTION_ALIGNMENT_KEYS: tuple[str, ...] = (
    "drive_alignment",
    "drive_alignments",
    "drives",
    "tier_weights",
    "signal_weights",
)


def _clamp(value: float, minimum: float = MIN_SIGNAL_VALUE, maximum: float = MAX_SIGNAL_VALUE) -> float:
    """Clamp a value into an inclusive numeric range."""

    return max(minimum, min(maximum, float(value)))


def _interpolate(start: float, end: float, amount: float) -> float:
    """Linearly interpolate between two values."""

    return start + ((end - start) * _clamp(amount, STAGE_MIN, STAGE_MAX))


@dataclass
class DriveSignal:
    """A single drive signal with current value and recent history."""

    name: str
    value: float
    baseline: float
    history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Normalize initial signal state."""

        self.value = _clamp(self.value)
        self.baseline = _clamp(self.baseline)
        self.history = [_clamp(entry) for entry in self.history[-DEFAULT_HISTORY_LIMIT:]]
        if not self.history:
            self.history.append(self.value)

    def update(self, new_value: float) -> None:
        """Update the signal value and append it to rolling history."""

        self.value = _clamp(new_value)
        self.history.append(self.value)
        if len(self.history) > DEFAULT_HISTORY_LIMIT:
            self.history = self.history[-DEFAULT_HISTORY_LIMIT:]

    def trend(self) -> float:
        """Return the recent directional trend for this signal."""

        if len(self.history) < 2:
            return 0.0

        midpoint = max(1, len(self.history) // 2)
        older = self.history[:midpoint]
        newer = self.history[midpoint:]
        older_avg = sum(older) / len(older)
        newer_avg = sum(newer) / len(newer)
        return newer_avg - older_avg


@dataclass
class DriveTier:
    """One tier of the drive hierarchy."""

    tier_id: int
    name: str
    signals: dict[str, DriveSignal]
    developmental_weight: float
    suppression_from_below: float = 1.0

    def raw_signal_strength(self) -> float:
        """Return the average activation of the signals in this tier."""

        if not self.signals:
            return 0.0
        return sum(signal.value for signal in self.signals.values()) / len(self.signals)

    def effective_weight(self) -> float:
        """Return this tier's current weight after lower-tier suppression."""

        return self.raw_signal_strength() * self.developmental_weight * self.suppression_from_below

    def dominant_signal(self) -> DriveSignal:
        """Return the strongest current signal in this tier."""

        return max(self.signals.values(), key=lambda signal: signal.value)

    def summary(self) -> dict[str, Any]:
        """Return a serialization-friendly summary of this tier."""

        return {
            "tier_id": self.tier_id,
            "name": self.name,
            "developmental_weight": round(self.developmental_weight, 4),
            "suppression_from_below": round(self.suppression_from_below, 4),
            "raw_signal_strength": round(self.raw_signal_strength(), 4),
            "effective_weight": round(self.effective_weight(), 4),
            "dominant_signal": self.dominant_signal().name,
            "signals": {
                name: {
                    "value": round(signal.value, 4),
                    "baseline": round(signal.baseline, 4),
                    "trend": round(signal.trend(), 4),
                }
                for name, signal in self.signals.items()
            },
        }


class DriveHierarchy:
    """Itera's complete internal motivation system."""

    def __init__(self, developmental_stage: float = 0.0) -> None:
        """Initialize the five-tier drive hierarchy."""

        self.developmental_stage = _clamp(developmental_stage, STAGE_MIN, STAGE_MAX)
        self.tiers: dict[int, DriveTier] = {}
        self._build_tiers()
        self._recalculate_developmental_weights()
        self._recalculate_suppression()

    def _build_tiers(self) -> None:
        """Create all drive tiers and signals for the current developmental stage."""

        for tier_id, tier_name in TIER_NAMES.items():
            signals: dict[str, DriveSignal] = {}
            for signal_name in TIER_SIGNAL_NAMES[tier_id]:
                baseline = self._baseline_for_signal(signal_name)
                signals[signal_name] = DriveSignal(
                    name=signal_name,
                    value=baseline,
                    baseline=baseline,
                    history=[baseline],
                )
            self.tiers[tier_id] = DriveTier(
                tier_id=tier_id,
                name=tier_name,
                signals=signals,
                developmental_weight=self._developmental_weight_for_tier(tier_id),
            )

    def _baseline_for_signal(self, signal_name: str) -> float:
        """Return the stage-adjusted baseline for a named signal."""

        start, end = SIGNAL_BASELINE_RANGES[signal_name]
        return _interpolate(start, end, self.developmental_stage)

    def _developmental_weight_for_tier(self, tier_id: int) -> float:
        """Return the stage-adjusted developmental weight for a tier."""

        start, end = TIER_DEVELOPMENTAL_WEIGHT_RANGES[tier_id]
        return _interpolate(start, end, self.developmental_stage)

    def _recalculate_developmental_weights(self) -> None:
        """Refresh tier weights and signal baselines after stage changes."""

        for tier_id, tier in self.tiers.items():
            tier.developmental_weight = self._developmental_weight_for_tier(tier_id)
            for signal_name, signal in tier.signals.items():
                signal.baseline = self._baseline_for_signal(signal_name)

    def _suppression_factor_for_tier(self, tier: DriveTier) -> float:
        """Calculate how strongly this tier suppresses tiers above it."""

        peak_signal = max(signal.value for signal in tier.signals.values())
        if peak_signal <= SUPPRESSION_THRESHOLD:
            return 1.0

        excess = (peak_signal - SUPPRESSION_THRESHOLD) / (MAX_SIGNAL_VALUE - SUPPRESSION_THRESHOLD)
        factor = 1.0 - (excess * SUPPRESSION_STRENGTH)
        return max(MIN_SUPPRESSION_FACTOR, factor)

    def _recalculate_suppression(self) -> None:
        """Apply cascading multiplicative suppression from lower tiers upward."""

        cumulative_factor = 1.0
        for tier_id in sorted(self.tiers):
            tier = self.tiers[tier_id]
            tier.suppression_from_below = cumulative_factor
            cumulative_factor *= self._suppression_factor_for_tier(tier)

    def _iter_signals(self) -> list[DriveSignal]:
        """Return a flat list of all signals while preserving tier ownership."""

        return [signal for tier in self.tiers.values() for signal in tier.signals.values()]

    def _extract_signal_updates(self, world_observations: dict[str, Any]) -> dict[str, float]:
        """Extract normalized signal updates from world-agnostic observations."""

        updates: dict[str, float] = {}
        for key, value in world_observations.items():
            if key in TIER_NAMES.values() and isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if nested_key in SIGNAL_BASELINE_RANGES and isinstance(nested_value, (int, float)):
                        updates[nested_key] = _clamp(float(nested_value))
                continue

            if key in SIGNAL_BASELINE_RANGES and isinstance(value, (int, float)):
                updates[key] = _clamp(float(value))

        return updates

    def update(self, world_observations: dict[str, Any]) -> None:
        """Update all drive signals based on current world state."""

        signal_updates = self._extract_signal_updates(world_observations)
        for tier in self.tiers.values():
            for signal_name, signal in tier.signals.items():
                if signal_name in signal_updates:
                    signal.update(signal_updates[signal_name])
                else:
                    eased_value = signal.value + ((signal.baseline - signal.value) * BASELINE_RETURN_RATE)
                    signal.update(eased_value)

        self._recalculate_suppression()

    def get_dominant_drive(self) -> tuple[str, float]:
        """Return the name and weight of the currently dominant drive."""

        dominant_tier = max(self.tiers.values(), key=lambda tier: tier.effective_weight())
        return dominant_tier.name, dominant_tier.effective_weight()

    def get_suppression_map(self) -> dict[int, float]:
        """Return the suppression factor for each tier."""

        return {
            tier_id: round(tier.suppression_from_below, 4)
            for tier_id, tier in self.tiers.items()
        }

    def _candidate_identifier(self, candidate: Any) -> str:
        """Return a stable label for an action candidate."""

        if isinstance(candidate, dict):
            for key in ("name", "id", "label"):
                if key in candidate:
                    return str(candidate[key])
        for attr in ("name", "id", "label"):
            if hasattr(candidate, attr):
                return str(getattr(candidate, attr))
        return str(candidate)

    def _extract_alignment_map(self, candidate: Any) -> dict[str, float]:
        """Extract drive-alignment metadata from a candidate action."""

        if isinstance(candidate, dict):
            for key in ACTION_ALIGNMENT_KEYS:
                value = candidate.get(key)
                if isinstance(value, dict):
                    return {str(name): _clamp(score) for name, score in value.items()}
            return {}

        for key in ACTION_ALIGNMENT_KEYS:
            value = getattr(candidate, key, None)
            if isinstance(value, dict):
                return {str(name): _clamp(score) for name, score in value.items()}
        return {}

    def get_action_weights(self, candidate_actions: list[Any]) -> dict[str, float]:
        """Weight a list of candidate actions by drive alignment."""

        tier_weights = {tier_id: tier.effective_weight() for tier_id, tier in self.tiers.items()}
        total_weight = sum(tier_weights.values())
        if total_weight == 0.0:
            return {self._candidate_identifier(candidate): 0.0 for candidate in candidate_actions}

        action_weights: dict[str, float] = {}
        for candidate in candidate_actions:
            alignment = self._extract_alignment_map(candidate)
            if not alignment:
                action_weights[self._candidate_identifier(candidate)] = 0.0
                continue

            weighted_score = 0.0
            for tier_id, tier in self.tiers.items():
                tier_alignment = alignment.get(str(tier_id), alignment.get(tier.name, 0.0))
                for signal_name in tier.signals:
                    tier_alignment = max(tier_alignment, alignment.get(signal_name, 0.0))
                weighted_score += tier_weights[tier_id] * _clamp(tier_alignment)

            action_weights[self._candidate_identifier(candidate)] = weighted_score / total_weight

        return action_weights

    def advance_developmental_stage(self, delta: float) -> None:
        """Shift baseline weights as Itera matures."""

        self.developmental_stage = _clamp(self.developmental_stage + delta, STAGE_MIN, STAGE_MAX)
        self._recalculate_developmental_weights()
        self._recalculate_suppression()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full hierarchy state for persistence."""

        return {
            "developmental_stage": self.developmental_stage,
            "tiers": {
                str(tier_id): {
                    "tier_id": tier.tier_id,
                    "name": tier.name,
                    "developmental_weight": tier.developmental_weight,
                    "suppression_from_below": tier.suppression_from_below,
                    "signals": {
                        signal_name: {
                            "name": signal.name,
                            "value": signal.value,
                            "baseline": signal.baseline,
                            "history": list(signal.history),
                        }
                        for signal_name, signal in tier.signals.items()
                    },
                }
                for tier_id, tier in self.tiers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DriveHierarchy":
        """Deserialize a hierarchy from saved state."""

        hierarchy = cls(developmental_stage=float(data.get("developmental_stage", 0.0)))
        for tier_id, tier_data in data.get("tiers", {}).items():
            tier = hierarchy.tiers[int(tier_id)]
            tier.developmental_weight = _clamp(
                float(tier_data.get("developmental_weight", tier.developmental_weight)),
                minimum=0.0,
                maximum=max(value[1] for value in TIER_DEVELOPMENTAL_WEIGHT_RANGES.values()),
            )
            for signal_name, signal_data in tier_data.get("signals", {}).items():
                if signal_name not in tier.signals:
                    continue
                signal = tier.signals[signal_name]
                signal.value = _clamp(float(signal_data.get("value", signal.value)))
                signal.baseline = _clamp(float(signal_data.get("baseline", signal.baseline)))
                history = signal_data.get("history", [signal.value])
                signal.history = [_clamp(float(entry)) for entry in history[-DEFAULT_HISTORY_LIMIT:]]
                if not signal.history:
                    signal.history = [signal.value]

        hierarchy._recalculate_suppression()
        return hierarchy

    def summary(self) -> str:
        """Return a concise human-readable snapshot of the hierarchy."""

        dominant_name, dominant_weight = self.get_dominant_drive()
        lines = [
            f"Developmental stage: {self.developmental_stage:.2f}",
            f"Dominant drive: {dominant_name} ({dominant_weight:.3f})",
        ]
        for tier_id in sorted(self.tiers):
            tier = self.tiers[tier_id]
            dominant_signal = tier.dominant_signal()
            lines.append(
                (
                    f"T{tier_id} {tier.name}: raw={tier.raw_signal_strength():.3f} "
                    f"dev={tier.developmental_weight:.3f} "
                    f"suppress={tier.suppression_from_below:.3f} "
                    f"effective={tier.effective_weight():.3f} "
                    f"top={dominant_signal.name}:{dominant_signal.value:.3f}"
                )
            )
        return "\n".join(lines)


if __name__ == "__main__":
    hierarchy = DriveHierarchy(developmental_stage=0.85)
    print("Initial hierarchy")
    print(hierarchy.summary())
    print()

    hierarchy.update({"threat_level": 0.9, "resource_scarcity": 0.82})
    print("After survival spike")
    print(hierarchy.summary())
    print()

    print("Suppression map:", hierarchy.get_suppression_map())
    actualization_weight = hierarchy.tiers[5].effective_weight()
    print(f"Actualization effective weight after spike: {actualization_weight:.3f}")

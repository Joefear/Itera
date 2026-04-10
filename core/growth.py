"""Itera's developmental tracking and capability emergence system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any
from uuid import uuid4

try:
    from core.memory import MemoryGraph
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    from memory import MemoryGraph

CAPABILITY_EMERGENCE_THRESHOLD = 3
DEPTH_SCORE_MAX = 10
DEPTH_CONFIRMATION_WEIGHT = 0.6
DEPTH_HYPOTHESIS_WEIGHT = 0.4
CAPABILITY_CONFIDENCE_STEP = 0.2
CAPABILITY_STAGE_WEIGHT = 0.5
CONFIRMATION_STAGE_WEIGHT = 0.3
DOMAIN_BREADTH_STAGE_WEIGHT = 0.2
MAX_STAGE = 1.0
MIN_STAGE = 0.0
DISCOVERY_MEMORY_VALENCE = 0.3
CAPABILITY_MEMORY_VALENCE = 0.45
DEFAULT_DEEPEST_LIMIT = 3
DEFAULT_SHALLOW_LIMIT = 3
MIN_DEPTH_FOR_ESTABLISHED_CAPABILITY = 0.5


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


def _normalize_domain(domain: str) -> str:
    """Normalize a domain label for storage and comparison."""

    cleaned = str(domain).strip().lower()
    if not cleaned:
        raise ValueError("domain must not be empty")
    return cleaned


def _normalize_tags(tags: list[str]) -> list[str]:
    """Normalize and de-duplicate a tag list while preserving order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = str(tag).strip().lower()
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


@dataclass
class Capability:
    """A capability Itera has developed through experience."""

    id: str
    name: str
    description: str
    emerged_at: float
    evidence_count: int
    confidence: float
    domain: str
    tags: list[str]

    def __post_init__(self) -> None:
        """Normalize capability fields for persistence and comparison."""

        self.name = str(self.name)
        self.description = str(self.description)
        self.emerged_at = float(self.emerged_at)
        self.evidence_count = max(0, int(self.evidence_count))
        self.confidence = _clamp(self.confidence, MIN_STAGE, MAX_STAGE)
        self.domain = _normalize_domain(self.domain)
        self.tags = _normalize_tags(self.tags)


@dataclass
class DomainDepth:
    """How deeply Itera has explored a domain of knowledge."""

    domain: str
    exploration_count: int
    hypothesis_count: int
    confirmation_count: int
    depth_score: float
    last_engaged: float

    def __post_init__(self) -> None:
        """Normalize counters and recalculate a bounded depth score."""

        self.domain = _normalize_domain(self.domain)
        self.exploration_count = max(0, int(self.exploration_count))
        self.hypothesis_count = max(0, int(self.hypothesis_count))
        self.confirmation_count = max(0, int(self.confirmation_count))
        self.last_engaged = float(self.last_engaged)
        self.depth_score = self.calculate_depth_score()

    def calculate_depth_score(self) -> float:
        """Calculate the normalized exploration depth for this domain."""

        weighted_total = (
            (self.confirmation_count * DEPTH_CONFIRMATION_WEIGHT)
            + (self.hypothesis_count * DEPTH_HYPOTHESIS_WEIGHT)
        )
        return _clamp(weighted_total / float(DEPTH_SCORE_MAX), MIN_STAGE, MAX_STAGE)

    def refresh(self, timestamp: float | None = None) -> None:
        """Recalculate depth and refresh the last-engaged timestamp."""

        self.depth_score = self.calculate_depth_score()
        if timestamp is not None:
            self.last_engaged = float(timestamp)


class GrowthTracker:
    """Tracks Itera's developmental history and capability emergence."""

    def __init__(self, memory: MemoryGraph) -> None:
        """Initialize a growth tracker bound to an existing memory graph."""

        self.memory = memory
        self.capabilities: dict[str, Capability] = {}
        self.domain_depths: dict[str, DomainDepth] = {}
        self.confirmation_history: list[dict[str, Any]] = []

    def record_experience(self, domain: str, outcome: dict[str, Any], valence: float) -> None:
        """Record that Itera had an experience in a domain."""

        normalized_domain = _normalize_domain(domain)
        timestamp = float(outcome.get("timestamp", time.time()))
        depth = self._ensure_domain(normalized_domain, timestamp)
        depth.exploration_count += 1
        depth.refresh(timestamp)
        self.check_capability_emergence(normalized_domain)

    def record_hypothesis(self, domain: str) -> None:
        """Increment hypothesis count for a domain."""

        normalized_domain = _normalize_domain(domain)
        timestamp = time.time()
        depth = self._ensure_domain(normalized_domain, timestamp)
        depth.hypothesis_count += 1
        depth.refresh(timestamp)

    def record_confirmation(self, domain: str, hypothesis_statement: str) -> None:
        """Record a confirmed hypothesis in a domain."""

        normalized_domain = _normalize_domain(domain)
        timestamp = time.time()
        depth = self._ensure_domain(normalized_domain, timestamp)
        depth.confirmation_count += 1
        depth.refresh(timestamp)
        self.confirmation_history.append(
            {
                "domain": normalized_domain,
                "hypothesis_statement": str(hypothesis_statement),
                "timestamp": timestamp,
            }
        )
        self.check_capability_emergence(normalized_domain)

    def check_capability_emergence(self, domain: str) -> list[Capability]:
        """Check whether a domain has accumulated enough evidence for a new capability."""

        normalized_domain = _normalize_domain(domain)
        depth = self.domain_depths.get(normalized_domain)
        if depth is None:
            return []

        target_capability_count = depth.confirmation_count // CAPABILITY_EMERGENCE_THRESHOLD
        existing = [capability for capability in self.capabilities.values() if capability.domain == normalized_domain]
        if target_capability_count <= len(existing):
            return []

        new_capabilities: list[Capability] = []
        for index in range(len(existing) + 1, target_capability_count + 1):
            evidence_count = depth.confirmation_count
            confidence = _clamp(index * CAPABILITY_CONFIDENCE_STEP, MIN_STAGE, MAX_STAGE)
            capability = Capability(
                id=str(uuid4()),
                name=f"{normalized_domain.title()} capability {index}",
                description=(
                    f"Emergent {normalized_domain} capability supported by "
                    f"{evidence_count} confirmed experiences."
                ),
                emerged_at=time.time(),
                evidence_count=evidence_count,
                confidence=confidence,
                domain=normalized_domain,
                tags=[normalized_domain, "capability", f"tier-{index}"],
            )
            self.capabilities[capability.id] = capability
            self.memory.create_node(
                node_type="concept",
                content={
                    "capability_id": capability.id,
                    "name": capability.name,
                    "description": capability.description,
                    "domain": capability.domain,
                    "confidence": capability.confidence,
                },
                valence=CAPABILITY_MEMORY_VALENCE,
                tags=capability.tags,
            )
            new_capabilities.append(capability)

        return new_capabilities

    def record_discovery(self, name: str, description: str, domain: str) -> "MemoryNode":
        """Record a named discovery and store it in the memory graph."""

        normalized_domain = _normalize_domain(domain)
        timestamp = time.time()
        depth = self._ensure_domain(normalized_domain, timestamp)
        depth.exploration_count += 1
        depth.refresh(timestamp)
        return self.memory.create_node(
            node_type="discovery",
            content={
                "name": str(name),
                "description": str(description),
                "domain": normalized_domain,
            },
            valence=DISCOVERY_MEMORY_VALENCE,
            tags=[normalized_domain, "discovery", str(name).strip().lower()],
        )

    def get_capabilities(self, domain: str | None = None) -> list[Capability]:
        """Return all capabilities, optionally filtered by domain."""

        if domain is None:
            return sorted(self.capabilities.values(), key=lambda capability: capability.emerged_at)
        normalized_domain = _normalize_domain(domain)
        return [
            capability
            for capability in sorted(self.capabilities.values(), key=lambda capability: capability.emerged_at)
            if capability.domain == normalized_domain
        ]

    def get_domain_depth(self, domain: str) -> DomainDepth | None:
        """Return the exploration-depth record for a specific domain."""

        return self.domain_depths.get(_normalize_domain(domain))

    def get_deepest_domains(self, limit: int = DEFAULT_DEEPEST_LIMIT) -> list[DomainDepth]:
        """Return the top N domains by depth score."""

        return sorted(
            self.domain_depths.values(),
            key=lambda depth: (depth.depth_score, depth.confirmation_count, depth.last_engaged),
            reverse=True,
        )[: max(0, int(limit))]

    def get_shallow_domains(self, limit: int = DEFAULT_SHALLOW_LIMIT) -> list[DomainDepth]:
        """Return domains that were touched but remain underexplored."""

        shallow = [depth for depth in self.domain_depths.values() if depth.exploration_count > 0]
        shallow.sort(key=lambda depth: (depth.depth_score, -depth.last_engaged))
        return shallow[: max(0, int(limit))]

    def overall_stage(self) -> float:
        """Calculate an overall developmental stage from growth evidence."""

        total_confirmations = sum(depth.confirmation_count for depth in self.domain_depths.values())
        confirmation_score = _clamp(total_confirmations / float(DEPTH_SCORE_MAX), MIN_STAGE, MAX_STAGE)
        capability_score = _clamp(
            len(self.capabilities) / float(max(1, CAPABILITY_EMERGENCE_THRESHOLD)),
            MIN_STAGE,
            MAX_STAGE,
        )
        breadth_score = _clamp(
            len([depth for depth in self.domain_depths.values() if depth.depth_score >= MIN_DEPTH_FOR_ESTABLISHED_CAPABILITY])
            / float(max(1, CAPABILITY_EMERGENCE_THRESHOLD)),
            MIN_STAGE,
            MAX_STAGE,
        )
        return _clamp(
            (capability_score * CAPABILITY_STAGE_WEIGHT)
            + (confirmation_score * CONFIRMATION_STAGE_WEIGHT)
            + (breadth_score * DOMAIN_BREADTH_STAGE_WEIGHT),
            MIN_STAGE,
            MAX_STAGE,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full growth-tracker state."""

        return {
            "capabilities": {
                capability_id: asdict(capability)
                for capability_id, capability in self.capabilities.items()
            },
            "domain_depths": {
                domain: asdict(depth)
                for domain, depth in self.domain_depths.items()
            },
            "confirmation_history": list(self.confirmation_history),
        }

    @staticmethod
    def _restore_capability(payload: dict[str, Any], fallback_id: str) -> "Capability":
        """Restore a Capability from persisted state without signaling emergence."""

        return Capability(
            id=str(payload.get("id", fallback_id)),
            name=str(payload["name"]),
            description=str(payload["description"]),
            emerged_at=float(payload["emerged_at"]),
            evidence_count=int(payload["evidence_count"]),
            confidence=float(payload["confidence"]),
            domain=str(payload["domain"]),
            tags=[str(tag) for tag in payload.get("tags", [])],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], memory: MemoryGraph) -> "GrowthTracker":
        """Deserialize a growth tracker from persisted state."""

        tracker = cls(memory=memory)
        for capability_id, payload in data.get("capabilities", {}).items():
            tracker.capabilities[capability_id] = GrowthTracker._restore_capability(payload, capability_id)
        for domain, payload in data.get("domain_depths", {}).items():
            tracker.domain_depths[domain] = DomainDepth(
                domain=str(payload.get("domain", domain)),
                exploration_count=int(payload.get("exploration_count", 0)),
                hypothesis_count=int(payload.get("hypothesis_count", 0)),
                confirmation_count=int(payload.get("confirmation_count", 0)),
                depth_score=float(payload.get("depth_score", 0.0)),
                last_engaged=float(payload.get("last_engaged", time.time())),
            )
        tracker.confirmation_history = [
            {
                "domain": str(entry.get("domain", "")),
                "hypothesis_statement": str(entry.get("hypothesis_statement", "")),
                "timestamp": float(entry.get("timestamp", time.time())),
            }
            for entry in data.get("confirmation_history", [])
        ]
        return tracker

    def summary(self) -> str:
        """Return a human-readable snapshot of growth state."""

        deepest = ", ".join(
            f"{depth.domain}:{depth.depth_score:.2f}" for depth in self.get_deepest_domains()
        ) or "none"
        return "\n".join(
            [
                f"Capabilities emerged: {len(self.capabilities)}",
                f"Tracked domains: {len(self.domain_depths)}",
                f"Overall stage: {self.overall_stage():.3f}",
                f"Deepest domains: {deepest}",
            ]
        )

    def _ensure_domain(self, domain: str, timestamp: float) -> DomainDepth:
        """Return an existing domain-depth record or create a new one."""

        normalized_domain = _normalize_domain(domain)
        if normalized_domain not in self.domain_depths:
            self.domain_depths[normalized_domain] = DomainDepth(
                domain=normalized_domain,
                exploration_count=0,
                hypothesis_count=0,
                confirmation_count=0,
                depth_score=0.0,
                last_engaged=float(timestamp),
            )
        return self.domain_depths[normalized_domain]


if __name__ == "__main__":
    memory = MemoryGraph()
    tracker = GrowthTracker(memory=memory)

    experiences = [
        ("physical", {"event": "moved stone", "result": "stone slid"}, 0.1),
        ("physical", {"event": "stacked stones", "result": "tower held"}, 0.25),
        ("social", {"event": "signaled entity", "result": "entity approached"}, 0.2),
        ("cognitive", {"event": "tracked pattern", "result": "pattern repeated"}, 0.3),
        ("survival", {"event": "avoided heat vent", "result": "damage prevented"}, 0.35),
    ]
    for domain, outcome, valence in experiences:
        tracker.record_experience(domain, outcome, valence)

    tracker.record_hypothesis("physical")
    tracker.record_hypothesis("physical")
    tracker.record_hypothesis("social")
    tracker.record_hypothesis("cognitive")

    tracker.record_confirmation("physical", "Stone stacks can remain stable when balanced.")
    tracker.record_confirmation("physical", "Sliding heavy stones predicts friction boundaries.")
    new_capabilities = tracker.record_confirmation("physical", "Repeated manipulation reveals structural affordances.")
    if new_capabilities is None:
        new_capabilities = []
    tracker.record_confirmation("social", "Nearby entities respond to simple repeated signals.")

    discovery = tracker.record_discovery(
        name="stone balance principle",
        description="Balanced stone placement reliably increases structural stability.",
        domain="physical",
    )

    print("New capabilities:", tracker.get_capabilities("physical"))
    print("Discovery node:", discovery)
    print("Deepest domains:", tracker.get_deepest_domains())
    print("Shallow domains:", tracker.get_shallow_domains())
    print("Overall stage:", round(tracker.overall_stage(), 4))
    print(tracker.summary())

    payload = tracker.to_dict()
    restored = GrowthTracker.from_dict(payload, memory=memory)
    print("Capability count before round-trip:", len(tracker.get_capabilities()))
    print("Capability count after round-trip:", len(restored.get_capabilities()))

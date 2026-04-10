"""Itera's persistent memory and living knowledge graph.

This module implements a world-agnostic memory system where experiences,
entities, discoveries, and concepts are stored as graph nodes connected by
typed relationships. Relevance decays over time but never reaches zero, so
early experience continues shaping later reasoning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4

MIN_VALENCE = -1.0
MAX_VALENCE = 1.0
MIN_EDGE_WEIGHT = 0.0
MAX_EDGE_WEIGHT = 1.0
MIN_RELEVANCE = 0.05
MAX_RELEVANCE = 1.0
DEFAULT_RELEVANCE = 1.0
DEFAULT_EDGE_WEIGHT = 0.5
DEFAULT_DECAY_RATE = 0.0025
DEFAULT_RETRIEVAL_LIMIT = 5
DEFAULT_MEMORY_FILENAME = "memory_graph.json"
DEFAULT_DATA_DIRECTORY = "data"
SECONDS_PER_HOUR = 3600.0
ACCESS_REINFORCEMENT = 0.04
VALENCE_RETRIEVAL_WEIGHT = 0.25
RELEVANCE_RETRIEVAL_WEIGHT = 0.45
TAG_MATCH_RETRIEVAL_WEIGHT = 0.2
ACCESS_RETRIEVAL_WEIGHT = 0.1
MAX_ACCESS_SCORE = 10.0

VALID_NODE_TYPES: frozenset[str] = frozenset({"experience", "entity", "concept", "discovery"})
VALID_EDGE_TYPES: frozenset[str] = frozenset(
    {"caused", "preceded", "relates_to", "contradicts", "confirms", "involves"}
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


def _normalize_tag(tag: str) -> str:
    """Normalize tags for stable indexing and lookup."""

    return str(tag).strip().lower()


def _normalize_tags(tags: list[str]) -> list[str]:
    """Normalize, de-duplicate, and preserve tag order."""

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        cleaned = _normalize_tag(tag)
        if not cleaned or cleaned in seen:
            continue
        normalized.append(cleaned)
        seen.add(cleaned)
    return normalized


@dataclass
class MemoryNode:
    """A single node in Itera's knowledge graph."""

    id: str
    created_at: float
    last_accessed: float
    node_type: str
    content: dict[str, Any]
    valence: float
    relevance: float
    access_count: int
    tags: list[str]

    def __post_init__(self) -> None:
        """Normalize node fields into valid memory ranges."""

        self.node_type = str(self.node_type)
        if self.node_type not in VALID_NODE_TYPES:
            raise ValueError(f"Unsupported memory node_type: {self.node_type}")
        self.created_at = float(self.created_at)
        self.last_accessed = float(self.last_accessed)
        self.content = dict(self.content)
        self.valence = _clamp(self.valence, MIN_VALENCE, MAX_VALENCE)
        self.relevance = _clamp(self.relevance, MIN_RELEVANCE, MAX_RELEVANCE)
        self.access_count = max(0, int(self.access_count))
        self.tags = _normalize_tags(self.tags)


@dataclass
class MemoryEdge:
    """A relationship between two memory nodes."""

    id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    valence: float
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize edge fields into valid memory ranges."""

        self.source_id = str(self.source_id)
        self.target_id = str(self.target_id)
        self.edge_type = str(self.edge_type)
        if self.edge_type not in VALID_EDGE_TYPES:
            raise ValueError(f"Unsupported memory edge_type: {self.edge_type}")
        self.weight = _clamp(self.weight, MIN_EDGE_WEIGHT, MAX_EDGE_WEIGHT)
        self.valence = _clamp(self.valence, MIN_VALENCE, MAX_VALENCE)
        self.created_at = float(self.created_at)
        self.metadata = dict(self.metadata)


class MemoryGraph:
    """Persistent knowledge graph that stores and retrieves identity-shaping memory."""

    def __init__(
        self,
        storage_path: str | Path | None = None,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ) -> None:
        """Initialize an empty memory graph with optional persistent storage."""

        self.storage_path = Path(storage_path) if storage_path is not None else Path(DEFAULT_DATA_DIRECTORY) / DEFAULT_MEMORY_FILENAME
        self.decay_rate = max(0.0, float(decay_rate))
        self.nodes: dict[str, MemoryNode] = {}
        self.edges: dict[str, MemoryEdge] = {}
        self._adjacency: dict[str, set[str]] = {}

    def create_node(
        self,
        node_type: str,
        content: dict[str, Any],
        valence: float = 0.0,
        relevance: float = DEFAULT_RELEVANCE,
        tags: list[str] | None = None,
        created_at: float | None = None,
    ) -> MemoryNode:
        """Create, store, and return a new memory node."""

        timestamp = time.time() if created_at is None else float(created_at)
        node = MemoryNode(
            id=str(uuid4()),
            created_at=timestamp,
            last_accessed=timestamp,
            node_type=node_type,
            content=dict(content),
            valence=valence,
            relevance=relevance,
            access_count=0,
            tags=[] if tags is None else list(tags),
        )
        self.add_node(node)
        return node

    def add_node(self, node: MemoryNode) -> None:
        """Store an existing memory node in the graph."""

        self.nodes[node.id] = node
        self._adjacency.setdefault(node.id, set())

    def connect(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = DEFAULT_EDGE_WEIGHT,
        valence: float = 0.0,
        metadata: dict[str, Any] | None = None,
        created_at: float | None = None,
    ) -> MemoryEdge:
        """Create a typed relationship between two stored memory nodes."""

        if source_id not in self.nodes:
            raise KeyError(f"Unknown source node: {source_id}")
        if target_id not in self.nodes:
            raise KeyError(f"Unknown target node: {target_id}")

        edge = MemoryEdge(
            id=str(uuid4()),
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            valence=valence,
            created_at=time.time() if created_at is None else float(created_at),
            metadata={} if metadata is None else dict(metadata),
        )
        self.add_edge(edge)
        return edge

    def add_edge(self, edge: MemoryEdge) -> None:
        """Store an existing memory edge and update graph adjacency."""

        if edge.source_id not in self.nodes:
            raise KeyError(f"Unknown source node: {edge.source_id}")
        if edge.target_id not in self.nodes:
            raise KeyError(f"Unknown target node: {edge.target_id}")
        self.edges[edge.id] = edge
        self._adjacency.setdefault(edge.source_id, set()).add(edge.id)
        self._adjacency.setdefault(edge.target_id, set()).add(edge.id)

    def get_node(self, node_id: str, reinforce: bool = True) -> MemoryNode:
        """Return a node by ID and optionally reinforce access."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown memory node: {node_id}")
        node = self.nodes[node_id]
        if reinforce:
            self._record_access(node)
        return node

    def get_edges_for_node(self, node_id: str) -> list[MemoryEdge]:
        """Return all edges touching a given node."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown memory node: {node_id}")
        return [self.edges[edge_id] for edge_id in sorted(self._adjacency.get(node_id, set()))]

    def related_nodes(
        self,
        node_id: str,
        edge_types: set[str] | None = None,
        limit: int | None = None,
    ) -> list[MemoryNode]:
        """Return nodes directly connected to a given node, ranked by edge weight."""

        if node_id not in self.nodes:
            raise KeyError(f"Unknown memory node: {node_id}")

        selected_types = VALID_EDGE_TYPES if edge_types is None else {str(edge_type) for edge_type in edge_types}
        ranked: list[tuple[float, MemoryNode]] = []
        for edge in self.get_edges_for_node(node_id):
            if edge.edge_type not in selected_types:
                continue
            other_id = edge.target_id if edge.source_id == node_id else edge.source_id
            ranked.append((edge.weight, self.nodes[other_id]))

        ranked.sort(key=lambda item: item[0], reverse=True)
        nodes = [node for _, node in ranked]
        if limit is None:
            return nodes
        return nodes[: max(0, int(limit))]

    def retrieve(
        self,
        query: str = "",
        tags: list[str] | None = None,
        node_types: set[str] | None = None,
        valence_bias: float | None = None,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ) -> list[MemoryNode]:
        """Retrieve nodes using content match, tags, relevance, access, and valence."""

        normalized_query = query.strip().lower()
        requested_tags = _normalize_tags([] if tags is None else list(tags))
        allowed_types = VALID_NODE_TYPES if node_types is None else {str(node_type) for node_type in node_types}
        scored: list[tuple[float, MemoryNode]] = []

        self.apply_decay()

        for node in self.nodes.values():
            if node.node_type not in allowed_types:
                continue

            tag_overlap = len(set(node.tags) & set(requested_tags))
            if requested_tags and tag_overlap == 0 and not normalized_query:
                continue

            content_blob = json.dumps(node.content, sort_keys=True).lower()
            query_match = 0.0
            if normalized_query:
                query_match = 1.0 if normalized_query in content_blob else 0.0
                if query_match == 0.0 and normalized_query not in " ".join(node.tags):
                    continue

            tag_score = 0.0
            if requested_tags:
                tag_score = tag_overlap / len(requested_tags)

            valence_score = abs(node.valence)
            if valence_bias is not None:
                bias = _clamp(valence_bias, MIN_VALENCE, MAX_VALENCE)
                valence_score = 1.0 - min(MAX_RELEVANCE, abs(node.valence - bias) / (MAX_VALENCE - MIN_VALENCE))

            access_score = min(node.access_count, MAX_ACCESS_SCORE) / MAX_ACCESS_SCORE
            query_score = max(query_match, tag_score)
            score = (
                (node.relevance * RELEVANCE_RETRIEVAL_WEIGHT)
                + (valence_score * VALENCE_RETRIEVAL_WEIGHT)
                + (tag_score * TAG_MATCH_RETRIEVAL_WEIGHT)
                + (access_score * ACCESS_RETRIEVAL_WEIGHT)
                + query_score
            )
            scored.append((score, node))

        scored.sort(key=lambda item: (item[0], item[1].last_accessed), reverse=True)
        retrieved = [node for _, node in scored[: max(0, int(limit))]]
        for node in retrieved:
            self._record_access(node)
        return retrieved

    def reinforce(self, node_id: str, amount: float = ACCESS_REINFORCEMENT) -> MemoryNode:
        """Increase a node's relevance in response to repeated use."""

        node = self.get_node(node_id, reinforce=False)
        node.relevance = _clamp(node.relevance + float(amount), MIN_RELEVANCE, MAX_RELEVANCE)
        self._record_access(node)
        return node

    def absorb_outcome(
        self,
        outcome: dict[str, Any],
        valence: float,
        tags: list[str] | None = None,
    ) -> MemoryNode:
        """Store a world outcome as an experience memory node."""

        outcome_tags = [] if tags is None else list(tags)
        hypothesis_id = outcome.get("hypothesis_id")
        if hypothesis_id is not None:
            outcome_tags.append(str(hypothesis_id))

        node = self.create_node(
            node_type="experience",
            content=dict(outcome),
            valence=valence,
            tags=outcome_tags,
        )

        if hypothesis_id is None:
            return node

        hypothesis_tag = _normalize_tag(str(hypothesis_id))
        for existing in self.nodes.values():
            if existing.id == node.id or hypothesis_tag not in existing.tags:
                continue
            self.connect(
                source_id=node.id,
                target_id=existing.id,
                edge_type="relates_to",
                weight=DEFAULT_EDGE_WEIGHT,
                valence=valence,
                metadata={"linked_by": "hypothesis_id"},
            )
        return node

    def update_valence(self, node_id: str, new_valence: float, blend: float = 0.5) -> MemoryNode:
        """Blend a new emotional assessment into an existing memory node."""

        node = self.get_node(node_id, reinforce=False)
        amount = _clamp(blend, 0.0, 1.0)
        node.valence = _clamp(
            (node.valence * (1.0 - amount)) + (_clamp(new_valence, MIN_VALENCE, MAX_VALENCE) * amount),
            MIN_VALENCE,
            MAX_VALENCE,
        )
        node.last_accessed = time.time()
        return node

    def apply_decay(self, now: float | None = None) -> None:
        """Decay relevance over elapsed time without ever fully erasing memory."""

        current_time = time.time() if now is None else float(now)
        for node in self.nodes.values():
            elapsed_seconds = max(0.0, current_time - node.last_accessed)
            elapsed_hours = elapsed_seconds / SECONDS_PER_HOUR
            if elapsed_hours <= 0.0:
                continue
            decay_amount = elapsed_hours * self.decay_rate
            node.relevance = _clamp(node.relevance - decay_amount, MIN_RELEVANCE, MAX_RELEVANCE)

    def faded_isolates(self, maximum_relevance: float = MIN_RELEVANCE) -> list[str]:
        """Return weak disconnected memories without erasing them from the graph."""

        threshold = _clamp(maximum_relevance, MIN_RELEVANCE, MAX_RELEVANCE)
        return [
            node_id
            for node_id, node in self.nodes.items()
            if not self._adjacency.get(node_id) and node.relevance <= threshold
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete memory graph."""

        return {
            "decay_rate": self.decay_rate,
            "nodes": {node_id: asdict(node) for node_id, node in self.nodes.items()},
            "edges": {edge_id: asdict(edge) for edge_id, edge in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], storage_path: str | Path | None = None) -> "MemoryGraph":
        """Deserialize a memory graph from a dictionary payload."""

        graph = cls(
            storage_path=storage_path,
            decay_rate=float(data.get("decay_rate", DEFAULT_DECAY_RATE)),
        )
        for node_id, payload in data.get("nodes", {}).items():
            graph.add_node(
                MemoryNode(
                    id=str(payload.get("id", node_id)),
                    created_at=float(payload["created_at"]),
                    last_accessed=float(payload["last_accessed"]),
                    node_type=str(payload["node_type"]),
                    content=dict(payload.get("content", {})),
                    valence=float(payload.get("valence", 0.0)),
                    relevance=float(payload.get("relevance", DEFAULT_RELEVANCE)),
                    access_count=int(payload.get("access_count", 0)),
                    tags=[str(tag) for tag in payload.get("tags", [])],
                )
            )

        for edge_id, payload in data.get("edges", {}).items():
            graph.add_edge(
                MemoryEdge(
                    id=str(payload.get("id", edge_id)),
                    source_id=str(payload["source_id"]),
                    target_id=str(payload["target_id"]),
                    edge_type=str(payload["edge_type"]),
                    weight=float(payload.get("weight", DEFAULT_EDGE_WEIGHT)),
                    valence=float(payload.get("valence", 0.0)),
                    created_at=float(payload.get("created_at", time.time())),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return graph

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the memory graph to JSON."""

        target_path = Path(path) if path is not None else self.storage_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return target_path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MemoryGraph":
        """Load a persisted memory graph, or return an empty graph if none exists."""

        source_path = Path(path) if path is not None else Path(DEFAULT_DATA_DIRECTORY) / DEFAULT_MEMORY_FILENAME
        if not source_path.exists():
            return cls(storage_path=source_path)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        return cls.from_dict(payload, storage_path=source_path)

    def summary(self) -> dict[str, Any]:
        """Return a compact snapshot of the current graph state."""

        type_counts = {
            node_type: sum(1 for node in self.nodes.values() if node.node_type == node_type)
            for node_type in sorted(VALID_NODE_TYPES)
        }
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "type_counts": type_counts,
            "average_relevance": 0.0 if not self.nodes else round(sum(node.relevance for node in self.nodes.values()) / len(self.nodes), 4),
            "average_valence": 0.0 if not self.nodes else round(sum(node.valence for node in self.nodes.values()) / len(self.nodes), 4),
        }

    def _record_access(self, node: MemoryNode) -> None:
        """Update access metadata and lightly reinforce a retrieved node."""

        node.last_accessed = time.time()
        node.access_count += 1
        node.relevance = _clamp(node.relevance + ACCESS_REINFORCEMENT, MIN_RELEVANCE, MAX_RELEVANCE)


if __name__ == "__main__":
    memory = MemoryGraph()

    experience = memory.create_node(
        node_type="experience",
        content={"event": "tested a glowing stone", "result": "stone emitted warmth"},
        valence=0.25,
        tags=["stone", "experiment", "warmth"],
    )
    discovery = memory.create_node(
        node_type="discovery",
        content={"insight": "dark stones may predict heat resistance"},
        valence=0.45,
        tags=["stone", "discovery", "heat"],
    )
    entity = memory.create_node(
        node_type="entity",
        content={"name": "river spirit", "trust": 0.6},
        valence=0.1,
        tags=["entity", "social", "river"],
    )
    concept = memory.create_node(
        node_type="concept",
        content={"idea": "warm stones may indicate hidden energy flows"},
        valence=0.2,
        tags=["concept", "stone", "theory"],
    )

    memory.connect(experience.id, discovery.id, edge_type="caused", weight=0.85, valence=0.35)
    memory.connect(discovery.id, entity.id, edge_type="relates_to", weight=0.4, valence=0.05)
    memory.connect(discovery.id, concept.id, edge_type="confirms", weight=0.7, valence=0.2)

    recalled = memory.retrieve(query="stone", tags=["discovery"], valence_bias=0.4, limit=2)
    before_decay = discovery.relevance
    memory.apply_decay(now=discovery.last_accessed + (SECONDS_PER_HOUR * 48.0))
    after_decay = discovery.relevance
    absorbed = memory.absorb_outcome(
        {"action": "examined stone", "result": "warmth detected"},
        valence=0.3,
    )
    saved_path = memory.save()

    print("Memory summary:", memory.summary())
    print("Retrieved node ids:", [node.id for node in recalled])
    print("Connected to discovery:", [node.id for node in memory.related_nodes(discovery.id)])
    print("Discovery relevance before decay:", round(before_decay, 4))
    print("Discovery relevance after decay:", round(after_decay, 4))
    print("Absorbed outcome node:", absorbed)
    print("Saved to:", saved_path)

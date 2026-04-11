"""Fast simulation loop for running Itera without external prompting."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Any

try:
    from adapters.text_sim.adapter import TextSimAdapter
    from core.growth import GrowthTracker
    from core.identity import Identity
    from interface.environment import Action
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from adapters.text_sim.adapter import TextSimAdapter  # type: ignore[no-redef]
    from core.growth import GrowthTracker  # type: ignore[no-redef]
    from core.identity import Identity  # type: ignore[no-redef]
    from interface.environment import Action  # type: ignore[no-redef]

DEFAULT_MAX_CYCLES = 1000
DEFAULT_MAX_DURATION_SECONDS = 1800.0
DEFAULT_IDENTITY_NAME = "Itera"
DEFAULT_DATA_DIR = "data/identity"
DEFAULT_LOG_INTERVAL = 10
DEFAULT_CHECKPOINT_INTERVAL = 100
DEFAULT_VERBOSE = True
CONSECUTIVE_FAILURE_THRESHOLD = 3
CHECKPOINT_FULL_SAVE = False
DEFAULT_EXAMINE_FALLBACK_ACTION = "examine"
DEFAULT_MOVE_FALLBACK_ACTION = "move"
DEFAULT_MOVE_DIRECTION = "north"
DEFAULT_FALLBACK_DRIVE_SOURCE = "MASTERY"
DEFAULT_FALLBACK_RADIUS = 1
ENTITY_PREDICTION_GAP_NORMALIZER = 5.0
TOP_MEMORY_LIMIT = 3
DEFAULT_DEMO_MAX_CYCLES = 200

DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "physical": ("move", "push", "heat", "cool", "combine", "rock", "water", "fire", "stone"),
    "social": ("creature", "entity", "social"),
    "cognitive": ("examine", "observe", "pattern", "hypothesis"),
    "creative": ("combine", "build", "arrange"),
    "survival": ("fire", "heat", "threat", "blocked", "resource"),
}


@dataclass
class SimConfig:
    """Configuration for a fast simulation run."""

    max_cycles: int = DEFAULT_MAX_CYCLES
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS
    world_seed: int | None = None
    identity_name: str = DEFAULT_IDENTITY_NAME
    data_dir: str = DEFAULT_DATA_DIR
    log_interval: int = DEFAULT_LOG_INTERVAL
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL
    verbose: bool = DEFAULT_VERBOSE


@dataclass
class SimResult:
    """Summary of a completed simulation run."""

    total_cycles: int
    total_duration_seconds: float
    final_developmental_stage: float
    hypotheses_generated: int
    hypotheses_confirmed: int
    capabilities_emerged: int
    memory_nodes: int
    termination_reason: str
    entities_modeled: int = 0
    emerged_capability_names: list[str] = field(default_factory=list)


class FastSim:
    """The core cognitive loop for Itera in fast simulation mode."""

    def __init__(self, config: SimConfig | None = None) -> None:
        """Initialize the simulation loop with config and empty runtime state."""

        self.config = SimConfig() if config is None else config
        self.identity = Identity(name=self.config.identity_name, data_dir=self.config.data_dir)
        self.adapter = TextSimAdapter(seed=self.config.world_seed)
        self.growth = GrowthTracker(memory=self.identity.memory)
        self._started_at = 0.0
        self._last_observation: Any | None = None
        self._last_action: Action | None = None
        self._last_tick_count = 0
        self._last_hypothesis_total = 0
        self._last_confirmed_total = 0
        self._progress_snapshot_paths: list[Path] = []

    def setup(self) -> None:
        """Initialize identity and adapter state and log the starting snapshot."""

        wake_summary = self.identity.wake()
        reset_observation = self.adapter.reset()
        self._last_observation = reset_observation
        self._last_tick_count = self._extract_tick_count({"resulting_state": self.adapter.world.get_state()})
        self._last_hypothesis_total = self.identity.developmental.total_hypotheses
        self._last_confirmed_total = self.identity.developmental.total_confirmed
        self.growth = GrowthTracker(memory=self.identity.memory)
        if self.config.verbose:
            print("FastSim setup")
            print("Identity:", wake_summary)
            print("Initial observation:", reset_observation)

    def run(self) -> SimResult:
        """Run the full simulation loop until a stop condition is reached."""

        self._started_at = time.time()
        consecutive_failures = 0
        termination_reason = "max_cycles"
        completed_cycles = 0

        for cycle in range(1, self.config.max_cycles + 1):
            if time.time() - self._started_at >= self.config.max_duration_seconds:
                termination_reason = "max_duration"
                break

            try:
                observation = self.adapter.perceive()
                self._last_observation = observation
                self.identity.perceive(self._translate_observation(observation))
                self._record_growth_hypothesis_delta()

                action_dict = self.identity.decide()
                action = self._translate_action(action_dict)
                self._last_action = action
                self.adapter.act(action)

                outcome = self.adapter.get_outcome()
                outcome_dict, valence = self._translate_outcome(outcome)
                self.identity.absorb(outcome_dict, valence)
                self._record_growth_outcome(outcome_dict, action, valence)
                self._maybe_advance_world(outcome_dict)

                completed_cycles = cycle
                consecutive_failures = 0

                if self.config.verbose and cycle % max(1, self.config.log_interval) == 0:
                    self._log_cycle(cycle)
                if cycle % max(1, self.config.checkpoint_interval) == 0:
                    self._checkpoint(cycle)
            except Exception as exc:
                consecutive_failures += 1
                if self.config.verbose:
                    print(f"Cycle {cycle} error: {exc}")
                if consecutive_failures >= CONSECUTIVE_FAILURE_THRESHOLD:
                    termination_reason = "error"
                    break
                continue
        else:
            termination_reason = "max_cycles"

        self.identity.sleep()
        self._cleanup_progress_snapshots()
        capability_summaries = self.growth.get_emerged_capabilities_summary()
        result = SimResult(
            total_cycles=completed_cycles,
            total_duration_seconds=time.time() - self._started_at,
            final_developmental_stage=self.identity.developmental_stage(),
            hypotheses_generated=self.identity.developmental.total_hypotheses,
            hypotheses_confirmed=self.identity.developmental.total_confirmed,
            capabilities_emerged=len(capability_summaries),
            memory_nodes=self.identity.memory.summary()["node_count"],
            termination_reason=termination_reason,
            entities_modeled=self.identity.empathy.known_entity_count(),
            emerged_capability_names=[str(item["name"]) for item in capability_summaries],
        )
        if self.config.verbose:
            self._print_final_summary(result, capability_summaries)
            print("FastSim result:", result)
        return result

    def _translate_observation(self, obs: Any) -> dict[str, Any]:
        """Translate an interface Observation into a plain dict for Identity."""

        observation = {
            **{str(key): float(value) for key, value in obs.percepts.items()},
            "novelty_hint": float(obs.novelty_hint),
            "available_actions_count": float(len(obs.available_actions)),
            "nearby_entities_count": float(len(obs.entities)),
            "available_actions": list(obs.available_actions),
            "entities": [dict(entity) for entity in obs.entities],
        }
        if obs.entities:
            observation["entity_prediction_gap"] = min(1.0, len(obs.entities) / ENTITY_PREDICTION_GAP_NORMALIZER)
            observation["relationship_depth"] = min(
                1.0,
                sum(1 for entity in obs.entities if "creature" in entity.get("tags", [])) / float(len(obs.entities)),
            )
        return observation

    def _translate_action(self, action_dict: dict[str, Any]) -> Action:
        """Translate Identity's action dict into an interface Action."""

        action_name = str(action_dict.get("action", "")).strip().lower()
        parameters = dict(action_dict.get("parameters", {}))
        drive_source = str(action_dict.get("drive_source", DEFAULT_FALLBACK_DRIVE_SOURCE))
        rationale = str(action_dict.get("rationale", ""))
        available = set(self.adapter.get_available_actions())

        if action_name not in available:
            fallback = self._fallback_action_parameters()
            action_name = fallback["name"]
            parameters = fallback["parameters"]
            drive_source = drive_source or DEFAULT_FALLBACK_DRIVE_SOURCE
            rationale = rationale or "Fallback to a valid local interaction."

        if action_name in {"examine", "push", "heat", "cool", "combine"} and "object_id" not in parameters:
            fallback = self._fallback_action_parameters()
            if fallback["name"] == DEFAULT_EXAMINE_FALLBACK_ACTION:
                action_name = fallback["name"]
                parameters = fallback["parameters"]

        if action_name == DEFAULT_MOVE_FALLBACK_ACTION and "direction" not in parameters:
            parameters["direction"] = DEFAULT_MOVE_DIRECTION

        return Action(
            name=action_name,
            parameters=parameters,
            drive_source=drive_source,
            expected_outcome={
                "rationale": rationale,
                **dict(action_dict.get("parameters", {})),
            },
        )

    def _translate_outcome(self, outcome: Any) -> tuple[dict[str, Any], float]:
        """Translate an interface Outcome into plain data for Identity.absorb()."""

        outcome_dict = {
            "action": outcome.action.name,
            "parameters": dict(outcome.action.parameters),
            "drive_source": outcome.action.drive_source,
            "expected_outcome": dict(outcome.action.expected_outcome),
            "hypothesis_id": outcome.action.parameters.get("hypothesis_id"),
            "timestamp": float(outcome.timestamp),
            "resulting_state": dict(outcome.resulting_state),
            "success": bool(outcome.success),
            "entities_affected": [dict(entity) for entity in outcome.entities_affected],
        }
        return outcome_dict, float(outcome.valence)

    def _log_cycle(self, cycle: int) -> None:
        """Print a readable simulation progress summary."""

        memory_nodes = self.identity.memory.summary()["node_count"]
        active_hypotheses = len(self.identity.hypothesis_engine.get_active())
        confirmed_hypotheses = len(self.identity.hypothesis_engine.get_confirmed())
        print(
            f"[cycle {cycle}] "
            f"drive={self.identity.current_drive()} "
            f"active_hypotheses={active_hypotheses} "
            f"confirmed={confirmed_hypotheses} "
            f"memory_nodes={memory_nodes} "
            f"stage={self.identity.developmental_stage():.4f}"
        )

    def _checkpoint(self, cycle: int) -> None:
        """Persist identity state mid-run."""

        if CHECKPOINT_FULL_SAVE:
            self.identity.sleep()
        else:
            snapshot_path = self._progress_snapshot_path(cycle)
            dominant_drive, dominant_weight = self.identity.drives.get_dominant_drive()
            snapshot = {
                "cycle": int(cycle),
                "stage": round(self.identity.developmental_stage(), 4),
                "phase": self.identity.developmental.phase_name,
                "hypotheses": int(self.identity.developmental.total_hypotheses),
                "confirmed": int(self.identity.developmental.total_confirmed),
                "dominant_drive": dominant_drive,
                "dominant_weight": round(dominant_weight, 4),
            }
            snapshot_path.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
            self._progress_snapshot_paths.append(snapshot_path)
        if self.config.verbose:
            print(f"Checkpoint saved at cycle {cycle}")

    def _fallback_action_parameters(self) -> dict[str, Any]:
        """Return a valid fallback action based on the current local context."""

        entities = [] if self._last_observation is None else list(self._last_observation.entities)
        if entities:
            return {
                "name": DEFAULT_EXAMINE_FALLBACK_ACTION,
                "parameters": {"object_id": entities[0]["id"]},
            }
        return {
            "name": DEFAULT_MOVE_FALLBACK_ACTION,
            "parameters": {"direction": DEFAULT_MOVE_DIRECTION},
        }

    def _record_growth_hypothesis_delta(self) -> None:
        """Update growth records when new hypotheses are generated."""

        delta = self.identity.developmental.total_hypotheses - self._last_hypothesis_total
        if delta <= 0:
            return
        domain = self._infer_domain_from_observation()
        for _ in range(delta):
            self.growth.record_hypothesis(domain)
        self._last_hypothesis_total = self.identity.developmental.total_hypotheses

    def _record_growth_outcome(self, outcome_dict: dict[str, Any], action: Action, valence: float) -> None:
        """Update growth records from action outcomes and confirmations."""

        domain = self._infer_domain_from_outcome(outcome_dict, action)
        self.growth.record_experience(domain, outcome_dict, valence)

        confirmed_delta = self.identity.developmental.total_confirmed - self._last_confirmed_total
        if confirmed_delta > 0:
            statement = str(action.expected_outcome.get("rationale", action.name))
            for _ in range(confirmed_delta):
                self.growth.record_confirmation(domain, statement)
            self._last_confirmed_total = self.identity.developmental.total_confirmed

    def _infer_domain_from_observation(self) -> str:
        """Infer a growth domain from the latest observation."""

        if self._last_observation is None:
            return "cognitive"
        entities = list(getattr(self._last_observation, "entities", []))
        entity_tags = {
            str(tag).lower()
            for entity in entities
            for tag in entity.get("tags", [])
        }
        if "creature" in entity_tags or "entity" in entity_tags:
            return "social"
        if "stone" in entity_tags or "rock" in entity_tags or "fire" in entity_tags or "water" in entity_tags:
            return "physical"
        return "cognitive"

    def _infer_domain_from_outcome(self, outcome_dict: dict[str, Any], action: Action) -> str:
        """Infer a growth domain from an action and its resulting state."""

        haystack = " ".join(
            [
                action.name,
                action.drive_source,
                " ".join(str(item) for item in outcome_dict.get("entities_affected", [])),
                " ".join(str(item) for item in outcome_dict.get("resulting_state", {}).get("nearby_objects", [])),
            ]
        ).lower()
        for domain, keywords in DOMAIN_KEYWORDS.items():
            if any(keyword in haystack for keyword in keywords):
                return domain
        return "cognitive"

    def _extract_tick_count(self, outcome_dict: dict[str, Any]) -> int:
        """Return the world's tick count from an outcome-like payload."""

        state = outcome_dict.get("resulting_state", {})
        return int(state.get("tick_count", self._last_tick_count))

    def _maybe_advance_world(self, outcome_dict: dict[str, Any]) -> None:
        """Advance the world one step only when the adapter has not already done so."""

        current_tick = self._extract_tick_count(outcome_dict)
        if current_tick > self._last_tick_count:
            self._last_tick_count = current_tick
            return

        tick_method = getattr(self.adapter, "tick", None)
        if callable(tick_method):
            tick_method()
            self._last_tick_count += 1

    def _progress_snapshot_path(self, cycle: int) -> Path:
        """Return the path for a lightweight progress snapshot."""

        root_dir = self.identity.data_dir.parent
        root_dir.mkdir(parents=True, exist_ok=True)
        return root_dir / f"progress_{self.identity.data_dir.name}_{int(cycle)}.json"

    def _cleanup_progress_snapshots(self) -> None:
        """Remove lightweight progress snapshots after the final full save."""

        for path in self._progress_snapshot_paths:
            if path.exists():
                path.unlink()
        self._progress_snapshot_paths = []

    def _top_memories(self) -> list[Any]:
        """Return the most relevant memories without mutating retrieval state."""

        memories = list(self.identity.memory.nodes.values())
        memories.sort(
            key=lambda node: (node.relevance, node.access_count, node.last_accessed),
            reverse=True,
        )
        return memories[:TOP_MEMORY_LIMIT]

    def _print_final_summary(self, result: SimResult, capability_summaries: list[dict[str, Any]]) -> None:
        """Print a human-readable end-of-run summary."""

        print("=== ITERA SESSION COMPLETE ===")
        print(f"Cycles: {result.total_cycles}")
        print(f"Duration: {result.total_duration_seconds:.2f}s")
        print(
            f"Final stage: {result.final_developmental_stage:.4f} "
            f"({self.identity.developmental.phase_name})"
        )
        print(
            f"Hypotheses: {result.hypotheses_generated} generated, "
            f"{result.hypotheses_confirmed} confirmed"
        )
        print(f"Memory nodes: {result.memory_nodes}")
        print(f"Entities modeled: {result.entities_modeled}")
        print(f"Dominant drive at end: {self.identity.current_drive()}")
        print("Empathy summary:")
        print(self.identity.empathy.summary())
        print()
        print("=== CAPABILITIES EMERGED ===")
        if not capability_summaries:
            print("None")
        else:
            for index, capability in enumerate(capability_summaries, start=1):
                print(f"{index}. {capability['name']} (domain: {capability['domain']})")
                print(f"   {capability['description']}")
                print(
                    f"   Confidence: {capability['confidence']:.2f} | "
                    f"Evidence: {capability['evidence_count']} experiences"
                )
        print()
        print("=== TOP MEMORIES ===")
        memories = self._top_memories()
        if not memories:
            print("None")
        else:
            for index, node in enumerate(memories, start=1):
                print(
                    f"{index}. {node.node_type} | tags={node.tags} | "
                    f"valence={node.valence:.2f} | relevance={node.relevance:.2f}"
                )


if __name__ == "__main__":
    # Clear data/identity between major test runs to avoid carrying large saved sessions forward.
    demo_data_dir = str(Path(DEFAULT_DATA_DIR) / f"fast_sim_demo_{int(time.time())}")
    config = SimConfig(max_cycles=DEFAULT_DEMO_MAX_CYCLES, verbose=True, data_dir=demo_data_dir)
    sim = FastSim(config=config)
    sim.setup()
    result = sim.run()
    print(result)

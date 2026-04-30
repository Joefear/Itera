"""World-agnostic contract between Itera's core and any environment adapter.

This module is the architectural boundary between Itera's identity core and any
world it inhabits. Core code should depend on these types and abstract methods
only; adapters translate world-specific behavior into this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0
MIN_VALENCE = -1.0
MAX_VALENCE = 1.0
DEFAULT_NOVELTY_HINT = 0.0
DEFAULT_ADAPTER_VERSION = "0.1.0"
DEFAULT_TIME_OF_DAY = 0.0
MOCK_TIME_OF_DAY = 0.3
MOCK_SELF_HEALTH = 1.0
MOCK_SELF_ENERGY = 0.9
MOCK_POSITION_X = 5
MOCK_POSITION_Y = 5
MOCK_OBSERVE_SECURITY_ALIGNMENT = 0.6
MOCK_OBSERVE_MASTERY_ALIGNMENT = 0.4
MOCK_WAIT_SECURITY_ALIGNMENT = 0.2
MOCK_OBSERVE_COST = 0.1
MOCK_WAIT_COST = 0.0


@dataclass
class Observation:
    """Raw world perception delivered by an adapter to Itera's core.

    This is the world-facing observation type. It is intentionally separate
    from any processed cognitive observation record inside `core/`.
    """

    timestamp: float
    percepts: dict[str, float]
    entities: list[dict[str, Any]]
    available_actions: list[str]
    novelty_hint: float
    self_state: dict = field(default_factory=dict)
    world_context: dict = field(default_factory=dict)
    time_of_day: float = DEFAULT_TIME_OF_DAY


@dataclass(eq=False)
class ActionDefinition:
    """
    A structured action that an adapter makes available to Itera.
    Replaces bare string action names with rich metadata.
    Itera uses drive_alignment and cost to choose between actions.
    """

    name: str
    category: str
    parameters: dict
    drive_alignment: dict[str, float]
    cost: float
    description: str

    def __eq__(self, other: object) -> bool:
        """Compare by action name so legacy string membership checks still work."""

        if isinstance(other, ActionDefinition):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False

    def __hash__(self) -> int:
        """Hash by action name for compatibility with old set-based callers."""

        return hash(self.name)


@dataclass
class Action:
    """World action requested by Itera's core and executed by an adapter."""

    name: str
    parameters: dict[str, Any]
    drive_source: str
    expected_outcome: dict[str, Any]


@dataclass
class Outcome:
    """Observed world result produced by an adapter after executing an action.

    The `valence` field is the adapter's honest raw assessment of the outcome.
    Core layers may reinterpret that signal through their own drive weighting.
    """

    action: Action
    timestamp: float
    resulting_state: dict[str, Any]
    success: bool
    valence: float
    entities_affected: list[dict[str, Any]]


class EnvironmentInterface(ABC):
    """Abstract contract between Itera's identity core and any world adapter.

    Every adapter must subclass this interface and implement all abstract
    methods. Core modules should never talk to adapters directly.
    """

    @abstractmethod
    def perceive(self) -> Observation:
        """Return the current world-facing observation for the decision cycle."""

    @abstractmethod
    def act(self, action: Action) -> None:
        """Execute an action in the world without directly returning its outcome."""

    @abstractmethod
    def get_outcome(self) -> Outcome:
        """Return the outcome of the most recent action taken in the world."""

    @abstractmethod
    def get_available_actions(self) -> list[ActionDefinition]:
        """Return structured actions available in the current world state."""

    def get_available_action_names(self) -> list[str]:
        """
        Convenience method returning just action name strings.
        Default implementation calls get_available_actions()
        and extracts names. Adapters may override for efficiency.
        """

        return [
            action.name if isinstance(action, ActionDefinition) else str(action)
            for action in self.get_available_actions()
        ]

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return the first observation of the new state."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return whether the current world state is terminal for the episode."""

    @property
    @abstractmethod
    def world_name(self) -> str:
        """Return a human-readable world name for logging only, never core branching."""

    @property
    @abstractmethod
    def adapter_version(self) -> str:
        """Return the semantic version string for this adapter implementation."""

    def validate(self) -> bool:
        """Run an optional adapter self-check before Itera begins operating."""

        return True


class MockAdapter(EnvironmentInterface):
    """Minimal in-memory adapter proving the environment contract is implementable."""

    def __init__(self) -> None:
        """Initialize a trivial mock world with deterministic stub state."""

        self._time = 0.0
        self._last_action: Action | None = None
        self._last_outcome: Outcome | None = None
        self._state: dict[str, Any] = {
            "energy": MAX_NORMALIZED_VALUE,
            "curiosity_signal": 0.4,
            "position": "origin",
        }
        self._entities: list[dict[str, Any]] = [
            {"name": "beacon", "distance": 1.0, "salience": 0.6},
        ]
        self._terminal = False

    def perceive(self) -> Observation:
        """Return a normalized snapshot of the mock world's current state."""

        return Observation(
            timestamp=self._time,
            percepts={
                "energy": float(self._state["energy"]),
                "curiosity_signal": float(self._state["curiosity_signal"]),
            },
            entities=[dict(entity) for entity in self._entities],
            available_actions=self.get_available_action_names(),
            novelty_hint=DEFAULT_NOVELTY_HINT,
            self_state={
                "health": MOCK_SELF_HEALTH,
                "energy": MOCK_SELF_ENERGY,
                "position": [MOCK_POSITION_X, MOCK_POSITION_Y],
            },
            world_context={"time_of_day": MOCK_TIME_OF_DAY, "weather": "clear"},
            time_of_day=MOCK_TIME_OF_DAY,
        )

    def act(self, action: Action) -> None:
        """Apply a trivial state transition and store the resulting mock outcome."""

        self._time += 1.0
        self._last_action = action

        if action.name == "observe_beacon":
            self._state["curiosity_signal"] = 0.7
            self._state["position"] = "beacon"
            success = True
            valence = 0.3
        else:
            success = False
            valence = MIN_NORMALIZED_VALUE

        self._last_outcome = Outcome(
            action=action,
            timestamp=self._time,
            resulting_state=dict(self._state),
            success=success,
            valence=max(MIN_VALENCE, min(MAX_VALENCE, valence)),
            entities_affected=[dict(entity) for entity in self._entities],
        )

    def get_outcome(self) -> Outcome:
        """Return the last stored mock outcome after an action has been executed."""

        if self._last_outcome is None:
            raise RuntimeError("get_outcome() called before any action was taken.")
        return self._last_outcome

    def get_available_actions(self) -> list[ActionDefinition]:
        """Return the small set of structured actions available in the mock world."""

        return [
            ActionDefinition(
                name="observe_beacon",
                category="exploration",
                parameters={"target": "string"},
                drive_alignment={
                    "SECURITY": MOCK_OBSERVE_SECURITY_ALIGNMENT,
                    "MASTERY": MOCK_OBSERVE_MASTERY_ALIGNMENT,
                },
                cost=MOCK_OBSERVE_COST,
                description="Observe the nearby beacon.",
            ),
            ActionDefinition(
                name="wait",
                category="interaction",
                parameters={},
                drive_alignment={"SECURITY": MOCK_WAIT_SECURITY_ALIGNMENT},
                cost=MOCK_WAIT_COST,
                description="Let the mock world advance without acting.",
            ),
        ]

    def reset(self) -> Observation:
        """Reset the mock world to its initial state and return the first observation."""

        self._time = 0.0
        self._last_action = None
        self._last_outcome = None
        self._state = {
            "energy": MAX_NORMALIZED_VALUE,
            "curiosity_signal": 0.4,
            "position": "origin",
        }
        self._entities = [{"name": "beacon", "distance": 1.0, "salience": 0.6}]
        self._terminal = False
        return self.perceive()

    def is_terminal(self) -> bool:
        """Return whether the mock world has reached a terminal state."""

        return self._terminal

    @property
    def world_name(self) -> str:
        """Return a logging-only world name that core must never branch on."""

        return "mock-world"

    @property
    def adapter_version(self) -> str:
        """Return the semantic version of the mock adapter implementation."""

        return DEFAULT_ADAPTER_VERSION


if __name__ == "__main__":
    adapter = MockAdapter()
    initial_observation = adapter.perceive()
    print("Initial observation:", initial_observation)

    action = Action(
        name="observe_beacon",
        parameters={"target": "beacon"},
        drive_source="SECURITY",
        expected_outcome={"position": "beacon", "curiosity_signal": 0.7},
    )
    adapter.act(action)
    outcome = adapter.get_outcome()

    print("Action:", action)
    print("Outcome:", outcome)
    print("Next observation:", adapter.perceive())

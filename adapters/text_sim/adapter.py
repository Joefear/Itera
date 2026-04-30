"""Adapter that exposes the text simulation through EnvironmentInterface."""

from __future__ import annotations

import json
import time
from typing import Any

try:
    from interface.environment import (
        Action,
        ActionDefinition,
        DEFAULT_ADAPTER_VERSION,
        MAX_VALENCE,
        MIN_VALENCE,
        Observation,
        Outcome,
        EnvironmentInterface,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from interface.environment import (  # type: ignore[no-redef]
        Action,
        ActionDefinition,
        DEFAULT_ADAPTER_VERSION,
        MAX_VALENCE,
        MIN_VALENCE,
        Observation,
        Outcome,
        EnvironmentInterface,
    )

from adapters.text_sim.world import DEFAULT_NEARBY_RADIUS, DIRECTIONS, TextWorld

DEFAULT_SEED = 7
NOVELTY_DIFFERENCE_SCALE = 6.0
POSITIVE_NOVELTY_VALENCE = 0.35
NEGATIVE_BLOCKED_VALENCE = -0.25
NEUTRAL_VALENCE = 0.0
EXAMINE_NOVELTY_BONUS = 0.15
TEXT_SIM_WORLD_NAME = "text_sim"
NOVELTY_CHANGE_FLOOR = 0.25
INTERACTION_NOVELTY_MINIMUM = 0.5
MINIMUM_NOVELTY_PER_CYCLE = 0.1
PROPERTY_CHANGE_NOVELTY_THRESHOLD = 0.05
TEXT_SIM_SELF_ENERGY = 1.0
TEXT_SIM_DAY_CYCLE_TICKS = 100

ACTION_MOVE = "move"
ACTION_EXAMINE = "examine"
ACTION_PUSH = "push"
ACTION_HEAT = "heat"
ACTION_COOL = "cool"
ACTION_COMBINE = "combine"

ACTION_CATEGORY_MOVEMENT = "movement"
ACTION_CATEGORY_EXPLORATION = "exploration"
ACTION_CATEGORY_INTERACTION = "interaction"

PARAMETER_DIRECTION = "direction"
PARAMETER_OBJECT_ID = "object_id"
PARAMETER_TARGET_ID = "target_id"
PARAMETER_DIRECTION_OPTIONS = "north|south|east|west"
PARAMETER_STRING = "string"

DRIVE_SECURITY = "SECURITY"
DRIVE_MASTERY = "MASTERY"
DRIVE_SURVIVAL = "SURVIVAL"
DRIVE_ACTUALIZATION = "ACTUALIZATION"

MOVE_SECURITY_ALIGNMENT = 0.7
MOVE_MASTERY_ALIGNMENT = 0.3
EXAMINE_MASTERY_ALIGNMENT = 0.8
EXAMINE_SECURITY_ALIGNMENT = 0.4
PUSH_MASTERY_ALIGNMENT = 0.6
PUSH_SURVIVAL_ALIGNMENT = 0.3
HEAT_MASTERY_ALIGNMENT = 0.7
HEAT_SECURITY_ALIGNMENT = 0.3
COOL_MASTERY_ALIGNMENT = 0.7
COOL_SECURITY_ALIGNMENT = 0.3
COMBINE_MASTERY_ALIGNMENT = 0.9
COMBINE_ACTUALIZATION_ALIGNMENT = 0.4

MOVE_ACTION_COST = 0.1
EXAMINE_ACTION_COST = 0.05
PUSH_ACTION_COST = 0.2
HEAT_ACTION_COST = 0.3
COOL_ACTION_COST = 0.3
COMBINE_ACTION_COST = 0.4


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric value into an inclusive range."""

    return max(minimum, min(maximum, float(value)))


class TextSimAdapter(EnvironmentInterface):
    """Translate the text simulation world into the environment interface."""

    def __init__(self, seed: int | None = DEFAULT_SEED) -> None:
        """Initialize the adapter with a reproducible text world."""

        self.world = TextWorld(seed=seed)
        self._last_action: Action | None = None
        self._last_outcome: Outcome | None = None
        self._last_perception_signature = ""

    def perceive(self) -> Observation:
        """Translate the current world state into an interface observation."""

        state = self.world.get_state()
        nearby = state["nearby_objects"]
        percepts = self._build_percepts(nearby)
        signature = self._state_signature(nearby)
        novelty_hint = self._novelty_from_signature(signature)
        self._last_perception_signature = signature
        time_of_day = (int(state["tick_count"]) % TEXT_SIM_DAY_CYCLE_TICKS) / float(TEXT_SIM_DAY_CYCLE_TICKS)
        return Observation(
            timestamp=time.time(),
            percepts=percepts,
            entities=self._build_entities(nearby),
            available_actions=self.get_available_action_names(),
            novelty_hint=novelty_hint,
            self_state={
                "position": list(state["itera_position"]),
                "energy": TEXT_SIM_SELF_ENERGY,
            },
            world_context={},
            time_of_day=time_of_day,
        )

    def act(self, action: Action) -> None:
        """Translate an interface action into a world mutation."""

        self._last_action = action
        success = False
        outcome_details: dict[str, Any]

        if action.name == ACTION_MOVE:
            direction = str(action.parameters.get(PARAMETER_DIRECTION, ""))
            success = self.world.move_itera(direction)
            outcome_details = {
                "success": success,
                "type": ACTION_MOVE,
                "direction": direction,
                "position": self.world.itera_position,
            }
        elif action.name in {ACTION_EXAMINE, ACTION_PUSH, ACTION_HEAT, ACTION_COOL, ACTION_COMBINE}:
            object_id = str(action.parameters.get(PARAMETER_OBJECT_ID, ""))
            outcome_details = self.world.interact(object_id, action.name)
            success = bool(outcome_details.get("success", False))
        else:
            outcome_details = {"success": False, "reason": "unknown_action", "action": action.name}

        self.world.tick()
        resulting_state = self.world.get_state()
        valence = self._assess_valence(action, outcome_details, resulting_state)
        self._last_outcome = Outcome(
            action=action,
            timestamp=time.time(),
            resulting_state=resulting_state,
            success=success,
            valence=valence,
            entities_affected=self._entities_affected(outcome_details),
        )

    def get_outcome(self) -> Outcome:
        """Return the outcome of the most recent action."""

        if self._last_outcome is None:
            raise RuntimeError("get_outcome() called before any action was taken.")
        return self._last_outcome

    def get_available_actions(self) -> list[ActionDefinition]:
        """Return all valid structured actions in the current local context."""

        actions = [self._move_action_definition()]
        nearby = self.world.get_nearby(self.world.itera_position, radius=DEFAULT_NEARBY_RADIUS)
        if nearby:
            actions.extend(
                [
                    self._examine_action_definition(),
                    self._push_action_definition(),
                    self._heat_action_definition(),
                    self._cool_action_definition(),
                    self._combine_action_definition(),
                ]
            )
        return actions

    def _move_action_definition(self) -> ActionDefinition:
        """Build the current move action definition."""

        return ActionDefinition(
            name=ACTION_MOVE,
            category=ACTION_CATEGORY_MOVEMENT,
            parameters={PARAMETER_DIRECTION: PARAMETER_DIRECTION_OPTIONS},
            drive_alignment={
                DRIVE_SECURITY: MOVE_SECURITY_ALIGNMENT,
                DRIVE_MASTERY: MOVE_MASTERY_ALIGNMENT,
            },
            cost=MOVE_ACTION_COST,
            description="Move Itera one cell in a cardinal direction.",
        )

    def _examine_action_definition(self) -> ActionDefinition:
        """Build the current examine action definition."""

        return ActionDefinition(
            name=ACTION_EXAMINE,
            category=ACTION_CATEGORY_EXPLORATION,
            parameters={PARAMETER_OBJECT_ID: PARAMETER_STRING},
            drive_alignment={
                DRIVE_MASTERY: EXAMINE_MASTERY_ALIGNMENT,
                DRIVE_SECURITY: EXAMINE_SECURITY_ALIGNMENT,
            },
            cost=EXAMINE_ACTION_COST,
            description="Inspect a nearby object and refine its perceived properties.",
        )

    def _push_action_definition(self) -> ActionDefinition:
        """Build the current push action definition."""

        return ActionDefinition(
            name=ACTION_PUSH,
            category=ACTION_CATEGORY_INTERACTION,
            parameters={PARAMETER_OBJECT_ID: PARAMETER_STRING},
            drive_alignment={
                DRIVE_MASTERY: PUSH_MASTERY_ALIGNMENT,
                DRIVE_SURVIVAL: PUSH_SURVIVAL_ALIGNMENT,
            },
            cost=PUSH_ACTION_COST,
            description="Push a nearby object if the world allows it to move.",
        )

    def _heat_action_definition(self) -> ActionDefinition:
        """Build the current heat action definition."""

        return ActionDefinition(
            name=ACTION_HEAT,
            category=ACTION_CATEGORY_INTERACTION,
            parameters={PARAMETER_OBJECT_ID: PARAMETER_STRING},
            drive_alignment={
                DRIVE_MASTERY: HEAT_MASTERY_ALIGNMENT,
                DRIVE_SECURITY: HEAT_SECURITY_ALIGNMENT,
            },
            cost=HEAT_ACTION_COST,
            description="Increase the temperature of a nearby object.",
        )

    def _cool_action_definition(self) -> ActionDefinition:
        """Build the current cool action definition."""

        return ActionDefinition(
            name=ACTION_COOL,
            category=ACTION_CATEGORY_INTERACTION,
            parameters={PARAMETER_OBJECT_ID: PARAMETER_STRING},
            drive_alignment={
                DRIVE_MASTERY: COOL_MASTERY_ALIGNMENT,
                DRIVE_SECURITY: COOL_SECURITY_ALIGNMENT,
            },
            cost=COOL_ACTION_COST,
            description="Decrease the temperature of a nearby object.",
        )

    def _combine_action_definition(self) -> ActionDefinition:
        """Build the current combine action definition."""

        return ActionDefinition(
            name=ACTION_COMBINE,
            category=ACTION_CATEGORY_INTERACTION,
            parameters={
                PARAMETER_OBJECT_ID: PARAMETER_STRING,
                PARAMETER_TARGET_ID: PARAMETER_STRING,
            },
            drive_alignment={
                DRIVE_MASTERY: COMBINE_MASTERY_ALIGNMENT,
                DRIVE_ACTUALIZATION: COMBINE_ACTUALIZATION_ALIGNMENT,
            },
            cost=COMBINE_ACTION_COST,
            description="Combine a nearby object with a compatible neighbor.",
        )

    def reset(self) -> Observation:
        """Reset the simulation world and return the first observation."""

        self.world.reset()
        self._last_action = None
        self._last_outcome = None
        self._last_perception_signature = ""
        return self.perceive()

    def is_terminal(self) -> bool:
        """Return whether the world has no valid actions left."""

        return len(self.get_available_actions()) == 0

    @property
    def world_name(self) -> str:
        """Return the logging-only world name for this adapter."""

        return TEXT_SIM_WORLD_NAME

    @property
    def adapter_version(self) -> str:
        """Return the semantic version string for this adapter."""

        return DEFAULT_ADAPTER_VERSION

    def validate(self) -> bool:
        """Run a simple self-check over the world and adapter boundary."""

        state = self.world.get_state()
        return bool(state["grid_size"] > 0 and isinstance(state["objects"], list))

    def _build_percepts(self, nearby: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate nearby object properties into normalized percepts."""

        if not nearby:
            return {
                "nearby_mass": 0.0,
                "nearby_temperature": 0.0,
                "nearby_brightness": 0.0,
                "nearby_interactables": 0.0,
            }

        def average(key: str) -> float:
            values = [float(item["properties"].get(key, 0.0)) for item in nearby]
            return sum(values) / len(values)

        return {
            "nearby_mass": average("mass"),
            "nearby_temperature": average("temperature"),
            "nearby_brightness": average("brightness"),
            "nearby_interactables": _clamp(
                sum(1 for item in nearby if item.get("interactable")) / float(max(1, len(nearby))),
                0.0,
                1.0,
            ),
        }

    def _build_entities(self, nearby: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate nearby world objects into interface entities."""

        entities: list[dict[str, Any]] = []
        for item in nearby:
            entities.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "position": tuple(item["position"]),
                    "interactable": bool(item["interactable"]),
                    "tags": list(item["tags"]),
                    "properties": dict(item["properties"]),
                }
            )
        return entities

    def _state_signature(self, nearby: list[dict[str, Any]]) -> str:
        """Build a stable signature for novelty comparison."""

        normalized = [
            {
                "name": item["name"],
                "position": list(item["position"]),
                "properties": {key: round(value, 2) for key, value in sorted(item["properties"].items())},
            }
            for item in sorted(nearby, key=lambda entry: entry["id"])
        ]
        return json.dumps(normalized, sort_keys=True)

    def _novelty_from_signature(self, signature: str) -> float:
        """Estimate novelty by comparing the current and previous local signatures."""

        if not self._last_perception_signature:
            return 1.0
        if signature == self._last_perception_signature:
            return MINIMUM_NOVELTY_PER_CYCLE

        current = json.loads(signature)
        previous = json.loads(self._last_perception_signature)
        previous_by_name = {
            (item["name"], tuple(item["position"])): item
            for item in previous
        }
        for item in current:
            match = previous_by_name.get((item["name"], tuple(item["position"])))
            if match is None:
                return max(MINIMUM_NOVELTY_PER_CYCLE, NOVELTY_CHANGE_FLOOR)
            for key, value in item["properties"].items():
                prior_value = float(match["properties"].get(key, value))
                if abs(float(value) - prior_value) > PROPERTY_CHANGE_NOVELTY_THRESHOLD:
                    return max(MINIMUM_NOVELTY_PER_CYCLE, NOVELTY_CHANGE_FLOOR)

        difference = abs(len(signature) - len(self._last_perception_signature))
        return max(
            MINIMUM_NOVELTY_PER_CYCLE,
            _clamp((difference / NOVELTY_DIFFERENCE_SCALE) + NOVELTY_CHANGE_FLOOR, 0.0, 1.0),
        )

    def _assess_valence(
        self,
        action: Action,
        outcome_details: dict[str, Any],
        resulting_state: dict[str, Any],
    ) -> float:
        """Return an honest outcome valence based on actual state change."""

        if not outcome_details.get("success", False):
            return NEGATIVE_BLOCKED_VALENCE

        nearby_signature = self._state_signature(resulting_state["nearby_objects"])
        novelty = self._novelty_from_signature(nearby_signature)
        valence = NEUTRAL_VALENCE
        if action.name == ACTION_MOVE:
            valence = novelty * POSITIVE_NOVELTY_VALENCE
        elif action.name == ACTION_EXAMINE:
            valence = min(MAX_VALENCE, (novelty * POSITIVE_NOVELTY_VALENCE) + EXAMINE_NOVELTY_BONUS)
        else:
            before = outcome_details.get("before", {})
            after = outcome_details.get("after", {})
            changed = before != after
            valence = (POSITIVE_NOVELTY_VALENCE if changed else NEUTRAL_VALENCE) * max(INTERACTION_NOVELTY_MINIMUM, novelty)
        return _clamp(valence, MIN_VALENCE, MAX_VALENCE)

    def _entities_affected(self, outcome_details: dict[str, Any]) -> list[dict[str, Any]]:
        """Return entities directly affected by the last outcome."""

        object_id = outcome_details.get("object_id")
        if object_id is None:
            return []
        obj = self.world.objects.get(str(object_id))
        if obj is None:
            return []
        return [obj.to_dict()]


if __name__ == "__main__":
    adapter = TextSimAdapter()
    print("Validate:", adapter.validate())
    print("Reset observation:", adapter.reset())

    observation = adapter.perceive()
    print("Perceive:", observation)

    move_action = Action(
        name="move",
        parameters={"direction": "north"},
        drive_source="SECURITY",
        expected_outcome={"position_change": "north"},
    )
    adapter.act(move_action)
    print("Move outcome:", adapter.get_outcome())

    nearby = adapter.perceive().entities
    target_id = nearby[0]["id"] if nearby else next(iter(adapter.world.objects))
    examine_action = Action(
        name="examine",
        parameters={"object_id": target_id},
        drive_source="MASTERY",
        expected_outcome={"object_id": target_id},
    )
    adapter.act(examine_action)
    print("Examine outcome:", adapter.get_outcome())
    print("World name:", adapter.world_name)
    print("Adapter version:", adapter.adapter_version)

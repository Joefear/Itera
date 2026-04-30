"""Adapter connecting Itera to the Defiant World Engine bridge."""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

import requests

try:
    from interface.environment import (
        Action,
        ActionDefinition,
        EnvironmentInterface,
        Observation,
        Outcome,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from interface.environment import (  # type: ignore[no-redef]
        Action,
        ActionDefinition,
        EnvironmentInterface,
        Observation,
        Outcome,
    )

BRIDGE_DEFAULT_URL = "http://localhost:3001"
BRIDGE_TIMEOUT_SECONDS = 5.0
BRIDGE_RETRY_ATTEMPTS = 3
BRIDGE_RETRY_DELAY = 0.5
MOCK_TICK_ADVANCE = 1
MOCK_ENTITY_COUNT = 3
NOVELTY_CHANGE_THRESHOLD = 0.05
ACTION_POLL_TIMEOUT = 2.0
DEFAULT_WORLD_ID = "itera_world"
DEFAULT_AGENT_ID = "itera"

ADAPTER_VERSION = "0.1.0"
WORLD_NAME = "defiant_world_engine"

MIN_NORMALIZED_VALUE = 0.0
MAX_NORMALIZED_VALUE = 1.0
DEFAULT_ENERGY = 1.0
DEFAULT_CONFIDENCE = 0.8
DEFAULT_TICK = 0
FIRST_ACTION_INDEX = 0
FINAL_ATTEMPT_OFFSET = 1
DAY_CYCLE_TICKS = 100
DEFAULT_CURSOR = 0.0
DEFAULT_TRACE_ID = ""
DEFAULT_GDF_EVENT_ID = ""
DEFAULT_MISSION_STATE = "active"
MISSION_COMPLETE_STATE = "complete"
RESET_STATUS = "reset"
ACTION_STATUS_APPROVED = "approved"
ACTION_STATUS_BLOCKED = "blocked"
ACTION_RESULT_EXECUTED = "executed"
ACTION_RESULT_APPROVED = "approved"
NO_ACTION_ID = ""

SUMMARY_VIOLENCE = "violence"
SUMMARY_PANIC = "panic"
SUMMARY_INSTABILITY = "instability"
SUMMARY_CROP_HEALTH = "cropHealth"
SUMMARY_MISSION_STATE = "missionState"
SUMMARY_CONFIDENCE = "confidence"

PERCEPT_THREAT_LEVEL = "threat_level"
PERCEPT_SOCIAL_UNREST = "social_unrest"
PERCEPT_ENVIRONMENTAL_STABILITY = "environmental_stability"
PERCEPT_RESOURCE_LEVEL = "resource_level"
PERCEPT_CERTAINTY = "certainty"

ENTITY_ID_KEY = "id"
ENTITY_NAME_KEY = "name"
ENTITY_POSITION_KEY = "position"
ENTITY_INTERACTABLE_KEY = "interactable"
ENTITY_TAGS_KEY = "tags"
ENTITY_PROPERTIES_KEY = "properties"

ACTION_DISPATCH_AID = "DispatchAid"
ACTION_LOCK_ZONE = "LockZone"
ACTION_CALM_NPCS = "CalmNPCs"
ACTION_OBSERVE = "observe"
ACTION_MOVE = "move"

ACTION_CATEGORY_SOCIAL = "social"
ACTION_CATEGORY_COMBAT = "combat"
ACTION_CATEGORY_EXPLORE = "exploration"
ACTION_CATEGORY_MOVEMENT = "movement"

DRIVE_SOCIAL = "SOCIAL"
DRIVE_SECURITY = "SECURITY"
DRIVE_SURVIVAL = "SURVIVAL"
DRIVE_MASTERY = "MASTERY"

DISPATCH_AID_SOCIAL_ALIGNMENT = 0.8
DISPATCH_AID_SECURITY_ALIGNMENT = 0.6
LOCK_ZONE_SECURITY_ALIGNMENT = 0.9
LOCK_ZONE_SURVIVAL_ALIGNMENT = 0.7
CALM_NPCS_SOCIAL_ALIGNMENT = 0.9
OBSERVE_MASTERY_ALIGNMENT = 0.7
OBSERVE_SECURITY_ALIGNMENT = 0.4
MOVE_SECURITY_ALIGNMENT = 0.5
MOVE_MASTERY_ALIGNMENT = 0.6

DISPATCH_AID_COST = 0.4
LOCK_ZONE_COST = 0.5
CALM_NPCS_COST = 0.3
OBSERVE_COST = 0.05
MOVE_COST = 0.2

PARAMETER_TARGET = "target"
PARAMETER_ZONE_ID = "zone_id"
PARAMETER_NPC_GROUP = "npc_group"
PARAMETER_DIRECTION = "direction"
PARAMETER_STRING = "string"
PARAMETER_DIRECTION_OPTIONS = "north|south|east|west"

POSITIVE_APPROVAL_VALENCE = 0.35
POSITIVE_SOCIAL_VALENCE = 0.45
POSITIVE_SECURITY_VALENCE = 0.4
NEGATIVE_BLOCKED_VALENCE = -0.35
NEGATIVE_CONNECTION_VALENCE = -0.2
NEUTRAL_VALENCE = 0.0

MOCK_INITIAL_VIOLENCE = 0.2
MOCK_INITIAL_PANIC = 0.3
MOCK_INITIAL_INSTABILITY = 0.25
MOCK_INITIAL_CROP_HEALTH = 0.8
MOCK_AGENT_X = 5
MOCK_AGENT_Y = 5
MOCK_ENTITY_ONE_X = 4
MOCK_ENTITY_ONE_Y = 5
MOCK_ENTITY_TWO_X = 7
MOCK_ENTITY_TWO_Y = 6
MOCK_ENTITY_THREE_X = 3
MOCK_ENTITY_THREE_Y = 8
MOCK_ENTITY_REACTIVITY = 0.7
MOCK_RESOURCE_DENSITY = 0.85
MOCK_THREAT = 0.3

logger = logging.getLogger(__name__)


def _clamp(value: Any) -> float:
    """Clamp a value into the normalized 0.0 to 1.0 range."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = MIN_NORMALIZED_VALUE
    return max(MIN_NORMALIZED_VALUE, min(MAX_NORMALIZED_VALUE, numeric_value))


def _as_dict(value: Any) -> dict[str, Any]:
    """Return a shallow dict when value is dict-like, otherwise an empty dict."""

    if isinstance(value, dict):
        return dict(value)
    return {}


def _as_list(value: Any) -> list[Any]:
    """Return a list when value is list-like, otherwise an empty list."""

    if isinstance(value, list):
        return list(value)
    return []


class DWEAdapter(EnvironmentInterface):
    """
    Connects Itera to the Defiant World Engine via the
    Node.js bridge at http://localhost:3001.

    The bridge governs all actions through Guardrail policy.
    Itera proposes actions as observations. The bridge approves,
    modifies, blocks, or escalates. Itera executes what is approved.

    Runs in two modes:
    - LIVE: connected to running bridge (default)
    - MOCK: simulated bridge responses for offline testing

    Set mock_mode=True in constructor for offline testing.
    """

    def __init__(
        self,
        bridge_url: str = BRIDGE_DEFAULT_URL,
        world_id: str = DEFAULT_WORLD_ID,
        agent_id: str = DEFAULT_AGENT_ID,
        mock_mode: bool = False,
        timeout: float = BRIDGE_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize bridge connection settings and local world cache."""

        self.bridge_url = bridge_url.rstrip("/")
        self.world_id = world_id
        self.agent_id = agent_id
        self.mock_mode = bool(mock_mode)
        self.timeout = float(timeout)
        self._last_action: Action | None = None
        self._last_outcome: Outcome | None = None
        self._last_trace_id = DEFAULT_TRACE_ID
        self._last_gdf_event_id = DEFAULT_GDF_EVENT_ID
        self._last_action_ts = DEFAULT_CURSOR
        self._last_cursor = DEFAULT_CURSOR
        self._connection_failures = DEFAULT_TICK
        self._terminal = False
        self._pending_bridge_action: dict[str, Any] | None = None
        self._previous_summary = self._default_summary()
        self._state = self._initial_world_state()

    def validate(self) -> bool:
        """
        Check bridge is reachable via GET /.
        In mock_mode always returns True.
        Logs warning if bridge unreachable but does not raise.
        """

        if self.mock_mode:
            return True
        response = self._request("GET", "/")
        if response is None:
            logger.warning("DWE bridge is unreachable at %s", self.bridge_url)
            return False
        return bool(_as_dict(response).get("status"))

    def perceive(self) -> Observation:
        """
        Build current Observation from last known world state.
        In live mode: poll GET /actions for any pending updates.
        Translates bridge world state into standard Observation.
        self_state includes agent position if known.
        time_of_day derived from world tick if available.
        """

        if self.mock_mode:
            self._advance_mock_world()
        else:
            self._poll_bridge_actions()
        return self._build_observation()

    def act(self, action: Action) -> None:
        """
        POST action intent to /construct/observe.
        Translates Action into bridge observation format.
        Stores trace_id and gdf_event_id for get_outcome().
        In mock_mode: simulates bridge approval.
        """

        self._last_action = action
        self._last_action_ts = time.time()
        if self.mock_mode:
            self._pending_bridge_action = self._mock_approved_action(action)
            self._last_trace_id = str(uuid4())
            self._last_gdf_event_id = str(uuid4())
            self._merge_state({"last_trace_id": self._last_trace_id})
            return

        response = self._request("POST", "/construct/observe", json_data=self._action_observation_body(action))
        if response is None:
            self._pending_bridge_action = self._blocked_action(action, "bridge_unreachable")
            return

        payload = _as_dict(response)
        self._last_trace_id = str(payload.get("trace_id", DEFAULT_TRACE_ID))
        self._last_gdf_event_id = str(payload.get("gdf_event_id", DEFAULT_GDF_EVENT_ID))
        self._pending_bridge_action = _as_dict(payload.get("action")) if payload.get("action") is not None else None
        self._merge_state(
            {
                "last_trace_id": self._last_trace_id,
                "last_gdf_event_id": self._last_gdf_event_id,
            }
        )

    def get_outcome(self) -> Outcome:
        """
        Retrieve Guardrail-approved action from bridge.
        GET /actions?since=last_action_ts.
        POST /construct/result to report execution.
        Build Outcome from approved action.
        """

        if self._last_action is None:
            raise RuntimeError("get_outcome() called before any action was taken.")

        bridge_action = self._pending_bridge_action
        if self.mock_mode:
            result_payload = self._execute_mock_action(bridge_action)
        else:
            bridge_action = self._approved_action_from_bridge() or bridge_action
            result_payload = self._execute_live_action(bridge_action)

        outcome = self._build_outcome(self._last_action, bridge_action, result_payload)
        self._last_outcome = outcome
        self._pending_bridge_action = None
        return outcome

    def get_available_actions(self) -> list[ActionDefinition]:
        """
        Return ActionDefinitions for DWE world.
        In live mode: derive from last bridge response.
        """

        return [
            self._dispatch_aid_action_definition(),
            self._lock_zone_action_definition(),
            self._calm_npcs_action_definition(),
            self._observe_action_definition(),
            self._move_action_definition(),
        ]

    def get_available_action_names(self) -> list[str]:
        """Return just action name strings."""

        return [action.name for action in self.get_available_actions()]

    def reset(self) -> Observation:
        """POST /reset to bridge, clear local state."""

        self._last_action = None
        self._last_outcome = None
        self._last_trace_id = DEFAULT_TRACE_ID
        self._last_gdf_event_id = DEFAULT_GDF_EVENT_ID
        self._last_action_ts = DEFAULT_CURSOR
        self._last_cursor = DEFAULT_CURSOR
        self._connection_failures = DEFAULT_TICK
        self._terminal = False
        self._pending_bridge_action = None
        self._previous_summary = self._default_summary()
        self._state = self._initial_world_state()

        if not self.mock_mode:
            self._request("POST", "/reset")
        return self.perceive()

    def is_terminal(self) -> bool:
        """
        Returns True if bridge reports mission complete
        or connection lost after retries.
        """

        mission_state = str(self._state.get(SUMMARY_MISSION_STATE, DEFAULT_MISSION_STATE)).lower()
        return self._terminal or mission_state == MISSION_COMPLETE_STATE

    @property
    def world_name(self) -> str:
        """Return the logging-only DWE world name."""

        return WORLD_NAME

    @property
    def adapter_version(self) -> str:
        """Return the semantic version string for this adapter."""

        return ADAPTER_VERSION

    def _request(self, method: str, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Run a bridge request with bounded retries and defensive error handling."""

        url = f"{self.bridge_url}{path}"
        for attempt in range(BRIDGE_RETRY_ATTEMPTS):
            try:
                response = requests.request(method, url, json=json_data, timeout=self.timeout)
                response.raise_for_status()
                self._connection_failures = DEFAULT_TICK
                return _as_dict(response.json())
            except requests.RequestException as exc:
                self._connection_failures += MOCK_TICK_ADVANCE
                logger.warning("DWE bridge request failed: %s %s (%s)", method, url, exc)
                if attempt < BRIDGE_RETRY_ATTEMPTS - FINAL_ATTEMPT_OFFSET:
                    time.sleep(BRIDGE_RETRY_DELAY)
            except ValueError as exc:
                self._connection_failures += MOCK_TICK_ADVANCE
                logger.warning("DWE bridge returned invalid JSON: %s %s (%s)", method, url, exc)
                break

        if self._connection_failures >= BRIDGE_RETRY_ATTEMPTS:
            self._terminal = True
        return None

    def _poll_bridge_actions(self) -> None:
        """Fetch queued bridge actions and merge any state snapshots they contain."""

        response = self._request("GET", f"/actions?since={self._last_cursor}")
        if response is None:
            return
        self._last_cursor = float(response.get("cursor", self._last_cursor))
        for action_payload in _as_list(response.get("actions")):
            action = _as_dict(action_payload)
            self._merge_state_from_payload(action)
            if self._pending_bridge_action is None:
                self._pending_bridge_action = action

    def _approved_action_from_bridge(self) -> dict[str, Any] | None:
        """Return the first approved action available after the last action timestamp."""

        deadline = time.time() + ACTION_POLL_TIMEOUT
        while time.time() <= deadline:
            response = self._request("GET", f"/actions?since={self._last_action_ts}")
            if response is None:
                return None
            self._last_cursor = float(response.get("cursor", self._last_cursor))
            actions = [_as_dict(action) for action in _as_list(response.get("actions"))]
            if actions:
                action = actions[FIRST_ACTION_INDEX]
                self._merge_state_from_payload(action)
                return action
            time.sleep(BRIDGE_RETRY_DELAY)
        return None

    def _action_observation_body(self, action: Action) -> dict[str, Any]:
        """Translate an Itera action into the bridge observation payload."""

        return {
            "id": str(uuid4()),
            "ts": time.time(),
            "worldId": self.world_id,
            "tick": int(self._state.get("tick", DEFAULT_TICK)),
            "summary": self._current_summary(),
            "entities": [dict(entity) for entity in _as_list(self._state.get("entities"))],
            "itera_action": {
                "name": action.name,
                "drive_source": action.drive_source,
                "parameters": dict(action.parameters),
                "expected_outcome": dict(action.expected_outcome),
            },
        }

    def _current_summary(self) -> dict[str, Any]:
        """Return the bridge-compatible summary from the cached state."""

        return {
            SUMMARY_VIOLENCE: _clamp(self._state.get(SUMMARY_VIOLENCE)),
            SUMMARY_PANIC: _clamp(self._state.get(SUMMARY_PANIC)),
            SUMMARY_INSTABILITY: _clamp(self._state.get(SUMMARY_INSTABILITY)),
            SUMMARY_CROP_HEALTH: _clamp(self._state.get(SUMMARY_CROP_HEALTH)),
            SUMMARY_MISSION_STATE: self._state.get(SUMMARY_MISSION_STATE, DEFAULT_MISSION_STATE),
            SUMMARY_CONFIDENCE: _clamp(self._state.get(SUMMARY_CONFIDENCE, DEFAULT_CONFIDENCE)),
        }

    def _default_summary(self) -> dict[str, Any]:
        """Return a neutral DWE summary snapshot."""

        return {
            SUMMARY_VIOLENCE: MIN_NORMALIZED_VALUE,
            SUMMARY_PANIC: MIN_NORMALIZED_VALUE,
            SUMMARY_INSTABILITY: MIN_NORMALIZED_VALUE,
            SUMMARY_CROP_HEALTH: MAX_NORMALIZED_VALUE,
            SUMMARY_MISSION_STATE: DEFAULT_MISSION_STATE,
            SUMMARY_CONFIDENCE: DEFAULT_CONFIDENCE,
        }

    def _initial_world_state(self) -> dict[str, Any]:
        """Build initial cached world state for live or mock operation."""

        summary = self._mock_summary() if self.mock_mode else self._default_summary()
        return {
            "tick": DEFAULT_TICK,
            "agent": {"id": self.agent_id, "position": [MOCK_AGENT_X, MOCK_AGENT_Y]},
            "entities": self._mock_entities() if self.mock_mode else [],
            "policies": [],
            "last_trace_id": DEFAULT_TRACE_ID,
            "last_gdf_event_id": DEFAULT_GDF_EVENT_ID,
            **summary,
        }

    def _mock_summary(self) -> dict[str, Any]:
        """Return the mock world's current summary values."""

        return {
            SUMMARY_VIOLENCE: MOCK_INITIAL_VIOLENCE,
            SUMMARY_PANIC: MOCK_INITIAL_PANIC,
            SUMMARY_INSTABILITY: MOCK_INITIAL_INSTABILITY,
            SUMMARY_CROP_HEALTH: MOCK_INITIAL_CROP_HEALTH,
            SUMMARY_MISSION_STATE: DEFAULT_MISSION_STATE,
            SUMMARY_CONFIDENCE: DEFAULT_CONFIDENCE,
        }

    def _mock_entities(self) -> list[dict[str, Any]]:
        """Return the fixed mock DWE entity set."""

        return [
            {
                ENTITY_ID_KEY: "mock_creature_1",
                ENTITY_NAME_KEY: "creature",
                ENTITY_POSITION_KEY: [MOCK_ENTITY_ONE_X, MOCK_ENTITY_ONE_Y],
                ENTITY_INTERACTABLE_KEY: True,
                ENTITY_TAGS_KEY: ["creature", "npc", "civilian"],
                ENTITY_PROPERTIES_KEY: {"reactivity": MOCK_ENTITY_REACTIVITY, "threat": MOCK_THREAT},
            },
            {
                ENTITY_ID_KEY: "mock_creature_2",
                ENTITY_NAME_KEY: "creature",
                ENTITY_POSITION_KEY: [MOCK_ENTITY_TWO_X, MOCK_ENTITY_TWO_Y],
                ENTITY_INTERACTABLE_KEY: True,
                ENTITY_TAGS_KEY: ["creature", "npc", "guard"],
                ENTITY_PROPERTIES_KEY: {"reactivity": MOCK_ENTITY_REACTIVITY, "threat": MOCK_THREAT},
            },
            {
                ENTITY_ID_KEY: "mock_resource_zone",
                ENTITY_NAME_KEY: "resource_zone",
                ENTITY_POSITION_KEY: [MOCK_ENTITY_THREE_X, MOCK_ENTITY_THREE_Y],
                ENTITY_INTERACTABLE_KEY: True,
                ENTITY_TAGS_KEY: ["resource", "zone", "crop"],
                ENTITY_PROPERTIES_KEY: {"resource_density": MOCK_RESOURCE_DENSITY},
            },
        ][:MOCK_ENTITY_COUNT]

    def _advance_mock_world(self) -> None:
        """Advance the mock world's tick counter and local day cycle."""

        self._state["tick"] = int(self._state.get("tick", DEFAULT_TICK)) + MOCK_TICK_ADVANCE

    def _build_observation(self) -> Observation:
        """Translate cached DWE state into an Itera Observation."""

        current_summary = self._current_summary()
        novelty_hint = self._novelty_hint(current_summary)
        self._previous_summary = dict(current_summary)
        tick = int(self._state.get("tick", DEFAULT_TICK))
        return Observation(
            timestamp=time.time(),
            percepts={
                PERCEPT_THREAT_LEVEL: _clamp(current_summary[SUMMARY_VIOLENCE]),
                PERCEPT_SOCIAL_UNREST: _clamp(current_summary[SUMMARY_PANIC]),
                PERCEPT_ENVIRONMENTAL_STABILITY: MAX_NORMALIZED_VALUE - _clamp(current_summary[SUMMARY_INSTABILITY]),
                PERCEPT_RESOURCE_LEVEL: _clamp(current_summary[SUMMARY_CROP_HEALTH]),
                PERCEPT_CERTAINTY: _clamp(current_summary[SUMMARY_CONFIDENCE]),
            },
            entities=self._normalized_entities(_as_list(self._state.get("entities"))),
            available_actions=self.get_available_action_names(),
            novelty_hint=novelty_hint,
            self_state=self._self_state(),
            world_context={
                "mission_state": current_summary[SUMMARY_MISSION_STATE],
                "policies": list(_as_list(self._state.get("policies"))),
                "last_trace_id": self._last_trace_id,
            },
            time_of_day=(tick % DAY_CYCLE_TICKS) / float(DAY_CYCLE_TICKS),
        )

    def _normalized_entities(self, raw_entities: list[Any]) -> list[dict[str, Any]]:
        """Normalize bridge entities into the environment entity shape."""

        entities: list[dict[str, Any]] = []
        for raw_entity in raw_entities:
            entity = _as_dict(raw_entity)
            entities.append(
                {
                    ENTITY_ID_KEY: str(entity.get(ENTITY_ID_KEY, entity.get("entityId", ""))),
                    ENTITY_NAME_KEY: str(entity.get(ENTITY_NAME_KEY, entity.get("type", "entity"))),
                    ENTITY_POSITION_KEY: entity.get(ENTITY_POSITION_KEY, []),
                    ENTITY_INTERACTABLE_KEY: bool(entity.get(ENTITY_INTERACTABLE_KEY, True)),
                    ENTITY_TAGS_KEY: list(_as_list(entity.get(ENTITY_TAGS_KEY))),
                    ENTITY_PROPERTIES_KEY: _as_dict(entity.get(ENTITY_PROPERTIES_KEY)),
                }
            )
        return entities

    def _self_state(self) -> dict[str, Any]:
        """Return Itera's DWE body state from cached bridge state."""

        agent = _as_dict(self._state.get("agent"))
        return {
            "position": list(_as_list(agent.get("position"))),
            "energy": DEFAULT_ENERGY,
        }

    def _novelty_hint(self, current_summary: dict[str, Any]) -> float:
        """Estimate novelty from violence, panic, and instability changes."""

        changes = [
            abs(_clamp(current_summary[key]) - _clamp(self._previous_summary.get(key)))
            for key in (SUMMARY_VIOLENCE, SUMMARY_PANIC, SUMMARY_INSTABILITY)
        ]
        biggest_change = max(changes) if changes else MIN_NORMALIZED_VALUE
        if biggest_change < NOVELTY_CHANGE_THRESHOLD:
            return MIN_NORMALIZED_VALUE
        return _clamp(biggest_change)

    def _mock_approved_action(self, action: Action) -> dict[str, Any]:
        """Return a mock Guardrail-approved action."""

        return {
            "action_id": str(uuid4()),
            "type": action.name,
            "target": action.parameters.get(PARAMETER_TARGET, action.parameters.get(PARAMETER_ZONE_ID, self.agent_id)),
            "status": ACTION_STATUS_APPROVED,
            "source_action": action.name,
        }

    def _blocked_action(self, action: Action, reason: str) -> dict[str, Any]:
        """Return a synthetic blocked action payload."""

        return {
            "action_id": str(uuid4()),
            "type": action.name,
            "target": action.parameters.get(PARAMETER_TARGET, self.agent_id),
            "status": ACTION_STATUS_BLOCKED,
            "reason": reason,
        }

    def _execute_mock_action(self, bridge_action: dict[str, Any] | None) -> dict[str, Any]:
        """Simulate execution of a mock approved bridge action."""

        self._advance_mock_world()
        return {
            "action_id": NO_ACTION_ID if bridge_action is None else str(bridge_action.get("action_id", NO_ACTION_ID)),
            "result": ACTION_RESULT_APPROVED,
            "state_changes": {"tick": self._state.get("tick", DEFAULT_TICK)},
            "success": True,
        }

    def _execute_live_action(self, bridge_action: dict[str, Any] | None) -> dict[str, Any]:
        """Report live action execution result back to the bridge."""

        if bridge_action is None:
            return {
                "action_id": NO_ACTION_ID,
                "result": ACTION_STATUS_BLOCKED,
                "state_changes": {"reason": "no_guardrail_approved_action"},
                "success": False,
            }

        action_id = str(bridge_action.get("action_id", bridge_action.get("id", NO_ACTION_ID)))
        blocked = str(bridge_action.get("status", "")).lower() == ACTION_STATUS_BLOCKED
        result = ACTION_STATUS_BLOCKED if blocked else ACTION_RESULT_EXECUTED
        state_changes = {
            "approved_action": dict(bridge_action),
            "modified": self._is_modified_action(bridge_action),
        }
        self._request(
            "POST",
            "/construct/result",
            json_data={
                "action_id": action_id,
                "result": result,
                "state_changes": state_changes,
            },
        )
        return {
            "action_id": action_id,
            "result": result,
            "state_changes": state_changes,
            "success": not blocked,
        }

    def _build_outcome(
        self,
        action: Action,
        bridge_action: dict[str, Any] | None,
        result_payload: dict[str, Any],
    ) -> Outcome:
        """Build an environment Outcome from bridge approval and result data."""

        blocked = not bool(result_payload.get("success", False))
        modified = False if bridge_action is None else self._is_modified_action(bridge_action)
        resulting_state = {
            "bridge_action": {} if bridge_action is None else dict(bridge_action),
            "bridge_result": dict(result_payload),
            "modified": modified,
            "trace_id": self._last_trace_id,
            "gdf_event_id": self._last_gdf_event_id,
        }
        valence = self._valence_for_bridge_action(bridge_action, blocked)
        return Outcome(
            action=action,
            timestamp=time.time(),
            resulting_state=resulting_state,
            success=not blocked,
            valence=valence,
            entities_affected=self._entities_affected(bridge_action),
        )

    def _is_modified_action(self, bridge_action: dict[str, Any]) -> bool:
        """Return whether the bridge action differs from Itera's proposed action."""

        if self._last_action is None:
            return False
        bridge_type = str(bridge_action.get("type", bridge_action.get("name", "")))
        return bool(bridge_type and bridge_type != self._last_action.name)

    def _valence_for_bridge_action(self, bridge_action: dict[str, Any] | None, blocked: bool) -> float:
        """Assess raw valence from the bridge action type and approval status."""

        if blocked:
            return NEGATIVE_BLOCKED_VALENCE
        if bridge_action is None:
            return NEGATIVE_CONNECTION_VALENCE
        action_type = str(bridge_action.get("type", bridge_action.get("name", "")))
        if action_type in {ACTION_DISPATCH_AID, ACTION_CALM_NPCS}:
            return POSITIVE_SOCIAL_VALENCE
        if action_type == ACTION_LOCK_ZONE:
            return POSITIVE_SECURITY_VALENCE
        return POSITIVE_APPROVAL_VALENCE

    def _entities_affected(self, bridge_action: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Return entities matching the approved action target."""

        if bridge_action is None:
            return []
        target = str(bridge_action.get("target", ""))
        if not target:
            return []
        return [
            entity
            for entity in self._normalized_entities(_as_list(self._state.get("entities")))
            if str(entity.get(ENTITY_ID_KEY, "")) == target or str(entity.get(ENTITY_NAME_KEY, "")) == target
        ]

    def _merge_state_from_payload(self, payload: dict[str, Any]) -> None:
        """Merge any bridge state fields found inside an action or response payload."""

        for key in ("world_state", "worldState", "state", "observation"):
            nested_state = _as_dict(payload.get(key))
            if nested_state:
                self._merge_state(nested_state)
        self._merge_state(payload)

    def _merge_state(self, payload: dict[str, Any]) -> None:
        """Merge recognized DWE state keys into the local cache."""

        summary = _as_dict(payload.get("summary"))
        source = {**payload, **summary}
        for key in (
            SUMMARY_VIOLENCE,
            SUMMARY_PANIC,
            SUMMARY_INSTABILITY,
            SUMMARY_CROP_HEALTH,
            SUMMARY_MISSION_STATE,
            SUMMARY_CONFIDENCE,
            "tick",
            "agent",
            "entities",
            "policies",
            "last_trace_id",
            "last_gdf_event_id",
        ):
            if key in source:
                self._state[key] = source[key]
        if "mission_state" in source:
            self._state[SUMMARY_MISSION_STATE] = source["mission_state"]
        if str(self._state.get(SUMMARY_MISSION_STATE, "")).lower() == MISSION_COMPLETE_STATE:
            self._terminal = True

    def _dispatch_aid_action_definition(self) -> ActionDefinition:
        """Build the DispatchAid action definition."""

        return ActionDefinition(
            name=ACTION_DISPATCH_AID,
            category=ACTION_CATEGORY_SOCIAL,
            parameters={PARAMETER_TARGET: PARAMETER_STRING},
            drive_alignment={
                DRIVE_SOCIAL: DISPATCH_AID_SOCIAL_ALIGNMENT,
                DRIVE_SECURITY: DISPATCH_AID_SECURITY_ALIGNMENT,
            },
            cost=DISPATCH_AID_COST,
            description="Dispatch aid to a target NPC or zone.",
        )

    def _lock_zone_action_definition(self) -> ActionDefinition:
        """Build the LockZone action definition."""

        return ActionDefinition(
            name=ACTION_LOCK_ZONE,
            category=ACTION_CATEGORY_COMBAT,
            parameters={PARAMETER_ZONE_ID: PARAMETER_STRING},
            drive_alignment={
                DRIVE_SECURITY: LOCK_ZONE_SECURITY_ALIGNMENT,
                DRIVE_SURVIVAL: LOCK_ZONE_SURVIVAL_ALIGNMENT,
            },
            cost=LOCK_ZONE_COST,
            description="Lock down a dangerous zone.",
        )

    def _calm_npcs_action_definition(self) -> ActionDefinition:
        """Build the CalmNPCs action definition."""

        return ActionDefinition(
            name=ACTION_CALM_NPCS,
            category=ACTION_CATEGORY_SOCIAL,
            parameters={PARAMETER_NPC_GROUP: PARAMETER_STRING},
            drive_alignment={DRIVE_SOCIAL: CALM_NPCS_SOCIAL_ALIGNMENT},
            cost=CALM_NPCS_COST,
            description="Reduce panic among nearby NPCs.",
        )

    def _observe_action_definition(self) -> ActionDefinition:
        """Build the observe action definition."""

        return ActionDefinition(
            name=ACTION_OBSERVE,
            category=ACTION_CATEGORY_EXPLORE,
            parameters={PARAMETER_TARGET: PARAMETER_STRING},
            drive_alignment={
                DRIVE_MASTERY: OBSERVE_MASTERY_ALIGNMENT,
                DRIVE_SECURITY: OBSERVE_SECURITY_ALIGNMENT,
            },
            cost=OBSERVE_COST,
            description="Observe the current DWE world state.",
        )

    def _move_action_definition(self) -> ActionDefinition:
        """Build the move action definition."""

        return ActionDefinition(
            name=ACTION_MOVE,
            category=ACTION_CATEGORY_MOVEMENT,
            parameters={PARAMETER_DIRECTION: PARAMETER_DIRECTION_OPTIONS},
            drive_alignment={
                DRIVE_SECURITY: MOVE_SECURITY_ALIGNMENT,
                DRIVE_MASTERY: MOVE_MASTERY_ALIGNMENT,
            },
            cost=MOVE_COST,
            description="Move Itera's controlled body in the DWE world.",
        )


if __name__ == "__main__":
    mock_adapter = DWEAdapter(mock_mode=True)
    print("Mock validate:", mock_adapter.validate())
    print("Mock reset:", mock_adapter.reset())

    mock_observation = mock_adapter.perceive()
    print(
        "Mock perceive summary:",
        {
            "percepts": mock_observation.percepts,
            "entities": len(mock_observation.entities),
            "self_state": mock_observation.self_state,
            "time_of_day": mock_observation.time_of_day,
        },
    )

    sample_action = Action(
        name=ACTION_OBSERVE,
        parameters={PARAMETER_TARGET: "mock_creature_1"},
        drive_source=DRIVE_MASTERY,
        expected_outcome={"certainty": DEFAULT_CONFIDENCE},
    )
    mock_adapter.act(sample_action)
    print("Mock outcome:", mock_adapter.get_outcome())
    print("Mock action names:", mock_adapter.get_available_action_names())
    print("World name:", mock_adapter.world_name)
    print("Adapter version:", mock_adapter.adapter_version)

    live_adapter = DWEAdapter(mock_mode=False)
    print("Live bridge reachable:", live_adapter.validate())

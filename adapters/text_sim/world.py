"""Minimal grid world for Itera's first text simulation."""

from __future__ import annotations

from dataclasses import asdict
import random
from typing import Any

from .objects import (
    WorldObject,
    make_creature,
    make_fire,
    make_rock,
    make_stone_deposit,
    make_water,
)

DEFAULT_GRID_SIZE = 10
DEFAULT_NEARBY_RADIUS = 2
DEFAULT_FIRE_HEAT_TRANSFER = 0.08
DEFAULT_WATER_COOLING = 0.12
DEFAULT_PUSH_DISTANCE = 1
MIN_COORDINATE = 0
DEFAULT_COMBINE_EFFECT = 0.06
ITERA_START_POSITION = (5, 5)
FIRE_SPREAD_PROBABILITY = 0.15
EMBER_INITIAL_TEMPERATURE = 0.6
EMBER_INITIAL_BRIGHTNESS = 0.55

DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}


def _clamp(value: float) -> float:
    """Clamp a property value into the normalized range."""

    return max(0.0, min(1.0, float(value)))


class TextWorld:
    """A 10x10 grid world with consistent physics."""

    grid_size: int = DEFAULT_GRID_SIZE

    def __init__(self, seed: int | None = None, grid_size: int = DEFAULT_GRID_SIZE) -> None:
        """Initialize the world with an optional random seed."""

        self.seed = seed
        self.grid_size = int(grid_size)
        self._rng = random.Random(seed)
        self.itera_position: tuple[int, int] = ITERA_START_POSITION
        self.objects: dict[str, WorldObject] = {}
        self.tick_count = 0
        self.reset()

    def reset(self) -> None:
        """Reset the world to a deterministic initial arrangement."""

        self._rng = random.Random(self.seed)
        self.tick_count = 0
        self.itera_position = (self.grid_size // 2, self.grid_size // 2)
        self.objects = {}
        for obj in [
            make_rock((2, 2)),
            make_fire((7, 2)),
            make_water((2, 7)),
            make_creature((7, 7)),
            make_stone_deposit((5, 3)),
        ]:
            self.spawn_object(obj)

    def get_cell(self, x: int, y: int) -> list[WorldObject]:
        """Return all objects occupying the given cell."""

        return [
            obj for obj in self.objects.values()
            if obj.position == (int(x), int(y))
        ]

    def get_nearby(self, position: tuple[int, int], radius: int = DEFAULT_NEARBY_RADIUS) -> list[WorldObject]:
        """Return objects within Manhattan radius of a position."""

        px, py = position
        return [
            obj
            for obj in self.objects.values()
            if abs(obj.position[0] - px) + abs(obj.position[1] - py) <= int(radius)
        ]

    def move_itera(self, direction: str) -> bool:
        """Move Itera by one cell in the requested cardinal direction."""

        if direction not in DIRECTIONS:
            return False
        dx, dy = DIRECTIONS[direction]
        new_x = self.itera_position[0] + dx
        new_y = self.itera_position[1] + dy
        if not self._in_bounds((new_x, new_y)):
            return False
        self.itera_position = (new_x, new_y)
        return True

    def interact(self, object_id: str, action: str) -> dict[str, Any]:
        """Apply an interaction to an object and return before/after state."""

        obj = self.objects.get(str(object_id))
        if obj is None:
            return {"success": False, "reason": "object_not_found"}

        before = dict(obj.properties)
        success = True
        details: dict[str, Any] = {}

        if action == "examine":
            details["observation"] = f"{obj.name} observed"
        elif action == "push":
            success = self._push_object(obj)
            details["movement"] = obj.position
        elif action == "heat":
            obj.properties["temperature"] = _clamp(obj.properties.get("temperature", 0.0) + DEFAULT_FIRE_HEAT_TRANSFER)
            obj.properties["brightness"] = _clamp(obj.properties.get("brightness", 0.0) + (DEFAULT_FIRE_HEAT_TRANSFER / 2.0))
        elif action == "cool":
            obj.properties["temperature"] = _clamp(obj.properties.get("temperature", 0.0) - DEFAULT_WATER_COOLING)
        elif action == "combine":
            success, details = self._combine_with_neighbor(obj)
        else:
            success = False
            details["reason"] = "unknown_action"

        after = dict(obj.properties)
        return {
            "success": success,
            "object_id": obj.id,
            "object_name": obj.name,
            "before": before,
            "after": after,
            "position": obj.position,
            "details": details,
        }

    def tick(self) -> None:
        """Advance the world by one step."""

        self.tick_count += 1
        self._move_creatures()
        self._spread_fire()
        self._flow_water()

    def get_state(self) -> dict[str, Any]:
        """Return the full visible world state."""

        nearby = self.get_nearby(self.itera_position)
        return {
            "grid_size": self.grid_size,
            "itera_position": self.itera_position,
            "tick_count": self.tick_count,
            "objects": [obj.to_dict() for obj in self.objects.values()],
            "nearby_objects": [obj.to_dict() for obj in nearby],
        }

    def spawn_object(self, obj: WorldObject) -> None:
        """Add a world object to the grid if it is in bounds."""

        if not self._in_bounds(obj.position):
            raise ValueError(f"Object position out of bounds: {obj.position}")
        self.objects[obj.id] = obj

    def _in_bounds(self, position: tuple[int, int]) -> bool:
        """Return whether a grid coordinate is inside the world."""

        x, y = position
        return MIN_COORDINATE <= x < self.grid_size and MIN_COORDINATE <= y < self.grid_size

    def _push_object(self, obj: WorldObject) -> bool:
        """Push an object one cell away from Itera if space exists."""

        dx = obj.position[0] - self.itera_position[0]
        dy = obj.position[1] - self.itera_position[1]
        if dx == 0 and dy == 0:
            return False
        step_x = 0 if dx == 0 else int(dx / abs(dx))
        step_y = 0 if dy == 0 else int(dy / abs(dy))
        new_position = (
            obj.position[0] + (step_x * DEFAULT_PUSH_DISTANCE),
            obj.position[1] + (step_y * DEFAULT_PUSH_DISTANCE),
        )
        if not self._in_bounds(new_position):
            return False
        obj.position = new_position
        return True

    def _combine_with_neighbor(self, obj: WorldObject) -> tuple[bool, dict[str, Any]]:
        """Blend properties with the nearest neighboring object."""

        neighbors = [
            other for other in self.get_nearby(obj.position, radius=1)
            if other.id != obj.id
        ]
        if not neighbors:
            return False, {"reason": "no_neighbor"}

        neighbor = neighbors[0]
        shared_keys = set(obj.properties) | set(neighbor.properties)
        for key in shared_keys:
            obj_value = obj.properties.get(key, 0.0)
            neighbor_value = neighbor.properties.get(key, 0.0)
            mixed_value = (obj_value + neighbor_value) / 2.0
            obj.properties[key] = _clamp(mixed_value + DEFAULT_COMBINE_EFFECT)
            neighbor.properties[key] = _clamp(mixed_value - DEFAULT_COMBINE_EFFECT)
        return True, {"combined_with": neighbor.id}

    def _move_creatures(self) -> None:
        """Move creatures in response to Itera proximity."""

        for obj in self.objects.values():
            if "creature" not in obj.tags:
                continue
            dx = self.itera_position[0] - obj.position[0]
            dy = self.itera_position[1] - obj.position[1]
            step_x = 0 if dx == 0 else int(dx / abs(dx))
            step_y = 0 if dy == 0 else int(dy / abs(dy))
            if abs(dx) + abs(dy) <= DEFAULT_NEARBY_RADIUS:
                candidate = (obj.position[0] + step_x, obj.position[1] + step_y)
            else:
                move = self._rng.choice(list(DIRECTIONS.values()))
                candidate = (obj.position[0] + move[0], obj.position[1] + move[1])
            if self._in_bounds(candidate):
                obj.position = candidate

    def _spread_fire(self) -> None:
        """Warm nearby objects and occasionally brighten adjacent cells."""

        fires = [obj for obj in self.objects.values() if obj.name == "fire"]
        for fire in fires:
            for other in self.get_nearby(fire.position, radius=1):
                if other.id == fire.id:
                    continue
                other.properties["temperature"] = _clamp(other.properties.get("temperature", 0.0) + DEFAULT_FIRE_HEAT_TRANSFER)
            if self._rng.random() < FIRE_SPREAD_PROBABILITY:
                target = self._random_adjacent_position(fire.position)
                if target is not None and not any(obj.position == target and obj.name == "fire" for obj in self.objects.values()):
                    ember = make_fire(target)
                    ember.properties["temperature"] = EMBER_INITIAL_TEMPERATURE
                    ember.properties["brightness"] = EMBER_INITIAL_BRIGHTNESS
                    self.spawn_object(ember)

    def _flow_water(self) -> None:
        """Move water slightly and cool neighboring objects."""

        for obj in self.objects.values():
            if obj.name != "water":
                continue
            for other in self.get_nearby(obj.position, radius=1):
                if other.id == obj.id:
                    continue
                other.properties["temperature"] = _clamp(other.properties.get("temperature", 0.0) - DEFAULT_WATER_COOLING)
            target = self._random_adjacent_position(obj.position)
            if target is not None:
                obj.position = target

    def _random_adjacent_position(self, position: tuple[int, int]) -> tuple[int, int] | None:
        """Return a random adjacent in-bounds position."""

        candidates = [
            (position[0] + dx, position[1] + dy)
            for dx, dy in DIRECTIONS.values()
            if self._in_bounds((position[0] + dx, position[1] + dy))
        ]
        if not candidates:
            return None
        return self._rng.choice(candidates)

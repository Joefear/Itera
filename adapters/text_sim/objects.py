"""World objects for Itera's minimal text simulation."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any
from uuid import uuid4

MIN_PROPERTY_VALUE = 0.0
MAX_PROPERTY_VALUE = 1.0
ROCK_MASS = 0.85
ROCK_TEMPERATURE = 0.15
ROCK_HARDNESS = 0.9
ROCK_BRIGHTNESS = 0.1
FIRE_TEMPERATURE = 0.95
FIRE_BRIGHTNESS = 0.98
FIRE_MASS = 0.2
FIRE_HARDNESS = 0.05
WATER_TEMPERATURE = 0.08
WATER_FLUIDITY = 0.95
WATER_MASS = 0.55
WATER_BRIGHTNESS = 0.2
CREATURE_TEMPERATURE = 0.45
CREATURE_MASS = 0.5
CREATURE_BRIGHTNESS = 0.35
CREATURE_REACTIVITY = 0.85
CREATURE_CHI_MIN = 0.3
CREATURE_CHI_MAX = 0.7
DEPOSIT_MASS = 0.75
DEPOSIT_HARDNESS = 0.8
DEPOSIT_BRIGHTNESS = 0.25
DEPOSIT_RESOURCE = 0.9


def _clamp(value: float) -> float:
    """Clamp a property value into the normalized range."""

    return max(MIN_PROPERTY_VALUE, min(MAX_PROPERTY_VALUE, float(value)))


@dataclass
class WorldObject:
    """A world object with normalized properties and a grid position."""

    id: str
    name: str
    position: tuple[int, int]
    properties: dict[str, float]
    interactable: bool
    tags: list[str]

    def __post_init__(self) -> None:
        """Normalize object properties and tags."""

        self.position = (int(self.position[0]), int(self.position[1]))
        self.properties = {
            str(name): _clamp(value)
            for name, value in self.properties.items()
        }
        self.tags = [str(tag).strip().lower() for tag in self.tags if str(tag).strip()]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly representation of this object."""

        return {
            "id": self.id,
            "name": self.name,
            "position": self.position,
            "properties": dict(self.properties),
            "interactable": self.interactable,
            "tags": list(self.tags),
        }


def make_rock(position: tuple[int, int]) -> WorldObject:
    """Create a heavy, cool, hard rock."""

    return WorldObject(
        id=str(uuid4()),
        name="rock",
        position=position,
        properties={
            "mass": ROCK_MASS,
            "temperature": ROCK_TEMPERATURE,
            "hardness": ROCK_HARDNESS,
            "brightness": ROCK_BRIGHTNESS,
        },
        interactable=True,
        tags=["mineral", "solid", "rock"],
    )


def make_fire(position: tuple[int, int]) -> WorldObject:
    """Create a hot, bright fire source."""

    return WorldObject(
        id=str(uuid4()),
        name="fire",
        position=position,
        properties={
            "mass": FIRE_MASS,
            "temperature": FIRE_TEMPERATURE,
            "hardness": FIRE_HARDNESS,
            "brightness": FIRE_BRIGHTNESS,
        },
        interactable=True,
        tags=["energy", "fire", "hot"],
    )


def make_water(position: tuple[int, int]) -> WorldObject:
    """Create a cool, fluid water source."""

    return WorldObject(
        id=str(uuid4()),
        name="water",
        position=position,
        properties={
            "mass": WATER_MASS,
            "temperature": WATER_TEMPERATURE,
            "fluidity": WATER_FLUIDITY,
            "brightness": WATER_BRIGHTNESS,
        },
        interactable=True,
        tags=["fluid", "water", "cool"],
    )


def make_creature(position: tuple[int, int], rng: random.Random | None = None) -> WorldObject:
    """Create a reactive creature that can move around the world."""

    source_rng = random if rng is None else rng
    return WorldObject(
        id=str(uuid4()),
        name="creature",
        position=position,
        properties={
            "mass": CREATURE_MASS,
            "temperature": CREATURE_TEMPERATURE,
            "brightness": CREATURE_BRIGHTNESS,
            "reactivity": CREATURE_REACTIVITY,
            "chi_level": source_rng.uniform(CREATURE_CHI_MIN, CREATURE_CHI_MAX),
        },
        interactable=True,
        tags=["creature", "mobile", "entity"],
    )


def make_stone_deposit(position: tuple[int, int]) -> WorldObject:
    """Create an extractable stone deposit resource."""

    return WorldObject(
        id=str(uuid4()),
        name="stone_deposit",
        position=position,
        properties={
            "mass": DEPOSIT_MASS,
            "hardness": DEPOSIT_HARDNESS,
            "brightness": DEPOSIT_BRIGHTNESS,
            "resource_density": DEPOSIT_RESOURCE,
        },
        interactable=True,
        tags=["deposit", "resource", "stone"],
    )

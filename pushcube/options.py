from __future__ import annotations

import math
from dataclasses import dataclass

_D = math.sqrt(0.5)
DIRECTIONS: dict[str, tuple[float, float]] = {
    "N": (0.0, -1.0),
    "NE": (_D, -_D),
    "E": (1.0, 0.0),
    "SE": (_D, _D),
    "S": (0.0, 1.0),
    "SW": (-_D, _D),
    "W": (-1.0, 0.0),
    "NW": (-_D, -_D),
}
DISTANCES: dict[str, float] = {"short": 10.0, "medium": 30.0, "long": 80.0}


@dataclass(frozen=True)
class Option:
    direction: str
    distance: str

    @property
    def id(self) -> str:
        return f"{self.direction}-{self.distance}"

    @classmethod
    def from_id(cls, s: str) -> Option:
        direction, distance = s.split("-")
        return cls(direction, distance)

    @property
    def vector(self) -> tuple[float, float]:
        return DIRECTIONS[self.direction]

    @property
    def travel(self) -> float:
        return DISTANCES[self.distance]


def all_options() -> list[Option]:
    return [Option(d, s) for d in DIRECTIONS for s in DISTANCES]

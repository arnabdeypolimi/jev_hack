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
# Where the pusher hits the face, as seen from behind the push. Left of centre always turns the block
# clockwise on screen, right of centre counterclockwise; the value is the fraction of the half-width.
AIMS: dict[str, float] = {"left": 0.6, "center": 0.0, "right": -0.6}


@dataclass(frozen=True)
class Option:
    direction: str
    distance: str
    aim: str = "center"

    @property
    def id(self) -> str:
        base = f"{self.direction}-{self.distance}"
        return base if self.aim == "center" else f"{base}-{self.aim}"

    @classmethod
    def from_id(cls, s: str) -> Option:
        parts = s.split("-")
        return cls(parts[0], parts[1], parts[2] if len(parts) > 2 else "center")

    @property
    def vector(self) -> tuple[float, float]:
        return DIRECTIONS[self.direction]

    @property
    def travel(self) -> float:
        return DISTANCES[self.distance]


def all_options() -> list[Option]:
    return [Option(d, s, a) for d in DIRECTIONS for s in DISTANCES for a in AIMS]

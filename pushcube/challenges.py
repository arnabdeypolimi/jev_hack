from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .scoring import clamp_goal
from .shapes import Pose
from .sim import SimState

Tier = Literal["easy", "medium", "hard", "far"]


@dataclass(frozen=True)
class Challenge:
    id: str
    name: str
    tier: Tier
    shape: str
    start: SimState
    goal: Pose


def _c(id: str, name: str, tier: Tier, shape: str, pusher, block, angle: float, goal: Pose) -> Challenge:
    return Challenge(id, name, tier, shape, SimState(pusher=pusher, cube=block, angle=angle), clamp_goal(goal, shape))


R = math.radians
PRESETS: list[Challenge] = [
    _c("easy-east", "Cube: short slide east", "easy", "square", (120, 256), (220, 256), 0.0, (330, 256, 0.0)),
    _c("easy-bar", "Bar: nudge it north", "easy", "I", (200, 450), (200, 330), 0.0, (200, 230, 0.0)),
    _c("medium-triangle", "Triangle: slide west", "medium", "triangle", (440, 256), (330, 256), 0.0, (220, 256, 0.0)),
    _c("medium-twist", "Cube: slide and twist", "medium", "square", (60, 60), (160, 160), 0.0, (310, 300, R(30))),
    _c("medium-l", "L: dog-leg with a turn", "medium", "L", (460, 60), (360, 170), 0.0, (210, 300, R(45))),
    _c("hard-bar-upright", "Bar: stand it up", "hard", "I", (100, 420), (220, 300), 0.0, (320, 230, R(90))),
    _c("hard-t-turn", "T: quarter turn", "hard", "T", (60, 460), (190, 200), 0.0, (310, 300, R(90))),
    _c("hard-z-turn", "Z: flip it round", "hard", "Z", (460, 460), (320, 320), 0.0, (200, 220, R(70))),
    _c("far-across", "Cube: across the table", "far", "square", (40, 150), (100, 100), 0.0, (410, 410, R(20))),
    _c("far-l-sweep", "L: long sweep", "far", "L", (470, 40), (390, 120), 0.0, (140, 360, R(-60))),
]

_BY_ID = {c.id: c for c in PRESETS}


def get_challenge(challenge_id: str) -> Challenge | None:
    return _BY_ID.get(challenge_id)

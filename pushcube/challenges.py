from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .sim import SimState, Vec

Tier = Literal["easy", "medium", "hard", "far"]


@dataclass(frozen=True)
class Challenge:
    id: str
    name: str
    tier: Tier
    start: SimState
    goal: Vec


PRESETS: list[Challenge] = [
    Challenge("easy-east", "A short slide east", "easy", SimState(pusher=(120, 256), cube=(220, 256)), (330, 256)),
    Challenge("easy-north", "Nudge it north", "easy", SimState(pusher=(200, 430), cube=(200, 320)), (200, 220)),
    Challenge("medium-diagonal", "Down the diagonal", "medium", SimState(pusher=(60, 60), cube=(150, 150)), (320, 330)),
    Challenge("medium-dogleg", "Dog-leg", "medium", SimState(pusher=(450, 90), cube=(360, 170)), (190, 320)),
    Challenge("hard-east-wall", "Park it on the east wall", "hard", SimState(pusher=(90, 256), cube=(240, 300)), (476, 256)),
    Challenge("hard-corner", "Tuck it in the corner", "hard", SimState(pusher=(420, 120), cube=(330, 200)), (35, 476)),
    Challenge("far-across", "Across the table", "far", SimState(pusher=(40, 140), cube=(90, 90)), (420, 420)),
    Challenge("far-sweep", "Long sweep west", "far", SimState(pusher=(470, 40), cube=(420, 100)), (90, 410)),
]

_BY_ID = {c.id: c for c in PRESETS}


def get_challenge(challenge_id: str) -> Challenge | None:
    return _BY_ID.get(challenge_id)

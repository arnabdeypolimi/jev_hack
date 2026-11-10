from __future__ import annotations

from .sim import CUBE_HALF, CUBE_SIZE, WALL_MAX, WALL_MIN, Vec

SUCCESS_OVERLAP = 0.90
GOAL_MIN, GOAL_MAX = WALL_MIN + CUBE_HALF, WALL_MAX - CUBE_HALF


def overlap(cube: Vec, goal: Vec) -> float:
    ox = max(0.0, CUBE_SIZE - abs(cube[0] - goal[0]))
    oy = max(0.0, CUBE_SIZE - abs(cube[1] - goal[1]))
    return ox * oy / (CUBE_SIZE * CUBE_SIZE)


def clamp_goal(goal: Vec) -> Vec:
    return tuple(min(max(v, GOAL_MIN), GOAL_MAX) for v in goal)


class PushCounter:
    """Counts contact episodes: each rising edge of pusher-cube contact is one push."""

    def __init__(self, in_contact: bool = False) -> None:
        self.count = 0
        self._prev = in_contact

    def update(self, contact: bool) -> None:
        if contact and not self._prev:
            self.count += 1
        self._prev = contact

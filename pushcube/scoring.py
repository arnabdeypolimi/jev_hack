from __future__ import annotations

from .shapes import outline, world_outline
from .sim import CUBE_HALF, WALL_MAX, WALL_MIN

SUCCESS_OVERLAP = 0.90
GOAL_MIN, GOAL_MAX = WALL_MIN + CUBE_HALF, WALL_MAX - CUBE_HALF


def _pose(p) -> tuple[float, float, float]:
    return (float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0)


def overlap(block, goal, shape: str = "square") -> float:
    """Area of the block inside the goal outline / block area. Poses are (x, y) or (x, y, angle)."""
    a = world_outline(shape, _pose(block))
    b = world_outline(shape, _pose(goal))
    return a.intersection(b).area / outline(shape).area


def clamp_goal(goal, shape: str = "square"):
    """Move the goal (keeping its angle) so the whole outline lies inside the walls."""
    pose = _pose(goal)
    minx, miny, maxx, maxy = world_outline(shape, pose).bounds
    dx = max(0.0, WALL_MIN - minx) - max(0.0, maxx - WALL_MAX)
    dy = max(0.0, WALL_MIN - miny) - max(0.0, maxy - WALL_MAX)
    x, y = round(pose[0] + dx, 6), round(pose[1] + dy, 6)
    x, y = (int(x) if x == int(x) else x), (int(y) if y == int(y) else y)
    return (x, y, pose[2]) if len(goal) > 2 else (x, y)


RELEASE_GAP = 8.0  # px: the pusher counts as "away" only beyond this gap
RELEASE_STEPS = 3  # control steps (0.3 s) it must stay away before the push is over


class PushCounter:
    """Counts pushes: a push starts on contact and ends once the pusher has stayed clearly away for 0.3 s.

    A spinning block can drift off the pusher and swing back into it; that is still the same push.
    The same rule applies to the human and to Jev.
    """

    def __init__(self, in_contact: bool = False) -> None:
        self.count = 0
        self._in_push = in_contact
        self._away = 0

    def update(self, contact: bool, near: bool | None = None) -> None:
        near = contact if near is None else near
        if contact:
            if not self._in_push:
                self.count += 1
                self._in_push = True
            self._away = 0
        elif near:
            self._away = 0
        else:
            self._away += 1
            if self._away >= RELEASE_STEPS:
                self._in_push = False

    def observe(self, sim) -> None:
        gap = sim.gap()
        self.update(gap <= 1.0, gap <= RELEASE_GAP)

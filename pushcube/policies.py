"""Non-Jev decision rules: the geometric fallback and a one-step oracle baseline."""

from __future__ import annotations

import math

from .executor import execute_option, is_feasible
from .options import DIRECTIONS, DISTANCES, Option, all_options
from .scoring import _pose
from .shapes import angle_error
from .sim import CubeSim

# Measured block travel along an axis push for each distance (square, centre aim).
EXPECTED_TRAVEL = {"short": 6.0, "medium": 26.0, "long": 75.0}
ANGLE_WEIGHT = 1.2  # px of cost per degree of turn still needed
TURN_THRESHOLD = math.radians(8)
FAR = 90.0  # px: beyond this, move first and turn later


def pose_cost(pose, goal, shape: str) -> float:
    (x, y, a), (gx, gy, ga) = _pose(pose), _pose(goal)
    return math.hypot(gx - x, gy - y) + ANGLE_WEIGHT * abs(math.degrees(angle_error(a, ga, shape)))


def geometric_option(sim: CubeSim, goal) -> Option:
    """Cheap rule of thumb: travel toward the goal, and push off-centre when the block must turn."""
    gx, gy, ga = _pose(goal)
    (cx, cy), turn = sim.cube, angle_error(sim.angle, ga, sim.shape)
    off = (gx - cx, gy - cy)
    dist = math.hypot(*off)
    turning = abs(turn) > TURN_THRESHOLD and dist < FAR
    aim = ("left" if turn > 0 else "right") if turning else "center"

    def toward(direction: str) -> float:
        d = DIRECTIONS[direction]
        return off[0] * d[0] + off[1] * d[1]

    def residual(o: Option) -> float:
        d, t = o.vector, EXPECTED_TRAVEL[o.distance]
        return math.hypot(off[0] - d[0] * t, off[1] - d[1] * t)

    if turning:
        distance = "medium" if abs(turn) > math.radians(25) else "short"
        ranked = sorted((Option(d, distance, aim) for d in DIRECTIONS), key=lambda o: -toward(o.direction))
    else:
        ranked = sorted((Option(d, s, aim) for d in DIRECTIONS for s in DISTANCES),
                        key=lambda o: (residual(o), DISTANCES[o.distance]))
    for o in ranked:
        if is_feasible(sim, o):
            return o
    return next((o for o in all_options() if is_feasible(sim, o)), ranked[0])


def oracle_option(sim: CubeSim, goal) -> tuple[Option, dict[str, float]]:
    """Simulate every feasible option and keep the one leaving the lowest pose cost (ties: shorter push)."""
    scores: dict[str, float] = {}
    for o in all_options():
        if not is_feasible(sim, o):
            continue
        trial = sim.copy()
        execute_option(trial, o)
        scores[o.id] = pose_cost(trial.pose, goal, sim.shape)
    best = min(scores, key=lambda k: (scores[k], DISTANCES[Option.from_id(k).distance]))
    return Option.from_id(best), scores


def fallback_option(sim: CubeSim, goal) -> Option:
    """What the AI plays when Jev is unavailable: the one-step oracle (fast enough to run every push)."""
    return oracle_option(sim, goal)[0]

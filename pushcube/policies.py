"""Non-Jev decision rules: the geometric fallback and a one-step oracle baseline."""

from __future__ import annotations

import math

from .executor import execute_option, is_feasible
from .options import DIRECTIONS, DISTANCES, Option, all_options
from .scoring import overlap
from .sim import CubeSim, Vec

# Measured cube travel along an axis push for each distance (see executor tests / ledger).
EXPECTED_TRAVEL = {"short": 6.0, "medium": 26.0, "long": 75.0}


def geometric_option(sim: CubeSim, goal: Vec) -> Option:
    cube = sim.cube
    off = (goal[0] - cube[0], goal[1] - cube[1])

    def projection(direction: str) -> float:
        d = DIRECTIONS[direction]
        return off[0] * d[0] + off[1] * d[1]

    def residual(o: Option) -> float:
        d, t = o.vector, EXPECTED_TRAVEL[o.distance]
        return (off[0] - d[0] * t) ** 2 + (off[1] - d[1] * t) ** 2

    feasible = [o for o in all_options() if is_feasible(cube, o)]
    toward = [o for o in feasible if projection(o.direction) > 0] or feasible
    return min(toward, key=lambda o: (residual(o), DISTANCES[o.distance]))


def oracle_option(sim: CubeSim, goal: Vec) -> tuple[Option, dict[str, float]]:
    """Simulate every feasible option; best overlap after, ties (e.g. all 0 when far) by distance left."""
    scores: dict[str, float] = {}
    remaining: dict[str, float] = {}
    for o in all_options():
        if not is_feasible(sim.cube, o):
            continue
        trial = sim.copy()
        execute_option(trial, o)
        scores[o.id] = overlap(trial.cube, goal)
        remaining[o.id] = math.dist(trial.cube, goal)
    best = max(scores, key=lambda k: (scores[k], -remaining[k], -DISTANCES[Option.from_id(k).distance]))
    return Option.from_id(best), scores

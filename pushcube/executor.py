"""Turns a discrete push option into pusher targets and executes it on a CubeSim."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

from .options import Option
from .scoring import PushCounter
from .sim import CUBE_HALF, PUSHER_RADIUS, WALL_MAX, WALL_MIN, CubeSim, SimState, Vec

CONTACT_GAP = 2.0  # px between pusher and cube surface at the start of the push
APPROACH_BACKOFF = 20.0  # px behind the contact point where the approach ends
AVOID_HALF = CUBE_HALF + PUSHER_RADIUS + 3  # pusher centre must stay outside this box while approaching
ROUTE_HALF = CUBE_HALF + PUSHER_RADIUS + 17  # corners used to route around the cube
APPROACH_SPACING = 20.0  # px between approach targets (one per control step)
PUSH_SPACING = 10.0  # px between push targets
RETREAT = 10.0
PUSH_DWELL = 3  # control steps holding the end target so PD lag doesn't eat short pushes
SETTLE_STEPS = 3
PUSHER_MIN, PUSHER_MAX = WALL_MIN + PUSHER_RADIUS, WALL_MAX - PUSHER_RADIUS


@dataclass
class PushPlan:
    approach: list[Vec]
    push: list[Vec]
    retreat: list[Vec]


@dataclass
class ExecResult:
    contact_episodes: int
    steps: int
    stopped_early: bool


def _standoff(d: Vec) -> float:
    """Distance from cube centre to pusher centre at contact, along push direction d."""
    return CUBE_HALF * (abs(d[0]) + abs(d[1])) + PUSHER_RADIUS + CONTACT_GAP


def _points(cube: Vec, option: Option) -> tuple[Vec, Vec]:
    d = option.vector
    s = _standoff(d)
    contact = (cube[0] - d[0] * s, cube[1] - d[1] * s)
    approach = (contact[0] - d[0] * APPROACH_BACKOFF, contact[1] - d[1] * APPROACH_BACKOFF)
    return approach, contact


def is_feasible(cube: Vec, option: Option) -> bool:
    approach, _ = _points(cube, option)
    return all(PUSHER_MIN <= v <= PUSHER_MAX for v in approach)


def _segment_hits_box(a: Vec, b: Vec, c: Vec, h: float) -> bool:
    """Liang-Barsky: does segment a->b pass through the open box centred at c with half-size h?"""
    t0, t1 = 0.0, 1.0
    for i in range(2):
        p = b[i] - a[i]
        lo, hi = c[i] - h - a[i], c[i] + h - a[i]
        if abs(p) < 1e-9:
            if not (lo < 0 < hi):
                return False
            continue
        ta, tb = sorted((lo / p, hi / p))
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 >= t1:
            return False
    return t1 - t0 > 1e-9


def _clamp(p: Vec) -> Vec:
    return (min(max(p[0], PUSHER_MIN), PUSHER_MAX), min(max(p[1], PUSHER_MIN), PUSHER_MAX))


def _route(start: Vec, goal: Vec, cube: Vec) -> list[Vec]:
    """Shortest corner path from start to goal that never enters the avoid box."""
    corners = [_clamp((cube[0] + sx * ROUTE_HALF, cube[1] + sy * ROUTE_HALF)) for sx in (-1, 1) for sy in (-1, 1)]
    best, best_len = None, math.inf
    for k in range(3):
        for via in itertools.permutations(corners, k):
            path = [start, *via, goal]
            if any(_segment_hits_box(p, q, cube, AVOID_HALF) for p, q in itertools.pairwise(path)):
                continue
            length = sum(math.dist(p, q) for p, q in itertools.pairwise(path))
            if length < best_len:
                best, best_len = list(via) + [goal], length
    return best if best is not None else [goal]


def _escape(start: Vec, cube: Vec) -> list[Vec]:
    """If the pusher starts inside the avoid box, first move straight out along the nearest axis."""
    dx, dy = start[0] - cube[0], start[1] - cube[1]
    if abs(dx) >= AVOID_HALF or abs(dy) >= AVOID_HALF:
        return []
    if abs(dx) >= abs(dy):
        out = (cube[0] + math.copysign(ROUTE_HALF, dx or 1.0), start[1])
    else:
        out = (start[0], cube[1] + math.copysign(ROUTE_HALF, dy or 1.0))
    return [_clamp(out)]


def _densify(start: Vec, waypoints: list[Vec], spacing: float) -> list[Vec]:
    out, prev = [], start
    for w in waypoints:
        n = max(1, math.ceil(math.dist(prev, w) / spacing))
        out += [(prev[0] + (w[0] - prev[0]) * i / n, prev[1] + (w[1] - prev[1]) * i / n) for i in range(1, n + 1)]
        prev = w
    return out


def plan_push(state: SimState, option: Option) -> PushPlan:
    cube, start = state.cube, state.pusher
    approach_pt, contact = _points(cube, option)
    escape = _escape(start, cube)
    route_from = escape[-1] if escape else start
    waypoints = escape + _route(route_from, approach_pt, cube)
    approach = _densify(start, waypoints, APPROACH_SPACING)
    approach += [approach_pt] * 2  # let the PD controller settle before contact
    d = option.vector
    end = (contact[0] + d[0] * option.travel, contact[1] + d[1] * option.travel)
    push = _densify(approach_pt, [contact, end], PUSH_SPACING) + [end] * PUSH_DWELL
    # The cube only ever moves forward, so backing off behind the original contact point always
    # breaks contact, even when the cube moved less than planned.
    back = _clamp((contact[0] - d[0] * RETREAT, contact[1] - d[1] * RETREAT))
    retreat = [back] * (1 + SETTLE_STEPS)
    return PushPlan(approach=approach, push=push, retreat=retreat)


def push_targets(state: SimState, option: Option) -> list[Vec]:
    plan = plan_push(state, option)
    return plan.approach + plan.push + plan.retreat


def execute_option(
    sim: CubeSim, option: Option, on_step: Callable[[CubeSim], bool] | None = None
) -> ExecResult:
    counter = PushCounter(in_contact=sim.in_contact())
    steps = 0
    for target in push_targets(sim.snapshot(), option):
        sim.step(target)
        steps += 1
        counter.update(sim.in_contact())
        if on_step is not None and on_step(sim):
            return ExecResult(counter.count, steps, True)
    return ExecResult(counter.count, steps, False)

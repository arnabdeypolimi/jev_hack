"""Turns a discrete push option into pusher targets and executes it on a CubeSim.

Geometry uses the block's real (rotated) outline: the pusher travels along a straight line in the
push direction, offset sideways by the option's aim, and touches the block where that line first
meets the outline grown by the pusher radius.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass

from shapely.geometry import LineString, Point

from .options import AIMS, Option
from .scoring import PushCounter
from .shapes import bounding_radius, world_outline
from .sim import PUSHER_RADIUS, WALL_MAX, WALL_MIN, CubeSim, Vec

CONTACT_GAP = 2.0  # px between pusher and block surface at the start of the push
APPROACH_BACKOFF = 20.0  # px behind the contact point where the approach ends
AVOID_MARGIN = 3.0  # the approach keeps the pusher this far from the block
ROUTE_MARGIN = 20.0  # waypoints sit on a circle this far outside the block's bounding circle
APPROACH_SPACING = 20.0  # px between approach targets (one per control step)
PUSH_SPACING = 10.0  # px between push targets
RETREAT = 25.0
PUSH_DWELL = 3  # control steps holding the end target so PD lag doesn't eat short pushes
SETTLE_STEPS = 4
PUSHER_MIN, PUSHER_MAX = WALL_MIN + PUSHER_RADIUS, WALL_MAX - PUSHER_RADIUS
RAY = 1000.0


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


def _inside(p: Vec) -> bool:
    return PUSHER_MIN <= p[0] <= PUSHER_MAX and PUSHER_MIN <= p[1] <= PUSHER_MAX


def clamp_pusher(p: Vec) -> Vec:
    return (min(max(p[0], PUSHER_MIN), PUSHER_MAX), min(max(p[1], PUSHER_MIN), PUSHER_MAX))


def _contact(sim: CubeSim, option: Option, block) -> Vec | None:
    """Pusher centre at the moment it would touch the block, or None if the push line misses it."""
    (cx, cy), (dx, dy) = sim.cube, option.vector
    left = (dy, -dx)  # left of the direction of travel on a y-down screen
    lateral = [(x - cx) * left[0] + (y - cy) * left[1] for x, y in block.exterior.coords]
    frac = AIMS[option.aim]
    s = frac * (max(lateral) if frac > 0 else -min(lateral))
    ox, oy = cx + left[0] * s, cy + left[1] * s
    ray = LineString([(ox - dx * RAY, oy - dy * RAY), (ox + dx * RAY, oy + dy * RAY)])
    hits = ray.intersection(block.buffer(PUSHER_RADIUS + CONTACT_GAP, quad_segs=16).exterior)
    if hits.is_empty:
        return None
    points = [hits] if hits.geom_type == "Point" else list(getattr(hits, "geoms", []))
    start = Point(ox - dx * RAY, oy - dy * RAY)
    first = min(points, key=start.distance)
    return (first.x, first.y)


def _route(start: Vec, goal: Vec, centre: Vec, radius: float, avoid) -> list[Vec] | None:
    """Shortest path via up to two adjacent waypoints on a circle around the block, never entering `avoid`."""
    ring = [clamp_pusher((centre[0] + radius * math.cos(k * math.pi / 4), centre[1] + radius * math.sin(k * math.pi / 4)))
            for k in range(8)]
    candidates = [[]] + [[w] for w in ring] + [[ring[k], ring[(k + j) % 8]] for k in range(8) for j in (1, -1)]
    best, best_len = None, math.inf
    for via in candidates:
        path = [start, *via, goal]
        if LineString(path).intersects(avoid):
            continue
        length = sum(math.dist(p, q) for p, q in itertools.pairwise(path))
        if length < best_len:
            best, best_len = [*via, goal], length
    return best


def _densify(start: Vec, waypoints: list[Vec], spacing: float) -> list[Vec]:
    out, prev = [], start
    for w in waypoints:
        n = max(1, math.ceil(math.dist(prev, w) / spacing))
        out += [(prev[0] + (w[0] - prev[0]) * i / n, prev[1] + (w[1] - prev[1]) * i / n) for i in range(1, n + 1)]
        prev = w
    return out


def plan_push(sim: CubeSim, option: Option) -> PushPlan | None:
    """Pusher targets for one push, or None when the push cannot be set up from here."""
    block = world_outline(sim.shape, sim.pose)
    contact = _contact(sim, option, block)
    if contact is None:
        return None
    d = option.vector
    approach_pt = (contact[0] - d[0] * APPROACH_BACKOFF, contact[1] - d[1] * APPROACH_BACKOFF)
    if not (_inside(contact) and _inside(approach_pt)):
        return None
    avoid = block.buffer(PUSHER_RADIUS + AVOID_MARGIN, quad_segs=8)
    start, centre = sim.pusher, sim.cube
    radius = bounding_radius(sim.shape) + PUSHER_RADIUS + ROUTE_MARGIN
    escape: list[Vec] = []
    if avoid.contains(Point(start)):  # still hugging the block after the last push: back straight out
        ux, uy = start[0] - centre[0], start[1] - centre[1]
        n = math.hypot(ux, uy) or 1.0
        escape = [clamp_pusher((centre[0] + ux / n * radius, centre[1] + uy / n * radius))]
    route = _route(escape[-1] if escape else start, approach_pt, centre, radius, avoid)
    if route is None:
        return None
    approach = _densify(start, escape + route, APPROACH_SPACING) + [approach_pt] * 2
    end = (contact[0] + d[0] * option.travel, contact[1] + d[1] * option.travel)
    push = _densify(approach_pt, [contact, end], PUSH_SPACING) + [clamp_pusher(end)] * PUSH_DWELL
    # Back off behind the original contact point: the block only moves away, so contact always breaks.
    back = clamp_pusher((contact[0] - d[0] * RETREAT, contact[1] - d[1] * RETREAT))
    return PushPlan(approach=approach, push=push, retreat=[back] * (1 + SETTLE_STEPS))


def is_feasible(sim: CubeSim, option: Option) -> bool:
    return plan_push(sim, option) is not None


def push_targets(sim: CubeSim, option: Option) -> list[Vec]:
    plan = plan_push(sim, option)
    return [] if plan is None else plan.approach + plan.push + plan.retreat


def execute_option(
    sim: CubeSim, option: Option, on_step: Callable[[CubeSim], bool] | None = None
) -> ExecResult:
    counter = PushCounter(in_contact=sim.in_contact())
    steps = 0
    for target in push_targets(sim, option):
        sim.step(target)
        steps += 1
        counter.observe(sim)
        if on_step is not None and on_step(sim):
            return ExecResult(counter.count, steps, True)
    return ExecResult(counter.count, steps, False)

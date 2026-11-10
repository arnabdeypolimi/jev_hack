"""Block pushing simulator (2D pymunk, top-down, y points down): the block slides and spins."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

from .shapes import local_polygons

BOARD = 512
WALL_MIN, WALL_MAX = 5.0, 506.0
PUSHER_RADIUS = 15.0

SIM_HZ = 100
CONTROL_HZ = 10
SUBSTEPS = SIM_HZ // CONTROL_HZ
K_P, K_V = 400.0, 40.0  # critically damped, 2x faster than Push-T so short pushes land
FRICTION_DECEL = 3000.0  # px/s^2 of table friction on the block, so it stops shortly after a push
ANGULAR_DECEL = 60.0  # rad/s^2 of table friction on the block's spin

Vec = tuple[float, float]


@dataclass(frozen=True)
class SimState:
    pusher: Vec
    cube: Vec  # block centroid position (the name predates other shapes)
    pusher_vel: Vec = (0.0, 0.0)
    cube_vel: Vec = (0.0, 0.0)
    angle: float = 0.0
    ang_vel: float = 0.0


class CubeSim:
    """One table with a pusher and a block of the given shape; the block slides and spins."""

    def __init__(self, shape: str = "square") -> None:
        self.shape = shape
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        r = 2.0
        lo, hi = WALL_MIN - r, WALL_MAX + r
        walls = [
            pymunk.Segment(self.space.static_body, (lo, lo), (hi, lo), r),
            pymunk.Segment(self.space.static_body, (hi, lo), (hi, hi), r),
            pymunk.Segment(self.space.static_body, (hi, hi), (lo, hi), r),
            pymunk.Segment(self.space.static_body, (lo, hi), (lo, lo), r),
        ]
        for w in walls:
            w.friction = 0.0
        parts = local_polygons(shape)
        areas = [abs(pymunk.Poly(None, p).area) for p in parts]
        total = sum(areas)
        moment = sum(pymunk.moment_for_poly(a / total, p) for a, p in zip(areas, parts))
        self._cube_body = pymunk.Body(1.0, moment)
        self._block_shapes = [pymunk.Poly(self._cube_body, p) for p in parts]
        for bs in self._block_shapes:
            bs.friction = 0.0
        self._pusher_body = pymunk.Body(10.0, math.inf)
        pusher_shape = pymunk.Circle(self._pusher_body, PUSHER_RADIUS)
        pusher_shape.friction = 0.0
        self.space.add(*walls, self._cube_body, *self._block_shapes, self._pusher_body, pusher_shape)
        self.steps = 0
        self.reset(SimState(pusher=(256.0, 400.0), cube=(256.0, 256.0)))

    @property
    def pusher(self) -> Vec:
        return tuple(self._pusher_body.position)

    @property
    def cube(self) -> Vec:
        return tuple(self._cube_body.position)

    @property
    def angle(self) -> float:
        return float(self._cube_body.angle)

    @property
    def pose(self) -> tuple[float, float, float]:
        return (*self.cube, self.angle)

    def reset(self, state: SimState) -> None:
        self.restore(state)
        self.steps = 0

    def snapshot(self) -> SimState:
        return SimState(
            pusher=self.pusher,
            cube=self.cube,
            pusher_vel=tuple(self._pusher_body.velocity),
            cube_vel=tuple(self._cube_body.velocity),
            angle=self.angle,
            ang_vel=float(self._cube_body.angular_velocity),
        )

    def restore(self, s: SimState) -> None:
        self._pusher_body.position, self._pusher_body.velocity = s.pusher, s.pusher_vel
        self._cube_body.position, self._cube_body.velocity = s.cube, s.cube_vel
        self._cube_body.angle, self._cube_body.angular_velocity = s.angle, s.ang_vel
        self.space.reindex_shapes_for_body(self._pusher_body)
        self.space.reindex_shapes_for_body(self._cube_body)

    def copy(self) -> CubeSim:
        other = CubeSim(self.shape)
        other.restore(self.snapshot())
        other.steps = self.steps
        return other

    def step(self, target: Vec | None) -> None:
        """One control step (1/10 s): PD-drive the pusher toward `target` (None = hold)."""
        dt = 1.0 / SIM_HZ
        body = self._pusher_body
        hold = body.position if target is None else pymunk.Vec2d(*target)
        for _ in range(SUBSTEPS):
            acc = K_P * (hold - body.position) - K_V * body.velocity
            body.velocity = body.velocity + acc * dt
            self._apply_table_friction(dt)
            self.space.step(dt)
        self.steps += 1

    def _apply_table_friction(self, dt: float) -> None:
        body = self._cube_body
        speed = body.velocity.length
        if speed:
            body.velocity = body.velocity * (max(0.0, speed - FRICTION_DECEL * dt) / speed)
        w = body.angular_velocity
        if w:
            body.angular_velocity = math.copysign(max(0.0, abs(w) - ANGULAR_DECEL * dt), w)

    def gap(self) -> float:
        """Distance between the pusher's edge and the block's surface (negative when overlapping)."""
        p = self.pusher
        return min(bs.point_query(p).distance for bs in self._block_shapes) - PUSHER_RADIUS

    def in_contact(self, eps: float = 1.0) -> bool:
        return self.gap() <= eps

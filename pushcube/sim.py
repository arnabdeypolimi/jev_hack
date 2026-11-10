"""Translation-only cube pushing simulator (2D pymunk, top-down, y points down)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

BOARD = 512
WALL_MIN, WALL_MAX = 5.0, 506.0
CUBE_SIZE = 60.0
CUBE_HALF = CUBE_SIZE / 2
PUSHER_RADIUS = 15.0

SIM_HZ = 100
CONTROL_HZ = 10
SUBSTEPS = SIM_HZ // CONTROL_HZ
K_P, K_V = 100.0, 20.0
FRICTION_DECEL = 3000.0  # px/s^2 of table friction on the cube, so it stops shortly after a push

Vec = tuple[float, float]


@dataclass(frozen=True)
class SimState:
    pusher: Vec
    cube: Vec
    pusher_vel: Vec = (0.0, 0.0)
    cube_vel: Vec = (0.0, 0.0)


class CubeSim:
    def __init__(self) -> None:
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
        self._cube_body = pymunk.Body(1.0, math.inf)
        cube_shape = pymunk.Poly.create_box(self._cube_body, (CUBE_SIZE, CUBE_SIZE))
        cube_shape.friction = 0.0
        self._pusher_body = pymunk.Body(10.0, math.inf)
        pusher_shape = pymunk.Circle(self._pusher_body, PUSHER_RADIUS)
        pusher_shape.friction = 0.0
        self.space.add(*walls, self._cube_body, cube_shape, self._pusher_body, pusher_shape)
        self.steps = 0
        self.reset(SimState(pusher=(256.0, 400.0), cube=(256.0, 256.0)))

    @property
    def pusher(self) -> Vec:
        return tuple(self._pusher_body.position)

    @property
    def cube(self) -> Vec:
        return tuple(self._cube_body.position)

    def reset(self, state: SimState) -> None:
        self.restore(state)
        self.steps = 0

    def snapshot(self) -> SimState:
        return SimState(
            pusher=self.pusher,
            cube=self.cube,
            pusher_vel=tuple(self._pusher_body.velocity),
            cube_vel=tuple(self._cube_body.velocity),
        )

    def restore(self, s: SimState) -> None:
        self._pusher_body.position, self._pusher_body.velocity = s.pusher, s.pusher_vel
        self._cube_body.position, self._cube_body.velocity = s.cube, s.cube_vel
        self._cube_body.angle = 0.0
        self.space.reindex_shapes_for_body(self._pusher_body)
        self.space.reindex_shapes_for_body(self._cube_body)

    def copy(self) -> CubeSim:
        other = CubeSim()
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
        v = self._cube_body.velocity
        speed = v.length
        if speed == 0.0:
            return
        self._cube_body.velocity = v * (max(0.0, speed - FRICTION_DECEL * dt) / speed)

    def in_contact(self, eps: float = 1.0) -> bool:
        (px, py), (cx, cy) = self.pusher, self.cube
        dx = max(abs(px - cx) - CUBE_HALF, 0.0)
        dy = max(abs(py - cy) - CUBE_HALF, 0.0)
        return math.hypot(dx, dy) <= PUSHER_RADIUS + eps

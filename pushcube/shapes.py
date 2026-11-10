"""Block shapes: convex parts in local coordinates centred on the centroid (y points down)."""

from __future__ import annotations

import math
from functools import cache

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

Vec = tuple[float, float]
Pose = tuple[float, float, float]


def _rect(x0: float, y0: float, x1: float, y1: float) -> list[Vec]:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


_RAW: dict[str, list[list[Vec]]] = {
    "square": [_rect(-30, -30, 30, 30)],
    "T": [_rect(-60, 0, 60, 30), _rect(-15, 30, 15, 120)],
    "L": [_rect(0, 0, 30, 90), _rect(30, 60, 75, 90)],
    "I": [_rect(-60, -15, 60, 15)],
    "Z": [_rect(-45, -30, 15, 0), _rect(-15, 0, 45, 30)],
    "triangle": [[(0.0, -46.188), (40.0, 23.094), (-40.0, 23.094)]],
}
_SYMMETRY = {"square": math.pi / 2, "T": 2 * math.pi, "L": 2 * math.pi, "I": math.pi, "Z": math.pi,
             "triangle": 2 * math.pi / 3}
SHAPES = tuple(_RAW)


@cache
def local_polygons(name: str) -> list[list[Vec]]:
    c = unary_union([Polygon(p) for p in _RAW[name]]).centroid
    return [[(round(x - c.x, 6) + 0.0, round(y - c.y, 6) + 0.0) for x, y in part] for part in _RAW[name]]


@cache
def outline(name: str) -> Polygon:
    return unary_union([Polygon(p) for p in local_polygons(name)])


@cache
def bounding_radius(name: str) -> float:
    return max(math.hypot(x, y) for x, y in outline(name).exterior.coords)


def symmetry(name: str) -> float:
    return _SYMMETRY[name]


def world_outline(name: str, pose: Pose) -> Polygon:
    x, y, theta = pose
    g = affinity.rotate(outline(name), theta, origin=(0, 0), use_radians=True)
    return affinity.translate(g, x, y)


def angle_error(angle: float, goal_angle: float, name: str) -> float:
    """Signed turn still needed (positive = clockwise on screen), using the shape's symmetry."""
    period = symmetry(name)
    return (goal_angle - angle + period / 2) % period - period / 2

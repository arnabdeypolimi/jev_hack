import math

import pytest

from pushcube.shapes import SHAPES, angle_error, local_polygons, outline, symmetry, world_outline


def test_shapes_available_and_centred_on_centroid():
    assert set(SHAPES) == {"square", "T", "L", "I", "Z", "triangle"}
    for name in SHAPES:
        c = outline(name).centroid
        assert c.x == pytest.approx(0, abs=1e-6) and c.y == pytest.approx(0, abs=1e-6)


def test_square_matches_old_cube():
    assert outline("square").area == pytest.approx(3600)
    assert local_polygons("square") == [[(-30, -30), (30, -30), (30, 30), (-30, 30)]]


def test_world_outline_rotates_like_pymunk():
    # pymunk local_to_world: x' = x cos - y sin; a quarter turn maps local +x to world +y (down on screen)
    g = world_outline("I", (100, 100, math.pi / 2))
    minx, miny, maxx, maxy = g.bounds
    assert maxx - minx == pytest.approx(30) and maxy - miny == pytest.approx(120)


def test_symmetry():
    assert symmetry("square") == pytest.approx(math.pi / 2)
    assert symmetry("I") == pytest.approx(math.pi)
    assert symmetry("T") == pytest.approx(2 * math.pi)
    assert symmetry("Z") == pytest.approx(math.pi)
    assert symmetry("triangle") == pytest.approx(2 * math.pi / 3)


def test_angle_error_uses_symmetry_and_sign():
    assert angle_error(0.0, math.radians(30), "T") == pytest.approx(math.radians(30))  # needs clockwise
    assert angle_error(0.0, math.radians(-30), "T") == pytest.approx(math.radians(-30))
    assert angle_error(0.0, math.radians(350), "T") == pytest.approx(math.radians(-10))
    assert angle_error(0.0, math.radians(100), "square") == pytest.approx(math.radians(10))
    assert angle_error(0.0, math.radians(170), "I") == pytest.approx(math.radians(-10))

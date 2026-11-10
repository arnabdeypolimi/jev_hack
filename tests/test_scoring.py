import pytest

from pushcube.options import DIRECTIONS, Option, all_options
from pushcube.scoring import PushCounter, clamp_goal, overlap


def test_overlap_values():
    assert overlap((100, 100), (100, 100)) == 1.0
    assert overlap((106, 100), (100, 100)) == pytest.approx(0.9)
    assert overlap((200, 100), (100, 100)) == 0.0


def test_overlap_is_symmetric_in_both_axes():
    assert overlap((103, 97), (100, 100)) == pytest.approx(57 * 57 / 3600)


def test_clamp_goal():
    assert clamp_goal((0, 600)) == (35, 476)
    assert clamp_goal((200, 300)) == (200, 300)


def test_push_counter_counts_contact_episodes():
    c = PushCounter()
    for x in [False, True, True, False, False, False, True, False]:  # 3 steps away (0.3 s) ends a push
        c.update(x)
    assert c.count == 2


def test_push_counter_ongoing_contact_at_start_not_counted():
    c = PushCounter(in_contact=True)
    for x in [True, False, False, False, True]:
        c.update(x)
    assert c.count == 1


def test_all_options_and_ids():
    opts = all_options()
    assert len(opts) == 72 and len(set(opts)) == 72
    assert Option.from_id("SW-long") == Option("SW", "long", "center")
    assert Option("NE", "medium").id == "NE-medium"
    assert Option("NE", "medium", "left").id == "NE-medium-left"
    assert Option.from_id("NE-medium-left") == Option("NE", "medium", "left")


def test_directions_are_unit_vectors_with_north_up():
    assert DIRECTIONS["N"] == (0.0, -1.0)
    for v in DIRECTIONS.values():
        assert (v[0] ** 2 + v[1] ** 2) == pytest.approx(1.0)


def test_overlap_with_angles_and_shapes():
    import math
    assert overlap((100, 100, math.pi / 2), (100, 100, 0.0)) == pytest.approx(1.0)  # square symmetry
    assert overlap((100, 100, math.pi / 4), (100, 100, 0.0)) < 0.9
    assert overlap((200, 200, 0.0), (200, 200, 0.0), "T") == pytest.approx(1.0)
    assert overlap((200, 200, math.pi), (200, 200, 0.0), "T") < 0.6
    assert overlap((200, 200, math.pi), (200, 200, 0.0), "I") == pytest.approx(1.0)


def test_clamp_goal_keeps_rotated_shape_inside_walls():
    import math
    from pushcube.shapes import world_outline
    g = clamp_goal((500, 20, math.pi / 3), "T")
    assert len(g) == 3 and g[2] == pytest.approx(math.pi / 3)
    minx, miny, maxx, maxy = world_outline("T", g).bounds
    assert minx >= 5 - 1e-6 and miny >= 5 - 1e-6 and maxx <= 506 + 1e-6 and maxy <= 506 + 1e-6


def test_push_counter_ignores_brief_separation_while_still_near():
    c = PushCounter()
    # (contact, near): touch, drift a few px (still near), touch, swing away briefly, touch, leave for 0.3 s, touch
    seq = [(True, True), (False, True), (True, True), (False, False), (False, False), (True, True),
           (False, False), (False, False), (False, False), (True, True)]
    for contact, near in seq:
        c.update(contact, near)
    assert c.count == 2


def test_push_counter_observe_uses_sim_gap():
    from pushcube.sim import CubeSim, SimState
    s = CubeSim()
    s.reset(SimState(pusher=(256 - 45, 256), cube=(256, 256)))
    c = PushCounter()
    c.observe(s)
    s.reset(SimState(pusher=(256 - 50, 256), cube=(256, 256)))  # 5 px gap: separated but still near
    c.observe(s)
    s.reset(SimState(pusher=(256 - 45, 256), cube=(256, 256)))
    c.observe(s)
    assert c.count == 1 and abs(s.gap()) < 1e-6

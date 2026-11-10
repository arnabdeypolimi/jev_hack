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
    for x in [False, True, True, False, True, False]:
        c.update(x)
    assert c.count == 2


def test_push_counter_ongoing_contact_at_start_not_counted():
    c = PushCounter(in_contact=True)
    for x in [True, False, True]:
        c.update(x)
    assert c.count == 1


def test_all_options_and_ids():
    opts = all_options()
    assert len(opts) == 24 and len(set(opts)) == 24
    assert Option.from_id("SW-long") == Option("SW", "long")
    assert Option("NE", "medium").id == "NE-medium"


def test_directions_are_unit_vectors_with_north_up():
    assert DIRECTIONS["N"] == (0.0, -1.0)
    for v in DIRECTIONS.values():
        assert (v[0] ** 2 + v[1] ** 2) == pytest.approx(1.0)

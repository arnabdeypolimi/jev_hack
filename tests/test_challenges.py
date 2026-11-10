from pushcube.challenges import PRESETS, get_challenge
from pushcube.scoring import SUCCESS_OVERLAP, clamp_goal, overlap
from pushcube.shapes import SHAPES


def test_presets_cover_tiers_and_shapes_and_goals_are_clamped():
    assert {c.tier for c in PRESETS} == {"easy", "medium", "hard", "far"}
    for tier in ("easy", "medium", "hard", "far"):
        assert sum(c.tier == tier for c in PRESETS) >= 2
    assert {c.shape for c in PRESETS} == set(SHAPES)
    assert all(len(c.goal) == 3 for c in PRESETS)
    assert all(clamp_goal(c.goal, c.shape) == c.goal for c in PRESETS)


def test_presets_start_unsolved_and_ids_unique():
    assert len({c.id for c in PRESETS}) == len(PRESETS)
    assert all(overlap((*c.start.cube, c.start.angle), c.goal, c.shape) < SUCCESS_OVERLAP for c in PRESETS)


def test_some_presets_need_turning():
    assert sum(abs(c.goal[2] - c.start.angle) > 0.3 for c in PRESETS) >= 4


def test_get_challenge_by_id():
    assert get_challenge(PRESETS[0].id) is PRESETS[0]
    assert get_challenge("nope") is None

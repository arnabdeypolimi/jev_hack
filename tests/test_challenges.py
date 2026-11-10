from pushcube.challenges import PRESETS, get_challenge
from pushcube.scoring import SUCCESS_OVERLAP, clamp_goal, overlap


def test_presets_cover_tiers_and_goals_are_clamped():
    assert {c.tier for c in PRESETS} == {"easy", "medium", "hard", "far"}
    for tier in ("easy", "medium", "hard", "far"):
        assert sum(c.tier == tier for c in PRESETS) >= 2
    assert all(clamp_goal(c.goal) == c.goal for c in PRESETS)


def test_presets_start_unsolved_and_ids_unique():
    assert len({c.id for c in PRESETS}) == len(PRESETS)
    assert all(overlap(c.start.cube, c.goal) < SUCCESS_OVERLAP for c in PRESETS)


def test_get_challenge_by_id():
    assert get_challenge(PRESETS[0].id) is PRESETS[0]
    assert get_challenge("nope") is None

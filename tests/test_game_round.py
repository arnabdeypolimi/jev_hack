import pytest

from pushcube.challenges import get_challenge
from pushcube.controller import SideResult
from pushcube.game_round import Round, clamp_mouse, decide_winner, human_target
from pushcube.sim import SimState


@pytest.mark.parametrize(
    "h,a,w",
    [
        (SideResult(12, True, 0.93, 40), SideResult(9, True, 0.91, 50), "ai"),
        (SideResult(7, True, 0.95, 30), SideResult(9, True, 0.91, 50), "human"),
        (SideResult(9, True, 0.9, 30), SideResult(9, True, 0.9, 50), "tie"),
        (SideResult(3, False, 0.8, None), SideResult(20, True, 0.9, 80), "ai"),
        (SideResult(3, False, 0.8, None), SideResult(4, False, 0.6, None), "human"),
        (SideResult(3, False, 0.6, None), SideResult(4, False, 0.603, None), "tie"),
    ],
)
def test_decide_winner(h, a, w):
    assert decide_winner(h, a) == w


def test_mouse_target_clamped():
    assert clamp_mouse(-50, 900) == (20, 491)
    assert clamp_mouse(100, 200) == (100, 200)


def test_human_target_only_when_down():
    assert human_target((600, 100), True) == (491, 100)
    assert human_target((100, 100), False) is None
    assert human_target(None, True) is None


def test_round_starts_both_sims_identically():
    c = get_challenge("hard-t-turn")
    r = Round(1, c.id, c.goal, c.start, shape=c.shape)
    assert r.human_sim.shape == r.ai_sim.shape == "T" and r.shape == "T"
    assert r.human_sim.cube == r.ai_sim.cube == c.start.cube
    assert r.human_sim.pusher == r.ai_sim.pusher == c.start.pusher
    assert r.human_sim is not r.ai_sim


def test_human_push_counted_once_per_contact():
    r = Round(1, None, (400, 256), SimState(pusher=(150, 256), cube=(256, 256)))
    for x in (200, 215, 230, 240):  # walk into the cube's west face and push
        r.human_step((x, 256), True)
    for _ in range(4):
        r.human_step((120, 256), True)  # back off
    for x in (200, 230, 250, 260):
        r.human_step((x, 256), True)  # second push
    assert r.human_pushes.count == 2


def test_human_result_tracks_reaching_goal_and_freezes_after():
    r = Round(1, None, (262, 256), SimState(pusher=(190, 256), cube=(250, 256)))
    for x in (215, 225, 235, 240, 245):
        r.human_step((x, 256), True)
    res = r.human_result()
    assert res.reached and res.finished_at_s is not None and res.pushes == 1
    cube = r.human_sim.cube
    r.human_step((300, 256), True)
    assert r.human_sim.cube == cube  # no more play once the goal is reached

import pytest

from pushcube.challenges import PRESETS
from pushcube.executor import execute_option, is_feasible
from pushcube.policies import geometric_option, oracle_option
from pushcube.scoring import SUCCESS_OVERLAP, overlap
from pushcube.sim import CubeSim, SimState

POLICIES = {"geometric": geometric_option, "oracle": lambda sim, goal: oracle_option(sim, goal)[0]}


def play(policy, challenge, max_pushes=25):
    sim = CubeSim()
    sim.reset(challenge.start)
    pushes = 0
    while overlap(sim.cube, challenge.goal) < SUCCESS_OVERLAP and pushes < max_pushes:
        execute_option(sim, POLICIES[policy](sim, challenge.goal))
        pushes += 1
    return overlap(sim.cube, challenge.goal) >= SUCCESS_OVERLAP, pushes


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("challenge", PRESETS, ids=lambda c: c.id)
def test_policy_solves_every_preset_within_25_pushes(policy, challenge):
    reached, pushes = play(policy, challenge)
    assert reached, f"{policy} failed {challenge.id} after {pushes} pushes"


def test_geometric_picks_direction_toward_goal():
    sim = CubeSim()
    sim.reset(SimState(pusher=(60, 60), cube=(200, 256)))
    assert geometric_option(sim, (400, 256)).id == "E-long"
    assert geometric_option(sim, (200, 230)).direction == "N"


def test_geometric_only_returns_feasible_options():
    sim = CubeSim()
    sim.reset(SimState(pusher=(300, 256), cube=(476, 256)))  # cube against the east wall
    opt = geometric_option(sim, (300, 256))                    # goal is west; W is infeasible
    assert is_feasible(sim.cube, opt)


@pytest.mark.parametrize("cube,goal", [((200, 200), (300, 260)), ((300, 300), (120, 330)), ((256, 256), (262, 240))])
def test_oracle_never_worse_than_geometric_on_one_step(cube, goal):
    def after(option):
        s = CubeSim()
        s.reset(SimState(pusher=(60, 60), cube=cube))
        execute_option(s, option)
        return overlap(s.cube, goal)

    s = CubeSim()
    s.reset(SimState(pusher=(60, 60), cube=cube))
    best, scores = oracle_option(s, goal)
    assert after(best) >= after(geometric_option(s, goal)) - 1e-9
    assert scores[best.id] == pytest.approx(max(scores.values()))
    assert s.cube == pytest.approx(cube)  # oracle simulates on copies only

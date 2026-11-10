import math

import pytest

from pushcube.challenges import PRESETS
from pushcube.executor import execute_option, is_feasible
from pushcube.policies import fallback_option, geometric_option, oracle_option, pose_cost
from pushcube.scoring import SUCCESS_OVERLAP, overlap
from pushcube.sim import CubeSim, SimState

POLICIES = {"geometric": geometric_option, "oracle": lambda sim, goal: oracle_option(sim, goal)[0]}


def play(policy, challenge, max_pushes=40):
    sim = CubeSim(challenge.shape)
    sim.reset(challenge.start)
    pushes = 0
    while overlap(sim.pose, challenge.goal, sim.shape) < SUCCESS_OVERLAP and pushes < max_pushes:
        execute_option(sim, POLICIES[policy](sim, challenge.goal))
        pushes += 1
    return overlap(sim.pose, challenge.goal, sim.shape) >= SUCCESS_OVERLAP, pushes


@pytest.mark.parametrize("challenge", PRESETS, ids=lambda c: c.id)
def test_oracle_solves_every_preset_within_40_pushes(challenge):
    reached, pushes = play("oracle", challenge)
    assert reached, f"oracle failed {challenge.id} after {pushes} pushes"


@pytest.mark.parametrize("challenge", [c for c in PRESETS if c.tier != "hard"], ids=lambda c: c.id)
def test_geometric_rule_solves_non_hard_presets(challenge):
    reached, pushes = play("geometric", challenge)
    assert reached, f"geometric failed {challenge.id} after {pushes} pushes"


def test_fallback_is_the_oracle_choice():
    s = CubeSim("T")
    s.reset(SimState(pusher=(60, 460), cube=(190, 200)))
    assert fallback_option(s, (310, 300, math.radians(90))) == oracle_option(s, (310, 300, math.radians(90)))[0]


def test_geometric_pushes_straight_toward_a_far_goal():
    sim = CubeSim()
    sim.reset(SimState(pusher=(60, 60), cube=(200, 256)))
    assert geometric_option(sim, (400, 256, 0.0)).id == "E-long"


def test_geometric_aims_off_centre_to_turn():
    sim = CubeSim("T")
    sim.reset(SimState(pusher=(60, 60), cube=(256, 256)))
    assert geometric_option(sim, (256, 256, math.radians(40))).aim == "left"  # clockwise
    assert geometric_option(sim, (256, 256, math.radians(-40))).aim == "right"


def test_geometric_only_returns_feasible_options():
    sim = CubeSim()
    sim.reset(SimState(pusher=(300, 256), cube=(476, 256)))
    assert is_feasible(sim, geometric_option(sim, (300, 256, 0.0)))


def test_pose_cost_combines_distance_and_turn():
    assert pose_cost((0, 0, 0), (30, 40, 0), "T") == pytest.approx(50)
    assert pose_cost((0, 0, 0), (0, 0, math.radians(10)), "T") > 0


def test_oracle_simulates_on_copies_and_picks_lowest_cost():
    s = CubeSim("L")
    s.reset(SimState(pusher=(60, 60), cube=(200, 200), angle=0.2))
    best, scores = oracle_option(s, (300, 260, 0.6))
    assert scores[best.id] == pytest.approx(min(scores.values()))
    assert s.pose == pytest.approx((200, 200, 0.2))

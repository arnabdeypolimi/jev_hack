import math

import numpy as np
import pytest

from pushcube.executor import execute_option, is_feasible, plan_push, push_targets
from pushcube.options import DIRECTIONS, DISTANCES, Option, all_options
from pushcube.sim import CubeSim, SimState

SHAPES = ["square", "T", "L", "I", "Z", "triangle"]


def sim_at(shape="square", pusher=(60, 60), cube=(256, 256), angle=0.0):
    s = CubeSim(shape)
    s.reset(SimState(pusher=pusher, cube=cube, angle=angle))
    return s


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("opt", all_options(), ids=lambda o: o.id)
def test_every_feasible_option_is_exactly_one_push(shape, opt):
    s = sim_at(shape, angle=0.4)
    if not is_feasible(s, opt):
        pytest.skip("infeasible here")
    assert execute_option(s, opt).contact_episodes == 1


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("direction", list(DIRECTIONS))
def test_centre_push_moves_block_along_direction(shape, direction):
    s = sim_at(shape)
    opt = Option(direction, "long", "center")
    before = np.array(s.cube)
    execute_option(s, opt)
    assert (np.array(s.cube) - before) @ np.array(DIRECTIONS[direction]) > 15


def test_left_aim_turns_clockwise_right_aim_counterclockwise():
    left, right = sim_at(), sim_at()
    execute_option(left, Option("E", "long", "left"))
    execute_option(right, Option("E", "long", "right"))
    assert left.angle > 0.1 and right.angle < -0.1


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("opt", [o for o in all_options() if o.distance == "medium"], ids=lambda o: o.id)
def test_approach_never_touches_block_before_push(shape, opt):
    s = sim_at(shape, pusher=(256, 480), angle=0.7)
    plan = plan_push(s, opt)
    if plan is None:
        pytest.skip("infeasible here")
    for t in plan.approach:
        s.step(t)
        assert not s.in_contact(), f"touched during approach for {opt.id}"


def test_push_targets_is_concatenation_of_phases():
    s = sim_at()
    plan = plan_push(s, Option("E", "medium"))
    assert push_targets(s, Option("E", "medium")) == plan.approach + plan.push + plan.retreat


def test_infeasible_when_block_against_wall():
    s = sim_at(cube=(476, 256), pusher=(300, 100))
    assert not is_feasible(s, Option("W", "short"))  # would need to approach from beyond the east wall
    assert is_feasible(s, Option("N", "short"))
    assert plan_push(s, Option("W", "short")) is None


def test_execute_stops_early_when_on_step_true():
    s = sim_at()
    calls = []

    def on_step(sim):
        calls.append(1)
        return len(calls) == 3

    r = execute_option(s, Option("E", "long"), on_step=on_step)
    assert r.stopped_early and r.steps == 3


def test_ongoing_contact_at_start_is_not_counted_twice():
    s = sim_at(pusher=(256 - 45, 256))  # already touching the west face
    assert execute_option(s, Option("E", "short")).contact_episodes == 1


def test_infeasible_option_executes_nothing():
    s = sim_at(cube=(476, 256), pusher=(300, 100))
    r = execute_option(s, Option("W", "short"))
    assert r.steps == 0 and r.contact_episodes == 0 and s.steps == 0

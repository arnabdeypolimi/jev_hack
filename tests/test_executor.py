import numpy as np
import pytest

from pushcube.executor import execute_option, is_feasible, plan_push, push_targets
from pushcube.options import DIRECTIONS, DISTANCES, Option, all_options
from pushcube.sim import CubeSim, SimState


@pytest.mark.parametrize("opt", all_options(), ids=lambda o: o.id)
def test_execute_option_is_one_contact_episode_and_moves_cube_in_direction(opt):
    s = CubeSim()
    s.reset(SimState(pusher=(60, 60), cube=(256, 256)))
    assert is_feasible(s.cube, opt)
    before = np.array(s.cube)
    r = execute_option(s, opt)
    moved = np.array(s.cube) - before
    assert r.contact_episodes == 1
    assert moved @ np.array(DIRECTIONS[opt.direction]) > 0.25 * DISTANCES[opt.distance]


@pytest.mark.parametrize("opt", all_options(), ids=lambda o: o.id)
def test_approach_never_touches_cube_before_push(opt):
    s = CubeSim()
    start = SimState(pusher=(256, 470), cube=(256, 256))  # pusher starts south of the cube
    s.reset(start)
    plan = plan_push(start, opt)
    for t in plan.approach:
        s.step(t)
        assert not s.in_contact(), f"touched during approach for {opt.id}"


def test_push_targets_is_concatenation_of_phases():
    st = SimState(pusher=(60, 60), cube=(256, 256))
    plan = plan_push(st, Option("E", "medium"))
    assert push_targets(st, Option("E", "medium")) == plan.approach + plan.push + plan.retreat


def test_infeasible_when_cube_against_wall():
    cube = (476, 256)  # touching the east wall
    assert not is_feasible(cube, Option("W", "short"))  # approach point east of the cube is outside
    assert is_feasible(cube, Option("N", "short"))
    assert is_feasible(cube, Option("E", "short"))  # approach is fine; the wall simply blocks the cube


def test_execute_stops_early_when_on_step_true():
    s = CubeSim()
    s.reset(SimState(pusher=(60, 60), cube=(256, 256)))
    calls = []

    def on_step(sim):
        calls.append(1)
        return len(calls) == 3

    r = execute_option(s, Option("E", "long"), on_step=on_step)
    assert r.stopped_early and r.steps == 3


def test_ongoing_contact_at_start_is_not_counted_twice():
    s = CubeSim()
    s.reset(SimState(pusher=(256 - 45, 256), cube=(256, 256)))  # already touching west face
    r = execute_option(s, Option("E", "short"))
    assert r.contact_episodes == 1

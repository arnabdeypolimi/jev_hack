import pytest

from pushcube.sim import CubeSim, SimState


def test_cube_does_not_rotate_when_pushed_off_centre():
    s = CubeSim()
    s.reset(SimState(pusher=(200, 300), cube=(256, 280)))
    for _ in range(20):
        s.step((320, 300))
    assert s._cube_body.angle == 0.0
    assert s.cube[0] > 256


def test_cube_stops_after_contact_ends():
    s = CubeSim()
    s.reset(SimState(pusher=(200, 256), cube=(256, 256)))
    for _ in range(8):
        s.step((300, 256))
    for _ in range(3):
        s.step((150, 256))
    x1 = s.cube[0]
    for _ in range(10):
        s.step((150, 256))
    assert abs(s.cube[0] - x1) < 0.5


def test_walls_contain_cube():
    s = CubeSim()
    s.reset(SimState(pusher=(400, 256), cube=(460, 256)))
    for _ in range(40):
        s.step((600, 256))
    assert s.cube[0] <= 506 - 30 + 0.5


def test_snapshot_restore_is_deterministic():
    a = CubeSim()
    a.reset(SimState(pusher=(100, 100), cube=(200, 200)))
    b = a.copy()
    for sim in (a, b):
        for _ in range(15):
            sim.step((260, 230))
    assert a.cube == pytest.approx(b.cube)
    assert a.pusher == pytest.approx(b.pusher)


def test_restore_returns_to_snapshot():
    s = CubeSim()
    s.reset(SimState(pusher=(100, 256), cube=(200, 256)))
    snap = s.snapshot()
    for _ in range(10):
        s.step((300, 256))
    s.restore(snap)
    assert s.cube == pytest.approx((200, 256))
    assert s.pusher == pytest.approx((100, 256))


def test_in_contact():
    s = CubeSim()
    s.reset(SimState(pusher=(256 - 30 - 15, 256), cube=(256, 256)))
    assert s.in_contact()
    s.reset(SimState(pusher=(50, 50), cube=(256, 256)))
    assert not s.in_contact()


def test_step_none_holds_pusher():
    s = CubeSim()
    s.reset(SimState(pusher=(100, 100), cube=(300, 300)))
    for _ in range(5):
        s.step(None)
    assert s.pusher == pytest.approx((100, 100), abs=0.5)
    assert s.steps == 5

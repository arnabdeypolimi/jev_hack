import pytest

from pushcube.controller import AIController
from pushcube.jev_judge import Decision, Pick
from pushcube.options import DIRECTIONS, DISTANCES, Option
from pushcube.scoring import overlap
from pushcube.sim import CubeSim, SimState

GOAL_FAR_EAST = (400, 256)


def dec(option_id, conf=0.9, source="jev", dir_p=None, dist_p=None):
    o = Option.from_id(option_id)
    dir_p = dir_p or {d: (0.9 if d == o.direction else 0.1 / 7) for d in DIRECTIONS}
    dist_p = dist_p or {s: (0.8 if s == o.distance else 0.1) for s in DISTANCES}
    return Decision(o, dir_p, dist_p, conf, source, 5.0, "jev-1.14.0" if source == "jev" else None)


class ScriptedJudge:
    def __init__(self, decisions, picks=(), on_decide=None, on_pick=None):
        self.decisions, self.picks = list(decisions), list(picks)
        self.decide_calls, self.pick_calls = [], []
        self.on_decide, self.on_pick = on_decide, on_pick

    async def decide(self, state, infeasible, fallback):
        self.decide_calls.append((state, infeasible, fallback))
        if self.on_decide:
            self.on_decide()
        return self.decisions.pop(0) if self.decisions else dec(fallback.id, 0.0, "fallback")

    async def pick(self, state, outcomes, fallback):
        self.pick_calls.append(outcomes)
        if self.on_pick:
            self.on_pick()
        o = Option.from_id(self.picks.pop(0))
        return Pick(o, {o.id: 0.7}, 0.6, "jev", 4.0)


class Run:
    def __init__(self, judge, cube=(200, 256), pusher=(100, 256), goal=GOAL_FAR_EAST, current=lambda rid: True,
                 max_pushes=1):
        self.sim = CubeSim()
        self.sim.reset(SimState(pusher=pusher, cube=cube))
        self.judge, self.goal, self.current, self.max_pushes = judge, goal, current, max_pushes
        self.msgs = []

    async def __call__(self):
        async def emit(m):
            self.msgs.append(m)

        ctl = AIController(self.sim, self.judge, emit, tau=0.5, step_delay_s=0)
        return await ctl.run(1, self.goal, self.current, max_pushes=self.max_pushes)

    def of(self, kind):
        return [m for m in self.msgs if m["type"] == kind]

    def statuses(self):
        return [m["status"] for m in self.of("ai_status")]

    def executed(self):
        return [m["option"] for m in self.of("ai_status") if m["status"] == "acting"]


async def test_confident_decision_executes_without_thinking_harder():
    r = Run(ScriptedJudge([dec("E-long", conf=0.9)]))
    res = await r()
    assert len(r.judge.decide_calls) == 1 and r.judge.pick_calls == []
    assert r.statuses()[:2] == ["deciding", "acting"] and r.executed() == ["E-long"]
    assert res.pushes == 1 and r.sim.cube[0] > 250
    assert r.of("decision")[0]["option"] == "E-long" and r.of("state")[-1]["board"] == "ai"


async def test_low_confidence_simulates_three_and_picks():
    r = Run(ScriptedJudge([dec("E-long", conf=0.3)], picks=["E-medium"]))
    await r()
    outcomes = r.of("imagination")[0]["outcomes"]
    assert [o["option"] for o in outcomes] == ["E-long", "E-short", "E-medium"]  # by p(dir) * p(dist)
    assert {"cube_after", "offset_after", "overlap_after", "touched_wall"} <= set(outcomes[0])
    assert "thinking_harder" in r.statuses() and r.executed() == ["E-medium"]
    assert r.of("pick")[0]["option"] == "E-medium"


async def test_imagination_sims_do_not_touch_real_sim():
    seen = {}
    r = None

    def capture():
        seen["cube"], seen["steps"] = r.sim.cube, r.sim.steps

    r = Run(ScriptedJudge([dec("E-long", conf=0.2)], picks=["E-long"], on_pick=capture))
    await r()
    assert seen["cube"] == pytest.approx((200, 256)) and seen["steps"] == 0


async def test_stops_mid_push_on_success():
    r = Run(ScriptedJudge([dec("E-long", conf=0.9)]), goal=(212, 256))
    res = await r()
    assert res.reached and res.pushes == 1 and res.finished_at_s is not None
    assert r.sim.cube[0] - 200 < 30
    assert overlap(r.sim.cube, (212, 256)) >= 0.9 and r.statuses()[-1] == "done"


async def test_stale_decision_dropped_after_new_round():
    state = {"current": True}
    r = Run(ScriptedJudge([dec("E-long")], on_decide=lambda: state.update(current=False)),
            current=lambda rid: state["current"])
    res = await r()
    assert r.sim.steps == 0 and res.pushes == 0 and r.executed() == []


async def test_infeasible_pick_falls_to_next_best():
    # cube against the east wall: W pushes need an approach point outside the table
    r = Run(ScriptedJudge([dec("N-medium", conf=0.2)], picks=["W-short"]), cube=(476, 256), pusher=(476, 400),
            goal=(476, 150))
    await r()
    executed = r.executed()
    assert executed and executed[0] != "W-short"


async def test_fallback_emits_judge_offline_and_uses_geometric_option():
    r = Run(ScriptedJudge([]))  # every decide returns a fallback decision
    await r()
    assert "judge_offline" in r.statuses() and r.of("decision")[0]["source"] == "fallback"
    assert r.executed()[0].startswith("E")


async def test_plays_until_goal_with_confident_judge():
    goal = (330, 256)
    judge = ScriptedJudge([dec("E-long"), dec("E-medium"), dec("E-medium"), dec("E-short"), dec("E-short")])
    r = Run(judge, goal=goal, max_pushes=10)
    res = await r()
    assert res.reached and 1 <= res.pushes <= 5

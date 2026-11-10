"""The AI side: Jev decides each push; when unsure it simulates its top options and asks again."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .executor import execute_option, is_feasible, push_targets
from .jev_judge import Decision, build_state, top_options
from .options import Option, all_options
from .policies import geometric_option
from .scoring import GOAL_MAX, GOAL_MIN, SUCCESS_OVERLAP, PushCounter, overlap
from .sim import CubeSim, Vec

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class SideResult:
    pushes: int
    reached: bool
    best_overlap: float
    finished_at_s: float | None


def _touched_wall(cube: Vec) -> bool:
    return any(v <= GOAL_MIN + 0.5 or v >= GOAL_MAX - 0.5 for v in cube)


def simulate_outcome(sim: CubeSim, option: Option, goal: Vec) -> dict:
    trial = sim.copy()
    execute_option(trial, option)
    cube = trial.cube
    return {
        "option": option.id,
        "cube_after": [round(cube[0], 1), round(cube[1], 1)],
        "offset_after": [round(goal[0] - cube[0], 1), round(goal[1] - cube[1], 1)],
        "overlap_after": round(overlap(cube, goal), 3),
        "touched_wall": _touched_wall(cube),
    }


class AIController:
    def __init__(self, sim: CubeSim, judge, emit: Emit, tau: float = 0.5, step_delay_s: float = 0.1) -> None:
        self.sim, self.judge, self.emit = sim, judge, emit
        self.tau, self.step_delay_s = tau, step_delay_s

    async def _status(self, status: str, **extra) -> None:
        await self.emit({"type": "ai_status", "status": status, **extra})

    async def _state(self, pushes: int, goal: Vec) -> None:
        await self.emit({"type": "state", "board": "ai", "pusher": list(self.sim.pusher), "cube": list(self.sim.cube),
                         "overlap": overlap(self.sim.cube, goal), "pushes": pushes})

    def _executable(self, option: Option, decision: Decision, infeasible: set[str], fallback: Option) -> Option:
        if is_feasible(self.sim.cube, option):
            return option
        return next(iter(top_options(decision, infeasible, k=len(all_options()))), fallback)

    async def run(self, round_id: int, goal: Vec, is_current: Callable[[int], bool], max_pushes: int = 40) -> SideResult:
        t0 = time.monotonic()
        counter = PushCounter(in_contact=self.sim.in_contact())
        best = overlap(self.sim.cube, goal)
        finished_at = None

        def done() -> bool:
            return best >= SUCCESS_OVERLAP or counter.count >= max_pushes or not is_current(round_id)

        while not done():
            infeasible = {o.id for o in all_options() if not is_feasible(self.sim.cube, o)}
            fallback = geometric_option(self.sim, goal)
            state = build_state(self.sim, goal, counter.count, infeasible)

            await self._status("deciding")
            decision = await self.judge.decide(state, infeasible, fallback)
            if not is_current(round_id):
                break
            await self.emit({"type": "decision", "option": decision.option.id, "direction_probs": decision.direction_probs,
                             "distance_probs": decision.distance_probs, "confidence": decision.confidence,
                             "source": decision.source, "latency_ms": decision.latency_ms})
            option = decision.option
            if decision.source != "jev":
                await self._status("judge_offline")
            elif decision.confidence < self.tau:
                await self._status("thinking_harder")
                outcomes = [simulate_outcome(self.sim, o, goal) for o in top_options(decision, infeasible, k=3)]
                await self.emit({"type": "imagination", "outcomes": outcomes})
                pick = await self.judge.pick(state, outcomes, option)
                if not is_current(round_id):
                    break
                await self.emit({"type": "pick", "option": pick.option.id, "probs": pick.probs,
                                 "confidence": pick.confidence, "source": pick.source})
                option = pick.option
            option = self._executable(option, decision, infeasible, fallback)

            await self._status("acting", option=option.id)
            for target in push_targets(self.sim.snapshot(), option):
                if not is_current(round_id):
                    break
                self.sim.step(target)
                counter.update(self.sim.in_contact())
                best = max(best, overlap(self.sim.cube, goal))
                await self._state(counter.count, goal)
                if best >= SUCCESS_OVERLAP:
                    finished_at = time.monotonic() - t0
                    break
                if self.step_delay_s:
                    await asyncio.sleep(self.step_delay_s)

        if is_current(round_id):
            await self._status("done")
        return SideResult(counter.count, best >= SUCCESS_OVERLAP, best, finished_at)

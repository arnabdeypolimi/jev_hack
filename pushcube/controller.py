"""The AI side: Jev decides each push; when unsure it simulates its top options and asks again."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .executor import execute_option, is_feasible, push_targets
from .jev_judge import Decision, build_state, top_options
from .options import Option, all_options
from .policies import fallback_option
from .scoring import SUCCESS_OVERLAP, PushCounter, _pose, overlap
from .shapes import angle_error, world_outline
from .sim import WALL_MAX, WALL_MIN, CubeSim

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class SideResult:
    pushes: int
    reached: bool
    best_overlap: float
    finished_at_s: float | None


def _touched_wall(sim: CubeSim) -> bool:
    minx, miny, maxx, maxy = world_outline(sim.shape, sim.pose).bounds
    return min(minx, miny) <= WALL_MIN + 0.5 or max(maxx, maxy) >= WALL_MAX - 0.5


def simulate_outcome(sim: CubeSim, option: Option, goal) -> dict:
    trial = sim.copy()
    execute_option(trial, option)
    (x, y), (gx, gy, ga) = trial.cube, _pose(goal)
    return {
        "option": option.id,
        "cube_after": [round(x, 1), round(y, 1)],
        "angle_after": round(trial.angle, 4),
        "angle_after_deg": round(math.degrees(trial.angle), 1),
        "offset_after": [round(gx - x, 1), round(gy - y, 1)],
        "turn_left_deg": round(math.degrees(angle_error(trial.angle, ga, trial.shape)), 1),
        "overlap_after": round(overlap(trial.pose, goal, trial.shape), 3),
        "touched_wall": _touched_wall(trial),
    }


class AIController:
    def __init__(self, sim: CubeSim, judge, emit: Emit, tau: float = 0.5, step_delay_s: float = 0.1) -> None:
        self.sim, self.judge, self.emit = sim, judge, emit
        self.tau, self.step_delay_s = tau, step_delay_s

    async def _status(self, status: str, **extra) -> None:
        await self.emit({"type": "ai_status", "status": status, **extra})

    async def _state(self, pushes: int, goal) -> None:
        await self.emit({"type": "state", "board": "ai", "pusher": list(self.sim.pusher), "cube": list(self.sim.cube),
                         "angle": self.sim.angle, "overlap": overlap(self.sim.pose, goal, self.sim.shape),
                         "pushes": pushes})

    def _executable(self, option: Option, decision: Decision, infeasible: set[str], fallback: Option) -> Option:
        if is_feasible(self.sim, option):
            return option
        return next(iter(top_options(decision, infeasible, k=len(all_options()))), fallback)

    async def run(self, round_id: int, goal, is_current: Callable[[int], bool], max_pushes: int = 40) -> SideResult:
        t0 = time.monotonic()
        counter = PushCounter(in_contact=self.sim.in_contact())
        best = overlap(self.sim.pose, goal, self.sim.shape)
        finished_at = None
        decisions = 0

        def done() -> bool:
            # The decision cap guards against moves that never touch the block (they cost no push).
            return (best >= SUCCESS_OVERLAP or counter.count >= max_pushes or decisions >= 2 * max_pushes
                    or not is_current(round_id))

        while not done():
            decisions += 1
            infeasible = {o.id for o in all_options() if not is_feasible(self.sim, o)}
            fallback = fallback_option(self.sim, goal)
            state = build_state(self.sim, goal, counter.count, infeasible)

            await self._status("deciding")
            decision = await self.judge.decide(state, infeasible, fallback)
            if not is_current(round_id):
                break
            await self.emit({"type": "decision", "option": decision.option.id, "direction_probs": decision.direction_probs,
                             "distance_probs": decision.distance_probs, "aim_probs": decision.aim_probs,
                             "confidence": decision.confidence,
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
            for target in push_targets(self.sim, option):
                if not is_current(round_id):
                    break
                self.sim.step(target)
                counter.observe(self.sim)
                best = max(best, overlap(self.sim.pose, goal, self.sim.shape))
                await self._state(counter.count, goal)
                if best >= SUCCESS_OVERLAP:
                    finished_at = time.monotonic() - t0
                    break
                if self.step_delay_s:
                    await asyncio.sleep(self.step_delay_s)

        if is_current(round_id):
            await self._status("done")
        return SideResult(counter.count, best >= SUCCESS_OVERLAP, best, finished_at)

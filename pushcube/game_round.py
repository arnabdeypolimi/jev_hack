from __future__ import annotations

import time
from typing import Literal

from .controller import SideResult
from .executor import clamp_pusher
from .scoring import SUCCESS_OVERLAP, PushCounter, overlap
from .sim import CubeSim, SimState, Vec

TIE_OVERLAP_EPS = 0.005


def decide_winner(human: SideResult, ai: SideResult) -> Literal["human", "ai", "tie"]:
    if human.reached and ai.reached:
        if human.pushes == ai.pushes:
            return "tie"
        return "human" if human.pushes < ai.pushes else "ai"
    if human.reached != ai.reached:
        return "human" if human.reached else "ai"
    if abs(human.best_overlap - ai.best_overlap) <= TIE_OVERLAP_EPS:
        return "tie"
    return "human" if human.best_overlap > ai.best_overlap else "ai"


def clamp_mouse(x: float, y: float) -> Vec:
    return clamp_pusher((x, y))


def human_target(mouse: Vec | None, down: bool) -> Vec | None:
    return clamp_mouse(*mouse) if (mouse is not None and down) else None


class Round:
    def __init__(self, id: int, challenge: str | None, goal, start: SimState, round_seconds: float = 90.0,
                 shape: str = "square") -> None:
        self.id, self.challenge, self.goal, self.start = id, challenge, goal, start
        self.round_seconds, self.shape = round_seconds, shape
        self.human_sim, self.ai_sim = CubeSim(shape), CubeSim(shape)
        self.human_sim.reset(start)
        self.ai_sim.reset(start)
        self.human_pushes = PushCounter(in_contact=self.human_sim.in_contact())
        self.started_at = time.monotonic()
        self._human_best = overlap((*start.cube, start.angle), goal, shape)
        self._human_finished_at: float | None = None

    @property
    def human_done(self) -> bool:
        return self._human_best >= SUCCESS_OVERLAP

    def human_step(self, mouse: Vec | None, down: bool) -> None:
        if self.human_done:
            return
        self.human_sim.step(human_target(mouse, down))
        self.human_pushes.observe(self.human_sim)
        self._human_best = max(self._human_best, overlap(self.human_sim.pose, self.goal, self.shape))
        if self.human_done:
            self._human_finished_at = time.monotonic() - self.started_at

    def human_result(self) -> SideResult:
        return SideResult(self.human_pushes.count, self.human_done, self._human_best, self._human_finished_at)

"""TypeSafe Jev as the AI's decision-maker: picks each push from raw coordinates."""

from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import httpx

from .options import AIMS, DIRECTIONS, DISTANCES, Option
from .scoring import _pose, overlap
from .shapes import angle_error
from .sim import WALL_MAX, WALL_MIN, CubeSim

API_URL = "https://api.typesafe.ai/v1/systemone"
MODELS_URL = "https://api.typesafe.ai/v1/models"
MODEL = "jev-1.14.0"

DIRECTION_TEXT = {
    "N": "push the cube up, toward the top edge (y decreases)",
    "NE": "push the cube up and to the right (x increases, y decreases)",
    "E": "push the cube right, toward the right edge (x increases)",
    "SE": "push the cube down and to the right (x increases, y increases)",
    "S": "push the cube down, toward the bottom edge (y increases)",
    "SW": "push the cube down and to the left (x decreases, y increases)",
    "W": "push the cube left, toward the left edge (x decreases)",
    "NW": "push the cube up and to the left (x decreases, y decreases)",
}
DISTANCE_TEXT = {
    "short": "a short nudge: the block moves about 6 px (or turns a little)",
    "medium": "a medium push: the block moves about 26 px",
    "long": "a long push: the block moves about 75 px",
}
AIM_TEXT = {
    "left": "hit the face left of centre (as seen from behind the push): the block turns clockwise",
    "center": "hit straight through the middle: the block slides with little turning",
    "right": "hit the face right of centre (as seen from behind the push): the block turns counterclockwise",
}
DECIDE_AIM = (
    "Where on the block should the push land? Use `turn_needed_deg` (positive = needs to turn clockwise, "
    "negative = counterclockwise); aim at the centre when the angle is already right."
)
DECIDE_DIRECTION = (
    "The block must end up on the goal outline at the goal angle. Which direction should the next push "
    "move the block so it gets closer to the goal? Options listed in `infeasible_options` cannot be executed."
)
DECIDE_DISTANCE = (
    "How far should the next push move the block, given `offset_to_goal`? Overshooting costs extra pushes."
)
PICK_INSTRUCTIONS = (
    "The AI simulated each candidate push with the real physics. Which candidate leaves the block "
    "best placed (position and angle) to reach the goal in the fewest further pushes?"
)


@dataclass
class Decision:
    option: Option
    direction_probs: dict[str, float]
    distance_probs: dict[str, float]
    confidence: float
    source: Literal["jev", "fallback"]
    latency_ms: float
    model: str | None = None
    aim_probs: dict[str, float] = field(default_factory=dict)


@dataclass
class Pick:
    option: Option
    probs: dict[str, float]
    confidence: float
    source: Literal["jev", "fallback"]
    latency_ms: float


@dataclass
class _Answers:
    answers: dict
    model: str | None
    latency_ms: float


def build_state(sim: CubeSim, goal, pushes: int, infeasible: set[str]) -> dict:
    (cx, cy), (px, py) = sim.cube, sim.pusher
    gx, gy, ga = _pose(goal)
    return {
        "coordinates": "pixels on a top-down table; x grows to the right, y grows downward",
        "turn_convention": "angles in degrees, positive = clockwise on screen",
        "table": {"width": 512, "height": 512, "walls_at": [WALL_MIN, WALL_MAX]},
        "block": {"shape": sim.shape, "x": round(cx, 1), "y": round(cy, 1),
                  "angle_deg": round(math.degrees(sim.angle), 1)},
        "pusher": {"x": round(px, 1), "y": round(py, 1)},
        "goal": {"x": round(gx, 1), "y": round(gy, 1), "angle_deg": round(math.degrees(ga), 1)},
        "offset_to_goal": {"dx": round(gx - cx, 1), "dy": round(gy - cy, 1)},
        "turn_needed_deg": round(math.degrees(angle_error(sim.angle, ga, sim.shape)), 1),
        "overlap_with_goal": round(overlap(sim.pose, goal, sim.shape), 3),
        "success_when_overlap_at_least": 0.9,
        "pushes_so_far": pushes,
        "infeasible_options": sorted(infeasible),
    }


def _ranked(direction_probs: dict[str, float], distance_probs: dict[str, float],
            aim_probs: dict[str, float] | None = None) -> list[tuple[float, Option]]:
    aim_probs = aim_probs or {"center": 1.0}
    triples = [
        (direction_probs.get(d, 0.0) * distance_probs.get(s, 0.0) * aim_probs.get(a, 0.0), Option(d, s, a))
        for d in DIRECTIONS
        for s in DISTANCES
        for a in AIMS
    ]
    return sorted(triples, key=lambda p: p[0], reverse=True)


def top_options(decision: Decision, infeasible: set[str], k: int = 3) -> list[Option]:
    if decision.source != "jev":
        return [decision.option]
    ranked = [o for p, o in _ranked(decision.direction_probs, decision.distance_probs, decision.aim_probs) if p > 0]
    return [o for o in ranked if o.id not in infeasible][:k]


def _choice(answer: dict) -> tuple[str, dict[str, float], float]:
    probs = answer.get("probabilities") or {answer["choice"]: 1.0}
    return answer["choice"], {k: float(v) for k, v in probs.items()}, float(answer.get("confidence") or 0.0)


def _parse(res: _Answers | None, *names: str) -> list[tuple[str, dict[str, float], float]] | None:
    """Parse the named Choice answers; None if the response is missing or malformed."""
    if res is None:
        return None
    try:
        return [_choice(res.answers[n]) for n in names]
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _describe(o: dict) -> str:
    turn = o.get("turn_left_deg", 0.0)
    turn_text = "angle right" if abs(turn) < 1 else f"still {abs(turn):.0f}° {'clockwise' if turn > 0 else 'counterclockwise'} to turn"
    return (
        f"block ends at ({o['cube_after'][0]:.0f}, {o['cube_after'][1]:.0f}) at {o.get('angle_after_deg', 0.0):.0f}°, "
        f"offset to goal ({o['offset_after'][0]:.0f}, {o['offset_after'][1]:.0f}), {turn_text}, "
        f"overlap {o['overlap_after'] * 100:.0f}%, touched wall: {'yes' if o['touched_wall'] else 'no'}"
    )


class JevJudge:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = MODEL,
        timeout_s: float = 4.0,
        transport: httpx.AsyncBaseTransport | None = None,
        log_path: str | None = "recordings/jev_log.jsonl",
    ) -> None:
        self._key = api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY")
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport
        self._log_path = Path(log_path) if log_path else None
        self._last_http: int | None = None
        self._unreachable = False

    @property
    def available(self) -> bool:
        return bool(self._key)

    @property
    def status(self) -> str:
        """missing | rejected | ok | unreachable | unknown — why Jev is (not) answering."""
        if not self._key:
            return "missing"
        if self._last_http in (401, 403):
            return "rejected"
        if self._last_http == 200:
            return "ok"
        return "unreachable" if self._unreachable else "unknown"

    def _record_http(self, status: int | None) -> None:
        self._last_http, self._unreachable = status, status is None

    async def check(self) -> str:
        """Ask TypeSafe whether the key works (lists models), without spending a decision."""
        if not self._key:
            return self.status
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_s) as client:
                r = await client.get(MODELS_URL, headers={"Authorization": f"Bearer {self._key}"})
            self._record_http(r.status_code)
        except httpx.HTTPError:
            self._record_http(None)
        return self.status

    async def _ask(self, kind: str, state: dict, questions: dict) -> _Answers | None:
        if not self._key:
            return None
        body = {"model": self.model, "state": state, "questions": questions}
        t0 = time.perf_counter()
        status, data = None, None
        try:
            async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout_s) as client:
                r = await client.post(API_URL, json=body, headers={"Authorization": f"Bearer {self._key}"})
            status = r.status_code
            data = r.json() if status == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        latency = (time.perf_counter() - t0) * 1000
        self._record_http(status)
        self._log(kind, state, questions, status, data, latency)
        if not data or not isinstance(data.get("answers"), dict):
            return None
        return _Answers(answers=data["answers"], model=data.get("model"), latency_ms=latency)

    def _log(self, kind, state, questions, status, data, latency) -> None:
        if self._log_path is None:
            return
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": time.time(), "kind": kind, "status": status, "latency_ms": round(latency, 1),
                "state": state, "questions": list(questions), "response": data}
        with self._log_path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    async def decide(self, state: dict, infeasible: set[str], fallback: Option) -> Decision:
        questions = {
            "direction": {"type": "choice", "instructions": DECIDE_DIRECTION, "criteria": DIRECTION_TEXT},
            "distance": {"type": "choice", "instructions": DECIDE_DISTANCE, "criteria": DISTANCE_TEXT},
            "aim": {"type": "choice", "instructions": DECIDE_AIM, "criteria": AIM_TEXT},
        }
        res = await self._ask("decide", state, questions)
        parsed = _parse(res, "direction", "distance", "aim")
        if parsed is None:
            return Decision(fallback, {}, {}, 0.0, "fallback", res.latency_ms if res else 0.0)
        (_, dir_p, dir_c), (_, dist_p, dist_c), (_, aim_p, aim_c) = parsed
        feasible = [o for p, o in _ranked(dir_p, dist_p, aim_p) if o.id not in infeasible]
        if not feasible:
            return Decision(fallback, dir_p, dist_p, 0.0, "fallback", res.latency_ms, res.model, aim_p)
        conf = min(dir_c, dist_c, aim_c)
        return Decision(feasible[0], dir_p, dist_p, conf, "jev", res.latency_ms, res.model, aim_p)

    async def pick(self, state: dict, outcomes: list[dict], fallback: Option) -> Pick:
        criteria = {o["option"]: _describe(o) for o in outcomes if "cube_after" in o}
        res = None
        if criteria:
            question = {"type": "choice", "instructions": PICK_INSTRUCTIONS, "criteria": criteria}
            res = await self._ask("pick", state, {"pick": question})
        parsed = _parse(res, "pick")
        if parsed is None or parsed[0][0] not in criteria:
            return Pick(fallback, {}, 0.0, "fallback", res.latency_ms if res else 0.0)
        chosen, probs, conf = parsed[0]
        return Pick(Option.from_id(chosen), probs, conf, "jev", res.latency_ms)

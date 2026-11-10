"""Headless play for measuring Jev against the geometric rule and the one-step oracle."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

from .challenges import Challenge
from .controller import AIController, SideResult
from .executor import execute_option
from .policies import geometric_option, oracle_option
from .scoring import SUCCESS_OVERLAP, overlap
from .sim import CubeSim

Policy = Literal["jev", "geometric", "oracle"]


@dataclass
class HeadlessRun:
    result: SideResult
    decisions: int = 0
    escalations: int = 0
    fallbacks: int = 0
    latencies_ms: list[float] = field(default_factory=list)


async def play_headless(policy: Policy, challenge: Challenge, judge=None, max_pushes: int = 40) -> HeadlessRun:
    sim = CubeSim()
    sim.reset(challenge.start)
    if policy == "jev":
        stats = {"decisions": 0, "escalations": 0, "fallbacks": 0, "latencies_ms": []}

        async def emit(msg: dict) -> None:
            if msg["type"] == "decision":
                stats["decisions"] += 1
                stats["fallbacks"] += msg["source"] != "jev"
                if msg["source"] == "jev":
                    stats["latencies_ms"].append(msg["latency_ms"])
            elif msg["type"] == "imagination":
                stats["escalations"] += 1

        ctl = AIController(sim, judge, emit, step_delay_s=0)
        result = await ctl.run(0, challenge.goal, lambda _: True, max_pushes=max_pushes)
        return HeadlessRun(result, **stats)

    choose = geometric_option if policy == "geometric" else (lambda s, g: oracle_option(s, g)[0])
    pushes, best = 0, overlap(sim.cube, challenge.goal)
    while best < SUCCESS_OVERLAP and pushes < max_pushes:
        execute_option(sim, choose(sim, challenge.goal))
        pushes += 1
        best = max(best, overlap(sim.cube, challenge.goal))
    return HeadlessRun(result=SideResult(pushes, best >= SUCCESS_OVERLAP, best, None))


def summarize(runs: list[tuple[str, str, HeadlessRun]]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], list[HeadlessRun]] = {}
    for tier, policy, run in runs:
        groups.setdefault((tier, policy), []).append(run)
    out = {}
    for key, rs in groups.items():
        lat = [v for r in rs for v in r.latencies_ms]
        decisions = sum(r.decisions for r in rs)
        out[key] = {
            "runs": len(rs),
            "success_rate": sum(r.result.reached for r in rs) / len(rs),
            "mean_pushes": statistics.mean(r.result.pushes for r in rs),
            "escalation_rate": (sum(r.escalations for r in rs) / decisions) if decisions else 0.0,
            "fallback_rate": (sum(r.fallbacks for r in rs) / decisions) if decisions else 0.0,
            "median_latency_ms": statistics.median(lat) if lat else None,
        }
    return out

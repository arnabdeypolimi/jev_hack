"""Live Jev dry run: checks the key, then lets Jev play two presets headless.

Run: uv run --env-file .env python scripts/dry_run_jev.py
"""

import asyncio
import os
import statistics

import httpx

from pushcube.challenges import get_challenge
from pushcube.executor import execute_option, is_feasible
from pushcube.jev_judge import JevJudge, build_state
from pushcube.options import all_options
from pushcube.policies import geometric_option
from pushcube.scoring import SUCCESS_OVERLAP, overlap
from pushcube.sim import CubeSim


def check_key() -> bool:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        print("TYPESAFE_API_KEY is not set (expected in .env)")
        return False
    r = httpx.get("https://api.typesafe.ai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    print(f"auth check: HTTP {r.status_code}")
    return r.status_code == 200


async def play(judge: JevJudge, challenge_id: str, max_pushes: int = 20) -> None:
    c = get_challenge(challenge_id)
    sim = CubeSim()
    sim.reset(c.start)
    latencies, pushes = [], 0
    while overlap(sim.cube, c.goal) < SUCCESS_OVERLAP and pushes < max_pushes:
        infeasible = {o.id for o in all_options() if not is_feasible(sim.cube, o)}
        d = await judge.decide(build_state(sim, c.goal, pushes, infeasible), infeasible, geometric_option(sim, c.goal))
        latencies.append(d.latency_ms)
        execute_option(sim, d.option)
        pushes += 1
        print(f"  push {pushes:2d}: {d.option.id:10s} conf {d.confidence:.2f} [{d.source}] "
              f"overlap {overlap(sim.cube, c.goal):.0%} ({d.latency_ms:.0f} ms)")
    print(f"{challenge_id}: reached={overlap(sim.cube, c.goal) >= SUCCESS_OVERLAP} pushes={pushes} "
          f"median latency={statistics.median(latencies):.0f} ms")


async def main() -> None:
    if not check_key():
        print("Jev unavailable: fix the key before the dry run.")
        return
    judge = JevJudge()
    for cid in ("easy-east", "medium-diagonal"):
        await play(judge, cid)


if __name__ == "__main__":
    asyncio.run(main())

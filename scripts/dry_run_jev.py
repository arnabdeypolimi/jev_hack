"""Live Jev dry run: checks the key, then lets Jev play two presets headless.

Run: uv run --env-file .env python scripts/dry_run_jev.py
"""

import asyncio
import statistics

from pushcube.challenges import get_challenge
from pushcube.executor import execute_option, is_feasible
from pushcube.jev_judge import JevJudge, build_state
from pushcube.options import all_options
from pushcube.policies import geometric_option
from pushcube.scoring import SUCCESS_OVERLAP, overlap
from pushcube.sim import CubeSim


async def play(judge: JevJudge, challenge_id: str, max_pushes: int = 20) -> None:
    c = get_challenge(challenge_id)
    sim = CubeSim(c.shape)
    sim.reset(c.start)
    latencies, pushes = [], 0
    while overlap(sim.pose, c.goal, sim.shape) < SUCCESS_OVERLAP and pushes < max_pushes:
        infeasible = {o.id for o in all_options() if not is_feasible(sim, o)}
        d = await judge.decide(build_state(sim, c.goal, pushes, infeasible), infeasible, geometric_option(sim, c.goal))
        latencies.append(d.latency_ms)
        execute_option(sim, d.option)
        pushes += 1
        print(f"  push {pushes:2d}: {d.option.id:16s} conf {d.confidence:.2f} [{d.source}] "
              f"overlap {overlap(sim.pose, c.goal, sim.shape):.0%} ({d.latency_ms:.0f} ms)")
    print(f"{challenge_id}: reached={overlap(sim.pose, c.goal, sim.shape) >= SUCCESS_OVERLAP} pushes={pushes} "
          f"median latency={statistics.median(latencies):.0f} ms")


async def main() -> None:
    judge = JevJudge()
    status = await judge.check()
    print(f"auth check: {status}")
    if status != "ok":
        print("Jev unavailable: fix TYPESAFE_API_KEY in .env before the dry run.")
        return
    for cid in ("easy-east", "medium-twist"):
        await play(judge, cid)


if __name__ == "__main__":
    asyncio.run(main())

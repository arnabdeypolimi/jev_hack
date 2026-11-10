"""Play-quality harness: Jev vs the geometric rule vs a one-step oracle on every preset.

Run: uv run --env-file .env python scripts/harness.py --repeats 3 --out recordings/harness.json
"""

import argparse
import asyncio
import json
from pathlib import Path

from pushcube.challenges import PRESETS
from pushcube.harness import play_headless, summarize
from pushcube.jev_judge import JevJudge


async def main(repeats: int, out: str) -> None:
    policies = ["geometric", "oracle"]
    judge = JevJudge(log_path="recordings/harness_jev_log.jsonl")
    if await judge.check() == "ok":
        policies.insert(0, "jev")
    else:
        print("Jev unavailable (missing or rejected key): reporting baselines only.\n")
    runs = []
    for c in PRESETS:
        for policy in policies:
            for _ in range(repeats if policy == "jev" else 1):
                runs.append((c.tier, policy, await play_headless(policy, c, judge=judge)))
    table = summarize(runs)
    print(f"{'tier':8s} {'policy':10s} {'runs':>4s} {'success':>8s} {'pushes':>7s} {'escalate':>9s} {'fallback':>9s} {'latency':>8s}")
    for (tier, policy), row in sorted(table.items()):
        lat = f"{row['median_latency_ms']:.0f}ms" if row["median_latency_ms"] else "–"
        print(f"{tier:8s} {policy:10s} {row['runs']:4d} {row['success_rate']:8.0%} {row['mean_pushes']:7.1f} "
              f"{row['escalation_rate']:9.0%} {row['fallback_rate']:9.0%} {lat:>8s}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({f"{t}/{p}": v for (t, p), v in table.items()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="recordings/harness.json")
    a = ap.parse_args()
    asyncio.run(main(a.repeats, a.out))

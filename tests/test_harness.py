from pushcube.challenges import get_challenge
from pushcube.harness import play_headless, summarize
from pushcube.jev_judge import Decision, Pick
from pushcube.options import DIRECTIONS, DISTANCES, Option


class EastJudge:
    """Always says east; low confidence on even calls so think-harder triggers."""

    def __init__(self):
        self.calls = 0

    async def decide(self, state, infeasible, fallback):
        self.calls += 1
        dist = "long" if state["offset_to_goal"]["dx"] > 60 else "medium" if state["offset_to_goal"]["dx"] > 18 else "short"
        dir_p = {d: (0.9 if d == "E" else 0.1 / 7) for d in DIRECTIONS}
        dist_p = {s: (0.8 if s == dist else 0.1) for s in DISTANCES}
        conf = 0.3 if self.calls % 2 == 0 else 0.9
        return Decision(Option("E", dist), dir_p, dist_p, conf, "jev", 12.0, "jev-1.14.0")

    async def pick(self, state, outcomes, fallback):
        best = max(outcomes, key=lambda o: o["overlap_after"])
        return Pick(Option.from_id(best["option"]), {best["option"]: 0.8}, 0.6, "jev", 8.0)


async def test_play_headless_geometric_reaches_easy_goal():
    run = await play_headless("geometric", get_challenge("easy-east"))
    assert run.result.reached and run.result.pushes <= 5 and run.escalations == 0


async def test_play_headless_oracle_reaches_easy_goal():
    run = await play_headless("oracle", get_challenge("easy-bar"))
    assert run.result.reached


async def test_play_headless_jev_counts_escalations_and_latency():
    run = await play_headless("jev", get_challenge("easy-east"), judge=EastJudge())
    assert run.result.reached
    assert run.decisions >= 1 and run.escalations >= 1 and run.fallbacks == 0
    assert run.latencies_ms and all(v > 0 for v in run.latencies_ms)


async def test_play_headless_oracle_handles_a_spinning_shape():
    run = await play_headless("oracle", get_challenge("hard-t-turn"))
    assert run.result.reached


async def test_summarize_groups_by_tier_and_policy():
    runs = [
        ("easy", "geometric", await play_headless("geometric", get_challenge("easy-east"))),
        ("easy", "geometric", await play_headless("geometric", get_challenge("easy-bar"))),
    ]
    table = summarize(runs)
    row = table[("easy", "geometric")]
    assert row["success_rate"] == 1.0 and row["runs"] == 2 and row["mean_pushes"] > 0

import json

import httpx
import pytest

from pushcube.jev_judge import JevJudge, build_state, top_options
from pushcube.options import DIRECTIONS, DISTANCES, Option
from pushcube.sim import CubeSim, SimState

FALLBACK = Option("S", "short")
STATE = {"cube": {"x": 200, "y": 256}, "goal": {"x": 330, "y": 256}}


def choice(name, probs):
    best = max(probs, key=probs.get)
    return {"type": "choice", "choice": best, "confidence": round(probs[best], 3), "probabilities": probs}


def answers(dir_p, dist_p, aim_p=None):
    aim_p = aim_p or {"center": 1.0}
    return {
        "model": "jev-1.14.0",
        "answers": {"direction": choice("direction", dir_p), "distance": choice("distance", dist_p),
                    "aim": choice("aim", aim_p)},
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }


def judge_with(handler, tmp_path=None, api_key="sk-test-key"):
    log = str(tmp_path / "jev.jsonl") if tmp_path else None
    return JevJudge(api_key=api_key, transport=httpx.MockTransport(handler), log_path=log)


def respond(payload, status=200, sink=None):
    def handler(request):
        if sink is not None:
            sink.append(request)
        return httpx.Response(status, json=payload)

    return handler


async def test_decide_combines_direction_and_distance():
    j = judge_with(respond(answers({"NE": 0.7, "E": 0.3}, {"medium": 0.9, "long": 0.1})))
    d = await j.decide(STATE, set(), FALLBACK)
    assert d.option == Option("NE", "medium")
    assert d.source == "jev" and d.model == "jev-1.14.0"
    assert d.confidence == pytest.approx(0.7)
    assert d.direction_probs["NE"] == pytest.approx(0.7)


async def test_request_shape_pinned_model_no_images():
    sink = []
    await judge_with(respond(answers({"E": 1.0}, {"long": 1.0}), sink=sink)).decide(STATE, set(), FALLBACK)
    req = sink[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer sk-test-key"
    assert "typesafe-danger-zone-images" not in req.headers
    assert body["model"] == "jev-1.14.0" and "assets" not in body
    assert body["state"] == STATE
    q = body["questions"]
    assert q["direction"]["type"] == "choice" and set(q["direction"]["criteria"]) == set(DIRECTIONS)
    assert q["distance"]["type"] == "choice" and set(q["distance"]["criteria"]) == set(DISTANCES)
    assert q["aim"]["type"] == "choice" and set(q["aim"]["criteria"]) == {"left", "center", "right"}
    assert "clockwise" in q["aim"]["criteria"]["left"]


async def test_decide_includes_aim():
    j = judge_with(respond(answers({"E": 1.0}, {"medium": 1.0}, {"left": 0.6, "center": 0.3, "right": 0.1})))
    d = await j.decide(STATE, set(), FALLBACK)
    assert d.option == Option("E", "medium", "left") and d.aim_probs["left"] == pytest.approx(0.6)
    assert d.confidence == pytest.approx(0.6)


async def test_infeasible_choice_falls_to_next_best_pair():
    j = judge_with(respond(answers({"NE": 0.7, "E": 0.3}, {"medium": 0.9, "long": 0.1})))
    d = await j.decide(STATE, {"NE-medium", "NE-long", "NE-short"}, FALLBACK)
    assert d.option == Option("E", "medium") and d.source == "jev"


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=r)),
        respond({"detail": "rate limited"}, status=429),
        respond({"detail": "bad key"}, status=401),
        respond({"model": "jev-1.14.0", "answers": {}}),
    ],
    ids=["timeout", "429", "401", "malformed"],
)
async def test_errors_fall_back(handler):
    d = await judge_with(handler).decide(STATE, set(), FALLBACK)
    assert d.source == "fallback" and d.option == FALLBACK and d.confidence == 0.0


async def test_missing_key_falls_back_without_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    calls = []
    d = await JevJudge(transport=httpx.MockTransport(respond({}, sink=calls)), log_path=None).decide(
        STATE, set(), FALLBACK
    )
    assert d.source == "fallback" and calls == []


async def test_pick_maps_choice_to_outcome_option():
    outcomes = [
        {"option": "E-long", "cube_after": [275, 256], "angle_after_deg": 12.0, "offset_after": [55, 0],
         "turn_left_deg": -12.0, "overlap_after": 0.08, "touched_wall": False},
        {"option": "E-medium", "cube_after": [226, 256], "angle_after_deg": 0.0, "offset_after": [104, 0],
         "turn_left_deg": 0.0, "overlap_after": 0.0, "touched_wall": False},
    ]
    sink = []
    payload = {"model": "jev-1.14.0", "answers": {"pick": choice("pick", {"E-long": 0.8, "E-medium": 0.2})}}
    p = await judge_with(respond(payload, sink=sink)).pick(STATE, outcomes, FALLBACK)
    assert p.option == Option("E", "long") and p.source == "jev" and p.probs["E-long"] == pytest.approx(0.8)
    criteria = json.loads(sink[0].content)["questions"]["pick"]["criteria"]
    assert set(criteria) == {"E-long", "E-medium"} and "overlap 8%" in criteria["E-long"]
    assert "12° counterclockwise" in criteria["E-long"]


async def test_pick_falls_back_to_given_option_on_error():
    p = await judge_with(respond({}, status=500)).pick(STATE, [{"option": "E-long"}], FALLBACK)
    assert p.source == "fallback" and p.option == FALLBACK


async def test_top_options_skips_infeasible_and_orders_by_product():
    d = await judge_with(respond(answers({"NE": 0.6, "E": 0.4}, {"medium": 0.7, "long": 0.3}))).decide(
        STATE, set(), FALLBACK
    )
    assert [o.id for o in top_options(d, {"NE-long"}, k=3)] == ["NE-medium", "E-medium", "E-long"]


async def test_log_never_contains_key(tmp_path):
    j = judge_with(respond(answers({"E": 1.0}, {"long": 1.0})), tmp_path=tmp_path, api_key="sk-SECRET-123")
    await j.decide(STATE, set(), FALLBACK)
    text = (tmp_path / "jev.jsonl").read_text()
    assert text and "sk-SECRET-123" not in text and json.loads(text.splitlines()[0])["kind"] == "decide"


def test_build_state_contents():
    import math
    sim = CubeSim("T")
    sim.reset(SimState(pusher=(100, 100), cube=(200, 256), angle=math.radians(10)))
    s = build_state(sim, (330, 256, math.radians(40)), pushes=2, infeasible={"W-long"})
    assert s["block"] == {"shape": "T", "x": 200.0, "y": 256.0, "angle_deg": 10.0}
    assert s["goal"] == {"x": 330.0, "y": 256.0, "angle_deg": 40.0}
    assert s["turn_needed_deg"] == pytest.approx(30.0) and "clockwise" in s["turn_convention"]
    assert s["offset_to_goal"] == {"dx": 130.0, "dy": 0.0}
    assert s["pushes_so_far"] == 2 and s["infeasible_options"] == ["W-long"]
    assert "y grows downward" in s["coordinates"]

import json

import httpx
from fastapi.testclient import TestClient

from pushcube.jev_judge import JevJudge
from pushcube.server import create_app


def offline_judge():
    return JevJudge(api_key="", log_path=None)


def make_client(judge=None, **kw):
    kw.setdefault("step_delay_s", 0)
    kw.setdefault("tick_s", 0.01)
    return TestClient(create_app(judge or offline_judge(), **kw))


def read_until(ws, pred, limit=3000):
    seen = []
    for _ in range(limit):
        m = ws.receive_json()
        seen.append(m)
        if pred(m):
            return m, seen
    raise AssertionError(f"predicate never matched; last types: {[s['type'] for s in seen[-10:]]}")


def test_snapshot_on_connect():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and snap["replay"] is False
        assert snap["shape"] == {"pusher_radius": 15, "walls": [5, 506], "board_size": 512}
        assert snap["round"]["phase"] == "setup" and snap["boards"]["human"]["cube"]
        assert snap["round"]["shape"] == "square" and snap["round"]["polygons"][0][0] == [-30, -30]
        assert len(snap["round"]["goal"]) == 3 and "angle" in snap["boards"]["ai"]
        assert snap["jev_available"] is False and snap["tau"] == 0.5


def test_challenges_endpoint():
    with make_client() as c:
        data = c.get("/challenges").json()
        assert {"id", "name", "tier", "shape", "goal", "start"} <= set(data[0])
        assert {c["shape"] for c in data} >= {"square", "T", "L", "I", "Z", "triangle"}


def test_start_broadcasts_round_and_states():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "challenge": "easy-east"})
        rnd, _ = read_until(ws, lambda m: m["type"] == "round" and m["phase"] == "playing")
        assert rnd["challenge"] == "easy-east" and rnd["id"] == 1
        read_until(ws, lambda m: m["type"] == "state" and m["board"] == "human")
        read_until(ws, lambda m: m["type"] == "state" and m["board"] == "ai")


def test_set_goal_is_clamped():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "set_goal", "goal": [500, 500]})
        rnd, _ = read_until(ws, lambda m: m["type"] == "round")
        assert rnd["goal"] == [476, 476, 0.0] and rnd["phase"] == "setup"
        ws.send_json({"type": "set_goal", "goal": [256, 256, 0.5]})
        rnd, _ = read_until(ws, lambda m: m["type"] == "round")
        assert rnd["goal"] == [256, 256, 0.5]


def test_selecting_a_shape_challenge_sends_its_outline():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "select", "challenge": "hard-t-turn"})
        rnd, _ = read_until(ws, lambda m: m["type"] == "round")
        assert rnd["shape"] == "T" and len(rnd["polygons"]) == 2 and rnd["start"]["angle"] == 0.0
        assert len(rnd["outline"]) == 8  # the T's outer boundary has 8 corners


def test_mouse_moves_human_pusher():
    with make_client() as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "challenge": "easy-east"})
        read_until(ws, lambda m: m["type"] == "round" and m["phase"] == "playing")
        ws.send_json({"type": "mouse", "x": 60, "y": 60, "down": True})
        read_until(ws, lambda m: m["type"] == "state" and m["board"] == "human" and m["pusher"][0] < 80)


def test_new_start_cancels_previous_ai_task():
    app = create_app(offline_judge(), step_delay_s=0.05, tick_s=0.01)
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "challenge": "far-across"})
        read_until(ws, lambda m: m["type"] == "ai_status" and m["status"] == "acting")
        first = app.state.game.ai_task
        ws.send_json({"type": "start", "challenge": "easy-east"})
        read_until(ws, lambda m: m["type"] == "round" and m["id"] == 2)
        read_until(ws, lambda m: m["type"] == "state")
        assert first.done() and app.state.game.ai_task is not first


def test_round_finishes_with_winner():
    with make_client(round_seconds=1.5) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "challenge": "easy-bar"})
        rnd, _ = read_until(ws, lambda m: m["type"] == "round" and m["phase"] == "finished")
        assert rnd["winner"] == "ai" and rnd["results"]["ai"]["reached"] is True
        assert rnd["results"]["human"]["pushes"] == 0


def test_api_key_never_sent_to_client():
    deny = httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "no"}))
    judge = JevJudge(api_key="sk-test-SECRET", transport=deny, log_path=None)
    with make_client(judge, round_seconds=1.0) as c, c.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        ws.send_json({"type": "start", "challenge": "easy-east"})
        _, seen = read_until(ws, lambda m: m["type"] == "round" and m["phase"] == "finished")
        blob = json.dumps([snap, *seen])
        assert "sk-test-SECRET" not in blob and snap["jev_available"] is True


def test_record_then_replay_same_message_types(tmp_path):
    rec = tmp_path / "round.jsonl"
    with make_client(record_to=str(rec), round_seconds=1.0) as c, c.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "challenge": "easy-bar"})
        read_until(ws, lambda m: m["type"] == "round" and m["phase"] == "finished")
    recorded = [json.loads(line)["msg"]["type"] for line in rec.read_text().splitlines()]
    assert recorded and "decision" in recorded

    with make_client(replay_from=str(rec), replay_speed=1000.0) as c, c.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["replay"] is True
        replayed = [ws.receive_json()["type"] for _ in recorded]
        assert replayed == recorded

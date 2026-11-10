"""FastAPI game server: two boards, the AI controller, and a WebSocket protocol for the browser.

Run: uv run --env-file .env uvicorn pushcube.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from .challenges import PRESETS, get_challenge
from .controller import AIController
from .game_round import Round, decide_winner
from .jev_judge import JevJudge
from .scoring import clamp_goal, overlap
from .sim import BOARD, CUBE_SIZE, PUSHER_RADIUS, WALL_MAX, WALL_MIN, SimState

STATIC_DIR = Path(__file__).parent / "static"
SHAPE = {"cube_size": int(CUBE_SIZE), "pusher_radius": int(PUSHER_RADIUS),
         "walls": [int(WALL_MIN), int(WALL_MAX)], "board_size": BOARD}


def _start_dict(s: SimState) -> dict:
    return {"pusher": list(s.pusher), "cube": list(s.cube)}


def _parse_start(d: dict | None, default: SimState) -> SimState:
    if not d:
        return default
    return SimState(pusher=tuple(d["pusher"]), cube=tuple(clamp_goal(d["cube"])))


class Game:
    def __init__(self, judge: JevJudge, tau: float, step_delay_s: float, tick_s: float, round_seconds: float,
                 record_to: str | None) -> None:
        self.judge, self.tau = judge, tau
        self.step_delay_s, self.tick_s, self.round_seconds = step_delay_s, tick_s, round_seconds
        self.clients: set[WebSocket] = set()
        first = PRESETS[0]
        self.challenge: str | None = first.id
        self.goal, self.start_state = first.goal, first.start
        self.round_id, self.phase = 0, "setup"
        self.round: Round | None = None
        self.mouse: tuple[float, float] | None = None
        self.down = False
        self.last_decision: dict | None = None
        self.ai_task: asyncio.Task | None = None
        self.tick_task: asyncio.Task | None = None
        self.ai_result = None
        self._last_ai_pushes = 0
        self.winner, self.results = None, None
        self._record = Path(record_to) if record_to else None
        self._record_t0: float | None = None

    # ---- messages -------------------------------------------------------------------------
    def round_msg(self) -> dict:
        return {"type": "round", "id": self.round_id, "phase": self.phase, "challenge": self.challenge,
                "goal": list(self.goal), "start": _start_dict(self.start_state), "winner": self.winner,
                "results": self.results, "round_seconds": self.round_seconds}

    def board_state(self, board: str) -> dict:
        if self.round is not None and self.phase != "setup":
            sim = self.round.human_sim if board == "human" else self.round.ai_sim
            pusher, cube = sim.pusher, sim.cube
            pushes = self.round.human_pushes.count if board == "human" else self._last_ai_pushes
        else:
            pusher, cube, pushes = self.start_state.pusher, self.start_state.cube, 0
        return {"type": "state", "board": board, "pusher": list(pusher), "cube": list(cube),
                "overlap": overlap(cube, self.goal), "pushes": pushes}

    def snapshot(self, replay: bool = False) -> dict:
        return {"type": "snapshot", "round": self.round_msg(),
                "boards": {"human": self.board_state("human"), "ai": self.board_state("ai")},
                "last_decision": self.last_decision, "replay": replay,
                "jev_available": self.judge.available, "shape": SHAPE}

    async def broadcast(self, msg: dict) -> None:
        if self._record is not None:
            now = time.monotonic()
            self._record_t0 = self._record_t0 or now
            self._record.parent.mkdir(parents=True, exist_ok=True)
            with self._record.open("a") as f:
                f.write(json.dumps({"t": now - self._record_t0, "msg": msg}) + "\n")
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                self.clients.discard(ws)

    async def _emit_ai(self, msg: dict) -> None:
        if msg["type"] == "decision":
            self.last_decision = msg
        if msg["type"] == "state":
            self._last_ai_pushes = msg["pushes"]
        await self.broadcast(msg)

    # ---- round lifecycle ------------------------------------------------------------------
    def _time_up(self) -> bool:
        return self.round is not None and time.monotonic() - self.round.started_at >= self.round_seconds

    def is_current(self, round_id: int) -> bool:
        return round_id == self.round_id and self.phase == "playing" and not self._time_up()

    async def cancel_tasks(self) -> None:
        current = asyncio.current_task()
        for task in (self.ai_task, self.tick_task):
            if task is not None and task is not current and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def start(self, challenge: str | None, goal, start: SimState) -> None:
        await self.cancel_tasks()
        self.round_id += 1
        rid = self.round_id
        self.challenge, self.goal, self.start_state = challenge, clamp_goal(goal), start
        self.round = Round(rid, challenge, self.goal, start, self.round_seconds)
        self.phase, self.ai_result, self.winner, self.results = "playing", None, None, None
        self.last_decision, self.mouse, self.down, self._last_ai_pushes = None, None, False, 0
        await self.broadcast(self.round_msg())
        await self.broadcast(self.board_state("human"))
        await self.broadcast(self.board_state("ai"))
        self.ai_task = asyncio.create_task(self._run_ai(rid))
        self.tick_task = asyncio.create_task(self._tick(rid))

    async def _run_ai(self, rid: int) -> None:
        ctl = AIController(self.round.ai_sim, self.judge, self._emit_ai, tau=self.tau, step_delay_s=self.step_delay_s)
        result = await ctl.run(rid, self.goal, self.is_current)
        if rid == self.round_id:
            self.ai_result = result
            await self._maybe_finish(rid)

    async def _tick(self, rid: int) -> None:
        while rid == self.round_id and self.phase == "playing":
            await asyncio.sleep(self.tick_s)
            if not self._time_up() and not self.round.human_done:
                self.round.human_step(self.mouse, self.down)
                await self.broadcast(self.board_state("human"))
            await self._maybe_finish(rid)

    async def _maybe_finish(self, rid: int) -> None:
        if rid != self.round_id or self.phase != "playing":
            return
        human_finished = self.round.human_done or self._time_up()
        if human_finished and self.ai_result is not None:
            human = self.round.human_result()
            self.phase = "finished"
            self.results = {"human": asdict(human), "ai": asdict(self.ai_result)}
            self.winner = decide_winner(human, self.ai_result)
            await self.broadcast(self.round_msg())

    async def to_setup(self, challenge: str | None, goal, start: SimState) -> None:
        await self.cancel_tasks()
        self.round_id += 1  # invalidates anything still in flight
        self.phase, self.round, self.winner, self.results = "setup", None, None, None
        self.challenge, self.goal, self.start_state = challenge, clamp_goal(goal), start
        await self.broadcast(self.round_msg())
        await self.broadcast(self.board_state("human"))
        await self.broadcast(self.board_state("ai"))

    async def handle(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "mouse":
            self.mouse, self.down = (float(msg["x"]), float(msg["y"])), bool(msg.get("down"))
        elif kind == "start":
            if msg.get("challenge"):
                c = get_challenge(msg["challenge"])
                if c is not None:
                    await self.start(c.id, c.goal, c.start)
            else:
                await self.start(None, msg.get("goal") or self.goal, _parse_start(msg.get("start"), self.start_state))
        elif kind == "select":
            c = get_challenge(msg.get("challenge", ""))
            if c is not None and self.phase != "playing":
                await self.to_setup(c.id, c.goal, c.start)
        elif kind == "set_goal" and self.phase != "playing":
            await self.to_setup(None, msg["goal"], self.start_state)


async def _replay(ws: WebSocket, path: Path, speed: float) -> None:
    prev = 0.0
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        await asyncio.sleep(max(0.0, rec["t"] - prev) / speed)
        prev = rec["t"]
        await ws.send_json(rec["msg"])


def create_app(judge: JevJudge, tau: float = 0.5, step_delay_s: float = 0.1, tick_s: float = 0.1,
               round_seconds: float = 90.0, record_to: str | None = None, replay_from: str | None = None,
               replay_speed: float = 1.0) -> FastAPI:
    game = Game(judge, tau, step_delay_s, tick_s, round_seconds, record_to)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await game.cancel_tasks()

    app = FastAPI(lifespan=lifespan)
    app.state.game = game

    @app.get("/challenges")
    def challenges() -> list[dict]:
        return [{"id": c.id, "name": c.name, "tier": c.tier, "goal": list(c.goal), "start": _start_dict(c.start)}
                for c in PRESETS]

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        if replay_from:
            await ws.send_json(game.snapshot(replay=True))
            try:
                await _replay(ws, Path(replay_from), replay_speed)
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                return
        game.clients.add(ws)
        try:
            await ws.send_json(game.snapshot())
            while True:
                await game.handle(await ws.receive_json())
        except WebSocketDisconnect:
            pass
        finally:
            game.clients.discard(ws)

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app


app = create_app(
    JevJudge(),
    tau=float(os.environ.get("JEV_TAU", "0.5")),
    record_to=os.environ.get("RECORD_TO") or None,
    replay_from=os.environ.get("REPLAY_FROM") or None,
)

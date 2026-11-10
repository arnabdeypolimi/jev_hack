# Push-Cube: Human vs. Jev — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser game where a human and TypeSafe's Jev race to push a sliding cube onto a goal square in fewer pushes; Jev decides each push from raw coordinates and "thinks harder" by simulating options when unsure.

**Architecture:** A Python game server (FastAPI + WebSockets) owns two 2D pymunk cube sims, scoring, push counting, a push executor, the AI controller and the Jev judge (the only external call). A single vanilla-JS page renders both boards in 3D with Three.js and shows Jev's decision panel.

**Tech Stack:** Python 3.11 (uv), FastAPI, uvicorn[standard], httpx, pymunk, pytest, pytest-asyncio; vanilla JS + Three.js **0.186.1** (ES module via import map from `https://cdn.jsdelivr.net/npm/three@0.186.1/`, no build step).

**Spec:** `docs/superpowers/specs/2026-11-10-push-cube-vs-jev-design.md`

## Global Constraints

- Board 512×512 px, walls at **5** and **506**; y points down. Cube side **60** px, translation only (infinite moment of inertia). Pusher radius **15** px, kinematic, PD control k_p **100**, k_v **20**; physics 100 Hz substeps, 10 Hz control (one `step` = 10 substeps).
- Overlap = `max(0, 60 − |dx|) · max(0, 60 − |dy|) / 3600`; success at overlap ≥ **0.90**. Goal centre clamped to **[35, 476]** on both axes.
- Push = one continuous pusher–cube contact episode, same code for both boards.
- AI options: directions `N, NE, E, SE, S, SW, W, NW` (N = −y) × distances `short` **10**, `medium` **30**, `long` **80** px of pusher travel. Option id format: `"NE-medium"`.
- Jev: `POST https://api.typesafe.ai/v1/systemone`, model **`jev-1.14.0`**, header `Authorization: Bearer <key>`, JSON body, no images. Timeout **4 s**. τ = **0.5** (env `JEV_TAU`), think-harder at most once per push, top **3** options.
- `TYPESAFE_API_KEY` lives in the user's gitignored `.env`; processes run with `uv run --env-file .env ...`; read with `os.environ.get`; never sent to the browser, never logged, never committed. `.env.example` lists `TYPESAFE_API_KEY=` and `JEV_TAU=`.
- 3D mapping: sim `(x, y)` → Three.js `(x − 256, 0, y − 256)`. All sizes in JS come from the server snapshot.
- Every non-Jev decision shown in the UI is labelled "judge offline".

## Review Focus

- **New round started while Jev is deciding** → the stale decision is never executed on the new round. Test in Task 5 (`test_stale_decision_dropped_after_new_round`).
- **Cube pinned against a wall** → options pushing away from that wall stay feasible, options needing an approach outside the walls are infeasible, listed to Jev, and skipped if picked. Tests in Task 2 (`test_infeasible_when_cube_against_wall`) and Task 5 (`test_infeasible_pick_falls_to_next_best`).
- **Goal reached mid-push** → execution stops at that step; no extra pushes counted. Test in Task 5 (`test_stops_mid_push_on_success`).
- **Browser reconnects mid-round** → full snapshot on connect, round keeps running. Test in Task 7 (`test_snapshot_on_connect`).
- **Mouse outside the table / goal dragged into a wall** → clamped. Tests in Task 6 (`test_mouse_target_clamped`) and Task 3 (`test_clamp_goal`).

---

## File Structure

```
pyproject.toml, .env.example          (.gitignore already committed)
pushcube/
  sim.py            CubeSim: pymunk world, step, snapshot/restore, contact
  scoring.py        overlap(), clamp_goal(), PushCounter
  options.py        DIRECTIONS, DISTANCES, Option, all_options()
  executor.py       push_targets(), is_feasible(), execute_option()
  challenges.py     Challenge, PRESETS
  policies.py       geometric_option(), oracle_option()
  jev_judge.py      JevJudge, Decision, Pick
  controller.py     AIController, SideResult
  game_round.py     Round, decide_winner(), clamp_mouse(), human_target()
  server.py         FastAPI app, WS protocol, record/replay
  static/index.html, app.js, board3d.js, panel.js, style.css
scripts/
  dry_run_jev.py, harness.py
tests/              test_<module>.py per module
```

---

### Task 1: Scaffold and cube simulator

**Files:** Create `pyproject.toml`, `.env.example`, `pushcube/__init__.py`, `pushcube/sim.py`; Test `tests/test_sim.py`

**Interfaces:**
- Produces: `@dataclass SimState(pusher: tuple[float, float], cube: tuple[float, float], pusher_vel: tuple = (0, 0), cube_vel: tuple = (0, 0))`; `CubeSim()` with `reset(state: SimState)`, `step(target: tuple[float, float] | None)` (target = absolute pusher target; None holds position), `snapshot() -> SimState`, `restore(s: SimState)`, `copy() -> CubeSim`, `in_contact(eps: float = 1.0) -> bool`, properties `pusher`, `cube`, `steps: int`.

- [ ] **Step 1:** `uv init --package --python 3.11 --name pushcube`; `uv add fastapi "uvicorn[standard]" httpx pymunk`; `uv add --dev pytest pytest-asyncio numpy`; set `asyncio_mode = "auto"` under `[tool.pytest.ini_options]`; write `.env.example`.
- [ ] **Step 2: Write failing tests**

```python
def test_cube_does_not_rotate_when_pushed_off_centre():
    s = CubeSim(); s.reset(SimState(pusher=(200, 300), cube=(256, 280)))
    for _ in range(20): s.step((320, 300))
    assert s._cube_body.angle == 0.0 and s.cube[0] > 256

def test_cube_stops_after_contact_ends():
    s = CubeSim(); s.reset(SimState(pusher=(200, 256), cube=(256, 256)))
    for _ in range(8): s.step((300, 256))
    for _ in range(3): s.step((150, 256))           # retreat
    x1 = s.cube[0]
    for _ in range(10): s.step((150, 256))
    assert abs(s.cube[0] - x1) < 0.5

def test_walls_contain_cube():
    s = CubeSim(); s.reset(SimState(pusher=(400, 256), cube=(460, 256)))
    for _ in range(40): s.step((600, 256))
    assert s.cube[0] <= 506 - 30 + 0.5

def test_snapshot_restore_is_deterministic():
    a = CubeSim(); a.reset(SimState(pusher=(100, 100), cube=(200, 200)))
    b = a.copy()
    for sim in (a, b):
        for _ in range(15): sim.step((260, 230))
    assert a.cube == pytest.approx(b.cube) and a.pusher == pytest.approx(b.pusher)

def test_in_contact():
    s = CubeSim(); s.reset(SimState(pusher=(256 - 30 - 15, 256), cube=(256, 256)))
    assert s.in_contact(); s.reset(SimState(pusher=(50, 50), cube=(256, 256))); assert not s.in_contact()
```

- [ ] **Step 3:** Run `uv run pytest tests/test_sim.py -v` → FAIL.
- [ ] **Step 4: Implement.** pymunk `Space(gravity=0)`, static segment walls (radius 2), cube `Body(1, float("inf"))` + `Poly.create_box(60, 60)`, pusher `KINEMATIC` circle r 15 with Push-T's PD update per substep; damping via `space.damping` (start 0.02, tune until the stop test passes). `copy()` = new sim + `restore(snapshot())`. Contact = distance from pusher centre to cube box ≤ 15 + eps.
- [ ] **Step 5:** Run → PASS. **Step 6:** `git commit -m "feat: translation-only cube simulator"`

---

### Task 2: Scoring, options, push executor

**Files:** Create `pushcube/scoring.py`, `pushcube/options.py`, `pushcube/executor.py`; Test `tests/test_scoring.py`, `tests/test_executor.py`

**Interfaces:**
- Consumes: `CubeSim`, `SimState`.
- Produces: `overlap(cube, goal) -> float`; `clamp_goal(goal) -> tuple`; `PushCounter.update(contact: bool)`, `.count`; `DIRECTIONS: dict[str, tuple[float, float]]` (unit vectors, N = (0, −1)); `DISTANCES = {"short": 10, "medium": 30, "long": 80}`; `@dataclass(frozen=True) Option(direction: str, distance: str)` with `.id` (`"NE-medium"`) and `Option.from_id(s)`; `all_options() -> list[Option]` (24); `is_feasible(cube, option) -> bool` (approach point = cube − d·(30+15+2+20) inside [20, 491]); `push_targets(state: SimState, option) -> list[tuple[float, float]]` (per-step pusher targets: route to the approach point avoiding the inflated box half-width 30+15+5 via its corners, move to contact point, push d·distance at ≤ 10 px per step, retreat 10 px); `execute_option(sim, option, on_step: Callable[[CubeSim], bool] | None = None) -> ExecResult(contact_episodes: int, steps: int, stopped_early: bool)` — `on_step` returning True stops early.

- [ ] **Step 1: Write failing tests**

```python
def test_overlap_values():
    assert overlap((100, 100), (100, 100)) == 1.0
    assert overlap((106, 100), (100, 100)) == pytest.approx(0.9)
    assert overlap((200, 100), (100, 100)) == 0.0

def test_clamp_goal():
    assert clamp_goal((0, 600)) == (35, 476)

def test_push_counter_counts_contact_episodes():
    c = PushCounter()
    for x in [False, True, True, False, True, False]: c.update(x)
    assert c.count == 2

def test_all_options_and_ids():
    assert len(all_options()) == 24 and Option.from_id("SW-long") == Option("SW", "long")

@pytest.mark.parametrize("opt", all_options())
def test_execute_option_is_one_contact_episode_and_moves_cube_in_direction(opt):
    s = CubeSim(); s.reset(SimState(pusher=(60, 60), cube=(256, 256)))
    if not is_feasible(s.cube, opt): pytest.skip()
    before = np.array(s.cube); r = execute_option(s, opt)
    moved = np.array(s.cube) - before
    assert r.contact_episodes == 1
    assert moved @ np.array(DIRECTIONS[opt.direction]) > 0.25 * DISTANCES[opt.distance]

def test_approach_never_touches_cube_before_push(): ...   # pusher starts on the far side; contact only begins on the final straight
def test_infeasible_when_cube_against_wall():
    cube = (476, 256)                                  # touching east wall
    assert not is_feasible(cube, Option("W", "short")) # approach point (553, 256) is outside the walls
    assert is_feasible(cube, Option("N", "short"))     # approach point (476, 333) is inside
    assert is_feasible(cube, Option("E", "short"))     # approach is fine; the wall simply blocks the cube
def test_execute_stops_early_when_on_step_true(): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: scoring, push options and executor"`

---

### Task 3: Challenges and baseline policies

**Files:** Create `pushcube/challenges.py`, `pushcube/policies.py`; Test `tests/test_challenges.py`, `tests/test_policies.py`

**Interfaces:**
- Consumes: `CubeSim`, `Option`, `all_options`, `is_feasible`, `execute_option`, `overlap`, `clamp_goal`.
- Produces: `@dataclass Challenge(id, name, tier: Literal["easy","medium","hard","far"], start: SimState, goal: tuple)`; `PRESETS` (≥ 2 per tier, ranges from spec §5.9); `geometric_option(sim, goal) -> Option` (feasible direction with largest cosine to goal − cube; distance with |travel − (|offset| − 8)| minimal, 8 px = expected slide, tune); `oracle_option(sim, goal) -> tuple[Option, dict[str, float]]` (simulate every feasible option on `sim.copy()`, return the max overlap-after, ties → shorter distance).

- [ ] **Step 1: Write failing tests**

```python
def test_presets_cover_tiers_and_goals_are_clamped():
    assert {c.tier for c in PRESETS} == {"easy", "medium", "hard", "far"}
    assert all(clamp_goal(c.goal) == c.goal for c in PRESETS)

@pytest.mark.parametrize("policy", ["geometric", "oracle"])
def test_policy_solves_every_preset_within_25_pushes(policy):
    for c in PRESETS:
        assert play(policy, c, max_pushes=25).reached, c.id   # helper loops policy → execute_option

def test_oracle_never_worse_than_geometric_on_one_step(): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run → PASS (if `hard` presets are unsolvable by the oracle, move their goals until they are — a preset nobody can finish is a bug).
- [ ] **Step 5:** `git commit -m "feat: challenge presets and baseline policies"`

---

### Task 4: Jev judge and Jev dry run

**Files:** Create `pushcube/jev_judge.py`, `scripts/dry_run_jev.py`; Test `tests/test_jev_judge.py`

**Interfaces:**
- Consumes: `Option`, `DIRECTIONS`, `DISTANCES`.
- Produces:
  - `@dataclass Decision(option: Option, direction_probs: dict[str, float], distance_probs: dict[str, float], confidence: float, source: Literal["jev","fallback"], latency_ms: float, model: str | None)`
  - `@dataclass Pick(option: Option, probs: dict[str, float], confidence: float, source: Literal["jev","fallback"], latency_ms: float)`
  - `JevJudge(api_key: str | None = None` (None → `os.environ.get("TYPESAFE_API_KEY")`)`, model="jev-1.14.0", timeout_s=4.0, transport: httpx.AsyncBaseTransport | None = None, log_path="recordings/jev_log.jsonl")`
  - `async decide(state: dict, infeasible: set[str], fallback: Option) -> Decision` — one request with Choice `direction` (8 options, described in words, e.g. `"N": "move the cube toward the top of the board (−y)"`) and Choice `distance` (`"short": "about 10 px"`, …). Option = argmax direction × argmax distance; if infeasible, the next-best feasible pair by product probability. Confidence = min of the two confidences. Any error / timeout / 429 / missing key → `Decision(fallback, source="fallback", confidence=0.0, …)`.
  - `async pick(state: dict, outcomes: list[dict], fallback: Option) -> Pick` — Choice `pick` over the outcome option ids; each description is generated from the outcome: `"cube ends at (x, y), offset to goal (dx, dy), overlap 72%, touched wall: no"`.
  - `top_options(decision: Decision, infeasible: set[str], k: int = 3) -> list[Option]` (by product probability).
- `state` built by `build_state(sim, goal, pushes, infeasible) -> dict` (also in this module): coordinate convention text, board size, walls, cube, pusher, goal, offset, overlap (rounded to 3 dp), pushes, infeasible ids.

- [ ] **Step 1: Write failing tests** (`httpx.MockTransport`)

```python
async def test_decide_combines_direction_and_distance():
    d = await judge(answers(direction="NE", dir_p={"NE": .7, "E": .3}, distance="medium", dist_p={"medium": .9, "long": .1})).decide(STATE, set(), FALLBACK)
    assert d.option == Option("NE", "medium") and d.source == "jev" and d.confidence == pytest.approx(min_conf)

async def test_request_shape_pinned_model_no_images():   # json["model"]=="jev-1.14.0", no "assets", Bearer header, questions direction+distance
async def test_infeasible_choice_falls_to_next_best_pair(): ...
async def test_timeout_and_429_and_missing_key_fall_back(): ...   # three cases → source "fallback", option == FALLBACK
async def test_pick_maps_choice_to_outcome_option(): ...
def test_top_options_skips_infeasible_and_orders_by_product(): ...
def test_log_never_contains_key(tmp_path): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** with `httpx.AsyncClient`. **Step 4:** Run → PASS.
- [ ] **Step 5: Dry run.** `scripts/dry_run_jev.py` plays one easy and one medium preset with Jev for up to 20 pushes (headless, think-harder on), printing per push: option, confidence, whether it escalated, overlap, latency. Run `uv run --env-file .env python scripts/dry_run_jev.py`. Expected: reaches ≥ 90% on the easy preset; record pushes, median latency and escalation rate in `AGENTS.md` under "Jev dry run".
- [ ] **Step 6:** `git commit -m "feat: Jev judge with fallback, plus dry run"`

---

### Task 5: AI controller

**Files:** Create `pushcube/controller.py`; Test `tests/test_controller.py`

**Interfaces:**
- Consumes: `CubeSim`, `PushCounter`, `overlap`, `is_feasible`, `all_options`, `execute_option`, `geometric_option`, `JevJudge`, `build_state`, `top_options`.
- Produces: `@dataclass SideResult(pushes: int, reached: bool, best_overlap: float, finished_at_s: float | None)`; `AIController(sim, judge, emit: Callable[[dict], Awaitable[None]], tau: float = 0.5, step_delay_s: float = 0.1)`; `async run(round_id, goal, is_current: Callable[[int], bool], max_pushes: int = 40) -> SideResult`.
- Per push: emit `ai_status` `deciding` → `decide()` → emit `decision` → if `source=="jev"` and confidence < τ: emit `ai_status` `thinking_harder`, simulate `top_options(..., k=3)` each on `sim.copy()` with `execute_option`, emit `imagination` (outcomes), `pick()`, emit `pick` → emit `ai_status` `acting` → `execute_option(sim, option, on_step=...)` sleeping `step_delay_s` per step, updating `PushCounter`, emitting `state` per step, stopping when overlap ≥ 0.90 or `not is_current(round_id)`. Fallback decisions emit `ai_status` `judge_offline`.

- [ ] **Step 1: Write failing tests** (scripted judge)

```python
async def test_confident_decision_executes_without_thinking_harder(): ...   # 1 decide call, 0 pick calls, statuses deciding→acting
async def test_low_confidence_simulates_three_and_picks():
    res = await run_with(ScriptedJudge(decisions=[dec(conf=0.3)], picks=[pick("E-long")]))
    msgs = emitted("imagination")[0]["outcomes"]; assert len(msgs) == 3
    assert executed_options() == [Option("E", "long")]
async def test_imagination_sims_do_not_touch_real_sim(): ...  # real sim state unchanged until acting
async def test_stops_mid_push_on_success(): ...               # goal 5 px ahead → stops early, pushes == 1
async def test_stale_decision_dropped_after_new_round(): ...  # is_current flips during decide → no step executed
async def test_infeasible_pick_falls_to_next_best(): ...
async def test_fallback_emits_judge_offline(): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: AI controller with think-harder and fallback"`

---

### Task 6: Round logic and human control

**Files:** Create `pushcube/game_round.py`; Test `tests/test_game_round.py`

**Interfaces:**
- Consumes: `SideResult`, `Challenge`, `CubeSim`, `PushCounter`, `overlap`, `clamp_goal`.
- Produces: `decide_winner(human: SideResult, ai: SideResult) -> Literal["human","ai","tie"]` (both reached → fewer pushes, equal → tie; one reached → it; neither → higher best_overlap, within 0.005 → tie); `clamp_mouse(x, y) -> tuple` into [20, 491]; `human_target(mouse: tuple | None, down: bool, pusher) -> tuple | None` (clamped mouse when down, else None); `Round(id, challenge | None, goal, start, round_seconds=90.0)` with `human_sim`, `ai_sim`, `human_pushes: PushCounter`, `human_step(mouse, down)` (one 10 Hz step + contact update), `human_result() -> SideResult`.

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("h,a,w", [
    (SideResult(12, True, .93, 40), SideResult(9, True, .91, 50), "ai"),
    (SideResult(7, True, .95, 30), SideResult(9, True, .91, 50), "human"),
    (SideResult(9, True, .9, 30), SideResult(9, True, .9, 50), "tie"),
    (SideResult(3, False, .8, None), SideResult(20, True, .9, 80), "ai"),
    (SideResult(3, False, .8, None), SideResult(4, False, .6, None), "human"),
])
def test_decide_winner(h, a, w): assert decide_winner(h, a) == w

def test_mouse_target_clamped(): assert clamp_mouse(-50, 900) == (20, 491)
def test_round_starts_both_sims_identically(): ...
def test_human_push_counted_once_per_contact(): ...   # drive human_step through two separate contacts → 2
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run → PASS.
- [ ] **Step 5:** `git commit -m "feat: round state, winner rule, human control"`

---

### Task 7: Game server, WebSocket protocol, record/replay

**Files:** Create `pushcube/server.py`; Test `tests/test_server.py`

**Interfaces:**
- Consumes: `Round`, `AIController`, `JevJudge`, `PRESETS`, `clamp_goal`, `decide_winner`.
- Produces: `create_app(judge: JevJudge) -> FastAPI`; module-level `app`. Run: `uv run --env-file .env uvicorn pushcube.server:app --port 8000`. Serves `static/` at `/`, WS `/ws`, `GET /challenges`.
- Server → client:
  - `snapshot`: `{round, boards, last_decision, replay: bool, shape: {"cube_size": 60, "pusher_radius": 15, "walls": [5, 506], "board_size": 512}}` on connect
  - `state`: `{"board": "human"|"ai", "pusher": [x, y], "cube": [x, y], "overlap": f, "pushes": n}`
  - `round`: `{"id", "phase": "setup"|"playing"|"finished", "challenge", "goal": [x, y], "start", "winner", "results"}`
  - `ai_status`: `{"status": "deciding"|"thinking_harder"|"acting"|"judge_offline"|"done"}`
  - `decision`: `{"option", "direction_probs", "distance_probs", "confidence", "source", "latency_ms"}`
  - `imagination`: `{"outcomes": [{"option", "cube_after": [x, y], "overlap_after", "touched_wall"}]}`
  - `pick`: `{"option", "probs", "confidence", "source"}`
- Client → server: `mouse {x, y, down}`, `start {challenge}` or `start {goal, start}`, `set_goal {goal}` (setup only; clamped).
- Human tick at 10 Hz; AI task per round; a new `start` increments the round id and cancels the previous task. `RECORD_TO=<path>` appends every broadcast with timestamps; `REPLAY_FROM=<path>` streams a recording instead of live play (`snapshot.replay = true`).

- [ ] **Step 1: Write failing tests** (`TestClient.websocket_connect`, judge without key → fallback)

```python
def test_snapshot_on_connect(): ...                 # first message snapshot; shape.cube_size == 60
def test_start_broadcasts_round_and_states(): ...
def test_set_goal_is_clamped(): ...
def test_new_start_cancels_previous_ai_task(): ...
def test_api_key_never_sent_to_client(): ...        # TYPESAFE_API_KEY="sk-test" never appears in any message
def test_record_then_replay_same_message_types(tmp_path): ...
```

- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run `uv run pytest -q` → all PASS.
- [ ] **Step 5:** `git commit -m "feat: game server, WS protocol, record/replay"`

---

### Task 8: Frontend — 3D JEV board and decision panel

**Files:** Create `pushcube/static/index.html`, `app.js`, `board3d.js`, `panel.js`, `style.css`

**Interfaces:**
- Consumes: WS protocol (Task 7).
- `board3d.js`: `createBoard3D(container, shape) -> {setState(pusher, cube, tMs), setGoal(goal), setView("tilt"|"top"), pickBoardPoint(ev) -> [x, y] | null, dispose()}`; exported `simToWorld`, `worldToSim`. Scene: table, walls, `BoxGeometry(60, 60, 60)` cube resting on the table, cylinder pusher r 15, emissive square goal outline, shadows; interpolate between the last two states; `V` toggles tilted perspective / top-down orthographic.
- `panel.js`: `createPanel(container) -> {showDecision(msg), showImagination(msg), showPick(msg), setStatus(status)}` — compass rose (8 wedges sized/labelled by `direction_probs`, chosen wedge highlighted), distance bars, confidence meter, "Jev isn't sure… simulating options" state, 3 mini top-down boards (2D canvas) for `imagination.outcomes` with pick probabilities, "judge offline" badge.
- `app.js`: layout — YOU and JEV boards side by side with HUD `pushes: N   overlap: NN%`; panel under JEV; wires WS messages.

- [ ] **Step 1: Implement** using the `frontend-design` skill (dark, projector-friendly).
- [ ] **Step 2: Verify manually** — `uv run --env-file .env uvicorn pushcube.server:app --port 8000`, open http://localhost:8000, start an easy preset. Expected: JEV cube slides on the 3D table; compass and distance bars update every push; low-confidence pushes show 3 simulated futures before acting; with `.env` absent the badge reads "judge offline" and play continues; top-down view (`V`) matches the sim coordinates; no console errors.
- [ ] **Step 3:** `git commit -m "feat: 3D Jev board and decision panel"`

---

### Task 9: Frontend — human board, picker, draggable goal, results

**Files:** Modify `pushcube/static/index.html`, `app.js`, `style.css`

**Interfaces:**
- Consumes: `pickBoardPoint`, `setGoal`, `setView` (Task 8); `GET /challenges`.
- Produces: pointer on the YOU board → `mouse` messages at ≤ 30 Hz during play (off-table → `down: false`); YOU board defaults to top-down; challenge picker by tier; goal draggable on either board in setup (sends `set_goal`, applies the clamped echo); results panel with winner, pushes and overlap.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify manually** — pusher follows the pointer to the exact table spot in both views; each separate contact adds one push; dragging the goal into a wall stops at the wall; winner banner matches `decide_winner`.
- [ ] **Step 3:** `git commit -m "feat: human board, picker, draggable goal, results"`

---

### Task 10: Play-quality harness

**Files:** Create `scripts/harness.py`; Test `tests/test_harness.py`

**Interfaces:**
- Consumes: `PRESETS`, `CubeSim`, `JevJudge`, `AIController` logic (headless, `step_delay_s=0`), `geometric_option`, `oracle_option`.
- Produces: `play_headless(policy: Literal["jev","geometric","oracle"], challenge, judge=None, max_pushes=40) -> SideResult`; CLI `uv run --env-file .env python scripts/harness.py --repeats 3 --out recordings/harness.json` printing per tier: success rate and mean pushes for each policy, Jev escalation rate and median latency.

- [ ] **Step 1: Write failing test** — `test_play_headless_geometric_reaches_easy_goal`.
- [ ] **Step 2:** Run → FAIL. **Step 3: Implement.** **Step 4:** Run → PASS.
- [ ] **Step 5:** Run the harness with the real key; record the table in `AGENTS.md`; tune `JEV_TAU` and the Jev wording if Jev trails the geometric rule badly.
- [ ] **Step 6:** `git commit -m "feat: play-quality harness"`

---

### Stretch (separate short plans once Tasks 1–10 are done)

Audience rules (Jev Noul per option) → sabotage → obstacles / two cubes → leaderboard.

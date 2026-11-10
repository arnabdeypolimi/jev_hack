# Push-T: Human vs. Imagination + Judgment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser game where a human and an AI (JEPA world model imagines candidate pushes, TypeSafe's Jev picks one) race to push a T block onto a goal outline in fewer pushes.

**Architecture:** A Python game server (FastAPI + WebSockets) runs two copies of the original Push-T pymunk env, scoring, push counting, the AI controller and the Jev judge. A separate planner service (FastAPI) loads `jepa_wm_pusht` via torch.hub, runs our own device-agnostic CEM, and returns K=4 diverse candidates with decoded pictures. A single vanilla-JS canvas page draws both boards and the decision/imagination strips.

**Tech Stack:** Python 3.11 (uv), FastAPI, uvicorn, httpx, pymunk, pygame, shapely, opencv-python-headless, scikit-image, gym 0.26.2, numpy<2, pytest, pytest-asyncio; planner extra: torch, torchvision, pyyaml, omegaconf, huggingface_hub, tensordict, einops; vanilla JS + canvas.

**Spec:** `docs/superpowers/specs/2026-11-10-push-t-vs-ai-design.md`

## Global Constraints

- The model-facing renderer is the vendored jepa-wms `PushTEnv` unchanged (render_size 224, `with_target=True`, `with_velocity=True`, default `goal_pose = [256, 256, π/4]` left as-is so frames match training). Our goal outline is drawn only on the audience canvas.
- Success threshold: overlap ≥ **0.90**. Overlap = area(block ∩ goal T) ÷ area(goal T), using the env's own `pymunk_to_shapely` and `block.shapes`.
- Push = one continuous pusher–T contact episode, same code for both boards.
- World model: `torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')`; context window 2 frames spaced 5 env steps apart (frameskip 5); proprio = `[agent_x, agent_y, agent_vx, agent_vy]`; cost = mean-squared latent distance at the final predicted step, visual + **alpha 0.1** × proprio.
- CEM defaults from repo config: horizon **6**, num_samples **300**, iterations **30**, num_elites **10**, var_scale **1.0**. Escalation doubles num_samples, warm-started.
- K = **4** candidates, labels **A–D**, option order shuffled per Jev request.
- Jev: `POST https://api.typesafe.ai/v1/systemone`, model **`jev-1.14.0`**, headers `Authorization: Bearer $TYPESAFE_API_KEY` and `TypeSafe-Danger-Zone-Images: true`, images as `assets` with `data_base64` + `format: "png"`, references are fresh UUID4 strings.
- Escalation threshold τ = **0.5** (env `JEV_TAU`), max **2** escalations. Jev timeout **4 s**, planner timeout **30 s**.
- `TYPESAFE_API_KEY` is read from the environment on the game server only; never sent to the browser, never committed (`.env` in `.gitignore`).
- Any fake planner must render a visible "FAKE PLANNER" banner in the UI.

## Review Focus

- **New round started while the AI is mid-plan** → the stale plan must never be applied to the new round. Test in Task 8 (`test_stale_plan_dropped_after_new_round`).
- **Browser reconnects mid-round** → server keeps running and sends a full snapshot on connect. Test in Task 10 (`test_snapshot_on_connect`).
- **Mouse outside the board** → target clamped into [5+15, 506−15] px. Test in Task 9 (`test_mouse_target_clamped`).
- **Goal dragged so the T would poke through a wall** → goal clamped so the whole T polygon is inside the walls. Test in Task 3 (`test_clamp_goal_inside_walls`).
- **T reaches 90% in the middle of an AI action chunk** → AI stops at that env step; push count not incremented further. Test in Task 8 (`test_stops_mid_chunk_on_success`).

---

## File Structure

```
pyproject.toml                    uv project; extras: planner, dev
.gitignore                        .venv, __pycache__, .env, recordings/, *.png in scratch/
pusht_game/
  vendor/pusht_env.py             verbatim copy of jepa-wms evals/simu_env_planning/envs/pusht_env/pusht_env.py (MIT)
  board.py                        Board: one sim instance + frame history + push counter
  geometry.py                     overlap(), in_contact(), clamp_goal()
  pushes.py                       PushCounter
  challenges.py                   Challenge dataclass, PRESETS
  goal_image.py                   render_goal(goal_pose) -> GoalImage
  imaging.py                      png_b64(ndarray) -> str
  planner_client.py               PlanRequest/Candidate/PlanResult types, PlannerClient
  jev_judge.py                    JevJudge, Decision
  controller.py                   AIController (imagine → decide → escalate → act)
  game_round.py                   Round, SideResult, decide_winner()
  server.py                       FastAPI app, WS protocol, human tick loop
  static/index.html, app.js, style.css
planner_service/
  model.py                        load_world_model()
  cem.py                          cem_plan()
  candidates.py                   select_diverse()
  backend.py                      JepaBackend, FakeBackend (both implement plan(PlanRequest)->PlanResult)
  service.py                      FastAPI POST /plan
scripts/
  dry_run_jepa.py, dry_run_jev.py, jev_harness.py
tests/                            mirrors modules: test_board.py, test_geometry.py, ...
```

Shared types (defined in Task 6, `pusht_game/planner_client.py`, imported by `planner_service`):

```python
@dataclass
class PlanRequest:
    obs_png_b64: list[str]          # 2 context frames, oldest first, 224×224 model renders
    proprio: list[list[float]]      # 2 × [ax, ay, vx, vy]
    goal_png_b64: str
    goal_proprio: list[float]       # [park_x, park_y, 0, 0]
    num_samples: int = 300
    iterations: int = 30
    warm_start_mean: list[list[float]] | None = None   # (horizon, model_action_dim), normalized

@dataclass
class Candidate:
    id: str                          # "c0".."c3", stable within one PlanResult
    env_actions: list[list[float]]   # horizon*5 relative env actions (denormalized, each [dx, dy])
    latent_cost: float
    ending_png_b64: str              # decoded final predicted frame
    frames_png_b64: list[str]        # decoded frames +1..+4

@dataclass
class PlanResult:
    candidates: list[Candidate]      # sorted by latent_cost ascending, len == 4
    mean: list[list[float]]          # final CEM mean, for warm start
    goal_decoded_png_b64: str
    fake: bool = False
```

---

### Task 1: Project scaffold, vendored env, Board

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `pusht_game/__init__.py`, `pusht_game/vendor/__init__.py`, `pusht_game/vendor/pusht_env.py`, `pusht_game/board.py`, `pusht_game/imaging.py`
- Test: `tests/test_board.py`

**Interfaces:**
- Produces: `Board(seed: int = 0)` with `reset(state: list[float]) -> None` (7-dim `[ax, ay, bx, by, θ, vx, vy]`), `step(rel_action: tuple[float, float] | None) -> None` (one env step = 10 substeps @100 Hz), `state -> np.ndarray` (7,), `agent_pos -> tuple`, `block_pose -> tuple[float, float, float]`, `proprio -> np.ndarray` (4,), `model_frame() -> np.ndarray` (224,224,3 uint8), `context_frames() -> tuple[np.ndarray, np.ndarray]` (frame 5 env steps ago, current; duplicates current when history is short), `context_proprio() -> tuple[np.ndarray, np.ndarray]`, `env` (the underlying `PushTEnv`). `png_b64(img: np.ndarray) -> str` in `imaging.py`.

- [ ] **Step 1: Scaffold.** `git init`; `uv init --package --python 3.11`; `uv add fastapi "uvicorn[standard]" httpx pymunk pygame shapely opencv-python-headless scikit-image "gym==0.26.2" "numpy<2" matplotlib pillow`; `uv add --optional planner torch torchvision pyyaml omegaconf huggingface_hub tensordict einops`; `uv add --dev pytest pytest-asyncio`. Copy the upstream env file verbatim into `pusht_game/vendor/pusht_env.py` (keep its MIT header). Set `SDL_VIDEODRIVER=dummy` at the top of `board.py` before importing the env.

- [ ] **Step 2: Write failing tests**

```python
def test_reset_sets_state():
    b = Board(); b.reset([100, 100, 256, 300, 0.5, 0, 0])
    assert np.allclose(b.block_pose, (256, 300, 0.5), atol=1e-3)
    assert np.allclose(b.agent_pos, (100, 100), atol=1e-3)

def test_model_frame_shape_and_dtype():
    b = Board(); b.reset([100, 100, 256, 300, 0.0, 0, 0])
    f = b.model_frame(); assert f.shape == (224, 224, 3) and f.dtype == np.uint8

def test_step_moves_agent_toward_relative_target():
    b = Board(); b.reset([100, 100, 400, 400, 0.0, 0, 0])
    for _ in range(5): b.step((0.3, 0.0))   # 30 px right per step target
    assert b.agent_pos[0] > 130

def test_context_frames_are_five_steps_apart():
    b = Board(); b.reset([100, 100, 400, 400, 0.0, 0, 0])
    for _ in range(6): b.step((0.2, 0.0))
    old, cur = b.context_frames()
    assert not np.array_equal(old, cur)
    assert np.array_equal(cur, b.model_frame())

def test_png_b64_roundtrip_header():
    assert base64.b64decode(png_b64(np.zeros((4, 4, 3), np.uint8)))[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 3: Run** `uv run pytest tests/test_board.py -v` → FAIL (import error).
- [ ] **Step 4: Implement `Board`** wrapping `PushTEnv(with_velocity=True, with_target=True)`; `reset` sets `env.reset_to_state` then calls `env.reset()`; history is a deque of the last 6 model frames and proprios appended after every `step`.
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** `git commit -m "feat: scaffold project, vendor Push-T env, Board wrapper"`

---

### Task 2: Geometry and push counting

**Files:**
- Create: `pusht_game/geometry.py`, `pusht_game/pushes.py`
- Test: `tests/test_geometry.py`, `tests/test_pushes.py`

**Interfaces:**
- Consumes: `Board` (Task 1).
- Produces: `overlap(board: Board, goal_pose: tuple[float, float, float]) -> float` in [0, 1]; `in_contact(board: Board, eps: float = 1.0) -> bool` (shapely distance from agent centre to block geometry ≤ 15 + eps); `PushCounter` with `update(contact: bool) -> None`, `count: int`.

- [ ] **Step 1: Write failing tests**

```python
def test_overlap_identical_pose_is_one():
    b = Board(); b.reset([50, 50, 256, 300, 0.7, 0, 0])
    assert overlap(b, (256, 300, 0.7)) == pytest.approx(1.0, abs=1e-3)

def test_overlap_far_pose_is_zero():
    b = Board(); b.reset([50, 50, 150, 150, 0.0, 0, 0])
    assert overlap(b, (400, 400, 0.0)) == 0.0

def test_overlap_wraps_angle():
    b = Board(); b.reset([50, 50, 256, 300, 0.1, 0, 0])
    assert overlap(b, (256, 300, 0.1 + 2 * math.pi)) == pytest.approx(1.0, abs=1e-3)

def test_in_contact_true_when_touching_false_when_far(): ...  # agent at block edge vs (50, 50) with block at (300, 300)

def test_push_counter_counts_contact_episodes():
    c = PushCounter()
    for x in [False, True, True, False, False, True, False]: c.update(x)
    assert c.count == 2
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (goal geometry via `env._get_goal_pose_body(np.array(goal_pose))` + `pymunk_to_shapely(goal_body, env.block.shapes)`). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: overlap scoring, contact detection, push counter`

---

### Task 3: Challenges, goal clamping, goal image

**Files:**
- Create: `pusht_game/challenges.py`, `pusht_game/goal_image.py`; Modify: `pusht_game/geometry.py`
- Test: `tests/test_challenges.py`, `tests/test_goal_image.py`

**Interfaces:**
- Produces: `@dataclass Challenge(id: str, name: str, tier: Literal["easy","medium","hard"], start_state: list[float], goal_pose: tuple[float,float,float])`; `PRESETS: list[Challenge]` (≥ 2 per tier; easy = shift ≤ 60 px and |Δθ| ≤ 20°, medium ≤ 120 px / ≤ 45°, hard = |Δθ| ≥ 90°); `clamp_goal(goal_pose) -> tuple` (moves the pose so the T polygon lies within x, y ∈ [5, 506]); `@dataclass GoalImage(png_b64: str, frame: np.ndarray, proprio: list[float])`; `render_goal(goal_pose) -> GoalImage` — a dedicated `Board` reset to `[park_x, park_y, gx, gy, gθ, 0, 0]`, where park is whichever of (40, 40), (472, 40), (40, 472), (472, 472) is farthest from the goal centre.

- [ ] **Step 1: Write failing tests**

```python
def test_presets_cover_all_tiers_and_are_inside_walls():
    tiers = {c.tier for c in PRESETS}; assert tiers == {"easy", "medium", "hard"}
    for c in PRESETS: assert clamp_goal(c.goal_pose) == pytest.approx(c.goal_pose)

def test_clamp_goal_inside_walls():
    g = clamp_goal((500, 500, 0.0))
    poly = goal_polygon(g)        # helper in geometry.py, shapely Polygon/MultiPolygon of the T at pose
    minx, miny, maxx, maxy = poly.bounds
    assert minx >= 5 and miny >= 5 and maxx <= 506 and maxy <= 506

def test_goal_image_parks_agent_far_from_goal():
    gi = render_goal((120, 120, 0.0))
    assert gi.proprio == [472, 472, 0, 0]
    assert gi.frame.shape == (224, 224, 3)
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (`clamp_goal` translates by the overshoot of the polygon bounds; angle unchanged). **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: challenge presets, goal clamping, goal image rendering`

---

### Task 4: World-model loader and JEPA dry run (milestone 0)

**Files:**
- Create: `planner_service/__init__.py`, `planner_service/model.py`, `scripts/dry_run_jepa.py`

**Interfaces:**
- Produces: `load_world_model(device: str | None = None) -> WorldModel` where `@dataclass WorldModel(wm, preprocessor, device: torch.device, model_action_dim: int, frameskip: int = 5, env_action_dim: int = 2)`; `wm` exposes repo methods `encode(obs_dict) -> TensorDict`, `unroll(z_ctxt, act_suffix=[T,B,A]) -> TensorDict [T+2, B, ...]`, `decode_unroll(encs, batch=True) -> uint8 [B, T, 224, 224, 3]`.

- [ ] **Step 1: Implement `load_world_model`**: `torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')`; device from arg or env `PLANNER_DEVICE` (default: cuda if available else cpu; "mps" → try `wm.to("mps")` and fall back to cpu with a logged warning); `model_action_dim = wm.action_dim`; assert `"image_head" in wm.heads` with message "decoder head missing — check heads_cfg in hub config".
- [ ] **Step 2: Write `scripts/dry_run_jepa.py`** that: builds a `Board` from the first easy preset; encodes 2 context frames + proprio (`visual` as float tensor [1, 2, 3, 224, 224] in 0–255, `proprio` [1, 2, 4]); unrolls one random normalized action sequence (6, 1, model_action_dim); decodes; saves `scratch/dry_run_decoded_{i}.png`; runs `cem_plan` (once Task 5 exists, behind `--cem`) for the easy and one hard preset; prints device, `model_action_dim`, encode / unroll / decode / CEM wall time, and the overlap reached after executing the plan in the env.
- [ ] **Step 3: Run** `uv run --extra planner python scripts/dry_run_jepa.py` on the Mac (CPU) and on the GPU box. Expected: prints `model_action_dim == 10` (2 × frameskip; if different, update `WorldModel` and the reshape in Task 5), decoded PNGs look like a Push-T board. Record timings and findings in `AGENTS.md` under "Dry-run results". Add any missing import surfaced by the hub repo with `uv add --optional planner <pkg>`.
- [ ] **Step 4: Commit** `feat: world-model loader and JEPA dry-run script`

---

### Task 5: CEM and diverse candidate selection

**Files:**
- Create: `planner_service/cem.py`, `planner_service/candidates.py`
- Test: `tests/test_cem.py`, `tests/test_candidates.py`

**Interfaces:**
- Produces: `latent_cost(pred: TensorDict, goal: TensorDict, alpha: float = 0.1) -> Tensor[B]` (final step only, mean-squared over non-batch dims, visual + alpha·proprio — same as repo `ReprTargetDistMPCObjective` with `sum_all_diffs=False`); `cem_plan(cost_fn: Callable[[Tensor[T,B,A]], Tensor[B]], horizon: int, action_dim: int, num_samples: int, iterations: int, num_elites: int, var_scale: float = 1.0, warm_start_mean: Tensor | None = None, device="cpu", generator=None) -> CEMOutput(mean: Tensor[T,A], elites: Tensor[T,E,A], elite_costs: Tensor[E])`; `select_diverse(elites: Tensor[T,E,A], costs: Tensor[E], k: int = 4, min_dist: float = 0.25) -> list[int]` (greedy by ascending cost, skipping elites whose first-step action is within `min_dist` L2 of an already-picked one; if fewer than k pass, fill with the next-cheapest unpicked).

- [ ] **Step 1: Write failing tests**

```python
def test_cem_finds_quadratic_minimum():
    target = torch.tensor([0.5, -0.3])
    cost = lambda a: ((a - target) ** 2).sum(dim=(0, 2))          # a: [T, B, A]
    out = cem_plan(cost, horizon=3, action_dim=2, num_samples=200, iterations=20, num_elites=10,
                   generator=torch.Generator().manual_seed(0))
    assert torch.allclose(out.mean, target.expand(3, 2), atol=0.05)
    assert out.elites.shape == (3, 10, 2) and out.elite_costs.shape == (10,)

def test_warm_start_used_as_initial_mean():
    ws = torch.full((3, 2), 0.7)
    cost = lambda a: ((a - ws[:, None]) ** 2).sum(dim=(0, 2))       # minimum exactly at ws
    out = cem_plan(cost, 3, 2, 50, 1, 5, warm_start_mean=ws, generator=torch.Generator().manual_seed(0))
    assert torch.equal(out.elites[:, 0], ws) and out.elite_costs[0] == 0

def test_select_diverse_skips_near_duplicates():
    elites = torch.tensor([[[0., 0.], [0.01, 0.], [1., 0.], [0., 1.], [-1., 0.]]])   # T=1, E=5
    idx = select_diverse(elites, torch.arange(5.), k=4, min_dist=0.25)
    assert idx == [0, 2, 3, 4]

def test_select_diverse_fills_when_not_enough_distinct():
    elites = torch.zeros(1, 5, 2)
    assert select_diverse(elites, torch.arange(5.), k=4) == [0, 1, 2, 3]

def test_latent_cost_uses_last_step_and_alpha(): ...   # pred visual zeros vs goal ones → 1.0; proprio diff 1 → +0.1
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — mirror the repo `CEMPlanner.plan` loop (sample `mean + std·randn`, keep the mean as sample 0, top-k elites by lowest cost, refit mean/std) without CUDA hardcoding or distributed code; return final-iteration elites. **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: device-agnostic CEM and diverse candidate selection`

---

### Task 6: Planner service (/plan) and client

**Files:**
- Create: `pusht_game/planner_client.py` (shared types above + client), `planner_service/backend.py`, `planner_service/service.py`
- Test: `tests/test_planner_service.py`, `tests/test_planner_client.py`

**Interfaces:**
- Consumes: `load_world_model`, `cem_plan`, `select_diverse`, `latent_cost`, `Board`, `png_b64`.
- Produces: `JepaBackend(wm: WorldModel).plan(req: PlanRequest) -> PlanResult`; `FakeBackend(seed=0).plan(req) -> PlanResult` (4 random action sequences with ascending fake costs; every image is a blank 224×224 board with the candidate id drawn on it; `fake=True`); `create_app(backend) -> FastAPI` with `POST /plan` (JSON of `PlanRequest` → `PlanResult`) and `GET /health -> {"ok": true, "fake": bool, "device": str}`; `PlannerClient(base_url: str | None, backend=None, timeout_s=30.0)` with `async plan(req) -> PlanResult` (HTTP when `base_url` set, else calls the in-process backend in a thread). Started via `uv run --extra planner uvicorn planner_service.service:app --port 8001` (env `PLANNER_FAKE=1` selects `FakeBackend`).
- JepaBackend steps: decode PNGs → encode context (`visual` [1,2,3,224,224], `proprio` [1,2,4]) and goal (repeat goal frame/proprio to 1 step) → `cost_fn = lambda a: latent_cost(wm.unroll(z_ctx, act_suffix=a), z_goal)` → `cem_plan(...)` → `select_diverse` → for the 4 picks, one batched `unroll` + `decode_unroll(batch=True)` → `ending` = last frame, `frames` = last 4 frames → `env_actions` = `preprocessor.denormalize_actions(a.reshape(horizon*5, 2))` (normalized per env step; verify against Task 4 output) → `goal_decoded` = decode of `z_goal`.

- [ ] **Step 1: Write failing tests** (FakeBackend only; JEPA path verified by Task 4's script)

```python
def test_plan_endpoint_returns_four_sorted_candidates():
    client = TestClient(create_app(FakeBackend()))
    r = client.post("/plan", json=asdict(sample_request())).json()
    assert len(r["candidates"]) == 4 and r["fake"] is True
    costs = [c["latent_cost"] for c in r["candidates"]]; assert costs == sorted(costs)
    assert all(len(c["env_actions"]) == 30 and len(c["frames_png_b64"]) == 4 for c in r["candidates"])

async def test_client_in_process_matches_http_shape():
    res = await PlannerClient(None, backend=FakeBackend()).plan(sample_request())
    assert isinstance(res, PlanResult) and res.candidates[0].id == "c0"

async def test_client_times_out(): ...   # backend that sleeps 1 s, timeout_s=0.1 → raises PlannerUnavailable
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS. **Step 5:** On the GPU box run the service with the real backend, `curl localhost:8001/health` → `{"ok": true, "fake": false, ...}`.
- [ ] **Step 6: Commit** `feat: planner service with JEPA and fake backends, async client`

---

### Task 7: Jev judge and Jev dry run

**Files:**
- Create: `pusht_game/jev_judge.py`, `scripts/dry_run_jev.py`
- Test: `tests/test_jev_judge.py`

**Interfaces:**
- Produces: `@dataclass Decision(pick: str, probabilities: dict[str, float], confidence: float, vetoed: set[str], source: Literal["jev", "fallback"], latency_ms: float, model: str | None)` — keys are candidate ids (`c0..c3`), never display labels; `JevJudge(api_key: str | None = os.environ["TYPESAFE_API_KEY"], model="jev-1.14.0", timeout_s=4.0, veto_threshold=0.8, transport: httpx.AsyncBaseTransport | None = None, rng: random.Random | None = None, log_path: str | None = "recordings/jev_log.jsonl")`; `async decide(board_png_b64: str, goal_png_b64: str, candidates: list[Candidate]) -> Decision` — raises nothing: on HTTP error / timeout / missing key returns `Decision(source="fallback", pick=<lowest latent_cost id>, probabilities={}, confidence=0.0, ...)`.
- Request: `state = {"board": ref_board, "goal": ref_goal}`; question `best` = Choice with instructions *"Each option is an imagined ending of a push. Which ending puts the gray T block closest to the gray T block's pose in `goal`, matching both position and rotation? Ignore the blue circle and the light-green T fixed at the centre; they are background."* and `criteria = {label: ref}` for shuffled labels A–D; questions `worse_A..D` = Noul *"Is the gray T block in this imagined ending {ref} further from its pose in `goal` than it is in `board`?"*. Pick = highest-probability non-vetoed id (veto: Noul ≥ 0.8); if all vetoed, highest probability.

- [ ] **Step 1: Write failing tests** (`httpx.MockTransport`)

```python
async def test_maps_shuffled_labels_back_to_candidate_ids():
    # rng seeded so labels A..D map to [c2, c0, c3, c1]; mock answers best.choice="A" with probs A=.7,B=.1,C=.1,D=.1
    d = await judge_with(mock_answers(...)).decide(B64, B64, cands())
    assert d.pick == "c2" and d.probabilities["c2"] == pytest.approx(0.7) and d.source == "jev"

async def test_request_has_images_header_pinned_model_and_all_assets_referenced(): ...
    # captured request: header TypeSafe-Danger-Zone-Images == "true", json["model"] == "jev-1.14.0",
    # every key of json["assets"] appears in state/instructions/criteria strings; 6 assets total

async def test_veto_skips_top_choice(): ...           # best=A(.6) but worse_A noul=.9 → pick is next best
async def test_timeout_falls_back_to_lowest_cost():   # transport raises httpx.ReadTimeout
    d = await judge_with(raise_timeout).decide(B64, B64, cands())
    assert d.source == "fallback" and d.pick == "c0"
async def test_missing_key_falls_back_without_network(): ...
async def test_429_falls_back(): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** with `httpx.AsyncClient`, JSON body (base64 assets). Append one JSONL line per call to `log_path` with request summary (no key, no image bytes), response, latency. **Step 4: Run** → PASS.
- [ ] **Step 5: Write `scripts/dry_run_jev.py`**: render 4 synthetic endings with `Board` (block shifted toward / away from a goal by known amounts), call `decide` 10×, print pick, probabilities, confidence and median latency; expected: picks the ending closest to the goal in most runs. Run with `TYPESAFE_API_KEY` set; record results in `AGENTS.md`.
- [ ] **Step 6: Commit** `feat: Jev judge with label shuffling, vetoes and fallback`

---

### Task 8: AI controller

**Files:**
- Create: `pusht_game/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `Board`, `PushCounter`, `overlap`, `in_contact`, `render_goal`/`GoalImage`, `PlannerClient`, `JevJudge`, `Decision`, `PlanRequest`, `PlanResult`.
- Produces: `@dataclass SideResult(pushes: int, reached: bool, best_overlap: float, finished_at_s: float | None)` (also used by Task 9); `AIController(board: Board, planner: PlannerClient, judge: JevJudge, emit: Callable[[dict], Awaitable[None]], tau: float = 0.5, max_escalations: int = 2, act_plan_steps: int = 1, step_delay_s: float = 0.1, goal_reference: Literal["crisp", "decoded"] = "crisp")` — `goal_reference` selects whether Jev gets `goal.png_b64` or the plan's `goal_decoded_png_b64`; `async run(round_id: int, goal: GoalImage, goal_pose: tuple, is_current: Callable[[int], bool]) -> SideResult` — loops until overlap ≥ 0.90, `is_current(round_id)` is false, or it's cancelled. Per decision: emit `ai_status` (`imagining` → `deciding` → `unsure` on escalation → `acting`; `judge_offline` / `planner_offline` on fallbacks), emit `decision` and `imagination` messages (shapes in Task 10), execute the first `act_plan_steps × 5` env actions of the picked candidate one env step at a time with `step_delay_s` sleeps, updating `PushCounter` and checking overlap after each step, then emit `reality` (real frame vs imagined +1).
- Escalation: if `decision.source == "jev"` and `confidence < tau` and escalations < max → re-plan with `num_samples *= 2`, `warm_start_mean = result.mean`, re-decide.
- Planner failure (`PlannerUnavailable`): emit `planner_offline`, sleep 1 s, retry; never crash the round.

- [ ] **Step 1: Write failing tests** (stub planner returning fixed `PlanResult`s whose c1 pushes the block onto the goal; stub judge returning scripted `Decision`s)

```python
async def test_low_confidence_escalates_then_acts():
    judge = ScriptedJudge([dec("c1", conf=0.3), dec("c1", conf=0.8)])
    res = await run_controller(judge=judge)
    assert planner.calls[1].num_samples == 600 and planner.calls[1].warm_start_mean is not None
    assert [m["status"] for m in emitted("ai_status")][:4] == ["imagining", "deciding", "unsure", "deciding"]

async def test_escalation_capped_at_two(): ...         # always conf 0.1 → exactly 3 plan calls before acting
async def test_fallback_decision_emits_judge_offline(): ...
async def test_stops_mid_chunk_on_success():
    # candidate actions reach ≥ 0.90 after env step 2 of 5 → board stepped exactly 2 times, result.reached is True
async def test_stale_plan_dropped_after_new_round():
    # is_current flips to False while planner awaits → no env step taken, run returns promptly
async def test_planner_offline_retries_without_crashing(): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: AI controller with Jev decisions, escalation and fallbacks`

---

### Task 9: Round logic and human control

**Files:**
- Create: `pusht_game/game_round.py`
- Test: `tests/test_game_round.py`

**Interfaces:**
- Consumes: `SideResult` from `pusht_game/controller.py` (Task 8).
- Produces: `decide_winner(human: SideResult, ai: SideResult) -> Literal["human", "ai", "tie"]` (both reached → fewer pushes, equal → tie; one reached → it wins; neither → higher best_overlap, equal within 0.005 → tie); `Round(id: int, challenge: Challenge, round_seconds: float = 90.0)` holding `human_board`, `ai_board`, `human_pushes`, `goal_pose`, `goal_image`; `human_action(mouse: tuple[float, float] | None) -> tuple[float, float]` — relative action `(target − agent_pos) / 100` toward the clamped mouse target when held, `(0, 0)` when not held (env keeps stepping so the block settles); `clamp_mouse(x, y) -> tuple` into [20, 491].

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("h,a,w", [
    (SideResult(12, True, .93, 40), SideResult(9, True, .91, 50), "ai"),
    (SideResult(7, True, .95, 30), SideResult(9, True, .91, 50), "human"),
    (SideResult(9, True, .9, 30), SideResult(9, True, .9, 50), "tie"),
    (SideResult(3, False, .80, None), SideResult(20, True, .9, 80), "ai"),
    (SideResult(3, False, .80, None), SideResult(4, False, .60, None), "human"),
])
def test_decide_winner(h, a, w): assert decide_winner(h, a) == w

def test_mouse_target_clamped():
    assert clamp_mouse(-50, 900) == (20, 491)

def test_human_action_points_at_mouse_and_zero_when_released(): ...
def test_round_starts_both_boards_from_same_state(): ...
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: round state, winner rule, human control mapping`

---

### Task 10: Game server and WebSocket protocol

**Files:**
- Create: `pusht_game/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Round`, `AIController`, `PlannerClient`, `JevJudge`, `PRESETS`, `clamp_goal`, `render_goal`, `overlap`, `in_contact`.
- Produces: `create_game_app(planner: PlannerClient, judge: JevJudge) -> FastAPI`; module-level `app` built from env (`PLANNER_URL` → HTTP client, unset → in-process `FakeBackend`; `TYPESAFE_API_KEY`). Run: `uv run uvicorn pusht_game.server:app --port 8000`. Serves `static/` at `/`, WS at `/ws`, `GET /challenges` → list of presets.
- Server → client messages:
  - `{"type":"snapshot", "round": {...}, "boards": {...}, "last_decision": {...}|null, "fake_planner": bool}` on connect
  - `{"type":"state","board":"human"|"ai","agent":[x,y],"block":[x,y,θ],"overlap":f,"pushes":n}` (human at 10 Hz tick, AI after each env step)
  - `{"type":"round","id":n,"phase":"setup"|"playing"|"finished","challenge":id|null,"goal":[x,y,θ],"winner":str|null,"results":{"human":{...},"ai":{...}}|null}`
  - `{"type":"ai_status","status":str,"escalation":n}`
  - `{"type":"decision","candidates":[{"id","label","ending_png_b64","prob","vetoed","latent_cost"}],"pick":id,"confidence":f,"source":"jev"|"fallback"}`
  - `{"type":"imagination","frames_png_b64":[...4]}`, `{"type":"reality","real_png_b64":s,"imagined_png_b64":s}`
- Client → server: `{"type":"mouse","x":f,"y":f,"down":bool}` (board coordinates 0–512), `{"type":"start","challenge":id}`, `{"type":"start","goal":[x,y,θ],"start_state":[7 floats]}`, `{"type":"set_goal","goal":[x,y,θ]}` (setup phase only; server clamps and re-renders the goal image).
- Starting a round increments the round id, cancels the previous AI task, starts the human 10 Hz tick and the AI task; round finishes when both sides finish or `round_seconds` elapses.

- [ ] **Step 1: Write failing tests** (`TestClient.websocket_connect`, fake planner, judge with no key → fallback)

```python
def test_snapshot_on_connect(): ...                 # first message type == "snapshot", fake_planner is True
def test_start_round_broadcasts_round_and_states(): ...  # after start: "round" phase playing, then "state" for both boards
def test_set_goal_is_clamped(): ...                 # set_goal [500,500,0] → round.goal within walls
def test_new_start_cancels_previous_ai_task(): ...  # two starts → only latest round id appears in later messages
def test_api_key_never_sent_to_client(): ...        # with TYPESAFE_API_KEY="sk-test", no message contains "sk-test"
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS; also `uv run pytest -q` (whole suite) → all PASS.
- [ ] **Step 5: Commit** `feat: game server with WebSocket protocol`

---

### Task 11: Frontend — AI board, decision strip, imagination strip

**Files:**
- Create: `pusht_game/static/index.html`, `pusht_game/static/app.js`, `pusht_game/static/style.css`

**Interfaces:**
- Consumes: the WS protocol from Task 10.
- Produces: layout from the brief — two 512-unit boards side by side (scaled to fit), each with a HUD line `pushes: N   overlap: NN%`; under the AI board the **decision strip** (4 tiles: ending image, label A–D, Jev probability bar, pick highlighted, veto mark, "judge offline" badge when `source=="fallback"`, spinner + "Jev isn't sure… imagining more" when status `unsure`) and the **imagination strip** ("What the AI imagines will happen", frames +1…+4, then real vs imagined). Boards drawn from state as vectors: T in slate gray, pusher in royal blue, goal outline in green (our goal, not the env's fixed one). "FAKE PLANNER" banner when `fake_planner`. Dark, projector-friendly theme.

- [ ] **Step 1: Implement** using the `frontend-design` skill for visual direction.
- [ ] **Step 2: Verify manually** — `PLANNER_FAKE=1 uv run uvicorn pusht_game.server:app --port 8000`, open http://localhost:8000, start an easy preset. Expected: AI board animates pushes; decision strip shows 4 tiles with bars that sum to ~100% (or "judge offline" without a key); imagination strip fills before each push; FAKE PLANNER banner visible; overlap and pushes update.
- [ ] **Step 3: Commit** `feat: frontend AI board with decision and imagination strips`

---

### Task 12: Frontend — human board, challenge picker, draggable goal, results

**Files:**
- Modify: `pusht_game/static/index.html`, `app.js`, `style.css`

**Interfaces:**
- Consumes: WS protocol (Task 10), `GET /challenges`.
- Produces: mouse on the YOU board sends `mouse` messages (board coordinates) at ≤ 30 Hz while the round is playing; challenge picker grouped by tier; in setup phase the goal outline on both boards can be dragged (move) and rotated (scroll wheel / Shift-drag), sending `set_goal`; results panel on `phase=="finished"` showing winner, pushes and overlap for both sides.

- [ ] **Step 1: Implement.**
- [ ] **Step 2: Verify manually** — drag the pusher to push the T on the YOU board; push counter increments once per contact episode; dragging the goal toward a wall stops at the wall; a round ends with the winner banner matching `decide_winner`.
- [ ] **Step 3: Commit** `feat: human board, challenge picker, draggable goal, results`

---

### Task 13: Jev accuracy harness

**Files:**
- Create: `scripts/jev_harness.py`
- Test: `tests/test_jev_harness.py`

**Interfaces:**
- Consumes: `Board`, `overlap`, `PlannerClient`, `JevJudge`, `PRESETS`, `render_goal`.
- Produces: `ground_truth_best(start_state: list[float], goal_pose, candidates: list[Candidate]) -> tuple[str, dict[str, float]]` (restores a fresh `Board` per candidate, executes all `env_actions`, returns the id with the highest final overlap plus all overlaps); CLI `uv run --extra planner python scripts/jev_harness.py --decisions 40 --reference crisp|decoded --out recordings/harness.json` that plays presets with the real planner, records each decision point, and prints top-1 accuracy for Jev, lowest latent cost, and random (0.25), plus Jev accuracy by confidence bucket (<0.3, 0.3–0.6, ≥0.6).

- [ ] **Step 1: Write failing test** — `test_ground_truth_picks_candidate_that_reaches_goal`: of two hand-built candidates, the one whose actions push the T onto the goal is returned.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Run the harness** on the GPU box with both `--reference crisp` and `--reference decoded`; record the table in `AGENTS.md`; set the judge's default goal reference and `JEV_TAU` from the results.
- [ ] **Step 6: Commit** `feat: Jev accuracy harness`

---

### Task 14: Stage fallback — record and replay a round

**Files:**
- Modify: `pusht_game/server.py`
- Test: `tests/test_replay.py`

**Interfaces:**
- Produces: env `RECORD_TO=recordings/round.jsonl` appends every broadcast message with a monotonic timestamp; env `REPLAY_FROM=<file>` makes `/ws` stream the recorded messages with original timing instead of running a live AI (human board still disabled in replay; UI shows a small "replay" badge via `snapshot.replay = true`).

- [ ] **Step 1: Write failing test** — record a short fake-planner round, then replay: the sequence of message `type`s received equals the recorded sequence.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Record** one good round on the GPU box and keep it next to the demo laptop.
- [ ] **Step 6: Commit** `feat: record and replay a round as a stage fallback`

---

### Stretch (separate plans if reached)

Instinct player (Jev-only third board, Choice over 8 push directions) → audience rules (Noul per candidate) → spot the fake → sabotage / leaderboard. Each gets its own short plan once Tasks 1–14 are done.

# Push-Cube: Human vs. Jev — Design

Status: approved direction · Date: 2026-11-10
History: started as Push-T with a JEPA world model; simplified to a translation-only cube with
TypeSafe's Jev as the only AI (no world model, no GPU).

## 1. Intent

A hackathon demo game built around TypeSafe's decision model **Jev**. A human and Jev race side by
side to push a cube onto a square goal outline on a top-down table, using a round pusher that can
push but never grab. The cube slides in x/y only (no rotation). Winner: fewer pushes to reach the goal.

The AI mirrors Kahneman's System 1 / System 2 split that Jev is named after:
- **Instinct (System 1)**: Jev reads the raw coordinates (pusher, cube, goal) and decides the next
  push — a direction and a distance — with calibrated probabilities.
- **Thinking harder (System 2)**: when Jev isn't confident, the AI simulates its top 3 options in a
  copy of the physics, shows those futures on screen, and asks Jev again with the outcomes.

Pitch line for judges:
> "Jev is a decision model, not a chatbot. It plays from raw coordinates by instinct, and when it
> isn't sure, it simulates its options before it commits — just like you would."

## 2. Goals, non-goals, anti-goals

**Goals**
- Two boards side by side (YOU vs JEV), same start, same goal; Three.js 3D view of a 2D physics sim.
- Cube 60 px, translation only; round pusher r = 15 px; table 512×512 px with walls.
- Score: overlap ≥ 90% between cube and goal square; winner = fewer pushes.
- AI turn: Jev decides direction (8 compass directions) + distance (short / medium / long) from JSON
  state → if confidence < τ, simulate top 3 options and re-ask Jev with outcomes → execute one push.
- **Decision panel**: compass rose with Jev's probability per direction, distance bars, confidence,
  "Jev isn't sure… simulating options" state, and the 3 simulated futures with Jev's final pick.
- Visible fallback ("judge offline") using a simple geometric rule when Jev is unavailable.
- Challenge picker (presets) plus a draggable goal.
- **Measured play quality**: offline harness comparing Jev's pushes-to-goal with a geometric rule
  and a one-step "oracle" that simulates every option.

**Stretch (in priority order)**
1. Audience rules: crowd types a rule ("never touch the walls"); a Jev Noul per option filters pushes.
2. Sabotage: nudge the AI's cube mid-round; it re-decides from the new coordinates.
3. Obstacles or two cubes, to make the puzzle harder for both sides.
4. Leaderboard.

**Non-goals**
- Rotation, the T shape, any learned world model, PyTorch, GPUs.
- Sending images to Jev (state is plain JSON; Jev's image support is beta and not needed).
- 3D physics — physics is 2D; 3D is only the view.
- Using Jev as the referee: overlap is computed exactly.

**Anti-goals**
- Faking decisions: every pick shown must be Jev's real answer or clearly labelled fallback.
- Simulated futures presented as anything other than "simulated with the same physics".
- The demo stalls when Jev is slow or down (every call has a timeout and a visible fallback).
- Leaking `TYPESAFE_API_KEY` to the browser, logs, or git.
- A 3D view that disagrees with the sim (sizes and positions come from the server).

## 3. Constraints and facts

**Jev (TypeSafe)**
- `POST https://api.typesafe.ai/v1/systemone`, header `Authorization: Bearer $TYPESAFE_API_KEY`;
  pin model `jev-1.14.0`. Early access; key provided by the user in a gitignored `.env`.
- Input `state` (string / JSON) + typed questions; **Choice** returns `choice`, per-option
  `probabilities`, `confidence`. Questions in one request are evaluated in parallel.
- Priced per input token ($0.042 per Mtok); rate limits per org (429 → fallback).
- Unknown: latency per request — measured in the Jev dry run.

**Physics**: pymunk 2D, 100 Hz substeps, 10 Hz control; pusher is a kinematic body PD-controlled
toward a target (k_p 100, k_v 20, as in Push-T); cube is a dynamic body with infinite moment of
inertia (cannot rotate) and linear damping so it stops shortly after contact ends.

**Hardware**: Apple M1 Pro Mac; everything runs locally except the Jev API call.

## 4. Approaches considered

**How the AI decides**

| | A. Jev only | **B. Jev + think harder (chosen)** | C. Learned world model (LeWM / JEPA) |
|---|---|---|---|
| Per push | 1 Jev call | 1 call, 2 when unsure | CEM planning + decoder |
| Runs on the Mac | yes | yes | slow / needs work |
| On-screen "thinking" | probabilities only | probabilities + simulated futures | imagined frames |
| Status | fine but less to show | picked | dropped by user (complexity, Mac speed) |

**How the app is built**: Python game server (FastAPI + WebSockets) owns both sims, scoring and
the AI; browser renders with Three.js. pygame is the fallback if the web layer becomes a time sink.

## 5. Design

### 5.1 Components

```
 Browser (Three.js)              Game server (Python, FastAPI)
 ┌───────────────────┐   WS      ┌──────────────────────────────┐   HTTPS
 │ YOU board (3D)    │◀────────▶ │ CubeSim #1 (human)           │
 │ JEV board (3D)    │ state,    │ CubeSim #2 (AI)              │
 │ decision panel    │ decisions,│ push executor, scoring       │
 │ picker, results   │ mouse     │ AI controller ── Jev judge ──┼──────▶ api.typesafe.ai
 └───────────────────┘           └──────────────────────────────┘
```

### 5.2 Simulation and scoring
- Table 512×512 px, walls at 5 and 506. Cube side 60 px; goal centre clamped to [35, 476].
- Overlap = (60 − |dx|)₊ · (60 − |dy|)₊ / 3600, where (dx, dy) = cube centre − goal centre.
  90% ≈ within ~6 px on one axis.
- Push (both boards) = one continuous pusher–cube contact episode.

### 5.3 AI push execution
- Options: 8 directions (N, NE, E, SE, S, SW, W, NW; y points down, N = −y) × distances short 10 px,
  medium 30 px, long 80 px of pusher travel = 24 options.
- Executor: contact point = cube centre − d·(30 + 15 + 2); approach point 20 px further back; route
  around the cube without touching it (via corners of the cube's inflated box when needed); push
  along d by the chosen distance; retreat 10 px to end contact. One executed option = exactly one
  contact episode = one push.
- An option is **infeasible** when its approach point would be outside the walls (cube against a wall).
  Infeasible options are listed in Jev's state and skipped if picked.

### 5.4 Jev decision (System 1)
- `state` (JSON): coordinate convention (x right, y down), board size, walls, cube {x, y, size},
  pusher {x, y}, goal {x, y}, offset {dx, dy} = goal − cube, overlap, pushes so far, infeasible options.
- Questions (one request): `direction` — Choice over 8 directions, each described in words ("move the
  cube toward the top of the board (−y)"); `distance` — Choice short / medium / long with px values.
- Confidence = min(direction.confidence, distance.confidence).

### 5.5 Thinking harder (System 2)
- Trigger: confidence < τ (start 0.5, env `JEV_TAU`), at most once per push.
- Take the 3 most probable options by p(direction)·p(distance), skipping infeasible ones. For each,
  copy the AI sim and run the executor headless → outcome: cube after, offset after, overlap after,
  touched wall.
- Second Jev request: `state` = current state + the 3 outcomes; question `pick` — Choice over the 3
  option ids, each described with its outcome numbers. Execute the pick.
- UI shows the 3 simulated futures (top-down mini boards with the cube's resulting position) with
  Jev's final probabilities; the executed push then plays out on the 3D board.

### 5.6 Fallback and baselines
- **Geometric rule** (fallback and baseline): direction closest to the goal offset; distance closest
  to the remaining distance minus expected slide. Used when Jev errors, times out (4 s), or the key
  is missing; UI shows "judge offline".
- **Oracle-greedy** (baseline only): simulate all feasible options, pick the max overlap after.

### 5.7 Round flow
1. Pick a preset or drag the goal; both sims reset to the same start.
2. Human drags the pusher (mouse → raycast onto the table → sim target), 10 Hz physics, client
   interpolation.
3. AI loop: Jev decides → maybe thinks harder → executes, animated in real time; repeats until
   ≥ 90% or timeout.
4. Winner: both reached → fewer pushes (equal → tie); one reached → it wins; neither → higher overlap.

### 5.8 3D view
- Three.js (pinned, CDN import map, no build step). Mapping sim (x, y) → world (x − 256, 0, y − 256).
- Cube = `BoxGeometry(60, 60, 60)`, pusher = cylinder r 15, goal = glowing square outline on the
  table, walls, soft shadows. Tilted camera on both boards; `V` toggles both to top-down together, so the two boards always match.
- Sizes come from the server's snapshot, never hardcoded in JS.

### 5.9 Challenges
- Presets: **easy** (goal 80–120 px away along one axis), **medium** (diagonal, 150–250 px),
  **hard** (goal against a wall or in a corner, so the approach side matters), **far** (across the
  board).

## 6. Milestones
1. Cube sim, scoring, push counter, push executor (headless, tested).
2. Jev judge + dry run with the real key (latency, sensible picks).
3. AI controller with think-harder and fallback.
4. Game server + WebSocket protocol.
5. Frontend: 3D JEV board + decision panel — complete "watch Jev decide" demo.
6. Human board, picker, draggable goal, results.
7. Harness (Jev vs geometric vs oracle), record/replay stage fallback.
8. Stretch goals.

## 7. Left to the builder
- Damping value, push speed, animation timing; exact Jev wording within §5.4–5.5.
- Visual styling (dark, projector-friendly); compass and strip design.
- Round timeout; preset coordinates.

## 8. Open questions
1. Push definition: one contact episode for both boards (current default) — OK?

## 9. v2 — more shapes, and they spin (2026-11-10)

Requested by the user: more shapes than the cube, the player only ever pushes, shapes rotate, and each
challenge comes with its own shape.

- **Shapes:** square, T, L, I-bar, Z, triangle (`pushcube/shapes.py`, convex parts centred on the
  centroid). Blocks are real pymunk bodies with mass and moment of inertia, plus table friction on both
  sliding and spinning.
- **Goal:** position *and* angle. Overlap = area(block ∩ goal outline) / block area (shapely), using the
  shape's symmetry (square 90°, I and Z 180°, triangle 120°). Still ≥ 90% to win.
- **Jev's choice:** 8 directions × 3 distances × **aim** (left of centre turns the block clockwise,
  centre slides it, right turns it counterclockwise) = 72 options. State adds the shape, block and goal
  angles and `turn_needed_deg`.
- **Push execution:** the pusher travels along the push line offset by the aim and touches the block where
  that line meets the block outline grown by the pusher radius; the approach routes around the real
  rotated outline.
- **Push counting:** a push ends only after the pusher has stayed clearly away (> 8 px) for 0.3 s, for
  both the human and Jev, so a spinning block that swings back into the pusher stays one push.
- **Fallback when Jev is unavailable:** the one-step simulated best move (oracle). The cheap geometric
  rule remains a harness baseline only (it can't solve the T quarter turn).
- **UI:** blocks are extruded from the server-sent outline and rotated; the goal outline rotates; scroll
  over the goal in setup to rotate it; the decision panel shows the aim bars.

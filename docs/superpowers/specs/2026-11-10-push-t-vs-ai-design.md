# Push-T: Human vs. World Model — Design

Status: draft for review · Date: 2026-11-10

## 1. Intent

A hackathon demo game. A human and a pretrained JEPA world model (`jepa_wm_pusht` from
`facebookresearch/jepa-wms`) race side by side to push a T-shaped block onto a green T outline
(right position and rotation) using a round pusher that can push but never grab.

The point is the story, not the game: the AI was never trained on this puzzle and gets no reward.
It learned how objects move from watching pushes, and solves new goals by *imagining* the outcome
of candidate pushes before acting. The "imagination strip" makes that visible: the audience sees
the AI's predicted future decoded into pictures, then sees whether reality matched.

Pitch line for judges:
> "The AI was never trained to solve this puzzle and gets no reward. It learned how objects move
> from watching pushes, and now it solves new goals by imagining the outcome of its moves before
> making them."

## 2. Goals, non-goals, anti-goals

**Goals**
- Two boards side by side, same start pose, same goal outline: YOU (mouse) vs AI (planner).
- AI plans from the current board image and a goal image, executes one push, replans (MPC loop).
- Imagination strip: 3–4 decoded predicted frames of the AI's chosen plan, shown before it acts.
- Live push counter and overlap % per board; round ends when someone reaches ≥ 90% overlap.
- Winner = fewer pushes to reach 90% (not speed).
- Challenge picker: preset challenges plus a draggable/rotatable goal outline.

**Stretch (only if time)**
- Sabotage mode: nudge the AI's T mid-plan; it replans and recovers.
- Spot the fake: show imagined vs real next frame; crowd guesses which is real.
- Leaderboard: best human push counts per challenge vs the AI's.

**Non-goals**
- Training or fine-tuning any model. We use the published checkpoint and decoder as-is.
- 3D graphics. Push-T is a top-down 2D task; a 3D look adds work and risk and nothing to the AI.
- Beating the AI on speed, or making the AI fast.

**Anti-goals (would make the demo fail even if it "works")**
- The AI board is fed images that differ from what the model was trained on (a re-implemented
  renderer, different colours/sizes/resolution). The model would be out of distribution and look dumb.
- Faking the imagination strip (e.g. showing the simulator's future instead of the model's decode).
- A scoring rule the AI can't win under or that's obviously unfair to the human.
- A demo that only works on the one GPU box and dies if the network hiccups, with no fallback.

## 3. Constraints and facts (from recon)

- **Model**: `jepa_wm_pusht` — DINOv2 ViT-S/14 encoder, predictor depth 6, 224×224 input,
  ~212 MB checkpoint. Load via
  `torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')`.
- **No hosted API.** Open weights only (license CC-BY-NC 4.0, fine for a non-commercial hackathon
  demo). We run it ourselves with PyTorch and wrap it in our own `/plan` endpoint (§5.1).
  `torch.hub.load` clones the repo, pulls weights from HF, and uses the repo's eval config
  `pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml` (CEM, horizon 6, 2 context
  frames, decode on). Without CUDA it falls back to CPU (MPS would need a manual `.to("mps")`).
- **Decoder** (needed for the strip): `dinov2_vits_224_INet` (the decoder assigned to JEPA-WM sim
  envs). Trained cross-environment on VideoMix2M, so Push-T decodes may be blurry — verify in dry run.
- **Planner**: repo ships CEM and NeverGrad planners (`evals/simu_env_planning/planning/planner.py`)
  minimizing L2 latent distance between predicted states and the encoded goal image. Model wrapper:
  `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.EncPredWM`.
- **Simulator**: the Push-T env used by DINO-WM / jepa-wms (pymunk physics, adapted from Diffusion
  Policy). The AI board **must** use this exact env and renderer so model inputs are in distribution.
  This rules out a Three.js physics re-implementation for the AI side.
- **Hardware**: dev machine is a Mac (Apple Silicon, no NVIDIA GPU). ViT-S may run on MPS/CPU but
  CEM with hundreds of samples will likely be several seconds per push; a CUDA GPU should give the
  ~1–3 s/push the brief expects. Where the model runs on demo day is open (see §8).
- **Goal image**: the model compares against a full rendered frame, so the goal image contains the
  pusher too. We render it by placing the T at the goal pose with the pusher parked at a fixed,
  out-of-the-way spot (to be tuned in dry run so the AI doesn't waste pushes chasing the pusher's
  goal position).
- **Inputs beyond images**: the Push-T world model may also take proprioception (pusher position /
  velocity) alongside images. The brief says "the AI sees only two images"; if proprio is used,
  narration should say "the board image, its own fingertip position, and a picture of the goal."
  Verify in dry run.
- **Known risk — rotation and distance**: published Push-T planning evals use goals sampled from the
  dataset a short horizon ahead of the start. Large rotations / far goals are untested territory.

## 4. Approaches considered

| | A. Python backend + browser canvas (**recommended**) | B. All-Python pygame | C. Three.js sim in browser + Python planner |
|---|---|---|---|
| Sim | Real Push-T env (Python) for both boards | Real Push-T env | Three.js/JS physics for display, Python env for AI |
| UI | HTML canvas, two boards + strip, styled freely | pygame window | Three.js |
| Remote GPU | Natural: planner is a separate service | Awkward (one process) | Natural |
| Fairness | Both boards share one physics engine | Same | Two physics engines → human and AI play different games |
| Effort | Medium | Lowest | Highest |
| Polish for judges | High | Low–medium | High but risky |

**Pick: A.** One physics engine for both boards (fair, and in distribution for the model), a
browser UI that's easy to make look good on a projector, and the planner can live on a GPU box
without changing the game. B is the fallback if the web layer becomes a time sink. C is rejected:
it's the most work and makes the two boards physically different.

*Pending your confirmation — this is my recommendation, not yet your pick.*

## 5. Design

### 5.1 Components

```
 Browser (laptop / projector)                Game server (Python, local)            Planner service (GPU or local)
 ┌──────────────────────────┐   WebSocket    ┌──────────────────────────┐   HTTP     ┌──────────────────────────┐
 │ canvas: YOU board        │ ◀────────────▶ │ PushT env #1 (human)     │            │ jepa_wm_pusht + decoder  │
 │ canvas: AI board         │  state @ 30Hz  │ PushT env #2 (AI)        │ ─────────▶ │ CEM planner              │
 │ imagination strip        │  mouse events  │ round / scoring / pushes │ obs, goal  │ returns: actions,        │
 │ challenge picker, HUD    │                │ goal image rendering     │ ◀───────── │ imagined frames, cost    │
 └──────────────────────────┘                └──────────────────────────┘            └──────────────────────────┘
```

- **Game server** (FastAPI + WebSockets): owns both env instances, round state, push counting,
  overlap scoring, challenge definitions, and renders 224×224 observation/goal images for the planner.
- **Planner service**: stateless `POST /plan {obs_image, goal_image, proprio?} → {actions,
  imagined_frames[3–4], cost}`. Runs in-process when co-located, or on a remote GPU box.
- **Frontend**: single HTML page, vanilla JS + canvas. Boards are drawn from env state (T pose,
  pusher position) as clean vector graphics for the audience; the model never sees these drawings,
  it sees the env's own renderer output.

### 5.2 Round flow (~60 s)

1. Pick a challenge (preset list, or drag/rotate the goal outline). Both envs reset to the same T start pose.
2. Server renders the goal image once and sends it to the planner with each request.
3. AI loop (on its own thread/task): render obs → `/plan` → show imagined frames in strip ("thinking…")
   → execute first action chunk → update push count/overlap → repeat until ≥ 90% or round timeout.
4. Human loop: mouse position streamed to server; pusher follows the mouse (velocity-limited as in
   the original env); server steps physics and broadcasts state.
5. Round ends for a side at ≥ 90% overlap. Winner = fewer pushes. Ties / neither reaches 90% →
   higher overlap wins.

### 5.3 Scoring

- **Overlap**: area of intersection of the T polygon and goal T polygon ÷ T area (shapely), the same
  coverage idea the original Push-T uses (it uses 95% for success; we use 90%).
- **Push definition — proposed**: a push = one continuous contact episode between pusher and T
  (starts on contact, ends on separation), applied identically to both boards by the same server
  code. This keeps the human's free mouse drag natural and treats both sides the same.
  Also displayed: AI planning steps, as a secondary stat. See §8 Q2.

### 5.4 Imagination strip

- After each plan, decode the predicted latents for the chosen action sequence at steps +1…+4 with
  `dinov2_vits_224_INet`, return as small PNGs, display left-to-right with a "What the AI imagines
  will happen" header.
- After the push executes, flash the real frame next to imagined frame +1 (also the basis for the
  "Spot the fake" stretch mode).

### 5.5 Challenges

- Presets tagged **easy** (small shift + small rotation), **medium**, **hard** (large rotation).
- If the dry run shows the AI fails large rotations: AI rounds use easy/medium presets; hard ones
  become "human challenge" rounds. Second fallback (needs honest narration): give the AI a short
  chain of intermediate goal images (interpolated T poses rendered by the sim) as waypoints.

## 6. Milestones (in order of risk)

0. **Dry run (before hackathon, on a GPU)**: load model + decoder, run one CEM planning episode in
   the Push-T env from the repo; measure seconds/push; test easy vs large-rotation goals; check
   decoder quality; confirm whether proprio is an input.
1. Game server with two env instances, overlap scoring, push counting; headless test.
2. Planner service wrapping the repo's planner behind `/plan`, returning actions + decoded frames.
3. Frontend: AI board + imagination strip only ("watch the AI think") — **this alone is a complete demo**.
4. Human board with mouse control, push counter, winner logic.
5. Challenge picker and draggable goal.
6. Stretch: sabotage, spot the fake, leaderboard.

## 7. Left to the builder

- CEM hyperparameters (samples, iterations, horizon) tuned for the ~1–3 s/push budget.
- Exact action chunk executed per AI step (first action vs first k actions of the plan).
- Visual styling of boards and strip (dark theme suited to a projector), fonts, animations.
- Round timeout length, tie-break details, preset challenge poses.
- Caching: precompute imagined frames for a scripted "safe" round as an offline fallback for the demo.

## 8. Open questions (yours to answer)

1. **Where does the model run on demo day** — this Mac, or a separate GPU machine (cloud or
   teammate's)? Approach A supports both, but it decides whether milestone 0 happens on this Mac
   and whether we need the remote-planner path at all.
2. **What counts as a push?** Proposed: one continuous contact episode with the T, same rule on both
   boards. Alternative: a push = one straight-line stroke (human: press-drag-release; AI: one executed
   action), which matches the AI's action space more literally but makes the human controls feel
   less free.
3. **Approach**: confirm A (Python backend + browser canvas), or prefer B (pure pygame) for speed of build.

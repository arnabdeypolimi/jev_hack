# Push-T: Human vs. Imagination + Judgment — Design

Status: approved direction (Jev integration chosen) · Date: 2026-11-10

## 1. Intent

A hackathon demo game built around TypeSafe's decision model **Jev**. A human and an AI race side
by side to push a T-shaped block onto a green T outline (right position and rotation) using a round
pusher that can push but never grab.

The AI is two models working together, mirroring Kahneman's System 1 / System 2 split that Jev is
named after:

- **Imagination (System 2, slow)**: a pretrained JEPA world model (`jepa_wm_pusht` from
  `facebookresearch/jepa-wms`) imagines many push sequences in its latent space and proposes its
  best few candidate futures, decoded back into pictures.
- **Judgment (System 1, fast)**: **Jev** looks at the candidate futures and the goal picture and
  decides which push to make, returning calibrated probabilities. When Jev isn't confident, the AI
  "thinks harder" (imagines more) and asks again.

The audience sees both: the candidate futures, Jev's probability bars over them, the pick, and then
whether reality matched.

Pitch line for judges:
> "Neither model was trained to solve this puzzle, and neither gets a reward. The world model learned
> how objects move from watching pushes and imagines what each move would do. Jev, a decision model,
> judges those imagined futures and picks one — and when it isn't sure, the AI imagines more before
> it acts."

## 2. Goals, non-goals, anti-goals

**Goals**
- Two boards side by side, same start pose, same goal outline: YOU (mouse) vs AI.
- AI loop per push: imagine (JEPA, CEM) → propose K=4 candidate futures → decide (Jev Choice) →
  escalate if Jev's confidence is low → act → replan.
- **Decision strip** under the AI board: the 4 candidate imagined endings, Jev's probability bar
  under each, the pick highlighted, plus a "Jev isn't sure… imagining more" state on escalation.
- Imagination strip for the chosen plan: decoded frames +1…+4, then real-vs-imagined after acting.
- **3D view** of both boards (Three.js) driven by the 2D sim's state: extruded T, cylinder pusher,
  glowing goal outline on a table; tilted camera with a top-down toggle. The strips stay flat 2D
  model pictures beside it ("you see a 3D table; the AI sees this flat picture").
- Live push counter and overlap % per board; winner = fewer pushes to reach ≥ 90% overlap.
- Challenge picker: preset challenges plus a draggable/rotatable goal outline.
- **Measured Jev accuracy**: an offline harness that reports how often Jev picks the truly best
  candidate vs the world model's own ranking (a real number for the judges).

**Stretch (only if time), in priority order**
1. **Instinct player** (Jev only, no world model): third board where Jev picks one of 8 push
   directions from the board + goal images. Race becomes Human vs Instinct vs Imagination+Judgment.
2. **Audience rules**: crowd types a natural-language rule ("never touch the top edge"); one Jev
   Noul per candidate rejects plans that break it.
3. **Spot the fake**: Jev and the crowd both guess which of two frames is real vs imagined.
4. Sabotage mode (nudge the AI's T mid-plan; it replans) and a leaderboard.

**Non-goals**
- Training or fine-tuning any model (JEPA or Jev; Jev is not customer-tunable anyway).
- 3D physics. Physics stays the original 2D pymunk sim (the world model only knows its motion and
  renders). 3D is a **view** for the audience only (§5.1 Frontend).
- Making the AI fast. Score is pushes, not time.
- Using Jev as the referee: the simulator measures overlap exactly.

**Anti-goals (the demo fails even if it "works")**
- Feeding the world model images that differ from its training renders (re-implemented renderer,
  other colours/sizes). It goes out of distribution and looks dumb.
- Faking any strip: showing simulator futures as "imagined", or a decision Jev didn't make.
- Jev being decorative: if its pick is always the world model's top-ranked candidate anyway, the
  "judgment" story is hollow. The accuracy harness must show what Jev adds (or we reframe honestly).
- The demo stalls if Jev or the GPU box is slow or down. Every remote call needs a timeout and a
  visible fallback.
- Leaking the TypeSafe API key to the browser or the repo.
- A 3D view that disagrees with the sim (different T shape, offsets, mirrored axes). The T outline is
  sent by the server from the sim's own shapes, never re-typed in JS.

## 3. Constraints and facts (from recon)

**World model (JEPA)**
- `jepa_wm_pusht`: DINOv2 ViT-S/14 encoder, predictor depth 6, 224×224 input, ~212 MB.
  `torch.hub.load('facebookresearch/jepa-wms', 'jepa_wm_pusht')`.
- No hosted API: open weights (CC-BY-NC 4.0, fine for a non-commercial demo). We run it with PyTorch
  behind our own `/plan` endpoint. The hub entry uses the repo config
  `pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml` (CEM, horizon 6, 2 context
  frames, decode on). Without CUDA it falls back to CPU; MPS needs a manual `.to("mps")`.
- Decoder for pictures: `dinov2_vits_224_INet` (trained cross-environment on VideoMix2M; Push-T
  decodes may be blurry — dry-run item).
- Planner: repo ships CEM and NeverGrad (`evals/simu_env_planning/planning/planner.py`), L2 latent
  cost to the encoded goal. Wrapper: `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.EncPredWM`.
- Simulator: the Push-T env used by DINO-WM / jepa-wms (pymunk, adapted from Diffusion Policy). The
  AI board must use this env and renderer.
- **Confirmed from the repo config**: the world model takes proprioception (pusher position and
  velocity, `proprio_dim: 4`) and a 2-frame context (frameskip 5). Narration: "it sees the board, its
  own fingertip position, and a picture of the goal."
- **Fixed background target**: the env always draws a light-green T at `goal_pose = [256, 256, π/4]`
  (training frames have it too). Model-facing renders keep it unchanged; our goal outline is drawn
  only in the audience 3D view, and overlap is computed against our goal pose.
- The repo's `CEMPlanner` hardcodes CUDA, so we implement a small device-agnostic CEM on top of the
  model's `encode` / `unroll` / `decode_unroll`, using the same cost (L2 at the final step, proprio
  weight alpha 0.1) and defaults (horizon 6, 300 samples, 30 iterations, 10 elites).
- Goal image is a full rendered frame, so it includes the pusher; we park the pusher in the board
  corner farthest from the goal.
- Known risk: the repo's Push-T eval uses dataset goals only `goal_H: 6` planning steps (30 env
  steps) ahead; large rotations / far goals are untested.

**Decision model (Jev, TypeSafe)**
- Hosted API: `POST https://api.typesafe.ai/v1/systemone`; Python SDK `typesafe_sdk`
  (`TypeSafeClient`, `Choice`, `Noul`, `Score`, `Image`). Auth via `TYPESAFE_API_KEY`. Early access.
- Input: a `state` (text/JSON/images) plus typed questions; returns typed answers with
  probabilities and confidence. No text generation, no prediction of futures — it can't replace the
  world model.
- Question types used: **Choice** (pick one candidate; up to 255 options; returns `choice`,
  per-option `probabilities`, `confidence`), **Noul** (yes/no probability, for rules), **Score**
  (optional per-candidate rating).
- Images: PNG/JPEG anywhere in the request — in `state`, in `instructions`, or as a Choice option's
  description (`criteria`). Needs header `TypeSafe-Danger-Zone-Images: true`. Images are labelled
  "Dreamer-ready" (beta): accuracy and format may change.
- Questions in one request are evaluated in parallel; adding questions barely changes latency.
- Price is per input token ($0.042 per Mtok); output free. Rate limits per org; SDK retries 429s.
- Pin `jev-1.14.0` (the `jev-latest` alias can move and shift probabilities).
- Unknowns: latency for image requests; accuracy on blurry decoded frames.

**Hardware**: dev machine is an Apple Silicon Mac, no NVIDIA GPU. A CUDA GPU should give ~1–3 s per
CEM plan; Jev adds one network round-trip per push.

## 4. Approaches considered

### 4.1 How Jev joins the AI

| | A. Jev picks among top-K candidates (**chosen**) | B. Jev scores every CEM sample | C. Jev alone ("instinct") |
|---|---|---|---|
| What Jev sees | 4 decoded candidate endings + goal + board | hundreds of decoded frames per push | board + goal only |
| Calls per push | 1 (2–3 with escalation) | dozens–hundreds | 1 |
| Latency / cost | low | prohibitive | lowest |
| Uses world model | yes, as proposer | yes | no |
| Story | "imagination proposes, judgment decides" | same, but too slow to demo | System 1 alone — likely plays badly |

**Chosen: A**, plus confidence-based escalation. C becomes stretch goal #1 as a contrasting third
player. B is rejected (latency and cost).

### 4.2 How the app is built

| | Python sim + browser 3D view (**chosen**) | All-Python pygame | 3D physics engine (Three.js/Rapier, MuJoCo) |
|---|---|---|---|
| Fairness | one physics engine, both boards | same | different physics from training |
| World model | in distribution | in distribution | out of distribution — would need retraining |
| Remote GPU / Jev | planner & Jev are services; key stays server-side | awkward | natural |
| Polish | high (3D view) | low–medium | high but breaks the AI |
| Effort | medium | lowest | highest |

pygame is the fallback if the web layer becomes a time sink. A 3D physics engine is rejected; Three.js
is used only to draw the 2D sim's state in 3D.

## 5. Design

### 5.1 Components

```
 Browser                        Game server (Python, local)                 Planner service (GPU/local)
 ┌──────────────────┐  WS       ┌─────────────────────────────┐   HTTP      ┌───────────────────────────┐
 │ YOU board        │◀────────▶ │ PushT env #1 (human)        │ ──────────▶ │ jepa_wm_pusht + decoder   │
 │ AI board         │ state,    │ PushT env #2 (AI)           │ /plan       │ CEM → top-K candidates    │
 │ decision strip   │ strips,   │ round / scoring / pushes    │ ◀────────── │ decoded endings + frames  │
 │ imagination strip│ mouse     │ AI controller (loop below)  │             └───────────────────────────┘
 │ picker, HUD      │           │ Jev judge  ───────────────────── HTTPS ──▶ api.typesafe.ai (Jev)
 └──────────────────┘           └─────────────────────────────┘
```

- **Game server** (FastAPI + WebSockets): both envs, round state, push counting, overlap scoring,
  challenges, goal/obs rendering, the **AI controller**, and the **Jev judge** module. Holds the
  TypeSafe key (env var); the browser never talks to TypeSafe.
- **Planner service**: `POST /plan {obs_images[ctx], goal_image, proprio?, n_samples, iters,
  warm_start?} → {candidates: [{id, actions, latent_cost, ending_png, frames_png[4]}] (K=4),
  goal_decoded_png}`. In-process when co-located; remote on a GPU box otherwise.
- **Jev judge**: builds and sends the decision request, parses `choice / probabilities / confidence`,
  enforces timeout, logs every request/response (model ID, latency, tokens) for the harness.
- **Frontend**: one HTML page, vanilla JS + Three.js (ES module from a CDN, pinned, no build step).
  Each board is a 3D scene built from env state; the models see only the env's own 2D renders.
  - Mapping: sim `(x, y)` px → Three.js `(x − 256, 0, y − 256)`; block angle `θ` → `rotation.y = −θ`
    (sim y points down, so this keeps the picture unmirrored).
  - Geometry: T = `ExtrudeGeometry` of the sim's T polygons (sent by the server on connect), pusher =
    cylinder r = 15, walls at 5 and 506 px, goal = flat emissive green T outline.
  - Motion: state arrives at 10 Hz; the client interpolates between the last two states.
  - Input: mouse → raycast onto the table plane → board coords → same `mouse` message as before.
  - Camera: tilted perspective for the audience; key `V` toggles a top-down orthographic view for
    precise human play.

### 5.2 AI controller loop (one push)

1. Render current obs (last 2 frames for context) and the goal image (rendered once per round).
2. **Imagine**: `/plan` with base budget → K=4 candidates. Candidates are the top elites from CEM's
   final iteration, filtered to be **diverse** (distinct first actions beyond a minimum distance) so
   Jev has a real choice.
3. **Decide**: one Jev request (§5.3) → `choice`, `probabilities`, `confidence`.
4. **Escalate** if `confidence < τ` (start τ = 0.5, tune in dry run): `/plan` again warm-started from
   the current elites with a larger budget (e.g. 2× samples), re-ask Jev. Max 2 escalations, then act
   on the best available pick.
5. **Act**: execute the chosen candidate's first action chunk in env #2. Update overlap / pushes.
   Push real-vs-imagined frame to the strip.
6. **Fallback**: if Jev errors or exceeds its timeout (start 4 s), pick the lowest `latent_cost`
   candidate and show "judge offline" in the UI. If the planner is down, the AI pauses with a
   visible status; a pre-recorded safe round is available for the stage.

### 5.3 Jev decision request

- `model`: `jev-1.14.0`; header `TypeSafe-Danger-Zone-Images: true`.
- `state`: `{"board": <current obs PNG>, "goal": <goal PNG>}`.
- Questions (one request, evaluated in parallel):
  - `best` — **Choice**: "Which imagined ending puts the T block closest to the T shape in `goal`,
    matching both position and rotation? Ignore the round pusher." `criteria`:
    `{"A": <ending A PNG>, "B": ..., "C": ..., "D": ...}`.
  - `worse_A..D` (speculative, optional) — **Noul**: "Does this ending move the T further from
    `goal` than `board` does?" Lets us veto clearly bad candidates and gives the UI a red mark.
- **Goal reference style**: decoded candidate endings are blurry while the goal render is crisp. The
  planner also returns `goal_decoded_png` (goal → encoder → decoder) so Jev can compare like with
  like. Which reference works better is decided by the harness (§5.6).
- Option order is shuffled per request (and mapped back) to avoid position bias.

### 5.4 Round flow (~60 s)

1. Pick a challenge (preset, or drag/rotate the goal). Both envs reset to the same T start pose.
2. Human: mouse streamed over WS; pusher follows (velocity-limited as in the original env).
3. AI: controller loop (§5.2) in its own task; status messages "imagining…", "Jev deciding…",
   "Jev isn't sure… imagining more" drive the strips.
4. A side finishes at ≥ 90% overlap. Winner = fewer pushes. Neither reaches 90% by timeout →
   higher overlap wins.

### 5.5 Scoring

- **Overlap**: area(T ∩ goal T) ÷ area(T), computed with shapely from env state (the original Push-T
  uses 95% coverage; we use 90%).
- **Push** (my proposed default): one continuous contact episode between pusher and T, same server
  code for both boards. AI planning steps and Jev escalations shown as secondary stats.

### 5.6 Jev accuracy harness (offline, before the demo)

- For N recorded decision points: clone the AI env, execute each of the K candidates in the real
  simulator, measure true overlap after the push → ground-truth best candidate.
- Report top-1 accuracy of: Jev pick, lowest latent cost (world model alone), random (25%).
  Also accuracy split by Jev confidence bucket (does low confidence really mean "unsure"?), and
  crisp-goal vs decoded-goal reference.
- Outcome drives the demo framing: if Jev beats latent cost, say so with the number; if not, keep
  Jev as the decision layer but tune prompt/reference/K before the event, and narrate honestly.

### 5.7 Strips (UI)

- **Decision strip**: 4 candidate endings (A–D) with Jev probability bars, pick highlighted, Noul
  vetoes marked; escalation shows a spinner and replaces candidates when the new plan arrives.
- **Imagination strip**: chosen plan decoded at +1…+4, header "What the AI imagines will happen";
  after the push, real frame shown next to imagined +1.

### 5.8 Challenges

- Presets tagged **easy** (small shift + small rotation), **medium**, **hard** (large rotation).
- If the dry run shows the AI fails large rotations: AI rounds use easy/medium; hard ones become
  "human challenge" rounds. Second fallback (narrate honestly): intermediate goal images
  (interpolated T poses rendered by the sim) as waypoints.

## 6. Milestones (in order of risk)

0. **Dry runs (before the hackathon)**
   - JEPA: load model + decoder, one CEM episode in the repo's Push-T env on a GPU; seconds/plan;
     easy vs large-rotation goals; decoder quality; proprio yes/no.
   - Jev: API key works; one image Choice request with 4 synthetic candidate images; measure latency.
1. Game server: two envs, overlap scoring, push counting, goal rendering; headless tests.
2. Planner service: `/plan` returning K diverse candidates with decoded endings and frames.
3. Jev judge module + AI controller loop with escalation and fallbacks; recorded decision points.
4. Accuracy harness; tune τ, K, prompt wording, goal reference.
5. Frontend: AI board + decision strip + imagination strip — **a complete "watch the AI think" demo**.
6. Human board with mouse control, push counter, winner logic; challenge picker and draggable goal.
7. Stretch: instinct player → audience rules → spot the fake → sabotage / leaderboard.

## 7. Left to the builder

- CEM budgets (base and escalated) to fit ~1–3 s per plan on the demo GPU.
- Candidate diversity threshold; action chunk length executed per push.
- Exact Jev prompt wording within §5.3's intent; timeouts; τ (tuned by harness).
- Visual styling (dark, projector-friendly), animations, fonts.
- Round timeout, tie-breaks, preset poses.
- Recording a safe pre-computed round (frames + Jev responses) as a stage fallback.

## 8. Open questions (yours to answer)

1. **TypeSafe access**: do you have a `TYPESAFE_API_KEY` with image requests enabled? (Blocks
   milestone 0's Jev dry run.)
2. **Where does the world model run on demo day** — this Mac or a GPU machine? The design supports
   both; it decides where milestone 0 happens.
3. **What counts as a push?** Default above (contact episode). Alternative: one straight-line stroke
   (human press-drag-release; AI one executed action).

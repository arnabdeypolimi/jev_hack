# Push·Cube — demo video plan

**What it is:** a browser game where you race TypeSafe's decision model Jev to push a block onto a green
outline (position and angle) in the fewest pushes.
**For:** hackathon judges and anyone curious what a "decision model" does differently from a chatbot.
**Sets it apart:** you watch Jev decide — probabilities for direction, distance and aim — and when it's
unsure it simulates three futures with the same physics before committing.
**Hook:** "Can you out-push a decision model?" over six tumbling block shapes.
**Show the thing:** real screen recordings of the running game (1920×1080).
**Tone:** clean demo-explainer (app-store pacing, the game's own lab-bench look).
**Share caption:** "Push·Cube: you vs TypeSafe's Jev. Push a block onto the outline in the fewest pushes,
and watch the decision model think."

**Honesty note:** TypeSafe currently rejects the Jev key (401), so the decision-panel footage was recorded
with sample Jev answers from a local stand-in. Every scene that shows the panel carries the label
"Sample Jev answers — live play needs a TypeSafe key". The human play, physics and 3D view are real.

**Length:** 32 s (a demo *and* how it works; longer than the usual 20 s on purpose).
**Music:** 120 BPM, A minor, soft synth pad + bass + light drums; every cut on the beat (0.5 s grid).

## Storyboard

| # | Time | Scene | On screen |
|---|---|---|---|
| 1 | 0–3 s | Hook | Six block silhouettes tumble in; "Can you out-push a decision model?" |
| 2 | 3–7.5 s | Reveal | Real game, full frame, slow push-in. "Push·Cube — you vs Jev, TypeSafe's decision model" |
| 3 | 7.5–11.5 s | You | Zoom on the YOU board while the pin pushes the L. "You drag the pin. You can only push." |
| 4 | 11.5–16 s | Instinct | Zoom on Jev's panel. "Jev reads raw coordinates and picks direction · distance · aim" |
| 5 | 16–21 s | Think harder | Jev board + panel with ghost futures. "Not sure? It simulates 3 futures with real physics, then decides." |
| 6 | 21–25 s | Shapes | 3×2 grid of real boards (square, I, triangle, L, T, Z). "Six shapes. They slide and spin." |
| 7 | 25–29 s | How it's built | Chips: 2D physics (pymunk) · 3D view (Three.js) · FastAPI · TypeSafe Jev. "Runs on a laptop. No GPU." |
| 8 | 29–32 s | Outro | "Push·Cube" wordmark, "Fewest pushes onto the outline wins." |

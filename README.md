# Push·Cube — You vs Jev

A human and TypeSafe's decision model **Jev** race to push a block (square, T, L, I-bar, Z or
triangle) onto a green goal outline, matching position and angle. Fewest pushes wins. Jev decides every push from raw coordinates (System 1); when it isn't sure, it
simulates its top three options with the same physics and decides again (System 2).

## Run

```bash
cp .env.example .env            # then put your TypeSafe key in TYPESAFE_API_KEY
uv sync
uv run --env-file .env uvicorn pushcube.server:app --port 8000
# open http://localhost:8000
```

Without a working key the game still runs: Jev's board falls back to a one-step oracle (it simulates
every push and plays the best) and the UI says why Jev is offline.

- Pick a challenge at the top, or drag the green goal on either board (scroll over it to rotate), then
  **Start round**.
- Hold the mouse on the YOU board to move your pusher. `V` toggles tilted / top-down view.

## Tools

```bash
uv run pytest -q                                              # test suite
uv run --env-file .env python scripts/dry_run_jev.py           # check the key + let Jev play two presets
uv run --env-file .env python scripts/harness.py --repeats 3   # Jev vs geometric rule vs one-step oracle
RECORD_TO=recordings/round.jsonl uv run --env-file .env uvicorn pushcube.server:app   # record rounds
REPLAY_FROM=recordings/round.jsonl uv run uvicorn pushcube.server:app                 # replay (stage fallback)
```

Design and plan: `docs/superpowers/specs/2026-11-10-push-cube-vs-jev-design.md`,
`docs/superpowers/plans/2026-11-10-push-cube-vs-jev.md`.

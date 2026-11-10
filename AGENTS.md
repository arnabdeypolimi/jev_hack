# AGENTS.md — Push·Cube

## Commands
- Install: `uv sync` (deps resolve with `exclude-newer` ≥ 7 days old, see pyproject).
- Tests: `uv run pytest -q` (≈740 tests, ~25 s; the oracle policy tests simulate every option).
- Lint (no config in repo; lines run to ~120): `uvx ruff check pushcube tests scripts --line-length 120 --select F,E4,E7,E9,I`.
- Run: `uv run --env-file .env uvicorn pushcube.server:app --port 8000`.
- Jev dry run: `uv run --env-file .env python scripts/dry_run_jev.py`.
- Harness: `uv run --env-file .env python scripts/harness.py --repeats 3`.

## Layout
- `pushcube/shapes.py` block shapes; `sim.py` 2D pymunk sim (blocks slide and spin); `scoring.py` overlap, goal clamping, push counting; `options.py` push options (8 directions × short/medium/long × aim left/centre/right); `executor.py` turns an option into pusher targets; `policies.py` geometric baseline, oracle, and the oracle-based fallback; `jev_judge.py` TypeSafe client; `controller.py` AI loop; `game_round.py` round state and winner rule; `challenges.py` presets; `harness.py` headless play; `server.py` FastAPI + WebSocket; `static/` Three.js UI.
- `brag-output/` holds the demo video and its HyperFrames source (not part of the game).

## Notes
- `TYPESAFE_API_KEY` lives only in `.env` (gitignored); never log or send it to the browser.
- Measured square travel per centre push: short ≈ 6 px, medium ≈ 26 px, long ≈ 75 px.
- A push ends only after the pusher is > 8 px away for 0.3 s (spinning blocks can swing back).
- Baseline harness (2026-11-10, all six shapes): oracle solves every preset (mean pushes easy 2.5, medium 5.7, hard 5.3, far 10); geometric misses one hard preset (easy 2.5, medium 9.3, hard 24.3, far 16).
- Jev dry run (2026-11-10): blocked — the provided key returned HTTP 401 on /v1/systemone and /v1/models.

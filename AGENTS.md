# AGENTS.md — Push·Cube

## Commands
- Install: `uv sync` (deps resolve with `exclude-newer` ≥ 7 days old, see pyproject).
- Tests: `uv run pytest -q` (≈135 tests, ~6 s).
- Run: `uv run --env-file .env uvicorn pushcube.server:app --port 8000`.
- Jev dry run: `uv run --env-file .env python scripts/dry_run_jev.py`.
- Harness: `uv run --env-file .env python scripts/harness.py --repeats 3`.

## Layout
- `pushcube/shapes.py` block shapes; `sim.py` 2D pymunk sim (blocks slide and spin); `executor.py` turns an option (8 directions × short/medium/long × aim left/centre/right) into pusher targets; `policies.py` geometric baseline, oracle, and the oracle-based fallback; `jev_judge.py` TypeSafe client; `controller.py` AI loop; `server.py` FastAPI + WebSocket; `static/` Three.js UI.

## Notes
- `TYPESAFE_API_KEY` lives only in `.env` (gitignored); never log or send it to the browser.
- Measured square travel per centre push: short ≈ 6 px, medium ≈ 26 px, long ≈ 75 px.
- A push ends only after the pusher is > 8 px away for 0.3 s (spinning blocks can swing back).
- Full test suite takes ~30 s (the oracle policy tests simulate every option).
- Baseline harness (2026-11-10): geometric/oracle solve every preset; mean pushes easy 2.5, medium 5, hard 5.5–6, far 8.
- Jev dry run (2026-11-10): blocked — the provided key returned HTTP 401 on /v1/systemone and /v1/models.

# AGENTS.md — Push·Cube

## Commands
- Install: `uv sync` (deps resolve with `exclude-newer` ≥ 7 days old, see pyproject).
- Tests: `uv run pytest -q` (≈135 tests, ~6 s).
- Run: `uv run --env-file .env uvicorn pushcube.server:app --port 8000`.
- Jev dry run: `uv run --env-file .env python scripts/dry_run_jev.py`.
- Harness: `uv run --env-file .env python scripts/harness.py --repeats 3`.

## Layout
- `pushcube/sim.py` 2D pymunk cube sim (translation only); `executor.py` turns an option (8 directions × short/medium/long) into pusher targets; `policies.py` geometric fallback + oracle; `jev_judge.py` TypeSafe client; `controller.py` AI loop; `server.py` FastAPI + WebSocket; `static/` Three.js UI.

## Notes
- `TYPESAFE_API_KEY` lives only in `.env` (gitignored); never log or send it to the browser.
- Measured cube travel per push: short ≈ 6 px, medium ≈ 26 px, long ≈ 75 px.
- Baseline harness (2026-11-10): geometric/oracle solve every preset; mean pushes easy 2.5, medium 5, hard 5.5–6, far 8.
- Jev dry run (2026-11-10): blocked — the provided key returned HTTP 401 on /v1/systemone and /v1/models.

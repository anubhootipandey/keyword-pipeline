# Keyword Intelligence Pipeline

**Status: Phase 1 (project scaffold) — not yet functional as a keyword tool.**

This project is being built incrementally. Right now, there is no keyword
extraction logic — this phase only sets up a working FastAPI backend and a
working React + TypeScript frontend that can talk to each other.

The full README (what it does, how the pipeline works, setup instructions,
API docs, limitations, etc.) will be written once there's an actual pipeline
and API to document — writing it now would just be describing a plan, not
a real, working project.

## What exists right now

- `backend/` — FastAPI app with a single `GET /api/health` endpoint
- `frontend/` — React + TypeScript + Vite app that calls the backend health
  endpoint and shows connection status
- Tests: `backend/tests/test_health.py` (backend boots, health check works)

## Running it locally (current scaffold only)

Backend:
```bash
cd backend
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend (separate terminal):
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 — it should show "Backend status: connected".

Full setup docs, including a dedicated Windows walkthrough, will be added
as the project nears completion.

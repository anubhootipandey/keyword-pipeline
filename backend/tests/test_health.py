"""
Phase 1 test: confirms the FastAPI app actually boots and the health
endpoint responds correctly. This is intentionally the only test in
Phase 1 — later phases add tests for the pipeline modules as they're
implemented (see section 27 of the project spec).
"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health_check_returns_ok():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app_name"] == "Keyword Intelligence Pipeline"


@pytest.mark.asyncio
async def test_root_endpoint_lists_docs_and_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["health"] == "/api/health"

"""
GET /api/health

A trivial liveness check. Deliberately has no dependencies on the NLP
pipeline (spaCy, sentence-transformers, etc.) so it stays fast and keeps
working even if a heavier component fails to load — useful once the
pipeline modules exist, and useful right now as the first thing we can
actually test end-to-end.
"""

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
    }

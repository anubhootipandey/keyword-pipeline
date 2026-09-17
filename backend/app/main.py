"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload

(from inside the backend/ directory, with the virtual environment active)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.core.config import get_settings
from app.core.logging import configure_logging

configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description="Local-first keyword extraction and ranking pipeline.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {
        "message": f"{settings.APP_NAME} API",
        "docs": "/docs",
        "health": "/api/health",
    }

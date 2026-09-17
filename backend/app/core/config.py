"""
Application configuration.

We use pydantic-settings so every environment variable the app reads is
declared in one typed place, instead of scattered `os.getenv()` calls with
no validation. This file only defines settings that exist and are used
today (Phase 1). Pipeline-specific settings (thresholds, weights, model
names, etc.) get added in the phases that actually implement them.

When the backend is run from the `backend/` directory (as documented in
the README), pydantic-settings will automatically pick up a `.env` file
sitting next to this project if one exists. No .env file is required to
run the app — every setting below has a sane default.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed application settings.

    Every field here maps to an environment variable of the same name
    (case-insensitive). See the root `.env.example` for documentation of
    each variable.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "Keyword Intelligence Pipeline"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    MAX_INPUT_LENGTH: int = 50_000

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    Using lru_cache means the .env file / environment is only read once
    per process, and the same Settings instance is reused everywhere via
    FastAPI's dependency injection (see api/dependencies.py, added when
    it's actually needed).
    """
    return Settings()

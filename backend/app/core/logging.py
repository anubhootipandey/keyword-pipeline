"""
Basic logging setup.

Nothing fancy: a single configured root logger with a consistent format.
This is intentionally simple for now. If the project ever needs
structured/JSON logging (e.g. for a hosted deployment), that can replace
this without touching call sites, since everything logs via
`logging.getLogger(__name__)`.
"""

import logging
import sys

from app.core.config import get_settings


def configure_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )

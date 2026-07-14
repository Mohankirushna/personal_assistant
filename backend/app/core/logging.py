"""Structured logging setup.

One consistent format across the app's own loggers and uvicorn's, configured
once at startup by :func:`setup_logging`.
"""

from __future__ import annotations

import logging.config

_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": _FORMAT},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "app": {"level": level},
                "uvicorn": {"level": level},
                "uvicorn.access": {"level": "WARNING"},
            },
            "root": {"level": level, "handlers": ["console"]},
        }
    )

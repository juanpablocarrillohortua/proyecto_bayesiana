"""Structured logging setup.

Emits JSON lines so a run can be post-processed, unless the output is a
TTY, in which case a human-readable renderer is used instead.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", force_json: bool = False):
    """Configure structlog once for the whole process."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    use_json = force_json or not sys.stdout.isatty()
    renderer = (
        structlog.processors.JSONRenderer()
        if use_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)

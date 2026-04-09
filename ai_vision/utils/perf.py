"""Lightweight performance logging helpers for probe hot paths."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Iterator


_LOGGER = logging.getLogger(__name__)


@contextlib.contextmanager
def timed_section(
    name: str,
    *,
    threshold_ms: float = 5.0,
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
) -> Iterator[None]:
    """Log sections that exceed the requested duration threshold."""
    active_logger = logger or _LOGGER
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if elapsed_ms >= threshold_ms and active_logger.isEnabledFor(level):
            active_logger.log(level, "[PERF] %s: %.1f ms", name, elapsed_ms)

"""Structured logging shared by every module."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "traffic_volume") -> logging.Logger:
    """Return a logger that writes one timestamped line per message to stderr."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOGGER = get_logger()

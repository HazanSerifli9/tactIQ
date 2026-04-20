"""
Centralised logger factory for TactIQ.

Usage:
    from shared.logger import get_logger
    logger = get_logger(__name__)
    logger.warning("something went wrong: %s", e)
"""

import logging
import os

# ---------------------------------------------------------------------------
# Log file sits at the project root (same directory as app.py / dash_app.log)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_PROJECT_ROOT, "dash_app.log")

_FORMATTER = logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, configuring handlers only once."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    # File handler — WARNING and above go to dash_app.log
    try:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.WARNING)
        fh.setFormatter(_FORMATTER)
        logger.addHandler(fh)
    except OSError:
        pass  # if log file is not writable, skip

    # Console handler — WARNING and above shown in terminal
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(_FORMATTER)
    logger.addHandler(ch)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    return logger

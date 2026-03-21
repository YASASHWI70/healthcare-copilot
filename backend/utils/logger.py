"""
utils/logger.py - Centralized logging configuration
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Configured Logger instance
    """
    from backend.config import LOG_LEVEL, LOG_FILE

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception:
        pass  # Non-fatal if file logging fails

    return logger

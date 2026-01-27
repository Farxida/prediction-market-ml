"""Logging setup for the Polymarket trading system."""

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create a logger with console and file output.

    Args:
        name: logger name (typically __name__)
        level: logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File
    file_handler = logging.FileHandler(LOG_DIR / "trading.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger

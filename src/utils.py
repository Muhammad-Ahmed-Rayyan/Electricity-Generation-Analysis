import logging
import sys
from pathlib import Path

from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to stdout with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_project_directories() -> None:
    """Create every directory the pipeline writes to, if it doesn't already exist."""
    for directory in config.ALL_DIRS:
        Path(directory).mkdir(parents=True, exist_ok=True)
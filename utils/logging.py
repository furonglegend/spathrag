# src/utils/logging.py
"""
Logging utilities.
Provides a get_logger convenience function that configures a logger
with a RotatingFileHandler and optional console output.
All comments and identifiers are in English.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional


def ensure_dir_for_file(filepath: str):
    """Ensure parent directory for filepath exists."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)


def get_logger(
    name: str = "s_path_rag",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Create or get a configured logger.

    Args:
      name: logger name.
      log_file: if provided, add a rotating file handler writing to this file.
      level: logging level.
      console: whether to also log to console (stderr).
      max_bytes: max size for rotating file handler.
      backup_count: number of rotated backups to keep.

    Returns:
      configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # logger already configured, adjust level and return
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if log_file:
        ensure_dir_for_file(log_file)
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # avoid duplicate logs in root logger
    logger.propagate = False
    return logger


# quick demo
if __name__ == "__main__":
    lg = get_logger("demo_logger", log_file="logs/demo.log", level=logging.DEBUG)
    lg.info("Logger initialized for demo")
    lg.debug("Debug message")

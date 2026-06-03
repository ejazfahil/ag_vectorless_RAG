"""
Logging configuration — uses loguru for structured, colorful logging.
"""

from __future__ import annotations

import sys

from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: str | None = None,
    serialize: bool = False,
) -> None:
    """
    Configure the global logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write logs to disk.
        serialize: If True, output structured JSON logs.
    """
    # Remove default handler
    logger.remove()

    # Console handler — rich formatting
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    # File handler (if requested)
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            serialize=serialize,
        )
        logger.info(f"File logging enabled: {log_file}")

    logger.info(f"Logger initialized at level={level}")

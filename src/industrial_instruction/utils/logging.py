"""Single place to configure logging for CLI and library use.

The public entry point is `configure_logging`. Modules should call
`get_logger`, which configures the handler on first use, so importing
the library never emits output on its own.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_HANDLER_TAG = "_industrial_instruction_handler"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: Optional[str] = None) -> None:
    """Attach a stderr handler once. Level falls back to ``II_LOG_LEVEL``."""
    resolved = (level or os.environ.get("II_LOG_LEVEL") or "INFO").upper()
    root = logging.getLogger("industrial_instruction")
    # Look for our own handler rather than trusting a module flag, so a
    # module reload (or another tool adding handlers) can't cause duplicates.
    if not any(getattr(h, _HANDLER_TAG, False) for h in root.handlers):
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, "%H:%M:%S"))
        setattr(handler, _HANDLER_TAG, True)
        root.addHandler(handler)
        root.propagate = False
    root.setLevel(getattr(logging, resolved, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a namespaced logger, configuring the root handler on first use."""
    configure_logging()
    if not name.startswith("industrial_instruction"):
        name = f"industrial_instruction.{name}"
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger"]

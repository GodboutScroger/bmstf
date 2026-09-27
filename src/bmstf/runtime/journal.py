"""Logging setup for the command-line entry points.

Ref: Code availability (the run record the study writes beside its results).
"""

from __future__ import annotations

import logging
import sys


def configure(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("bmstf")
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    return root


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"bmstf.{name}")

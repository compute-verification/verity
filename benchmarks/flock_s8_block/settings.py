"""The statement's shape, which `run.sh` sets per statement for `program.py`: ``K`` (the template's K and L), ``ROWS`` (its N,
the rows of one proof block; 8 unset) and ``N`` (the instances)."""
from __future__ import annotations

import os


def k() -> int:
    return int(os.environ["K"])


def rows() -> int:
    return int(os.environ.get("ROWS", "8"))


def n() -> int:
    return int(os.environ["N"])

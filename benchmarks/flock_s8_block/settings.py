"""The statement's shape, which `run.sh` sets per statement for `program.py`: ``K`` (the template's K and L), ``ROWS`` (its N,
the rows of one proof block; 8 unset), ``N`` (the instances) and ``TILE`` (``PxC``: the template is one coordinate, staged as
P x C tiles under ``FLOCK_GEMM_TILE``; unset, a block of ``ROWS`` rows)."""
from __future__ import annotations

import os


def k() -> int:
    return int(os.environ["K"])


def rows() -> int:
    return int(os.environ.get("ROWS", "8"))


def n() -> int:
    return int(os.environ["N"])


def tile() -> tuple[int, int] | None:
    t = os.environ.get("TILE")
    return tuple(int(x) for x in t.split("x")) if t else None

"""The v1 / v2 switch of value acquisition.  `ACQUIRE_ENGINE=v2` selects the plan-installed hooks; anything else (unset, `v1`) is v1."""
from __future__ import annotations

import os

ENV = "ACQUIRE_ENGINE"
ENGINES = ("v1", "v2")
DEFAULT = "v1"


def engine(value: str | None = None) -> str:
    v = (os.environ.get(ENV, DEFAULT) if value is None else value).strip().lower() or DEFAULT
    if v not in ENGINES:
        raise ValueError(f"{ENV}={v!r}: expected one of {ENGINES}")
    return v


def v2(value: str | None = None) -> bool:
    return engine(value) == "v2"

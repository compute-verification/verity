"""The value-acquisition engine: `ACQUIRE_ENGINE` (default `v2`, the plan-installed hooks -- THE integration since main 22e10e0 + this flip).

`ACQUIRE_ENGINE=v1` stays selectable for ONE more release as the REGRESSION ORACLE ONLY: it reproduces the acquisition the frozen R19
records (`tests/regression/fixtures.toml`, the `fixture/v1` trees on R2) were made with, so a difference can be attributed.  Nothing of
record runs it; a row of record under `v1` is not a row of record.  The oracle itself is the frozen records, never the v1 code.
"""
from __future__ import annotations

import os

ENV = "ACQUIRE_ENGINE"
ENGINES = ("v1", "v2")
DEFAULT = "v2"
ORACLE = "v1"                     # regression oracle only; removed one release after the flip


def engine(value: str | None = None) -> str:
    v = (os.environ.get(ENV, DEFAULT) if value is None else value).strip().lower() or DEFAULT
    if v not in ENGINES:
        raise ValueError(f"{ENV}={v!r}: expected one of {ENGINES}")
    return v


def v2(value: str | None = None) -> bool:
    return engine(value) == "v2"

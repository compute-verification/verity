"""circuit-check's bindings of `verity_flock.rec_link`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s link check (`verity_flock.rec_link`) at its smallest shapes: trees of depth 0 (each leaf its own root: no climb, and
    every field must be zero) and of depth 2 (one unit bit and one port bit: a level of each), each with its two hm96 leaves
    and the row's checks. A larger population adds levels of the same climb."""
    from verity_flock.rec_link import definition

    return [
        definition(0, 0),
        definition(1, 1),
    ]

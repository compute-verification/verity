"""circuit-check's bindings of `verity_flock.rec_open`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s opening check (`verity_flock.rec_open`) at its smallest shapes: a row of 8 lanes (one SHA-512 block) and one
    level, four compressions (the plain leaf's two and the level's two) and the consistency term's 16 products; and the same
    row with no level (``H = 0``: the leaf is the top, ``aux`` and ``dirs`` unread). Wider rows add message blocks and
    products, and higher openings repeat the level (above 16 reading the side bits from ``dirs``' word 1)."""
    from verity_flock.rec_open import definition

    return [
        definition(8, 0),
        definition(8, 1),
    ]

"""circuit-check's bindings of `verity_flock.rec_open`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s opening check (`verity_flock.rec_open`) at its smallest shape: a row of 8 lanes (one SHA-512 block) and one
    level, six compressions and the consistency term's 16 products. Wider rows add message blocks and products, and higher
    openings repeat the level."""
    from verity_flock.rec_open import definition

    return [
        definition(8, 1),
    ]

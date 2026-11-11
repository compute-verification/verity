"""circuit-check's bindings of `verity_flock.rec_sparse`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _rec_roots() -> list[Any]:
    """V*'s sparse phase (`verity_flock.rec_sparse`) at its smallest shape with every part: 4 entries, the skip's 6 row bits,
    two slots (a round over the slot bit) of two stacked types (the selects, ``vt`` and the plane batch). A larger shape adds
    rounds, Φ factors and planes of the same Calls."""
    from verity_flock.rec_sparse import definition

    return [
        definition(2, 6, 1, 1),
    ]

"""circuit-check's bindings of `verity.definitions.registry.pearl_c_forming`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`).

Every Boolean Definition bound here is within the 65,536 ANDs circuit-check lowers a Definition whole at, and pinned, but
one: the forming of a whole row, `PearlCForm_v2`, whose 32 scaled codes alone are 84,608. It is σ of Calls of three
Definitions that are bound and pinned here: the statistics, the scaled code and the formed code."""
from __future__ import annotations

from typing import Any


def _pearl_c_forming_roots() -> list[Any]:
    """Forming v1, the chain, the Gram matrix, the peel factors and the peel, on words and on bits (`pearl_c_forming.bindings`):
    the served sizes (rank 32, k in the thousands) are the same bodies over more columns and more atoms."""
    from verity.definitions.registry import pearl_c_forming as F

    return F.bindings()

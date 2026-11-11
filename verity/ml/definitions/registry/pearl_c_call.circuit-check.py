"""circuit-check's bindings of `verity.ml.definitions.registry.pearl_c_call`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`).

A row of a call is over the 65,536 ANDs circuit-check lowers a Definition whole at, at every K, so neither row is lowered
whole and neither has a pin: each is the σ of Calls of Definitions `pearl_c_noise` and `pearl_c_forming` bind, whose
units are pinned there (`pearl_c_call.units` lays a row out in them). Their other checks run on the whole, at the least
row: the word view, the partition, the proof block and the gates."""
from __future__ import annotations

from typing import Any


def _pearl_c_call_roots() -> list[Any]:
    """A call's row of A (its noise line derived from the salt and the row) and its row of B, on words and on bits, on the
    H100's atom over one atom of the row at rank 1."""
    from verity.ml.definitions.registry import pearl_c_call as K

    return K.bindings()

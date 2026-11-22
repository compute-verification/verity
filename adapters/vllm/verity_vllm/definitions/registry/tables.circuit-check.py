"""circuit-check's bindings of `verity_vllm.definitions.registry.tables`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _table_read_roots() -> list[Any]:
    """The word reads at small tables, each with its Boolean form (`verity.circuits.boolean.table`, which Flock lowers it
    by): one row of 2^6 one-word entries, `exp2_lut127`'s 2^8 entries of two words, and 2^10 one-word entries in 16 rows."""
    from verity.circuits.boolean import table as BT
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import tables

    out: list[Any] = []
    for ib, ew in ((6, 1), (8, 2), (10, 1)):
        word = tables.table_read(ib, ew)
        out += [word, bind(BT.table_read(word, ib))]
    return out

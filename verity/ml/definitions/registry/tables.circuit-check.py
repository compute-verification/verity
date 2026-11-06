"""circuit-check's bindings of `verity.ml.definitions.registry.tables`' Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import tables

    return [
        # the Definitions that read a registered table (`gelu_tanh_bf16`, `tanh_rn`) through `TableRead16x1_v1`, `TableRead27x2_v1`;
        # no rule binds them (`boolean_tanh` computes both tables' entries)
        ("GeluTanhMulBf16_v3", lambda: bind(tables.GeluTanhMulBf16V3)),
        ("TanhF32Rn_v2", lambda: bind(tables.TanhF32RnV2)),
    ]


def _table_read_roots() -> list[Any]:
    """The word reads at small tables, each with its Boolean form (`verity.ml.circuits.boolean.table`, which Flock lowers it
    by): one row of 2^6 one-word entries, `exp2_lut127`'s 2^8 entries of two words, and 2^10 one-word entries in 16 rows."""
    from verity.ml.circuits.boolean import table as BT
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import tables

    out: list[Any] = []
    for ib, ew in ((6, 1), (8, 2), (10, 1)):
        word = tables.table_read(ib, ew)
        out += [word, bind(BT.table_read(word, ib))]
    return out

"""circuit-check's bindings of `verity.ml.definitions.registry.dsv4_decode_metadata`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Both rows at DeepSeek-V4-Pro's statics (`dsv4_decode_metadata.DSV4_DECODE_METADATA`: the SWA row's window of 128 over its
    block table's row of blocks of 64 at max_model_len 1,048,576 and 4096; the c128a row at 4096's width of 128 over the
    compressed block table's row of blocks of 2), which reach their window, lanes, slot and length, and at small ones; the
    length also at the widest active top-k width, 8192.  The c128a row is not bound at 8192: its `Dsv4ShortContextTopK_v1{8192,
    128}` spells two 32-bit literals per lane, and each `Const32[...]_v1` a body reaches is a target of its own (16,328 here)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsv4_decode_metadata as X

    S = X.DSV4_DECODE_METADATA
    return [
        ("Dsv4SwaIndices_v1:1", lambda: bind(X.Dsv4SwaIndices, **S["swa"])),
        ("Dsv4SwaIndices_v1:2", lambda: bind(X.Dsv4SwaIndices, **S["swa-4096"])),
        ("Dsv4SwaIndices_v1:3", lambda: bind(X.Dsv4SwaIndices, SW=8, BS=4, NB=8)),
        ("Dsv4C128aGlobalTopk_v1:1", lambda: bind(X.Dsv4C128aGlobalTopk, **S["c128-4096"])),
        ("Dsv4C128aGlobalTopk_v1:2", lambda: bind(X.Dsv4C128aGlobalTopk, W=16, R=4, BS=2, NB=8)),
        ("Dsv4C128aLen_v1", lambda: bind(X.Dsv4C128aLen, W=S["c128"]["W"])),
    ]

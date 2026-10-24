"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsv4_decode_metadata`'s Definitions, in
`circuit_check.targets`' forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """Both rows at max_model_len 4096's statics (the SWA row over a block-table row of 64, its lanes the served ones; the c128a
    row at width 128 over 16 blocks, its slot the served one, and 96 lanes past the row) and at small ones that reach odd
    multiplexer levels and lanes past a row of 4 blocks; the served SWA window (a block-table row of 16,384, over
    `targets.MAX_GATES`, so checked when named) and the served c128a length (width 8192).  The served c128a row is
    `Dsv4ShortContextTopK_v2{8192,128}`, 4096 Calls of the slot and the length."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsv4_decode_metadata as BX

    return [
        ("Dsv4SwaIndices_v2:1", lambda: bind(BX.Dsv4SwaIndicesV2, SW=128, BS=64, NB=64)),
        ("Dsv4SwaIndices_v2:2", lambda: bind(BX.Dsv4SwaIndicesV2, SW=8, BS=4, NB=8)),
        ("Dsv4SwaIndices_v2:3", lambda: bind(BX.Dsv4SwaIndicesV2, SW=5, BS=2, NB=3)),
        ("Dsv4SwaWindow_v2", lambda: bind(BX.Dsv4SwaWindowV2, SW=128, BS=64, NB=16384)),
        ("Dsv4C128aGlobalTopk_v2:1", lambda: bind(BX.Dsv4C128aGlobalTopkV2, W=128, R=128, BS=2, NB=16)),
        ("Dsv4C128aGlobalTopk_v2:2", lambda: bind(BX.Dsv4C128aGlobalTopkV2, W=16, R=4, BS=2, NB=8)),
        ("Dsv4C128aGlobalTopk_v2:3", lambda: bind(BX.Dsv4C128aGlobalTopkV2, W=12, R=3, BS=2, NB=4)),
        ("Dsv4C128aLen_v2", lambda: bind(BX.Dsv4C128aLenV2, W=8192)),
    ]

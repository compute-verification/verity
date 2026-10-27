"""circuit-check's bindings of `verity.ml.definitions.registry.dsv4_indexer`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at DeepSeek-V4-Pro's served shapes: the indexer's 256-wide state score, 8-row columns, D = 128
    entries with a 64-wide rotary suffix, heads of 4 x 128 with 64 rotary, and the top-1024 fill at ratio 4."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsv4_indexer as I

    return [
        ("Dsv4CompressorScore_v1", lambda: bind(I.Dsv4CompressorScore, W=256)),
        ("Dsv4CompressColumn_v1", lambda: bind(I.Dsv4CompressColumn, ROWS=8)),
        ("Dsv4IndexerNormRope_v1", lambda: bind(I.Dsv4IndexerNormRope, D=128, ROPE=64, EPS=1e-6)),
        ("Dsv4IndexerKQuant_v1", lambda: bind(I.Dsv4IndexerKQuant, N=128)),
        ("Dsv4IndexerCompress_v1", lambda: bind(I.Dsv4IndexerCompress, D=128, ROPE=64, EPS=1e-6)),
        ("Dsv4InvRope_v1", lambda: bind(I.Dsv4InvRope, ROPE=64)),
        ("Dsv4InvRopeQuant_v1", lambda: bind(I.Dsv4InvRopeQuant, N=128)),
        ("Dsv4InvRopeQuantHead_v1", lambda: bind(I.Dsv4InvRopeQuantHead, CHUNKS=4, QG=128, ROPE=64)),
        ("Dsv4ShortContextTopK_v1", lambda: bind(I.Dsv4ShortContextTopK, TOPK=1024, RATIO=4)),
    ]

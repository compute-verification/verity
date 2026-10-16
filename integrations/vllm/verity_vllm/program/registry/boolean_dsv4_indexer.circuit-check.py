"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsv4_indexer`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at small statics: the column at the kernel's 8 rows, an entry at one column per lane (D = 32),
    a head of two groups of 8 with a rotary suffix of 4, the fill at 16 slots."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsv4_indexer as BI

    return [
        ("Dsv4CompressorScore_v2", lambda: bind(BI.Dsv4CompressorScoreV2, W=4)),
        ("Dsv4CompressColumn_v2", lambda: bind(BI.Dsv4CompressColumnV2, ROWS=8)),
        ("Dsv4IndexerNormRope_v2", lambda: bind(BI.Dsv4IndexerNormRopeV2, D=32, ROPE=16, EPS=1e-6)),
        ("Dsv4IndexerKQuant_v2", lambda: bind(BI.Dsv4IndexerKQuantV2, N=4)),
        ("Dsv4IndexerCompress_v2", lambda: bind(BI.Dsv4IndexerCompressV2, D=32, ROPE=16, EPS=1e-6)),
        ("Dsv4InvRope_v2", lambda: bind(BI.Dsv4InvRopeV2, ROPE=4)),
        ("Dsv4InvRopeQuant_v2", lambda: bind(BI.Dsv4InvRopeQuantV2, N=4)),
        ("Dsv4InvRopeQuantHead_v2", lambda: bind(BI.Dsv4InvRopeQuantHeadV2, CHUNKS=2, QG=8, ROPE=4)),
        ("Dsv4ShortContextTopK_v2", lambda: bind(BI.Dsv4ShortContextTopKV2, TOPK=16, RATIO=4)),
    ]

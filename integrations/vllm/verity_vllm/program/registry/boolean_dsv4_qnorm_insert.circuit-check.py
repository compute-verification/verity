"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsv4_qnorm_insert`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """One binding per family at small statics (one warp's 32 elements, a 16-wide quant block): every one is flat and over
    `targets.MAX_GATES` (a head is 120,000 to 160,000 ANDs, the insert of one head and its pad 380,000), so `select` keeps a
    family's smallest whatever is listed."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsv4_qnorm_insert as BQ

    return [
        ("Dsv4QNormRope_v2", lambda: bind(BQ.Dsv4QNormRopeV2, D=32, DR=8, EPS=1e-6)),
        ("Dsv4KvRopeQuant_v2", lambda: bind(BQ.Dsv4KvRopeQuantV2, DN=16, DR=16, QB=16)),
        ("Dsv4QNormRopeKvInsert_v2", lambda: bind(BQ.Dsv4QNormRopeKvInsertV2, NH=1, NHP=2, DN=16, DR=16, QB=16, EPS=1e-6)),
        ("Dsv4IndexerQHead_v2", lambda: bind(BQ.Dsv4IndexerQHeadV2, D=32, DR=16, SCALE=0.75)),
        ("Dsv4IndexerQ_v2", lambda: bind(BQ.Dsv4IndexerQV2, H=2, D=32, DR=16, SCALE=0.75)),
    ]

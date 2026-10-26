"""circuit-check's bindings of `verity.ml.definitions.registry.dsv4_qnorm_insert`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions at DeepSeek-V4-Pro's statics on the H200: one query head (D = 512) and the KV row, a TP8 rank's
    whole insert (16 heads padded to 64, `dsv4_qnorm_insert.DSV4_INSERT`), and the indexer's head and token (64 heads of 128,
    `DSV4_INDEXER`)."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import dsv4_qnorm_insert as Q

    S, X = Q.DSV4_INSERT, Q.DSV4_INDEXER
    return [
        ("Dsv4QNormRope_v1", lambda: bind(Q.Dsv4QNormRope, D=S["DN"] + S["DR"], DR=S["DR"], EPS=S["EPS"])),
        ("Dsv4KvRopeQuant_v1", lambda: bind(Q.Dsv4KvRopeQuant, DN=S["DN"], DR=S["DR"], QB=S["QB"])),
        ("Dsv4QNormRopeKvInsert_v1", lambda: bind(Q.Dsv4QNormRopeKvInsert, **S)),
        ("Dsv4IndexerQHead_v1", lambda: bind(Q.Dsv4IndexerQHead, D=X["D"], DR=X["DR"], SCALE=X["SCALE"])),
        ("Dsv4IndexerQ_v1", lambda: bind(Q.Dsv4IndexerQ, **X)),
    ]

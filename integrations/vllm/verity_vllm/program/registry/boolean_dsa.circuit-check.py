"""circuit-check's bindings of `verity_vllm.program.registry.boolean_dsa`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import boolean_dsa

    return [
        ("DsaUe8m0Quant_v2", lambda: bind(boolean_dsa.DsaUe8m0QuantV2, N=4)),
        ("DsaIndexerKRow_v2", lambda: bind(boolean_dsa.DsaIndexerKRowV2, D=32, ROPE=16, EPS=1e-6)),
        ("DsaIndexerQHead_v2", lambda: bind(boolean_dsa.DsaIndexerQHeadV2, D=32, ROPE=16, NH=32)),
        ("DsaIndexerQ_v2", lambda: bind(boolean_dsa.DsaIndexerQV2, H=2, D=32, ROPE=16)),
        ("DsaMqaLogit_v2", lambda: bind(boolean_dsa.DsaMqaLogitV2, H=8, D=32)),
        ("DsaMqaLogits_v2", lambda: bind(boolean_dsa.DsaMqaLogitsV2, N=2, H=8, D=32)),
        ("DsaTopKShortRow_v2", lambda: bind(boolean_dsa.DsaTopKShortRowV2, TOPK=16)),
        ("DsaTopKSelection_v2", lambda: bind(boolean_dsa.DsaTopKSelectionV2, TOPK=16, N=64)),
        ("DsaTopKSelection_v2 ragged", lambda: bind(boolean_dsa.DsaTopKSelectionV2, TOPK=5, N=37)),
        # four chunks of 16; and four of 8 with a ragged last one of 5 positions (W < 2^w)
        ("DsaTopKSelection_v4", lambda: bind(boolean_dsa.DsaTopKSelectionV4, TOPK=16, N=64, C=16)),
        ("DsaTopKSelection_v4 ragged", lambda: bind(boolean_dsa.DsaTopKSelectionV4, TOPK=5, N=37, C=8)),
        # v4 is traced, so it reaches neither: each chunk of both, and their finals
        *((f"DsaTopKChunk_v1 H={h}", lambda h=h: bind(boolean_dsa.DsaTopKChunk, TOPK=16, N=64, C=16, H=h)) for h in range(4)),
        *((f"DsaTopKChunk_v1 ragged H={h}", lambda h=h: bind(boolean_dsa.DsaTopKChunk, TOPK=5, N=37, C=8, H=h)) for h in range(5)),
        ("DsaTopKFinal_v1", lambda: bind(boolean_dsa.DsaTopKFinal, TOPK=16, N=64, C=16)),
        ("DsaTopKFinal_v1 ragged", lambda: bind(boolean_dsa.DsaTopKFinal, TOPK=5, N=37, C=8)),
    ]

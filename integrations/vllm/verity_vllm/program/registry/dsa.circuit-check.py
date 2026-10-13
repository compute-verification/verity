"""circuit-check's bindings of `verity_vllm.program.registry.dsa`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsa

    return [
        # GLM-5.3's DSA indexer at its served shapes (D = 128, 32 heads, the top-2048 row), and on bits at one warp
        ("DsaUe8m0Quant_v1", lambda: bind(dsa.DsaUe8m0Quant, N=128)),
        ("DsaIndexerKRow_v1", lambda: bind(dsa.DsaIndexerKRow, D=128, ROPE=64, EPS=1e-6)),
        ("DsaIndexerQHead_v1", lambda: bind(dsa.DsaIndexerQHead, D=128, ROPE=64, NH=32)),
        ("DsaIndexerQ_v1", lambda: bind(dsa.DsaIndexerQ, H=32, D=128, ROPE=64)),
        ("DsaMqaLogit_v1", lambda: bind(dsa.DsaMqaLogit, H=32, D=128)),
        ("DsaMqaLogits_v1", lambda: bind(dsa.DsaMqaLogits, N=4, H=32, D=128)),
        ("DsaTopKShortRow_v1", lambda: bind(dsa.DsaTopKShortRow, TOPK=2048)),
        # the long-row check at a top-16 of 64 keys and at a ragged row (no power of two in either static)
        ("DsaTopKSelection_v1", lambda: bind(dsa.DsaTopKSelection, TOPK=16, N=64)),
        ("DsaTopKSelection_v1 ragged", lambda: bind(dsa.DsaTopKSelection, TOPK=5, N=37)),
        ("DsaTopKSelection_v3", lambda: bind(dsa.DsaTopKSelectionV3, TOPK=16, N=64)),
        ("DsaTopKSelection_v3 ragged", lambda: bind(dsa.DsaTopKSelectionV3, TOPK=5, N=37)),
    ]

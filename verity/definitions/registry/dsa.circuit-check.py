"""circuit-check's bindings of `verity.definitions.registry.dsa`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's DSA indexer logits at its served shapes over every key count a served row takes, checked at these spots
#: (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaMqaLogits_v3``).  Past about 32,480 keys the parameters are over
#: the partition's leaf budget, so the last spot is partitioned by its `DsaMqaLogit_v3` Calls (`circuit_check.checks.by_callees`);
#: serving cuts a row into N = 1 slices (`verity_vllm.pipeline.serving_templates.SLICES`), the unit N = 1 checks whole
RANGES = {
    "DsaMqaLogits_v3{N=1..32533,H=32,D=128}": (
        # the ends: one key, and the longest served row
        "N=1", "N=32533",
        # a short row, the top-k's identity limit and one past it, and a long row
        "N=288", "N=2048", "N=2049", "N=8192",
    ),
}
#: `DsaMqaLogits_v3`'s struct budget (its pins' ``struct_max_gates``, `circuit_check.checks.struct_budget`) is 204 IR gates a
#: key times 2,048 keys: a row of at most 2,048 keys is realized by its structural lowering too, within about ten minutes a
#: check; a longer row is not (C-Flock's template format holds no more than about 3,900 keys of it unrolled)


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa

    return [
        # GLM-5.3's DSA indexer at its served shapes (D = 128, 32 heads, the top-2048 row), and on bits at one warp
        ("DsaUe8m0Quant_v1", lambda: bind(dsa.DsaUe8m0Quant, N=128)),
        ("DsaIndexerKRow_v1", lambda: bind(dsa.DsaIndexerKRow, D=128, ROPE=64, EPS=1e-6)),
        ("DsaIndexerQHead_v1", lambda: bind(dsa.DsaIndexerQHead, D=128, ROPE=64, NH=32)),
        ("DsaIndexerQ_v1", lambda: bind(dsa.DsaIndexerQ, H=32, D=128, ROPE=64)),
        ("DsaMqaLogit_v1", lambda: bind(dsa.DsaMqaLogit, H=32, D=128)),
        ("DsaMqaLogits_v1", lambda: bind(dsa.DsaMqaLogits, N=4, H=32, D=128)),
        ("DsaMqaLogit_v3", lambda: bind(dsa.DsaMqaLogitV3, H=32, D=128)),
        ("DsaMqaLogits_v3", lambda: bind(dsa.DsaMqaLogitsV3, N=4, H=32, D=128)),
        ("DsaTopKShortRow_v1", lambda: bind(dsa.DsaTopKShortRow, TOPK=2048)),
    ]

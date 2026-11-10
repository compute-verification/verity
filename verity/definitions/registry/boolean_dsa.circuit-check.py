"""circuit-check's bindings of `verity.definitions.registry.boolean_dsa`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable

#: GLM-5.3's DSA indexer logits on bits at its served shapes over every key count a 128k-context row takes, checked at these spots
#: (`circuit_check.ranges`; ``circuit-check --as-call --ranges DsaMqaLogits_v4``).  Its description is the same at every N (one
#: `DsaMqaLogitsQ_v1{32,128}` Call and a batch of `DsaMqaLogitsKey_v1{32,128}` over the keys); on bits k is N x 1,024 leaves, so
#: past 4,096 keys it is over the input budget.  From N = 2 it is over the partition's gate budget, and is partitioned by those
#: two Calls (`circuit_check.checks.by_callees`)
RANGES = {
    "DsaMqaLogits_v4{N=1..131072,H=32,D=128}": (
        # the ends: one key, and a 128k-context row
        "N=1", "N=131072",
        # the first batch of more than one key, the top-k's identity limit and one past it
        "N=2", "N=2048", "N=2049",
        # the last key count whose k is within the input budget and the first past it, v3's longest registered row, and 64k
        "N=4096", "N=4097", "N=32533", "N=65536",
    ),
}


def _glm53_served_roots() -> list[Any]:
    """`DsaMqaLogits_v4`'s two blocks at GLM-5.3's served H and D, each checked whole: past 4,096 keys the composite's
    correspondence is unchecked (its word view compares it), so its blocks' is here.  Both are within their family's struct
    budget (this module's pins' ``struct_max_gates``), so each is realized by its structural lowering, and over
    `targets.MAX_GATES`, so each holds a ``digests`` pin, which keeps it in ``--all``."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa

    return [bind(boolean_dsa.DsaMqaLogitsQ, H=32, D=128), bind(boolean_dsa.DsaMqaLogitsKey, H=32, D=128)]


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, the vLLM tests'), and the
    leaf families they do not reach, at small statics."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import boolean_dsa

    return [
        ("DsaUe8m0Quant_v2", lambda: bind(boolean_dsa.DsaUe8m0QuantV2, N=4)),
        ("DsaIndexerKRow_v2", lambda: bind(boolean_dsa.DsaIndexerKRowV2, D=32, ROPE=16, EPS=1e-6)),
        ("DsaIndexerQHead_v2", lambda: bind(boolean_dsa.DsaIndexerQHeadV2, D=32, ROPE=16, NH=32)),
        ("DsaIndexerQ_v2", lambda: bind(boolean_dsa.DsaIndexerQV2, H=2, D=32, ROPE=16)),
        ("DsaMqaLogit_v2", lambda: bind(boolean_dsa.DsaMqaLogitV2, H=8, D=32)),
        ("DsaMqaLogits_v2", lambda: bind(boolean_dsa.DsaMqaLogitsV2, N=2, H=8, D=32)),
        # H = 16: each lane sums two heads, so the contracted first FFMA is in the circuit (at H = 8 v4 is v2)
        ("DsaMqaLogit_v4", lambda: bind(boolean_dsa.DsaMqaLogitV4, H=16, D=32)),
        ("DsaMqaLogits_v4", lambda: bind(boolean_dsa.DsaMqaLogitsV4, N=2, H=16, D=32)),
        ("DsaTopKShortRow_v2", lambda: bind(boolean_dsa.DsaTopKShortRowV2, TOPK=16)),
    ]

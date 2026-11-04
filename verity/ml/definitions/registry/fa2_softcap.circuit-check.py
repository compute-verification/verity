"""circuit-check's bindings of `verity.ml.definitions.registry.fa2_softcap`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """Soft-capped FA2 at CAP = 30 on the Ampere k16 step: the frontend's `attention_dot` instance with a CAP (T = 5, two query
    heads on one KV head, D = 16, one block of 16 under Check_inf), and one head over 17 keys in blocks of 16 at MASKED_FROM 0
    and 1, which reach the second block with and without Check_inf."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import fa2_softcap as FS
    from verity.ml.definitions.registry import prims as P

    on = {"CAP": 30.0, "DOT": P.AmpereBF16TcDot16, "INV": P.Fa2InvSum}
    return [
        bind(FS.AttentionSoftcapV2, T=5, NH=2, KVH=1, D=16, BN=16, MASKED_FROM=0, **on),
        *[bind(FS.AttentionHeadSoftcapV2, T=17, D=16, BN=16, MASKED_FROM=mf, **on) for mf in (0, 1)],
    ]

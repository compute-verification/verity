"""circuit-check's bindings of `verity_vllm.definitions.registry.attention_key_units`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any, Callable


def _model_roots() -> list[tuple[str, Callable[[], Any]]]:
    """FA2's head at D 16, kBlockN 32 and P chunks of 16 keys, a key's value 4 slices of 4 columns: v21's head over a first block
    of 16 keys and one block below it (48 keys), and over a first block of 8 (40 keys, its P·V chunk padded to a k-step), and
    its attention; the softmax step first and below with its block guarded or not, and P·V from zero (padded), rescaled and
    carried.  Then Llama-3.2-1B's two largest units at the window's grain (D 64, kBlockN 192, P chunks of 48 keys, value
    slices of 16 columns): the softmax step of a full block below the first, and a rescaled P·V chunk of 48 keys."""
    from verity.circuits.defs import bind

    from verity_vllm.definitions.registry import attention_key_units as AK
    from verity_vllm.definitions.registry import hopper
    from verity_vllm.definitions.registry import prims as P

    dot = hopper.HopperBF16WgmmaDot16
    fa2 = {"D": 16, "BN": 32, "DOT": dot, "INV": P.Fa2InvSum, "PK": 16, "VS": 4}
    soft = {"D": 16, "DOT": dot, "PK": 16}
    pv = {"D": 16, "DOT": dot, "VS": 4}
    return [
        ("AttentionHead_v21", lambda: bind(AK.AttentionHeadV21, T=48, MASKED_FROM=1, **fa2)),
        ("AttentionHead_v21-padded", lambda: bind(AK.AttentionHeadV21, T=40, MASKED_FROM=0, **fa2)),
        ("Attention_v21", lambda: bind(AK.AttentionV21, T=48, NH=2, KVH=1, MASKED_FROM=1, **fa2)),
        ("AttnSoftmaxStep_v1-first", lambda: bind(AK.AttnSoftmaxStepV1, NVIS=24, FIRST=True, CHECK=True, **soft)),
        ("AttnSoftmaxStep_v1", lambda: bind(AK.AttnSoftmaxStepV1, NVIS=32, FIRST=False, CHECK=True, **soft)),
        ("AttnSoftmaxStep_v1-unguarded", lambda: bind(AK.AttnSoftmaxStepV1, NVIS=32, FIRST=False, CHECK=False, **soft)),
        ("AttnPV_v1-first", lambda: bind(AK.AttnPVV1, NV=8, FIRST=True, RESCALE=False, **pv)),
        ("AttnPV_v1", lambda: bind(AK.AttnPVV1, NV=16, FIRST=False, RESCALE=True, **pv)),
        ("AttnPV_v1-carried", lambda: bind(AK.AttnPVV1, NV=16, FIRST=False, RESCALE=False, **pv)),
        ("AttnSoftmaxStep_v1-llama", lambda: bind(AK.AttnSoftmaxStepV1, D=64, NVIS=192, FIRST=False, CHECK=False, DOT=dot, PK=48)),
        ("AttnPV_v1-llama", lambda: bind(AK.AttnPVV1, D=64, NV=48, FIRST=False, RESCALE=True, DOT=dot, VS=16)),
    ]

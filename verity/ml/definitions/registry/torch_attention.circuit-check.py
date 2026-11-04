"""circuit-check's bindings of `verity.ml.definitions.registry.torch_attention`'s Definitions, in `circuit_check.targets`'
forms (`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """torch's eager attention kernels at small rows: the softmax and its backward at N = 6 (P = 8, one element a lane), the
    eager row and its backward at smollm2's D = 64, RoPE and its backward on a head of 8, and repeat_kv's backward at
    smollm2's three query heads a KV head. The step's blocks at one sequence of 16 tokens with D = 16, two query heads a KV
    head."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_attention as TA

    block = dict(N=16, NH=4, KVH=2, D=16)
    return [
        bind(TA.EagerAttentionTorch, **block),
        bind(TA.EagerAttentionTorchBackward, **block),
        bind(TA.SoftmaxTorch, N=6),
        bind(TA.SoftmaxTorchBackward, N=6),
        bind(TA.EagerSoftmax, N=6, D=64),
        bind(TA.EagerSoftmaxBackward, N=6, D=64),
        bind(TA.RopeHf, D=8),
        bind(TA.RopeHfBackward, D=8),
        bind(TA.RepeatKvBackward, NREP=3, N=4),
    ]

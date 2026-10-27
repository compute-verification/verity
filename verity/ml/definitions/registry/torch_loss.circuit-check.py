"""circuit-check's bindings of `verity.ml.definitions.registry.torch_loss`'s Definitions, in `circuit_check.targets`' forms
(`circuit_check.sidecars`)."""
from __future__ import annotations

from typing import Any


def _boolean_roots() -> list[Any]:
    """torch's cross-entropy kernels at the least vocabulary the block kernel runs at, V = 12,288 (three loads a thread): a
    warp's max and sum (reaching the lanes' at KL = 3), the label's slots and a warp's backward (reaching
    `DsaTopKOneHot_v1{N=384}`), and the loss over 16 rows with and without the mean, with its backward; then the step's blocks,
    a row and a row's backward, reaching the row's end."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import torch_loss as TL

    return [
        bind(TL.LogSoftmaxWarpMax, V=12288),
        bind(TL.LogSoftmaxWarpSumExp, V=12288),
        bind(TL.LogSoftmaxNllSlots, V=12288),
        bind(TL.LogSoftmaxNllBackwardWarp, V=12288),
        bind(TL.NllLossTorch, M=16, V=12288, MEAN=1),
        bind(TL.NllLossTorch, M=16, V=12288, MEAN=0),
        bind(TL.NllLossTorchBackward, MEAN=1),
        bind(TL.NllLossTorchBackward, MEAN=0),
        bind(TL.LogSoftmaxRow, V=12288),
        bind(TL.LogSoftmaxNllBackwardRow, V=12288),
    ]

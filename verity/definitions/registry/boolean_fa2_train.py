"""`fa2_train`'s Definitions on bits: version 2 of each, σ of its word composite (`sigma_lift`), so each one's word view is its
`_v1` at the same statics.

    Fa2TrainFirst_v2{D,BN,NVIS}    Fa2TrainBlock_v2{D,BN}    Fa2TrainRow_v2{D,BN,NB,NVIS}    Fa2TrainHead_v2{T,D,BN}
    Fa2TrainPart_v2{D,BN,NVIS}     Fa2TrainCombine_v2{D}     Fa2TrainSplitRow_v2{D,BN,NS,NVIS}    Fa2TrainSplitHead_v2{T,D,BN}

The word bodies' callees each have a Boolean version: `DotBf16_v3` over `HopperBF16WgmmaDot16_v2`, the f32 arithmetic's and
comparisons', `Ex2Approx_v2`, `NvExpf_v2`, `MufuLg2Ftz_v2` and `NvLogf_v3`.
"""
from __future__ import annotations

from verity.definitions.boolean import sigma_lift
from verity.definitions.registry import fa2_train as W

Fa2TrainFirst = sigma_lift(W.Fa2TrainFirst, 2)
Fa2TrainBlock = sigma_lift(W.Fa2TrainBlock, 2)
Fa2TrainRow = sigma_lift(W.Fa2TrainRow, 2)
Fa2TrainHead = sigma_lift(W.Fa2TrainHead, 2)
Fa2TrainPart = sigma_lift(W.Fa2TrainPart, 2)
Fa2TrainCombine = sigma_lift(W.Fa2TrainCombine, 2)
Fa2TrainSplitRow = sigma_lift(W.Fa2TrainSplitRow, 2)
Fa2TrainSplitHead = sigma_lift(W.Fa2TrainSplitHead, 2)

__all__ = ["Fa2TrainBlock", "Fa2TrainCombine", "Fa2TrainFirst", "Fa2TrainHead", "Fa2TrainPart", "Fa2TrainRow",
           "Fa2TrainSplitHead", "Fa2TrainSplitRow"]

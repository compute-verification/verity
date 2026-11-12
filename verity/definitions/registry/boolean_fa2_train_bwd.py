"""`fa2_train_bwd`'s Definitions on bits: version 2 of each, σ of its word composite (`sigma_lift`), so each one's word view is
its `_v1` at the same statics.

    Fa2TrainBwdPre_v2{D}    Fa2TrainBwdRow_v2{D,BN,NVIS}    Fa2TrainBwdClear_v2{D,BN}    Fa2TrainBwdKey_v2{D,BM}
    Fa2TrainBwdTile_v2{D,BM,BN,OFF}    Fa2TrainBwdDkDv_v2{D}    Fa2TrainBwdDq_v2{D,NP,NS}    Fa2TrainBwdHead_v2{T,D,BM,BN,NS}

The word bodies' callees each have a Boolean version: `DotBf16_v3` over `HopperBF16WgmmaDot16_v2`, the f32 arithmetic's, the
bf16 conversions' and `Ex2Approx_v2`.
"""
from __future__ import annotations

from verity.definitions.boolean import sigma_lift
from verity.definitions.registry import fa2_train_bwd as W

Fa2TrainBwdPre = sigma_lift(W.Fa2TrainBwdPre, 2)
Fa2TrainBwdRow = sigma_lift(W.Fa2TrainBwdRow, 2)
Fa2TrainBwdClear = sigma_lift(W.Fa2TrainBwdClear, 2)
Fa2TrainBwdKey = sigma_lift(W.Fa2TrainBwdKey, 2)
Fa2TrainBwdTile = sigma_lift(W.Fa2TrainBwdTile, 2)
Fa2TrainBwdDkDv = sigma_lift(W.Fa2TrainBwdDkDv, 2)
Fa2TrainBwdDq = sigma_lift(W.Fa2TrainBwdDq, 2)
Fa2TrainBwdHead = sigma_lift(W.Fa2TrainBwdHead, 2)

__all__ = ["Fa2TrainBwdClear", "Fa2TrainBwdDkDv", "Fa2TrainBwdDq", "Fa2TrainBwdHead", "Fa2TrainBwdKey", "Fa2TrainBwdPre",
           "Fa2TrainBwdRow", "Fa2TrainBwdTile"]

"""The fused-MoE router row (`MoeRouterTopK_v1` / `MoeRouterTopKNorm_v1`, one token row) as the replay row kernels evaluate it
(`verity_vllm.program.kernels.rows`: the router rows and the padded block)."""
from __future__ import annotations

import numpy as np


def moe_router_topk(logits: np.ndarray, *, E: int, TOPK: int, VPT: int, renormalize: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """`MoeRouterTopK_v1{E,TOPK,VPT}` (`_moe_C.topk_softmax`, one token row) evaluated with the registry's scalar primitives, statement for
    statement the body of `verity_vllm.program.registry.moe.MoeRouterTopK`: E bf16 logits -> f32 (Bf16ToF32); thread t of E/VPT holds columns
    t*VPT .. t*VPT+VPT-1: thread max (F32Max, ascending), warp butterfly xor T/2 .. 1 (lane 0); `ex_i = NvExpf(x_i - max)` (F32Sub, the
    CUDA math-library expf sequence, MUFU.EX2 from the measured table); thread sum from 0.f (F32Add, ascending), butterfly; `rcp = 1.f / sum`
    (F32Div, IEEE); `p_i = ex_i * rcp` (F32Mul), non-finite -> +0; TOPK rounds: lowest-index strict argmax (F32GtStrict scan from p[0]),
    the winner masked to -10000.f.  `renormalize` (MoeRouterTopKNorm_v1, MOE-04): selected sum ascending from 0.f, denom = sum > 0 ? sum :
    1.f, scale = 1.f / denom (F32Div), w_k = w_k * scale (F32Mul).  Returns (router_w f32 words [TOPK] u32, router_ids i32 words [TOPK] u32)."""
    from verity_vllm.program.registry import moe as RMOE
    from verity_vllm.program.kernels import moe_twins
    from verity_vllm.program.registry import prims as P
    from verity_vllm.program.registry import ref_prims as R

    if E % VPT:
        raise ValueError(f"E = {E} is not a multiple of VPT = {VPT}")
    T = E // VPT
    x = [P.Bf16ToF32.evaluate(int(v)) for v in np.asarray(logits, dtype=np.uint16).reshape(E)]

    def butterfly(op, vals: list[int]) -> int:
        mask = len(vals) // 2
        while mask >= 1:
            vals = [op(vals[t], vals[t ^ mask]) for t in range(len(vals))]
            mask //= 2
        return vals[0]

    tmax = []
    for t in range(T):
        m = x[t * VPT]
        for j in range(1, VPT):
            m = P.F32Max.evaluate(m, x[t * VPT + j])
        tmax.append(m)
    rmax = butterfly(P.F32Max.evaluate, tmax)
    ex = [moe_twins.nv_expf_bits(R.F32Sub.evaluate(x[i], rmax)) for i in range(E)]
    tsum = []
    for t in range(T):
        s = 0
        for j in range(VPT):
            s = P.F32Add.evaluate(s, ex[t * VPT + j])
        tsum.append(s)
    rsum = butterfly(P.F32Add.evaluate, tsum)
    one = 0x3F800000
    rcp = P.F32Div.evaluate(one, rsum)
    p = []
    for i in range(E):
        q = P.F32Mul.evaluate(ex[i], rcp)
        p.append(q if RMOE.F32IsFinite.evaluate(q) else 0)
    neg_big = 0xC61C4000                                                          # -10000.f
    weights, ids = [], []
    for _ in range(TOPK):
        best, best_i = p[0], 0
        for i in range(1, E):
            if R.F32GtStrict.evaluate(p[i], best):
                best, best_i = p[i], i
        weights.append(best)
        ids.append(best_i)
        p = [neg_big if i == best_i else p[i] for i in range(E)]
    if renormalize:
        acc = 0
        for k in range(TOPK):
            acc = P.F32Add.evaluate(acc, weights[k])
        denom = acc if R.F32GtStrict.evaluate(acc, 0) else one
        scale = P.F32Div.evaluate(one, denom)
        weights = [P.F32Mul.evaluate(w, scale) for w in weights]
    return (np.asarray([w & 0xFFFFFFFF for w in weights], dtype=np.uint32), np.asarray([i & 0xFFFFFFFF for i in ids], dtype=np.uint32))


# ---------------------------------------------------------------------------------------------------------------- registration

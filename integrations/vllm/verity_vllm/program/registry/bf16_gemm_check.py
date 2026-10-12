"""numpy models of GLM-5.3's bf16 GEMMs on sm_90, the arithmetic `ll_bf16` and `cublas_bf16` compute, and vLLM d9105ea80's choice
of kernel for the MoE gate (`GateLinear.forward`, `LLBf16Gemm.dispatch`):

* `gate_rows`: x (M, K) bf16 against the gate's W (N, K) -> f32 words (M, N); M <= 16 is `ll_bf16_gemm` (`dispatch_sm90`), past
  it cuBLAS's ``torch.mm(x, W.T, out_dtype=float32)`` (`f32out_rows`);
* `dotprod_rows`: `LlBf16DotprodGemm_v1` per token row: BS lane chains from +0 (FFMA, or FMUL then FADD), each warp's butterfly,
  then the warp sums in order from +0;
* `splitk_rows`: `LlBf16SplitKGemm_v1{DOT=HopperBF16WgmmaDot16_v1}` per token row: each (cluster rank, MMA warp)'s k16 blocks
  ascend, so its chain is `DotBf16` over those columns (`mla_check.dot_chain`), then the warps' and the ranks' ordered sums;
* `f32out_rows`: `GemmF32Out_v1{DOT=HopperBF16WgmmaDot16_v1}`; `linear_rows`: core's `Gemm_v2` (lm_head, the indexer's
  `wk_weights_proj`); `bmm_rows`: `BatchedGemm_v1`, `mla_check.bmm_rows`.

NaN payloads are not part of any comparison (`mla_check._words_equal`): the GPU's FFMA and FADD write 0x7FFFFFFF, numpy keeps
an operand's.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import mla_check as MK

U32 = np.uint32
WARP = 32
#: `LLBf16Gemm`'s defaults and the sm_90 tables at GLM-5.3's gate (K 6144, N 256), vLLM d9105ea80 `cute_dsl/ll_bf16.py`
DOTPROD_MAX_M, DOTPROD_BS, DOTPROD_VEC = 4, 128, 8
SPLITK_DEFAULT, SPLITK_TILE_K, SPLITK_WARPS = 6, 256, 4
SM90_DOTPROD_BS = {(4096, 256): {1: 256, 3: 256}, (7168, 384): {1: 256, 2: 256}}
SM90_SPLITK = {(4096, 256): {**{m: 8 for m in range(5, 8)}, 10: 8, 11: 8, 13: 8, 14: 8, 15: 8, 16: 8},
               (7168, 256): {8: 6},
               (6144, 256): {9: 8, 11: 8},
               (7168, 384): {m: (8 if m in (6, 8, 12) else 7) for m in range(6, 17)}}
LL_MAX_M = 16


def dispatch_sm90(M: int, K: int, N: int) -> tuple[str, dict]:
    """The kernel `GateLinear` runs for a bf16 gate on sm_90 at M tokens: ``("dotprod", {BS, VEC})``, ``("splitk", {SPLIT_K,
    TILE_K, WARPS})`` (`ll_bf16_gemm`), or ``("cublas", {})`` (tier 4).  The split-K tables' second entry, ``num_stages``, only
    pipelines the loads."""
    if M <= 0:
        raise ValueError(f"M = {M}: no token rows")
    if M > LL_MAX_M:
        return "cublas", {}
    if M <= DOTPROD_MAX_M or K < 2048:
        return "dotprod", {"BS": SM90_DOTPROD_BS.get((K, N), {}).get(M, DOTPROD_BS), "VEC": DOTPROD_VEC}
    return "splitk", {"SPLIT_K": SM90_SPLITK.get((K, N), {}).get(M, SPLITK_DEFAULT), "TILE_K": SPLITK_TILE_K,
                      "WARPS": SPLITK_WARPS}


def _ordered_sum(parts) -> np.ndarray:
    out = np.zeros(np.shape(parts[0]), U32)
    for p in parts:
        out = MK.add(out, p)
    return out


def dotprod_rows(x: np.ndarray, w: np.ndarray, BS: int = DOTPROD_BS, VEC: int = DOTPROD_VEC, FMA: int = 1) -> np.ndarray:
    """`LlBf16DotprodGemm_v1{N, K, BS, VEC, FMA}` of every row of x (M, K) against w (N, K): f32 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    (M, K), N, tile = x.shape, w.shape[0], VEC * BS
    if K % tile or BS % WARP or w.shape[1] != K:
        raise ValueError(f"dotprod: K {K} a multiple of VEC BS {tile}, BS {BS} of 32 and W [N, K] (W {w.shape})")
    xf = DR.bf16_to_f32_words(x).reshape(M, K // tile, BS, VEC)
    wf = DR.bf16_to_f32_words(w).reshape(N, K // tile, BS, VEC)
    acc = np.zeros((M, N, BS), U32)
    for i in range(K // tile):
        for v in range(VEC):
            a, b = xf[:, None, i, :, v], wf[None, :, i, :, v]
            acc = MK.fma(a, b, acc) if FMA else MK.add(acc, MK.mul(a, b))
    lanes = acc.reshape(M, N, BS // WARP, WARP)
    o = WARP // 2
    while o:
        lanes = MK.add(lanes[..., :o], lanes[..., o:2 * o])
        o //= 2
    return _ordered_sum([lanes[:, :, j, 0] for j in range(BS // WARP)])


def splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int, TILE_K: int = SPLITK_TILE_K, WARPS: int = SPLITK_WARPS) -> np.ndarray:
    """`LlBf16SplitKGemm_v1{N, K, SPLIT_K, TILE_K, WARPS, HopperBF16WgmmaDot16_v1}` of every row of x (M, K) against w (N, K)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    K = x.shape[1]
    tiles, blocks = K // TILE_K, TILE_K // 16
    if K % TILE_K or TILE_K % 16 or blocks % WARPS or not 0 < SPLIT_K <= tiles or w.shape[1] != K:
        raise ValueError(f"splitk: K {K}, TILE_K {TILE_K}, WARPS {WARPS}, SPLIT_K {SPLIT_K} (W {w.shape})")
    ranks = []
    for z in range(SPLIT_K):
        parts = []
        for j in range(WARPS):
            cols = np.concatenate([np.arange(16) + TILE_K * t + 16 * (j + WARPS * ki)
                                   for t in range(z, tiles, SPLIT_K) for ki in range(blocks // WARPS)])
            parts.append(MK.dot_chain(x[:, cols], w[:, cols]))
        ranks.append(_ordered_sum(parts))
    return _ordered_sum(ranks)


def f32out_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`GemmF32Out_v1{K, N, HopperBF16WgmmaDot16_v1}`: x (M, K), w (N, K) -> f32 words (M, N)."""
    return MK.dot_chain(x, w)


def linear_rows(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """core's `Gemm_v2{K, N, HopperBF16WgmmaDot16_v1}`: x (M, K), w (N, K) -> bf16 words (M, N)."""
    return MK.bf16_of(MK.dot_chain(x, w))


def bmm_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`BatchedGemm_v1{B, K, N, HopperBF16WgmmaDot16_v1}`: a (B, M, K), b (B, K, N) -> bf16 words (B, M, N)."""
    return MK.bmm_rows(a, b)


def gate_rows(x: np.ndarray, w: np.ndarray, FMA: int = 1) -> np.ndarray:
    """GLM-5.3's MoE gate on sm_90: the kernel `dispatch_sm90` names at x's M, as the Definition that is its function."""
    kind, st = dispatch_sm90(x.shape[0], x.shape[1], w.shape[0])
    if kind == "dotprod":
        return dotprod_rows(x, w, FMA=FMA, **st)
    if kind == "splitk":
        return splitk_rows(x, w, **st)
    return f32out_rows(x, w)

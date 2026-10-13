"""numpy models of GLM-5.3's and DeepSeek-V4-Pro's bf16 GEMMs on sm_90, the arithmetic `ll_bf16` and `cublas_bf16` compute, and
vLLM d9105ea80's choice of kernel for the MoE gate (`GateLinear.forward`, `LLBf16Gemm.dispatch`):

* `gate_rows`: x (M, K) bf16 against the gate's W (N, K) -> f32 words (M, N); M <= 16 is `ll_bf16_gemm` (`dispatch_sm90`), past
  it cuBLAS's ``torch.mm(x, W.T, out_dtype=float32)`` (`f32out_rows`);
* `dotprod_rows`: `LlBf16DotprodGemm_v1` per token row: BS lane chains from +0 (FFMA, or FMUL then FADD) over `dotprod_rounds`,
  each warp's butterfly, then the warp sums in order from +0;
* `splitk_rows`: `LlBf16SplitKGemm_v1{DOT=HopperBF16WgmmaDot16_v1}` per token row: each (cluster rank, MMA warp)'s k16 blocks
  ascend, so its chain is `DotBf16` over those columns (`mla_check.dot_chain`), then the warps' and the ranks' ordered sums;
* `f32out_rows`: `GemmF32Out_v1{DOT=HopperBF16WgmmaDot16_v1}` (also DeepSeek-V4-Pro's compressor GEMMs); `linear_rows`: core's
  `Gemm_v2` (lm_head, GLM-5.3's indexer `wk_weights_proj`, DeepSeek-V4-Pro's `weights_proj`); `bmm_rows`: `BatchedGemm_v1`,
  `mla_check.bmm_rows`;
* `cublas_splitk_rows`: cuBLAS 12.9.1's H200 route in `cublas_bf16`: equal 64-element-aligned contiguous chunks, a final
  remainder, f32 partials and a serial increasing-order f32 reduction. `default_cublas_split_k` pins the selected split count.

NaN payloads are not part of any comparison (`mla_check._words_equal`): the GPU's FFMA and FADD write 0x7FFFFFFF, numpy keeps
an operand's.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import mla_check as MK

U32 = np.uint32
WARP = 32
#: `LLBf16Gemm`'s defaults and its sm_90 tables by gate (K, N), vLLM d9105ea80 `cute_dsl/ll_bf16.py`
DOTPROD_MAX_M, DOTPROD_BS, DOTPROD_VEC = 4, 128, 8
SPLITK_DEFAULT, SPLITK_TILE_K, SPLITK_WARPS = 6, 256, 4
SM90_DOTPROD_BS = {(4096, 256): {1: 256, 3: 256}, (7168, 384): {1: 256, 2: 256}}
SM90_SPLITK = {(4096, 256): {**{m: 8 for m in range(5, 8)}, 10: 8, 11: 8, 13: 8, 14: 8, 15: 8, 16: 8},
               (7168, 256): {8: 6},
               (6144, 256): {9: 8, 11: 8},
               (7168, 384): {m: (8 if m in (6, 8, 12) else 7) for m in range(6, 17)}}
LL_MAX_M = 16
DEFAULT_CUBLAS_SPLIT_K = {
    ("router", 6144, 256): {17: 4, 32: 4, 128: 4, 256: 3},
    ("router", 7168, 384): {17: 5, 32: 5, 128: 4, 256: 4},
    ("linear", 6144, 160): {m: 4 for m in (1, 4, 16, 128, 256)},
    ("linear", 7168, 64): {m: 5 for m in (1, 4, 16, 128, 256)},
    ("f32out", 7168, 2048): {1: 3, 4: 3, 16: 3, 128: 2},
    ("f32out", 7168, 1024): {1: 5, 4: 5, 16: 5, 128: 3, 256: 2},
    ("f32out", 7168, 512): {1: 5, 4: 5, 16: 5, 128: 3, 256: 3},
}


def default_cublas_split_k(phase: str, M: int, K: int, N: int) -> int | None:
    """The split count selected by cuBLAS 12.9.1 on the pinned H200, or None for its non-split routes."""
    return DEFAULT_CUBLAS_SPLIT_K.get((phase, K, N), {}).get(M)


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


def _reduce(parts: list[np.ndarray], order: str) -> np.ndarray:
    """F32-add partials in a hypothesized reduction order."""
    if order == "forward":
        return _ordered_sum(parts)
    if order == "reverse":
        return _ordered_sum(parts[::-1])
    if order == "tree":
        level = list(parts)
        while len(level) > 1:
            level = [MK.add(level[i], level[i + 1]) if i + 1 < len(level) else level[i] for i in range(0, len(level), 2)]
        return level[0]
    raise ValueError(f"unknown reduction order {order!r}")


def _split_cols(K: int, splits: int, partition: str) -> list[np.ndarray]:
    """k16-block columns under the contiguous or round-robin partition hypothesis."""
    if K <= 0 or K % 16 or not 1 < splits <= K // 16:
        raise ValueError(f"cublas split-k: K {K}, splits {splits}")
    blocks = K // 16
    if partition == "contiguous":
        chunk = ((K + splits - 1) // splits + 63) // 64 * 64
        bounds = list(range(0, K, chunk)) + [K]
        if len(bounds) != splits + 1:
            bounds = [16 * (z * blocks // splits) for z in range(splits)] + [K]
        return [np.arange(lo, hi) for lo, hi in zip(bounds, bounds[1:])]
    if partition == "strided":
        return [np.concatenate([np.arange(16 * b, 16 * b + 16) for b in range(z, blocks, splits)]) for z in range(splits)]
    raise ValueError(f"unknown split partition {partition!r}")


def cublas_splitk_rows(x: np.ndarray, w: np.ndarray, SPLIT_K: int, *, partition: str = "contiguous",
                       partial: str = "f32", reduction: str = "forward", out: str = "f32") -> np.ndarray:
    """One cuBLAS split-K hypothesis: k16 chains from +0, then a reduction of the partials and the requested output rounding."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError(f"cublas split-k: x {x.shape}, W {w.shape}")
    parts = [MK.dot_chain(x[:, cols], w[:, cols]) for cols in _split_cols(x.shape[1], SPLIT_K, partition)]
    if partial == "bf16":
        parts = [DR.bf16_to_f32_words(MK.bf16_of(p)) for p in parts]
    elif partial != "f32":
        raise ValueError(f"unknown partial precision {partial!r}")
    result = _reduce(parts, reduction)
    if out == "bf16":
        return MK.bf16_of(result)
    if out != "f32":
        raise ValueError(f"unknown output precision {out!r}")
    return result


CUBLAS_SPLITK_HYPOTHESIS = "contiguous-f32-forward"
CUBLAS_SPLITK_ALTERNATIVES = (
    CUBLAS_SPLITK_HYPOTHESIS,
    "strided-f32-forward",
    "contiguous-f32-reverse",
    "contiguous-f32-tree",
    "contiguous-bf16-forward",
)


def cublas_splitk_hypotheses(x: np.ndarray, w: np.ndarray, SPLIT_K: int, *, out: str) -> dict[str, np.ndarray]:
    """The capture probe's split partition, partial precision and reduction-order alternatives."""
    ans = {}
    for name in CUBLAS_SPLITK_ALTERNATIVES:
        partition, partial, reduction = name.split("-")
        ans[name] = cublas_splitk_rows(x, w, SPLIT_K, partition=partition, partial=partial, reduction=reduction, out=out)
    return ans


def dotprod_rounds(K: int, BS: int, VEC: int) -> list[np.ndarray]:
    """`LLBf16Dotprod`'s K loop as rounds, each the k every thread adds then (-1: none): the vector loads at width VEC over K's
    whole tiles of VEC BS, then at width VEC/2 over what is left (its 64-bit tail at VEC 8), then scalar rounds of BS, the last
    ragged."""
    t, rounds, k0 = np.arange(BS), [], 0
    for width in (VEC, VEC // 2):
        tiles = (K - k0) // (width * BS) if width else 0
        rounds += [k0 + width * (BS * i + t) + v for i in range(tiles) for v in range(width)]
        k0 += tiles * width * BS
    rounds += [np.where(s + t < K, s + t, -1) for s in range(k0, K, BS)]
    return rounds


def dotprod_rows(x: np.ndarray, w: np.ndarray, BS: int = DOTPROD_BS, VEC: int = DOTPROD_VEC, FMA: int = 1) -> np.ndarray:
    """`LlBf16DotprodGemm_v1{N, K, BS, VEC, FMA}` of every row of x (M, K) against w (N, K): f32 words (M, N)."""
    x, w = np.asarray(x, np.uint16), np.asarray(w, np.uint16)
    (M, K), N = x.shape, w.shape[0]
    if K < BS or BS % WARP or VEC < 1 or w.shape[1] != K:
        raise ValueError(f"dotprod: K {K} at least BS {BS}, BS of 32, VEC {VEC} >= 1 and W [N, K] (W {w.shape})")
    xf, wf = DR.bf16_to_f32_words(x), DR.bf16_to_f32_words(w)
    acc = np.zeros((M, N, BS), U32)
    for ks in dotprod_rounds(K, BS, VEC):
        a, b = xf[:, None, np.maximum(ks, 0)], wf[None, :, np.maximum(ks, 0)]
        acc = np.where(ks >= 0, MK.fma(a, b, acc) if FMA else MK.add(acc, MK.mul(a, b)), acc)
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
    """A bf16 MoE gate on sm_90: the kernel `dispatch_sm90` names at x's M and the gate's (K, N), as the Definition that is its
    function."""
    kind, st = dispatch_sm90(x.shape[0], x.shape[1], w.shape[0])
    if kind == "dotprod":
        return dotprod_rows(x, w, FMA=FMA, **st)
    if kind == "splitk":
        return splitk_rows(x, w, **st)
    return f32out_rows(x, w)

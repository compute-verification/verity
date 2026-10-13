"""numpy models of GLM-5.3's TP8 all-reduces on sm_90, the arithmetic `tp_allreduce` states, over many tokens at once:

* `mnnvl_rows`: FlashInfer's MNNVL all-reduce (one- or two-shot) of (WORLD, ..., N) bf16 words;
* `mnnvl_norm_rows`: its fused residual add + RMSNorm, (WORLD, M, N) partials with (M, N) residuals and an N weight;
* `custom_rows`: vLLM's custom all-reduce (1- or 2-stage) of (WORLD, SIZE) bf16 words;
* `multimem_rows`: torch's multimem all-reduce of (WORLD, SIZE) bf16 words at an owner's tree each.

Written from the kernels (FlashInfer 0.6.18 `trtllm_mnnvl_allreduce.cuh`, vLLM d9105ea80 `csrc/custom_all_reduce.cuh`, torch 2.13
`CUDASymmetricMemoryOps.cu`), not from the Definitions, so that a test comparing the two compares two statements of the kernels.
NaN payloads never reach an output: every output is F2FP's bf16.
"""
from __future__ import annotations

import numpy as np

from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import tp_allreduce as T

U32 = np.uint32


def _widen(x: np.ndarray) -> np.ndarray:
    return DR.bf16_to_f32_words(x)


def _sanitize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint16)
    return np.where(x == 0x8000, np.uint16(0), x)


def mnnvl_sum(parts: np.ndarray, shot: int) -> np.ndarray:
    """f32 words of the MNNVL sum (before F2FP): +0, then ranks ascending, FADD.FTZ; two-shot once more onto +0."""
    zero = np.zeros(parts.shape[1:], U32)
    acc = zero
    for r in range(parts.shape[0]):
        acc = DR.add_ftz(acc, _widen(_sanitize(parts[r])))
    return DR.add_ftz(zero, acc) if shot == 2 else acc


def mnnvl_rows(parts: np.ndarray, shot: int) -> np.ndarray:
    return DR.f2fp_bf16(mnnvl_sum(np.asarray(parts, np.uint16), shot))


def _butterfly0(v: np.ndarray) -> np.ndarray:
    """Lane 0 of the xor butterfly over the last axis (32 lanes)."""
    m = T.WARP // 2
    while m:
        v = DR.add_ftz(v[..., :m], v[..., m:2 * m])
        m //= 2
    return v[..., 0]


def mnnvl_norm_rows(parts: np.ndarray, res: np.ndarray, w: np.ndarray, shot: int, cluster: int, eps: float,
                    fma: bool) -> tuple[np.ndarray, np.ndarray]:
    """(normed, residual') bf16 words (M, N) of `trtllm_allreduce_fusion(kARResidualRMSNorm)` on the MNNVL backend."""
    parts, res, w = np.asarray(parts, np.uint16), np.asarray(res, np.uint16), np.asarray(w, np.uint16)
    M, N = res.shape
    block = N // (8 * cluster)
    a = mnnvl_rows(parts, shot)
    r = DR.f2fp_bf16(DR.add_rn(_widen(a), _widen(res)))
    rf = _widen(r)
    if shot == 1:
        sq = _widen(DR.f2fp_bf16(DR.mul_rn(rf, rf))).reshape(M, cluster, block, 8)
        acc = np.zeros(sq.shape[:-1], U32)
        for i in range(8):
            acc = DR.add_ftz(acc, sq[..., i])
    else:
        v = rf.reshape(M, cluster, block, 8)
        acc = np.zeros(v.shape[:-1], U32)
        for i in range(8):
            acc = DR.fma_ftz(v[..., i], v[..., i], acc)
    warps = block // T.WARP
    warp_sums = _butterfly0(acc.reshape(M, cluster, warps, T.WARP))
    padded = np.concatenate([warp_sums, np.zeros((M, cluster, T.WARP - warps), U32)], axis=-1)
    sums = _butterfly0(padded)
    total = sums[:, 0]
    if cluster > 1:
        total = np.zeros(M, U32)
        for b in range(cluster):
            total = DR.add_ftz(total, sums[:, b])
    rcp = DR.mufu_rcp_ftz(np.array([np.float32(N)]).view(U32))[0]
    e = U32(T.eps_bits(eps))
    arg = DR.fma_ftz(total, rcp, e) if fma else DR.add_ftz(DR.mul_ftz(total, rcp), e)
    rs = DR.rsqrt_approx(arg)[:, None]
    g = DR.add_ftz(np.zeros(N, U32), _widen(w))[None, :]
    y = DR.mul_ftz(DR.mul_ftz(rf, rs), g) if shot == 1 else DR.mul_ftz(DR.mul_ftz(g, rf), rs)
    return DR.f2fp_bf16(y), r


def custom_rows(parts: np.ndarray, stage: int) -> np.ndarray:
    """vLLM's custom all-reduce of (WORLD, SIZE) bf16 words: IEEE f32 from each element's owner rank, F2FP."""
    parts = np.asarray(parts, np.uint16)
    W, size = parts.shape
    out = np.empty(size, np.uint16)
    for lo, hi, owner in T.custom_owners(W, size, stage):
        order = T.custom_order(W, owner)
        acc = _widen(parts[order[0], lo:hi])
        for r in order[1:]:
            acc = DR.add_rn(acc, _widen(parts[r, lo:hi]))
        out[lo:hi] = DR.f2fp_bf16(acc)
    return out


def tree_sum(cols: np.ndarray, tree, ftz: bool) -> np.ndarray:
    """f32 words of `tree`'s adds over the (WORLD, ...) bf16 words `cols`."""
    add = DR.add_ftz if ftz else DR.add_rn
    if isinstance(tree, int):
        return _widen(cols[tree])
    return add(tree_sum(cols, tree[0], ftz), tree_sum(cols, tree[1], ftz))


def multimem_rows(parts: np.ndarray, trees, ftz: bool) -> np.ndarray:
    """torch's multimem all-reduce of (WORLD, SIZE) bf16 words: each owner's slice (`multimem_owners`) the f32 adds of its tree,
    RN to bf16."""
    parts = np.asarray(parts, np.uint16)
    W, size = parts.shape
    trees = T.multimem_trees(trees, W)
    out = np.empty(size, np.uint16)
    for lo, hi, owner in T.multimem_owners(W, size):
        out[lo:hi] = DR.f2fp_bf16(tree_sum(parts[:, lo:hi], trees[owner], ftz))
    return out

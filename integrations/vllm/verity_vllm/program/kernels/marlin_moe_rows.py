"""The Marlin W4A16 MXFP4 expert GEMMs' twin (`registry.marlin_moe`: `MarlinMxfp4Coordinate(W)_v1` and the schedule
`MarlinMoeSchedule_v1`), vectorised over a launch's (pair, column) coordinates with core's k16 step; `moe_block`, the moe block
`fused_marlin_moe` picks for a batch; and `auto_config`, ops.cu's choice of thread config and blocks per SM (`determine_exec_config`)
at that block over the per-config limits a driver's register count sets.  The scalar Definitions are the reference:
tests/program/test_marlin_moe.py checks this twin against them word for word, and the H200 capture
(tests/properties/dsv4_marlin_capture_gpu.py) checks the kernel's words against it."""
from __future__ import annotations

import numpy as np

from verity.ml.kernels.tc import group_sum_total_batch
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
from verity_vllm.program.kernels.derived_rows import add_rn, f2fp_bf16, mul_rn
from verity_vllm.program.registry import marlin_moe as MM

E2M1_BF16 = np.array([MM.e2m1_to_bf16(q) for q in range(16)], dtype=np.uint16)

#: ops.cu's `small_batch_thread_configs` and `large_batch_thread_configs` (thread_k, thread_n, num_threads), by priority
#: (ops.cu:131-141): a moe block of 8 or 16 has thread_m_blocks 1 and takes the small ones, a larger block the large ones (285-292)
SMALL_BATCH_CONFIGS = ((128, 128, 256), (64, 128, 128), (128, 64, 128))
LARGE_BATCH_CONFIGS = ((64, 256, 256), (64, 128, 128), (128, 64, 128))
#: `fused_marlin_moe`'s moe block candidates (vLLM d9105ea80 experts/marlin_moe.py:334)
MOE_BLOCKS = (8, 16, 32, 48, 64)


def moe_block(M: int, top_k: int, E: int) -> int:
    """The moe block `fused_marlin_moe` gives M tokens over E local experts with bf16 activations (marlin_moe.py:332-339, no expert
    map): the first of MOE_BLOCKS at which M * top_k / E / block < 0.9, in its float arithmetic, else the last."""
    for b in MOE_BLOCKS:
        if M * top_k / E / b < 0.9:
            return b
    return MOE_BLOCKS[-1]


def thread_m_blocks(block: int) -> int:
    """The template's m blocks of 16 for a moe block (ops.cu:362, div_ceil(moe_block_size, 16)); 8 is one, `m_block_size_8`."""
    return -(-block // 16)


def thread_configs(block: int) -> tuple[tuple[int, int, int], ...]:
    return SMALL_BATCH_CONFIGS if thread_m_blocks(block) == 1 else LARGE_BATCH_CONFIGS


def max_blocks_per_sm(block: int) -> int:
    """The clamp on `determine_exec_config`'s per-config allow count (ops.cu:325-328): 4 at one m block, 2 at more."""
    return 4 if thread_m_blocks(block) == 1 else 2


def dequant(wq: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """`MarlinMxfp4Group_v1` over the last axis: codes (..., K) and scale bytes (..., K / 32) -> bf16 words (..., K)."""
    a = E2M1_BF16[np.asarray(wq, dtype=np.uint8) & 0xF].astype(np.uint32) << 16
    b = np.repeat(np.asarray(ws, dtype=np.uint32) << 23, MM.GROUP, axis=-1)
    return f2fp_bf16(mul_rn(a, b))


def schedule(KT: int, NT: int, GRID: int, blk: int, par: int) -> np.ndarray:
    """`MarlinMoeSchedule_v1`: (NT, KT - 1) bools, C `int` arithmetic as the template's (`marlin_moe.i32_divrem`)."""
    MM.check_schedule(KT, NT, GRID)
    M32 = MM.M32

    def s(v: int) -> int:
        v &= M32
        return v - (1 << 32) if v >> 31 else v

    def rem(a: int, b: int) -> int:
        return s(MM.i32_divrem(a, b)[1])

    G = s(par * NT)
    p2 = G
    if G > GRID:
        p2 = rem(G, GRID)
        if s(p2 * 3) <= GRID:
            p2 = s(p2 + GRID)
    iters = s(MM.i32_divrem(s(p2 * KT + GRID - 1), GRID)[0])
    first = s(G - p2)
    out = np.zeros((NT, KT - 1), dtype=bool)
    for J in range(NT):
        g = s(blk * NT + J)
        if first > g:
            continue
        r = rem(s(s(g - first) * KT), iters)
        for t in range(1, KT):
            r = s(r + 1)
            if r == iters:
                r, out[J, t - 1] = 0, True
    return out


def _warp_tree(acc: list[np.ndarray]) -> np.ndarray:
    return MM.warp_tree(add_rn, acc)


def coordinates(x: np.ndarray, cut: np.ndarray, wq: np.ndarray, ws: np.ndarray, *, TK: int, g: np.ndarray | None = None) -> np.ndarray:
    """`MarlinMxfp4Coordinate_v1` (``g`` None) or `MarlinMxfp4CoordinateW_v1` on n coordinates at once: x (n, K) bf16 words, cut
    (n, K / TK - 1) bools, wq (n, K) codes, ws (n, K / 32) scale bytes, g (n,) the routed weights as bf16 words -> (n,) bf16 words."""
    x = np.asarray(x, dtype=np.uint16)
    n, K = x.shape
    MM.check_statics(K, TK)
    W, KT = TK // 32, K // TK
    w = dequant(wq, ws)
    zero = np.zeros(n, dtype=np.uint32)
    acc = [zero] * W
    trees = []
    for t in range(KT):
        if t:
            acc = [np.where(cut[:, t - 1], zero, a) for a in acc]
        for r in range(W):
            for j in range(2):
                k = t * TK + 32 * r + 16 * j
                acc[r] = np.asarray(group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc[r], x[:, k:k + 16], w[:, k:k + 16])[0],
                                    dtype=np.uint32)
        trees.append(_warp_tree(acc))
    R = trees[-1]
    for t in reversed(range(KT - 1)):
        R = np.where(cut[:, t], add_rn(trees[t], R), R)
    y = f2fp_bf16(R)
    if g is None:
        return y
    return f2fp_bf16(mul_rn(y.astype(np.uint32) << 16, np.asarray(g, dtype=np.uint32) << 16))


def launch(x: np.ndarray, blk: np.ndarray, par: int, e: np.ndarray, wq: np.ndarray, ws: np.ndarray, cols: np.ndarray, *, TK: int,
           TN: int, GRID: int, g: np.ndarray | None = None) -> np.ndarray:
    """One launch's words at the columns ``cols`` of every pair: x (P, K) the pairs' activation rows, blk (P,) their moe blocks
    (sorted position / the moe block), par the launch's num_tokens_post_padded / the moe block, e (P,) their experts, wq (E, N, K)
    codes, ws (E, N, K / 32) scale bytes, g (P,) f32 routed weights for the w2 launch -> (P, len(cols)) bf16 words; the cuts are
    `schedule`'s at (K / TK, N / TN, GRID)."""
    P, K = x.shape
    N = wq.shape[1]
    chunk = max(1, (1 << 24) // K)
    KT, NT = K // TK, N // TN
    if N % TN:
        raise ValueError(f"N = {N} is not a whole number of TN = {TN} tile columns")
    cols = np.asarray(cols, dtype=np.int64)
    cuts = {b: schedule(KT, NT, GRID, int(b), int(par)) for b in np.unique(blk)}
    gb = None if g is None else f2fp_bf16(np.asarray(g, dtype=np.uint32))
    pi, ci = np.meshgrid(np.arange(P), np.arange(len(cols)), indexing="ij")
    pi, ci = pi.ravel(), ci.ravel()
    out = np.empty(P * len(cols), dtype=np.uint16)
    for lo in range(0, len(pi), chunk):
        p, c = pi[lo:lo + chunk], ci[lo:lo + chunk]
        n = cols[c]
        cut = np.stack([cuts[int(blk[i])][j] for i, j in zip(p, n // TN)]) if KT > 1 else np.zeros((len(p), 0), dtype=bool)
        out[lo:lo + chunk] = coordinates(x[p], cut, wq[e[p], n], ws[e[p], n], TK=TK, g=None if gb is None else gb[p])
    return out.reshape(P, len(cols))


def auto_config(prob_m: int, prob_n: int, top_k: int, sms: int, allow: dict[tuple[int, int], int], *,
                block: int) -> tuple[int, int, int]:
    """`determine_exec_config` at a moe block of ``block`` (ops.cu:274-342): (TK, TN, blocks_per_sm) over `thread_configs(block)`,
    each config's ``allow`` its min(255 * 1024 / (numRegs * threads * 4), max_shared / (cache + 1536)) at that block's kernel,
    clamped to [1, `max_blocks_per_sm(block)`] (a config absent from it is invalid here), lowered to
    max(prob_n / TN * prob_m * top_k * 4 / sms, 1) when that count is under sms * allow; the first config with the strictly largest
    count wins."""
    best, cfg = 0, None
    cap = max_blocks_per_sm(block)
    for tk, tn, _threads in thread_configs(block):
        if (tk, tn) not in allow:
            continue
        a = max(min(allow[(tk, tn)], cap), 1)
        work = prob_n // tn * prob_m * top_k * 4
        if work < sms * a:
            a = max(work // sms, 1)
        if a > best:
            best, cfg = a, (tk, tn, a)
    if cfg is None:
        raise ValueError("no valid thread config")
    return cfg

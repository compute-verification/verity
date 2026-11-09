"""The Marlin W4A16 MXFP4 expert GEMMs' twin (`registry.marlin_moe`: `MarlinMxfp4Coordinate(W)_v1` and the schedule
`MarlinMoeSchedule_v1`), vectorised over a launch's (pair, column) coordinates with core's k16 step; `moe_block`, the moe block
`fused_marlin_moe` picks for a batch; `auto_config`, ops.cu's choice of thread config and blocks per SM (`determine_exec_config`)
at that block over the per-config limits a driver's register count sets; and `served_config`, the config a launch runs at a batch
by a capture's sweep, where the sweep holds it or determines it under that rule.  The scalar Definitions are the reference:
tests/program/test_marlin_moe.py checks this twin against them word for word, and the H200 capture
(tests/properties/dsv4_marlin_capture_gpu.py) checks the kernel's words against it.  `expert_gemm` is the replay row kernel of
`MarlinMoeExpertGemm(W)_v1`, one pair by `launch`; it registers from `rows` (kernel "rows")."""
from __future__ import annotations

import itertools
from functools import partial
from typing import Any, Callable, Sequence

import numpy as np

from verity.ml.circuits.evaluation.batch import type_dtype
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
from verity.ml.kernels.tc import group_sum_total_batch
from verity.ml.definitions.numerics.derived_rows import add_rn, f2fp_bf16, mul_rn
from verity.ml.definitions.numerics.kernel_registry import Declined, prim_static
from verity.ml.definitions.registry import marlin_moe as MM

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


def pack_e2m1(wq: np.ndarray) -> np.ndarray:
    """Codes (..., K) as the checkpoint stores them and the Definitions read them (..., K / 2): two a byte, the even k in the low
    nibble."""
    q = np.asarray(wq, dtype=np.uint8) & 0xF
    return q[..., 0::2] | q[..., 1::2] << 4


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


EVALUATORS = {
    "MarlinMoeExpertGemm_v1": "verity.ml.definitions.numerics.marlin_moe_rows.expert_gemm (`launch` on the pair: MarlinMoeSchedule_v1's cuts, "
                              "MarlinMxfp4Coordinate_v1 per column over expert e's e2m1 codes and e8m0 bytes, DOT's k16 step) -> N bf16",
    "MarlinMoeExpertGemmW_v1": "verity.ml.definitions.numerics.marlin_moe_rows.expert_gemm (as MarlinMoeExpertGemm_v1, MarlinMxfp4CoordinateW_v1: "
                               "the routed f32 weight rounded to bf16 once, times every word) -> N bf16",
}
#: every word of a pair whose expert is outside [0, E): its gathers' fills dequantise to NaN (0 * inf), F2FP's NaN
NAN_BF16 = 0x7FFF


def unpack_e2m1(packed: np.ndarray) -> np.ndarray:
    """`pack_e2m1`'s inverse: bytes (..., K / 2) -> codes (..., K), the even k from the low nibble."""
    b = np.asarray(packed, dtype=np.uint8)
    q = np.empty(b.shape[:-1] + (2 * b.shape[-1],), dtype=np.uint8)
    q[..., 0::2], q[..., 1::2] = b & 0xF, b >> 4
    return q


def _i32(op: np.ndarray) -> int:
    v = int(np.asarray(op).reshape(-1)[0]) & MM.M32
    return v - (1 << 32) if v >> 31 else v


def expert_gemm(statics: dict[str, Any], ops: Sequence[np.ndarray], *, routed: bool) -> dict[str, np.ndarray]:
    """`MarlinMoeExpertGemm_v1{E,K,N,TK,TN,GRID,DOT}(x, e, blk, par, wq, ws)` or, `routed`, `MarlinMoeExpertGemmW_v1(x, e, g, blk, par,
    wq, ws)`: the pair's N words by `launch` over expert e's [N][K/2] packed codes and [N][K/32] scale bytes; an expert outside [0, E)
    reads the gathers' fills, NaN in every word.  A DOT other than the twin's step declines by name."""
    fam = "MarlinMoeExpertGemmW_v1" if routed else "MarlinMoeExpertGemm_v1"
    dot = prim_static(statics, "DOT", fam)
    if dot != MM.DOT.id:
        raise Declined(f"no registered evaluator for {fam} with DOT={dot} (the twin steps {MM.DOT.id})")
    E, K, N, TK, TN, GRID = (int(statics[k]) for k in ("E", "K", "N", "TK", "TN", "GRID"))
    o = 3 if routed else 2
    e, blk, par = _i32(ops[1]), _i32(ops[o]), _i32(ops[o + 1])
    if not 0 <= e < E:
        return {"out": np.full(N, NAN_BF16, dtype=np.uint16)}
    wq = np.asarray(ops[o + 2], dtype=np.uint8).reshape(E, N, K // 2)
    ws = np.asarray(ops[o + 3], dtype=np.uint8).reshape(E, N, K // MM.GROUP)
    g = np.asarray(ops[2], dtype=np.uint32).reshape(1) if routed else None
    y = launch(np.asarray(ops[0], dtype=np.uint16).reshape(1, K), np.array([blk]), par, np.zeros(1, dtype=np.int64),
               unpack_e2m1(wq[e])[None], ws[e][None], np.arange(N), TK=TK, TN=TN, GRID=GRID, g=g)
    return {"out": y.reshape(N)}


def _sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
    """Pairs whose words stay finite: bf16 rows at a scale per row, experts in range, a block below its count, every code byte, scale
    bytes near 2^0, and (W) a routed weight in [0.05, 1)."""
    E, K, N = (int(statics[k]) for k in ("E", "K", "N"))
    x = ((rng.standard_normal((n, K)).astype(np.float32) * np.exp2(rng.integers(-4, 4, (n, 1))).astype(np.float32)).view(np.uint32) >> 16)
    par = rng.integers(1, 6, n)
    g = [rng.uniform(0.05, 1.0, n).astype(np.float32).view(np.uint32)] if len(params) == 7 else []
    out = [x, rng.integers(0, E, n), *g, (rng.random(n) * par).astype(np.int64), par, rng.integers(0, 256, (n, E, N, K // 2)),
           rng.integers(118, 130, (n, E, N, K // MM.GROUP))]
    return [np.asarray(a).astype(type_dtype(pt)) for a, pt in zip(out, params)]


#: family -> (row kernel, the members it returns in the Definition's return order, the Definition)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...], Any]] = {
    "MarlinMoeExpertGemm_v1": (partial(expert_gemm, routed=False), ("out",), MM.MarlinMoeExpertGemm),
    "MarlinMoeExpertGemmW_v1": (partial(expert_gemm, routed=True), ("out",), MM.MarlinMoeExpertGemmW),
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds: three experts, two tile columns, two and four k tiles (TK 128, 64)."""
    from verity.ml.circuits.defs import bind
    return [bind(ROWS[family][2], E=3, K=256, N=16, TK=tk, TN=8, GRID=5, DOT=MM.DOT) for tk in MM.TKS]


def register_rows(evaluators: dict, row: Callable) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    for family, (fn, returns, _d) in ROWS.items():
        row(family, fn, returns, sample=_sample, check_targets=lambda rng, f=family: check_targets(f))


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


def _auto(prob_m: int, prob_n: int, top_k: int, sms: int, allow: dict[tuple[int, int], int], block: int) -> tuple[int, int, int] | None:
    try:
        return auto_config(prob_m, prob_n, top_k, sms, allow, block=block)
    except ValueError:
        return None


def allows(served: dict[int, tuple[int, int, int]], block: int, prob_n: int, top_k: int, launch_top_k: int, E: int,
           sms: int) -> list[dict[tuple[int, int], int]]:
    """Every `allow` of a moe block (each of its thread configs' limit in 0 to the block's clamp, 0 invalid) under which `auto_config`
    gives the config ``served`` holds at each of its batches at that block: ``served`` maps M tokens to the (TK, TN, blocks per SM) a
    launch of N = ``prob_n`` ran at (a capture's sweep), routed top ``top_k`` over E experts, its prob_m the M * top_k pairs over the
    launch's top_k."""
    configs = thread_configs(block)
    obs = [(M, tuple(c)) for M, c in sorted(served.items()) if c and moe_block(M, top_k, E) == block]
    out = []
    for a in itertools.product(range(max_blocks_per_sm(block) + 1), repeat=len(configs)):
        allow = {(tk, tn): v for (tk, tn, _), v in zip(configs, a) if v}
        if obs and all(_auto(M * top_k // launch_top_k, prob_n, launch_top_k, sms, allow, block) == c for M, c in obs):
            out.append(allow)
    return out


def served_config(served: dict[int, tuple[int, int, int]], M: int, prob_n: int, top_k: int, launch_top_k: int, E: int,
                  sms: int) -> tuple[int, int, int] | None:
    """The config a launch runs at M tokens: ``served``'s at a batch it holds, else the one `auto_config` gives at M under every
    `allows` of M's moe block; None when there is none or they differ."""
    if M in served:
        return tuple(served[M])
    block = moe_block(M, top_k, E)
    got = {_auto(M * top_k // launch_top_k, prob_n, launch_top_k, sms, a, block) for a in allows(served, block, prob_n, top_k, launch_top_k, E, sms)}
    return next(iter(got)) if len(got) == 1 and None not in got else None

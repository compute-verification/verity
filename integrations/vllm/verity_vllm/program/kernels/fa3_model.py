"""Independent CPU model of vLLM's FlashAttention-3 forward on Hopper (sm_90, BF16, head dims 64, 128 and 256) — lane HP.

The arithmetic is ``fa3_rows`` in ``cpp/fa2_model.cpp``: W11's FA2 per-row recurrence instantiated with the two things
Hopper computes differently (REPORT §2 W4d) —

* the tensor-core step is ``wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16`` (``veritor.core.silicon_total
  HOPPER_BF16_WGMMA_K16``: one group of 16 products + accumulator, 26-bit adder, exponent floor -133; pinned on an
  H100 by ``gpu/tc-hopper``) instead of ``mma.sync.m16n8k16`` (two groups of 8, 25 bits, -132);
* ``kBlockN = 128`` keys per online-softmax block (``hopper/tile_size.h``: hdim 128, causal) instead of 64, and the
  epilogue ``inv_sum = (sum == 0 || sum != sum) ? 0 : 1/sum`` (``hopper/softmax.h`` finalize) instead of ``? 1``.

Everything else is FA2's recurrence read from ``hopper/softmax.h`` / ``mainloop_fwd_sm90_tma_gmma_ws.hpp`` at
vllm-flash-attn 506341a1: blocks visited last -> first, ``scores_scale = exp2((m_prev - m_cur) * scale_log2)``,
``row_sum = row_sum * scores_scale`` contracted by nvcc into the first FADD of the block (one FFMA — the FA2
finding, *assumed* for FA3 and confirmed only through the production correspondence below), ``p = exp2(fma(s,
scale_log2, -m * scale_log2))``, per-thread partial sums in accumulator-fragment order (cols ``8*nj + 2*(lane%4) +
j``, j fastest — the wgmma accumulator layout is the ``mma.sync`` one), ``O = O * scores_scale`` then the P.V
wgmma chain, ``sum = (l0 + l2) + (l1 + l3)`` (``Allreduce<4>``), ``out = bf16(O * inv)``; FTZ everywhere
(``--use_fast_math``), MUFU.EX2 / MUFU.RCP from the sm_89 tables (assumed identical on sm_90 — tested only through
the same correspondence).

Kernel (torch.profiler record, results/CONF-b1-hopper-*/kernel_set.json): ``flash::FlashAttnFwdSm90<
CollectiveMainloopFwdSm90<2, ClusterShape 1x1x1, TileShape (128|64)x128x128, 128, bf16, f32, Sm90, Is_causal=true,
Is_local=false, Has_softcap=false, Varlen=true, PagedKVNonTMA=true, AppendKV=false, HasQv=false, MmaPV_is_RS=true,
IntraWGOverlap=true, PackGQA=true, Split=false, V_colmajor=false>>``, kBlockM 128 (prefill) / 64 (decode),
num_splits = 1 (``VLLM_BATCH_INVARIANT``), no ``FlashAttnFwdCombine``.

Head dim 256 (GLM-5.3's dense MHA prefill, ``run_prefill_new_tokens``: qk 192 nope + 64 rope, v 256, 8 heads a rank, causal varlen) runs
``TileShape 128x80x256`` (``tile_size_fwd_sm90``: hdim > 192, not local -> {128, 80, MmaPV_is_RS, IntraWGOverlap}, whatever seqlen_q or
paging), so a block is 80 keys: 16 QK k16 steps, 5 P.V k16 steps.  ``RescaleOBeforeGemm = kHeadDim > 128 && IntraWGOverlap`` is true
there: block n's rescale of O moves from the end of its own step to just before the next block's P.V gemm, which leaves every element's
``O = O * scores_scale + P V`` (and the FFMA-contracted row sum) as here.  kBlockM 128 > kBlockN 80 makes the first rows of an M tile visit
up to two fully-masked key blocks first; ``kernel_geometry(kblock_m=)`` gives that count and the twin shows it neutral.  vLLM launches it
with ``num_splits = 0``: :func:`num_splits` says which sequences the heuristic splits (GLM-5.3 per rank, 132 SMs: 321..1664 tokens), and
those run the Split kernel and ``FlashAttnFwdCombine``, whose rows are ``AttentionSplit_v1`` (``registry.fa3_split``; its twin
``fa3_split_twins.split_call`` runs :func:`attention_rows` split by split).  Not yet measured on an H200 (``MEASURED``; the capture is
``tests/properties/glm53_fa3_prefill_capture_gpu.py``).

Measured (typed program `AttentionFA3@1` = this arithmetic gate for gate, correspondence against the H100 production
capture, request 917b9faa, positions 0..526, layers 0/1/13/27).  Exactness
against this module is checked by tests/test_fa3_model.py (model vs the typed composite on random operands).
"""

from __future__ import annotations

import ctypes

import numpy as np

from verity_vllm.program.kernels import fa2_model as _fa2
from verity_vllm.program.kernels.fa2_model import (  # noqa: F401  (the shared pieces: tables, scale, build)
    HEAD_DIM,
    M_LOG2E,
    MufuTables,
    RowsResult,
    _nudge,
    build,
    mufu_ex2,
    mufu_rcp,
    scale_log2,
)

KBLOCK_N = 128           # hdim 128 (causal / paged non-TMA), and hdim 64 outside the one-MMA-warpgroup regime
KBLOCK_N_HD64_ONE_WG = 192   # hdim 64, one MMA warpgroup (seqlen_q * qheads_per_kvhead <= 64): 64 x 192 tile
KBLOCK_N_HD256 = 80      # hdim 256, not local: the 128 x 80 tile ("128 x 80 hits the limit of smem"), whatever seqlen_q
KBLOCK_M_PREFILL = 128
KBLOCK_M_DECODE = 64
SUPPORTED_HEAD_DIMS = (64, 128, 256)
SUPPORTED_KBLOCK_N = (80, 128, 192)


def kblock_n_for(D: int, seqlen_q: int, qh_per_kvh: int) -> int:
    """[R13 hopper] kBlockN of the FA3 forward vLLM launches for head dim ``D`` (vllm-flash-attn 506341a1 ``hopper/tile_size.h::
    tile_size_fwd_sm90``, element_size 2, causal, paged KV without TMA) with ``use_one_mma_wg = D in {64, 128} and seqlen_q *
    qheads_per_kvhead <= 64`` (``hopper/heuristics.h``; ``seqlen_q`` = the launch's max_seqlen_q under varlen): D <= 64 -> 192 under
    one MMA warpgroup (64 x 192 tile) else 128 (192 x 128 tile); D <= 128 -> 128; D = 256 -> 80 (the 128 x 80 tile, paged or not).
    The same rule as ``verity_vllm.program.frontend.targets.fa3_kblock_n`` (the registered one; ``check`` cross-checks the two)."""
    D, seqlen_q, qh = int(D), int(seqlen_q), int(qh_per_kvh)
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"FA3 head dim {D}: not modelled (supported {SUPPORTED_HEAD_DIMS})")
    if D == 256:
        return KBLOCK_N_HD256
    one_wg = seqlen_q * qh <= 64
    if D <= 64:
        return KBLOCK_N_HD64_ONE_WG if one_wg else KBLOCK_N
    return KBLOCK_N


def kblock_m_for(D: int, seqlen_q: int, qh_per_kvh: int) -> int:
    """kBlockM of the same tile (``targets.fa3_tile_m``'s rule): D <= 64 -> 64 under one MMA warpgroup else 192; D <= 128 -> 64 under
    one MMA warpgroup else 128; D = 256 -> 128."""
    D, seqlen_q, qh = int(D), int(seqlen_q), int(qh_per_kvh)
    if D not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"FA3 head dim {D}: not modelled (supported {SUPPORTED_HEAD_DIMS})")
    if D == 256:
        return KBLOCK_M_PREFILL
    one_wg = seqlen_q * qh <= 64
    if D <= 64:
        return KBLOCK_M_DECODE if one_wg else 192
    return KBLOCK_M_DECODE if one_wg else KBLOCK_M_PREFILL

KERNEL = {
    "source": "vllm-flash-attn (vllm-project/flash-attention) commit 506341a143fcabd4bb79052a7605ada727d6b3f5, hopper/ (FA3)",
    "build": "--use_fast_math (ftz, MUFU.EX2 for exp2f, MUFU.RCP for 1.f/x); sm_90a cubin (_vllm_fa3_C)",
    "kernel": "flash::FlashAttnFwdSm90<CollectiveMainloopFwdSm90<2, 1x1x1, (128|64)x128x128, 128, bf16, f32, Sm90, Is_causal, "
              "!Is_local, !Has_softcap, Varlen, PagedKVNonTMA, !AppendKV, !HasQv, MmaPV_is_RS, IntraWGOverlap, PackGQA, !Split, !V_colmajor>>",
    "mma": "wgmma.mma_async.sync.aligned.m64nNk16.f32.bf16.bf16 (veritor.core.silicon_total HOPPER_BF16_WGMMA_K16: groups (16,), width 26, floor -133)",
    "block": {"kBlockM": [KBLOCK_M_PREFILL, KBLOCK_M_DECODE], "kBlockN": KBLOCK_N, "head_dim": HEAD_DIM},
    "block_hd256": {"kernel": "TileShape 128x80x256, RescaleOBeforeGemm", "kBlockM": KBLOCK_M_PREFILL, "kBlockN": KBLOCK_N_HD256},
    "inv_sum": "(sum == 0.f || sum != sum) ? 0.f : 1.f / sum   (FA2: ? 1.f)",
}

_lib = None


def lib() -> ctypes.CDLL:
    """The shared FA2/FA3 library with `fa3_rows` bound (same signature as `fa2_rows`)."""
    global _lib
    if _lib is None:
        L = _fa2.lib()
        L.fa3_rows.restype = None
        L.fa3_rows.argtypes = L.fa2_rows.argtypes
        _lib = L
    return _lib


def kernel_geometry(seqlen_q: int, seqlen_k: int, rows: list[int] | np.ndarray, kblock_n: int = KBLOCK_N, *,
                    kblock_m: int | None = None, qh_per_kvh: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Per query row ``r``: (visible columns, n_block_max).  FA3 keeps ``Is_causal`` for ``seqlen_q == 1`` (the decode
    kernel instantiation above is causal), so every launch is causal, bottom-right aligned: ``col_limit = min(seqlen_k,
    r + 1 + seqlen_k - seqlen_q)``.  ``BlockMN::get_n_block_min_max`` gives ``n_block_max = min(ceil(seqlen_k /
    kBlockN), ceil((m_idx_max + seqlen_k - seqlen_q) / kBlockN))`` with ``m_idx_max = (m_block + 1) * kBlockM``
    (PackGQA: divided by the GQA ratio); with kBlockM = kBlockN = 128 in prefill and one packed M tile in decode
    this equals ``ceil(col_limit / kBlockN)`` — no fully-masked block is ever visited (the typed program's
    ``AttentionHead`` block count).  [R13 hopper] With kBlockM > kBlockN (hdim 64 prefill: 192 x 128; hdim 256: 128 x 80) the
    kernel's per-tile count makes the first rows of an M tile visit fully-masked key blocks first; those are arithmetically
    neutral in this recurrence (S = -inf -> P = 0, running max -inf, l = o = 0, and the next block's rescale of that zero state is
    0 * x = 0 -- the kernel's own first-block path), so the per-row count is what is modelled.  ``kblock_m`` gives the kernel's
    per-tile count instead (packed row ``r * qh_per_kvh``'s M tile), which the twin runs to show that neutrality."""
    rows = np.asarray(rows, dtype=np.int64)
    off = seqlen_k - seqlen_q
    col_limit = np.minimum(seqlen_k, rows + 1 + off)
    BN = int(kblock_n)
    if kblock_m is None:
        nb = -(-col_limit // BN)
    else:
        BM, g = int(kblock_m), int(qh_per_kvh)
        m_idx_max = (((rows * g) // BM + 1) * BM - 1) // g + 1
        nb = np.minimum(-(-seqlen_k // BN), -(-(m_idx_max + off) // BN))
    return col_limit.astype(np.int32), nb.astype(np.int32)


#: hopper/flash_api.cpp PREPARE_VARLEN_MAX_BATCHES_1CTA: a varlen launch of more sequences runs one split
PREPARE_VARLEN_MAX_BATCHES_1CTA = 992


def num_splits_heuristic(total_mblocks: int, num_sms: int, num_n_blocks: int, num_m_blocks: int, size_one_kv_head: int,
                         is_causal_or_local: bool, max_splits: int) -> int:
    """``hopper/heuristics.h::num_splits_heuristic`` at vllm-flash-attn 506341a1, its float arithmetic in f32 as the host compiles it."""
    f32 = np.float32
    if f32(total_mblocks) >= f32(0.8) * f32(num_sms):
        size_l2 = 50 * 1024 * 1024
        if size_one_kv_head > size_l2 and num_m_blocks >= num_sms * 2 and not is_causal_or_local:
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff = []
    for ns in range(1, max_splits + 1):
        n_waves = f32(total_mblocks * ns) / f32(num_sms)
        eff.append(n_waves / np.ceil(n_waves))
    best = max(eff)
    return next((ns for ns, e in enumerate(eff, 1) if float(e) >= 0.85 * float(best)), 1)


def num_splits(seqlens: list[int] | np.ndarray, H: int, KVH: int, D: int, *, num_sms: int, Dv: int | None = None) -> list[int]:
    """The split count FA3 runs each sequence of one causal varlen self-attention launch at: :func:`split_plan`'s second part."""
    return split_plan(seqlens, H, KVH, D, num_sms=num_sms, Dv=Dv)[1]


def split_plan(seqlens: list[int] | np.ndarray, H: int, KVH: int, D: int, *, num_sms: int,
               Dv: int | None = None) -> tuple[int, list[int]]:
    """(the launch's ``params.num_splits``, each sequence's split count) of one causal varlen self-attention launch (``cu_seqlens_q =
    cu_seqlens_k``, the sequences' lengths ``seqlens``) when the caller passes ``num_splits = 0`` (vLLM's MLA prefill outside batch
    invariance).  The launch runs the ``Split`` mainloop and ``FlashAttnFwdCombine`` exactly when the first exceeds 1:
    ``flash_api.cpp::get_num_splits``' static bound (the tile of max_seqlen_q, ``total_mblocks = h_k * num_m_blocks`` as if batch = 1,
    PackGQA rows), then ``flash_prepare_scheduler.cu``'s dynamic split per sequence (``blocks_per_sm = ceilf(total_blocks * 1.1f * h_k /
    num_sm)`` over every sequence's ``num_m_blocks * num_n_blocks``; ``max(min(ceil(num_n_blocks / blocks_per_sm), static), 1)``); one
    split past ``PREPARE_VARLEN_MAX_BATCHES_1CTA`` sequences.  The host heuristic is IEEE f32; the device's ``/`` is ``div.full.ftz.f32``
    (``--use_fast_math``: the numerator times the measured MUFU.RCP of ``num_sm``, ``DivFullRcp_v1``), assumed to lower as Triton's does.
    A sequence counted above 1 is combined from its splits (``fa3_split_twins.split_call``, ``AttentionSplit_v1``); one counted 1
    is written directly."""
    from verity_vllm.program.kernels import derived_rows as DR
    L = [int(x) for x in seqlens]
    Dv = int(D if Dv is None else Dv)
    g = int(H) // int(KVH)
    if not L or len(L) > PREPARE_VARLEN_MAX_BATCHES_1CTA:
        return 1, [1] * len(L)
    max_q = max(L)
    BM, BN = kblock_m_for(D, max_q, g), kblock_n_for(D, max_q, g)
    nmb_static = -(-(max_q * g) // BM)
    nnb_static = -(-max_q // BN)
    static = num_splits_heuristic(int(KVH) * nmb_static, int(num_sms), nnb_static, nmb_static, max_q * (int(D) + Dv) * 2, True, 128)
    if static == 1:
        return 1, [1] * len(L)
    nmb = [-(-(x * g) // BM) for x in L]
    nnb = [-(-x // BN) for x in L]
    total = sum(m * n for m, n in zip(nmb, nnb))
    f32 = np.float32
    a = np.array([f32(f32(f32(total) * f32(1.1)) * f32(KVH))], np.float32).view(np.uint32)
    b = np.array([f32(num_sms)], np.float32).view(np.uint32)
    bps = int(np.ceil(DR.as_f32(DR.mul_ftz(DR.div_full_scale_a(a, b), DR.div_full_rcp(b)))[0]))
    return static, [max(min(-(-n // bps), static), 1) for n in nnb]


def attention_rows(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, kvh: np.ndarray, col_limit: np.ndarray, n_block_max: np.ndarray,
    softmax_scale: float, tables: MufuTables, *, debug: bool = False, variant: int = 0, scale_log2_nudge: int = 0,
    kblock_n: int = KBLOCK_N,
) -> RowsResult:
    """``q`` (n, D) u16 words (one head's query per row); ``k``/``v`` (Lk, KVH, D) u16 canonical KV rows;
    ``kvh`` (n,) KV head per row; ``col_limit``/``n_block_max`` from :func:`kernel_geometry` (same ``kblock_n``)."""
    BN = int(kblock_n)
    if BN not in SUPPORTED_KBLOCK_N:
        raise ValueError(f"kBlockN {BN}: not an FA3 sm90 bf16 forward tile modelled here {SUPPORTED_KBLOCK_N}")
    q = np.ascontiguousarray(q, dtype=np.uint16)
    k = np.ascontiguousarray(k, dtype=np.uint16)
    v = np.ascontiguousarray(v, dtype=np.uint16)
    n, D = q.shape
    Lk, KVH, D2 = k.shape
    if D not in SUPPORTED_HEAD_DIMS or D2 != D or v.shape != k.shape or (D == 256) != (BN == KBLOCK_N_HD256):
        raise ValueError(f"shapes: q {q.shape} k {k.shape} v {v.shape}, kBlockN {BN} (head dims modelled: {SUPPORTED_HEAD_DIMS}; "
                         f"hdim 256 runs kBlockN {KBLOCK_N_HD256} only)")
    kvh = np.ascontiguousarray(kvh, dtype=np.int32)
    col_limit = np.ascontiguousarray(col_limit, dtype=np.int32)
    n_block_max = np.ascontiguousarray(n_block_max, dtype=np.int32)
    if col_limit.max(initial=0) > Lk or (n_block_max * BN).max(initial=0) < col_limit.max(initial=0):
        raise ValueError("col_limit exceeds the KV rows / block range")
    out = np.zeros((n, D), np.uint16)
    out_f32 = np.zeros((n, D), np.uint32)
    flags = np.zeros(n, np.uint8)
    max_blocks = int(n_block_max.max(initial=1))
    dbg = None
    ptrs = [None] * 5
    if debug:
        dbg = {
            "s": np.zeros((n, max_blocks * BN), np.uint32),
            "p": np.zeros((n, max_blocks * BN), np.uint32),
            "m": np.zeros((n, max_blocks), np.uint32),
            "l": np.zeros((n, 4), np.uint32),
            "o_pre": np.zeros((n, D), np.uint32),
        }
        ptrs = [dbg[x].ctypes.data for x in ("s", "p", "m", "l", "o_pre")]
    lib().fa3_rows(
        q.ctypes.data, k.ctypes.data, v.ctypes.data, KVH, D, kvh.ctypes.data, col_limit.ctypes.data,
        n_block_max.ctypes.data, n, BN, ctypes.c_float(float(_nudge(scale_log2(softmax_scale), scale_log2_nudge))),
        tables.ex2.ctypes.data, tables.rcp.ctypes.data, out.ctypes.data, out_f32.ctypes.data, *ptrs, max_blocks, int(variant),
        flags.ctypes.data,
    )
    if (flags & 2).any():
        raise RuntimeError("fa3_rows refused the block size")
    return RowsResult(out, out_f32, dbg, (flags & 1).astype(bool))


def attention_call(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, rows: np.ndarray, seqlen_q: int, seqlen_k: int, softmax_scale: float,
    tables: MufuTables, *, debug: bool = False, variant: int = 0, scale_log2_nudge: int = 0, kblock_n: int | None = None,
    launch_max_seqlen_q: int | None = None,
) -> RowsResult:
    """One kernel call for one sequence: ``q`` (len(rows), H, D) words for the query rows ``rows`` (0-based in the
    sequence), ``k``/``v`` (>= seqlen_k, KVH, D).  Returns (len(rows), H, D) words.  ``kblock_n`` defaults to the tile rule
    :func:`kblock_n_for` on the LAUNCH's max_seqlen_q (``launch_max_seqlen_q``, default ``seqlen_q``: one sequence per launch)."""
    nrow, H, D = q.shape
    KVH = k.shape[1]
    ngroups = H // KVH
    BN = int(kblock_n) if kblock_n is not None else kblock_n_for(D, launch_max_seqlen_q if launch_max_seqlen_q is not None else seqlen_q, ngroups)
    rows = np.asarray(rows, dtype=np.int64)
    col_limit, nb = kernel_geometry(seqlen_q, seqlen_k, rows, BN)
    qf = q.reshape(nrow * H, D)
    kvh = (np.arange(H, dtype=np.int32) // ngroups)[None, :].repeat(nrow, 0).reshape(-1)
    res = attention_rows(qf, k[:seqlen_k], v[:seqlen_k], kvh, np.repeat(col_limit, H), np.repeat(nb, H), softmax_scale, tables,
                         debug=debug, variant=variant, scale_log2_nudge=scale_log2_nudge, kblock_n=BN)
    res.out = res.out.reshape(nrow, H, D)
    res.out_f32 = res.out_f32.reshape(nrow, H, D)
    res.nonfinite = res.nonfinite.reshape(nrow, H)
    return res


MEASURED = {
    # what AttentionFA3@1 (= this arithmetic) was measured bit-exact against: the H100 production capture of the
    # standard workload, request 917b9faa (prompt 512 + 16 decode), layers 0/1/13/27 (hopper-profile.md)
    "arch": ("sm_90",),
    "head_dim": (128,),
    "heads_kv_heads": ((12, 2),),
    "block_size": (16,),
    "seqlen_q": (1, 512),
    "seqlen_k": (1, 527),
    "fa_version": 3, "num_splits": 1, "window": ([-1, -1], [-1, 0]), "softcap": 0.0, "alibi": False,
    "dtype": "bfloat16", "finite_operands": True,
}


def coverage(seqlen_q: int, seqlen_k: int, H: int, KVH: int, D: int, consts: dict, *, arch: str | None = None) -> str | None:
    """None when the call is inside the measured family, else the reason it is ``unresolved`` (the gate is what was
    measured, as in :func:`fa2_model.coverage`)."""
    a = arch or consts.get("arch")
    if a is None:
        name = str(consts.get("gpu") or consts.get("gpu_name") or "")
        a = "sm_90" if ("H100" in name or "H200" in name or "H800" in name) else None
    if a is None:
        return "arch not declared (consts.arch / consts.gpu); measured only on sm_90"
    if a not in MEASURED["arch"]:
        return f"arch {a} not measured (measured: {MEASURED['arch']})"
    if D not in SUPPORTED_HEAD_DIMS:
        return f"head_dim {D} not modelled (kernel traits differ; modelled: {SUPPORTED_HEAD_DIMS})"
    if D not in MEASURED["head_dim"]:
        return f"head_dim {D} modelled but not measured (measured: {MEASURED['head_dim']})"
    if int(consts.get("fa_version", 3)) != 3:
        return "fa_version != 3"
    if int(consts.get("num_splits", 1)) != 1:
        return "num_splits != 1 (split-KV combine not modelled)"
    ws = consts.get("window_size") or consts.get("sliding_window") or [-1, -1]
    if list(ws) not in MEASURED["window"]:
        return "sliding window not modelled"
    if float(consts.get("softcap", consts.get("logits_soft_cap", 0)) or 0) != 0:
        return "softcap not modelled"
    if consts.get("alibi") or consts.get("alibi_slopes"):
        return "alibi not modelled"
    if str(consts.get("dtype", "bfloat16")).replace("torch.", "") != "bfloat16":
        return "dtype != bfloat16"
    if (H, KVH) not in MEASURED["heads_kv_heads"]:
        return f"(num_heads, num_kv_heads) = ({H}, {KVH}) not measured (measured: {MEASURED['heads_kv_heads']})"
    if not (MEASURED["seqlen_k"][0] <= seqlen_k <= MEASURED["seqlen_k"][1]):
        return f"seqlen_k {seqlen_k} outside the measured range {MEASURED['seqlen_k']}"
    if not (MEASURED["seqlen_q"][0] <= seqlen_q <= MEASURED["seqlen_q"][1]) or seqlen_q > seqlen_k:
        return f"seqlen_q {seqlen_q} outside the measured range {MEASURED['seqlen_q']} (and <= seqlen_k)"
    return None

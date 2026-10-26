"""numpy twin of `registry.flashmla` (DeepSeek-V4-Pro's FlashMLA sparse decode on the H200, 6bc49418) and a port of its decode
scheduler.

`decode` computes `FlashMlaDecode_v1` over numpy words, its heads at once; `split`, `combine` and the libm pieces are its parts.
The arithmetic is written here again from the kernel's SASS rather than read from the registry (FADD / FMUL / FFMA round to
nearest with subnormals, every NaN result 0x7FFFFFFF; MUFU through core's measured tables; the wgmma k16 step through
`verity.ml.kernels.tc`), so the self-check compares two writings.  `sched_meta` is `get_mla_metadata_kernel`
(csrc/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu:12-104) line for line, its one shared array included, and
`request_splits` a request's SPLITS from its output, as the decode kernel walks it (splitkv_mla.cuh:164-183).

`registry.flashmla` registers it (`register_twin`) with `kernel_registry` as kernel ``flashmla-twin`` of `NvExp2f_v1`,
`NvLog2f_v1`, `NvFdividefRcp_v1`, `FlashMlaKvRow_v1`, `FlashMlaSplit_v1`, `FlashMlaCombine_v1` and `FlashMlaDecode_v1` (``DOT``
the Hopper wgmma k16 step).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.ops import mufu as MU
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16
from verity.ml.circuits.silicon.total import HOPPER_BF16_WGMMA_K16
from verity.ml.kernels import tc as CK
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.numerics.kernel_registry import Declined, register

U32 = np.uint32
KERNEL = "flashmla-twin"
DOMAIN = "every word; DOT the Hopper wgmma k16 step (HopperBF16WgmmaDot16_v1)"

GPU_NAN = 0x7FFFFFFF
F_NINF, F_PINF, F_ONE, F_HALF = 0xFF800000, 0x7F800000, 0x3F800000, 0x3F000000
F_NEG126, F_2M126, F_2P23, F_2P24, F_2M23, F_NEG23, F_NEG1 = (0xC2FC0000, 0x00800000, 0x4B000000, 0x4B800000, 0x34000000,
                                                            0xC1B80000, 0xBF800000)
F_MAX_INIT = int(np.float32(-1e30).view(U32))
L2E = 0x3FB8AA3B
#: `__nv_log2f`'s exponent offset and mask and its polynomial in Horner order, as the decode's SASS has them
LOG2F_OFFSET, LOG2F_MASK = 0xC0CAFB0D, 0xFF800000
LOG2F_POLY = (0x3DC6B27F, 0xBE2C7F30, 0x3E2FCF2A, 0xBE374E43, 0x3E520BF4, 0xBE763C8B, 0x3E93BF99, 0xBEB8AA49, 0x3EF6384A,
              0xBF38AA3B)
#: FlashMLA's `LOG_2_E` (api/common.h:31)
LOG_2_E = 1.44269504
#: the sm_90 decode's tile (`TOPK_BLOCK_SIZE`), the scheduler's fixed overhead per request, and the H200's SMs
BLOCK, FIXED_OVERHEAD, H200_SMS = 64, 5, 132


def scale_log2(scale: float) -> int:
    """`sm_scale_div_log2`: `sm_scale * LOG_2_E` in double, narrowed to f32 (api/sparse_decode.h:414)."""
    return int(np.float32(float(scale) * float(np.float32(LOG_2_E))).view(U32))


# ---- the SASS's f32 steps ---------------------------------------------------------------------------------------------------

def _w(x) -> np.ndarray:
    return np.asarray(x, dtype=U32)


def is_nan(u) -> np.ndarray:
    return (_w(u) & U32(0x7FFFFFFF)) > U32(0x7F800000)


def _gpu(u) -> np.ndarray:
    u = _w(u)
    return np.where(is_nan(u), U32(GPU_NAN), u).astype(U32)


def _f64(u) -> np.ndarray:
    return np.ascontiguousarray(_w(u)).view(np.float32).astype(np.float64)


def fadd(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(_w(a), _w(b))
    with np.errstate(all="ignore"):
        return _gpu(DR.add_rn(a, b))


def fmul(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(_w(a), _w(b))
    with np.errstate(all="ignore"):
        return _gpu(DR.mul_rn(a, b))


def ffma(a, b, c) -> np.ndarray:
    with np.errstate(all="ignore"):
        return _gpu(DR.fma_rn(a, b, c))


def neg(x) -> np.ndarray:
    return _w(x) ^ U32(0x80000000)


def feq(a, b) -> np.ndarray:
    """`F32Eq_v1`: NaN unequal to everything, +0 == -0."""
    a, b = np.broadcast_arrays(_w(a), _w(b))
    return ~is_nan(a) & ~is_nan(b) & ((a == b) | (((a | b) & U32(0x7FFFFFFF)) == 0))


def geu(a, b) -> np.ndarray:
    """`setp.geu.f32`: a >= b, or either a NaN."""
    a, b = np.broadcast_arrays(_w(a), _w(b))
    return is_nan(a) | is_nan(b) | (_f64(a) >= _f64(b))


def fmaxf(a, b) -> np.ndarray:
    """`max.f32`: the larger, a NaN yields the other operand (two: 0x7FFFFFFF), max(+0, -0) = +0."""
    a, b = np.broadcast_arrays(_w(a), _w(b))
    na, nb = is_nan(a), is_nan(b)
    zeros = ((a & U32(0x7FFFFFFF)) == 0) & ((b & U32(0x7FFFFFFF)) == 0)
    r = np.where(_f64(b) > _f64(a), b, a)
    r = np.where(zeros, np.where((a == 0) | (b == 0), U32(0), U32(0x80000000)), r)
    r = np.where(na, b, np.where(nb, a, r))
    return np.where(na & nb, U32(GPU_NAN), r).astype(U32)


def ex2(x) -> np.ndarray:
    return _w(MU.mufu_ex2_words(_w(x), MU.table("ex2")))


def rcp(x) -> np.ndarray:
    return _w(MU.mufu_rcp_words(_w(x), MU.table("rcp")))


def exp2f(x) -> np.ndarray:
    """`NvExp2f_v1`."""
    x = _w(x)
    c = geu(x, F_NEG126)
    e = ex2(np.where(c, x, fmul(x, F_HALF)))
    return np.where(c, e, fmul(e, e)).astype(U32)


def log2f(x) -> np.ndarray:
    """`NvLog2f_v1`."""
    x = _w(x)
    p0 = geu(x, F_2M126)
    xs = np.where(p0, x, fmul(x, F_2P23)).astype(U32)
    e = (xs + U32(LOG2F_OFFSET)) & U32(LOG2F_MASK)
    f = fadd(xs - e, F_NEG1)
    p = ffma(f, LOG2F_POLY[0], LOG2F_POLY[1])
    for c in LOG2F_POLY[2:]:
        p = ffma(f, p, c)
    ef = ffma(e.view(np.int32).astype(np.float32).view(U32), F_2M23, np.where(p0, U32(0), U32(F_NEG23)))
    y = fadd(ef, ffma(f, L2E, fmul(f, fmul(f, p))))
    y = np.where(xs >= U32(F_PINF), ffma(xs, F_PINF, F_PINF), y)
    return np.where(feq(xs, 0), U32(F_NINF), y).astype(U32)


def rcp1(x) -> np.ndarray:
    """`NvFdividefRcp_v1`: `__fdividef(1, x)`."""
    x = _w(x)
    p = (x & U32(0x7FFFFFFF)) >= U32(F_2M126)
    r = rcp(np.where(p, x, fmul(x, F_2P24)))
    return fmul(r, np.where(p, U32(F_ONE), U32(F_2P24)))


def _bf16(w) -> np.ndarray:
    return np.asarray(w, dtype=np.uint16).astype(U32) << U32(16)


# ---- the rows ---------------------------------------------------------------------------------------------------------------

def e8m0_to_bf16(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint16) & np.uint16(0xFF)
    return np.where(x == 0xFF, np.uint16(0x7FFF), np.where(x == 0, np.uint16(0x0040), x << np.uint16(7))).astype(np.uint16)


def kv_rows(nope, scale, rope, valid, QT: int) -> np.ndarray:
    """`FlashMlaKvRow_v1` over rows: (N, DN) e4m3 bytes, (N, DN / QT) e8m0 bytes, (N, DR) bf16, (N,) valid -> (N, DN + DR) bf16."""
    nope, rope = np.asarray(nope), np.asarray(rope, dtype=np.uint16)
    x = DR.f2fp_bf16(CK.e4m3_to_f32_batch(nope)[0])
    s = np.repeat(e8m0_to_bf16(scale), QT, axis=-1)
    el = DR.f2fp_bf16(fmul(_bf16(x), _bf16(s)))
    rows = np.concatenate([el, rope], axis=-1)
    return np.where(np.asarray(valid, dtype=bool)[..., None], rows, np.uint16(0)).astype(np.uint16)


def row_valid(idx, lens, NW: int) -> np.ndarray:
    """`FlashMlaRowValid_v1` per row in kernel order: the window's NW rows against lens[0], the extra rows against lens[1]."""
    idx = np.asarray(idx, dtype=U32).view(np.int32)
    pos = np.arange(idx.shape[-1])
    n = [np.asarray(v, dtype=U32).view(np.int32) for v in lens]
    within = pos < n[0] if len(n) == 1 else np.where(pos < NW, pos < n[0], pos - NW < n[1])
    return (idx != -1) & within


# ---- one block, one split, the combine -------------------------------------------------------------------------------------

def _dot_step(acc, a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    shape = a.shape[:-1]
    acc = np.broadcast_to(_w(acc), shape).reshape(-1)
    return _w(CK.group_sum_total_batch(HOPPER_BF16_WGMMA_K16, acc, a.reshape(-1, 16), b.reshape(-1, 16))[0]).reshape(shape)


def scores(q, k) -> np.ndarray:
    """`DotBf16_v2{K = D, HopperBF16WgmmaDot16}` of every head (H, D) with every row (N, D): (H, N) f32."""
    q, k = np.asarray(q, dtype=np.uint16), np.asarray(k, dtype=np.uint16)
    acc = np.zeros((q.shape[0], k.shape[0]), dtype=U32)
    for s in range(q.shape[1] // 16):
        acc = _dot_step(acc, q[:, None, 16 * s:16 * s + 16], k[None, :, 16 * s:16 * s + 16])
    return acc


@dataclass
class State:
    m: np.ndarray        # (H,)
    l: np.ndarray        # (H, 4) the four threads' sums
    o: np.ndarray        # (H, D)


def block(q, k, valid, scale_bits: int, prev: State | None) -> State:
    """`FlashMlaBlock_v1` for every head: q (H, D), k (BN, D) staged rows, valid (BN,); ``prev`` None on a split's first block."""
    BN = k.shape[0]
    s = np.where(np.asarray(valid, dtype=bool)[None, :], scores(q, k), U32(F_NINF))
    bmax = np.full(s.shape[0], F_NINF, dtype=U32)
    for c in range(BN):
        bmax = fmaxf(bmax, s[:, c])
    m_old = np.full(s.shape[0], F_MAX_INIT, dtype=U32) if prev is None else prev.m
    m_new = fmaxf(fmul(bmax, scale_bits), m_old)
    nm = neg(m_new)
    p = exp2f(ffma(s, scale_bits, nm[:, None]))
    lanes = []
    for t in range(4):
        cols = [c for c in range(BN) if (c % 8) // 2 == t]
        acc = p[:, cols[0]]
        for c in cols[1:]:
            acc = fadd(acc, p[:, c])
        lanes.append(acc)
    lanes = np.stack(lanes, axis=1)
    if prev is None:
        l, o = lanes, np.zeros((q.shape[0], q.shape[1]), dtype=U32)
    else:
        r = exp2f(fadd(m_old, nm))
        l, o = ffma(prev.l, r[:, None], lanes), fmul(prev.o, r[:, None])
    pb = DR.f2fp_bf16(p)
    for ks in range(BN // 16):
        a = pb[:, None, 16 * ks:16 * ks + 16]                    # (H, 1, 16)
        b = np.asarray(k[16 * ks:16 * ks + 16], dtype=np.uint16).T[None, :, :]   # (1, D, 16)
        o = _dot_step(o, a, b)
    return State(m_new, l, o)


def split(q, k, valid, scale_bits: int, BN: int, sink=None):
    """`FlashMlaSplit_v1` for every head over k (NB * BN, D): bf16 (H, D) with ``sink`` (H,), else the partial (oacc (H, D), lse (H,))."""
    st = None
    for b in range(k.shape[0] // BN):
        st = block(q, k[b * BN:(b + 1) * BN], valid[b * BN:(b + 1) * BN], scale_bits, st)
    total = fadd(fadd(st.l[:, 0], st.l[:, 1]), fadd(st.l[:, 2], st.l[:, 3]))
    empty = feq(total, 0)
    if sink is not None:
        den = fadd(total, exp2f(fadd(fmul(sink, L2E), neg(st.m))))
        return DR.f2fp_bf16(fmul(st.o, np.where(empty, U32(0), rcp1(den))[:, None]))
    oacc = fmul(st.o, np.where(empty, U32(0), rcp1(total))[:, None])
    return oacc, np.where(empty, U32(F_NINF), fadd(log2f(total), st.m)).astype(U32)


def combine(lse, oacc, sink) -> np.ndarray:
    """`FlashMlaCombine_v1` for every head: lse (H, NS), oacc (H, NS, D), sink (H,) -> bf16 (H, D)."""
    lse, oacc, sink = _w(lse), _w(oacc), _w(sink)
    NS = lse.shape[1]
    mx = np.full(lse.shape[0], F_NINF, dtype=U32)
    for s in range(NS):
        mx = fmaxf(mx, lse[:, s])
    mx = np.where(mx == U32(F_NINF), U32(0), mx)
    e = exp2f(fadd(lse, neg(mx)[:, None]))
    lanes = []
    for ln in range(32):
        acc = None
        for s in range(ln, NS, 32):
            acc = e[:, s] if acc is None else fadd(acc, e[:, s])
        lanes.append(acc)
    while len(lanes) > 1:
        h = len(lanes) // 2
        lanes = [a if b is None else b if a is None else fadd(a, b) for a, b in zip(lanes[:h], lanes[h:])]
    total = lanes[0]
    g = np.where(feq(total, 0), U32(F_PINF), fadd(log2f(total), mx))
    g_sink = fadd(g, log2f(fadd(F_ONE, exp2f(ffma(sink, L2E, neg(g))))))
    g_none = np.where(feq(sink, F_NINF), U32(F_PINF), fmul(sink, L2E))
    g = np.where(feq(g, F_PINF), g_none, g_sink).astype(U32)
    res = np.zeros(oacc.shape[::2], dtype=U32)
    for s in range(NS):
        res = ffma(exp2f(fadd(lse[:, s], neg(g)))[:, None], oacc[:, s], res)
    return DR.f2fp_bf16(res)


def decode(q, nope, scale, rope, idx, lens, sink, *, SPLITS, NW: int, QT: int, BN: int, SCALE: float) -> np.ndarray:
    """`FlashMlaDecode_v1`: q (NH, D) bf16, the N rows' nope / scale / rope and indices, lens (swa_len[, extra_len]), sink (NH,)
    -> (NH, D) bf16."""
    sp = tuple(SPLITS)
    valid = row_valid(idx, lens, NW)
    k = kv_rows(nope, scale, rope, valid, QT)
    q, sc = np.asarray(q, dtype=np.uint16), scale_log2(SCALE)
    if len(sp) == 2:
        return split(q, k, valid, sc, BN, _w(sink))
    parts = [split(q, k[a * BN:b * BN], valid[a * BN:b * BN], sc, BN) for a, b in zip(sp, sp[1:])]
    return combine(np.stack([p[1] for p in parts], axis=1), np.stack([p[0] for p in parts], axis=1), sink)


# ---- the scheduler ----------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedMeta:
    """`DecodingSchedMeta` (csrc/params.h:13-19), its seven words in order (the eighth is padding the kernel never writes)."""
    begin_req_idx: int
    end_req_idx: int
    begin_block_idx: int
    end_block_idx: int
    begin_split_idx: int
    is_first_req_splitted: int
    is_last_req_splitted: int

    def words(self) -> tuple[int, ...]:
        return (self.begin_req_idx, self.end_req_idx, self.begin_block_idx, self.end_block_idx, self.begin_split_idx,
                self.is_first_req_splitted, self.is_last_req_splitted)


def num_sm_parts(h_q: int, s_q: int = 1, num_sms: int = H200_SMS) -> int:
    """`Decode_Sm90_Impl::get_meta` (api/sparse_decode.h:62-68)."""
    return max(num_sms // s_q // (h_q // 64), 1)


def sched_meta(topk_length, extra_topk_length=None, *, topk: int, extra_topk: int = 0, parts: int = H200_SMS,
               block_size_n: int = BLOCK, fixed_overhead: int = FIXED_OVERHEAD) -> tuple[list[SchedMeta], list[int]]:
    """`get_mla_metadata_kernel` for the sparse model: (the ``parts`` metadata, num_splits [b + 1]).  The kernel's shared array is
    one array here as well: past the last request, `first_block_idx_shared[b]` reads `last_block_idx_shared[0]`."""
    b = len(topk_length)
    sh = [0] * (5 * b + 1)
    NB, NS, SK, FIRST, LAST = 0, b, 2 * b + 1, 3 * b + 1, 4 * b + 1
    total = 0
    for i in range(b):
        cur = int(topk_length[i]) if topk_length is not None else topk
        if cur == 0:
            cur = 1
        if extra_topk:
            cur = -(-cur // block_size_n) * block_size_n
            cur += int(extra_topk_length[i]) if extra_topk_length is not None else extra_topk
        sh[SK + i] = cur
        first, last = 0, max(cur - 1, 0) // block_size_n
        total += last - first + 1 + fixed_overhead
        sh[NB + i], sh[FIRST + i], sh[LAST + i] = last - first + 1, first, last
    payload = -(-total // parts) + fixed_overhead
    now_req = now_block = now_split = cum = 0
    sh[NS] = 0
    metas = []
    for _ in range(parts):
        begin_req, begin_block, begin_split = now_req, now_block + sh[FIRST + now_req], now_split
        first_split = int(now_block != 0)
        remain = payload
        while now_req < b:
            left = sh[NB + now_req] - now_block
            if remain >= left + fixed_overhead:
                cum += now_split + 1
                sh[NS + now_req + 1] = cum
                remain -= left + fixed_overhead
                now_req, now_block, now_split = now_req + 1, 0, 0
            else:
                if remain - fixed_overhead > 0:
                    now_block += remain - fixed_overhead
                    now_split += 1
                    remain = 0
                break
        end_req = now_req if now_block > 0 else now_req - 1
        end_block = now_block + sh[FIRST + now_req] if now_block > 0 else (0 if sh[SK + now_req - 1] == 0 else sh[LAST + now_req - 1] + 1)
        last_split = int(end_block != sh[LAST + end_req] + 1 and sh[SK + end_req] != 0)
        if begin_req == end_req:
            first_split = last_split = int(first_split or last_split)
        metas.append(SchedMeta(begin_req, end_req, begin_block, end_block, begin_split, first_split, last_split))
    if not (now_req == b and now_block == 0 and now_split == 0):
        raise AssertionError("get_mla_metadata_kernel: FLASH_DEVICE_ASSERT(now_req_idx == batch_size && ...) fails")
    return metas, sh[NS:NS + b + 1]


def request_blocks(topk_len: int, extra_len: int | None, block_size_n: int = BLOCK) -> tuple[int, int]:
    """(NW, N) in rows: the window rows max(ceil(topk_len), 64) and every row of request (`get_cur_req_info`, :164-176)."""
    nw = max(-(-topk_len // block_size_n), 1) * block_size_n
    return nw, nw + (-(-(extra_len or 0) // block_size_n)) * block_size_n


def request_splits(metas, num_splits, req: int, nblocks: int) -> tuple[int, ...]:
    """SPLITS of request ``req`` (``nblocks`` blocks): the block spans the partitions that do work give it, in partition order
    (each partition's start and end, splitkv_mla.cuh:178-179); their count is num_splits[req + 1] - num_splits[req]."""
    b = len(num_splits) - 1
    spans = []
    for m in metas:
        if m.begin_req_idx >= b or not m.begin_req_idx <= req <= m.end_req_idx:
            continue
        spans.append((m.begin_block_idx if req == m.begin_req_idx else 0, m.end_block_idx if req == m.end_req_idx else nblocks))
    bounds = (0,) + tuple(e for _, e in spans)
    if [s for s, _ in spans] != list(bounds[:-1]) or bounds[-1] != nblocks or len(spans) != num_splits[req + 1] - num_splits[req]:
        raise AssertionError(f"request {req}: the partitions' spans {spans} do not tile its {nblocks} blocks")
    return bounds


# ---- registration ----------------------------------------------------------------------------------------------------------

def _hopper(statics: dict) -> str | None:
    dot = statics.get("DOT")
    return None if dot is None or getattr(dot, "id", "") == "HopperBF16WgmmaDot16_v1" else f"DOT={getattr(dot, 'id', dot)}"


def _decode_fn(q, nope, scale, rope, idx, *rest, NH, SPLITS, NW, DN, DR, QT, BN, DOT, SCALE):
    D = DN + DR
    return decode(np.asarray(q).reshape(NH, D), np.asarray(nope).reshape(-1, DN), np.asarray(scale).reshape(-1, DN // QT),
                  np.asarray(rope).reshape(-1, DR), idx, rest[:-1], rest[-1], SPLITS=SPLITS, NW=NW, QT=QT, BN=BN,
                  SCALE=SCALE).reshape(-1)


def _split_fn(q, k, *sink, NB, D, BN, PARTIAL, DOT, SCALE):
    k = np.asarray(k).reshape(NB * BN, D + 1)
    out = split(np.asarray(q).reshape(1, D), k[:, :D], k[:, D], scale_log2(SCALE), BN, None if PARTIAL else _w(sink[0]).reshape(1))
    return np.concatenate([out[0].reshape(-1), out[1].reshape(-1)]) if PARTIAL else out.reshape(-1)


def _combine_fn(lse, oacc, sink, *, NS, D):
    return combine(_w(lse).reshape(1, NS), _w(oacc).reshape(1, NS, D), _w(sink).reshape(1)).reshape(-1)


def _row_fn(nope, scale, rope, valid, *, DN, DR, QT):
    return kv_rows(np.asarray(nope).reshape(1, DN), np.asarray(scale).reshape(1, DN // QT), np.asarray(rope).reshape(1, DR),
                   np.asarray(valid).reshape(1), QT).reshape(-1)


_F32_EDGE = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF, 0xFFFFFFFF, 1, 0x80000001, 0x00800000,
                      0x3F800000, 0xC2FC0000, 0xC3000000, 0x42FE0000, 0x7F7FFFFF, 0xFF7FFFFF], dtype=U32)


_BF16_EDGE = np.array([0, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0x0001, 0x7F7F], dtype=np.uint16)
_E4M3_EDGE = np.array([0x7F, 0xFF, 0x00, 0x80, 0x7E, 0xFE, 0x01], dtype=np.uint8)
_E8M0_EDGE = np.array([0, 0xFF, 1, 0xFE], dtype=np.uint8)


def _edges(rng, a, pool, frac: float = 0.2, k: int = 2) -> np.ndarray:
    """``a`` with ``k`` elements replaced from ``pool`` in about ``frac`` of its instances (axis 0): one NaN in a row reaches every
    output of a head, so the other instances stay finite."""
    a = np.array(a)
    flat = a.reshape(a.shape[0], -1)
    for i in np.flatnonzero(rng.random(a.shape[0]) < frac):
        flat[i, rng.integers(0, flat.shape[1], k)] = rng.choice(pool, k)
    return a


def _f32(rng, shape, scale: float = 4.0, p: float = 0.1) -> np.ndarray:
    v = (rng.standard_normal(shape) * scale * 2.0 ** rng.integers(-6, 6, shape)).astype(np.float32).view(U32)
    return np.where(rng.random(shape) < p, rng.choice(_F32_EDGE, shape), v).astype(U32)


def _bf16_words(rng, shape, scale: float = 1.0) -> np.ndarray:
    return _edges(rng, ((rng.standard_normal(shape) * scale).astype(np.float32).view(U32) >> U32(16)).astype(np.uint16), _BF16_EDGE)


def _e4m3(rng, shape) -> np.ndarray:
    a = rng.integers(0, 256, shape).astype(np.uint8)
    a[(a & 0x7F) == 0x7F] ^= 1
    return _edges(rng, a, _E4M3_EDGE)


def _e8m0(rng, shape) -> np.ndarray:
    return _edges(rng, rng.integers(116, 128, shape).astype(np.uint8), _E8M0_EDGE)


def _sample_libm(rng, n, params, **statics):
    return [_f32(rng, (n,), scale=64.0, p=0.25)]


def _sample_row(rng, n, params, *, DN, DR, QT):
    return [_e4m3(rng, (n, DN)), _e8m0(rng, (n, DN // QT)), _bf16_words(rng, (n, DR)), (rng.random(n) < 0.8).astype(np.uint8)]


def _staged(rng, n, N, D):
    rows = _bf16_words(rng, (n, N, D), scale=2.0)
    valid = (rng.random((n, N, 1)) < 0.8).astype(np.uint16)
    valid[: max(1, n // 6)] = 0
    return np.concatenate([rows, valid], axis=2)


def _sample_split(rng, n, params, *, NB, D, BN, PARTIAL, DOT, SCALE):
    out = [_bf16_words(rng, (n, D), scale=2.0), _staged(rng, n, NB * BN, D)]
    return out if PARTIAL else out + [_f32(rng, (n,), scale=2.0, p=0.3)]


def _sample_combine(rng, n, params, *, NS, D):
    lse = _f32(rng, (n, NS), scale=8.0, p=0.0)
    lse = np.where(rng.random((n, NS)) < 0.1, U32(F_NINF), _edges(rng, lse, _F32_EDGE, k=1)).astype(U32)
    lse[: max(1, n // 6)] = F_NINF
    return [lse, _edges(rng, _f32(rng, (n, NS, D), scale=1.0, p=0.0), _F32_EDGE, k=1), _f32(rng, (n,), scale=2.0, p=0.3)]


def _sample_decode(rng, n, params, *, NH, SPLITS, NW, DN, DR, QT, BN, DOT, SCALE):
    N = tuple(SPLITS)[-1] * BN
    idx = np.where(rng.random((n, N)) < 0.85, rng.integers(0, 1 << 20, (n, N)), -1).astype(np.int64).astype(U32)
    lens = [rng.integers(-1, NW + 2, n).astype(np.int64).astype(U32)]
    if N > NW:
        lens.append(rng.integers(-1, N - NW + 2, n).astype(np.int64).astype(U32))
    sink = _f32(rng, (n, NH), scale=2.0, p=0.3)
    return [_bf16_words(rng, (n, NH, DN + DR), scale=2.0), _e4m3(rng, (n, N, DN)), _e8m0(rng, (n, N, DN // QT)),
            _bf16_words(rng, (n, N, DR), scale=2.0), idx, *lens, sink]


def _targets(defs: dict, family: str) -> list:
    """The specialisations the self-check binds: small statics, every arm of SPLITS and PARTIAL, and the combine at 3, 33 and 127
    splits (one, two and four partials on a lane; 127 is the most any batch gives a request)."""
    from verity.ml.circuits.defs import bind

    d, dot, s = defs, HopperBF16WgmmaDot16, 32 ** -0.5
    decode = dict(NH=2, DN=16, DR=16, QT=16, BN=16, DOT=dot, SCALE=s)
    return {
        "NvExp2f_v1": [bind(d["NvExp2f_v1"])], "NvLog2f_v1": [bind(d["NvLog2f_v1"])],
        "NvFdividefRcp_v1": [bind(d["NvFdividefRcp_v1"])],
        "FlashMlaKvRow_v1": [bind(d["FlashMlaKvRow_v1"], DN=32, DR=16, QT=16)],
        "FlashMlaSplit_v1": [bind(d["FlashMlaSplit_v1"], NB=2, D=32, BN=16, PARTIAL=False, DOT=dot, SCALE=s),
                             bind(d["FlashMlaSplit_v1"], NB=1, D=32, BN=16, PARTIAL=True, DOT=dot, SCALE=s)],
        "FlashMlaCombine_v1": [bind(d["FlashMlaCombine_v1"], NS=3, D=16), bind(d["FlashMlaCombine_v1"], NS=33, D=2),
                               bind(d["FlashMlaCombine_v1"], NS=127, D=2)],
        "FlashMlaDecode_v1": [bind(d["FlashMlaDecode_v1"], SPLITS=(0, 1, 2), NW=16, **decode),
                              bind(d["FlashMlaDecode_v1"], SPLITS=(0, 2), NW=32, **decode)],
    }[family]


def _instance(fn):
    def run(*args, **statics):
        why = _hopper(statics)
        if why:
            raise Declined(why)
        return fn(*args, **statics)
    return run


KERNELS = {
    "NvExp2f_v1": (lambda x: exp2f(x).reshape(-1), _sample_libm), "NvLog2f_v1": (lambda x: log2f(x).reshape(-1), _sample_libm),
    "NvFdividefRcp_v1": (lambda x: rcp1(x).reshape(-1), _sample_libm), "FlashMlaKvRow_v1": (_row_fn, _sample_row),
    "FlashMlaSplit_v1": (_split_fn, _sample_split), "FlashMlaCombine_v1": (_combine_fn, _sample_combine),
    "FlashMlaDecode_v1": (_decode_fn, _sample_decode),
}


def register_twin(defs: dict) -> None:
    """Register the twin of each family in `KERNELS`; ``defs`` is `registry.flashmla`'s composites by id, which binds the
    self-check's specialisations (the registry calls this, so neither module imports the other both ways)."""
    for family, (fn, sample) in KERNELS.items():
        register(family, _instance(fn), kernel=KERNEL, domain=DOMAIN, sample=sample,
                 check_targets=lambda rng, f=family: _targets(defs, f), covers=_hopper)

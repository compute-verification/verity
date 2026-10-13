"""[GLM-5.3] Sparse MLA attention with the key range split, as vLLM d9105ea80's `FLASH_ATTN_MLA_SPARSE` runs it by default on the
H200 (vllm-flash-attn 506341a1, FA3, sm_90, 132 SMs): each split is `sparse_mla`'s online softmax over its blocks, ended with its
partial output and log-sum-exp, and `FlashAttnFwdCombine` merges a token's splits.

How many splits.  `forward_mqa` (flashattn_mla_sparse.py:201-250) passes no `num_splits`, so `mha_fwd` (hopper/flash_api.cpp:1055-1070)
takes `get_num_splits`: `num_splits_heuristic(total_mblocks = 1, num_sm, ceil(max_seqlen_k / 64), ceil(8 / 64), max_seqlen_k * 576 * 2,
..., 128)` (heuristics.h:32-66; varlen counts the batch once), with `max_seqlen_k` the top-k width 2048: a bound of 28 at 132 SMs
(`static_num_splits`), and 1 for a batch of more than 992 tokens.  Above 1, `prepare_varlen_num_blocks` (flash_prepare_scheduler.cu:143-163)
gives token b its count from the whole batch: `blocks_per_sm = ceilf(float(total) * 1.1f * float(num_head) / float(num_sm))` (total
the batch's 64-key blocks, num_head 1 under PackGQA, `/` the `div.full.f32` of `--use_fast_math`) and `ns_b = max(min(ceil(nblk_b /
blocks_per_sm), bound), 1)` (`dynamic_num_splits`).  The mainloop cuts token b's blocks (`get_n_block_min_max`): per = ceil(nblk / ns),
split s the blocks [s per, min((s + 1) per, nblk)), empty once s per >= nblk (`split_blocks`).  A token of one split is written as
the unsplit kernel writes it (the epilogue's `is_split = num_splits > 1`, the combine's `num_splits <= 1` return): `SparseMlaHead_v1`.

A split.  Its blocks are visited last to first, its first visited block the one `SparseMlaBlock_v1{FIRST}` (`Is_first`,
`Check_inf`); only the token's last block is partial.  `Softmax::finalize` (hopper/softmax.h:137-152, and the SASS of
`FlashAttnFwdSm90<64x64x64, 512, ..., PackGQA, Split>`) ends it: `sum = (l0 + l2) + (l1 + l3)`, the partial output `o * INV(sum)` in
f32, and `lse = sum != 0 ? fma(row_max, scale_log2 * LN2, lg2(sum) * LN2) : -inf` (`__logf` is MUFU.LG2 times LN2, the test
FSETP.NE.FTZ), row_max the running max unguarded (`SparseMlaSplitEnd_v1`).

The combine.  `FlashAttnFwdCombine<8x128, kLogMaxSplits 5, ..., Varlen>` (flash_fwd_combine_kernel.h:453-690, and its SASS): lane i
of a warp holds split i's lse and the lanes past the token's splits -inf (an empty split's lse is -inf too, so it is a pad: NSPLIT
counts the non-empty ones).  `max` is the warp's FMNMX.FTZ butterfly as lane 0 sees it (offsets 16, 8, 4, 2, 1, its own operand
first), guarded to 0 at -inf; `scale_i = ex2((lse_i - max) * LOG2E)` (`expf`); `sum` the warp's FADD.FTZ butterfly of `0 + scale_i`;
`w_i = scale_i * INV(sum)`; then over the splits in order `acc = w_s > 0 ? fma(w_s, O_s, acc) : acc` from +0 (FSETP.GT.FTZ), and
`out = bf16(acc)` (`SparseMlaCombine_v1`).  FMNMX.FTZ is `F32Max_v1` (the block's row max) up to the sign of a zero and a NaN
operand; every reader of the max (the -inf guard, FADD.FTZ) maps either to the same word.

MUFU.LG2 (`MufuLg2Ftz_v1`, `lg2.approx.ftz.f32`).  The fraction v(m) of log2(1.m) in units of 2^-36 is the special-function unit's
quadratic interpolator, `verity.ml.circuits.ops.boolean.mufu.Fit` over `mufu_lg2.json` (64 segments of 2^17 mantissas, the square
from column 19, square_shift 1, shift 2), and v(0) = 0.  With E the exponent field and t = (E + 1) mod 128, M is the 43-bit
`t << 36 | v` for E >= 127 and its complement below (|log2 x| 2^36 less one), and the word is M truncated to 24 significant bits:
exponent bit_length(M) + 90, sign E < 127, +0 at M = 0.  A zero or subnormal gives -inf, a negative word or a NaN the MUFU NaN
0x7FFFFFFF, +inf +inf.  The coefficients reproduce every mantissa at [1, 2) and 2^20 random words measured on sm_120 (run
r20261004-060203-ba52); the measurements leave C0 a range in 39 segments, and `mufu_lg2_undetermined` names the words the range's ends
disagree on and the mantissa-0 words below 1.  FA3 takes the log of a split's sum, in [1 - 2^-24, 2^12), where none is.  sm_90 is
unmeasured: `tests/properties/glm53_fa3_split_capture_gpu.py` measures it.

Registered functions (all `Name_v1`):
  MufuLg2Ftz                                    (x: f32) -> f32
  F32NeZeroFtz, F32GtZeroFtz                    (x: f32) -> Value<1>   x != 0 (false on NaN), x > 0, a subnormal x read as 0
  SparseMlaSplitEnd{DV,INV,SCALE}               (m, l[4], o[DV]) -> (o_partial[DV], lse)
  SparseMlaCombine{NSPLIT,DV,INV}               (lse: NSPLIT f32, o: NSPLIT x DV f32) -> DV bf16
  SparseMlaHeadSplit{T,NS,DR,DV,BN,DOT,INV,SCALE}  (q_nope: DV bf16, q_rope: DR bf16, kv: T x (DV+DR) bf16) -> DV bf16
  SparseMlaSplit{T,NS,NH,DR,DV,BN,DOT,INV,SCALE}   (q_nope: NH x DV, q_rope: NH x DR, kv: T x (DV+DR)) -> NH x DV bf16

NS is the token's split count (`dynamic_num_splits`), the boundaries `split_blocks(T, NS, BN)`.
"""
from __future__ import annotations

import json
from functools import cache
from importlib import resources

import numpy as np

from verity.ml.circuits.defs import bind, composite, primitive
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.boolean.mufu import Fit
from verity.ml.circuits.ops.mufu import DivFullRcp
from verity.ml.circuits.ops.prims import const
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.registry.prims import BF16, F32

BIT = SC.BIT
LG2_TABLE = "mufu_lg2.json"
LG2_FORMAT = "verity/mufu-lg2-quadratic/v1"
NEG_INF, INF, QNAN = 0xFF800000, 0x7F800000, 0x7FFFFFFF
LN2, LOG2E = 0x3F317218, 0x3FB8AA3B          # float(M_LN2), float(M_LOG2E)
#: the combine's warp: kLogMaxSplits 5 for every bound up to 32 (flash_fwd_combine_launch_template.h:77)
LANES = 32
#: FA3's kBlockN at hdim 64 / hdimv 512, and the batch above which no varlen launch splits (PREPARE_VARLEN_MAX_BATCHES_1CTA)
BLOCK_N, MAX_SPLIT_BATCH = 64, 992

_CONF = ("model-untested: read from vllm-flash-attn 506341a1's FA3 split path, combine kernel and their SASS (the module "
         "docstring's files) at hdim 64 / hdimv 512; tests/properties/glm53_fa3_split_capture_gpu.py is its hardware check")
_LG2_CONF = ("model-untested on sm_90: fit to sm_120's lg2.approx.ftz.f32 (every mantissa at [1, 2) and 2^20 random words, run "
             "r20261004-060203-ba52, 0 mismatches); total on encodings, `mufu_lg2_undetermined` names the words the measurement "
             "leaves open; tests/properties/glm53_fa3_split_capture_gpu.py measures sm_90")


# --- MUFU.LG2 ---------------------------------------------------------------------------------------------------------------

@cache
def lg2_table() -> dict:
    """`mufu_lg2.json`, its format checked."""
    data = json.loads((resources.files(__package__) / LG2_TABLE).read_text())
    if data.get("format") != LG2_FORMAT:
        raise ValueError(f"{LG2_TABLE}: unknown format {data.get('format')!r}")
    return data


@cache
def lg2_fit(c0: str = "C0") -> Fit:
    """The interpolator with the table's C0 (the primitive's) or its other end ``C0_max``."""
    return Fit("lg2", {**lg2_table(), "C0": lg2_table()[c0]})


@cache
def _square() -> tuple[int, ...]:
    f = lg2_fit()
    return tuple(int(q) for q in f.square(np.arange(1 << f.nb)))


def lg2_fraction(m: int, c0: str = "C0") -> int:
    """v(m): the fraction of log2(1.m) in units of 2^-36, 0 at m = 0."""
    if m == 0:
        return 0
    f = lg2_fit(c0)
    g, u = m >> f.nb, m & ((1 << f.nb) - 1)
    return (f.C0[g] + f.C1[g] * u + ((f.C2[g] * _square()[u]) << f.square_shift)) >> f.shift


def mufu_lg2_bits(x: int, c0: str = "C0") -> int:
    """`lg2.approx.ftz.f32` on an f32 word (the module docstring's rule)."""
    x &= 0xFFFFFFFF
    E, m = (x >> 23) & 0xFF, x & 0x7FFFFF
    if E == 0:
        return NEG_INF
    if x >> 31 or (E == 255 and m):
        return QNAN
    if E == 255:
        return INF
    neg = E < 127
    M = (((E + 1) & 0x7F) << 36 | lg2_fraction(m, c0)) ^ (((1 << 43) - 1) if neg else 0)
    if M == 0:
        return 0
    n = M.bit_length()
    mant = (M >> (n - 24) if n >= 24 else M << (24 - n)) & 0x7FFFFF
    return neg << 31 | (n + 90) << 23 | mant


def mufu_lg2_undetermined(x: int) -> bool:
    """Whether the measurements leave the word of ``x`` open: a positive normal x whose word the two ends of C0's range give
    differently, or a power of two below 1 (v(0) = 0 is measured at E = 127 only)."""
    x &= 0xFFFFFFFF
    E, m = (x >> 23) & 0xFF, x & 0x7FFFFF
    if x >> 31 or E in (0, 255):
        return False
    return (m == 0 and E < 127) or mufu_lg2_bits(x) != mufu_lg2_bits(x, "C0_max")


@primitive("MufuLg2Ftz", 1, [("x", F32)], F32, conformance=_LG2_CONF)
def MufuLg2Ftz(x: int) -> int:
    """`lg2.approx.ftz.f32` (MUFU.LG2): the quadratic interpolator of `mufu_lg2.json` inside the exact exponent rule."""
    return mufu_lg2_bits(x)


def _nan(x: int) -> bool:
    return (x & 0x7F800000) == 0x7F800000 and bool(x & 0x7FFFFF)


@primitive("F32NeZeroFtz", 1, [("x", F32)], BIT, conformance="exact-model-tested")
def F32NeZeroFtz(x: int) -> int:
    """`setp.ne.ftz.f32 x, 0` (FSETP.NE.FTZ): x is neither a zero nor a subnormal nor a NaN."""
    return int(bool(x & 0x7F800000) and not _nan(x))


@primitive("F32GtZeroFtz", 1, [("x", F32)], BIT, conformance="exact-model-tested")
def F32GtZeroFtz(x: int) -> int:
    """`setp.gt.ftz.f32 x, 0` (FSETP.GT.FTZ): x positive, not subnormal, not NaN."""
    return int(not x & 0x80000000 and bool(x & 0x7F800000) and not _nan(x))


# --- the split counts ----------------------------------------------------------------------------------------------------------

def num_splits_heuristic(total_mblocks: int, num_sms: int, num_n_blocks: int, num_m_blocks: int, size_one_kv_head: int,
                         is_causal_or_local: bool, max_splits: int) -> int:
    """hopper/heuristics.h `num_splits_heuristic` (host code, IEEE f32 and the one double comparison)."""
    f = np.float32
    if f(total_mblocks) >= f(0.8) * f(num_sms):
        size_l2 = 50 * 1024 * 1024
        if size_one_kv_head > size_l2 and num_m_blocks >= num_sms * 2 and not is_causal_or_local:
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        return 1
    if num_n_blocks <= 4:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff = []
    for ns in range(1, max_splits + 1):
        n_waves = f(total_mblocks * ns) / f(num_sms)
        eff.append(n_waves / f(np.ceil(n_waves)))
    best = max(eff)
    return next((ns for ns, e in enumerate(eff, 1) if float(e) >= 0.85 * float(best)), 1)


def static_num_splits(num_sm: int, max_seqlen_k: int, batch: int, *, heads: int = 8, heads_kv: int = 1, d: int = 64,
                      dv: int = 512, block_m: int = 64, block_n: int = BLOCK_N) -> int:
    """`mha_fwd`'s `params.num_splits` for a varlen decode batch of ``batch`` tokens (seqlen_q 1, so not causal, and no
    `num_splits` passed): `get_num_splits` with the dynamic split's batch of one, 1 above `MAX_SPLIT_BATCH` tokens."""
    if batch > MAX_SPLIT_BATCH:
        return 1
    m_blocks = -(-(heads // heads_kv) // block_m)
    return num_splits_heuristic(heads_kv * m_blocks, num_sm, -(-max_seqlen_k // block_n), m_blocks, max_seqlen_k * (d + dv) * 2,
                                False, 128)


def blocks_per_sm(total_blocks: int, num_sm: int, num_head: int = 1) -> int:
    """`ceilf(float(total_blocks) * 1.1f * float(num_head) / float(num_sm))` on the device: RN products, `div.full.f32` as
    `RN(a * DivFullRcp(b))` (b is a small normal number, so its numerator scaling is the identity)."""
    f = np.float32
    a = f(f(f(total_blocks) * f(1.1)) * f(num_head))
    rcp = np.array([DivFullRcp.evaluate(P.bits_of(f(num_sm)))], np.uint32).view(np.float32)[0]
    return int(np.ceil(f(a * rcp)))


def dynamic_num_splits(valid, num_sm: int, bound: int, block_n: int = BLOCK_N) -> list[int]:
    """Each token's split count (`prepare_varlen_num_blocks`): ``valid`` the batch's valid key counts, ``bound`` its
    `static_num_splits`.  A batch with no key divides by zero there: refused."""
    valid = [int(t) for t in valid]
    if bound <= 1 or len(valid) > MAX_SPLIT_BATCH:
        return [1] * len(valid)
    nblk = [-(-t // block_n) for t in valid]
    bps = blocks_per_sm(sum(nblk), num_sm)
    if bps <= 0:
        raise ValueError(f"dynamic_num_splits: {sum(nblk)} blocks give blocks_per_sm = {bps} (the kernel divides by it)")
    return [max(min((n + bps - 1) // bps, bound), 1) for n in nblk]


def split_blocks(T: int, NS: int, BN: int = BLOCK_N) -> list[tuple[int, int]]:
    """The non-empty splits of T keys in NS (`get_n_block_min_max`): [b0, b1) block ranges in split order."""
    nb = -(-T // BN)
    per = -(-nb // NS)
    return [(s * per, min((s + 1) * per, nb)) for s in range(NS) if s * per < nb]


def check_split_statics(T: int, NS: int) -> None:
    if T < 1 or not 1 <= NS <= LANES:
        raise ValueError(f"SparseMlaHeadSplit: T={T} selected keys in NS={NS} splits (1 <= NS <= {LANES})")


# --- the bodies (one per Definition, over `sparse_mla.Steps`) ------------------------------------------------------------

WORD = {**M.WORD, "lg2": MufuLg2Ftz, "nez": F32NeZeroFtz, "gtz": F32GtZeroFtz, "sel": SC.SelectI32}


def word_steps(B) -> M.Steps:
    return M.Steps(B, WORD, lambda bits: B.call(const(32, bits)), lambda: B.call(const(16, 0)), 1)


def lse_scale_bits(scale: float) -> int:
    """`softmax_scale_log2 * float(M_LN2)` (FMUL.FTZ of two normal words)."""
    f = np.float32
    c = np.array([M.scale_log2_bits(scale), LN2], np.uint32).view(np.float32)
    return P.bits_of(f(c[0] * c[1]))


def split_end(st: M.Steps, S, INV, m, l, o):
    """`SparseMlaSplitEnd`'s body: (o * INV(sum), lse)."""
    tot = st.add(st.add(l[0], l[2]), st.add(l[1], l[3]))
    part = st.B.batch(st._defs["mul"], o, st.B.call(INV, tot), axes=(0, None))
    lse = st.sel(st.nez(tot), st.fma(m, st.word(lse_scale_bits(S.SCALE)), st.mul(st.lg2(tot), st.word(LN2))), st.word(NEG_INF))
    return tuple_of(part, lse)


def _lane_tree(op, lanes: list, keys: list):
    """Lane 0's value after the warp butterfly of ``op`` over ``lanes`` (``op(own, partner)``, offsets LANES/2 .. 1), each
    distinct (own, partner) pair computed once (``keys`` name the lanes' values; equal keys are one value)."""
    memo: dict = {}
    off = len(lanes) // 2
    while off:
        nxt, nk = [], []
        for i in range(off):
            k = (keys[i], keys[i + off])
            if k not in memo:
                memo[k] = op(lanes[i], lanes[i + off])
            nxt.append(memo[k])
            nk.append(k)
        lanes, keys, off = nxt, nk, off // 2
    return lanes[0]


def combine(st: M.Steps, S, INV, lse, o):
    """`SparseMlaCombine`'s body: the warp's max and sum over NSPLIT lanes and the -inf pads, the weights, the ordered FMAs."""
    NSPLIT, DV, B = S.NSPLIT, S.DV, st.B
    if not 1 <= NSPLIT <= LANES:
        raise ValueError(f"SparseMlaCombine: NSPLIT={NSPLIT} splits, the warp holds 1 to {LANES}")
    pad = st.word(NEG_INF)
    vals, keys = [lse[i] for i in range(NSPLIT)] + [pad] * (LANES - NSPLIT), list(range(NSPLIT)) + ["pad"] * (LANES - NSPLIT)
    cur = st.guard(_lane_tree(st.max, vals, keys))
    log2e, zero = st.word(LOG2E), st.word(0)
    scale = [st.ex2(st.mul(st.sub(lse[i], cur), log2e)) for i in range(NSPLIT)]
    pad_scale = st.add(zero, st.ex2(st.mul(st.sub(pad, cur), log2e))) if NSPLIT < LANES else None
    terms = [st.add(zero, s) for s in scale] + [pad_scale] * (LANES - NSPLIT)
    inv = B.call(INV, _lane_tree(st.add, terms, keys))
    acc = array_of([zero] * DV)
    for s in range(NSPLIT):
        w = st.mul(scale[s], inv)
        acc = B.batch(st._defs["sel"], st.gtz(w), B.batch(st._defs["fma"], w, o[s], acc, axes=(None, 0, 0)), acc,
                      axes=(None, 0, 0))
    return B.batch(st._defs["cvt"], acc, axes=(0,))


def head_split(st: M.Steps, S, block_of, unsplit, end, combine_of, q_nope, q_rope, kv):
    """`SparseMlaHeadSplit`'s body: ``unsplit`` the token's head when NS = 1; else each split's blocks (``block_of(nvis,
    first)``), ``end`` and ``combine_of(nsplit)``."""
    B, T, NS, DR, DV, BN = st.B, S.T, S.NS, S.DR, S.DV, S.BN
    check_split_statics(T, NS)
    if NS == 1:
        return B.call(unsplit, q_nope, q_rope, kv)
    KW = DR + DV
    q = Coll(Array(KW, st.bf16), concat([q_rope.refs, q_nope.refs]))
    row = None
    lses, parts = [], []
    for b0, b1 in split_blocks(T, NS, BN):
        m = l = o = None
        for b in range(b1 - 1, b0 - 1, -1):
            lo, hi = b * BN, min(T, (b + 1) * BN)
            kvb = kv[lo:hi]
            if hi - lo < BN:  # FA3 clears the out-of-bounds rows; their columns are masked and never read
                if row is None:
                    row = Coll(Array(KW, st.bf16), concat([st.zero16().refs] * KW))
                kvb = Coll(Array(BN, Array(KW, st.bf16)), concat([kvb.refs] + [row.refs] * (BN - hi + lo)))
            first = b == b1 - 1
            r = B.call(block_of(hi - lo, True), q, kvb) if first else B.call(block_of(hi - lo, False), q, kvb, m, l, o)
            m, l, o = r[0], r[1], r[2]
        e = B.call(end, m, l, o)
        parts.append(e[0])
        lses.append(e[1])
    return B.call(combine_of(len(parts)), array_of(lses), array_of(parts))


# --- the Definitions --------------------------------------------------------------------------------------------------------

def _end_type(S, f32=F32):
    return ((("m", f32), ("l", Array(4, f32)), ("o", Array(S.DV, f32))), Tuple(Array(S.DV, f32), f32))


@composite("SparseMlaSplitEnd", 1, ["DV", "INV", "SCALE"], _end_type, conformance=_CONF)
def SparseMlaSplitEnd(B, S, m, l, o):
    """A split's end (`Softmax::finalize` at Split): sum = (l0 + l2) + (l1 + l3), the partial output o * INV(sum), and lse =
    sum != 0 ? fma(m, scale_log2 * LN2, MufuLg2Ftz(sum) * LN2) : -inf (`F32NeZeroFtz_v1`, `SelectI32_v1`)."""
    return split_end(word_steps(B), S, S.INV, m, l, o)


def _combine_type(S, bf16=BF16, f32=F32):
    return ((("lse", Array(S.NSPLIT, f32)), ("o", Array(S.NSPLIT, Array(S.DV, f32)))), Array(S.DV, bf16))


@composite("SparseMlaCombine", 1, ["NSPLIT", "DV", "INV"], _combine_type, conformance=_CONF)
def SparseMlaCombine(B, S, lse, o):
    """`FlashAttnFwdCombine` for one query row of NSPLIT splits: the warp max of the lse (lane 0's `F32Max_v1` butterfly over
    32 lanes, the lanes past NSPLIT -inf) guarded, the scales `MufuEx2Ftz((lse - max) * LOG2E)`, the butterfly sum of `0 + scale`,
    the weights `scale * INV(sum)`, then `acc = w > 0 ? fma(w, o_s, acc) : acc` over the splits in order from +0, bf16(acc)."""
    return combine(word_steps(B), S, S.INV, lse, o)


def _head_split_type(S, bf16=BF16):
    return M._head_type(S, bf16)


@composite("SparseMlaHeadSplit", 1, ["T", "NS", "DR", "DV", "BN", "DOT", "INV", "SCALE"], _head_split_type, conformance=_CONF)
def SparseMlaHeadSplit(B, S, q_nope, q_rope, kv):
    """One query head over T selected KV rows in NS splits (`split_blocks`): `SparseMlaHead_v1` at NS = 1; else each split's
    `SparseMlaBlock_v1`s last to first (its first visited one FIRST, the token's last block zero-padded), its
    `SparseMlaSplitEnd_v1`, and `SparseMlaCombine_v1` over the non-empty splits."""
    def block_of(nvis, first):
        return bind(M.SparseMlaBlock, DR=S.DR, DV=S.DV, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    unsplit = bind(M.SparseMlaHead, T=S.T, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    end = bind(SparseMlaSplitEnd, DV=S.DV, INV=S.INV, SCALE=S.SCALE)
    return head_split(word_steps(B), S, block_of, unsplit, end, lambda n: bind(SparseMlaCombine, NSPLIT=n, DV=S.DV, INV=S.INV),
                      q_nope, q_rope, kv)


def _split_type(S, bf16=BF16):
    return M._mla_type(S, bf16)


@composite("SparseMlaSplit", 1, ["T", "NS", "NH", "DR", "DV", "BN", "DOT", "INV", "SCALE"], _split_type, conformance=_CONF)
def SparseMlaSplit(B, S, q_nope, q_rope, kv):
    """One query token's NH heads over the same T selected rows in its NS splits: head h `SparseMlaHeadSplit_v1`."""
    hd = bind(SparseMlaHeadSplit, T=S.T, NS=S.NS, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    return B.batch(hd, q_nope, q_rope, kv, axes=(0, 0, None))


SPARSE_MLA_SPLIT_COMPOSITES = [SparseMlaSplitEnd, SparseMlaCombine, SparseMlaHeadSplit, SparseMlaSplit]

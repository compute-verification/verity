"""torch's FlashAttention-2 forward as a training step runs it on cc 12.0: `aten._scaled_dot_product_flash_attention` (causal,
no dropout, the default scale) of bf16 values, torch's `pytorch_flash` kernels, one Definition per head.

    Fa2TrainFirst_v1{D,BN,NVIS}         a row's first visited key block, its diagonal one: NVIS keys visible, the rest masked
    Fa2TrainBlock_v1{D,BN}              a full key block after the first, from the running max, lane sums and accumulator
    Fa2TrainRow_v1{D,BN,NB,NVIS}        a query row of `flash_fwd_kernel` over NB blocks, the diagonal one first, then NB - 2
                                        down to 0: O (bf16) and the LSE
    Fa2TrainHead_v1{T,D,BN}             a head of T rows of `flash_fwd_kernel` (BM = BN)
    Fa2TrainPart_v1{D,BN,NVIS}          one split of `flash_fwd_splitkv_kernel`, one block: its O (f32, normalized) and LSE
    Fa2TrainCombine_v1{D}               `flash_fwd_splitkv_combine_kernel` over the four splits
    Fa2TrainSplitRow_v1{D,BN,NS,NVIS}   a query row over its NS non-empty splits (the last one its diagonal block) and the combine
    Fa2TrainSplitHead_v1{T,D,BN}        a head of T = 4 BN rows in four splits

torch builds these kernels without fast math, so every f32 operation is the GPU's without flush-to-zero: FADD, FMUL, FFMA, FMNMX,
`ex2.approx.f32` (`dsv4_moe.Ex2Approx`) and the combine's `expf` (`moe.NvExpf`).  The serving kernels flush and contract the
exponent (`fa2_check_inf`); these don't.  A block's scores are `DotBf16_v2{D, HopperBF16WgmmaDot16}`, a masked key's -inf; its
max is FMNMX over the visible scores (and -inf once if a key is masked); past the first block the max folds in the running one
and the rescale `ex2((m_prev - m) * L)` reads it unguarded (FA2's `Check_inf` is off past the masking step); each probability is
`ex2(s * L - guard(m) * L)`, an FMUL then an FADD, a masked one +0; lane j of four sums the probabilities of columns c with
(c mod 8) div 2 = j in ascending c, the first add past the first block contracted with the rescale; P V is `D` k16 chains over
the block's every key, a masked key's V row read (+0 times an inf is a NaN).  The row's end: total (l0 + l2) + (l1 + l3); a
total of 0 or NaN reads 1 as its inverse and +inf as its LSE (-inf as a split's), else O = o * RN(1 / total) and LSE =
fma(m, D^-0.5, lg2(total) * ln 2) (the total is at least 1 then, so MUFU.LG2's flush is never reached).  The combine: lmax =
FMNMX of the splits' LSEs, guarded; e_s = expf(lse_s - lmax); lsum = (e0 + e2) + (e1 + e3); LSE = +inf at a lsum of 0 or NaN,
else logf(lsum) + lmax (lsum is at least 1, so `NvLogf_v2`'s flush is never reached); O = bf16 of the ascending fma chain of
expf(lse_s - LSE) times the split's O.  An empty split is O 0 and LSE -inf.
"""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.defs import bind, composite
from verity.ml.circuits.ops import scalar as SC
from verity.ml.circuits.ops.gemm import DotBf16 as DotBf16V2
from verity.ml.circuits.ops.prims import HopperBF16WgmmaDot16, ZERO32, const
from verity.ml.circuits.refs import Affine, Coll as _Coll, Explicit, array_of, concat, strided_view, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.definitions.evidence import EVIDENCE
from verity.ml.definitions.registry import prims as P
from verity.ml.definitions.registry.b1 import NEG_INF32, ZERO16, _lane, scale_log2_bits
from verity.ml.definitions.registry.dsv4_moe import Ex2Approx
from verity.ml.definitions.registry.moe import NvExpf
from verity.ml.definitions.registry.prims import A16, BF16, F32
from verity.ml.definitions.registry.sparse_mla_split import MufuLg2Ftz
from verity.ml.definitions.registry.topp_word_gates import NvLogfV2

#: the split-KV kernel's split count at every shape it serves here
SPLITS = 4
POS_INF, ONE, LN2 = 0x7F800000, 0x3F800000, 0x3F317218

#: torch's flash attention forward at each (B, H, T, D) the capture covers, on 188 SMs: four splits of one block of 256 keys
#: each where B * H * T / 128 < 0.8 * 188 (`flash_fwd_splitkv_kernel` and its combine), else `flash_fwd_kernel` at BM = BN = 128
TRAIN_SDPA = {(2, 8, 1024, 64): ("split", 256), (2, 12, 1024, 64): ("unsplit", 128), (2, 16, 1024, 64): ("unsplit", 128),
              (2, 24, 1024, 64): ("unsplit", 128)}

_FA2 = EVIDENCE["cc12_fa2_forward_capture"]
CONFORMANCE = ("exact-model-tested: every word of O and the LSE of torch's flash attention forward "
               "(aten._scaled_dot_product_flash_attention, causal, bf16, head dim 64) on cc 12.0 with 188 SMs (torch "
               "2.14.0+cu130) at B = 2, T = 1,024 and H = 12, 16 and 24 (`flash_fwd_kernel`, BM = BN = 128) and H = 8 "
               "(`flash_fwd_splitkv_kernel`, four splits of 256 keys, and the combine), on each batch member's first and last "
               "heads with q and k N(0, 1) and N(0, 9) and v N(0, 1) "
               f"(capture {_FA2['run_record']}, fits {_FA2['fit']} and {_FA2['fit_split']}), and on heads whose words tell each "
               "flush-to-zero apart from none, and a masked key's V row read from one not read "
               f"(captures {_FA2['probe']} and {_FA2['probe_ftz']}, fits {_FA2['fit_ftz']})")


def _refuse(b: dict) -> str | None:
    D, BN, NVIS, NB, NS = (b.get(n) for n in ("D", "BN", "NVIS", "NB", "NS"))
    if D is not None and (D < 16 or D % 16):
        return f"D = {D}: the scores are whole k16 steps (D a positive multiple of 16)"
    if BN is not None and (BN < 16 or BN % 16):
        return f"BN = {BN}: P V is whole k16 steps over the block's keys (BN a positive multiple of 16)"
    if NVIS is not None and BN is not None and not 1 <= NVIS <= BN:
        return f"NVIS = {NVIS}: a row's diagonal block shows 1 to BN = {BN} keys"
    if NB is not None and NB < 1:
        return f"NB = {NB}: a row visits at least its diagonal block"
    if NS is not None and not 1 <= NS <= SPLITS:
        return f"NS = {NS}: a row has 1 to {SPLITS} non-empty splits"
    return None


def _refuse_head(b: dict) -> str | None:
    T, BN = b.get("T"), b.get("BN")
    if T is not None and BN is not None and (T < BN or T % BN):
        return f"T = {T}: the head is whole blocks of BN = {BN} rows"
    return _refuse(b)


def _refuse_split_head(b: dict) -> str | None:
    T, BN = b.get("T"), b.get("BN")
    if T is not None and BN is not None and T != SPLITS * BN:
        return f"T = {T}: the split head is {SPLITS} splits of one block of BN = {BN} keys"
    return _refuse(b)


def _c(B, bits: int):
    return B.call(const(32, bits))


def _sub(B, a, b):
    """a - b: FADD of a and b negated."""
    return B.call(P.F32AddV2, a, B.call(SC.F32Neg, b))


def _bad(B, x):
    """x is 0 or NaN."""
    return B.call(SC.BitOr, B.call(SC.F32Eq, x, B.call(ZERO32)), B.call(SC.BitNot, B.call(SC.F32Eq, x, x)))


def _zeros(B, n: int, zero=None):
    z = zero if zero is not None else B.call(ZERO32)
    return _Coll(Array(n, F32), concat([z.refs] * n))


def _block(B, D: int, BN: int, NVIS: int, first: bool, q, k, v, m_prev=None, l_prev=None, o_prev=None):
    L = _c(B, scale_log2_bits(D))
    dot = bind(DotBf16V2, K=D, DOT=HopperBF16WgmmaDot16)
    s = [B.call(dot, q, k[c]) for c in range(NVIS)]
    tmax = s[0]
    for c in range(1, NVIS):
        tmax = B.call(SC.F32Fmaxf, tmax, s[c])
    if NVIS < BN:
        tmax = B.call(SC.F32Fmaxf, tmax, B.call(NEG_INF32))
    m = tmax if first else B.call(SC.F32Fmaxf, m_prev, tmax)
    if not first:
        f = B.call(Ex2Approx, B.call(P.F32MulV2, _sub(B, m_prev, m), L))
    ms = B.call(P.F32MulV2, B.call(P.GuardNegInfZero, m), L)
    p = [B.call(Ex2Approx, _sub(B, B.call(P.F32MulV2, s[c], L), ms)) for c in range(NVIS)]
    zero = None
    lanes = []
    for ln in range(4):
        cols = [c for c in range(NVIS) if _lane(c) == ln]
        if not first:
            acc = B.call(P.F32FmaV2, l_prev[ln], f, p[cols[0]])
        elif cols:
            acc = p[cols[0]]
        else:
            zero = zero if zero is not None else B.call(ZERO32)
            acc = zero
        for c in cols[1:]:
            acc = B.call(P.F32AddV2, acc, p[c])
        lanes.append(acc)
    pb = [B.call(P.F2fpBf16, x) for x in p]
    if NVIS < BN:
        pb += [B.call(ZERO16)] * (BN - NVIS)
    o = _zeros(B, D, zero) if first else B.batch(P.F32MulV2, o_prev, f, axes=(0, None))
    for ks in range(BN // 16):
        k0 = 16 * ks
        prow = _Coll(A16, concat([pb[k0 + j].refs for j in range(16)]))
        vblk = v[k0:k0 + 16]
        vt = strided_view(vblk, 0, 16, 1, D, 16 * D) if isinstance(vblk.refs, Affine) else \
            Explicit(tuple(vblk.refs[j * D + d] for d in range(D) for j in range(16)))
        o = B.batch(HopperBF16WgmmaDot16, o, prow, _Coll(Array(D, A16), vt), axes=(0, None, 0))
    return tuple_of(m, array_of(lanes), o)


def _end(B, D: int, st, bad_word: int):
    """A row's (or a split's) state -> (o / total, LSE), `bad_word` the LSE at a total of 0 or NaN."""
    m, l, o = st[0], st[1], st[2]
    tot = B.call(P.F32AddV2, B.call(P.F32AddV2, l[0], l[2]), B.call(P.F32AddV2, l[1], l[3]))
    bad = _bad(B, tot)
    one = _c(B, ONE)
    inv = B.call(P.SelectF32, bad, one, B.call(P.F32DivV2, one, tot))
    lg = B.call(P.F32MulV2, B.call(MufuLg2Ftz, tot), _c(B, LN2))
    lse = B.call(P.F32FmaV2, m, _c(B, P.bits_of(np.float32(D ** -0.5))), lg)
    return B.batch(P.F32MulV2, o, inv, axes=(0, None)), B.call(P.SelectF32, bad, _c(B, bad_word), lse)


def _block_type(S, state: bool):
    kv = Array(S.BN, Array(S.D, BF16))
    params = (("q", Array(S.D, BF16)), ("k", kv), ("v", kv))
    if state:
        params += (("m_prev", F32), ("l_prev", Array(4, F32)), ("o_prev", Array(S.D, F32)))
    return params, Tuple(F32, Array(4, F32), Array(S.D, F32))


def _row_type(S, n: int):
    kv = Array(n, Array(S.D, BF16))
    return (("q", Array(S.D, BF16)), ("k", kv), ("v", kv)), Tuple(Array(S.D, BF16), F32)


def _head_type(S):
    rows = Array(S.T, Array(S.D, BF16))
    return (("q", rows), ("k", rows), ("v", rows)), Tuple(rows, Array(S.T, F32))


@composite("Fa2TrainFirst", 1, ["D", "BN", "NVIS"], lambda S: _block_type(S, False), conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainFirst(B, S, q, k, v):
    """A row's first visited block, its diagonal one: keys 0..NVIS-1 of the block visible, the rest masked.  Returns the max,
    the four lane sums and the unnormalized accumulator."""
    return _block(B, S.D, S.BN, S.NVIS, True, q, k, v)


@composite("Fa2TrainBlock", 1, ["D", "BN"], lambda S: _block_type(S, True), conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainBlock(B, S, q, k, v, m_prev, l_prev, o_prev):
    """A full key block after a row's first: the running max, lane sums and accumulator in, the same out."""
    return _block(B, S.D, S.BN, S.BN, False, q, k, v, m_prev, l_prev, o_prev)


@composite("Fa2TrainRow", 1, ["D", "BN", "NB", "NVIS"], lambda S: _row_type(S, S.NB * S.BN), conformance=CONFORMANCE,
           refuse=_refuse)
def Fa2TrainRow(B, S, q, k, v):
    """A query row over keys 0..NB*BN-1: `Fa2TrainFirst_v1{NVIS}` on block NB - 1, `Fa2TrainBlock_v1` on NB - 2 down to 0, then
    O = bf16(o / total) and the LSE (+inf at a total of 0 or NaN)."""
    D, BN, NB = S.D, S.BN, S.NB
    lo = (NB - 1) * BN
    st = B.call(bind(Fa2TrainFirst, D=D, BN=BN, NVIS=S.NVIS), q, k[lo:lo + BN], v[lo:lo + BN])
    block = bind(Fa2TrainBlock, D=D, BN=BN)
    for b in range(NB - 2, -1, -1):
        st = B.call(block, q, k[b * BN:(b + 1) * BN], v[b * BN:(b + 1) * BN], st[0], st[1], st[2])
    o, lse = _end(B, D, st, POS_INF)
    return tuple_of(B.batch(P.F2fpBf16, o, axes=(0,)), lse)


@composite("Fa2TrainHead", 1, ["T", "D", "BN"], _head_type, conformance=CONFORMANCE, refuse=_refuse_head)
def Fa2TrainHead(B, S, q, k, v):
    """Causal attention of a head's T rows: row r is `Fa2TrainRow_v1{NB = r div BN + 1, NVIS = r mod BN + 1}` over keys
    0..NB*BN-1.  Returns O (T rows) and the LSE (T words)."""
    outs, lses = [], []
    for r in range(S.T):
        nb = r // S.BN + 1
        y = B.call(bind(Fa2TrainRow, D=S.D, BN=S.BN, NB=nb, NVIS=r % S.BN + 1), q[r], k[0:nb * S.BN], v[0:nb * S.BN])
        outs.append(y[0])
        lses.append(y[1])
    return tuple_of(array_of(outs), array_of(lses))


@composite("Fa2TrainPart", 1, ["D", "BN", "NVIS"],
           lambda S: (_block_type(S, False)[0], Tuple(Array(S.D, F32), F32)), conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainPart(B, S, q, k, v):
    """One split of one block (`Fa2TrainFirst_v1{NVIS}`): its O normalized, in f32, and its LSE (-inf at a total of 0 or
    NaN)."""
    st = B.call(bind(Fa2TrainFirst, D=S.D, BN=S.BN, NVIS=S.NVIS), q, k, v)
    o, lse = _end(B, S.D, st, 0xFF800000)
    return tuple_of(o, lse)


@composite("Fa2TrainCombine", 1, ["D"],
           lambda S: ((("o", Array(SPLITS, Array(S.D, F32))), ("lse", Array(SPLITS, F32))), Tuple(Array(S.D, BF16), F32)),
           conformance=CONFORMANCE, refuse=_refuse)
def Fa2TrainCombine(B, S, o, lse):
    """The four splits' O and LSE -> the row's O (bf16) and LSE."""
    lmax = lse[0]
    for s in range(1, SPLITS):
        lmax = B.call(SC.F32Fmaxf, lmax, lse[s])
    lmax = B.call(P.GuardNegInfZero, lmax)
    e = [B.call(NvExpf, _sub(B, lse[s], lmax)) for s in range(SPLITS)]
    lsum = B.call(P.F32AddV2, B.call(P.F32AddV2, e[0], e[2]), B.call(P.F32AddV2, e[1], e[3]))
    logsum = B.call(P.SelectF32, _bad(B, lsum), _c(B, POS_INF), B.call(P.F32AddV2, B.call(NvLogfV2, lsum), lmax))
    acc = _zeros(B, S.D)
    for s in range(SPLITS):
        acc = B.batch(P.F32FmaV2, B.call(NvExpf, _sub(B, lse[s], logsum)), o[s], acc, axes=(None, 0, 0))
    return tuple_of(B.batch(P.F2fpBf16, acc, axes=(0,)), logsum)


@composite("Fa2TrainSplitRow", 1, ["D", "BN", "NS", "NVIS"], lambda S: _row_type(S, S.NS * S.BN), conformance=CONFORMANCE,
           refuse=_refuse)
def Fa2TrainSplitRow(B, S, q, k, v):
    """A query row over keys 0..NS*BN-1: split s < NS is `Fa2TrainPart_v1` on block s (NVIS keys visible in the last, all in the
    others), splits NS..3 empty, then `Fa2TrainCombine_v1`."""
    D, BN, NS = S.D, S.BN, S.NS
    parts = [B.call(bind(Fa2TrainPart, D=D, BN=BN, NVIS=S.NVIS if s == NS - 1 else BN), q, k[s * BN:(s + 1) * BN],
                    v[s * BN:(s + 1) * BN]) for s in range(NS)]
    os, ls = [p[0].refs for p in parts], [p[1].refs for p in parts]
    if NS < SPLITS:
        os += [_zeros(B, D).refs] * (SPLITS - NS)
        ls += [B.call(NEG_INF32).refs] * (SPLITS - NS)
    return B.call(bind(Fa2TrainCombine, D=D), _Coll(Array(SPLITS, Array(D, F32)), concat(os)),
                  _Coll(Array(SPLITS, F32), concat(ls)))


@composite("Fa2TrainSplitHead", 1, ["T", "D", "BN"], _head_type, conformance=CONFORMANCE, refuse=_refuse_split_head)
def Fa2TrainSplitHead(B, S, q, k, v):
    """Causal attention of a head's T = 4 BN rows in four splits: row r is `Fa2TrainSplitRow_v1{NS = r div BN + 1, NVIS =
    r mod BN + 1}` over keys 0..NS*BN-1."""
    outs, lses = [], []
    for r in range(S.T):
        ns = r // S.BN + 1
        y = B.call(bind(Fa2TrainSplitRow, D=S.D, BN=S.BN, NS=ns, NVIS=r % S.BN + 1), q[r], k[0:ns * S.BN],
                   v[0:ns * S.BN])
        outs.append(y[0])
        lses.append(y[1])
    return tuple_of(array_of(outs), array_of(lses))


def train_sdpa(B: int, H: int, T: int, D: int):
    """The head Definition of torch's flash attention forward at (B, H, T, D) on cc 12.0, bound (`TRAIN_SDPA`); ValueError at a
    shape the capture doesn't cover."""
    route = TRAIN_SDPA.get((B, H, T, D))
    if route is None:
        raise ValueError(f"(B, H, T, D) = {(B, H, T, D)} is not captured: torch's flash attention forward is fitted at "
                         f"{sorted(TRAIN_SDPA)}")
    kind, BN = route
    return bind(Fa2TrainSplitHead if kind == "split" else Fa2TrainHead, T=T, D=D, BN=BN)


__all__ = ["CONFORMANCE", "Fa2TrainBlock", "Fa2TrainCombine", "Fa2TrainFirst", "Fa2TrainHead", "Fa2TrainPart", "Fa2TrainRow",
           "Fa2TrainSplitHead", "Fa2TrainSplitRow", "SPLITS", "TRAIN_SDPA", "train_sdpa"]

"""The word twins of `mla`'s Boolean MLA decode kernels: the same bodies on f32 / bf16 words, for word Programs (the Build and
Match evaluate words; circuit-check proves the Boolean). Each step is the core word primitive the Boolean step computes:

    TritonExpSubWord_v1                   (x, m: F32) -> F32      ex2_approx(RN(RN(x - m) * 0x3FB8AA3B)), `TritonExpSub_v1`
    TritonMlaScoreWord_v1{K, SCALE, DOT}  (q, k: Array<K, BF16>) -> F32           RN(DotBf16_v2{K, DOT}(q, k) * SCALE)
    TritonMlaBlockWord_v1{DL, DR, NVIS, BN, FIRST, SCALE, DOT}                    `TritonMlaBlock_v1`
    TritonMlaGateWord_v1                  (e_max, e_sum: F32) -> Value<1>         `TritonMlaGate_v1`
    TritonMlaOutWord_v1                   (acc, e_sum, rcp: F32, gate: Value<1>) -> BF16                  `TritonMlaOut_v1`
    TritonMlaDecodeHeadWord_v1{T, DL, DR, BN, SPLITS, SCALE, DOT}                 `TritonMlaDecodeHead_v1`
    TritonMlaDecodeWord_v1{H, T, DL, DR, BN, SPLITS, SCALE, DOT}                  `TritonMlaDecode_v1`
    MlaBmmWord_v1{NHEADS, K, N, DOT}                                              `MlaBmm_v1`: Gemm_v2{K, N, DOT} per head

DOT is the word k16 step (`HopperBF16WgmmaDot16_v1` on sm_120, the word of the Boolean `HopperBF16WgmmaDot16_v2`).
``x - m`` is `F32Sub_v2` (FSUB); the Boolean's canonical NaNs and the word FMUL's propagated ones differ only in payloads that
``MufuEx2Ftz`` or ``F2fpBf16`` maps to one word, so the twins agree on every input word.  The first block (``FIRST=1``) starts
from ``(-inf, 0, 0)`` itself, so its ``EXP(-inf - n)`` and ``0 * re`` are each computed once (the Boolean computes them per
masked column and per coordinate, the same words).
The gate reads e_max finite as ``-inf < e_max < +inf`` and e_sum a positive normal as ``0x007FFFFF < e_sum < +inf`` (NaN compares
false). The output's ``+ 0`` turns a -0 quotient into +0, as the Boolean's cleared sign.
"""
from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.refs import Coll, concat, strided_view, tuple_of
from verity.ir.types import Array, Tuple
from verity.ml import fp32 as W
from verity.ml import gemm as G
from verity.ml import mufu as M
from verity.ml import scalar as SC
from verity.ml.prims import ZERO32, F2fpBf16, const
from verity_vllm.program.registry.prims import A16, BF16, BIT, F32, F32SubV2
from verity_vllm.program.registry.ref_prims import F32GtStrict

__all__ = ["MlaBmmWord", "TritonExpSubWord", "TritonMlaBlockWord", "TritonMlaDecodeHeadWord", "TritonMlaDecodeWord",
           "TritonMlaGateWord", "TritonMlaOutWord", "TritonMlaScoreWord", "MLA_WORD_COMPOSITES"]

NEG_INF = 0xFF800000
POS_INF = 0x7F800000
MAX_SUBNORMAL = 0x007FFFFF
LOG2E = 0x3FB8AA3B
HALF = 0x3F000000
MINUS_126 = 0xC2FC0000
NAN_F2FP = 0x7FFF
BLOCK_N = 32
CONFORMANCE = "exact-model-tested: equals mla_check's model, as the Boolean twin"


def _check(name: str, **ok: bool) -> None:
    bad = [k for k, v in ok.items() if not v]
    if bad:
        raise ValueError(f"{name}: unsupported statics ({', '.join(bad)})")


def _c32(B, w: int) -> Coll:
    return B.call(const(32, w))


@composite("TritonExpSubWord", 1, [], lambda S: ((("x", F32), ("m", F32)), F32), conformance=CONFORMANCE)
def TritonExpSubWord(B, S, x, m):
    """Triton's ``tl.exp(x - m)`` on sm_120: ``t = RN(RN(x - m) * log2 e)``, then ``t < -126 ? MUFU.EX2(t * 0.5)^2 : MUFU.EX2(t)``."""
    t = B.call(W.F32Mul, B.call(F32SubV2, x, m), _c32(B, LOG2E))
    small = B.call(F32GtStrict, _c32(B, MINUS_126), t)
    y = B.call(M.MufuEx2Ftz, B.call(SC.SelectF32, small, B.call(W.F32Mul, t, _c32(B, HALF)), t))
    return B.call(SC.SelectF32, small, B.call(W.F32Mul, y, y), y)


@composite("TritonMlaScoreWord", 1, ["K", "SCALE", "DOT"],
           lambda S: ((("q", Array(S.K, BF16)), ("k", Array(S.K, BF16))), F32), conformance=CONFORMANCE)
def TritonMlaScoreWord(B, S, q, k):
    """``RN(DotBf16_v2{K, DOT}(q, k) * SCALE)``: one key's scaled score."""
    return B.call(W.F32Mul, B.call(bind(G.DotBf16, K=S.K, DOT=S.DOT), q, k), _c32(B, S.SCALE))


class _Memo:
    """Calls by callee and operand objects, so a body computes no value twice (the masked columns' sums)."""

    def __init__(self, B):
        self.B, self.done = B, {}

    def __call__(self, fn, *args):
        key = (fn.id, *map(id, args))
        if key not in self.done:
            self.done[key] = (self.B.call(fn, *args), args)
        return self.done[key][0]


def _block_type(S):
    D = S.DL + S.DR
    state = () if S.FIRST else (("e_max", F32), ("e_sum", F32), ("acc", Array(S.DL, F32)))
    return ((("q", Array(D, BF16)), ("kv", Array(S.NVIS, Array(D, BF16))), *state), Tuple(F32, F32, Array(S.DL, F32)))


@composite("TritonMlaBlockWord", 1, ["DL", "DR", "NVIS", "BN", "FIRST", "SCALE", "DOT"], _block_type, conformance=CONFORMANCE)
def TritonMlaBlockWord(B, S, q, kv, *state):
    """One BN-key block of `_fwd_grouped_kernel_stage1` for one head, NVIS visible keys (the rest masked: score -inf, value row
    +0): ``n = max(scores, e_max)``, ``re = EXP(e_max - n)``, ``p = EXP(s - n)``, ``acc = DOT steps over RN(acc * re)``,
    ``e_sum = fma(e_sum, re, rowsum(p))``; returns ``(n, e_sum, acc)``.  ``FIRST=1`` takes no state: ``(e_max, e_sum, acc)`` is
    ``(-inf, 0, 0)``."""
    DL, D, NVIS, BN = S.DL, S.DL + S.DR, S.NVIS, S.BN
    _check("TritonMlaBlockWord_v1", BN_is_32=BN == BLOCK_N, NVIS_in_1_BN=1 <= NVIS <= BN,
           DL_DR_multiples_of_16=DL % 16 == 0 == S.DR % 16, FIRST_is_0_or_1=S.FIRST in (0, 1))
    call = _Memo(B)
    s = B.batch(bind(TritonMlaScoreWord, K=D, SCALE=S.SCALE, DOT=S.DOT), q, kv, axes=(None, 0))
    m = s[0]
    for j in range(1, NVIS):
        m = B.call(SC.F32Fmaxf, m, s[j])
    ninf = _c32(B, NEG_INF) if NVIS < BN or S.FIRST else None
    if NVIS < BN:
        m = B.call(SC.F32Fmaxf, m, ninf)
    e_max, e_sum, acc = (ninf, B.call(ZERO32), None) if S.FIRST else state
    n = B.call(SC.F32Fmaxf, m, e_max)
    re = call(TritonExpSubWord, e_max, n)
    p = B.batch(TritonExpSubWord, s, n, axes=(0, None))
    pm = call(TritonExpSubWord, ninf, n) if NVIS < BN else None
    cols = [p[j] for j in range(NVIS)] + [pm] * (BN - NVIS)

    def add(a, b):
        return call(W.F32Add, a, b)

    w = []
    for k in range(4):
        x = cols[8 * k:8 * k + 8]
        w.append(add(add(add(x[0], x[1]), add(x[4], x[5])), add(add(x[2], x[3]), add(x[6], x[7]))))
    rowsum = add(add(w[0], w[2]), add(w[1], w[3]))
    if S.FIRST:
        zr = B.call(W.F32Mul, e_sum, re)
        e_sum = B.call(W.F32Fma, e_sum, re, rowsum)
        acc = Coll(Array(DL, F32), concat([zr.refs] * DL))
    else:
        e_sum = B.call(W.F32Fma, e_sum, re, rowsum)
        acc = B.batch(W.F32Mul, acc, re, axes=(0, None))
    pb = B.batch(F2fpBf16, p, axes=(0,))
    pmb = B.call(F2fpBf16, pm) if pm is not None else None
    zero16 = B.call(const(16, 0)) if NVIS < BN else None
    for g in range(BN // 16):
        k0 = 16 * g
        prow = Coll(A16, concat([(pb[k0 + j] if k0 + j < NVIS else pmb).refs for j in range(16)]))
        if k0 + 16 <= NVIS:
            vt = strided_view(kv, k0 * D, 16, 1, D, 16 * DL)
        else:
            vt = concat([(kv[k0 + j][d] if k0 + j < NVIS else zero16).refs for d in range(DL) for j in range(16)])
        acc = B.batch(S.DOT, acc, prow, Coll(Array(DL, A16), vt), axes=(0, None, 0))
    return tuple_of(n, e_sum, acc)


@composite("TritonMlaGateWord", 1, [], lambda S: ((("e_max", F32), ("e_sum", F32)), BIT), conformance=CONFORMANCE)
def TritonMlaGateWord(B, S, e_max, e_sum):
    """One split's ``e_max + logf(e_sum)`` is finite: e_max finite and e_sum a positive normal (FTZ logf)."""
    inf = _c32(B, POS_INF)
    finite = B.call(SC.BitAnd, B.call(F32GtStrict, inf, e_max), B.call(F32GtStrict, e_max, _c32(B, NEG_INF)))
    normal = B.call(SC.BitAnd, B.call(F32GtStrict, e_sum, _c32(B, MAX_SUBNORMAL)), B.call(F32GtStrict, inf, e_sum))
    return B.call(SC.BitAnd, finite, normal)


@composite("TritonMlaOutWord", 1, [], lambda S: ((("acc", F32), ("e_sum", F32), ("rcp", F32), ("gate", BIT)), BF16),
           conformance=CONFORMANCE)
def TritonMlaOutWord(B, S, acc, e_sum, rcp, gate):
    """``gate ? F2fp(div.full(acc, e_sum) + 0) : 0x7FFF``, with ``rcp = DivFullRcp(e_sum)``."""
    t = B.call(W.F32Mul, B.call(M.DivFullScaleA, acc, e_sum), rcp)
    t = B.call(W.F32Add, t, B.call(ZERO32))
    return B.call(SC.SelectBf16, gate, B.call(F2fpBf16, t), B.call(const(16, NAN_F2FP)))


def _head_type(S):
    D = S.DL + S.DR
    return ((("q", Array(D, BF16)), ("kv", Array(S.T, Array(D, BF16)))), Array(S.DL, BF16))


@composite("TritonMlaDecodeHeadWord", 1, ["T", "DL", "DR", "BN", "SPLITS", "SCALE", "DOT"], _head_type, conformance=CONFORMANCE)
def TritonMlaDecodeHeadWord(B, S, q, kv):
    """One head of a one-split Triton MLA decode over T keys: ceil(T / BN) `TritonMlaBlockWord_v1` in ascending order from
    ``(-inf, 0, 0)``, then `TritonMlaOutWord_v1` on each coordinate with ``rcp = DivFullRcp_v1(e_sum)`` and the gate
    `TritonMlaGateWord_v1(e_max, e_sum)`."""
    T, BN = S.T, S.BN
    _check("TritonMlaDecodeHeadWord_v1", SPLITS_is_1=S.SPLITS == 1, BN_is_32=BN == BLOCK_N, T_positive=T >= 1)
    state: tuple = ()
    for lo in range(0, T, BN):
        hi = min(T, lo + BN)
        blk = bind(TritonMlaBlockWord, DL=S.DL, DR=S.DR, NVIS=hi - lo, BN=BN, FIRST=int(lo == 0), SCALE=S.SCALE, DOT=S.DOT)
        st = B.call(blk, q, kv[lo:hi], *state)
        state = (st[0], st[1], st[2])
    e_max, e_sum, acc = state
    rcp = B.call(M.DivFullRcp, e_sum)
    gate = B.call(TritonMlaGateWord, e_max, e_sum)
    return B.batch(TritonMlaOutWord, acc, e_sum, rcp, gate, axes=(0, None, None, None))


def _decode_type(S):
    D = S.DL + S.DR
    return ((("q", Array(S.H, Array(D, BF16))), ("kv", Array(S.T, Array(D, BF16)))), Array(S.H, Array(S.DL, BF16)))


@composite("TritonMlaDecodeWord", 1, ["H", "T", "DL", "DR", "BN", "SPLITS", "SCALE", "DOT"], _decode_type,
           conformance=CONFORMANCE)
def TritonMlaDecodeWord(B, S, q, kv):
    """`TritonMLAImpl.forward_mqa`'s attention for one query token: `TritonMlaDecodeHeadWord_v1` on each of the H heads, the
    request's cache rows shared (one latent KV head)."""
    head = bind(TritonMlaDecodeHeadWord, T=S.T, DL=S.DL, DR=S.DR, BN=S.BN, SPLITS=S.SPLITS, SCALE=S.SCALE, DOT=S.DOT)
    return B.batch(head, q, kv, axes=(0, None))


@composite("MlaBmmWord", 1, ["NHEADS", "K", "N", "DOT"],
           lambda S: ((("x", Array(S.NHEADS, Array(S.K, BF16))), ("w", Array(S.NHEADS, Array(S.N, Array(S.K, BF16))))),
                      Array(S.NHEADS, Array(S.N, BF16))), conformance=CONFORMANCE)
def MlaBmmWord(B, S, x, w):
    """vLLM's batch-invariant `bmm_kernel` for one token row: `Gemm_v2{K, N, DOT}` of ``x[h]`` against ``w[h]`` on each head,
    ``w[h][n][k]`` the right operand at ``[h, k, n]``."""
    return B.batch(bind(G.Gemm, K=S.K, N=S.N, DOT=S.DOT), x, w, axes=(0, 0))


MLA_WORD_COMPOSITES = [TritonExpSubWord, TritonMlaScoreWord, TritonMlaBlockWord, TritonMlaGateWord, TritonMlaOutWord,
                       TritonMlaDecodeHeadWord, TritonMlaDecodeWord, MlaBmmWord]

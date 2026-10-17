"""`sparse_mla_split`'s split sparse MLA attention on bits: version 2 of each Definition, its word body line for line over Boolean
Definitions (the bodies are `sparse_mla_split`'s, written once over `sparse_mla.Steps`), so each one's word view is its `_v1` at
the same statics with the Boolean ``DOT`` and ``INV`` read as their word views.

    MufuLg2Ftz_v2                  (x: Array<32, Value<1>>) -> Array<32, Value<1>>      word view: MufuLg2Ftz_v1
    F32NeZeroFtz_v2, F32GtZeroFtz_v2  (x: Array<32, Value<1>>) -> Value<1>             word view: their _v1
    SparseMlaSplitEnd_v2{DV,INV,SCALE}                    INV: `boolean_attention.Fa3InvSum_v2`
    SparseMlaCombine_v2{NSPLIT,DV,INV}
    SparseMlaHeadSplit_v2{T,NS,DR,DV,BN,DOT,INV,SCALE}    DOT: `gemm.HopperBF16WgmmaDot16_v2`; the blocks and the NS = 1 head are
    SparseMlaSplit_v2{T,NS,NH,DR,DV,BN,DOT,INV,SCALE}     `boolean_sparse_mla`'s

MUFU.LG2 on bits is core's table read (`verity.ml.circuits.ops.boolean.mufu.table_word`) over `mufu_lg2.json`: 2 v as one
carry-save sum of the rows ``u XOR n_k``, ``Q(u) XOR n_k`` (Q the truncated square from column 19, at square_shift 1) and C0 with
the digits' constants folded in, every n_k and C0 bit the XOR of the monomials of the 6 segment bits (`_lg2_plan`), so the
products cost no AND.  Then the module docstring's rule: v cleared at m = 0, M = (t << 36 | v) XOR (E < 127), its leading-zero
count, the 23 bits below the leading one and the exponent 133 - count, and the specials.  The rest are `sparse_mla`'s callees
(`boolean_sparse_mla.BITS`) and `SelectI32_v2`; each composite is one traced circuit (`boolean_dsa.flat`).
"""
from __future__ import annotations

from functools import cache

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import NOT, ONE, OR, ZERO, add, and_reduce, compress_columns, const_bits, lzc, mux, or_reduce, shl_window, sub
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import mufu as BM
from verity.ml.circuits.ops.boolean import scalar as BSC
from verity_vllm.program.registry import boolean_sparse_mla as BSM
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.registry import sparse_mla_split as W
from verity_vllm.program.registry.boolean_attention import BF16, F32, _Bits
from verity_vllm.program.registry.boolean_dsa import flat

BIT = BL.BIT
#: v's bits: log2(1.m) < 1 in units of 2^-36
FRACTION_BITS = 36


# --- MUFU.LG2 ------------------------------------------------------------------------------------------------------------------

@cache
def _lg2_sums() -> np.ndarray:
    """S = C0 + C1 u + (C2 Q(u)) << square_shift at every mantissa (int64, by segment then offset); v = S >> shift."""
    f = W.lg2_fit()
    u = np.arange(1 << f.nb, dtype=np.int64)
    q = f.square(u)
    c0, c1, c2 = (np.asarray(c, np.int64)[:, None] for c in (f.C0, f.C1, f.C2))
    return c0 + c1 * u + ((c2 * q) << f.square_shift)


@cache
def _lg2_plan() -> tuple[tuple[int, int, int], tuple[tuple[int, int], ...], tuple[tuple[int, ...], ...]]:
    """((K1, K2, W), pairs, sums): the digit counts of C1 and C2, the bits W of 2S, and the segment functions (C1's digit signs,
    C2's, C0's W folded bits) as `_shared_sums` of their algebraic normal forms over the segment bits.  The sum is checked
    nonnegative at every mantissa but 0 and v below 2^36, so the circuit's ``t << 36 | v`` is the model's."""
    f = W.lg2_fit()
    S = _lg2_sums().reshape(-1)[1:]
    if S.min() < 0 or S.max() >> f.shift >= 1 << FRACTION_BITS:
        raise ValueError(f"{W.LG2_TABLE}: v leaves [0, 2^{FRACTION_BITS}) at a nonzero mantissa")
    Wd = int(2 * S.max()).bit_length()
    c0, rows, Ks = [2 * c for c in f.C0], [], []
    for n, cs, at in ((f.nb, f.C1, 0), (2 * f.nb - f.p, f.C2, f.square_shift)):
        K, B = BM._digits(cs)
        Ks.append(K)
        rows += [[1 - ((b >> k) & 1) for b in B] for k in range(K)]
        c0 = [c - (((1 << n) - 1) * ((((1 << K) - 1) ^ b) + 1) << at) for c, b in zip(c0, B)]
    rows += [[(c >> j) & 1 for c in c0] for j in range(Wd)]
    anf = np.array(rows, dtype=np.uint8)
    for i in range(f.bits - f.nb):
        anf = anf.reshape(len(rows), -1, 2, 1 << i)
        anf[:, :, 1, :] ^= anf[:, :, 0, :]
    pairs, sums = BM._shared_sums(anf.reshape(len(rows), -1))
    return (Ks[0], Ks[1], Wd), pairs, sums


def lg2_fraction(C, m, m_nz):
    """v(m) on bits (`FRACTION_BITS`, LSB first), 0 where ``m_nz`` (m != 0) is not."""
    f = W.lg2_fit()
    (K1, K2, Wd), pairs, sums = _lg2_plan()
    u = list(m[:f.nb])
    terms = BM._monomials(C, m[f.nb:])
    for a, b in pairs:
        terms.append(terms[a] ^ terms[b])
    fns = [BM._xor_all(terms[i] for i in s) for s in sums]
    cols: list[list] = [[] for _ in range(Wd)]
    # 2 c x = sum_k (x XOR n_k) 2^k + ~x - (2^len(x) - 1)(N + 1), the constant folded into C0 (core's `table_word`)
    for x, n, at in ((u, fns[:K1], 0), (BM.truncated_square(C, u, f.p), fns[K1:K1 + K2], f.square_shift)):
        for k, n_k in enumerate(n):
            for j, b in enumerate(x):
                if j + k + at < Wd:
                    cols[j + k + at].append(b ^ n_k)
        for j, b in enumerate(x):
            if j + at < Wd:
                cols[j + at].append(NOT(b))
    for j, b in enumerate(fns[K1 + K2:]):
        cols[j].append(b)
    v = compress_columns(C, cols)[f.shift + 1:Wd]
    return [C.AND(m_nz, b) for b in v] + [ZERO] * (FRACTION_BITS - len(v))


def mufu_lg2(C, x):
    """``MufuLg2Ftz_v1`` on bits."""
    s, E, m = x[31], list(x[23:31]), list(x[:23])
    lo7 = and_reduce(C, E[:7])
    e_max = C.AND(E[7], lo7)
    neg = NOT(OR(C, E[7], lo7))                                   # E < 127
    e_zero = NOT(or_reduce(C, E))
    m_nz = or_reduce(C, m)
    t = add(C, E[:7], [], ONE, width=7)                           # (E + 1) mod 128
    L = [ZERO] * (64 - FRACTION_BITS - 7) + [b ^ neg for b in lg2_fraction(C, m, m_nz) + t]
    az, lz = lzc(C, L)
    body = shl_window(C, L, lz, 40, 63) + sub(C, const_bits(133, 8), lz + [ZERO] * (8 - len(lz)), 8)[0] + [neg]
    return BM._word(C, [(e_zero, const_bits(W.NEG_INF, 32)), (s, const_bits(W.QNAN, 32)),
                        (e_max, mux(C, m_nz, const_bits(W.QNAN, 32), const_bits(W.INF, 32))), (az, const_bits(0, 32))], body)


def _nonzero(C, x):
    """x is neither a zero nor a subnormal nor a NaN."""
    E = list(x[23:31])
    return C.AND(or_reduce(C, E), NOT(C.AND(and_reduce(C, E), or_reduce(C, x[:23]))))


class _Cse:
    """``C`` whose AND of two forms already ANDed (in either order) reuses that bit, as the trace's does: C-Flock lowers a word
    primitive through its Boolean Definition's builder (`ir_lower._boolean_build`) on a circuit without CSE, and these builders
    take some products twice (an OR's AND is an AND's, the square's and the monomials' are the mantissa OR's)."""

    def __init__(self, C):
        self.C, self.ands = C, {}

    def AND(self, a, b):
        key = frozenset((a.key(), b.key()))
        if key not in self.ands:
            self.ands[key] = self.C.AND(a, b)
        return self.ands[key]

    def __getattr__(self, name):
        return getattr(self.C, name)


def _cse(build):
    return lambda C, *x: build(_Cse(C), *x)


MufuLg2Ftz = definition("MufuLg2Ftz", 2, _cse(mufu_lg2), (("x", F32),), F32, word=W.MufuLg2Ftz, conformance=W.MufuLg2Ftz.conformance,
                        doc="lg2.approx.ftz.f32 on bits: MufuLg2Ftz_v1, the interpolator of mufu_lg2.json as one carry-save sum")
F32NeZeroFtz = definition("F32NeZeroFtz", 2, _cse(lambda C, x: [_nonzero(C, x)]), (("x", F32),), BIT, word=W.F32NeZeroFtz,
                          doc="setp.ne.ftz.f32 x, 0 on bits: F32NeZeroFtz_v1")
F32GtZeroFtz = definition("F32GtZeroFtz", 2, _cse(lambda C, x: [C.AND(NOT(x[31]), _nonzero(C, x))]), (("x", F32),), BIT,
                          word=W.F32GtZeroFtz, doc="setp.gt.ftz.f32 x, 0 on bits: F32GtZeroFtz_v1")


# --- the composites --------------------------------------------------------------------------------------------------------------

BITS = {**BSM.BITS, "lg2": MufuLg2Ftz, "nez": F32NeZeroFtz, "gtz": F32GtZeroFtz, "sel": BSC.SelectI32}
#: GLM-5.3 at TP8 on bits
SPARSE_MLA_GLM53 = BSM.SPARSE_MLA_GLM53


def bit_steps(B) -> M.Steps:
    c = _Bits(B)
    return M.Steps(B, BITS, lambda bits: c.word(bits, 32), lambda: c.word(0, 16), 16, BF16, F32)


def _end(B, S, m, l, o):
    """`SparseMlaSplitEnd_v1`'s body on bits."""
    return W.split_end(bit_steps(B), S, S.INV, m, l, o)


SparseMlaSplitEndV2 = flat("SparseMlaSplitEnd", 2, ["DV", "INV", "SCALE"], lambda S: W._end_type(S, F32), _end,
                           W.SparseMlaSplitEnd, lifted=("INV",))


def _combine(B, S, lse, o):
    """`SparseMlaCombine_v1`'s body on bits."""
    return W.combine(bit_steps(B), S, S.INV, lse, o)


SparseMlaCombineV2 = flat("SparseMlaCombine", 2, ["NSPLIT", "DV", "INV"], lambda S: W._combine_type(S, BF16, F32), _combine,
                          W.SparseMlaCombine, lifted=("INV",))


def _head_split(B, S, q_nope, q_rope, kv):
    """`SparseMlaHeadSplit_v1`'s body on bits over `SparseMlaBlock_v2`, `SparseMlaHead_v2` (NS = 1), `SparseMlaSplitEnd_v2` and
    `SparseMlaCombine_v2`."""
    def block_of(nvis, first):
        return bind(BSM.SparseMlaBlockV2, DR=S.DR, DV=S.DV, NVIS=nvis, FIRST=first, BN=S.BN, DOT=S.DOT, SCALE=S.SCALE)
    unsplit = bind(BSM.SparseMlaHeadV2, T=S.T, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    end = bind(SparseMlaSplitEndV2, DV=S.DV, INV=S.INV, SCALE=S.SCALE)
    return W.head_split(bit_steps(B), S, block_of, unsplit, end,
                        lambda n: bind(SparseMlaCombineV2, NSPLIT=n, DV=S.DV, INV=S.INV), q_nope, q_rope, kv)


SparseMlaHeadSplitV2 = flat("SparseMlaHeadSplit", 2, ["T", "NS", "DR", "DV", "BN", "DOT", "INV", "SCALE"],
                            lambda S: M._head_type(S, BF16), _head_split, W.SparseMlaHeadSplit, lifted=("DOT", "INV"))


def _split(B, S, q_nope, q_rope, kv):
    """`SparseMlaSplit_v1`'s body on bits over `SparseMlaHeadSplit_v2`."""
    hd = bind(SparseMlaHeadSplitV2, T=S.T, NS=S.NS, DR=S.DR, DV=S.DV, BN=S.BN, DOT=S.DOT, INV=S.INV, SCALE=S.SCALE)
    return B.batch(hd, q_nope, q_rope, kv, axes=(0, 0, None))


SparseMlaSplitV2 = flat("SparseMlaSplit", 2, ["T", "NS", "NH", "DR", "DV", "BN", "DOT", "INV", "SCALE"],
                        lambda S: M._mla_type(S, BF16), _split, W.SparseMlaSplit, lifted=("DOT", "INV"))

BOOLEAN_SPARSE_MLA_SPLIT = [MufuLg2Ftz, F32NeZeroFtz, F32GtZeroFtz, SparseMlaSplitEndV2, SparseMlaCombineV2, SparseMlaHeadSplitV2,
                            SparseMlaSplitV2]

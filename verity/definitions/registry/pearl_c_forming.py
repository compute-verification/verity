"""Pearl-C's forming v1, its chain, the Gram matrix, the peel factors and the peel as registered Definitions, each on words (``_v1``)
and on bits (``_v2``): what a row unit of Pearl-C's tile check computes from a row and its noise, stated on the catalog's
primitives alone, so a Program outside PoUW can call it.

    PearlCRowStats{K}        (x: f32[K]) -> (s, ρ, α, β_row)                           `pearl_c.s5_stats`
    PearlCScaledCode         (beta: f32, e: e4m3) -> e4m3                              one code of E′ = e4m3(β·E), `pearl_c.eprime`
    PearlCFormCode{DOT}      (alpha: f32, x: f32, ep: e4m3[32], f: e4m3[32]) -> e4m3   e4m3(DOT(α·x, E′, F_t)), one code of the row
    PearlCForm{K,DOT}        (x: f32[K], e: e4m3[32], f: e4m3[K][32])                  `pearl_c.form_v1` on one row
                               -> (codes: e4m3[K], ep: e4m3[32], s, ρ, α, β_row)
    PearlCChain{K,DOT,G}     (a, b: e4m3[K]) -> f32                                    `pearl_c.chain`
    PearlCGram{K,R,DOT,G}    (fb, fa: e4m3[K][R]) -> f32[R][R]                         `pearl_c.gram`: G_pq = chain(F_B[:, p], F_A[:, q])
    PearlCPeelA{K,R,DOT,G}   (codes: e4m3[K], ep: e4m3[R], fb: e4m3[K][R]) -> bf16[2R]  one row of `pearl_c.peel_factors_a`
    PearlCPeelB{K,R,DOT,G}   (codes: e4m3[K], ep: e4m3[R], fa: e4m3[K][R], g: f32[R][R]) -> bf16[2R]
                                                                                       one row of `pearl_c.peel_factors_b`
    PearlCPeel{R,STEP}       (c: f32, pa, pb: bf16[2R]) -> f32                         `pearl_c.peel`: U from C̃ and a row of each P

A row's α, which the unscale of its useful output reads, is `PearlCRowStats`' third field and `PearlCForm`'s fifth.

``DOT`` is the device's FP8 atom (E4M3 x E4M3 into an FP32 accumulator, k = 32: `HopperE4m3QgmmaDot32` on the H100,
`BlackwellE4m3QmmaDot32` on sm_120) and ``G`` its promotion period, the atoms a fresh +0 accumulator chains before one FP32
nearest-even add promotes it into the total; G = 0 is the unpromoted record, one accumulator from +0 over all of k. A last
window short of G atoms is promoted as it is, as `pearl_c.chain` promotes it. ``R`` is the noise's rank, 32 wherever a
tile is checked: the Gram matrix and the peel factors read each line only by its columns, so they are stated at any R, and
at R = 1 or 2 each is small enough to lower whole. ``STEP`` is the device's BF16 step (BF16 x BF16 into an FP32
accumulator, k = 16: `HopperBF16WgmmaDot16` on every record), which the peel takes over the 2R operands of a row of P_A
and of P_B sixteen at a time, so there R is a multiple of 8.

These are the functions of PoUW's row units (`verity.protocols.pouw.prover.circuit.pc8`: `Pc8RowA`, `Pc8RowB`), which
compute α and keep it: the same primitives in the same order, so each equals the unit's on every input, a row the unit
refuses included. Three differences are of form only. The square root is `PearlCSqrt`, a Definition whose body is the
one Call of `PouwF32SqrtRn_v1`, so that its Boolean version is found here and not in PoUW's package. −bf16(E′) is
`PearlCBf16Neg`, the sign bit flipped, as PoUW's `PouwBf16Neg` is. And the chain is a Definition of its own, where the row
units inline it. `pearl_c.chain`'s −0 → +0 after a promotion has no gate, as in `pc8`: the atom's zero is +0, the total
starts at +0, and a nearest-even sum is −0 only of two −0s.

The reference rejects a non-finite accumulator where these compute on (a NaN is 0x7FFFFFFF throughout); the row's
admission (`Pc8RowA`'s ``ok``) is not one of these Definitions.

On bits each is σ of its word form (`verity.definitions.boolean.sigma_lift`): the same Calls, each of the Boolean version
of its Definition. `PearlCSqrt_v2` is `verity.definitions.ops.boolean.sqrt.f32_sqrt_rn` traced, and `PearlCBf16Neg_v2` one
NOT.
"""
from __future__ import annotations

import struct

from verity.circuits import boolean as BL
from verity.circuits.boolean.forms import NOT
from verity.circuits.boolean.trace import definition
from verity.circuits.defs import CompositeDefinition, bind, composite, primitive
from verity.definitions.ops.boolean.sqrt import f32_sqrt_rn
from verity.definitions.ops.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.definitions.ops.prims import E4m3ToF32, F2fpBf16, F32ToE4m3Sat, const
from verity.definitions.ops.scalar import F32Fabs, F32Fmaxf, F32Neg, I32Le, SelectI32
from verity.definitions.ops.sqrt import PouwF32SqrtRn
from verity.circuits.refs import array_of, tuple_of
from verity.circuits.types import Array, Tuple, Value
from verity.definitions.boolean import sigma_lift

__all__ = ["PearlCBf16Neg", "PearlCBf16NegV2", "PearlCChain", "PearlCChainV2", "PearlCForm", "PearlCFormCode",
           "PearlCFormCodeV2", "PearlCFormV2", "PearlCGram", "PearlCGramV2", "PearlCPeel", "PearlCPeelV2", "PearlCPeelA",
           "PearlCPeelAV2", "PearlCPeelB",
           "PearlCPeelBV2", "PearlCRowStats", "PearlCRowStatsV2", "PearlCScaledCode", "PearlCScaledCodeV2", "PearlCSqrt",
           "PearlCSqrtV2", "bindings"]

F32, BF16, E4M3 = Value(32), Value(16), Value(8)
#: the atom's k: the codes of a noise line the forming reads, and of one step of the chain
SLICE = 32
LINE = Array(SLICE, E4M3)
#: §5's statistics (`pearl_c.S5_*`): ρ's sampling stride, the floor 2^-32 of α's divisor, 4, q_max = 448, and
#: β_row's factor √32 / (256 · 16), each an FP32 word but the stride
STRIDE, FLOOR, FOUR, QMAX, LINE_FACTOR = 8, 0x2F800000, 0x40800000, 0x43E00000, 0x3AB504F3

_TESTS = "verity/definitions/tests/test_pearl_c_forming.py"
_HELD = f"protocol-defined; equal to `verity.protocols.pouw.prover.schemes.pearl_c.{{}}` on its vectors ({_TESTS})"


class _Emit:
    """A body under construction: an integer operand is a constant of its parameter's width, one node per value, and an
    operation on constants alone is evaluated, not emitted."""

    def __init__(self, B) -> None:
        self.B = B
        self._consts: dict = {}

    def call(self, fn, *args):
        if all(isinstance(a, int) for a in args):
            return fn.evaluate(*args)
        return self.B.call(fn, *(self._const(t.w, a) if isinstance(a, int) else a for a, (_n, t) in zip(args, fn.params)))

    def _const(self, w: int, v: int):
        if (w, v) not in self._consts:
            self._consts[(w, v)] = self.B.call(const(w, v))
        return self._consts[(w, v)]


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


@composite("PearlCSqrt", 1, [], lambda S: ((("a", F32),), F32), conformance="protocol-defined")
def PearlCSqrt(B, S, a):
    """sqrt.rn.f32 as the GPU computes it: the one Call of `PouwF32SqrtRn_v1` (ρ's root)."""
    return B.call(PouwF32SqrtRn, a)


PearlCSqrtV2 = definition("PearlCSqrt", 2, f32_sqrt_rn, (("a", BL.bit_type(F32)),), BL.bit_type(F32), word=bind(PearlCSqrt),
                          conformance="protocol-defined", doc="sqrt.rn.f32 on bits: PearlCSqrt_v1")


@primitive("PearlCBf16Neg", 1, [("a", BF16)], BF16, conformance="protocol-defined")
def PearlCBf16Neg(a: int) -> int:
    """−a on a BF16 encoding: the sign bit flipped (−bf16(E′) in P_B)."""
    return (a ^ 0x8000) & 0xFFFF


PearlCBf16NegV2 = definition("PearlCBf16Neg", 2, lambda C, a: [*a[:15], NOT(a[15])], (("a", BL.bit_type(BF16)),),
                             BL.bit_type(BF16), word=PearlCBf16Neg, conformance="protocol-defined",
                             doc="−a on a BF16 encoding, on bits: PearlCBf16Neg_v1")


# ---- forming v1 --------------------------------------------------------------------------------------------------------------

def _stats_sig(S):
    return (("x", Array(S.K, F32)),), Tuple(F32, F32, F32, F32)


def _stats(B, S, x):
    e, k = _Emit(B), S.K
    mag = [e.call(F32Fabs, x[t]) for t in range(k)]
    s = mag[0]
    for m in mag[1:]:
        s = e.call(SelectI32, e.call(I32Le, s, m), m, s)
    positions = range(0, k, STRIDE)
    part = [0] * 32
    for i, p in enumerate(positions):
        part[i % 32] = e.call(F32FmaV2, x[p], x[p], part[i % 32])
    o = 16
    while o:
        part = [e.call(F32AddV2, part[j], part[j + o]) for j in range(o)]
        o //= 2
    rho = B.call(PearlCSqrt, e.call(F32DivV2, part[0], _f32(float(len(positions)))))
    t = e.call(F32AddV2, s, e.call(F32MulV2, FOUR, rho))
    alpha = e.call(F32DivV2, QMAX, e.call(F32Fmaxf, t, FLOOR))
    beta = e.call(F32MulV2, e.call(F32MulV2, rho, alpha), LINE_FACTOR)
    return tuple_of(s, rho, alpha, beta)


PearlCRowStats = CompositeDefinition(
    "PearlCRowStats", 1, ("K",), _stats_sig, _stats, conformance=_HELD.format("s5_stats"),
    refuse=lambda b: None if b.get("K", 1) >= 1 else "a row has at least one word",
    doc="§5's statistics of one row of K FP32 words, as FP32 words: s = max |x|; ρ = sqrt.rn of the mean of x_p² over p ≡ 0 "
        "(mod 8), 32 FMA partial sums by sample index mod 32 then a pairwise tree; α = 448 / fmaxf(s + 4·ρ, 2^-32); and "
        "β_row = (ρ·α)·RN(√32 / 4096).")


@composite("PearlCScaledCode", 1, [], lambda S: ((("beta", F32), ("e", E4M3)), E4M3), conformance=_HELD.format("eprime"))
def PearlCScaledCode(B, S, beta, e):
    """One code of the row's scaled noise line E′ = e4m3(β_row·E): the code's exact FP32 decode, one FP32 product, and the
    nearest-even saturating cast."""
    return B.call(F32ToE4m3Sat, B.call(F32MulV2, beta, B.call(E4m3ToF32, e)))


@composite("PearlCFormCode", 1, ["DOT"], lambda S: ((("alpha", F32), ("x", F32), ("ep", LINE), ("f", LINE)), E4M3),
           conformance=_HELD.format("form_v1"))
def PearlCFormCode(B, S, alpha, x, ep, f):
    """One code of a formed row: the device's atom on E′ and the basis' line F_t from the accumulator α·x_t, then the cast."""
    return B.call(F32ToE4m3Sat, B.call(S.DOT, B.call(F32MulV2, alpha, x), ep, f))


def _form_sig(S):
    k = S.K
    return ((("x", Array(k, F32)), ("e", LINE), ("f", Array(k, LINE))),
            Tuple(Array(k, E4M3), LINE, F32, F32, F32, F32))


@composite("PearlCForm", 1, ["K", "DOT"], _form_sig, conformance=_HELD.format("form_v1"))
def PearlCForm(B, S, x, e, f):
    """§5's forming of one row of K FP32 words against its noise line E and the basis F (K lines): the row's codes, its
    scaled line E′, and its statistics s, ρ, α, β_row."""
    stats = B.call(bind(PearlCRowStats, K=S.K), x)
    s, rho, alpha, beta = (stats[i] for i in range(4))
    ep = B.batch(PearlCScaledCode, beta, e, axes=(None, 0))
    codes = B.batch(bind(PearlCFormCode, DOT=S.DOT), alpha, x, ep, f, axes=(None, 0, None, 0))
    return tuple_of(codes, ep, s, rho, alpha, beta)


# ---- the chain, the Gram matrix and the peel factors --------------------------------------------------------------------------

def _refuse_chain(b: dict) -> str | None:
    if b.get("K", SLICE) < SLICE or b.get("K", SLICE) % SLICE:
        return f"the chain multiplies in atoms of k = {SLICE}: K is a positive multiple of it"
    if b.get("G", 0) < 0 or b.get("R", 1) < 1:
        return "G is a number of atoms (0: no promotion) and R at least 1"
    return None


@composite("PearlCChain", 1, ["K", "DOT", "G"], lambda S: ((("a", Array(S.K, E4M3)), ("b", Array(S.K, E4M3))), F32),
           conformance=_HELD.format("chain"), refuse=_refuse_chain)
def PearlCChain(B, S, a, b):
    """The device's replay of a·b over K E4M3 codes, an FP32 word: per window of G atoms a fresh +0 accumulator chains
    them and one FP32 add promotes it into the total, from +0; at G = 0 one accumulator from +0 chains every atom."""
    e, k, window = _Emit(B), S.K, SLICE * S.G
    step = lambda acc, at: e.call(S.DOT, acc, a[at:at + SLICE], b[at:at + SLICE])   # noqa: E731
    total = 0
    if not window:
        for at in range(0, k, SLICE):
            total = step(total, at)
        return total
    for start in range(0, k, window):
        local = 0
        for at in range(start, min(start + window, k), SLICE):
            local = step(local, at)
        total = e.call(F32AddV2, total, local)
    return total


def _basis(S):
    return Array(S.K, Array(S.R, E4M3))


def _chain_of(S):
    return bind(PearlCChain, K=S.K, DOT=S.DOT, G=S.G)


@composite("PearlCGram", 1, ["K", "R", "DOT", "G"],
           lambda S: ((("fb", _basis(S)), ("fa", _basis(S))), Array(S.R, Array(S.R, F32))),
           conformance=_HELD.format("gram"), refuse=_refuse_chain)
def PearlCGram(B, S, fb, fa):
    """G = F_Bᵀ·F_A as FP32 words by rows p: G_pq the chain of F_B's column p and F_A's column q."""
    chain = _chain_of(S)
    columns = [fa.take(1, q) for q in range(S.R)]
    return array_of([array_of([B.call(chain, fb.take(1, p), column) for column in columns]) for p in range(S.R)])


def _peel_sig(S, basis: str, extra: tuple = ()):
    params = (("codes", Array(S.K, E4M3)), ("ep", Array(S.R, E4M3)), (basis, _basis(S))) + extra
    return params, Array(2 * S.R, BF16)


@composite("PearlCPeelA", 1, ["K", "R", "DOT", "G"], lambda S: _peel_sig(S, "fb"),
           conformance=_HELD.format("peel_factors_a"), refuse=_refuse_chain)
def PearlCPeelA(B, S, codes, ep, fb):
    """Row i of P_A = [bf16(E′_A,i) ‖ RNE_bf16(A′_i·F_B)]: E′'s exact BF16 words, then the R fused columns' chains."""
    chain = _chain_of(S)
    noise = [B.call(F2fpBf16, B.call(E4m3ToF32, ep[p])) for p in range(S.R)]
    return array_of(noise + [B.call(F2fpBf16, B.call(chain, codes, fb.take(1, p))) for p in range(S.R)])


@composite("PearlCPeelB", 1, ["K", "R", "DOT", "G"],
           lambda S: _peel_sig(S, "fa", (("g", Array(S.R, Array(S.R, F32))),)),
           conformance=_HELD.format("peel_factors_b"), refuse=_refuse_chain)
def PearlCPeelB(B, S, codes, ep, fa, g):
    """Row j of P_B = [RNE_bf16(E′_B,j·G − B̃_j·F_A) ‖ −bf16(E′_B,j)]: per column q the FP32 FMA chain of E′'s decode and
    G's column from +0 in p order, one FP32 add of the negated chain B̃_j·F_A's column q, and the cast."""
    e, chain = _Emit(B), _chain_of(S)
    words = [B.call(E4m3ToF32, ep[p]) for p in range(S.R)]
    mid = []
    for q in range(S.R):
        acc = 0
        for p in range(S.R):
            acc = e.call(F32FmaV2, words[p], g[p][q], acc)
        fused = B.call(chain, codes, fa.take(1, q))
        mid.append(B.call(F2fpBf16, B.call(F32AddV2, acc, B.call(F32Neg, fused))))
    return array_of(mid + [B.call(PearlCBf16Neg, B.call(F2fpBf16, w)) for w in words])


def _refuse_peel(b: dict) -> str | None:
    if b.get("R", 8) < 8 or b.get("R", 8) % 8:
        return "the peel takes the 2R operands of a row of P sixteen at a time: R is a positive multiple of 8"
    return None


@composite("PearlCPeel", 1, ["R", "STEP"],
           lambda S: ((("c", F32), ("pa", Array(2 * S.R, BF16)), ("pb", Array(2 * S.R, BF16))), F32),
           conformance=_HELD.format("peel"), refuse=_refuse_peel)
def PearlCPeel(B, S, c, pa, pb):
    """U from C̃, an FP32 word: the device's BF16 step over a row of P_A and a row of P_B, sixteen operands a step in
    order, into C̃'s word."""
    acc = c
    for at in range(0, 2 * S.R, 16):
        acc = B.call(S.STEP, acc, pa[at:at + 16], pb[at:at + 16])
    return acc


PearlCRowStatsV2 = sigma_lift(PearlCRowStats, 2)
PearlCScaledCodeV2 = sigma_lift(PearlCScaledCode, 2)
PearlCFormCodeV2 = sigma_lift(PearlCFormCode, 2)
PearlCFormV2 = sigma_lift(PearlCForm, 2)
PearlCChainV2 = sigma_lift(PearlCChain, 2)
PearlCGramV2 = sigma_lift(PearlCGram, 2)
PearlCPeelAV2 = sigma_lift(PearlCPeelA, 2)
PearlCPeelBV2 = sigma_lift(PearlCPeelB, 2)
PearlCPeelV2 = sigma_lift(PearlCPeel, 2)


def bindings() -> list:
    """The bindings circuit-check takes (the sidecar), each Definition on words and on bits: on the H100's atom and
    sm_120's at G = 4, the chain over one promotion window and, unpromoted, over two atoms; the Gram matrix and the peel
    factors at rank 2 over one atom, and the peel factors at rank 1 over a window; the statistics of 32 words; and, on
    the H100's atom, the forming of a row of eight and the chain over a window and a short one (K = 160), as the tile's
    read calls it; and the peel at rank 32. But for the forming of a row, each is within the size circuit-check lowers a
    Definition whole at."""
    from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32, HopperBF16WgmmaDot16, HopperE4m3QgmmaDot32
    from verity.definitions.boolean import boolean_version

    out = []
    for bits in (False, True):
        pick = (lambda word, lifted: lifted) if bits else (lambda word, lifted: word)
        atoms = [boolean_version(d) if bits else d for d in (HopperE4m3QgmmaDot32, BlackwellE4m3QmmaDot32)]
        out += [bind(pick(PearlCSqrt, PearlCSqrtV2)), bind(pick(PearlCRowStats, PearlCRowStatsV2), K=32),
                bind(pick(PearlCScaledCode, PearlCScaledCodeV2))]
        if bits:
            out.append(bind(PearlCBf16NegV2))
        for dot in atoms:
            out += [bind(pick(PearlCFormCode, PearlCFormCodeV2), DOT=dot),
                    bind(pick(PearlCChain, PearlCChainV2), K=128, DOT=dot, G=4),
                    bind(pick(PearlCGram, PearlCGramV2), K=32, R=2, DOT=dot, G=4)]
            for k, r in ((128, 1), (32, 2)):
                out += [bind(pick(PearlCPeelA, PearlCPeelAV2), K=k, R=r, DOT=dot, G=4),
                        bind(pick(PearlCPeelB, PearlCPeelBV2), K=k, R=r, DOT=dot, G=4)]
        out += [bind(pick(PearlCForm, PearlCFormV2), K=8, DOT=atoms[0]),
                bind(pick(PearlCChain, PearlCChainV2), K=64, DOT=atoms[1], G=0),
                bind(pick(PearlCChain, PearlCChainV2), K=160, DOT=atoms[0], G=4),
                bind(pick(PearlCPeel, PearlCPeelV2), R=32,
                     STEP=boolean_version(HopperBF16WgmmaDot16) if bits else HopperBF16WgmmaDot16)]
    return out

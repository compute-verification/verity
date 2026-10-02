"""`pearl-c-nvfp4-v0`'s tile check as a Program: Pearl-C4's forming, chain and clean-up (`schemes.pearl_c4`) cut into proof
units, so sampled proofs can establish a drawn tile's checked words (C̃ and U) and its rows' admission without the verifier
reading a row.

A weight's side is formed once per job and weight, by `Pc4Weight{K,N}`, whose root body holds the `Pc4RowB` instances.
One call's side is `Pc4Call{M,K,N,TM,TN}`, which reads that weight's B̃ codes, scale bytes and P_B rows as committed
inputs.  `Pc4Check{M,K,N,TM,TN}` is both in one root: the whole check of an M × N block, recompute-free.  Each root body
holds only instances of the unit templates below (the partition is `Q_template_instances` over them, `query`), and every
value is computed in exactly one of them.

    template          one instance per        reads                                    computes and returns
    Pc4RowB{K}        row j of B              B_j, E_B,j, F_B, F_A, G                   B̃_j's codes and scale bytes, P_B,j, ok
      (in Pc4Weight, once per job and weight)
    Pc4RowA{K}        row i of A              A_i, E_A,i, F_A, F_B                      A′_i's codes and scale bytes, P_A,i, ok
    Pc4Tile{K,TM,TN}  tile of TM × TN words   its TM rows A′, P_A and TN rows B̃, P_B    C̃ and U of its words

A tile unit, not a word unit: a unit reads another unit's rows through their commitments, which under hiding it hashes in
the circuit, and a word unit would hash about 4.9 KiB of rows for one word; a tile (64 × 64 in the scheme, `TILE`) hashes
each of its rows once.

A row unit computes the row's statistics, its noise atoms, block scales and casts (the forming), Ẽ, and its half of the
clean-up: P_A,i = [Ẽ_A,i ‖ RNE_bf16(A′_i·F_B)], P_B,j = [RNE_bf16(Ẽ_B,j·G − B̃_j·F_A) ‖ −Ẽ_B,j].  ``ok`` is the row's
admission (`schemes.pearl_c4.row_passes`: every BF16 word finite and D-NF), one public bit per row that an honest prover's
tile always sets.  Every operation is the scheme's own, in its order: core's FP32 add, mul, FMA and divide (``_v2``), its
casts and its sm_120 NVF4 and BF16 steps, and three PoUW primitives (`words`: sqrt.rn, Ẽ's exact decode, BF16 negation).
Three of the scheme's integer operations are written as float operations they equal on every row that ``ok`` admits:
``max(ue4m3(v), 0x01)`` is ``ue4m3(fmaxf(v, 2^-9))`` (the cast is monotone and ue4m3(2^-9) = 0x01), a block's
``max |D|`` is an `F32Fmaxf` reduction (D is finite on an admitted row), and ``ue4m3(β)``'s mask of bit 7 is dropped (β is
non-negative there).  ``max |x|`` stays the integer maximum, so a non-finite word reaches ``ok``.

The noise lines (E_A, E_B, F_A, F_B) are inputs, anchored like `ncp2`'s salt: they are TurboSHAKE128 of seeds the
verifier derives from public values (the salt, the commitments' roots, the unit index), so they carry no row's value.  G =
F_Bᵀ·F_A (`pearl_c4.gram`) is a function of F_A and F_B alone, so it is anchored too: the verifier computes it, and no unit
does.
"""
from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from verity.ir import partition_object as PO
from verity.ir.cut import CallGraph, check_cut
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.partition import CutVerdict
from verity.ir.refs import array_of, tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.prims import (BlackwellNvf4OmmaDot64, Bf16ToF32, E4m3ToF32, F2fpBf16, F32ToE2m1Sat, F32ToE4m3Sat,
                             HopperBF16WgmmaDot16, const)
from verity.ml.scalar import BitAnd, F32Fabs, F32Fmaxf, F32Neg, I32Le, SelectI32

from ..protocol import DomainError
from ..schemes import pearl_c as C
from ..schemes import pearl_c4 as S
from . import words as W

F32, BF16, E2M1, U8, BIT = W.F32, W.BF16, W.E2M1, W.U8, Value(1)
R, ATOM, BLOCK, PEEL = S.R, S.ATOM, S.NVFP4.block, S.PEEL_K
LINE = Array(R, E2M1)
P_ROW = Array(PEEL, BF16)
F32_MAX = 0x7F7FFFFF
DNF_LOW = 0x3C700000               # 7.5·2^-9: β at or above it casts to a UE4M3 byte of at least 0x08 (the tie rounds up)
UE4M3_MIN_WORD = 0x3B000000        # 2^-9, whose UE4M3 byte is 0x01
TILE = (64, 64)                    # the scheme's tile: rows of A, rows of B

GRAM = Array(R, Array(R, F32))
BASIS = lambda k: Array(k, LINE)                                                    # noqa: E731
CODES = lambda n, k: Array(n, Array(k, E2M1))                                       # noqa: E731
SCALES = lambda n, k: Array(n, Array(k // BLOCK, U8))                               # noqa: E731


def _f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


class _Emit:
    """A body under construction: integer operands are constants of the parameter's width, one node per value and body, and
    an operation on constants alone is evaluated, not emitted."""

    def __init__(self, B) -> None:
        self.B = B
        self._consts: dict[tuple[int, int], Any] = {}

    def c(self, w: int, v: int):
        got = self._consts.get((w, v))
        if got is None:
            got = self._consts[(w, v)] = self.B.call(const(w, v))
        return got

    def call(self, fn, *args):
        if all(isinstance(a, int) for a in args):
            return fn.evaluate(*args)
        return self.B.call(fn, *(self.c(t.w, a) if isinstance(a, int) else a for a, (_n, t) in zip(args, fn.params)))


# ---- the pieces of a unit ------------------------------------------------------------------------------------------------

def _chain(e: _Emit, a, b, sa, sb, k: int):
    """The NVF4 atom chained from +0 over k, no promotion (`pearl_c4.chain`): k/64 steps in ascending k."""
    acc = 0
    for s in range(k // ATOM):
        acc = e.call(BlackwellNvf4OmmaDot64, acc, a[ATOM * s:ATOM * (s + 1)], b[ATOM * s:ATOM * (s + 1)],
                     sa[4 * s:4 * (s + 1)], sb[4 * s:4 * (s + 1)])
    return acc


def _ones(e: _Emit, k: int):
    """The scale bytes of 1.0 for k codes (the lines' scale in the fused columns and G)."""
    return array_of([e.c(8, S.NVFP4.one)] * (k // BLOCK))


def _stats(e: _Emit, x, k: int):
    """(α, a_E, ok) of one row of k BF16 words (`pearl_c4.stats`, `pearl_c.s5_stats`, `row_passes`), and its FP32 words."""
    w = [e.call(Bf16ToF32, x[t]) for t in range(k)]
    mag = [e.call(F32Fabs, wt) for wt in w]
    s = mag[0]
    for m in mag[1:]:
        s = e.call(SelectI32, e.call(I32Le, s, m), m, s)
    part = [0] * 32
    positions = range(0, k, C.S5_STRIDE)
    for i, p in enumerate(positions):
        part[i % 32] = e.call(F32FmaV2, w[p], w[p], part[i % 32])
    o = 16
    while o:
        part = [e.call(F32AddV2, part[j], part[j + o]) for j in range(o)]
        o //= 2
    rho = e.call(W.PouwF32SqrtRn, e.call(F32DivV2, part[0], _f32(float(len(positions)))))
    sigma = e.call(F32MulV2, rho, S.DELTA)
    t = e.call(F32AddV2, s, e.call(F32MulV2, S.FOUR, sigma))
    alpha = e.call(F32DivV2, S.TOP, e.call(F32Fmaxf, t, S.FLOOR))
    beta = e.call(F32MulV2, e.call(F32MulV2, sigma, alpha), S.K_BETA)
    ae = e.call(F32ToE4m3Sat, beta)
    dnf = e.call(BitAnd, e.call(I32Le, DNF_LOW, beta), e.call(I32Le, beta, F32_MAX))
    ok = e.call(BitAnd, e.call(I32Le, s, F32_MAX), dnf)
    return w, alpha, ae, ok


def _form(e: _Emit, w, alpha, ae, line, basis, k: int):
    """A row's codes and scale bytes (`pearl_c4.form`): per element the noise atom D_t = NVF4.step(α·x_t, E ‖ 0 at a_E,
    F_t ‖ 0 at 2^-4), per block of 16 its scale s_b = ue4m3(fmaxf(f32(1/6)·max|D|, 2^-9)) and r_b = 1/s_b, and per element
    e2m1(D_t·r_b)."""
    zeros = [e.c(4, 0)] * (ATOM - R)
    e_op = array_of([line[p] for p in range(R)] + zeros)
    sa = array_of([ae] * 4)
    sb = array_of([e.c(8, S.NOISE_F_SCALE)] * 4)
    d = [e.call(BlackwellNvf4OmmaDot64, e.call(F32MulV2, alpha, w[t]), e_op, array_of([basis[t][p] for p in range(R)] + zeros),
                sa, sb) for t in range(k)]
    codes, scales = [], []
    for b in range(0, k, BLOCK):
        amax = e.call(F32Fabs, d[b])
        for t in range(b + 1, b + BLOCK):
            amax = e.call(F32Fmaxf, amax, e.call(F32Fabs, d[t]))
        s = e.call(F32ToE4m3Sat, e.call(F32Fmaxf, e.call(F32MulV2, amax, S.SIXTH), UE4M3_MIN_WORD))
        r = e.call(F32DivV2, S.ONE, e.call(E4m3ToF32, s))
        scales.append(s)
        codes += [e.call(F32ToE2m1Sat, e.call(F32MulV2, d[t], r)) for t in range(b, b + BLOCK)]
    return array_of(codes), array_of(scales)


def _e_tilde(e: _Emit, line, ae):
    return [e.call(W.PouwETilde, line[p], ae) for p in range(R)]


# ---- the unit templates --------------------------------------------------------------------------------------------------

def _row_sig(S_, second: str, extra: tuple = ()):
    k = S_.K
    first = "fb" if second == "fa" else "fa"
    params = (("x", Array(k, BF16)), ("e", LINE), (first, Array(k, LINE)), (second, Array(k, LINE))) + extra
    return params, Tuple(Array(k, E2M1), Array(k // BLOCK, U8), P_ROW, BIT)


def _row_a_body(B, S_, x, line, fa, fb):
    e, k = _Emit(B), S_.K
    w, alpha, ae, ok = _stats(e, x, k)
    codes, scales = _form(e, w, alpha, ae, line, fa, k)
    ones = _ones(e, k)
    af = [e.call(F2fpBf16, _chain(e, codes, fb.take(1, p), scales, ones, k)) for p in range(R)]
    return tuple_of(codes, scales, array_of(_e_tilde(e, line, ae) + af), ok)


Pc4RowADef = CompositeDefinition(
    "Pc4RowA", 1, ("K",), lambda S_: _row_sig(S_, "fb"), _row_a_body,
    doc="Row i of A: its admission, A′_i's codes and scale bytes formed against F_A, and P_A,i = [Ẽ_A,i ‖ RNE_bf16(A′_i·F_B)].",
    conformance="protocol-defined")


def _row_b_body(B, S_, x, line, fb, fa, g):
    e, k = _Emit(B), S_.K
    w, alpha, ae, ok = _stats(e, x, k)
    codes, scales = _form(e, w, alpha, ae, line, fb, k)
    ones = _ones(e, k)
    et = _e_tilde(e, line, ae)
    ew = [e.call(Bf16ToF32, v) for v in et]
    mid = []
    for q in range(R):
        acc = 0
        for p in range(R):
            acc = e.call(F32FmaV2, ew[p], g[p][q], acc)
        bf = _chain(e, codes, fa.take(1, q), scales, ones, k)
        mid.append(e.call(F2fpBf16, e.call(F32AddV2, acc, e.call(F32Neg, bf))))
    return tuple_of(codes, scales, array_of(mid + [e.call(W.PouwBf16Neg, v) for v in et]), ok)


Pc4RowBDef = CompositeDefinition(
    "Pc4RowB", 1, ("K",), lambda S_: _row_sig(S_, "fa", (("g", GRAM),)), _row_b_body,
    doc="Row j of B: its admission, B̃_j's codes and scale bytes formed against F_B, and P_B,j = [RNE_bf16(Ẽ_B,j·G − "
        "B̃_j·F_A) ‖ −Ẽ_B,j] (Ẽ_B,j·G the FP32 FMA chain from +0 in p order); once per job and weight.",
    conformance="protocol-defined")


def _tile_body(B, S_, ac, as_, bc, bs, pa, pb):
    e, k = _Emit(B), S_.K
    cs, us = [], []
    for i in range(S_.TM):
        crow, urow = [], []
        for j in range(S_.TN):
            c = _chain(e, ac[i], bc[j], as_[i], bs[j], k)
            u = c
            for s in range(0, PEEL, 16):
                u = e.call(HopperBF16WgmmaDot16, u, pa[i][s:s + 16], pb[j][s:s + 16])
            crow.append(c)
            urow.append(u)
        cs.append(array_of(crow))
        us.append(array_of(urow))
    return tuple_of(array_of(cs), array_of(us))


def _tile_sig(S_):
    k, tm, tn = S_.K, S_.TM, S_.TN
    return ((("ac", CODES(tm, k)), ("as", SCALES(tm, k)), ("bc", CODES(tn, k)), ("bs", SCALES(tn, k)),
             ("pa", Array(tm, P_ROW)), ("pb", Array(tn, P_ROW))), Tuple(Array(tm, Array(tn, F32)), Array(tm, Array(tn, F32))))


Pc4TileDef = CompositeDefinition(
    "Pc4Tile", 1, ("K", "TM", "TN"), _tile_sig, _tile_body,
    doc="One TM x TN tile's checked words: C̃_ij, the NVF4 atom chained from +0 over A′_i·B̃_j, and U_ij, four sm_120 BF16 "
        "m16n8k16 steps from C̃_ij over P_A,i·P_B,j (`pearl_c4.chain`, `pearl_c4.peel`).", conformance="protocol-defined")


@lru_cache(maxsize=16)
def templates(k: int, tile: tuple[int, int] = TILE) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k`` and tile ``tile`` (rows, columns), by role."""
    return {"row_b": bind(Pc4RowBDef, K=k), "row_a": bind(Pc4RowADef, K=k),
            "tile": bind(Pc4TileDef, K=k, TM=tile[0], TN=tile[1])}


# ---- the Programs --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """M rows of A and N rows of B at depth k, cut into tiles of ``tile`` (rows of A, rows of B; the scheme's is 64 x 64)."""

    m: int
    k: int
    n: int
    tile: tuple[int, int] = TILE

    def check(self) -> None:
        tm, tn = self.tile
        if self.k % ATOM or min(self.m, self.k, self.n, tm, tn) <= 0 or self.m % tm or self.n % tn:
            raise DomainError(f"Pearl-C4's circuit needs 64 | k and whole tiles on each side: {self}")


def _weight_units(B, k: int, b, eb, fa, fb, g):
    rows = B.batch(templates(k)["row_b"], b, eb, fb, fa, g, axes=(0, 0, None, None, None))
    n = len(b)
    return [array_of([rows[j][f] for j in range(n)]) for f in range(4)]


def _call_units(B, k: int, tile: tuple[int, int], a, ea, fa, fb, bc, bs, pb):
    tm, tn = tile
    tp = templates(k, tile)
    rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
    m, n = len(a), len(bc)
    ms, ns = m // tm, n // tn
    ac, as_, pa = (array_of([rows[i][f] for i in range(m)]) for f in range(3))
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    acs, ass, pas = blocks(ac, ms, tm), blocks(as_, ms, tm), blocks(pa, ms, tm)
    bcs, bss, pbs = blocks(bc, ns, tn), blocks(bs, ns, tn), blocks(pb, ns, tn)
    tiles = [B.batch(tp["tile"], acs[rs], ass[rs], bcs, bss, pas[rs], pbs, axes=(None, None, 0, 0, None, 0))
             for rs in range(ms)]
    cw, uw = (array_of([array_of([tiles[i // tm][j // tn][f][i % tm][j % tn] for j in range(n)]) for i in range(m)])
              for f in range(2))
    return cw, uw, array_of([rows[i][3] for i in range(m)])


def _weight_sig(S_):
    k, n = S_.K, S_.N
    return ((("b", Array(n, Array(k, BF16))), ("eb", Array(n, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(CODES(n, k), SCALES(n, k), Array(n, P_ROW), Array(n, BIT)))


Pc4WeightDef = CompositeDefinition(
    "Pc4Weight", 1, ("K", "N"), _weight_sig, lambda B, S_, *a: tuple_of(*_weight_units(B, S_.K, *a)),
    doc="A weight's side, once per job and weight: per row B̃'s codes and scale bytes, P_B and its admission (G is anchored).",
    conformance="protocol-defined")


def _call_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, BF16))), ("ea", Array(m, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)),
             ("bc", CODES(n, k)), ("bs", SCALES(n, k)), ("pb", Array(n, P_ROW))),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, BIT)))


Pc4CallDef = CompositeDefinition(
    "Pc4Call", 1, ("M", "K", "N", "TM", "TN"), _call_sig,
    lambda B, S_, *a: tuple_of(*_call_units(B, S_.K, (S_.TM, S_.TN), *a)),
    doc="One call's side of an M x N block: its A rows formed, C̃ and U for every word, tile by tile, and each A row's "
        "admission; its weight's B̃ codes, scale bytes and P_B rows are committed inputs (`Pc4Weight`).",
    conformance="protocol-defined")


def _check_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, BF16))), ("ea", Array(m, LINE)), ("b", Array(n, Array(k, BF16))), ("eb", Array(n, LINE)),
             ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, BIT), Array(n, BIT)))


def _check_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, bs, pb, ok_b = _weight_units(B, S_.K, b, eb, fa, fb, g)
    cw, uw, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb)
    return tuple_of(cw, uw, ok_a, ok_b)


Pc4CheckDef = CompositeDefinition(
    "Pc4Check", 1, ("M", "K", "N", "TM", "TN"), _check_sig, _check_body,
    doc="The check of an M x N block in one root: the weight's units, then the call's, reading them.",
    conformance="protocol-defined")


def weight(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc4WeightDef, K=shape.k, N=shape.n)


def call(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc4CallDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0], TN=shape.tile[1])


def check(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc4CheckDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0], TN=shape.tile[1])


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = {"weight": ("row_b",), "call": ("row_a", "tile"), "check": ("row_b", "row_a", "tile")}
_ROLE_OF = {id(Pc4RowBDef): "row_b", id(Pc4RowADef): "row_a", id(Pc4TileDef): "tile"}


def query(shape: Shape, side: str = "check") -> dict[str, Any]:
    """`Q_template_instances` over the side's unit templates (`ROLES`), in the order its root body first uses them."""
    tp = templates(shape.k, shape.tile)
    return PO.template_instances_query([tp[r] for r in ROLES[side]])


@dataclass(frozen=True)
class Unit:
    index: int
    role: str                   # a key of `templates`
    node: int                   # its root-body node
    member: int                 # its member of that batch node


def units(fn: SpecializedDefinition, role_of: dict[int, str] | None = None) -> tuple[Unit, ...]:
    """Every unit of a Program built from the unit templates, in the partition's numbering (node order, then member);
    ``role_of`` maps each template's Definition (by id) to its role, Pearl-C4's by default."""
    role_of = _ROLE_OF if role_of is None else role_of
    out: list[Unit] = []
    for j, nd in enumerate(fn.body.nodes):
        for m in range(nd.n if nd.form == "batch" else 1):
            out.append(Unit(len(out), role_of[id(nd.fn.defn)], j, m))
    return tuple(out)


def unit_cut(fn: SpecializedDefinition, role_of: dict[int, str] | None = None) -> tuple[CallGraph, list[int], CutVerdict]:
    """The whole Program's cut with each computing gate owned by its root node's member, checked by `cut.check_cut`
    (every gate certified once, reads across units only through committed values, no value computed in two units), with no
    width rule: no unit here reaches a served output."""
    body, us = fn.body, units(fn, role_of)
    first, at = [], 0
    for nd in body.nodes:
        first.append(at)
        at += nd.n if nd.form == "batch" else 1
    g = CallGraph(fn)
    owner = []
    for c in range(g.n):
        j = g.keep[c]
        node = bisect.bisect_right(body.offsets, j) - 1
        nd = body.nodes[node]
        owner.append(first[node] + ((j - body.offsets[node]) // nd.fn.gates if nd.form == "batch" else 0))
    assert len(us) == at
    return g, owner, check_cut(g, owner, limits=None)


__all__ = ["BASIS", "GRAM", "LINE", "P_ROW", "ROLES", "TILE", "Shape", "Unit", "call", "check", "query", "templates", "unit_cut",
           "units", "weight"]

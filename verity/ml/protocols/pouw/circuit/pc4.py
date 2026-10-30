"""`pearl-c-nvfp4-v0`'s tile check as a Program: Pearl-C4's forming, chain and clean-up (`schemes.pearl_c4`) cut into proof
blocks, so sampled proofs can establish a drawn tile's checked words (C̃ and U) and its rows' admission without the verifier
reading a row.

A weight's side is formed once per job and weight, by `Pc4Weight{K,N}`, whose root body holds the `Pc4RowB` instances.
One call's side is `Pc4Call{M,K,N,TM,TN}`, which reads that weight's B̃ codes, scale bytes and P_B rows as committed
inputs.  `Pc4Check{M,K,N,TM,TN}` is both in one root: the whole check of an M × N block, recompute-free.  Each root body
holds only instances of the unit templates below (the partition is `Q_template_instances` over them, `query`), and every
value is computed in exactly one of them.  ``Shape.row_k`` cuts each row unit along k into a head, a form per segment of
row_k positions and a tail (`rowk`), so that a row unit fits C-Flock's block at every served k; off, nothing here changes.

    template          one instance per        reads                                    computes and returns
    Pc4RowB{K}        row j of B              B_j, E_B,j, F_B, F_A, G                   B̃_j's codes and scale bytes, P_B,j, ok
      (in Pc4Weight, once per job and weight)
    Pc4RowA{K}        row i of A              A_i, E_A,i, F_A, F_B                      A′_i's codes and scale bytes, P_A,i, ok
    Pc4Tile{K,TM,TN}  tile of TM × TN words   its TM rows A′, P_A and TN rows B̃, P_B    C̃ and U of its words

A tile unit, not a word unit: a unit reads another unit's rows through their commitments, which under hiding the proof
hashes, and a word unit would hash about 4.9 KiB of rows for one word; a tile (64 × 64 in the scheme, `TILE`) hashes each of
its rows once.

The hiding Programs (``Shape.hidden``: `Pc4CallHidden`, `Pc4CheckHidden`) differ only in the tile: `Pc4TileHidden` is
`Pc4Tile` and its tile digest (`leaves.PearlTileDigest`, `pearl_c.leaf` of C̃ and U), which it returns in place of C̃ and U.
The weight's Program is `Pc4Weight` under hiding too.  The hiding is C-Flock's: every row a unit reads or writes is a
committed port, which C-Flock commits as an hm96-sha512 row and hashes in the proof, so these Programs hold no leaf and no
value is computed in two units.  `PORTS` names each unit's committed rows by schema (`leaves.SCHEMAS`): a row unit's input
row (bf16), the nvfp4-p row it writes (A′_i ‖ P_A,i or B̃_j ‖ P_B,j) and its ``ok`` row (C-Flock commits every output,
#757), and a tile unit's TM + TN nvfp4-p rows and its tile digest.  A tile reads each row as its row unit wrote it, the
same words under the same schema, so the units link by equal commit strings.  The rest are the anchored inputs, each a
public input region that `PORTS` names with its recomputation (`DERIVATIONS`).

A row unit computes the row's statistics, its noise atoms, block scales and casts (the forming), Ẽ, and its half of the
clean-up: P_A,i = [Ẽ_A,i ‖ RNE_bf16(A′_i·F_B)], P_B,j = [RNE_bf16(Ẽ_B,j·G − B̃_j·F_A) ‖ −Ẽ_B,j].  ``ok`` is the row's
admission (`schemes.pearl_c4.row_passes`: every BF16 word finite and D-NF), one bit per row that an honest prover's tile
always sets, returned as its ok row's one word with bits 1-15 zero in gates (`ok_word`).  Every operation is the scheme's
own, in its order: core's FP32 add, mul, FMA and divide (``_v2``), its casts and its sm_120 NVF4 and BF16 steps, and three
PoUW primitives (`words`: sqrt.rn, Ẽ's exact decode, BF16 negation).
Three of the scheme's integer operations are written as float operations they equal on every row that ``ok`` admits:
``max(ue4m3(v), 0x01)`` is ``ue4m3(fmaxf(v, 2^-9))`` (the cast is monotone and ue4m3(2^-9) = 0x01), a block's
``max |D|`` is an `F32Fmaxf` reduction (D is finite on an admitted row), and ``ue4m3(β)``'s mask of bit 7 is dropped (β is
non-negative there).  ``max |x|`` stays the integer maximum, so a non-finite word reaches ``ok``.

The noise lines (E_A, E_B, F_A, F_B) and G = F_Bᵀ·F_A (`pearl_c4.gram`) are verifier-derived public inputs, as `ncp2`'s
anchored inputs are (`anchors.TileAnchors`, `DERIVATIONS`): the lines are SHA-512 of the scheme's seeds (`pearl_c4.line`),
which read public values alone (seed_A the salt, A's root, B's registered root and the matmul index; seed_B the salt and B's
root), so they carry no row's value, and G is a function of F_A and F_B, so no unit computes it.  In a unit's proof each is a C-Flock
public input region, its values unpacked one u16 word each (G's FP32 words two, low half first), which the verifier takes
from its own `--public-inputs` file (`TileAnchors.regions`).
"""
from __future__ import annotations

import bisect
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, ClassVar

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.cut import CallGraph, check_cut
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.partition import CutVerdict
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple
from verity.ml.circuits.ops.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.circuits.ops.prims import (BlackwellNvf4OmmaDot64, Bf16ToF32, E4m3ToF32, F2fpBf16, F32ToE2m1Sat, F32ToE4m3Sat,
                             HopperBF16WgmmaDot16, const)
from verity.ml.circuits.ops.scalar import BitAnd, F32Fabs, F32Fmaxf, F32Neg, I32Le, SelectBf16, SelectI32

from verity.core.protocols.pouw.protocol import DomainError
from ..schemes import pearl_c as C
from ..schemes import pearl_c4 as S
from . import anchors as A
from . import boolean as _boolean  # noqa: F401  registers the _v2 Definitions C-Flock's lowering falls back to
from . import leaves as L
from . import words as W

F32, BF16, E2M1, U8, OK = W.F32, W.BF16, W.E2M1, W.U8, L.OK
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

def _chain(e: _Emit, a, b, sa, sb, k: int, acc=0):
    """The NVF4 atom chained from +0 over k, no promotion (`pearl_c4.chain`): k/64 steps in ascending k.  ``acc`` is the
    accumulator it starts from: +0, or the carry of the k positions before ``a`` and ``b`` (`rowk`)."""
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


def ok_word(e: _Emit, ok):
    """The ok row's word of the admission bit ``ok`` (`leaves.OK`): a select between the constants 1 and 0, so bit 0 is
    ``ok`` and bits 1-15 are zero in gates, never free bits of the committed row."""
    return e.call(SelectBf16, ok, 1, 0)


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
    return params, Tuple(Array(k, E2M1), Array(k // BLOCK, U8), P_ROW, OK)


def _row_a_body(B, S_, x, line, fa, fb):
    e, k = _Emit(B), S_.K
    w, alpha, ae, ok = _stats(e, x, k)
    codes, scales = _form(e, w, alpha, ae, line, fa, k)
    ones = _ones(e, k)
    af = [e.call(F2fpBf16, _chain(e, codes, fb.take(1, p), scales, ones, k)) for p in range(R)]
    return tuple_of(codes, scales, array_of(_e_tilde(e, line, ae) + af), ok_word(e, ok))


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
    return tuple_of(codes, scales, array_of(mid + [e.call(W.PouwBf16Neg, v) for v in et]), ok_word(e, ok))


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


# ---- the hiding tile: its checked words behind its digest --------------------------------------------------------------

def _tile_hidden_body(B, S_, ac, as_, bc, bs, pa, pb):
    k, tm, tn = S_.K, S_.TM, S_.TN
    out = B.call(templates(k, (tm, tn))["tile"], ac, as_, bc, bs, pa, pb)
    return B.call(L.tile_digest(tm, tn), out[0], out[1])


Pc4TileHiddenDef = CompositeDefinition(
    "Pc4TileHidden", 1, ("K", "TM", "TN"), lambda S_: (_tile_sig(S_)[0], L.DIGEST), _tile_hidden_body,
    doc="`Pc4Tile` with its checked words hidden: their tile digest (`leaves.PearlTileDigest`, `pearl_c.leaf` of C̃ and U).",
    conformance="protocol-defined")

#: each unit's ports under hiding, by role: its C-Flock-committed rows, each (schema, role, ports) whose words are its ports'
#: in order (`leaves.words_*`; parameters by name, outputs by index, row r of each where a tile reads TM or TN), inputs and
#: outputs alike, and its public input regions, each parameter's anchor (`DERIVATIONS`).  Every port is in exactly one.  A
#: row's role is its tree's, which the row's registration domain binds and which picks its schema (`leaves.ROW_SCHEMA`: a
#: per-call row is a row-seg row): its digest (`leaves.commit`) does not name it.  `leaves.statement_rows` is a unit's
#: C-Flock rows from them.
PORTS = {
    "row_b": {"rows": (("bf16", ROLE_W, ("x",)), ("nvfp4-p", ROLE_W, (0, 1, 2)), ("ok", ROLE_W, (3,))),
              "regions": {"e": "eb", "fb": "fb", "fa": "fa", "g": "g"}},
    "row_a": {"rows": (("bf16", ROLE_X, ("x",)), ("nvfp4-p", ROLE_X, (0, 1, 2)), ("ok", ROLE_X, (3,))),
              "regions": {"e": "ea", "fa": "fa", "fb": "fb"}},
    "tile": {"rows": (("nvfp4-p", ROLE_X, ("ac", "as", "pa")), ("nvfp4-p", ROLE_W, ("bc", "bs", "pb")),
                      ("tile", ROLE_X, (0,))), "regions": {}},
}


# ---- the anchors: public input regions, each with its recomputation ------------------------------------------------------

def line_ea(shape, seed_a: bytes, i: int) -> list[int]:
    """E_A's line for A's global row i: `pearl_c4.line(seed_A, ROLE_EA, i)`."""
    return S.line(seed_a, S.ROLE_EA, i)


def line_eb(shape, seed_b: bytes, j: int) -> list[int]:
    """E_B's line for B's global row j: `pearl_c4.line(seed_B, ROLE_EB, j)`."""
    return S.line(seed_b, S.ROLE_EB, j)


def basis_fa(shape, seed_b: bytes, _i=None) -> list[int]:
    """F_A, line t for position t < k: `pearl_c4.basis(seed_B, ROLE_FA, k)`."""
    return [c for ln in S.basis(seed_b, S.ROLE_FA, shape.k) for c in ln]


def basis_fb(shape, seed_b: bytes, _i=None) -> list[int]:
    """F_B, line t for position t < k: `pearl_c4.basis(seed_B, ROLE_FB, k)`."""
    return [c for ln in S.basis(seed_b, S.ROLE_FB, shape.k) for c in ln]


def gram_g(shape, seed_b: bytes, _i=None) -> list[int]:
    """G = F_Bᵀ·F_A by row p, then column q: `pearl_c4.gram(NVFP4, seed_B, k)`."""
    return [w for row in S.gram(S.NVFP4, seed_b, shape.k) for w in row]


_LINE = (f"bytes 16·(index mod {S.LINES_PER_BLOCK}) to +15 of SHA-512(\"{S.LABELS['line'].decode()}\" ‖ \\0 zero-padded to "
         f"128 bytes ‖ seed ‖ role ‖ ⌊index/{S.LINES_PER_BLOCK}⌋ u32 LE), low nibble first; role ")
#: each anchor's recomputation from `pearl_c4`'s seeds (`noise`: seed_A = H_"pearl-c4/v1/seed-A"(salt ‖ root_A ‖ root_B ‖
#: matmul u64 LE), seed_B = H_"pearl-c4/v1/seed-B"(salt ‖ root_B), H_ℓ `pearl_c4.labelled`'s SHA-512), each under its rule's
#: versioned id
DERIVATIONS = {
    "ea": A.Derivation("verity/pouw/anchor/pearl-c4/ea/v2", "seed_a",
                       _LINE + f"{S.ROLE_EA:#04x} (ROLE_EA), index A's global row", line_ea, 4, per_row=True),
    "eb": A.Derivation("verity/pouw/anchor/pearl-c4/eb/v2", "seed_b",
                       _LINE + f"{S.ROLE_EB:#04x} (ROLE_EB), index B's global row", line_eb, 4, per_row=True),
    "fa": A.Derivation("verity/pouw/anchor/pearl-c4/fa/v2", "seed_b",
                       _LINE + f"{S.ROLE_FA:#04x} (ROLE_FA), index the position t < k", basis_fa, 4),
    "fb": A.Derivation("verity/pouw/anchor/pearl-c4/fb/v2", "seed_b",
                       _LINE + f"{S.ROLE_FB:#04x} (ROLE_FB), index the position t < k", basis_fb, 4),
    "g": A.Derivation("verity/pouw/anchor/pearl-c4/g/v2", "seed_b",
                      f"G_pq = the NVF4 atom chained from +0 over F_B[:, p]·F_A[:, q], k/64 steps, every scale "
                      f"byte UE4M3 1.0 ({S.NVFP4.one:#04x}); FP32 words", gram_g, 32),
}
#: each seed as a public input port names it: the scheme's label for it
SEEDS = {"seed_a": S.LABELS["seed-A"].decode(), "seed_b": S.LABELS["seed-B"].decode()}


@lru_cache(maxsize=16)
def templates(k: int, tile: tuple[int, int] = TILE, hidden: bool = False) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k`` and tile ``tile`` (rows, columns), by role; ``hidden``, the hiding tile."""
    return {"row_b": bind(Pc4RowBDef, K=k), "row_a": bind(Pc4RowADef, K=k),
            "tile": bind(Pc4TileHiddenDef if hidden else Pc4TileDef, K=k, TM=tile[0], TN=tile[1])}


# ---- the Programs --------------------------------------------------------------------------------------------------------

class RowK:
    """A tile check's `Shape` attribute (``ports``, ``derivations``) that is ``today`` on the class and on a shape without
    ``row_k``, and with ``row_k`` set `rowk`'s attribute ``name``: the cut's roles and anchors.  So the class and every
    shape without the cut read today's dicts, which `rowk` leaves alone."""

    def __init__(self, today: dict, name: str) -> None:
        self.today, self.name = today, name

    def __get__(self, shape, cls=None) -> dict:
        if shape is None or shape.row_k is None:
            return self.today
        from . import rowk
        return getattr(rowk, self.name)


@dataclass(frozen=True)
class Shape:
    """    M rows of A and N rows of B at depth k, cut into tiles of ``tile`` (rows of A, rows of B; the scheme's is 64 x 64);
    ``hidden``, the hiding Programs; ``row_k``, each row unit cut into k / row_k segments (`rowk`; None, today's
    Programs).  `anchors.TileAnchors` reads its units' ports and its anchors' derivations."""

    m: int
    k: int
    n: int
    tile: tuple[int, int] = TILE
    hidden: bool = False
    row_k: int | None = None
    ports = RowK(PORTS, "PORTS4")
    derivations = RowK(DERIVATIONS, "DERIVATIONS4")
    seeds: ClassVar[dict] = SEEDS

    def check(self) -> None:
        tm, tn = self.tile
        if self.k % ATOM or min(self.m, self.k, self.n, tm, tn) <= 0 or self.m % tm or self.n % tn:
            raise DomainError(f"Pearl-C4's circuit needs 64 | k and whole tiles on each side: {self}")
        if self.row_k is not None:
            from . import rowk
            rowk.check4(self)


def _weight_units(B, k: int, b, eb, fa, fb, g, row_k: int | None = None):
    if row_k is not None:
        from . import rowk
        return rowk.rows4(B, k, row_k, "b", b, eb, fb, fa, g)
    rows = B.batch(templates(k)["row_b"], b, eb, fb, fa, g, axes=(0, 0, None, None, None))
    n = len(b)
    return [array_of([rows[j][f] for j in range(n)]) for f in range(4)]


def _call_units(B, k: int, tile: tuple[int, int], a, ea, fa, fb, bc, bs, pb, hidden: bool = False,
                row_k: int | None = None):
    """The call's units: (C̃, U, ok), or under hiding (the tile digests by tile row and column, ok); with ``row_k``, each
    A row's unit cut into `rowk`'s pieces."""
    tm, tn = tile
    tp = templates(k, tile, hidden)
    m, n = len(a), len(bc)
    if row_k is None:
        rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
        ac, as_, pa, ok = (array_of([rows[i][f] for i in range(m)]) for f in range(4))
    else:
        from . import rowk
        ac, as_, pa, ok = rowk.rows4(B, k, row_k, "a", a, ea, fa, fb)
    ms, ns = m // tm, n // tn
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    acs, ass, pas = blocks(ac, ms, tm), blocks(as_, ms, tm), blocks(pa, ms, tm)
    bcs, bss, pbs = blocks(bc, ns, tn), blocks(bs, ns, tn), blocks(pb, ns, tn)
    tiles = [B.batch(tp["tile"], acs[rs], ass[rs], bcs, bss, pas[rs], pbs, axes=(None, None, 0, 0, None, 0))
             for rs in range(ms)]
    if hidden:
        return array_of([array_of([tiles[rs][cs] for cs in range(ns)]) for rs in range(ms)]), ok
    cw, uw = (array_of([array_of([tiles[i // tm][j // tn][f][i % tm][j % tn] for j in range(n)]) for i in range(m)])
              for f in range(2))
    return cw, uw, ok


def _weight_sig(S_):
    k, n = S_.K, S_.N
    return ((("b", Array(n, Array(k, BF16))), ("eb", Array(n, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(CODES(n, k), SCALES(n, k), Array(n, P_ROW), Array(n, OK)))


Pc4WeightDef = CompositeDefinition(
    "Pc4Weight", 1, ("K", "N"), _weight_sig, lambda B, S_, *a: tuple_of(*_weight_units(B, S_.K, *a)),
    doc="A weight's side, once per job and weight: per row B̃'s codes and scale bytes, P_B and its admission (G is anchored).",
    conformance="protocol-defined")


def _call_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, BF16))), ("ea", Array(m, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)),
             ("bc", CODES(n, k)), ("bs", SCALES(n, k)), ("pb", Array(n, P_ROW))),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, OK)))


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
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, OK), Array(n, OK)))


def _check_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, bs, pb, ok_b = _weight_units(B, S_.K, b, eb, fa, fb, g)
    cw, uw, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb)
    return tuple_of(cw, uw, ok_a, ok_b)


Pc4CheckDef = CompositeDefinition(
    "Pc4Check", 1, ("M", "K", "N", "TM", "TN"), _check_sig, _check_body,
    doc="The check of an M x N block in one root: the weight's units, then the call's, reading them.",
    conformance="protocol-defined")


# ---- the hiding Programs: a tile's C̃ and U only as its tile digest --------------------------------------------------------

DIGESTS = lambda S_: Array(S_.M // S_.TM, Array(S_.N // S_.TN, L.DIGEST))          # noqa: E731


def _check_hidden_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, bs, pb, ok_b = _weight_units(B, S_.K, b, eb, fa, fb, g)
    digests, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), a, ea, fa, fb, bc, bs, pb, hidden=True)
    return tuple_of(digests, ok_a, ok_b)


Pc4CallHiddenDef = CompositeDefinition(
    "Pc4CallHidden", 1, ("M", "K", "N", "TM", "TN"),
    lambda S_: (_call_sig(S_)[0], Tuple(DIGESTS(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*_call_units(B, S_.K, (S_.TM, S_.TN), *a, hidden=True)),
    doc="`Pc4Call` hiding: each tile's digest of C̃ and U by tile row and column, and each A row's admission.",
    conformance="protocol-defined")

Pc4CheckHiddenDef = CompositeDefinition(
    "Pc4CheckHidden", 1, ("M", "K", "N", "TM", "TN"),
    lambda S_: (_check_sig(S_)[0], Tuple(DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK))), _check_hidden_body,
    doc="`Pc4Check` hiding: `Pc4CallHidden`'s outputs, then each B row's admission.", conformance="protocol-defined")


def weight(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.row_k is not None:
        from . import rowk
        return rowk.root4(shape, "weight")
    return bind(Pc4WeightDef, K=shape.k, N=shape.n)


def call(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.row_k is not None:
        from . import rowk
        return rowk.root4(shape, "call")
    return bind(Pc4CallHiddenDef if shape.hidden else Pc4CallDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1])


def check(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.row_k is not None:
        from . import rowk
        return rowk.root4(shape, "check")
    return bind(Pc4CheckHiddenDef if shape.hidden else Pc4CheckDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1])


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = {"weight": ("row_b",), "call": ("row_a", "tile"), "check": ("row_b", "row_a", "tile")}
_ROLE_OF = {id(Pc4RowBDef): "row_b", id(Pc4RowADef): "row_a", id(Pc4TileDef): "tile", id(Pc4TileHiddenDef): "tile"}


def _role_of() -> dict[int, str]:
    from . import rowk
    return _ROLE_OF | rowk.ROLE_OF


def query(shape: Shape, side: str = "check") -> dict[str, Any]:
    """`Q_template_instances` over the side's unit templates (`ROLES`, or with ``row_k`` `rowk.ROLES`), in the order its
    root body first uses them."""
    if shape.row_k is not None:
        from . import rowk
        return PO.template_instances_query(rowk.unit_templates4(shape, side))
    tp = templates(shape.k, shape.tile, shape.hidden)
    return PO.template_instances_query([tp[r] for r in ROLES[side]])


@dataclass(frozen=True)
class Unit:
    index: int
    role: str                   # a key of `templates`, or of `rowk.ROLES`
    node: int                   # its root-body node
    member: int                 # its member of that batch node
    template: str = ""          # its template's descriptor id (`plan.template_strata`'s stratum)
    where: tuple[int, ...] = field(default=())    # a row unit's row; a tile or tally unit's (tile row, tile column); a
                                                  # word unit's (row, column); a `rowk` form's (row, segment)

#: the roles whose units sit on a grid, one batch node per grid row (`cap`'s word, row and tally units beside the tile)
GRID_ROLES = ("tile", "word", "row_digest", "row_tally", "tally")


def units(fn: SpecializedDefinition, role_of: dict[int, str] | None = None) -> tuple[Unit, ...]:
    """Every unit of a Program built from the unit templates, in the partition's numbering (node order, then member);
    ``role_of`` maps each template's Definition (by id) to its role, Pearl-C4's by default."""
    from . import rowk

    role_of = _role_of() if role_of is None else role_of
    out: list[Unit] = []
    grid_rows = dict.fromkeys(GRID_ROLES, 0)
    for j, nd in enumerate(fn.body.nodes):
        role, tid = role_of[id(nd.fn.defn)], descriptor_id(nd.fn)
        for m in range(nd.n if nd.form == "batch" else 1):
            where = (grid_rows[role], m) if role in grid_rows else (m, rowk.segment(nd.fn)) if role in rowk.FORMS else (m,)
            out.append(Unit(len(out), role, j, m, tid, where))
        if role in grid_rows:
            grid_rows[role] += 1
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


__all__ = ["BASIS", "DERIVATIONS", "GRAM", "GRID_ROLES", "LINE", "PORTS", "P_ROW", "ROLES", "SEEDS", "TILE", "Shape", "Unit",
           "basis_fa", "basis_fb", "call", "check", "gram_g", "line_ea", "line_eb", "ok_word", "query", "templates", "unit_cut",
           "units", "weight"]

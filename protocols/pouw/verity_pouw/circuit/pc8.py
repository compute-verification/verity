"""Pearl-C's tile check as a Program (``pearl-c-h100-v1``, ``pearl-c-sm120-v1``, and the unpromoted sm_120 record): forming
v1, the chain and the clean-up (`schemes.pearl_c`) cut into proof units as `pc4` cuts Pearl-C4's, so sampled proofs can
establish a drawn tile's checked words (C̃ and U) and its rows' admission without the verifier reading a row.

    template              one instance per        reads                                   computes and returns
    Pc8RowB{K,DEV}        row j of B              B_j, E_B,j, F_B, F_A, G                  B̃_j's codes, P_B,j, ok
      (in Pc8Weight, once per job and weight)
    Pc8RowA{K,DEV}        row i of A              A_i, E_A,i, F_A, F_B                     A′_i's codes, P_A,i, ok
    Pc8Tile{K,TM,TN,DEV}  tile of TM × TN words   its TM rows A′, P_A and TN rows B̃, P_B   C̃ and U of its words

The roots are `pc4`'s: `Pc8Weight{K,N,DEV}` holds the B rows; `Pc8Call{M,K,N,TM,TN,DEV}` holds the A rows and the tiles,
reading the weight's B̃ and P_B as committed inputs; `Pc8Check` is both in one root.  The partition is
`Q_template_instances` over the templates (`query`), and every value is computed in exactly one unit.  Forming v0 (#311's
port) has no circuit here.  The hiding Programs are `pc4`'s (``Shape.hidden``): `Pc8TileHidden` returns its tile digest in
place of C̃ and U, and the rows are C-Flock-committed ports (`PORTS`), with fp32 input rows, fp8-p rows (A′_i ‖ P_A,i,
B̃_j ‖ P_B,j; `leaves`) and each row's ``ok`` row, and the anchored inputs public input regions.  ``Shape.row_k`` cuts
each row unit along k into a head, a form per segment of row_k positions and a tail (`rowk`), so that a row unit fits
C-Flock's block at every served k, and ``Shape.tile_sub`` each tile unit into sub-tiles (`tilesub`); off, nothing here
changes.

``DEV`` names a device record (`pearl_c_device.DEVICES`): its E4M3 k32 atom (`HopperE4m3QgmmaDot32` on the H100,
`BlackwellE4m3QmmaDot32` on sm_120) and its promotion period G.  The peel's BF16 k16 step is `HopperBF16WgmmaDot16` on
every record (sm_120's stand-in, `HOPPER_BF16_M16N8K16`, has its parameters).  The scheme's −0 → +0 after a promotion never
fires: the atom's zero is +0, the total starts at +0, and a nearest-even sum is −0 only of two −0s.

A row unit computes `s5_stats`, E′ = e4m3(β·E), the forming atoms and their casts, and its half of the clean-up: P_A,i =
[bf16(E′_A,i) ‖ RNE_bf16(A′_i·F_B)], P_B,j = [RNE_bf16(Σ_p fma(E′_B,jp, G_pq) − B̃_j·F_A) ‖ −bf16(E′_B,j)].  ``ok`` is the
row's admission: `parse_f32_row`'s and `s5_stats`'s finiteness, then TT_OUT's `OperandOK` (`PearlC.admissible`), as

    s <= F32_MAX              every word finite
    ρ <= F32_MAX              every partial and butterfly sum finite: they are sums of squares, so an overflow reaches ρ
    fma(ρ, α, −1) >= +0       the noise floor ρ·α >= 1 on the exact product, which also refuses a non-finite t (α = 0):
                              where ρ·α is near 1, ρ·α − 1 is zero or at least 2^-49, so the one rounding keeps its sign
    #dead <= k/64             `live_row`, exactly: 2^300·x_l² against ⌈64·Σ_p 2^300·x_p² / (k/8)⌉ (`words.PouwSq*`)

No other rejection of the scheme's is reachable on such a row: β·E and α·x are bounded (ρ·α <= 112 and |α·x| <= 448), so
every atom, cast, chain and peel factor is finite.  E′_B's FP32 decode is the FMA's operand: bf16(E′) is exact, so it widens
back to that decode.

The noise lines (E_A, E_B, F_A, F_B) and G = F_Bᵀ·F_A (`pearl_c.gram`) are verifier-derived public inputs
(`anchors.TileAnchors`, `DERIVATIONS`): the verifier derives the lines from public seeds and computes G from them, so no
unit computes G.  In a unit's proof each is a C-Flock public input region, as in `pc4`: one u16 word per E4M3 code, two
per FP32 word of G, low half first.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar

from verity.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ir import partition_object as PO
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import array_of, tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.prims import (BlackwellE4m3QmmaDot32, E4m3ToF32, F2fpBf16, F32ToE4m3Sat, HopperBF16WgmmaDot16,
                             HopperE4m3QgmmaDot32)
from verity.ml.scalar import BitAnd, F32Fabs, F32Fmaxf, F32Neg, I32Add, I32Le, SelectI32

from ..protocol import DomainError
from ..schemes import pearl_c as C
from ..schemes import pearl_kw as P
from ..schemes.pearl_c_device import DEVICES
from . import anchors as A
from . import leaves as L
from . import pc4
from . import words as W

F32, BF16, E4M3, OK = W.F32, W.BF16, Value(8), L.OK
R, SLICE, PEEL = C.R, C.SLICE, C.PEEL_K
LINE = Array(R, E4M3)
P_ROW = Array(PEEL, BF16)
GRAM = Array(R, Array(R, F32))
F32_MAX, MINUS_ONE = 0x7F7FFFFF, 0xBF800000
TILE = (C.TILE, C.TILE)
#: each device's FP8 atom as a core primitive, by its model's name; the peel's step is the same primitive on every record
ATOMS = {"hopper_e4m3_wgmma_k32": HopperE4m3QgmmaDot32, "blackwell_sm120_e4m3_m16n8k32": BlackwellE4m3QmmaDot32}
PEELS = {"hopper_bf16_wgmma_k16": HopperBF16WgmmaDot16, "hopper_bf16_m16n8k16": HopperBF16WgmmaDot16}

BASIS = lambda k: Array(k, LINE)                                                    # noqa: E731
CODES = lambda n, k: Array(n, Array(k, E4M3))                                       # noqa: E731


def _chain(e: pc4._Emit, a, b, k: int, dev: str, acc=0):
    """The device's replay of a·b (`pearl_c.chain`): per window of G atoms a fresh +0 accumulator chains them, and one
    FP32 add promotes it into the total; unpromoted, one accumulator from +0 over all of k.  ``acc`` is the total (or the
    accumulator) the chain starts from: +0, or the carry of the k positions before ``a`` and ``b`` (`rowk`)."""
    d = DEVICES[dev]
    atom = ATOMS[d.atom.name]
    step = lambda acc, s: e.call(atom, acc, a[s:s + SLICE], b[s:s + SLICE])   # noqa: E731
    if d.g is None:
        for s in range(0, k, SLICE):
            acc = step(acc, s)
        return acc
    total = acc
    for ws in range(0, k, d.window):
        local = 0
        for s in range(ws, ws + d.window, SLICE):
            local = step(local, s)
        total = e.call(F32AddV2, total, local)
    return total


def _moments(e: pc4._Emit, x, k: int):
    """(s, ρ, α, β_row, θ) of one row of k FP32 words: max |x|, `pearl_c.s5_stats` and `live_row`'s threshold."""
    mag = [e.call(F32Fabs, x[t]) for t in range(k)]
    s = mag[0]
    for m in mag[1:]:
        s = e.call(SelectI32, e.call(I32Le, s, m), m, s)
    positions = range(0, k, C.S5_STRIDE)
    part = [0] * 32
    for i, p in enumerate(positions):
        part[i % 32] = e.call(F32FmaV2, x[p], x[p], part[i % 32])
    o = 16
    while o:
        part = [e.call(F32AddV2, part[j], part[j + o]) for j in range(o)]
        o //= 2
    rho = e.call(W.PouwF32SqrtRn, e.call(F32DivV2, part[0], pc4._f32(float(len(positions)))))
    t = e.call(F32AddV2, s, e.call(F32MulV2, C.S5_FOUR, rho))
    alpha = e.call(F32DivV2, C.S5_QMAX, e.call(F32Fmaxf, t, C.S5_FLOOR))
    beta = e.call(F32MulV2, e.call(F32MulV2, rho, alpha), C.S5_LINE)
    total = 0
    for p in positions:
        total = e.call(W.PouwSqAcc, total, x[p])
    th = e.call(W.PouwSqCeil, total, len(positions))
    return s, rho, alpha, beta, th


def _dead(e: pc4._Emit, x, th, lo: int, hi: int, dead=0):
    """``dead`` plus the number of positions t in [lo, hi) whose word is dead at the threshold θ (`live_row`)."""
    for t_ in range(lo, hi):
        dead = e.call(I32Add, dead, e.call(SelectI32, e.call(W.PouwSqGe, x[t_], th), 1, 0))
    return dead


def _bounds(e: pc4._Emit, s, rho, alpha):
    """The row's noise floor bit and its finiteness bit (module docstring)."""
    floor = e.call(I32Le, 0, e.call(F32FmaV2, rho, alpha, MINUS_ONE))
    finite = e.call(BitAnd, e.call(I32Le, s, F32_MAX), e.call(I32Le, rho, F32_MAX))
    return floor, finite


def _stats(e: pc4._Emit, x, k: int):
    """(α, β_row, ok) of one row of k FP32 words (`pearl_c.s5_stats`, `live_row`, the noise floor; module docstring)."""
    s, rho, alpha, beta, th = _moments(e, x, k)
    dead = _dead(e, x, th, 0, k)
    floor, finite = _bounds(e, s, rho, alpha)
    ok = e.call(BitAnd, finite, e.call(BitAnd, floor, e.call(I32Le, dead, k // 64)))
    return alpha, beta, ok


def _ep(e: pc4._Emit, beta, line):
    """E′ = e4m3(β·E), the row's scaled noise line."""
    return [e.call(F32ToE4m3Sat, e.call(F32MulV2, beta, e.call(E4m3ToF32, line[p]))) for p in range(R)]


def _codes(e: pc4._Emit, x, alpha, ep_line, basis, lo: int, hi: int, dev: str):
    """The codes e4m3(atom(α·x_t, E′, F_t)) of positions t in [lo, hi), ``basis`` holding F's lines from position lo."""
    atom = ATOMS[DEVICES[dev].atom.name]
    return [e.call(F32ToE4m3Sat, e.call(atom, e.call(F32MulV2, alpha, x[t]), ep_line, basis[t - lo])) for t in range(lo, hi)]


def _form(e: pc4._Emit, x, alpha, beta, line, basis, k: int, dev: str):
    """A row's E′ = e4m3(β·E) (with E′'s FP32 decode) and its codes e4m3(atom(α·x_t, E′, F_t)) (`pearl_c.form_v1`)."""
    ep = _ep(e, beta, line)
    ew = [e.call(E4m3ToF32, c) for c in ep]
    return array_of(_codes(e, x, alpha, array_of(ep), basis, 0, k, dev)), ew


def _row_sig(S_, second: str, extra: tuple = ()):
    k = S_.K
    first = "fb" if second == "fa" else "fa"
    params = (("x", Array(k, F32)), ("e", LINE), (first, BASIS(k)), (second, BASIS(k))) + extra
    return params, Tuple(Array(k, E4M3), P_ROW, OK)


def _row_a_body(B, S_, x, line, fa, fb):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    alpha, beta, ok = _stats(e, x, k)
    codes, ew = _form(e, x, alpha, beta, line, fa, k, dev)
    af = [e.call(F2fpBf16, _chain(e, codes, fb.take(1, p), k, dev)) for p in range(R)]
    return tuple_of(codes, array_of([e.call(F2fpBf16, w) for w in ew] + af), pc4.ok_word(e, ok))


Pc8RowADef = CompositeDefinition(
    "Pc8RowA", 1, ("K", "DEV"), lambda S_: _row_sig(S_, "fb"), _row_a_body,
    doc="Row i of A: its admission, A′_i's codes formed against F_A, and P_A,i = [bf16(E′_A,i) ‖ RNE_bf16(A′_i·F_B)].",
    conformance="protocol-defined")


def _row_b_body(B, S_, x, line, fb, fa, g):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    alpha, beta, ok = _stats(e, x, k)
    codes, ew = _form(e, x, alpha, beta, line, fb, k, dev)
    mid = []
    for q in range(R):
        acc = 0
        for p in range(R):
            acc = e.call(F32FmaV2, ew[p], g[p][q], acc)
        bf = _chain(e, codes, fa.take(1, q), k, dev)
        mid.append(e.call(F2fpBf16, e.call(F32AddV2, acc, e.call(F32Neg, bf))))
    return tuple_of(codes, array_of(mid + [e.call(W.PouwBf16Neg, e.call(F2fpBf16, w)) for w in ew]), pc4.ok_word(e, ok))


Pc8RowBDef = CompositeDefinition(
    "Pc8RowB", 1, ("K", "DEV"), lambda S_: _row_sig(S_, "fa", (("g", GRAM),)), _row_b_body,
    doc="Row j of B: its admission, B̃_j's codes formed against F_B, and P_B,j = [RNE_bf16(E′_B,j·G − B̃_j·F_A) ‖ "
        "−bf16(E′_B,j)] (E′_B,j·G the FP32 FMA chain from +0 in p order); once per job and weight.",
    conformance="protocol-defined")


def _tile_body(B, S_, ac, bc, pa, pb):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    peel = PEELS[DEVICES[dev].peel.name]
    cs, us = [], []
    for i in range(S_.TM):
        crow, urow = [], []
        for j in range(S_.TN):
            c = _chain(e, ac[i], bc[j], k, dev)
            u = c
            for s in range(0, PEEL, 16):
                u = e.call(peel, u, pa[i][s:s + 16], pb[j][s:s + 16])
            crow.append(c)
            urow.append(u)
        cs.append(array_of(crow))
        us.append(array_of(urow))
    return tuple_of(array_of(cs), array_of(us))


def _tile_sig(S_):
    k, tm, tn = S_.K, S_.TM, S_.TN
    return ((("ac", CODES(tm, k)), ("bc", CODES(tn, k)), ("pa", Array(tm, P_ROW)), ("pb", Array(tn, P_ROW))),
            Tuple(Array(tm, Array(tn, F32)), Array(tm, Array(tn, F32))))


Pc8TileDef = CompositeDefinition(
    "Pc8Tile", 1, ("K", "TM", "TN", "DEV"), _tile_sig, _tile_body,
    doc="One TM x TN tile's checked words: C̃_ij, the device's chain over A′_i·B̃_j, and U_ij, four BF16 k16 steps from "
        "C̃_ij over P_A,i·P_B,j (`pearl_c.chain`, `pearl_c.peel`).", conformance="protocol-defined")


# ---- the hiding tile (`pc4`'s) ------------------------------------------------------------------------------------------

def _tile_hidden_body(B, S_, ac, bc, pa, pb):
    k, tm, tn = S_.K, S_.TM, S_.TN
    out = B.call(templates(k, (tm, tn), S_.DEV)["tile"], ac, bc, pa, pb)
    return B.call(L.tile_digest(tm, tn), out[0], out[1])


Pc8TileHiddenDef = CompositeDefinition(
    "Pc8TileHidden", 1, ("K", "TM", "TN", "DEV"), lambda S_: (_tile_sig(S_)[0], L.DIGEST), _tile_hidden_body,
    doc="`Pc8Tile` with its checked words hidden: their tile digest (`leaves.PearlTileDigest`, `pearl_c.leaf` of C̃ and U).",
    conformance="protocol-defined")

#: each unit's ports under hiding, as `pc4.PORTS`: fp32 input rows, fp8-p rows and ok rows, and the anchors' regions
PORTS = {
    "row_b": {"rows": (("fp32", ROLE_W, ("x",)), ("fp8-p", ROLE_W, (0, 1)), ("ok", ROLE_W, (2,))),
              "regions": {"e": "eb", "fb": "fb", "fa": "fa", "g": "g"}},
    "row_a": {"rows": (("fp32", ROLE_X, ("x",)), ("fp8-p", ROLE_X, (0, 1)), ("ok", ROLE_X, (2,))),
              "regions": {"e": "ea", "fa": "fa", "fb": "fb"}},
    "tile": {"rows": (("fp8-p", ROLE_X, ("ac", "pa")), ("fp8-p", ROLE_W, ("bc", "pb")), ("tile", ROLE_X, (0,))),
             "regions": {}},
}


# ---- the anchors: public input regions, each with its recomputation ------------------------------------------------------

def line_ea(shape, seed_a: bytes, i: int) -> list[int]:
    """E_A's line for A's global row i: `pearl_kw.sample_line(seed_A, 0, 0, i)` (norm 256)."""
    return P.sample_line(seed_a, 0, 0, i)


def line_eb(shape, seed_b: bytes, j: int) -> list[int]:
    """E_B's line for B's global row j: `pearl_kw.sample_line(seed_B, 1, 0, j)` (norm 256)."""
    return P.sample_line(seed_b, 1, 0, j)


def basis_fa(shape, seed_b: bytes, _i=None) -> list[int]:
    """F_A, line t for position t < k: `pearl_c._basis(seed_B, 0, k, LINE_NORM_V1)`."""
    return [c for ln in C._basis(seed_b, 0, shape.k, C.LINE_NORM_V1) for c in ln]


def basis_fb(shape, seed_b: bytes, _i=None) -> list[int]:
    """F_B, line t for position t < k: `pearl_c._basis(seed_B, 1, k, LINE_NORM_V1)`."""
    return [c for ln in C._basis(seed_b, 1, shape.k, C.LINE_NORM_V1) for c in ln]


def gram_g(shape, seed_b: bytes, _i=None) -> list[int]:
    """G = F_Bᵀ·F_A by row p, then column q, on the record's chain: `pearl_c.gram(seed_B, k, LINE_NORM_V1, DEVICES[dev])`."""
    return [w for row in C.gram(seed_b, shape.k, C.LINE_NORM_V1, DEVICES[shape.dev]) for w in row]


_LINE = ("BLAKE3 keyed by keyed(seed, \"pearl/v4/FP8/noise-line\") over side ‖ factor ‖ index u32 LE zero-padded to 64 "
         "bytes; byte b is ±((b & 0x7F) + 1), bit 7 the sign; scaled in BF16 to L2 norm {} and cast to E4M3 "
         "(`pearl_kw.sample_line`); ")
#: each anchor's recomputation from `pearl_c`'s seeds (`noise`: seed_A = BLAKE3 keyed by blake3("pearl-c/v0/seed-A") over
#: salt ‖ root_A ‖ root_B ‖ matmul u64 LE, seed_B keyed by blake3("pearl-c/v0/seed-B") over salt ‖ root_B), each under its
#: rule's versioned id (G's chain is the device's, which the unit's ``DEV`` binds)
DERIVATIONS = {
    "ea": A.Derivation("verity/pouw/anchor/pearl-c/ea/v1", "seed_a",
                       _LINE.format(256) + "side 0, factor 0, index A's global row", line_ea, 8, per_row=True),
    "eb": A.Derivation("verity/pouw/anchor/pearl-c/eb/v1", "seed_b",
                       _LINE.format(256) + "side 1, factor 0, index B's global row", line_eb, 8, per_row=True),
    "fa": A.Derivation("verity/pouw/anchor/pearl-c/fa/v1", "seed_b",
                       _LINE.format(16) + "side 0, factor 1, index the position t < k", basis_fa, 8),
    "fb": A.Derivation("verity/pouw/anchor/pearl-c/fb/v1", "seed_b",
                       _LINE.format(16) + "side 1, factor 1, index the position t < k", basis_fb, 8),
    "g": A.Derivation("verity/pouw/anchor/pearl-c/g/v1", "seed_b",
                      "G_pq = the record's chain (`pearl_c.chain`: its atom and promotion) over F_B[:, p]·F_A[:, q]; "
                      "FP32 words", gram_g, 32),
}
#: each seed as a public input port names it: the scheme's label for it
SEEDS = {"seed_a": C.LABELS["seed-A"].decode(), "seed_b": C.LABELS["seed-B"].decode()}


@lru_cache(maxsize=16)
def templates(k: int, tile: tuple[int, int] = TILE, dev: str = "sm120",
              hidden: bool = False) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k``, tile ``tile`` (rows, columns) and device ``dev``, by role; ``hidden``, the
    hiding tile."""
    return {"row_b": bind(Pc8RowBDef, K=k, DEV=dev), "row_a": bind(Pc8RowADef, K=k, DEV=dev),
            "tile": bind(Pc8TileHiddenDef if hidden else Pc8TileDef, K=k, TM=tile[0], TN=tile[1], DEV=dev)}


# ---- the Programs --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """M rows of A and N rows of B at depth k on device ``dev``, cut into tiles of ``tile`` (the scheme's is 64 x 64);
    ``hidden``, the hiding Programs; ``row_k``, each row unit cut into k / row_k segments (`rowk`); ``tile_sub``, each tile
    unit cut into sub-tiles of (a, b) words (`tilesub`); None, today's Programs.  `anchors.TileAnchors` reads its units'
    ports and its anchors' derivations."""

    m: int
    k: int
    n: int
    tile: tuple[int, int] = TILE
    dev: str = "sm120"
    hidden: bool = False
    row_k: int | None = None
    tile_sub: tuple[int, int] | None = None
    ports = pc4.Cut(PORTS, "PORTS8")
    derivations = pc4.Cut(DERIVATIONS, "DERIVATIONS8")
    seeds: ClassVar[dict] = SEEDS

    def check(self) -> None:
        d = DEVICES.get(self.dev)
        tm, tn = self.tile
        if d is None or d.atom.name not in ATOMS or d.peel.name not in PEELS:
            raise DomainError(f"Pearl-C's circuit has no device {self.dev!r}: one of {', '.join(DEVICES)}")
        if (self.k % d.window or not 0 < self.k <= d.k_max or min(self.m, self.n, tm, tn) <= 0 or self.m % tm
                or self.n % tn):
            raise DomainError(f"Pearl-C's circuit needs {d.window} | k <= {d.k_max} and whole tiles on each side: {self}")
        if self.row_k is not None:
            from . import rowk
            rowk.check8(self)
        if self.tile_sub is not None:
            from . import tilesub
            tilesub.check(self)


def _weight_units(B, k: int, dev: str, b, eb, fa, fb, g, row_k: int | None = None):
    if row_k is not None:
        from . import rowk
        return rowk.rows8(B, k, row_k, dev, "b", b, eb, fb, fa, g)
    rows = B.batch(templates(k, TILE, dev)["row_b"], b, eb, fb, fa, g, axes=(0, 0, None, None, None))
    return [array_of([rows[j][f] for j in range(len(b))]) for f in range(3)]


def _call_units(B, k: int, tile: tuple[int, int], dev: str, a, ea, fa, fb, bc, pb, hidden: bool = False,
                row_k: int | None = None, sub: tuple[int, int] | None = None):
    """The call's units: (C̃, U, ok), or under hiding (the tile digests by tile row and column, ok); with ``row_k``, each
    A row's unit cut into `rowk`'s pieces, and with ``sub`` each tile's into `tilesub`'s sub-tiles and digest units."""
    tm, tn = sub or tile
    tp = templates(k, sub or tile, dev, hidden and sub is None)
    m, n = len(a), len(bc)
    if row_k is None:
        rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
        ac, pa, ok = (array_of([rows[i][f] for i in range(m)]) for f in range(3))
    else:
        from . import rowk
        ac, pa, ok = rowk.rows8(B, k, row_k, dev, "a", a, ea, fa, fb)
    ms, ns = m // tm, n // tn
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    acs, pas, bcs, pbs = blocks(ac, ms, tm), blocks(pa, ms, tm), blocks(bc, ns, tn), blocks(pb, ns, tn)
    if sub is not None:
        from . import tilesub
        subs = [B.batch(tilesub.sub_tile8(k, sub, dev), acs[rs], bcs, pas[rs], pbs, axes=(None, 0, None, 0))
                for rs in range(ms)]
        return (*tilesub.outputs(B, subs, m, n, tile, sub, hidden), ok)
    tiles = [B.batch(tp["tile"], acs[rs], bcs, pas[rs], pbs, axes=(None, 0, None, 0)) for rs in range(ms)]
    if hidden:
        return array_of([array_of([tiles[rs][cs] for cs in range(ns)]) for rs in range(ms)]), ok
    cw, uw = (array_of([array_of([tiles[i // tm][j // tn][f][i % tm][j % tn] for j in range(n)]) for i in range(m)])
              for f in range(2))
    return cw, uw, ok


def _weight_sig(S_):
    k, n = S_.K, S_.N
    return ((("b", Array(n, Array(k, F32))), ("eb", Array(n, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(CODES(n, k), Array(n, P_ROW), Array(n, OK)))


Pc8WeightDef = CompositeDefinition(
    "Pc8Weight", 1, ("K", "N", "DEV"), _weight_sig, lambda B, S_, *a: tuple_of(*_weight_units(B, S_.K, S_.DEV, *a)),
    doc="A weight's side, once per job and weight: per row B̃'s codes, P_B and its admission (G is anchored).",
    conformance="protocol-defined")


def _call_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, F32))), ("ea", Array(m, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)),
             ("bc", CODES(n, k)), ("pb", Array(n, P_ROW))),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, OK)))


Pc8CallDef = CompositeDefinition(
    "Pc8Call", 1, ("M", "K", "N", "TM", "TN", "DEV"), _call_sig,
    lambda B, S_, *a: tuple_of(*_call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a)),
    doc="One call's side of an M x N block: its A rows formed, C̃ and U for every word, tile by tile, and each A row's "
        "admission; its weight's B̃ codes and P_B rows are committed inputs (`Pc8Weight`).",
    conformance="protocol-defined")


def _check_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, F32))), ("ea", Array(m, LINE)), ("b", Array(n, Array(k, F32))), ("eb", Array(n, LINE)),
             ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, OK), Array(n, OK)))


def _check_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, pb, ok_b = _weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    cw, uw, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb)
    return tuple_of(cw, uw, ok_a, ok_b)


Pc8CheckDef = CompositeDefinition(
    "Pc8Check", 1, ("M", "K", "N", "TM", "TN", "DEV"), _check_sig, _check_body,
    doc="The check of an M x N block in one root: the weight's units, then the call's, reading them.",
    conformance="protocol-defined")


# ---- the hiding Programs (`pc4`'s) ---------------------------------------------------------------------------------------

def _check_hidden_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, pb, ok_b = _weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    digests, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, hidden=True)
    return tuple_of(digests, ok_a, ok_b)


Pc8CallHiddenDef = CompositeDefinition(
    "Pc8CallHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (_call_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*_call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a, hidden=True)),
    doc="`Pc8Call` hiding: each tile's digest of C̃ and U by tile row and column, and each A row's admission.",
    conformance="protocol-defined")

Pc8CheckHiddenDef = CompositeDefinition(
    "Pc8CheckHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (_check_sig(S_)[0], Tuple(pc4.DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK))), _check_hidden_body,
    doc="`Pc8Check` hiding: `Pc8CallHidden`'s outputs, then each B row's admission.", conformance="protocol-defined")


def weight(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "weight")
    return bind(Pc8WeightDef, K=shape.k, N=shape.n, DEV=shape.dev)


def call(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.tile_sub is not None:
        from . import tilesub
        return tilesub.root8(shape, "call")
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "call")
    return bind(Pc8CallHiddenDef if shape.hidden else Pc8CallDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


def check(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.tile_sub is not None:
        from . import tilesub
        return tilesub.root8(shape, "check")
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "check")
    return bind(Pc8CheckHiddenDef if shape.hidden else Pc8CheckDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = pc4.ROLES
_ROLE_OF = {id(Pc8RowBDef): "row_b", id(Pc8RowADef): "row_a", id(Pc8TileDef): "tile", id(Pc8TileHiddenDef): "tile"}


def _role_of() -> dict[int, str]:
    from . import rowk, tilesub
    return _ROLE_OF | rowk.ROLE_OF | tilesub.ROLE_OF


def query(shape: Shape, side: str = "check") -> dict[str, Any]:
    """`Q_template_instances` over the side's unit templates (`ROLES`, or with a cut `rowk.ROLES` and `tilesub.ROLES`), in
    the order its root body first uses them."""
    if shape.tile_sub is not None:
        from . import tilesub
        return PO.template_instances_query(tilesub.unit_templates8(shape, side))
    if shape.row_k is not None:
        from . import rowk
        return PO.template_instances_query(rowk.unit_templates8(shape, side))
    tp = templates(shape.k, shape.tile, shape.dev, shape.hidden)
    return PO.template_instances_query([tp[r] for r in ROLES[side]])


def units(fn: SpecializedDefinition) -> tuple[pc4.Unit, ...]:
    """Every unit of a Program built from the unit templates (`pc4.units`)."""
    return pc4.units(fn, _role_of())


def unit_cut(fn: SpecializedDefinition):
    """The whole Program's cut, checked by `cut.check_cut` (`pc4.unit_cut`)."""
    return pc4.unit_cut(fn, _role_of())


__all__ = ["ATOMS", "BASIS", "DERIVATIONS", "GRAM", "LINE", "PEELS", "PORTS", "P_ROW", "ROLES", "SEEDS", "TILE", "Shape",
           "basis_fa", "basis_fb", "call", "check", "gram_g", "line_ea", "line_eb", "query", "templates", "unit_cut", "units", "weight"]

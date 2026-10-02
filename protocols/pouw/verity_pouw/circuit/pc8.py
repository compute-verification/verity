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
port) has no circuit here.  The hiding Programs are `pc4`'s (``Shape.hidden``), with fp32 input rows and fp8-p rows
(A′_i ‖ P_A,i, B̃_j ‖ P_B,j; `leaves`).

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

The noise lines (E_A, E_B, F_A, F_B) and G = F_Bᵀ·F_A (`pearl_c.gram`) are anchored inputs: the verifier derives the lines
from public seeds and computes G from them, so no unit computes G.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from verity.ir import partition_object as PO
from verity.ir.cut import RECOMPUTE
from verity.ir.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import array_of, tuple_of
from verity.ir.types import Array, Tuple, Value
from verity.ml.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.prims import (BlackwellE4m3QmmaDot32, E4m3ToF32, F2fpBf16, F32ToE4m3Sat, HopperBF16WgmmaDot16,
                             HopperE4m3QgmmaDot32)
from verity.ml.scalar import BitAnd, F32Fabs, F32Fmaxf, F32Neg, I32Add, I32Le, SelectI32

from ..protocol import DomainError
from ..schemes import pearl_c as C
from ..schemes.pearl_c_device import DEVICES
from . import leaves as L
from . import pc4
from . import words as W

F32, BF16, E4M3, BIT = W.F32, W.BF16, Value(8), Value(1)
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


def _chain(e: pc4._Emit, a, b, k: int, dev: str):
    """The device's replay of a·b (`pearl_c.chain`): per window of G atoms a fresh +0 accumulator chains them, and one
    FP32 add promotes it into the total; unpromoted, one accumulator from +0 over all of k."""
    d = DEVICES[dev]
    atom = ATOMS[d.atom.name]
    step = lambda acc, s: e.call(atom, acc, a[s:s + SLICE], b[s:s + SLICE])   # noqa: E731
    if d.g is None:
        acc = 0
        for s in range(0, k, SLICE):
            acc = step(acc, s)
        return acc
    total = 0
    for ws in range(0, k, d.window):
        local = 0
        for s in range(ws, ws + d.window, SLICE):
            local = step(local, s)
        total = e.call(F32AddV2, total, local)
    return total


def _stats(e: pc4._Emit, x, k: int):
    """(ρ, α, β_row, ok) of one row of k FP32 words (`pearl_c.s5_stats`, `live_row`, the noise floor; module docstring)."""
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
    dead = 0
    for t_ in range(k):
        dead = e.call(I32Add, dead, e.call(SelectI32, e.call(W.PouwSqGe, x[t_], th), 1, 0))
    floor = e.call(I32Le, 0, e.call(F32FmaV2, rho, alpha, MINUS_ONE))
    finite = e.call(BitAnd, e.call(I32Le, s, F32_MAX), e.call(I32Le, rho, F32_MAX))
    ok = e.call(BitAnd, finite, e.call(BitAnd, floor, e.call(I32Le, dead, k // 64)))
    return alpha, beta, ok


def _form(e: pc4._Emit, x, alpha, beta, line, basis, k: int, dev: str):
    """A row's E′ = e4m3(β·E) (with E′'s FP32 decode) and its codes e4m3(atom(α·x_t, E′, F_t)) (`pearl_c.form_v1`)."""
    atom = ATOMS[DEVICES[dev].atom.name]
    ep = [e.call(F32ToE4m3Sat, e.call(F32MulV2, beta, e.call(E4m3ToF32, line[p]))) for p in range(R)]
    ew = [e.call(E4m3ToF32, c) for c in ep]
    ep_line = array_of(ep)
    codes = [e.call(F32ToE4m3Sat, e.call(atom, e.call(F32MulV2, alpha, x[t]), ep_line, basis[t])) for t in range(k)]
    return array_of(codes), ew


def _row_sig(S_, second: str, extra: tuple = ()):
    k = S_.K
    first = "fb" if second == "fa" else "fa"
    params = (("x", Array(k, F32)), ("e", LINE), (first, BASIS(k)), (second, BASIS(k))) + extra
    return params, Tuple(Array(k, E4M3), P_ROW, BIT)


def _row_a_body(B, S_, x, line, fa, fb):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    alpha, beta, ok = _stats(e, x, k)
    codes, ew = _form(e, x, alpha, beta, line, fa, k, dev)
    af = [e.call(F2fpBf16, _chain(e, codes, fb.take(1, p), k, dev)) for p in range(R)]
    return tuple_of(codes, array_of([e.call(F2fpBf16, w) for w in ew] + af), ok)


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
    return tuple_of(codes, array_of(mid + [e.call(W.PouwBf16Neg, e.call(F2fpBf16, w)) for w in ew]), ok)


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


# ---- the hiding units (`pc4`'s, with fp32 input rows and fp8-p rows) -----------------------------------------------------

def _row_hidden_sig(S_, second: str, extra: tuple = ()):
    params, out = _row_sig(S_, second, extra)
    return params + (("y_row", L.SALT), ("y_mid", L.SALT)), Tuple(*out.fields, L.BC, L.BC)


def _row_hidden_body(plain):
    def body(B, S_, x, *rest):
        e, k, (*args, y_row, y_mid) = pc4._Emit(B), S_.K, rest
        out = B.call(bind(plain, K=k, DEV=S_.DEV), x, *args)
        codes, p, ok = (out[f] for f in range(3))
        return tuple_of(codes, p, ok, L.leaf(e, "fp32", L.fp32_row(e, x, k), y_row),
                        L.leaf(e, "fp8-p", L.fp8_p_row(e, codes, p, k), y_mid))
    return body


Pc8RowAHiddenDef = CompositeDefinition(
    "Pc8RowAHidden", 1, ("K", "DEV"), lambda S_: _row_hidden_sig(S_, "fb"), _row_hidden_body(Pc8RowADef),
    doc="`Pc8RowA` and the commit strings of A_i (fp32, salt y_row) and of A′_i ‖ P_A,i (fp8-p, salt y_mid), each the row "
        "C-Flock commits (`leaves.ROW_SCHEMA`).", conformance="protocol-defined")

Pc8RowBHiddenDef = CompositeDefinition(
    "Pc8RowBHidden", 1, ("K", "DEV"), lambda S_: _row_hidden_sig(S_, "fa", (("g", GRAM),)),
    _row_hidden_body(Pc8RowBDef),
    doc="`Pc8RowB` and the commit strings of B_j (fp32, salt y_row) and of B̃_j ‖ P_B,j (fp8-p, salt y_mid), each the row "
        "C-Flock commits (`leaves.ROW_SCHEMA`).", conformance="protocol-defined")


def _tile_hidden_body(B, S_, ac, bc, pa, pb, ya, yb, yt):
    e, k, tm, tn = pc4._Emit(B), S_.K, S_.TM, S_.TN
    out = B.call(templates(k, (tm, tn), S_.DEV)["tile"], ac, bc, pa, pb)
    reads_a = [L.leaf(e, "fp8-p", L.fp8_p_row(e, ac[i], pa[i], k), ya[i]) for i in range(tm)]
    reads_b = [L.leaf(e, "fp8-p", L.fp8_p_row(e, bc[j], pb[j], k), yb[j]) for j in range(tn)]
    return tuple_of(array_of(reads_a), array_of(reads_b), L.tile_leaf(e, tm, tn, out[0], out[1], yt))


def _tile_hidden_sig(S_):
    params, _ = _tile_sig(S_)
    return (params + (("ya", pc4.SALTS(S_.TM)), ("yb", pc4.SALTS(S_.TN)), ("yt", L.SALT)),
            Tuple(pc4.BCS(S_.TM), pc4.BCS(S_.TN), L.BC))


Pc8TileHiddenDef = CompositeDefinition(
    "Pc8TileHidden", 1, ("K", "TM", "TN", "DEV"), _tile_hidden_sig, _tile_hidden_body,
    doc="`Pc8Tile` with its checked words hidden: the commit strings of the fp8-p rows it reads (A′_i ‖ P_A,i under ya_i, "
        "B̃_j ‖ P_B,j under yb_j) and of its tile digest (`leaves.PearlTileDigest`, a tile row, salt yt).",
    conformance="protocol-defined")


@lru_cache(maxsize=16)
def templates(k: int, tile: tuple[int, int] = TILE, dev: str = "sm120",
              hidden: bool = False) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k``, tile ``tile`` (rows, columns) and device ``dev``, by role; ``hidden``, the
    hiding units."""
    if hidden:
        return {"row_b": bind(Pc8RowBHiddenDef, K=k, DEV=dev), "row_a": bind(Pc8RowAHiddenDef, K=k, DEV=dev),
                "tile": bind(Pc8TileHiddenDef, K=k, TM=tile[0], TN=tile[1], DEV=dev)}
    return {"row_b": bind(Pc8RowBDef, K=k, DEV=dev), "row_a": bind(Pc8RowADef, K=k, DEV=dev),
            "tile": bind(Pc8TileDef, K=k, TM=tile[0], TN=tile[1], DEV=dev)}


# ---- the Programs --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    """M rows of A and N rows of B at depth k on device ``dev``, cut into tiles of ``tile`` (the scheme's is 64 x 64);
    ``hidden``, the hiding Programs."""

    m: int
    k: int
    n: int
    tile: tuple[int, int] = TILE
    dev: str = "sm120"
    hidden: bool = False

    def check(self) -> None:
        d = DEVICES.get(self.dev)
        tm, tn = self.tile
        if d is None or d.atom.name not in ATOMS or d.peel.name not in PEELS:
            raise DomainError(f"Pearl-C's circuit has no device {self.dev!r}: one of {', '.join(DEVICES)}")
        if (self.k % d.window or not 0 < self.k <= d.k_max or min(self.m, self.n, tm, tn) <= 0 or self.m % tm
                or self.n % tn):
            raise DomainError(f"Pearl-C's circuit needs {d.window} | k <= {d.k_max} and whole tiles on each side: {self}")


def _weight_units(B, k: int, dev: str, b, eb, fa, fb, g):
    rows = B.batch(templates(k, TILE, dev)["row_b"], b, eb, fb, fa, g, axes=(0, 0, None, None, None))
    return [array_of([rows[j][f] for j in range(len(b))]) for f in range(3)]


def _call_units(B, k: int, tile: tuple[int, int], dev: str, a, ea, fa, fb, bc, pb):
    tm, tn = tile
    tp = templates(k, tile, dev)
    rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
    m, n = len(a), len(bc)
    ms, ns = m // tm, n // tn
    ac, pa = (array_of([rows[i][f] for i in range(m)]) for f in range(2))
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    acs, pas, bcs, pbs = blocks(ac, ms, tm), blocks(pa, ms, tm), blocks(bc, ns, tn), blocks(pb, ns, tn)
    tiles = [B.batch(tp["tile"], acs[rs], bcs, pas[rs], pbs, axes=(None, 0, None, 0)) for rs in range(ms)]
    cw, uw = (array_of([array_of([tiles[i // tm][j // tn][f][i % tm][j % tn] for j in range(n)]) for i in range(m)])
              for f in range(2))
    return cw, uw, array_of([rows[i][2] for i in range(m)])


def _weight_sig(S_):
    k, n = S_.K, S_.N
    return ((("b", Array(n, Array(k, F32))), ("eb", Array(n, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)), ("g", GRAM)),
            Tuple(CODES(n, k), Array(n, P_ROW), Array(n, BIT)))


Pc8WeightDef = CompositeDefinition(
    "Pc8Weight", 1, ("K", "N", "DEV"), _weight_sig, lambda B, S_, *a: tuple_of(*_weight_units(B, S_.K, S_.DEV, *a)),
    doc="A weight's side, once per job and weight: per row B̃'s codes, P_B and its admission (G is anchored).",
    conformance="protocol-defined")


def _call_sig(S_):
    m, k, n = S_.M, S_.K, S_.N
    return ((("a", Array(m, Array(k, F32))), ("ea", Array(m, LINE)), ("fa", BASIS(k)), ("fb", BASIS(k)),
             ("bc", CODES(n, k)), ("pb", Array(n, P_ROW))),
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, BIT)))


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
            Tuple(Array(m, Array(n, F32)), Array(m, Array(n, F32)), Array(m, BIT), Array(n, BIT)))


def _check_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, pb, ok_b = _weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    cw, uw, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb)
    return tuple_of(cw, uw, ok_a, ok_b)


Pc8CheckDef = CompositeDefinition(
    "Pc8Check", 1, ("M", "K", "N", "TM", "TN", "DEV"), _check_sig, _check_body,
    doc="The check of an M x N block in one root: the weight's units, then the call's, reading them.",
    conformance="protocol-defined")


# ---- the hiding Programs (`pc4`'s) ---------------------------------------------------------------------------------------

def _weight_hidden_units(B, k: int, dev: str, b, eb, fa, fb, g, yb, ybm):
    rows = B.batch(templates(k, TILE, dev, True)["row_b"], b, eb, fb, fa, g, yb, ybm,
                   axes=(0, 0, None, None, None, 0, 0))
    return [array_of([rows[j][f] for j in range(len(b))]) for f in range(5)]


def _call_hidden_units(B, k: int, tile: tuple[int, int], dev: str, a, ea, fa, fb, bc, pb, ya, yam, ybm, yt):
    tm, tn = tile
    tp = templates(k, tile, dev, True)
    rows = B.batch(tp["row_a"], a, ea, fa, fb, ya, yam, axes=(0, 0, None, None, 0, 0))
    m, n = len(a), len(bc)
    ms, ns = m // tm, n // tn
    ac, pa, ok, bca, bcam = (array_of([rows[i][f] for i in range(m)]) for f in range(5))
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    acs, pas, yams = blocks(ac, ms, tm), blocks(pa, ms, tm), blocks(yam, ms, tm)
    bcs, pbs, ybms = blocks(bc, ns, tn), blocks(pb, ns, tn), blocks(ybm, ns, tn)
    tiles = [B.batch(tp["tile"], acs[rs], bcs, pas[rs], pbs, yams[rs], ybms, yt[rs], axes=(None, 0, None, 0, None, 0, 0))
             for rs in range(ms)]
    reads_a, reads_b, tile_bc = (array_of([array_of([tiles[rs][cs][f] for cs in range(ns)]) for rs in range(ms)])
                                 for f in range(3))
    return ok, bca, bcam, reads_a, reads_b, tile_bc


def _weight_hidden_sig(S_):
    params, _ = _weight_sig(S_)
    return params + (("yb", pc4.SALTS(S_.N)), ("ybm", pc4.SALTS(S_.N))), Tuple(Array(S_.N, BIT), pc4.BCS(S_.N),
                                                                               pc4.BCS(S_.N))


def _call_hidden_sig(S_):
    params, _ = _call_sig(S_)
    return params + pc4._call_hidden_salts(S_), Tuple(*pc4._call_hidden_out(S_))


def _check_hidden_sig(S_):
    params, _ = _check_sig(S_)
    ya, yam, ybm, yt = pc4._call_hidden_salts(S_)
    return (params + (ya, yam, ("yb", pc4.SALTS(S_.N)), ybm, yt),
            Tuple(*pc4._call_hidden_out(S_), Array(S_.N, BIT), pc4.BCS(S_.N), pc4.BCS(S_.N)))


def _check_hidden_body(B, S_, a, ea, b, eb, fa, fb, g, ya, yam, yb, ybm, yt):
    bc, pb, ok_b, bcb, bcbm = _weight_hidden_units(B, S_.K, S_.DEV, b, eb, fa, fb, g, yb, ybm)
    out = _call_hidden_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, ya, yam, ybm, yt)
    return tuple_of(*out, ok_b, bcb, bcbm)


Pc8WeightHiddenDef = CompositeDefinition(
    "Pc8WeightHidden", 1, ("K", "N", "DEV"), _weight_hidden_sig,
    lambda B, S_, *a: tuple_of(*_weight_hidden_units(B, S_.K, S_.DEV, *a)[2:]),
    doc="`Pc8Weight` hiding: per row of B its admission and the commit strings of B_j and of B̃_j ‖ P_B,j.",
    conformance="protocol-defined")

Pc8CallHiddenDef = CompositeDefinition(
    "Pc8CallHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"), _call_hidden_sig,
    lambda B, S_, *a: tuple_of(*_call_hidden_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a)),
    doc="`Pc8Call` hiding: per row of A its admission and the commit strings of A_i and of A′_i ‖ P_A,i; per tile the "
        "commit strings of the rows it reads (each equal to its row unit's) and of its tile digest.",
    conformance="protocol-defined")

Pc8CheckHiddenDef = CompositeDefinition(
    "Pc8CheckHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"), _check_hidden_sig, _check_hidden_body,
    doc="`Pc8Check` hiding: `Pc8CallHidden`'s outputs, then `Pc8WeightHidden`'s.", conformance="protocol-defined")


def weight(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc8WeightHiddenDef if shape.hidden else Pc8WeightDef, K=shape.k, N=shape.n, DEV=shape.dev)


def call(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc8CallHiddenDef if shape.hidden else Pc8CallDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


def check(shape: Shape) -> SpecializedDefinition:
    shape.check()
    return bind(Pc8CheckHiddenDef if shape.hidden else Pc8CheckDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = pc4.ROLES
_ROLE_OF = {id(Pc8RowBDef): "row_b", id(Pc8RowADef): "row_a", id(Pc8TileDef): "tile", id(Pc8RowBHiddenDef): "row_b",
            id(Pc8RowAHiddenDef): "row_a", id(Pc8TileHiddenDef): "tile"}


def query(shape: Shape, side: str = "check") -> dict[str, Any]:
    """`Q_template_instances` over the side's unit templates (`ROLES`), in the order its root body first uses them."""
    tp = templates(shape.k, shape.tile, shape.dev, shape.hidden)
    return PO.template_instances_query([tp[r] for r in ROLES[side]])


def units(fn: SpecializedDefinition) -> tuple[pc4.Unit, ...]:
    """Every unit of a Program built from the unit templates (`pc4.units`)."""
    return pc4.units(fn, _ROLE_OF)


def unit_cut(fn: SpecializedDefinition, recompute: str = RECOMPUTE[1]):
    """The whole Program's cut, checked by `cut.check_cut` (`pc4.unit_cut`, whose ``recompute`` a hiding Program needs)."""
    return pc4.unit_cut(fn, _ROLE_OF, recompute)


__all__ = ["ATOMS", "BASIS", "GRAM", "LINE", "PEELS", "P_ROW", "ROLES", "TILE", "Shape", "call", "check", "query",
           "templates", "unit_cut", "units", "weight"]

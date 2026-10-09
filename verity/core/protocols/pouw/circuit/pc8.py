"""Pearl-C's tile check as a Program (``pearl-c-h100-v1``, ``pearl-c-sm120-v1``, and the unpromoted sm_120 record): forming
v1, the chain and the clean-up (`schemes.pearl_c`) cut into proof units as `pc4` cuts Pearl-C4's, so sampled proofs can
establish a drawn tile's checked words (C̃ and U) and its rows' admission without the verifier reading a row.

    template              one instance per        reads                                   computes and returns
    Pc8RowB{K,DEV}        row j of B              B_j, E_B,j, F_B, F_A, G                  B̃_j's codes, P_B,j, ok
      (in Pc8Weight, once per job and weight)
    Pc8RowA{K,DEV}        row i of A              A_i, E_A,i, F_A, F_B                     A′_i's codes, P_A,i, ok
    Pc8Tile{K,TM,TN,DEV}  tile of TM × TN words   its TM rows A′, P_A and TN rows B̃, P_B   C̃ and U of its words

The roots are `pc4`'s: `Pc8Weight{K,N,DEV}` holds the B rows; `Pc8Call{M,K,N,TM,TN,DEV}` holds the A rows and the tiles,
reading the weight's B̃ and P_B as committed inputs; `Pc8Check` is both in one root.  Down the rows the domain is the
scheme's (whole 16-row ticket subtiles), so a short last tile row, r = m mod TM rows, is its own batch of `Pc8Tile` at
r × TN (`Shape.tile_rows`, `tile_templates`).  The partition is `Q_template_instances` over the templates (`query`), and
every value is computed in exactly one unit.  Forming v0 (#311's
port) has no circuit here.  The hiding Programs are `pc4`'s (``Shape.hidden``): `Pc8TileHidden` returns its tile digest in
place of C̃ and U (under -h3, `Pc8TileHiddenH3` returns the tile's messages, the row its ``sha512/row-seg/v1`` leaf
hashes, a value for the unit that hashes it: `leaves.tile_messages`; no root uses it yet; and an audit window's tile,
`Pc8TileSide`, is `Pc8TileHidden` on the window's side rows, each carrying its index field: `SIDE_PORTS`), and the rows are
C-Flock-committed ports (`PORTS`), with fp32 input rows, fp8-p rows (A′_i ‖ P_A,i, B̃_j ‖ P_B,j; `leaves`) and each
row's ``ok`` row, and the anchored inputs public input regions.  ``Shape.cap`` adds the per-tile cap to the hiding call,
cut into a unit per word, the tile and its tally (`cap`).  ``Shape.row_k`` cuts each row unit along k into a head, a form
per segment of row_k positions and a tail (`rowk`), so that a row unit fits C-Flock's block at every served k.  A shape
takes at most one of the two, and each takes whole tiles only (`Shape.check` refuses a short last tile row there); off,
nothing here changes.

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

from verity.ml.commitments.rowleaf import ROLE_W, ROLE_X
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.defs import CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import array_of, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.circuits.ops.fp32 import F32AddV2, F32DivV2, F32FmaV2, F32MulV2
from verity.ml.circuits.ops.prims import (BlackwellE4m3QmmaDot32, E4m3ToF32, F2fpBf16, F32ToE4m3Sat, HopperBF16WgmmaDot16,
                             HopperE4m3QgmmaDot32)
from verity.ml.circuits.ops.scalar import BitAnd, F32Fabs, F32Fmaxf, F32Neg, I32Add, I32Le, SelectI32

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


def _codes(e: pc4._Emit, x, alpha, ep_line, basis, lo: int, hi: int, dev: str, accs: list | None = None):
    """The codes e4m3(atom(α·x_t, E′, F_t)) of positions t in [lo, hi), ``basis`` holding F's lines from position lo; each
    accumulator α·x_t is appended to ``accs`` when given."""
    atom = ATOMS[DEVICES[dev].atom.name]
    out = []
    for t in range(lo, hi):
        acc = e.call(F32MulV2, alpha, x[t])
        if accs is not None:
            accs.append(acc)
        out.append(e.call(F32ToE4m3Sat, e.call(atom, acc, ep_line, basis[t - lo])))
    return out


def _form(e: pc4._Emit, x, alpha, beta, line, basis, k: int, dev: str):
    """A row's E′ = e4m3(β·E) (with E′'s FP32 decode), its codes e4m3(atom(α·x_t, E′, F_t)) (`pearl_c.form_v1`) and their
    accumulators α·x_t."""
    ep = _ep(e, beta, line)
    ew = [e.call(E4m3ToF32, c) for c in ep]
    accs: list = []
    return array_of(_codes(e, x, alpha, array_of(ep), basis, 0, k, dev, accs)), ew, accs


def _row_sig(S_, second: str, extra: tuple = ()):
    k = S_.K
    first = "fb" if second == "fa" else "fa"
    params = (("x", Array(k, F32)), ("e", LINE), (first, BASIS(k)), (second, BASIS(k))) + extra
    return params, Tuple(Array(k, E4M3), P_ROW, OK)


def _row_a(e: pc4._Emit, k: int, dev: str, x, line, fa, fb):
    """Row i of A's outputs (`Pc8RowA`), then its forming's accumulators α·x_t and β, which the cap's row reads."""
    alpha, beta, ok = _stats(e, x, k)
    codes, ew, accs = _form(e, x, alpha, beta, line, fa, k, dev)
    af = [e.call(F2fpBf16, _chain(e, codes, fb.take(1, p), k, dev)) for p in range(R)]
    return (codes, array_of([e.call(F2fpBf16, w) for w in ew] + af), pc4.ok_word(e, ok)), accs, beta


def _row_a_body(B, S_, x, line, fa, fb):
    outs, _accs, _beta = _row_a(pc4._Emit(B), S_.K, S_.DEV, x, line, fa, fb)
    return tuple_of(*outs)


Pc8RowADef = CompositeDefinition(
    "Pc8RowA", 1, ("K", "DEV"), lambda S_: _row_sig(S_, "fb"), _row_a_body,
    doc="Row i of A: its admission, A′_i's codes formed against F_A, and P_A,i = [bf16(E′_A,i) ‖ RNE_bf16(A′_i·F_B)].",
    conformance="protocol-defined")


def _row_b_body(B, S_, x, line, fb, fa, g):
    e, k, dev = pc4._Emit(B), S_.K, S_.DEV
    alpha, beta, ok = _stats(e, x, k)
    codes, ew, _accs = _form(e, x, alpha, beta, line, fb, k, dev)
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


def _tile_hidden_h3_body(B, S_, ac, bc, pa, pb):
    k, tm, tn = S_.K, S_.TM, S_.TN
    out = B.call(templates(k, (tm, tn), S_.DEV)["tile"], ac, bc, pa, pb)
    return L.tile_messages(out[0], out[1], tm, tn)


Pc8TileHiddenH3Def = CompositeDefinition(
    "Pc8TileHiddenH3", 1, ("K", "TM", "TN", "DEV"), lambda S_: (_tile_sig(S_)[0], L.messages_type(S_.TM, S_.TN)),
    _tile_hidden_h3_body,
    doc="`Pc8TileHidden` under hashing -h3: `Pc8Tile`'s checked words as the tile's messages (`pearl_c.messages`, "
        "`leaves.tile_messages`), whose bits are the row `Sha512RowSeg_v1{N_BITS = 64·TM·TN}` hashes into `pearl_c.leaf`; "
        "a value for that unit, not a public output.", conformance="protocol-defined")


# ---- the audit window's tile ------------------------------------------------------------------------------------------------

def _tile_side_sig(S_):
    return _tile_sig(S_)[0] + (("ia", Array(S_.TM, L.INDEX)), ("jb", Array(S_.TN, L.INDEX))), L.DIGEST


def _tile_side_body(B, S_, ac, bc, pa, pb, ia, jb):
    return B.call(templates(S_.K, (S_.TM, S_.TN), S_.DEV, True)["tile"], ac, bc, pa, pb)


Pc8TileSideDef = CompositeDefinition(
    "Pc8TileSide", 1, ("K", "TM", "TN", "DEV"), _tile_side_sig, _tile_side_body,
    doc="`Pc8TileHidden` on an audit window's side rows (`window.side_row`): each A row A′_i ‖ P_A,i ‖ I and B row "
        "B̃_j ‖ P_B,j ‖ J carries its place in its call, I or J, in an index field (`leaves.INDEX`) that the unit reads as a "
        "value of its committed row; its tile digest is `Pc8TileHidden`'s.", conformance="protocol-defined")

#: `Pc8TileSide`'s ports, as `PORTS["tile"]`: its rows are fp8-pi rows, each index field a value in its row, so the rows a
#: unit commits are the side trees' leaves (`window.side_row`)
SIDE_PORTS = {"rows": (("fp8-pi", ROLE_X, ("ac", "pa", "ia")), ("fp8-pi", ROLE_W, ("bc", "pb", "jb")), ("tile", ROLE_X, (0,))),
              "regions": {}, "advice": ()}


@lru_cache(maxsize=8)
def side_tile(k: int, tile: tuple[int, int] = (1, 1), dev: str = "sm120") -> SpecializedDefinition:
    """The audit window's tile unit at depth ``k``: `Pc8TileSide`."""
    return bind(Pc8TileSideDef, K=k, TM=tile[0], TN=tile[1], DEV=dev)


#: each unit's ports under hiding, as `pc4.PORTS`: fp32 input rows, fp8-p rows and ok rows, the anchors' regions, and its
#: advice: parameters the prover supplies that no unit computes and nothing binds (the cap word's certificates,
#: `cap.ADVICE`), which C-Flock proves as private inputs.  An advice port is in the unit's io and in no read set: in no
#: committed row (so in no link either) and in no public input region, which `leaves.statement_rows` enforces.  Under the
#: cap the ports are `cap.PORTS` (``Shape.ports``).
PORTS = {
    "row_b": {"rows": (("fp32", ROLE_W, ("x",)), ("fp8-p", ROLE_W, (0, 1)), ("ok", ROLE_W, (2,))),
              "regions": {"e": "eb", "fb": "fb", "fa": "fa", "g": "g"}, "advice": ()},
    "row_a": {"rows": (("fp32", ROLE_X, ("x",)), ("fp8-p", ROLE_X, (0, 1)), ("ok", ROLE_X, (2,))),
              "regions": {"e": "ea", "fa": "fa", "fb": "fb"}, "advice": ()},
    "tile": {"rows": (("fp8-p", ROLE_X, ("ac", "pa")), ("fp8-p", ROLE_W, ("bc", "pb")), ("tile", ROLE_X, (0,))),
             "regions": {}, "advice": ()},
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
              hidden: bool = False, hashing: str = "h0") -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k``, tile ``tile`` (rows, columns) and device ``dev``, by role; ``hidden``, the
    hiding tile, under ``hashing`` h0 (`Pc8TileHidden`) or h3 (`Pc8TileHiddenH3`)."""
    if hashing not in C.HASHINGS or (hashing != "h0" and not hidden):
        raise ValueError(f"Pearl-C's tile takes hashing {' or '.join(C.HASHINGS)}, and h3 only hiding: {hashing!r}")
    td = Pc8TileDef if not hidden else Pc8TileHiddenDef if hashing == "h0" else Pc8TileHiddenH3Def
    return {"row_b": bind(Pc8RowBDef, K=k, DEV=dev), "row_a": bind(Pc8RowADef, K=k, DEV=dev),
            "tile": bind(td, K=k, TM=tile[0], TN=tile[1], DEV=dev)}


# ---- the Programs --------------------------------------------------------------------------------------------------------

class _Ports(pc4.RowK):
    """`Shape.ports`: `cap.PORTS` on a shape with the cap, else `pc4.RowK`'s (today's, or with ``row_k`` the cut's)."""

    def __get__(self, shape, cls=None) -> dict:
        if shape is not None and shape.cap:
            from . import cap
            return cap.PORTS
        return super().__get__(shape, cls)


@dataclass(frozen=True)
class Shape:
    """M rows of A and N rows of B at depth k on device ``dev``, cut into tiles of ``tile`` (the scheme's is 64 x 64);
    ``hidden``, the hiding Programs; ``cap``, the hiding Programs with the per-tile cap, cut into word, tile and tally
    units (`cap`); ``row_k``, each row unit cut into k / row_k segments (`rowk`; None, today's Programs).
    `anchors.TileAnchors` reads its units' ports and its anchors' derivations."""

    m: int
    k: int
    n: int
    tile: tuple[int, int] = TILE
    dev: str = "sm120"
    hidden: bool = False
    cap: bool = False
    row_k: int | None = None
    ports = _Ports(PORTS, "PORTS8")
    derivations = pc4.RowK(DERIVATIONS, "DERIVATIONS8")
    seeds: ClassVar[dict] = SEEDS

    def check(self) -> None:
        """`PearlC.check`'s rows and columns: whole tiles across, and down the rows whole 16-row ticket subtiles (whole
        tiles, for a tile under 16 rows), so the last tile row may be short (`tile_rows`).  The cap (``cap``, `cap`) and
        the row cut (``row_k``, `rowk`) take whole tiles only."""
        d = DEVICES.get(self.dev)
        tm, tn = self.tile
        if d is None or d.atom.name not in ATOMS or d.peel.name not in PEELS:
            raise DomainError(f"Pearl-C's circuit has no device {self.dev!r}: one of {', '.join(DEVICES)}")
        if (self.k % d.window or not 0 < self.k <= d.k_max or min(self.m, self.n, tm, tn) <= 0
                or self.m % min(tm, P.TILE) or self.n % tn):
            raise DomainError(f"Pearl-C's circuit needs {d.window} | k <= {d.k_max}, {tn} | n and {min(tm, P.TILE)} | m "
                              f"(whole tiles across, whole ticket subtiles down): {self}")
        if self.cap and self.row_k is not None:
            raise DomainError(f"Pearl-C's circuit has no cap on rows cut along k: {self}")
        if self.cap:
            if self.m % tm:
                raise DomainError(f"Pearl-C's cap takes whole tiles only, {tm} | m: {self}")
            from . import cap
            cap.check_shape(self)
        if self.row_k is not None:
            if self.m % tm:
                raise DomainError(f"Pearl-C's row cut takes whole tiles only, {tm} | m: {self}")
            from . import rowk
            rowk.check8(self)

    def tile_rows(self) -> tuple[int, ...]:
        """Each tile row's height, top to bottom: TM, then the short last one (m mod TM rows) where TM does not divide m."""
        tm = self.tile[0]
        return (tm,) * (self.m // tm) + ((self.m % tm,) if self.m % tm else ())


def _weight_units(B, k: int, dev: str, b, eb, fa, fb, g, row_k: int | None = None):
    if row_k is not None:
        from . import rowk
        return rowk.rows8(B, k, row_k, dev, "b", b, eb, fb, fa, g)
    rows = B.batch(templates(k, TILE, dev)["row_b"], b, eb, fb, fa, g, axes=(0, 0, None, None, None))
    return [array_of([rows[j][f] for j in range(len(b))]) for f in range(3)]


def _call_units(B, k: int, tile: tuple[int, int], dev: str, a, ea, fa, fb, bc, pb, hidden: bool = False,
                row_k: int | None = None):
    """The call's units: (C̃, U, ok), or under hiding (the tile digests by tile row and column, ok); with ``row_k``, each
    A row's unit cut into `rowk`'s pieces.  A short last tile row (r = m mod TM rows) is its own batch of the tile template
    at r × TN."""
    tm, tn = tile
    tp = templates(k, tile, dev, hidden)
    m, n = len(a), len(bc)
    if row_k is None:
        rows = B.batch(tp["row_a"], a, ea, fa, fb, axes=(0, 0, None, None))
        ac, pa, ok = (array_of([rows[i][f] for i in range(m)]) for f in range(3))
    else:
        from . import rowk
        ac, pa, ok = rowk.rows8(B, k, row_k, dev, "a", a, ea, fa, fb)
    ms, ns, r = m // tm, n // tn, m % tm
    blocks = lambda x, h, t: x.reshape(Array(h, Array(t, x.type.elem)))            # noqa: E731
    whole_a, whole_p = (ac[:ms * tm], pa[:ms * tm]) if r else (ac, pa)
    acs, pas, bcs, pbs = blocks(whole_a, ms, tm), blocks(whole_p, ms, tm), blocks(bc, ns, tn), blocks(pb, ns, tn)
    tiles = [B.batch(tp["tile"], acs[rs], bcs, pas[rs], pbs, axes=(None, 0, None, 0)) for rs in range(ms)]
    if r:
        short = templates(k, (r, tn), dev, hidden)["tile"]
        tiles.append(B.batch(short, ac[ms * tm:], bcs, pa[ms * tm:], pbs, axes=(None, 0, None, 0)))
    if hidden:
        return array_of([array_of([tiles[rs][cs] for cs in range(ns)]) for rs in range(len(tiles))]), ok
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

#: the tile digests by tile row (a short last one included) and tile column
DIGESTS = lambda S_: Array(-(-S_.M // S_.TM), Array(S_.N // S_.TN, L.DIGEST))       # noqa: E731


def _check_hidden_body(B, S_, a, ea, b, eb, fa, fb, g):
    bc, pb, ok_b = _weight_units(B, S_.K, S_.DEV, b, eb, fa, fb, g)
    digests, ok_a = _call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, a, ea, fa, fb, bc, pb, hidden=True)
    return tuple_of(digests, ok_a, ok_b)


Pc8CallHiddenDef = CompositeDefinition(
    "Pc8CallHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (_call_sig(S_)[0], Tuple(DIGESTS(S_), Array(S_.M, OK))),
    lambda B, S_, *a: tuple_of(*_call_units(B, S_.K, (S_.TM, S_.TN), S_.DEV, *a, hidden=True)),
    doc="`Pc8Call` hiding: each tile's digest of C̃ and U by tile row and column, and each A row's admission.",
    conformance="protocol-defined")

Pc8CheckHiddenDef = CompositeDefinition(
    "Pc8CheckHidden", 1, ("M", "K", "N", "TM", "TN", "DEV"),
    lambda S_: (_check_sig(S_)[0], Tuple(DIGESTS(S_), Array(S_.M, OK), Array(S_.N, OK))), _check_hidden_body,
    doc="`Pc8Check` hiding: `Pc8CallHidden`'s outputs, then each B row's admission.", conformance="protocol-defined")


def weight(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "weight")
    return bind(Pc8WeightDef, K=shape.k, N=shape.n, DEV=shape.dev)


def call(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.cap:
        from . import cap
        return cap.call(shape)
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "call")
    return bind(Pc8CallHiddenDef if shape.hidden else Pc8CallDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


def check(shape: Shape) -> SpecializedDefinition:
    shape.check()
    if shape.cap:
        from . import cap
        return cap.check(shape)
    if shape.row_k is not None:
        from . import rowk
        return rowk.root8(shape, "check")
    return bind(Pc8CheckHiddenDef if shape.hidden else Pc8CheckDef, M=shape.m, K=shape.k, N=shape.n, TM=shape.tile[0],
                TN=shape.tile[1], DEV=shape.dev)


# ---- the partition -------------------------------------------------------------------------------------------------------

ROLES = pc4.ROLES
_ROLE_OF = {id(Pc8RowBDef): "row_b", id(Pc8RowADef): "row_a", id(Pc8TileDef): "tile", id(Pc8TileHiddenDef): "tile",
            id(Pc8TileHiddenH3Def): "tile"}


def _role_of() -> dict[int, str]:
    from . import cap, rowk
    return _ROLE_OF | cap.ROLE_OF | rowk.ROLE_OF


def tile_templates(shape: Shape) -> list[SpecializedDefinition]:
    """The tile template of each distinct tile row height (`Shape.tile_rows`), top to bottom: TM × TN, then r × TN."""
    return [templates(shape.k, (h, shape.tile[1]), shape.dev, shape.hidden)["tile"] for h in dict.fromkeys(shape.tile_rows())]


def query(shape: Shape, side: str = "check") -> dict[str, Any]:
    """`Q_template_instances` over the side's unit templates (`ROLES`, under the cap `cap.ROLES`, or with ``row_k``
    `rowk.ROLES`), in the order its root body first uses them: a short last tile row's template after the whole tiles'."""
    if shape.row_k is not None:
        from . import rowk
        return PO.template_instances_query(rowk.unit_templates8(shape, side))
    if shape.cap:
        from . import cap
        tp = cap.unit_templates(shape.k, shape.dev, shape.tile, shape.n)
        return PO.template_instances_query([tp[r] for r in cap.ROLES[side]])
    tp = templates(shape.k, shape.tile, shape.dev, shape.hidden)
    return PO.template_instances_query([t for r in ROLES[side] for t in (tile_templates(shape) if r == "tile" else [tp[r]])])


def units(fn: SpecializedDefinition) -> tuple[pc4.Unit, ...]:
    """Every unit of a Program built from the unit templates (`pc4.units`)."""
    return pc4.units(fn, _role_of())


def unit_cut(fn: SpecializedDefinition):
    """The whole Program's cut, checked by `cut.check_cut` (`pc4.unit_cut`)."""
    return pc4.unit_cut(fn, _role_of())


__all__ = ["ATOMS", "BASIS", "DERIVATIONS", "DIGESTS", "GRAM", "LINE", "PEELS", "PORTS", "P_ROW", "ROLES", "SEEDS",
           "SIDE_PORTS", "TILE", "Shape", "basis_fa", "basis_fb", "call", "check", "gram_g", "line_ea", "line_eb", "query",
           "side_tile", "templates", "tile_templates", "unit_cut", "units", "weight"]

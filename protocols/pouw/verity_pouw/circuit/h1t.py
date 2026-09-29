"""H-1T as a Program under layout A: Track H's tagged one-sided FP8 construction, cut into its proof units.

One linear call ``A·B′ᵀ`` (A: m × k rows of the model's E4M3 codes X_q, legal when finite with |x| <= 416; B′: the
weight's n rows of registered E4M3 codes, with its fixed tag rows) is `H1tLinear{M,K,N}`.  Its root body holds only
instances of these templates, so the partition is `Q_template_instances` over them (`circuit.partition`):

    template        one instance per        reads                                              returns
    H1tKey{K}       16-row strip of A       its 16 rows, its strip id, salt, forward,          the strip's key K_s
                                            position, weight id
    H1tFormX{K}     16-row strip of A       its 16 rows, their row ids, K_s                    [x1 | x2] per row (2 × 32T codes)
    H1tTile{K}      16 × 16 block           its x strip, 16 weight rows, their tag tuples      the checked words, and Z
    H1tDequant      output entry            Z, the row scale, the column scale, the bias       y (BF16)

**The map** is Track H's pinned `h1_tagged.h1t_ref` (docs/pouw/hardness-shaped-matrices.md §3.1), one primitive per lane
(`h1t_words`): T = ceil(k/29) slices of 29 real lanes (A's codes, 0 past k) and 3 tag lanes; per slice, the floor F from
its real lanes, x1 = enc(x + s·mu) and x2 = enc(−s·mu) on real lanes, x1 = v and x2 = −v on tag lanes.  B′'s slice is
29 real codes (0 past k) and the column's tag tuple (t_0, t_1, t_2) = (A0[j mod 23], A1[(j div 23) mod 62],
A1[(j div 1426) mod 62]), anchored inputs.  **The chain** per cell: I_τ = HOPPER_E4M3_K32(+0, x1[S_τ], B′[S_τ]) for
τ = 1 … T, then I_(T+τ) over x2; R_1 = I_1 and R_τ = RN_f32(R_(τ−1) + I_τ).  The tile returns the credited words I_1 …
I_2T and R_2 … R_(2T−2) (4T − 3 per cell), and Z = R_2T, the useful output; R_(2T−1) stays inside the tile.

**The salt stream** is tile-local (the FP8 red team's X-SPW-5: the strip digest D_s in root_A's place, and the global
row index, X-SPW-2), in `ncp-v2`'s formats (a 64-bit field is one lane; a digest word is its big-endian value; an index
word ι(i) = i << 32 | i is written into all five Keccak columns so that every gate depends on it):

    r_i  = SHA-512("h1t-row\\0" ‖ row i's k codes)       D_s = SHA-512("h1t-stp\\0" ‖ r_16s ‖ … ‖ r_16s+15)
    K_s  = SHAKE256(ι(strip id) ×5 ‖ "H1T-KEY\\0" ‖ salt ‖ forward ‖ position ‖ weight id ‖ D_s)[:32]
    row i's salt words = SHAKE256(ι(row id) ×5 ‖ "h1t-slt\\0" ‖ K_s)[:72T], little-endian: slice τ's r0 … r17

Track H's reference fixes the map, the words' bit layout and the chain, but not this derivation; it is the circuit's.
The strip id and the row ids are separate anchored inputs (the global strip index and row indices), so that the key's and
the stream's index words are never the same input.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from verity.ir.defs import REGISTRY, CompositeDefinition, SpecializedDefinition, bind
from verity.ir.refs import Coll, array_of, concat, tuple_of
from verity.ir.types import Array, Tuple, Value

import verity.ml.fp32  # noqa: F401  (registers F32Add_v1, F32Mul_v2, Bf16Add_v1)
import verity.ml.prims  # noqa: F401  (registers F32ToBf16Rn_v1, HopperE4m3QgmmaDot32_v1)
from verity.ml.prims import const

from ..protocol import DomainError
from . import h1t_words as HW
from .hashes import Emit, keccak_lanes, sha512, sha_words, shake256

U32, U8, F32, BF16 = HW.U32, HW.U8, Value(32), Value(16)
D = 16
SLICE = 32
REAL = HW.REAL
TAG_ROW, TAG_STRIP, TAG_KEY, TAG_SALT = b"h1t-row\0", b"h1t-stp\0", b"H1T-KEY\0", b"h1t-slt\0"
LANE = Array(2, U32)                                        # a 64-bit field as (lo, hi)
WORDS_PER_SLICE = 18                                        # r0 … r17
MAX_K = 32_768                                              # the absorption bound's headroom (|x| <= 416)
A0 = tuple(range(0x68, 0x7F))                               # 23 positive codes, [64, 448]
A1 = tuple(c | s for s in (0, 0x80) for c in range(0x60, 0x7F))   # 62 codes, ±[32, 448]


def slices(k: int) -> int:
    """T = ceil(k / 29)."""
    return -(-k // REAL)


def tag_tuple(j: int) -> tuple[int, int, int]:
    """Column j's tag codes (`h1_tagged.tag_tuples`): injective for j < 23 · 62 · 62 = 88,412."""
    return A0[j % 23], A1[(j // 23) % 62], A1[(j // (23 * 62)) % 62]


def words_per_cell(k: int) -> int:
    return 4 * slices(k) - 3


def _pack_row(e: Emit, row, k: int) -> list:
    """A row's k codes as SHA-512 message word halves (hi, lo)."""
    out = []
    for w in range(k // 8):
        b = [row[8 * w + j] for j in range(8)]
        out.append((e.call(HW.PouwPackBE4Bytes, *b[:4]), e.call(HW.PouwPackBE4Bytes, *b[4:])))
    return out


def _key_body(B, S, rows, strip, salt, fwd, pos, wid):
    e = Emit(B)
    r = [sha512(e, sha_words(TAG_ROW) + _pack_row(e, rows[j], S.K)) for j in range(D)]
    ds = sha512(e, sha_words(TAG_STRIP) + [(ri[2 * t], ri[2 * t + 1]) for ri in r for t in range(8)])
    lanes = [(strip, strip)] * 5 + keccak_lanes(TAG_KEY) + [(salt[2 * t], salt[2 * t + 1]) for t in range(4)]
    lanes += [(fwd[0], fwd[1]), (pos[0], pos[1]), (wid[0], wid[1])]
    lanes += [(ds[2 * t + 1], ds[2 * t]) for t in range(8)]             # a digest word as a lane: its big-endian value
    return e.array(shake256(e, lanes, 32))


H1tKeyDef = CompositeDefinition(
    "H1tKey", 1, ("K",),
    lambda S: ((("rows", Array(D, Array(S.K, U8))), ("strip", U32), ("salt", Array(8, U32)), ("forward", LANE),
                ("position", LANE), ("weight", LANE)), Array(8, U32)), _key_body,
    doc="A strip's salt key K_s = SHAKE256(ι(strip) ×5 ‖ tag ‖ salt ‖ forward ‖ position ‖ weight id ‖ D_s)[:32], D_s the "
        "strip digest over its rows' SHA-512 digests.", conformance="protocol-defined")


def _form_x_body(B, S, rows, rid, key):
    e = Emit(B)
    k, T = S.K, slices(S.K)
    zero = e.B.call(const(8, 0))
    key_lanes = [(key[2 * t], key[2 * t + 1]) for t in range(4)]
    out = []
    for i in range(D):
        r = shake256(e, [(rid[i], rid[i])] * 5 + keccak_lanes(TAG_SALT) + key_lanes, 4 * WORDS_PER_SLICE * T)
        codes = [rows[i][j] for j in range(k)] + [zero] * (REAL * T - k)
        x1, x2 = [], []
        for t in range(T):
            x = codes[REAL * t:REAL * (t + 1)]
            w = r[WORDS_PER_SLICE * t:WORDS_PER_SLICE * (t + 1)]
            ef = e.B.call(HW.PouwH1tSliceExp, array_of(x))
            for lane in range(REAL):
                x1.append(e.call(HW.PouwH1tX1, x[lane], ef, w[lane // 2], lane % 2))
                x2.append(e.call(HW.PouwH1tX2, x[lane], ef, w[lane // 2], lane % 2))
            for word, sel in ((w[17], 1), (w[16], 0), (w[16], 1)):           # tag lanes 29, 30, 31
                v = e.call(HW.PouwH1tTag, ef, word, sel)
                x1.append(v)
                x2.append(e.call(HW.PouwE4m3Neg, v))
        out.append(array_of(x1 + x2))
    return array_of(out)


H1tFormXDef = CompositeDefinition(
    "H1tFormX", 1, ("K",),
    lambda S: ((("rows", Array(D, Array(S.K, U8))), ("rowids", Array(D, U32)), ("key", Array(8, U32))),
               Array(D, Array(2 * SLICE * slices(S.K), U8))), _form_x_body,
    doc="An activation strip's two blocks per row, [x1 | x2] (T slices of 32 codes each): Track H's pinned H-1T map on "
        "its salt words.", conformance="protocol-defined")


def _tile_body(B, S, xs, bs, tags):
    k, T = S.K, slices(S.K)
    atom, add = REGISTRY["HopperE4m3QgmmaDot32_v1"], REGISTRY["F32Add_v1"]
    zero32, zero8 = B.call(const(32, 0)), B.call(const(8, 0))
    bp = [[array_of([bs[c][REAL * t + lane] if REAL * t + lane < k else zero8 for lane in range(REAL)]
                    + [tags[c][i] for i in range(3)]) for t in range(T)] for c in range(D)]
    words, z = [], []
    for rr in range(D):
        zr = []
        for c in range(D):
            atoms = [B.call(atom, zero32, xs[rr][SLICE * (blk * T + t):SLICE * (blk * T + t + 1)], bp[c][t])
                     for blk in (0, 1) for t in range(T)]
            run = [atoms[0]]
            for a in atoms[1:]:
                run.append(B.call(add, run[-1], a))
            words += atoms + run[1:2 * T - 2]
            zr.append(run[-1])
        z.append(array_of(zr))
    return tuple_of(array_of(words), array_of(z))


H1tTileDef = CompositeDefinition(
    "H1tTile", 1, ("K",),
    lambda S: ((("xs", Array(D, Array(2 * SLICE * slices(S.K), U8))), ("bs", Array(D, Array(S.K, U8))),
                ("tags", Array(D, Array(3, U8)))),
               Tuple(Array(D * D * words_per_cell(S.K), U32), Array(D, Array(D, F32)))), _tile_body,
    doc="A 16 × 16 tile's chain over [x1 | x2]·[B′; B′]: per cell the credited words I_1 … I_2T and R_2 … R_(2T−2), "
        "cell-major, and Z = R_2T, the useful output.", conformance="protocol-defined")


def _dequant_body(B, S, z, s, t, bias):
    st = B.call(REGISTRY["F32Mul_v2"], s, t)
    v = B.call(REGISTRY["F32Mul_v2"], st, z)
    return B.call(REGISTRY["Bf16Add_v1"], B.call(REGISTRY["F32ToBf16Rn_v1"], v), bias)


H1tDequantDef = CompositeDefinition(
    "H1tDequant", 1, (), lambda S: ((("z", F32), ("s", F32), ("t", F32), ("bias", BF16)), BF16), _dequant_body,
    doc="y = bf16(s_i·t_j·Z) + bias_j: a width-rule unit after the Z separator.", conformance="protocol-defined")

H1tDequant = bind(H1tDequantDef)


@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int

    def check(self) -> None:
        if self.m % D or self.n % D or self.k % 8 or min(self.m, self.n) <= 0:
            raise DomainError(f"the H-1T circuit needs 16 | m, 16 | n, 8 | k: {self}")
        if not REAL < self.k <= MAX_K:
            raise DomainError(f"the H-1T circuit needs 29 < k <= {MAX_K} (at least two slices): {self}")
        if self.n > 23 * 62 * 62:
            raise DomainError(f"the tag tuples are injective for n <= 88,412 columns: {self}")


def _linear_sig(S):
    m, k, n = S.M, S.K, S.N
    return ((("a", Array(m, Array(k, U8))), ("s", Array(m, F32)), ("b", Array(n, Array(k, U8))), ("t", Array(n, F32)),
             ("bias", Array(n, BF16)), ("salt", Array(8, U32)), ("forward", LANE), ("position", LANE), ("weight", LANE),
             ("stripids", Array(m // D, U32)), ("rowids", Array(m, U32)), ("tags", Array(n, Array(3, U8)))),
            Tuple(Array(m, Array(n, BF16)), Array((m // D) * (n // D), Array(D * D * words_per_cell(k), U32))))


def _linear_body(B, S, a, s, b, t, bias, salt, fwd, pos, wid, sid, rid, tags):
    m, k, n = S.M, S.K, S.N
    ms, ns = m // D, n // D
    tm = templates(k)
    strips_a = a.reshape(Array(ms, Array(D, Array(k, U8))))
    keys = B.batch(tm["key"], strips_a, sid, salt, fwd, pos, wid, axes=(0, 0, None, None, None, None))
    xs = B.batch(tm["form_x"], strips_a, rid.reshape(Array(ms, Array(D, U32))), keys, axes=(0, 0, 0))
    b_strips = b.reshape(Array(ns, Array(D, Array(k, U8))))
    t_strips = tags.reshape(Array(ns, Array(D, Array(3, U8))))
    tiles = [B.batch(tm["tile"], xs[rs], b_strips, t_strips, axes=(None, 0, 0)) for rs in range(ms)]
    out = {}
    for rs in range(ms):
        for cs in range(ns):
            z = tiles[rs][cs][1]
            zz = Coll(Array(D * D, F32), z.refs)
            s_rep = Coll(Array(D * D, F32), concat([s[D * rs + r].refs for r in range(D) for _ in range(D)]))
            t_rep = Coll(Array(D * D, F32), concat([t[D * cs + c].refs for _ in range(D) for c in range(D)]))
            b_rep = Coll(Array(D * D, BF16), concat([bias[D * cs + c].refs for _ in range(D) for c in range(D)]))
            out[rs, cs] = B.batch(H1tDequant, zz, s_rep, t_rep, b_rep, axes=(0, 0, 0, 0))
    y = Coll(Array(m, Array(n, BF16)),
             concat([out[i // D, j // D][D * (i % D) + j % D].refs for i in range(m) for j in range(n)]))
    words = array_of([tiles[rs][cs][0] for rs in range(ms) for cs in range(ns)])
    return tuple_of(y, words)


H1tLinearDef = CompositeDefinition(
    "H1tLinear", 1, ("M", "K", "N"), _linear_sig, _linear_body,
    doc="One H-1T linear call under sampled proofs (layout A): y = dequant(R_2T) with the credited chain words, as "
        "instances of the unit templates only.", conformance="protocol-defined")


@lru_cache(maxsize=None)
def templates(k: int) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k``, by role."""
    return {"key": bind(H1tKeyDef, K=k), "form_x": bind(H1tFormXDef, K=k), "tile": bind(H1tTileDef, K=k),
            "dequant": H1tDequant}


def linear(shape: Shape) -> SpecializedDefinition:
    """`H1tLinear{M,K,N}` for one call."""
    shape.check()
    return bind(H1tLinearDef, M=shape.m, K=shape.k, N=shape.n)


__all__ = ["A0", "A1", "BF16", "D", "F32", "H1tDequant", "LANE", "MAX_K", "REAL", "SLICE", "Shape", "TAG_KEY", "TAG_ROW",
           "TAG_SALT", "TAG_STRIP", "WORDS_PER_SLICE", "linear", "slices", "tag_tuple", "templates", "words_per_cell"]

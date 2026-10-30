"""`ncp-v2` as a Program: NCP-INT's noised GEMM with the two-level key, cut into its proof blocks.

A weight's Y strips and c₀ depend only on B, the salt, the weight id and the column ids, so they are formed once per run
and weight, by `NcpWeight{K,N,PERM}`, whose root body holds only `NcpFormY` instances.  One linear call ``A·Bᵀ`` (A: m × k
activations, B: the weight's n rows along k, int7 entries) is `NcpLinear{M,K,N,PERM}`, which reads that weight's Y strips
and c₀ as committed inputs.  Each root body holds only instances of the unit templates below (the partition is
`Q_template_instances` over them, `circuit.partition`), and every value is computed in exactly one of them.
`NcpWindow{M,K,N,PERM,C}` is C calls on one weight in one root, the weight's Y units once: the composition a window of
calls needs, recompute-free.

    template            one instance per                 reads (committed or input)                      computes and returns
    NcpFormY{K,PERM}    16-column strip of the weight    16 weight rows, salt, weight id, column ids     the Y strip and c₀
      (in NcpWeight, once per run and weight)
    NcpDigest{K}        16-row strip of A                its 16 A rows                                   D_s
    NcpNode             node of the call's tree          its two children (D_s or a node's parent)       the parent; the root is D_A
    NcpKey              call                             D_A, salt, forward, position, weight id         the call key K
    NcpFormX{K,PERM}    16-row strip of A                16 A rows, K, row ids                           the X strip (X + β)
    NcpTile{K}          16 × 16 block                    its X strip, its Y strip, c₀                    the credited words, and Z
    NcpDequant          output entry                     Z, the row scale, the column scale, the bias    y (BF16)

**The key** (a two-level key):

    r_i  = SHA-512("ncp2-row" ‖ row i's int7 bytes)                   D_s = SHA-512("ncp2-stp" ‖ r_16s ‖ … ‖ r_16s+15)
    D_A  = the RFC 6962 tree over D_1 … D_(m/16), node = SHA-512("ncp2-nod" ‖ left ‖ right), one leaf's root the leaf
    K    = SHAKE256(D_A ‖ "ncp2-key" ‖ salt ‖ forward ‖ position ‖ weight id)[:32]
    E₁ row i    = noise(SHAKE256(i ×5 ‖ "ncp2-E1" ‖ K)[:k])              F₁ column j = noise(SHAKE256(j ×5 ‖ "ncp2-F1" ‖ salt ‖ weight id)[:k])

Every field is a 64-bit word: SHA-512's big-endian, SHAKE256's little-endian (a digest word is its big-endian value); the
position word is ``layer << 32 | site``; an index word ``i`` is the lane ``i << 32 | i``, repeated five times so that each of
Keccak's five columns holds it (`hashes.KeccakF1600`'s θ reads it first).  K's message begins with D_A's eight lanes for the
same reason: every column's first lane is the call's own, so no two calls' key units compute a common value.  noise(b) = (b mod 64) − 31.  X and Y are
`schemes.ncp.assemble`'s, route U's checked values `schemes.ncp.checked_route_u`'s: c₀ = −β·Σ_l Y[l][j] and one
16-deep step per checkpoint; the tile returns checkpoints 1 … 3k/16 − 1 (the credited words) and Z, the last.

**Index binding**: the forward index, the position, the weight id and the global row and column
indices are Program inputs, anchored like the salt: `Q_template_instances` admits no constant node at the root, so they
cannot be structure there.

The permutation P is `schemes.ncp`'s (``PERM`` = "word" or "shift24"); both have every sign +1, so E₂ and F₂'s placement
is wiring and only F₂'s negation is a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, SpecializedDefinition, bind
from verity.ml.circuits.refs import Coll, array_of, concat, tuple_of
from verity.ml.circuits.types import Array, Tuple, Value

import verity.ml.circuits.ops.fp32  # noqa: F401  (registers F32Mul_v2, Bf16Add_v1)
import verity.ml.circuits.ops.prims  # noqa: F401  (registers F32ToBf16Rn_v1)
from verity.ml.circuits.ops.ncp import NcpDequant, NcpDequantDef  # noqa: F401  (the dequantization unit, registered in ops)

from ..protocol import DomainError
from ..schemes.ncp import BETA, admissible_perm, shift_perm, word_perm
from . import words as W
from .hashes import Emit, keccak_lanes, sha512, sha_words, shake256

U32, U8, I7, F32, BF16 = W.U32, W.U8, W.I7, Value(32), Value(16)
D = 16
TAG_ROW, TAG_STRIP, TAG_NODE, TAG_KEY, TAG_E1, TAG_F1 = b"ncp2-row", b"ncp2-stp", b"ncp2-nod", b"ncp2-key", b"ncp2-E1\0", b"ncp2-F1\0"
DIGEST = Array(16, U32)                                     # a SHA-512 digest: 8 words as (hi, lo)
LANE = Array(2, U32)                                        # a 64-bit input word as (lo, hi)
assert BETA == W.BETA


@lru_cache(maxsize=16)
def perm(kind: str, k: int) -> tuple[int, ...]:
    """π of `schemes.ncp`'s production P at ``k`` (every sign +1), after its rejection rule."""
    pi, sigma = word_perm(k) if kind == "word" else shift_perm(k) if kind == "shift24" else (None, None)
    if pi is None:
        raise ValueError("PERM is 'word' or 'shift24'")
    if set(sigma) != {1} or not admissible_perm(k, pi, sigma):
        raise DomainError(f"the {kind} permutation fails the rejection rule at k = {k}")
    return pi


def _index_lanes(i) -> list:
    return [(i, i)] * 5


def _noise(e: Emit, halves, k: int) -> list:
    """k noise entries from XOF output half-lanes (byte j is byte j mod 4 of half j // 4)."""
    return [e.call(W.PouwNoiseByte, halves[j // 4], j % 4) for j in range(k)]


# ---- the unit templates --------------------------------------------------------------------------------------------------

def _digest_body(B, S, rows):
    e = Emit(B)
    r = []
    for row in rows:
        words = sha_words(TAG_ROW)
        for w in range(S.K // 8):
            x = [row[8 * w + b] for b in range(8)]
            words.append((e.call(W.PouwPackBE4, *x[:4]), e.call(W.PouwPackBE4, *x[4:])))
        r.append(sha512(e, words))
    words = sha_words(TAG_STRIP) + [(ri[2 * t], ri[2 * t + 1]) for ri in r for t in range(8)]
    return e.array(sha512(e, words))


NcpDigestDef = CompositeDefinition(
    "NcpDigest", 1, ("K",), lambda S: ((("rows", Array(D, Array(S.K, I7))),), DIGEST), _digest_body,
    doc="A strip's digest D_s: SHA-512 over its 16 row digests (the per-row form).", conformance="protocol-defined")


def _node_body(B, S, left, right):
    e = Emit(B)
    words = sha_words(TAG_NODE) + [(left[2 * t], left[2 * t + 1]) for t in range(8)] + [(right[2 * t], right[2 * t + 1]) for t in range(8)]
    return e.array(sha512(e, words))


NcpNodeDef = CompositeDefinition(
    "NcpNode", 1, (), lambda S: ((("left", DIGEST), ("right", DIGEST)), DIGEST), _node_body,
    doc="One node of a call's tree: SHA-512 over its two children, domain-separated from the leaves.",
    conformance="protocol-defined")


def _key_body(B, S, salt, fwd, pos, wid, da):
    e = Emit(B)
    lanes = [(da[2 * t + 1], da[2 * t]) for t in range(8)]            # a digest word as a lane: its big-endian value
    lanes += keccak_lanes(TAG_KEY) + [(salt[2 * t], salt[2 * t + 1]) for t in range(4)]
    lanes += [(fwd[0], fwd[1]), (pos[0], pos[1]), (wid[0], wid[1])]
    return e.array(shake256(e, lanes, 32))


NcpKeyDef = CompositeDefinition(
    "NcpKey", 1, (), lambda S: ((("salt", Array(8, U32)), ("forward", LANE), ("position", LANE), ("weight", LANE),
                                 ("root", DIGEST)), Array(8, U32)), _key_body,
    doc="The call key K = SHAKE256(D_A, salt, forward, position, weight id)[:32]: E₁'s key, once per call; D_A's lanes "
        "first, so every Keccak column begins with the call's own lane.", conformance="protocol-defined")


def _form_y_body(B, S, wrows, salt, wid, cid):
    e = Emit(B)
    k, pi = S.K, perm(S.PERM, S.K)
    inv = [0] * k
    for lo, hi in enumerate(pi):
        inv[hi] = lo
    salt_lanes = [(salt[2 * t], salt[2 * t + 1]) for t in range(4)]
    cols, c0 = [], []
    for c in range(D):
        f = _noise(e, shake256(e, _index_lanes(cid[c]) + keccak_lanes(TAG_F1) + salt_lanes + [(wid[0], wid[1])], k), k)
        b = [wrows[c][l] for l in range(k)]
        y = ([e.call(W.PouwAdd8, b[l], f[l]) for l in range(k)] + [e.call(W.PouwSub8, b[p], f[inv[p]]) for p in range(k)]
             + [e.call(W.PouwAddSub8, b[p], f[p], f[inv[p]]) for p in range(k)])
        acc = 0
        for t in range(0, 3 * k, D):
            acc = e.B.call(W.PouwNegBetaSum16, e.g(acc), array_of(y[t:t + D]))
        cols.append(array_of(y))
        c0.append(acc)
    return tuple_of(array_of(cols), array_of(c0))


NcpFormYDef = CompositeDefinition(
    "NcpFormY", 1, ("K", "PERM"),
    lambda S: ((("wrows", Array(D, Array(S.K, I7))), ("salt", Array(8, U32)), ("weight", LANE), ("cols", Array(D, U32))),
               Tuple(Array(D, Array(3 * S.K, U8)), Array(D, U32))), _form_y_body,
    doc="A weight strip's F₁ columns, its Y strip Y = [B + F₁ ; B + F₂ ; B + F₃] (s8) and c₀ = −β·Σ_l Y[l][j].",
    conformance="protocol-defined")


def _form_x_body(B, S, rows, key, rid):
    e = Emit(B)
    k, pi = S.K, perm(S.PERM, S.K)
    key_lanes = [(key[2 * t], key[2 * t + 1]) for t in range(4)]
    out = []
    for r in range(D):
        n = _noise(e, shake256(e, _index_lanes(rid[r]) + keccak_lanes(TAG_E1) + key_lanes, k), k)
        a = [rows[r][j] for j in range(k)]
        x = ([e.call(W.PouwFormX, a[j], n[j]) for j in range(k)] + [e.call(W.PouwFormX, a[j], n[pi[j]]) for j in range(k)]
             + [e.call(W.PouwFormX3, a[j], n[j], n[pi[j]]) for j in range(k)])
        out.append(array_of(x))
    return array_of(out)


NcpFormXDef = CompositeDefinition(
    "NcpFormX", 1, ("K", "PERM"),
    lambda S: ((("rows", Array(D, Array(S.K, I7))), ("key", Array(8, U32)), ("rowids", Array(D, U32))),
               Array(D, Array(3 * S.K, U8))), _form_x_body,
    doc="An activation strip's E₁ rows and its X strip X + β = [A + E₁ | A + E₂ | −(A + E₃)] + β (u8).",
    conformance="protocol-defined")


def _tile_body(B, S, xs, ys, c0):
    steps = 3 * S.K // D
    acc = [[c0[c] for c in range(D)] for _ in range(D)]
    words = []
    for t in range(steps):
        acc = [[B.call(W.PouwDot16U8S8, acc[r][c], xs[r][D * t:D * (t + 1)], ys[c][D * t:D * (t + 1)]) for c in range(D)]
               for r in range(D)]
        if t < steps - 1:
            words += [acc[r][c] for r in range(D) for c in range(D)]
    return tuple_of(array_of(words), array_of([array_of(row) for row in acc]))


NcpTileDef = CompositeDefinition(
    "NcpTile", 1, ("K",),
    lambda S: ((("xs", Array(D, Array(3 * S.K, U8))), ("ys", Array(D, Array(3 * S.K, U8))), ("c0", Array(D, U32))),
               Tuple(Array((3 * S.K // D - 1) * D * D, U32), Array(D, Array(D, U32)))), _tile_body,
    doc="A 16 × 16 tile's route-U accumulator: the credited words (checkpoints 1 … 3k/16 − 1, step-major) and Z, the last "
        "(A·Bᵀ exactly, mod 2^32).", conformance="protocol-defined")


NcpNode = bind(NcpNodeDef)
NcpKey = bind(NcpKeyDef)


# ---- the linear call -----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Shape:
    m: int
    k: int
    n: int

    def check(self) -> None:
        if self.m % D or self.n % D or self.k % 64 or min(self.m, self.k, self.n) <= 0:
            raise DomainError(f"ncp-v2's circuit needs 16 | m, 16 | n, 64 | k: {self}")

    @property
    def steps(self) -> int:
        return 3 * self.k // D


def rfc6962(lo: int, hi: int, leaf, node):
    """The RFC 6962 Merkle tree hash over leaves lo … hi − 1: one leaf is its own hash; otherwise split at the largest power
    of two below the count."""
    if hi - lo == 1:
        return leaf(lo)
    split = 1
    while split * 2 < hi - lo:
        split *= 2
    return node(rfc6962(lo, lo + split, leaf, node), rfc6962(lo + split, hi, leaf, node))


YS = lambda k, ns: Array(ns, Array(D, Array(3 * k, U8)))  # noqa: E731  a weight's Y strips
C0S = lambda ns: Array(ns, Array(D, U32))                   # noqa: E731  and their c₀


def _weight_sig(S):
    k, n = S.K, S.N
    return ((("b", Array(n, Array(k, I7))), ("salt", Array(8, U32)), ("weight", LANE), ("colids", Array(n, U32))),
            Tuple(YS(k, n // D), C0S(n // D)))


def _weight_units(B, S, b, salt, wid, cid):
    k, ns = S.K, S.N // D
    ys = B.batch(templates(k, S.PERM)["form_y"], b.reshape(Array(ns, Array(D, Array(k, I7)))), salt, wid,
                 cid.reshape(Array(ns, Array(D, U32))), axes=(0, None, None, 0))
    return array_of([ys[c][0] for c in range(ns)]), array_of([ys[c][1] for c in range(ns)])


NcpWeightDef = CompositeDefinition(
    "NcpWeight", 1, ("K", "N", "PERM"), _weight_sig, lambda B, S, *a: tuple_of(*_weight_units(B, S, *a)),
    doc="One run's weight side: its Y strips and c₀, formed once per run and weight and read by every call on it.",
    conformance="protocol-defined")


def _linear_sig(S):
    m, k, n = S.M, S.K, S.N
    return ((("a", Array(m, Array(k, I7))), ("s", Array(m, F32)), ("ys", YS(k, n // D)), ("c0s", C0S(n // D)),
             ("t", Array(n, F32)), ("bias", Array(n, BF16)), ("salt", Array(8, U32)), ("forward", LANE),
             ("position", LANE), ("weight", LANE), ("rowids", Array(m, U32))),
            Tuple(Array(m, Array(n, BF16)), Array((m // D) * (n // D), Array((3 * k // D - 1) * D * D, U32))))


def _call_units(B, m: int, k: int, n: int, kind: str, a, s, ys, c0s, t, bias, salt, fwd, pos, wid, rid):
    """One call's units, reading its weight's committed Y strips and c₀: (y, the credited words)."""
    ms, ns = m // D, n // D
    tm = templates(k, kind)
    strips_a = a.reshape(Array(ms, Array(D, Array(k, I7))))
    dig = B.batch(tm["digest"], strips_a, axes=(0,))
    root = rfc6962(0, ms, lambda i: dig[i], lambda left, right: B.call(NcpNode, left, right))
    key = B.call(NcpKey, salt, fwd, pos, wid, root)
    xs = B.batch(tm["form_x"], strips_a, key, rid.reshape(Array(ms, Array(D, U32))), axes=(0, None, 0))
    tiles = [B.batch(tm["tile"], xs[rs], ys, c0s, axes=(None, 0, 0)) for rs in range(ms)]
    out = {}
    for rs in range(ms):
        for cs in range(ns):
            z = tiles[rs][cs][1]
            zz = Coll(Array(D * D, U32), z.refs)
            s_rep = Coll(Array(D * D, F32), concat([s[D * rs + r].refs for r in range(D) for _ in range(D)]))
            t_rep = Coll(Array(D * D, F32), concat([t[D * cs + c].refs for _ in range(D) for c in range(D)]))
            b_rep = Coll(Array(D * D, BF16), concat([bias[D * cs + c].refs for _ in range(D) for c in range(D)]))
            out[rs, cs] = B.batch(NcpDequant, zz, s_rep, t_rep, b_rep, axes=(0, 0, 0, 0))
    y = Coll(Array(m, Array(n, BF16)),
             concat([out[i // D, j // D][D * (i % D) + j % D].refs for i in range(m) for j in range(n)]))
    return y, array_of([tiles[rs][cs][0] for rs in range(ms) for cs in range(ns)])


def _linear_body(B, S, a, s, ys, c0s, t, bias, salt, fwd, pos, wid, rid):
    return tuple_of(*_call_units(B, S.M, S.K, S.N, S.PERM, a, s, ys, c0s, t, bias, salt, fwd, pos, wid, rid))


NcpLinearDef = CompositeDefinition(
    "NcpLinear", 1, ("M", "K", "N", "PERM"), _linear_sig, _linear_body,
    doc="One NCP-INT linear call under sampled proofs (ncp-v2): y = dequant(A·Bᵀ) with the noised route-U words, as "
        "instances of the unit templates only; its weight's Y strips and c₀ are committed inputs (`NcpWeight`).",
    conformance="protocol-defined")


def _window_sig(S):
    m, k, n, c = S.M, S.K, S.N, S.C
    return ((("b", Array(n, Array(k, I7))), ("salt", Array(8, U32)), ("weight", LANE), ("colids", Array(n, U32)),
             ("t", Array(n, F32)), ("bias", Array(n, BF16)), ("a", Array(c, Array(m, Array(k, I7)))),
             ("s", Array(c, Array(m, F32))), ("forward", Array(c, LANE)), ("position", Array(c, LANE)),
             ("rowids", Array(c, Array(m, U32)))),
            Tuple(Array(c, Array(m, Array(n, BF16))),
                  Array(c, Array((m // D) * (n // D), Array((3 * k // D - 1) * D * D, U32)))))


def _window_body(B, S, b, salt, wid, cid, t, bias, a, s, fwd, pos, rid):
    ys, c0s = _weight_units(B, S, b, salt, wid, cid)
    calls = [_call_units(B, S.M, S.K, S.N, S.PERM, a[c], s[c], ys, c0s, t, bias, salt, fwd[c], pos[c], wid, rid[c])
             for c in range(S.C)]
    return tuple_of(array_of([y for y, _ in calls]), array_of([w for _, w in calls]))


NcpWindowDef = CompositeDefinition(
    "NcpWindow", 1, ("M", "K", "N", "PERM", "C"), _window_sig, _window_body,
    doc="C calls on one weight in one root: the weight's Y units once, then each call's units reading them.",
    conformance="protocol-defined")


@lru_cache(maxsize=16)
def templates(k: int, kind: str) -> dict[str, SpecializedDefinition]:
    """The unit templates at depth ``k`` and permutation ``kind``, by role."""
    return {"digest": bind(NcpDigestDef, K=k), "node": NcpNode, "key": NcpKey,
            "form_y": bind(NcpFormYDef, K=k, PERM=kind), "form_x": bind(NcpFormXDef, K=k, PERM=kind),
            "tile": bind(NcpTileDef, K=k), "dequant": NcpDequant}


def linear(shape: Shape, kind: str = "shift24") -> SpecializedDefinition:
    """`NcpLinear{M,K,N,PERM}` for one call."""
    shape.check()
    perm(kind, shape.k)
    return bind(NcpLinearDef, M=shape.m, K=shape.k, N=shape.n, PERM=kind)


def weight(shape: Shape, kind: str = "shift24") -> SpecializedDefinition:
    """`NcpWeight{K,N,PERM}`: the weight side of every call of this shape's (k, n)."""
    shape.check()
    perm(kind, shape.k)
    return bind(NcpWeightDef, K=shape.k, N=shape.n, PERM=kind)


def window(shape: Shape, calls: int, kind: str = "shift24") -> SpecializedDefinition:
    """`NcpWindow{M,K,N,PERM,C}`: ``calls`` calls of this shape on one weight, in one root."""
    shape.check()
    perm(kind, shape.k)
    return bind(NcpWindowDef, M=shape.m, K=shape.k, N=shape.n, PERM=kind, C=calls)


__all__ = ["BF16", "D", "F32", "NcpDequant", "NcpKey", "NcpNode", "Shape", "TAG_E1", "TAG_F1", "TAG_KEY", "TAG_NODE",
           "TAG_ROW", "TAG_STRIP", "linear", "perm", "rfc6962", "templates", "weight", "window"]

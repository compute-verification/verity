"""BLAKE3's compression with committed XOR-output bits (hash-proving survey §3.2), and the leaf scheme ``blake3-xob`` that
proves the ``blake3`` leaf with it: the same committed object (schema ``blake3-keyed/row/v2``, keys, framing, digest and
carry layout -- every committer / verifier method is ``Blake3Leaf``'s), another circuit, so other system pins.  The witness
op it adds, ``xadd``, runs in every witness interpreter (``witness.py``, ``witness_device.py``).

The pinned gadget commits every addition's output bits (32 + carry rows) and pays one product row per XOR bit: 268 rows
per G.  Here a G commits the bits of its four XOR outputs instead, and no addition output:

* ``e = other ^ (t_1 + ... + t_n mod 2^32)`` is 32 fresh boolean rows (the ``xadd`` program op), ``other`` the XOR partner
  (bit rows or constants), the addends ``t`` as 16-bit limbs;
* the sum's bits are ``s_i = (other_i - e_i)^2``, boolean for boolean ``other_i`` and ``e_i``.  Its limbs ``sum 2^i s_i``
  are affine in 8 product rows per limb: ``2^i x^2 + 2^j y^2 = (r^i x + u r^j y)(r^i x - u r^j y)`` with ``r^2 = 2`` and
  ``u^2 = -1`` (both exist: ``p = 1 mod 8``).  A constant ``other_i`` makes ``s_i`` affine and costs no product;
* the addition is checked on limbs, ``L_lo = lo + 2^16 c_lo`` and ``L_hi + c_lo = hi + 2^16 c_hi``, each carry the sum of
  ``n - 1`` boolean ``sels`` rows (``[L >= 2^16]``, ``[L >= 2^17]``).  Every quantity is an integer below ``2^19``, so the
  field identities are integer identities; ``lo`` / ``hi`` lie in ``[0, 2^16)`` by construction, hence ``other ^ e`` is the
  sum mod 2^32 and ``e`` is unique.

In G the XOR partner of each addition is the word the next step needs as bits: ``a1`` pairs with ``d`` (``d1 = (d ^ a1)
>>> 16`` is ``e`` re-indexed), ``c1`` with ``b``, ``a2`` with ``d1``, ``c2`` with ``b1``.  So the ``b`` / ``d`` words are
always bit rows and the ``a`` / ``c`` words always limbs (the diagonal step keeps the roles): 32 + 16 + carries per addition,
204 rows per G against 268.  Finalisation ``h_w = v_w ^ v_{w+8}``: for w < 4 both words exist only as ``other ^ e``, a
4-way XOR of bit rows (one ``[s >= 2]`` row for three of them, one product for the fourth); for w >= 4 a product XOR.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..compile import Expr, LigeroCtx, Linear, P, Quadratic
from .blake3 import (BLOCK_LEN, IV, KEYED_HASH, K_VU, MSG_PERMUTATION, ROUNDS, Blake3Leaf, _W32, _const, _fresh_ctx,
                     _prod, compress_gadget)

__all__ = ["NAME", "R2", "U", "Blake3XobLeaf", "compress_xob", "catalog/census", "run_xadd"]

NAME = "blake3-xob"


def _sqrt(a: int) -> int:
    """A square root of ``a`` mod P (Tonelli-Shanks); ValueError for a non-residue."""
    a %= P
    if a == 0:
        return 0
    if pow(a, (P - 1) // 2, P) != 1:
        raise ValueError(f"{a} is not a square mod p")
    q, s = P - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (P - 1) // 2, P) != P - 1:
        z += 1
    m, c, t, r = s, pow(z, q, P), pow(a, q, P), pow(a, (q + 1) // 2, P)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % P
            i += 1
        b = pow(c, 1 << (m - i - 1), P)
        m, c, t, r = i, b * b % P, t * b * b % P, r * b % P
    return r


R2 = _sqrt(2)
U = _sqrt(P - 1)


class _XW:
    """A u32 held as limbs (affine in product rows) whose bits are ``other_i ^ e_i`` (never rows)."""
    __slots__ = ("other", "e", "limbs")

    def __init__(self, other: _W32, e: list, limbs: tuple):
        self.other, self.e, self.limbs = other, e, limbs


def _bit_terms(w, i: int) -> list:
    return [w.other.bits[i], w.e[i]] if isinstance(w, _XW) else [w.bits[i]]


def _xadd_rows(ctx: LigeroCtx, lo: Expr, hi: Expr, olo: Expr, ohi: Expr, name: str) -> list[Expr]:
    rows = [ctx._row("bit", f"{name}.e{i}", 0, 2, ("xadd", i)) for i in range(32)]
    idxs = [next(iter(r.lin)) for r in rows]
    if idxs != list(range(idxs[0], idxs[0] + 32)):
        raise RuntimeError("xadd rows must be consecutive")
    ctx.sys.program.append(("xadd", idxs[0], lo, hi, olo, ohi))
    for i, r in enumerate(rows):
        ctx.sys.quadratic.append(Quadratic(f"{name}.bool{i}", r, r, r))
    return rows


def run_xadd(lo: np.ndarray, hi: np.ndarray, olo: np.ndarray, ohi: np.ndarray) -> np.ndarray:
    """The ``xadd`` op on honest integer values: (32, l) bits of ``(other) ^ ((lo + 2^16 hi) mod 2^32)``."""
    s = (np.asarray(lo, dtype=np.int64) + (np.asarray(hi, dtype=np.int64) << 16)) & 0xFFFFFFFF
    o = np.asarray(olo, dtype=np.int64) + (np.asarray(ohi, dtype=np.int64) << 16)
    x = s ^ o
    return (x[None, :] >> np.arange(32, dtype=np.int64)[:, None]) & 1


def _carries(ctx: LigeroCtx, key: Expr, n: int, name: str) -> Expr:
    """``floor(key / 2^16)`` for an honest ``key`` in ``[0, n 2^16)``: ``n - 1`` boolean ``[key >= k 2^16]`` rows, summed."""
    rows = [ctx._row("sel", f"{name}{k}", 0, 2, (key, k)) for k in range(1, n)]
    idxs = [next(iter(r.lin)) for r in rows]
    ctx.sys.program.append(("sels", idxs[0], [k << 16 for k in range(1, n)], [n << 16] * (n - 1), key))
    for r in rows:
        ctx.sys.quadratic.append(Quadratic(f"{name}.bool", r, r, r))
    return sum(rows[1:], rows[0])


def _limb_of_squares(ctx: LigeroCtx, other: _W32, e: list, i0: int, scope: str) -> Expr:
    """``sum_{i < 16} 2^i (other_{i0+i} - e_{i0+i})^2`` as an affine expression: constant ``other`` bits directly, the
    rest in pairs through one product each."""
    acc = _const(ctx, 0)
    sq: list[tuple[int, Expr]] = []
    for i in range(16):
        o, x = other.bits[i0 + i], e[i0 + i]
        if o.is_const():
            acc = acc + (1 << i) * (x if o.c % P == 0 else 1 - x)
        else:
            sq.append((i, o - x))
    for k in range(0, len(sq) - 1, 2):
        (i, x), (j, y) = sq[k], sq[k + 1]
        a, b = pow(R2, i, P), U * pow(R2, j, P) % P
        acc = acc + _prod(ctx, a * x + b * y, a * x - b * y, scope + ".sq")
    if len(sq) % 2:
        i, x = sq[-1]
        acc = acc + (1 << i) * _prod(ctx, x, x, scope + ".sq")
    return acc


def _add_xor(ctx: LigeroCtx, terms: Sequence, other: _W32, name: str) -> tuple[_XW, list[Expr]]:
    """``(sum, e)``: the sum mod 2^32 of 2 or 3 words (limbs) as an :class:`_XW`, and ``e = other ^ sum`` as 32 bit rows."""
    n = len(terms)
    lo = sum((t.limbs[0] for t in terms), _const(ctx, 0))
    hi = sum((t.limbs[1] for t in terms), _const(ctx, 0))
    olo, ohi = other.limbs
    e = _xadd_rows(ctx, lo, hi, olo, ohi, name)
    s_lo = _limb_of_squares(ctx, other, e, 0, name)
    s_hi = _limb_of_squares(ctx, other, e, 16, name)
    c_lo = _carries(ctx, lo, n, name + ".cl")
    r = lo - s_lo - (1 << 16) * c_lo
    ctx.sys.linear.append(Linear(f"{name}.lo", dict(r.lin), r.c, None))
    c_hi = _carries(ctx, hi + c_lo, n, name + ".ch")
    r = hi + c_lo - s_hi - (1 << 16) * c_hi
    ctx.sys.linear.append(Linear(f"{name}.hi", dict(r.lin), r.c, None))
    return _XW(other, e, (s_lo, s_hi)), e


def _carry3(ctx: LigeroCtx, s: Expr, name: str) -> Expr:
    """``[s >= 2]`` for a sum of 3 booleans: one row, ``c^2 = c`` and ``(s - 2c)^2 = s - 2c`` (lane b-ligero-sha256's
    ``_carry``); ``s - 2c`` is the XOR."""
    c = ctx._row("sel", name, 0, 2, (s, 2))
    idx = next(iter(c.lin))
    ctx.sys.program.append(("sels", idx, [2], [3], s))
    ctx.sys.quadratic.append(Quadratic(f"{name}.bool", c, c, c))
    x = s - 2 * c
    ctx.sys.quadratic.append(Quadratic(f"{name}.xor", x, x, x))
    return c


def _xor_bits(ctx: LigeroCtx, xs: Sequence[Expr], name: str) -> Expr:
    """The XOR of up to 4 boolean expressions (constants folded)."""
    flip, var = 0, []
    for x in xs:
        if x.is_const():
            flip ^= x.c % P & 1
        else:
            var.append(x)
    if len(var) > 4:
        raise ValueError("at most 4 XOR inputs")
    if not var:
        r = _const(ctx, 0)
    elif len(var) == 1:
        r = var[0]
    elif len(var) == 2:
        r = var[0] + var[1] - 2 * _prod(ctx, var[0], var[1], name)
    else:
        s = var[0] + var[1] + var[2]
        r = s - 2 * _carry3(ctx, s, name + ".c3")
        if len(var) == 4:
            r = r + var[3] - 2 * _prod(ctx, r, var[3], name)
    return 1 - r if flip else r


def _g(ctx: LigeroCtx, v: list, a: int, b: int, c: int, d: int, mx, my, scope: str) -> None:
    v[a], e = _add_xor(ctx, (v[a], v[b], mx), v[d], f"{scope}.a1")
    v[d] = _W32(e).rotr(16)
    v[c], e = _add_xor(ctx, (v[c], v[d]), v[b], f"{scope}.c1")
    v[b] = _W32(e).rotr(12)
    v[a], e = _add_xor(ctx, (v[a], v[b], my), v[d], f"{scope}.a2")
    v[d] = _W32(e).rotr(8)
    v[c], e = _add_xor(ctx, (v[c], v[d]), v[b], f"{scope}.c2")
    v[b] = _W32(e).rotr(7)


def compress_xob(ctx: LigeroCtx, cv: list[_W32], msg: list[_W32], counter: _W32, block_len: _W32, flags: _W32,
                 scope: str) -> list[_W32]:
    """The compression's output chaining value ``h_w = v_w ^ v_{w+8}`` (8 words as bits) on circuit words: ``cv`` 8 words
    (0..3 may be limb-only, 4..7 need bits), ``msg`` 16 words (limbs), ``counter`` / ``block_len`` / ``flags`` as bits."""
    v = list(cv) + [_W32.const(ctx, IV[i]) for i in range(4)] + [counter, _W32.const(ctx, 0), block_len, flags]
    m = list(msg)
    for r in range(ROUNDS):
        s = f"{scope}.r{r}"
        _g(ctx, v, 0, 4, 8, 12, m[0], m[1], s + ".g0")
        _g(ctx, v, 1, 5, 9, 13, m[2], m[3], s + ".g1")
        _g(ctx, v, 2, 6, 10, 14, m[4], m[5], s + ".g2")
        _g(ctx, v, 3, 7, 11, 15, m[6], m[7], s + ".g3")
        _g(ctx, v, 0, 5, 10, 15, m[8], m[9], s + ".g4")
        _g(ctx, v, 1, 6, 11, 12, m[10], m[11], s + ".g5")
        _g(ctx, v, 2, 7, 8, 13, m[12], m[13], s + ".g6")
        _g(ctx, v, 3, 4, 9, 14, m[14], m[15], s + ".g7")
        if r != ROUNDS - 1:
            m = [m[i] for i in MSG_PERMUTATION]
    return [_W32([_xor_bits(ctx, _bit_terms(v[w], i) + _bit_terms(v[w + 8], i), f"{scope}.fin{w}.{i}") for i in range(32)])
            for w in range(8)]


def _rows_of(build) -> tuple[int, dict]:
    ctx = _fresh_ctx()
    cv = [_W32([ctx.hint(f"cv{w}.b{i}") for i in range(32)]) for w in range(8)]
    msg = [_W32(limbs=(ctx.hint(f"m{w}.lo"), ctx.hint(f"m{w}.hi"))) for w in range(16)]
    ctr = _W32([ctx.hint("ctr.b0")] + [_const(ctx, 0)] * 31)
    fl = _W32([ctx.hint("cs"), ctx.hint("ce"), _const(ctx, 0), _const(ctx, 0), _const(ctx, 1)] + [_const(ctx, 0)] * 27)
    m0 = ctx.sys.m
    build(ctx, cv, msg, ctr, _W32.const(ctx, BLOCK_LEN), fl)
    kinds: dict[str, int] = {}
    for r in ctx.sys.rows[m0:]:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    return ctx.sys.m - m0, kinds


def _pinned(ctx, cv, msg, ctr, bl, fl):
    from .blake3 import _xor
    v = compress_gadget(ctx, cv, msg, ctr, bl, fl, "c")
    for w in range(8):
        _xor(ctx, v[w], v[w + 8], "fin")


def census(word_bits: int = 8) -> dict:
    """Rows per 64-byte block (compression + ``h = v ^ v'``), both designs compiled on the same private inputs (cv bits,
    message limbs, a one-bit counter and the chunk flags as hint rows: the column gadget's shapes), and the per-VU hash
    rows of one role pair: blocks per VU x rows per block (the column's CV selection, holds and copies excluded: they
    are the same in both designs)."""
    new, new_kinds = _rows_of(lambda *a: compress_xob(*a, "c"))
    old, old_kinds = _rows_of(_pinned)
    blocks_per_vu = 2 * (K_VU * word_bits // 8 // BLOCK_LEN)
    return {"word_bits": word_bits, "per_block_pinned": old, "per_block_pinned_kinds": old_kinds,
            "per_block_xob": new, "per_block_xob_kinds": new_kinds, "ratio": round(new / old, 4),
            "blocks_per_vu": blocks_per_vu, "hash_rows_per_vu_pinned": blocks_per_vu * old,
            "hash_rows_per_vu_xob": blocks_per_vu * new, "keyed_flag": KEYED_HASH}


class Blake3XobLeaf(Blake3Leaf):
    """``Blake3Leaf`` with :func:`compress_xob` as the column's compression (204 rows per G against 268)."""

    name = NAME

    def _compress(self, ctx: LigeroCtx, cv: list[_W32], msg: list[_W32], counter: _W32, block_len: _W32, flags: _W32,
                  scope: str):
        h = compress_xob(ctx, cv, msg, counter, block_len, flags, scope)
        return lambda w: h[w]

    def census(self) -> dict:
        c = census(8)
        return {"blake3_compression_rows_per_block": c["per_block_xob"], "blake3_compression": "xor-output bits (xadd)",
                "blake3_framing": "chunk CVs in-circuit, parent fold by the verifier", "same_leaf_as": "blake3"}


try:
    from . import registry as _registry
    _registry.register(Blake3XobLeaf)
except ImportError:                                           # pragma: no cover
    pass


if __name__ == "__main__":
    import json
    import sys as _sys
    print(json.dumps(census(int(_sys.argv[1]) if len(_sys.argv) > 1 else 8), indent=1))

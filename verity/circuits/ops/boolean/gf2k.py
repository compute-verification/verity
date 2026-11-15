"""GF(2^128) and GF(2^256) as Flock's verifier has them (its `Flock.Field`), their products as Boolean Definitions,
and the sums of products that a verifier's algebra over them reduces to once its coins are values.

F = GF(2)[x]/(x^128 + x^7 + x^2 + x + 1). An element is an int below 2^128 whose bit i is the coefficient of x^i: the Lean
verifier's `F128` ⟨lo, hi⟩ is ``lo | hi << 64``, its bytes are ``le64(lo) ‖ le64(hi)`` (`to_bytes`) and its record hex is
``{hi:016x}{lo:016x}`` (`hex128`). K = F[u]/(u^2 + u + x^-1); an element is a pair (c0, c1), c0 + c1·u, and

    (a0 + a1 u)(b0 + b1 u) = (p0 + x^-1 p1) + (p2 + p0) u,   p0 = a0 b0, p1 = a1 b1, p2 = (a0 + a1)(b0 + b1)

as the Lean verifier's `F256.mul` computes it. `mul`, `kmul` and `inv` are the references.

    Gf128Mul_v1(a: Array<128, Value<1>>, b: Array<128, Value<1>>) -> Array<128, Value<1>>
    Gf256Mul_v1(a: Array<2, Array<128, Value<1>>>, b: Array<2, Array<128, Value<1>>>) -> Array<2, Array<128, Value<1>>>
    GfScale_v1{L}(a: F, b: Array<L, F>) -> Array<L, F>
    GfResiduals_v1{S}(w: Array<S.nw, F>, v: Array<S.nv, F>) -> Array<len(S.res), F>

Bit i of a port's 128-bit element is the coefficient of x^i, and a K port is (c0, c1).

**Gf128Mul** is `verity.circuits.boolean.gf128`'s product, Karatsuba down to single bits and the reduction, 2,187 ANDs
(F's arithmetic is there, below this unit, so that every unit that may import `verity.circuits` reaches it; `clmul`,
`reduce`, `mul`, `karatsuba`, `reduce_forms` and `mul_forms` are its). **Gf256Mul** is three of them and the product by the
constant x^-1 (linear), 6,561 ANDs.
**GfScale{L}** multiplies one element into each of L (L Gf128Mul circuits).

**GfResiduals{S}** evaluates residuals, each a sum of products of sums of input elements. ``S`` is a dict:

    nw, nv   the port lengths
    ops      the operands: each a list of terms whose sum it is; a term is ["w", i] or ["v", i] (an element of a port), or
             ["wt", i, u], the element whose bit j is bit u of w[i + j] (the transpose of a block of 128 elements)
    res      the residuals: each {"mul": [[a, b], ...], "add": [term, ...]}, the sum of ops[a]·ops[b] over its products and
             of its added terms

Each product is one Gf128Mul (2,187 ANDs); sums are XORs. `residuals` is the reference.

**Constants.** A product with a constant operand is F2-linear: a circuit that multiplies by a constant can call these with
the constant's bits as `Const1` gates, and Flock's lowering folds every AND with a constant operand, leaving the linear map.
"""
from __future__ import annotations

import struct
from typing import Sequence

from verity.circuits import boolean as BL
from verity.circuits.boolean.gf128 import MASK, clmul, karatsuba, mul, mul_forms, reduce, reduce_forms
from verity.circuits.defs import CompositeDefinition
from verity.circuits.refs import Coll
from verity.circuits.types import Array

from verity.circuits.boolean import trace as TR

__all__ = ["F", "Gf128Mul", "Gf256Mul", "GfResiduals", "GfScale", "K", "MASK", "XINV", "check_structure", "clmul", "from_bytes",
           "hex128", "inv", "karatsuba", "kmul", "kmul_forms", "mul", "mul_forms", "of_hex", "phi8", "reduce", "reduce_forms",
           "residuals", "smul", "to_bytes"]

BIT = BL.BIT
#: one F element: 128 bits, bit i the coefficient of x^i
F = Array(128, BIT)
#: one K element: (c0, c1)
K = Array(2, F)
#: x^-1 in F: `F256.xinv` = ⟨0x43, 0x8000000000000000⟩
XINV = 0x80000000000000000000000000000043


def inv(a: int) -> int:
    """``a^(2^128 - 2)``: the inverse, with ``inv(0) = 0`` (``Flock.Field``)."""
    acc = 1
    for _ in range(127):
        acc = mul(mul(acc, acc), a)
    return mul(acc, acc)


def kmul(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    p0, p1, p2 = mul(a[0], b[0]), mul(a[1], b[1]), mul(a[0] ^ a[1], b[0] ^ b[1])
    return (p0 ^ mul(XINV, p1), p2 ^ p0)


def smul(a: tuple[int, int], s: int) -> tuple[int, int]:
    """K · F, coordinate-wise (`F256.smul`)."""
    return (mul(a[0], s), mul(a[1], s))


def of_hex(h: str) -> int:
    """An F element from its record hex ``{hi:016x}{lo:016x}``: the 128-bit big-endian number."""
    if len(h) != 32:
        raise ValueError(f"an F128 is 32 hex digits, not {len(h)}")
    return int(h, 16)


def hex128(a: int) -> str:
    return f"{a:032x}"


def to_bytes(a: int) -> bytes:
    return struct.pack("<QQ", a & ((1 << 64) - 1), a >> 64)


def from_bytes(b: bytes) -> int:
    lo, hi = struct.unpack("<QQ", b)
    return lo | hi << 64


#: `φ8`'s generator g, a root of X^8 + X^4 + X^3 + X + 1 in F (``Flock.Field``)
PHI8_GEN = (0x0dcb364640a222fe << 64) | 0x6b8330483c2e9849


def _phi8_table() -> list[int]:
    pw = [1]
    for _ in range(7):
        pw.append(mul(pw[-1], PHI8_GEN))
    out = []
    for v in range(256):
        s = 0
        for i in range(8):
            if (v >> i) & 1:
                s ^= pw[i]
        out.append(s)
    return out


_PHI8 = _phi8_table()


def phi8(v: int) -> int:
    """``φ8(v) = Σ bit_i(v) · g^i`` for a byte ``v``."""
    return _PHI8[v % 256]


# --- circuits ----------------------------------------------------------------------------------------------------------------

def mul_const_forms(a: list, c: int) -> list:
    """``a · c`` for a constant ``c``: bit l is the XOR of the bits a[i] for which bit l of ``c·x^i`` is set."""
    out = [None] * 128
    col = c
    for i in range(128):
        for l in range(128):
            if (col >> l) & 1:
                out[l] = a[i] if out[l] is None else out[l] ^ a[i]
        col = mul(col, 2)
    zero = a[0] ^ a[0]
    return [zero if o is None else o for o in out]


def kmul_forms(C, a0: list, a1: list, b0: list, b1: list) -> list[list]:
    p0 = mul_forms(C, a0, b0)
    p1 = mul_forms(C, a1, b1)
    p2 = mul_forms(C, [x ^ y for x, y in zip(a0, a1)], [x ^ y for x, y in zip(b0, b1)])
    xp1 = mul_const_forms(p1, XINV)
    return [[x ^ y for x, y in zip(p0, xp1)], [x ^ y for x, y in zip(p2, p0)]]


Gf128Mul = TR.definition(
    "Gf128Mul", 1, lambda C, a, b: mul_forms(C, a, b), (("a", F), ("b", F)), F,
    doc="a · b in GF(2^128) = GF(2)[x]/(x^128 + x^7 + x^2 + x + 1), bit i the coefficient of x^i (C-Flock's F): Karatsuba to "
        "single bits, 2,187 ANDs")

Gf256Mul = TR.definition(
    "Gf256Mul", 1, lambda C, a0, a1, b0, b1: kmul_forms(C, a0, a1, b0, b1), (("a", K), ("b", K)), K,
    doc="a · b in GF(2^256) = F[u]/(u^2 + u + x^-1), an element (c0, c1) = c0 + c1 u over C-Flock's F: (p0 + x^-1 p1, p2 + p0) "
        "for p0 = a0 b0, p1 = a1 b1, p2 = (a0 + a1)(b0 + b1), three Gf128Mul circuits, 6,561 ANDs")


def _scale_sig(S):
    return (("a", F), ("b", Array(S.L, F))), Array(S.L, F)


def _scale_body(B, S, a, b):
    return B.batch(Gf128Mul, a, b, axes=(None, 0))


GfScale = CompositeDefinition(
    "GfScale", 1, ("L",), _scale_sig, _scale_body,
    doc="a · b[l] in GF(2^128) for each of L elements: L Gf128Mul circuits sharing the operand a")


# --- residuals ---------------------------------------------------------------------------------------------------------------

def _term_value(t: Sequence, w: Sequence[int], v: Sequence[int]) -> int:
    if t[0] == "w":
        return w[t[1]]
    if t[0] == "v":
        return v[t[1]]
    if t[0] == "wt":
        i, u = t[1], t[2]
        return sum(((w[i + j] >> u) & 1) << j for j in range(128))
    raise ValueError(f"unknown term {t!r}")


def check_structure(S: dict) -> None:
    """Refuse an S out of range: every ``w``/``v`` term's element inside its port, a ``wt`` term's 128 elements inside w and
    its bit ``u < 128``, every product's operands among ``ops``. Past a port's end a slice comes up short and `zip`
    truncates, so the Python builders and Lean's `residualForms` agree only on an in-range S."""
    def nat(x) -> bool:
        return isinstance(x, int) and not isinstance(x, bool) and x >= 0

    if not (nat(S["nw"]) and nat(S["nv"])):
        raise ValueError(f"GfResiduals' S: port lengths {S['nw']!r} and {S['nv']!r}")
    size = {"w": S["nw"], "v": S["nv"]}

    def term(t) -> None:
        if len(t) == 2 and t[0] in size and nat(t[1]) and t[1] < size[t[0]]:
            return
        if len(t) == 3 and t[0] == "wt" and nat(t[1]) and nat(t[2]) and t[1] + 128 <= S["nw"] and t[2] < 128:
            return
        raise ValueError(f"GfResiduals' S: term {list(t)!r} is out of range for ports of {S['nw']} and {S['nv']} elements")

    for o in S["ops"]:
        for t in o:
            term(t)
    for r in S["res"]:
        for t in r["add"]:
            term(t)
        for pair in r["mul"]:
            if len(pair) != 2 or not all(nat(k) and k < len(S["ops"]) for k in pair):
                raise ValueError(f"GfResiduals' S: product {list(pair)!r} of {len(S['ops'])} operands")


def residuals(S: dict, w: Sequence[int], v: Sequence[int]) -> list[int]:
    """`GfResiduals{S}` on F elements: each residual's products and added terms, summed."""
    check_structure(S)
    if len(w) != S["nw"] or len(v) != S["nv"]:
        raise ValueError(f"ports of {len(w)} and {len(v)} elements, S wants {S['nw']} and {S['nv']}")
    ops: list[int | None] = [None] * len(S["ops"])

    def op(k: int) -> int:
        if ops[k] is None:
            x = 0
            for t in S["ops"][k]:
                x ^= _term_value(t, w, v)
            ops[k] = x
        return ops[k]

    out = []
    for r in S["res"]:
        x = 0
        for t in r["add"]:
            x ^= _term_value(t, w, v)
        for a, b in r["mul"]:
            x ^= mul(op(a), op(b))
        out.append(x)
    return out


def _term_refs(t: Sequence) -> list[tuple]:
    if t[0] in ("w", "v"):
        p = 0 if t[0] == "w" else 1
        return [("p", p, 128 * t[1] + j) for j in range(128)]
    if t[0] == "wt":
        return [("p", 0, 128 * (t[1] + j) + t[2]) for j in range(128)]
    raise ValueError(f"unknown term {t!r}")


def _xor_sums(B, groups: list[list[list[tuple]]]) -> list[list[tuple] | None]:
    """Each group of equal-width ref vectors summed, level by level: one batched XOR over every pair of a level."""
    cur = [list(g) for g in groups]
    while any(len(g) > 1 for g in cur):
        pairs = [(gi, k) for gi, g in enumerate(cur) for k in range(0, len(g) - 1, 2)]
        lhs = [r for gi, k in pairs for r in cur[gi][k]]
        rhs = [r for gi, k in pairs for r in cur[gi][k + 1]]
        out = B.batch(BL.XOR, Coll(Array(2, Array(len(lhs), BIT)), TR._refs(lhs + rhs)), axes=(1,))
        nxt: list[list] = [[] for _ in cur]
        o = 0
        for gi, k in pairs:
            n = len(cur[gi][k])
            nxt[gi].append([out.refs[o + j] for j in range(n)])
            o += n
        for gi, g in enumerate(cur):
            if len(g) % 2:
                nxt[gi].append(g[-1])
        cur = nxt
    return [g[0] if g else None for g in cur]


def _res_sig(S):
    s = S.S
    check_structure(s)
    return (("w", Array(s["nw"], F)), ("v", Array(s["nv"], F))), Array(len(s["res"]), F)


def _res_body(B, S, w, v):
    s = S.S
    used = sorted({k for r in s["res"] for pair in r["mul"] for k in pair})
    opref = dict(zip(used, _xor_sums(B, [[_term_refs(t) for t in s["ops"][k]] for k in used])))
    pairs = [pair for r in s["res"] for pair in r["mul"]]
    groups = []
    if pairs:
        A = Coll(Array(len(pairs), F), TR._refs([x for a, _ in pairs for x in opref[a]]))
        Bo = Coll(Array(len(pairs), F), TR._refs([x for _, b in pairs for x in opref[b]]))
        prod = B.batch(Gf128Mul, A, Bo, axes=(0, 0))
    q = 0
    for r in s["res"]:
        g = [_term_refs(t) for t in r["add"]]
        for _ in r["mul"]:
            g.append([prod.refs[128 * q + j] for j in range(128)])
            q += 1
        groups.append(g)
    sums = _xor_sums(B, groups)
    if any(x is None for x in sums):
        zero = B.call(BL.constant(0)).refs[0]
        sums = [[zero] * 128 if x is None else x for x in sums]
    return Coll(Array(len(sums), F), TR._refs([x for e in sums for x in e]))


GfResiduals = CompositeDefinition(
    "GfResiduals", 1, ("S",), _res_sig, _res_body,
    doc="residuals over GF(2^128), each a sum of products of sums of input elements (S: the operands, and each residual's "
        "products and added terms), one Gf128Mul per product")

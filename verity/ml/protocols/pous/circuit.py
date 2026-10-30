"""P2's modular squaring as a Boolean circuit: x ↦ x² mod p for p = 2^W − C0, the step P2's decode does twice per label
(`schemes.p2`: ``m = sq(σ_K(sq(c))) − t mod p``), so that PoUS's setup can be proved over committed data.

    PousModSquare_v1{W, C0}   (x: Array<W, Value<1>>) -> Array<W, Value<1>>      x² mod (2^W − C0), every x < 2^W
    PousSquare_v1{N}          (x: Array<N, Value<1>>) -> Array<2N, Value<1>>     x²
    PousModFold_v1{W, C0}     (x: Array<2W>) -> Array<W>                         x mod (2^W − C0), every x < 2^2W
    PousToom3Eval_v1{N}       (x: Array<N>) -> Array<3m + 6>                     Toom-3's evaluation, m = ⌈N/3⌉
    PousToom3Interp_v1{N}     (v0: Array<2m>, v1: Array<2m + 3>, vm1: Array<2m + 2>, v2: Array<2m + 6>,
                               vinf: Array<2(N − 2m)>) -> Array<2N>              Toom-3's interpolation
    PousSquareJoin_v1{N, H}   (a0sq: Array<2H>, a1sq: Array<2(N − H)>, ssq: Array<N + 1>) -> Array<2N>   Karatsuba's join
    PousAdd_v1{A, B}          (a: Array<A>, b: Array<B>) -> Array<max(A, B) + 1>  a + b

Integers are unsigned, bits least significant first, and every array is of `Value<1>`.  At P2's point, W = 16,448 and
C0 = 21,065 (`PousModSquare_v1{W=16448,C0=21065}`, 14,040,852 ANDs).

**The square.** `PousSquare_v1{N}` takes one of three forms by width:

- up to `SCHOOL` bits, schoolbook: each cross product x_i x_j (i < j) once, at weight 2^(i+j+1), and the diagonal x_i free
  at 2^2i, summed by `column_sum` (N² − N ANDs at N = 24);
- up to `TOOM` bits, Karatsuba: x = a0 + a1 2^H with H = ⌊N/2⌋ gives x² = a0² + 2^H (s² − a0² − a1²) + 2^2H a1² with
  s = a0 + a1, so three squarings (a batch of two when a0 and a1 are equally wide) and the join.  M = s² − a0² − a1² =
  2 a0 a1 < 2^(N+1) is summed mod 2^(N+1) over the low N + 1 bits of each square (two ANDs a bit), then added at bit H
  into a1² ‖ a0², whose halves don't overlap (one AND a bit over M, then a carry chain above);
- above, Toom-3: x = a0 + a1 2^m + a2 2^2m with m = ⌈N/3⌉ is a polynomial p at 2^m, and five squarings, p(0)² = a0²,
  p(1)², p(−1)² = |a0 − a1 + a2|², p(2)² and p(∞)² = a2² (a batch when a0 and a2 are equally wide), give x²'s five
  coefficients by Bodrato's sequence (`toom3_interp`), every step of which is a non-negative combination, so unsigned
  sums mod a power of two are exact.  Its one division by 3 is `div3`, one AND a bit.

Where the Karatsuba join leaves the top bit of s², and Toom-3 the top bit of p(1)², unread, the caller slices them off.  At
P2's width, Toom-3 above 80 to 200 bits and schoolbook up to 20 to 32 bits are the cheapest of the thresholds traced (all
give the same 13,892,750 ANDs for the square; Karatsuba alone gives 23,375,083).

**The reduction.** `PousModFold_v1{W, C0}` folds 2^W ≡ C0 twice: x = h 2^W + l gives y = l + C0 h < 2^(W + b) (b the width
of C0 + 1), then y = h' 2^W + l' gives z = l' + C0 h' ≤ 2^W − 1 + C0², so z < 2p while C0² + 2 C0 < 2^W.  The result is
u = z + C0 mod 2^W when bit W of u is set (z ≥ p), else z: one carry chain and one multiplexer.  The output is fully
reduced, below p, for every input.

**Sums.** Every sum is `column_sum`: weighted columns of bits reduced from the least significant column up, full adders
while a column has three bits, a half adder when it has two (free when one of them is the constant 1), and no carry out of
the top column.  A carry is ``maj(x, y, z) = z ⊕ (x ⊕ z)(y ⊕ z)``, one AND, with z the smallest of the three forms, so a
carry chain never carries the chain's forms along.

Each piece is traced from a builder over `forms` (`trace`); the levels of the square are bodies of Calls, so no body is
wider than its own level's sums and a 16,448-bit square is 130 Definitions.  None names a word Definition; Python's
integers are the reference (the tests).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Sequence

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.boolean.forms import ONE, ZERO, L, maj
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.refs import Coll, concat
from verity.ml.circuits.types import Array, Type

__all__ = ["P2_C0", "P2_W", "SCHOOL", "TOOM", "PousAdd", "PousModFold", "PousModSquare", "PousSquare", "PousSquareJoin",
           "PousToom3Eval", "PousToom3Interp", "column_sum", "div3", "fold", "join", "school_square",
           "toom3_eval", "toom3_interp"]

BIT = BL.BIT

P2_W, P2_C0 = 16448, 21065
"""P2's registered point (`schemes.p2.POINT`)."""

SCHOOL = 24
"""The widest square written schoolbook."""

TOOM = 100
"""The widest square split Karatsuba's way; Toom-3 above."""


def bits(n: int) -> Type:
    return Array(n, BIT)


def column_sum(C, cols: Sequence[Sequence[L]], width: int) -> list[L]:
    """Σ_i 2^i Σ cols[i] mod 2^width, as ``width`` forms.  Constant bits are summed first, into at most one 1 per column."""
    k = sum(sum(f.c for f in col if f.is_const()) << i for i, col in enumerate(cols[:width])) % (1 << width)
    cols = [[f for f in col if not f.is_const()] for col in cols[:width]] + [[] for _ in range(width - len(cols))]
    return _column_sum(C, cols, width, k)


def _column_sum(C, cols: list[list[L]], width: int, k: int) -> list[L]:
    one = [bool(k >> i & 1) for i in range(width)]
    out = []
    for i in range(width):
        col, top = cols[i], i == width - 1
        nxt = cols[i + 1] if not top else None
        while len(col) + one[i] >= 3:
            if len(col) >= 3:
                x, y, z = col.pop(0), col.pop(0), col.pop(0)
            else:
                (x, y), z = (col.pop(0), col.pop(0)), ONE
                one[i] = False
            col.append(x ^ y ^ z)
            if not top:
                nxt.append(_carry(C, x, y, z))
        if len(col) == 2:
            x, y = col
            col[:] = [x ^ y]
            if not top:
                nxt.append(C.AND(x, y))
        if one[i]:
            if col:
                (x,) = col
                col[:] = [x.inv()]
                if not top:
                    nxt.append(x)
            else:
                col.append(ONE)
        out.append(col[0] if col else ZERO)
    return out


def _carry(C, x: L, y: L, z: L) -> L:
    x, y, z = sorted((x, y, z), key=lambda f: -len(f.s))
    return maj(C, x, y, z)


def _columns(width: int, *terms: tuple[int, Sequence[L]]) -> list[list[L]]:
    """The columns of Σ 2^shift · term, below ``width``."""
    cols: list[list[L]] = [[] for _ in range(width)]
    for shift, term in terms:
        for j, f in enumerate(term):
            if 0 <= shift + j < width:
                cols[shift + j].append(f)
    return cols


def school_square(C, x: Sequence[L]) -> list[L]:
    """x² (2 len(x) bits): the cross products once each, doubled by their weight, and the diagonal bits."""
    n = len(x)
    cols: list[list[L]] = [[] for _ in range(2 * n)]
    for i in range(n):
        cols[2 * i].append(x[i])
        for j in range(i + 1, n):
            cols[i + j + 1].append(C.AND(x[i], x[j]))
    return column_sum(C, cols, 2 * n)


def join(C, N: int, H: int, a0sq: Sequence[L], a1sq: Sequence[L], ssq: Sequence[L]) -> list[L]:
    """a1² 2^2H + (s² − a0² − a1²) 2^H + a0², 2N bits, from the squares of x's halves and of their sum (its low N + 1 bits)."""
    w = N + 1
    neg = lambda v: [(v[j] if j < len(v) else ZERO).inv() for j in range(w)]
    cols = _columns(w, (0, ssq[:w]), (0, neg(a0sq)), (0, neg(a1sq)))
    cols[1].append(ONE)
    m = column_sum(C, cols, w)
    return column_sum(C, _columns(2 * N, (0, a0sq), (2 * H, a1sq), (H, m)), 2 * N)


def fold(C, W: int, C0: int, x: Sequence[L]) -> list[L]:
    """x mod (2^W − C0) for x < 2^2W (2W bits LSB first): two folds of 2^W ≡ C0, then one conditional subtraction of p."""
    taps = [s for s in range(C0.bit_length()) if C0 >> s & 1]
    b = (C0 + 1).bit_length()
    y = column_sum(C, _columns(W + b, (0, x[:W]), *((s, x[W:]) for s in taps)), W + b)
    z = column_sum(C, _columns(W + 1, (0, y[:W]), *((s, y[W:]) for s in taps)), W + 1)
    cols = _columns(W + 1, (0, z))
    for s in taps:
        cols[s].append(ONE)
    u = column_sum(C, cols, W + 1)
    ge = u[W]
    return [b0 ^ C.AND(ge, a ^ b0) for a, b0 in zip(u[:W], z[:W])]


def add(C, a: Sequence[L], b: Sequence[L]) -> list[L]:
    w = max(len(a), len(b)) + 1
    return column_sum(C, _columns(w, (0, a), (0, b)), w)


def _neg(v: Sequence[L], w: int, shift: int = 0) -> list[L]:
    """The complement of 2^shift · v over w bits (its two's-complement negation less 1)."""
    return [(v[j - shift] if 0 <= j - shift < len(v) else ZERO).inv() for j in range(w)]


def _sub(C, x: Sequence[L], w: int, *ys: tuple[int, Sequence[L]]) -> list[L]:
    """x − Σ 2^shift · y mod 2^w."""
    cols = _columns(w, (0, x), *((0, _neg(y, w, s)) for s, y in ys))
    cols[0] += [ONE] * len(ys)
    return column_sum(C, cols, w)


def div3(C, v: Sequence[L], w: int) -> list[L]:
    """v / 3 mod 2^w for v a multiple of 3, one AND a bit.  3q = q + 2q, so bit i of q is v_i ⊕ q_{i−1} ⊕ c_i with c the
    carries of that sum; the state carried is a_i = q_{i−1} ⊕ c_i, and a_{i+1} = (v_i ⊕ c_i)(v_i ⊕ q_{i−1}) (x ⊕ maj(x, y, z)
    = (x ⊕ y)(x ⊕ z) at x = q_i), a fresh AND, so no form grows along the chain."""
    q, a, qm1 = [], ZERO, ZERO
    for i in range(w):
        q.append(v[i] ^ a)
        if i < w - 1:
            a, qm1 = C.AND(v[i] ^ a ^ qm1, v[i] ^ qm1), q[-1]
    return q


def _toom_m(N: int) -> int:
    return -(-N // 3)


def toom3_eval(C, N: int, x: Sequence[L]) -> list[L]:
    """p(1) = a0 + a1 + a2 (m + 2 bits), |p(−1)| = |a0 − a1 + a2| (m + 1) and p(2) = a0 + 2 a1 + 4 a2 (m + 3), concatenated,
    for x = a0 + a1 2^m + a2 2^2m."""
    m = _toom_m(N)
    a0, a1, a2 = x[:m], x[m:2 * m], x[2 * m:]
    e = column_sum(C, _columns(m + 1, (0, a0), (0, a2)), m + 1)
    p1 = column_sum(C, _columns(m + 2, (0, e), (0, a1)), m + 2)
    u = _sub(C, e, m + 2, (0, a1))
    neg = u[m + 1]
    pm1 = column_sum(C, _columns(m + 1, (0, [b ^ neg for b in u[:m + 1]]), (0, [neg])), m + 1)
    p2 = column_sum(C, _columns(m + 3, (0, a0), (1, a1), (2, a2)), m + 3)
    return p1 + pm1 + p2


def toom3_interp(C, N: int, v0, v1, vm1, v2, vinf) -> list[L]:
    """x² from the squares at 0, 1, −1, 2 and ∞ (v1 to its low 2m + 3 bits).  With x² = Σ c_k 2^km:

        t1 = (v2 − vm1) / 3 = c1 + c2 + 3 c3 + 5 c4      t2 = (v1 − vm1) / 2 = c1 + c3      t3 = v1 − v0 = c1 + c2 + c3 + c4
        t4 = (t1 − t3) / 2 = c3 + 2 c4                   c2 = t3 − t2 − c4    c3 = t4 − 2 c4    c1 = t2 − c3

    Every step is a non-negative combination of the c_k, so each is exact mod 2^w at the width its bound needs."""
    m = _toom_m(N)
    t1 = div3(C, _sub(C, v2, 2 * m + 6, (0, vm1)), 2 * m + 4)
    t2 = _sub(C, v1, 2 * m + 3, (0, vm1))[1:]
    t3 = _sub(C, v1, 2 * m + 3, (0, v0))
    t4 = _sub(C, t1, 2 * m + 3, (0, t3))[1:]
    c2 = _sub(C, t3, 2 * m + 2, (0, t2), (0, vinf))
    c3 = _sub(C, t4, 2 * m + 1, (1, vinf))
    c1 = _sub(C, t2, 2 * m + 1, (0, c3))
    return column_sum(C, _columns(2 * N, (0, v0), (m, c1), (2 * m, c2), (3 * m, c3), (4 * m, vinf)), 2 * N)


# --- the Definitions -----------------------------------------------------------------------------------------------------------

def _check(name: str, **kw: int) -> None:
    for k, v in kw.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name}'s {k} is an int >= 1, not {v!r}")


@lru_cache(maxsize=None)
def _traced(key: tuple, build: Callable, params: tuple[Type, ...], ret: Type) -> TR.Traced:
    return TR.trace(build, list(params), ret)


def _traced_body(key: tuple, build: Callable, signature: Callable) -> Callable:
    def body(B, S, *colls):
        params, ret = signature(S)
        types = tuple(t for _, t in params)
        return TR.emit(_traced(key(S), build(S), types, ret), types, ret, B, *colls)
    return body


def _add_sig(S):
    _check("PousAdd", A=S.A, B=S.B)
    return (("a", bits(S.A)), ("b", bits(S.B))), bits(max(S.A, S.B) + 1)


PousAdd = CompositeDefinition(
    "PousAdd", 1, ("A", "B"), _add_sig,
    _traced_body(lambda S: ("add", S.A, S.B), lambda S: lambda C, a, b: add(C, a, b), _add_sig),
    conformance="exact-model-tested", doc="a + b of an A-bit and a B-bit unsigned integer, max(A, B) + 1 bits")


def _join_sig(S):
    _check("PousSquareJoin", N=S.N, H=S.H)
    if not 2 * S.H <= S.N:
        raise ValueError(f"PousSquareJoin's H is at most N/2, not {S.H} for N = {S.N}")
    return (("a0sq", bits(2 * S.H)), ("a1sq", bits(2 * (S.N - S.H))), ("ssq", bits(S.N + 1))), bits(2 * S.N)


PousSquareJoin = CompositeDefinition(
    "PousSquareJoin", 1, ("N", "H"), _join_sig,
    _traced_body(lambda S: ("join", S.N, S.H), lambda S: lambda C, a0, a1, s: join(C, S.N, S.H, a0, a1, s), _join_sig),
    conformance="exact-model-tested",
    doc="Karatsuba's join for squaring: a1sq 2^2H + (ssq − a0sq − a1sq mod 2^(N+1)) 2^H + a0sq mod 2^2N")


def _square_sig(S):
    _check("PousSquare", N=S.N)
    return (("x", bits(S.N)),), bits(2 * S.N)


def _squares(B, *parts):
    """The squares of the operands ``parts``: one batch when the two are equally wide."""
    a, b = parts
    if a.type == b.type:
        sq = B.batch(bind(PousSquare, N=a.type.n), Coll(Array(2, a.type), concat([a.refs, b.refs])), axes=(0,))
        return sq[0], sq[1]
    return B.call(bind(PousSquare, N=a.type.n), a), B.call(bind(PousSquare, N=b.type.n), b)


def _square_body(B, S, x):
    N = S.N
    if N <= SCHOOL:
        types, ret = (bits(N),), bits(2 * N)
        return TR.emit(_traced(("school", N), school_square, types, ret), types, ret, B, x)
    if N > TOOM:
        m = _toom_m(N)
        ev = B.call(bind(PousToom3Eval, N=N), x)
        v0, vinf = _squares(B, x[0:m], x[2 * m:N])
        v1 = B.call(bind(PousSquare, N=m + 2), ev[0:m + 2])
        vm1 = B.call(bind(PousSquare, N=m + 1), ev[m + 2:2 * m + 3])
        v2 = B.call(bind(PousSquare, N=m + 3), ev[2 * m + 3:3 * m + 6])
        return B.call(bind(PousToom3Interp, N=N), v0, v1[0:2 * m + 3], vm1, v2, vinf)
    h = N // 2
    a0, a1 = x[0:h], x[h:N]
    a0sq, a1sq = _squares(B, a0, a1)
    s = B.call(bind(PousAdd, A=h, B=N - h), a0, a1)
    ssq = B.call(bind(PousSquare, N=N - h + 1), s)
    return B.call(bind(PousSquareJoin, N=N, H=h), a0sq, a1sq, ssq[0:N + 1])


PousSquare = CompositeDefinition(
    "PousSquare", 1, ("N",), _square_sig, _square_body, conformance="exact-model-tested",
    doc="x² of an N-bit unsigned integer, 2N bits: schoolbook up to SCHOOL bits, Karatsuba (three squarings of about N/2 bits) "
        "up to TOOM, Toom-3 (five of about N/3) above")


def _toom_sig(name: str, S) -> int:
    _check(name, N=S.N)
    if S.N < 5:
        raise ValueError(f"{name}'s N is at least 5 (three nonempty parts), not {S.N}")
    return _toom_m(S.N)


def _eval_sig(S):
    m = _toom_sig("PousToom3Eval", S)
    return (("x", bits(S.N)),), bits(3 * m + 6)


PousToom3Eval = CompositeDefinition(
    "PousToom3Eval", 1, ("N",), _eval_sig,
    _traced_body(lambda S: ("toom3-eval", S.N), lambda S: lambda C, x: toom3_eval(C, S.N, x), _eval_sig),
    conformance="exact-model-tested",
    doc="Toom-3's evaluation for squaring, x = a0 + a1 2^m + a2 2^2m (m = ceil(N/3)): a0 + a1 + a2 (m + 2 bits), "
        "|a0 − a1 + a2| (m + 1) and a0 + 2 a1 + 4 a2 (m + 3), concatenated")


def _interp_sig(S):
    m = _toom_sig("PousToom3Interp", S)
    return ((("v0", bits(2 * m)), ("v1", bits(2 * m + 3)), ("vm1", bits(2 * m + 2)), ("v2", bits(2 * m + 6)),
             ("vinf", bits(2 * (S.N - 2 * m)))), bits(2 * S.N))


PousToom3Interp = CompositeDefinition(
    "PousToom3Interp", 1, ("N",), _interp_sig,
    _traced_body(lambda S: ("toom3-interp", S.N), lambda S: lambda C, *v: toom3_interp(C, S.N, *v), _interp_sig),
    conformance="exact-model-tested",
    doc="Toom-3's interpolation for squaring: x² (2N bits) from the squares of x's polynomial at 0, 1 (its low 2m + 3 "
        "bits), −1, 2 and ∞")


def _fold_sig(S):
    _check("PousModFold", W=S.W, C0=S.C0)
    if not S.C0 * S.C0 + 2 * S.C0 < 1 << S.W:
        raise ValueError(f"PousModFold needs C0² + 2 C0 < 2^W, so two folds leave z < 2p: not C0 = {S.C0}, W = {S.W}")
    return (("x", bits(2 * S.W)),), bits(S.W)


PousModFold = CompositeDefinition(
    "PousModFold", 1, ("W", "C0"), _fold_sig,
    _traced_body(lambda S: ("fold", S.W, S.C0), lambda S: lambda C, x: fold(C, S.W, S.C0, x), _fold_sig),
    conformance="exact-model-tested", doc="x mod (2^W − C0) of a 2W-bit unsigned integer, fully reduced")


def _mod_square_sig(S):
    _fold_sig(S)
    return (("x", bits(S.W)),), bits(S.W)


def _mod_square_body(B, S, x):
    sq = B.call(bind(PousSquare, N=S.W), x)
    return B.call(bind(PousModFold, W=S.W, C0=S.C0), sq)


PousModSquare = CompositeDefinition(
    "PousModSquare", 1, ("W", "C0"), _mod_square_sig, _mod_square_body, conformance="exact-model-tested",
    doc="x² mod (2^W − C0) of a W-bit unsigned integer, fully reduced: P2's squaring at W = 16448, C0 = 21065")

"""BabyBear^6 reference: naive and packed (base-field-first) provers for the degree-3 eq-weighted sumcheck, pure Python.

Field (shared with lane ``tc-sumcheck``'s ``backends/gkr/tensor/reference.py``, same constants so the two kernels agree
on the limb layout): BabyBear ``p = 2^31 - 2^27 + 1``; challenge field ``F_{p^6} = F_p[X] / (X^6 - W)``, ``W = 31`` the
smallest non-square non-cube (``x^6 - a`` is irreducible iff ``a`` is neither a square nor a cube: ``2, 3 | p - 1``).
Elements are 6-tuples in ``[0, p)``; a base element is ``(a, 0, 0, 0, 0, 0)``.  Limbs: 4 balanced signed 8-bit limbs
per coefficient (``limbs_balanced``), 24 per challenge element.

The sumcheck (Candidate A's GKR c-rounds, PROTOCOL.md 4.1, as ``explore.tensor`` prices them)::

    claim = sum_{c in {0,1}^n} eq(gamma, c) g(c) h(c),      gamma in F_{p^6}^n,  g, h : {0,1}^n -> F_p (layer values)

Variable ``j`` of ``c`` is bit ``j`` of the index (bit 0 = the first variable bound); the round-``i`` polynomial is
``S_i(x) = sum_b eq(gamma, r_1..r_{i-1}, x, b) g(r, x, b) h(r, x, b)`` (degree 3), sent as ``S_i(0..3)``.  The
verifier: ``S_1(0) + S_1(1) = claim``, ``S_i(0) + S_i(1) = S_{i-1}(r_{i-1})``, and finally
``S_n(r_n) = eq(gamma, r) g(r) h(r)``.  Nothing below changes the protocol: the three provers emit the same ``S_i``.

Provers:

* :func:`naive_prover` -- textbook: three tables ``f = eq(gamma, .)``, ``g``, ``h`` (any tables, all in ``F_{p^6}``),
  each round the 4 values by :func:`round_polynomial` and all three tables folded by ``r_i``.  The oracle.
* :func:`packed_prover` -- the algorithm the kernel implements (DESIGN.md 3):

  1. *eq-trick* (Gruen 2024/108 s.3, Dao-Thaler 2024/1210): ``eq(gamma, c)`` is never a table.
     ``S_i(x) = eq(gamma_{<i-1}, r_{<i}) . eq(gamma_{i-1}, x) . Q_i(x)`` with
     ``Q_i(x) = sum_b eq(gamma_{>=i}, b) g_i(x, b) h_i(x, b)`` of degree 2, computed at ``x in {0, 1, 2}``; ``Q_i(3)``
     by extrapolation and ``S_i(0..3)`` from it.  The weights ``eq(gamma_{>=i}, b)`` are the suffix of one eq table.
  2. *packed rounds* ``i <= k`` (Bagad-Domb-Thaler 2025/1117, the "small-value" idea): ``g, h`` stay base-field
     tables; ``g_i(x, b) = sum_{a in {0,1}^{i-1}} eq(r_{<i}, a) g(a, x, b)`` is expanded, so
     ``Q_i(x) = sum_{a, a'} eq(r, a) eq(r, a') T_i[x, a, a']``, ``T_i[x, a, a'] = sum_b eq(gamma_{>=i}, b) g(a, x, b) h(a', x, b)``:
     ``3 . 4^(i-1)`` inner products of a *base x base* Hadamard against the extension weights (base x extension MACs).
     ``g(a, 2, b) = 2 g(a, 1, b) - g(a, 0, b)`` is still base.
  3. after round ``k`` the tables are folded once by all ``k`` challenges (``g'(b) = sum_a eq(r_{1..k}, a) g(a, b)``,
     base x extension), and rounds ``k+1..n`` are ordinary extension rounds (Hadamard ``g_i(x) o h_i(x)`` in
     ``F_{p^6}``, weighted inner product, fold by ``r_i``).

``k = 0`` is the ordinary algorithm; ``k = 1`` is the ordinary algorithm with the natural base-field first round.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

P = 2**31 - 2**27 + 1
DEGREE = 6
LIMB_BITS = 8
LIMBS_PER_BASE = 4
LIMBS_PER_EXT = DEGREE * LIMBS_PER_BASE

Ext = tuple[int, int, int, int, int, int]


def _find_w() -> int:
    for a in range(2, P):
        if pow(a, (P - 1) // 2, P) != 1 and pow(a, (P - 1) // 3, P) != 1:
            return a
    raise RuntimeError("no non-residue found")


W = _find_w()
ZERO: Ext = (0,) * DEGREE
ONE: Ext = (1,) + (0,) * (DEGREE - 1)


# ---------------------------------------------------------------------------------------------------------------------
# field
# ---------------------------------------------------------------------------------------------------------------------


def base(a: int) -> Ext:
    return (a % P,) + (0,) * (DEGREE - 1)


def add(a: Ext, b: Ext) -> Ext:
    return tuple((x + y) % P for x, y in zip(a, b))


def sub(a: Ext, b: Ext) -> Ext:
    return tuple((x - y) % P for x, y in zip(a, b))


def scale(a: Ext, c: int) -> Ext:
    return tuple(x * c % P for x in a)


def mul(a: Ext, b: Ext) -> Ext:
    acc = [0] * (2 * DEGREE - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                acc[i + j] += x * y
    out = [acc[d] + W * acc[d + DEGREE] for d in range(DEGREE - 1)] + [acc[DEGREE - 1]]
    return tuple(v % P for v in out)


def is_base(a: Ext) -> bool:
    return all(v == 0 for v in a[1:])


def random_ext(rng: random.Random) -> Ext:
    return tuple(rng.randrange(P) for _ in range(DEGREE))


def random_base(rng: random.Random) -> Ext:
    return base(rng.randrange(P))


def eval_at(even: Ext, odd: Ext, x: int) -> Ext:
    """``(1 - x) even + x odd`` for a small integer ``x``."""
    return add(scale(even, (1 - x) % P), scale(odd, x % P))


def eval_at_ext(even: Ext, odd: Ext, r: Ext) -> Ext:
    return add(mul(sub(ONE, r), even), mul(r, odd))


def eq1(gamma: Ext, x: Ext) -> Ext:
    """``eq(gamma, x) = (1 - gamma)(1 - x) + gamma x`` on one variable."""
    return add(mul(sub(ONE, gamma), sub(ONE, x)), mul(gamma, x))


def eq_table(gammas: Sequence[Ext]) -> list[Ext]:
    """``eq(gammas, b)`` for ``b`` in ``{0,1}^m``, bit ``j`` of ``b`` = variable ``j``."""
    tbl = [ONE]
    for gm in gammas:
        one_m = sub(ONE, gm)
        tbl = [mul(one_m, v) for v in tbl] + [mul(gm, v) for v in tbl]
    return tbl


def eq_points(gammas: Sequence[Ext], rs: Sequence[Ext]) -> Ext:
    out = ONE
    for gm, r in zip(gammas, rs):
        out = mul(out, eq1(gm, r))
    return out


def lagrange_0123(values: Sequence[Ext], r: Ext) -> Ext:
    """The degree-3 polynomial through ``(0..3, values)`` evaluated at ``r``."""
    out = ZERO
    for i, v in enumerate(values):
        num, den = ONE, 1
        for j in range(4):
            if j != i:
                num = mul(num, sub(r, base(j)))
                den *= i - j
        out = add(out, mul(v, scale(num, pow(den % P, P - 2, P))))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# naive prover (three arbitrary tables)
# ---------------------------------------------------------------------------------------------------------------------


def round_polynomial(f: Sequence[Ext], g: Sequence[Ext], h: Sequence[Ext], points: Sequence[int] = (0, 1, 2, 3)) -> list[Ext]:
    """``S(x) = sum_k f_k(x) g_k(x) h_k(x)`` at ``points``, ``f_k(x) = (1 - x) f[2k] + x f[2k + 1]``."""
    t = len(f)
    if t % 2 or len(g) != t or len(h) != t:
        raise ValueError("tables must have equal even length")
    out = []
    for x in points:
        s = ZERO
        for k in range(t // 2):
            fk = eval_at(f[2 * k], f[2 * k + 1], x)
            gk = eval_at(g[2 * k], g[2 * k + 1], x)
            hk = eval_at(h[2 * k], h[2 * k + 1], x)
            s = add(s, mul(mul(fk, gk), hk))
        out.append(s)
    return out


def fold(table: Sequence[Ext], r: Ext) -> list[Ext]:
    """``table'[k] = (1 - r) table[2k] + r table[2k + 1]`` (binds the low bit)."""
    one_minus_r = sub(ONE, r)
    return [add(mul(one_minus_r, table[2 * k]), mul(r, table[2 * k + 1])) for k in range(len(table) // 2)]


Challenger = Callable[[int, list[Ext]], Ext]   # (round index i >= 1, S_i(0..3)) -> r_i


def naive_prover(f: Sequence[Ext], g: Sequence[Ext], h: Sequence[Ext], challenger: Challenger) -> tuple[list[list[Ext]], list[Ext], tuple[Ext, Ext, Ext]]:
    """All rounds of the textbook prover: returns (``[S_1, .., S_n]``, ``[r_1, .., r_n]``, final ``(f(r), g(r), h(r))``)."""
    f, g, h = list(f), list(g), list(h)
    rounds, rs = [], []
    i = 1
    while len(f) > 1:
        S = round_polynomial(f, g, h)
        r = challenger(i, S)
        rounds.append(S)
        rs.append(r)
        f, g, h = fold(f, r), fold(g, r), fold(h, r)
        i += 1
    return rounds, rs, (f[0], g[0], h[0])


# ---------------------------------------------------------------------------------------------------------------------
# verifier
# ---------------------------------------------------------------------------------------------------------------------


def verify(claim: Ext, rounds: Sequence[Sequence[Ext]], rs: Sequence[Ext], final: tuple[Ext, Ext, Ext]) -> bool:
    """The sumcheck verifier given the (oracle) final evaluations ``f(r), g(r), h(r)``."""
    expect = claim
    for S, r in zip(rounds, rs):
        if add(S[0], S[1]) != expect:
            return False
        expect = lagrange_0123(S, r)
    return expect == mul(mul(final[0], final[1]), final[2])


# ---------------------------------------------------------------------------------------------------------------------
# packed prover (eq-weighted, base-field g, h)
# ---------------------------------------------------------------------------------------------------------------------


def _q_to_s(Q: Sequence[Ext], prefix: Ext, gamma_i: Ext) -> list[Ext]:
    """``S(x) = prefix . eq(gamma_i, x) . Q(x)`` at ``x = 0..3`` from ``Q(0), Q(1), Q(2)`` (degree 2)."""
    Q3 = add(sub(scale(Q[2], 3), scale(Q[1], 3)), Q[0])
    out = []
    for x, q in enumerate((Q[0], Q[1], Q[2], Q3)):
        e = eval_at(sub(ONE, gamma_i), gamma_i, x)
        out.append(mul(mul(prefix, e), q))
    return out


def packed_prover(gammas: Sequence[Ext], g: Sequence[int], h: Sequence[int], k: int, challenger: Challenger
                  ) -> tuple[list[list[Ext]], list[Ext], tuple[Ext, Ext, Ext]]:
    """The eq-trick prover with ``k`` base-field ("packed") rounds; ``g, h`` are base-field integers.  Returns the same
    triple as :func:`naive_prover` on ``(eq_table(gammas), g, h)``."""
    n = len(gammas)
    t = 1 << n
    if len(g) != t or len(h) != t:
        raise ValueError("tables must have 2^n entries")
    if not 0 <= k <= n:
        raise ValueError("0 <= k <= n")
    g = [x % P for x in g]
    h = [x % P for x in h]
    weights = eq_table(gammas)          # eq(gamma, c); the round-i weights eq(gamma_{>=i}, b) are weights[b << i] / eq(gamma_{<i}, 0)
    rounds, rs = [], []
    prefix = ONE                          # eq(gamma_{<i-1}, r_{<i})
    # -- packed rounds: g, h untouched --
    for i in range(1, k + 1):
        m = t >> i
        w = _suffix_weights(gammas, i)    # eq(gamma_{i..n-1}, b), b in {0,1}^(n-i)
        eq_r = eq_table(rs)               # eq(r_{<i}, a), a in {0,1}^(i-1)
        Q = []
        for x in range(3):
            acc = ZERO
            for a in range(1 << (i - 1)):
                for a2 in range(1 << (i - 1)):
                    T = ZERO
                    for b in range(m):
                        gv = _base_eval(g, b, a, x, i)
                        hv = _base_eval(h, b, a2, x, i)
                        T = add(T, scale(w[b], gv * hv % P))
                    acc = add(acc, mul(mul(eq_r[a], eq_r[a2]), T))
            Q.append(acc)
        S = _q_to_s(Q, prefix, gammas[i - 1])
        r = challenger(i, S)
        rounds.append(S)
        rs.append(r)
        prefix = mul(prefix, eq1(gammas[i - 1], r))
    # -- one fold by all k challenges --
    if k:
        eq_r = eq_table(rs)
        gE = [_fold_k(g, b, eq_r, k) for b in range(t >> k)]
        hE = [_fold_k(h, b, eq_r, k) for b in range(t >> k)]
    else:
        gE, hE = [base(v) for v in g], [base(v) for v in h]
    # -- extension rounds --
    for i in range(k + 1, n + 1):
        m = len(gE) // 2
        w = _suffix_weights(gammas, i)
        Q = []
        for x in range(3):
            acc = ZERO
            for b in range(m):
                gv = eval_at(gE[2 * b], gE[2 * b + 1], x)
                hv = eval_at(hE[2 * b], hE[2 * b + 1], x)
                acc = add(acc, mul(w[b], mul(gv, hv)))
            Q.append(acc)
        S = _q_to_s(Q, prefix, gammas[i - 1])
        r = challenger(i, S)
        rounds.append(S)
        rs.append(r)
        prefix = mul(prefix, eq1(gammas[i - 1], r))
        gE, hE = fold(gE, r), fold(hE, r)
    return rounds, rs, (prefix, gE[0], hE[0])


def _suffix_weights(gammas: Sequence[Ext], i: int) -> list[Ext]:
    return eq_table(gammas[i:])


def _base_eval(tbl: Sequence[int], b: int, a: int, x: int, i: int) -> int:
    """``g(a, x, b)`` for ``x in {0, 1, 2}``: index ``(b << i) | (x << (i-1)) | a``; ``x = 2`` is ``2 g(a,1,b) - g(a,0,b)``."""
    lo = (b << i) | a
    v0 = tbl[lo]
    v1 = tbl[lo | (1 << (i - 1))]
    if x == 0:
        return v0
    if x == 1:
        return v1
    return (2 * v1 - v0) % P


def _fold_k(tbl: Sequence[int], b: int, eq_r: Sequence[Ext], k: int) -> Ext:
    acc = ZERO
    for a in range(1 << k):
        acc = add(acc, scale(eq_r[a], tbl[(b << k) | a]))
    return acc


def multilinear_eval(tbl: Sequence[Ext], rs: Sequence[Ext]) -> Ext:
    t = list(tbl)
    for r in rs:
        t = fold(t, r)
    return t[0]


# ---------------------------------------------------------------------------------------------------------------------
# limbs (the INT8 layout)
# ---------------------------------------------------------------------------------------------------------------------


def limbs_balanced(a: int, n: int = LIMBS_PER_BASE) -> list[int]:
    """``n`` signed 8-bit limbs ``l_j in [-128, 128)`` with ``a = sum l_j 2^(8j)``."""
    out = []
    for _ in range(n):
        l = ((a + 128) & 255) - 128
        out.append(l)
        a = (a - l) >> LIMB_BITS
    if a != 0:
        raise ValueError("value does not fit the limb count")
    return out


def from_limbs(ls: Sequence[int]) -> int:
    return sum(l << (LIMB_BITS * j) for j, l in enumerate(ls))


def seeded_challenger(seed: int) -> Challenger:
    """Deterministic challenges (a stand-in for the verifier's coins; not Fiat-Shamir)."""
    rng = random.Random(seed)

    def ch(_i: int, _S: list[Ext]) -> Ext:
        return random_ext(rng)

    return ch

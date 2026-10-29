"""N_L at toy sizes: a search for circuits of 6b's model that return a square's digits with few budgeted gates.

**The conjecture.** For every vector Q of integer quadratics in
the digits of a and every lattice Λ ⊆ ℤ^L of rank r, the a ∈ [0, p) for which some canonical output lies in Q(a) + Λ number
at most B^(r + d₀ − L)·p, with d₀ an absolute constant. At B = 256 that is 6b's `Renormalization F f w d ε` with
d = d₀ + log₂(1/ε)/8, and the formula under test is d ≈ 4 + log₂(1/ε)/8.

**The reduction the search uses.** Write a = Σ a_l B^l, s(a) for the honest folded column sums (t_l + δ·t_{l+L}, with
δ = B^L − p), and y(a) for the canonical digits of a² mod p. Then y = s + C·c(a) for one cyclic carry vector c(a) ∈ ℤ^L,
where C moves B units of column l into one of column l + 1 (and the top column's into δ of column 0); C has determinant p.
A circuit that outputs s + C·(W⁻¹·w), with k coordinates of w = W·c free (an integer quadratic q_j(a) plus a constant m_j)
and the other L − k budgeted, is correct exactly where the k constraints (W·c(a))_j = q_j(a) + m_j hold. So each constraint
set is a circuit with r = L − k budgeted gates (`build`), and every circuit of the model reduces to one by the closure lemma
(§R3.3 step 1). The search measures the largest fiber N_k it finds for each k, so ε = N_k/p and the measured d is k; the
formula then needs d₀ ≥ k − log_B(p/N_k) (`implied_d0`).

**The model.** `Op`, `Gate` and the functions below port `Security/Definitions/Accounting/Gates/Model.lean` (6b's branch
cursor/circuit-encode-proofs-6cd3 at f7a216e236): `raw`, `out`, `Ok`, `InWindow`, `derive`, `Proven`, `DeclaredHold`,
`FanIn`, `Width`, `budgeted`, and `Restricted` (`Codec.lean`). One generalization: an `mma` operand format has b bits, not 8,
so that at a toy base B = 2^b a digit fills its format as a byte does at B = 256 (the closure lemma's degree bound needs that).
At B = 256 the formats are Model.lean's s8 and u8.

Exhaustive over a ∈ [0, p) at every toy size; the constraint sets are searched exactly where stated (modes, every subset of
carries) and heuristically elsewhere (lines through the heaviest points, small and LLL-short constraint rows, a beam).
Past the exhaustive sizes, byte fields of L limbs take product families of small digits whose squares never carry
(`greedy_series`, and p1024's own): a circuit with no budgeted gate that is exact on all of them.
"""

from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

S32_MIN, S32_MAX = -(2 ** 31), 2 ** 31 - 1
P1024 = 2 ** 1024 - 105  # p1024 (verity.core.protocols.remote_bandwidth.codec, Field("p1024", 1024, 105))


# ---- the model: Model.lean's gates, values, window and interval analysis -------------------------------------------------


def fmt_range(fmt: tuple[bool, int]) -> tuple[int, int]:
    """An operand format (signed, bits): s8 is (True, 8), [-128, 127]; u8 is (False, 8), [0, 255]."""
    signed, bits = fmt
    return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2 ** bits - 1)


S8, U8 = (True, 8), (False, 8)


@dataclass(frozen=True)
class Op:
    """`Op ι`: an input slot (by binding), a constant, an affine map `c + Σ coeff·v`, an s8-family coordinate `mma`, a
    `gather`, or a `budgeted` primitive of class `cls` (any function of its operands, vectorized over inputs)."""

    kind: str
    c: int = 0
    coeff: tuple[int, ...] = ()
    fa: tuple[bool, int] = S8
    fb: tuple[bool, int] = S8
    k: int = 0
    sat: bool = True
    slot: int = -1
    binding: str = "registered"
    cls: str = "rounding"
    f: Callable[[list[np.ndarray]], np.ndarray] | None = field(default=None, compare=False)

    @property
    def derived(self) -> bool:
        return self.kind in ("const", "affine", "mma")

    @property
    def charged(self) -> bool:
        return self.kind in ("gather", "budgeted") or (self.kind == "input" and self.binding == "run")

    @property
    def budget(self) -> str | None:
        return "gather" if self.kind == "gather" else self.cls if self.kind == "budgeted" else None


@dataclass(frozen=True)
class Gate:
    op: Op
    args: tuple[int, ...]
    lo: int
    hi: int

    @property
    def bits(self) -> int:
        return max(0, math.ceil(math.log2(max(1, self.hi + 1 - self.lo))))


def _dot(k: int, v: Sequence[np.ndarray], n: int) -> np.ndarray:
    a, b = v[:k], v[k:]
    out = np.zeros(n, dtype=object if any(x.dtype == object for x in v) else np.int64)
    for x, y in zip(a, b):
        out = out + x * y
    return out


def raw(op: Op, x: dict[int, np.ndarray], v: list[np.ndarray], n: int) -> np.ndarray:
    """`Op.raw`: the exact value before an s8-family step's clamp or wrap."""
    if op.kind == "input":
        return x[op.slot]
    if op.kind == "const":
        return np.full(n, op.c, dtype=np.int64)
    if op.kind == "affine":
        out = np.full(n, op.c, dtype=np.int64)
        for cf, val in zip(op.coeff, v):
            out = out + cf * val
        return out
    if op.kind == "mma":
        head = v[0] if v else np.zeros(n, dtype=np.int64)
        return head + _dot(op.k, v[1:], n)
    if op.kind == "gather":
        idx = np.maximum(v[0], 0) if v else np.zeros(n, dtype=np.int64)
        col = np.stack(v[1:]) if len(v) > 1 else np.zeros((1, n), dtype=np.int64)
        ok = idx < len(v) - 1
        return np.where(ok, col[np.minimum(idx, max(0, len(v) - 2)), np.arange(n)], 0)
    return np.asarray(op.f(v), dtype=np.int64)  # type: ignore[misc]


def out(op: Op, z: np.ndarray) -> np.ndarray:
    """`Op.out`: an s8-family step clamps (`satfinite`) or wraps to s32; every other op passes its exact value."""
    if op.kind != "mma":
        return z
    if op.sat:
        return np.clip(z, S32_MIN, S32_MAX)
    return (z - S32_MIN) % 2 ** 32 + S32_MIN


def operands_ok(op: Op, v: list[np.ndarray], n: int) -> np.ndarray:
    if op.kind != "mma":
        return np.ones(n, dtype=bool)
    if len(v) != 2 * op.k + 1:
        return np.zeros(n, dtype=bool)
    ok = np.ones(n, dtype=bool)
    for z in v[1:op.k + 1]:
        lo, hi = fmt_range(op.fa)
        ok &= (lo <= z) & (z <= hi)
    for z in v[op.k + 1:]:
        lo, hi = fmt_range(op.fb)
        ok &= (lo <= z) & (z <= hi)
    return ok


def evaluate(C: Sequence[Gate], x: dict[int, np.ndarray], n: int) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """`values` with, per input, `InWindow` (every gate `Ok`) and `DeclaredHold` (inputs and budgeted gates in their
    declared intervals), as boolean arrays over the n inputs."""
    vs: list[np.ndarray] = []
    window = np.ones(n, dtype=bool)
    hold = np.ones(n, dtype=bool)
    for g in C:
        v = [vs[j] if j < len(vs) else np.zeros(n, dtype=np.int64) for j in g.args]
        z = raw(g.op, x, v, n)
        inside = (g.lo <= z) & (z <= g.hi)
        ok = inside & operands_ok(g.op, v, n)
        if g.op.kind == "mma" and not (S32_MIN <= g.lo and g.hi <= S32_MAX):
            ok &= False
        window &= ok
        if not g.op.derived:
            hold &= inside
        vs.append(out(g.op, z))
    return vs, window, hold


Iv = tuple[int, int]


def derive(g: Gate, ivs: Sequence[Iv]) -> Iv:
    """`Gate.derive`: a constant's value, an affine map's or an s8-family step's interval image, else the declared one."""
    a = [ivs[j] if j < len(ivs) else (0, 0) for j in g.args]
    if g.op.kind == "const":
        return (g.op.c, g.op.c)
    if g.op.kind == "affine":
        lo = hi = g.op.c
        for cf, (l, h) in zip(g.op.coeff, a):
            lo, hi = lo + min(cf * l, cf * h), hi + max(cf * l, cf * h)
        return (lo, hi)
    if g.op.kind == "mma":
        lo, hi = a[0] if a else (0, 0)
        k = g.op.k
        for (l1, h1), (l2, h2) in zip(a[1:k + 1], a[k + 1:]):
            ps = (l1 * l2, l1 * h2, h1 * l2, h1 * h2)
            lo, hi = lo + min(ps), hi + max(ps)
        return (lo, hi)
    return (g.lo, g.hi)


def intervals(C: Sequence[Gate]) -> list[Iv]:
    ivs: list[Iv] = []
    for g in C:
        ivs.append(derive(g, ivs))
    return ivs


def proven(C: Sequence[Gate]) -> list[str]:
    """`Proven C`'s failures (empty when it holds): a derived interval outside its declared one, or an s8-family step whose
    operands' derived intervals leave their formats or whose declared interval leaves s32."""
    bad = []
    ivs: list[Iv] = []
    for i, g in enumerate(C):
        d = derive(g, ivs)
        if g.op.derived and not (g.lo <= d[0] and d[1] <= g.hi):
            bad.append(f"gate {i}: derived {d} outside declared {(g.lo, g.hi)}")
        if g.op.kind == "mma":
            k = g.op.k
            if len(g.args) != 2 * k + 1 or not (S32_MIN <= g.lo and g.hi <= S32_MAX):
                bad.append(f"gate {i}: mma shape or declared interval outside s32")
            for j in g.args[1:k + 1]:
                lo, hi = fmt_range(g.op.fa)
                if not (lo <= ivs[j][0] and ivs[j][1] <= hi):
                    bad.append(f"gate {i}: operand {j}'s derived {ivs[j]} outside {(lo, hi)}")
            for j in g.args[k + 1:]:
                lo, hi = fmt_range(g.op.fb)
                if not (lo <= ivs[j][0] and ivs[j][1] <= hi):
                    bad.append(f"gate {i}: operand {j}'s derived {ivs[j]} outside {(lo, hi)}")
        ivs.append(d)
    return bad


def fan_in(f: int, C: Sequence[Gate]) -> bool:
    return all(not g.op.charged or g.op.budget == "gather" or len(g.args) <= f for g in C)


def width(w: int, C: Sequence[Gate]) -> bool:
    return all(-(2 ** w) <= g.lo and g.hi < 2 ** w for g in C)


def restricted(f: int, w: int, C: Sequence[Gate]) -> bool:
    """`Restricted f w C` (Codec.lean): `Proven C ∧ FanIn f C ∧ Width w C`."""
    return not proven(C) and fan_in(f, C) and width(w, C)


def budgeted(C: Sequence[Gate]) -> int:
    return sum(1 for g in C if g.op.budget is not None)


def run_bits(C: Sequence[Gate]) -> int:
    return sum(g.bits for g in C if g.op.kind == "input" and g.op.binding == "run")


# ---- toy fields and their tables ------------------------------------------------------------------------------------------


def is_prime(n: int) -> bool:
    """Miller–Rabin on the first 20 prime bases: deterministic below 3.3·10^24, and a fixed-base test above."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)
    for q in small:
        if n % q == 0:
            return n == q
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


@dataclass(frozen=True)
class Field:
    """A toy field: base B = 2^b, L digits, and p the largest prime below B^L with p ≡ 3 (mod 4), as p1024 is."""

    B: int
    L: int
    p: int

    @property
    def delta(self) -> int:
        return self.B ** self.L - self.p

    @property
    def b(self) -> int:
        return self.B.bit_length() - 1

    @staticmethod
    def toy(B: int, L: int) -> "Field":
        p = B ** L - 1
        while not (p % 4 == 3 and is_prime(p)):
            p -= 1
        return Field(B, L, p)


@dataclass
class Table:
    """Every a ∈ [0, p): its digits, folded column sums s, canonical digits y of a² mod p, and cyclic carries c."""

    F: Field
    digits: np.ndarray
    s: np.ndarray
    y: np.ndarray
    c: np.ndarray

    @property
    def n(self) -> int:
        return self.F.p


def column_sums(d: np.ndarray, L: int, delta: int) -> np.ndarray:
    """The honest free part: t_l = Σ_{i+j=l} a_i a_j (one mma each), folded s_l = t_l + δ·t_{l+L}."""
    n = d.shape[0]
    t = np.zeros((n, 2 * L - 1), dtype=np.int64)
    for i in range(L):
        t[:, i:i + L] += d[:, i:i + 1] * d
    s = t[:, :L].copy()
    s[:, :L - 1] += delta * t[:, L:]
    return s


def carries(s: np.ndarray, y: np.ndarray, B: int, p: int, delta: int) -> np.ndarray:
    """The cyclic carry vector: y_l = s_l + c_{l−1} − B·c_l, with c_{−1} read as δ·c_{L−1}; c_{L−1} = (val s − val y)/p."""
    n, L = s.shape
    pw = np.array([B ** l for l in range(L)], dtype=np.int64)
    K, r = np.divmod(s @ pw - y @ pw, p)
    assert not r.any()
    c = np.zeros((n, L), dtype=np.int64)
    prev = delta * K
    for l in range(L):
        num = s[:, l] + prev - y[:, l]
        assert not (num % B).any()
        c[:, l] = num // B
        prev = c[:, l]
    assert (c[:, L - 1] == K).all()
    return c


def table(F: Field) -> Table:
    a = np.arange(F.p, dtype=np.int64)
    pw = [F.B ** l for l in range(F.L)]
    d = np.stack([(a // q) % F.B for q in pw], axis=1)
    sq = (a * a) % F.p
    y = np.stack([(sq // q) % F.B for q in pw], axis=1)
    s = column_sums(d, F.L, F.delta)
    return Table(F, d, s, y, carries(s, y, F.B, F.p, F.delta))


# ---- integer linear algebra: unimodular completion, LLL -------------------------------------------------------------------


def _colreduce(rows: list[list[int]], L: int) -> tuple[list[list[int]], list[list[int]], int]:
    """Column operations to a lower echelon form: rows·V = A with V unimodular; returns (A, V, rank). The columns of V past
    the rank are a ℤ-basis of the integer kernel (saturated, since V is unimodular)."""
    A = [list(r) for r in rows]
    V = [[int(i == j) for j in range(L)] for i in range(L)]

    def colop(i: int, j: int, a: int, b: int, c: int, d: int) -> None:  # (col_i, col_j) <- (a·col_i + b·col_j, c·col_i + d·col_j)
        for M in (A, V):
            for row in M:
                x, y = row[i], row[j]
                row[i], row[j] = a * x + b * y, c * x + d * y

    col = 0
    for i in range(len(A)):
        if col == L:
            break
        piv = next((j for j in range(col, L) if A[i][j]), None)
        if piv is None:
            continue
        if piv != col:
            colop(col, piv, 0, 1, 1, 0)
        for j in range(col + 1, L):
            x, y = A[i][col], A[i][j]
            if y == 0:
                continue
            g, u, v = _egcd(x, y)
            colop(col, j, u, v, -y // g, x // g)
        col += 1
    return A, V, col


def integer_kernel(rows: list[list[int]], L: int) -> list[list[int]]:
    """A ℤ-basis of {v ∈ ℤ^L : rows·v = 0}, saturated."""
    _A, V, rank = _colreduce(rows, L)
    return [[V[i][j] for i in range(L)] for j in range(rank, L)]


def unimodular_completion(Psi: list[list[int]], L: int) -> list[list[int]] | None:
    """An L×L integer matrix W with det ±1 whose first k rows are `Psi`, or None when Psi's rows don't span a saturated
    lattice of rank k (then no circuit with L − k budgeted gates reads its fiber off free values)."""
    k = len(Psi)
    if k == 0:
        return [[int(i == j) for j in range(L)] for i in range(L)]
    A, V, rank = _colreduce(Psi, L)
    if rank != k or any(A[i][j] for i in range(k) for j in range(k, L)):
        return None
    H = [row[:k] for row in A]
    det = _det(H)
    if abs(det) != 1:
        return None
    Vinv = _inverse_unimodular(V)
    top = [[sum(H[i][t] * Vinv[t][j] for t in range(k)) for j in range(L)] for i in range(k)]
    W = top + [Vinv[i] for i in range(k, L)]
    assert [r for r in W[:k]] == [list(r) for r in Psi]
    return W


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    """g = gcd(a, b) ≥ 0 with u·a + v·b = g (and the 2×2 column step (u v; −b/g a/g) unimodular)."""
    x0, y0, x1, y1, r0, r1 = 1, 0, 0, 1, a, b
    while r1:
        q = r0 // r1
        r0, r1, x0, x1, y0, y1 = r1, r0 - q * r1, x1, x0 - q * x1, y1, y0 - q * y1
    if r0 < 0:
        r0, x0, y0 = -r0, -x0, -y0
    return r0, x0, y0


def _det(M: list[list[int]]) -> int:
    from fractions import Fraction

    n = len(M)
    A = [[Fraction(x) for x in r] for r in M]
    det = Fraction(1)
    for i in range(n):
        piv = next((r for r in range(i, n) if A[r][i] != 0), None)
        if piv is None:
            return 0
        if piv != i:
            A[i], A[piv] = A[piv], A[i]
            det = -det
        det *= A[i][i]
        for r in range(i + 1, n):
            f = A[r][i] / A[i][i]
            for c in range(i, n):
                A[r][c] -= f * A[i][c]
    return int(det)


def _inverse_unimodular(M: list[list[int]]) -> list[list[int]]:
    from fractions import Fraction

    n = len(M)
    A = [[Fraction(x) for x in r] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(M)]
    for i in range(n):
        piv = next(r for r in range(i, n) if A[r][i] != 0)
        A[i], A[piv] = A[piv], A[i]
        pv = A[i][i]
        A[i] = [x / pv for x in A[i]]
        for r in range(n):
            if r != i and A[r][i] != 0:
                f = A[r][i]
                A[r] = [x - f * y for x, y in zip(A[r], A[i])]
    out = [[A[i][n + j] for j in range(n)] for i in range(n)]
    assert all(x.denominator == 1 for r in out for x in r)
    return [[int(x) for x in r] for r in out]


def lll(G: np.ndarray, delta: float = 0.99) -> list[list[int]]:
    """An LLL-reduced basis of ℤ^n under the quadratic form G (positive definite): integer vectors φ with φᵀGφ small first.
    Floating Gram–Schmidt, exact integer basis; n ≤ about 40."""
    n = G.shape[0]
    Bm = np.eye(n, dtype=np.int64)

    def gs() -> tuple[np.ndarray, np.ndarray]:
        Bf = Bm.astype(float)
        R = np.linalg.cholesky(Bf @ G @ Bf.T)  # Gb = R·Rᵀ: mu = R/diag, b*² = diag²
        dg = np.diag(R)
        return R / dg[None, :], dg ** 2

    mu, bstar = gs()
    k, guard = 1, 0
    while k < n and guard < 20000:
        guard += 1
        for j in range(k - 1, -1, -1):
            q = round(mu[k, j])
            if q:
                Bm[k] -= q * Bm[j]
                mu, bstar = gs()
        if bstar[k] >= (delta - mu[k, k - 1] ** 2) * bstar[k - 1]:
            k += 1
        else:
            Bm[[k, k - 1]] = Bm[[k - 1, k]]
            mu, bstar = gs()
            k = max(k - 1, 1)
    return [list(map(int, r)) for r in Bm]


# ---- fibers ---------------------------------------------------------------------------------------------------------------

_RNG = np.random.default_rng(20261017)
_MIX = _RNG.integers(1, 2 ** 62, size=256, dtype=np.int64) | 1


def fiber(M: np.ndarray) -> tuple[int, np.ndarray]:
    """The largest set of rows of M with one value, and that value (exact: the hashed winner is recounted)."""
    if M.shape[1] == 0:
        return M.shape[0], M[:0].reshape(0)
    with np.errstate(over="ignore"):
        h = (M * _MIX[:M.shape[1]]).sum(axis=1)
    vals, inv, cnt = np.unique(h, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)
    best, value = 0, M[0]
    for t in order[:4]:
        rows = M[inv == t]
        u, uc = np.unique(rows, axis=0, return_counts=True)
        i = int(np.argmax(uc))
        if uc[i] > best:
            best, value = int(uc[i]), u[i]
        if cnt[order[0]] <= best:
            break
    return best, value


def monomials(d: np.ndarray) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """The integer monomials of degree 1 and 2 in the digits: a_i, and a_i·a_j for i ≤ j."""
    L = d.shape[1]
    names: list[tuple[int, ...]] = [(i,) for i in range(L)] + [(i, j) for i in range(L) for j in range(i, L)]
    cols = [d[:, i] for i in range(L)] + [d[:, i] * d[:, j] for i in range(L) for j in range(i, L)]
    return np.stack(cols, axis=1), names


@dataclass
class Constraint:
    """k rows over (c, monomials): row j says Σ psi[j]·c(a) − Σ q[j]·mono(a) = m[j]. Its fiber is where all hold."""

    psi: list[list[int]]
    q: list[list[int]]
    m: list[int]
    count: int
    method: str

    @property
    def k(self) -> int:
        return len(self.psi)


def evaluate_rows(T: Table, mono: np.ndarray, psi: list[list[int]], q: list[list[int]]) -> np.ndarray:
    P = np.array(psi, dtype=np.int64).reshape(len(psi), T.F.L)
    out = T.c @ P.T
    if q and any(any(r) for r in q):
        Q = np.array(q, dtype=np.int64).reshape(len(q), mono.shape[1])
        out = out - mono @ Q.T
    return out


def constraint(T: Table, mono: np.ndarray, psi: list[list[int]], q: list[list[int]] | None, method: str) -> Constraint:
    q = q or [[0] * mono.shape[1] for _ in psi]
    cnt, val = fiber(evaluate_rows(T, mono, psi, q))
    return Constraint(psi, q, [int(v) for v in val], cnt, method)


# ---- the search -----------------------------------------------------------------------------------------------------------


def search(T: Table, *, lines_bases: int = 200, small_coeff: int = 1, max_rows: int = 1500, beam: int = 8,
           lll_quadratic: bool = True, seed: int = 1) -> dict[int, Constraint]:
    """The largest fiber found for each k = 1..L constraints (r = L − k budgeted gates), by:

    * ``subset``: every subset S of the carries, guessed at their joint mode (exact over subsets);
    * ``mode``: k = L, the whole carry vector at its mode (exact), and ``carry-free`` (c = 0: y = s, the column sums);
    * ``line``: k = L − 1, the heaviest affine line of carry vectors through one of the `lines_bases` heaviest points;
    * ``row``: k = 1, every row with entries in [−small_coeff, small_coeff] (`max_rows` random ones when there are more),
      and LLL-short rows of the carries' covariance (and of the carries' with the quadratic monomials', ``row+q``);
    * ``beam``: k rows from the pool of the best single rows, extended greedily (`beam` states kept).
    """
    F, L = T.F, T.F.L
    mono, _names = monomials(T.digits)
    D = mono.shape[1]
    zero_q = [0] * D
    best: dict[int, Constraint] = {}

    def offer(cn: Constraint) -> None:
        if cn.k == 0 or unimodular_completion(cn.psi, L) is None:
            return
        cur = best.get(cn.k)
        if cur is None or cn.count > cur.count:
            best[cn.k] = cn

    eye = [[int(i == j) for j in range(L)] for i in range(L)]
    for k in range(1, L + 1):
        for S in itertools.combinations(range(L), k):
            offer(constraint(T, mono, [eye[i] for i in S], None, "subset"))
    cf = int((T.c == 0).all(axis=1).sum())
    offer(Constraint(eye, [zero_q] * L, [0] * L, cf, "carry-free"))
    offer(constraint(T, mono, eye, None, "mode"))

    # lines: k = L − 1
    if L >= 2:
        V, w = np.unique(T.c, axis=0, return_counts=True)
        order = np.argsort(-w)[:lines_bases]
        top = None
        for i in order:
            dv = V - V[i]
            nz = np.abs(dv).max(axis=1) > 0
            dv, wv = dv[nz], w[nz]
            g = np.gcd.reduce(np.abs(dv), axis=1)
            dv = dv // g[:, None]
            first = np.argmax(dv != 0, axis=1)
            sgn = np.sign(dv[np.arange(len(dv)), first])
            dv = dv * sgn[:, None]
            u, inv = np.unique(dv, axis=0, return_inverse=True)
            tot = np.bincount(inv.ravel(), weights=wv)
            j = int(np.argmax(tot))
            cnt = int(w[i] + tot[j])
            if top is None or cnt > top[0]:
                top = (cnt, V[i], u[j])
        if top is not None:
            cnt, base, dirn = top
            psi = integer_kernel([list(map(int, dirn))], L)
            cn = constraint(T, mono, psi, None, "line")
            if cn.count >= cnt:
                offer(cn)

    # single rows (k = 1)
    pool: list[Constraint] = []
    span = range(-small_coeff, small_coeff + 1)
    if (2 * small_coeff + 1) ** L <= 2 * max_rows:
        cands = list(itertools.product(span, repeat=L))
    else:
        r = random.Random(seed)
        cands = [tuple(r.choice(span) for _ in range(L)) for _ in range(max_rows)]
    for row in dict.fromkeys(cands):
        if not any(row) or next(x for x in row if x) < 0 or math.gcd(*row) != 1:
            continue
        pool.append(constraint(T, mono, [list(row)], None, "row"))
    cov = np.cov(T.c.T.astype(float)).reshape(L, L) + 1e-9 * np.eye(L)
    for v in lll(cov):
        if any(v) and math.gcd(*v) == 1:
            pool.append(constraint(T, mono, [v], None, "row-lll"))
    if lll_quadratic and L + D <= 64:
        X = np.concatenate([T.c, mono], axis=1).astype(float)
        G = np.cov(X.T) + 1e-6 * np.eye(L + D)
        for v in lll(G):
            psi, q = v[:L], [-x for x in v[L:]]
            if any(psi) and math.gcd(*psi) == 1:
                pool.append(constraint(T, mono, [psi], [q], "row+q-lll"))
    pool.sort(key=lambda cn: -cn.count)
    for cn in pool[:1]:
        offer(cn)

    # beam over the pool's best rows (and the carries' unit rows)
    singles = pool[:16] + [constraint(T, mono, [eye[i]], None, "subset") for i in range(L)]
    states = [cn for cn in singles[:beam]]
    for k in range(2, L + 1):
        nxt: list[Constraint] = []
        seen: set = set()
        for st in states:
            for s1 in singles:
                psi = st.psi + s1.psi
                if np.linalg.matrix_rank(np.array(psi, dtype=float)) < k:
                    continue
                key = tuple(sorted(map(tuple, psi)))
                if key in seen:
                    continue
                seen.add(key)
                nxt.append(constraint(T, mono, psi, st.q + s1.q, "beam"))
        nxt.sort(key=lambda cn: -cn.count)
        states = nxt[:beam]
        for cn in states:
            offer(cn)
    # rows of a saturated set are saturated, and fewer rows hold on a superset: N_{k−1} ≥ N_k
    for k in range(L, 1, -1):
        cn = best.get(k)
        if cn is not None and (k - 1 not in best or best[k - 1].count < cn.count):
            offer(constraint(T, mono, cn.psi[:-1], cn.q[:-1], cn.method + "-row"))
    return best


# ---- circuits: a constraint set as a circuit of the model ----------------------------------------------------------------


@dataclass
class Built:
    C: list[Gate]
    out: list[int]
    slots: list[int]


def build(T: Table, cn: Constraint | None, mono_names: list[tuple[int, ...]]) -> Built:
    """The circuit whose outputs are s + C·(W⁻¹·w): w's first k coordinates free (q_j(a) + m_j: s8-family products and
    affine maps), its last L − k budgeted gates (each any function of the word a, one operand), and W a unimodular
    completion of the constraint rows. `cn = None` is the honest circuit (k = 0, L budgeted carries)."""
    F, L, B = T.F, T.F.L, T.F.B
    fu = (False, F.b)
    C: list[Gate] = []
    ivs: list[Iv] = []

    def add(op: Op, args: Sequence[int], lo: int | None = None, hi: int | None = None) -> int:
        g = Gate(op, tuple(args), 0, 0)
        if op.derived:
            dlo, dhi = derive(g, ivs)
            g = Gate(op, tuple(args), dlo, dhi)
        else:
            g = Gate(op, tuple(args), int(lo), int(hi))  # type: ignore[arg-type]
        C.append(g)
        ivs.append(derive(g, ivs))
        return len(C) - 1

    dig = [add(Op("input", slot=l, binding="registered"), (), 0, B - 1) for l in range(L)]
    zero = add(Op("const", c=0), ())
    A = add(Op("affine", c=0, coeff=tuple(B ** l for l in range(L))), dig)
    t = []
    for col in range(2 * L - 1):
        pairs = [(i, col - i) for i in range(L) if 0 <= col - i < L]
        t.append(add(Op("mma", fa=fu, fb=fu, k=len(pairs)), [zero] + [dig[i] for i, _ in pairs] + [dig[j] for _, j in pairs]))
    s = [add(Op("affine", c=0, coeff=(1, F.delta)), (t[l], t[l + L])) if l + L < 2 * L - 1 else
         add(Op("affine", c=0, coeff=(1,)), (t[l],)) for l in range(L)]

    k = cn.k if cn else 0
    W = unimodular_completion(cn.psi if cn else [], L)
    assert W is not None
    Winv = _inverse_unimodular(W)
    w_full = T.c @ np.array(W, dtype=np.int64).T  # w(a) = W·c(a), for every a
    good = np.ones(F.p, dtype=bool)
    if cn:
        mono, _ = monomials(T.digits)
        good = (evaluate_rows(T, mono, cn.psi, cn.q) == np.array(cn.m)).all(axis=1)
    w = []
    prods: dict[tuple[int, ...], int] = {}
    for j in range(k):
        terms, coefs = [], []
        for cf, name in zip(cn.q[j], mono_names):  # type: ignore[union-attr]
            if not cf:
                continue
            if len(name) == 1:
                g = dig[name[0]]
            else:
                if name not in prods:
                    prods[name] = add(Op("mma", fa=fu, fb=fu, k=1), (zero, dig[name[0]], dig[name[1]]))
                g = prods[name]
            terms.append(g)
            coefs.append(cf)
        w.append(add(Op("affine", c=cn.m[j], coeff=tuple(coefs)), terms))  # type: ignore[union-attr]
    for j in range(k, L):
        col = w_full[:, j]
        lut = col.copy()
        lo, hi = int(col[good].min()), int(col[good].max())
        w.append(add(Op("budgeted", cls="rounding", f=lambda v, lut=lut: lut[v[0]]), (A,), lo, hi))
    c = [add(Op("affine", c=0, coeff=tuple(Winv[l])), w) for l in range(L)]
    y = [add(Op("affine", c=0, coeff=(1, F.delta, -B)), (s[0], c[L - 1], c[0]))] + \
        [add(Op("affine", c=0, coeff=(1, 1, -B)), (s[l], c[l - 1], c[l])) for l in range(1, L)]
    return Built(C, y, dig)


def renormalization_witness(T: Table, bc: Built, f: int = 4, w: int = 31) -> dict:
    """Renormalization's premises for `bc` (Restricted f w, no run input, every free input a digit of a) and the a on which
    it holds its declared intervals and outputs the canonical digits: the `good` set, and its budgeted gates."""
    F = T.F
    x = {l: T.digits[:, l] for l in range(F.L)}
    vs, window, hold = evaluate(bc.C, x, F.p)
    outs = np.stack([vs[i] for i in bc.out], axis=1)
    good = hold & (outs == T.y).all(axis=1)
    return {"restricted": restricted(f, w, bc.C), "proven_failures": proven(bc.C)[:3], "run_bits": run_bits(bc.C),
            "budgeted": budgeted(bc.C), "good": int(good.sum()), "window_on_good": bool(window[good].all()),
            "gates": len(bc.C), "good_mask": good}


# ---- measured d against the formula ---------------------------------------------------------------------------------------


def implied_d0(F: Field, k: int, count: int) -> float:
    """The d₀ the formula d = d₀ + log_B(1/ε) needs to cover a circuit with L − k budgeted gates on `count` of the p inputs:
    d = k at ε = count/p."""
    return k - math.log(F.p / count, F.B)


def run_field(B: int, L: int, **kw) -> dict:
    F = Field.toy(B, L)
    T = table(F)
    best = search(T, **kw)
    _m, names = monomials(T.digits[:1])
    rows = []
    for k in sorted(best):
        cn = best[k]
        wit = renormalization_witness(T, build(T, cn, names))
        assert wit["restricted"] and wit["run_bits"] == 0 and wit["window_on_good"], wit
        assert wit["budgeted"] == L - k and wit["good"] == cn.count, (wit, cn)
        eps = cn.count / F.p
        d0 = implied_d0(F, k, cn.count)
        rows.append({"k": k, "r": L - k, "count": cn.count, "method": cn.method, "log2_inv_eps": math.log2(1 / eps),
                     "cost_digits": math.log(1 / eps, B), "implied_d0": d0, "implied_d0_bits": d0 * F.b,
                     "rows": cn.psi if k <= 2 else None, "quadratic": any(any(r) for r in cn.q)})
    honest = renormalization_witness(T, build(T, None, names))
    assert honest["good"] == F.p and honest["budgeted"] == L
    return {"B": B, "L": L, "p": F.p, "delta": F.delta, "carry_vectors": int(len(np.unique(T.c, axis=0))),
            "carry_free": int((T.c == 0).all(axis=1).sum()), "honest_budgeted": honest["budgeted"],
            "frontier": rows, "max_implied_d0": max(r["implied_d0"] for r in rows)}


def carry_free_count(F: Field) -> int:
    """The a ∈ [0, p) whose folded column sums are all digits (< B) with val(s) < p, so y = s and the column-sum circuit
    (no budgeted gate) is exact on them: counted by a depth-first walk over the digits, pruned when a partial folded sum
    reaches B (sums only grow with a digit). Exact for any (B, L); every such a has a_l² < B, so the walk stays small."""
    B, L, delta, p = F.B, F.L, F.delta, F.p
    t = [0] * (2 * L - 1)
    digits = [0] * L
    count = 0

    def folded(m: int) -> int:
        return t[m] + (delta * t[m + L] if m + L < 2 * L - 1 else 0)

    def walk(l: int) -> None:
        nonlocal count
        if l == L:
            if sum(folded(m) * B ** m for m in range(L)) < p:
                count += 1
            return
        for a in range(B):
            digits[l] = a
            t[2 * l] += a * a
            for i in range(l):
                t[l + i] += 2 * a * digits[i]
            good = all(folded(m) < B for m in range(L))
            if good:
                walk(l + 1)
            t[2 * l] -= a * a
            for i in range(l):
                t[l + i] -= 2 * a * digits[i]
            if not good and a > 0:
                break  # a larger digit only adds to every sum it touches
        digits[l] = 0

    walk(0)
    return count


def carry_free_series(B: int = 256, Ls: Sequence[int] = range(1, 9)) -> list[dict]:
    """The carry-free count at base B (the real one, 256) for each L: the r = 0 fiber N_L's d₀ has to cover, exactly."""
    out = []
    for L in Ls:
        F = Field.toy(B, L)
        n = carry_free_count(F)
        out.append({"B": B, "L": L, "p_log2": math.log2(F.p), "delta": F.delta, "carry_free": n,
                    "implied_d0": implied_d0(F, L, n), "implied_d0_bits": implied_d0(F, L, n) * F.b})
    return out


# ---- p1024: the small-digit families and the cost of guessing a carry ------------------------------------------------------


def p1024_column_sums(digits: Sequence[int], L: int = 128, delta: int = 105) -> list[int]:
    t = [0] * (2 * L - 1)
    nz = [(i, a) for i, a in enumerate(digits) if a]
    for i, a in nz:
        for j, b in nz:
            t[i + j] += a * b
    return [t[l] + (delta * t[l + L] if l + L < 2 * L - 1 else 0) for l in range(L)]


def p1024_family(maxdigit: Sequence[int], p: int = P1024) -> dict:
    """A product family {a : 0 ≤ a_l ≤ maxdigit[l]} on which the column-sum circuit (r = 0) is exact: its folded column
    sums at the all-max member bound every member's (products are monotone in nonnegative digits), so if those are ≤ 255
    and their value is below p, every member's folded sums are its canonical digits: y = s on all of them. The field is
    p1024 unless `p` (just below 256^len(maxdigit)) says otherwise."""
    L = len(maxdigit)
    s = p1024_column_sums(maxdigit, L, 256 ** L - p)
    exact = max(s) <= 255 and sum(v << (8 * l) for l, v in enumerate(s)) < p
    log2 = sum(math.log2(m + 1) for m in maxdigit)
    return {"exact": exact, "max_column": max(s), "log2_size": log2, "log256_size": log2 / 8,
            "eps_log2": log2 - math.log2(p), "formula_d_at_eps": 4 + (math.log2(p) - log2) / 8,
            "implied_d0": L - (math.log2(p) - log2) / 8}


def p1024_greedy_family(L: int = 128, delta: int = 105) -> list[int]:
    """A larger exact product family: from all-zero, repeatedly raise the digit bound that adds the most log-size while
    the all-max member's folded column sums stay ≤ 255 and its value below p (`p1024_family` checks the result)."""
    D = [0] * L
    t = [0] * (2 * L - 1)

    def fits(l: int, new: int) -> bool:
        old = D[l]
        dt = {2 * l: new * new - old * old}
        for i in range(L):
            if i != l and D[i]:
                dt[l + i] = dt.get(l + i, 0) + 2 * (new - old) * D[i]
        cols: dict[int, int] = {}
        for m, v in dt.items():
            cols[m % L] = cols.get(m % L, 0) + (v if m < L else delta * v)
        return all(t[c] + (delta * t[c + L] if c + L < 2 * L - 1 else 0) + v <= 255 for c, v in cols.items())

    while True:
        best, gain = None, 0.0
        for l in range(L):
            g = math.log2(D[l] + 2) - math.log2(D[l] + 1)
            if g > gain and fits(l, D[l] + 1):
                best, gain = l, g
        if best is None:
            break
        l, old, new = best, D[best], D[best] + 1
        t[2 * l] += new * new - old * old
        for i in range(L):
            if i != l and D[i]:
                t[l + i] += 2 * (new - old) * D[i]
        D[l] = new
    return D


def greedy_series(Ls: Sequence[int] = range(1, 41)) -> list[dict]:
    """The greedy exact family at each byte field `Field.toy(256, L)`: a Renormalization counterexample to the formula
    wherever `implied_d0` exceeds 4 (no budgeted gate, every member's canonical digits, `formula_d_at_eps` below L)."""
    rows = []
    for L in Ls:
        F = Field.toy(256, L)
        fam = p1024_family(p1024_greedy_family(L, F.delta), F.p)
        rows.append({"L": L, "delta": F.delta, **fam})
    return rows


def p1024_sample(n: int, digit_hi: int, positions: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    return np.array([[rng.randint(0, digit_hi) if l < positions else 0 for l in range(128)] for _ in range(n)], dtype=np.int64)


def p1024_carry_free_fraction(n: int = 2000, digit_hi: int = 3, positions: int = 64, seed: int = 7) -> dict:
    """Monte Carlo: the fraction of a with digits uniform in [0, digit_hi] on `positions` low positions whose folded column
    sums are all ≤ 255 (so y = s with no budgeted gate)."""
    d = p1024_sample(n, digit_hi, positions, seed)
    s = column_sums(d, 128, 105)
    ok = (s <= 255).all(axis=1)
    frac = float(ok.mean())
    log2 = positions * math.log2(digit_hi + 1) + (math.log2(frac) if frac else -math.inf)
    return {"n": n, "digit_hi": digit_hi, "positions": positions, "carry_free": int(ok.sum()), "fraction": frac,
            "log2_size": log2, "implied_d0": log2 / 8}


def p1024_carries(n: int = 4000, seed: int = 11) -> np.ndarray:
    """The cyclic carry vectors of n uniform a ∈ [0, p1024) (exact big-integer squares)."""
    rng = random.Random(seed)
    A = [rng.randrange(P1024) for _ in range(n)]
    d = np.array([list(a.to_bytes(128, "little")) for a in A], dtype=np.int64)
    y = np.array([list((a * a % P1024).to_bytes(128, "little")) for a in A], dtype=np.int64)
    s = column_sums(d, 128, 105)
    c = np.zeros_like(s)
    for i in range(n):
        vs = sum(int(v) << (8 * l) for l, v in enumerate(s[i]))
        K, r = divmod(vs - A[i] * A[i] % P1024, P1024)
        assert r == 0
        prev = 105 * K
        for l in range(128):
            num = int(s[i, l]) + prev - int(y[i, l])
            assert num % 256 == 0
            c[i, l] = num // 256
            prev = int(c[i, l])
    return c


def p1024_guess_costs(c: np.ndarray) -> dict:
    """The bits a guessed carry costs: −log₂ of its mode's frequency (and a Gaussian estimate from its spread, for carries
    whose mode the sample can't resolve), per column, and for the cheapest columns guessed jointly."""
    n = c.shape[0]
    per = []
    for l in range(c.shape[1]):
        _v, cnt = np.unique(c[:, l], return_counts=True)
        sd = float(c[:, l].std())
        per.append({"l": l, "mode_bits": -math.log2(cnt.max() / n), "gauss_bits": math.log2(sd * math.sqrt(2 * math.pi)) if sd else 0.0})
    cheapest = sorted(per, key=lambda r: r["gauss_bits"])
    return {"n": n, "cheapest": cheapest[:6], "median_gauss_bits": float(np.median([r["gauss_bits"] for r in per])),
            "min_gauss_bits": cheapest[0]["gauss_bits"]}


def p1024_top_constraint(n: int = 4000, seed: int = 11) -> dict:
    """The toy search's best single row, carried to p1024: the top carry less the digit sum, `c_127 − Σ a_l = m`, against
    the top carry alone. t_127's linear part is 255·Σ(a_l − μ), so c_127 ≈ Σ a_l + const with only the bilinear part left
    (spread ≈ (2/256)·8·σ²): one budgeted gate saved for the bits this costs. Costs are −log₂ of the density at the
    centre, by a Gaussian fit and by a ±16 window (the sample can't resolve a single value's frequency)."""
    rng = random.Random(seed)
    A = [rng.randrange(P1024) for _ in range(n)]
    d = np.array([list(a.to_bytes(128, "little")) for a in A], dtype=np.int64)
    c = p1024_carries(n, seed)
    out = {}
    for name, v in (("c127", c[:, 127]), ("c127-digitsum", c[:, 127] - d.sum(axis=1))):
        med = np.median(v)
        win = float((np.abs(v - med) <= 16).mean()) / 33
        out[name] = {"std": float(v.std()), "gauss_bits": math.log2(float(v.std()) * math.sqrt(2 * math.pi)),
                     "window_bits": -math.log2(win) if win else math.inf,
                     "corr_digitsum": float(np.corrcoef(c[:, 127], d.sum(axis=1))[0, 1])}
    return out


def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="renorm.json")
    ap.add_argument("--sizes", default="4:3,4:4,4:5,4:6,4:7,4:8,16:2,16:3,16:4,256:1,256:2")
    ap.add_argument("--carry-free-L", type=int, default=10, help="the exact carry-free series at B = 256, L = 1..this")
    ap.add_argument("--greedy-L", type=int, default=40, help="the greedy exact family at B = 256, L = 1..this")
    ap.add_argument("--p1024-samples", type=int, default=4000)
    args = ap.parse_args(argv)
    res: dict = {"toy": [], "carry_free_256": [], "p1024": {}}
    for row in carry_free_series(256, range(1, args.carry_free_L + 1)):
        res["carry_free_256"].append(row)
        print(f"B=256 L={row['L']:2d} δ={row['delta']:>4d} carry-free={row['carry_free']:>9d} "
              f"d0≥{row['implied_d0']:.2f} bytes ({row['implied_d0_bits']:.1f} bits)", flush=True)
    for sz in args.sizes.split(","):
        B, L = map(int, sz.split(":"))
        r = run_field(B, L)
        res["toy"].append(r)
        print(f"B={B:3d} L={L:2d} p={r['p']:>9d} δ={r['delta']:>4d} carry-free={r['carry_free']:>6d} "
              f"max implied d0={r['max_implied_d0']:.2f}", flush=True)
        for row in r["frontier"]:
            print(f"   k={row['k']:2d} r={row['r']:2d} N={row['count']:>8d} log2(1/ε)={row['log2_inv_eps']:6.2f} "
                  f"cost={row['cost_digits']:5.2f} d0≥{row['implied_d0']:5.2f} {row['method']}", flush=True)
    res["greedy_256"] = greedy_series(range(1, args.greedy_L + 1))
    for row in res["greedy_256"]:
        print(f"greedy B=256 L={row['L']:3d} δ={row['delta']:>5d} exact={row['exact']} log2|family|={row['log2_size']:7.2f} "
              f"log2(1/ε)={-row['eps_log2']:7.2f} formula d={row['formula_d_at_eps']:6.2f} d0≥{row['implied_d0']:.2f}",
              flush=True)
    greedy = p1024_greedy_family()
    fam = {"{0,1,2}^63x{0,1}": p1024_family([2] * 63 + [1] + [0] * 64),
           "{0,1}^64": p1024_family([1] * 64 + [0] * 64),
           "greedy": {**p1024_family(greedy), "bounds": greedy}}
    res["p1024"]["families"] = fam
    res["p1024"]["monte_carlo"] = [p1024_carry_free_fraction(digit_hi=h, positions=m)
                                   for h, m in ((2, 64), (3, 64), (3, 62), (4, 48))]
    res["p1024"]["guess"] = p1024_guess_costs(p1024_carries(args.p1024_samples))
    res["p1024"]["top_constraint"] = p1024_top_constraint(args.p1024_samples)
    print(json.dumps(res["p1024"], indent=1, default=float))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=float)


if __name__ == "__main__":
    main()

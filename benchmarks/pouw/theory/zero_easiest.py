"""ZERO-EASIEST's exact check: how the cost of Dither's transcript depends on the instance (A, B) fixed before the sample.

Dither: A is m×k, B is k×n, R ∈ {0,1}^{m×k} and U ∈ {±1..±64}^{n×L} are
sampled after, S_{tj} = U_{j, t mod L} (S = ΠUᵀ, U shared by every segment), and the transcript is every L-step prefix
sum T_{A,B}[i, j, ℓ] = Σ_{t < L(ℓ+1)} (A+R)_{it} (B+S)_{tj} in Z/2³². U's bits are planes, U = −64 + Σ_p w_p β_p with
w = (1, 2, 4, 8, 16, 32, 65). X♭ Y is the prefix-summed product, and A♯[i, ℓ, q] = Σ_{ℓ' ≤ ℓ} A[i, Lℓ' + q] the fold.

Parts:
- A. The decomposition T_{A,B} = (AB)♭ + (RB)♭ + A♯Uᵀ + R♯Uᵀ, in s32 at served shapes and over the ring (32-bit A, B).
  The zero instance's transcript is T₀ = R♯Uᵀ, one ⟨mk/L, L, n⟩ product.
- B. Theorem Z. ℓ_{A,B} = T_{A,B} − T₀ is affine in R's bits and U's planes (Möbius, exhaustive). So C ↦ C + ℓ maps a
  circuit for T₀ to one for T_{A,B} with the same × gates, the same per-word product reads and the same error at every
  point. Checked on naive, Strassen and Strassen·(1 − δ) circuits, and backwards from a naive circuit for T_{A,B}.
- C. The antipodal reduction. z̄ = (J − R, −U) is a measure-preserving involution of the sample, and
  T_{A,B}(z) + T_{A,B}(z̄) = 2T₀(z) + ((2A+J)B)♭ − (JS)♭. Two calls give T₀ (recovered from 2T₀ mod 2³²), right
  wherever both calls are, so with probability ≥ 2δ − 1; tight. The one-sided flips give (A,0) and (0,B).
  The mask-sign family g_{f,c} = (R ⊕ 1⊗f̃, c·U∘σ) preserves the law; a combination Σ α_c T(g_c z) leaves the data
  terms R·diag(Σ α_c σ_c)·B and A·diag(Σ α_c c_c σ_c)·S, so one call never cancels them and the only two-call
  combinations that do are antipodal pairs, with an even coefficient on T₀.
- D. A model where zero is not easiest: with zero products skipped, the expected work is Σ Pr[A+R ≠ 0]·Pr[B+S ≠ 0].
  S is never 0 but B+S is, w.p. 1/128, so (0, B) with B nowhere 0 costs 127/128 of the zero instance, the minimum.
- E. Special instances: periodic A (A = WΠᵀ), periodic B (B = ΠVᵀ, the zero instance on the shifted alphabet U + V)
  and low-rank B, with what their correction ℓ costs.
- F. Luck: a word's largest point mass (what a free guess wins), exactly. At L = 1, (0, b ≠ 0) is likelier than zero
  (129/256 against 1/2); at L = 32 no instance tried beats zero.

Run: `uv run --no-project --with numpy --with-editable benchmarks/pouw/exact_kit python -B zero_easiest.py --mutations`
(from this directory, or with the path).
"""
from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np

from exact_kit import cube
from exact_kit.checks import Checks

W_RE = np.array([1, 2, 4, 8, 16, 32, 65], dtype=np.int64)
NP = 7
ALPHABET = np.array([u for u in range(-64, 65) if u != 0], dtype=np.int64)
MASK = np.uint64((1 << 32) - 1)
RNG = np.random.default_rng(20261017)
np.seterr(over="ignore")  # uint64 scalars wrap: that is Z/2^64


def ring(x):
    """Into Z/2^64 (uint64 wraps), which reduces consistently to Z/2^32."""
    if isinstance(x, (int, np.integer)):
        return np.uint64(int(x) % (1 << 64))
    a = np.asarray(x)
    return a if a.dtype == np.uint64 else a.astype(np.int64).astype(np.uint64)


def m32(x):
    return ring(x) & MASK


# ---------------------------------------------------------------- the transcript and its parts (batched over draws)


def pref(X, Y, L):
    """X♭Y: (..., m, k) × (..., k, n) -> (..., m, n, k/L), the prefix sums of the segment products, in Z/2^64."""
    X, Y = ring(X), ring(Y)
    *lead, m, k = X.shape
    n = Y.shape[-1]
    s = k // L
    Xs = np.swapaxes(X.reshape(*lead, m, s, L), -3, -2)  # (..., s, m, L)
    Ys = Y.reshape(*Y.shape[:-2], s, L, n)  # (..., s, L, n)
    inc = np.matmul(Xs, Ys)  # (..., s, m, n)
    return np.moveaxis(np.cumsum(inc, axis=-3, dtype=np.uint64), -3, -1)


def spread(U, k):
    """S = ΠUᵀ: S[t, j] = U[j, t mod L]."""
    L = U.shape[-1]
    return np.swapaxes(U, -1, -2)[..., np.arange(k) % L, :]


def fold(A, L):
    """A♯[..., i, ℓ, q] = Σ_{ℓ' ≤ ℓ} A[..., i, Lℓ' + q]."""
    A = ring(A)
    *lead, m, k = A.shape
    return np.cumsum(A.reshape(*lead, m, k // L, L), axis=-2, dtype=np.uint64)


def fold_times(Fd, U):
    """F♯Uᵀ: (..., m, s, L) and (..., n, L) -> (..., m, n, s)."""
    return np.einsum("...isq,...jq->...ijs", ring(Fd), ring(U))


def transcript(A, B, R, U):
    L = U.shape[-1]
    k = A.shape[-1]
    return m32(pref(ring(A) + ring(R), ring(B) + ring(spread(U, k)), L))


def t_zero(R, U):
    return m32(fold_times(fold(R, U.shape[-1]), U))


def ell(A, B, R, U):
    """ℓ_{A,B} = (AB)♭ + (RB)♭ + A♯Uᵀ: the affine correction."""
    L = U.shape[-1]
    return m32(pref(A, B, L) + pref(R, B, L) + fold_times(fold(A, L), U))


def instance(m, k, n, rng=RNG):
    return rng.integers(-128, 127, size=(m, k)), rng.integers(-63, 64, size=(k, n))


def draw(m, k, n, L, N=None, rng=RNG):
    sh = () if N is None else (N,)
    return rng.integers(0, 2, size=sh + (m, k)), rng.choice(ALPHABET, size=sh + (n, L))


# ---------------------------------------------------------------- bits: R's coins and U's planes


class Bits:
    """Bit layout: R(i, t) at i·k + t, then U's plane p of entry (j, q) at mk + (j·L + q)·7 + p."""

    def __init__(self, m, k, n, L):
        self.m, self.k, self.n, self.L = m, k, n, L
        self.nR = m * k
        self.V = m * k + n * L * NP

    def r(self, i, t):
        return i * self.k + t

    def u(self, j, q, p):
        return self.nR + (j * self.L + q) * NP + p

    def decode(self, z):
        z = np.asarray(z, dtype=np.int64)
        N = z.shape[0]
        R = z[:, : self.nR].reshape(N, self.m, self.k)
        planes = z[:, self.nR :].reshape(N, self.n, self.L, NP)
        return R, -64 + planes @ W_RE


def cube_points(V):
    idx = np.arange(1 << V, dtype=np.int64)
    return ((idx[:, None] >> np.arange(V)) & 1).astype(np.uint8)


# ---------------------------------------------------------------- straight-line programs over Z/2^32 on the bits


class Slp:
    """Gates: ('lin', row over the bits, constant), ('comb', constant, [(coef, gate)]) and ('mul', g, h). Every value is
    tracked symbolically as (product reads, affine part): a dict product-gate -> coefficient, and a bit row plus a
    constant. `muls` counts the × gates."""

    def __init__(self, V):
        self.V = V
        self.gates = []
        self.reads = []
        self.aff = []

    def lin(self, coef: dict, const=0):
        row = np.zeros(self.V, dtype=np.uint64)
        for v, c in coef.items():
            row[v] = row[v] + ring(c)
        self.gates.append(("lin", row, ring(const)))
        self.reads.append({})
        self.aff.append((row, ring(const)))
        return len(self.gates) - 1

    def comb(self, const, terms):
        terms = [(ring(c), g) for c, g in terms]
        self.gates.append(("comb", ring(const), terms))
        rd: dict = {}
        row = np.zeros(self.V, dtype=np.uint64)
        cst = ring(const)
        for c, g in terms:
            for p, cp in self.reads[g].items():
                rd[p] = (rd.get(p, np.uint64(0)) + c * cp) & MASK
            row = row + c * self.aff[g][0]
            cst = cst + c * self.aff[g][1]
        self.reads.append({p: c for p, c in rd.items() if c})
        self.aff.append((row & MASK, cst & MASK))
        return len(self.gates) - 1

    def mul(self, g, h):
        self.gates.append(("mul", g, h))
        self.reads.append({len(self.gates) - 1: np.uint64(1)})
        self.aff.append((np.zeros(self.V, dtype=np.uint64), np.uint64(0)))
        return len(self.gates) - 1

    @property
    def muls(self):
        return sum(1 for g in self.gates if g[0] == "mul")

    def evaluate(self, z, wanted):
        z = np.asarray(z, dtype=np.uint64)
        vals = []
        for g in self.gates:
            if g[0] == "lin":
                vals.append(g[2] + z @ g[1])
            elif g[0] == "comb":
                acc = np.full(z.shape[0], g[1], dtype=np.uint64)
                for c, h in g[2]:
                    acc = acc + c * vals[h]
                vals.append(acc)
            else:
                vals.append(vals[g[1]] * vals[g[2]])
        return np.stack([vals[w] & MASK for w in wanted], axis=1)


def word_order(m, n, s):
    return [(i, j, l) for i in range(m) for j in range(n) for l in range(s)]


def operands_zero(P: Slp, bits: Bits):
    """T₀'s operands: X[(i, ℓ)][q] = R♯[i, ℓ, q] and Y[q][j] = U[j, q], as lin gates."""
    m, k, n, L = bits.m, bits.k, bits.n, bits.L
    s = k // L
    X = [[P.lin({bits.r(i, L * l2 + q): 1 for l2 in range(l + 1)}) for q in range(L)] for i in range(m) for l in range(s)]
    Y = [[P.lin({bits.u(j, q, p): int(W_RE[p]) for p in range(NP)}, -64) for j in range(n)] for q in range(L)]
    return X, Y


def matmul_naive(P, X, Y):
    N, K, M = len(X), len(Y), len(Y[0])
    return [[P.comb(0, [(1, P.mul(X[r][q], Y[q][c])) for q in range(K)]) for c in range(M)] for r in range(N)]


def blocks(Mx):
    h, w = len(Mx) // 2, len(Mx[0]) // 2
    return ([row[:w] for row in Mx[:h]], [row[w:] for row in Mx[:h]], [row[:w] for row in Mx[h:]],
            [row[w:] for row in Mx[h:]])


def addm(P, X, Y, sy=1):
    return [[P.comb(0, [(1, a), (sy, b)]) for a, b in zip(ra, rb)] for ra, rb in zip(X, Y)]


def matmul_strassen(P, X, Y, delta=None):
    """One level of Strassen on 2×2 blocks, each block product naive: 7/8 of the × gates. With `delta`, M1's right
    operand gets δ added at its (0, 0) entry, which makes C11 and C22 wrong exactly where δ = 1 (RS-E-REF VIII)."""
    X11, X12, X21, X22 = blocks(X)
    Y11, Y12, Y21, Y22 = blocks(Y)
    S1 = addm(P, Y11, Y22)
    if delta is not None:
        S1[0][0] = P.comb(0, [(1, S1[0][0]), (1, delta)])
    M1 = matmul_naive(P, addm(P, X11, X22), S1)
    M2 = matmul_naive(P, addm(P, X21, X22), Y11)
    M3 = matmul_naive(P, X11, addm(P, Y12, Y22, -1))
    M4 = matmul_naive(P, X22, addm(P, Y21, Y11, -1))
    M5 = matmul_naive(P, addm(P, X11, X12), Y22)
    M6 = matmul_naive(P, addm(P, X21, X11, -1), addm(P, Y11, Y12))
    M7 = matmul_naive(P, addm(P, X12, X22, -1), addm(P, Y21, Y22))
    h, w = len(M1), len(M1[0])
    C = [[None] * (2 * w) for _ in range(2 * h)]
    for r in range(h):
        for c in range(w):
            C[r][c] = P.comb(0, [(1, M1[r][c]), (1, M4[r][c]), (-1, M5[r][c]), (1, M7[r][c])])
            C[r][c + w] = P.comb(0, [(1, M3[r][c]), (1, M5[r][c])])
            C[r + h][c] = P.comb(0, [(1, M2[r][c]), (1, M4[r][c])])
            C[r + h][c + w] = P.comb(0, [(1, M1[r][c]), (-1, M2[r][c]), (1, M3[r][c]), (1, M6[r][c])])
    return C


def circuit_zero(bits: Bits, kind: str, live_u=None):
    """A circuit for T₀ = R♯Uᵀ: 'naive', 'strassen' or 'strassen-delta' (δ = ∏(1 − z) over the U bits in `live_u`)."""
    P = Slp(bits.V)
    X, Y = operands_zero(P, bits)
    if kind == "naive":
        C = matmul_naive(P, X, Y)
    else:
        delta = None
        if kind == "strassen-delta":
            vs = list(live_u)
            delta = P.lin({vs[0]: -1}, 1)
            for v in vs[1:]:
                delta = P.mul(delta, P.lin({v: -1}, 1))
        C = matmul_strassen(P, X, Y, delta)
    s = bits.k // bits.L
    words = {(i, j, l): C[i * s + l][j] for i in range(bits.m) for j in range(bits.n) for l in range(s)}
    return P, words


def ell_lin(P: Slp, bits: Bits, A, B, sign=1):
    """ℓ_{A,B} as one lin gate a word, its coefficients read off A and B (not from T): constant (AB)♭ − 64·ΣA♯,
    R-bit (i, t) coefficient B[t, j]·[t < L(ℓ+1)], U-plane (j, q, p) coefficient w_p·A♯[i, ℓ, q]."""
    m, k, n, L = bits.m, bits.k, bits.n, bits.L
    s = k // L
    AB = np.einsum("isq,sqj->ijs", np.asarray(A, dtype=np.int64).reshape(m, s, L),
                   np.asarray(B, dtype=np.int64).reshape(s, L, n))
    AB = np.cumsum(AB, axis=-1)  # (m, n, s), exact over Z
    Ad = np.cumsum(np.asarray(A, dtype=np.int64).reshape(m, s, L), axis=1)
    out = {}
    for i in range(m):
        for j in range(n):
            for l in range(s):
                coef = {bits.r(i, t): int(B[t, j]) for t in range(L * (l + 1))}
                for q in range(L):
                    for p in range(NP):
                        coef[bits.u(j, q, p)] = int(Ad[i, l, q]) * int(W_RE[p])
                const = int(AB[i, j, l]) - 64 * int(Ad[i, l, :].sum())
                out[(i, j, l)] = P.lin({v: sign * c for v, c in coef.items()}, sign * const)
    return out


def shift(P: Slp, bits: Bits, words, A, B, sign=1):
    """C ↦ C + sign·ℓ_{A,B}: one free comb a word, every × gate kept."""
    L = ell_lin(P, bits, A, B, sign)
    return {w: P.comb(0, [(1, g), (1, L[w])]) for w, g in words.items()}


def circuit_naive_instance(bits: Bits, A, B):
    """The honest form for T_{A,B}: products (A + R)_{it}(B + S)_{tj}, prefix-summed: mnk × gates of mixed affine
    operands (constants on both sides)."""
    m, k, n, L = bits.m, bits.k, bits.n, bits.L
    s = k // L
    P = Slp(bits.V)
    Xa = [[P.lin({bits.r(i, t): 1}, int(A[i, t])) for t in range(k)] for i in range(m)]
    Yb = [[P.lin({bits.u(j, t % L, p): int(W_RE[p]) for p in range(NP)}, int(B[t, j]) - 64) for j in range(n)]
          for t in range(k)]
    words = {}
    for i in range(m):
        for j in range(n):
            acc = []
            for l in range(s):
                acc += [(1, P.mul(Xa[i][t], Yb[t][j])) for t in range(L * l, L * (l + 1))]
                words[(i, j, l)] = P.comb(0, list(acc))
    return P, words


# ---------------------------------------------------------------- the parts


def part_a(ch: Checks, mut=None):
    shapes = [(3, 8, 2, 2), (8, 128, 8, 32), (16, 512, 16, 32), (64, 4096, 64, 32)]
    for (m, k, n, L) in shapes:
        A, B = instance(m, k, n)
        R, U = draw(m, k, n, L)
        S = spread(U, k)
        T = transcript(A, B, R, U)
        Ad = fold(A, L)
        if mut == "no-fold":
            Ad = ring(A).reshape(m, k // L, L)
        parts = m32(pref(A, B, L) + pref(R, B, L) + fold_times(Ad, U) + fold_times(fold(R, L), U))
        ok = np.array_equal(T, parts)
        ok_fold = np.array_equal(m32(pref(A, S, L)), m32(fold_times(Ad, U)))
        ok_zero = np.array_equal(t_zero(R, U), transcript(np.zeros_like(A), np.zeros_like(B), R, U))
        ch.check(f"A decomposition (m,k,n,L)=({m},{k},{n},{L})", ok and ok_fold and ok_zero,
                 f"T = (AB)♭+(RB)♭+A♯Uᵀ+R♯Uᵀ on {T.size} words; (AS)♭ = A♯Uᵀ; T₀ = R♯Uᵀ")
    m, k, n, L = 4, 64, 4, 32
    A = RNG.integers(0, 1 << 32, size=(m, k), dtype=np.uint64)
    B = RNG.integers(0, 1 << 32, size=(k, n), dtype=np.uint64)
    R, U = draw(m, k, n, L)
    Ad = fold(A, L) if mut != "no-fold" else ring(A).reshape(m, k // L, L)
    parts = m32(pref(A, B, L) + pref(R, B, L) + fold_times(Ad, U) + fold_times(fold(R, L), U))
    ch.check("A decomposition over the ring (32-bit A, B, wrap)", np.array_equal(transcript(A, B, R, U), parts),
             "an identity of Z/2³²")


def part_b_affine(ch: Checks, mut=None):
    """ℓ = T_{A,B} − T₀ has Möbius degree ≤ 1 in Z/2^32 on the whole cube, while T₀ has the target's degree-2 part."""
    for (m, k, n, L) in [(1, 4, 1, 2), (2, 2, 2, 1)]:
        bits = Bits(m, k, n, L)
        z = cube_points(bits.V)
        R, U = bits.decode(z)
        for trial in range(2):
            A, B = instance(m, k, n)
            T = transcript(A, B, R, U)
            T0 = t_zero(R, U)
            D = (T - T0) & MASK
            if mut == "bilinear-ell":
                D = (D + T0) & MASK
            N = z.shape[0]
            Dw = D.reshape(N, -1)
            T0w = T0.reshape(N, -1)
            deg = cube.degree(bits.V)
            hi = 0
            for w in range(Dw.shape[1]):
                c = cube.mobius(Dw[:, w].astype(np.uint32), bits.V, bits=32)
                hi += int(np.count_nonzero(c[deg >= 2]))
            formula = ell(A, B, R, U).reshape(N, -1)
            # T0's degree-2 part for word (i,j,l): w_p at (R(i,t), U(j, t mod L, p)) for t < L(l+1), 0 elsewhere
            bad2 = 0
            s = k // L
            for wi, (i, j, l) in enumerate(word_order(m, n, s)):
                c = cube.mobius(T0w[:, wi].astype(np.uint32), bits.V, bits=32)
                want = np.zeros(1 << bits.V, dtype=np.uint32)
                for t in range(L * (l + 1)):
                    for p in range(NP):
                        want[(1 << bits.r(i, t)) | (1 << bits.u(j, t % L, p))] = W_RE[p]
                bad2 += int(np.count_nonzero(c[deg == 2] != want[deg == 2]))
                bad2 += int(np.count_nonzero(c[deg >= 3]))
            ch.check(f"B1 ℓ affine on the {bits.V}-cube (m,k,n,L)=({m},{k},{n},{L}) draw {trial}",
                     hi == 0 and np.array_equal(Dw, formula), f"{hi} coefficients of degree ≥ 2 in ℓ over "
                     f"{Dw.shape[1]} words and {N} points; ℓ = (AB)♭+(RB)♭+A♯Uᵀ at every point")
            ch.check(f"B1 T₀'s cross part is the target's (m,k,n,L)=({m},{k},{n},{L}) draw {trial}", bad2 == 0,
                     "every (R bit, U plane) coefficient w_p·[t mod L = q]·[t < L(ℓ+1)], nothing of degree ≥ 3")


def part_b_affine_served(ch: Checks):
    """At a served shape: second differences of ℓ vanish on random (base, pair), including the pairs where T₀'s don't."""
    m, k, n, L = 8, 256, 8, 32
    bits = Bits(m, k, n, L)
    A, B = instance(m, k, n)
    nz_ell = 0
    nz_t0 = 0
    tried = 0
    for _ in range(12):
        base = RNG.integers(0, 2, size=bits.V).astype(np.uint8)
        pairs = []
        for _ in range(24):
            i, t = RNG.integers(m), RNG.integers(k)
            j, p = RNG.integers(n), RNG.integers(NP)
            pairs.append((bits.r(i, t), bits.u(j, t % L, p)))  # where T₀ has a cross term
            pairs.append(tuple(RNG.choice(bits.V, size=2, replace=False)))
        pts = []
        for u, v in pairs:
            for a in (0, 1):
                for b in (0, 1):
                    zz = base.copy()
                    zz[u], zz[v] = a, b
                    pts.append(zz)
        R, U = bits.decode(np.stack(pts))
        T = transcript(A, B, R, U).astype(np.int64)
        T0 = t_zero(R, U).astype(np.int64)
        for x, ctr in ((T - T0, "ell"), (T0, "t0")):
            x = x.reshape(len(pairs), 4, -1)
            dd = (x[:, 3] - x[:, 2] - x[:, 1] + x[:, 0]) % (1 << 32)
            if ctr == "ell":
                nz_ell += int(np.count_nonzero(dd))
            else:
                nz_t0 += int(np.count_nonzero(dd.any(axis=1)))
        tried += len(pairs)
    ch.check("B1 ℓ's second differences vanish at a served shape (8, 256, 8, 32)", nz_ell == 0 and nz_t0 >= tried // 2,
             f"{tried} (base, pair) tries × {m * n * (k // L)} words: ℓ nonzero {nz_ell}; T₀ nonzero at {nz_t0} pairs")


def part_b_transform(ch: Checks, mut=None):
    """Theorem Z on circuits: C ↦ C + ℓ keeps every × gate and every product read, and moves no error."""
    m, k, n, L = 1, 4, 2, 2  # T₀ = R♯Uᵀ is ⟨2, 2, 2⟩: Strassen applies
    bits = Bits(m, k, n, L)
    live_u = [bits.u(j, q, p) for j in range(n) for q in range(L) for p in (0, 6)]
    live = list(range(bits.nR)) + live_u
    fixed = RNG.integers(0, 2, size=bits.V).astype(np.uint8)
    sub = np.repeat(fixed[None], 1 << len(live), axis=0)
    sub[:, live] = cube_points(len(live))
    full = RNG.integers(0, 2, size=(20000, bits.V)).astype(np.uint8)
    special = np.repeat(fixed[None], 16, axis=0)
    special[:, bits.nR :] = 0  # U = −64 everywhere: where δ over all U bits is 1
    special[:, : bits.nR] = cube_points(bits.nR)
    pts_sets = {"subcube": sub, "random": full, "U=−64": special}
    s = k // L
    wanted = word_order(m, n, s)
    for kind in ("naive", "strassen", "strassen-delta"):
        for trial in range(2):
            A, B = instance(m, k, n)
            P, words = circuit_zero(bits, kind, live_u)
            mul0 = P.muls
            reads0 = {w: dict(P.reads[g]) for w, g in words.items()}
            words2 = shift(P, bits, words, A, B, sign=1)
            if mut == "ell-off-by-one":
                B2 = B.copy()
                B2[-1] += 1
                words2 = shift(P, bits, words, A, B2, sign=1)
            mul1 = P.muls
            same_reads = all(P.reads[words2[w]] == reads0[w] for w in wanted)
            moved = 0
            errs = 0
            npts = 0
            for name, pts in pts_sets.items():
                R, U = bits.decode(pts)
                o0 = P.evaluate(pts, [words[w] for w in wanted])
                o1 = P.evaluate(pts, [words2[w] for w in wanted])
                T0 = t_zero(R, U).reshape(len(pts), -1)
                T1 = transcript(A, B, R, U).reshape(len(pts), -1)
                e0 = (o0 - T0) & MASK
                e1 = (o1 - T1) & MASK
                moved += int(np.count_nonzero(e0 != e1))
                errs += int(np.count_nonzero(e0))
                npts += len(pts)
            ch.check(f"B2 Theorem Z on {kind}, instance {trial}", mul1 == mul0 and same_reads and moved == 0,
                     f"× gates {mul0} → {mul1}; product reads equal in every word; errors moved {moved} of "
                     f"{npts * len(wanted)} (point, word); errors present {errs}")
    # the δ circuit really is wrong, and only at δ = 1
    P, words = circuit_zero(bits, "strassen-delta", live_u)
    R, U = bits.decode(sub)
    e = (P.evaluate(sub, [words[w] for w in wanted]) - t_zero(R, U).reshape(len(sub), -1)) & MASK
    delta_pts = np.all(sub[:, live_u] == 0, axis=1)
    ch.check("B2 Strassen·(1−δ) is wrong only where δ = 1 (subcube)", np.count_nonzero(e[~delta_pts]) == 0
             and np.count_nonzero(e[delta_pts]) > 0, f"{np.count_nonzero(e.any(axis=1))} wrong points of {len(sub)}, "
             f"all at δ = 1; × gates {P.muls} (naive 8, Strassen 7, δ chain {len(live_u) - 1})")
    # backwards: the honest-form circuit for T_{A,B}, minus ℓ, is a circuit for T₀ with the same mnk products
    for trial in range(2):
        A, B = instance(m, k, n)
        P, words = circuit_naive_instance(bits, A, B)
        mul0 = P.muls
        back = shift(P, bits, words, A, B, sign=-1)
        bad = 0
        for pts in (sub, full):
            R, U = bits.decode(pts)
            o = P.evaluate(pts, [back[w] for w in wanted])
            o_inst = P.evaluate(pts, [words[w] for w in wanted])
            bad += int(np.count_nonzero((o - t_zero(R, U).reshape(len(pts), -1)) & MASK))
            bad += int(np.count_nonzero((o_inst - transcript(A, B, R, U).reshape(len(pts), -1)) & MASK))
        ch.check(f"B2 backwards: honest form for T_(A,B) minus ℓ computes T₀, instance {trial}",
                 bad == 0 and P.muls == mul0 == m * k * n, f"{mul0} × gates both ways; 0 errors on "
                 f"{len(sub) + len(full)} points")


def part_c(ch: Checks, mut=None):
    """The antipodal reduction."""
    alphabet = ALPHABET if mut != "asym-alphabet" else np.arange(1, 129)
    neg_ok = set((-alphabet).tolist()) == set(alphabet.tolist())
    flips_ok = set((1 - np.array([0, 1])).tolist()) == {0, 1}
    planes = cube_points(NP).astype(np.int64)
    letters = -64 + planes @ W_RE
    ch.check("C0 the planes are a bijection onto ±1..±64", sorted(letters.tolist()) == sorted(ALPHABET.tolist()))
    ch.check("C0 z̄ = (J − R, −U) maps the draw's domain onto itself (an involution, so it preserves the law)",
             neg_ok and flips_ok, f"−alphabet = alphabet: {neg_ok}; 1 − {{0,1}} = {{0,1}}: {flips_ok}")
    shapes = [(3, 8, 2, 2), (8, 64, 8, 32), (64, 4096, 64, 32), (32, 8192, 32, 32)]
    for (m, k, n, L) in shapes:
        A, B = instance(m, k, n)
        R, U = draw(m, k, n, L)
        Rb, Ub = 1 - R, -U
        if mut == "flip-R-only":
            Ub = U
        T = transcript(A, B, R, U)
        Tb = transcript(A, B, Rb, Ub)
        J = np.ones((m, k), dtype=np.int64)
        K = pref(2 * A + J, B, L)
        sig = pref(J, spread(U, k), L)  # (JS)♭[i, j, ℓ] = (ℓ+1)·Σ_q U[j, q]
        sig_closed = ((np.arange(k // L) + 1)[None, None, :] * U.sum(axis=1)[None, :, None]) * np.ones((m, 1, 1), int)
        two_t0 = m32(T + Tb - K + sig)
        ok_id = np.array_equal(two_t0, m32(2 * t_zero(R, U)))
        half = (two_t0 >> np.uint64(1)).astype(np.int64)
        rec = np.where(half >= (1 << 30), half - (1 << 31), half)
        t0s = t_zero(R, U).astype(np.int64)
        t0s = np.where(t0s >= (1 << 31), t0s - (1 << 32), t0s)
        ok_rec = np.array_equal(rec, t0s) and int(np.abs(t0s).max()) < (1 << 30) and 64 * k < (1 << 30)
        ch.check(f"C1 T(z) + T(z̄) = 2T₀ + ((2A+J)B)♭ − (JS)♭ at (m,k,n,L)=({m},{k},{n},{L})",
                 ok_id and np.array_equal(m32(sig), m32(sig_closed)) and ok_rec,
                 f"{T.size} words; T₀ recovered from 2T₀ mod 2³² (|T₀| ≤ 64k = {64 * k} < 2³⁰)")
    m, k, n, L = 4, 64, 4, 32
    A = RNG.integers(0, 1 << 32, size=(m, k), dtype=np.uint64)
    B = RNG.integers(0, 1 << 32, size=(k, n), dtype=np.uint64)
    R, U = draw(m, k, n, L)
    J = np.ones((m, k), dtype=np.uint64)
    lhs = m32(transcript(A, B, R, U) + transcript(A, B, 1 - R, -U if mut != "flip-R-only" else U))
    rhs = m32(2 * ring(t_zero(R, U)) + pref(ring(2) * A + J, B, L) - pref(J, spread(U, k), L))
    ch.check("C1 the antipodal identity over the ring (32-bit A, B)", np.array_equal(lhs, rhs))
    # one-sided flips: (A,B) → (A,0) by U ↦ −U, (A,B) → (0,B) by R ↦ J − R
    m, k, n, L = 8, 128, 8, 32
    A, B = instance(m, k, n)
    R, U = draw(m, k, n, L)
    J = np.ones((m, k), dtype=np.int64)
    zA = m32(transcript(A, B, R, U) - transcript(A, B, R, -U)) == m32(2 * ring(transcript(A, 0 * B, R, U)))
    zB = m32(transcript(A, B, R, U) - transcript(A, B, 1 - R, U)) == m32(
        2 * ring(transcript(0 * A, B, R, U)) - pref(J, ring(B) + ring(spread(U, k)), L))
    ch.check("C2 one-sided flips: T(R,U) − T(R,−U) = 2T_(A,0); T(R,U) − T(J−R,U) = 2T_(0,B) − (J(B+S))♭",
             bool(zA.all() and zB.all()), "each a two-call reduction to an instance with one data term removed")
    # the reduction as a program, against cheaters with chosen error sets, exhaustively on a small domain
    m, k, n, L = 1, 2, 1, 2
    A, B = instance(m, k, n)
    Rs = cube_points(m * k).reshape(-1, m, k).astype(np.int64)
    Us = np.array(np.meshgrid(*([ALPHABET] * (n * L)), indexing="ij")).reshape(n * L, -1).T.reshape(-1, n, L)
    R = np.repeat(Rs, len(Us), axis=0)
    U = np.tile(Us, (len(Rs), 1, 1))
    N = len(R)
    T = transcript(A, B, R, U)
    # index of z̄ for every z: R index flips every bit, U index maps each letter to its negative
    ridx = np.repeat(np.arange(len(Rs)), len(Us))
    uidx = np.tile(np.arange(len(Us)), len(Rs))
    let = {int(v): a for a, v in enumerate(ALPHABET)}
    neg = np.array([let[-int(v)] for v in ALPHABET])
    ucoords = np.array(np.unravel_index(uidx, [len(ALPHABET)] * (n * L)))
    ubar = np.ravel_multi_index(tuple(neg[ucoords]), [len(ALPHABET)] * (n * L))
    bar = ((len(Rs) - 1 - ridx) * len(Us) + ubar)
    ok_bar = np.array_equal(R[bar], 1 - R) and np.array_equal(U[bar], -U) and np.array_equal(bar[bar], np.arange(N))
    J = np.ones((m, k), dtype=np.int64)
    K = pref(2 * A + J, B, L)
    sig = pref(np.broadcast_to(J, (N, m, k)), spread(U, k), L)
    t0 = t_zero(R, U)

    def reduce_with(P):
        two = m32(ring(P) + ring(P[bar]) - K + sig)
        half = (two >> np.uint64(1)).astype(np.int64)
        return m32(np.where(half >= (1 << 30), half - (1 << 31), half))

    rows = []
    for name, wrong in (("random 10%", RNG.random(N) < 0.10), ("random 40%", RNG.random(N) < 0.40),
                        ("wrong iff R₀₀ = 1", R[:, 0, 0] == 1)):
        P = T.copy()
        P[wrong] = m32(ring(P[wrong]) + ring(2 * RNG.integers(1, 1 << 19, size=P[wrong].shape)))
        Z = reduce_with(P)
        zwrong = np.any((Z != t0).reshape(N, -1), axis=1)
        union = wrong | wrong[bar]
        delta = 1 - wrong.mean()
        rows.append((name, delta, 1 - zwrong.mean(), np.all(zwrong <= union), 2 * delta - 1))
    ok = ok_bar and all(r[3] and r[2] >= r[4] - 1e-12 for r in rows) and rows[2][2] == 0.0
    detail = "; ".join(f"{r[0]}: δ = {r[1]:.4f}, reduction right w.p. {r[2]:.4f} ≥ 2δ−1 = {r[4]:.4f}" for r in rows)
    ch.check(f"C3 the reduction is right wherever both calls are, exhaustively over {N} draws", ok, detail)


def g_image(R, U, f, c):
    """g_{f,c}(R, U) = (R ⊕ (1 ⊗ f̃), c·U∘σ), σ = 1 − 2f, f ∈ {0,1}^L read periodically along t."""
    L = U.shape[-1]
    ft = f[np.arange(R.shape[-1]) % L]
    return np.where(ft == 1, 1 - R, R), c * U * (1 - 2 * f)


def part_c_masks(ch: Checks, mut=None):
    """The mask-sign family: what each image costs, and that only the antipodal pair cancels both data terms."""
    m, k, n, L = 8, 256, 8, 32
    s = k // L
    A, B = instance(m, k, n)
    R, U = draw(m, k, n, L)
    T0 = ring(t_zero(R, U))
    ok = True
    for trial in range(4):
        f = RNG.integers(0, 2, size=L)
        sig_t = (1 - 2 * f)[np.arange(k) % L]
        for c in (1, -1):
            Rg, Ug = g_image(R, U, f, c)
            corr = ring(-c * (np.arange(s) + 1)[None, None, :] * U[:, f == 1].sum(axis=1)[None, :, None]
                        * np.ones((m, 1, 1), dtype=np.int64))
            M = np.broadcast_to(f[np.arange(k) % L], (m, k))
            rhs = (pref(A + M, B, L) + ring(c) * fold_times(fold(A * sig_t, L), U) + pref(R * sig_t, B, L)
                   + ring(c) * T0 + corr)
            ok &= np.array_equal(t_zero(Rg, Ug), m32(ring(c) * T0 + corr))
            ok &= np.array_equal(transcript(A, B, Rg, Ug), m32(rhs))
    ch.check("C4 g_(f,c) = (R ⊕ 1⊗f̃, c·U∘σ): T₀(gz) = c·T₀ + row-constant term; T_(A,B)(gz) = ((A+M)B)♭ "
             "+ c·(A∘σ̃)♯Uᵀ + (R·diag σ̃·B)♭ + c·T₀ + row-constant term", bool(ok),
             "8 images at (8,256,8,32); the row-constant term is −c(ℓ+1)Σ_{f_q=1}U[j,q], O(nL) work")
    # Σ_c α_c T(g_c z) has data terms R·diag(D_R)·B and A·diag(D_U)·S, D_R = Σ α_c σ_c, D_U = Σ α_c c_c σ_c, and
    # T₀'s coefficient Σ α_c c_c. Enumerate every two-call combination over Z/2^w at small L.
    w, Ls = 8, 3
    mod = 1 << w
    sigs = np.array(np.meshgrid(*([[1, -1]] * Ls), indexing="ij")).reshape(Ls, -1).T  # (8, L)
    gs = [(sg, c) for sg in sigs for c in (1, -1)]
    al = np.arange(mod)
    a1, a2 = np.meshgrid(al, al, indexing="ij")
    a1, a2 = a1.ravel(), a2.ravel()
    found, one_call = [], True
    for (s1, c1) in gs:
        one_call &= not np.all((al[1:, None] * s1[None, :]) % mod == 0, axis=1).any()
        for (s2, c2) in gs:
            DR = (a1[:, None] * s1 + a2[:, None] * s2) % mod
            DU = (a1[:, None] * c1 * s1 + a2[:, None] * c2 * s2) % mod
            coef = (a1 * c1 + a2 * c2) % mod
            hit = np.all(DR == 0, axis=1) & np.all(DU == 0, axis=1) & (coef != 0)
            if mut == "two-call-any":
                hit = np.all(DR == 0, axis=1) & (coef != 0)
            for i in np.flatnonzero(hit):
                found.append((tuple(s1), c1, tuple(s2), c2, int(a1[i]), int(a2[i]), int(coef[i])))
    anti = all(s2 == tuple(-np.array(s1)) and c2 == c1 and x1 == x2 and co % 2 == 0
               for (s1, c1, s2, c2, x1, x2, co) in found)
    ch.check(f"C5 over Z/2^{w}, L = {Ls}: no one-call combination cancels the data terms; every two-call one is the "
             "antipodal pair (σ₂ = −σ₁, c₂ = c₁, α₂ = α₁), and T₀'s coefficient is even",
             bool(one_call and found and anti), f"{len(found)} cancelling two-call combinations of {len(gs) ** 2 * mod ** 2}")


def part_d(ch: Checks, mut=None):
    """Zero-skipping: expected nonzero products, exactly."""
    alphabet = ALPHABET if mut != "zero-in-alphabet" else np.arange(-64, 64)
    al = set(alphabet.tolist())

    def p_left(a):
        return Fraction(1, 2) if a in (0, -1) else Fraction(1)

    def p_right(b):
        return Fraction(1) - Fraction(int(-b in al), len(alphabet))

    lefts = {a: p_left(a) for a in range(-128, 127)}
    rights = {b: p_right(b) for b in range(-63, 64)}
    pl, pr = min(lefts.values()), min(rights.values())
    zero = lefts[0] * rights[0]
    best = pl * pr
    argl = sorted(a for a, v in lefts.items() if v == pl)
    argr = [b for b, v in rights.items() if v == pr]
    ch.check("D1 zero-skipping: the zero instance is not the easiest; (A ∈ {0,−1}, B nowhere 0) is, at 127/128",
             best < zero and best / zero == Fraction(127, 128) and argl == [-1, 0] and 0 not in argr
             and len(argr) == 126, f"per (i,t,j): zero {zero}, minimum {best} = {best / zero}·zero, at A ∈ {argl} and "
             f"B ∈ [−63,63]∖{{0}} ({len(argr)} values)")
    # exhaustive at a small shape: the skipping program's work over every sample
    m, k, n, L = 1, 2, 1, 2
    Rs = cube_points(m * k).reshape(-1, m, k).astype(np.int64)
    Us = np.array(np.meshgrid(*([alphabet] * (n * L)), indexing="ij")).reshape(n * L, -1).T.reshape(-1, n, L)
    R = np.repeat(Rs, len(Us), axis=0)
    U = np.tile(Us, (len(Rs), 1, 1))
    res = {}
    for name, (A, B) in {"zero": (np.zeros((m, k), int), np.zeros((k, n), int)),
                         "(0, J)": (np.zeros((m, k), int), np.ones((k, n), int)),
                         "(0, ΠVᵀ), V = (5, −7)": (np.zeros((m, k), int), spread(np.array([[5, -7]]), k))}.items():
        X = A[None] + R
        Y = B[None] + spread(U, k)
        prod = X[:, :, :, None] * Y[:, None, :, :]  # (N, m, k, n)
        work = int(np.count_nonzero(prod))
        sk = np.where(prod != 0, prod, 0).cumsum(axis=2)[:, :, L - 1 :: L, :]
        right = np.array_equal(m32(np.moveaxis(sk, 2, 3)), transcript(A, B, R, U))
        res[name] = (Fraction(work, len(R) * m * k * n), right)
    z = res["zero"][0]
    ok = all(v[1] for v in res.values()) and all(v[0] / z == Fraction(127, 128) for kname, v in res.items()
                                                  if kname != "zero")
    ch.check(f"D2 exhaustively over {len(R)} draws at (1,2,1,2): skipped work per product",
             ok, "; ".join(f"{kname} {v[0]}" for kname, v in res.items()) + "; every skipping run right")


def part_e(ch: Checks):
    """Special instances and what their correction ℓ costs."""
    m, k, n, L = 8, 256, 8, 32
    s = k // L
    R, U = draw(m, k, n, L)
    W = RNG.integers(-128, 127, size=(m, L))
    A = W[:, np.arange(k) % L]  # periodic A = WΠᵀ
    lhs = m32(fold_times(fold(A, L), U))
    rhs = m32((np.arange(s) + 1)[None, None, :] * (W @ U.T)[:, :, None])
    ch.check("E1 periodic A = WΠᵀ: A♯Uᵀ = (ℓ+1)·WUᵀ", np.array_equal(lhs, rhs),
             f"the A-term is one ⟨m, L, n⟩ product (L/k = {L / k:.4f} of the GEMM) and a scale a word")
    A2, _ = instance(m, k, n)
    V = RNG.integers(-63, 64, size=(n, L))
    B = spread(V, k)  # periodic B = ΠVᵀ
    lhs = transcript(A2, B, R, U)
    rhs = m32(pref(A2, B, L) + fold_times(fold(A2, L), U) + fold_times(fold(R, L), U + V))
    ch.check("E2 periodic B = ΠVᵀ: T_(A,ΠVᵀ)(R,U) = (AB)♭ + A♯Uᵀ + R♯(U+V)ᵀ", np.array_equal(lhs, rhs),
             "with A = 0 it is the zero instance on the shifted alphabet U + V, which holds 0 w.p. 1/128 (D)")
    r = 2
    X = RNG.integers(-8, 8, size=(k, r))
    Y = RNG.integers(-8, 8, size=(r, n))
    Bl = X @ Y
    lhs = m32(pref(R, Bl, L))
    RX = np.cumsum(np.einsum("isq,sqr->isr", ring(R).reshape(m, s, L), ring(X).reshape(s, L, r)), axis=1,
                   dtype=np.uint64)  # (m, s, r)
    rhs = m32(np.einsum("isr,rj->ijs", RX, ring(Y)))
    ch.check(f"E3 rank-{r} B = XY: (RB)♭ = (RX)♭·Y", np.array_equal(lhs, rhs),
             f"the B-term is mkr + (mk/L)·r·n MACs: r/L = {r / L:.4f} of the GEMM")


SLOT = 33  # bytes a coefficient: every count is ≤ 256^32 = 2^256


def poly_mul(p, q):
    """Exact product of two count polynomials (lists of ints) by Kronecker substitution."""
    enc = [int.from_bytes(b"".join(c.to_bytes(SLOT, "little") for c in x), "little") for x in (p, q)]
    n = len(p) + len(q) - 1
    raw = (enc[0] * enc[1]).to_bytes(n * SLOT, "little")
    return [int.from_bytes(raw[i * SLOT:(i + 1) * SLOT], "little") for i in range(n)]


def term_poly(a, b, alphabet):
    """(lo, counts) of X = (a + r)(b + u), r ∈ {0,1}, u ∈ alphabet: counts out of 2·|alphabet|."""
    vals = [(a + r) * (b + int(u)) for r in (0, 1) for u in alphabet]
    lo, hi = min(vals), max(vals)
    cnt = [0] * (hi - lo + 1)
    for v in vals:
        cnt[v - lo] += 1
    return lo, cnt


def word_poly(pairs, alphabet):
    """(lo, counts) of one segment's word Σ_q (a_q + r_q)(b_q + u_q), the q independent."""
    lo, acc = 0, [1]
    for (a, b) in pairs:
        l2, c2 = term_poly(a, b, alphabet)
        lo, acc = lo + l2, poly_mul(acc, c2)
    return lo, acc


def part_f(ch: Checks, mut=None):
    """Luck: a word's largest point mass, the success of a guess that costs nothing."""
    alphabet = ALPHABET if mut != "zero-in-alphabet" else np.arange(-64, 64)
    _, c0 = term_poly(0, 0, alphabet)
    p0 = Fraction(max(c0), sum(c0))
    pb = [Fraction(max(term_poly(0, b, alphabet)[1]), 2 * len(alphabet)) for b in range(-63, 64) if b != 0]
    hits = sum(1 for r in (0, 1) for u in alphabet if r * int(u) == -r * 5)
    aff = Fraction(hits, 2 * len(alphabet))
    ch.check("F1 L = 1: a free constant guess is right w.p. 1/2 on zero and 129/256 on every (0, b ≠ 0); zero's "
             "affine guess −r·b also gets 129/256 (Theorem Z)", p0 == Fraction(1, 2)
             and all(p == Fraction(129, 256) for p in pb) and aff == Fraction(129, 256),
             "so at L = 1 zero is not the easiest where only constants are free, by 1/256")
    L = 32
    fam = [(a, b) for a in (-2, -1, 0, 1) for b in (-63, -32, -8, -1, 0, 1, 8, 32, 63)]
    rng = np.random.default_rng(32)
    mixed = [list(zip(rng.integers(-2, 2, size=L).tolist(), rng.integers(-63, 64, size=L).tolist()))
             for _ in range(6)]
    mixed += [list(zip([0] * L, rng.integers(-63, 64, size=L).tolist())) for _ in range(3)]
    mixed += [list(zip(rng.integers(-1, 1, size=L).tolist(), [0] * L)) for _ in range(3)]
    tot = (2 * len(alphabet)) ** L
    best = {}
    for (a, b) in fam:
        best[f"({a},{b})"] = max(word_poly([(a, b)] * L, alphabet)[1])
    for i, pr in enumerate(mixed):
        best[f"mixed {i}"] = max(word_poly(pr, alphabet)[1])
    z = best["(0,0)"]
    ties = sorted(kk for kk, v in best.items() if v == z)
    ok = all(v <= z for v in best.values())
    runner = max((kk for kk, v in best.items() if v < z), key=lambda kk: best[kk])
    ch.check("F2 L = 32, one segment's word, exactly: no instance tried has a word likelier to guess than zero's",
             ok, f"zero's largest point mass {float(Fraction(z, tot)):.6f}; ties: {len(ties)} "
             f"(a ∈ {{0,−1}}, b = 0); the next best, {runner}, is {float(Fraction(best[runner], z)):.5f}× zero's; "
             f"{len(best)} instances (a, b) constant along q or mixed")


def main():
    ch = Checks()
    part_a(ch)
    part_b_affine(ch)
    part_b_affine_served(ch)
    part_b_transform(ch)
    part_c(ch)
    part_c_masks(ch)
    part_d(ch)
    part_e(ch)
    part_f(ch)
    if ch.mutations:
        with ch.mutant("A without the fold (A's block in place of A♯)"):
            part_a(ch, mut="no-fold")
        with ch.mutant("B1 ℓ with T₀ added (a bilinear correction)"):
            part_b_affine(ch, mut="bilinear-ell")
        with ch.mutant("B2 ℓ read off a perturbed B"):
            part_b_transform(ch, mut="ell-off-by-one")
        with ch.mutant("C flipping R only"):
            part_c(ch, mut="flip-R-only")
        with ch.mutant("C on an asymmetric alphabet 1..128"):
            part_c(ch, mut="asym-alphabet")
        with ch.mutant("C5 cancelling only the R-side data term"):
            part_c_masks(ch, mut="two-call-any")
        with ch.mutant("D with 0 in U's alphabet (−64..63)"):
            part_d(ch, mut="zero-in-alphabet")
        with ch.mutant("F with 0 in U's alphabet (−64..63)"):
            part_f(ch, mut="zero-in-alphabet")
    sys.exit(ch.finish())


if __name__ == "__main__":
    main()

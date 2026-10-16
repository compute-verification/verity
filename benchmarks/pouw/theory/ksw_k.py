"""KSW-K's exact check: what Theorem K's counting gives on Komargodski–Schen–Weinstein's low-rank transcript.

The transcript (KSW Def 6.1, MatMul_L): A′ = A + E_L·E_R and B′ = B + F_L·F_R, with E_L, F_L (n × r) and E_R, F_R (r × n).
The committed words are the prefix tiles T_ℓ[a, c] = Σ_{k < (ℓ+1)L} A′[a, k]·B′[k, c], for rows a, columns c < n and
depths ℓ < n/L: n³/L words. KSW take L = r. A word's cell is (a, c) and its cell-block (a, c, ℓ). Noise entries are bits
(the factor alphabet {0, 1}); A and B are int8; arithmetic is exact (int64), and in Z/2³² where the kit's Möbius reads it.

Parts:
- A. The zero-instance identity T_ℓ = (E_L)_i·S_ℓ·(F_R)_j with S_ℓ = Σ_{ℓ′ ≤ ℓ} (E_R)_{ℓ′}(F_L)_{ℓ′} (r × r), the general
  instance's four-term split, and the natural algorithm's MAC count n·r² + n²r²/L + n³r/L, against n³ direct.
- B. Theorem K as stated, all four factors coins: a word is degree 4 at A = B = 0 with no degree-2 part; for every A, B
  its degree-2 part is A·F_L·F_R + E_L·E_R·B, with no (E bit, F bit) cross pair; at A, B even its F₂ rank is 0.
- C. Conditioned on E_L, F_R (public), coins E_R, F_L: every word is bilinear plus affine; an increment's cross block
  has F₂ rank L (iff the row of E_L and the column of F_R are nonzero mod 2); a cell's chain has rank n; the class
  count (charge-pure, one-sided) is n³ at every L. KSW's own algorithm, n·r² one-sided products shared by every cell, is
  exact, so the class is what carries the count; at L = n (output only) the class count n³ is "beaten" by KSW §2's
  A = 0 attack. The luck of a one-product-short word.
- D. The other two conditionings (mixed: coins E_L, F_L; outer: coins E_L, F_R) and Dither's contrast: in Dither every
  coin-pair monomial is in exactly one cell-block's target, in KSW each is shared by Θ(n) cells under every
  conditioning, and the span of all words' degree-2 parts is ≤ n²r² (inner: (n/L)·r²).

Run: `uv run --no-project --with numpy --with-editable ../exact_kit python -B ksw_k.py --mutations` from this
directory.
"""
from __future__ import annotations

import sys
from itertools import combinations

import numpy as np

from exact_kit import cube, f2
from exact_kit.checks import Checks

GROUPS = ("EL", "ER", "FL", "FR")
MOD = 1 << 32


def shape_of(g, n, r):
    return (n, r) if g in ("EL", "FL") else (r, n)


class Layout:
    """Bits of the coin groups, in GROUPS order; the other groups are public (fixed matrices)."""

    def __init__(self, n, r, coins):
        self.n, self.r, self.coins = n, r, tuple(g for g in GROUPS if g in coins)
        self.off, v = {}, 0
        for g in self.coins:
            self.off[g] = v
            v += n * r
        self.V = v

    def idx(self, g, i, j):
        rows, cols = shape_of(g, self.n, self.r)
        return self.off[g] + i * cols + j

    def bits_of(self, g):
        return list(range(self.off[g], self.off[g] + self.n * self.r))

    def mats(self, pts, fixed):
        """Each group as an array (P, rows, cols): coins read from the points, public groups broadcast."""
        out = {}
        for g in GROUPS:
            rows, cols = shape_of(g, self.n, self.r)
            if g in self.off:
                sh = np.arange(self.n * self.r, dtype=np.uint64) + np.uint64(self.off[g])
                b = (pts[:, None] >> sh[None, :]) & np.uint64(1)
                out[g] = b.astype(np.int64).reshape(-1, rows, cols)
            else:
                out[g] = np.broadcast_to(fixed[g], (len(pts), rows, cols))
        return out


def prefix_words(Ap, Bp, L):
    """(P, n, n) operands -> (P, n, n, T) prefix words T_ℓ[a, c]."""
    n = Ap.shape[-1]
    inc = [np.einsum("pak,pkc->pac", Ap[:, :, t * L:(t + 1) * L], Bp[:, t * L:(t + 1) * L, :]) for t in range(n // L)]
    return np.cumsum(np.stack(inc, axis=-1), axis=-1)


def words_at(lay, pts, fixed, A, B, L):
    m = lay.mats(pts, fixed)
    Ap = A[None] + np.einsum("zap,zpk->zak", m["EL"], m["ER"])
    Bp = B[None] + np.einsum("zkq,zqc->zkc", m["FL"], m["FR"])
    return prefix_words(Ap, Bp, L)


def mob32(vals, V):
    return cube.mobius(np.asarray(vals % MOD, dtype=np.uint32), V, bits=32)


def rand_bits(rng, shape):
    return rng.integers(0, 2, size=shape).astype(np.int64)


def rand_bits_nonzero(rng, g, n, r):
    """A public factor whose rows (EL, FL) or columns (ER, FR) are nonzero mod 2."""
    while True:
        M = rand_bits(rng, shape_of(g, n, r))
        lines = M if g in ("EL", "FL") else M.T
        if all(line.any() for line in lines):
            return M


# ---------------------------------------------------------------- A. the transcript and the natural algorithm


class Macs:
    def __init__(self):
        self.count = 0

    def mm(self, X, Y):
        self.count += X.shape[0] * X.shape[1] * Y.shape[1]
        return X @ Y


def natural_zero_instance(EL, ER, FL, FR, L, macs):
    """KSW's zero-instance algorithm: M_ℓ = (E_R)_ℓ (F_L)_ℓ, S_ℓ its prefix, G_iℓ = (E_L)_i S_ℓ, tile = G_iℓ (F_R)_j."""
    n, r = EL.shape
    T = n // L
    S, out = np.zeros((r, r), dtype=np.int64), np.zeros((n, n, T), dtype=np.int64)
    for t in range(T):
        S = S + macs.mm(ER[:, t * L:(t + 1) * L], FL[t * L:(t + 1) * L, :])
        for i in range(T):
            G = macs.mm(EL[i * L:(i + 1) * L, :], S)
            for j in range(T):
                out[i * L:(i + 1) * L, j * L:(j + 1) * L, t] = macs.mm(G, FR[:, j * L:(j + 1) * L])
    return out


def part_a(ck, rng):
    for n, r, L in [(8, 2, 1), (8, 2, 2), (8, 2, 4), (8, 2, 8), (16, 4, 4), (16, 4, 2), (16, 2, 2)]:
        EL, ER, FL, FR = (rand_bits(rng, shape_of(g, n, r)) for g in GROUPS)
        A = rng.integers(-128, 128, size=(n, n)).astype(np.int64)
        B = rng.integers(-128, 128, size=(n, n)).astype(np.int64)
        E, F = EL @ ER, FL @ FR
        full = prefix_words((A + E)[None], (B + F)[None], L)[0]
        AF = np.cumsum(np.stack([A[:, t * L:(t + 1) * L] @ FL[t * L:(t + 1) * L] for t in range(n // L)], -1), -1)
        EB = np.cumsum(np.stack([ER[:, t * L:(t + 1) * L] @ B[t * L:(t + 1) * L] for t in range(n // L)], -1), -1)
        macs = Macs()
        nn = natural_zero_instance(EL, ER, FL, FR, L, macs)
        split = (prefix_words(A[None], B[None], L)[0] + np.einsum("aqt,qc->act", AF, FR)
                 + np.einsum("ap,pct->act", EL, EB) + nn)
        zero = prefix_words(E[None], F[None], L)[0]
        formula = n * r * r + n * n * r * r // L + n ** 3 * r // L
        ck.check(f"A ({n},{r},{L}) zero instance = E_L·S_ℓ·F_R tile-wise, general = four-term split",
                 np.array_equal(zero, nn) and np.array_equal(full, split))
        ck.check(f"A ({n},{r},{L}) natural algorithm MACs = n r² + n²r²/L + n³r/L",
                 macs.count == formula, f"{macs.count} = {formula}, direct n³ = {n ** 3}, ratio {macs.count / n ** 3:.4f}")


# ---------------------------------------------------------------- B. Theorem K as stated: all four factors coins


def part_b(ck, rng, mutate_even=False):
    n, r, L = 4, 1, 1
    lay = Layout(n, r, GROUPS)
    pts = cube.points(lay.V)
    deg = cube.degree(lay.V)
    Ebits = set(lay.bits_of("EL") + lay.bits_of("ER"))
    Fbits = set(lay.bits_of("FL") + lay.bits_of("FR"))
    zero = np.zeros((n, n), dtype=np.int64)
    odd = (2 * rng.integers(-64, 64, size=(n, n)) + 1).astype(np.int64)
    even = (2 * rng.integers(-64, 64, size=(n, n))).astype(np.int64)
    if mutate_even:
        even = odd
    cases = {"zero": (zero, zero), "odd": (odd, odd.T.copy()), "even": (even, even.T.copy())}
    for name, (A, B) in cases.items():
        W = words_at(lay, pts, {}, A, B, L)
        maxdeg, low_zero, cross_zero, ranks = 0, True, True, []
        for a in range(n):
            for c in range(n):
                for t in range(n // L):
                    co = mob32(W[:, a, c, t], lay.V)
                    nz = co != 0
                    maxdeg = max(maxdeg, int(deg[nz].max()) if nz.any() else 0)
                    if name == "zero":
                        low_zero &= not (nz & (deg < 4)).any()
                    for u in Ebits:
                        for v in Fbits:
                            cross_zero &= int(co[(1 << u) | (1 << v)]) == 0
                    ranks.append(f2.rank(cube.quadratic_rows(co, lay.V, 0)))
        if name == "zero":
            ck.check("B (4,1,1) zero instance: every word has degree 4 and nothing below", low_zero and maxdeg == 4)
        ck.check(f"B (4,1,1) {name} instance: no (E bit, F bit) degree-2 coefficient, degree ≤ 4",
                 cross_zero and maxdeg <= 4, f"max degree {maxdeg}")
        if name == "even":
            ck.check("B (4,1,1) A, B even: every word's degree-2 part has F₂ rank 0 (K vacuous)", max(ranks) == 0,
                     f"max rank {max(ranks)}")
        if name == "odd":
            ck.check("B (4,1,1) A, B odd: degree-2 part (A·F_L·F_R + E_L·E_R·B) is nonzero mod 2", max(ranks) > 0,
                     f"ranks {sorted(set(ranks))}")
    # L = r = 2, n = 4, all 32 bits: pair differences on the ≤2-ball.
    n, r, L = 4, 2, 2
    lay = Layout(n, r, GROUPS)
    V = lay.V
    ball = [0] + [1 << u for u in range(V)] + [(1 << u) | (1 << v) for u, v in combinations(range(V), 2)]
    ball = np.array(ball, dtype=np.uint64)
    pos = {int(p): i for i, p in enumerate(ball)}
    Eb = lay.bits_of("EL") + lay.bits_of("ER")
    Fb = lay.bits_of("FL") + lay.bits_of("FR")
    for name, (A, B) in {"zero": (np.zeros((n, n), np.int64),) * 2,
                         "random": (rng.integers(-128, 128, (n, n)), rng.integers(-128, 128, (n, n)))}.items():
        W = words_at(lay, ball, {}, np.asarray(A, np.int64), np.asarray(B, np.int64), L)
        f0 = W[pos[0]]
        d2_all_zero, cross_zero = True, True
        for u, v in combinations(range(V), 2):
            pd = W[pos[(1 << u) | (1 << v)]] - W[pos[1 << u]] - W[pos[1 << v]] + f0
            nzp = bool(pd.any())
            d2_all_zero &= not nzp
            if (u in Eb) != (v in Eb):
                cross_zero &= not nzp
        if name == "zero":
            ck.check("B (4,2,2) zero instance, all 496 pairs: degree-2 part is 0 in every word", d2_all_zero)
        else:
            ck.check("B (4,2,2) random instance, all (E, F) pairs: no cross degree-2 coefficient", cross_zero)


# ---------------------------------------------------------------- C. conditioned on E_L, F_R: coins E_R, F_L


def cross_inner(n, r, L, EL, FR, a, c, t, prefix=True):
    """Coefficient matrix (E_R bits × F_L bits) of word (a, c, t), mod 2, under the inner conditioning."""
    M = np.zeros((r * n, n * r), dtype=np.int64)
    for k in range(n):
        blk = k // L
        if (blk <= t) if prefix else (blk == t):
            for p in range(r):
                for q in range(r):
                    M[p * n + k, k * r + q] = EL[a, p] * FR[q, c]
    return M % 2


def rows_of(M):
    return f2.from_matrix(np.asarray(M) % 2)


def part_c(ck, rng, mutate_prefix=False, mutate_dup=False):
    n, r, L = 4, 2, 2
    lay = Layout(n, r, ("ER", "FL"))
    pts = cube.points(lay.V)
    deg = cube.degree(lay.V)
    EL = rand_bits_nonzero(rng, "EL", n, r)
    FR = rand_bits_nonzero(rng, "FR", n, r)
    A = rng.integers(-128, 128, (n, n)).astype(np.int64)
    B = rng.integers(-128, 128, (n, n)).astype(np.int64)
    fixed = {"EL": EL, "FR": FR}
    W = words_at(lay, pts, fixed, A, B, L)
    ok_deg, ok_formula = True, True
    for a in range(n):
        for c in range(n):
            for t in range(n // L):
                co = mob32(W[:, a, c, t], lay.V)
                ok_deg &= not ((co != 0) & (deg > 2)).any()
                Mx = cross_inner(n, r, L, EL, FR, a, c, t)
                for p in range(r):
                    for k in range(n):
                        for kk in range(n):
                            for q in range(r):
                                u, v = lay.idx("ER", p, k), lay.idx("FL", kk, q)
                                want = EL[a, p] * FR[q, c] * (k == kk) * (k // L <= t)
                                ok_formula &= int(co[(1 << u) | (1 << v)]) == want % MOD
                del Mx
    ck.check("C (4,2,2) inner conditioning: every word is bilinear + affine in (E_R, F_L) on the full cube", ok_deg)
    ck.check("C (4,2,2) cross coefficient of (E_R[p,k], F_L[k',q]) = [k=k'][⌊k/L⌋ ≤ ℓ]·E_L[a,p]·F_R[q,c]", ok_formula)

    for (n, r, L) in [(4, 2, 2), (8, 2, 2), (8, 4, 4), (16, 4, 4), (16, 2, 2), (32, 4, 4)]:
        EL = rand_bits(rng, (n, r))
        FR = rand_bits(rng, (r, n))
        T = n // L
        inc_ok, chain_ok, cls = True, True, 0
        for a in range(n):
            for c in range(n):
                live = EL[a].any() and FR[:, c].any()
                for t in range(T):
                    rk = f2.rank(rows_of(cross_inner(n, r, L, EL, FR, a, c, t, prefix=False)))
                    inc_ok &= rk == (L if live else 0)
                # chain: flatten the stacked cross blocks along the F_L side.
                stack = np.concatenate([cross_inner(n, r, L, EL, FR, a, c, t) for t in range(T)], axis=0)
                chain_ok &= f2.rank(rows_of(stack.T)) == (n if live else 0)
                cls += n if live else 0
        span_rows = []
        for a in range(n):
            for c in range(n):
                for t in range(T):
                    span_rows.append(int("".join(map(str, cross_inner(n, r, L, EL, FR, a, c, t, prefix=False)
                                                     .reshape(-1)[::-1])), 2))
        span = f2.rank(span_rows)
        live_cells = sum(EL[a].any() and FR[:, c].any() for a in range(n) for c in range(n))
        ck.check(f"C ({n},{r},{L}) increment cross block has F₂ rank L at live cells, 0 elsewhere", inc_ok)
        ck.check(f"C ({n},{r},{L}) a live cell's chain (T = n/L words) has rank n", chain_ok,
                 f"class count {cls} = n·#live cells ({live_cells}/{n * n}); n³ = {n ** 3}")
        ck.check(f"C ({n},{r},{L}) span of all words' increments ≤ (n/L)·r², KSW's products n·r²",
                 span <= T * r * r, f"span {span}, (n/L)r² = {T * r * r}, n r² = {n * r * r}, "
                 f"class/span = {cls / max(span, 1):.0f}")

    # KSW's algorithm under the conditioning: products π[k,p,q] = E_R[p,k]·F_L[k,q], n·r² of them, read by every cell.
    for (n, r, L) in [(4, 2, 2), (4, 2, 4), (4, 1, 4)]:
        lay = Layout(n, r, ("ER", "FL"))
        pts = cube.points(lay.V)
        m = None
        EL = rand_bits_nonzero(rng, "EL", n, r)
        FR = rand_bits_nonzero(rng, "FR", n, r)
        A = rng.integers(-128, 128, (n, n)).astype(np.int64)
        B = rng.integers(-128, 128, (n, n)).astype(np.int64)
        fixed = {"EL": EL, "FR": FR}
        W = words_at(lay, pts, fixed, A, B, L)
        m = lay.mats(pts, fixed)
        ER, FL = m["ER"], m["FL"]
        if mutate_dup:
            ER = ER.copy()
            ER[:, :, 1::L] = ER[:, :, 0::L][:, :, : ER[:, :, 1::L].shape[2]]
        pi = np.einsum("zpk,zkq->zkpq", ER, FL)
        nprod = n * r * r
        T = n // L
        S = np.zeros((len(pts), r, r), dtype=np.int64)
        alg = np.zeros_like(W)
        AFp = np.zeros((len(pts), n, r), dtype=np.int64)
        EBp = np.zeros((len(pts), r, n), dtype=np.int64)
        ABp = np.zeros((n, n), dtype=np.int64)
        for t in range(T):
            blk = slice(t * L, (t + 1) * L)
            Mt = pi[:, blk].sum(axis=1)
            S = Mt if (mutate_prefix and t > 0) else S + Mt
            AFp = AFp + np.einsum("ak,zkq->zaq", A[:, blk], FL[:, blk])
            EBp = EBp + np.einsum("zpk,kc->zpc", ER[:, :, blk], B[blk])
            ABp = ABp + A[:, blk] @ B[blk]
            alg[..., t] = (ABp[None] + np.einsum("zaq,qc->zac", AFp, FR) + np.einsum("ap,zpc->zac", EL, EBp)
                           + np.einsum("ap,zpq,qc->zac", EL, S, FR))
        exact = np.array_equal(alg % MOD, W % MOD)
        tag = "output only, KSW §2's A = 0 attack" if L == n else "L = r" if L == r else f"L = {L}"
        ck.check(f"C ({n},{r},{L}) [{tag}] KSW's shared algorithm is exact on the full cube with n·r² products",
                 exact, f"{nprod} one-sided products; class count n³ = {n ** 3}")
        readers = max(int((EL[:, p] % 2).sum() * (FR[q] % 2).sum()) for p in range(r) for q in range(r))
        ck.check(f"C ({n},{r},{L}) a product is read oddly by several cell-blocks of one frame (not charge-pure)",
                 readers >= 2, f"max odd readers at its own depth {readers}")
        del m

    # Luck: drop one term x_k·y_k of a live cell's increment; pass = the dropped term vanishes.
    n, r, L = 4, 2, 2
    lay = Layout(n, r, ("ER", "FL"))
    pts = cube.points(lay.V)
    EL = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    FR = np.array([[1, 0, 1, 1], [1, 1, 0, 1]], dtype=np.int64)
    m = lay.mats(pts, {"EL": EL, "FR": FR})
    for a, c in [(0, 0), (1, 1), (0, 3)]:
        x = np.einsum("p,zp->z", EL[a], m["ER"][:, :, 0])
        y = np.einsum("q,zq->z", FR[:, c], m["FL"][:, 0, :])
        wa, wc = int(EL[a].sum()), int(FR[:, c].sum())
        p_int = float(((x * y) == 0).mean())
        p_two = float(((x * y) % 2 == 0).mean())
        want = 1 - (1 - 2.0 ** -wa) * (1 - 2.0 ** -wc)
        ck.check(f"C luck, one product short at cell ({a},{c}), |e_a| = {wa}, |f_c| = {wc}",
                 abs(p_int - want) < 1e-12 and abs(p_two - 0.75) < 1e-12,
                 f"pass {p_int:.4f} = 1 − (1 − 2^−{wa})(1 − 2^−{wc}) in Z, {p_two:.4f} mod 2")


# ---------------------------------------------------------------- D. other conditionings, and Dither


def part_d(ck, rng, mutate_lowrank_dither=False):
    # Mixed: coins E_L, F_L; public E_R, F_R. Row-pure affine algorithm: products e_a[p]·S_ℓ[p,q].
    n, r, L = 4, 2, 2
    lay = Layout(n, r, ("EL", "FL"))
    pts = cube.points(lay.V)
    ER = rand_bits_nonzero(rng, "ER", n, r)
    FR = rand_bits_nonzero(rng, "FR", n, r)
    A = rng.integers(-128, 128, (n, n)).astype(np.int64)
    B = rng.integers(-128, 128, (n, n)).astype(np.int64)
    W = words_at(lay, pts, {"ER": ER, "FR": FR}, A, B, L)
    m = lay.mats(pts, {"ER": ER, "FR": FR})
    EL, FL = m["EL"], m["FL"]
    T = n // L
    S = np.zeros((len(pts), r, r), dtype=np.int64)
    alg = np.zeros_like(W)
    nprod = 0
    for t in range(T):
        blk = slice(t * L, (t + 1) * L)
        S = S + np.einsum("pk,zkq->zpq", ER[:, blk], FL[:, blk])
        pref_AB = sum(A[:, s * L:(s + 1) * L] @ B[s * L:(s + 1) * L] for s in range(t + 1))
        pref_AF = sum(np.einsum("ak,zkq->zaq", A[:, s * L:(s + 1) * L], FL[:, s * L:(s + 1) * L]) for s in range(t + 1))
        pref_EB = sum(ER[:, s * L:(s + 1) * L] @ B[s * L:(s + 1) * L] for s in range(t + 1))
        prods = np.einsum("zap,zpq->zapq", EL, S)
        nprod += n * r * r
        alg[..., t] = (pref_AB[None] + np.einsum("zaq,qc->zac", pref_AF, FR) + np.einsum("zap,pc->zac", EL, pref_EB)
                       + np.einsum("zapq,qc->zac", prods, FR))
    ok_formula = True
    for a in range(n):
        for c in range(n):
            for t in range(T):
                co = mob32(W[:, a, c, t], lay.V)
                for p in range(r):
                    for k in range(n):
                        for q in range(r):
                            want = ER[p, k] * FR[q, c] * (k < (t + 1) * L)
                            ok_formula &= int(co[(1 << lay.idx("EL", a, p)) | (1 << lay.idx("FL", k, q))]) == want
    ck.check("D (4,2,2) mixed: cross coefficient of (E_L[a,p], F_L[k,q]) = [k < (ℓ+1)L]·E_R[p,k]·F_R[q,c]", ok_formula)
    ck.check("D (4,2,2) mixed conditioning (coins E_L, F_L): row-pure affine-operand algorithm is exact",
             np.array_equal(alg % MOD, W % MOD), f"{nprod} products = n²r²/L, class count n³ = {n ** 3}")

    def mixed_row_rank(n, r, L, ER, FR, a):
        """Row a's words, flattened along F_L: rows F_L bits, columns (word, E_L[a,·] bit)."""
        T = n // L
        M = np.zeros((n * r, n * T * r), dtype=np.int64)
        for c in range(n):
            for t in range(T):
                for k in range((t + 1) * L):
                    for p in range(r):
                        for q in range(r):
                            M[k * r + q, (c * T + t) * r + p] ^= (ER[p, k] * FR[q, c]) & 1
        return f2.rank(rows_of(M))

    for n, r, L in [(8, 2, 2), (16, 2, 2), (16, 4, 4)]:
        ER = rand_bits_nonzero(rng, "ER", n, r)
        FR = rand_bits_nonzero(rng, "FR", n, r)
        rk = sum(mixed_row_rank(n, r, L, ER, FR, a) for a in range(n))
        ck.check(f"D ({n},{r},{L}) mixed: Σ_rows chain rank (K's affine row-pure count) ≤ n·(n/L)·r²",
                 rk <= n * (n // L) * r * r, f"{rk} vs n²r²/L = {n * (n // L) * r * r}, n³ = {n ** 3}")

    # Outer: coins E_L, F_R; public cores S_ℓ. A cell's chain rank is dim span{S_ℓ mod 2} ≤ r².
    for n, r, L in [(16, 2, 2), (32, 2, 2), (32, 4, 4)]:
        ERp, FLp = rand_bits(rng, (r, n)), rand_bits(rng, (n, r))
        Ss = np.cumsum(np.stack([ERp[:, t * L:(t + 1) * L] @ FLp[t * L:(t + 1) * L] for t in range(n // L)]), 0) % 2
        span = f2.rank([int("".join(map(str, S.reshape(-1)[::-1])), 2) for S in Ss])
        ck.check(f"D ({n},{r},{L}) outer: a cell's T = {n // L} prefix words have chain rank ≤ r² = {r * r}",
                 span <= r * r, f"rank {span} against T·L = n = {n} for a K-style count")

    # Monomial multiplicity: how many cell-blocks' increments carry a coin-pair monomial with odd coefficient.
    n, r, L = 16, 4, 4
    T = n // L
    EL, ER, FL, FR = (rand_bits(rng, shape_of(g, n, r)) for g in GROUPS)
    inner = [int((EL[:, p] % 2).sum() * (FR[q] % 2).sum()) for k in range(n) for p in range(r) for q in range(r)]
    mixed = [int(sum((ER[p, k] * FR[q, c]) & 1 for c in range(n))) for a in range(n) for k in range(n)
             for p in range(r) for q in range(r)]
    Ms = [(ER[:, t * L:(t + 1) * L] @ FL[t * L:(t + 1) * L]) % 2 for t in range(T)]
    outer = [int(sum(Mt[p, q] for Mt in Ms)) for p in range(r) for q in range(r)]
    # Dither: words (i, j, l) = Σ_m R[i, L·l + m]·U[j, m]; R full rank (one bit an entry), U shared across blocks l.
    nd, nprime, K = 8, 8, 16
    Rl, Rr = rand_bits(rng, (nd, 2)), rand_bits(rng, (2, K))
    mult = {}
    for i in range(nd):
        for j in range(nprime):
            for l in range(K // L):
                for mm in range(L):
                    if mutate_lowrank_dither:
                        keys = [("RL", i, s, j, mm) for s in range(2) if Rl[i, s] and Rr[s, L * l + mm]]
                    else:
                        keys = [("R", i, L * l + mm, j, mm)]
                    for key in keys:
                        mult[key] = mult.get(key, 0) + 1
    dmax = max(mult.values())
    ck.check("D Dither: every coin-pair monomial R[i,k]·U[j,k mod L] is in exactly one cell-block's increment",
             dmax == 1, f"{len(mult)} monomials = n·K·n′ = {nd * K * nprime} honest MACs, max multiplicity {dmax}")
    ck.check(f"D KSW ({n},{r},{L}) shares monomials under every conditioning",
             max(inner) > 1 and max(mixed) > 1 and max(outer) > 1,
             f"mean odd multiplicity: inner {np.mean(inner):.1f} (≈ n²/4 = {n * n / 4:.0f}), "
             f"mixed {np.mean(mixed):.1f} (≈ n/4 = {n / 4:.0f}), outer {np.mean(outer):.1f} (≈ T/2 = {T / 2:.0f})")


def main():
    ck = Checks()
    rng = np.random.default_rng(20261016)
    part_a(ck, rng)
    part_b(ck, rng)
    part_c(ck, rng)
    part_d(ck, rng)
    if ck.mutations:
        with ck.mutant("B: A, B odd in place of even (the rank-0 check must fail)"):
            part_b(ck, np.random.default_rng(1), mutate_even=True)
        with ck.mutant("C: S_ℓ without the prefix in KSW's shared algorithm"):
            part_c(ck, np.random.default_rng(2), mutate_prefix=True)
        with ck.mutant("C: duplicated E_R columns inside a block (rank < L, algorithm on the wrong coins)"):
            part_c(ck, np.random.default_rng(3), mutate_dup=True)
        with ck.mutant("D: Dither's R made rank 2 (monomials shared)"):
            part_d(ck, np.random.default_rng(4), mutate_lowrank_dither=True)
    sys.exit(ck.finish())


if __name__ == "__main__":
    main()

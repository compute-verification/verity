"""NEEKON-FORM's exact check: certification on the ≤2-ball does not bound a general circuit's products.

The design is Lean's (`Definitions.Pouw.RelativeSecurity`): bits are U's planes β_p(i, j) (row i < n, column j < L,
plane p < 7, weights wRe = (1, 2, 4, 8, 16, 32, 65)) and the coins b(t, j, l) (block t < B, j < L, column l < n′).
P̃(i, tL+j) = pc(i, t, j) + Σ_p w_p β_p(i, j), X(tL+j, l) = vq(t, j, l) + b(t, j, l), and the committed word of
cell-block (i, l, t) is the prefix T_t(i, l) = Σ_{t′ ≤ t} Σ_j P̃ X (`dPrefix`), all in Z/2³².

Parts:
- A. The product rule on the ball: pairDiff(fg) = f_u g_v + f_v g_u + f̂·g(e_u+e_v) + ĝ·f(e_u+e_v) − f̂ĝ, for any f, g
  over Z/2³²; with f, g affine it is f_u g_v + f_v g_u (alternating rank ≤ 2).
- B. The ball circuit: n·n′·B + n + 2·log₂L + 1 multiplications, every word equal to dPrefix at every bit point with
  ≤ 2 ones (exhaustive), for every committed pc and vq drawn. One equality gadget E (1 at e_β + e_b iff the U bit's
  column equals the coin's j) serves every word; ρ_i = E·W_i; word = affine + ρ_i·S_tl.
- C. Its products, read mod 2 on the ball: E and ρ_i have cross blocks of F₂ rank L, against ≤ 2 for an affine product.
- D. It fails the spot check: its pass rate at uniform coins.
- E. The comparison: its count against naive L·n·n′·B and the elementary lower bound on C_F₂(⟨n, L, n′B⟩).
- F. At n = 1 and n′B < 20 its count is below L ≤ MC_∧(⟨1, L, n′B⟩), the AND count of every F₂ circuit exact on the
  whole cube (restriction: t ANDs leave the outputs affine on a subspace of codimension ≤ t, whose direction is
  isotropic for every output's alternating form).
- G. The restriction bound's lemma, exhaustively at small L, N: the largest subspace isotropic for every output form of
  ⟨1, L, N⟩ has dimension L·N, so its codimension is L.

Run: `uv run --no-project --with numpy --with-editable benchmarks/pouw/exact_kit python -B neekon_form.py --mutations`
(from this directory, or with the path).
"""
from __future__ import annotations

import sys

import numpy as np

from exact_kit import f2
from exact_kit.checks import Checks

M32 = np.uint32
W_RE = np.array([1, 2, 4, 8, 16, 32, 65], dtype=M32)
NP = 7


class Shape:
    def __init__(self, n, L, n2, B):
        self.n, self.L, self.n2, self.B = n, L, n2, B
        self.nU = n * L * NP
        self.nb = B * L * n2
        self.V = self.nU + self.nb

    def u(self, i, j, p):
        return (i * self.L + j) * NP + p

    def b(self, t, j, l):
        return self.nU + (t * self.L + j) * self.n2 + l

    def __str__(self):
        return f"(n, L, n', B) = ({self.n}, {self.L}, {self.n2}, {self.B})"


# ---------------------------------------------------------------- straight-line program over Z/2^32 on the bits


class Slp:
    """Gates: ('lin', dense coefficient row over the bits plus a constant), ('comb', constant, [(coef, gate)]),
    ('mul', g, h). `muls` counts the multiplications of two gates (every product here reads the bits on both sides)."""

    def __init__(self, V):
        self.V = V
        self.gates = []
        self.lin_rows = []
        self.lin_const = []

    def lin(self, coef: dict, const=0):
        row = np.zeros(self.V + 1, dtype=M32)
        for k, c in coef.items():
            row[k] = M32((row[k] + np.uint64(c % 2**32)) % 2**32)
        self.lin_rows.append(row)
        self.lin_const.append(M32(const % 2**32))
        self.gates.append(("lin", len(self.lin_rows) - 1))
        return len(self.gates) - 1

    def lin_dense(self, row, const):
        r = np.zeros(self.V + 1, dtype=M32)
        r[: self.V] = row
        self.lin_rows.append(r)
        self.lin_const.append(M32(const % 2**32))
        self.gates.append(("lin", len(self.lin_rows) - 1))
        return len(self.gates) - 1

    def comb(self, const, terms):
        self.gates.append(("comb", M32(const % 2**32), [(M32(c % 2**32), g) for c, g in terms]))
        return len(self.gates) - 1

    def mul(self, g, h):
        self.gates.append(("mul", g, h))
        return len(self.gates) - 1

    @property
    def muls(self):
        return sum(1 for g in self.gates if g[0] == "mul")

    def _lin_values(self, pts):
        R = np.stack(self.lin_rows)  # (F, V+1)
        c = np.array(self.lin_const, dtype=M32)[:, None]
        if pts[0] == "ball":
            _, u, v = pts
            return c + R[:, u] + R[:, v]
        _, z = pts  # dense (N, V) in {0,1}
        return c + (z.astype(M32) @ R[:, : self.V].T).T

    def evaluate(self, pts, wanted):
        lv = self._lin_values(pts)
        vals = {}
        for k, g in enumerate(self.gates):
            if g[0] == "lin":
                vals[k] = lv[g[1]]
            elif g[0] == "comb":
                acc = np.full(lv.shape[1], g[1], dtype=M32)
                for c, h in g[2]:
                    acc = acc + c * vals[h]
                vals[k] = acc
            else:
                vals[k] = vals[g[1]] * vals[g[2]]
        return {w: vals[w] for w in wanted}, vals


def ball_circuit(S: Shape, pc, vq, *, drop_bit=None, unit_weights=False, no_prefix=False):
    """The ball circuit. Returns (slp, words[(i, l, t)] -> gate, named gates)."""
    n, L, n2, B = S.n, S.L, S.n2, S.B
    P = Slp(S.V)
    w = np.ones(NP, dtype=M32) if unit_weights else W_RE
    nbits = max(1, (L - 1).bit_length())
    Xtot = P.lin({S.u(i, j, p): 1 for i in range(n) for j in range(L) for p in range(NP)})
    Ytot = P.lin({S.b(t, j, l): 1 for t in range(B) for j in range(L) for l in range(n2)})
    E = P.mul(Xtot, Ytot)
    for r in range(nbits):
        if r == drop_bit:
            continue
        Xr = P.lin({S.u(i, j, p): (j >> r) & 1 for i in range(n) for j in range(L) for p in range(NP)})
        Yr = P.lin({S.b(t, j, l): (j >> r) & 1 for t in range(B) for j in range(L) for l in range(n2)})
        m = P.mul(Xr, Yr)
        eq = P.comb(1, [(-1, Xr), (-1, Yr), (2, m)])
        E = P.mul(E, eq)
    rho = []
    for i in range(n):
        Wi = P.lin({S.u(i, j, p): int(w[p]) for j in range(L) for p in range(NP)})
        rho.append(P.mul(E, Wi))
    words = {}
    for t in range(B):
        for l in range(n2):
            ts = [t] if no_prefix else range(t + 1)
            Stl = P.lin({S.b(tt, j, l): 1 for tt in ts for j in range(L)})
            for i in range(n):
                # the word's constant and linear part, from the committed values (free: Lean's `off`)
                row = np.zeros(S.V, dtype=np.uint64)
                c0 = np.uint64(0)
                for tt in range(t + 1):
                    for j in range(L):
                        c0 += np.uint64(pc[i, tt, j]) * np.uint64(vq[tt, j, l])
                        row[S.b(tt, j, l)] += np.uint64(pc[i, tt, j])
                        for p in range(NP):
                            row[S.u(i, j, p)] += np.uint64(W_RE[p]) * np.uint64(vq[tt, j, l])
                aff = P.lin_dense((row % 2**32).astype(M32), int(c0 % 2**32))
                q = P.mul(rho[i], Stl)
                words[(i, l, t)] = P.comb(0, [(1, aff), (1, q)])
    return P, words, {"E": E, "rho": rho}


# ---------------------------------------------------------------- the target, independently (Lean's dP, dX, dPrefix)


def target(S: Shape, pc, vq, z):
    """dPrefix at dense bit points z (N, V) -> (N, n, n', B) words mod 2^32."""
    n, L, n2, B = S.n, S.L, S.n2, S.B
    N = z.shape[0]
    zu = z[:, : S.nU].reshape(N, n, L, NP).astype(M32)
    zb = z[:, S.nU :].reshape(N, B, L, n2).astype(M32)
    Upart = zu @ W_RE  # (N, n, L)
    Pt = pc[None, :, :, :] + Upart[:, :, None, :]  # (N, n, B, L)
    X = vq[None] + zb  # (N, B, L, n2)
    incr = np.matmul(Pt.transpose(0, 2, 1, 3), X)  # (N, B, n, n2)
    pref = np.cumsum(incr, axis=1, dtype=M32)  # prefix over blocks
    return pref.transpose(0, 2, 3, 1)  # (N, n, n2, B)


def ball_points(V):
    us, vs = [], []
    us.append(np.array([V]))
    vs.append(np.array([V]))
    us.append(np.arange(V))
    vs.append(np.full(V, V))
    iu, iv = np.triu_indices(V, 1)
    us.append(iu)
    vs.append(iv)
    return np.concatenate(us), np.concatenate(vs)


def dense_from_ball(V, u, v):
    z = np.zeros((len(u), V + 1), dtype=np.uint8)
    z[np.arange(len(u)), u] = 1
    z[np.arange(len(v)), v] = 1
    return z[:, :V]


def ball_check(S, P, words, pc, vq, chunk=4096, sample=None, rng=None):
    """Exhaustive over every bit point with <= 2 ones, or a uniform sample of `sample` of them plus the zero point and
    every singleton. Returns (#points, #word errors)."""
    u, v = ball_points(S.V)
    if sample is not None:
        keep = np.concatenate([np.arange(S.V + 1), rng.choice(np.arange(S.V + 1, len(u)), sample, replace=False)])
        u, v = u[keep], v[keep]
    keys = sorted(words)
    bad = 0
    for s in range(0, len(u), chunk):
        uu, vv = u[s : s + chunk], v[s : s + chunk]
        out, _ = P.evaluate(("ball", uu, vv), [words[k] for k in keys])
        T = target(S, pc, vq, dense_from_ball(S.V, uu, vv))
        for k in keys:
            i, l, t = k
            bad += int(np.count_nonzero(out[words[k]] != T[:, i, l, t]))
    return len(u), bad


def cube_pass(S, P, words, pc, vq, rng, N=2000):
    z = rng.integers(0, 2, size=(N, S.V), dtype=np.uint8)
    keys = sorted(words)
    out, _ = P.evaluate(("dense", z), [words[k] for k in keys])
    T = target(S, pc, vq, z)
    ok = np.stack([out[words[k]] == T[:, k[0], k[1], k[2]] for k in keys], axis=1)
    return float(ok.mean()), float(ok.all(axis=1).mean())


def cross_rank_mod2(S, P, gate, i):
    """F2 rank of a gate's pair differences on (β_0(i, ·), every coin): its cross block at row i, plane 0."""
    rows_u = [S.u(i, j, 0) for j in range(S.L)]
    cols = [S.b(t, j, l) for t in range(S.B) for j in range(S.L) for l in range(S.n2)]
    V = S.V
    uu = np.array([a for a in rows_u for _ in cols])
    vv = np.array([c for _ in rows_u for c in cols])
    f_uv = P.evaluate(("ball", uu, vv), [gate])[0][gate]
    f_u = P.evaluate(("ball", np.array(rows_u), np.full(len(rows_u), V)), [gate])[0][gate]
    f_v = P.evaluate(("ball", np.array(cols), np.full(len(cols), V)), [gate])[0][gate]
    f_0 = P.evaluate(("ball", np.array([V]), np.array([V])), [gate])[0][gate][0]
    D = (f_uv.reshape(len(rows_u), len(cols)) - f_u[:, None] - f_v[None, :] + f_0) & M32(1)
    masks = [int(sum(int(D[r, c]) << c for c in range(len(cols)))) for r in range(len(rows_u))]
    return f2.rank(masks)


def committed(S, rng):
    pc = (rng.integers(-192, 64, size=(S.n, S.B, S.L)) % 2**32).astype(M32)
    vq = (rng.integers(-128, 127, size=(S.B, S.L, S.n2)) % 2**32).astype(M32)
    return pc, vq


# ---------------------------------------------------------------- A. the product rule on the ball


def part_a(checks, rng):
    V = 9
    u, v = ball_points(V)
    idx = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(u, v))}

    def val(f, a=None, b=None):
        if a is None:
            return f[idx[(V, V)]]
        if b is None:
            return f[idx[(a, V)]]
        return f[idx[(min(a, b), max(a, b))]]

    def pd(f, a, b):
        return val(f, a, b) - val(f, a) - val(f, b) + val(f)

    worst = 0
    for _ in range(50):
        f = rng.integers(0, 2**32, size=len(u), dtype=np.uint64).astype(M32)
        g = rng.integers(0, 2**32, size=len(u), dtype=np.uint64).astype(M32)
        fg = f * g
        for a in range(V):
            for b in range(a + 1, V):
                fu, fv, gu, gv = val(f, a) - val(f), val(f, b) - val(f), val(g, a) - val(g), val(g, b) - val(g)
                fh, gh = pd(f, a, b), pd(g, a, b)
                rhs = fu * gv + fv * gu + fh * val(g, a, b) + gh * val(f, a, b) - fh * gh
                worst += int(pd(fg, a, b) != rhs)
    checks.check("A1 product rule on the ball, any f, g over Z/2^32", worst == 0, f"{worst} mismatches over 50 pairs")
    bad = 0
    for _ in range(50):
        c = rng.integers(0, 2**32, size=(2, V + 1), dtype=np.uint64).astype(M32)
        c[:, V] = 0
        k0 = rng.integers(0, 2**32, size=2, dtype=np.uint64).astype(M32)
        f = k0[0] + c[0, u] + c[0, v]
        g = k0[1] + c[1, u] + c[1, v]
        for a in range(V):
            for b in range(a + 1, V):
                bad += int(pd(f * g, a, b) != c[0, a] * c[1, b] + c[0, b] * c[1, a])
    checks.check("A2 affine f, g: pairDiff(fg) = f_u g_v + f_v g_u (rank <= 2)", bad == 0, f"{bad} mismatches")


# ---------------------------------------------------------------- B-E


def naive(S):
    return S.L * S.n * S.n2 * S.B


def c_lower(S):
    """Elementary lower bound on C_F2(<n, L, n'B>): max(dimension n·n'B, half the largest flattening rank)."""
    N = S.n2 * S.B
    return max(S.n * N, -(-max(S.n * S.L, S.L * N, S.n * N) // 2))


def mc_lower(S):
    """Lower bound on MC_and(<n, L, n'B>) over F2, general AND/XOR circuits exact on the whole cube: max(dimension
    n·n'B, L at n = 1). A circuit with t ANDs is affine on an affine subspace of codimension <= t (fix one input of
    each AND in order), whose direction is then isotropic for every output's alternating form; at n = 1 those spaces
    have dimension <= L·N, so codimension >= L."""
    N = S.n2 * S.B
    return max(S.n * N, S.L if S.n == 1 else 0)


def max_common_isotropic(L, N):
    """Exhaustive: the largest subspace of F2^(L + L·N) isotropic for every output form of <1, L, N>
    (B_l((x, y), (x', y')) = x·y'_l + x'·y_l), level by level over subspaces deduplicated by their span (tiny L, N)."""
    dim = L + L * N
    mx = (1 << L) - 1

    def form(l, a, b):
        ya, yb = (a >> (L + l * L)) & mx, (b >> (L + l * L)) & mx
        return (bin(a & mx & yb).count("1") + bin(b & mx & ya).count("1")) & 1

    level = {1: ((), frozenset([0]))}
    best = 0
    while level:
        nxt = {}
        for basis, span in level.values():
            for v in range(1, 1 << dim):
                if v in span or any(form(l, v, b) for b in basis for l in range(N)):
                    continue
                s2 = span | {s ^ v for s in span}
                key = sum(1 << s for s in s2)
                if key not in nxt:
                    nxt[key] = (basis + (v,), frozenset(s2))
        if nxt:
            best += 1
        level = nxt
    return best


def main():
    np.seterr(over="ignore")
    checks = Checks()
    rng = np.random.default_rng(20261016)
    part_a(checks, rng)

    full = [Shape(1, 32, 8, 4), Shape(4, 32, 4, 2), Shape(2, 4, 2, 2), Shape(1, 32, 1, 1), Shape(1, 32, 4, 4)]
    sampled = [Shape(8, 32, 8, 4)]
    for S in full + sampled:
        for draw in range(2 if S in full else 1):
            pc, vq = committed(S, rng)
            P, words, named = ball_circuit(S, pc, vq)
            nbits = (S.L - 1).bit_length()
            want = S.n * S.n2 * S.B + S.n + 2 * nbits + 1
            if draw == 0:
                checks.check(f"B0 {S}: multiplications", P.muls == want,
                             f"{P.muls} = n·n'B + n + 2·log2 L + 1; naive {naive(S)} "
                             f"(ratio {P.muls / naive(S):.4f}); C_F2 >= {c_lower(S)}")
            sample = None if (S in full or "--full" in checks.args) else 400_000
            npts, bad = ball_check(S, P, words, pc, vq, sample=sample, rng=rng)
            scope = "every" if sample is None else f"{npts:,} sampled (zero, every singleton, uniform pairs)"
            checks.check(f"B1 {S} draw {draw}: every word = dPrefix at {scope} bit points with <= 2 ones",
                         bad == 0, f"{npts:,} points x {len(words)} words, {bad} errors")
            if draw == 0:
                rE = cross_rank_mod2(S, P, named["E"], 0)
                rR = cross_rank_mod2(S, P, named["rho"][0], 0)
                checks.check(f"C {S}: E and rho_0 cross blocks mod 2 (row 0, plane 0) have rank L", rE == S.L == rR,
                             f"rank E {rE}, rank rho_0 {rR}; an affine product has <= 2")
                wpass, rpass = cube_pass(S, P, words, pc, vq, rng)
                checks.check(f"D {S}: uniform coins, words right / all words right", rpass < 0.01,
                             f"{wpass:.4f} / {rpass:.4f} over 2000 points")
        if S in full and S.L == 32 and S.n2 * S.B > 1:
            checks.check(f"E {S}: below the elementary C_F2 bound", P.muls < c_lower(S),
                         f"{P.muls} < {c_lower(S)} <= C_F2(<{S.n},{S.L},{S.n2 * S.B}>)")
        if S in full and S.n == 1 and S.n2 * S.B < 20:
            checks.check(f"F {S}: below MC_and of the F2 target (general circuits exact on the cube)",
                         P.muls < mc_lower(S),
                         f"{P.muls} < {mc_lower(S)} <= MC_and(<1,{S.L},{S.n2 * S.B}>)")

    for L, N in [(2, 1), (2, 2), (3, 1), (2, 3), (4, 1)]:
        iso = max_common_isotropic(L, N)
        checks.check(f"G <1,{L},{N}>: largest common isotropic subspace has dimension L·N", iso == L * N,
                     f"{iso} (so codimension L = {L} <= MC_and)")

    if checks.mutations:
        S = Shape(2, 4, 2, 2)
        pc, vq = committed(S, rng)
        for name, kw in [("E without its bit-0 equality factor", {"drop_bit": 0}),
                         ("W_i with unit weights in place of wRe", {"unit_weights": True}),
                         ("S_tl over block t only (no prefix)", {"no_prefix": True})]:
            P, words, _ = ball_circuit(S, pc, vq, **kw)
            with checks.mutant(name):
                _, bad = ball_check(S, P, words, pc, vq)
                checks.check(f"M {name}: ball check", bad == 0, f"{bad} errors")

    sys.exit(checks.finish())


if __name__ == "__main__":
    main()

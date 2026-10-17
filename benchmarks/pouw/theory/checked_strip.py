"""CHECKED-STRIP's exact check: CO-Dither, a two-sided dither whose served output is a checked word.

Coordinates are Lean's (`Definitions.Pouw.RelativeSecurity`): the peeled side P (n×k, in [−63, 62]) carries U, the
served side V (k×n′) carries the dither b. U(i, ρ) = −64 + Σ_p w_p β_p(i, ρ) over 7 planes, wRe = (1, 2, 4, 8, 16, 32, 65),
ρ < r; c ∈ {0,1}^{n×k} (two-sided designs only) and b ∈ {0,1}^{k×n′} are fresh coins. With P̂ = P + c,
P̃(i, κ) = P̂(i, κ) + U(i, κ mod r) and X = V + b, a cell (i, l) has, over Z/2³²:
- the transcript words T_t = Σ_{κ < (t+1)L} P̃ X, t < k/L;
- the strip P2 = Σ_ρ U(i, ρ) Y(ρ, l), with Y(ρ, l) = Σ_{κ ≡ ρ (mod r)} X(κ, l) the residue sums;
- the served output O = Σ_κ P̂ X = T_last − P2.
The honest prover computes T (k products a cell) and P2 (r products a cell) and derives O. Designs, by their committed
words: `dither` (Dither as adopted: T, with no c), `checked1` (T and O, no c), `co` (CO-Dither: T and O, with c) and
`co_p2` (T and P2, with c). At r = 32 with 8-bit X, Y needs more than one int8 limb, and `co32c` commits each of its
redundant base-128 digits: A_d = (R_d mod 128) − 128 z_d, R_{d+1} = ⌊R_d/128⌋ + z_d (R_0 = Y, z fresh carry coins),
words T, D_d = U·A_d (d ≥ 1) and O′ = T_last − U·A_0, with the served O = O′ − Σ 128^d D_d. The carry coin keeps D_1
fresh where Y < 128, which is every column at A = B = 0, where co32's high limb is 0 and its product free.

Parts:
- A. The s32 identity O = T_last − P2, the int8 ranges, the strip operand Y's int8 limbs at the worst instance, at
  the zero instance and at a structured one, and co32c's digits (int8, reconstructing Y and O; D_1 fresh at zero).
- B. Per-cell F₂ cross blocks from pair differences of the words evaluated at bit points with ≤ 2 ones, in both
  orientations (rows (word, peeled bit) or (word, served coin)), at the zero, a random and a structured instance; the
  same-side blocks vanish exactly (Lemma G's hypothesis on either side). CO's swapped block has rank k + r, the honest
  count; Dither's and checked1's best is k, r short.
- C. The F₂ tensor rank of the coin–coin target with affine-operand products of any mixing and no purity, by SAT at
  tiny shapes (one cell, a column of two rows, a row of two columns, a 2×2 block); each SAT model is checked.
- D. The collapses: one-sided, P2 = T_last at P = 0; KSW's four products equal at the zero instance; per-product
  independent noise admits no recovering combination; CO at zero serves the random 0/1 product c·b.
- E. Priced strategies, each verified word for word against the honest words: int8 MAC = 1, FP4 (E2M1) MAC = ½, one
  product per int8 limb, additions free. The ε table at production k, and Llama-3.1 8B/70B linears MAC-weighted. Under
  co32c the cheater at the zero and the structured instance pays k + m·r products a cell, as honest does.
- F. Served error: exact mean and variance of O − PV over the coins (Fractions, every coin point), against Dither's.
- G. Theorem N's bound ε ≥ (1 − θ)/(q − 1 + θ) for schemes whose q checked products are recovered by a per-coordinate
  identity, and the q it needs for ε < 1% and < 0.1%.
- H. The tensor–scalar model (note:20261017T0310Z-report-eps1-program): a 128-cell tile's MMAs and other warp
  instructions, honest against the cheapest cheater found, timed on the measured facets (the tensor pipe and JI3e), for
  CO at r = 32 (co32, co32c), CO at r = k/2 (cohalf) and Dither as built, at fold costs 3, 1.5 and 1 a word, on
  Llama-3.1 8B and 70B linears and on QKᵀ (k = 128). At A = B = 0, O = c·b is a depth-k binary product with no commit
  inside, so a cheater may make any fraction of it off its strip: bit-sliced on the CUDA cores (x warp instructions
  per 32 × 128 bit products, a parameter), with b1 MMAs (emulated on sm_120; native elsewhere) or FP4 MMAs. The other
  words are narrow there too (T at 8 binary products an IMMA, co32c's D at 7), and every route's fraction is chosen at
  once by an exact minimum over the vertices of a small linear program. O's route gains nothing at r = 32 and beats
  r = 32's ε at r = k/2 for any x below about 24; co32c's linears gain nothing from any route at fold 3 for x ≥ 4, or
  at cheaper folds for x ≥ 8.

Run from the repository root: `uv run --no-project --with numpy --with python-sat --with-editable
benchmarks/pouw/exact_kit python -B benchmarks/pouw/theory/checked_strip.py --mutations` (`--full` adds the 2×2
block's SAT case: 12 products, model checked, and 11 under a conflict budget, which so far leaves it undecided).
`benchmarks/pouw/tests/test_theory_checked_strip.py` runs it in the pouw suite.
"""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction

import numpy as np
from pysat.solvers import Solver

from exact_kit import f2
from exact_kit.checks import Checks
from exact_kit.sat import SOLVER, Cnf

M32 = np.uint32
MASK = (1 << 32) - 1
W_RE = np.array([1, 2, 4, 8, 16, 32, 65], dtype=np.int64)
W_EVEN = np.array([2, 2, 4, 8, 16, 32, 64], dtype=np.int64)
FP4 = {0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6}
DESIGNS = {"dither": (False, ("T",)), "checked1": (False, ("T", "O")), "co": (True, ("T", "O")),
           "co_p2": (True, ("T", "P2"))}


def s32(x):
    return (np.asarray(x, dtype=np.int64) & MASK).astype(M32)


class Block:
    """A block of n rows and n′ columns of one design, with its bit layout: U's planes, then c, then b."""

    def __init__(self, design, n, k, n2, r, L, w=W_RE):
        assert k % L == 0 and k % r == 0
        self.design, self.n, self.k, self.n2, self.r, self.L, self.w = design, n, k, n2, r, L, np.asarray(w)
        self.two, self.kinds = DESIGNS[design]
        self.np = len(self.w)
        self.nU = n * r * self.np
        self.nc = n * k if self.two else 0
        self.nb = k * n2
        self.nbits = self.nU + self.nc + self.nb
        self.left = list(range(self.nU + self.nc))
        self.right = list(range(self.nU + self.nc, self.nbits))
        self.T = k // L
        self.per_cell = self.T * ("T" in self.kinds) + sum(x != "T" for x in self.kinds)

    def __str__(self):
        return f"{self.design} (n, k, n', r, L) = ({self.n}, {self.k}, {self.n2}, {self.r}, {self.L})"

    def fields(self, Z):
        N = Z.shape[0]
        Z = Z.astype(np.int64)
        beta = Z[:, :self.nU].reshape(N, self.n, self.r, self.np)
        U = -64 + beta @ self.w
        c = Z[:, self.nU:self.nU + self.nc].reshape(N, self.n, self.k) if self.two else np.zeros((N, self.n, self.k),
                                                                                                    np.int64)
        b = Z[:, self.nU + self.nc:].reshape(N, self.k, self.n2)
        return U, c, b

    def words(self, P, V, Z):
        """Every committed word of every cell at the bit points Z (N × nbits), over Z/2³² (as int64 mod 2³²)."""
        U, c, b = self.fields(Z)
        Ph = P[None] + c
        Pt = Ph + U[:, :, np.arange(self.k) % self.r]
        X = V[None] + b
        out, Ts, acc = [], [], 0
        for t in range(self.T):
            sl = slice(t * self.L, (t + 1) * self.L)
            acc = acc + np.einsum("nik,nkl->nil", Pt[:, :, sl], X[:, sl])
            Ts.append(acc)
        O = np.einsum("nik,nkl->nil", Ph, X)
        Y = X.reshape(X.shape[0], self.k // self.r, self.r, self.n2).sum(1)
        P2 = np.einsum("nir,nrl->nil", U, Y)
        for i in range(self.n):
            for l in range(self.n2):
                for kind in self.kinds:
                    if kind == "T":
                        out.extend(T[:, i, l] for T in Ts)
                    else:
                        out.append((O if kind == "O" else P2)[:, i, l])
        return np.stack(out, axis=1) & MASK


def instance(B: Block, kind, rng, vbits=8):
    vlo, vhi = (-128, 126) if vbits == 8 else (-64, 62)
    if kind == "zero":
        return np.zeros((B.n, B.k), np.int64), np.zeros((B.k, B.n2), np.int64)
    if kind == "random":
        return rng.integers(-63, 63, (B.n, B.k)), rng.integers(vlo, vhi + 1, (B.k, B.n2))
    if kind == "balanced":
        P = rng.integers(-63, 63, (B.n, B.k))
        a = rng.integers(1, min(-vlo, vhi) + 1, (B.r, B.n2))
        sign = np.where(np.arange(B.k // B.r) % 2 == 0, 1, -1)
        V = (sign[:, None, None] * a[None]).reshape(B.k, B.n2)
        if (B.k // B.r) % 2:
            V[-B.r:] = 0
        return P, V
    raise ValueError(kind)


def unit(nbits, idx):
    Z = np.zeros((len(idx), nbits), np.uint8)
    for row, ii in enumerate(idx):
        for j in ii:
            Z[row, j] = 1
    return Z


def pair_diffs(B: Block, P, V, pairs, chunk=8192):
    """pairDiff of every word at every pair (x, y), x ≠ y: W(e_x + e_y) − W(e_x) − W(e_y) + W(0), mod 2³²."""
    W0 = B.words(P, V, np.zeros((1, B.nbits), np.uint8))[0]
    W1 = B.words(P, V, np.eye(B.nbits, dtype=np.uint8))
    out = np.empty((len(pairs), W0.shape[0]), np.int64)
    for s in range(0, len(pairs), chunk):
        part = pairs[s:s + chunk]
        Wxy = B.words(P, V, unit(B.nbits, part))
        xs = np.array([x for x, _ in part])
        ys = np.array([y for _, y in part])
        out[s:s + len(part)] = (Wxy - W1[xs] - W1[ys] + W0[None]) & MASK
    return out


def cross_ranks(B: Block, P, V):
    """(rank of rows (word, peeled bit) × served coins, rank of rows (word, served coin) × peeled bits, same-side
    pair differences all zero over Z/2³²), for a one-cell block."""
    pairs = [(x, y) for x in B.left for y in B.right]
    D = pair_diffs(B, P, V, pairs) & 1
    nw = D.shape[1]
    D = D.reshape(len(B.left), len(B.right), nw)
    rows1 = f2.from_matrix(D.transpose(2, 0, 1).reshape(nw * len(B.left), len(B.right)))
    rows2 = f2.from_matrix(D.transpose(2, 1, 0).reshape(nw * len(B.right), len(B.left)))
    same = [p for p in itertools.combinations(B.left, 2)] + [p for p in itertools.combinations(B.right, 2)]
    S = pair_diffs(B, P, V, same)
    return f2.rank(rows1), f2.rank(rows2), int(np.count_nonzero(S))


# ---------------------------------------------------------------- C: SAT tensor rank over F2


def f2_targets(B: Block, rng):
    """Each word's alternating form mod 2 (a set of pairs u < v), from pair differences at a random instance."""
    P, V = instance(B, "random", rng)
    pairs = list(itertools.combinations(range(B.nbits), 2))
    D = pair_diffs(B, P, V, pairs) & 1
    return [{pairs[q] for q in np.flatnonzero(D[:, w])} for w in range(D.shape[1])]


def sat_products(targets, nb, S, budget=None):
    """Is there a program of S products of F₂-affine operands (any bits on either side) whose linear combinations
    have exactly the target alternating forms? Returns (True, (ℓ, m, γ)), (False, None) or (None, None) on budget."""
    cnf = Cnf()
    ell = [cnf.word(f"l{s}", nb) for s in range(S)]
    m = [cnf.word(f"m{s}", nb) for s in range(S)]
    g = [[cnf.var(f"g{w},{s}") for s in range(S)] for w in range(len(targets))]
    for s in range(S):
        _echelon(cnf, ell[s], m[s])
    for s in range(S - 1):
        _lex_le(cnf, ell[s] + m[s], ell[s + 1] + m[s + 1])
    for u, v in itertools.combinations(range(nb), 2):
        q = [cnf.xor(cnf.and_(ell[s][u], m[s][v]), cnf.and_(ell[s][v], m[s][u])) for s in range(S)]
        for w, tw in enumerate(targets):
            cnf.parity([cnf.and_(g[w][s], q[s]) for s in range(S)], (u, v) in tw)
    with Solver(name=SOLVER, bootstrap_with=cnf.clauses) as sv:
        if budget is not None:
            sv.conf_budget(budget)
            res = sv.solve_limited()
        else:
            res = sv.solve()
        if res is None:
            return None, None
        if not res:
            return False, None
        mod = {abs(x): x > 0 for x in sv.get_model()}
    val = lambda lit: mod.get(abs(lit), False) == (lit > 0)
    L_ = [[val(x) for x in ell[s]] for s in range(S)]
    M_ = [[val(x) for x in m[s]] for s in range(S)]
    G_ = [[val(g[w][s]) for s in range(S)] for w in range(len(targets))]
    return True, (L_, M_, G_)


def _echelon(cnf: Cnf, ell, m):
    """(ℓ, m) is the reduced echelon basis of its span: ℓ ≠ 0, m's leading one after ℓ's, ℓ zero at m's leading one.
    A product's alternating form ℓ_u m_v + ℓ_v m_u is ℓ ∧ m, which depends only on span{ℓ, m}, so every product has
    this form (m = 0 is the zero product, kept for padding)."""
    cnf.add(ell)
    seen_l = seen_m = -cnf.true
    for v in range(len(ell)):
        cnf.add([-m[v], seen_l])
        cnf.add([-m[v], seen_m, -ell[v]])
        seen_l, seen_m = cnf.or_(seen_l, ell[v]), cnf.or_(seen_m, m[v])


def _lex_le(cnf: Cnf, a, b):
    """a ≤ b as bit strings, most significant first (symmetry breaking over the order of the products)."""
    eq = cnf.true
    for x, y in zip(a, b):
        cnf.add([-eq, -x, y])
        eq = cnf.and_(eq, -cnf.xor(x, y))


def check_program(targets, nb, prog):
    L_, M_, G_ = prog
    for w, tw in enumerate(targets):
        for u, v in itertools.combinations(range(nb), 2):
            bit = 0
            for s in range(len(L_)):
                if G_[w][s]:
                    bit ^= (L_[s][u] & M_[s][v]) ^ (L_[s][v] & M_[s][u])
            if bit != ((u, v) in tw):
                return False
    return True


# ---------------------------------------------------------------- E: priced strategies


def limbs(lo, hi):
    """int8 limbs an operand with values in [lo, hi] needs: 1 if it is s8 or u8, else the signed 8m-bit width."""
    if (-128 <= lo and hi <= 127) or (0 <= lo and hi <= 255):
        return 1
    m = 2
    while not (-(1 << (8 * m - 1)) <= lo and hi < (1 << (8 * m - 1))):
        m += 1
    return m


class Ledger:
    """A strategy's products, priced per cell (MACs per output cell times the unit's price)."""

    def __init__(self):
        self.items = []

    def int8(self, A, B, what):
        A, B = np.asarray(A), np.asarray(B)
        for M in (A, B):
            if not ((M.min() >= -128 and M.max() <= 127) or (M.min() >= 0 and M.max() <= 255)):
                raise ValueError(f"{what}: an operand is not s8 or u8")
        self.items.append((what, A.shape[-1], Fraction(1)))
        return A.astype(np.int64) @ B.astype(np.int64)

    def fp4(self, A, B, what):
        A, B = np.asarray(A), np.asarray(B)
        if not (set(np.unique(A).tolist()) <= FP4 and set(np.unique(B).tolist()) <= FP4):
            raise ValueError(f"{what}: an operand is not E2M1")
        out = A.astype(np.float32) @ B.astype(np.float32)
        if np.abs(out).max() >= 2 ** 24:
            raise ValueError(f"{what}: FP32 accumulation is not exact")
        self.items.append((what, A.shape[-1], Fraction(1, 2)))
        return out.astype(np.int64)

    def limbed(self, A, Y, what):
        """A·Y with Y split into int8 limbs (low limbs u8, the top one s8)."""
        lo, hi = int(Y.min()), int(Y.max())
        n = limbs(lo, hi)
        if n == 1:
            return self.int8(A, Y if lo >= -128 and hi <= 127 else Y, what)
        acc = np.zeros((A.shape[0], Y.shape[1]), np.int64)
        rest = Y.astype(np.int64)
        for j in range(n):
            part = rest & 255 if j < n - 1 else rest
            acc += self.int8(A, part, f"{what} limb {j}") << (8 * j)
            rest = rest >> 8
        return acc

    def price(self):
        return sum(K * p for _, K, p in self.items)


def n_digits(lo, hi):
    """Base-128 digits `digits` needs for values in [lo, hi]: the least m with −128^m ≤ lo and ⌊hi/128^{m−1}⌋ ≤ 127."""
    m = 1
    while not (lo >= -(128 ** m) and hi // 128 ** (m - 1) <= 127):
        m += 1
    return m


def digits(Y, z, m):
    """Y = Σ_d 128^d A_d with m int8 digits: A_d = (R_d mod 128) − 128 z_d for d < m − 1 and A_{m−1} = R_{m−1}, where
    R_0 = Y and R_{d+1} = ⌊R_d/128⌋ + z_d. Each fresh coin z_d (one an entry) picks one of the digit's two int8
    representatives; the last is taken as 0 where ⌊R/128⌋ = 127, so that the top digit stays int8."""
    R, A = np.asarray(Y, np.int64), []
    for d in range(m - 1):
        q = R >> 7
        zd = np.where(q == 127, 0, z[d]) if d == m - 2 else z[d]
        A.append((R & 127) - 128 * zd)
        R = q + zd
    A.append(R)
    return A


def co32c_words(P, V, U, c, b, z, r, L, m, led=None):
    """CO with committed digits: T_t, D_d = U·A_d (1 ≤ d < m) and O′ = T_last − U·A_0, for Y's digits A_d; the
    served output is O = O′ − Σ_d 128^d D_d. With a ledger, every product is priced as computed."""
    k = P.shape[1]
    mm = led.int8 if led is not None else (lambda A_, B_, what: A_.astype(np.int64) @ B_.astype(np.int64))
    Pt = P + c + U[:, np.arange(k) % r]
    X = V + b
    T, acc = [], np.zeros((P.shape[0], V.shape[1]), np.int64)
    for t in range(k // L):
        sl = slice(t * L, (t + 1) * L)
        acc = acc + mm(Pt[:, sl], X[sl], f"T segment {t}")
        T.append(acc.copy())
    A = digits(X.reshape(k // r, r, -1).sum(0), z, m)
    D = [mm(U, A[d], f"D_{d}") for d in range(1, m)]
    Op = T[-1] + mm(-U, A[0], "O′")
    return [s32(x) for x in T], [s32(x) for x in D], s32(Op)


def honest_words(P, V, U, c, b, r, L):
    k = P.shape[1]
    Ph = P + c
    Pt = Ph + U[:, np.arange(k) % r]
    X = V + b
    T = [Pt[:, :(t + 1) * L] @ X[:(t + 1) * L] for t in range(k // L)]
    O = Ph @ X
    return [s32(x) for x in T], s32(O)


def strategy(name, P, V, U, c, b, r, L):
    """Run one strategy; return its words and its ledger."""
    k = P.shape[1]
    led = Ledger()
    Ph = P + c
    Pt = Ph + U[:, np.arange(k) % r]
    X = V + b
    T, acc = [], np.zeros((P.shape[0], V.shape[1]), np.int64)
    for t in range(k // L):
        sl = slice(t * L, (t + 1) * L)
        acc = acc + led.int8(Pt[:, sl], X[sl], f"T segment {t}")
        T.append(acc.copy())
    if name == "strip":
        Y = X.reshape(k // r, r, -1).sum(0)
        O = T[-1] - led.limbed(U, Y, "P2 = U·Y")
    elif name == "direct":
        O = led.int8(Ph, X, "O = P̂X")
    elif name == "fp4":
        O = led.fp4(Ph, X, "O = P̂X in E2M1")
    else:
        raise ValueError(name)
    return [s32(x) for x in T], s32(O), led


def draw(rng, n, k, n2, r, two=True):
    beta = rng.integers(0, 2, (n, r, 7))
    U = -64 + beta @ W_RE
    c = rng.integers(0, 2, (n, k)) if two else np.zeros((n, k), np.int64)
    b = rng.integers(0, 2, (k, n2))
    return U, c, b


# ---------------------------------------------------------------- F: served error moments


def error_moments(P, V, two):
    """Exact mean and variance of Σ_κ ((P + c)(V + b) − PV) over uniform coins, at every coin point."""
    k = len(P)
    vals = []
    for cs in itertools.product((0, 1), repeat=k if two else 0):
        cs = cs if two else (0,) * k
        for bs in itertools.product((0, 1), repeat=k):
            vals.append(sum((P[j] + cs[j]) * (V[j] + bs[j]) - P[j] * V[j] for j in range(k)))
    N = len(vals)
    mean = Fraction(sum(vals), N)
    var = Fraction(sum(v * v for v in vals), N) - mean * mean
    return mean, var


def theorem_n(q, theta):
    return (1 - theta) / (q - 1 + theta)


# ---------------------------------------------------------------- H: the tensor–scalar model (note:20261017T0310Z)

#: (row, MMA coefficient, coefficient of every other vector-pipe warp instruction) in SMSP-cycles:
#: `M` (the tensor pipe) and JI3e_0..2, benchmarks/pouw/machine_model/data/rows.csv on main.
TS_FACETS = [("M", 16.0, 0.0), ("JI3e_0", 8.115, 1.0), ("JI3e_1", 15.030, 0.317), ("JI3e_2", 15.539, 0.172)]
#: Llama-3.1 linears as served by vLLM (q, k, v fused; gate, up fused): (name, k, n).
TS_SHAPES = {"Llama-3.1-8B": [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672), ("down", 14336, 4096)],
             "Llama-3.1-70B": [("qkv", 8192, 10240), ("o", 8192, 8192), ("gate_up", 8192, 57344),
                               ("down", 28672, 8192)]}


def ts_time(M, O):
    return max(a * M + b * O for _, a, b in TS_FACETS)


def ts_binding(M, O):
    return max(TS_FACETS, key=lambda f: f[1] * M + f[2] * O)[0]


def ts_tile(design, k, n, fold):
    """((MMAs, other warp instructions) of honest, of the cheapest cheater found) for one m16n8 tile of 128 cells at
    depth k, n rows sharing the served side's residue sums. A word group (128 words) is folded at `fold` instructions a
    word, 4·fold warp instructions (4 words a lane). Honest's residue sums are one add an X entry, k·n′ adds for n·n′
    cells (4k/n a tile; 2k/n at r = k/2); the cheater's are taken as free (its b-only sums are popcounts). co32c
    commits each digit's product (`co32c_words`), so the cheater does every MMA honest does and saves only the
    served output's recombination O = O′ − Σ 128^d D_d (4 IMADs a digit) and the residue sums."""
    g = 4 * fold
    if design == "co32":
        honest = (k // 32 + 2, g * (k // 32 + 1) + 4 + 4 * k / n)
        cheat = (k // 32 + 1, g * (k // 32 + 1))
    elif design == "cohalf":
        honest = (k // 32 + k // 64, g * (k // 32 + 1) + 2 * k / n)
        cheat = (k // 32 + k // 64, g * (k // 32 + 1))
    elif design == "co32c":
        m = n_digits((k // 32) * -128, (k // 32) * 127)
        honest = (k // 32 + m, g * (k // 32 + m) + 4 * (m - 1) + 4 * k / n)
        cheat = (k // 32 + m, g * (k // 32 + m))
    elif design == "dither":
        honest = (k // 32 + 2, g * (k // 32) + 4 + 4 * k / n)
        cheat = (k // 32, g * (k // 32))
    else:
        raise ValueError(design)
    return honest, cheat


#: SMSP-cycles a warp b1 and.popc m16n8k256 MMA takes on sm_120, which emulates it (note:20261015T0608Z-draft-census-sm120).
B1_EMULATED = 193.0


def ts_mix(design, k, n, fold, dM, dO):
    """(honest's time, the best cheater's) at A = B = 0 when the cheater may move any fraction φ of its cells' O off
    its strip, changing a tile's (MMAs, other instructions) by φ·(dM, dO). The time is convex and piecewise linear in
    φ, so its minimum is at φ ∈ {0, 1} or where two facets meet."""
    hon, (Mc, Oc) = ts_tile(design, k, n, fold)
    cands = {0.0, 1.0}
    for (_, a1, b1), (_, a2, b2) in itertools.combinations(TS_FACETS, 2):
        den = (a1 - a2) * dM + (b1 - b2) * dO
        if den:
            phi = -((a1 - a2) * Mc + (b1 - b2) * Oc) / den
            if 0 < phi < 1:
                cands.add(phi)
    return ts_time(*hon), min(ts_time(Mc + p * dM, Oc + p * dO) for p in cands)


def o_routes(design, k):
    """At A = B = 0, O = c·b is a depth-k binary product with no commit inside. Each route replaces the cheater's
    strip (1 IMMA at r = 32, k/64 at r = k/2) by: `scalar(x)`, bit-sliced on the CUDA cores at x warp instructions per
    32 × 128 bit products; `b1_emulated`, sm_120's b1 MMAs priced as issue; `b1_native`, b1 MMAs at an IMMA's cost
    (sm_80 to sm_89); `fp4`, FP4 k64 MMAs plus NOBITS (one instruction a word). Returns {route: (dM, dO)} with
    `scalar` a function of x."""
    strip = k // 64 if design == "cohalf" else 1
    return {"scalar": lambda x: (-strip, x * k / 32), "b1_emulated": (-strip, B1_EMULATED * k / 256),
            "b1_native": (-strip + k / 256, 0.0), "fp4": (-strip + k / 64, 4.0)}


def word_routes(design, k, x):
    """At A = B = 0 the other committed words are narrow products too, each a k32 IMMA honest pays: a T_t increment is
    Σ (c + U)·b, 8 binary products (c and U's 7 planes; Σb is shared by the rows), and co32c's D_d = U·A_d is U's 7
    planes against fresh carry bits. Bit-sliced at x per binary product: [(dM, dO)] for all the T words, then all the
    D words."""
    routes = [(-(k // 32), 8 * x * (k // 32))]
    if design == "co32c":
        m = n_digits((k // 32) * -128, (k // 32) * 127)
        routes.append((-(m - 1), 7 * x * (m - 1)))
    return routes


def ts_best(design, k, n, fold, routes):
    """(honest's time, the best cheater's) when the cheater moves any fraction φ_j ∈ [0, 1] of each route's work off
    its tile, changing (MMAs, other instructions) by Σ φ_j (dM_j, dO_j). Minimising the max of the facets over the box
    is a linear program in (φ, t); its optimum is a vertex, where len(routes) + 1 of the facet and box constraints
    are tight, so every such vertex is solved for and the feasible minimum taken."""
    hon, (Mc, Oc) = ts_tile(design, k, n, fold)
    R = len(routes)
    dM = np.array([d[0] for d in routes], float)
    dO = np.array([d[1] for d in routes], float)
    # Each constraint is a·(φ, t) = rhs when tight: a facet t = α(Mc + φ·dM) + β(Oc + φ·dO), or a box face φ_j ∈ {0, 1}.
    cons = [(np.r_[a * dM + b * dO, -1.0], -(a * Mc + b * Oc)) for _, a, b in TS_FACETS]
    cons += [(np.r_[np.eye(R)[j], 0.0], v) for j in range(R) for v in (0.0, 1.0)]
    best = ts_time(Mc, Oc)
    for tight in itertools.combinations(cons, R + 1):
        A = np.array([c_[0] for c_ in tight])
        if not any(c_[0][-1] for c_ in tight) or abs(np.linalg.det(A)) < 1e-12:
            continue
        phi = np.linalg.solve(A, np.array([c_[1] for c_ in tight]))[:R]
        if np.all(phi >= -1e-12) and np.all(phi <= 1 + 1e-12):
            best = min(best, ts_time(Mc + phi @ dM, Oc + phi @ dO))
    return ts_time(*hon), best


def main():
    checks = Checks()
    rng = np.random.default_rng(20261017)
    out = {}

    # ------------------------------------------------------------ A
    n, k, n2, r, L = 4, 256, 4, 32, 32
    for two in (False, True):
        P, V = rng.integers(-63, 63, (n, k)), rng.integers(-128, 127, (k, n2))
        U, c, b = draw(rng, n, k, n2, r, two)
        T, O = honest_words(P, V, U, c, b, r, L)
        Y = (V + b).reshape(k // r, r, n2).sum(0)
        P2 = s32(U @ Y)
        checks.check(f"A1 O = T_last − P2 in s32 ({'two' if two else 'one'}-sided, k={k}, r={r})",
                     np.array_equal(O, s32(T[-1].astype(np.int64) - P2.astype(np.int64))))
    alphabet = sorted({-64 + int(np.dot(bits, W_RE)) for bits in itertools.product((0, 1), repeat=7)})
    Ph_rng = (-63, 62 + 1)
    Pt_rng = (Ph_rng[0] + alphabet[0], Ph_rng[1] + alphabet[-1])
    X_rng = (-128, 126 + 1)
    checks.check("A2 U's 128 letters lie in [−64, 64]; P̂ = P + c, P̃ = P̂ + U and X = V + b are int8",
                 len(alphabet) == 128 and -128 <= Pt_rng[0] and Pt_rng[1] <= 127 and X_rng[1] <= 127,
                 f"U in [{alphabet[0]}, {alphabet[-1]}], P̂ in {list(Ph_rng)}, P̃ in {list(Pt_rng)}, X in {list(X_rng)}")
    lt = {}
    for kk, rr, vb in [(4096, 32, 8), (14336, 32, 8), (128, 32, 8), (128, 64, 8), (128, 64, 7), (4096, 2048, 7)]:
        g = kk // rr
        vlo, vhi = (-128, 126) if vb == 8 else (-64, 62)
        worst = limbs(g * vlo, g * (vhi + 1))
        zero = limbs(0, g)
        lt[(kk, rr, vb)] = (worst, zero)
    out["limbs"] = {f"k={a},r={b_},served={c_}bit": v for (a, b_, c_), v in lt.items()}
    g = 14336 // 32
    hoeff = 2 * np.exp(-2 * 127 ** 2 / g)
    checks.check("A3 strip operand Y (r = 32, 8-bit X): k = 4096 worst 2 limbs, zero 1; k = 14336 worst 3, zero 2 raw, "
                 "and 1 once centred (Y − k/2r is s8 but with probability ≤ 2e^{−2·127²/(k/r)})",
                 lt[(4096, 32, 8)] == (2, 1) and lt[(14336, 32, 8)] == (3, 2) and hoeff < 1e-30,
                 f"{out['limbs']}; centring miss ≤ {hoeff:.1e}")
    checks.check("A4 r = k/2 with a 7-bit served side: Y is int8 at every instance (1 limb)",
                 lt[(128, 64, 7)] == (1, 1) and lt[(4096, 2048, 7)] == (1, 1) and lt[(128, 64, 8)] == (2, 1))
    B = Block("co", 1, k, n2, r, L)
    Pb, Vb = instance(B, "balanced", rng)
    _, _, bb = draw(rng, 1, k, n2, r)
    Yb = (Vb + bb).reshape(k // r, r, n2).sum(0)
    checks.check("A5 structured instance (Y(V) = 0): Y = residue sums of b, in [0, k/r], 1 limb",
                 np.array_equal(Yb, bb.reshape(k // r, r, n2).sum(0)) and limbs(int(Yb.min()), int(Yb.max())) == 1)
    dig, dig_ok = {}, True
    for kk in (256, 4096, 8192, 14336, 28672):
        lo, hi = (kk // 32) * -128, (kk // 32) * 127
        m = n_digits(lo, hi)
        dig[kk] = m
        Ys = np.concatenate([np.arange(lo, lo + 300), np.arange(hi - 300, hi + 1), np.arange(-300, 300),
                             rng.integers(lo, hi + 1, 2000)])
        for zs in (np.zeros((m - 1, Ys.size), np.int64), np.ones((m - 1, Ys.size), np.int64),
                   rng.integers(0, 2, (m - 1, Ys.size))):
            A = digits(Ys, zs, m)
            dig_ok &= all(int(a.min()) >= -128 and int(a.max()) <= 127 for a in A)
            dig_ok &= np.array_equal(sum(a * 128 ** d for d, a in enumerate(A)), Ys)
    out["digits"] = dig
    checks.check("A6 the strip operand's redundant base-128 digits (r = 32, 8-bit X): every digit int8 and Σ 128^d A_d = "
                 "Y at both ends of the range, around 0 and at random, for coins all 0, all 1 and random",
                 dig_ok and dig == {256: 2, 4096: 2, 8192: 3, 14336: 3, 28672: 3}, f"digits {dig}")
    nz, k6, r6 = 16, 256, 32
    m6 = n_digits((k6 // r6) * -128, (k6 // r6) * 127)
    U6, c6, b6 = draw(rng, nz, k6, nz, r6)
    z6 = rng.integers(0, 2, (m6 - 1, r6, nz))
    Z6 = np.zeros((nz, k6), np.int64), np.zeros((k6, nz), np.int64)
    T6, D6, Op6 = co32c_words(*Z6, U6, c6, b6, z6, r6, L, m6)
    Pr, Vr = rng.integers(-63, 63, (nz, k6)), rng.integers(-128, 127, (k6, nz))
    Tr, Dr, Opr = co32c_words(Pr, Vr, U6, c6, b6, z6, r6, L, m6)
    served = lambda Op, D: s32(Op.astype(np.int64) - sum(Dd.astype(np.int64) << (7 * (d + 1)) for d, Dd in enumerate(D)))
    checks.check("A7 committed digits: the served output O = O′ − Σ 128^d D_d in s32 (random and zero instance); at "
                 "A = B = 0 D_1 = U·z_0 (Y < 128 there), a fresh word, nonzero in ≥ 95% of 256 cells",
                 np.array_equal(served(Opr, Dr), s32((Pr + c6) @ (Vr + b6))) and np.array_equal(served(Op6, D6),
                 s32(c6 @ b6)) and np.array_equal(D6[0], s32(U6 @ z6[0])) and np.count_nonzero(D6[0]) >= 0.95 * nz * nz,
                 f"m = {m6}, D_1 nonzero in {np.count_nonzero(D6[0])} of {nz * nz} cells")

    # ------------------------------------------------------------ B
    one = lambda k, r, L: ((k // L) * min(r, L), r)
    want = {"dither": one, "checked1": one, "co": lambda k, r, L: (k, k + r), "co_p2": lambda k, r, L: (k, k + r)}
    rank_tab = {}
    for (kk, rr, LL) in [(128, 32, 32), (64, 32, 32), (96, 16, 32)]:
        for design in DESIGNS:
            Bc = Block(design, 1, kk, 1, rr, LL)
            res = []
            for kind in ("zero", "random", "balanced"):
                P, V = instance(Bc, kind, rng)
                res.append(cross_ranks(Bc, P, V))
            r1, r2, same = res[0]
            w_ = want[design](kk, rr, LL)
            rank_tab[f"{design} k={kk} r={rr} L={LL}"] = [r1, r2]
            checks.check(f"B {Bc}: cross ranks (peeled rows, coin rows) = {w_}, the same at the zero, a random and a "
                         f"structured instance; same-side blocks 0",
                         all(x[:2] == w_ for x in res) and all(x[2] == 0 for x in res),
                         f"{[x[:2] for x in res]}, same-side nonzero {[x[2] for x in res]}; honest count {kk + rr}")
    out["ranks"] = rank_tab
    if checks.mutations:
        for name, kw in [("co with an even alphabet (U's parity constant)", {"w": W_EVEN}),
                         ("co without c (one-sided)", {"design": "checked1"})]:
            Bm = Block(kw.get("design", "co"), 1, 64, 1, 32, 32, w=kw.get("w", W_RE))
            P, V = instance(Bm, "random", rng)
            r1, r2, _ = cross_ranks(Bm, P, V)
            with checks.mutant(name):
                checks.check(f"M {name}: certified max rank = k + r", max(r1, r2) == 64 + 32, f"ranks {r1}, {r2}")

    # ------------------------------------------------------------ C
    sat_tab = {}
    cases = [("co", 1, 2, 1, 1, 1, 3), ("co", 1, 3, 1, 1, 1, 4), ("co", 1, 4, 1, 2, 2, 6), ("co", 1, 4, 1, 1, 2, 5),
             ("co", 2, 2, 1, 1, 1, 6), ("co", 1, 2, 2, 1, 1, 6), ("dither", 1, 4, 1, 2, 2, 4),
             ("checked1", 1, 4, 1, 2, 2, 4), ("co_p2", 1, 4, 1, 2, 2, 6)]
    for design, nn, kk, nn2, rr, LL, target in cases:
        Bs = Block(design, nn, kk, nn2, rr, LL, w=np.array([1]))
        tg = f2_targets(Bs, rng)
        lo_ok, _ = sat_products(tg, Bs.nbits, target - 1)
        hi_ok, prog = sat_products(tg, Bs.nbits, target)
        valid = hi_ok and check_program(tg, Bs.nbits, prog)
        sat_tab[str(Bs)] = target
        honest = nn * nn2 * (kk + rr)
        checks.check(f"C {Bs}: F2 tensor rank (affine products, no purity) = {target}; honest count {honest}",
                     lo_ok is False and valid, f"{target - 1} UNSAT, {target} SAT (model checked)")
    if "--full" in checks.args:
        Bs = Block("co", 2, 2, 2, 1, 1, w=np.array([1]))
        tg = f2_targets(Bs, rng)
        lo_ok, _ = sat_products(tg, Bs.nbits, 11, budget=3_000_000)
        hi_ok, prog = sat_products(tg, Bs.nbits, 12)
        sat_tab[str(Bs)] = {"11": {False: "UNSAT", None: "undecided"}.get(lo_ok, "SAT"), "12": bool(hi_ok)}
        checks.check(f"C {Bs}: 12 products suffice (model checked); 11: {sat_tab[str(Bs)]['11']}",
                     hi_ok and check_program(tg, Bs.nbits, prog) and lo_ok is not True)
    out["sat"] = sat_tab

    # ------------------------------------------------------------ D
    P0 = np.zeros((n, k), np.int64)
    V = rng.integers(-128, 127, (k, n2))
    U, c, b = draw(rng, n, k, n2, r, two=False)
    T, O = honest_words(P0, V, U, c, b, r, L)
    Y = (V + b).reshape(k // r, r, n2).sum(0)
    checks.check("D1 one-sided at P = 0: O = 0 and P2 = T_last (the checked strip is free)",
                 not O.any() and np.array_equal(s32(U @ Y), T[-1]))
    R, S = rng.integers(0, 2 ** 32, (n, k)), rng.integers(0, 2 ** 32, (k, n2))
    A, Bm = rng.integers(0, 2 ** 32, (n, k)), rng.integers(0, 2 ** 32, (k, n2))
    def mm(X_, Y_):
        X_, Y_ = np.asarray(X_, np.int64) & MASK, np.asarray(Y_, np.int64) & MASK
        lo = (X_ & 0xFFFF) @ (Y_ & 0xFFFF)
        mid = ((X_ >> 16) @ (Y_ & 0xFFFF) + (X_ & 0xFFFF) @ (Y_ >> 16)) & 0xFFFF
        return s32(lo + (mid << 16))
    M = [mm(A + R, Bm + S), mm(R, Bm + S), mm(A + R, S), mm(R, S)]
    rec = s32(M[0].astype(np.int64) - M[1] - M[2] + M[3])
    checks.check("D2 KSW: (A+R)(B+S) − R(B+S) − (A+R)S + RS = AB over Z/2^32", np.array_equal(rec, mm(A, Bm)))
    Z0 = [mm(R, S), mm(R, S), mm(R, S), mm(R, S)]
    M0 = [mm(0 * A + R, 0 * Bm + S), mm(R, 0 * Bm + S), mm(0 * A + R, S), mm(R, S)]
    checks.check("D3 KSW at A = B = 0: all four checked products equal RS (one product, not four)",
                 all(np.array_equal(x, y) for x, y in zip(M0, Z0)))
    q = 4
    nv = 3
    forms = []
    for i in range(q):
        Mf = np.zeros((q * nv, q * nv), np.int64)
        Mf[i * nv:(i + 1) * nv, i * nv:(i + 1) * nv] = rng.integers(1, 3, (nv, nv))
        forms.append(Mf.ravel())
    checks.check("D4 independent noise per product: the q noise–noise forms are independent, so no combination with "
                 "Σλ = 1 cancels them (a recovery needs unchecked strips)",
                 np.linalg.matrix_rank(np.stack(forms).astype(float)) == q)
    Pz, Vz = np.zeros((n, k), np.int64), np.zeros((k, n2), np.int64)
    U, c, b = draw(rng, n, k, n2, r)
    T, O = honest_words(Pz, Vz, U, c, b, r, L)
    checks.check("D5 CO at the zero instance: O = c·b (the random 0/1 product) = T_last − U·Y(b)",
                 np.array_equal(O, s32(c @ b)) and np.array_equal(O, s32(T[-1].astype(np.int64)
                                                                          - U @ b.reshape(k // r, r, n2).sum(0))))

    # ------------------------------------------------------------ E
    price = {}
    for rr, vb in [(32, 8), (128, 7)]:
        vlo, vhi = (-128, 126) if vb == 8 else (-64, 62)
        rows = {}
        for inst, strat in [("random", "strip"), ("random", "direct"), ("zero", "strip"), ("zero", "fp4"),
                            ("balanced", "strip"), ("small", "fp4")]:
            Bc = Block("co", n, k, n2, rr, L)
            if inst == "small":
                P, V = np.zeros((n, k), np.int64), rng.choice([1, 3], (k, n2))
            else:
                P, V = instance(Bc, inst, rng, vbits=vb)
            if inst == "random":
                V = np.where(rng.random((k, n2)) < 0.5, vlo, vhi)
            U, c, b = draw(rng, n, k, n2, rr)
            Th, Oh = honest_words(P, V, U, c, b, rr, L)
            Ts, Os, led = strategy(strat, P, V, U, c, b, rr, L)
            ok = all(np.array_equal(x, y) for x, y in zip(Th, Ts)) and np.array_equal(Oh, Os)
            rows[f"{inst}/{strat}"] = led.price()
            checks.check(f"E r={rr} {vb}-bit X, {inst} instance, {strat}: every word equals the honest word",
                         ok, f"price {led.price()} a cell (k = {k}, r = {rr})")
        price[f"r={rr},{vb}bit"] = {a: str(v) for a, v in rows.items()}
        hon = rows["random/strip"]
        adv = min(v for a, v in rows.items() if a != "random/strip")
        if rr == 32:
            checks.check("E r = 32: honest k + 2r; best cheater k + r (zero or structured instance, 1-limb strip)",
                         hon == k + 2 * rr and adv == k + rr, f"{hon} vs {adv}")
        else:
            checks.check("E r = k/2, 7-bit X: honest k + k/2, every enumerated cheater ≥ k + k/2 (tie)",
                         hon == k + k // 2 and adv == k + k // 2, f"{hon} vs {adv}")
    out["prices_test"] = price
    m = n_digits((k // 32) * -128, (k // 32) * 127)
    U, c, b = draw(rng, n, k, n2, 32)
    z = rng.integers(0, 2, (m - 1, 32, n2))
    cprice = {}
    for inst in ("random", "zero", "balanced"):
        P, V = instance(Block("co", n, k, n2, 32, L), inst, rng)
        if inst == "random":
            V = np.where(rng.random((k, n2)) < 0.5, -128, 126)
        led = Ledger()
        got = co32c_words(P, V, U, c, b, z, 32, L, m, led)
        want = co32c_words(P, V, U, c, b, z, 32, L, m)
        same = all(np.array_equal(x, y) for x, y in zip(got[0] + got[1] + [got[2]], want[0] + want[1] + [want[2]]))
        Y = (V + b).reshape(k // 32, 32, n2).sum(0)
        skip = s32(want[0][-1].astype(np.int64) - U @ Y)
        cprice[inst] = (led.price(), same, not np.array_equal(skip, want[2]))
    out["prices_co32c"] = {a: str(v[0]) for a, v in cprice.items()}
    checks.check("E committed digits at r = 32 (k = 256): every product is a committed word's, so honest and the cheater "
                 "at the zero and the structured instance pay k + m·r alike, and the 1-limb shortcut O′ = T_last − U·Y "
                 "gives a wrong O′", all(v[0] == k + m * 32 and v[1] and v[2] for v in cprice.values()),
                 f"m = {m}, prices {out['prices_co32c']}")

    def eps_cell(kk, rr, vb):
        g = kk // rr
        vlo, vhi = (-128, 126) if vb == 8 else (-64, 62)
        hon = kk + min(limbs(g * vlo, g * (vhi + 1)), 2) * rr
        adv = min(kk + rr, kk + Fraction(kk, 2), 2 * kk)
        return hon, adv, Fraction(hon) / adv - 1

    tab = {}
    for name, kk, rr, vb in [("linear k=4096", 4096, 32, 8), ("linear k=8192", 8192, 32, 8),
                             ("linear k=14336", 14336, 32, 8), ("linear k=28672", 28672, 32, 8),
                             ("QK^T k=128, r=32", 128, 32, 8), ("QK^T k=128, r=64, 7-bit", 128, 64, 7),
                             ("PV ctx=2048", 2048, 32, 8), ("PV ctx=8192", 8192, 32, 8),
                             ("PV ctx=32768", 32768, 32, 8), ("linear k=4096, r=2048, 7-bit", 4096, 2048, 7)]:
        hon, adv, e = eps_cell(kk, rr, vb)
        dither = Fraction(kk + min(limbs((kk // rr) * -128, (kk // rr) * 127), 2) * rr, kk) - 1
        tab[name] = {"honest": str(hon), "adversary": str(adv), "eps_CO": f"{float(e):.5%}",
                     "gamma_CO_macs": f"{float(Fraction(hon, kk) - 1):.4%}", "eps_dither": f"{float(dither):.4%}"}
    out["eps_table"] = tab
    checks.check("E ε table: r = 32 linears ε = r/(k + r) (0.775% at 4096, 0.223% at 14336); 7-bit r = k/2: 0",
                 tab["linear k=4096"]["eps_CO"] == f"{32 / 4128:.5%}" and tab["linear k=14336"]["eps_CO"]
                 == f"{32 / 14368:.5%}" and tab["QK^T k=128, r=64, 7-bit"]["eps_CO"] == f"{0:.5%}", str(tab))
    models = {"Llama-3.1-8B": (4096, 1024, 14336), "Llama-3.1-70B": (8192, 1024, 28672)}
    mw = {}
    for mname, (h, kv, ff) in models.items():
        lin = [(h, h), (h, kv), (h, kv), (h, h), (h, ff), (h, ff), (ff, h)]
        H = sum(nn * eps_cell(kk, 32, 8)[0] for kk, nn in lin)
        Aa = sum(nn * eps_cell(kk, 32, 8)[1] for kk, nn in lin)
        D = sum(nn * (kk + 2 * 32) for kk, nn in lin)
        Dd = sum(nn * kk for kk, nn in lin)
        mw[mname] = {"eps_CO_linears": f"{float(Fraction(H) / Aa - 1):.4%}",
                     "eps_dither_linears_2limb": f"{float(Fraction(D, Dd) - 1):.4%}"}
    out["model_weighted"] = mw
    checks.check("E Llama-3.1 linears, MAC-weighted CO ε < 1%", all(float(v["eps_CO_linears"][:-1]) < 1
                                                                      for v in mw.values()), str(mw))

    # ------------------------------------------------------------ F
    bad = 0
    for _ in range(6):
        kk = 3
        P = rng.integers(-63, 63, kk).tolist()
        V = rng.integers(-128, 127, kk).tolist()
        mean, var = error_moments(P, V, True)
        f_mean = Fraction(sum(V), 2) + Fraction(sum(P), 2) + Fraction(kk, 4)
        f_var = sum(Fraction(v * v + p * p + v + p, 4) + Fraction(3, 16) for p, v in zip(P, V))
        m1, v1 = error_moments(P, V, False)
        bad += (mean, var) != (f_mean, f_var) or (m1, v1) != (Fraction(sum(P), 2), Fraction(sum(p * p for p in P), 4))
    checks.check("F served error over the coins: CO mean ½ΣV + ½ΣP + k/4, variance Σ(V² + P² + V + P)/4 + 3k/16; "
                 "Dither mean ½ΣP, variance ΣP²/4", bad == 0, f"{bad} mismatches over 6 instances (every coin point)")
    EV2 = Fraction(sum(v * v for v in range(-128, 127)), 255)
    EP2 = Fraction(sum(p * p for p in range(-63, 63)), 126)
    EV2_7 = Fraction(sum(v * v for v in range(-64, 63)), 127)
    ratios = {"uniform entries, 8-bit served": float((EV2 + EP2 + Fraction(3, 4)) / EP2),
              "same rms in LSBs on both sides": 2.0,
              "served rms half the peeled rms (outlier-heavy activations)": 1.25}
    out["variance_ratio_CO_over_dither"] = ratios
    out["relative_error_note"] = ("relative to the cell's signal, Dither's error power is 1/(4 σ_V²) and CO's is "
                                  "1/(4 σ_V²) + 1/(4 σ_P²) (σ in LSBs): ratio 1 + σ_V²/σ_P²")
    checks.check("F CO/Dither error variance ratio = 1 + E[V²]/E[P²] + O(1/E[P²])", ratios["uniform entries, "
                 "8-bit served"] > 1, str(ratios))

    # ------------------------------------------------------------ G
    gt = {}
    for qq in (2, 4, 10, 11, 51, 501):
        gt[qq] = {"theta=0": f"{theorem_n(qq, 0):.4%}", "theta=1/2": f"{theorem_n(qq, 0.5):.4%}"}
    out["theorem_n"] = gt
    need = {e: next(qq for qq in itertools.count(2) if theorem_n(qq, 0.5) < e) for e in (0.01, 0.001)}
    out["theorem_n_q_needed"] = need
    checks.check("G Theorem N at γ ≤ 10 (q ≤ 11 checked products, θ = ½ for a 0/1 dither product): ε ≥ 4.76%",
                 abs(theorem_n(11, 0.5) - 0.5 / 10.5) < 1e-12 and theorem_n(11, 0.5) > 0.047, str(gt))
    checks.check("G ε < 1% needs q ≥ 51 checked products (γ ≥ 50); ε < 0.1% needs q ≥ 501",
                 need == {0.01: 51, 0.001: 501}, str(need))
    checks.check("G KSW (q = 4, θ = 0, exact recovery): the bound 1/3 ≤ the actual 3 (four products collapse to one)",
                 theorem_n(4, 0) <= 3)

    # ------------------------------------------------------------ H
    ts = {}
    for fold in (3, 1.5, 1):
        for mname, shapes in TS_SHAPES.items():
            rows = {}
            tot = {d: [0.0, 0.0, 0.0] for d in ("co32", "co32c", "cohalf", "dither")}
            for name, kk, nn in shapes:
                for d in tot:
                    (Mh, Oh), (Mc, Oc) = ts_tile(d, kk, nn, fold)
                    th, tc = ts_time(Mh, Oh), ts_time(Mc, Oc)
                    tot[d][0] += th * nn
                    tot[d][1] += tc * nn
                    tot[d][2] += 16 * (kk // 32) * nn
                    rows[f"{name} {d}"] = f"{th / tc - 1:.4%}"
                    rows[f"{name} gamma {d} over plain int8"] = f"{th / (16 * (kk // 32)) - 1:.4%}"
            for d, (h, c_, p_) in tot.items():
                rows[f"weighted {d}"] = f"{h / c_ - 1:.4%}"
                rows[f"weighted gamma {d} over plain int8"] = f"{h / p_ - 1:.4%}"
            ts[f"fold={fold} {mname}"] = rows
    out["ts"] = ts
    w8 = ts["fold=3 Llama-3.1-8B"]
    checks.check("H TS model, duo fold (3 a word): CO r = 32 ε < 1% on every Llama-3.1-8B linear; r = k/2 smaller",
                 all(float(v[:-1]) < 1 for kk_, v in w8.items() if kk_.endswith(" co32")) and
                 float(w8["weighted cohalf"][:-1]) < float(w8["weighted co32"][:-1]), str(w8))
    k4 = ts_tile("co32", 4096, 4096, 3)
    checks.check("H honest at r = 32 is issue-bound under the duo fold (JI3e_0 binds), so the second limb costs its "
                 "issue (8.115 + 4) rather than its 16 tensor cycles",
                 ts_binding(*k4[0]) == "JI3e_0", f"binding {ts_binding(*k4[0])}, times {ts_time(*k4[0]):.1f} vs "
                                                  f"{ts_time(*k4[1]):.1f} SMSP-cycles a tile")

    def weighted(design, route_of_k, mname):
        h = c_ = 0.0
        for _, kk, nn in TS_SHAPES[mname]:
            th, tb = ts_mix(design, kk, nn, 3, *route_of_k(kk))
            h, c_ = h + th * nn, c_ + tb * nn
        return h / c_ - 1

    def breakeven(design, kk, nn):
        """The least x (to 0.01) at which moving O to scalar units gains nothing over the strip-only cheater."""
        strip_only = ts_time(*ts_tile(design, kk, nn, 3)[1])
        lo, hi = 0.0, 256.0
        while hi - lo > 0.01:
            mid = (lo + hi) / 2
            if ts_mix(design, kk, nn, 3, *o_routes(design, kk)["scalar"](mid))[1] < strip_only - 1e-9:
                lo = mid
            else:
                hi = mid
        return hi

    zero = {}
    for design in ("co32", "co32c", "cohalf"):
        for mname in TS_SHAPES:
            for x in (8, 12, 16, 24):
                zero[f"{design} {mname} scalar x={x}"] = weighted(design, lambda kk: o_routes(design, kk)["scalar"](x),
                                                                  mname)
            for route in ("b1_emulated", "b1_native", "fp4"):
                zero[f"{design} {mname} {route}"] = weighted(design, lambda kk: o_routes(design, kk)[route], mname)
            zero[f"{design} {mname} breakeven x"] = max(breakeven(design, kk, nn) for _, kk, nn in TS_SHAPES[mname])
        for x in (8, 12, 16, 24):
            th, tb = ts_mix(design, 128, 4096, 3, *o_routes(design, 128)["scalar"](x))
            zero[f"{design} QK^T scalar x={x}"] = th / tb - 1
    out["ts_zero_instance_O"] = {kk_: (f"{v:.4%}" if "breakeven" not in kk_ else f"{v:.2f}") for kk_, v in zero.items()}
    base = {(d, m): weighted(d, lambda kk: (0.0, 0.0), m) for d in ("co32", "co32c", "cohalf") for m in TS_SHAPES}
    checks.check("H r = 32 (co32, co32c) at A = B = 0: making O = c·b on scalar units (any x ≥ 2), with b1 MMAs "
                 "(emulated or native) or with FP4 gains nothing over the cheater's one-IMMA strip",
                 all(abs(zero[f"{d} {m} {r}"] - base[(d, m)]) < 1e-12 for d in ("co32", "co32c") for m in TS_SHAPES
                     for r in ("scalar x=8", "b1_emulated", "b1_native", "fp4"))
                 and all(breakeven(d, kk, nn) <= 2 for d in ("co32", "co32c") for m in TS_SHAPES
                         for _, kk, nn in TS_SHAPES[m]),
                 str({k_: v for k_, v in out["ts_zero_instance_O"].items() if k_.startswith("co32")}))
    checks.check("H committed digits (co32c) close the second-limb gap: the cheater's saving is the recombination "
                 "(4 IMADs a digit) and the residue sums, below co32's ε on every Llama-3.1 linear at every fold",
                 all(float(ts[f"fold={f} {m}"][f"{name} co32c"][:-1]) < float(ts[f"fold={f} {m}"][f"{name} co32"][:-1])
                     for f in (3, 1.5, 1) for m, shapes in TS_SHAPES.items() for name, _, _ in shapes),
                 str({f"fold={f} {m}": {d: ts[f"fold={f} {m}"][f"weighted {d}"] for d in ("co32", "co32c")}
                      for f in (3, 1.5, 1) for m in TS_SHAPES}))
    checks.check("H cohalf at A = B = 0: a cheater that makes a fraction of O = c·b on scalar units beats co32's ε "
                 "for x ≤ 16 warp instructions per IMMA-equivalent (Llama-3.1-8B and 70B, weighted); FP4 never helps",
                 all(zero[f"cohalf {m} scalar x={x}"] > base[("co32", m)] for m in TS_SHAPES for x in (8, 12, 16))
                 and all(abs(zero[f"cohalf {m} fp4"] - base[("cohalf", m)]) < 1e-12 for m in TS_SHAPES),
                 str({k_: v for k_, v in out["ts_zero_instance_O"].items() if k_.startswith("cohalf")}))

    agree = [(d, kk, f, x) for d in ("co32c", "cohalf") for kk in (128, 4096, 14336) for f in (3, 1.5, 1)
             for x in (4, 12, 24)]
    checks.check("H the joint minimiser (vertices of the (φ, t) program) agrees with the one-route kink minimum",
                 all(abs(ts_best(d, kk, 4096, f, [o_routes(d, kk)["scalar"](x)])[1]
                         - ts_mix(d, kk, 4096, f, *o_routes(d, kk)["scalar"](x))[1]) < 1e-9 for d, kk, f, x in agree))

    def all_routes(design, kk, x):
        return word_routes(design, kk, x) + [o_routes(design, kk)["scalar"](x)]

    def weighted_best(design, fold, x, mname):
        h = c_ = 0.0
        for _, kk, nn in TS_SHAPES[mname]:
            th, tb = ts_best(design, kk, nn, fold, all_routes(design, kk, x) if x else [])
            h, c_ = h + th * nn, c_ + tb * nn
        return h / c_ - 1

    joint = {f"fold={f} {m} x={x}": weighted_best("co32c", f, x, m) for f in (3, 1.5, 1) for m in TS_SHAPES
             for x in (4, 8, 12, 16)}
    base_f = {(f, m): weighted_best("co32c", f, None, m) for f in (3, 1.5, 1) for m in TS_SHAPES}
    out["ts_zero_instance_co32c_all_routes"] = {k_: f"{v:.4%}" for k_, v in joint.items()}
    checks.check("H co32c at A = B = 0 with every word bit-sliced at once (T at 8x, D at 7x an IMMA, O′'s c·b at "
                 "x·k/32): no gain on the Llama-3.1 linears at fold 3 for x ≥ 4, and at folds 1.5 and 1 for x ≥ 8",
                 all(abs(joint[f"fold={f} {m} x={x}"] - base_f[(f, m)]) < 1e-12 for m in TS_SHAPES
                     for f, xs in ((3, (4, 8, 12, 16)), (1.5, (8, 12, 16)), (1, (8, 12, 16))) for x in xs),
                 str(out["ts_zero_instance_co32c_all_routes"]))
    qk = {}
    for f in (3, 1.5, 1):
        for d in ("co32", "co32c", "cohalf"):
            th, tc = (ts_time(*t_) for t_ in ts_tile(d, 128, 4096, f))
            qk[f"fold={f} {d} strip-only"] = f"{th / tc - 1:.4%}"
            for x in (4, 8, 12, 16, 24):
                th, tb = ts_best(d, 128, 4096, f, all_routes(d, 128, x))
                qk[f"fold={f} {d} x={x}"] = f"{th / tb - 1:.4%}"
    out["ts_qk_k128_all_routes"] = qk
    checks.check("H QK^T (k = 128) at fold 3 with every route: cohalf below co32c at every x ≥ 4, co32c below co32",
                 all(float(qk[f"fold=3 cohalf x={x}"][:-1]) < float(qk[f"fold=3 co32c x={x}"][:-1])
                     < float(qk[f"fold=3 co32 x={x}"][:-1]) for x in (4, 8, 12, 16, 24)), str(qk))
    recomb = {}
    for f in (3, 1.5, 1):
        for mname, shapes in TS_SHAPES.items():
            h = c_ = 0.0
            for _, kk, nn in shapes:
                (Mh, Oh), (Mc, Oc) = ts_tile("co32c", kk, nn, f)
                h += ts_time(Mh, Oh) * nn
                c_ += ts_time(Mc, Oc + 4 * (n_digits((kk // 32) * -128, (kk // 32) * 127) - 1)) * nn
            recomb[f"fold={f} {mname}"] = h / c_ - 1
    out["ts_co32c_recombination_charged"] = {k_: f"{v:.4%}" for k_, v in recomb.items()}
    checks.check("H co32c when the served output feeds a checked value (the cheater recombines O too): ε < 0.1% on "
                 "Llama-3.1 8B and 70B linears at every fold", all(v < 1e-3 for v in recomb.values()),
                 str(out["ts_co32c_recombination_charged"]))

    import json
    print("\nRESULTS " + json.dumps(out, default=str, sort_keys=True))
    sys.exit(checks.finish())


if __name__ == "__main__":
    main()

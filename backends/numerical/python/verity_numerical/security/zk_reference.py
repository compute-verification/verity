"""Pure-Python reference of the masking pieces of ``backends/ZK_CONSTRUCTION.md`` at toy sizes (zk-construction lane,
2026-09-22).  Nothing here is fast or a proof system; it is the executable form of the two privacy arguments the
document relies on, small enough to check exhaustively or statistically in a test:

1. **ZK Ligero masking** (AHIV22 section 4.6 / Lemma 4.15; libzk ``random_row`` + ``layout_zk_rows``): a witness
   matrix is encoded row by row as a Reed-Solomon codeword of a degree ``< k`` polynomial whose values at the first
   ``l`` payload points are the witness and at the remaining ``k - l`` points are fresh randomizers.  Any ``t <= k - l``
   opened columns of a row are then uniform (:func:`opened_columns_rank` shows the randomizer-to-opened-symbols
   map has full rank -- a Cauchy-matrix minor); the three tests (interleaved, linear, quadratic) return responses
   blinded by one committed random row each (a random codeword; a random degree ``< k + l - 1`` polynomial whose
   payload sums to zero; a random degree ``< 2k - 1`` polynomial vanishing on the payload points).
   :class:`LigeroToy` runs prover and verifier; :func:`simulate_view` produces the verifier's view without the
   witness; :func:`view_statistics` compares the two distributions.

2. **Zero-knowledge sumcheck** for a degree-3 product ``f = f1 f2 f3`` over ``n`` variables with Libra's masking
   (Xie et al. 2019, section 4.1, following Chiesa-Forbes-Spooner 2017): the prover commits to a random
   ``g(x) = a_0 + sum_i g_i(x_i)`` with univariate ``g_i`` of degree 3, reveals ``H_g``, receives ``rho`` and runs the
   sumcheck on ``f + rho g``.  :func:`masked_sumcheck` runs it; :func:`transcript_map_rank` shows that, for fixed
   challenges, the affine map from ``g`` to the transcript is onto the whole space of consistent transcripts (so the
   transcript is uniform there and independent of ``f``); :func:`padded_sumcheck` is the protocols' own variant
   (FS24 / libzk padded transcript: one committed pad per transcript element, consistency deferred to linear
   constraints on the pads).

The field is a small prime ``p`` (default 97) so the statistical tests have room to fail; all arithmetic is plain
Python integers.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "LigeroToy",
    "LigeroView",
    "Poly",
    "chi_square_uniform",
    "interpolate",
    "masked_sumcheck",
    "mle_eval",
    "opened_columns_rank",
    "padded_sumcheck",
    "poly_eval",
    "rank_mod_p",
    "simulate_view",
    "transcript_map_rank",
    "view_statistics",
]

Poly = list[int]        # coefficients, low degree first


# ---------------------------------------------------------------------------------------------------------------------
# Field / polynomial helpers
# ---------------------------------------------------------------------------------------------------------------------

def poly_eval(c: Sequence[int], x: int, p: int) -> int:
    acc = 0
    for a in reversed(c):
        acc = (acc * x + a) % p
    return acc


def _poly_add(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    n = max(len(a), len(b))
    return [((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)]


def _poly_scale(a: Sequence[int], s: int, p: int) -> Poly:
    return [(x * s) % p for x in a]


def _poly_mul(a: Sequence[int], b: Sequence[int], p: int) -> Poly:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % p
    return out


def interpolate(xs: Sequence[int], ys: Sequence[int], p: int) -> Poly:
    """Lagrange interpolation: the unique polynomial of degree < len(xs) through the points (coefficients)."""
    n = len(xs)
    out = [0] * n
    for i in range(n):
        num = [1]
        den = 1
        for j in range(n):
            if j != i:
                num = _poly_mul(num, [(-xs[j]) % p, 1], p)
                den = den * (xs[i] - xs[j]) % p
        coef = ys[i] * pow(den, p - 2, p) % p
        out = _poly_add(out, _poly_scale(num, coef, p), p)
    return out


def _degree(c: Sequence[int]) -> int:
    for i in range(len(c) - 1, -1, -1):
        if c[i]:
            return i
    return -1


def rank_mod_p(rows: list[list[int]], p: int) -> int:
    """Rank of a matrix over F_p by Gaussian elimination."""
    m = [r[:] for r in rows]
    rank = 0
    ncols = len(m[0]) if m else 0
    for col in range(ncols):
        piv = next((i for i in range(rank, len(m)) if m[i][col] % p), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][col], p - 2, p)
        m[rank] = [x * inv % p for x in m[rank]]
        for i in range(len(m)):
            if i != rank and m[i][col] % p:
                f = m[i][col]
                m[i] = [(x - f * y) % p for x, y in zip(m[i], m[rank])]
        rank += 1
        if rank == len(m):
            break
    return rank


def chi_square_uniform(samples: Sequence[int], p: int) -> float:
    """Chi-square statistic of ``samples`` against the uniform distribution on ``F_p`` (``p - 1`` degrees of freedom)."""
    counts = [0] * p
    for s in samples:
        counts[s % p] += 1
    e = len(samples) / p
    return sum((c - e) ** 2 / e for c in counts)


# ---------------------------------------------------------------------------------------------------------------------
# 1. ZK Ligero at toy size
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LigeroToy:
    """Reed-Solomon ``[n, k]`` over ``F_p`` with payload ``l`` per row and ``t`` opened columns.  Evaluation points:
    ``zetas[0:l]`` payload, ``zetas[l:k]`` randomizers, ``etas`` the ``n`` codeword positions (all distinct)."""

    p: int = 97
    n: int = 32
    k: int = 12
    l: int = 4
    t: int = 6

    def __post_init__(self) -> None:
        if not (0 < self.l < self.k < self.n) or self.n <= 2 * self.k:
            raise ValueError("need 0 < l < k < n and n > 2k (quadratic test)")
        if self.k - self.l < self.t:
            raise ValueError(f"zero knowledge needs k - l >= t (AHIV22 Lemma 4.15: k > l + t): k-l={self.k - self.l}, t={self.t}")
        if self.p < self.n + self.k:
            raise ValueError("need n + k distinct points")

    @property
    def zetas(self) -> list[int]:
        return list(range(1, self.k + 1))

    @property
    def etas(self) -> list[int]:
        return list(range(self.k + 1, self.k + 1 + self.n))

    # -- encoding ----------------------------------------------------------------------------------------------------

    def row_poly(self, payload: Sequence[int], rng: random.Random, zk: bool = True) -> Poly:
        """Degree ``< k`` polynomial: payload on ``zetas[:l]``, fresh randomizers (or zeros without ZK) on the rest."""
        if len(payload) != self.l:
            raise ValueError("payload length must be l")
        vals = [v % self.p for v in payload] + [rng.randrange(self.p) if zk else 0 for _ in range(self.k - self.l)]
        return interpolate(self.zetas, vals, self.p)

    def encode(self, poly: Sequence[int]) -> list[int]:
        return [poly_eval(poly, e, self.p) for e in self.etas]

    def is_codeword(self, word: Sequence[int], degree_bound: int | None = None) -> bool:
        """``word`` (length ``n``) is the evaluation of a polynomial of degree < ``degree_bound`` (default ``k``)."""
        d = self.k if degree_bound is None else degree_bound
        c = interpolate(self.etas, list(word), self.p)
        return _degree(c) < d

    # -- blinding rows -----------------------------------------------------------------------------------------------

    def blind_irs(self, rng: random.Random) -> Poly:
        return [rng.randrange(self.p) for _ in range(self.k)]                                   # random codeword

    def blind_linear(self, rng: random.Random) -> Poly:
        """Random polynomial of degree < k + l - 1 with sum over the payload points zero (AHIV22 4.6.2)."""
        deg = self.k + self.l - 1
        c = [rng.randrange(self.p) for _ in range(deg)]
        s = sum(poly_eval(c, z, self.p) for z in self.zetas[: self.l]) % self.p
        # subtract s/l from the constant coefficient: the payload sum becomes 0
        c[0] = (c[0] - s * pow(self.l, self.p - 2, self.p)) % self.p
        return c

    def blind_quadratic(self, rng: random.Random) -> Poly:
        """Random polynomial of degree < 2k - 1 vanishing on the payload points (AHIV22 4.6.3)."""
        van = [1]
        for z in self.zetas[: self.l]:
            van = _poly_mul(van, [(-z) % self.p, 1], self.p)
        q = [rng.randrange(self.p) for _ in range(2 * self.k - 1 - self.l)]
        return _poly_mul(van, q, self.p)

    # -- the three tests (prover side) --------------------------------------------------------------------------------

    def irs_response(self, polys: Sequence[Poly], r: Sequence[int], u0: Poly) -> Poly:
        v = list(u0)
        for ri, pi in zip(r, polys):
            v = _poly_add(v, _poly_scale(pi, ri, self.p), self.p)
        return v

    def linear_response(self, polys: Sequence[Poly], A: Sequence[Sequence[int]], r: Sequence[int], u_add: Poly) -> tuple[Poly, list[Poly]]:
        """``q(X) = sum_i r_i(X) p_i(X) + u_add(X)`` where ``r_i`` interpolates ``(r^T A)`` restricted to row ``i`` on the
        payload points (AHIV22 4.7 step 2 / 4.6.2).  Returns ``q`` and the public ``r_i``."""
        m = len(polys)
        rtA = [sum(r[c] * A[c][j] for c in range(len(A))) % self.p for j in range(m * self.l)]
        r_polys = [interpolate(self.zetas[: self.l], rtA[i * self.l:(i + 1) * self.l], self.p) for i in range(m)]
        q = list(u_add)
        for ri, pi in zip(r_polys, polys):
            q = _poly_add(q, _poly_mul(ri, pi, self.p), self.p)
        return q, r_polys

    def quadratic_response(self, triples: Sequence[tuple[Poly, Poly, Poly]], r: Sequence[int], u_q: Poly) -> Poly:
        p0 = list(u_q)
        for ri, (px, py, pz) in zip(r, triples):
            prod = _poly_add(_poly_mul(px, py, self.p), _poly_scale(pz, self.p - 1, self.p), self.p)
            p0 = _poly_add(p0, _poly_scale(prod, ri, self.p), self.p)
        return p0

    # -- verifier side -----------------------------------------------------------------------------------------------

    def verify(self, view: LigeroView, A: Sequence[Sequence[int]], b: Sequence[int], triple_index: Sequence[tuple[int, int, int]]) -> bool:
        p = self.p
        Q = view.Q
        # IRS: v is a codeword and agrees with the opened columns
        if not self.is_codeword(view.v):
            return False
        for j in Q:
            lhs = view.v[j]
            rhs = (view.opened["u0"][j] + sum(view.r[i] * view.opened[i][j] for i in range(view.m))) % p
            if lhs != rhs:
                return False
        # linear: degree, payload sum, column consistency
        if _degree(view.q) >= self.k + self.l - 1:
            return False
        if sum(poly_eval(view.q, z, p) for z in self.zetas[: self.l]) % p != sum(view.r_lin[c] * b[c] for c in range(len(b))) % p:
            return False
        for j in Q:
            eta = self.etas[j]
            rhs = (view.opened["u_add"][j] + sum(poly_eval(view.r_polys[i], eta, p) * view.opened[i][j] for i in range(view.m))) % p
            if poly_eval(view.q, eta, p) != rhs:
                return False
        # quadratic: degree, vanishing on payload points, column consistency
        if _degree(view.p0) >= 2 * self.k - 1:
            return False
        if any(poly_eval(view.p0, z, p) for z in self.zetas[: self.l]):
            return False
        for j in Q:
            eta = self.etas[j]
            rhs = view.opened["u_q"][j]
            for ri, (ix, iy, iz) in zip(view.r_quad, triple_index):
                rhs = (rhs + ri * (view.opened[ix][j] * view.opened[iy][j] - view.opened[iz][j])) % p
            if poly_eval(view.p0, eta, p) != rhs:
                return False
        return True


@dataclass
class LigeroView:
    """What the verifier sees: its own coins, the three blinded responses, and the ``t`` opened columns of every
    committed row (witness rows ``0..m-1`` and the three blinding rows)."""

    m: int
    Q: list[int]
    r: list[int]
    r_lin: list[int]
    r_quad: list[int]
    r_polys: list[Poly]
    v: list[int]
    q: Poly
    p0: Poly
    opened: dict = field(default_factory=dict)      # row id -> {column j: symbol}

    def projection(self, p: int) -> list[int]:
        """A fixed low-dimensional projection used by the statistical comparison: the opened symbols of witness row
        0, one interior value of each response."""
        out = [self.opened[0][j] for j in self.Q]
        out += [self.v[self.Q[0]], poly_eval(self.q, 0, p), poly_eval(self.p0, 0, p)]
        return out


def _coins(toy: LigeroToy, m: int, n_lin: int, n_quad: int, rng: random.Random) -> tuple[list[int], list[int], list[int], list[int]]:
    r = [rng.randrange(toy.p) for _ in range(m)]
    r_lin = [rng.randrange(toy.p) for _ in range(n_lin)]
    r_quad = [rng.randrange(toy.p) for _ in range(n_quad)]
    Q = sorted(rng.sample(range(toy.n), toy.t))
    return r, r_lin, r_quad, Q


def prove(toy: LigeroToy, witness_rows: Sequence[Sequence[int]], A: Sequence[Sequence[int]], triple_index: Sequence[tuple[int, int, int]],
          rng: random.Random, coins: tuple | None = None, zk: bool = True) -> LigeroView:
    """Honest prover against an honest verifier's coins (``coins`` from :func:`_coins`; drawn here if None -- in the
    real protocol they arrive after the commitment; here the commitment is implicit, the order does not matter for
    an honest verifier)."""
    m = len(witness_rows)
    polys = [toy.row_poly(w, rng, zk) for w in witness_rows]
    u0 = toy.blind_irs(rng) if zk else [0] * toy.k
    u_add = toy.blind_linear(rng) if zk else [0]
    u_q = toy.blind_quadratic(rng) if zk else [0]
    r, r_lin, r_quad, Q = coins if coins else _coins(toy, m, len(A), len(triple_index), rng)
    v = toy.irs_response(polys, r, u0)
    q, r_polys = toy.linear_response(polys, A, r_lin, u_add)
    p0 = toy.quadratic_response([(polys[a], polys[b], polys[c]) for a, b, c in triple_index], r_quad, u_q)
    codewords = {i: toy.encode(pi) for i, pi in enumerate(polys)}
    codewords["u0"] = toy.encode(u0)
    codewords["u_add"] = toy.encode(u_add)
    codewords["u_q"] = toy.encode(u_q)
    opened = {key: {j: cw[j] for j in Q} for key, cw in codewords.items()}
    return LigeroView(m, Q, r, r_lin, r_quad, r_polys, toy.encode(v), q, p0, opened)


def simulate_view(toy: LigeroToy, m: int, A: Sequence[Sequence[int]], b: Sequence[int], triple_index: Sequence[tuple[int, int, int]],
                  rng: random.Random, coins: tuple | None = None) -> LigeroView:
    """The honest-verifier simulator (AHIV22 Lemma 4.15 / 4.6): opened witness symbols uniform; ``v`` a uniform
    codeword; ``q`` uniform of degree ``< k + l - 1`` with the right payload sum; ``p0`` uniform of degree ``< 2k - 1``
    vanishing on the payload points; the blinding rows' opened symbols set to make the column checks pass.  Uses
    the public data only (``A``, ``b``, the triple layout, the coins)."""
    p = toy.p
    r, r_lin, r_quad, Q = coins if coins else _coins(toy, m, len(A), len(triple_index), rng)
    opened = {i: {j: rng.randrange(p) for j in Q} for i in range(m)}
    v_poly = [rng.randrange(p) for _ in range(toy.k)]
    v = toy.encode(v_poly)
    q = toy.blind_linear(rng)
    target = sum(r_lin[c] * b[c] for c in range(len(b))) % p
    q[0] = (q[0] + target * pow(toy.l, p - 2, p)) % p                    # payload sum = r_lin^T b
    p0 = toy.blind_quadratic(rng)
    rtA = [sum(r_lin[c] * A[c][j] for c in range(len(A))) % p for j in range(m * toy.l)]
    r_polys = [interpolate(toy.zetas[: toy.l], rtA[i * toy.l:(i + 1) * toy.l], p) for i in range(m)]
    opened["u0"] = {j: (v[j] - sum(r[i] * opened[i][j] for i in range(m))) % p for j in Q}
    opened["u_add"] = {}
    opened["u_q"] = {}
    for j in Q:
        eta = toy.etas[j]
        opened["u_add"][j] = (poly_eval(q, eta, p) - sum(poly_eval(r_polys[i], eta, p) * opened[i][j] for i in range(m))) % p
        acc = 0
        for ri, (ix, iy, iz) in zip(r_quad, triple_index):
            acc = (acc + ri * (opened[ix][j] * opened[iy][j] - opened[iz][j])) % p
        opened["u_q"][j] = (poly_eval(p0, eta, p) - acc) % p
    return LigeroView(m, Q, r, r_lin, r_quad, r_polys, v, q, p0, opened)


def opened_columns_rank(toy: LigeroToy, Q: Sequence[int]) -> int:
    """Rank of the map randomizers (``k - l`` values) -> opened symbols (``t`` values) of one row.  Equal to ``t``
    means the opened symbols are uniform whatever the payload (the affine offset is the payload's contribution)."""
    p = toy.p
    rows = []
    for j in Q:
        eta = toy.etas[j]
        row = []
        for c in range(toy.l, toy.k):
            # Lagrange basis polynomial of zeta_c over all zetas, evaluated at eta
            num = den = 1
            for c2 in range(toy.k):
                if c2 != c:
                    num = num * (eta - toy.zetas[c2]) % p
                    den = den * (toy.zetas[c] - toy.zetas[c2]) % p
            row.append(num * pow(den, p - 2, p) % p)
        rows.append(row)
    return rank_mod_p(rows, p)


def view_statistics(toy: LigeroToy, witness_a: Sequence[Sequence[int]], witness_b: Sequence[Sequence[int]], A, b, triple_index,
                    trials: int = 400, seed: int = 1) -> dict:
    """Chi-square statistics (against uniform on ``F_p``) of each projected coordinate of the verifier's view for two
    witnesses and for the simulator, all under fresh coins.  Under zero knowledge every coordinate is uniform for
    all three; without ZK (``zk=False``) the opened witness symbols are deterministic given the coins."""
    rng = random.Random(seed)
    out = {}
    for name, w in (("witness_a", witness_a), ("witness_b", witness_b), ("simulator", None), ("witness_a_no_zk", witness_a)):
        proj = []
        for _ in range(trials):
            if name == "simulator":
                view = simulate_view(toy, len(witness_a), A, b, triple_index, rng)
            else:
                view = prove(toy, w, A, triple_index, rng, zk=not name.endswith("no_zk"))
            proj.append(view.projection(toy.p))
        out[name] = [chi_square_uniform([pr[i] for pr in proj], toy.p) for i in range(len(proj[0]))]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# 2. Zero-knowledge sumcheck (degree-3 product) at toy size
# ---------------------------------------------------------------------------------------------------------------------

def mle_eval(table: Sequence[int], point: Sequence[int], p: int) -> int:
    """Multilinear extension of ``table`` (length ``2^n``, index bits low first) at ``point``."""
    cur = list(table)
    for x in point:
        half = len(cur) // 2
        cur = [((1 - x) * cur[i] + x * cur[i + half]) % p for i in range(half)]
    return cur[0] % p


def _fold(table: Sequence[int], x: int, p: int) -> list[int]:
    half = len(table) // 2
    return [((1 - x) * table[i] + x * table[i + half]) % p for i in range(half)]


def _round_poly_product(tables: Sequence[Sequence[int]], p: int) -> Poly:
    """Degree-3 round polynomial of ``prod_k tables_k`` over the first variable: evaluated at X = 0,1,2,3 then
    interpolated."""
    half = len(tables[0]) // 2
    vals = []
    for X in range(4):
        s = 0
        for i in range(half):
            prod = 1
            for tb in tables:
                prod = prod * (((1 - X) * tb[i] + X * tb[i + half]) % p) % p
            s = (s + prod) % p
        vals.append(s)
    return interpolate([0, 1, 2, 3], vals, p)


def masked_sumcheck(tables: Sequence[Sequence[int]], p: int, rng: random.Random, challenges: Sequence[int] | None = None,
                    rho: int | None = None, mask: dict | None = None, cheat_sum: int | None = None) -> dict:
    """Libra-masked sumcheck of ``H = sum_x f1 f2 f3`` over ``{0,1}^n``.

    Prover: draws ``g = a_0 + sum_i g_i(x_i)`` (``g_i`` degree 3), publishes ``H_g = sum g`` and a commitment to ``g``
    (here: the coefficients, kept in the transcript's ``mask`` field and opened only at the end); verifier sends
    ``rho``; sumcheck on ``h = f + rho g`` with claim ``H + rho H_g``; final check ``s_n(r_n) = f(r) + rho g(r)``.
    Returns the transcript (claim, round polynomials, challenges, final values) and ``accepted``."""
    n = len(tables[0]).bit_length() - 1
    if any(len(tb) != 2**n for tb in tables):
        raise ValueError("tables must have length 2^n")
    if mask is None:
        mask = {"a0": rng.randrange(p), "g": [[rng.randrange(p) for _ in range(4)] for _ in range(n)]}
    a0, gs = mask["a0"], mask["g"]
    H = sum(_prod(tb[i] for tb in tables) for i in range(2**n)) % p
    Hg = (pow(2, n, p) * a0 + pow(2, n - 1, p) * sum(poly_eval(g, 0, p) + poly_eval(g, 1, p) for g in gs)) % p
    if rho is None:
        rho = rng.randrange(p)
    claim = ((cheat_sum if cheat_sum is not None else H) + rho * Hg) % p
    cur = [list(tb) for tb in tables]
    rounds: list[Poly] = []
    rs: list[int] = []
    prev = claim
    accepted = True
    for j in range(n):
        sf = _round_poly_product(cur, p)
        # g's contribution: 2^(n-j-1) [ a0 + sum_{i<j} g_i(r_i) + g_j(X) ] + 2^(n-j-2) sum_{i>j} (g_i(0)+g_i(1))
        fixed = (a0 + sum(poly_eval(gs[i], rs[i], p) for i in range(j))) % p
        tail = sum(poly_eval(gs[i], 0, p) + poly_eval(gs[i], 1, p) for i in range(j + 1, n)) % p
        scale_now = pow(2, n - j - 1, p)
        scale_tail = pow(2, n - j - 2, p) if n - j - 2 >= 0 else pow(2, p - 2, p)     # 2^-1 when n-j-2 = -1
        sg = _poly_add([fixed] + [0, 0, 0], gs[j], p)
        sg = _poly_scale(sg, scale_now, p)
        sg[0] = (sg[0] + scale_tail * tail) % p
        s = _poly_add(sf, _poly_scale(sg, rho, p), p)
        while len(s) < 4:
            s.append(0)
        if (poly_eval(s, 0, p) + poly_eval(s, 1, p)) % p != prev:
            accepted = False
        r = challenges[j] if challenges else rng.randrange(p)
        rs.append(r)
        rounds.append(s)
        prev = poly_eval(s, r, p)
        cur = [_fold(tb, r, p) for tb in cur]
    f_r = _prod(tb[0] for tb in cur) % p
    g_r = (a0 + sum(poly_eval(gs[i], rs[i], p) for i in range(n))) % p
    if (f_r + rho * g_r) % p != prev:
        accepted = False
    return {"n": n, "H": H, "H_g": Hg, "rho": rho, "claim": claim, "rounds": rounds, "challenges": rs,
            "f_r": f_r, "g_r": g_r, "mask": mask, "accepted": accepted}


def _prod(it) -> int:
    out = 1
    for x in it:
        out *= x
    return out


def transcript_vector(tr: dict) -> list[int]:
    """The transcript as a vector: claim, the 4 coefficients of every round polynomial, ``g(r)``."""
    out = [tr["claim"]]
    for s in tr["rounds"]:
        out += list(s[:4])
    out.append(tr["g_r"])
    return out


def transcript_map_rank(tables: Sequence[Sequence[int]], p: int, challenges: Sequence[int], rho: int) -> dict:
    """For fixed ``(challenges, rho)`` the transcript is an affine function of the ``4n + 1`` mask coefficients.  The
    consistent transcripts (``s_j(0) + s_j(1) = s_{j-1}(r_{j-1})``, ``s_0(0)+s_0(1) = claim``, ``s_n(r_n) = f(r) + rho g(r)``
    with ``f(r)`` public) form an affine space of dimension ``3n + 1``; if the linear part has that rank, the masked
    transcript is uniform on it and independent of ``f``."""
    n = len(tables[0]).bit_length() - 1
    rng = random.Random(0)
    zero_mask = {"a0": 0, "g": [[0] * 4 for _ in range(n)]}
    base = transcript_vector(masked_sumcheck(tables, p, rng, challenges, rho, zero_mask))
    cols = []
    for idx in range(4 * n + 1):
        m = {"a0": 0, "g": [[0] * 4 for _ in range(n)]}
        if idx == 0:
            m["a0"] = 1
        else:
            m["g"][(idx - 1) // 4][(idx - 1) % 4] = 1
        vec = transcript_vector(masked_sumcheck(tables, p, rng, challenges, rho, m))
        cols.append([(a - b) % p for a, b in zip(vec, base)])
    rank = rank_mod_p([list(c) for c in zip(*cols)], p)        # rows = transcript coordinates, columns = mask coefficients
    return {"rank": rank, "consistent_space_dim": 3 * n + 1, "transcript_len": len(base), "mask_dim": 4 * n + 1,
            "uniform_on_consistent_space": rank == 3 * n + 1}


def padded_sumcheck(tables: Sequence[Sequence[int]], p: int, rng: random.Random, challenges: Sequence[int] | None = None,
                    cheat_sum: int | None = None) -> dict:
    """The protocols' padded-transcript variant (FS24 2.3 / libzk 6.5): every round polynomial coefficient and the
    final evaluation are sent as ``value + pad`` with pads committed beforehand (here: returned); the verifier's
    checks become linear constraints on the pads with public coefficients, deferred (here: evaluated at the end).
    Returns the padded messages, the pads, and whether the deferred constraints hold."""
    n = len(tables[0]).bit_length() - 1
    H = sum(_prod(tb[i] for tb in tables) for i in range(2**n)) % p
    claim = cheat_sum if cheat_sum is not None else H
    pads = [[rng.randrange(p) for _ in range(4)] for _ in range(n)] + [[rng.randrange(p)]]
    cur = [list(tb) for tb in tables]
    padded: list[Poly] = []
    rs: list[int] = []
    clear: list[Poly] = []
    for j in range(n):
        s = _round_poly_product(cur, p)
        while len(s) < 4:
            s.append(0)
        clear.append(s)
        padded.append([(a + b) % p for a, b in zip(s, pads[j])])
        r = challenges[j] if challenges else rng.randrange(p)
        rs.append(r)
        cur = [_fold(tb, r, p) for tb in cur]
    f_r = _prod(tb[0] for tb in cur) % p
    padded_final = (f_r + pads[n][0]) % p
    # deferred constraints, linear in the pads: <s_j>(0)+<s_j>(1) - <s_{j-1}>(r_{j-1}) = pad_j(0)+pad_j(1) - pad_{j-1}(r_{j-1})
    ok = True
    prev_padded = claim
    for j in range(n):
        lhs = (poly_eval(padded[j], 0, p) + poly_eval(padded[j], 1, p) - (prev_padded if j == 0 else poly_eval(padded[j - 1], rs[j - 1], p))) % p
        rhs = (poly_eval(pads[j], 0, p) + poly_eval(pads[j], 1, p) - (0 if j == 0 else poly_eval(pads[j - 1], rs[j - 1], p))) % p
        ok &= lhs == rhs
        prev_padded = padded[j]
    # final: <s_n>(r_n) - <f(r)> = pad_n(r_n) - pad_final   (the real protocol reduces f(r) further; here it is the oracle)
    ok &= (poly_eval(padded[n - 1], rs[n - 1], p) - padded_final) % p == (poly_eval(pads[n - 1], rs[n - 1], p) - pads[n][0]) % p
    return {"n": n, "claim": claim, "padded_rounds": padded, "padded_final": padded_final, "pads": pads, "challenges": rs,
            "clear_rounds": clear, "deferred_constraints_hold": bool(ok)}

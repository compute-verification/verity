"""`verity_flock.rec_sparse`, V*'s sparse phase (F4 candidate 3): an honest prover written from the design note (§1.3 and §5.4
"10 Oct, 10:00Z") runs a rep of a session of S slots of hidden types over a stacked table of T random templates, from the
hidden matrix's comb to the registered claim, and the reference accepts it; a forged round message, a wrong registered
claim, a slot's type swapped for another and a type outside the bucket are refused; the Definition computes the reference
bit for bit, honest and forged; and its ANDs are the products' 2,187 each and the selects'."""
from __future__ import annotations

import random
from functools import reduce

import numpy as np
import pytest
from verity.ml.circuits.ops.boolean.gf2k import mul
from verity_flock import rec_sparse as RS
from verity_flock.rec_algebra import skip_weights

eq_table = RS.eq_table


def _xor(xs) -> int:
    out = 0
    for x in xs:
        out ^= x
    return out


def _mle(vals, r):
    return RS._dot(eq_table(r), vals)


def _poly_mul(polys):
    """The product of polynomials given by coefficients, lowest first."""
    out = [1]
    for p in polys:
        nxt = [0] * (len(out) + len(p) - 1)
        for i, a in enumerate(out):
            for j, b in enumerate(p):
                nxt[i + j] ^= mul(a, b)
        out = nxt
    return out


def _eval(poly, r):
    acc = 0
    for c in reversed(poly):
        acc = mul(acc, r) ^ c
    return acc


def _sumcheck(g: random.Random, tables: list[list[int]], factors, claim: int, degs: list[int]):
    """The honest prover of ``Σ_b ∏ factors(the tables at b)`` (each factor affine in the tables), one round a variable, lowest
    first: per round ``(c_0, c_2, .., c_deg)`` (the degree bound and the sum rule asserted) and its coin. Returns the messages,
    the coins, the last claim and the tables at the coins."""
    msgs, coins = [], []
    for deg in degs:
        poly = [0] * (deg + 2)
        for b in range(0, len(tables[0]), 2):
            lo, hi = factors([t[b] for t in tables]), factors([t[b + 1] for t in tables])
            for i, c in enumerate(_poly_mul([[x, x ^ y] for x, y in zip(lo, hi, strict=True)])):
                poly[i] ^= c
        assert not any(poly[deg + 1:]), "a round polynomial past its degree bound"
        assert _xor(poly[1:]) == claim, "the honest round breaks the sum rule"
        msgs.append([poly[0]] + poly[2:deg + 1])
        r = g.getrandbits(128)
        coins.append(r)
        claim = _eval(poly, r)
        tables = [[t[b] ^ mul(r, t[b] ^ t[b + 1]) for b in range(0, len(t), 2)] for t in tables]
    return msgs, coins, claim, [t[0] for t in tables]


def _template(g: random.Random, sh: RS.Shape) -> list[list[int]]:
    """A registered type's P' planes over 2^NU entries: random row and column bits and sides, some entries a copy of the one
    before (they cancel), the last one padding (a = b = 0, junk indices), the planes past P zero."""
    n = 1 << sh.NU
    planes = [[0] * n for _ in range(sh.PP)]
    for k in range(n):
        if k and g.random() < 0.2:
            for p in range(sh.P):
                planes[p][k] = planes[p][k - 1]
            continue
        row, col = g.getrandbits(sh.ELL), g.getrandbits(sh.ELL)
        a, b = (0, 0) if k == n - 1 else [(1, 0), (0, 1), (1, 1)][g.randrange(3)]
        for j in range(sh.ELL):
            planes[j][k], planes[sh.ELL + j][k] = (row >> j) & 1, (col >> j) & 1
        planes[2 * sh.ELL][k], planes[2 * sh.ELL + 1][k] = a, b
    return planes


def session(sh: RS.Shape, seed: int = 0):
    """One honest rep: ``(coins, messages, tables)``. Slot σ's template fills the unit at high bits σ of the hidden matrix
    (rows and columns of ELL + S_LOG + 1 bits, the other units public); the comb is the lincheck's, ``Σ_i e[i] M[i, c]`` with
    ``e[s + 64 u] = L_s(z) eq(ρ_in, u)``, folded at the column coins; every message comes from the tables and the coins alone."""
    g = random.Random(seed)
    E, hb, n = sh.ELL, sh.S_LOG + 1, 1 << sh.NU
    tables = [_template(g, sh) for _ in range(sh.T)]
    tau = [g.randrange(sh.T) for _ in range(sh.S)]
    alpha, z = g.getrandbits(128), g.getrandbits(128)
    rho_in = [g.getrandbits(128) for _ in range(E - 6 + hb)]
    t_top = [g.getrandbits(128) for _ in range(E - 6 + hb)]
    zeta_c = [g.getrandbits(128) for _ in range(6)]
    y = zeta_c + t_top
    entries = []
    for s in range(sh.S):
        P = tables[tau[s]]
        for k in range(n):
            val = mul(alpha, P[2 * E][k]) ^ P[2 * E + 1][k]
            if val:
                row, col = (sum(P[o + j][k] << j for j in range(E)) for o in (0, E))
                entries.append((row + (s << E), col + (s << E), val))
    L, eqr, eqy = skip_weights(z), eq_table(rho_in), eq_table(y)
    comb = [0] * (1 << (E + hb))
    for i, c, val in entries:
        comb[c] ^= mul(mul(L[i & 63], eqr[i >> 6]), val)
    eqt = eq_table(t_top)
    h = [_xor(mul(eqt[c >> 6], comb[c]) for c in range(s, len(comb), 64)) for s in range(64)]
    # H1 and H2: Σ_s ŵ(s) V(s), V(s) = Σ_u eq(ρ_in, u) M̂((s, u), y)
    Vs = [0] * 64
    for i, c, val in entries:
        Vs[i & 63] ^= mul(mul(eqr[i >> 6], val), eqy[c])
    pre, zeta_r, claim, (_, v2) = _sumcheck(g, [L, Vs], lambda t: t, _mle(h, zeta_c), [2] * 6)
    x = zeta_r + rho_in
    x_lo, x_hi, y_lo, y_hi = x[:E], x[E:], y[:E], y[E:]
    assert claim == mul(_mle(L, zeta_r), v2)
    # H3: Σ_{k, σ} Ũ(σ) Φ(Q̂(σ, ·, k)), the entry bits low
    U = [mul(eq_table(x_hi)[s], eq_table(y_hi)[s]) for s in range(sh.S)]
    Q = [[tables[tau[s]][p][k] for s in range(sh.S) for k in range(n)] for p in range(sh.PP)]

    def factors(t):
        q = t[:-1]
        return ([t[-1], mul(alpha, q[2 * E]) ^ q[2 * E + 1]] + [x_lo[j] ^ 1 ^ q[j] for j in range(E)]
                + [y_lo[j] ^ 1 ^ q[E + j] for j in range(E)])

    sc, rounds, claim, last = _sumcheck(g, Q + [[U[j // n] for j in range(n * sh.S)]], factors, v2, sh.degrees)
    u, ur = last[:-1], last[-1]
    assert claim == mul(ur, reduce(mul, factors(u + [1])[1:], 1))
    # H4: the plane batch, the types' values at (ζ_p, r_k), and the registered claim at ρ_t
    r_k = rounds[:sh.NU]
    zeta_p = [g.getrandbits(128) for _ in range(sh.P_LOG)]
    vt = [_mle([_mle(P[p], r_k) for p in range(sh.PP)], zeta_p) for P in tables]
    rho = [g.getrandbits(128) for _ in range(sh.T_LOG)]
    coins = RS.Coins(alpha, z, tuple(rho_in), tuple(t_top), tuple(zeta_c), tuple(zeta_r), tuple(rounds), tuple(zeta_p),
                     tuple(rho), tuple(U))
    msgs = {"h": h, "pre": pre, "v2": v2, "sc": sc, "u": u, "vt": vt, "claim": _mle(vt, rho), "tau": tau}
    if not sh.stacked:
        msgs["claim"] = _mle(u, zeta_p)
        assert msgs["claim"] == vt[0]
    return coins, msgs, tables


SHAPES = [RS.Shape(2, 6, 1, 1), RS.Shape(3, 7, 2, 2), RS.Shape(2, 6, 0, 0), RS.Shape(1, 6, 2, 0)]


def _ref(sh, msgs, v):
    return RS.sparse_ref(sh, RS.rows(sh, msgs, v))


@pytest.mark.parametrize("sh", SHAPES, ids=str)
def test_the_reference_accepts_an_honest_rep(sh):
    for seed in range(2):
        coins, msgs, _ = session(sh, seed)
        assert _ref(sh, msgs, RS.verifier_row(sh, coins)) == [0] * RS.out_words(sh)


def _forgeries(sh, msgs, g):
    """(name, the residual it should hit, messages) for each forgery."""
    out = []
    for j, at in ((0, 0), (len(sh.degrees) - 1, -1)):
        m = {**msgs, "sc": [list(r) for r in msgs["sc"]]}
        m["sc"][j][at] ^= 1 + g.getrandbits(127)
        out.append((f"sc{j}", "h3", m))
    m = {**msgs, "pre": [list(r) for r in msgs["pre"]]}
    m["pre"][3][1] ^= 1
    out.append(("pre3", "h2", m))
    out.append(("h", "h2", {**msgs, "h": [msgs["h"][0] ^ 2] + msgs["h"][1:]}))
    out.append(("u", "h3", {**msgs, "u": [msgs["u"][0] ^ 1] + msgs["u"][1:]}))
    out.append(("claim", "claim", {**msgs, "claim": msgs["claim"] ^ 1}))
    if sh.stacked:
        out.append(("vt", "h4", {**msgs, "vt": [msgs["vt"][0] ^ 1] + msgs["vt"][1:]}))
        swapped = list(msgs["tau"])
        swapped[0] = (swapped[0] + 1) % sh.T
        out.append(("tau-swapped", "h4", {**msgs, "tau": swapped}))
        out.append(("tau-outside", "tau", {**msgs, "tau": [msgs["tau"][0] | sh.T] + list(msgs["tau"][1:])}))
    return out


def _hits(sh, out):
    names = RS.residual_names(sh)
    hit = {nm for i, nm in enumerate(names) if any(out[8 * i:8 * i + 8])}
    if any(out[8 * len(names):]):
        hit.add("tau")
    return hit


@pytest.mark.parametrize("sh", SHAPES, ids=str)
def test_forged_messages_wrong_claims_and_wrong_type_selects_are_refused(sh):
    g = random.Random(5)
    coins, msgs, _ = session(sh, 1)
    v = RS.verifier_row(sh, coins)
    for name, where, m in _forgeries(sh, msgs, g):
        assert where in _hits(sh, _ref(sh, m, v)), name


@pytest.mark.parametrize("bad, error", [((0, 6, 0, 0), ValueError), ((2, 5, 0, 0), ValueError), ((2, 6, -1, 0), ValueError),
                                        ((2, 6, 0, 17), ValueError), ((2.0, 6, 0, 0), TypeError), ((True, 6, 0, 0), TypeError)])
def test_the_shape_is_checked(bad, error):
    with pytest.raises(error, match="RecSparse"):
        RS.definition(*bad)


@pytest.mark.parametrize("sh", [RS.Shape(2, 6, 1, 1), RS.Shape(20, 15, 4, 2), RS.Shape(18, 14, 4, 0)], ids=str)
def test_the_products_are_the_structures_and_the_chains(sh):
    S, _ = RS.products_structure(sh)
    chains = RS.PRE_ROUNDS + len(sh.degrees) + 2 * sh.ELL + 1 + (sh.T if sh.stacked else 0)
    assert sum(len(r["mul"]) for r in S["res"]) + chains == RS.products(sh)


def _bits(words):
    return np.array([[(w >> j) & 1 for j in range(16)] for w in words], dtype=np.uint8)


def test_the_definition_computes_the_reference():
    """`RecSparse_v1{NU=2,ELL=6,S_LOG=1,T_LOG=1}` (the circuit-check binding) evaluated bit-sliced from its descriptor on an
    honest rep, each forgery, and random words, against the reference; and its ANDs."""
    from verity.ml.circuits.evaluation import evaluate_batch

    sh = RS.Shape(2, 6, 1, 1)
    d = RS.definition(2, 6, 1, 1)
    assert d.id == "RecSparse_v1{NU=2,ELL=6,S_LOG=1,T_LOG=1}"
    g = random.Random(9)
    coins, msgs, _ = session(sh, 0)
    v = RS.verifier_row(sh, coins)
    cases = [RS.rows(sh, msgs, v)] + [RS.rows(sh, m, v) for _, _, m in _forgeries(sh, msgs, g)]
    cases.append({nm: [g.getrandbits(16) for _ in range(n)] for nm, n in RS.ports(sh)})
    args = [np.array([_bits(c[nm]) for c in cases]) for nm, _ in RS.ports(sh)]
    res = np.asarray(evaluate_batch(d, *args)).reshape(len(cases), RS.out_words(sh), 16)
    for i, c in enumerate(cases):
        got = [int(sum(int(b) << j for j, b in enumerate(w))) for w in res[i]]
        assert got == RS.sparse_ref(sh, c), i
    assert got != [0] * RS.out_words(sh)


def test_the_ands_are_the_products_and_the_selects():
    from verity.ml.circuits.evaluation import bits as BS

    sh = RS.Shape(2, 6, 1, 1)
    assert BS.compile(RS.definition(2, 6, 1, 1)).counts["and"] == RS.ands(sh) == 151 * 2187 + 256

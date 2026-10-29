"""`verity_flock.rec_sparse`, V*'s sparse phase (F4 candidate 3): an honest prover written from the design note (§1.3 and §5.4
"10 Oct, 10:00Z") runs a rep of a session of S slots of hidden types over a stacked table of T random templates, from the
hidden matrix's comb to the registered claim, and the reference accepts it; a forged round message, a wrong registered
claim, a slot's type swapped for another and a type outside the bucket are refused; the Definition computes the reference
bit for bit, honest and forged; and its ANDs are the products' 2,187 each and the selects'."""
from __future__ import annotations

import json
import random
import subprocess
from functools import reduce
from pathlib import Path

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


def session(sh: RS.Shape, seed: int = 0, inner: dict | None = None, tau: list[int] | None = None, tables=None):
    """One honest rep: ``(coins, messages, tables)``. Slot σ's template fills the unit at high bits σ of the hidden matrix
    (rows and columns of ELL + S_LOG + 1 bits, the other units public); the comb is the lincheck's, ``Σ_i e[i] M[i, c]`` with
    ``e[s + 64 u] = L_s(z) eq(ρ_in, u)``, folded at the column point (bit j of a column against its coordinate j); every
    message comes from the tables and the coins alone. ``inner``: alpha, z and the points from the inner rep's coins
    (`rec_sparse.inner_points`), else drawn; ``tau``, ``tables``: the session's, else drawn."""
    g = random.Random(seed)
    E, hb, n = sh.ELL, sh.S_LOG + 1, 1 << sh.NU
    tables = tables if tables is not None else [_template(g, sh) for _ in range(sh.T)]
    tau = list(tau) if tau is not None else [g.randrange(sh.T) for _ in range(sh.S)]
    if inner is None:
        alpha, z = g.getrandbits(128), g.getrandbits(128)
        rho_in = [g.getrandbits(128) for _ in range(E - 6 + hb)]
        t_top = [g.getrandbits(128) for _ in range(E - 6 + hb)]
    else:
        alpha, z = inner["alpha"], inner["z"]
        rho_in, t_top = list(inner["x_rest"]), list(inner["y_rest"])
        assert len(rho_in) == len(t_top) == E - 6 + hb
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
    eqk, eqp = eq_table(r_k), eq_table(zeta_p)
    vt = [RS._dot(eqp, [_xor(e for e, bit in zip(eqk, P[p]) if bit) for p in range(sh.PP)]) for P in tables]
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


def _inner_rounds(inner):
    """An inner rep's coins by round, as many as `rec_algebra.run` draws at the shape."""
    from verity_flock import rec_algebra as A

    rec = A._dummy_coins(inner)
    A.run(inner, rec, A.symbolic(inner), A._extras_like(inner))
    return rec.rounds


def test_y_rest_is_the_inner_column_point_in_bit_order():
    """`inner_points` takes ``y_rest`` from the inner rep's coins through `rec_algebra.run`: the column point the inner claim
    on z has (its coordinates after the skip's), the lincheck's coins t reversed. An honest rep on those points is accepted,
    and the same rep with ``y_rest`` in the lincheck's round order is refused (k_log - 6 = ELL - 6 + S_LOG + 1, so every
    coordinate is read)."""
    from verity_flock import rec_algebra as A

    sh, inner = RS.Shape(2, 8, 1, 1), A.Shape(22, 10, 2)
    rounds = _inner_rounds(inner)
    pts = RS.inner_points(inner, rounds)
    info = A.run(inner, A.Coins(rounds), A.symbolic(inner), A._extras_like(inner))[1]
    ts = info["fold"]["ts"]
    assert pts["y_rest"] == info["claims"][0].x[:inner.k_log - 6] == tuple(ts[::-1]) != tuple(ts)
    assert pts["x_rest"] == tuple(info["fold"]["rho_in"]) and pts["alpha"] == info["fold"]["alpha"] and pts["z"] == info["z"]
    coins, msgs, _ = session(sh, 0, inner=pts)
    assert _ref(sh, msgs, RS.verifier_row(sh, coins)) == [0] * RS.out_words(sh)
    rounds_order = RS.Coins(**{**coins.__dict__, "y_rest": tuple(ts)})
    assert "h3" in _hits(sh, _ref(sh, msgs, RS.verifier_row(sh, rounds_order)))


def test_tau_is_one_row_that_every_reps_instance_reads():
    """Two reps of one session over the same tables and τ: each instance holds on `session_rows`, which takes τ once, so the
    reps' τ rows are one row, read at row 0 by both (`reads`) while every other port reads row r. Different τ per rep can't
    be expressed: a rep's messages carrying a τ of their own are refused."""
    sh = RS.Shape(2, 6, 1, 1)
    g = random.Random(3)
    tables = [_template(g, sh) for _ in range(sh.T)]
    tau = [1, 0]
    reps = [session(sh, seed, tau=tau, tables=tables) for seed in (0, 1)]
    msgs = [{k: x for k, x in m.items() if k != "tau"} for _, m, _ in reps]
    vs = [RS.verifier_row(sh, c) for c, _, _ in reps]
    inst = RS.session_rows(sh, msgs, vs, tau)
    assert all(RS.sparse_ref(sh, r) == [0] * RS.out_words(sh) for r in inst)
    assert inst[0]["tau"] == inst[1]["tau"] and inst[0]["h"] != inst[1]["h"]
    rd = RS.reads(sh, 2)
    assert rd["tau"] == [0, 0] and all(rd[nm] == [0, 1] for nm, _ in RS.ports(sh) if nm != "tau")
    with pytest.raises(ValueError, match="one row"):
        RS.session_rows(sh, [msgs[0], {**msgs[1], "tau": [0, 1]}], vs, tau)
    with pytest.raises(ValueError, match="tau missing"):
        RS.session_rows(sh, msgs, vs)
    assert "tau" not in RS.reads(RS.Shape(2, 6, 1, 0), 2)


@pytest.mark.parametrize("sh", [RS.Shape(2, 6, 1, 1), RS.Shape(3, 7, 2, 2)], ids=str)
def test_the_registered_claim_is_a_row_at_the_coins_point_never_a_public_extra_claim(sh):
    """`registered_claim`'s point evaluates the stacked table (entries, then planes, then types) to the honest ``claim``; it
    names row r of the ``claim`` registration and has no value, and `rec_algebra.run` refuses it as an extra claim."""
    from verity_flock import rec_algebra as A

    coins, msgs, tables = session(sh, 0)
    rc = RS.registered_claim(sh, coins, 1)
    flat = [P[p][k] for P in tables for p in range(sh.PP) for k in range(1 << sh.NU)]
    rest = eq_table(rc.x)
    assert _xor(mul(mul(rc.skip[i & 63], rest[i >> 6]), x) for i, x in enumerate(flat) if x) == msgs["claim"]
    assert (rc.row, rc.port, rc.value) == (1, "claim", None)
    inner = A.Shape(22, 10, 3)
    with pytest.raises(TypeError, match="registered claim"):
        A.run(inner, A.Coins(_inner_rounds(A.Shape(22, 10, 3))), A.symbolic(inner), [rc])


def test_h_is_the_inner_sessions_h_round_read_at_row_r():
    """RecSparse's ``h`` is the holo inner session's h round: the same 64 elements, one 512-word row port ``m<i>`` of
    `InnerRepCheck`, read at row r by both; a wrong h moves RecSparse's H2 (and the algebra's lincheck,
    `test_rec_residuals`)."""
    from verity_flock import rec_algebra as A
    from verity_flock import rec_residuals as RR

    sh, inner = RS.Shape(2, 6, 1, 1), A.Shape(22, 10, 2, holo=True)
    i = A.h_round(inner)
    assert A.rounds(inner)[i] == [("h", l) for l in range(64)] and A.rounds(inner)[i - 1] == [("zp", l) for l in range(64)]
    assert RR.spec(inner)["ports"][i] == [f"m{i}", 64] and dict(RS.ports(sh))["h"] == RR.port_words(64) == 512
    assert RS.reads(sh, 2)["h"] == [0, 1]
    coins, msgs, _ = session(sh, 0)
    assert "h2" in _hits(sh, _ref(sh, {**msgs, "h": [msgs["h"][0] ^ 1] + msgs["h"][1:]}, RS.verifier_row(sh, coins)))


def _cases(sh: RS.Shape, seed: int = 0) -> list[dict]:
    """An honest rep's ports, each forgery's and random words."""
    g = random.Random(9)
    coins, msgs, _ = session(sh, seed)
    v = RS.verifier_row(sh, coins)
    cases = [RS.rows(sh, msgs, v)] + [RS.rows(sh, m, v) for _, _, m in _forgeries(sh, msgs, g)]
    return cases + [{nm: [g.getrandbits(16) for _ in range(n)] for nm, n in RS.ports(sh)}]


@pytest.mark.slow
@pytest.mark.parametrize("sh", [RS.Shape(2, 6, 1, 1), RS.Shape(1, 6, 2, 0), RS.Shape(2, 6, 0, 1)], ids=str)
def test_the_unit_is_the_definitions_ands_and_computes_the_reference(sh):
    """The M0 unit (through its layout, what Flock proves) on an honest rep, each forgery and random words equals the reference;
    its inputs are the ports' bits, and its ANDs are the Definition's (`ands`: 2,187 a product and the selects) and
    `copy_rows`' rows of ``B = 1`` (the leaf sums and the committed operands; one stacked slot's all-ones ``w_t`` is one AND)."""
    from verity.ml.circuits.boolean.forms import ONE

    low = RS.lowering(sh)
    C = low.circuit
    ands = [v for v in C.live() if C.kind[v] == "and"]
    copies = sum(1 for v in ands if C.data[v][1].key() == ONE.key())
    assert (len(ands) - copies, copies) == (RS.ands(sh), RS.copy_rows(sh))
    assert C.counts()["inputs"] == 16 * sum(n for _, n in RS.ports(sh))
    cases = _cases(sh)
    got, _cut, ok = low.evaluate(np.array([sum((c[nm] for nm, _ in RS.ports(sh)), []) for c in cases], np.uint64))
    assert int(ok) == (1 << len(cases)) - 1
    for i, c in enumerate(cases):
        assert got[i].astype(np.uint16).tolist() == RS.sparse_ref(sh, c), i
    assert RS.sparse_ref(sh, cases[0]) == [0] * RS.out_words(sh) and all(any(RS.sparse_ref(sh, c)) for c in cases[1:])


def test_the_copy_rows_at_the_binding_shape():
    """`copy_rows` at the circuit-check binding: 6 residuals of several products (H1, the 3 H3 rounds, the plane batch, the
    registered claim) and 35 committed operands (6 H2 claims, 2 H3 claims, 25 in Φ's tree of 14 factors, 2 ``w_t``)."""
    assert RS.copy_rows(RS.Shape(2, 6, 1, 1)) == 6 * 2187 + 128 * 35 == 17_602


BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"


def _lean_statement(d: Path):
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"),
                           "--program", str(d / "program.json"), "--registered", str(d / "registered.json")],
                          capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_a_sessions_statement_reads_tau_at_row_0_and_every_other_port_at_row_r(tmp_path):
    """One RecSparse statement over two reps of one session (`stage`): every row port is a read of the prover's registration, τ
    at row 0 of its one row in both instances and every other port at row r; ``v`` is the verifier's own and ``out`` its zero
    rows. Both instances hold and the unit agrees with the reference; with rep 1's first H3 round forged (the prover's
    registration of another row), rep 1 alone fails at H3. When Lean is built, its `statement` accepts the honest files, and
    refuses the forged prover's row against the verifier's own entry of the row the ZK server committed (one salt each)."""
    from verity.ml.commitments import hm96

    sh = RS.Shape(2, 6, 1, 1)
    g = random.Random(3)
    tables = [_template(g, sh) for _ in range(sh.T)]
    tau = [1, 0]
    reps = [session(sh, seed, tau=tau, tables=tables) for seed in (0, 1)]
    msgs = [{k: x for k, x in m.items() if k != "tau"} for _, m, _ in reps]
    inst = RS.session_rows(sh, msgs, [RS.verifier_row(sh, c) for c, _, _ in reps], tau)
    program = "ab" * 64
    salts = {nm: [hm96.SHA512.fresh_salt() for _ in range(1 if nm == "tau" else 2)] for nm, _ in RS.ports(sh) if nm != "v"}
    honest = RS.registrations(sh, program, inst, salts)
    s = RS.stage(sh, inst, tmp_path / "honest", program, regs=honest)
    assert s["holds"] == [True, True] and s["unit_agrees"] == [True, True] and s["nonzero_residuals"] == [[], []]
    assert s["unit"] == "rec-sparse/n2-e6-s1-t1" and s["template"] == "RecSparse_v1{NU=2,ELL=6,S_LOG=1,T_LOG=1}"
    reads = json.loads((tmp_path / "honest" / "registered.json").read_text())["ports"]
    assert sorted(reads) == sorted([nm for nm, _ in RS.ports(sh)] + ["out"])
    assert reads["tau"]["reads"] == [0, 0] and reads["tau"]["leaves"] == 1
    assert all(reads[nm]["reads"] == [0, 1] for nm in reads if nm != "tau")
    assert all(reads[nm]["root"] == honest[nm].root.hex() for nm in honest)
    bad = [dict(r) for r in inst]
    bad[1]["sc0"] = [bad[1]["sc0"][0] ^ 1] + bad[1]["sc0"][1:]
    forged = RS.registrations(sh, program, bad, salts)["sc0"]
    f = RS.stage(sh, inst, tmp_path / "forged", program, regs={**honest, "sc0": forged}, own={"sc0": honest["sc0"].entry()})
    assert f["holds"] == [True, False] and f["nonzero_residuals"] == [[], ["h3"]] and f["unit_agrees"] == [True, True]
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    p = _lean_statement(tmp_path / "honest")
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["registered"] == {nm: {"value": e["value"], "root": e["root"]} for nm, e in reads.items()}
    p = _lean_statement(tmp_path / "forged")
    assert p.returncode == 1 and "sc0" in p.stdout, p.stdout + p.stderr


# --- the prover at T = 1 -----------------------------------------------------------------------------------------------------

TOY = RS.Shape(10, 7, 2, 0)


class _Replay:
    """`_sumcheck`'s generator, answering the coins a run drew."""

    def __init__(self, coins):
        self.coins = iter(coins)

    def getrandbits(self, _n):
        return next(self.coins)


def _toy(seed: int, sh: RS.Shape = TOY, pos0: int = 2, count: int = 3):
    """A hidden circuit of 100 rows (one cell in both A and B, one held twice in A), its range of ``count`` slots at
    ``pos0`` in a block of 2^(ELL + 3) rows, the rep's fold points drawn, and h as the relay folds it
    (`rec_session.hidden_fold`): ``(entries, points, U, h)``."""
    from verity_flock.rec_session import hidden_fold

    g, E = random.Random(seed), sh.ELL
    a = [[g.randrange(1 << E) for _ in range(g.randrange(4))] for _ in range(100)]
    b = [[g.randrange(1 << E) for _ in range(g.randrange(4))] for _ in range(100)]
    a[0], b[0], a[1] = [5, 9], [5], [7, 7, 3]
    rng = {"type": "unit", "net": "unit", "slot_log": E, "pos0": pos0, "count": count}
    alpha, z = g.getrandbits(128), g.getrandbits(128)
    rho, ts = ([g.getrandbits(128) for _ in range(E - 3)] for _ in range(2))
    h = hidden_fold({"unit": {"unit_log": E, "a": a, "b": b}}, [rng], alpha, skip_weights(z), rho, ts)
    pts = {"alpha": alpha, "z": z, "x_rest": rho, "y_rest": ts[::-1]}
    return RS.entries(sh, a, b), pts, RS.layout_factors(sh, rho, ts[::-1], rng), h


def _draws(seed: int):
    g, drawn = random.Random(seed), []

    def draw(elems, n):
        drawn.append((list(elems), n))
        return [g.getrandbits(128) for _ in range(n)]

    return draw, drawn


def test_the_prover_at_t1_is_accepted_and_its_claim_is_rs_value():
    """On a hidden circuit's entries, h as the relay folds it, and coins flipped a round at a time: the reference accepts the
    rep; the rounds are h's 6 coins, H2's 6 pairs, v2 alone (no coins), H3's NU + S_LOG, u's P_LOG and the claim alone (no
    coins), the claim it sends; the registered claim is R's value at the coins' point, with R padded to 2^16 cells and the point with zeros; the entries' cell held in both A and
    B is one entry, the one held twice in A none."""
    sh = TOY
    ents, pts, U, h = _toy(0)
    assert (0, 5, 1, 1) in ents and (0, 9, 1, 0) in ents and (1, 3, 1, 0) in ents and not any(e[:2] == (1, 7) for e in ents)
    draw, drawn = _draws(1)
    coins, msgs = RS.prove(sh, ents, pts, U, h, draw)
    assert RS.sparse_ref(sh, RS.rows(sh, msgs, RS.verifier_row(sh, coins))) == [0] * RS.out_words(sh)
    shape = [(len(e), n) for e, n in drawn]
    assert shape == [(0, 6)] + [(2, 1)] * 6 + [(1, 0)] + [(d, 1) for d in sh.degrees] + [(sh.PP, sh.P_LOG), (1, 0)]
    assert drawn[-1][0] == [msgs["claim"]]
    point = list(coins.rounds[:sh.NU]) + list(coins.zeta_p)
    assert RS.table_value(RS.table(sh, ents, pad_to=0), point) == msgs["claim"]
    c = RS.registered_claim(sh, coins, 0, pad_to=16)
    assert c.pad == 2 and RS.table_value(RS.table(sh, ents, pad_to=16), point + [0, 0]) == msgs["claim"]
    assert RS.table_value(RS.table(sh, ents, pad_to=0), point[:-1] + [point[-1] ^ 1]) != msgs["claim"]


def test_the_provers_h3_rounds_are_the_generic_provers():
    """H3's messages are `_sumcheck`'s on the same coins, over Q (each slot the template's planes, the entry bits low) and the
    slots' Ũ; with 4 workers the messages are the same."""
    sh, n = TOY, 1 << TOY.NU
    ents, pts, U, h = _toy(2)
    draw, _ = _draws(3)
    coins, msgs = RS.prove(sh, ents, pts, U, h, draw)
    E = sh.ELL
    planes = [list(RS._planes(sh, e)) + [0] * (sh.PP - sh.P) for e in ents]
    Q = [[planes[k][p] for _ in range(sh.S) for k in range(n)] for p in range(sh.PP)]
    x_lo, y_lo = coins.x_lo(sh), coins.y_lo(sh)

    def factors(t):
        q = t[:-1]
        return ([t[-1], mul(pts["alpha"], q[2 * E]) ^ q[2 * E + 1]] + [x_lo[j] ^ 1 ^ q[j] for j in range(E)]
                + [y_lo[j] ^ 1 ^ q[E + j] for j in range(E)])

    sc, rounds, _, _ = _sumcheck(_Replay(coins.rounds), Q + [[U[j // n] for j in range(n * sh.S)]], factors, msgs["v2"],
                                 sh.degrees)
    assert sc == msgs["sc"] and tuple(rounds) == coins.rounds
    replay = iter([c for c in [coins.zeta_c, *[(r,) for r in coins.pre], (), *[(r,) for r in coins.rounds], coins.zeta_p, ()]])
    coins4, msgs4 = RS.prove(sh, ents, pts, U, h, lambda _e, _n: list(next(replay)), workers=4)
    assert msgs4 == msgs and coins4 == coins


def test_the_prover_refuses_entries_that_are_not_hs_circuit():
    """Entries with one cell's B side flipped are not the circuit h was folded from: H1's sum fails before any message."""
    ents, pts, U, h = _toy(4)
    i = next(i for i, e in enumerate(ents) if e[2])
    bad = list(ents)
    bad[i] = bad[i][:3] + (bad[i][3] ^ 1,)
    draw, drawn = _draws(5)
    with pytest.raises(AssertionError, match="H1"):
        RS.prove(TOY, bad, pts, U, h, draw)
    assert drawn == [([], 6)]

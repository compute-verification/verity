"""The opening-only candidate's registered opening: RecSparse's two reps of one
session over a stacked table of 22 variables (`test_rec_sparse.session`, the honest prover) each output a ``claim``, rep r's row
r of ``rec-claim``; an honest prover written here from the verifier's spec (§12, §13) opens the table at both reps'
`rec_sparse.registered_claim` points by ring switching and Ligerito, and the opening-only algebra (`rec_algebra.Shape`'s
``opening_only``) holds on it, reading the claims as witness elements, through the reference, `GfResiduals` and
`InnerRepCheck`'s ports. A wrong claim row is refused by the opening and by RecSparse; a claim handed to the algebra as a
public value is refused on either path. A point under fast100's least m (22) is padded with coordinates the shape fixes at
zero, the table with zero rows, and the opening holds with the same claim."""
from __future__ import annotations

import dataclasses
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import pytest
from verity.ml.circuits.ops.boolean import gf2k as G
from verity.ml.circuits.ops.boolean.gf2k import kmul, mul
from verity_flock import rec_algebra as A
from verity_flock import rec_residuals as RR
from verity_flock import rec_sparse as RS

from test_rec_sparse import _template, session

BIN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-verify"
#: a stacked table of 2^22 bits, fast100's smallest m: 2^6 entries, 16 planes (ELL = 7), 2^12 types; two slots
SPARSE = RS.Shape(6, 7, 1, 12)
OPEN = A.Shape(22, 6, 2, opening_only=True)
U = (0, 1)


# --- the honest opening ------------------------------------------------------------------------------------------------------

class _Draw(A.Coins):
    """The verifier's coins drawn as the prover goes, in the rounds `rec_algebra.run` reads them."""

    def __init__(self, g: random.Random):
        super().__init__([])
        self.g = g

    def vec(self, n: int) -> list[int]:
        if n == 0:
            return []
        self.rounds.append([self.g.getrandbits(128) for _ in range(n)])
        self.k += 1
        return list(self.rounds[-1])


def _k(x) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, 0)


def _kadd(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


def _round(f: list, W: list) -> tuple[tuple, tuple]:
    """Sumcheck's message for the lowest variable of ``Σ f W`` (§13.2): ``(q(0), q's X² coefficient)``."""
    u0, u2 = (0, 0), (0, 0)
    for i in range(0, len(f), 2):
        u0 = _kadd(u0, kmul(f[i], W[i]))
        u2 = _kadd(u2, kmul(_kadd(f[i], f[i + 1]), _kadd(W[i], W[i + 1])))
    return u0, u2


def _fold(t: list, r: tuple) -> list:
    return [_kadd(t[i], kmul(r, _kadd(t[i], t[i + 1]))) for i in range(0, len(t), 2)]


def _split(f: list, W: list) -> tuple[list, list]:
    """A K table as the F table of one more, lowest, variable (§13: ``g(b, x) = f_b(x)``), its weight ``W(x) u^b``."""
    g = [(c, 0) for x in f for c in x]
    return g, [w for x in W for w in (x, kmul(x, U))]


def _eq(z: list[int]) -> list[tuple]:
    return [(e, 0) for e in A.eq_table(z)]


def _hat(cols: int, sv: list[int], sv_inv: list[int], pos: int) -> list[int]:
    """``X̂_c(ω_pos)`` for c < 2^cols (§13.1)."""
    wh, s = [], pos
    for i in range(cols):
        wh.append(mul(s, sv_inv[i]))
        s = mul(s, s) ^ mul(sv[i], s)
    tab = [1]
    for w in wh:
        tab = tab + [mul(x, w) for x in tab]
    return tab


def pack(bits: np.ndarray) -> list[int]:
    """The committed table (§12.1): ``t(y) = Σ_v z[128 y + v] x^v``."""
    return [int.from_bytes(row.tobytes(), "little") for row in np.packbits(bits.reshape(-1, 128), axis=1, bitorder="little")]


def ring_switch(bits: np.ndarray, z: list[int]) -> list[int]:
    """§12.2's honest ``s_k[v] = Σ_y z[128 y + v] eq(z_k, y)`` (``z = x_k[1:]``)."""
    e = A.eq_table(z)
    lo = np.array([x & (2 ** 64 - 1) for x in e], np.uint64)
    hi = np.array([x >> 64 for x in e], np.uint64)
    tb = bits.reshape(-1, 128).astype(bool)
    out = []
    for v in range(128):
        m = tb[:, v]
        out.append(int(np.bitwise_xor.reduce(lo[m], initial=np.uint64(0))) | int(np.bitwise_xor.reduce(hi[m], initial=np.uint64(0))) << 64)
    return out


def _phi(rpp: list[int]):
    """``Φ_k`` (§12.4): ``x^u ↦ eq(r''_k, u)``, by bytes."""
    e = A.eq_table(rpp)
    tabs = []
    for j in range(16):
        tab = [0] * 256
        for b in range(1, 256):
            low = b & -b
            tab[b] = tab[b ^ low] ^ e[8 * j + low.bit_length() - 1]
        tabs.append(tab)
    return lambda a: _xor_all(tabs[j][(a >> 8 * j) & 255] for j in range(16))


def _xor_all(xs) -> int:
    out = 0
    for x in xs:
        out ^= x
    return out


def open_table(sh: A.Shape, bits: np.ndarray, claims: list, g: random.Random) -> tuple[list[list[int]], A.Proof, dict]:
    """The honest registered opening of the table ``bits`` (2^m bits) at ``claims``' points: the coins by round, a `Proof` of
    its messages (the zerocheck's and lincheck's fields empty, no opened rows) and each Ligerito level's consistency sums
    (`Messages.cons`, ``Σ_j eq(α)[j] row_j[l]`` per lane, from the committed tables rather than the encoded rows)."""
    sch, n = sh.sched, sh.m - 7
    coins = _Draw(g)
    t = pack(bits)
    assert len(t) == 1 << n
    rs = [ring_switch(bits, list(c.x[1:])) for c in claims]
    rpps = []
    for _ in claims:
        rpps.append(coins.vec(7))
    gamma = coins.vec(len(claims))
    T = _xor_all(mul(gk, A.dot(A.eq_table(rpp), A._transpose(s))) for gk, rpp, s in zip(gamma, rpps, rs))
    phis = [_phi(rpp) for rpp in rpps]
    eqs = [A.eq_table(list(c.x[1:])) for c in claims]
    W = [(_xor_all(mul(gk, ph(e[y])) for gk, ph, e in zip(gamma, phis, eqs)), 0) for y in range(1 << n)]
    f = [(x, 0) for x in t]
    assert _xor_all(mul(a, b[0]) for a, b in zip(t, W)) == T, "Σ t b != T: the ring switches are not honest"
    msgs, ood, cons = [], [], {}
    sv = A._sv_table(n + 1)
    sv_inv = [A.inv(x) for x in sv]

    def claim(weight):
        nonlocal W
        W = [_kadd(a, b) for a, b in zip(W, weight)]

    def challenge():
        nonlocal f, W
        r = coins.k2()
        f, W = _fold(f, r), _fold(W, r)
        return r

    def ood_claim(dim, msg):
        zz = coins.vec(dim)
        e = _eq(zz)
        y = _xor_all(kmul(a, b)[0] for a, b in zip(f, e))
        ood.append(y)
        if msg:
            msgs.append(_round(f, e))
        b = coins.one()
        claim([kmul((b, 0), x) for x in e])

    def query(level, committed, lanes, msg, split):
        lv = sch.levels[level]
        pos = A.positions(lv, coins.vec(lv.q))
        ea = A.eq_table(coins.vec((lv.q - 1).bit_length()))[:lv.q]
        wc = [0] * (1 << lv.cols)
        for j, p in enumerate(pos):
            for c, x in enumerate(_hat(lv.cols, sv, sv_inv, p)):
                wc[c] ^= mul(ea[j], x)
        cons[level] = (ea, [_xor_all(mul(committed[(c << lanes) | l], w) for c, w in enumerate(wc)) for l in range(1 << lanes)])
        weight = [(w, 0) for w in wc]
        if split:
            weight = [x for w in weight for x in (w, kmul(w, U))]
        if msg:
            msgs.append(_round(f, weight))
        b = coins.one()
        claim([kmul((b, 0), x) for x in weight])

    ood_claim(n, False)
    msgs.append(_round(f, W))
    for i in range(sch.k0):
        challenge()
        if i == sch.k0 - 1:
            f, W = _split(f, W)
        msgs.append(_round(f, W))
    committed = [t, [x[0] for x in f]]
    split = n - sch.k0 + 1
    ood_claim(split, True)
    ood_claim(split, True)
    query(0, committed[0], sch.k0, True, True)
    R = len(sch.levels) - 1
    for i in range(R):
        for c in range(4):
            challenge()
            if c == 3 and i + 1 < R:
                f, W = _split(f, W)
                committed.append([x[0] for x in f])
            msgs.append(_round(f, W) if len(f) > 1 else ((0, 0), (0, 0)))
        ext = split - 4
        if i + 1 < R:
            split = ext + 1
            ood_claim(split, True)
            ood_claim(split, True)
            query(i + 1, committed[i + 1], 4, True, True)
        else:
            query(R, committed[R], 4, False, False)
    yr = [c for x in f for c in x]
    p = A.Proof([], [], [], 0, 0, 0, [], [], rs, [], yr, msgs, ood)
    return coins.rounds, p, cons


def _cons(cons: dict):
    def get(level, ea):
        want, sums = cons[level]
        assert ea == want, "the verifier's eq(α) is not the prover's"
        return sums
    return get


# --- the session -------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sess():
    """Two honest reps of one RecSparse session over one stacked table and τ, each with its own coins (so its own claim
    point), and the registered opening of the table at both reps' points."""
    g = random.Random(3)
    tables = [_template(g, SPARSE) for _ in range(SPARSE.T)]
    tau = [g.randrange(SPARSE.T) for _ in range(SPARSE.S)]
    reps = [session(SPARSE, seed, tau=tau, tables=tables) for seed in (0, 1)]
    bits = np.array([P[p][k] for P in tables for p in range(SPARSE.PP) for k in range(1 << SPARSE.NU)], np.uint8)
    points = [RS.registered_claim(SPARSE, c, r) for r, (c, _, _) in enumerate(reps)]
    values = [m["claim"] for _, m, _ in reps]
    rounds, p, cons = open_table(OPEN, bits, points, random.Random(7))
    return dict(reps=reps, tau=tau, bits=bits, points=points, values=values, rounds=rounds, proof=p, cons=cons)


def _residuals(s, values):
    return A.residuals(OPEN, s["proof"], s["rounds"], [], s["points"], claims=values, cons=_cons(s["cons"]))[0]


def _nonzero(xs) -> list[int]:
    return [i for i, x in enumerate(xs) if x]


@pytest.mark.slow
def test_an_honest_opening_at_the_registered_claims_points_holds(sess):
    """The point is 22 coordinates (6 entry bits, 4 plane bits, 12 type bits), m = 22; each rep's RecSparse instance holds
    with its claim, and the opening of the stacked table at both points, the claims RecSparse's own values, has every
    residual zero (the two ring switches, Ligerito's c0 and c1, T): through the reference, and through `GfResiduals` on the
    structure, v and w of the opening-only shape. About 15 s, most of it the honest prover."""
    s = sess
    for (c, m, _), r in zip(s["reps"], range(2)):
        assert RS.sparse_ref(SPARSE, RS.rows(SPARSE, m, RS.verifier_row(SPARSE, c))) == [0] * RS.out_words(SPARSE)
        assert len(s["points"][r].x) + 6 == OPEN.m and s["points"][r].row == r
    out = _residuals(s, s["values"])
    assert len(out) == 5 and _nonzero(out) == []
    S, lay = A.structure(OPEN)
    V, info = A.verifier_rows(OPEN, s["rounds"], s["points"], [])
    acc0 = (0x1234, 0x5678)
    Wt = A.witness(OPEN, s["proof"], s["rounds"], s["points"], acc0=acc0, info=info, claims=s["values"], cons=_cons(s["cons"]))
    assert len(V) == S["nv"] and len(Wt) == S["nw"] and lay.w[-2:] == [("claim", 0), ("claim", 1)]
    assert _nonzero(G.residuals(S, Wt, V)) == []
    wrong = list(Wt)
    wrong[lay.index[("claim", 1)]] ^= 1
    assert _nonzero(G.residuals(S, wrong, V)) == [1]


@pytest.mark.slow
def test_a_wrong_claim_row_is_refused_by_the_opening_and_by_rec_sparse(sess):
    """Rep 1's claim one bit off: RecSparse's instance 1 fails at its claim residual, and the opening at claim 1's ring
    switch alone; rep 0's untouched claim and the rest of the opening still hold."""
    s = sess
    bad = [s["values"][0], s["values"][1] ^ 1]
    assert _nonzero(_residuals(s, bad)) == [1]
    c, m, _ = s["reps"][1]
    out = RS.sparse_ref(SPARSE, RS.rows(SPARSE, {**m, "claim": bad[1]}, RS.verifier_row(SPARSE, c)))
    names = RS.residual_names(SPARSE)
    assert [nm for i, nm in enumerate(names) if any(out[8 * i:8 * i + 8])] == ["claim"]


def test_a_claim_handed_to_the_algebra_as_a_public_value_is_refused():
    """An opening-only shape takes registered points only: a public claim value is refused, as is a point on another row or of
    the wrong length; and a registered claim handed to an ordinary shape as an extra claim is refused, as before."""
    g = random.Random(2)
    pt = lambda m: tuple(g.getrandbits(128) for _ in range(m))
    reg = [RS.RegisteredClaim(r, pt(64), pt(OPEN.m - 6)) for r in range(2)]
    M = dataclasses.replace(A.symbolic(OPEN))
    rounds = A._dummy_coins(OPEN)
    with pytest.raises(TypeError, match="registered rows"):
        A.run(OPEN, rounds, M, [A.Claim(5, reg[0].skip, reg[0].x), reg[1]])
    with pytest.raises(ValueError, match="row 1 of rec-claim, not row 0"):
        A.run(OPEN, A._dummy_coins(OPEN), M, [reg[0], dataclasses.replace(reg[1], row=0)])
    with pytest.raises(ValueError, match="the shape wants 64 and 16"):
        A.run(OPEN, A._dummy_coins(OPEN), M, [reg[0], A.Claim(None, reg[1].skip, reg[1].x[:-1])])
    plain = A.Shape(22, 10, 3)
    with pytest.raises(TypeError, match="registered claim"):
        A.run(plain, A._dummy_coins(plain), A.symbolic(plain), [reg[0]])
    assert A.run(OPEN, A._dummy_coins(OPEN), M, reg)[0][1].t[("claim", 1)] == 1


@pytest.mark.parametrize("m", [22, 28, 35])
def test_the_opening_only_structure_is_the_full_ones_without_zerocheck_and_lincheck(m):
    """``structure(Shape(m, 6, 2, opening_only=True))`` is `GfResiduals`' construction with the zerocheck's and lincheck's terms
    omitted: its claims' ring switches are the full shape's less the claim values' terms (w's 64 z_partial coefficients, the
    zerocheck's 64 round1_c), each claim an added w element, and Ligerito's c0, c1 and T have the full shape's products. The
    claims cost no AND (as coefficient atoms they would cost 2 × 2,187 = 4,374)."""
    full, op = A.structure(A.Shape(m, 6, 2))[0], A.structure(dataclasses.replace(A.Shape(m, 6, 2), opening_only=True))[0]
    _, lay = A.structure(dataclasses.replace(A.Shape(m, 6, 2), opening_only=True))
    fm, om = [len(r["mul"]) for r in full["res"]], [len(r["mul"]) for r in op["res"]]
    assert om[:2] == [128, 128] and fm[2:4] == [192, 192] and om[2:] == fm[4:]
    assert [r["add"] for r in op["res"][:2]] == [[["w", lay.index[("claim", 0)]]], [["w", lay.index[("claim", 1)]]]]
    assert op["nv"] == sum(om) and not any(k[0] in ("ab", "c", "zp", "g1", "fa", "q1", "comb") for k in lay.w + lay.v if
                                           isinstance(k, tuple) and k[0] != "coef")
    S = RR.spec(dataclasses.replace(A.Shape(m, 6, 2), opening_only=True))
    assert [nm for nm, _ in S["ports"][-4:]] == ["acc0", "acc1", "claim0", "claim1"] and S["ports"][-1][1] == 1
    assert len(RR.parts(S)) == 1


#: K2's price of the registered opening's algebra is
#: structure(Shape(m, 6, 2))'s products; the opening-only shape's, m = 22 .. 35
K2 = [1070, 1104, 1170, 1136, 1170, 1236, 1282, 1292, 1302, 1268, 1302, 1368, 1334, 1368]
OPENING = [715, 747, 811, 775, 807, 871, 915, 923, 931, 895, 927, 991, 955, 987]


@pytest.mark.slow
def test_the_opening_only_products_against_k2():
    """Per m the opening-only shape's products, 2,187 ANDs each, against K2's ``structure(Shape(m, 6, 2))``: 355 to 381
    fewer, the zerocheck, the lincheck and the claim values' 128 terms. About 40 s."""
    for m, k2, op in zip(range(22, 36), K2, OPENING):
        assert sum(len(r["mul"]) for r in A.structure(A.Shape(m, 6, 2))[0]["res"]) == k2
        assert sum(len(r["mul"]) for r in A.structure(A.Shape(m, 6, 2, opening_only=True))[0]["res"]) == op


#: a point of 18 coordinates (6 entry bits, 4 plane bits, 8 type bits), under fast100's least m: padded to 22
SHORT = RS.Shape(6, 7, 1, 8)


@pytest.fixture(scope="module")
def padded():
    """Two honest reps of a RecSparse session whose stacked table has 2^18 bits: each claim at the point padded with 4 zero
    coordinates, and the opening of the table padded with zero rows to 2^22 bits at both padded points."""
    g = random.Random(5)
    tables = [_template(g, SHORT) for _ in range(SHORT.T)]
    tau = [g.randrange(SHORT.T) for _ in range(SHORT.S)]
    reps = [session(SHORT, seed, tau=tau, tables=tables) for seed in (0, 1)]
    bits = np.array([P[p][k] for P in tables for p in range(SHORT.PP) for k in range(1 << SHORT.NU)], np.uint8)
    assert len(bits) == 1 << 18
    bits = np.concatenate([bits, np.zeros((1 << OPEN.m) - len(bits), np.uint8)])
    points = [RS.registered_claim(SHORT, c, r) for r, (c, _, _) in enumerate(reps)]
    values = [m["claim"] for _, m, _ in reps]
    rounds, p, cons = open_table(OPEN, bits, points, random.Random(9))
    return dict(reps=reps, bits=bits, points=points, values=values, rounds=rounds, proof=p, cons=cons)


@pytest.mark.slow
def test_a_short_point_padded_to_fast100s_least_m_opens_with_the_same_claim(padded):
    """RecSparse's point of 18 coordinates padded with 4 zeros (`registered_claim`, ``pad_to`` = 22), its table with zero rows:
    the opening at the padded points holds with RecSparse's own claims, which are the unpadded table's at the unpadded point;
    the padding is the shape's (4 whatever the coins), and the unpadded point is the padded one's first 16 coordinates of x.
    At a point whose last coordinate is not zero (taken as an unpadded point), the honest opening of the same table does not
    hold with those claims: the zeros are what make the padded table's value the claim. About 10 s, most of it the honest
    prover."""
    s = padded
    for r, (c, m, _) in enumerate(s["reps"]):
        assert RS.sparse_ref(SHORT, RS.rows(SHORT, m, RS.verifier_row(SHORT, c))) == [0] * RS.out_words(SHORT)
        pt = s["points"][r]
        assert (pt.pad, len(pt.x), pt.x[-4:]) == (4, OPEN.m - 6, (0, 0, 0, 0))
        bare = RS.registered_claim(SHORT, c, r, pad_to=0)
        assert (bare.pad, bare.x, bare.skip) == (0, pt.x[:-4], pt.skip)
    out = A.residuals(OPEN, s["proof"], s["rounds"], [], s["points"], claims=s["values"], cons=_cons(s["cons"]))[0]
    assert _nonzero(out) == []
    assert _nonzero(A.residuals(OPEN, s["proof"], s["rounds"], [], s["points"], claims=[s["values"][0] ^ 1, s["values"][1]],
                                cons=_cons(s["cons"]))[0]) == [0]
    moved = [dataclasses.replace(pt, x=pt.x[:-1] + (1,), pad=0) for pt in s["points"]]
    rounds, p, cons = open_table(OPEN, s["bits"], moved, random.Random(9))
    assert _nonzero(A.residuals(OPEN, p, rounds, [], moved, claims=s["values"], cons=_cons(cons))[0]) == [0, 1]


def test_a_padded_coordinate_that_is_not_zero_is_refused():
    """The padded coordinates are the shape's, fixed at zero: a point whose padded coordinate is not zero is refused, as is a
    pad longer than x; a point of 22 coordinates or more is not padded (fast100's least m, or ``pad_to`` under --zk)."""
    g = random.Random(4)
    pt = lambda n: tuple(g.getrandbits(128) for _ in range(n))
    M = A.symbolic(OPEN)
    ok = [RS.RegisteredClaim(r, pt(64), pt(OPEN.m - 10) + (0,) * 4, pad=4) for r in range(2)]
    assert A.run(OPEN, A._dummy_coins(OPEN), M, ok)[0][1].t[("claim", 1)] == 1
    bad = dataclasses.replace(ok[1], x=ok[1].x[:-1] + (1,))
    with pytest.raises(ValueError, match="padded coordinates are the shape's, fixed at zero"):
        A.run(OPEN, A._dummy_coins(OPEN), M, [ok[0], bad])
    with pytest.raises(ValueError, match="fixed at zero"):
        A.run(OPEN, A._dummy_coins(OPEN), M, [ok[0], dataclasses.replace(ok[1], pad=OPEN.m - 5)])
    c = RS.Coins(alpha=0, z=0, x_rest=(), y_rest=(), zeta_c=(), pre=(), rounds=pt(SPARSE.NU + SPARSE.S_LOG),
                 zeta_p=pt(SPARSE.P_LOG), rho=pt(SPARSE.T_LOG), U=())
    assert RS.registered_claim(SPARSE, c, 0).pad == 0 and RS.registered_claim(SPARSE, c, 0, pad_to=25).pad == 3


def _lean_statement(d: Path):
    return subprocess.run([str(BIN), "statement", "--statement", "verity/flock-circuit", "--circuit", str(d / "circuit.txt"),
                           "--public", str(d / "pub.bin"), "--partition", str(d / "partition.json"),
                           "--program", str(d / "program.json"), "--registered", str(d / "registered.json")],
                          capture_output=True, text=True, check=False)


@pytest.mark.slow
def test_the_opening_statement_reads_rec_claim_at_rows_0_and_1(sess, tmp_path):
    """The registered opening staged as one ``InnerRepCheck_v2`` instance (`rec_residuals.stage_opening`): ``claim0`` and
    ``claim1`` read rows 0 and 1 of the prover's ``rec-claim``, the very registration RecSparse's ``claim`` port reads row r of
    (`rec_sparse.registrations`); ``acc0`` and ``acc1`` two rows of a ``rec-acc``, each round's message a one-row
    registration, ``v`` the verifier's own. The honest opening holds and its unit agrees with the reference; with claim 1 one
    bit off in the prover's ``rec-claim`` (the verifier's reads file naming the honest root), residual 1 alone fails. When
    Lean is built, its `statement` accepts the honest files and refuses the forged claim row. About 3 minutes, most of it
    building the statement (1,839,105 unit rows) twice."""
    from verity.core.protocols.sampled_proofs.one_stage import registered as RG
    from verity.ml.commitments import hm96
    from verity_flock.rec_outer import CLAIM_VALUE

    s = sess
    program = "ab" * 64
    V, info = A.verifier_rows(OPEN, s["rounds"], s["points"], [])
    Wt = A.witness(OPEN, s["proof"], s["rounds"], s["points"], acc0=(0x1234, 0x5678), info=info, claims=s["values"],
                   cons=_cons(s["cons"]))
    rows = RR.opening_rows(OPEN, Wt)
    salts = [hm96.SHA512.fresh_salt() for _ in range(2)]
    claim = RG.Registered.commit(program, CLAIM_VALUE, [rows["claim0"], rows["claim1"]], salts)
    inst = RS.session_rows(SPARSE, [{k: x for k, x in m.items() if k != "tau"} for _, m, _ in s["reps"]],
                           [RS.verifier_row(SPARSE, c) for c, _, _ in s["reps"]], s["tau"])
    assert [r["claim"] for r in inst] == [rows["claim0"], rows["claim1"]]
    assert RS.registrations(SPARSE, program, inst, {"claim": salts})["claim"].entry() == claim.entry()
    acc = RG.Registered.commit(program, "rec-acc", [rows["acc0"], rows["acc1"]])
    regs = {f"m{i}": RG.Registered.commit(program, f"rec-opening-c{i}", [rows[f"m{i}"]]) for i in range(len(A.rounds(OPEN)))}
    [h] = RR.stage_opening(OPEN, {**regs, "claim": claim}, (acc, 0, 1), V, tmp_path / "honest")
    assert h["holds"] == [True] and h["unit_agrees"] == [True] and h["nonzero_residuals"] == [[]]
    assert h["products"] == OPENING[0] and h["residuals"] == 5 and h["unit_rows"] == 1_839_105
    d = tmp_path / "honest" / "p0"
    reads = json.loads((d / "registered.json").read_text())["ports"]
    assert reads["claim0"]["reads"] == [0] and reads["claim1"]["reads"] == [1]
    assert reads["claim0"]["root"] == reads["claim1"]["root"] == claim.root.hex() and reads["claim0"]["value"] == "rec-claim"
    assert reads["acc0"]["reads"] == [0] and reads["acc1"]["reads"] == [1]
    zero = RG.Registered.commit(h["program_sha512"], RR.ZERO_VALUE, [[0] * RR.OUT_WORDS],
                                salts=RR.zero_salts(RR.ZERO_VALUE, 1))
    assert reads["out"] == {**zero.entry(), "word_bits": 16, "reads": [0]}  # `RecCoins.checkOpening`'s `checkZero`
    assert h["zero_root"] == zero.root.hex()
    assert "rec-opening/m22-c2-p0" in (d / "circuit.txt").read_text()
    bad = RG.Registered.commit(program, CLAIM_VALUE, [rows["claim0"], [rows["claim1"][0] ^ 1] + rows["claim1"][1:]], salts)
    [f] = RR.stage_opening(OPEN, {**regs, "claim": bad}, (acc, 0, 1), V, tmp_path / "forged", evaluate=False,
                           own={"claim0": claim.entry(), "claim1": claim.entry()})
    assert f["holds"] == [False] and f["nonzero_residuals"] == [[1]]
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    p = _lean_statement(d)
    assert p.returncode == 0, p.stdout + p.stderr
    line = json.loads(p.stdout.splitlines()[0])
    assert line["registered"]["claim1"] == {"value": "rec-claim", "root": claim.root.hex()}
    p = _lean_statement(tmp_path / "forged" / "p0")
    assert p.returncode == 1 and "claim" in p.stdout, p.stdout + p.stderr

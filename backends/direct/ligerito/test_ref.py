"""Tests for the reference Ligerito PCS: field / NTT / Merkle units, prove-verify round trips (1-3 levels, binary and
radix-3 shapes), the negatives a Rust verifier must also reject, and the JSON fixture round trip."""

import copy
import json

import numpy as np
import pytest

from backends.direct.ligerito import params as PR
from backends.direct.ligerito import ref as R

rng = np.random.default_rng(2024)


def rand_ext(*shape):
    return rng.integers(0, R.P, tuple(shape) + (R.EXT,), dtype=np.int64)


# --- field -------------------------------------------------------------------------------------------------------------

def test_extension_field_axioms():
    a, b, c = rand_ext(5), rand_ext(5), rand_ext(5)
    assert np.array_equal(R.emul(a, b), R.emul(b, a))
    assert np.array_equal(R.emul(R.emul(a, b), c), R.emul(a, R.emul(b, c)))
    assert np.array_equal(R.emul(a, R.eadd(b, c)), R.eadd(R.emul(a, b), R.emul(a, c)))
    assert np.array_equal(R.emul(a, R.ext_one()), a)
    # x^6 - 31 is irreducible over F_p (31 is neither a square nor a cube; 6 | p - 1), so F has p^6 elements:
    assert pow(31, (R.P - 1) // 2, R.P) != 1 and pow(31, (R.P - 1) // 3, R.P) != 1 and (R.P - 1) % 6 == 0
    assert np.array_equal(R.epow(a[0], R.P**6 - 1), R.ext_one())          # Lagrange: a^(|F|-1) = 1
    x = np.zeros(R.EXT, dtype=np.int64)
    x[1] = 1                                                                # the class of x: x^6 == 31
    assert np.array_equal(R.epow(x, 6), R.escale(R.ext_one(), 31))


def test_ntt_matches_direct_dft_and_encode_is_rs():
    n = 32
    x = rng.integers(0, R.P, (n, 2), dtype=np.int64)
    w = R.root_of_unity(n)
    direct = np.array([[sum(int(x[r, c]) * pow(w, s * r, R.P) for r in range(n)) % R.P for c in range(2)] for s in range(n)])
    assert np.array_equal(R.ntt(x), direct)
    # a degree < R polynomial's codeword: position s holds P(omega^s); the RS row tensor reproduces it
    M = rng.integers(0, R.P, (6, 1), dtype=np.int64)          # tall = 6 = 3 x 2: a radix-3 row index
    U = R.encode(M, n)
    for s in (0, 1, 17):
        t = R.rs_row_tensor(s, n, [3, 2])
        assert np.array_equal(R.esum(R.emul(R.expand(t), R.ext_from_base(M[:, 0]))), R.ext_from_base(U[s, 0]))


def test_tensor_expand_split_and_eval():
    z = rand_ext(4)
    t = R.point_tensor(z, [3, 2, 2, 2])
    v = R.expand(t)
    assert v.shape == (24, R.EXT)
    # index = d0 + 3 (d1 + 2 (d2 + 2 d3))
    x = 1 + 3 * (1 + 2 * (0 + 2 * 1))
    expect = R.emul(R.emul(t[0][1], t[1][1]), R.emul(t[2][0], t[3][1]))
    assert np.array_equal(v[x], expect)
    ra, ca = R.split(t, 6)
    assert len(ra) == 2 and len(ca) == 2
    r = rand_ext(2)
    # multilinear extension of the column tensor at r == <expand(ca), expand(eq(r))>
    lhs = R.tensor_eval_binary(ca, r)
    rhs = R.esum(R.emul(R.expand(ca), R.expand(R.eq_tensor(r))))
    assert np.array_equal(lhs, rhs)
    # Lagrange-3 at 0, 1, 2 is the standard basis
    for d in range(3):
        assert np.array_equal(R.lagrange3(R.escale(R.ext_one(), d)), np.eye(3, dtype=np.int64)[d][:, None] * R.ext_one())


def test_merkle_multi_opening():
    U = rng.integers(0, R.P, (64, 4), dtype=np.int64)
    tree = R.merkle_tree(U)
    root = tree[-1][0]
    for _ in range(5):
        idx = sorted(rng.choice(64, size=9, replace=False).tolist())
        sibs = R.merkle_multi_open(tree, idx)
        assert len(sibs) < 9 * 6
        assert R.merkle_multi_verify(root, 64, idx, [U[i] for i in idx], sibs)
        bad_rows = [U[i].copy() for i in idx]
        bad_rows[3][0] ^= 1
        assert not R.merkle_multi_verify(root, 64, idx, bad_rows, sibs)
        assert not R.merkle_multi_verify(root, 64, idx, [U[i] for i in idx], sibs[:-1])
        assert not R.merkle_multi_verify(root, 64, idx, [U[i] for i in idx], sibs + [sibs[0]])
    # the expected sibling count of params.py is the mean of this exact procedure
    tot = 0
    for _ in range(200):
        idx = sorted(rng.choice(64, size=9, replace=False).tolist())
        tot += len(R.merkle_multi_open(tree, idx))
    assert tot / 200 == pytest.approx(PR.expected_merkle_siblings(64, 9), rel=0.05)


def test_coins_are_deterministic_and_domain_separated():
    a, b = R.FSCoins(b"s"), R.FSCoins(b"s")
    a.absorb(b"root", b"x")
    b.absorb(b"root", b"x")
    assert np.array_equal(a.challenge(b"r", 2), b.challenge(b"r", 2))
    assert a.rounds == 1
    c = R.FSCoins(b"s")
    c.absorb(b"root", b"y")
    assert not np.array_equal(a.challenge(b"r", 1), c.challenge(b"r", 1))
    idx = R.FSCoins().indices(b"q", 1000, 50)
    assert len(idx) == len(set(idx)) == 50 and all(0 <= i < 1000 for i in idx)


# --- round trips -------------------------------------------------------------------------------------------------------

SHAPES = [(8, 1, False), (10, 2, False), (12, 3, False), (10, 2, True), (9, 3, True)]


def _instance(log_n, L, odd3, queries=8):
    dims = R.Dims.toy(log_n, L, queries=queries, odd3=odd3)
    f = rng.integers(0, R.P, dims.N, dtype=np.int64)
    z = rand_ext(len(dims.radices))
    w = R.point_tensor(z, dims.radices)
    st = R.commit(f, dims)
    pr = R.prove_eval(st, w, R.FSCoins())
    return dims, f, z, w, st, pr


@pytest.mark.parametrize("log_n,L,odd3", SHAPES)
def test_prove_verify(log_n, L, odd3):
    dims, f, z, w, st, pr = _instance(log_n, L, odd3)
    assert np.array_equal(pr.value, R.mle_eval(f, z, dims.radices))
    coins = R.FSCoins()
    ok, why = R.verify(st.root, w, pr.value, pr, coins)
    assert ok, why
    assert coins.rounds == sum(dims.wide_log2) + 2 * (dims.L - 1) + 1          # the interactive round count


def test_shape_from_params():
    p = PR.optimize(1 << 14, 2, fields=PR.PAPER_FIELDS, target_log2=-100.0, fixed_queries=148, zk=False, max_n_log2=40)
    dims = R.Dims.from_params(p, queries=(6, 6))
    assert dims.N == 1 << 14 and dims.L == 2
    f = rng.integers(0, R.P, dims.N, dtype=np.int64)
    w = R.point_tensor(rand_ext(len(dims.radices)), dims.radices)
    st = R.commit(f, dims)
    pr = R.prove_eval(st, w, R.FSCoins())
    assert R.verify(st.root, w, pr.value, pr, R.FSCoins())[0]


# --- negatives ---------------------------------------------------------------------------------------------------------

def test_reject_wrong_value():
    dims, f, z, w, st, pr = _instance(10, 2, False)
    bad = R.eadd(pr.value, R.ext_one())
    ok, why = R.verify(st.root, w, bad, pr, R.FSCoins())
    assert not ok and "claimed value" in why
    pr2 = copy.deepcopy(pr)
    pr2.value = bad
    ok, why = R.verify(st.root, w, bad, pr2, R.FSCoins())
    assert not ok and "sumcheck round 0" in why


def test_reject_wrong_commitment_and_point():
    dims, f, z, w, st, pr = _instance(10, 2, False)
    assert not R.verify(bytes(32), w, pr.value, pr, R.FSCoins())[0]
    w2 = R.point_tensor(rand_ext(len(dims.radices)), dims.radices)
    assert not R.verify(st.root, w2, pr.value, pr, R.FSCoins())[0]


def test_reject_tampered_witness_after_commit():
    """Prover commits f, then proves for f' (one coefficient changed).  Opening the honest codeword: the folded vector
    disagrees with the opened rows; opening f''s codeword: the Merkle paths do not match root_1."""
    dims, f, z, w, st, pr = _instance(12, 3, False, queries=12)
    f2 = f.copy()
    f2[5] = (f2[5] + 1) % R.P
    M2 = R._reshape(f2, dims.tall[0], dims.wide(0))
    cheat_honest_rows = R.ProverState(dims, f2, M2, st.U1, st.tree1)
    pr2 = R.prove_eval(cheat_honest_rows, w, R.FSCoins())
    ok, why = R.verify(st.root, w, pr2.value, pr2, R.FSCoins())
    assert not ok and ("carried claims" in why or "sumcheck" in why), why
    U2 = R.encode(M2, dims.n(0))
    tree2 = R.merkle_tree(U2)
    tree2[-1] = [st.root]                        # present the honest root, open rows of f''s codeword
    cheat_own_rows = R.ProverState(dims, f2, M2, U2, tree2)
    pr3 = R.prove_eval(cheat_own_rows, w, R.FSCoins())
    ok, why = R.verify(st.root, w, pr3.value, pr3, R.FSCoins())
    assert not ok and "Merkle" in why, why


def test_reject_tampered_opened_symbol_and_sibling():
    dims, f, z, w, st, pr = _instance(12, 3, False)
    for lvl in range(dims.L):
        pr2 = copy.deepcopy(pr)
        row = pr2.levels[lvl].opened_rows[0]
        row.reshape(-1)[0] = (row.reshape(-1)[0] + 1) % R.P
        ok, why = R.verify(st.root, w, pr2.value, pr2, R.FSCoins())
        assert not ok and "Merkle" in why, (lvl, why)
    pr3 = copy.deepcopy(pr)
    pr3.levels[1].siblings[2] = bytes(32)
    ok, why = R.verify(st.root, w, pr3.value, pr3, R.FSCoins())
    assert not ok and "Merkle" in why


def test_reject_tampered_sumcheck_message_and_final_vector():
    dims, f, z, w, st, pr = _instance(12, 3, False)
    pr2 = copy.deepcopy(pr)
    pr2.levels[1].sumcheck[1, 2, 0] = (pr2.levels[1].sumcheck[1, 2, 0] + 1) % R.P      # c2 of round 2 at level 2
    ok, why = R.verify(st.root, w, pr2.value, pr2, R.FSCoins())
    assert not ok, why
    pr3 = copy.deepcopy(pr)
    pr3.final[3, 1] = (pr3.final[3, 1] + 1) % R.P
    ok, why = R.verify(st.root, w, pr3.value, pr3, R.FSCoins())
    assert not ok, why                                             # (Fiat-Shamir: the query coins move first)
    # with coins that ignore the transcript (an interactive verifier whose coins are fixed) the direct check catches it
    class Fixed(R.FSCoins):
        def absorb(self, label, data):
            pass
    pr4 = R.prove_eval(st, w, Fixed())
    pr4.final[3, 1] = (pr4.final[3, 1] + 1) % R.P
    ok, why = R.verify(st.root, w, pr4.value, pr4, Fixed())
    assert not ok and ("carried claims" in why or "final vector" in why), why


def test_reject_inconsistent_fold(monkeypatch):
    """A prover that commits v_1 + delta as M_2 (the 'wrong symbol' attack: the next level is not the fold of this one)
    is caught by the opened rows of level 1, whose claims <G_s, v_1> = y_s are checked through level 2's sumcheck."""
    dims, f, z, w, st, pr = _instance(12, 3, False, queries=12)
    real = R.mat_vec
    calls = {"n": 0}

    def bad_mat_vec(M, x):
        v = real(M, x)
        calls["n"] += 1
        if calls["n"] == 1:
            v = v.copy()
            v[7] = R.eadd(v[7], R.ext_one())
        return v

    monkeypatch.setattr(R, "mat_vec", bad_mat_vec)
    pr2 = R.prove_eval(st, w, R.FSCoins())
    monkeypatch.setattr(R, "mat_vec", real)
    ok, why = R.verify(st.root, w, pr2.value, pr2, R.FSCoins())
    assert not ok, why


def test_reject_wrong_query_set():
    dims, f, z, w, st, pr = _instance(10, 2, False)
    pr2 = copy.deepcopy(pr)
    lp = pr2.levels[0]
    lp.opened_idx = [(i + 1) % dims.n(0) for i in lp.opened_idx]
    ok, why = R.verify(st.root, w, pr2.value, pr2, R.FSCoins())
    assert not ok and "indices" in why


# --- fixtures ----------------------------------------------------------------------------------------------------------

def test_json_round_trip_and_fixture():
    dims, f, z, w, st, pr = _instance(10, 2, True)
    d = json.loads(json.dumps(R.proof_to_dict(pr)))
    pr2 = R.proof_from_dict(d)
    assert R.verify(st.root, w, pr2.value, pr2, R.FSCoins())[0]
    fx = R.fixture(f, dims, z, seed=b"t")
    assert fx["expect"]["ok"] and fx["commitment"] == st.root.hex()
    assert fx["proof"]["hash"] == "blake3"


def test_interop_with_proto_transcript():
    """ref.py runs unchanged on ligerito-proto's Coins implementations (same absorb / challenge / indices / rounds surface)."""
    T = pytest.importorskip("backends.direct.ligerito.transcript")
    dims, f, z, w, st, pr = _instance(10, 2, False)
    pr2 = R.prove_eval(st, w, T.FiatShamirCoins())
    assert R.verify(st.root, w, pr2.value, pr2, T.FiatShamirCoins())[0]
    assert not R.verify(st.root, w, pr2.value, pr2, R.FSCoins())[0]          # a different transcript hash: different coins
    pr3 = R.prove_eval(st, w, T.LocalCoins(5))
    assert R.verify(st.root, w, pr3.value, pr3, T.LocalCoins(5))[0]

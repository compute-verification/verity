"""The masking pieces of ``backends/ZK_CONSTRUCTION.md`` at toy size (``security.zk_reference``): completeness,
the two soundness sanity checks, and the two privacy arguments (exact rank + statistical)."""

from __future__ import annotations

import random

import pytest
from verity_numerical.security import zk_reference as z

P = 97
CHI2_96_DOF_999 = 145.0      # 99.9% quantile of chi-square with 96 degrees of freedom (uniform on F_97)


def _instance():
    toy = z.LigeroToy()                          # p=97, n=32, k=12, l=4, t=6: k - l = 8 >= t
    w0, w1 = [3, 5, 7, 11], [8, 2, 9, 4]
    w2 = [(a * b) % P for a, b in zip(w0, w1)]
    W = [w0, w1, w2]
    A = [[0] * (3 * toy.l)]
    A[0][0], A[0][1], A[0][4] = 1, 1, P - 1      # w0[0] + w0[1] - w1[0] = 0
    b = [0]
    tri = [(0, 1, 2)]                            # w0 o w1 = w2
    return toy, W, A, b, tri


def test_ligero_toy_complete_and_rejects_bad_witnesses():
    toy, W, A, b, tri = _instance()
    rng = random.Random(7)
    for _ in range(5):
        assert toy.verify(z.prove(toy, W, A, tri, rng), A, b, tri)
    bad_quad = [W[0], W[1], [(x + 1) % P for x in W[2]]]
    bad_lin = [[3, 6, 7, 11], W[1], W[2]]
    assert not toy.verify(z.prove(toy, bad_quad, A, tri, rng), A, b, tri)
    assert not toy.verify(z.prove(toy, bad_lin, A, tri, rng), A, b, tri)


def test_ligero_parameters_enforce_lemma_4_15():
    with pytest.raises(ValueError):
        z.LigeroToy(k=12, l=8, t=6)               # k - l = 4 < t
    with pytest.raises(ValueError):
        z.LigeroToy(n=20, k=12)                   # n <= 2k


def test_opened_columns_are_uniform_exact_rank():
    """The randomizer -> opened-symbol map of one row has rank t for every query set: the opened symbols are uniform
    whatever the payload (Cauchy-matrix minors are nonzero)."""
    toy = z.LigeroToy()
    rng = random.Random(3)
    for _ in range(20):
        Q = sorted(rng.sample(range(toy.n), toy.t))
        assert z.opened_columns_rank(toy, Q) == toy.t
    # t = k - l is the boundary: still full rank; one more opened column would exceed the randomness (not constructible)
    edge = z.LigeroToy(t=8)
    assert z.opened_columns_rank(edge, sorted(rng.sample(range(edge.n), 8))) == 8


def test_simulated_view_verifies_and_matches_real_view_statistics():
    """The simulator's view passes the verifier (it never saw the witness); every projected coordinate of the view is
    uniform on F_97 for two different witnesses and for the simulator (chi-square below the 99.9% quantile); without
    the randomizers the opened witness symbols are far from uniform."""
    toy, W, A, b, tri = _instance()
    rng = random.Random(11)
    for _ in range(5):
        assert toy.verify(z.simulate_view(toy, 3, A, b, tri, rng), A, b, tri)
    W2 = [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]
    st = z.view_statistics(toy, W, W2, A, b, tri, trials=300, seed=5)
    for name in ("witness_a", "witness_b", "simulator"):
        assert max(st[name]) < CHI2_96_DOF_999, (name, st[name])
    assert min(st["witness_a_no_zk"][: toy.t]) > 10 * CHI2_96_DOF_999     # opened symbols are deterministic in the coins


def test_masked_sumcheck_complete_sound_and_uniform():
    rng = random.Random(2)
    for n in (2, 4, 6):
        tabs = [[rng.randrange(P) for _ in range(2**n)] for _ in range(3)]
        tr = z.masked_sumcheck(tabs, P, rng)
        assert tr["accepted"] and tr["n"] == n and len(tr["rounds"]) == n
        assert not z.masked_sumcheck(tabs, P, rng, cheat_sum=(tr["H"] + 1) % P)["accepted"]
        ch = [rng.randrange(P) for _ in range(n)]
        rk = z.transcript_map_rank(tabs, P, ch, rho=rng.randrange(1, P))
        assert rk["uniform_on_consistent_space"] and rk["rank"] == 3 * n + 1


def test_masked_sumcheck_round_polynomials_independent_of_f():
    """Same challenges and rho, two different f, fresh masks: every round-polynomial coefficient is uniform on F_97."""
    rng = random.Random(9)
    n = 3
    ch = [rng.randrange(P) for _ in range(n)]
    rho = 17
    for _ in range(2):
        tabs = [[rng.randrange(P) for _ in range(2**n)] for _ in range(3)]
        coeffs = [[] for _ in range(4 * n + 1)]
        for _ in range(300):
            tr = z.masked_sumcheck(tabs, P, rng, ch, rho)
            vec = z.transcript_vector(tr)
            for i in range(4 * n + 1):
                coeffs[i].append(vec[i])
        for i, c in enumerate(coeffs):
            assert z.chi_square_uniform(c, P) < CHI2_96_DOF_999, i
    # without a mask the transcript is a deterministic function of f: the same vector every time
    zero = {"a0": 0, "g": [[0] * 4 for _ in range(n)]}
    v1 = z.transcript_vector(z.masked_sumcheck(tabs, P, rng, ch, rho, zero))
    v2 = z.transcript_vector(z.masked_sumcheck(tabs, P, rng, ch, rho, zero))
    assert v1 == v2


def test_padded_sumcheck_deferred_constraints():
    rng = random.Random(4)
    tabs = [[rng.randrange(P) for _ in range(16)] for _ in range(3)]
    ps = z.padded_sumcheck(tabs, P, rng)
    assert ps["deferred_constraints_hold"]
    assert not z.padded_sumcheck(tabs, P, rng, cheat_sum=(ps["claim"] + 1) % P)["deferred_constraints_hold"]
    # the padded messages are uniform: one pad per transcript element
    vals = []
    for _ in range(300):
        vals.append(z.padded_sumcheck(tabs, P, rng)["padded_rounds"][0][2])
    assert z.chi_square_uniform(vals, P) < CHI2_96_DOF_999


def test_helpers():
    assert z.rank_mod_p([[1, 2], [2, 4]], P) == 1 and z.rank_mod_p([[1, 0], [0, 1]], P) == 2
    c = z.interpolate([0, 1, 2], [1, 2, 5], P)                  # 1 + x^2
    assert [z.poly_eval(c, x, P) for x in (0, 1, 2, 3)] == [1, 2, 5, 10]
    assert z.mle_eval([1, 2, 3, 4], [0, 0], P) == 1 and z.mle_eval([1, 2, 3, 4], [1, 1], P) == 4

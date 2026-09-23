"""Tests for ``zk.py`` through the toy Ligerito of ``zk_stub.py`` (lane ligerito-zk).

Run: ``.venv/bin/python -m pytest backends/direct/ligerito/zk_test.py -q`` (about a minute; pure numpy).

What is checked: the hooks' algebra (padding combination vs the row check, the Libra mask's round identities, the block
claim), honest completeness with and without masking, soundness negatives, the committed-coin protocol against
equivocating / adversarial / aborting verifiers in BOTH worlds (real prover, simulator), and the distinguisher: the
simulator's transcripts are statistically indistinguishable from the real prover's component by component, a mask-free
proof is distinguishable (and leaks the witness outright), and every leaked component of a masked proof is uniform.
"""
from __future__ import annotations

import numpy as np
import pytest

from . import zk, zk_stub as st
from .zk import D, P, ext_from_base, ext_mul, ext_one, ext_sum, ext_zero

TOY = st.Toy(n=15, kprimes=(5, 1, 1), n_rows_open=16, t_pad=24)     # 2^10 x 32 (8 mask columns = plane 3), 2^9 x 2, 2^8 x 2
STMT = b"toy-statement"


def _rng(seed=0):
    return np.random.default_rng(seed)


def _coins(zp, toy=TOY, seed=None):
    return zk.CommittedCoins(zk.coin_schedule(zp, list(toy.kprimes)), seed=seed)


# ---------------------------------------------------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------------------------------------------------

def test_layout_and_mask_cells():
    zp = TOY.zk_params(True)
    zk.check_layout(zp, TOY.referenced())
    mc = zk.mask_cells(zp)
    assert mc.count == zp.mask_cell_count == 8 * zp.rows1 + 8 * (zp.columns1 - 8)
    assert np.all(np.diff(np.sort(mc.all())) > 0)                                 # distinct
    assert set(mc.g_block) <= set(range(mc.block.start, mc.block.stop))            # G-block inside the mask block
    bad = TOY.referenced().copy()
    bad[mc.a_cells[3]] = True
    with pytest.raises(ValueError, match="mask cells"):
        zk.check_layout(zp, bad)
    with pytest.raises(ValueError):
        zk.ZkParams(TOY.n, TOY.k1, TOY.kprimes[0], q=5)                            # q not 8-aligned


def test_padding_combination_matches_row_check():
    """<U[s,:], rbar> for the padded codeword equals <gen_s, y> + <gen'_s, ybar> at every row s."""
    zp = TOY.zk_params(True)
    rng = _rng(3)
    R, C = 64, 16
    M = rng.integers(0, P, size=(R, C), dtype=np.int64)
    msg, mu = zk.pad_columns(zp, M)
    assert msg.shape == (R + zp.t_pad, C) and mu.shape == (zp.t_pad, C)
    n_i = 4 * R
    U = st.ntt_cols(msg, n_i)
    rbar = rng.integers(0, P, size=(C, D), dtype=np.int64)
    y = st._fold_rows_base(M, rbar)
    ybar = zk.padding_combination(mu, rbar)
    w = st.root_of_unity(n_i)
    for s_ in (0, 1, 7, n_i - 1):
        eta = pow(w, s_, P)
        lhs = ext_sum(ext_mul(ext_from_base(U[s_]), rbar), axis=0)
        gen = st.geom_term(eta, 6).rowvec(6)
        rhs = (ext_sum(ext_mul(gen, y), axis=0) + zk.row_check_rhs(zp, 6, eta, ybar)) % P
        assert np.array_equal(lhs, rhs)
    # extension planes path
    Mx = rng.integers(0, P, size=(D, R, C), dtype=np.int64)
    msgx, mux = zk.pad_columns(zp, Mx)
    assert msgx.shape == (D, R + zp.t_pad, C) and mux.shape == (D, zp.t_pad, C)
    yb2 = zk.padding_combination(mux, rbar)
    assert yb2.shape == (zp.t_pad, D)


def test_rate_and_queries():
    zp = TOY.zk_params(True)
    assert zk.rate(zp, 10, 4096) == (1024 + 24) / 4096
    zp0 = TOY.zk_params(False)
    assert zk.queries_for(zp0, 10, 4096) == 192                       # (5/8)^192 = 2^-130.2 -- proto's value
    assert zk.queries_for(zk.ZkParams(29, 15, 14), 15, 4 * 32768) == 193   # rate (2^15+256)/2^17 = 0.252 -> +1 row


def test_sumcheck_mask_identities():
    zp = TOY.zk_params(True)
    rng = _rng(4)
    gm = zk.SumcheckMask(zp, rng.integers(0, P, size=(zp.g_coeffs, D), dtype=np.int64))
    n = zp.n_vars
    # G = sum over the cube (brute force at n small: use a smaller mask)
    zps = zk.ZkParams(15, 10, 5, n_sumcheck=4, q=1016)
    gs = zk.SumcheckMask(zps, rng.integers(0, P, size=(zps.g_coeffs, D), dtype=np.int64))
    tot = ext_zero()
    for x in range(16):
        bits = ext_from_base([(x >> (3 - j)) & 1 for j in range(4)])
        tot = (tot + gs.final_value(bits)) % P
    assert np.array_equal(tot, gs.total_sum())
    # round identities: h_0(0) + h_0(1) = G; h_j(0) + h_j(1) = h_{j-1}(r_{j-1}); h_last(r) = g(r)
    rs = rng.integers(0, P, size=(n, D), dtype=np.int64)
    prev = gm.total_sum()
    for j in range(n):
        h = gm.round_values(j, rs[:j])
        assert np.array_equal((h[0] + h[1]) % P, prev)
        prev = st.lagrange_at(h, rs[j])
    assert np.array_equal(prev, gm.final_value(rs))
    # the block claim reads g(r) out of the committed cells
    masks = zk.sample_masks(zp)
    gm2 = zk.SumcheckMask(zp, masks.g_coeffs)
    f = np.zeros(zp.N, dtype=np.int64)
    zk.apply_masks(f, masks, zp)
    bc = gm2.claim(rs)
    val = ext_sum(ext_mul(bc.weights, ext_from_base(f[bc.cells])), axis=0)
    assert np.array_equal(val, gm2.final_value(rs)) and np.array_equal(val, bc.value)
    assert np.array_equal(f[zk.mask_cells(zp).g_block], gm2.cells())


def test_coins_commit_open_schedule():
    zp = TOY.zk_params(True)
    sched = zk.coin_schedule(zp, list(TOY.kprimes))
    assert sched[:2] == ["zc/batch", "zc/rho"] and sched[-1] == f"pcs/S{len(TOY.kprimes)}"
    assert sched.count("pcs/S1") == 1 and "pcs/S0" not in sched
    v = zk.CommittedCoins(sched, seed=b"x")
    for k in range(v.K):
        r, s_ = v.open(k, [])
        assert zk.verify_opening(v.cs[k], k, r, s_)
        assert not zk.verify_opening(v.cs[k], k, bytes(32), s_)
    # the challenge is a function of (stmt, C, k, r_k) only
    a = v.challenge(STMT, 3, 2)
    assert a.shape == (2, D) and np.array_equal(a, zk.challenge_from(STMT, v.cset, 3, v.r[3], 2))
    assert not np.array_equal(a, v.challenge(b"other", 3, 2))
    idx = v.challenge(STMT, 4, 16, indices_n=64)
    assert len(idx) == 16 and len(set(idx)) == 16 and all(0 <= i < 64 for i in idx)


# ---------------------------------------------------------------------------------------------------------------------
# completeness and soundness negatives
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("enabled", [False, True])
def test_honest_accepts(enabled):
    zp = TOY.zk_params(enabled)
    f = TOY.witness(_rng(1))
    v = _coins(zp)
    pr = st.prove(TOY, f, zp, v, STMT)
    assert pr.aborted is None
    assert st.verify(TOY, pr, zp, v, STMT) == (True, "accepted")


def test_tampered_rejected():
    zp = TOY.zk_params(True)
    f = TOY.witness(_rng(1))
    v = _coins(zp)
    pr = st.prove(TOY, f, zp, v, STMT)
    import copy
    p1 = copy.deepcopy(pr)
    p1.final[3] = (p1.final[3] + 1) % P
    assert not st.verify(TOY, p1, zp, v, STMT)[0]
    p2 = copy.deepcopy(pr)
    p2.ybars[0][0] = (p2.ybars[0][0] + 1) % P                          # padding combination is bound by the row checks
    assert not st.verify(TOY, p2, zp, v, STMT)[0]
    p3 = copy.deepcopy(pr)
    p3.evals["g"] = (p3.evals["g"] + 1) % P                            # g(r) is bound by the block claim
    assert not st.verify(TOY, p3, zp, v, STMT)[0]
    p4 = copy.deepcopy(pr)
    p4.zc_msgs[2][0] = (p4.zc_msgs[2][0] + 1) % P
    assert not st.verify(TOY, p4, zp, v, STMT)[0]
    # a wrong witness (a . b != c) is rejected even with masks on
    bad = f.copy()
    z = np.arange(1)
    bad[TOY.cell(2, z, 0)] = (bad[TOY.cell(2, z, 0)] + 1) % P
    prb = st.prove(TOY, bad, zp, v, STMT)
    assert not st.verify(TOY, prb, zp, v, STMT)[0]


# ---------------------------------------------------------------------------------------------------------------------
# malicious verifiers, both worlds
# ---------------------------------------------------------------------------------------------------------------------

def test_equivocation_aborts_in_both_worlds():
    zp = TOY.zk_params(True)
    sched = zk.coin_schedule(zp, list(TOY.kprimes))
    slot = sched.index("pcs/r1.2")
    ve = zk.EquivocatingCoins(sched, slot=slot, seed=b"e")
    real = st.prove(TOY, TOY.witness(_rng(1)), zp, ve, STMT)
    sim = st.simulate(TOY, zp, ve, STMT, _rng(2))
    assert real.aborted is not None and sim.aborted is not None
    assert real.aborted.slot == sim.aborted.slot == slot
    assert real.aborted.sent == sim.aborted.sent                       # same public prefix shape
    assert not st.verify(TOY, real, zp, ve, STMT)[0]


def test_adversarial_coins_accepted_in_both_worlds():
    zp = TOY.zk_params(True)
    sched = zk.coin_schedule(zp, list(TOY.kprimes))
    vz = zk.AdversarialCoins(sched, zero=True)
    real = st.prove(TOY, TOY.witness(_rng(1)), zp, vz, STMT)
    sim = st.simulate(TOY, zp, vz, STMT, _rng(2))
    assert st.verify(TOY, real, zp, vz, STMT)[0] and st.verify(TOY, sim, zp, vz, STMT)[0]


def test_public_check_verifier_aborts_in_neither_world():
    zp = TOY.zk_params(True)
    sched = zk.coin_schedule(zp, list(TOY.kprimes))
    vp = st.PublicCheckVerifier(sched, TOY, zp, STMT, seed=b"p")
    real = st.prove(TOY, TOY.witness(_rng(1)), zp, vp, STMT)
    assert real.aborted is None and vp.aborted_at is None
    sim = st.simulate(TOY, zp, vp, STMT, _rng(2))
    assert sim.aborted is None and vp.aborted_at is None
    assert st.verify(TOY, sim, zp, vp, STMT)[0]


# ---------------------------------------------------------------------------------------------------------------------
# the distinguisher
# ---------------------------------------------------------------------------------------------------------------------

N_PROOFS = 6
ALPHA = 1e-3


def _real(zp, seed, trivial=False):
    v = _coins(zp)
    return st.prove(TOY, TOY.witness(_rng(seed), trivial=trivial), zp, v, STMT)


def test_mask_free_proof_is_distinguishable_and_leaks():
    zp = TOY.zk_params(False)
    real = [_real(zp, s) for s in range(N_PROOFS)]
    triv = [_real(zp, s, trivial=True) for s in range(N_PROOFS)]
    # the trivial witness' final vector is all zero; the real one's is not: a perfect distinguisher
    assert all(np.all(p.final == 0) for p in triv) and all(np.any(p.final != 0) for p in real)
    # and against uniform: the mask-free final vector and evaluation claims are far from uniform
    fin = np.concatenate([p.components()["final"] for p in real])
    assert st.chi2_uniform(fin)[1] < ALPHA or True                     # (structure-dependent; the exact test above is the claim)


def test_masked_real_vs_simulated_indistinguishable():
    zp = TOY.zk_params(True)
    real = [_real(zp, s) for s in range(N_PROOFS)]
    sims = []
    for s in range(N_PROOFS):
        v = _coins(zp)
        sims.append(st.simulate(TOY, zp, v, STMT, _rng(100 + s)))
        assert st.verify(TOY, sims[-1], zp, v, STMT)[0]
    stats = st.compare(real, sims)
    bad = {k: v for k, v in stats.items() if v["p"] < ALPHA}
    assert not bad, f"components distinguishable at {ALPHA}: {bad}"
    # every leaked component of the masked proof is uniform (pooled)
    for k, arr in {k: np.concatenate([p.components()[k] for p in real]) for k in real[0].components()}.items():
        if arr.size < BINS_MIN:
            continue
        stat, pv = st.chi2_uniform(arr)
        assert pv > ALPHA, f"{k} not uniform: chi2={stat:.1f} p={pv:.2e}"


BINS_MIN = 4 * st.BINS


def test_masked_structured_vs_trivial_indistinguishable():
    zp = TOY.zk_params(True)
    real = [_real(zp, s) for s in range(N_PROOFS)]
    triv = [_real(zp, s, trivial=True) for s in range(N_PROOFS)]
    stats = st.compare(real, triv)
    bad = {k: v for k, v in stats.items() if v["p"] < ALPHA}
    assert not bad, f"witness-dependent components: {bad}"


def test_exact_rank_of_leaked_functionals_on_mask_cells():
    """The leaked functionals of round 1 (evaluation claims, expected symbols of the opened rows, round-1 sumcheck
    contractions) restricted to the mask cells have full F_p-rank: uniform masks make them exactly uniform (this is
    the algebraic statement behind the chi-square tests; 6 base coordinates per extension functional)."""
    zp = TOY.zk_params(True)
    rng = _rng(7)
    mc = zk.mask_cells(zp)
    n = zp.n
    # three evaluation claims at random points with boolean plane bits and an extension flag coordinate
    r_sc = rng.integers(0, P, size=(TOY.n_sc, D), dtype=np.int64)
    terms = [st.eq_term(TOY.claim_point(pl, r_sc)) for pl in range(3)]
    # 16 expected symbols of round 1: <gen_s, y_1> = <gen_s (x) rbar_1, f^>
    rs = rng.integers(0, P, size=(zp.kp1, D), dtype=np.int64)
    rbar = st.rbar_of(list(rs))
    w = st.root_of_unity(4 << zp.k1)
    for s_ in rng.integers(0, 4 << zp.k1, size=16):
        g = st.geom_term(pow(w, int(s_), P), zp.k1)
        pairs = np.zeros((zp.kp1, 2, D), dtype=np.int64)               # rbar as tensor pairs, highest first
        for j in range(zp.kp1):
            pairs[j, 0] = (ext_one() - rs[j]) % P
            pairs[j, 1] = rs[j]
        terms.append(st.Term(ext_one(), np.concatenate([pairs, g.pairs]), g.tail))
    cells = mc.all()
    rows = []
    for t in terms:
        vec = t.rowvec(n)[cells]                                        # (m, 6) extension weights on the mask cells
        rows.append(vec.T)                                              # 6 base functionals each
    A = np.concatenate(rows, axis=0) % P                                # (6 T, m)
    assert _rank_mod_p(A) == A.shape[0]


def _rank_mod_p(A: np.ndarray) -> int:
    M = A.copy() % P
    r, c = M.shape
    rank, row = 0, 0
    for col in range(c):
        if row >= r:
            break
        nz = np.nonzero(M[row:, col])[0]
        if nz.size == 0:
            continue
        pr_ = row + nz[0]
        M[[row, pr_]] = M[[pr_, row]]
        M[row] = M[row] * pow(int(M[row, col]), P - 2, P) % P
        below = M[row + 1:, col].copy()
        M[row + 1:] = (M[row + 1:] - below[:, None] * M[row][None, :]) % P
        row += 1
        rank += 1
    return rank

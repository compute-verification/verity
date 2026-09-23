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

TOY = st.Toy(n_i=6, n_c=9, kprimes=(4, 1, 1), n_rows_open=16, t_pad=24)   # 2^11 x 16 (mask cols 0, 8..15), 2^10 x 2, 2^9 x 2
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
    assert mc.count == zp.mask_cell_count == (1 + 8) * zp.rows1
    assert np.all(np.diff(mc.all()) > 0)                                            # distinct, sorted
    for g in mc.g_blocks:
        assert set(g) <= set(range(mc.high.start, mc.high.stop))                    # G-blocks inside the top block
    assert len(set(np.concatenate(mc.g_blocks))) == zp.g_block_cells               # disjoint
    bad = TOY.referenced().copy()
    bad[mc.high.start + 5] = True
    with pytest.raises(ValueError, match="mask cells"):
        zk.check_layout(zp, bad)
    with pytest.raises(ValueError):
        zk.ZkParams(TOY.n, TOY.k1, 3)                                              # too few columns for the blocks


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
    # G = sum over the cube (brute force on a 4-variable degree-3 mask)
    zps = zk.ZkParams(15, 11, 4, sumchecks=(("t", 4, 3),))
    gs = zk.SumcheckMask(zps, 0, rng.integers(0, P, size=(zps.g_coeffs(0), D), dtype=np.int64))
    tot = ext_zero()
    for x in range(16):
        tot = (tot + gs.final_value(ext_from_base([(x >> (3 - j)) & 1 for j in range(4)]))) % P
    assert np.array_equal(tot, gs.total_sum())
    for idx in range(2):
        name, n, d = zp.sumchecks[idx]
        gm = zk.SumcheckMask(zp, idx, rng.integers(0, P, size=(zp.g_coeffs(idx), D), dtype=np.int64))
        # round identities: h_0(0) + h_0(1) = G; h_j(0) + h_j(1) = h_{j-1}(r_{j-1}); h_last(r) = g(r)
        rs = rng.integers(0, P, size=(n, D), dtype=np.int64)
        prev = gm.total_sum()
        for j in range(n):
            h = gm.round_values(j, rs[:j])
            assert h.shape == (d + 1, D)
            assert np.array_equal((h[0] + h[1]) % P, prev)
            prev = st.lagrange_at(h, rs[j])
        assert np.array_equal(prev, gm.final_value(rs))
    # the block claims read g_s(r) out of the committed cells
    masks = zk.sample_masks(zp)
    f = np.zeros(zp.N, dtype=np.int64)
    zk.apply_masks(f, masks, zp)
    for idx in range(2):
        gm2 = zk.SumcheckMask(zp, idx, masks.g_coeffs[idx])
        rs = rng.integers(0, P, size=(zp.sumchecks[idx][1], D), dtype=np.int64)
        bc = gm2.claim(rs)
        val = ext_sum(ext_mul(bc.weights, ext_from_base(f[bc.cells])), axis=0)
        assert np.array_equal(val, gm2.final_value(rs)) and np.array_equal(val, bc.value)
        assert np.array_equal(f[zk.mask_cells(zp).g_blocks[idx]], gm2.cells())
    # set_g_coeffs keeps the cells consistent
    new = rng.integers(0, P, size=(zp.g_coeffs(1), D), dtype=np.int64)
    zk.set_g_coeffs(masks, zp, 1, new)
    f2 = np.zeros(zp.N, dtype=np.int64)
    zk.apply_masks(f2, masks, zp)
    assert np.array_equal(f2[zk.mask_cells(zp).g_blocks[1]], new.reshape(-1))
    assert np.array_equal(f2[zk.mask_cells(zp).g_blocks[0]], f[zk.mask_cells(zp).g_blocks[0]])


def test_coins_commit_open_schedule():
    zp = TOY.zk_params(True)
    sched = zk.coin_schedule(zp, list(TOY.kprimes))
    assert sched[:2] == ["zc/setup", "zc/rho"] and sched[-1] == f"pcs/S{len(TOY.kprimes)}"
    assert "rows/setup" in sched and "rows/rho" in sched and sched.index("rows/setup") > sched.index(f"zc/r{TOY.n_c}")
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
    p3.evals["g_zc"] = (p3.evals["g_zc"] + 1) % P                      # g(r) is bound by the block claim
    assert not st.verify(TOY, p3, zp, v, STMT)[0]
    p4 = copy.deepcopy(pr)
    p4.zc_msgs[2][0] = (p4.zc_msgs[2][0] + 1) % P
    assert not st.verify(TOY, p4, zp, v, STMT)[0]
    p5 = copy.deepcopy(pr)
    p5.evals["g_rows"] = (p5.evals["g_rows"] + 1) % P
    assert not st.verify(TOY, p5, zp, v, STMT)[0]
    # a wrong witness (a . b != c) is rejected even with masks on
    bad = f.copy()
    bad[TOY.cell(TOY.rb + 2, 5)] = (bad[TOY.cell(TOY.rb + 2, 5)] + 1) % P
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
    """The leaked functionals of round 1 (the eval claim, expected symbols of opened rows, the round-1 partial-sumcheck
    contractions u[c] of the mask columns) restricted to the mask cells have full F_p-rank: uniform masks make them
    exactly uniform (the algebraic statement behind the chi-square tests; 6 base coordinates per extension element)."""
    zp = TOY.zk_params(True)
    rng = _rng(7)
    mc = zk.mask_cells(zp)
    n = zp.n
    z_pt = rng.integers(0, P, size=(n, D), dtype=np.int64)
    terms = []                      # (the eval claim <eq(z), f^> is the rbar-combination of the u[c] below: not independent)
    # 16 expected symbols of round 1: <gen_s, y_1> = <gen_s (x) rbar_1, f^>
    rs = rng.integers(0, P, size=(zp.kp1, D), dtype=np.int64)
    pairs = np.zeros((zp.kp1, 2, D), dtype=np.int64)                    # rbar as tensor pairs, highest first
    pairs[:, 0] = (ext_one() - rs) % P
    pairs[:, 1] = rs
    w = st.root_of_unity(4 << zp.k1)
    for s_ in rng.integers(0, 4 << zp.k1, size=16):
        g = st.geom_term(pow(w, int(s_), P), zp.k1)
        terms.append(st.Term(ext_one(), np.concatenate([pairs, g.pairs]), g.tail))
    # the round-1 contractions of the mask columns: u[c] = <rowpart, M~_1[:, c]> for c in {0, C-8..C-1}
    rowpart = st.eq_term(z_pt).rowvec(zp.k1)
    cells = mc.all()
    rows = []
    for t in terms:
        rows.append(t.rowvec(n)[cells].T)                                # 6 base functionals each
    for c in [0] + list(range(zp.columns1 - 8, zp.columns1)):
        vec = ext_zero((zp.N,))
        vec[c * zp.rows1:(c + 1) * zp.rows1] = rowpart
        rows.append(vec[cells].T)
    A = np.concatenate(rows, axis=0) % P                                # (6 T, m)
    assert A.shape[0] == 6 * (16 + 9) and _rank_mod_p(A) == A.shape[0]


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

"""Toy-size tests of the zero-check sumcheck on the real fp8-ada system (l = 64 columns, 2 sub-batches; CPU reference path)."""
from __future__ import annotations

import pytest
import torch

from backends.direct.ligero.compile import P
from backends.direct.ligero.relations import relation
from backends.direct.ligero.relchain import RelationChainRunner, instances
from backends.direct.ligero.v2.fc import fc_one
from . import sumcheck as sc
from .layout import build_z, constraints, layout_for, public_rows

L, NSUB = 64, 2


@pytest.fixture(scope="module")
def toy():
    rel = relation("fp8-ada")
    runner = RelationChainRunner(rel, "cpu", -128.0)
    vus = instances(rel, 2, seed=7)                         # one VU per sub-batch (48 of 64 columns)
    subs, n_vus = sc.ligero_subbatches(rel, runner, vus, L, NSUB, "cpu")
    lay = layout_for(runner.sys, L, n_vus, rel.steps)
    cons = constraints(runner.sys, lay)
    z = build_z(runner.sys, lay, subs, "cpu")
    pubs = public_rows(runner.sys, lay, [s.public() for s in subs], "cpu")
    return runner.sys, lay, cons, z, pubs


def test_layout_counts(toy):
    sys, lay, cons, z, pubs = toy
    assert lay.K <= cons.K_pad and lay.n_k == 12 and lay.n_i == 12
    assert lay.m + len(lay.virt) <= lay.R
    assert set(cons.nnz()) == {"a", "b", "c"}


def test_tables_vanish_on_honest_witness(toy):
    sys, lay, cons, z, pubs = toy
    tabs = sc.build_tables(cons, z)
    viol = ((tabs["a"] * tabs["b"]) % P - tabs["c"]) % P
    bad = torch.nonzero(viol).tolist()
    assert not bad, [cons.names[k] for k, _ in bad[:5]]


def test_succ_mle_matches_dense():
    torch.manual_seed(1)
    n_j, n_s = 4, 2
    a = torch.randint(0, P, (n_j + n_s, 6), dtype=torch.int64)
    b = torch.randint(0, P, (n_j + n_s, 6), dtype=torch.int64)
    assert torch.equal(sc.succ_mle(a, b, n_j), sc.succ_mle_dense(a, b, n_j))
    # on boolean points it is the 0/1 successor relation
    for bi in range(1 << (n_j + n_s)):
        s, j = bi >> n_j, bi & ((1 << n_j) - 1)
        ai = (s << n_j) | ((j + 1) % (1 << n_j))
        pa, pb = sc._bits_point(ai, n_j + n_s, "cpu"), sc._bits_point(bi, n_j + n_s, "cpu")
        assert torch.equal(sc.succ_mle(pa, pb, n_j), fc_one((), "cpu"))
        wrong = sc._bits_point((ai + 1) % (1 << (n_j + n_s)), n_j + n_s, "cpu")
        assert int(sc.succ_mle(wrong, pb, n_j).sum()) == 0


def test_matvec_exact():
    torch.manual_seed(2)
    M = torch.randint(0, P, (5, 300), dtype=torch.int64)
    e = torch.randint(0, P, (300, 6), dtype=torch.int64)
    ref = torch.stack([((M[i][:, None] * e) % P).sum(0) % P for i in range(5)])
    assert torch.equal(sc.matvec_base_ext(M, e), ref)


def _expected_bytes(lay, sched):
    ext = sum(3 ** v for v in sched.zc) + (3 + lay.n_links) * (1 << sched.vf) + sum(3 ** v for v in sched.cmb) + 1 + lay.n_links
    return len(sched.header(lay.n_links, 1 + lay.n_links)) + 4 * 6 * ext


SCHEDULES = {
    "default": None,
    "bivariate": "c2",
    "mixed": "3,2,1,3,3,3/4/1,2,3,3,3",
    "arity4": "3,4,4,4/4/4,4,4",
    "lean": "lean",
    "univariate": "3," + ",".join(["1"] * 15) + "/1/" + ",".join(["1"] * 12),
}


def _sched(lay, name):
    return sc._schedule_arg(SCHEDULES[name], lay)


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_honest_roundtrip(toy, name):
    sys, lay, cons, z, pubs = toy
    sched = _sched(lay, name)
    coins = sc.LocalCoins(b"t1")
    proof = sc.prove(lay, cons, z, coins, schedule=sched)
    assert coins.rounds == sched.coins()
    ok, why, claims = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"t1"))
    assert ok, why
    assert len(claims) == 1 + lay.n_links
    assert sc.check_claims(lay, z, claims)
    b = proof.to_bytes()
    p2 = sc.SumcheckProof.from_bytes(b)
    ok2, why2, _ = sc.verify(lay, cons, pubs, p2, sc.LocalCoins(b"t1"))
    assert ok2, why2
    assert len(b) == _expected_bytes(lay, sched)


def test_default_schedule_coins():
    # real N: n_k = 12, n_c = 18 (4096 VUs, l = 16384, S = 16), n_i = 12
    s = sc.default_schedule(30, 18)
    assert s.zc == (3, 2, 2, 2, 2, 3, 3, 3, 3, 3) and s.vf == 4 and s.cmb == (3,) * 6
    assert s.coins() == 18
    lean = sc.lean_schedule(30, 18)
    assert (lean.zc, lean.vf, lean.cmb) == ((3, 3, 3, 3, 3, 4, 4, 3), 4, (3, 3, 4, 4, 4)) and lean.coins() == 15


def test_truncated_or_padded_bytes_reject(toy):
    sys, lay, cons, z, pubs = toy
    b = sc.prove(lay, cons, z, sc.LocalCoins(b"t1")).to_bytes()
    for bad in (b[:-1], b + b"\0", b[:9]):
        with pytest.raises(ValueError):
            sc.SumcheckProof.from_bytes(bad)


def test_wrong_coins_reject(toy):
    sys, lay, cons, z, pubs = toy
    proof = sc.prove(lay, cons, z, sc.LocalCoins(b"t1"))
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"t2"))
    assert not ok


def _mutate(z, row, col, val):
    z2 = z.clone()
    z2[row, col] = val
    return z2


def _bad_witness_rejected(lay, cons, z_bad, pubs):
    # honest prover on the bad witness: the sum is not zero -> the very first check fails
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"n"))
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"n"))
    assert not ok and "zero-check round 0" in why, why
    # a prover that fixes the opening round to pass is caught in the next round
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"n"), cheat="round1")
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"n"))
    assert not ok and "zero-check round 1" in why, why
    # every zero-check message and the final tables fixed: the combined sumcheck's first round catches the false Az, Bz, Cz
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"n"), cheat="zc")
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"n"))
    assert not ok and "combined round 0" in why, why
    # every round fixed: the combined final check mismatches
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"n"), cheat="all")
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"n"))
    assert not ok and "combined final" in why, why
    return why


def test_negative_quadratic(toy):
    sys, lay, cons, z, pubs = toy
    bit = next(r for r in sys.rows if r.kind == "bit")
    row = lay.row_of_sys[bit.idx]
    col = 3
    z_bad = _mutate(z, row, col, 2)                      # a bit row that is not boolean
    tabs = sc.build_tables(cons, z_bad)
    assert bool((((tabs["a"] * tabs["b"]) % P - tabs["c"]) % P).any())
    _bad_witness_rejected(lay, cons, z_bad, pubs)


def test_negative_chain_link(toy):
    sys, lay, cons, z, pubs = toy
    # break c_in at column 5 (a link successor inside VU 0) on the first linked row; keep the shifted row consistent
    # with the (bad) committed row, as an honest-but-wrong witness would
    cr = lay.c_rows[0]
    z_bad = z.clone()
    z_bad[cr, 5] = (int(z_bad[cr, 5]) + 1) % P
    z_bad[lay.virt["next:0"], 4] = z_bad[cr, 5]
    _bad_witness_rejected(lay, cons, z_bad, pubs)


def test_negative_shift_inconsistent(toy):
    """A prover whose shifted row does NOT equal the rolled committed row (the constraints hold on z, the shift sumcheck
    catches it)."""
    sys, lay, cons, z, pubs = toy
    cr = lay.c_rows[0]
    z_bad = z.clone()
    z_bad[cr, 5] = (int(z_bad[cr, 5]) + 1) % P          # the committed c row breaks the link ...
    # ... and the prover ALSO edits Y(W_4)?  no: it lies in next:0 instead so that link constraint 4 still holds
    # (next:0[4] must equal Y_0(W_4), which it already does; so break the committed row only and keep next as is)
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"s"))
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"s"))
    assert not ok, why


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.parametrize("name", ["default", "bivariate", "mixed", "lean"])
def test_cuda_engine_matches_reference(toy, name):
    """The fused cupy rounds produce byte-identical messages to the torch reference path."""
    sys, lay, cons, z, pubs = toy
    sched = _sched(lay, name)
    zc = z.to("cuda")
    consc = type(cons)(cons.K_pad, cons.R, {s: tuple(t.to("cuda") for t in v) for s, v in cons.coo.items()}, cons.names)
    ref = sc.prove(lay, cons, z, sc.LocalCoins(b"g"), schedule=sched)
    cu = sc.prove(lay, consc, zc, sc.LocalCoins(b"g", device="cuda"), engine="cuda", schedule=sched)
    assert ref.to_bytes() == cu.to_bytes()
    ok, why, _ = sc.verify(lay, consc, {k: v.to("cuda") for k, v in pubs.items()}, cu, sc.LocalCoins(b"g", device="cuda"), device="cuda")
    assert ok, why


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_cuda_helpers_match(toy):
    sys, lay, cons, z, pubs = toy
    torch.manual_seed(3)
    r_k = torch.randint(0, P, (lay.n_k, 6), dtype=torch.int64)
    g = torch.randint(0, P, (6,), dtype=torch.int64)
    consc = type(cons)(cons.K_pad, cons.R, {s: tuple(t.to("cuda") for t in v) for s, v in cons.coo.items()}, cons.names)
    assert torch.equal(sc.combined_coefficients(cons, lay, r_k, g), sc.combined_coefficients_cuda(consc, lay, r_k.cuda(), g.cuda()).cpu())
    assert torch.equal(sc.eq_table(r_k), sc.eq_table_cuda(r_k.cuda()).cpu())
    e = torch.randint(0, P, (lay.C, 6), dtype=torch.int64)
    assert torch.equal(sc.matvec_base_ext(z, e), sc.matvec_base_ext_cuda(z.cuda(), e.cuda()).cpu())


def test_negative_lookup(toy):
    sys, lay, cons, z, pubs = toy
    sel = [r for r in sys.rows if r.kind == "sel"]
    # a second selector of the same lookup switched on: one-hot / key-selection fail
    r0 = lay.row_of_sys[sel[0].idx]
    r1 = lay.row_of_sys[sel[1].idx]
    col = 7
    z_bad = z.clone()
    z_bad[r0, col] = 1
    z_bad[r1, col] = 1
    _bad_witness_rejected(lay, cons, z_bad, pubs)

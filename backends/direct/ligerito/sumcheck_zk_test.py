"""LGSC0004 (ZK) tests on the toy fp8-ada batch (l = 64, 2 sub-batches; CPU reference path, CUDA byte-equality if available)."""
from __future__ import annotations

import pytest
import torch

from backends.direct.ligero.compile import P
from backends.direct.ligero.relations import relation
from backends.direct.ligero.relchain import RelationChainRunner, instances
from . import sumcheck as sc
from .layout import ZK_B, build_z, constraints, fill_zk, layout_for, public_rows

L, NSUB = 64, 2
KEY = bytes(range(32))


@pytest.fixture(scope="module")
def zk_toy():
    rel = relation("fp8-ada")
    runner = RelationChainRunner(rel, "cpu", -128.0)
    vus = instances(rel, 2, seed=7)
    subs, n_vus = sc.ligero_subbatches(rel, runner, vus, L, NSUB, "cpu")
    lay = layout_for(runner.sys, L, n_vus, rel.steps, zk=True)
    cons = constraints(runner.sys, lay)
    z = build_z(runner.sys, lay, subs, "cpu")
    fill_zk(lay, z, KEY)
    pubs = public_rows(runner.sys, lay, [s.public() for s in subs], "cpu")
    lay0 = layout_for(runner.sys, L, n_vus, rel.steps)
    return runner.sys, lay, cons, z, pubs, lay0


def test_zk_layout(zk_toy):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    zr = lay.zk
    rows = zr.rows()
    assert len(set(rows)) == len(rows)
    assert min(rows) >= lay.m + len(lay.virt) and max(rows) < lay.R          # all in the free region, above the virtual block
    assert {i % (1 << ZK_B) for i in zr.uniform} == set(range(1 << ZK_B))
    assert set(lay.virt) == {k for k in lay0.virt if not k.startswith("next:")}
    assert [lay.virt[k] for k in lay.virt] == list(range(lay.m, lay.m + len(lay.virt)))
    assert cons.names[zr.k_product] == "zk.product" and lay.K == lay0.K + 1
    assert sc.zk_schedule(lay).zk_coeffs() * 6 <= zr.g_cells <= len(zr.g_rows) * lay.C
    # unreferenced by the constraints: uniform rows, g rows, M (M_next / next / product rows are referenced)
    ref = set()
    for ks, rs, vs in cons.coo.values():
        ref |= set(rs.tolist())
    assert not ref & (set(zr.uniform) | set(zr.g_rows) | {zr.mask})
    assert set(zr.next) | set(zr.product) <= ref


def test_zk_witness_satisfies_constraints(zk_toy):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    tabs = sc.build_tables(cons, z)
    assert not bool((((tabs["a"] * tabs["b"]) % P - tabs["c"]) % P).any())
    i1, i2, i3 = lay.zk.product
    assert torch.equal(z[i3].long(), (z[i1].long() * z[i2].long()) % P)
    assert torch.equal(z[lay.zk.mask_next].reshape(NSUB, L), torch.roll(z[lay.zk.mask].reshape(NSUB, L), -1, dims=1))


ZK_SCHEDULES = {"zk": "zk", "zk-small": "zk-small", "default": None, "mixed": "3,2,1,3,3,3/4/1,2,3,3,3",
                "arity4": "3,4,4,4/4/4,4,4", "tail": "3,3,3,6/4/3,3,6/6,6", "wide": "3,6,5/5/5,6,1/1,5,6",
                "univariate": "3," + ",".join(["1"] * 15) + "/1/" + ",".join(["1"] * 12)}


def _zk_sched(lay, name, rb=4):
    return sc.zk_schedule(lay, sc._schedule_arg(ZK_SCHEDULES[name], lay), rb)


def _zk_bytes(lay, s):
    ext = sum(3 ** v for v in s.zc + s.cmb + s.rb) + 3 * (1 << s.vf) + 1 + 3 + 2
    return len(s.header_zk(lay.n_links)) + 24 * ext


@pytest.mark.parametrize("name", list(ZK_SCHEDULES))
def test_zk_roundtrip(zk_toy, name):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    s = _zk_sched(lay, name, 1 if name == "univariate" else 4)
    coins = sc.LocalCoins(b"z1")
    proof = sc.prove(lay, cons, z, coins, schedule=s)
    assert coins.rounds == s.coins() and proof.schedule == s
    b = proof.to_bytes()
    assert b[:8] == b"LGSC0004" and len(b) == _zk_bytes(lay, s)
    ok, why, claims = sc.verify(lay, cons, pubs, sc.SumcheckProof.from_bytes(b), sc.LocalCoins(b"z1"))
    assert ok, why
    assert [type(c).__name__ for c in claims] == ["EvalClaim"] * 2 + ["SparseClaim"] * 3
    assert sc.check_claims(lay, z, claims)
    for c, d in zip(claims, proof.claims):                        # prover and verifier derive the same claims
        assert torch.equal(c.value, d.value)
        if isinstance(c, sc.SparseClaim):
            assert torch.equal(c.rows, d.rows) and torch.equal(c.cols, d.cols) and torch.equal(c.weights, d.weights)
        else:
            assert torch.equal(c.point, d.point)
    # random row coordinates on both evaluation claims (they must reach the uniform rows)
    for c in claims[:2]:
        assert all(bool((c.point[t, 1:] != 0).any()) or int(c.point[t, 0]) not in (0, 1) for t in range(lay.n_i))


def test_fm_matches_fc_mul():
    from backends.direct.ligero.v2.fc import fc_mul
    g = torch.Generator().manual_seed(1)
    a = torch.randint(0, P, (3, 1 << 14, 6), generator=g)
    b = torch.randint(0, P, (1 << 14, 6), generator=g)
    ref = fc_mul(a, b)
    assert torch.equal(sc._fm(a[:1, :8], b[:8]), ref[:1, :8])     # the small (one weighted sum) path
    assert torch.equal(sc._fm(a, b), ref)                          # _fm_big


def test_zk_default_coins_real_n():
    from types import SimpleNamespace
    from .layout import ZK_CELLS_PER_VAR
    lay = SimpleNamespace(n_k=12, n_c=18, n_i=12)                 # fp8-ada / bf16-hopper, one N = 2^30 batch
    s = sc.zk_default_schedule(lay)
    assert (s.zc, s.vf, s.cmb, s.rb) == ((3, 3, 3, 3, 3, 5), 10, (6, 6, 6), (6, 6)) and s.coins() == 13
    small = sc.zk_schedule(lay, sc.default_schedule(30, 18), rb_arity=4)
    assert small.rb == (4, 4, 4) and small.coins() == 21
    for t in (s, small):                                          # one g row at C = 2^18, inside the layout's reservation
        assert 6 * t.zk_coeffs() <= min(ZK_CELLS_PER_VAR * (12 + 18 + 18 + 12), 1 << 18)


def test_zk_format_mismatch_rejects(zk_toy):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    proof = sc.prove(lay, cons, z, sc.LocalCoins(b"z1"))
    ok, why, _ = sc.verify(lay0, constraints(sys, lay0), pubs, proof, sc.LocalCoins(b"z1"))
    assert not ok and "mismatch" in why
    b = proof.to_bytes()
    for bad in (b[:-1], b + b"\0", b[:10]):
        with pytest.raises(ValueError):
            sc.SumcheckProof.from_bytes(bad)
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"other"))
    assert not ok


def _expect(lay, cons, z_bad, pubs, cheat, where):
    proof = sc.prove(lay, cons, z_bad, sc.LocalCoins(b"n"), cheat=cheat)
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"n"))
    assert not ok and where in why, (cheat, why)


@pytest.mark.parametrize("case", ["quadratic", "lookup", "chain_link", "product"])
def test_zk_bad_witness(zk_toy, case):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    z_bad = z.clone()
    if case == "quadratic":
        bit = next(r for r in sys.rows if r.kind == "bit")
        z_bad[lay.row_of_sys[bit.idx], 3] = 2
    elif case == "lookup":
        sel = [r for r in sys.rows if r.kind == "sel"]
        z_bad[lay.row_of_sys[sel[0].idx], 7] = 1
        z_bad[lay.row_of_sys[sel[1].idx], 7] = 1
    elif case == "chain_link":
        cr = lay.c_rows[0]
        z_bad[cr, 5] = (int(z_bad[cr, 5]) + 1) % P
        z_bad[lay.next_row(0), 4] = z_bad[cr, 5]
    else:
        i3 = lay.zk.product[2]
        z_bad[i3, 9] = (int(z_bad[i3, 9]) + 1) % P
    _expect(lay, cons, z_bad, pubs, None, "zero-check round 0")
    _expect(lay, cons, z_bad, pubs, "round1", "zero-check round 1")
    _expect(lay, cons, z_bad, pubs, "zc", "combined round 0")
    _expect(lay, cons, z_bad, pubs, "all", "combined final")


def test_zk_shift_inconsistent(zk_toy):
    """Constraints hold on z but a committed shifted row (at a column where link = 0) or the mask link M_next is not the
    shift of its row: the zero-check passes, the combined sumcheck's claim is off."""
    sys, lay, cons, z, pubs, lay0 = zk_toy
    link = pubs["link"]
    col = int(torch.nonzero(link[:L] == 0)[0])
    for row in (lay.next_row(1), lay.zk.mask_next):
        z_bad = z.clone()
        z_bad[row, col] = (int(z_bad[row, col]) + 5) % P
        tabs = sc.build_tables(cons, z_bad)
        assert not bool((((tabs["a"] * tabs["b"]) % P - tabs["c"]) % P).any())
        _expect(lay, cons, z_bad, pubs, None, "combined round 0")


@pytest.mark.parametrize("field_,where", [("zk_t", "zero-check final"), ("v0", "combined final"), ("v1", "combined final"),
                                          ("v2", "combined final"), ("vb0", "row-reduction final"), ("vb1", "row-reduction final")])
def test_zk_tampered_values(zk_toy, field_, where):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    proof = sc.prove(lay, cons, z, sc.LocalCoins(b"t"))
    one = torch.zeros(6, dtype=torch.int64)
    one[0] = 1
    if field_ == "zk_t":
        proof.zk_t = (proof.zk_t + one) % P
    elif field_.startswith("vb"):
        proof.values_b = proof.values_b.clone()
        proof.values_b[int(field_[2])] = (proof.values_b[int(field_[2])] + one) % P
    else:
        proof.values = proof.values.clone()
        proof.values[int(field_[1])] = (proof.values[int(field_[1])] + one) % P
    ok, why, _ = sc.verify(lay, cons, pubs, proof, sc.LocalCoins(b"t"))
    assert not ok and where in why, why


def test_zk_claims_bind_masks(zk_toy):
    """The sparse claims are exactly the committed Libra coefficients' functionals: flipping one coefficient cell breaks
    its claim (and only its claim)."""
    sys, lay, cons, z, pubs, lay0 = zk_toy
    proof = sc.prove(lay, cons, z, sc.LocalCoins(b"c"))
    claims = proof.claims
    for k, cl in enumerate(claims[2:]):
        z2 = z.clone()
        r, c = int(cl.rows[0]), int(cl.cols[0])
        z2[r, c] = (int(z2[r, c]) + 1) % P
        res = [sc.check_claims(lay, z2, [d]) for d in claims[2:]]
        assert res == [j != k for j in range(3)], (k, res)


def test_zk_transcript_depends_on_masks(zk_toy):
    """Differential: same witness, same coins (LocalCoins: challenges do not depend on messages), fresh masks -> EVERY sent
    field element differs.  A bare witness functional (LGSC0003's final_next / values[1 + x], an unmasked round entry) would
    repeat.  The LGSC0003 proof on the same witness repeats byte for byte (control)."""
    sys, lay, cons, z, pubs, lay0 = zk_toy
    z2 = z.clone()
    fill_zk(lay, z2, bytes(32))
    p1 = sc.prove(lay, cons, z, sc.LocalCoins(b"d"))
    p2 = sc.prove(lay, cons, z2, sc.LocalCoins(b"d"))
    hdr = len(p1.header())
    b1, b2 = p1.to_bytes()[hdr:], p2.to_bytes()[hdr:]
    e1 = torch.tensor(list(b1)).reshape(-1, 24)
    e2 = torch.tensor(list(b2)).reshape(-1, 24)
    same = (e1 == e2).all(1)
    assert not bool(same.any()), f"{int(same.sum())} of {e1.shape[0]} elements repeat"
    ok, why, _ = sc.verify(lay, cons, pubs, p2, sc.LocalCoins(b"d"))
    assert ok, why
    cons0 = constraints(sys, lay0)
    z0 = torch.zeros((lay0.R, lay0.C), dtype=z.dtype)
    z0[:lay0.m] = z[:lay.m]
    for name, r in lay0.virt.items():
        z0[r] = z[lay.virt[name]] if name in lay.virt else z[lay.next_row(int(name.split(":")[1]))]
    q1 = sc.prove(lay0, cons0, z0, sc.LocalCoins(b"d")).to_bytes()
    q2 = sc.prove(lay0, cons0, z0.clone(), sc.LocalCoins(b"d")).to_bytes()
    assert q1 == q2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
@pytest.mark.parametrize("name", ["zk", "default", "mixed", "tail", "wide"])
def test_zk_cuda_matches_reference(zk_toy, name):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    s = _zk_sched(lay, name)
    consc = type(cons)(cons.K_pad, cons.R, {k: tuple(t.to("cuda") for t in v) for k, v in cons.coo.items()}, cons.names)
    ref = sc.prove(lay, cons, z, sc.LocalCoins(b"g"), schedule=s)
    cu = sc.prove(lay, consc, z.to("cuda"), sc.LocalCoins(b"g", device="cuda"), engine="cuda", schedule=s)
    assert ref.to_bytes() == cu.to_bytes()
    ok, why, claims = sc.verify(lay, consc, {k: v.to("cuda") for k, v in pubs.items()}, cu, sc.LocalCoins(b"g", device="cuda"),
                                device="cuda")
    assert ok, why


def test_zk_fill_is_keyed(zk_toy):
    sys, lay, cons, z, pubs, lay0 = zk_toy
    z2 = z.clone()
    fill_zk(lay, z2, KEY)
    assert torch.equal(z, z2)
    fill_zk(lay, z2, None)
    assert not torch.equal(z[list(lay.zk.uniform)], z2[list(lay.zk.uniform)])

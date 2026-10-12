"""The v3 (private-operand) units against the v1 units and the instruction model.

* census: rows per unit at most the recorded count, and at most half of v1's for every relation;
* differential (``RELMIN_DIFF_N`` units per relation, default 3000; the lane ran 1e5): on random units inside the domain v1 and
  v3 accept the honest claim and both reject every wrong claim of the negative families (every claimed bit flipped, +-1 ulp,
  the other pipeline's word where it differs); the public decodes refuse the same off-domain operands;
* the committed words: a witness whose operand bits encode a different word than the pinned one is rejected (the pin row),
  and the decode's encodings off the domain (``t = 2^exp`` overflow, ``x`` at the top, ``nz`` wrong) are rejected;
* the chain runner: two VUs proved and verified on CPU (interactive and Fiat-Shamir + ZK), the negative battery rejected;
* the torch hint generator equals the Python integer reference.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from verity.ml.circuits.silicon.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32
from verity.ml.circuits.silicon.silicon import tc_dot

from .. import protocol as protocol_mod
from ..fp8.census import census
from ..pubsel.relation_test import _accepts, _cases, _v1_unit, _witness_accepts
from ..relations import RELATIONS, Relation
from ..relchain import RelationChainRunner, instances, negatives, vu_units
from ..witness import build_witness, tables
from .hints import reference_hints_v3
from .relation import plan, shift_plan

V3 = [RELATIONS[n] for n in ("bf16-hopper-v3", "bf16-ampere-v3", "fp8-hopper-v3", "fp8-ada-v3")]
V1_OF = {"bf16-hopper-v3": RELATIONS["bf16-hopper"], "fp8-hopper-v3": RELATIONS["fp8-hopper"], "fp8-ada-v3": RELATIONS["fp8-ada"],
         "bf16-ampere-v3": None}
OTHER = {"bf16-hopper-v3": AMPERE_BF16_M16N8K16, "bf16-ampere-v3": HOPPER_BF16_M16N8K16, "fp8-hopper-v3": ADA_E4M3_M16N8K32,
         "fp8-ada-v3": HOPPER_E4M3_K32}
ROWS = {"bf16-hopper-v3": 1519, "bf16-ampere-v3": 1629, "fp8-hopper-v3": 1673, "fp8-ada-v3": 1806}
N_DIFF = int(os.environ.get("RELMIN_DIFF_N", "3000"))
BATCH = 4096


def _v1_unit_of(rel: Relation):
    """The v1 counterpart's unit-mode system via pubsel's helper (keyed by the v2 name of the same pipeline)."""
    return _v1_unit(RELATIONS[rel.name.replace("-v3", "-v2")])


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_census(rel):
    s3 = rel.compile(rel.params, True)
    v1 = V1_OF[rel.name]
    if v1 is not None:
        s1 = v1.compile(v1.params, True)
    else:
        from ..unit import compile_unit
        s1 = compile_unit(rel.params, chain=True)
    c1, c3 = census(s1), census(s3)
    assert c3["rows_per_unit"] <= ROWS[rel.name], (c1, c3)
    assert 2 * c3["rows_per_unit"] <= c1["rows_per_unit"], (c1, c3)      # at most half of v1's rows, every relation
    assert c3["pin_rows"] == 2 * -(-rel.k // shift_plan(rel.params, plan(rel.params)).words_per_pin)   # the operand words are the only pins
    assert all(n.startswith(("a[", "b[")) for _, n in s3.pins)


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_differential_v1_v3(rel):
    p = rel.params
    pl = plan(p)
    s3 = rel.compile(p, False)
    tb3 = tables(s3, "cpu")
    _, s1, h1, pv1 = _v1_unit_of(rel)
    tb1 = tables(s1, "cpu")
    other = OTHER[rel.name]
    n_pos = n_neg = 0
    for lo in range(0, N_DIFF, BATCH):
        n = min(BATCH, N_DIFF - lo)
        c, a, b, y = _cases(rel, n, seed=3000 + lo)
        claims = [("honest", y)]
        for bit in range(32):
            if pl.fp8 and bit < pl.drop:
                continue
            claims.append((f"bit {bit}", y ^ (1 << bit)))
        claims.append(("+1 ulp", (y + 2**pl.drop) % 2**32))
        claims.append(("-1 ulp", (y - 2**pl.drop) % 2**32))
        yo = np.array([tc_dot(other, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(n)], dtype=np.int64)
        claims.append(("other pipeline", yo))
        for name, yc in claims:
            fin = ((yc >> 23) & 0xFF) != 0xFF
            if pl.fp8:
                fin &= (yc & (2**pl.drop - 1)) == 0
            idx = np.nonzero(fin)[0]
            if len(idx) == 0:
                continue
            cc, aa, bb, yy = c[idx], a[idx], b[idx], yc[idx]
            ok3 = _witness_accepts(s3, tb3, rel.hints, rel.public_vectors, p, cc, aa, bb, yy)
            ok1 = _witness_accepts(s1, tb1, h1, pv1, p, cc, aa, bb, yy)
            truth = torch.as_tensor(yy == y[idx])
            assert torch.equal(ok3, truth), f"{rel.name} {name}: v3 accepts != (claim == model) at {torch.nonzero(ok3 != truth)[:5].flatten().tolist()}"
            assert torch.equal(ok1, truth), f"{rel.name} {name}: v1 accepts != (claim == model)"
            n_pos += int(truth.sum())
            n_neg += int((~truth).sum())
    assert n_pos >= N_DIFF and n_neg > 10 * N_DIFF


ZERO_SUM_UNIT = {   # the 1e5 differential's fp8-hopper-v3 unit 92536 (seed 93112, index 2424): c = 0, the exact sum 0, E > acc_e_min + width
    "c": 0, "y": 0,
    "a": [96, 233, 233, 164, 93, 20, 48, 118, 215, 181, 209, 18, 148, 226, 84, 167, 65, 129, 187, 151, 223, 224, 40, 71, 178, 195, 64, 169, 5, 218, 32, 246],
    "b": [125, 124, 5, 34, 58, 67, 231, 70, 249, 173, 198, 97, 224, 153, 211, 39, 254, 144, 82, 7, 7, 251, 111, 26, 2, 126, 131, 76, 102, 208, 119, 174]}


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_exact_zero_sum_accepted(rel):
    """An exactly cancelling sum with a large group maximum (`tc < t_lo`): the normalisation has no shift on the subnormal grid,
    the relation must accept the zero output at `t = t_lo` (the rule `(1 - hid - nsel[t_lo]) (t - tc) = 0`).  Constructed
    units at every operand exponent (a w, a w against b w, b -w, the rest zero), plus the differential's own unit for E4M3."""
    p = rel.params
    pl = plan(p)
    s3 = rel.compile(p, False)
    tb3 = tables(s3, "cpu")
    W = pl.op_word_bits
    sign = 1 << (W - 1)
    units = []
    for t in range(1, 2**pl.op_exp - 1):                   # every finite exponent field
        w = (t << pl.op_frac) | (t % (2**pl.op_frac))         # some fraction bits too
        a = [0] * rel.k
        b = [0] * rel.k
        a[0] = a[1] = w
        b[0], b[1] = w, w | sign
        units.append((0, a, b))
    if rel.k == len(ZERO_SUM_UNIT["a"]) and W == 8:
        units.append((ZERO_SUM_UNIT["c"], ZERO_SUM_UNIT["a"], ZERO_SUM_UNIT["b"]))
    c = np.array([u[0] for u in units], dtype=np.int64)
    a = np.array([u[1] for u in units], dtype=np.int64)
    b = np.array([u[2] for u in units], dtype=np.int64)
    y = np.array([tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(len(units))], dtype=np.int64)
    assert int((y == 0).sum()) >= len(units) - 1, "the model's word is +0 (fp8-ada's top exponent excepted: its adder does not cancel there)"
    ok3 = _witness_accepts(s3, tb3, rel.hints, rel.public_vectors, p, c, a, b, y)
    assert bool(ok3.all()), f"{rel.name}: honest exact-zero sums rejected at {torch.nonzero(~ok3).flatten().tolist()}"
    _, s1, h1, pv1 = _v1_unit_of(rel)
    assert bool(_witness_accepts(s1, tables(s1, "cpu"), h1, pv1, p, c, a, b, y).all())
    assert torch.equal(reference_hints_v3(p, s3, [(int(c[i]), a[i].tolist(), b[i].tolist(), None) for i in range(len(units))], "cpu"),
                       rel.hints(p, s3, c, a, b, None, "cpu"))


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_off_domain_operands_refused_by_both_decodes(rel):
    p = rel.params
    _, _, _, pv1 = _v1_unit_of(rel)
    rng = np.random.default_rng(3)
    a = rel.operands(rng, (4, rel.k))
    b = rel.operands(rng, (4, rel.k))
    for w in rel.off_domain:
        bad = a.copy()
        bad[1, 2] = w
        with pytest.raises(ValueError):
            rel.public_vectors(p, None, bad, b, None)
        with pytest.raises(ValueError):
            pv1(p, None, bad, b, None)


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_committed_words_are_bound_and_canonical(rel):
    """The witness's operand bits must encode the pinned word (a different word in the bits is rejected), and the encodings
    off the domain -- ``t = 2^exp`` (overflow into the sign bit), ``x`` at the top (BF16: ``t = 255``), ``nz`` inconsistent
    with ``t`` -- are rejected even when the pinned word is the one they collide with."""
    p = rel.params
    pl = plan(p)
    s = rel.compile(p, False)
    tb = tables(s, "cpu")
    c, a, b, y = _cases(rel, 8, seed=77)
    pub = rel.public_vectors(p, c, a, b, y, device="cpu")
    names = [n for _, n in s.hints]
    pos = {n: i for i, n in enumerate(names)}
    hints = rel.hints(p, s, c, a, b, None, "cpu")
    assert _accepts(s, tb, build_witness(s, pub, hints, "cpu"), pub).all()

    def rejected(edit) -> bool:
        h = hints.clone()
        edit(h)
        return not _accepts(s, tb, build_witness(s, pub, h, "cpu"), pub).any()

    # another word in the committed bits (the pin row disagrees)
    assert rejected(lambda h: h.__setitem__((pos["a_words[3].f[0]"], slice(None)), 1 - h[pos["a_words[3].f[0]"]]))
    assert rejected(lambda h: h.__setitem__((pos["b_words[0].s"], slice(None)), 1 - h[pos["b_words[0].s"]]))
    # the (x = 2^exp - 1, nz = 1) encoding of word (s = 1, t = 0, f): flips the sign bit's meaning
    top = 2**pl.op_exp - 1

    def collide(h):
        i = pos["a_words[1].s"]
        h[i] = 0
        h[pos["a_words[1].nz"]] = 1
        for j in range(pl.op_exp):
            h[pos[f"a_words[1].x[{j}]"]] = (top >> j) & 1
    # pin the word (s = 1, t = 0, f) and try the colliding encoding
    a2 = a.copy()
    a2[:, 1] = (1 << (pl.op_word_bits - 1)) | (a2[:, 1] & (2**pl.op_frac - 1))
    y2 = np.array([tc_dot(rel.model, int(c[i]), a2[i].tolist(), b[i].tolist()) for i in range(len(c))], dtype=np.int64)
    pub2 = rel.public_vectors(p, c, a2, b, y2, device="cpu")
    h2 = rel.hints(p, s, c, a2, b, None, "cpu")
    assert _accepts(s, tb, build_witness(s, pub2, h2, "cpu"), pub2).all()
    h2c = h2.clone()
    collide(h2c)
    for j in range(pl.op_frac):
        h2c[pos[f"a_words[1].f[{j}]"]] = torch.as_tensor((a2[:, 1] >> j) & 1)
    assert not _accepts(s, tb, build_witness(s, pub2, h2c, "cpu"), pub2).any()
    # nz = 0 with t != 0 (the hidden bit dropped: x = t) and nz = 1 with t = 0 (x = -1: not a bit vector) on the pinned word
    edited = []

    def drop_nz(h):
        for col in range(h.shape[1]):
            t = (int(a[col, 2]) >> pl.op_frac) & top
            if t:
                edited.append(col)
                h[pos["a_words[2].nz"], col] = 0
                for j in range(pl.op_exp):
                    h[pos[f"a_words[2].x[{j}]"], col] = (t >> j) & 1
    h3 = hints.clone()
    drop_nz(h3)
    ok = _accepts(s, tb, build_witness(s, pub, h3, "cpu"), pub)
    assert edited and not ok[edited].any() and ok[[c_ for c_ in range(len(c)) if c_ not in edited]].all()


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_reference_hints_equal_torch(rel):
    p = rel.params
    s = rel.compile(p, True)
    vus = instances(rel, 1, seed=7)
    units = vu_units(rel, *vus[0][:3])
    c = np.asarray([u[0] for u in units]); a = np.asarray([u[1] for u in units]); b = np.asarray([u[2] for u in units])
    y = np.asarray([u[3] for u in units])
    assert torch.equal(reference_hints_v3(p, s, units, "cpu"), rel.hints(p, s, c, a, b, y, "cpu"))


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_chain_two_vus_cpu(rel, zk, mode):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=zk, mode=mode)
    vus = instances(rel, 2, seed=5)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    a, b, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])


@pytest.mark.parametrize("rel", V3, ids=lambda r: r.name)
def test_negative_battery_rejected(rel):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, instances(rel, 2, seed=5), row_negatives=40)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
    assert len(negs) >= rel.y_bits + 2 + len(rel.off_domain) + 1 + 1 + 4 + 40

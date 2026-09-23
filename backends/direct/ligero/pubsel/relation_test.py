"""The v2 (public-selection) units against the v1 units and the instruction model.

* census: every v2 chain unit has at most half the rows of its v1 counterpart;
* differential (``RELMIN_DIFF_N`` units per relation, default 3000; the lane ran 1e5): on random units inside the domain
  (random and chained accumulators, zeros, subnormals) v1 and v2 accept the honest claim and both reject every wrong claim of
  the negative families -- every bit of the claimed word flipped, +-1 ulp, the other pipeline's word where it differs -- and
  agree column by column; the public decodes refuse the same off-domain operands;
* the chain runner: two VUs proved and verified on CPU (interactive and Fiat-Shamir + ZK), the negative battery rejected;
* the torch hint generator equals the Python integer reference.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from verity.ml.tc.models import ADA_E4M3_M16N8K32, AMPERE_BF16_M16N8K16, HOPPER_BF16_M16N8K16, HOPPER_E4M3_K32
from verity.ml.tc.silicon import tc_dot

from .. import protocol as protocol_mod
from ..compile import P
from ..fp8.census import census
from ..relations import RELATIONS, Relation
from ..relchain import RelationChainRunner, instances, negatives
from ..witness import apply_sparse, build_witness, tables
from .hints import reference_hints_v2
from .relation import plan

V2 = [RELATIONS[n] for n in ("bf16-hopper-v2", "bf16-ampere-v2", "fp8-hopper-v2", "fp8-ada-v2")]
V1_OF = {"bf16-hopper-v2": RELATIONS["bf16-hopper"], "fp8-hopper-v2": RELATIONS["fp8-hopper"], "fp8-ada-v2": RELATIONS["fp8-ada"],
         "bf16-ampere-v2": None}
OTHER = {"bf16-hopper-v2": AMPERE_BF16_M16N8K16, "bf16-ampere-v2": HOPPER_BF16_M16N8K16, "fp8-hopper-v2": ADA_E4M3_M16N8K32,
         "fp8-ada-v2": HOPPER_E4M3_K32}
N_DIFF = int(os.environ.get("RELMIN_DIFF_N", "3000"))
BATCH = 4096


def _v1_unit(rel_v2: Relation):
    """The v1 counterpart's unit-mode system (bf16-ampere-v2 has none registered on the generic runner: the Ampere BF16 unit
    is ``unit.compile_unit`` at REAL)."""
    v1 = V1_OF[rel_v2.name]
    if v1 is not None:
        return v1, v1.compile(v1.params, False), v1.hints, v1.public_vectors
    from ..hints_torch import hints_torch
    from ..unit import compile_unit, public_vectors
    return None, compile_unit(rel_v2.params, chain=False), (lambda p, s, c, a, b, y, d: hints_torch(p, s, c, a, b, None, d)), \
        (lambda p, c, a, b, y, device="cpu": public_vectors(p, c, a, b, y, device=device))


def _finite_fp32(rng: np.random.Generator, n: int) -> np.ndarray:
    kind = rng.integers(0, 16, size=n)
    sign = rng.integers(0, 2, size=n, dtype=np.int64) << 31
    t = rng.integers(40, 220, size=n, dtype=np.int64)
    f = rng.integers(0, 1 << 23, size=n, dtype=np.int64)
    c = sign | (t << 23) | f
    c = np.where(kind == 0, sign, c)
    c = np.where(kind == 1, sign | f, c)
    return c


def _cases(rel: Relation, n: int, seed: int):
    """(c, a, b, y) inside the domain: accumulators random finite FP32, zeros, subnormals, or the model's own state after a
    few chained steps (redrawn when the step leaves the finite range)."""
    rng = np.random.default_rng(seed)
    k = rel.k
    a = rel.operands(rng, (n, k))
    b = rel.operands(rng, (n, k))
    c = _finite_fp32(rng, n)
    chained = rng.integers(0, 3, size=n) == 0
    for i in np.nonzero(chained)[0]:
        cc = 0
        for _ in range(int(rng.integers(1, 4))):
            cc = tc_dot(rel.model, cc, rel.operands(rng, (k,)).tolist(), rel.operands(rng, (k,)).tolist())
        c[i] = cc
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        while True:
            yy = tc_dot(rel.model, int(c[i]), a[i].tolist(), b[i].tolist())
            if ((yy >> 23) & 0xFF) != 0xFF:
                break
            c[i] = _finite_fp32(rng, 1)[0]
        y[i] = yy
    return c, a, b, y


def _accepts(sys, tb, W: torch.Tensor, pub: dict) -> torch.Tensor:
    """Per column: every linear and quadratic constraint holds."""
    lin = apply_sparse((tb.lin_q, tb.lin_j, tb.lin_coef), tb.lin_c, tb.L, W)
    ok = torch.ones(W.shape[1], dtype=torch.bool)
    for q, name in enumerate(tb.lin_pub):
        rhs = pub[name] % P if name is not None else torch.zeros_like(lin[q])
        ok &= lin[q] == rhs
    A = apply_sparse(tb.quad["a"][:3], tb.quad["a"][3], tb.Q, W)
    B = apply_sparse(tb.quad["b"][:3], tb.quad["b"][3], tb.Q, W)
    C = apply_sparse(tb.quad["c"][:3], tb.quad["c"][3], tb.Q, W)
    ok &= ~(((A * B) % P != C).any(dim=0))
    return ok


def _witness_accepts(sys, tb, hints_fn, pub_fn, p, c, a, b, y_claim):
    """Honest witness (hints from the true unit) against the claim ``y_claim``: the acceptance mask."""
    pub = pub_fn(p, c, a, b, y_claim, device="cpu")
    hints = hints_fn(p, sys, c, a, b, None, "cpu")
    W = build_witness(sys, pub, hints, "cpu")
    return _accepts(sys, tb, W, pub)


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_census_at_most_half_of_v1(rel):
    s2 = rel.compile(rel.params, True)
    _, s1u, _, _ = _v1_unit(rel)
    v1 = V1_OF[rel.name]
    if v1 is not None:
        s1 = v1.compile(v1.params, True)
    else:
        from ..unit import compile_unit
        s1 = compile_unit(rel.params, chain=True)
    c1, c2 = census(s1), census(s2)
    assert 2 * c2["rows_per_unit"] <= c1["rows_per_unit"], (c1, c2)
    assert c2["rows_per_unit"] <= {"bf16-hopper-v2": 328, "bf16-ampere-v2": 505, "fp8-hopper-v2": 208, "fp8-ada-v2": 370}[rel.name]


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_differential_v1_v2(rel):
    p = rel.params
    pl = plan(p)
    s2 = rel.compile(p, False)
    tb2 = tables(s2, "cpu")
    _, s1, h1, pv1 = _v1_unit(rel)
    tb1 = tables(s1, "cpu")
    other = OTHER[rel.name]
    ybits = 32
    n_pos = n_neg = 0
    for lo in range(0, N_DIFF, BATCH):
        n = min(BATCH, N_DIFF - lo)
        c, a, b, y = _cases(rel, n, seed=1000 + lo)
        claims = [("honest", y)]
        for bit in range(ybits):
            if pl.fp8 and bit < pl.drop:
                continue                                     # the FP8 claim is the 22-bit packed word: low bits are not claimed
            claims.append((f"bit {bit}", y ^ (1 << bit)))
        claims.append(("+1 ulp", (y + 2**pl.drop) % 2**32))
        claims.append(("-1 ulp", (y - 2**pl.drop) % 2**32))
        yo = np.array([tc_dot(other, int(c[i]), a[i].tolist(), b[i].tolist()) for i in range(n)], dtype=np.int64)
        claims.append(("other pipeline", yo))
        for name, yc in claims:
            fin = ((yc >> 23) & 0xFF) != 0xFF                 # the verifier's decode refuses non-finite claims: not a witness question
            if pl.fp8:
                fin &= (yc & (2**pl.drop - 1)) == 0
            idx = np.nonzero(fin)[0]
            if len(idx) == 0:
                continue
            cc, aa, bb, yy = c[idx], a[idx], b[idx], yc[idx]
            ok2 = _witness_accepts(s2, tb2, rel.hints, rel.public_vectors, p, cc, aa, bb, yy)
            ok1 = _witness_accepts(s1, tb1, h1, pv1, p, cc, aa, bb, yy)
            truth = torch.as_tensor(yy == y[idx])
            assert torch.equal(ok2, truth), f"{rel.name} {name}: v2 accepts != (claim == model) at {torch.nonzero(ok2 != truth)[:5].flatten().tolist()}"
            assert torch.equal(ok1, truth), f"{rel.name} {name}: v1 accepts != (claim == model)"
            n_pos += int(truth.sum())
            n_neg += int((~truth).sum())
    assert n_pos >= N_DIFF and n_neg > 10 * N_DIFF


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_off_domain_operands_refused_by_both_decodes(rel):
    p = rel.params
    _, _, _, pv1 = _v1_unit(rel)
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


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_reference_hints_equal_torch(rel):
    p = rel.params
    s = rel.compile(p, True)
    vus = instances(rel, 1, seed=7)
    from ..relchain import vu_units
    units = vu_units(rel, *vus[0][:3])
    c = np.asarray([u[0] for u in units]); a = np.asarray([u[1] for u in units]); b = np.asarray([u[2] for u in units])
    y = np.asarray([u[3] for u in units])
    assert torch.equal(reference_hints_v2(p, s, units, "cpu"), rel.hints(p, s, c, a, b, y, "cpu"))


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
@pytest.mark.parametrize("zk,mode", [(False, protocol_mod.MODE_INTERACTIVE), (True, protocol_mod.MODE_FIAT_SHAMIR)])
def test_chain_two_vus_cpu(rel, zk, mode):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=zk, mode=mode)
    vus = instances(rel, 2, seed=5)
    proof, pubs, lay = R.prove_vus(vus, check=True)
    ok, why = R.verify_vus(proof, pubs, lay)
    assert ok, why
    a, b, ypub = pubs
    assert int(ypub[rel.steps - 1]) == rel.y_public(vus[0][3]) and int(ypub[2 * rel.steps - 1]) == rel.y_public(vus[1][3])


@pytest.mark.parametrize("rel", V2, ids=lambda r: r.name)
def test_negative_battery_rejected(rel):
    R = RelationChainRunner(rel, "cpu", -128.0, zk=False, mode=protocol_mod.MODE_FIAT_SHAMIR)
    negs = negatives(R, instances(rel, 2, seed=5), row_negatives=40)
    accepted = [name for name, (ok, _) in negs if ok]
    assert accepted == [], f"{rel.name}: negatives accepted: {accepted}"
    assert len(negs) >= rel.y_bits + 2 + len(rel.off_domain) + 1 + 1 + 4 + 40

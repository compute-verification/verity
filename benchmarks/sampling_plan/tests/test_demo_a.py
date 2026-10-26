"""Demo A's frontier (`demo_a.py`): every unit within C-Flock's k_log 27 and its work its own ANDs, every unit its output
bits, statements shared, the certificate `certify`'s where core's exact count is cheap and its location term never below
it, at each budget the plan under top's rate no worse than either half's best alone, within the budget, and the least
sampling block the first whose share of its GPU time affords a plan."""

from __future__ import annotations

from math import floor

import pytest

import demo_a as D
import h200 as H
from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import location_bits
from verity.core.protocols.sampled_proofs.optimize import Objective, certify

SMALL = H.Shape(layers=1, width=1024, heads=16, mlp=4096, context=16, prompt=4, batch=4, step_s=0.01, steps=16)


def test_every_unit_fits_the_cap_and_its_work_is_its_own_ands():
    ts = H.templates(D.SHAPE)
    fitted = D.fit(ts)
    assert len(fitted) == len(ts) and max(map(D.k_log, fitted)) <= D.K_LOG_MAX
    assert not any(t.id.startswith("stub:RmsNormRow") for t in fitted)
    for t, f in zip(ts, fitted):
        p = int(f.id.split(":")[1]) if f.id.startswith("split:") else 1
        assert f.units == p * t.units and (p == 1 or D.k_log(D._part(f, 1)) <= D.K_LOG_MAX < D.k_log(D._part(t, p // 2)))
    one, week = D.candidates(), D.candidates(blocks=7)
    for c, w in zip(one, week):
        assert c.work == c.ands and c.influence == 0 and c.commit_bits == c.out_bits > 0 and w.units == 7 * c.units
    cm = D.costs()
    assert cm.shared and cm.commit_per_bit == pytest.approx(1 / (8 * 225e9))
    assert D.costs(blocks=7).replay["request"].units == 7 * D.SHAPE.batch
    assert D.SHAPE.block_s == pytest.approx(32 * 7.84)
    g = D.ands_against_gpu()
    assert sum(d["ands_share"] for d in g["definitions"].values()) == pytest.approx(1)
    assert g["s8"]["gpu_share"] == pytest.approx(5.08 / 5.33)


def test_location_is_location_bits_at_the_exact_count():
    law = Stratified((Stratum("a", 400, 30), Stratum("b", 90, 9)))
    profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, 0.01)
    assert D.location(law.population, law.count_bound(0.01)) == location_bits(profile)
    assert floor(law.harm_bound({"a": 1.0, "b": 1.0}, 0.01)) >= law.count_bound(0.01)


def test_the_certificate_is_certifys_on_a_small_shape():
    cands = D.candidates(SMALL)
    sizes = {c.template: min(c.units, 40) for c in cands}
    objective = Objective(0.01, work=1e-9, bits=1e-3)
    ours, core = D.certificate(sizes, cands, objective), certify(sizes, cands, objective)
    assert ours.free_compute == core.free_compute and ours.harm == core.harm
    assert ours.leak >= core.leak and ours.leak == pytest.approx(core.leak, rel=1e-6)


def test_each_budget_certifies_no_less_than_either_half_alone():
    cands, cm = D.candidates(SMALL), D.costs(SMALL)
    s = D.sweep(cands, cm, SMALL, budgets=(1e-3, 3e4))
    assert s["skipped"] == [1e-3 * SMALL.block_s]
    (p,) = s["points"]
    assert p.plan.cost["total"] <= p.budget + s["commit"] * (1 + 1e-9)
    at = p.at["0.01"]
    assert at["free_compute"] >= p.h_work * (1 - 1e-9) and at["leak"] >= p.h_bits * (1 - 1e-9)
    assert p.at["1e-06"]["free_compute"] >= at["free_compute"] and p.at["1e-06"]["leak"] >= at["leak"]


def test_the_least_sampling_block_is_the_first_that_affords_a_plan():
    n = D.least_blocks()
    assert n > 1
    assert D.cheapest(D.candidates(blocks=n), D.costs(blocks=n)) <= D.BUDGET_SHARE * n * D.SHAPE.block_s
    assert D.cheapest(D.candidates(blocks=n - 1), D.costs(blocks=n - 1)) > D.BUDGET_SHARE * (n - 1) * D.SHAPE.block_s
    one = D.plan_at(D.SHAPE, "one block", 1)
    assert not one["affordable"] and "at" not in one
    assert D.spans(periods={"week": 7 * 86400})["week"] == floor(7 * 86400 / D.SHAPE.block_s)


def test_the_knee_is_where_the_certified_shares_stop_falling_fast():
    def at(share):
        return {"0.01": {"free_compute": share, "leak": share}}

    pts = [D.Point(b, 1.0, 1.0, None, at(s)) for b, s in ((1, 1.0), (2, 1e-3), (4, 5e-4), (8, 2.5e-4))]
    assert D.knee(pts, 1.0, 1.0) == 1

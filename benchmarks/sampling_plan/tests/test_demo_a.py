"""Demo A's frontier (`demo_a.py`): only PoUW's units carry credited work, every unit its output bits, statements are
shared, the certificate is `certify`'s where core's exact count is cheap and its location term never below it, and at
each budget the plan under top's rate certifies no less than either half's best alone, within the budget."""

from __future__ import annotations

from math import floor

import pytest

import demo_a as D
import h200 as H
from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import location_bits
from verity.core.protocols.sampled_proofs.optimize import Objective, certify

SMALL = H.Shape(layers=1, width=1024, heads=16, mlp=4096, context=16, prompt=4, batch=4, step_s=0.01, steps=16)


def test_only_pouws_coordinates_carry_work():
    for c in D.candidates():
        assert c.work == (int(c.template.split("K=")[1].split(",")[0]) if c.template.startswith(D.S8) else 0)
        assert c.influence == 0 and c.commit_bits == c.out_bits > 0
        assert c.per_replay * D.SHAPE.batch == c.units
    cm = D.costs()
    assert cm.shared and cm.commit_per_bit == pytest.approx(1 / (8 * 225e9))
    assert D.SHAPE.block_s == pytest.approx(32 * 7.84)


def test_location_is_location_bits_at_the_exact_count():
    law = Stratified((Stratum("a", 400, 30), Stratum("b", 90, 9)))
    profile = IntegrityProfile(("unit",), (law.population,), {"unit": law}, 0.01)
    assert D.location(law.population, law.count_bound(0.01)) == location_bits(profile)
    assert floor(law.harm_bound({"a": 1.0, "b": 1.0}, 0.01)) >= law.count_bound(0.01)


def test_the_certificate_is_certifys_on_a_small_shape():
    cands = D.candidates(SMALL)
    sizes = {c.template: min(c.units, 40) for c in cands}
    objective = Objective(0.01, work=1e-4, bits=1e-3)
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

"""Every named vector re-solves from data/ to the numbers it was published with, and passes its check."""
from __future__ import annotations

import dataclasses
import functools

import pytest

from verity_machine_model import attacks as atk
from verity_machine_model.lp import price
from verity_machine_model.model import Attack, measured_cost, vectors
from verity_machine_model.resolve import resolve

NAMES = vectors()


@functools.cache
def solved(name):
    return resolve(name)


def test_every_named_vector_is_here():
    assert NAMES[0] == "VF" and NAMES[-1] == "VF3j-16"
    assert {"VF2", "VF3", "VF3c", "VF3d", "VF3e", "VF3f", "VF3g", "VF3h", "VF3i", "VF3i-strict", "VF3j", "VF3j-8"} < set(NAMES)


# τ, τ at 600 W, U/F_low and margin as the round notes published them
PUBLISHED = {
    "VF3h": (0.1246, 0.0975, 1.0938, 1.935),          # note:20261016T1538Z-draft-rs-vf3i-textures-uniform
    "VF3i-strict": (0.0698, 0.0546, 1.0938, 1.935),   # note:20261016T1640Z-draft-rs-vf3j-strict-closure, on valid rows
    "VF3j": (0.0634, 0.0495, 1.0938, 1.935),
    "VF3j-16": (0.0610, 0.0477, 1.1563, 2.000),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_reproduces_the_round_notes(name):
    r = solved(name)
    assert (round(r.tau, 4), round(r.tau_600w, 4), round(r.uf_low, 4), round(r.margin, 3)) == PUBLISHED[name]


@pytest.mark.parametrize("name", NAMES)
def test_reproduces_its_published_numbers(name):
    r = solved(name)
    assert {"tau", "uf_low", "margin"} <= set(r.vector.published), f"{name}: record what `solve` prints under [published]"
    assert r.unreproduced() == []


@pytest.mark.parametrize("name", [n for n in NAMES if solved(n).vector.legacy])
def test_matches_the_legacy_scripts(name):
    """The legacy scripts' values in full: a re-solve on the same data and solver lands on the same vertex."""
    r, legacy = solved(name), solved(name).vector.legacy
    assert r.tau == pytest.approx(legacy["tau"], rel=1e-12)
    assert r.solution.infl == pytest.approx(legacy["infl"], rel=1e-12)
    assert r.margin == pytest.approx(legacy["margin"], rel=1e-12)
    assert len(r.solution.attacks) == legacy["attacks"]


@pytest.mark.parametrize("name", NAMES)
def test_every_price_at_most_measured(name):
    r = solved(name)
    mc = measured_cost(r.census)
    over = {k: price(r.census, r.solution.lam, k) / r.solution.sigma / mc[k] for k in r.census}
    assert all(x <= 1 + 1e-9 for x in over.values()), {k: x for k, x in over.items() if x > 1 + 1e-9}


@pytest.mark.parametrize("name", NAMES)
def test_every_attack_costs_at_least_its_macs(name):
    r = solved(name)
    xs = {(a.design, a.family, a.name): atk.x_of(r.census, r.solution, a.program) for a in r.attacks}
    assert len(xs) == len(r.attacks) and not r.solution.refused
    assert min(xs.values()) >= 1 - 1e-9, min(xs.items(), key=lambda t: t[1])


@pytest.mark.parametrize("name", NAMES)
def test_passes_its_check(name):
    assert solved(name).failures == []


def test_check_names_what_fails():
    r = solved("VF3j")
    cheap = Attack("U±64, F±1", "(x) test", "one ALU a MAC", {"ALU": 1 / 4096})
    out = atk.check(r.model, r.census, r.solution, r.vector.credits, r.attacks + [cheap], r.model.x_min, r.model.rho)
    assert len(out) == 1 and out[0].startswith("attack U±64, F±1 / (x) test / one ALU a MAC costs X = ")
    inflated = dataclasses.replace(r.solution, sigma=r.solution.sigma / 1000)   # every price 1000 times its convex combination
    out = atk.check(r.model, r.census, inflated, (), [], r.model.x_min, r.model.rho)
    assert any(f.startswith("FFMA's price ") and "exceeds its measured cost" in f for f in out)
    out = atk.check(r.model, r.census, r.solution, [("ALU", 10 ** 6)], [], r.model.x_min, r.model.rho)
    assert out == [out[0]] and out[0].startswith("credit ALU ≥ 1000000 MACs unpaid")


def test_margin_is_the_adopted_designs_cheapest():
    r = solved("VF3j-16")
    xs = [atk.x_of(r.census, r.solution, a.program) for a in r.attacks if a.design == r.model.adopted]
    assert r.margin == min(xs) and r.cheapest[0][0] == r.margin
    assert r.three_piece == min(atk.x_of(r.census, r.solution, a.program) for a in r.attacks
                                if a.design == r.model.adopted and a.family.startswith("(iv)"))

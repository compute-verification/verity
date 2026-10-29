"""Every named vector re-solves from data/ to the numbers it was published with, and passes its check."""
from __future__ import annotations

import dataclasses
import functools
import shutil

import pytest

from verity_machine_model import attacks as atk
from verity_machine_model.lp import price
from verity_machine_model.model import DATA, Attack, census, measured_cost, model, vectors
from verity_machine_model.resolve import resolve

NAMES = vectors()


@functools.cache
def solved(name):
    return resolve(name)


def test_every_named_vector_is_here():
    assert NAMES[0] == "VF" and model().current in NAMES and set(PUBLISHED) < set(NAMES)
    assert {"VF2", "VF3", "VF3c", "VF3d", "VF3e", "VF3f", "VF3g", "VF3h", "VF3i", "VF3i-strict", "VF3j", "VF3j-8",
            "VF3j-16", "VF3k-16"} < set(NAMES)


# τ, U/F_low and margin as the round notes published them, through VF3s-16. A later vector is held to its
# own file's [published] (test_reproduces_its_published_numbers), so a new vector touches no test.
PUBLISHED = {
    "VF3h": (0.1246, 1.0938, 1.935),
    "VF3i-strict": (0.0698, 1.0938, 1.935),   # on valid rows
    "VF3j": (0.0634, 1.0938, 1.935),
    "VF3j-16": (0.0610, 1.1563, 2.000),
    "VF3k-16": (0.0603, 1.1563, 2.000),
    "VF3l-16": (0.0603, 1.1563, 2.000),       # with its copy prices
    "VF3l-16-copy16": (0.0525, 1.1563, 2.000),
    "VF3l-16-copy16-fan32": (0.0460, 1.1563, 2.000),
    "VF3l-16-copy32-fan16": (0.0600, 1.3125, 2.000),
    "VF3l-16-copy32": (0.0518, 1.3125, 2.000),
    "VF3m-16": (0.0591, 1.1563, 2.000),
    "VF3n-16": (0.0586, 1.1563, 2.000),
    "VF3o-16": (0.0588, 1.1563, 2.000),
    "VF3p-16": (0.0588, 1.1563, 2.000),
    "VF3q-16": (0.0564, 1.1563, 2.000),
    "VF3r-16": (0.0556, 1.1563, 2.000),
    "VF3s-16": (0.0556, 1.1563, 2.000),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_reproduces_the_round_notes(name):
    r = solved(name)
    assert (round(r.tau, 4), round(r.uf_low, 4), round(r.margin, 3)) == PUBLISHED[name]


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
    r = solved(model().current)
    xs = [atk.x_of(r.census, r.solution, a.program) for a in r.attacks if a.design == r.model.adopted]
    assert r.margin == min(xs) and r.cheapest[0][0] == r.margin
    assert r.three_piece == min(atk.x_of(r.census, r.solution, a.program) for a in r.attacks
                                if a.design == r.model.adopted and a.family.startswith("(iv)"))


# What each claim of data/floors.csv finds below its floor, by vector: (π₃₂) and its load half
CLAIMED = {
    "VF3j": ([], ["LDS128 8.00 a word < 16", "LDG128 8.00 a word < 16", "LDC64 0.00 a word < 16"]),
    "VF3j-16": ([], []),
    "VF3k-16": ([], []),
    "VF3l-16": ([], []),
    "VF3l-16-copy16": ([], []),
    "VF3l-16-copy32": ([], []),
    "VF3m-16": ([], []),
    "VF3n-16": ([], []),
    "VF3o-16": ([], []),
    "VF3p-16": ([], []),
}


@pytest.mark.parametrize("name", sorted(CLAIMED))
def test_claims(name):
    r = solved(name)
    assert (r.floors["pi32"][1], r.floors["loads16"][1]) == CLAIMED[name]
    assert r.floors["loads16"][0] == 9 and r.floors["pi32"][0] == 84 + ("ATOME32P" in r.census)


# π_c ≥ 16 by vector: what is under it, and its cheapest copy route a word. VF3k-16 and VF3l-16 price the UR fan-out
# under it; each copy candidate raises the routes it names; VF3p-16's UNI on the issue rows lifts UNI's route over it
UR_FANOUT = ["UNI 2.00 a word < 16", "UNI64 1.00 a word < 16"]
COPIES = {"VF3k-16": (UR_FANOUT, 1), "VF3l-16": (UR_FANOUT, 1), "VF3l-16-copy16": ([], 16),
          "VF3l-16-copy16-fan32": ([], 16), "VF3l-16-copy32-fan16": ([], 16), "VF3l-16-copy32": ([], 32),
          "VF3m-16": (UR_FANOUT, 1), "VF3n-16": (UR_FANOUT, 1), "VF3o-16": (UR_FANOUT, 1),
          "VF3p-16": (UR_FANOUT[1:], 1)}


@pytest.mark.parametrize("name", sorted(COPIES))
def test_copy_routes(name):
    r = solved(name)
    below, cheapest = COPIES[name]
    assert r.floors["copy16"][1] == below and r.floors["copy16"][0] == len(r.copies) == 63 + 14 * ("IMMA16832" in r.census)
    assert r.copies[0][2] == pytest.approx(cheapest)
    assert all(p <= mc * (1 + 1e-9) for _k, _w, p, mc, _ in r.copies)  # no copy route priced over its measured cost


def test_vf3j16_rested_on_slow_rows():
    """VF3j-16's θ on VF3k-16's census, the gated probes' faster rows: its (π₃₂) for ATOMS64 doesn't hold there."""
    j = solved("VF3j-16")
    out = atk.check(j.model, census(j.model, "VF3k-16"), j.solution, j.vector.credits, [], j.model.x_min, j.model.rho)
    assert [f.split(":")[0] for f in out] == ["credit ATOMS64 ≥ 2048 MACs unpaid", "credit LDSM_X4T ≥ 4096 MACs unpaid"]


def test_vf3k16_loses_tau_on_atoms64_alone(tmp_path):
    """Every adopted line but ATOMS64's, and the peer class, leave VF3j-16's τ as it was, to 1e-6 (LDSM_X4T's faster
    line costs 7.5e-7 of it): the 1.1% VF3k-16 loses is the price of ATOMS64's credit on its faster row."""
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    p = data / "census" / "VF3k-16.csv"
    c = p.read_text()
    line = "ATOMS64,ROW_ATOMS64,0.480156,0.02,,,VF3k-16,VF3k-16,r20261016-183617-3914\n"
    assert c.count(line) == 1
    p.write_text(c.replace(line, line.replace("0.480156", "0.29523")))
    assert resolve("VF3k-16", data).tau == pytest.approx(solved("VF3j-16").tau, rel=1e-6)
    assert solved("VF3k-16").tau < solved("VF3j-16").tau * (1 - 0.01)


def test_vf3m16_loses_tau_on_the_new_word_writers(tmp_path):
    """Without the credits of the three classes VF3m-16 adds (cp.async, bulk copy, DSMEM stores, each at 32 MACs a word),
    its refits and rows leave VF3l-16's τ as it was: the 2.1% it loses is the price of those credits."""
    data = tmp_path / "data"
    shutil.copytree(DATA, data)
    p = data / "vectors" / "VF3m-16.toml"
    credits = '    ["LDGSTS16", 4096], ["UBLKCP4K", 32768], ["DSMST4P", 4096],\n'
    assert p.read_text().count(credits) == 1
    p.write_text(p.read_text().replace(credits, ""))
    assert resolve("VF3m-16", data).tau == pytest.approx(solved("VF3l-16").tau, rel=1e-9)
    assert solved("VF3m-16").tau < solved("VF3l-16").tau * (1 - 0.02)

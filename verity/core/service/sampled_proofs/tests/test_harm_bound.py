"""`certify`'s harm bounds held to Lean's closed form (`SampledProofs.Bound.densityBound`, the `harm_bound` piece of
`sampled-proofs-difftest`): the greedy fill that `stratified_isHarmBound_fill` proves a harm bound, at the budget
`lnUpper δ >= ln(1/δ)`, in rationals.

`optimize.fill` at `optimize.ln_upper`, on rationals, is the reference, and must agree with the Lean exactly on drawn rows
and δ, invalid ones included. `certify` reports floats, so it is held to the Lean's value on Demo A's plan of record at a
one-week sampling block (`plan_of_record.json`, the draw PRF-166 posts, with what core certified for it) and on random
draws: its `density` (`density_bound`) at least the Lean's value and within `SLACK` of it, and its exact fields (core's
`Stratified.harm_bound`, whose exact risk is at least the closed form's) at most the Lean's value within `SLACK`.
"""

from __future__ import annotations

import json
import random
import shutil
from fractions import Fraction
from math import floor, inf, nextafter
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.core.protocols.profile import IntegrityProfile, Stratified, Stratum
from verity.core.protocols.sampled_proofs.consumers import location_bits
from verity.core.protocols.sampled_proofs.optimize import (Candidate, Objective, certify, density_bound, fill,
                                                           ln_upper)

from test_difftest import DELTA, EXACT_DELTA, JSON, PACKAGE, PIECES, _exactly, _json, _nat, ratio

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PLAN = json.loads((Path(__file__).with_name("plan_of_record.json")).read_text())
#: how far above the Lean's rational `certify`'s floats may sit: `HARM_MARGIN` on the budget and again on the value
SLACK = Fraction(1, 2 ** 18)


def harm_bound(j):
    j = _json(j)
    _exactly(j, {"delta", "rows"}, "the input")
    if not isinstance(j["rows"], list):
        raise D.Reject("rows is not a list")
    rows = []
    for r in j["rows"]:
        _exactly(r, {"harm", "k", "size"}, "a row")
        if not (_nat(r["size"]) and _nat(r["k"]) and r["k"] <= r["size"]):
            raise D.Reject("a row's size and k are natural numbers, k <= size")
        rows.append((r["size"], r["k"], ratio(r["harm"])))
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    budget = ln_upper(delta)
    value = Fraction(fill(rows, budget))
    return {"budget": str(budget), "floor": floor(value), "value": str(value)}


def _row(n: int, mode: int, k: int, harm: str) -> dict:
    return {"harm": harm, "k": min(n, (0, 1, 3, n // 2, n, k)[mode]), "size": n}


HARM = st.one_of(st.floats(0, 1e30).map(lambda f: str(Fraction(f))),
                 st.builds(lambda n, d: str(Fraction(n, d)), st.integers(0, 10 ** 6), st.integers(1, 1000)),
                 st.sampled_from(["0", "1", "1/3"]))
SIZE = st.one_of(st.integers(0, 20), st.integers(0, 10 ** 6), st.integers(0, 10 ** 18))
ROWS = st.lists(st.builds(_row, SIZE, st.integers(0, 5), st.integers(0, 2000), HARM), max_size=8)
BAD_ROW = st.sampled_from([{"harm": "1", "k": 3, "size": 2}, {"harm": "1", "k": True, "size": 2},
                           {"harm": 1, "k": 0, "size": 2}, {"harm": "-1", "k": 0, "size": 2},
                           {"harm": "2/4", "k": 0, "size": 2}, {"k": 0, "size": 2},
                           {"harm": "1", "k": 0, "size": 2, "x": 1}, {"harm": "1", "k": -1, "size": 2}, ["1", 0, 2]])
HARM_BOUND = D.Port(PIECES, "harm_bound", harm_bound, st.one_of(
    st.builds(lambda rows, d: {"delta": d, "rows": rows}, ROWS, st.one_of(DELTA, EXACT_DELTA)),
    st.builds(lambda rows, d: {"delta": d, "rows": rows}, ROWS, st.one_of(DELTA, EXACT_DELTA)),
    st.builds(lambda rows, d: {"delta": d, "rows": rows}, ROWS,
              st.sampled_from(["0", "1", "3/2", "2/4", "a", "1/0", 1, None])),
    st.builds(lambda rows, d, i, bad: {"delta": d, "rows": [*rows[:i], bad, *rows[i:]]}, ROWS, DELTA,
              st.integers(0, 8), BAD_ROW),
    st.sampled_from([{"delta": "1/2", "rows": {}}, {"delta": "1/2", "rows": [], "x": 1}, {"rows": []}]),
    JSON), package=PACKAGE)

server = D.server_fixture(HARM_BOUND)


def lean(server, rows, delta) -> Fraction:
    """The Lean's harm bound of rows `(n, k, h)` at delta, floats read as the rationals they are."""
    kind, out = server.call("harm_bound", {"delta": str(Fraction(delta)), "rows": [
        {"harm": str(Fraction(h)), "k": k, "size": n} for n, k, h in rows]})
    assert kind == "output", out
    return Fraction(json.loads(out)["value"])


def held(density: float, exact: Fraction) -> bool:
    """`density_bound`'s float is at least the Lean's value, and within `SLACK` of it."""
    return exact <= Fraction(density) <= exact * (1 + SLACK)


def under(core: float, exact: Fraction) -> bool:
    """Core's exact-risk bound is at most the closed form's, within `SLACK` (and its least float above 0 at no harm)."""
    return core <= exact * (1 + SLACK) or (exact == 0 and core <= nextafter(0.0, inf))


def test_the_harm_bound_agrees(server):
    stats = D.run(HARM_BOUND, examples=500, server=server)
    assert stats.outputs > 125 and stats.rejects > 0, stats


@pytest.mark.parametrize("delta", sorted(PLAN["certified"], key=float))
def test_certify_is_held_to_lean_on_the_plan_of_record(server, delta):
    """The objective's harm, the credited work and the output bits of the plan of record. The leak's location term
    (`location_bits`, core's exact `count_bound` over 1.6e16 units) is no harm bound and is left out."""
    d, (w, b) = float(delta), (PLAN["weights"]["work"], PLAN["weights"]["bits"])
    law = Stratified(tuple(Stratum(t, n, k) for t, n, k, _, _ in PLAN["rows"]))
    for name, harm in (("objective", lambda a, o: w * a + b * o), ("work", lambda a, o: float(a)),
                       ("bits", lambda a, o: float(o))):
        exact = lean(server, [(n, k, harm(a, o)) for _, n, k, a, o in PLAN["rows"]], d)
        assert held(density_bound([(n, k, harm(a, o)) for _, n, k, a, o in PLAN["rows"]], d), exact), name
        assert under(law.harm_bound({t: harm(a, o) for t, _, _, a, o in PLAN["rows"]}, d), exact), name
    assert law.harm_bound({t: float(a) for t, _, _, a, _ in PLAN["rows"]}, d) == PLAN["certified"][delta]["free_compute"]


@pytest.mark.parametrize("seed", range(16))
def test_certify_is_held_to_lean_on_random_draws(server, seed):
    r = random.Random(seed)
    cands = [Candidate(f"t{i}", f"t{i}", r.choice([1, 2, 10, 1000, 10 ** 6, 10 ** 12]) * r.randint(1, 9),
                       r.randint(0, 10 ** 7), r.randint(0, 16)) for i in range(r.randint(1, 6))]
    sizes = {c.template: min(c.units, r.choice([1, 2, 5, 40, 300, c.units])) for c in cands}
    objective = Objective(r.choice([0.5, 0.01, 1e-6]), work=r.random(), bits=r.choice([0.0, r.random() * 1e4]))
    cert, inexact = certify(sizes, cands, objective), certify(sizes, cands, objective, exact=False)

    def bound(h) -> Fraction:
        return lean(server, [(c.units, sizes[c.template], h(c)) for c in cands], objective.delta)

    exact, work = bound(objective.harm), bound(lambda c: float(c.work))
    assert held(cert.density, exact) and inexact.harm == inexact.density == cert.density
    assert held(inexact.free_compute, work)
    assert under(cert.harm, exact) and under(cert.free_compute, work)
    if cert.leak is not None:
        law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in cands))
        loc = location_bits(IntegrityProfile(("unit",), (law.population,), {"unit": law}, objective.delta))
        bits = bound(lambda c: float(c.out_bits))
        assert Fraction(cert.leak) <= (bits * (1 + SLACK) + Fraction(loc)) * (1 + Fraction(1, 2 ** 52)) \
            or (bits == 0 and cert.leak <= nextafter(loc, inf))

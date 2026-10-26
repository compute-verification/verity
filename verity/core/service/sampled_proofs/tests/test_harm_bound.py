"""`certify`'s harm bounds held to `Flock.Guarantees.JointStratified`'s exact supremum, `StratifiedBlock.maxHarm`: the
most ANDs, or output bits, a wrong set whose exact stratified escape is at least δ holds, at the same δ and the same
weights.  The supremum is bracketed in Lean (`SampledProofs.Bound.maxHarmCheck`, the `max_harm` piece of
`sampled-proofs-difftest`) from a certificate `optimize.max_harm` finds: `lower` the harm of one set of the confidence
set, `upper` a dual bound.  Also the closed form (`Bound.densityBound`, the `harm_bound` piece) that
`stratified_isHarmBound_fill` proves a harm bound.

`optimize.max_harm_check` and `optimize.fill` at `optimize.ln_upper`, on rationals, are the references, and must agree
with the Lean exactly, invalid inputs included.  `certify` reports floats, so it is held on Demo A's plan of record at a
one-week sampling block (`plan_of_record.json`, the draw PRF-166 posts, with what core certified for it) and on random
draws: each exact field (core's `Stratified.harm_bound`) at least the supremum's `lower` and at most its `upper` within
`SLACK`, the bracket itself within `WIDTH` units' harm, and `density` (`density_bound`) at least the closed form's value
within `SLACK`.
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
                                                           ln_upper, max_harm, max_harm_check)

from test_difftest import DELTA, EXACT_DELTA, JSON, PACKAGE, PIECES, _exactly, _json, _nat, ratio

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

PLAN = json.loads((Path(__file__).with_name("plan_of_record.json")).read_text())
#: how far above the Lean's rational `certify`'s floats may sit: `HARM_MARGIN` on the budget and again on the value
SLACK = Fraction(1, 2 ** 18)
#: how many units' harm (the largest per-unit harm) the supremum's bracket may be wide, beyond 2^-40 of it
WIDTH = 16


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


def max_harm_ref(j):
    j = _json(j)
    _exactly(j, {"delta", "multiplier", "rows"}, "the input")
    if not isinstance(j["rows"], list):
        raise D.Reject("rows is not a list")
    rows = []
    for r in j["rows"]:
        _exactly(r, {"harm", "hi", "k", "lo", "size", "take"}, "a row")
        if not (all(_nat(r[f]) for f in ("size", "k", "take", "lo", "hi")) and r["k"] <= r["size"]):
            raise D.Reject("a row's size, k, take, lo and hi are natural numbers, k <= size")
        rows.append((r["size"], r["k"], ratio(r["harm"]), r["take"], r["lo"], r["hi"]))
    delta = ratio(j["delta"])
    if not 0 < delta < 1:
        raise D.Reject("delta is not in (0, 1)")
    lower, upper = max_harm_check(rows, delta, ratio(j["multiplier"]))
    return {"lower": str(lower), "upper": str(upper)}


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


@st.composite
def _cert_row(draw) -> dict:
    """A row with a certificate that may fail: `lo = 0`, `hi = size - k` and `take = 0` always hold."""
    n = draw(st.one_of(st.integers(0, 20), st.integers(0, 10 ** 6), st.integers(0, 10 ** 18)))
    k = min(n, draw(st.sampled_from([0, 1, 3, min(n // 2, 1000), n, draw(st.integers(0, 300))])))
    top = n - k
    lo = draw(st.one_of(st.just(0), st.integers(0, top)))
    return {"harm": draw(HARM), "hi": draw(st.one_of(st.just(top), st.integers(lo, top + 2))), "k": k, "lo": lo,
            "size": n, "take": draw(st.one_of(st.just(0), st.integers(0, min(top, 3)), st.integers(0, top + 1)))}


def _found(rows: list[dict], delta: str) -> dict:
    """The certificate `max_harm` finds for the rows at delta."""
    _, _, lam, cert = max_harm([(r["size"], r["k"], ratio(r["harm"])) for r in rows], ratio(delta))
    return {"delta": delta, "multiplier": str(lam),
            "rows": [{**r, "hi": hi, "lo": lo, "take": take} for r, (take, lo, hi) in zip(rows, cert)]}


MULTIPLIER = st.one_of(st.floats(1e-30, 1e30).map(lambda f: str(Fraction(f))),
                       st.builds(lambda n, d: str(Fraction(n, d)), st.integers(1, 10 ** 6), st.integers(1, 1000)),
                       st.sampled_from(["0", "1", "1/3", "a", 1]))
CERT_ROWS = st.lists(_cert_row(), max_size=6)
SMALL_ROWS = st.lists(st.builds(_row, st.integers(0, 2000), st.integers(0, 5), st.integers(0, 300), HARM), max_size=5)
MAX_HARM = D.Port(PIECES, "max_harm", max_harm_ref, st.one_of(
    st.builds(lambda rows, d, m: {"delta": d, "multiplier": m, "rows": rows}, CERT_ROWS, st.one_of(DELTA, EXACT_DELTA),
              MULTIPLIER),
    st.builds(_found, SMALL_ROWS, st.one_of(DELTA, EXACT_DELTA)),
    st.builds(lambda rows, d: {"delta": d, "multiplier": "1", "rows": rows}, CERT_ROWS,
              st.sampled_from(["0", "1", "3/2", "2/4", "a", None])),
    st.sampled_from([{"delta": "1/2", "multiplier": "1", "rows": {}}, {"delta": "1/2", "rows": []},
                     {"delta": "1/2", "multiplier": "1", "rows": [{"harm": "1", "hi": 1, "k": 0, "lo": 0, "size": 1}]},
                     {"delta": "1/2", "multiplier": "1", "rows": [], "x": 1}]),
    JSON), package=PACKAGE, rejects=(D.Reject, ValueError))

server = D.server_fixture(HARM_BOUND)


def lean(server, rows, delta) -> Fraction:
    """The Lean's closed-form harm bound of rows `(n, k, h)` at delta, floats read as the rationals they are."""
    kind, out = server.call("harm_bound", {"delta": str(Fraction(delta)), "rows": [
        {"harm": str(Fraction(h)), "k": k, "size": n} for n, k, h in rows]})
    assert kind == "output", out
    return Fraction(json.loads(out)["value"])


def supremum(server, rows, delta) -> tuple[Fraction, Fraction]:
    """The Lean's bracket around `maxHarm` for rows `(n, k, h)` at delta, from the certificate `max_harm` finds, which
    is within `WIDTH` units' harm."""
    lower, upper, lam, cert = max_harm(rows, delta)
    kind, out = server.call("max_harm", {"delta": str(Fraction(delta)), "multiplier": str(lam), "rows": [
        {"harm": str(Fraction(h)), "hi": hi, "k": k, "lo": lo, "size": n, "take": take}
        for (n, k, h), (take, lo, hi) in zip(rows, cert)]})
    assert kind == "output", out
    assert json.loads(out) == {"lower": str(lower), "upper": str(upper)}
    assert upper - lower <= WIDTH * max((Fraction(h) for _, _, h in rows), default=0) + upper / 2 ** 40
    return lower, upper


def held(density: float, exact: Fraction) -> bool:
    """`density_bound`'s float is at least the Lean's closed form, and within `SLACK` of it."""
    return exact <= Fraction(density) <= exact * (1 + SLACK)


def at_supremum(core: float, bracket: tuple[Fraction, Fraction]) -> bool:
    """Core's bound is at least the supremum's `lower` and at most its `upper` within `SLACK` (or its least float above
    0 at no harm)."""
    lower, upper = bracket
    return lower <= Fraction(core) and (Fraction(core) <= upper * (1 + SLACK) or core <= nextafter(0.0, inf))


def test_the_harm_bound_agrees(server):
    stats = D.run(HARM_BOUND, examples=500, server=server)
    assert stats.outputs > 125 and stats.rejects > 0, stats


def test_the_max_harm_check_agrees(server):
    stats = D.run(MAX_HARM, examples=400, server=server)
    assert stats.outputs > 100 and stats.rejects > 0, stats


@pytest.mark.parametrize("delta", sorted(PLAN["certified"], key=float))
def test_certify_is_held_to_the_supremum_on_the_plan_of_record(server, delta):
    """The objective's harm, the credited work and the output bits of the plan of record. The leak's location term
    (`location_bits`, core's exact `count_bound` over 1.6e16 units) is no harm bound and is left out."""
    d, (w, b) = float(delta), (PLAN["weights"]["work"], PLAN["weights"]["bits"])
    law = Stratified(tuple(Stratum(t, n, k) for t, n, k, _, _ in PLAN["rows"]))
    for name, harm in (("objective", lambda a, o: w * a + b * o), ("ands", lambda a, o: a), ("bits", lambda a, o: o)):
        rows = [(n, k, harm(a, o)) for _, n, k, a, o in PLAN["rows"]]
        assert at_supremum(law.harm_bound({t: float(harm(a, o)) for t, _, _, a, o in PLAN["rows"]}, d),
                           supremum(server, rows, d)), name
        assert held(density_bound([(n, k, float(h)) for n, k, h in rows], d), lean(server, rows, d)), name
    assert law.harm_bound({t: float(a) for t, _, _, a, _ in PLAN["rows"]}, d) == PLAN["certified"][delta]["free_compute"]


@pytest.mark.parametrize("seed", range(16))
def test_certify_is_held_to_the_supremum_on_random_draws(server, seed):
    r = random.Random(seed)
    cands = [Candidate(f"t{i}", f"t{i}", r.choice([1, 2, 10, 1000, 10 ** 6, 10 ** 12]) * r.randint(1, 9),
                       r.randint(0, 10 ** 7), r.randint(0, 16)) for i in range(r.randint(1, 6))]
    sizes = {c.template: min(c.units, r.choice([1, 2, 5, 40, 300, c.units])) for c in cands}
    objective = Objective(r.choice([0.5, 0.01, 1e-6]), work=r.random(), bits=r.choice([0.0, r.random() * 1e4]))
    cert, inexact = certify(sizes, cands, objective), certify(sizes, cands, objective, exact=False)

    def rows(h) -> list[tuple[int, int, float]]:
        return [(c.units, sizes[c.template], h(c)) for c in cands]

    harm, work = rows(objective.harm), rows(lambda c: c.work + c.influence)
    assert at_supremum(cert.harm, supremum(server, harm, objective.delta))
    assert at_supremum(cert.free_compute, supremum(server, work, objective.delta))
    assert held(cert.density, lean(server, harm, objective.delta)) and inexact.harm == inexact.density == cert.density
    assert held(inexact.free_compute, lean(server, work, objective.delta))
    if cert.leak is not None:
        law = Stratified(tuple(Stratum(c.template, c.units, sizes[c.template]) for c in cands))
        loc = location_bits(IntegrityProfile(("unit",), (law.population,), {"unit": law}, objective.delta))
        lower, upper = supremum(server, rows(lambda c: c.out_bits), objective.delta)
        assert lower <= Fraction(cert.leak) <= (upper * (1 + SLACK) + Fraction(loc)) * (1 + Fraction(1, 2 ** 52))

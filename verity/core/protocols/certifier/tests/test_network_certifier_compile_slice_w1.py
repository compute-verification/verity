"""The compile slice (`compile_slice.CompileSlice_v1`) on an async served window: window 1 of a vLLM run,
requests 4 to 7 of the session, 3 decisions and 38 units, whose sync schedule isn't the served one
(the encodings' `test_served_window.py`). `fixtures/compile_slice_w1.json` is the window at its own statics (R, W, A) =
(4, 1, 5), Q 4, D 3, P 38, K 4, and `fixtures/compile_slice_w1_class.json` at the engine's public class (8, 1, 9), Q 4,
D 32, P 54, K 4, both under its async scheduling, AS = 1 (`replay_vectors.py --slice 1 --AS 1`). The charge is from the
session's carry into w1 (`CARRY`: the chain over window 0, whose decisions 1, 1, 1 charge 24 bits from the priors).

At AS = 1 the slice's ``rows`` holds the served rows to the scan's through ``wit``'s ``map``, a witnessed one-to-one map
from the runner's batch positions onto the scheduler's slots with rows: the runner's batch isn't the slots at units 0 and
5 here."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import served_rows as SR
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK

FIXTURES = Path(__file__).parent / "fixtures"
OWN, CLASS = "compile_slice_w1.json", "compile_slice_w1_class.json"
ANDS = {OWN: 314936, CLASS: 1021302}
#: the session's carry into window 1 (the module's docstring)
CARRY = AE.Carry(((11, 4), (11, 0), (30, 1)), 0)


def _load(name):
    fx = json.loads((FIXTURES / name).read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    served = [SR.step_rows(S, step, fx["ids"]) for step in fx["served"]]
    return SimpleNamespace(name=name, st=st, S=S, sl=sl, fx=fx, served=served)


@pytest.fixture(scope="module", params=[OWN, CLASS])
def w1(request):
    return _load(request.param)


def _cases(x):
    """The honest unit and the negatives of w212's: a charge one below the advice's, request 5's prompt a token longer
    than certified, unit 2's served rows missing their last; the one the map opens, unit 0's first two served rows
    (requests 4 and 6, 78 and 157 prompt tokens) with their requests swapped; and at the class, a request admitted a
    step late in the window row (a pass inserted after the first ask; window 1 asks no 0 to flip), whose fourth
    decision is past the window's own D = 3."""
    S, sl, served = x.S, x.sl, x.served
    kw = {"served": served, "carry": CARRY}
    honest = CS.case(S, sl, BOOK, **kw)
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, *cert[1][2:])
    a, b = served[0][:2]
    swap = [((a[0], b[1], *a[2:]), (b[0], a[1], *b[2:])) + served[0][2:]] + served[1:]
    cases = {"honest": honest, "charge": CS.case(S, sl, BOOK, charge=honest.length - 1, **kw),
             "arrival": CS.case(S, sl, BOOK, cert=cert, **kw),
             "served": CS.case(S, sl, BOOK, **{**kw, "served": served[:2] + [served[2][:-1]] + served[3:]}),
             "swap": CS.case(S, sl, BOOK, **{**kw, "served": swap})}
    if x.name == CLASS:
        cases["late"] = CS.case(S, sl, BOOK, window=sl.decisions[:1] + (0,) + sl.decisions[1:], **kw)
    return cases


def test_the_served_rows_are_the_replays_up_to_their_slots(w1):
    """The seal's rows are the async replay's, unit for unit, as (request, kind, start, n); their batch positions are
    the scheduler's slots but at unit 0 (the runner's batch is requests 4, 6, 5, 7) and unit 5 (request 5 holds slot 0
    while its last output is in flight, so request 6, in slot 1, is the batch's first row), and the map the honest
    prover puts in ``map`` (`compile_slice.slot_map`) is the identity on the other units' rows."""
    served, rows = [[list(r) for r in s] for s in w1.served], w1.fx["rows"]
    assert [sorted(r[1:] for r in s) for s in served] == [sorted(r[1:] for r in s) for s in rows]
    assert [u for u, (a, b) in enumerate(zip(served, rows)) if a != b] == [0, 5]
    maps = [CS.slot_map(w1.S, [tuple(r) for r in rows[u]], w1.served[u]) for u in range(len(rows))]
    slots = [[r.index(1) if 1 in r else None for r in m] for m in maps]
    assert slots[0][:4] == [0, 2, 1, 3] and slots[5][:2] == [1, None]
    assert all(slots[u][r[0]] == r[0] for u in range(len(rows)) if u not in (0, 5) for r in w1.served[u])
    assert w1.S.AS == 1 and w1.fx["steps"] == 38 and [n for n, _a, _b in CS.fields(w1.S)["wit"]][-1] == "map"


def test_the_slice_accepts_w1(w1):
    """The honest unit is accepted (0x3F) with the served rows in ``srv``, its scan asking the 3 committed decisions,
    which charge 21 bits from the session's carry; each negative is refused as at w212: the charge by ``charge``, the
    longer prompt and the dropped row by ``rows``, the late admission by ``load`` and ``rows``; and the swapped requests
    by ``rows``, as no slot's row is either served row."""
    cases = _cases(w1)
    honest = cases["honest"]
    assert honest.verdict == CS.ACCEPT and honest.asks == w1.sl.decisions and honest.length == 21
    want = {"honest": set(), "charge": {"charge"}, "arrival": {"rows"}, "served": {"rows"}, "swap": {"rows"}}
    if w1.name == CLASS:
        want["late"] = {"load", "rows"}
    assert {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()} == want


@pytest.mark.slow
@pytest.mark.parametrize("name", [OWN, CLASS])
def test_the_circuit_is_its_reference_on_w1(name):
    """The unit's verdict word is the reference's on the honest window and each negative, and its size, at w1's own
    statics and at the engine class. About two minutes, at 7.3 GB: it builds and evaluates the 314,936- and
    1,021,302-AND units at the served engine's widths under async scheduling, the map at units 0 and 5 not the
    identity, which the async demonstration slice's test doesn't reach."""
    x = _load(name)
    cases = _cases(x)
    assert CS.evaluate(x.st, list(cases.values())) == [c.verdict for c in cases.values()]
    assert CS.size(**x.st)["ands"] == ANDS[name]

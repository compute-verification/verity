"""The compile slice (`compile_slice.CompileSlice_v1`) on a served window: window 212 of a vLLM run,
which `replay_vectors.py --slice 212` builds from the run's scheduler trace and its seal. Its three
requests' 14 decisions and 20 steps, at the served engine with the slice's sync ReplayStep_v1, whose schedule on this
window is the served one (the encodings' `test_served_window.py`). `fixtures/compile_slice_w212.json` is the window at
its own statics; `fixtures/compile_slice_w212_class.json` is the same window padded into the engine's public sizes
(R, W, A) = (8, 1, 9), Q 4, D 32, P 54, K 4 under a sync scheduler, the sync class. The engine's own class is its
AsyncScheduler's (AS = 1), where `benchmarks/network_traces/compile_slice_zk.py` proves w212. The charge is from the
session's carry into w212 (`CARRY`: the chain over windows 0 … 211, each window's decisions as its own replay asks
them, under the async-light book; `benchmarks/network_traces/compile_slice_zk.py` derives it from the run's records)."""
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
OWN, CLASS = "compile_slice_w212.json", "compile_slice_w212_class.json"
ANDS = {OWN: 117164, CLASS: 857866}
#: the session's carry into window 212 (the module's docstring)
CARRY = AE.Carry(((58, 50), (58, 0), (392, 48)), 0)
#: counts fitted to the window, a restart that charges it 25 bits in place of 27
FITTED = AE.Carry(((1, 0), (16, 0), (64, 16)), 0)


def _load(name):
    fx = json.loads((FIXTURES / name).read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    served = [SR.step_rows(S, step, fx["ids"]) for step in fx["served"]]
    return SimpleNamespace(name=name, st=st, S=S, sl=sl, fx=fx, served=served)


@pytest.fixture(scope="module", params=[OWN, CLASS])
def w212(request):
    return _load(request.param)


def _cases(x):
    """The honest unit and each negative: request 310 admitted a step early in the window row, a charge one below the
    advice's, request 309's prompt a token longer than certified, step 7's served rows missing request 309's, advice
    that leaves the last asked decision uncommitted (``cleared``: the window row keeps it; ``cleared-read``: the window
    row drops it too, and the scan reads past the count), a carry-out other than the charge's carry', and six
    passes at w212 (``sixpass``: advice that registers a pass at every ask, its own scan's twenty, where
    the served window admitted requests 309 and 310 at the seventh and fourteenth), with the served rows in ``srv`` and
    with its own. The last three read or commit more decisions than the window's own D = 14, so they are the class's
    only."""
    S, sl, served = x.S, x.sl, x.served
    kw = {"served": served, "carry": CARRY}
    honest = CS.case(S, sl, BOOK, **kw)
    early = list(sl.decisions)
    early[12] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    cut = sl.decisions[:-1]
    six = CS.Slice(sl.arrivals, (0,) * 20, sl.steps)
    cases = {"honest": honest, "early": CS.case(S, sl, BOOK, window=early, **kw),
             "charge": CS.case(S, sl, BOOK, charge=honest.length - 1, **kw),
             "arrival": CS.case(S, sl, BOOK, cert=cert, **kw),
             "served": CS.case(S, sl, BOOK, **{**kw, "served": served[:7] + [served[7][:1]] + served[8:]}),
             "cleared": CS.case(S, sl, BOOK, advice=cut, **kw),
             "carry": CS.case(S, sl, BOOK, cout=FITTED, **kw)}
    if x.name == CLASS:
        cases.update({"cleared-read": CS.case(S, sl, BOOK, advice=cut, window=cut, **kw),
                      "sixpass": CS.case(S, six, BOOK, **kw), "sixpass-own": CS.case(S, six, BOOK, carry=CARRY)})
    return cases


def test_the_served_rows_are_the_replays(w212):
    """The seal's rows (the model runner's, by `served_rows.step_rows`) are the slice's scan of the window, step for step;
    at the class, the 34 steps past the window's 20 are idle."""
    x = w212
    assert [[list(r) for r in step] for step in x.served] == x.fx["rows"]
    assert [list(map(list, s)) for s in CS.case(x.S, x.sl, BOOK).steps] == x.fx["rows"]
    assert x.fx["steps"] == 20 and x.S.P == {OWN: 20, CLASS: 54}[x.name]


def test_the_slice_accepts_w212(w212):
    """The honest unit is accepted (0x3F) with the served rows in ``srv``: its scan asks the 14 committed decisions in
    commit order, which charge 27 bits from the session's carry (28 from the priors, 25 from the fitted counts). Each
    negative is refused by its own checks: the early admission by ``load`` and ``rows`` (its steps leave the last
    decision unread, which the chain carries, so ``steps`` holds), the advice that drops an asked decision by ``load``
    or, with the window row cut too, by ``steps`` and ``rows`` (the scan's last ask reads past the count, a pass where
    the served window admitted), the forged carry-out by ``code``, and the twenty passes (23 bits) by ``rows`` alone
    against the served rows; with its own rows in ``srv`` the unit accepts them, and the native b‖c check refuses
    ``srv``. The padding (an empty arrival slot, 18 decisions past the count, 34 idle steps) changes no verdict and no
    charge."""
    cases = _cases(w212)
    honest = cases["honest"]
    assert honest.verdict == CS.ACCEPT and honest.asks == w212.sl.decisions and honest.length == 27
    cb, prior = AE.charge_book(BOOK, w212.S.D)
    en, bits = [1] * 14 + [0] * (w212.S.D - 14), list(w212.sl.decisions) + [0] * (w212.S.D - 14)
    assert [AE.reference_charge(w212.S.D, cb, c, en, bits)[1] for c in (prior, FITTED)] == [28, 25]
    flags = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    want = {"honest": set(), "early": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"}, "served": {"rows"},
            "cleared": {"load"}, "carry": {"code"}}
    if w212.name == CLASS:
        want.update({"cleared-read": {"steps", "rows"}, "sixpass": {"rows"}, "sixpass-own": set()})
        assert cases["sixpass"].length == 23
    assert flags == want and cases["cleared"].length < honest.length


@pytest.mark.slow
@pytest.mark.parametrize("name", [OWN, CLASS])
def test_the_circuit_is_its_reference_on_w212(name):
    """The unit's verdict word is the reference's on the honest window and each negative, and its size at w212's shape
    and at the engine class. About two minutes: it builds and evaluates the 117,164- and 857,866-AND units at the served
    engine's widths (L = 512, NB = 101,626; P = 20 and 54, D = 14 and 32), which the smallkv slice's test doesn't
    reach."""
    x = _load(name)
    cases = _cases(x)
    assert CS.evaluate(x.st, list(cases.values())) == [c.verdict for c in cases.values()]
    assert CS.size(**x.st)["ands"] == ANDS[name]

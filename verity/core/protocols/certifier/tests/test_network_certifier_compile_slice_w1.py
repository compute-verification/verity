"""The compile slice (`compile_slice.CompileSlice_v1`) on an async served window: window 1 of vLLM's run
r20261024-141912-8eb1, requests 4 to 7 of the session, 3 decisions and 38 units, whose sync schedule isn't the served one
(the encodings' `test_served_window.py`). `fixtures/compile_slice_w1_class.json` is the window at the engine's public
class (R, W, A) = (8, 1, 9), Q 4, D 32, P 54, K 4 under its async scheduling, AS = 1 (`replay_vectors.py --slice 1 --AS
1`). The charge is from the session's carry into w1 (`CARRY`: the chain over window 0, whose decisions 1, 1, 1 charge 24
bits from the priors).

The slice doesn't take the window yet (`protocol/served-shape.md`, "For the 25 async windows"), so the test of what it
must do is a strict xfail, which its fix turns into a pass:
- its reference is sync: its rows' entries carry the async fields (`replay_chain`'s at AS = 1), but `_state`, `_row`,
  `_inputs` and `check`'s arrivals build sync tuples, so `case` raises `MalformedRow` on the window's first state row;
- its ``rows`` check reads the served rows by batch position (`served_rows.step_rows`' slot), and the runner's batch
  isn't the scheduler's slots at units 0 and 5."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import served_rows as SR
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_w1_class.json"
#: the session's carry into window 1 (the module's docstring)
CARRY = AE.Carry(((11, 4), (11, 0), (30, 1)), 0)
NOT_YET = ("the slice's reference builds sync entries where its rows' are async, and its rows check reads the served rows "
           "by batch position (protocol/served-shape.md, 'For the 25 async windows')")


@pytest.fixture(scope="module")
def w1():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    served = [SR.step_rows(S, step, fx["ids"]) for step in fx["served"]]
    return SimpleNamespace(st=st, S=S, sl=sl, fx=fx, served=served)


def _cases(x):
    """The honest unit and the four negatives of w212's: a request admitted a step late in the window row (a pass
    inserted after the first ask; window 1 asks no 0 to flip), a charge one below the advice's, request 5's prompt a
    token longer than certified, and unit 2's served rows missing their last."""
    S, sl, served = x.S, x.sl, x.served
    kw = {"served": served, "carry": CARRY}
    honest = CS.case(S, sl, BOOK, **kw)
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, *cert[1][2:])
    return {"honest": honest, "late": CS.case(S, sl, BOOK, window=sl.decisions[:1] + (0,) + sl.decisions[1:], **kw),
            "charge": CS.case(S, sl, BOOK, charge=honest.length - 1, **kw),
            "arrival": CS.case(S, sl, BOOK, cert=cert, **kw),
            "served": CS.case(S, sl, BOOK, **{**kw, "served": served[:2] + [served[2][:-1]] + served[3:]})}


def test_the_served_rows_are_the_replays_up_to_their_slots(w1):
    """The seal's rows are the async replay's, unit for unit, as (request, kind, start, n); their slots, the runner's
    batch positions, are the scheduler's but at unit 0 (the runner's batch is requests 4, 6, 5, 7) and unit 5 (request 5
    holds its slot while its last output is in flight, so request 6 is the batch's first row)."""
    served, rows = [[list(r) for r in s] for s in w1.served], w1.fx["rows"]
    assert [sorted(r[1:] for r in s) for s in served] == [sorted(r[1:] for r in s) for s in rows]
    assert [u for u, (a, b) in enumerate(zip(served, rows)) if a != b] == [0, 5]
    assert w1.S.AS == 1 and w1.fx["steps"] == 38


@pytest.mark.xfail(strict=True, reason=NOT_YET)
def test_the_slice_accepts_w1(w1):
    """The honest unit is accepted (0x3F) with the served rows in ``srv``, its scan asking the 3 committed decisions,
    which charge 21 bits from the session's carry; each negative is refused as at w212: the late admission by ``load``
    and ``rows``, the charge by ``charge``, the longer prompt and the dropped row by ``rows``."""
    cases = _cases(w1)
    honest = cases["honest"]
    assert honest.verdict == CS.ACCEPT and honest.asks == w1.sl.decisions and honest.length == 21
    assert {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()} == {
        "honest": set(), "late": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"}, "served": {"rows"}}

"""The compile slice (`compile_slice.CompileSlice_v1`) on a served window: window 212 of vLLM's run
r20261024-141912-8eb1 (`fixtures/compile_slice_w212.json`, which `replay_vectors.py --slice 212` builds from the run's
scheduler trace and its seal). Its three requests' 14 decisions and 20 steps, at the served engine with the slice's
sync ReplayStep_v1, whose schedule on this window is the served one (the encodings' `test_served_window.py`)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import served_rows as SR

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_w212.json"


@pytest.fixture(scope="module")
def w212():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    served = [SR.step_rows(S, step, fx["ids"]) for step in fx["served"]]
    return SimpleNamespace(st=st, S=S, sl=sl, fx=fx, served=served)


def _cases(x):
    S, sl, served = x.S, x.sl, x.served
    honest = CS.case(S, sl, served=served)
    early = list(sl.decisions)
    early[12] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    return {"honest": honest, "early": CS.case(S, sl, served=served, window=early),
            "charge": CS.case(S, sl, served=served, charge=honest.length - 1),
            "arrival": CS.case(S, sl, served=served, cert=cert),
            "served": CS.case(S, sl, served=served[:7] + [served[7][:1]] + served[8:])}


def test_the_served_rows_are_the_replays(w212):
    """The seal's rows (the model runner's, by `served_rows.step_rows`) are the slice's scan of the window, step for step."""
    x = w212
    assert [[list(r) for r in step] for step in x.served] == x.fx["rows"]
    assert [list(map(list, s)) for s in CS.case(x.S, x.sl).steps] == x.fx["rows"]


def test_the_slice_accepts_w212(w212):
    """The honest unit is accepted (0x3F) with the served rows in ``srv``: the 14 decisions code in 13 bits. Each negative is
    refused by its own checks: request 310 admitted a step early in the window row (the decision it leaves unread
    carries, so ``steps`` holds), a charge one below the coded length, request 309's prompt a token longer than
    certified, and step 7's served rows missing request 309's."""
    cases = _cases(w212)
    assert cases["honest"].verdict == CS.ACCEPT and cases["honest"].length == 13 and len(cases["honest"].symbols) == 14
    flags = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    assert flags == {"honest": set(), "early": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "served": {"rows"}}


@pytest.mark.slow
def test_the_circuit_is_its_reference_on_w212(w212):
    """The unit's verdict word is the reference's on the honest window and each negative, and its size at w212's shape.
    About a minute: it builds and evaluates the 222,668-AND unit at the served engine's widths (L = 512, NB = 101,626,
    P = 20), which the smallkv slice's test doesn't reach."""
    cases = _cases(w212)
    assert CS.evaluate(w212.st, list(cases.values())) == [c.verdict for c in cases.values()]
    assert CS.size(**w212.st)["ands"] == 222668

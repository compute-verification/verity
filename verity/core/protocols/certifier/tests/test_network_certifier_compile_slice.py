"""The compile slice (`compile_slice.CompileSlice_v1`) on corpus-smallkv's first seven steps: the reference's scan is the
replay's (its rows and its symbols' contexts), the circuit's verdict is `check`'s on the honest rows and on each
negative (a decision flipped in the window row, a charge one below the coded length, an arrival not the certified
one, served rows not the run's) and on each of REDTEAM's phase 9 choices (a context, frequencies, a prefix hit, the heads left in
place), the codeword is the encoder's, and the cost `size` reports."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import compile_slice as CS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"


@pytest.fixture(scope="module")
def slice_():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"])
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, rows=rows)


def _negatives(x):
    S, sl = x.S, x.sl
    honest = CS.case(S, sl, served=x.rows)
    flip = list(sl.decisions)
    flip[5] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    shorter = [((0, 0, 0, 0, 255),)] + x.rows[1:]
    return {"honest": honest, "flip": CS.case(S, sl, served=x.rows, window=flip),
            "charge": CS.case(S, sl, served=x.rows, charge=honest.length - 1),
            "arrival": CS.case(S, sl, served=x.rows, cert=cert),
            "served": CS.case(S, sl, served=shorter)}


def test_the_scan_is_the_replay(slice_):
    """The honest scan's rows are the replay's, its ``srv`` row is the clear rows', and its symbols are the plain contexts'
    (ENC-REPLAY's KT counts, scaled as advice v1 does): the queue head's passes 0, 1, 2, 2, 2 and 2, then a fresh head."""
    x = slice_
    c = CS.case(x.S, x.sl)
    assert list(c.steps) == x.rows and c.rows["srv"] == CS.srv_row(x.S, x.rows)
    assert c.verdict == CS.ACCEPT and all(c.flags.values())
    want, n = [], {k: (0, 0) for k in range(3)}
    for ctx, bit in zip((0, 1, 2, 2, 2, 2, 0), x.sl.decisions):
        want.append((bit, *AE.scaled(2 * n[ctx][0] + 1, 2 * n[ctx][1] + 1)))
        n[ctx] = (n[ctx][0] + (not bit), n[ctx][1] + bit)
    assert list(c.symbols) == want


def test_the_circuit_is_its_reference(slice_):
    """The unit's verdict word is the reference's on the honest rows and on every negative, and each negative is
    refused by exactly the checks the reference names."""
    cases = _negatives(slice_)
    got = CS.evaluate(slice_.st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()]
    flags = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    assert flags == {"honest": set(), "flip": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "served": {"rows"}}


def test_redteams_phase9_choices_are_refused(slice_):
    """Every interface REDTEAM's phase 9 found free in a piece alone (`compile_slice.phase9`) is the unit's own here: no
    witness field carries a context, a frequency, a hit, an eviction, a return or a consumed count, and each choice is
    refused by the proved verdict. The codes are shorter than the honest one (8 bits), and the unit holds the charge to
    its own count; the hit schedules 32 tokens fewer, and the unit's step takes no hit (the cache counts' stand-in);
    the heads must move by ReplayStep_v1's own taken and used."""
    x = slice_
    assert [name for name, _n, _w in CS.fields(x.S)["wit"]] == ["state", "window", "arr", "bits", "seg"]
    cases = {k: CS.case(x.S, x.sl, served=x.rows, **kw) for k, kw in CS.phase9(x.S, x.sl).items()}
    assert tuple(cases) == CS.PHASE9
    assert CS.evaluate(x.st, list(cases.values())) == [c.verdict for c in cases.values()]
    flags = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    assert flags == {"context": {"code", "charge"}, "frequency": {"code", "charge"}, "hit": {"steps", "rows"},
                     "reread": {"steps", "rows"}, "untaken": {"steps", "rows"}}
    assert (cases["context"].length, cases["frequency"].length) == (4, 2)
    assert cases["hit"].steps[0] == ((0, 0, 0, 32, 256),) and x.rows[0] == ((0, 0, 0, 0, 256),)


def test_the_codeword_is_the_encoders():
    """`codeword` is the segment `arith_encode`'s reference accepts, and no bit of it can flip."""
    rng = random.Random(7)
    for _ in range(40):
        syms = [(rng.randrange(2), rng.randrange(1, 300), rng.randrange(1, 300)) for _ in range(rng.randrange(1, 12))]
        seg = CS.codeword(syms)
        assert AE.check_window(AE.K_MAX, 1, syms, seg)
        k = rng.randrange(len(seg))
        assert not AE.check_window(AE.K_MAX, 1, syms, seg[:k] + [1 - seg[k]] + seg[k + 1:])


def test_size(slice_):
    s = CS.size(**slice_.st)
    assert s == {"ands": 60519, "pub_bits": 16, "adv_bits": 16, "cert_bits": 112, "srv_bits": 432,
                 "wit_bits": 2160}

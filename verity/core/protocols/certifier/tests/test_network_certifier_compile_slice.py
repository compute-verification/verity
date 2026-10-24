"""The compile slice (`compile_slice.CompileSlice_v1`) on corpus-smallkv's first seven steps: the reference's scan is the
replay's (its rows and its symbols' contexts), the circuit's verdict is the reference's on the honest rows and on each
negative (a decision flipped in the window row, a charge one below the coded length, an arrival not the certified
one, a wrong digest), the codeword is the encoder's, and the cost `size` reports."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import compile_slice as CS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
SALT = bytes(range(32))


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
    honest = CS.case(S, sl, SALT)
    digest = CS.rows_digest(S, x.rows, SALT)
    flip = list(sl.decisions)
    flip[5] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    return {"honest": honest, "flip": CS.case(S, sl, SALT, digest=digest, window=flip),
            "charge": CS.case(S, sl, SALT, digest=digest, charge=honest.length - 1),
            "arrival": CS.case(S, sl, SALT, digest=digest, cert=cert),
            "digest": CS.case(S, sl, SALT, digest=bytes(64)),
            "salt": CS.case(S, sl, bytes(32), digest=digest)}


def test_the_scan_is_the_replay(slice_):
    """The honest scan's rows are the replay's, its digest is the clear rows', and its symbols are the plain contexts'
    (ENC-REPLAY's KT counts, scaled as advice v1 does): the queue head's passes 0, 1, 2, 2, 2 and 2, then a fresh head."""
    x = slice_
    c = CS.case(x.S, x.sl, SALT)
    assert list(c.steps) == x.rows
    assert c.verdict == CS.ACCEPT and all(c.flags.values())
    want, n = [], {k: (0, 0) for k in range(3)}
    for ctx, bit in zip((0, 1, 2, 2, 2, 2, 0), x.sl.decisions):
        want.append((bit, *AE.scaled(2 * n[ctx][0] + 1, 2 * n[ctx][1] + 1)))
        n[ctx] = (n[ctx][0] + (not bit), n[ctx][1] + bit)
    assert list(c.symbols) == want
    assert CS.rows_digest(x.S, x.rows, SALT) == CS.rows_digest(x.S, c.steps, SALT)


def test_the_circuit_is_its_reference(slice_):
    """The unit's verdict word is the reference's on the honest rows and on every negative, and each negative is
    refused by exactly the checks the reference names."""
    cases = _negatives(slice_)
    got = CS.evaluate(slice_.st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()]
    flags = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    assert flags == {"honest": set(), "flip": {"load", "steps", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "digest": {"rows"}, "salt": {"rows"}}


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
    assert s == {"ands": 139327, "pub_bits": 528, "adv_bits": 16, "cert_bits": 112, "wit_bits": 2416}

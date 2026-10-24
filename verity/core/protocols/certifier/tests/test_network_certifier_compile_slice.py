"""The compile slice (`compile_slice.CompileSlice_v1`) on corpus-smallkv's first seven steps: the reference's scan is the
replay's (its rows, and its asks the committed decisions in commit order), its charge `ChargeLink_v1`'s from the
window's carry-in, and `check`'s verdict on each negative (a decision flipped in the window row, a charge one below the
advice's, an arrival not the certified one, served rows not the run's, advice that drops an asked decision, a
carry-out not the charge's carry') and on each of REDTEAM's phase 9 choices the slice still has (a prefix hit, the heads
left in place) and the prefix cache's (a made-up table, made-up tokens). The circuit is its reference on the
demonstration slice (`compile_slice.demo`), whose candidates take real prefix hits (the corpus class, about 7.6M ANDs,
is past what a test evaluates), and the cost `size` reports. The registered book is the arith fixture's async-light
stream's (fit on that run, never on the slice's)."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE, charge_chain as CC
from verity.core.protocols.certifier import compile_slice as CS

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "compile_slice_smallkv.json"


def book() -> dict:
    """The registered book: the arith fixture's async-light stream's (`arith_vectors.decisions`)."""
    streams = json.loads((FIXTURES / "arith_encode_vectors.json").read_text())["streams"]
    return next(s for s in streams if s["source"].startswith("async-light"))["decisions"]["book"]


BOOK = book()


@pytest.fixture(scope="module")
def slice_():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"])
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, rows=rows)


@pytest.fixture(scope="module")
def synth():
    """The demonstration slice (`compile_slice.demo`)."""
    st, sl = CS.demo()
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


def _refused(c: CS.Case) -> set[str]:
    return {f for f, ok in c.flags.items() if not ok}


def _negatives(x, flip_at: int):
    S, sl = x.S, x.sl
    honest = CS.case(S, sl, BOOK)
    served = list(honest.steps)
    first = served[0][0]
    flip = list(sl.decisions)
    flip[flip_at] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    shorter = [((*first[:4], first[4] - 1),) + served[0][1:]] + served[1:]
    return {"honest": honest, "flip": CS.case(S, sl, BOOK, served=served, window=flip),
            "charge": CS.case(S, sl, BOOK, served=served, charge=honest.length - 1),
            "arrival": CS.case(S, sl, BOOK, served=served, cert=cert),
            "served": CS.case(S, sl, BOOK, served=shorter),
            "cleared": CS.case(S, sl, BOOK, served=served, advice=sl.decisions[:-1]),
            "carry": CS.case(S, sl, BOOK, served=served, cout=AE.first_carry(BOOK))}


def test_the_scan_is_the_replay(slice_):
    """The honest scan's rows are the replay's and its ``srv`` row is the clear rows'; it asks the committed decisions in
    commit order, and the charge is `ChargeCount_v1`'s on them from the book's priors (the session's first window), with
    ``cout`` its carry'. No request shares a block with another, so every hit is 0, and the table holds request 0's
    units as its rows complete them, then request 1's first."""
    x = slice_
    c = CS.case(x.S, x.sl, BOOK)
    assert list(c.steps) == x.rows and c.rows["srv"] == CS.srv_row(x.S, x.rows)
    assert c.verdict == CS.ACCEPT == 0x3F and all(c.flags.values())
    assert c.hits == ((0, 0, 0),) * 7 and c.asks == x.sl.decisions
    cb, prior = AE.charge_book(BOOK, x.S.D)
    n = len(x.sl.decisions)
    want = AE.reference_charge(x.S.D, cb, prior, [1] * n + [0] * (x.S.D - n), list(x.sl.decisions) + [0] * (x.S.D - n))
    assert (1, c.length, c.carry) == want and c.length == 22
    assert c.rows["cin"] == list(CC.encode(prior)) and c.rows["cout"] == list(CC.encode(c.carry))
    sc = CS.scan(x.S, x.sl)
    assert [len(t.table) for t in sc.caches] == [0, 16, 23, 23, 23, 23, 23, 24]
    assert [t.pfx for t in sc.caches][-1] == (22, 23)


def test_the_negatives(slice_):
    """Each negative is refused by exactly the checks the reference names: advice that leaves the window's last asked
    decision uncommitted (REDTEAM phase 14's cleared enable) by ``load``, since the window row still holds it, and a
    carry-out other than the charge's carry' by ``code``."""
    cases = _negatives(slice_, 5)
    flags = {k: _refused(c) for k, c in cases.items()}
    assert flags == {"honest": set(), "flip": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "served": {"rows"}, "cleared": {"load"}, "carry": {"code"}}


def test_redteams_phase9_choices_are_refused(slice_):
    """Every interface REDTEAM's phase 9 found free in a piece alone (`compile_slice.phase9`) is the unit's own here: no
    witness field carries a context, a frequency, an eviction, a return or a consumed count, and each choice the slice
    still has is refused by the verdict. The hit schedules 32 tokens fewer, and CacheStep_v1 refuses it (the table is
    empty at step 0); the heads must move by ReplayStep_v1's own taken and used."""
    x = slice_
    assert [name for name, _n, _w in CS.fields(x.S)["wit"]] == ["state", "window", "arr", "bits", "hit", "cache"]
    cases = {k: CS.case(x.S, x.sl, BOOK, served=x.rows, **kw) for k, kw in CS.phase9(x.S, x.sl).items()}
    assert tuple(cases) == CS.PHASE9
    flags = {k: _refused(c) for k, c in cases.items()}
    assert flags == {"hit": {"cache", "rows"}, "reread": {"steps", "rows"}, "untaken": {"steps", "rows"}}
    assert cases["hit"].steps[0] == ((0, 0, 0, 32, 256),) and x.rows[0] == ((0, 0, 0, 0, 256),)


def test_the_cache_binds_the_hit(slice_):
    """REDTEAM phase 9 (4)'s binding on the corpus: CacheStep_v1's hit pinned to the committed table and its tokens free
    in the piece alone (`test_network_certifier_cache_step_phase9.py`) are bound in the slice. A made-up first table
    (request 1's first block under ROOT) gives request 1 a free 16-token hit at step 6, and the unit refuses it by its
    ``cache`` check alone when the prover serves that schedule (its own rows in ``srv``), since the table chains from
    the empty row; made-up tokens (request 1's first block as request 0's) give the same hit through a consistent
    table, and ``rows`` refuses them against the served rows, as ``cert`` is the certified record's, which the verifier
    checks natively."""
    x = slice_
    table = CS.case(x.S, x.sl, BOOK, **CS.made_up(x.S, x.sl, 1))
    assert table.steps[6] == ((0, 0, 1, 376, 1), (1, 1, 0, 16, 13)) and table.hits[6] == (0, 1, 0)
    assert _refused(table) == {"cache"}
    assert _refused(CS.case(x.S, x.sl, BOOK, served=x.rows, **CS.made_up(x.S, x.sl, 1))) == {"cache", "rows"}
    toks = [list(t) for t in x.sl.tokens]
    toks[1][:x.S.B] = toks[0][:x.S.B]
    made = CS.case(x.S, x.sl, BOOK, served=x.rows, tokens=toks)
    assert made.steps[6] == table.steps[6] and _refused(made) == {"rows"}


def test_the_circuit_is_its_reference(synth):
    """The unit's verdict word is the reference's on a synthetic slice where every hit path is taken: request 1 is the
    first candidate left unadmitted at steps 0 and 1 (its 2-block hit looked up at the step's last table), is admitted
    at step 2 with that hit, and request 2 is left unadmitted at steps 2 and 3 with a 1-block hit, after request 1 in
    the same step. Then on every negative, the hits claimed shorter or made up by a table or tokens, REDTEAM's phase 9
    hit, and a restarted carry-in, which the unit accepts at its lower charge (the chain refuses it,
    `test_network_certifier_compile_slice_phase14.py`)."""
    x = synth
    cases = _negatives(x, 2)
    honest = cases["honest"]
    assert honest.verdict == CS.ACCEPT
    assert honest.hits == ((0, 2, 0), (0, 2, 0), (0, 2, 1), (0, 1, 0))
    assert honest.steps[2] == ((0, 0, 1, 10, 1), (1, 1, 0, 8, 3))
    served = list(honest.steps)
    cases["short"] = CS.case(x.S, x.sl, BOOK, served=served, hits={2: (0, 1, 1)})
    cases["unasked"] = CS.case(x.S, x.sl, BOOK, served=served, hits={0: (0, 0, 0)})
    cases["table"] = CS.case(x.S, x.sl, BOOK, **CS.made_up(x.S, x.sl, 2, 2))
    toks = [list(t) for t in x.sl.tokens]
    toks[1][:4] = [9, 9, 9, 9]
    cases["tokens"] = CS.case(x.S, x.sl, BOOK, served=served, tokens=toks)
    cases["hit"] = CS.case(x.S, x.sl, BOOK, served=served, **CS.phase9(x.S, x.sl)["hit"])
    cases["restart"] = CS.case(x.S, x.sl, BOOK, served=served, carry=AE.Carry(((0, 0),) * AE.RUNS, 0))
    got = CS.evaluate(x.st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()]
    flags = {k: _refused(c) for k, c in cases.items()}
    assert flags == {"honest": set(), "flip": {"load", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "served": {"rows"}, "cleared": {"load"}, "carry": {"code"}, "short": {"cache", "rows"},
                     "unasked": {"cache"}, "table": {"cache"}, "tokens": {"rows"}, "hit": {"cache", "rows"},
                     "restart": set()}
    assert cases["restart"].length < honest.length
    assert cases["table"].hits[0] == (1, 2, 0) and cases["table"].steps[0] == ((0, 0, 0, 4, 5),)
    assert cases["tokens"].hits[2] == (0, 0, 1) and cases["tokens"].steps[2][1] == (1, 1, 0, 0, 11)


def test_size(synth, slice_):
    s = CS.size(**synth.st)
    assert s == {"ands": 59341, "pub_bits": 256, "adv_bits": 16, "cert_bits": 256, "srv_bits": 144, "cin_bits": 96,
                 "cout_bits": 96, "wit_bits": 1568}
    assert CS.row_bits(slice_.S) == {"pub": 256, "adv": 16, "cert": 20512, "srv": 432, "cin": 96, "cout": 96,
                                     "wit": 56928}

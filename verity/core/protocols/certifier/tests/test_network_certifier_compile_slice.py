"""The compile slice (`compile_slice.CompileSlice_v1`) on corpus-smallkv's first seven steps: the reference's scan is the
replay's (its rows and its symbols' contexts), `check`'s verdict on each negative (a decision flipped in the window
row, a charge one below the coded length, an arrival not the certified one, a wrong digest) and on each of REDTEAM's
phase 9 choices (a context, frequencies, a prefix hit, the heads left in place) and the prefix cache's (a made-up table,
made-up tokens), the codeword is the encoder's. The circuit is its reference on a synthetic slice whose candidates take
real prefix hits (the corpus class, about 7.6M ANDs, is held to its reference by the ZK run's staged verdicts), and the
cost `size` reports."""
from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import cache_step as CST
from verity.core.protocols.certifier import compile_slice as CS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
SALT = bytes(range(32))
#: a synthetic engine small enough to trace: 4-token blocks, 16 tokens a step
ENGINE = {"max_num_seqs": 4, "max_num_batched_tokens": 16, "max_model_len": 64, "num_gpu_blocks": 20, "block_size": 4,
          "enable_prefix_caching": True, "enable_chunked_prefill": True, "scheduler_reserve_full_isl": True}


@pytest.fixture(scope="module")
def slice_():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"])
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"], tuple(map(tuple, fx["tokens"])))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, rows=rows)


@pytest.fixture(scope="module")
def synth():
    """Three requests on `ENGINE`: request 1's prompt shares request 0's first two blocks and request 2's its first,
    request 1 arrives at step 2 and request 2 never in the window (its asks are 0)."""
    st = CS.statics(2, 1, 2, ENGINE, 3, Q=3, D=5, P=4, K=3, SEG=32, CN=6, CTW=4, CU=4)
    tokens = ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2), (1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 3, 5, 7, 9),
              (1, 2, 3, 4, 9, 9, 6, 6, 6))
    sl = CS.Slice(((0, 9, 8), (1, 11, 4), (2, 6, 3)), (0, 0, 1, 0, 0), 4, tokens)
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl)


def _refused(c: CS.Case) -> set[str]:
    return {f for f, ok in c.flags.items() if not ok}


def _negatives(x, flip_at: int):
    S, sl = x.S, x.sl
    honest = CS.case(S, sl, SALT)
    digest = CS.rows_digest(S, honest.steps, SALT)
    flip = list(sl.decisions)
    flip[flip_at] ^= 1
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    return {"honest": honest, "flip": CS.case(S, sl, SALT, digest=digest, window=flip),
            "charge": CS.case(S, sl, SALT, digest=digest, charge=honest.length - 1),
            "arrival": CS.case(S, sl, SALT, digest=digest, cert=cert),
            "digest": CS.case(S, sl, SALT, digest=bytes(64)),
            "salt": CS.case(S, sl, bytes(32), digest=digest)}


def _made_up(x, request: int, units: int = 1) -> dict:
    """`case`'s keywords for a made-up first table: `request`'s first `units` units under ROOT, owned by request 0."""
    S, toks = x.S, x.sl.tokens[request]
    return {"fabricate": {0: tuple(CST.Entry(S.CN if j == 0 else j - 1, tuple(toks[j * S.B:(j + 1) * S.B]), 0)
                                   for j in range(units))}}


def test_the_scan_is_the_replay(slice_):
    """The honest scan's rows are the replay's, its digest is the clear rows', and its symbols are the plain contexts'
    (ENC-REPLAY's KT counts, scaled as advice v1 does): the queue head's passes 0, 1, 2, 2, 2 and 2, then a fresh head.
    No request shares a block with another, so every hit is 0, and the table holds request 0's units as its rows
    complete them, then request 1's first."""
    x = slice_
    c = CS.case(x.S, x.sl, SALT)
    assert list(c.steps) == x.rows
    assert c.verdict == CS.ACCEPT == 0x3F and all(c.flags.values())
    assert c.hits == ((0, 0, 0),) * 7
    want, n = [], {k: (0, 0) for k in range(3)}
    for ctx, bit in zip((0, 1, 2, 2, 2, 2, 0), x.sl.decisions):
        want.append((bit, *AE.scaled(2 * n[ctx][0] + 1, 2 * n[ctx][1] + 1)))
        n[ctx] = (n[ctx][0] + (not bit), n[ctx][1] + bit)
    assert list(c.symbols) == want
    assert CS.rows_digest(x.S, x.rows, SALT) == CS.rows_digest(x.S, c.steps, SALT)
    sc = CS.scan(x.S, x.sl)
    assert [len(t.table) for t in sc.caches] == [0, 16, 23, 23, 23, 23, 23, 24]
    assert [t.pfx for t in sc.caches][-1] == (22, 23)


def test_the_negatives(slice_):
    """Each negative is refused by exactly the checks the reference names."""
    flags = {k: _refused(c) for k, c in _negatives(slice_, 5).items()}
    assert flags == {"honest": set(), "flip": {"load", "steps", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "digest": {"rows"}, "salt": {"rows"}}


def test_redteams_phase9_choices_are_refused(slice_):
    """Every interface REDTEAM's phase 9 found free in a piece alone (`compile_slice.phase9`) is the unit's own here: no
    witness field carries a context, a frequency, an eviction, a return or a consumed count, and each choice is
    refused by the verdict. The codes are shorter than the honest one (8 bits), and the unit holds the charge to its own
    count; the hit schedules 32 tokens fewer, and CacheStep_v1 refuses it (the table is empty at step 0); the heads must
    move by ReplayStep_v1's own taken and used."""
    x = slice_
    assert [name for name, _n, _w in CS.fields(x.S)["wit"]] == ["state", "window", "arr", "bits", "hit", "cache", "seg",
                                                                 "salt"]
    digest = CS.rows_digest(x.S, x.rows, SALT)
    cases = {k: CS.case(x.S, x.sl, SALT, digest=digest, **kw) for k, kw in CS.phase9(x.S, x.sl).items()}
    assert tuple(cases) == CS.PHASE9
    flags = {k: _refused(c) for k, c in cases.items()}
    assert flags == {"context": {"code", "charge"}, "frequency": {"code", "charge"}, "hit": {"cache", "rows"},
                     "reread": {"steps", "rows"}, "untaken": {"steps", "rows"}}
    assert (cases["context"].length, cases["frequency"].length) == (4, 2)
    assert cases["hit"].steps[0] == ((0, 0, 0, 32, 256),) and x.rows[0] == ((0, 0, 0, 0, 256),)


def test_the_cache_binds_the_hit(slice_):
    """REDTEAM phase 9 (4)'s binding on the corpus: CacheStep_v1's hit pinned to the committed table and its tokens free
    in the piece alone (`test_network_certifier_cache_step_phase9.py`) are bound in the slice. A made-up first table
    (request 1's first block under ROOT) gives request 1 a free 16-token hit at step 6, and the unit refuses it by its
    ``cache`` check alone when the prover serves that schedule (its own digest), since the table chains from the empty
    row; made-up tokens (request 1's first block as request 0's) give the same hit through a consistent table, and the
    rows' digest refuses them as ``cert`` is the certified record's, which the verifier checks natively."""
    x = slice_
    table = CS.case(x.S, x.sl, SALT, **_made_up(x, 1))
    assert table.steps[6] == ((0, 0, 1, 376, 1), (1, 1, 0, 16, 13)) and table.hits[6] == (0, 1, 0)
    assert _refused(table) == {"cache"}
    digest = CS.rows_digest(x.S, x.rows, SALT)
    assert _refused(CS.case(x.S, x.sl, SALT, digest=digest, **_made_up(x, 1))) == {"cache", "rows"}
    toks = [list(t) for t in x.sl.tokens]
    toks[1][:x.S.B] = toks[0][:x.S.B]
    made = CS.case(x.S, x.sl, SALT, digest=digest, tokens=toks)
    assert made.steps[6] == table.steps[6] and _refused(made) == {"rows"}


def test_the_circuit_is_its_reference(synth):
    """The unit's verdict word is the reference's on a synthetic slice where every hit path is taken: request 1 is the
    first candidate left unadmitted at steps 0 and 1 (its 2-block hit looked up at the step's last table), is admitted
    at step 2 with that hit, and request 2 is left unadmitted at steps 2 and 3 with a 1-block hit, after request 1 in
    the same step. Then on every negative, the hits claimed shorter or made up by a table or tokens, and REDTEAM's
    phase 9 choices the slice can take."""
    x = synth
    cases = _negatives(x, 2)
    honest = cases["honest"]
    assert honest.verdict == CS.ACCEPT
    assert honest.hits == ((0, 2, 0), (0, 2, 0), (0, 2, 1), (0, 1, 0))
    assert honest.steps[2] == ((0, 0, 1, 10, 1), (1, 1, 0, 8, 3))
    digest = CS.rows_digest(x.S, honest.steps, SALT)
    cases["short"] = CS.case(x.S, x.sl, SALT, digest=digest, hits={2: (0, 1, 1)})
    cases["unasked"] = CS.case(x.S, x.sl, SALT, digest=digest, hits={0: (0, 0, 0)})
    cases["table"] = CS.case(x.S, x.sl, SALT, **_made_up(x, 2, 2))
    toks = [list(t) for t in x.sl.tokens]
    toks[1][:4] = [9, 9, 9, 9]
    cases["tokens"] = CS.case(x.S, x.sl, SALT, digest=digest, tokens=toks)
    red = CS.phase9(x.S, x.sl)
    for k in ("context", "frequency", "hit"):
        cases[k] = CS.case(x.S, x.sl, SALT, digest=digest, **red[k])
    got = CS.evaluate(x.st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()]
    flags = {k: _refused(c) for k, c in cases.items()}
    assert flags == {"honest": set(), "flip": {"load", "steps", "rows"}, "charge": {"charge"}, "arrival": {"rows"},
                     "digest": {"rows"}, "salt": {"rows"}, "short": {"cache", "rows"}, "unasked": {"cache"},
                     "table": {"cache"}, "tokens": {"rows"}, "context": {"code", "charge"},
                     "frequency": {"code", "charge"}, "hit": {"cache", "rows"}}
    assert cases["table"].hits[0] == (1, 2, 0) and cases["table"].steps[0] == ((0, 0, 0, 4, 5),)
    assert cases["tokens"].hits[2] == (0, 0, 1) and cases["tokens"].steps[2][1] == (1, 1, 0, 0, 11)


def test_the_codeword_is_the_encoders():
    """`codeword` is the segment `arith_encode`'s reference accepts, and no bit of it can flip."""
    rng = random.Random(7)
    for _ in range(40):
        syms = [(rng.randrange(2), rng.randrange(1, 300), rng.randrange(1, 300)) for _ in range(rng.randrange(1, 12))]
        seg = CS.codeword(syms)
        assert AE.check_window(AE.K_MAX, 1, syms, seg)
        k = rng.randrange(len(seg))
        assert not AE.check_window(AE.K_MAX, 1, syms, seg[:k] + [1 - seg[k]] + seg[k + 1:])


def test_size(synth, slice_):
    s = CS.size(**synth.st)
    assert s == {"ands": 127800, "pub_bits": 528, "adv_bits": 16, "cert_bits": 256, "wit_bits": 1856}
    assert CS.row_bits(slice_.S) == {"pub": 528, "adv": 16, "cert": 20512, "wit": 57216}

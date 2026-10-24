"""The served rows (`served_rows`) on the compile slice's corpus-smallkv window: the server's rows as the model runner laid
them out give the slice's rows, and under the digest the Commit side publishes, never the prover's own, the slice refuses
REDTEAM's phase 10 finding 2 (six passes registered where user-1 was admitted at the sixth ask) at ``rows``; the honest run
is accepted. The registration's row is bound by b‖c to its leaf, its run root and its index, and the public record holds
no row or salt."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
ROWS = 1 << CS.FLAGS.index("rows")
RUN_ROOT = bytes(range(64))


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    prompt = {a[0]: a[1] for a in fx["arrivals"]}
    ids = {f"user-{a[0]}-0a1b2c3d": a[0] for a in fx["arrivals"]}
    by_step = {}
    for u, step in enumerate(rows):   # the runner's layout of the fixture's rows: token rows [lo, hi) in slot order
        lo, got = 0, []
        for _slot, req, _kind, start, n in step:
            got.append((f"user-{req}-0a1b2c3d", lo, lo + n, start, prompt[req]))
            lo += n
        by_step[u] = got
    session = SR.windows(S, by_step, ids, fx["steps"])
    served = SR.register(S, session, RUN_ROOT)
    return SimpleNamespace(st=st, S=S, sl=sl, rows=rows, ids=ids, by_step=by_step, session=session, served=served,
                           record=served.record())


def test_the_runner_rows_are_the_slices(x):
    assert len(x.session) == 1 and list(x.session[0]) == x.rows
    salt = x.served.digest_salts[0]
    assert x.served.digests[0] == CS.rows_digest(x.S, x.rows, salt)


def test_the_slice_refuses_six_passes_under_the_published_digest(x):
    """The honest run under the Commit's digest salt is accepted with the verifier's own ``pub``; REDTEAM's cheap run, six
    passes and a 5-bit code, accepted at 0x1F under its own digest, is refused at ``rows`` under the published one,
    whatever salt the prover puts in its witness."""
    S, sl, salt = x.S, x.sl, x.served.digest_salts[0]
    honest = CS.case(S, sl, salt)
    charge = honest.length
    pub = SR.pub(S, x.record, 0, charge)
    honest = CS.case(S, sl, salt, digest=bytes.fromhex(x.record["digests"][0]), charge=charge)
    assert CS.words(honest.rows["pub"]) == pub
    cheap_sl = CS.Slice(sl.arrivals, (0,) * 6, sl.steps)
    own = CS.case(S, cheap_sl, salt)
    assert own.length == 5 < charge
    held = [CS.case(S, cheap_sl, s, digest=bytes.fromhex(x.record["digests"][0]), charge=own.length)
            for s in (salt, bytes(CS.SALT))]
    got = CS.evaluate(x.st, [honest, own, *held])
    assert got == [CS.ACCEPT, CS.ACCEPT, CS.ACCEPT & ~ROWS, CS.ACCEPT & ~ROWS]
    assert [c.verdict for c in (honest, own, *held)] == got


def test_the_row_is_bound_by_bc_to_its_leaf_run_and_index(x):
    reg = x.served.registration
    path = reg.open(0).path
    assert SR.leaf_ok(x.record, RUN_ROOT, 0, x.served.bc(0), path)
    assert not SR.leaf_ok(x.record, bytes(64), 0, x.served.bc(0), path)
    assert not SR.leaf_ok(x.record, RUN_ROOT, 1, x.served.bc(0), path)
    cheap = CS.case(x.S, CS.Slice(x.sl.arrivals, (0,) * 6, x.sl.steps), x.served.digest_salts[0])
    forged = SR.register(x.S, [list(cheap.steps)], RUN_ROOT, salts=reg.salts)
    assert forged.registration.rows != reg.rows
    assert not SR.leaf_ok(x.record, RUN_ROOT, 0, forged.bc(0), path)


def test_two_windows_and_an_idle_tail(x):
    """Steps past the first window open a second, its idle steps zero; each window has its own leaf and digest."""
    by_step = {**x.by_step, 8: x.by_step[2]}
    session = SR.windows(x.S, by_step, x.ids, 10)
    assert [len(w) for w in session] == [7, 3] and session[1] == [None, x.session[0][2], None]
    served = SR.register(x.S, session, RUN_ROOT)
    rec = served.record()
    assert rec["count"] == 2 and served.digests[1] == CS.rows_digest(x.S, session[1], served.digest_salts[1])
    for w in (0, 1):
        assert SR.leaf_ok(rec, RUN_ROOT, w, served.bc(w), served.registration.open(w).path)


def test_the_record_is_public_only(x):
    text = json.dumps(x.record)
    private = [*x.served.registration.salts, *x.served.digest_salts, *x.served.registration.rows]
    assert not any(p.hex() in text for p in private)
    assert set(x.record) == {"schema", "binding", "root", "count", "n_bits", "digests"}


@pytest.mark.parametrize("bad, why", [
    ({0: [("user-9-0a1b2c3d", 0, 1, 0, 1)]}, "no certified arrival"),
    ({0: [("user-0-0a1b2c3d", 0, 1, 1 << 20, 1)]}, "start"),
    ({0: [("user-0-0a1b2c3d", 0, 1, 0, 1)] * 3}, "over R"),
])
def test_rows_it_cant_pack_are_refused(x, bad, why):
    with pytest.raises(ValueError, match=why):
        SR.windows(x.S, bad, x.ids, 1)
